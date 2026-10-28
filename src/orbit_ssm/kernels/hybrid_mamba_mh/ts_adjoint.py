"""TIME-SPLIT backward (TS): the incoming adjoint of every super-chunk.

`_mh_bwd_kernel` walks the chunks of one (b, head-tile) in REVERSE and carries
the state adjoint across them, so one CTA owns the whole sequence.  At small
B * D / HP (long L, small batch) that leaves most of the machine idle: B2 D2048
L8192 N16 runs 512 one-warp CTAs.

The carry is LINEAR in the adjoint, so the sequence can be cut into TS
super-chunks that run in parallel if each one is handed the adjoint that would
have arrived from its right:

    G_{s-1} = lam0_local(s) + exp(sum_{t in s} log a_t) * G_s,   G_{TS-1} = gh_final

where lam0_local(s) is the carry the chunk walk of super-chunk s would emit at
its left edge from a ZERO seed (`al0 = a_t0 * lam_t0`, the kernel's own carry
convention) and the decay product is that of the whole super-chunk.  This
module computes lam0_local and the per-super-chunk delta sums with one small
chunk-parallel Triton kernel (only the adjoint recurrence -- no state replay,
no gradients), and the host combines them in TS - 1 fused steps.  The main
kernel then seeds each super-chunk's walk with its G_s.

Exact up to fp32 rounding: every term is the one the sequential walk adds,
only the association order across super-chunk boundaries changes.

ENVELOPE (refused by name in `ts_refusal`): the real LAW_A recurrence of the
Mamba-1 function -- delta (+bias, softplus), A, B, C, gy, optional output
gate z and D skip -- with a caller checkpoint (`hck`, so pass 1 is skipped)
and none of: rotation, pair maps, B/C biases or gates, per-head B/C, delta_w,
the MU/DYN laws, the trapezoid endpoint, an n window.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

LOG2E = 1.4426950408889634


def ts_refusal(*, ROT, pmap, LAW_MASK, hck, delta_w, beta_b, beta_c, g_B, g_C,
               endpoint_own, bc_head, nr, N, NP):
    """None when the time split may serve this launch, else why not."""
    if hck is None:
        return "needs the forward's checkpoints (pass 1 is sequential)"
    if ROT or pmap:
        return "rotation not served"
    if LAW_MASK != 1:
        return "only the LAW_A decay law"
    if delta_w is not None:
        return "delta_w not served"
    if beta_b is not None or beta_c is not None:
        return "B/C biases not served"
    if g_B is not None or g_C is not None:
        return "B/C gates not served"
    if endpoint_own is not None:
        return "trapezoid endpoint not served"
    if bc_head:
        return "per-head B/C not served"
    if nr != N:
        return "n window not served"
    if NP != 1:
        return "p shards not served"
    return None


@triton.autotune(
    configs=[triton.Config(dict(BLOCK_C=bc, BLOCK_T=bt), num_warps=w)
             for bc in (16, 32, 64) for bt in (32, 64, 128) for w in (2, 4, 8)
             if bc * bt // (32 * w) >= 4],
    # plain stores, no atomics: re-running a trial is harmless
    key=["H", "P", "N", "L", "SPAN", "Bs"])
@triton.jit
def _ts_adj_kernel(gy_ptr, z_ptr, delta_ptr, dbias_ptr, Ae_ptr, C_ptr,
                   lam_ptr, dsum_ptr,
                   Bs, H, P, N, L, SPAN,
                   BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
                   HAS_Z: tl.constexpr, HAS_DBIAS: tl.constexpr,
                   SOFTPLUS: tl.constexpr):
    """grid (cdiv(H*P, BLOCK_C), B, TS).  One program = BLOCK_C channels of
    one batch row over one super-chunk [t0, t1).

    No scan: the zero-seeded left-edge carry is a WEIGHTED SUM,

        al0_local(n) = sum_{t in [t0,t1)} exp(A_n * cd_t) * gyp_t * C_t(n),
        cd_t = sum_{k=t0..t} delta_k   (n-free, inclusive),

    because a_{t0} lam_{t0} = sum_t (prod_{k=t0..t} a_k) beta_t.  So gy, z and
    delta are read ONCE per tile and every n rides the same registers.

    Writes lam_ptr (TS, B, H*P, N) and dsum_ptr (TS, B, H*P) (= cd_{t1-1}).
    """
    pc = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    s = tl.program_id(2)
    D = H * P
    c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < D
    h = c // P
    t0 = s * SPAN
    t1 = tl.minimum(t0 + SPAN, L)
    k = tl.arange(0, BLOCK_T)
    nn = tl.arange(0, BLOCK_N)
    nm = nn < N
    ae = tl.load(Ae_ptr + c[:, None] * N + nn[None, :], mask=cm[:, None] & nm[None, :], other=0.0)
    if HAS_DBIAS:
        db = tl.load(dbias_ptr + h, mask=cm, other=0.0)
    acc = tl.zeros([BLOCK_C, BLOCK_N], tl.float32)
    cd0 = tl.zeros([BLOCK_C], tl.float32)
    for tt in range(t0, t1, BLOCK_T):
        t = tt + k
        tm = t < t1
        m2 = cm[:, None] & tm[None, :]
        off = (b * D + c)[:, None] * L + t[None, :]
        gy = tl.load(gy_ptr + off, mask=m2, other=0.0).to(tl.float32)
        if HAS_Z:
            zt = tl.load(z_ptr + off, mask=m2, other=0.0).to(tl.float32)
            gy = gy * (zt / (1.0 + tl.exp2(-zt * 1.4426950408889634)))
        dl = tl.load(delta_ptr + (b * H + h)[:, None] * L + t[None, :], mask=m2, other=0.0).to(tl.float32)
        if HAS_DBIAS:
            dl = tl.where(m2, dl + db[:, None], 0.0)
        if SOFTPLUS:
            dl = tl.where(m2, tl.where(dl > 20.0, dl, libdevice.log1p(tl.exp(dl))), 0.0)
        cd = tl.cumsum(dl, axis=1) + cd0[:, None]                         # inclusive
        cd0 = tl.sum(dl, axis=1) + cd0
        for n in range(N):
            a_n = tl.sum(tl.where(nn[None, :] == n, ae, 0.0), axis=1)        # (BLOCK_C,)
            cr = tl.load(C_ptr + (b * N + n) * L + t, mask=tm, other=0.0).to(tl.float32)
            w = tl.exp2(a_n[:, None] * 1.4426950408889634 * cd)
            v = tl.sum(w * gy * cr[None, :], axis=1)
            acc += tl.where(nn[None, :] == n, v[:, None], 0.0)
    tl.store(lam_ptr + ((s * Bs + b) * D + c)[:, None] * N + nn[None, :], acc,
             mask=cm[:, None] & nm[None, :])
    tl.store(dsum_ptr + (s * Bs + b) * D + c, cd0, mask=cm)


@triton.jit
def _ts_combine_kernel(lam_ptr, dsum_ptr, Ae_ptr, gh_ptr, out_ptr,
                       TS, Bs, H, P, N, BLOCK: tl.constexpr, HAS_GH: tl.constexpr,
                       PACKED: tl.constexpr):
    """G_{TS-1} = gh (or 0); G_{s-1} = lam[s] + exp(A * dsum[s]) * G_s, written
    straight into the main kernel's `alam` layout: PACKED (TS,B,N,D), else
    (TS,B,H,N,P).  One launch instead of a host loop (whose TS-1 small launches
    the search's GPU-event timing does not see)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    D = H * P
    tot = Bs * D * N
    m = i < tot
    n = i % N
    c = (i // N) % D
    b = i // (N * D)
    ae = tl.load(Ae_ptr + c * N + n, mask=m, other=0.0)
    if HAS_GH:
        g = tl.load(gh_ptr + i, mask=m, other=0.0)
    else:
        g = tl.zeros([BLOCK], tl.float32)
    h = c // P
    p = c % P
    for jj in range(TS):
        s = TS - 1 - jj
        if PACKED:
            o = ((s * Bs + b) * N + n) * D + c
        else:
            o = (((s * Bs + b) * H + h) * N + n) * P + p
        tl.store(out_ptr + o, g, mask=m)
        lam = tl.load(lam_ptr + (s * Bs + b) * D * N + c * N + n, mask=m, other=0.0)
        ds = tl.load(dsum_ptr + (s * Bs + b) * D + c, mask=m, other=0.0)
        g = lam + tl.exp2(ae * 1.4426950408889634 * ds) * g


def ts_seeds(TS, span, x, delta, A_log, C_t, gy, *, delta_bias=None,
             delta_softplus=True, z=None, gh_final=None, out=None, packed=False):
    """(TS, B, H, P, N) fp32: the adjoint arriving at the RIGHT edge of each
    super-chunk (the value the main kernel seeds its walk with).

    `span` is the super-chunk length in tokens (a multiple of the kernel's
    chunk).  The last super-chunk is seeded with `gh_final` (or zero)."""
    Bs, H, P, L = x.shape
    N = A_log.shape[-1]
    D = H * P
    dev = x.device
    Ae = (-torch.exp(A_log.float())).reshape(D, N).contiguous()
    lam = torch.empty(TS, Bs, D, N, device=dev, dtype=torch.float32)
    dsum = torch.empty(TS, Bs, D, device=dev, dtype=torch.float32)
    BLOCK_N = triton.next_power_of_2(N)
    grid = lambda meta: (triton.cdiv(D, meta["BLOCK_C"]), Bs, TS)
    from .autotune import carry_launcher
    carry_launcher(_ts_adj_kernel, "bwd")[grid](
        gy.contiguous(), z.contiguous() if z is not None else gy,
        delta.contiguous(), delta_bias.float().contiguous() if delta_bias is not None else Ae,
        Ae, C_t.contiguous(), lam, dsum,
        Bs, H, P, N, L, span,
        BLOCK_N=BLOCK_N, HAS_Z=z is not None,
        HAS_DBIAS=delta_bias is not None, SOFTPLUS=bool(delta_softplus))
    ret = None
    if out is None:
        out = torch.empty(TS, Bs, H, N, P, device=dev, dtype=torch.float32)
        packed, ret = False, out.permute(0, 1, 2, 4, 3)         # (TS,B,H,P,N) view
    gh = gh_final.float().reshape(Bs, D, N).contiguous() if gh_final is not None else lam
    BLK = 256
    _ts_combine_kernel[(triton.cdiv(Bs * D * N, BLK),)](
        lam, dsum, Ae, gh, out, TS, Bs, H, P, N, BLOCK=BLK,
        HAS_GH=gh_final is not None, PACKED=bool(packed))
    return out if ret is None else ret
