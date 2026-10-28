"""TIME-SPLIT forward (TS): the incoming state of every super-chunk.

`_mh_fwd_kernel` walks the chunks of one (b, head, p-tile) in order and
carries the state across them, so one CTA owns the whole sequence.  At long L
and small B * D that is a short grid of long walks: B2 D2048 L8192 runs 4096
one-warp CTAs of 8192 steps, one wave and a tail on 114-132 SMs.

The carry is LINEAR in the state, so the sequence can be cut into TS
super-chunks that run in parallel if each one is handed the state that would
have arrived from its left -- the forward twin of `ts_adjoint`:

    H_0 = h_init,   H_s = exp(A * dsum[s-1]) * H_{s-1} + hloc[s-1]

where hloc(s) is the end state super-chunk s reaches from a ZERO seed and
dsum(s) the sum of its step sizes.  hloc needs no scan: it is a weighted sum,

    hloc(n) = sum_{t in s} exp(A_n * (cd_end - cd_t)) * dt_t * B_t(n) * x_t,

computed tile by tile with a running rescale so every exponent is <= 0.
The remaining step-size sum is formed directly by an exclusive reverse
scan, never by subtracting rounded prefixes (large reset / slow-tail case).
The main kernel then seeds each super-chunk's walk with its H_s.

Exact up to fp32 rounding: every term is the one the sequential walk adds,
only the association order across super-chunk boundaries changes.

ENVELOPE (refused by name in `ts_fwd_refusal`): the real LAW_A recurrence of
the Mamba-1 function -- delta (+bias, softplus), A, B, C, x, optional h_init,
D skip, output gate z, checkpoints -- and none of: rotation, pair maps, B/C
biases or gates, beta_a, per-head B/C, delta_w, the MU/DYN laws, the
trapezoid endpoint, an n window, channel packing (FPACK).
"""
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice


def ts_fwd_refusal(*, ROT, pmap, LAW_MASK, delta_w, beta_a, beta_b, beta_c,
                   g_B, g_C, beta_gB, beta_gC, endpoint_own, bc_head, n_off,
                   nr, N, fpack):
    """None when the time split may serve this forward launch, else why not."""
    if ROT or pmap:
        return "rotation not served"
    if LAW_MASK != 1:
        return "only the LAW_A decay law"
    if delta_w is not None:
        return "delta_w not served"
    if beta_a is not None:
        return "beta_a not served"
    if any(t is not None for t in (beta_b, beta_c, g_B, g_C, beta_gB, beta_gC)):
        return "B/C biases or gates not served"
    if endpoint_own is not None:
        return "trapezoid endpoint not served"
    if bc_head:
        return "per-head B/C not served"
    if n_off != 0 or nr != N:
        return "n window not served"
    if fpack:
        return "channel packing not served"
    return None


@gluon.jit
def _add(a, b):
    return a + b


@triton.autotune(
    configs=[triton.Config(dict(BLOCK_C=bc, BLOCK_T=bt), num_warps=w)
             for bc in (16, 32, 64) for bt in (32, 64, 128) for w in (2, 4, 8)
             if bc * bt // (32 * w) >= 4],
    # plain stores, no atomics: re-running a trial is harmless
    key=["H", "P", "N", "L", "SPAN", "Bs"])
@gluon.jit
def _ts_fwd_kernel(x_ptr, delta_ptr, dbias_ptr, Ae_ptr, B_ptr,
                   hloc_ptr, dsum_ptr,
                   Bs, H, P, N, L, SPAN,
                   BLOCK_C: gl.constexpr, BLOCK_T: gl.constexpr, BLOCK_N: gl.constexpr,
                   HAS_DBIAS: gl.constexpr, SOFTPLUS: gl.constexpr):
    """grid (cdiv(H*P, BLOCK_C), B, TS - 1).  One program = BLOCK_C channels
    of one batch row over one super-chunk [t0, t1) (the last super-chunk's
    end state is nobody's seed, so it is not computed).

    Per tile, with rem_t = sum_{j>t} dt_j and T its total:
        acc(n) <- exp(A_n T) acc(n) + sum_t exp(A_n rem_t) dt_t B_t(n) x_t
    x, delta are read ONCE per tile and every n rides the same registers.

    Writes hloc_ptr (TS-1, B, H*P, N) and dsum_ptr (TS-1, B, H*P).
    """
    pc = gl.program_id(0)
    b = gl.program_id(1).to(gl.int64)
    s = gl.program_id(2)
    D = H * P
    tile: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [1, gl.num_warps()], [1, 0])
    c = pc * BLOCK_C + gl.arange(0, BLOCK_C, layout=gl.SliceLayout(1, tile))
    cm = c < D
    h = c // P
    t0 = s * SPAN
    t1 = gl.minimum(t0 + SPAN, L)
    k = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(0, tile))
    nn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, tile))
    nm = nn < N
    ae = gl.load(Ae_ptr + c[:, None] * N + nn[None, :], mask=cm[:, None] & nm[None, :], other=0.0)
    if HAS_DBIAS:
        db = gl.load(dbias_ptr + h, mask=cm, other=0.0)
    acc = gl.zeros([BLOCK_C, BLOCK_N], gl.float32, layout=tile)
    ds = gl.zeros([BLOCK_C], gl.float32, layout=gl.SliceLayout(1, tile))
    for tt in range(t0, t1, BLOCK_T):
        t = tt + k
        tm = t < t1
        m2 = cm[:, None] & tm[None, :]
        xt = gl.load(x_ptr + (b * D + c)[:, None] * L + t[None, :], mask=m2, other=0.0).to(gl.float32)
        dl = gl.load(delta_ptr + (b * H + h)[:, None] * L + t[None, :], mask=m2, other=0.0).to(gl.float32)
        if HAS_DBIAS:
            dl = dl + db[:, None]
        if SOFTPLUS:
            dl = gl.where(dl > 20.0, dl, libdevice.log1p(gl.exp(dl)))
        dl = gl.where(m2, dl, 0.0)
        # Exclusive suffix by a reverse scan and shift. Total - prefix
        # cancels after a large reset followed by small positive steps.
        suffix = gl.associative_scan(dl, 1, _add, reverse=True)
        idx = gl.minimum(k[None, :] + 1, BLOCK_T - 1) + gl.zeros(
            [BLOCK_C, 1], gl.int32, layout=tile)
        rem = gl.where(k[None, :] < BLOCK_T - 1,
                       gl.gather(suffix, idx, axis=1), 0.0)
        tot = gl.sum(dl, axis=1)
        xw = dl * xt
        ds += tot
        for n in range(N):
            a_n = gl.sum(gl.where(nn[None, :] == n, ae, 0.0), axis=1)        # (BLOCK_C,)
            br = gl.load(B_ptr + (b * N + n) * L + t, mask=tm, other=0.0).to(gl.float32)
            w = gl.exp2(a_n[:, None] * 1.4426950408889634 * rem)
            v = gl.sum(w * xw * br[None, :], axis=1)
            dec = gl.exp2(a_n * 1.4426950408889634 * tot)
            sel = nn[None, :] == n
            acc = gl.where(sel, acc * dec[:, None] + v[:, None], acc)
    gl.store(hloc_ptr + ((s * Bs + b) * D + c)[:, None] * N + nn[None, :], acc,
             mask=cm[:, None] & nm[None, :])
    gl.store(dsum_ptr + (s * Bs + b) * D + c, ds, mask=cm)


@gluon.jit
def _ts_fwd_combine_kernel(hloc_ptr, dsum_ptr, Ae_ptr, h0_ptr, out_ptr,
                           TS, Bs, H, P, N, BLOCK: gl.constexpr, HAS_H0: gl.constexpr):
    """H_0 = h_init (or 0); H_s = exp(A * dsum[s-1]) * H_{s-1} + hloc[s-1],
    written straight into the main kernel's carry layout (TS, B, H, N, P).
    `h0_ptr` is that same layout's slab 0 order, (B, H, N, P)."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    i = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    D = H * P
    tot = Bs * D * N
    m = i < tot
    # i enumerates (b, h, n, p), p fastest -- the carry layout
    p = i % P
    n = (i // P) % N
    h = (i // (P * N)) % H
    b = i // (P * N * H)
    c = h * P + p
    ae = gl.load(Ae_ptr + c * N + n, mask=m, other=0.0)
    if HAS_H0:
        g = gl.load(h0_ptr + i, mask=m, other=0.0)
    else:
        g = gl.zeros([BLOCK], gl.float32, layout=layout)
    gl.store(out_ptr + i, g, mask=m)
    for s in range(1, TS):
        hl = gl.load(hloc_ptr + ((s - 1) * Bs + b) * D * N + c * N + n, mask=m, other=0.0)
        dd = gl.load(dsum_ptr + ((s - 1) * Bs + b) * D + c, mask=m, other=0.0)
        g = gl.exp2(ae * 1.4426950408889634 * dd) * g + hl
        gl.store(out_ptr + s * tot + i, g, mask=m)


def ts_fwd_seeds(TS, span, x, delta, A_log, B_t, out, *, delta_bias=None,
                 delta_softplus=True, h0=None):
    """Fill `out` (TS, B, H, N, P) fp32 with the state arriving at the LEFT
    edge of each super-chunk: slab 0 is `h0` ((B, H, N, P), or zero).

    `span` is the super-chunk length in tokens (a multiple of the kernel's
    chunk)."""
    Bs, H, P, L = x.shape
    N = A_log.shape[-1]
    D = H * P
    dev = x.device
    Ae = (-torch.exp(A_log.float())).reshape(D, N).contiguous()
    hloc = torch.empty(max(TS - 1, 1), Bs, D, N, device=dev, dtype=torch.float32)
    dsum = torch.empty(max(TS - 1, 1), Bs, D, device=dev, dtype=torch.float32)
    if TS > 1:
        from .autotune import carry_launcher
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_C"]), Bs, TS - 1)
        carry_launcher(_ts_fwd_kernel, "fwd")[grid](
            x.contiguous(), delta.contiguous(),
            delta_bias.float().contiguous() if delta_bias is not None else Ae,
            Ae, B_t.contiguous(), hloc, dsum,
            Bs, H, P, N, L, span,
            BLOCK_N=triton.next_power_of_2(N),
            HAS_DBIAS=delta_bias is not None, SOFTPLUS=bool(delta_softplus))
    BLK = 256
    _ts_fwd_combine_kernel[(triton.cdiv(Bs * D * N, BLK),)](
        hloc, dsum, Ae, h0 if h0 is not None else Ae, out, TS, Bs, H, P, N,
        BLOCK=BLK, HAS_H0=h0 is not None)
    return out
