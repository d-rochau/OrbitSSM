"""Host-side maps of the exact-Mamba-3 (`m3`) mode, as Triton launches.

1. **The per-chunk phase OFFSET build.**  `theta` holds per-token angle
   INCREMENTS; the kernels that read the phase need the phase ENTERING
   each chunk, `off[b, h, c, k] = sum_{c' < c} sum_{t in c'} dtheta`.
   `phase_offset_build` is that reduction and scan in ONE kernel (grid
   (B, H), serial over CHUNKS, reducing the [T, K] tile the phase already
   lives in), and `phase_offset_bwd_add` is its adjoint: the cross-chunk term `sum_{c' > c} dOff[c']`, broadcast over
   the chunk's tokens and ADDED into the increment gradient the backward
   kernels already produced.  Neither tensor leaves the device and no
   autograd node is built, so the two dTheta terms meet in one kernel's
   `+` rather than in an autograd add over [B, L, H, N/2].

2. **The two-tap trapezoid's ADJOINT** (backward, 2 Triton launches).
   The trapezoid's forward lives in the scan kernel's store and its two
   activation-sized adjoint terms at Kc's dY load; the rest is a PROLOGUE
   that rebuilds the two-tap write scale and an EPILOGUE for dB / dC /
   dbeta / dgamma and the write scale's own chain rule.

   `trap_prologue` is the first: `n_t = dt_{t+1}(1 - lam_{t+1})`, the
   collapsed write scale `w_t = lam_t dt_t + n_t`, the product
   `n_t d_t` Kc loads, and the zero-fill of the `dSdot` buffer Kc
   accumulates into -- one pass over four [B, H, L] vectors.

   `trap_adjoint` is the second, and it runs AFTER Kc because both of
   its inputs are produced there: `A_t = dL/dw_t` (`dDW`) and
   `s_t = sum_g dY_{t,g} v_{t,g}` (`dSdot`) are atomic reductions over
   the G tiles, i.e. over CTAs.  It is chunk-parallel, grid (B, nc),
   with the HEAD loop INSIDE:

       dd_t^h   = own_t^h s_t^h                      (the k.q gradient)
       dB[b,l]  = (sum_h dd) C[b,l] + dd . gamma     ([T, HB] x [HB, N])
       dC[b,l]  = (sum_h dd) B[b,l] + dd . beta
       dGamma   = dd^T B + beta  sum_t dd            (atomic, [H, N])
       dBeta    = dd^T C + gamma sum_t dd
       M_t      = A_t                                (dL/dn_t)
       ddt_t    = (A_t + d_t s_t) lam_t + M_{t-1}(1 - lam_t)
       dlam_t   = dt_t (A_t + d_t s_t - M_{t-1})

   TRAPFOLD (fused_fwd_kernel `_TRAPFOLD_NOTE`): A_t is the STRICT-tile
   dL/dw_t (Kc's score mask excludes the diagonal under HAS_TRAP_ADJ) and
   the diagonal -- weight own_t = lam_t dt_t in the forward -- is carried
   here, exactly, as the per-token terms d_t s_t (write scale) and
   own_t s_t (k.q).  This avoids an O(n_t) pair cancelling to the
   O(own_t) truth (reset heads carry n_t / own_t up to 2e6).

   The head sum is taken FIRST (it is a [B, H, L] reduction, and the
   remaining per-head coupling to B / C is the rank-`H` product with
   beta / gamma, i.e. a `tl.dot`), so the only atomics on
   activation-sized tensors are ONE dB and ONE dC tile per (b, chunk) --
   H times less atomic traffic than a per-head grid.  The accumulation
   rule: a CTA that is not the sole fan-in owner adds atomically, and
   here no CTA ever owns a tile because Kbc-hb / Kc wrote it first.

Stability by construction: the trapezoid is a WRITE-side rank-1 term and
the phase is a ROTATION.  Neither reaches `a_val * D`, so every decay factor
is still `exp(a * (a non-negative chunk-local difference))` formed
difference-first.  No clamp is added here and none is needed.
"""

import torch
from ..autograd_guard import no_double_backward as _no_double_backward
import triton
import triton.language as tl
from .common import pdot  # FP32DOT operand rounding
from triton.language.extra import libdevice

from .fused_fwd_kernel import _env
from .common import ACC_SEM, _cos_approx, _sin_approx, _wrap2pi


# ══════════════════════════════════════════════════════════════════════
# 1. The per-chunk phase offset
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def _phase_offset_kernel(
    Th_ptr, Off_ptr, nc,
    s_tb, s_tc, s_tt, s_th, s_tk,
    s_ob, s_oh, s_oc, s_ok,
    T: tl.constexpr, K: tl.constexpr,
):
    """off[b, h, c, k] = sum_{c' < c} sum_{t} dtheta[b, c', t, h, k].

    One CTA per (b, h): the scan is over CHUNKS (nc, not L) and the
    reduction over T rides the [T, K] tile the phase already lives in,
    so the whole build is one streaming pass with `nc` serial adds.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_t = tl.arange(0, T)
    offs_k = tl.arange(0, K)
    run = tl.zeros([K], dtype=tl.float32)
    tbase = Th_ptr + b * s_tb + h * s_th
    obase = Off_ptr + b * s_ob + h * s_oh + offs_k * s_ok
    for c in range(nc):
        tl.store(obase + c * s_oc, run)
        tile = tl.load(
            tbase + c * s_tc
            + offs_t[:, None] * s_tt + offs_k[None, :] * s_tk,
        ).to(tl.float32)
        run = _wrap2pi(run + tl.sum(tile, axis=0))     # PHASEWRAP


@triton.jit
def _phase_offset_bwd_kernel(
    dOff_ptr, dTh_ptr, nc,
    s_ob, s_oh, s_oc, s_ok,
    s_tb, s_tc, s_tt, s_th, s_tk,
    T: tl.constexpr, K: tl.constexpr,
):
    """dTheta += sum_{c' > c} dOff[b, h, c', k], broadcast over the chunk.

    An increment in chunk `c` moves the offset of every LATER chunk, so
    the cross-chunk term is a reverse EXCLUSIVE prefix sum over chunks.
    It is added to the intra-chunk reverse cumsum Kbc-hb already stored,
    which is why this is a read-modify-write and not a store.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_t = tl.arange(0, T)
    offs_k = tl.arange(0, K)
    run = tl.zeros([K], dtype=tl.float32)
    obase = dOff_ptr + b * s_ob + h * s_oh + offs_k * s_ok
    tbase = dTh_ptr + b * s_tb + h * s_th
    for i in range(nc):
        c = nc - 1 - i
        p = (tbase + c * s_tc
             + offs_t[:, None] * s_tt + offs_k[None, :] * s_tk)
        tl.store(p, tl.load(p) + run[None, :])
        run += tl.load(obase + c * s_oc).to(tl.float32)


@triton.jit
def _phase_prefix_bwd_kernel(
    dTh_ptr, dOff_ptr, Out_ptr, nc,
    s_ob, s_oh, s_oc, s_ok,
    s_tb, s_tc, s_tt, s_th, s_tk,
    s_qb, s_qc, s_qt, s_qh, s_qk,
    T: tl.constexpr, K: tl.constexpr, KT: tl.constexpr,
    ACC_CROSS: tl.constexpr, WRITE_OFF: tl.constexpr,
):
    """The WHOLE increment-reparameterisation adjoint, in ONE pass.

    `dTh` arrives holding the RAW `dL/dtheta_cum` -- what the five
    backward kernels accumulate before any reparameterisation -- and
    leaves holding `dL/d incr`:

        dL/d off_c  = sum_t  dL/dtheta_cum[c, t]                  (the chunk total)
        dL/d incr_t = sum_{t' >= t, same chunk} dL/dtheta_cum[t']  (in-chunk reverse cumsum)
                      + sum_{c' > c} dL/d off_c'                   (the cross-chunk term)

    All three are LINEAR in the phase gradient, so running them AFTER
    the cross-CTA accumulation is exact and needs no ordering rule the
    kernels did not already have.

    One CTA per (b, h), the same streaming shape as
    `_phase_offset_bwd_kernel` (which did only the third line).  Doing
    this here keeps the in-chunk scan, a cross-thread scan that would run
    once per HEAD inside a register-starved body, out of the backward
    kernels.

    `WRITE_OFF` returns `dL/d off_c` to a caller that supplied its own
    `theta_off` (and then `ACC_CROSS` is off: that caller owns the
    cross-chunk term on its own graph).

    `Out_ptr` is where `dL/d incr` lands: the same tensor as `dTh_ptr`
    for the in-place callers (`_SSDMimoFn`'s own `dTheta` buffer), a
    FRESH buffer for `phase_law_bwd`, whose `dTh` is an autograd
    `grad_output` that may be the caller's own tensor.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    # The chunk axis is SERIAL (the cross-chunk term is a scan over it),
    # so the only parallelism beyond (b, h) is the PAIR axis: KT lanes
    # per CTA.
    kt = tl.program_id(2)
    offs_t = tl.arange(0, T)
    offs_k = kt * KT + tl.arange(0, KT)
    run = tl.zeros([KT], dtype=tl.float32)
    obase = dOff_ptr + b * s_ob + h * s_oh + offs_k * s_ok
    tbase = dTh_ptr + b * s_tb + h * s_th
    qbase = Out_ptr + b * s_qb + h * s_qh
    for i in range(nc):
        c = nc - 1 - i
        p = (tbase + c * s_tc
             + offs_t[:, None] * s_tt + offs_k[None, :] * s_tk)
        q = (qbase + c * s_qc
             + offs_t[:, None] * s_qt + offs_k[None, :] * s_qk)
        tile = tl.load(p).to(tl.float32)
        tot = tl.sum(tile, axis=0)
        if WRITE_OFF:
            tl.store(obase + c * s_oc, tot)
        tile = tl.cumsum(tile, axis=0, reverse=True)
        if ACC_CROSS:
            tile = tile + run[None, :]
            run += tot
        tl.store(q, tile)


def _phase_shapes(theta):
    """(B, nc, T, H_axis, K) and the 5-stride list for a 4- or 5-d phase.

    A 4-d `[B, nc, T, K]` phase is head-SHARED: the head axis is length
    one with stride 0, exactly as the kernels index it.
    """
    if theta.dim() == 5:
        B, nc, T, H, K = theta.shape
        st = list(theta.stride())
    else:
        B, nc, T, K = theta.shape
        H = 1
        s = theta.stride()
        st = [s[0], s[1], s[2], 0, s[3]]
    return (B, nc, T, H, K), st


def phase_offset_build(theta: torch.Tensor) -> torch.Tensor:
    """[B, nc, T, (H,) K] increments -> [B, H|1, nc, K] chunk-entry phase.

    ONE launch; builds no autograd graph.
    """
    (B, nc, T, H, K), st = _phase_shapes(theta)
    off = torch.empty(B, H, nc, K, device=theta.device, dtype=torch.float32)
    so = off.stride()
    _phase_offset_kernel[(B, H)](
        theta, off, nc,
        st[0], st[1], st[2], st[3], st[4],
        so[0], so[1], so[2], so[3],
        T=T, K=K, num_warps=4,
    )
    return off


def phase_offset_bwd_add(dOff: torch.Tensor, dTheta: torch.Tensor) -> None:
    """dTheta += the cross-chunk term of the increment reparameterisation."""
    (B, nc, T, H, K), st = _phase_shapes(dTheta)
    so = dOff.stride()
    _phase_offset_bwd_kernel[(B, H)](
        dOff, dTheta, nc,
        so[0], 0 if dOff.shape[1] == 1 else so[1], so[2], so[3],
        st[0], st[1], st[2], st[3], st[4],
        T=T, K=K, num_warps=4,
    )


@triton.jit
def _phase_prefix_kernel(
    Th_ptr, Out_ptr, nc,
    s_tb, s_tc, s_tt, s_th, s_tk,
    s_ob, s_oc, s_ot, s_oh, s_ok,
    T: tl.constexpr, K: tl.constexpr, KT: tl.constexpr,
):
    """theta_cum[b, c, t, h, k] = sum_{(c', t') <= (c, t)} incr[b, c', t', h, k].

    The running phase is built ONCE, in one streaming pass, instead of
    `nc` times per head inside every kernel that reads it.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    kt = tl.program_id(2)
    offs_t = tl.arange(0, T)
    offs_k = kt * KT + tl.arange(0, KT)
    run = tl.zeros([KT], dtype=tl.float32)
    tbase = Th_ptr + b * s_tb + h * s_th
    obase = Out_ptr + b * s_ob + h * s_oh
    for c in range(nc):
        tile = tl.load(
            tbase + c * s_tc
            + offs_t[:, None] * s_tt + offs_k[None, :] * s_tk,
        ).to(tl.float32)
        cum = _wrap2pi(tl.cumsum(tile, axis=0) + run[None, :])   # PHASEWRAP
        tl.store(
            obase + c * s_oc
            + offs_t[:, None] * s_ot + offs_k[None, :] * s_ok,
            cum)
        run = _wrap2pi(run + tl.sum(tile, axis=0))


def phase_prefix_build(theta: torch.Tensor) -> torch.Tensor:
    """[B, nc, T, (H,) K] increments -> the same shape, CUMULATED."""
    (B, nc, T, H, K), st = _phase_shapes(theta)
    out = torch.empty_like(theta, dtype=torch.float32)
    (_, _, _, _, _), so = _phase_shapes(out)
    kt = _prefix_kt(B, H, K)
    _phase_prefix_kernel[(B, H, K // kt)](
        theta, out, nc,
        st[0], st[1], st[2], st[3], st[4],
        so[0], so[1], so[2], so[3], so[4],
        T=T, K=K, KT=kt,
        num_warps=int(_env("PHASE_PREFIX_WARPS", "4")),
    )
    return out


def _prefix_kt(B, H, K):
    """The pair-axis tile the prefix kernels launch with.

    `tl.arange(0, KT)` needs a POWER OF TWO, and the grid's z axis is
    `K // KT`, so KT must also divide K.  The start is therefore the
    largest power of two dividing K (`K & -K`), not K itself: at a
    power-of-two K that IS K, and at K = 96 (N = 192) it is 32 where K
    itself would have been rejected by `tl.arange`.
    """
    kt = int(_env("PHASE_PREFIX_KT", "0")) or (K & -K)
    while kt > 16 and (K % kt or B * H * (K // kt) < 264):
        kt //= 2
    return kt


def phase_prefix_bwd(dTheta: torch.Tensor, dOff=None,
                     cross: bool = True, out: torch.Tensor = None) -> None:
    """Raw `dL/dtheta_cum` -> `dL/d increment`, in place unless `out` is
    given.

    `dOff` (optional) receives the per-chunk total; `cross` adds the
    reverse exclusive prefix over chunks.  Replaces
    `phase_offset_bwd_add` on the internal-offset path.  `out` (same
    shape as `dTheta`) receives the result and `dTheta` is left
    untouched -- `phase_law_bwd` uses it because its `dTheta` is an
    autograd `grad_output`.
    """
    (B, nc, T, H, K), st = _phase_shapes(dTheta)
    if out is None:
        out, sq = dTheta, st
    else:
        assert out.shape == dTheta.shape and out.dtype == dTheta.dtype
        sq = _phase_shapes(out)[1]
    if dOff is None:
        dOff = dTheta                       # never written (WRITE_OFF=False)
        so = (0, 0, 0, 0)
    else:
        so = dOff.stride()
        so = (so[0], 0 if dOff.shape[1] == 1 else so[1], so[2], so[3])
    kt = _prefix_kt(B, H, K)
    _phase_prefix_bwd_kernel[(B, H, K // kt)](
        dTheta, dOff, out, nc,
        so[0], so[1], so[2], so[3],
        st[0], st[1], st[2], st[3], st[4],
        sq[0], sq[1], sq[2], sq[3], sq[4],
        T=T, K=K, KT=kt, ACC_CROSS=cross, WRITE_OFF=dOff is not dTheta,
        num_warps=int(_env("PHASE_PREFIX_WARPS", "4")),
    )


# ══════════════════════════════════════════════════════════════════════
# 1b. The phase ACTIVATION
# ══════════════════════════════════════════════════════════════════════
#
# Mamba-3's angle law, which callers of the `m3` route would otherwise
# write as a host chain over a [B, L, H, N/2] fp32 tensor:
#
#     incr = tanh(Angles) * pi * dt.permute(0, 2, 1).unsqueeze(-1)
#     theta = F.pad(-incr, (0, N // 2 - ka))
#
# `m3_phase_increments` is that law in ONE kernel and its adjoint in one
# more.  The zero pad is a STORE MASK rather than a tensor op, so the
# [B, L, H, N/2] buffer is written exactly once.
#
# It is not folded into `_phase_offset_kernel`: the offsets are per CHUNK,
# and the chunk is chosen by `_ssd_dispatch` AFTER the shared-memory
# ladder, so a caller-side offset build could be built for the wrong T.
# The offset build therefore lives INSIDE `_SSDMimoFn`.


@triton.jit
def _phase_activation_kernel(
    Ang_ptr, DT_ptr, Th_ptr, L,
    s_ab, s_al, s_ah, s_ak,
    s_db, s_dh, s_dl,
    s_tb, s_tl, s_th, s_tk,
    H: tl.constexpr, KA: tl.constexpr, K: tl.constexpr,
    BL: tl.constexpr, SIGN: tl.constexpr,
):
    """theta[b, l, h, k] = SIGN * pi * tanh(ang[b, l, h, k]) * dt[b, h, l].

    Grid (B, H, ceil(L / BL)).  `k >= KA` is Mamba-3's unrotated tail:
    it is not loaded and not computed, only STORED as zero, which is the
    `F.pad` this replaces.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_l = tl.program_id(2) * BL + tl.arange(0, BL)
    offs_k = tl.arange(0, K)
    m_l = offs_l < L
    m_k = offs_k < KA
    ang = tl.load(
        Ang_ptr + b * s_ab + h * s_ah
        + offs_l[:, None] * s_al + offs_k[None, :] * s_ak,
        mask=m_l[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    dt = tl.load(DT_ptr + b * s_db + h * s_dh + offs_l * s_dl,
                 mask=m_l, other=0.0).to(tl.float32)
    th = (SIGN * 3.141592653589793) * libdevice.tanh(ang) * dt[:, None]
    tl.store(
        Th_ptr + b * s_tb + h * s_th
        + offs_l[:, None] * s_tl + offs_k[None, :] * s_tk,
        tl.where(m_k[None, :], th, 0.0), mask=m_l[:, None])


@triton.jit
def _phase_activation_bwd_kernel(
    dTh_ptr, Ang_ptr, DT_ptr, dAng_ptr, dDT_ptr, L,
    s_ub, s_ul, s_uh, s_uk,
    s_ab, s_al, s_ah, s_ak,
    s_db, s_dh, s_dl,
    s_gb, s_gl, s_gh, s_gk,
    H: tl.constexpr, KA: tl.constexpr, K: tl.constexpr,
    BL: tl.constexpr, SIGN: tl.constexpr,
):
    """(dAngles, dDT) from dTheta, one pass over the same tile.

    d theta / d ang = SIGN pi dt (1 - tanh^2 ang)   -- elementwise
    d theta / d dt  = SIGN pi tanh(ang)             -- summed over k

    `dDT` is a plain STORE, not an atomic: the grid gives each
    (b, h, L-tile) to exactly one program and dDT's other users (the
    write scale, a_dt) collect their terms through autograd's own add.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_l = tl.program_id(2) * BL + tl.arange(0, BL)
    offs_k = tl.arange(0, K)
    m_l = offs_l < L
    m_k = offs_k < KA
    mm = m_l[:, None] & m_k[None, :]
    dth = tl.load(
        dTh_ptr + b * s_ub + h * s_uh
        + offs_l[:, None] * s_ul + offs_k[None, :] * s_uk,
        mask=mm, other=0.0).to(tl.float32)
    ang = tl.load(
        Ang_ptr + b * s_ab + h * s_ah
        + offs_l[:, None] * s_al + offs_k[None, :] * s_ak,
        mask=mm, other=0.0).to(tl.float32)
    dt = tl.load(DT_ptr + b * s_db + h * s_dh + offs_l * s_dl,
                 mask=m_l, other=0.0).to(tl.float32)
    tanh_a = libdevice.tanh(ang)
    coef = SIGN * 3.141592653589793
    tl.store(
        dAng_ptr + b * s_gb + h * s_gh
        + offs_l[:, None] * s_gl + offs_k[None, :] * s_gk,
        (coef * dt[:, None]) * (1.0 - tanh_a * tanh_a) * dth, mask=mm)
    tl.store(dDT_ptr + b * s_db + h * s_dh + offs_l * s_dl,
             coef * tl.sum(tl.where(mm, dth * tanh_a, 0.0), axis=1),
             mask=m_l)


class _M3PhaseIncrements(torch.autograd.Function):
    """`theta = -pi * tanh(Angles) * dt`, zero-padded to K pairs."""

    @staticmethod
    def forward(ctx, angles, dt, k_out, sign):
        Bsz, L, H, KA = angles.shape
        K = triton.next_power_of_2(max(k_out, 1))
        assert K == k_out, ("m3_phase_increments needs a power-of-two pair "
                            f"count, got {k_out}")
        theta = torch.empty(Bsz, L, H, K, device=angles.device,
                            dtype=torch.float32)
        sa, sd, st = angles.stride(), dt.stride(), theta.stride()
        BL = 64
        _phase_activation_kernel[(Bsz, H, triton.cdiv(L, BL))](
            angles, dt, theta, L,
            sa[0], sa[1], sa[2], sa[3],
            sd[0], sd[1], sd[2],
            st[0], st[1], st[2], st[3],
            H=H, KA=KA, K=K, BL=BL, SIGN=float(sign), num_warps=4)
        ctx.save_for_backward(angles, dt)
        ctx.sign = float(sign)
        ctx.KA, ctx.K, ctx.L = KA, K, L
        ctx.need = (angles.requires_grad, dt.requires_grad)
        return theta

    @staticmethod
    @_no_double_backward("SSD-MIMO phase law")
    def backward(ctx, dTheta):
        angles, dt = ctx.saved_tensors
        if not any(ctx.need):
            return None, None, None, None
        dTheta = dTheta.contiguous()
        Bsz, L, H, _ = angles.shape
        dAng = torch.empty_like(angles, dtype=torch.float32)
        dDT = torch.empty_like(dt, dtype=torch.float32)
        su, sa, sd, sg = (dTheta.stride(), angles.stride(), dt.stride(),
                          dAng.stride())
        BL = 64
        _phase_activation_bwd_kernel[(Bsz, H, triton.cdiv(L, BL))](
            dTheta, angles, dt, dAng, dDT, L,
            su[0], su[1], su[2], su[3],
            sa[0], sa[1], sa[2], sa[3],
            sd[0], sd[1], sd[2],
            sg[0], sg[1], sg[2], sg[3],
            H=H, KA=ctx.KA, K=ctx.K, BL=BL, SIGN=ctx.sign, num_warps=4)
        return (dAng.to(angles.dtype) if ctx.need[0] else None,
                dDT.to(dt.dtype) if ctx.need[1] else None, None, None)


def m3_phase_increments(angles: torch.Tensor, dt: torch.Tensor,
                        n_pairs_out: int, sign: float = -1.0
                        ) -> torch.Tensor:
    """Mamba-3's per-token angle INCREMENT, in one launch.

    `angles` [B, L, H, ka] (raw, pre-tanh), `dt` [B, H, L]; returns
    [B, L, H, n_pairs_out] fp32 with the unrotated tail at zero -- the
    tensor `ssd_mimo_scan(theta=..., theta_increment=True)` takes.

    `sign = -1` gives `theta = -Theta`, because our rotation convention
    is the transpose of Mamba-3's.
    """
    if angles.dim() != 4:
        raise ValueError(f"angles must be [B, L, H, ka], got {tuple(angles.shape)}")
    if dt.dim() != 3:
        raise ValueError(f"dt must be [B, H, L], got {tuple(dt.shape)}")
    if angles.shape[-1] > n_pairs_out:
        raise ValueError(
            f"n_pairs_out={n_pairs_out} < the angle count {angles.shape[-1]}")
    return _M3PhaseIncrements.apply(angles, dt, int(n_pairs_out), float(sign))


# ══════════════════════════════════════════════════════════════════════
# 1c. The FUSED phase law
# ══════════════════════════════════════════════════════════════════════
#
# `phase_law_fwd`: angles + dt + omega -> the CUMULATED phase per kernel
# head in one launch, the increment never materialised.
# `phase_law_bwd` is its adjoint in two launches: the reparameterisation
# adjoint (`_phase_prefix_bwd_kernel`, streaming the raw dL/dtheta_cum the
# scan kernels accumulate into a fresh buffer -- the incoming grad_output
# is never written) and one pass that turns dL/d incr into dAngles
# (summed over the siblings and, for a head-shared angle, over the heads
# -- inside the CTA, no atomics, so it is deterministic), dDT and dOmega.
#
# Same arithmetic and op order as sections 1/1b (`(SIGN pi) tanh(ang)
# dt`, then `* omega`, then the in-block `tl.cumsum` + running total), so
# the result differs from the unfused chain only by the block the running
# sum is folded at.  `SSD_MIMO_PHASE_FUSED=0` restores the chain.


@triton.jit
def _phase_law_fwd_kernel(
    Ang_ptr, DT_ptr, Om_ptr, Th_ptr, ThLo_ptr, L, nblk,
    s_ab, s_al, s_ah, s_ak,
    s_db, s_dh, s_dl,
    s_tb, s_tl, s_th, s_tk,
    H: tl.constexpr, KA: tl.constexpr, K: tl.constexpr,
    KB: tl.constexpr,
    BL: tl.constexpr, SIGN: tl.constexpr, HAS_OMEGA: tl.constexpr,
    STORE_LO: tl.constexpr,
    # The omega layout and the per-head angle.
    # OM_MODE 0: omega[k] (pair); 1: omega[h] (head); 2: omega[h, k]
    # (head_pair, row stride `s_omh`), h = hk mod H.  ANG_K1: the angle has
    # ONE column per head, broadcast over the KA rotated pairs (tanh on the
    # [BL] column only) -- the one-eigenvalue-per-head law.
    ANG_K1: tl.constexpr = False, OM_MODE: tl.constexpr = 0, s_omh=0,
):
    """theta_cum[b, l, hk, k] = sum_{l' <= l} SIGN pi tanh(ang[b, l', h, k])
    dt[b, hk, l'] omega[k], h = hk mod H (kernel heads are r-major).

    Grid (B, H_k).  `k >= KA` is the unrotated tail: stored as zero.
    A head-SHARED angle is the same tensor with `s_ah = 0`.

    STORE_LO: the same tile is ALSO stored, rounded
    to `ThLo_ptr`'s element type (bf16), through the same strides -- the
    16-bit twin the scan's backward reads, produced here instead of by a
    second pass over the fp32 tensor."""
    b = tl.program_id(0)
    hk = tl.program_id(1)
    h = hk % H
    offs_k = tl.arange(0, KB)
    m_k = offs_k < KA
    coef = SIGN * 3.141592653589793
    if HAS_OMEGA:
        if OM_MODE == 1:
            om = tl.load(Om_ptr + h).to(tl.float32)
        elif OM_MODE == 2:
            om = tl.load(Om_ptr + h * s_omh + offs_k, mask=m_k,
                         other=0.0).to(tl.float32)
        else:
            om = tl.load(Om_ptr + offs_k, mask=m_k, other=0.0).to(tl.float32)
    run = tl.zeros([KB], dtype=tl.float32)
    # `KB` is the TILE and `K` the stored width.  They differ only where K
    # is not a power of two (N = 192 -> K = 96 -> KB = 128), and then the
    # store needs the column mask too.
    abase = Ang_ptr + b * s_ab + h * s_ah
    dbase = DT_ptr + b * s_db + hk * s_dh
    tbase = Th_ptr + b * s_tb + hk * s_th
    lobase = ThLo_ptr + b * s_tb + hk * s_th
    for blk in range(nblk):
        offs_l = blk * BL + tl.arange(0, BL)
        m_l = offs_l < L
        if KB != K:
            m_st = m_l[:, None] & (offs_k < K)[None, :]
        else:
            m_st = m_l[:, None]
        if ANG_K1:
            ang = tl.load(abase + offs_l * s_al, mask=m_l, other=0.0
                          ).to(tl.float32)[:, None]
        else:
            ang = tl.load(abase + offs_l[:, None] * s_al
                          + offs_k[None, :] * s_ak,
                          mask=m_l[:, None] & m_k[None, :], other=0.0
                          ).to(tl.float32)
        dt = tl.load(dbase + offs_l * s_dl, mask=m_l, other=0.0
                     ).to(tl.float32)
        th = coef * libdevice.tanh(ang) * dt[:, None]
        th = tl.where(m_k[None, :], th, 0.0)
        if HAS_OMEGA:
            if OM_MODE == 1:
                th = th * om
            else:
                th = th * om[None, :]
        # PHASEWRAP: the running phase is stored reduced mod 2 pi (the
        # model uses only its cos / sin: exact), and the block carry `run`
        # too -- a trained phase reaches 4e5 rad, where fp32 keeps 0.03
        # rad and a bf16 twin none.
        cum = _wrap2pi(tl.cumsum(th, axis=0) + run[None, :])
        tl.store(tbase + offs_l[:, None] * s_tl + offs_k[None, :] * s_tk,
                 cum, mask=m_st)
        if STORE_LO:
            tl.store(lobase + offs_l[:, None] * s_tl + offs_k[None, :] * s_tk,
                     cum.to(ThLo_ptr.dtype.element_ty), mask=m_st)
        if KB < K:
            # Structural real half: same required zero stores, without
            # tanh/prefix/carry arithmetic on those inactive columns.
            tail = tbase + offs_l[:,None]*s_tl + (offs_k[None,:]+KB)*s_tk
            tl.store(tail,tl.zeros((BL,KB),tl.float32),mask=m_l[:,None])
            if STORE_LO:
                tail_lo = lobase + offs_l[:,None]*s_tl + (offs_k[None,:]+KB)*s_tk
                tl.store(tail_lo,tl.zeros((BL,KB),tl.float32),mask=m_l[:,None])
        run = _wrap2pi(run + tl.sum(th, axis=0))


@triton.jit
def _phase_blocksum_kernel(
    G_ptr, Tot_ptr, L,
    s_gb, s_gl, s_gh, s_gk,
    s_tb, s_th, s_tn, s_tk,
    KA: tl.constexpr, KB: tl.constexpr, BL: tl.constexpr,
):
    """Tot[b, hk, blk, k] = sum_{l in block blk} dL/dtheta_cum[b, l, hk, k]
    for k < KA (the rotated pairs; the tail is never read downstream).
    Grid (B, H_k, ceil(L / BL)).  Pass 1 of the fused law adjoint."""
    b = tl.program_id(0)
    hk = tl.program_id(1)
    blk = tl.program_id(2)
    offs_l = blk * BL + tl.arange(0, BL)
    offs_k = tl.arange(0, KB)
    mm = (offs_l < L)[:, None] & (offs_k < KA)[None, :]
    g = tl.load(G_ptr + b * s_gb + hk * s_gh + offs_l[:, None] * s_gl
                + offs_k[None, :] * s_gk, mask=mm, other=0.0).to(tl.float32)
    tl.store(Tot_ptr + b * s_tb + hk * s_th + blk * s_tn + offs_k * s_tk,
             tl.sum(g, axis=0), mask=offs_k < KA)


@triton.jit
def _phase_exclusive_suffix_kernel(Tot, Suf, NB, K: tl.constexpr, KA: tl.constexpr,
                                   BN: tl.constexpr, BK: tl.constexpr):
    # Load the NEXT block, then reverse-scan: exclusive suffix without
    # total-minus-prefix cancellation and without flip/cumsum/copy passes.
    b=tl.program_id(0);h=tl.program_id(1);hk=tl.num_programs(1)
    n=tl.arange(0,BN);k=tl.program_id(2)*BK+tl.arange(0,BK)
    base=(b*hk+h)*NB*K
    x=tl.load(Tot+base+(n[:,None]+1)*K+k[None,:],
              mask=(n[:,None]+1<NB)&(k[None,:]<KA),other=0.0)
    y=tl.cumsum(x,axis=0,reverse=True)
    tl.store(Suf+base+n[:,None]*K+k[None,:],y,
             mask=(n[:,None]<NB)&(k[None,:]<K))


@triton.jit
def _phase_law_bwd_kernel(
    G_ptr, Ang_ptr, DT_ptr, Om_ptr, dAng_ptr, dDT_ptr, dOm_ptr, L,
    HO, NJ, HS,
    s_gb, s_gl, s_gh, s_gk,
    s_ab, s_al, s_ah, s_ak,
    s_db, s_dh, s_dl,
    s_ub, s_ul, s_uh, s_uk,
    s_vb, s_vh, s_vl,
    s_ob, s_ol, s_ok,
    KA: tl.constexpr, K: tl.constexpr, KB: tl.constexpr,
    BL: tl.constexpr,
    SIGN: tl.constexpr, HAS_OMEGA: tl.constexpr,
    # The increment reparameterisation FUSED in.
    # With FUSED_PREFIX, `G_ptr` holds the RAW dL/dtheta_cum and each
    # (head, block) tile becomes dL/d incr here -- the in-block reverse
    # cumsum plus `Suf` (the sum over every LATER block, from
    # `_phase_blocksum_kernel` + one small suffix scan) -- instead of in
    # a separate pass that wrote and re-read a [B, L, H_k, K] fp32
    # tensor.  Masked on k < KA like every other load here, so the
    # unrotated tail is not even read.
    Suf_ptr=None, s_fb=0, s_fh=0, s_fn=0, s_fk=0,
    FUSED_PREFIX: tl.constexpr = False,
    # The omega layout (see the forward).
    # ANG_K1: dAng is the k-sum of the pair tile, stored as ONE column per
    # (token, angle head).  OM_MODE 1 (head): omega[ho] -- the angle head
    # IS the head (the per-head eigenvalue always has per-head angles) --
    # and one dOmega scalar per (b, block, head) at column ho.  OM_MODE 2
    # (head_pair): omega[hk mod H, k] and one dOmega row per (b, block,
    # KERNEL head) at `s_oh` (the host sums the siblings), so a head-shared
    # angle needs no cross-iteration omega state.
    ANG_K1: tl.constexpr = False, OM_MODE: tl.constexpr = 0, s_omh=0,
    s_oh=0, NH=1, HEADPAR: tl.constexpr = False,
):
    """(dAng, dDT, dOmega partials) from g = dL/d incr [B, L, H_k, K].

    Grid (B, ceil(L / BL)).  One CTA owns a token block and walks EVERY
    kernel head: the outer loop is the angle's own head axis (HO = H per
    head, 1 head-shared), the inner loop the NJ kernel heads that share
    that angle (hk = j * HS + ho), so the head reduction of dAng is a
    register accumulation, not an atomic.  dDT is per kernel head and is
    a plain store; dOmega is one partial per CTA.  Opt-in HEADPAR adds a
    head grid axis for independent per-head angles without omega: each CTA
    handles one angle head, retaining its sibling reduction in the same order."""
    b = tl.program_id(0)
    blk = tl.program_id(1)
    offs_l = blk * BL + tl.arange(0, BL)
    # `KB` is the TILE (a power of two), `K` the width of the
    # [.., K] buffers.  EVERY load and store in this body is already
    # masked on `offs_k < KA <= K`, so widening the tile needs no new
    # mask -- only the two tile shapes below move.
    offs_k = tl.arange(0, KB)
    m_l = offs_l < L
    m_k = offs_k < KA
    mm = m_l[:, None] & m_k[None, :]
    coef = SIGN * 3.141592653589793
    if HAS_OMEGA and OM_MODE == 0:
        om = tl.load(Om_ptr + offs_k, mask=m_k, other=0.0).to(tl.float32)
    dom = tl.zeros([KB], dtype=tl.float32)
    for hi in range(1 if HEADPAR else HO):
        ho = tl.program_id(2) if HEADPAR else hi
        if ANG_K1:
            ang = tl.load(Ang_ptr + b * s_ab + ho * s_ah + offs_l * s_al,
                          mask=m_l, other=0.0).to(tl.float32)[:, None]
        else:
            ang = tl.load(Ang_ptr + b * s_ab + ho * s_ah
                          + offs_l[:, None] * s_al + offs_k[None, :] * s_ak,
                          mask=mm, other=0.0).to(tl.float32)
        t = libdevice.tanh(ang)
        if HAS_OMEGA and OM_MODE == 1:
            om = tl.load(Om_ptr + ho).to(tl.float32)
            dom_h = tl.zeros([KB], dtype=tl.float32)
        acc = tl.zeros([BL, KB], dtype=tl.float32)
        for j in range(NJ):
            hk = j * HS + ho
            g = tl.load(G_ptr + b * s_gb + hk * s_gh
                        + offs_l[:, None] * s_gl + offs_k[None, :] * s_gk,
                        mask=mm, other=0.0).to(tl.float32)
            if FUSED_PREFIX:
                g = tl.cumsum(g, axis=0, reverse=True) + tl.load(
                    Suf_ptr + b * s_fb + hk * s_fh + blk * s_fn
                    + offs_k * s_fk, mask=m_k, other=0.0)[None, :]
                g = tl.where(mm, g, 0.0)
            dt = tl.load(DT_ptr + b * s_db + hk * s_dh + offs_l * s_dl,
                         mask=m_l, other=0.0).to(tl.float32)
            if HAS_OMEGA and OM_MODE == 1:
                dom_h += tl.sum(g * (coef * t * dt[:, None]), axis=0)
                gp = g * om
            elif HAS_OMEGA and OM_MODE == 2:
                om2 = tl.load(Om_ptr + (hk % NH) * s_omh + offs_k, mask=m_k,
                              other=0.0).to(tl.float32)
                tl.store(dOm_ptr + b * s_ob + blk * s_ol + hk * s_oh
                         + offs_k * s_ok,
                         tl.sum(g * (coef * t * dt[:, None]), axis=0),
                         mask=m_k)
                gp = g * om2[None, :]
            elif HAS_OMEGA:
                dom += tl.sum(g * (coef * t * dt[:, None]), axis=0)
                gp = g * om[None, :]
            else:
                gp = g
            acc += (coef * dt[:, None]) * (1.0 - t * t) * gp
            tl.store(dDT_ptr + b * s_vb + hk * s_vh + offs_l * s_vl,
                     coef * tl.sum(tl.where(mm, gp * t, 0.0), axis=1),
                     mask=m_l)
        if ANG_K1:
            # acc is 0 on the masked pairs / tokens (gp is), so the k-sum
            # of the tile is the adjoint of the broadcast
            tl.store(dAng_ptr + b * s_ub + ho * s_uh + offs_l * s_ul,
                     tl.sum(acc, axis=1), mask=m_l)
        else:
            tl.store(dAng_ptr + b * s_ub + ho * s_uh
                     + offs_l[:, None] * s_ul + offs_k[None, :] * s_uk,
                     acc, mask=mm)
        if HAS_OMEGA and OM_MODE == 1:
            tl.store(dOm_ptr + b * s_ob + blk * s_ol + ho * s_ok,
                     tl.sum(dom_h, axis=0))
    if HAS_OMEGA and OM_MODE == 0:
        tl.store(dOm_ptr + b * s_ob + blk * s_ol + offs_k * s_ok, dom,
                 mask=m_k)


_PHASE_LAW_BL = 64
# The fused law adjoint's token block and warps (`phase_law_bwd` only; the
# forward keeps _PHASE_LAW_BL): it runs an in-block reverse cumsum per
# (head, block) tile, so its block and CTA width are separate levers.
_PHASE_LAW_BWD_BL = int(_env("PHASE_LAW_BWD_BL", "32"))
_PHASE_LAW_BWD_WARPS = int(_env("PHASE_LAW_BWD_WARPS", "4"))


_OM_MODES = {"pair": 0, "head": 1, "head_pair": 2}


def _head_law(angles, omega, H: int, ka: int, omega_layout: str):
    """(KA, ANG_K1, OM_MODE, s_omh) of a call: "head" passes
    angles [B, L, H, 1] with `ka` rotated pairs and omega [H]; "head_pair"
    passes omega [H, K]; "pair" is the original [K] contract."""
    KA = angles.shape[-1] if ka <= 0 else int(ka)
    ang_k1 = angles.shape[-1] == 1 and KA > 1
    assert angles.shape[-1] in (1, KA), (tuple(angles.shape), KA)
    mode = _OM_MODES[omega_layout]
    if mode == 1:
        assert angles.shape[2] == H and (omega is None
                                         or tuple(omega.shape) == (H,)), (
            tuple(angles.shape), None if omega is None else tuple(omega.shape))
    if mode == 2 and omega is not None:
        assert omega.dim() == 2 and omega.shape[0] == H and omega.stride(1) == 1, (
            tuple(omega.shape), omega.stride())
    s_omh = omega.stride(0) if (mode == 2 and omega is not None) else 0
    return KA, ang_k1, (mode if omega is not None else 0), s_omh


def phase_law_fwd(angles: torch.Tensor, dt: torch.Tensor, omega,
                  H: int, K: int, sign: float, lo_dtype=None, ka: int = 0,
                  omega_layout: str = "pair"):
    """`angles` [B, L, HA, ka] (HA in {1, H}; raw, pre-tanh), `dt`
    [B, H_k, L] (H_k = R * H, r-major), `omega` [K] or None ->
    `(theta, lo)`: the CUMULATED phase [B, L, H_k, K] fp32 and, when
    `lo_dtype` is given, the same tensor rounded to it in the SAME
    launch (else None).

    `omega_layout`: "pair" -- `omega` [K];
    "head_pair" -- `omega` [H, K]; "head" -- one eigenvalue per head,
    `angles` [B, L, H, 1] broadcast over `ka` rotated pairs, `omega` [H]."""
    Bsz, L, HA, _ = angles.shape
    KA, ang_k1, om_mode, s_omh = _head_law(angles, omega, H, ka, omega_layout)
    Hk = dt.shape[1]
    assert HA in (1, H) and Hk % H == 0, (HA, H, Hk)
    # K = N/2 is the PAIR COUNT and need not be a power of two (N = 192 ->
    # K = 96, ...).  The tile is padded to `next_power_of_2(K)` and the
    # store masked; the tensor keeps its K columns.
    assert KA <= K, (KA, K)
    from . import levers as _lv
    _grp = _lv.group_of(dt.shape[1] // H, 1)
    narrow = _lv.on("phase_law_fwd_active_k", _grp,
                    guard=(K == 2*KA and KA == triton.next_power_of_2(KA)))
    narrow = narrow and K == 2*KA and KA == triton.next_power_of_2(KA)
    KB = KA if narrow else triton.next_power_of_2(K)
    theta = torch.empty(Bsz, L, Hk, K, device=angles.device,
                        dtype=torch.float32)
    lo = (torch.empty_like(theta, dtype=lo_dtype) if lo_dtype is not None
          else None)
    sa, sd, st = angles.stride(), dt.stride(), theta.stride()
    BL = _PHASE_LAW_BL
    _phase_law_fwd_kernel[(Bsz, Hk)](
        angles, dt, omega if omega is not None else dt, theta,
        lo if lo is not None else theta, L,
        triton.cdiv(L, BL),
        sa[0], sa[1], 0 if HA == 1 else sa[2], sa[3],
        sd[0], sd[1], sd[2],
        st[0], st[1], st[2], st[3],
        H=H, KA=KA, K=K, KB=KB, BL=BL, SIGN=float(sign),
        HAS_OMEGA=omega is not None, STORE_LO=lo is not None,
        ANG_K1=ang_k1, OM_MODE=om_mode, s_omh=s_omh, num_warps=4)
    return theta, lo


def _prefix_chunk(L: int) -> int:
    for T in (64, 32, 16, 8, 4, 2):
        if L % T == 0:
            return T
    return 1


def phase_law_bwd(dcum: torch.Tensor, angles: torch.Tensor,
                  dt: torch.Tensor, omega, H: int, K: int, sign: float,
                  ka: int = 0, omega_layout: str = "pair"):
    """dL/dtheta_cum [B, L, H_k, K] -> (dAngles [B, L, HA, ka] fp32,
    dDT [B, H_k, L] fp32, dOmega [K] fp32 or None).  `dcum` is NOT
    written: it is an autograd `grad_output`, and under
    `torch.autograd.grad(..., grad_outputs=g)` it IS the caller's `g`
    (and what any tensor hook on `theta` observes).  Default (fused):
    dL/d incr is formed per tile inside the law
    kernel from `dcum` plus per-block suffix totals, so no [B, L, H_k, K]
    buffer is written; SSD_MIMO_PHASE_LAW_BWD_FUSED=0 selects the fresh
    buffer + `phase_prefix_bwd` pass.

    `omega_layout`: dOmega has omega's shape ([K], [H] or [H, K]); under
    "head" dAngles is [B, L, H, 1] (the k-sum, in-kernel)."""
    Bsz, L, HA, _ = angles.shape
    KA, ang_k1, om_mode, s_omh = _head_law(angles, omega, H, ka, omega_layout)
    Hk = dt.shape[1]
    fused = _env("PHASE_LAW_BWD_FUSED", "1") == "1"
    BL = _PHASE_LAW_BWD_BL if fused else _PHASE_LAW_BL
    nblk = triton.cdiv(L, BL)
    # Opt-in: only KA pairs contribute to the phase law. Keeping the
    # storage strides at K while narrowing the register tile removes the
    # inactive columns without reducing across heads or dropping dDT.
    # All k reductions retain the same masked-zero mathematical terms.
    from . import levers as _lv
    _grp = _lv.group_of(Hk // H, 1)
    active_k = _lv.on("phase_law_active_k", _grp)
    KB = triton.next_power_of_2(KA if active_k else K)
    if fused:
        # No [B, L, H_k, K] dL/d incr buffer.  Pass
        # 1 reduces each (head, token block) to its total, a suffix scan
        # over the [B, H_k, nblk, K] totals gives every block's
        # later-block sum, and the law kernel below forms dL/d incr per
        # tile (`FUSED_PREFIX`).  `dcum` is read, never written.
        g = dcum if dcum.stride(-1) == 1 else dcum.contiguous()
        # zeros only for the unread k >= KA columns; every k < KA entry is
        # stored by the kernel
        tot = torch.zeros(Bsz, Hk, nblk, K, device=dcum.device,
                          dtype=torch.float32)
        sg0, st0 = g.stride(), tot.stride()
        _phase_blocksum_kernel[(Bsz, Hk, nblk)](
            g, tot, L, sg0[0], sg0[1], sg0[2], sg0[3],
            st0[0], st0[1], st0[2], st0[3],
            KA=KA, KB=KB, BL=BL, num_warps=4)
        # exclusive suffix over blocks: the sum of every LATER block's total,
        # as the inclusive suffix SHIFTED by one block (not `incl - tot`,
        # which cancels when a block's total dominates what follows it)
        if _lv.on("phase_law_suffix", _grp, guard=nblk <= 1024) and nblk <= 1024:
            suf = torch.empty_like(tot)
            _phase_exclusive_suffix_kernel[(Bsz,Hk,triton.cdiv(K,16))](
                tot,suf,nblk,K=K,KA=KA,BN=triton.next_power_of_2(nblk),BK=16,
                num_warps=4)
            del tot
        else:
            incl = tot.flip(2).cumsum(2).flip(2)
            suf = torch.zeros_like(tot)
            suf[:, :, :-1] = incl[:, :, 1:]
            del tot, incl
    else:
        src = dcum.contiguous()
        g = torch.empty_like(src)
        T = _prefix_chunk(L)
        phase_prefix_bwd(src.view(Bsz, L // T, T, Hk, K), None, cross=True,
                         out=g.view(Bsz, L // T, T, Hk, K))
        del src
        suf = None
    dAng = torch.empty(Bsz, L, HA, angles.shape[-1], device=angles.device,
                       dtype=torch.float32)
    dDT = torch.empty(Bsz, Hk, L, device=dt.device, dtype=torch.float32)
    if omega is None:
        dOm = None
    elif om_mode == 2:
        # one row per (b, block, KERNEL head); every k < KA is stored
        dOm = torch.zeros(Bsz, nblk, Hk, K, device=dt.device,
                          dtype=torch.float32)
    else:
        dOm = torch.zeros(Bsz, nblk, H if om_mode == 1 else K,
                          device=dt.device, dtype=torch.float32)
    sg, sa, sd, su, sv = (g.stride(), angles.stride(), dt.stride(),
                          dAng.stride(), dDT.stride())
    # (b, block, column) strides of the dOmega partials; head_pair's
    # [B, nblk, Hk, K] passes its KERNEL-head stride separately (`s_oh`)
    so = ((dOm.stride(0), dOm.stride(1), dOm.stride(-1)) if omega is not None
          else (0, 0, 0))
    per_head = HA == H
    # Only independent, per-head raw angles without shared omega reduction.
    _hp_ok = HA == H and omega is None
    headpar = _lv.on("phase_law_headpar", _grp, guard=_hp_ok) and _hp_ok
    _phase_law_bwd_kernel[(Bsz, nblk, HA if headpar else 1)](
        g, angles, dt, omega if omega is not None else dt,
        dAng, dDT, dOm if dOm is not None else dDT, L,
        HA, (Hk // H) if per_head else Hk, H if per_head else 1,
        sg[0], sg[1], sg[2], sg[3],
        sa[0], sa[1], 0 if HA == 1 else sa[2], sa[3],
        sd[0], sd[1], sd[2],
        su[0], su[1], su[2], su[3],
        sv[0], sv[1], sv[2],
        so[0], so[1], so[2],
        KA=KA, K=K, KB=KB, BL=BL, SIGN=float(sign),
        HAS_OMEGA=omega is not None,
        **(dict(Suf_ptr=suf, s_fb=suf.stride(0), s_fh=suf.stride(1),
                s_fn=suf.stride(2), s_fk=suf.stride(3), FUSED_PREFIX=True)
           if suf is not None else {}),
        ANG_K1=ang_k1, OM_MODE=om_mode, s_omh=s_omh,
        s_oh=dOm.stride(2) if om_mode == 2 else 0, NH=H, HEADPAR=headpar,
        num_warps=_PHASE_LAW_BWD_WARPS if fused else 4)
    if dOm is not None and om_mode == 2:
        # (b, block) partials, then the R siblings of each head (r-major)
        return dAng, dDT, dOm.sum(dim=(0, 1)).view(Hk // H, H, K).sum(0)
    return dAng, dDT, (dOm.sum(dim=(0, 1)) if dOm is not None else None)


class _PhaseLawFn(torch.autograd.Function):
    """The cumulated phase from (angles, dt, omega) in one launch, and
    its adjoint in two.  Saves only the three inputs."""

    @staticmethod
    def forward(ctx, angles, dt, omega, H, K, sign, lo_dtype, ka=0,
                omega_layout="pair"):
        theta, lo = phase_law_fwd(angles, dt, omega, H, K, sign, lo_dtype,
                                  ka, omega_layout)
        ctx.ka, ctx.omega_layout = ka, omega_layout
        ctx.save_for_backward(angles, dt,
                              omega if omega is not None
                              else angles.new_empty(0))
        ctx.has_omega = omega is not None
        ctx.H, ctx.K, ctx.sign = H, K, float(sign)
        ctx.need = (angles.requires_grad, dt.requires_grad,
                    omega is not None and omega.requires_grad)
        if lo is None:
            return theta
        # The twin is DATA only: the scan's `theta` input stays the fp32
        # tensor, so dL/dtheta_cum reaches `backward` below in fp32 (the
        # engine casts a gradient to its input's dtype at every Function
        # boundary -- a bf16 `theta` would hand the reverse cumsum a
        # bf16 dL/dtheta_cum).
        ctx.mark_non_differentiable(lo)
        return theta, lo

    @staticmethod
    @_no_double_backward("SSD-MIMO phase law")
    def backward(ctx, dcum, _dlo=None):
        if not any(ctx.need):
            return None, None, None, None, None, None, None, None, None
        angles, dt, om = ctx.saved_tensors
        omega = om if ctx.has_omega else None
        dAng, dDT, dOm = phase_law_bwd(dcum, angles, dt, omega, ctx.H,
                                       ctx.K, ctx.sign, ctx.ka,
                                       ctx.omega_layout)
        return (dAng.to(angles.dtype) if ctx.need[0] else None,
                dDT.to(dt.dtype) if ctx.need[1] else None,
                dOm if ctx.need[2] else None,
                None, None, None, None, None, None)


# ── the compiled route: a registered custom-op pair ──────────────────────
# Under `torch.compile` the law is ONE opaque op each way (fake impls give
# the shapes; the launchers own all arithmetic), so the graph keeps no
# Triton wrapper: Inductor's post-grad decomposition of raw Triton
# launches fails on a recompile at a second sequence length (dynamic
# shapes).  Eager uses `_PhaseLawFn` directly (no dispatcher cost).
@torch.library.custom_op("orbit_ssm::phase_law_fwd", mutates_args=())
def _phase_law_fwd_op(angles: torch.Tensor, dt: torch.Tensor,
                      omega: torch.Tensor | None, H: int, K: int,
                      sign: float, lo_dtype: torch.dtype | None,
                      ka: int = 0, omega_layout: str = "pair"
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    theta, lo = phase_law_fwd(angles, dt, omega, H, K, sign, lo_dtype, ka,
                              omega_layout)
    return theta, (lo if lo is not None else theta.new_empty(0))


@_phase_law_fwd_op.register_fake
def _phase_law_fwd_fake(angles, dt, omega, H, K, sign, lo_dtype, ka=0,
                        omega_layout="pair"):
    Bsz, L = angles.shape[0], angles.shape[1]
    theta = angles.new_empty((Bsz, L, dt.shape[1], K), dtype=torch.float32)
    lo = (theta.new_empty(theta.shape, dtype=lo_dtype)
          if lo_dtype is not None else theta.new_empty(0))
    return theta, lo


@torch.library.custom_op("orbit_ssm::phase_law_bwd", mutates_args=())
def _phase_law_bwd_op(dcum: torch.Tensor, angles: torch.Tensor,
                      dt: torch.Tensor, omega: torch.Tensor | None, H: int,
                      K: int, sign: float, ka: int = 0,
                      omega_layout: str = "pair"
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dAng, dDT, dOm = phase_law_bwd(dcum, angles, dt, omega, H, K, sign, ka,
                                   omega_layout)
    return dAng, dDT, (dOm if dOm is not None else dAng.new_empty(0))


@_phase_law_bwd_op.register_fake
def _phase_law_bwd_fake(dcum, angles, dt, omega, H, K, sign, ka=0,
                        omega_layout="pair"):
    dAng = angles.new_empty(angles.shape, dtype=torch.float32)
    dDT = dt.new_empty(dt.shape, dtype=torch.float32)
    dOm = (dt.new_empty(tuple(omega.shape), dtype=torch.float32)
           if omega is not None else dt.new_empty(0, dtype=torch.float32))
    return dAng, dDT, dOm


def _phase_law_setup(ctx, inputs, output):
    angles, dt, omega, H, K, sign, _lo_dtype, ka, omega_layout = inputs
    ctx.ka, ctx.omega_layout = ka, omega_layout
    ctx.save_for_backward(angles, dt, omega if omega is not None
                          else angles.new_empty(0))
    ctx.has_omega = omega is not None
    ctx.H, ctx.K, ctx.sign = H, K, float(sign)
    ctx.mark_non_differentiable(output[1])


@_no_double_backward("SSD-MIMO phase law")
def _phase_law_backward(ctx, dcum, _dlo):
    angles, dt, om = ctx.saved_tensors
    omega = om if ctx.has_omega else None
    need = ctx.needs_input_grad
    if not (need[0] or need[1] or need[2]):
        return None, None, None, None, None, None, None, None, None
    dAng, dDT, dOm = _phase_law_bwd_op(dcum, angles, dt, omega, ctx.H,
                                       ctx.K, ctx.sign, ctx.ka,
                                       ctx.omega_layout)
    return (dAng.to(angles.dtype) if need[0] else None,
            dDT.to(dt.dtype) if need[1] else None,
            dOm if (ctx.has_omega and need[2]) else None,
            None, None, None, None, None, None)


_phase_law_fwd_op.register_autograd(_phase_law_backward,
                                    setup_context=_phase_law_setup)


def phase_law(angles: torch.Tensor, dt: torch.Tensor, omega,
              H: int, K: int, sign: float = -1.0, lo_dtype=None, *,
              ka: int = 0, omega_layout: str = "pair"):
    """Mamba-3's angle law, cumulated, in one launch (see the section
    note).  `angles` [B, L, HA, ka] with HA in {1, H}; `dt` [B, R*H, L];
    `omega` [K] (zero beyond ka) or None.  Returns [B, L, R*H, K] fp32.

    `lo_dtype`: also return the phase rounded to
    that dtype -- `(theta, lo)` -- stored by the same launch.  `lo` is
    not differentiable; `ssd_mimo_scan(theta=theta, theta_lo=lo)` saves
    it for the backward in place of its own `theta.to(bf16)` pass.

    `omega_layout`: "pair" (default) -- `omega`
    [K]; "head_pair" -- `omega` [H, K], one frequency per (head, pair);
    "head" -- ONE complex eigenvalue per head: `angles` [B, L, H, 1] (one
    angle feature per head, broadcast in-kernel over the `ka` rotated
    pairs) and `omega` [H], so every rotated pair of a head carries the
    SAME phase; the adjoint returns dAngles [B, L, H, 1].  dOmega has
    omega's shape in every layout."""
    if angles.dim() != 4:
        raise ValueError(f"angles must be [B, L, HA, ka], got {tuple(angles.shape)}")
    if dt.dim() != 3:
        raise ValueError(f"dt must be [B, H_k, L], got {tuple(dt.shape)}")
    if omega_layout not in _OM_MODES:
        raise ValueError(f"omega_layout must be one of {tuple(_OM_MODES)}, "
                         f"got {omega_layout!r}")
    if omega_layout == "head":
        if tuple(angles.shape[2:]) != (H, 1) or not 1 <= ka <= K:
            raise ValueError(
                "omega_layout='head' takes angles [B, L, H, 1] and 1 <= ka <= K, "
                f"got {tuple(angles.shape)}, ka={ka}, K={K}")
        if omega is not None and tuple(omega.shape) != (H,):
            raise ValueError(f"omega_layout='head': omega must be [{H}], got "
                             f"{tuple(omega.shape)}")
    else:
        if ka not in (0, angles.shape[-1]):
            raise ValueError(f"ka={ka} != the angle count {angles.shape[-1]}")
        if angles.shape[-1] > K:
            raise ValueError(f"K={K} < the angle count {angles.shape[-1]}")
        want = (K,) if omega_layout == "pair" else (H, K)
        if omega is not None and tuple(omega.shape) != want:
            raise ValueError(f"omega_layout={omega_layout!r}: omega must be "
                             f"{list(want)}, got {tuple(omega.shape)}")
        if omega is not None and omega_layout == "head_pair":
            omega = omega.contiguous()
    ka = int(ka) if omega_layout == "head" else 0
    if lo_dtype is not None and lo_dtype not in (torch.bfloat16,
                                                 torch.float16):
        raise ValueError(f"lo_dtype must be a 16-bit float, got {lo_dtype}")
    if torch.compiler.is_compiling():
        theta, lo = _phase_law_fwd_op(angles, dt, omega, int(H), int(K),
                                      float(sign), lo_dtype, ka,
                                      omega_layout)
        return theta if lo_dtype is None else (theta, lo)
    return _PhaseLawFn.apply(angles, dt, omega, int(H), int(K), float(sign),
                             lo_dtype, ka, omega_layout)


# ══════════════════════════════════════════════════════════════════════
# 2. The two-tap trapezoid's adjoint
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def _trap_prologue_kernel(
    DW_ptr, Lam_ptr, Dot_ptr,
    Own_ptr, W_ptr, OD_ptr, dSdot_ptr,
    H, L,
    s_wb, s_wh, s_wl,
    s_lb, s_lh, s_ll,
    s_qb, s_qh, s_ql,
    BL: tl.constexpr,
):
    """n_t, w_t, n_t d_t and a zeroed dSdot, in one pass.

    `DW_ptr` is Mamba-3's RAW dt in this mode.  The `t+1` taps are
    MASKED at the sequence end, which is the zero the host's
    `F.pad(x[..., 1:], (0, 1))` put there; the L padding rides the same
    arithmetic because dt is zero in it.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.program_id(2) * BL + tl.arange(0, BL)
    m = offs < L
    m1 = (offs + 1) < L
    dt = tl.load(DW_ptr + b * s_wb + h * s_wh + offs * s_wl,
                 mask=m, other=0.0).to(tl.float32)
    lam = tl.load(Lam_ptr + b * s_lb + h * s_lh + offs * s_ll,
                  mask=m, other=0.0).to(tl.float32)
    dt1 = tl.load(DW_ptr + b * s_wb + h * s_wh + (offs + 1) * s_wl,
                  mask=m1, other=0.0).to(tl.float32)
    lam1 = tl.load(Lam_ptr + b * s_lb + h * s_lh + (offs + 1) * s_ll,
                   mask=m1, other=0.0).to(tl.float32)
    dot = tl.load(Dot_ptr + b * s_qb + h * s_qh + offs * s_ql,
                  mask=m, other=0.0).to(tl.float32)
    n = dt1 * (1.0 - lam1)
    own = lam * dt
    flat = (b * H + h) * L + offs
    # TRAPFOLD: the buffers carry the OWN tap -- the diagonal's weight --
    # not the next-token tap.
    tl.store(Own_ptr + flat, own, mask=m)
    tl.store(W_ptr + flat, lam * dt + n, mask=m)
    tl.store(OD_ptr + flat, own * dot, mask=m)
    tl.store(dSdot_ptr + flat, 0.0, mask=m)


@triton.jit
def _trap_prologue_or_kernel(
    DW_ptr, Lam_ptr, Dot_ptr,
    Own_ptr, W_ptr, OD_ptr, dSdot_ptr,
    H, L,
    s_wb, s_wh, s_wl,
    s_lb, s_lh, s_ll,
    s_qb, s_qh, s_ql, s_qo, s_qr,
    s_nb, s_nh, s_nl, s_no, s_nr,
    BL: tl.constexpr, O: tl.constexpr, R: tl.constexpr,
):
    """The same pass at R, O >= 1.

    `n_t` and `w_t` are per (kernel head, token) -- the write scale owns
    no (o, r) axis ($\\Delta_t$ is per (h, r, t) and the sibling `r` IS the
    kernel head here) -- so those two
    stores are the R = O = 1 stores verbatim.  What grows is the PRODUCT
    `n_t d_t[o, r]` Kc loads and the `dSdot[o, r]` accumulator it
    atomically reduces into: both carry the (O, R) tail of the saved
    `Dot`.  A separate kernel rather than a constexpr branch, so the
    R = O = 1 launch above is unchanged.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.program_id(2) * BL + tl.arange(0, BL)
    m = offs < L
    m1 = (offs + 1) < L
    dt = tl.load(DW_ptr + b * s_wb + h * s_wh + offs * s_wl,
                 mask=m, other=0.0).to(tl.float32)
    lam = tl.load(Lam_ptr + b * s_lb + h * s_lh + offs * s_ll,
                  mask=m, other=0.0).to(tl.float32)
    dt1 = tl.load(DW_ptr + b * s_wb + h * s_wh + (offs + 1) * s_wl,
                  mask=m1, other=0.0).to(tl.float32)
    lam1 = tl.load(Lam_ptr + b * s_lb + h * s_lh + (offs + 1) * s_ll,
                   mask=m1, other=0.0).to(tl.float32)
    n = dt1 * (1.0 - lam1)
    own = lam * dt
    flat = (b * H + h) * L + offs
    tl.store(Own_ptr + flat, own, mask=m)          # TRAPFOLD: own tap
    tl.store(W_ptr + flat, lam * dt + n, mask=m)
    q_base = Dot_ptr + b * s_qb + h * s_qh + offs * s_ql
    n_base = OD_ptr + b * s_nb + h * s_nh + offs * s_nl
    s_base = dSdot_ptr + b * s_nb + h * s_nh + offs * s_nl
    for o in tl.static_range(O):
        for r in tl.static_range(R):
            dot = tl.load(q_base + o * s_qo + r * s_qr,
                          mask=m, other=0.0).to(tl.float32)
            tl.store(n_base + o * s_no + r * s_nr, own * dot, mask=m)
            tl.store(s_base + o * s_no + r * s_nr, 0.0, mask=m)


def trap_prologue(dw, trap_lam, trap_dot):
    """(own_t, w_t, own_t d_t, a zeroed dSdot) -- one launch for six torch ops.

    TRAPFOLD (fused_fwd_kernel `_TRAPFOLD_NOTE`): the first and third
    buffers carry the OWN tap own_t = lam_t dt_t (the forward's Score
    diagonal weight) and own_t d_t (Kc's diagonal dx term), not the
    next-token tap n_t.

    Returns four [B, H, L] fp32 CONTIGUOUS buffers; `w` is the collapsed
    write scale the scan kernels take as their `dw`.  `dSdot` is zeroed
    here rather than by a `zeros_like` memset, because Kc accumulates
    into it atomically and the zero-fill is free in a pass that is
    already writing three vectors of the same shape.
    """
    Bsz, H, L = dw.shape
    sw, sl, sq = dw.stride(), trap_lam.stride(), trap_dot.stride()
    BL = 256
    if trap_dot.dim() == 5:
        # The saved dot carries an (O, R) tail, so `nd` and `dSdot` do too.  `nxt` / `w` stay [B, H, L].
        O, R = trap_dot.shape[3], trap_dot.shape[4]
        out = torch.empty(2, Bsz, H, L, device=dw.device, dtype=torch.float32)
        nxt, w = out[0], out[1]
        nd = torch.empty(Bsz, H, L, O, R, device=dw.device,
                         dtype=torch.float32)
        ds = torch.empty_like(nd)
        sn = nd.stride()
        _trap_prologue_or_kernel[(Bsz, H, triton.cdiv(L, BL))](
            dw, trap_lam, trap_dot, nxt, w, nd, ds,
            H, L,
            sw[0], sw[1], sw[2],
            sl[0], sl[1], sl[2],
            sq[0], sq[1], sq[2], sq[3], sq[4],
            sn[0], sn[1], sn[2], sn[3], sn[4],
            BL=BL, O=O, R=R, num_warps=4,
        )
        return nxt, w, nd, ds
    out = torch.empty(4, Bsz, H, L, device=dw.device, dtype=torch.float32)
    nxt, w, nd, ds = out[0], out[1], out[2], out[3]
    _trap_prologue_kernel[(Bsz, H, triton.cdiv(L, BL))](
        dw, trap_lam, trap_dot, nxt, w, nd, ds,
        H, L,
        sw[0], sw[1], sw[2],
        sl[0], sl[1], sl[2],
        sq[0], sq[1], sq[2],
        BL=BL, num_warps=4,
    )
    return nxt, w, nd, ds


# ── `_trap_adjoint_kernel` schedule mask and launch knobs ──────────────
#
# Bit 1 (`AC`) of `SSD_MIMO_TRAP_SCHED` issues the head block's four dots
# together and applies every non-wgmma accumulator write after the last of
# them.  It is a static no-op (byte for byte `SCHED = 0`) and ships OFF.
# The ptxas wgmma serialization diagnostic (`C7515`) comes from the body
# holding FOUR [T, N] fp32 tiles (`Bt`, `Ct`, `dB_add`, `dC_add`): at
# N = 128 and 4 warps that is 256 registers of tile alone.  The real lever
# is the warp count: at 8 warps each tile is 32 registers instead of 64.
_TRAP_SCHED_DEFAULT = 0


def trap_sched() -> int:
    return int(_env("TRAP_SCHED", str(_TRAP_SCHED_DEFAULT)))


_TRAP_WARPS_DEFAULT = 8


def trap_warps() -> int:
    return int(_env("TRAP_WARPS", str(_TRAP_WARPS_DEFAULT)))


# The GEN_OR branch's own warp count.  There is no MMA in this branch, so
# the warp count only partitions the tile (occupancy / register pressure).
_TRAP_WARPS_OR_DEFAULT = 8


def trap_warps_or() -> int:
    return int(_env("TRAP_WARPS_OR", str(_TRAP_WARPS_OR_DEFAULT)))


# The HAS_THH-only route's dGamma / dBeta / dThH atomics are contended by
# every (b, c) CTA -- `nc * Bsz` deep on the un-slotted [H, N] / [H, N//2]
# buffers.  Each CTA instead writes `bslot = c % BSLOT`, giving
# `B * ceil(nc / BSLOT)`-deep contention, and the host sums the BSLOT axis
# once.  Default is capped so the slot planes stay small.
_TRAP_BSLOTS_DEFAULT = 32


def trap_bslots(nc: int) -> int:
    n = int(_env("TRAP_BSLOTS", str(_TRAP_BSLOTS_DEFAULT)))
    n = n if n > 0 else nc
    return max(1, min(n, nc))


# ── The general-rank nests are RUNTIME loops ──
#
# `_trap_adjoint_kernel`'s general-(O, R) branches (GEN_VEC below, and the
# PAIR_HALF general-rank body above it) must not unroll R, O and GB all
# statically: GB = Gb = R at the production sibling gate, so the emitted
# tile bodies would grow like R*GB*(O + R*O), and the TritonGPU layout
# pass is superlinear on top of that (cold compile of minutes to hours at
# R = O = 4..8).
#
# The OUTER axes (`r` in pass 1, `o` in pass 2, the group walk `g`, and
# the (oo, rr) M-sum) enter their bodies ONLY as address arithmetic and a
# `r == 0` predicate, so they are `tl.range(..., loop_unroll_factor=1)`.
# The INNER axis (`o` in pass 1, `r` in pass 2) is where the MMAs and
# their shared operands live, and it stays `tl.static_range` -- it is also
# what the K-merge (`THH_W & 4`) builds its `tl.join` operand out of,
# which is a FRONTEND tuple and needs the Python-level unroll.  Emitted
# tile bodies go from R*GB*(O + R*O) to O + R.
#
# The inner loops are renamed `oi` / `ri` / `om` / `rm` wherever a RUNTIME
# loop of the same name is still in scope: a `tl.range` target OUTLIVES
# its loop, and rebinding it to a `tl.static_range` constexpr is a
# loop-carried type change the frontend refuses.
#
# There is no unroll knob: `loop_unroll_factor=R` (unrolling in the TTIR
# pass instead of the frontend) crashes `TritonGPUCoalesce`.


@triton.jit
def _trap_adjoint_kernel(
    B_ptr, C_ptr, Beta_ptr, Gamma_ptr,
    DW_ptr, Lam_ptr, Dot_ptr,
    Own_ptr, A_ptr, S_ptr,
    dDW_ptr, dLam_ptr,
    dB_ptr, dC_ptr, dBeta_ptr, dGamma_ptr,
    # ── the static READ offset theta_h [H, N//2]: the dot
    # is d_t = (B+beta_h) . R(nu_h)^T (C+gamma_h), so its adjoint needs
    # the per-head rotation of the read tile and returns d nu_h too.
    ThH_ptr, dThH_ptr,
    H, L,
    s_bb, s_bc, s_bt, s_bn,
    s_cb, s_cc, s_ct, s_cn,
    s_dbb, s_dbc, s_dbt, s_dbn,
    s_dcb, s_dcc, s_dct, s_dcn,
    s_wb, s_wh, s_wl,
    s_lb, s_lh, s_ll,
    s_qb, s_qh, s_ql,
    # ── grouped B / C: the read map is per KERNEL HEAD when
    # the static read offset theta_h is host-folded into it (Gc = Hk), and
    # the write map is per group under a grouped gate.  `s_?h` is the
    # group stride (0 when head-shared) and `hpg_?` the heads per group.
    s_bh, s_ch, s_dbh, s_dch, hpg_b, hpg_c,
    # ── the (O, R) tail ─────────────────────────────────────────────
    # `d_t` is the O x R matrix C~_t^T B~_t, so B / C / dB / dC carry a
    # trailing R / O axis and the saved dot and its adjoint `dSdot` carry
    # BOTH.  All eleven are 0 at R = O = 1 and no R = O = 1 branch reads
    # any of them.
    s_br, s_co, s_dbr, s_dco,
    s_qo, s_qr,
    s_sb, s_sh, s_sl, s_so, s_sr,
    # ── THH slot buffers: dBeta/dGamma/dThH are contended
    # by every (b, c) CTA (nc * Bsz deep on the un-vectorised path).
    # `bslot = c % BSLOT` gives each CTA a private plane; the host sums
    # the BSLOT axis after the launch.  `s_?s` is the slot stride (0 and
    # BSLOT=1 recovers the direct-write behaviour).
    s_dgs, s_dbes, s_dths, BSLOT: tl.constexpr,
    T: tl.constexpr, N: tl.constexpr, HB: tl.constexpr,
    # TS ("T-sub") is the actual per-CTA t-extent; a
    # third grid axis (`tl.program_id(2)`) tiles a chunk's T positions
    # into T // TS pieces when the HAS_THH vectorised path's six
    # loop-invariant [T, N] fp32 operand tiles (Bt/Ct/Btx/Ctx/BtCt/BtCtx,
    # kept live for the whole HB loop as MMA operands) overflow shared
    # memory at a wide N tile (e.g. N=192's 128-wide tile).  This kernel
    # does no scan WITHIN a chunk (A/S/Nxt/dot are read, not recurred), so
    # splitting T is exact -- every `l` position is independent.  TS = T
    # is the unsplit behaviour.
    TS: tl.constexpr,
    PREC: tl.constexpr,
    HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr,
    NEED_DLAM: tl.constexpr,
    # ── the LIVE-RANGE / accumulator schedule (`SSD_MIMO_TRAP_SCHED`)
    SCHED: tl.constexpr = 0,
    B_GROUPED: tl.constexpr = False, C_GROUPED: tl.constexpr = False,
    HAS_THH: tl.constexpr = False,
    # ── the write / read ranks.  O = R = 1 compiles none of the general-rank
    # branches.
    O: tl.constexpr = 1, R: tl.constexpr = 1,
    # The number of B WRITE groups (1 when the write map is
    # head-shared).  Only the head-vectorised GEN_VEC branch reads it --
    # it walks `hpg_b` heads per group instead of masking a full-H block.
    GB: tl.constexpr = 1,
    # Which MERGE the GEN_VEC branch's theta_h fold uses (a BITMASK,
    # `SSD_MIMO_TRAP_THH_MERGE`; see the comment at the head of the
    # branch).  1 = the `ddT`-side N-merge, 2 = the `dd`-side N-merge,
    # 4 = the (h, o) K-merge.  0 is the one-MMA-per-term form.  All are
    # ALGEBRAICALLY the same sum; 1 and 2 are bit-identical to 0 (same K,
    # same operand values, same reduction order), 4 re-associates the
    # o-sum inside one MMA accumulator.
    THH_W: tl.constexpr = 0,
    # The rotate-half pair layout.  Only the
    # HAS_THH branch is layout-sensitive (the static read offset is the
    # kernel's only pair rotation), so PAIR_HALF with no theta_h falls
    # through to the unchanged head-blocked path.
    PAIR_HALF: tl.constexpr = False,
):
    """Everything the trapezoid still owed the host, chunk-parallel.

    d_t = (B_t + beta_h) . (C_t + gamma_h) in the UNROTATED frame (the
    rotation is orthogonal and cancels at equal t; a host-folded read
    offset lives inside the per-head C tile and needs no term of its
    own).  Head-shared B/C take the blocked (HB heads at a time) path,
    which is the original kernel bit for bit; a grouped map takes the
    per-head path below, whose only difference is WHICH tile it reads
    and where its dB/dC land.
    """
    SCHED_AC: tl.constexpr = (SCHED.value & 1) != 0
    # HAS_THH alone (no B/C grouping) takes the vectorised HB path below;
    # GROUPED is for the B_GROUPED / C_GROUPED maps (which may still carry theta_h and use
    # the per-head fold inside this branch).
    GROUPED: tl.constexpr = B_GROUPED or C_GROUPED
    b = tl.program_id(0)
    c = tl.program_id(1)
    t0 = tl.program_id(2) * TS
    offs_t = t0 + tl.arange(0, TS)
    offs_n = tl.arange(0, N)
    l = c * T + offs_t                                    # [TS]

    # ══════════════════════════════════════════════════════════════════
    # The adjoint at R, O >= 1.
    #
    # The forward's correction is  y_t[o, g] -= n_t sum_r d_t[o, r] x_t[g, r]
    # with  d_t[o, r] = sum_n C~_t[n, o] B~_t[n, r]  (C~ carrying the static
    # read offset R(nu_h) when `theta_h` is present; the per-token phase
    # R(Theta_t) cancels at equal t and both tiles are read UNROTATED
    # here, exactly as at R = O = 1).  With  s_t[o, r] = sum_g dY_t[o, g]
    # x_t[g, r]  (Kc's atomic reduction, now an [O, R] matrix per token):
    #
    #   dd_t[o, r]  = -n_t s_t[o, r]
    #   dB~_t[n, r] = sum_o dd_t[o, r] C~_t[n, o]
    #   dC~_t[n, o] = sum_r dd_t[o, r] B~_t[n, r]      (un-rotated by nu_h)
    #   dnu_h[k]    = sum_{t,o,r} dd_t[o,r] sum_{n in k} sgn_n B~[n,r] C~x[n,o]
    #   dbeta_h[n, r] = sum_t dB~_t[n, r],  dgamma_h[n, o] = sum_t dC~_t[n, o]
    #   dn_t        = -sum_{o,r} d_t[o,r] s_t[o,r]
    #
    # and the write scale's own chain rule is UNCHANGED except that the
    # scalar `d_t s_t` inside M_t becomes that (o, r) sum -- w_t and n_t
    # own no (o, r) axis (Delta is per kernel head, and the sibling r IS
    # the kernel head).
    #
    # Structurally this is the GROUPED branch's per-head loop with two
    # static (o, r) loops inside it, and it is the branch EVERY R, O > 1
    # call takes (grouped or not).  At R = O = 1 it is not compiled.
    GEN_OR: tl.constexpr = (O > 1) or (R > 1)
    # The head-VECTORISED general-(O, R) branch.  It needs the
    # per-head factors to be DIAGONAL in n, so a grouped READ map (a
    # per-head C TILE) keeps the per-head branch; a grouped WRITE
    # map is walked group by group inside it.
    GEN_VEC: tl.constexpr = GEN_OR and not C_GROUPED
    # ══ rotate-half: ONE dispatch for the layout, as an if/ELSE ═══════
    # NOT two sequential `if ...: return` blocks.  Triton keeps
    # generating code AFTER an early `return` taken inside a static
    # branch, and `tl.static_assert` fires at TRACE time regardless of
    # reachability -- so a sibling body's rank assert would trip on a
    # call that never reaches it.  One if/else, one `return`.
    if PAIR_HALF and HAS_THH:
        if GEN_OR:
            tl.static_assert(GEN_VEC,
                             "PAIR_HALF: the general-rank half adjoint "
                             "is the head-vectorised branch (an "
                             "ungrouped C map)")
            # adjoint on the ROTATE-HALF layout.
            #
            # The theta_h fold dominates this kernel's cost: it emits 9 MMAs
            # per (head block, o, r) in pass 1 and 7 in pass 2 against 2 in
            # the non-THH branch.  Under this layout two of them vanish and
            # four merge into two, because
            #
            #   P_Bx = P_B on the OTHER HALF,   P_Cx = P_C on the other half
            #   ch g_1 + sh g_2  is ONE [HB, N/2] operand
            #
            # and BOTH lanes of a pair land on the same `dThH` slot, so they
            # are summed in registers and written with ONE atomic instead of
            # two.  Per (head block, o) in pass 1 the count goes 9 -> 6
            # full-width-equivalents (12 half-width MMAs), and per
            # (head block, r) in pass 2 it goes 6 -> 3 (6 half-width).  The
            # identity is per (o, r), so it applies to every one of the
            # general branch's MMAs.
            #
            # The MMA K of the `ddT` dots is HB (the head block, capped at
            # 16 by `1 << (hpg_b - 1).bit_length()`), NOT N -- halving N
            # halves the OUTPUT width and leaves the minimum-legal-K floor
            # exactly where it was, so the count reduction is a pure win.
            # A grouped WRITE map is walked group by group (`GB`), exactly
            # as the interleaved GEN_VEC branch walks it.  A grouped READ
            # map is what `GEN_VEC` itself excludes, and the host refuses it.
            tl.static_assert(not C_GROUPED,
                             "PAIR_HALF: the half adjoint needs an "
                             "ungrouped C (read) map")
            KH: tl.constexpr = N // 2
            offs_p = tl.arange(0, KH)
            offs_h2n = KH + tl.arange(0, KH)
            b_base = B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
            c_base = C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
            bslot = c % BSLOT

            # ── pass 1, (r, g_b) OUTER: dB~, dbeta, dnu_h, and ddt/dlam ──
            for r in tl.range(R, loop_unroll_factor=1):
                for g in tl.range(GB, loop_unroll_factor=1):
                    dB1 = tl.zeros([TS, KH], dtype=tl.float32)
                    dB2 = tl.zeros([TS, KH], dtype=tl.float32)
                    Bt1_r = tl.load(b_base + g * s_bh + offs_p[None, :] * s_bn
                                    + r * s_br).to(tl.float32)
                    Bt2_r = tl.load(b_base + g * s_bh + offs_h2n[None, :] * s_bn
                                    + r * s_br).to(tl.float32)
                    for h0 in range(0, hpg_b, HB):
                        hs = g * hpg_b + h0 + tl.arange(0, HB)
                        hm = (h0 + tl.arange(0, HB) < hpg_b)[:, None]
                        flat = (b * H + hs[:, None]) * L + l[None, :]
                        n1 = tl.load(Own_ptr + flat, mask=hm, other=0.0)
                        if r == 0:
                            # the write scale's own chain rule, once per head
                            dt1 = tl.load(DW_ptr + b * s_wb + hs[:, None] * s_wh
                                          + l[None, :] * s_wl, mask=hm,
                                          other=0.0).to(tl.float32)
                            lam1 = tl.load(Lam_ptr + b * s_lb + hs[:, None] * s_lh
                                           + l[None, :] * s_ll, mask=hm,
                                           other=0.0).to(tl.float32)
                            A1 = tl.load(A_ptr + flat, mask=hm, other=0.0)
                            pm = hm & (l[None, :] > 0)
                            Ap1 = tl.load(A_ptr + (b * H + hs[:, None]) * L
                                          + (l[None, :] - 1), mask=pm, other=0.0)
                            msum_p = tl.zeros([HB, TS], dtype=tl.float32)
                            for oo in tl.range(O, loop_unroll_factor=1):
                                for rr in tl.range(R, loop_unroll_factor=1):
                                    dp = tl.load(Dot_ptr + b * s_qb
                                                 + hs[:, None] * s_qh
                                                 + l[None, :] * s_ql
                                                 + oo * s_qo + rr * s_qr,
                                                 mask=hm, other=0.0
                                                 ).to(tl.float32)
                                    sp = tl.load(S_ptr + b * s_sb
                                                 + hs[:, None] * s_sh
                                                 + l[None, :] * s_sl
                                                 + oo * s_so + rr * s_sr,
                                                 mask=hm, other=0.0)
                                    msum_p += dp * sp
                            Mp1 = Ap1  # TRAPFOLD: M_{t-1} = A_{t-1}; dL/down_t = A_t + d_t s_t
                            A1 = A1 + msum_p
                            tl.store(dDW_ptr + flat,
                                     A1 * lam1 + Mp1 * (1.0 - lam1), mask=hm)
                            if NEED_DLAM:
                                tl.store(dLam_ptr + flat, dt1 * (A1 - Mp1),
                                         mask=hm)
                        bet1 = tl.zeros([HB, KH], dtype=tl.float32)
                        bet2 = tl.zeros([HB, KH], dtype=tl.float32)
                        if HAS_BETA:
                            bet1 = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                           + offs_p[None, :] * R + r,
                                           mask=hm, other=0.0).to(tl.float32)
                            bet2 = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                           + offs_h2n[None, :] * R + r,
                                           mask=hm, other=0.0).to(tl.float32)
                        th_h = tl.load(ThH_ptr + hs[:, None] * KH
                                       + offs_p[None, :], mask=hm, other=0.0
                                       ).to(tl.float32)
                        cth = tl.cos(th_h)
                        sth = tl.sin(th_h)
                        dth_hb = tl.zeros([HB, KH], dtype=tl.float32)
                        dBe1_hb = tl.zeros([HB, KH], dtype=tl.float32)
                        dBe2_hb = tl.zeros([HB, KH], dtype=tl.float32)
                        for oi in tl.range(O, loop_unroll_factor=1):
                            s1 = tl.load(S_ptr + b * s_sb + hs[:, None] * s_sh
                                         + l[None, :] * s_sl + oi * s_so
                                         + r * s_sr, mask=hm, other=0.0)
                            dd = n1 * s1                        # [HB, TS]
                            ddT = tl.trans(dd)                     # [TS, HB]
                            ddrow = tl.sum(dd, axis=1)             # [HB]
                            Ct1_o = tl.load(c_base + offs_p[None, :] * s_cn
                                            + oi * s_co).to(tl.float32)
                            Ct2_o = tl.load(c_base + offs_h2n[None, :] * s_cn
                                            + oi * s_co).to(tl.float32)
                            ga1 = tl.zeros([HB, KH], dtype=tl.float32)
                            ga2 = tl.zeros([HB, KH], dtype=tl.float32)
                            if HAS_GAMMA:
                                ga1 = tl.load(Gamma_ptr + hs[:, None] * (N * O)
                                              + offs_p[None, :] * O + oi,
                                              mask=hm, other=0.0
                                              ).to(tl.float32)
                                ga2 = tl.load(Gamma_ptr + hs[:, None] * (N * O)
                                              + offs_h2n[None, :] * O + oi,
                                              mask=hm, other=0.0
                                              ).to(tl.float32)
                            # the four DISTINCT head contractions (the
                            # interleaved branch needs six: its `x` twins are
                            # these on the other half)
                            P_C1 = pdot(dd, Ct1_o, input_precision=PREC)
                            P_C2 = pdot(dd, Ct2_o, input_precision=PREC)
                            P_B1 = pdot(dd, Bt1_r, input_precision=PREC)
                            P_B2 = pdot(dd, Bt2_r, input_precision=PREC)
                            P_BC1 = pdot(dd, Bt1_r * Ct1_o,
                                           input_precision=PREC)
                            P_BC2 = pdot(dd, Bt2_r * Ct2_o,
                                           input_precision=PREC)
                            P_BCx1 = pdot(dd, Bt1_r * Ct2_o,
                                            input_precision=PREC)
                            P_BCx2 = pdot(dd, Bt2_r * Ct1_o,
                                            input_precision=PREC)
                            if HAS_BETA:
                                dBe1_hb += (cth * (P_C1 + ga1 * ddrow[:, None])
                                            + sth * (P_C2
                                                     + ga2 * ddrow[:, None]))
                                dBe2_hb += (cth * (P_C2 + ga2 * ddrow[:, None])
                                            - sth * (P_C1
                                                     + ga1 * ddrow[:, None]))
                            # both lanes of the pair, summed in REGISTERS
                            dth_hb += (
                                (cth * (P_BCx1 + ga2 * P_B1 + bet1 * P_C2
                                        + bet1 * ga2 * ddrow[:, None])
                                 - sth * (P_BC1 + ga1 * P_B1 + bet1 * P_C1
                                          + bet1 * ga1 * ddrow[:, None]))
                                - (cth * (P_BCx2 + ga1 * P_B2 + bet2 * P_C1
                                          + bet2 * ga1 * ddrow[:, None])
                                   + sth * (P_BC2 + ga2 * P_B2 + bet2 * P_C2
                                            + bet2 * ga2 * ddrow[:, None])))
                            Dc = pdot(ddT, cth, input_precision=PREC)
                            Ds = pdot(ddT, sth, input_precision=PREC)
                            dB1 += Ct1_o * Dc + Ct2_o * Ds
                            dB2 += Ct2_o * Dc - Ct1_o * Ds
                            if HAS_GAMMA:
                                # the merged gamma legs: ONE [HB, N/2]
                                # operand each, two MMAs where the
                                # interleaved branch needs four
                                dB1 += pdot(ddT, cth * ga1 + sth * ga2,
                                              input_precision=PREC)
                                dB2 += pdot(ddT, cth * ga2 - sth * ga1,
                                              input_precision=PREC)
                        if HAS_BETA:
                            tl.atomic_add(dBeta_ptr + bslot * s_dbes
                                          + hs[:, None] * (N * R)
                                          + offs_p[None, :] * R + r,
                                          dBe1_hb, mask=hm, sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + bslot * s_dbes
                                          + hs[:, None] * (N * R)
                                          + offs_h2n[None, :] * R + r,
                                          dBe2_hb, mask=hm, sem=ACC_SEM)
                        tl.atomic_add(dThH_ptr + bslot * s_dths
                                      + hs[:, None] * KH + offs_p[None, :],
                                      dth_hb, mask=hm, sem=ACC_SEM)
                    db_p = (dB_ptr + b * s_dbb + c * s_dbc + g * s_dbh
                            + offs_t[:, None] * s_dbt + r * s_dbr)
                    tl.atomic_add(db_p + offs_p[None, :] * s_dbn, dB1, sem=ACC_SEM)
                    tl.atomic_add(db_p + offs_h2n[None, :] * s_dbn, dB2, sem=ACC_SEM)

            # ── pass 2, o OUTER: dC~[:, :, o] (un-rotated) and dgamma ────
            for o in tl.range(O, loop_unroll_factor=1):
                dC1 = tl.zeros([TS, KH], dtype=tl.float32)
                dC2 = tl.zeros([TS, KH], dtype=tl.float32)
                for g in tl.range(GB, loop_unroll_factor=1):
                    for h0 in range(0, hpg_b, HB):
                        hs = g * hpg_b + h0 + tl.arange(0, HB)
                        hm = (h0 + tl.arange(0, HB) < hpg_b)[:, None]
                        flat = (b * H + hs[:, None]) * L + l[None, :]
                        n1 = tl.load(Own_ptr + flat, mask=hm, other=0.0)
                        th_h = tl.load(ThH_ptr + hs[:, None] * KH
                                       + offs_p[None, :], mask=hm, other=0.0
                                       ).to(tl.float32)
                        cth = tl.cos(th_h)
                        sth = tl.sin(th_h)
                        dG1_hb = tl.zeros([HB, KH], dtype=tl.float32)
                        dG2_hb = tl.zeros([HB, KH], dtype=tl.float32)
                        # `ri`, not `r` -- pass 1's `r` is a
                        # RUNTIME loop target and is still in scope here,
                        # so a static rebinding of the same name is a
                        # loop-carried type change.
                        for ri in tl.range(R, loop_unroll_factor=1):
                            s1 = tl.load(S_ptr + b * s_sb + hs[:, None] * s_sh
                                         + l[None, :] * s_sl + o * s_so
                                         + ri * s_sr, mask=hm, other=0.0)
                            dd = n1 * s1
                            ddT = tl.trans(dd)
                            Bt1_r = tl.load(b_base + g * s_bh
                                            + offs_p[None, :] * s_bn
                                            + ri * s_br).to(tl.float32)
                            Bt2_r = tl.load(b_base + g * s_bh
                                            + offs_h2n[None, :] * s_bn
                                            + ri * s_br).to(tl.float32)
                            bet1 = tl.zeros([HB, KH], dtype=tl.float32)
                            bet2 = tl.zeros([HB, KH], dtype=tl.float32)
                            if HAS_BETA:
                                bet1 = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                               + offs_p[None, :] * R + ri,
                                               mask=hm, other=0.0
                                               ).to(tl.float32)
                                bet2 = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                               + offs_h2n[None, :] * R + ri,
                                               mask=hm, other=0.0
                                               ).to(tl.float32)
                            if HAS_GAMMA:
                                ddrow = tl.sum(dd, axis=1)
                                P_B1 = pdot(dd, Bt1_r, input_precision=PREC)
                                P_B2 = pdot(dd, Bt2_r, input_precision=PREC)
                                dG1_hb += (cth * (P_B1 + bet1 * ddrow[:, None])
                                           - sth * (P_B2
                                                    + bet2 * ddrow[:, None]))
                                dG2_hb += (cth * (P_B2 + bet2 * ddrow[:, None])
                                           + sth * (P_B1
                                                    + bet1 * ddrow[:, None]))
                            Dc = pdot(ddT, cth, input_precision=PREC)
                            Ds = pdot(ddT, sth, input_precision=PREC)
                            dC1 += Bt1_r * Dc - Bt2_r * Ds
                            dC2 += Bt2_r * Dc + Bt1_r * Ds
                            if HAS_BETA:
                                dC1 += pdot(ddT, cth * bet1 - sth * bet2,
                                              input_precision=PREC)
                                dC2 += pdot(ddT, cth * bet2 + sth * bet1,
                                              input_precision=PREC)
                        if HAS_GAMMA:
                            tl.atomic_add(dGamma_ptr + bslot * s_dgs
                                          + hs[:, None] * (N * O)
                                          + offs_p[None, :] * O + o,
                                          dG1_hb, mask=hm, sem=ACC_SEM)
                            tl.atomic_add(dGamma_ptr + bslot * s_dgs
                                          + hs[:, None] * (N * O)
                                          + offs_h2n[None, :] * O + o,
                                          dG2_hb, mask=hm, sem=ACC_SEM)
                dc_p = (dC_ptr + b * s_dcb + c * s_dcc
                        + offs_t[:, None] * s_dct + o * s_dco)
                tl.atomic_add(dc_p + offs_p[None, :] * s_dcn, dC1, sem=ACC_SEM)
                tl.atomic_add(dc_p + offs_h2n[None, :] * s_dcn, dC2, sem=ACC_SEM)
        else:
            # ═══ the ROTATE-HALF pair layout ═══════
            # The trapezoid adjoint's only layout-sensitive object is the
            # static read offset's pair partner.  Under this layout the
            # partner of half 1 IS half 2, so the two `offs_n ^ 1` loads
            # (Btx, Ctx) and the two per-head partner loads (betax, gammax)
            # disappear, and with them TWO of the six head-contraction MMAs:
            #   P_Bx = P_B on the other half,  P_Cx = P_C on the other half.
            # The gamma / beta legs of dB_add / dC_add also merge, because
            # `cth*g1 + sin*g2` is one [HB, N/2] operand -- two MMAs where
            # the interleaved branch needs four.  All of it is the same sum:
            # even lane <-> half 1, odd lane <-> half 2, sgn = (+1, -1).
            tl.static_assert(not GROUPED,
                             "PAIR_HALF: head-shared B/C only")
            # R = O = 1 BY CONSTRUCTION -- the general-rank half body above
            # takes every (R, O) > 1 and returns.
            tl.static_assert(O == 1 and R == 1,
                             "PAIR_HALF: the R = O = 1 half adjoint was "
                             "reached with GEN_OR false at R, O > 1")
            KH: tl.constexpr = N // 2
            offs_p = tl.arange(0, KH)
            offs_h2n = KH + tl.arange(0, KH)
            Bt1 = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
                          + offs_p[None, :] * s_bn).to(tl.float32)
            Bt2 = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
                          + offs_h2n[None, :] * s_bn).to(tl.float32)
            Ct1 = tl.load(C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
                          + offs_p[None, :] * s_cn).to(tl.float32)
            Ct2 = tl.load(C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
                          + offs_h2n[None, :] * s_cn).to(tl.float32)
            dB1 = tl.zeros([TS, KH], dtype=tl.float32)
            dB2 = tl.zeros([TS, KH], dtype=tl.float32)
            dC1 = tl.zeros([TS, KH], dtype=tl.float32)
            dC2 = tl.zeros([TS, KH], dtype=tl.float32)
            BC11 = Bt1 * Ct1
            BC22 = Bt2 * Ct2
            BC12 = Bt1 * Ct2
            BC21 = Bt2 * Ct1
            bslot = c % BSLOT

            for h0 in range(0, H, HB):
                hs = h0 + tl.arange(0, HB)
                hm = (hs < H)[:, None]
                dt1 = tl.load(DW_ptr + b * s_wb + hs[:, None] * s_wh
                              + l[None, :] * s_wl, mask=hm, other=0.0
                              ).to(tl.float32)
                lam1 = tl.load(Lam_ptr + b * s_lb + hs[:, None] * s_lh
                               + l[None, :] * s_ll, mask=hm, other=0.0
                               ).to(tl.float32)
                flat = (b * H + hs[:, None]) * L + l[None, :]
                n1 = tl.load(Own_ptr + flat, mask=hm, other=0.0)
                A1 = tl.load(A_ptr + flat, mask=hm, other=0.0)
                s1 = tl.load(S_ptr + flat, mask=hm, other=0.0)
                pm = hm & (l[None, :] > 0)
                flatp = (b * H + hs[:, None]) * L + (l[None, :] - 1)
                Ap1 = tl.load(A_ptr + flatp, mask=pm, other=0.0)
                d1 = tl.load(Dot_ptr + b * s_qb + hs[:, None] * s_qh
                             + l[None, :] * s_ql, mask=hm, other=0.0
                             ).to(tl.float32)
                Mp1 = Ap1  # TRAPFOLD: M_{t-1} = A_{t-1}; dL/down_t = A_t + d_t s_t
                A1 = A1 + d1 * s1
                tl.store(dDW_ptr + flat, A1 * lam1 + Mp1 * (1.0 - lam1), mask=hm)
                if NEED_DLAM:
                    tl.store(dLam_ptr + flat, dt1 * (A1 - Mp1), mask=hm)

                dd = n1 * s1                                  # [HB, T]
                ddrow = tl.sum(dd, axis=1)                       # [HB]
                ddT = tl.trans(dd)                               # [T, HB]

                be1 = tl.zeros([HB, KH], dtype=tl.float32)
                be2 = tl.zeros([HB, KH], dtype=tl.float32)
                ga1 = tl.zeros([HB, KH], dtype=tl.float32)
                ga2 = tl.zeros([HB, KH], dtype=tl.float32)
                if HAS_BETA:
                    be1 = tl.load(Beta_ptr + hs[:, None] * N + offs_p[None, :],
                                  mask=hm, other=0.0).to(tl.float32)
                    be2 = tl.load(Beta_ptr + hs[:, None] * N + offs_h2n[None, :],
                                  mask=hm, other=0.0).to(tl.float32)
                if HAS_GAMMA:
                    ga1 = tl.load(Gamma_ptr + hs[:, None] * N + offs_p[None, :],
                                  mask=hm, other=0.0).to(tl.float32)
                    ga2 = tl.load(Gamma_ptr + hs[:, None] * N + offs_h2n[None, :],
                                  mask=hm, other=0.0).to(tl.float32)
                th_h = tl.load(ThH_ptr + hs[:, None] * KH + offs_p[None, :],
                               mask=hm, other=0.0).to(tl.float32)
                cth = tl.cos(th_h)
                sth = tl.sin(th_h)

                P_B1 = pdot(dd, Bt1, input_precision=PREC)
                P_B2 = pdot(dd, Bt2, input_precision=PREC)
                P_C1 = pdot(dd, Ct1, input_precision=PREC)
                P_C2 = pdot(dd, Ct2, input_precision=PREC)
                P_BC1 = pdot(dd, BC11, input_precision=PREC)
                P_BC2 = pdot(dd, BC22, input_precision=PREC)
                P_BCx1 = pdot(dd, BC12, input_precision=PREC)
                P_BCx2 = pdot(dd, BC21, input_precision=PREC)

                if HAS_GAMMA:
                    dG1 = (cth * (P_B1 + be1 * ddrow[:, None])
                           - sth * (P_B2 + be2 * ddrow[:, None]))
                    dG2 = (cth * (P_B2 + be2 * ddrow[:, None])
                           + sth * (P_B1 + be1 * ddrow[:, None]))
                    tl.atomic_add(dGamma_ptr + bslot * s_dgs
                                  + hs[:, None] * N + offs_p[None, :],
                                  dG1, mask=hm, sem=ACC_SEM)
                    tl.atomic_add(dGamma_ptr + bslot * s_dgs
                                  + hs[:, None] * N + offs_h2n[None, :],
                                  dG2, mask=hm, sem=ACC_SEM)
                if HAS_BETA:
                    dBe1 = (cth * (P_C1 + ga1 * ddrow[:, None])
                            + sth * (P_C2 + ga2 * ddrow[:, None]))
                    dBe2 = (cth * (P_C2 + ga2 * ddrow[:, None])
                            - sth * (P_C1 + ga1 * ddrow[:, None]))
                    tl.atomic_add(dBeta_ptr + bslot * s_dbes
                                  + hs[:, None] * N + offs_p[None, :],
                                  dBe1, mask=hm, sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + bslot * s_dbes
                                  + hs[:, None] * N + offs_h2n[None, :],
                                  dBe2, mask=hm, sem=ACC_SEM)
                # the two lanes of a pair land on the SAME dThH slot, so they
                # are summed in registers and written with ONE atomic
                dth_pair = (
                    (cth * (P_BCx1 + ga2 * P_B1 + be1 * P_C2
                            + be1 * ga2 * ddrow[:, None])
                     - sth * (P_BC1 + ga1 * P_B1 + be1 * P_C1
                              + be1 * ga1 * ddrow[:, None]))
                    - (cth * (P_BCx2 + ga1 * P_B2 + be2 * P_C1
                              + be2 * ga1 * ddrow[:, None])
                       + sth * (P_BC2 + ga2 * P_B2 + be2 * P_C2
                                + be2 * ga2 * ddrow[:, None])))
                tl.atomic_add(dThH_ptr + bslot * s_dths
                              + hs[:, None] * KH + offs_p[None, :],
                              dth_pair, mask=hm, sem=ACC_SEM)

                Dc = pdot(ddT, cth, input_precision=PREC)       # [TS, N/2]
                Ds = pdot(ddT, sth, input_precision=PREC)
                dB1 += (Ct1 * Dc + Ct2 * Ds
                        + pdot(ddT, cth * ga1 + sth * ga2,
                                 input_precision=PREC))
                dB2 += (Ct2 * Dc - Ct1 * Ds
                        + pdot(ddT, cth * ga2 - sth * ga1,
                                 input_precision=PREC))
                dC1 += (Bt1 * Dc - Bt2 * Ds
                        + pdot(ddT, cth * be1 - sth * be2,
                                 input_precision=PREC))
                dC2 += (Bt2 * Dc + Bt1 * Ds
                        + pdot(ddT, cth * be2 + sth * be1,
                                 input_precision=PREC))

            db_p = dB_ptr + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt
            dc_p = dC_ptr + b * s_dcb + c * s_dcc + offs_t[:, None] * s_dct
            tl.atomic_add(db_p + offs_p[None, :] * s_dbn, dB1, sem=ACC_SEM)
            tl.atomic_add(db_p + offs_h2n[None, :] * s_dbn, dB2, sem=ACC_SEM)
            tl.atomic_add(dc_p + offs_p[None, :] * s_dcn, dC1, sem=ACC_SEM)
            tl.atomic_add(dc_p + offs_h2n[None, :] * s_dcn, dC2, sem=ACC_SEM)
        return


    if GEN_OR:
        if GEN_VEC:
            # ══════════════════════════════════════════════════════════════════
            # The HEAD-VECTORISED general-(O, R) adjoint.
            #
            # The per-head GEN_OR branch is elementwise: it would emit
            # Hk*(R + O) atomic [TS, N] tiles per (b, chunk) and as many
            # cross-thread `tl.sum(., axis=0)` reductions per CTA.  The head
            # reduction is a CONTRACTION, so it is `tl.dot` whenever the
            # per-head factors are DIAGONAL in n:
            #
            #   dB~[g_b, t, n, r] = sum_{h in g_b, o} dd[h,o,r,t] C~[h, t, n, o]
            #   dC~[t, n, o]      = sum_{h,r}        dd[h,o,r,t] B~[h, t, n, r]
            #
            # with C~[h,t,n,o] = cth[h,n](C[t,n,o] + gamma[h,n,o])
            #                  + sth[h,n](Cx[t,n,o] + gammax[h,n,o])   (HAS_THH)
            # and B~[h,t,n,r] = B[g_b, t, n, r] + beta[h,n,r].  Every term is then
            # a [TS, HB] x [HB, N] MMA against a t-independent per-head operand,
            # times a loop-invariant [TS, N] tile -- and the six P_* MMAs give
            # dbeta / dgamma / dnu their t-sums for free (no `tl.sum(., axis=0)`).
            #
            # The (o, r) axes are NOT promoted to tile axes (r is dB's
            # innermost memory axis).  The r loop
            # is OUTER for dB~ and the o loop OUTER for dC~, so each output tile
            # is finished in registers and stored ONCE: Gb*R + O atomic tiles per
            # (b, chunk) instead of Hk*(R + O).  Triton cannot hold a VARIABLE
            # number of [TS, N] accumulators live (a Python list of tile values is
            # not expressible), which is what forces the two passes; the re-read
            # operands are L2 hits (2(Gb*R + O) distinct tiles per CTA).
            #
            # A GROUPED WRITE MAP (`B_GROUPED`, the `Gb = R` sibling gate) is
            # walked GROUP BY GROUP -- the head loop
            # runs over `hpg_b` heads starting at `g * hpg_b` -- rather than
            # masked inside a full-H block, so the grouping costs no extra lanes:
            # Gb blocks of hpg_b heads is the same total work as one pass over H.
            # A grouped READ map (Gc > 1) has a per-head C TILE, which is not a
            # diagonal factor and cannot be contracted this way; it keeps the
            # per-head branch (the host-folded-theta_h route, which does not
            # take the in-scan trapezoid anyway).
            sgn = tl.where((offs_n % 2) == 0, 1.0, -1.0)
            b_base = B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
            c_base = C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
            bslot = c % BSLOT
            # ── the theta_h fold's MMA WIDTH ──────
            # The dots of the fold share a LEFT operand (`ddT`, resp. `dd`)
            # and differ only in an [HB, N] / [TS, N] right operand, so they
            # merge along the OUTPUT (N) axis -- one MMA with a 2N- or
            # 4N-wide right operand instead of two or four at the same K.
            # Same K, same operand values, same reduction order, so the
            # result is BIT-IDENTICAL.  Pass 1 goes 9 -> 4 MMAs per (o), pass
            # 2 goes 7 -> 2 per (r).  The interleaved layout is `n * W + j`,
            # which is exactly what `tl.reshape(., (., N, 2, 2))` + two
            # `tl.split`s undoes.  (The (h, o) K-merge only applies to the
            # two gamma terms: the `cth` / `sth` products are scaled by an
            # o-DEPENDENT [TS, N] tile OUTSIDE the contraction.)
            offs_w4 = tl.arange(0, 4 * N)
            n4 = offs_w4 // 4
            j4 = offs_w4 % 4
            offs_w2 = tl.arange(0, 2 * N)
            n2 = offs_w2 // 2
            j2 = offs_w2 % 2
            sgn4 = tl.where((n4 % 2) == 0, 1.0, -1.0)
            sgn2 = tl.where((n2 % 2) == 0, 1.0, -1.0)
            W_ROW: tl.constexpr = (THH_W & 1) != 0
            W_COL: tl.constexpr = (THH_W & 2) != 0
            # the K-merge needs a POWER-OF-TWO pair of (h, o) rows to
            # build its concatenated operand with one `tl.join`; O = 2 /
            # R = 2 is the production cell and the only one it serves.
            W_KM1: tl.constexpr = ((THH_W & 4) != 0 and O == 2
                                   and not (THH_W & 1))
            W_KM2: tl.constexpr = ((THH_W & 4) != 0 and R == 2
                                   and not (THH_W & 1))

            # ── pass 1, (r, g_b) OUTER: dB~, dbeta, dnu, and ddt/dlam ─────────
            for r in tl.range(R, loop_unroll_factor=1):
                for g in tl.range(GB, loop_unroll_factor=1):
                    dB_add = tl.zeros([TS, N], dtype=tl.float32)
                    Bt_r = tl.load(b_base + g * s_bh + offs_n[None, :] * s_bn
                                   + r * s_br).to(tl.float32)
                    if HAS_THH:
                        Btx_r = tl.load(b_base + g * s_bh
                                        + (offs_n ^ 1)[None, :] * s_bn
                                        + r * s_br).to(tl.float32)
                        if W_COL:
                            # B~ duplicated on the pair-merged N axis, so
                            # `Btw2 * [C | Cx]` IS [B C | B Cx]: the two
                            # P_BC MMAs become one.  Same bytes (an L1
                            # hit on the tile above), same values.
                            Btw2 = tl.load(b_base + g * s_bh
                                           + n2[None, :] * s_bn
                                           + r * s_br).to(tl.float32)
                    for h0 in range(0, hpg_b, HB):
                        hs = g * hpg_b + h0 + tl.arange(0, HB)         # [HB]
                        hm = (h0 + tl.arange(0, HB) < hpg_b)[:, None]
                        flat = (b * H + hs[:, None]) * L + l[None, :]
                        n1 = tl.load(Own_ptr + flat, mask=hm, other=0.0)
                        if r == 0:
                            # the write scale's own chain rule, once per head: it
                            # carries no (o, r) axis (Delta is owned per KERNEL
                            # head and the sibling r IS that head)
                            dt1 = tl.load(DW_ptr + b * s_wb + hs[:, None] * s_wh
                                          + l[None, :] * s_wl, mask=hm, other=0.0
                                          ).to(tl.float32)
                            lam1 = tl.load(Lam_ptr + b * s_lb + hs[:, None] * s_lh
                                           + l[None, :] * s_ll, mask=hm, other=0.0
                                           ).to(tl.float32)
                            A1 = tl.load(A_ptr + flat, mask=hm, other=0.0)
                            pm = hm & (l[None, :] > 0)
                            Ap1 = tl.load(A_ptr + (b * H + hs[:, None]) * L
                                          + (l[None, :] - 1), mask=pm, other=0.0)
                            msum_p = tl.zeros([HB, TS], dtype=tl.float32)
                            for oo in tl.range(O, loop_unroll_factor=1):
                                for rr in tl.range(R, loop_unroll_factor=1):
                                    dp = tl.load(Dot_ptr + b * s_qb
                                                 + hs[:, None] * s_qh
                                                 + l[None, :] * s_ql
                                                 + oo * s_qo + rr * s_qr,
                                                 mask=hm, other=0.0).to(tl.float32)
                                    sp = tl.load(S_ptr + b * s_sb + hs[:, None] * s_sh
                                                 + l[None, :] * s_sl
                                                 + oo * s_so + rr * s_sr,
                                                 mask=hm, other=0.0)
                                    msum_p += dp * sp
                            Mp1 = Ap1  # TRAPFOLD: M_{t-1} = A_{t-1}; dL/down_t = A_t + d_t s_t
                            A1 = A1 + msum_p
                            tl.store(dDW_ptr + flat,
                                     A1 * lam1 + Mp1 * (1.0 - lam1), mask=hm)
                            if NEED_DLAM:
                                tl.store(dLam_ptr + flat, dt1 * (A1 - Mp1), mask=hm)
                        bet_r = tl.zeros([HB, N], dtype=tl.float32)
                        betx_r = tl.zeros([HB, N], dtype=tl.float32)
                        if HAS_BETA:
                            bet_r = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                            + offs_n[None, :] * R + r,
                                            mask=hm, other=0.0).to(tl.float32)
                            if HAS_THH:
                                betx_r = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                                 + (offs_n ^ 1)[None, :] * R + r,
                                                 mask=hm, other=0.0).to(tl.float32)
                        if HAS_THH:
                            th_h = tl.load(ThH_ptr + hs[:, None] * (N // 2)
                                           + (offs_n // 2)[None, :],
                                           mask=hm, other=0.0).to(tl.float32)
                            cth = tl.cos(th_h)
                            sth = sgn[None, :] * tl.sin(th_h)
                            dth_hb = tl.zeros([HB, N], dtype=tl.float32)
                            if W_ROW:
                                if HAS_GAMMA:
                                    th_w = tl.load(
                                        ThH_ptr + hs[:, None] * (N // 2)
                                        + (n4 // 2)[None, :],
                                        mask=hm, other=0.0).to(tl.float32)
                                    cth_w = tl.cos(th_w)
                                    sth_w = sgn4[None, :] * tl.sin(th_w)
                                else:
                                    th_w = tl.load(
                                        ThH_ptr + hs[:, None] * (N // 2)
                                        + (n2 // 2)[None, :],
                                        mask=hm, other=0.0).to(tl.float32)
                                    cth_w = tl.cos(th_w)
                                    sth_w = sgn2[None, :] * tl.sin(th_w)
                        dBeta_hb = tl.zeros([HB, N], dtype=tl.float32)
                        ddT_l = ()
                        for oi in tl.static_range(O):
                            s1 = tl.load(S_ptr + b * s_sb + hs[:, None] * s_sh
                                         + l[None, :] * s_sl
                                         + oi * s_so + r * s_sr,
                                         mask=hm, other=0.0)
                            dd = n1 * s1                            # [HB, TS]
                            ddT = tl.trans(dd)                         # [TS, HB]
                            Ct_o = tl.load(c_base + offs_n[None, :] * s_cn
                                           + oi * s_co).to(tl.float32)
                            gam_o = tl.zeros([HB, N], dtype=tl.float32)
                            gamx_o = tl.zeros([HB, N], dtype=tl.float32)
                            if HAS_GAMMA:
                                gam_o = tl.load(Gamma_ptr + hs[:, None] * (N * O)
                                                + offs_n[None, :] * O + oi,
                                                mask=hm, other=0.0).to(tl.float32)
                            if HAS_THH:
                                Ctx_o = tl.load(c_base + (offs_n ^ 1)[None, :] * s_cn
                                                + oi * s_co).to(tl.float32)
                                if HAS_GAMMA:
                                    gamx_o = tl.load(
                                        Gamma_ptr + hs[:, None] * (N * O)
                                        + (offs_n ^ 1)[None, :] * O + oi,
                                        mask=hm, other=0.0).to(tl.float32)
                                ddrow = tl.sum(dd, axis=1)             # [HB]
                                if W_ROW:
                                    # ── the N-merged theta_h fold ──────
                                    # one MMA for the four `ddT` terms
                                    # (cth | sth | cth gam | sth gamx):
                                    # 4 MMAs -> 1, at a 4N-wide OUTPUT.
                                    if HAS_GAMMA:
                                        gam_w = tl.load(
                                            Gamma_ptr + hs[:, None] * (N * O)
                                            + n4[None, :] * O + oi,
                                            mask=hm, other=0.0).to(tl.float32)
                                        gamx_w = tl.load(
                                            Gamma_ptr + hs[:, None] * (N * O)
                                            + (n4 ^ 1)[None, :] * O + oi,
                                            mask=hm, other=0.0).to(tl.float32)
                                        RA = tl.where(
                                            j4[None, :] == 0, cth_w,
                                            tl.where(j4[None, :] == 1, sth_w,
                                                     tl.where(j4[None, :] == 2,
                                                              cth_w * gam_w,
                                                              sth_w * gamx_w)))
                                        MA = pdot(ddT, RA,
                                                    input_precision=PREC)
                                        # two 3-D splits, not one 4-D
                                        # one: `tl.split` takes the LAST
                                        # axis of a rank <= 3 tile.
                                        ma0, ma1 = tl.split(
                                            tl.reshape(MA, (TS, 2 * N, 2)))
                                        D_c, D_cg = tl.split(
                                            tl.reshape(ma0, (TS, N, 2)))
                                        D_s, D_sg = tl.split(
                                            tl.reshape(ma1, (TS, N, 2)))
                                        dB_add += (Ct_o * D_c + Ctx_o * D_s
                                                   + D_cg + D_sg)
                                    else:
                                        RA = tl.where(j2[None, :] == 0,
                                                      cth_w, sth_w)
                                        MA = pdot(ddT, RA,
                                                    input_precision=PREC)
                                        D_c, D_s = tl.split(
                                            tl.reshape(MA, (TS, N, 2)))
                                        dB_add += Ct_o * D_c + Ctx_o * D_s
                                else:
                                    dB_add += (Ct_o * pdot(ddT, cth, input_precision=PREC)
                                               + Ctx_o * pdot(ddT, sth,
                                                                input_precision=PREC))
                                    if HAS_GAMMA and not W_KM1:
                                        dB_add += (pdot(ddT, cth * gam_o,
                                                          input_precision=PREC)
                                                   + pdot(ddT, sth * gamx_o,
                                                            input_precision=PREC))
                                    if W_KM1:
                                        # The o-sum of the two
                                        # GAMMA terms IS a contraction --
                                        # their right operand carries the
                                        # o axis and nothing outside the
                                        # dot does -- so the (h, o) axes
                                        # merge into ONE K = HB*O dot.
                                        # The `cth` / `sth` terms above
                                        # do NOT fold: each is scaled by
                                        # the o-dependent [TS, N] tile
                                        # `Ct_o` OUTSIDE the contraction.
                                        ddT_l = ddT_l + (ddT,)
                                if W_COL:
                                    # the `dd`-side pairs share a LEFT
                                    # operand and differ only in an
                                    # [TS, N] right one: (C | Cx) and
                                    # (B C | B Cx) are two MMAs, not four,
                                    # at a 2N-wide [HB, .] output.
                                    RBa = tl.where(
                                        j2[None, :] == 0,
                                        tl.load(c_base + n2[None, :] * s_cn
                                                + oi * s_co).to(tl.float32),
                                        tl.load(c_base + (n2 ^ 1)[None, :] * s_cn
                                                + oi * s_co).to(tl.float32))
                                    P_C, P_Cx = tl.split(tl.reshape(
                                        pdot(dd, RBa, input_precision=PREC),
                                        (HB, N, 2)))
                                    P_BC, P_BCx = tl.split(tl.reshape(
                                        pdot(dd, Btw2 * RBa,
                                               input_precision=PREC),
                                        (HB, N, 2)))
                                else:
                                    P_C = pdot(dd, Ct_o, input_precision=PREC)
                                    P_Cx = pdot(dd, Ctx_o, input_precision=PREC)
                                    P_BC = pdot(dd, Bt_r * Ct_o,
                                                  input_precision=PREC)
                                    P_BCx = pdot(dd, Bt_r * Ctx_o,
                                                   input_precision=PREC)
                                if HAS_BETA:
                                    dBeta_hb += (cth * (P_C + gam_o * ddrow[:, None])
                                                 + sth * (P_Cx
                                                          + gamx_o * ddrow[:, None]))
                                P_B = pdot(dd, Bt_r, input_precision=PREC)
                                dth_hb += sgn[None, :] * (
                                    cth * (P_BCx + gamx_o * P_B + bet_r * P_Cx
                                           + bet_r * gamx_o * ddrow[:, None])
                                    - sth * (P_BC + gam_o * P_B + bet_r * P_C
                                             + bet_r * gam_o * ddrow[:, None]))
                            else:
                                dB_add += tl.sum(dd, axis=0)[:, None] * Ct_o
                                if HAS_GAMMA:
                                    dB_add += pdot(ddT, gam_o, input_precision=PREC)
                                if HAS_BETA:
                                    db = pdot(dd, Ct_o, input_precision=PREC)
                                    if HAS_GAMMA:
                                        db += gam_o * tl.sum(dd, axis=1)[:, None]
                                    dBeta_hb += db
                        if HAS_THH and HAS_GAMMA and W_KM1:
                            # [TS, HB*O], column h*O + o -- the join's own
                            # interleave, which is why the rows below are
                            # indexed the same way.
                            ddcT = tl.reshape(tl.join(ddT_l[0], ddT_l[1]),
                                              (TS, HB * O))
                            hoc = tl.arange(0, HB * O)
                            hoh = hoc // O
                            hoo = hoc % O
                            hsc = g * hpg_b + h0 + hoh
                            hmc = (h0 + hoh < hpg_b)[:, None]
                            thc = tl.load(ThH_ptr + hsc[:, None] * (N // 2)
                                          + (offs_n // 2)[None, :],
                                          mask=hmc, other=0.0).to(tl.float32)
                            cthc = tl.cos(thc)
                            sthc = sgn[None, :] * tl.sin(thc)
                            gamc = tl.load(Gamma_ptr + hsc[:, None] * (N * O)
                                           + offs_n[None, :] * O + hoo[:, None],
                                           mask=hmc, other=0.0).to(tl.float32)
                            gamxc = tl.load(Gamma_ptr + hsc[:, None] * (N * O)
                                            + (offs_n ^ 1)[None, :] * O
                                            + hoo[:, None],
                                            mask=hmc, other=0.0).to(tl.float32)
                            dB_add += (pdot(ddcT, cthc * gamc,
                                              input_precision=PREC)
                                       + pdot(ddcT, sthc * gamxc,
                                                input_precision=PREC))
                        if HAS_BETA:
                            tl.atomic_add(dBeta_ptr + bslot * s_dbes
                                          + hs[:, None] * (N * R)
                                          + offs_n[None, :] * R + r,
                                          dBeta_hb, mask=hm, sem=ACC_SEM)
                        if HAS_THH:
                            tl.atomic_add(dThH_ptr + bslot * s_dths
                                          + hs[:, None] * (N // 2)
                                          + (offs_n // 2)[None, :], dth_hb, mask=hm, sem=ACC_SEM)
                    tl.atomic_add(dB_ptr + b * s_dbb + c * s_dbc + g * s_dbh
                                  + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn
                                  + r * s_dbr, dB_add, sem=ACC_SEM)

            # ── pass 2, o OUTER: dC~[:, :, o] (un-rotated) and dgamma[:, :, o] ─
            for o in tl.range(O, loop_unroll_factor=1):
                dC_add = tl.zeros([TS, N], dtype=tl.float32)
                for g in tl.range(GB, loop_unroll_factor=1):
                    for h0 in range(0, hpg_b, HB):
                        hs = g * hpg_b + h0 + tl.arange(0, HB)
                        hm = (h0 + tl.arange(0, HB) < hpg_b)[:, None]
                        flat = (b * H + hs[:, None]) * L + l[None, :]
                        n1 = tl.load(Own_ptr + flat, mask=hm, other=0.0)
                        if HAS_THH:
                            th_h = tl.load(ThH_ptr + hs[:, None] * (N // 2)
                                           + (offs_n // 2)[None, :],
                                           mask=hm, other=0.0).to(tl.float32)
                            cth = tl.cos(th_h)
                            sth = sgn[None, :] * tl.sin(th_h)
                            if W_ROW:
                                # same N-merge as pass 1
                                if HAS_BETA:
                                    th_w = tl.load(
                                        ThH_ptr + hs[:, None] * (N // 2)
                                        + (n4 // 2)[None, :],
                                        mask=hm, other=0.0).to(tl.float32)
                                    cth_w = tl.cos(th_w)
                                    sth_w = sgn4[None, :] * tl.sin(th_w)
                                else:
                                    th_w = tl.load(
                                        ThH_ptr + hs[:, None] * (N // 2)
                                        + (n2 // 2)[None, :],
                                        mask=hm, other=0.0).to(tl.float32)
                                    cth_w = tl.cos(th_w)
                                    sth_w = sgn2[None, :] * tl.sin(th_w)
                        dGamma_hb = tl.zeros([HB, N], dtype=tl.float32)
                        ddT_l = ()
                        # `ri`, not `r` (see pass 1)
                        for ri in tl.static_range(R):
                            s1 = tl.load(S_ptr + b * s_sb + hs[:, None] * s_sh
                                         + l[None, :] * s_sl
                                         + o * s_so + ri * s_sr,
                                         mask=hm, other=0.0)
                            dd = n1 * s1
                            ddT = tl.trans(dd)
                            Bt_r = tl.load(b_base + g * s_bh
                                           + offs_n[None, :] * s_bn
                                           + ri * s_br).to(tl.float32)
                            bet_r = tl.zeros([HB, N], dtype=tl.float32)
                            betx_r = tl.zeros([HB, N], dtype=tl.float32)
                            if HAS_BETA:
                                bet_r = tl.load(Beta_ptr + hs[:, None] * (N * R)
                                                + offs_n[None, :] * R + ri,
                                                mask=hm, other=0.0).to(tl.float32)
                            if HAS_THH:
                                Btx_r = tl.load(b_base + g * s_bh
                                                + (offs_n ^ 1)[None, :] * s_bn
                                                + ri * s_br).to(tl.float32)
                                if HAS_BETA:
                                    betx_r = tl.load(
                                        Beta_ptr + hs[:, None] * (N * R)
                                        + (offs_n ^ 1)[None, :] * R + ri,
                                        mask=hm, other=0.0).to(tl.float32)
                                if W_ROW:
                                    if HAS_BETA:
                                        bet_w = tl.load(
                                            Beta_ptr + hs[:, None] * (N * R)
                                            + n4[None, :] * R + ri,
                                            mask=hm, other=0.0).to(tl.float32)
                                        betx_w = tl.load(
                                            Beta_ptr + hs[:, None] * (N * R)
                                            + (n4 ^ 1)[None, :] * R + ri,
                                            mask=hm, other=0.0).to(tl.float32)
                                        RA2 = tl.where(
                                            j4[None, :] == 0, cth_w,
                                            tl.where(j4[None, :] == 1, sth_w,
                                                     tl.where(j4[None, :] == 2,
                                                              cth_w * bet_w,
                                                              sth_w * betx_w)))
                                        MA2 = pdot(ddT, RA2,
                                                     input_precision=PREC)
                                        m20, m21 = tl.split(
                                            tl.reshape(MA2, (TS, 2 * N, 2)))
                                        D_c, D_cb = tl.split(
                                            tl.reshape(m20, (TS, N, 2)))
                                        D_s, D_sb = tl.split(
                                            tl.reshape(m21, (TS, N, 2)))
                                        dC_add += (Bt_r * D_c - Btx_r * D_s
                                                   + D_cb - D_sb)
                                    else:
                                        RA2 = tl.where(j2[None, :] == 0,
                                                       cth_w, sth_w)
                                        MA2 = pdot(ddT, RA2,
                                                     input_precision=PREC)
                                        D_c, D_s = tl.split(
                                            tl.reshape(MA2, (TS, N, 2)))
                                        dC_add += Bt_r * D_c - Btx_r * D_s
                                else:
                                    dC_add += (Bt_r * pdot(ddT, cth,
                                                             input_precision=PREC)
                                               - Btx_r * pdot(ddT, sth,
                                                                input_precision=PREC))
                                    if HAS_BETA and not W_KM2:
                                        dC_add += (pdot(ddT, cth * bet_r,
                                                          input_precision=PREC)
                                                   - pdot(ddT, sth * betx_r,
                                                            input_precision=PREC))
                                    if W_KM2:
                                        ddT_l = ddT_l + (ddT,)
                                if HAS_GAMMA:
                                    ddrow = tl.sum(dd, axis=1)
                                if W_COL and HAS_GAMMA:
                                    RB2 = tl.where(
                                        j2[None, :] == 0,
                                        tl.load(b_base + g * s_bh
                                                + n2[None, :] * s_bn
                                                + ri * s_br).to(tl.float32),
                                        tl.load(b_base + g * s_bh
                                                + (n2 ^ 1)[None, :] * s_bn
                                                + ri * s_br).to(tl.float32))
                                    P_B, P_Bx = tl.split(tl.reshape(
                                        pdot(dd, RB2, input_precision=PREC),
                                        (HB, N, 2)))
                                elif HAS_GAMMA:
                                    P_B = pdot(dd, Bt_r,
                                                 input_precision=PREC)
                                    P_Bx = pdot(dd, Btx_r,
                                                  input_precision=PREC)
                                if HAS_GAMMA:
                                    dGamma_hb += (cth * (P_B + bet_r * ddrow[:, None])
                                                  - sth * (P_Bx
                                                           + betx_r * ddrow[:, None]))
                            else:
                                dC_add += tl.sum(dd, axis=0)[:, None] * Bt_r
                                if HAS_BETA:
                                    dC_add += pdot(ddT, bet_r, input_precision=PREC)
                                if HAS_GAMMA:
                                    dg = pdot(dd, Bt_r, input_precision=PREC)
                                    if HAS_BETA:
                                        dg += bet_r * tl.sum(dd, axis=1)[:, None]
                                    dGamma_hb += dg
                        if HAS_THH and HAS_BETA and W_KM2:
                            # The C-side K-merge: the r-sum of the
                            # two BETA terms merges into K = HB*R.
                            ddcT = tl.reshape(tl.join(ddT_l[0], ddT_l[1]),
                                              (TS, HB * R))
                            hrc = tl.arange(0, HB * R)
                            hrh = hrc // R
                            hrr = hrc % R
                            hsc = g * hpg_b + h0 + hrh
                            hmc = (h0 + hrh < hpg_b)[:, None]
                            thc = tl.load(ThH_ptr + hsc[:, None] * (N // 2)
                                          + (offs_n // 2)[None, :],
                                          mask=hmc, other=0.0).to(tl.float32)
                            cthc = tl.cos(thc)
                            sthc = sgn[None, :] * tl.sin(thc)
                            betc = tl.load(Beta_ptr + hsc[:, None] * (N * R)
                                           + offs_n[None, :] * R + hrr[:, None],
                                           mask=hmc, other=0.0).to(tl.float32)
                            betxc = tl.load(Beta_ptr + hsc[:, None] * (N * R)
                                            + (offs_n ^ 1)[None, :] * R
                                            + hrr[:, None],
                                            mask=hmc, other=0.0).to(tl.float32)
                            dC_add += (pdot(ddcT, cthc * betc,
                                              input_precision=PREC)
                                       - pdot(ddcT, sthc * betxc,
                                                input_precision=PREC))
                        if HAS_GAMMA:
                            tl.atomic_add(dGamma_ptr + bslot * s_dgs
                                          + hs[:, None] * (N * O)
                                          + offs_n[None, :] * O + o,
                                          dGamma_hb, mask=hm, sem=ACC_SEM)
                tl.atomic_add(dC_ptr + b * s_dcb + c * s_dcc
                              + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn
                              + o * s_dco, dC_add, sem=ACC_SEM)
        else:
            # Two passes over the head loop, one per output tile family:
            # `dB~[:, r] = sum_o dd[o,r] C~[:, o]` wants all o for a fixed r
            # and `dC~[:, o] = sum_r dd[o,r] B~[:, r]` all r for a fixed o,
            # and Triton has no way to keep a VARIABLE number of [TS, N]
            # accumulators live across one pass (a Python list of tile values
            # is not expressible in the frontend).  The repeated operand
            # loads are L2 hits -- one (b, chunk) CTA reads only 2(R + O)
            # DISTINCT tiles.
            #
            # Promoting r to a TILE axis ([R, TS, N] operands and a
            # `tl.sum(., axis=0)` contraction) was measured twice as slow:
            # the r axis is the INNERMOST axis of dB in memory, so a tile
            # that carries it as its slowest axis scatters every store.
            sgn = tl.where((offs_n % 2) == 0, 1.0, -1.0)
            b_base = B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
            c_base = C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
            for h in range(0, H):
                dt_h = tl.load(DW_ptr + b * s_wb + h * s_wh + l * s_wl
                               ).to(tl.float32)
                lam_h = tl.load(Lam_ptr + b * s_lb + h * s_lh + l * s_ll
                                ).to(tl.float32)
                flat = (b * H + h) * L + l
                n_h = tl.load(Own_ptr + flat)
                A_h = tl.load(A_ptr + flat)
                pm = l > 0
                Ap = tl.load(A_ptr + (b * H + h) * L + (l - 1),
                             mask=pm, other=0.0)
                q_base = Dot_ptr + b * s_qb + h * s_qh
                sd_base = S_ptr + b * s_sb + h * s_sh
                # TRAPFOLD: M_{t-1} = A_{t-1} (the strict-tile A); the own
                # tap's diagonal sum_{o,r} d_t[o,r] s_t[o,r] joins A_t
                msum_p = tl.zeros([TS], dtype=tl.float32)
                for om in tl.range(O, loop_unroll_factor=1):
                    for rm in tl.range(R, loop_unroll_factor=1):
                        dp = tl.load(q_base + l * s_ql
                                     + om * s_qo + rm * s_qr).to(tl.float32)
                        sp = tl.load(sd_base + l * s_sl
                                     + om * s_so + rm * s_sr)
                        msum_p += dp * sp
                Mp = Ap
                A_h = A_h + msum_p
                tl.store(dDW_ptr + flat, A_h * lam_h + Mp * (1.0 - lam_h))
                if NEED_DLAM:
                    tl.store(dLam_ptr + flat, dt_h * (A_h - Mp))
                bh_base = b_base + (h // hpg_b) * s_bh
                ch_base = c_base + (h // hpg_c) * s_ch
                if HAS_THH:
                    th_h = tl.load(ThH_ptr + h * (N // 2) + offs_n // 2)
                    cth = tl.cos(th_h)
                    sth = sgn * tl.sin(th_h)
                    # dnu accumulates as a TILE and is reduced over t ONCE
                    # per head; the cross-thread `tl.sum(., axis=0)` is the
                    # expensive part, so R*O of them per head is not paid.
                    dth_tile = tl.zeros([TS, N], dtype=tl.float32)
                # ── pass over r: dB~[:, r]  (and dbeta, and dnu's operand)
                for r in tl.range(R, loop_unroll_factor=1):
                    Bh = tl.load(bh_base + offs_n[None, :] * s_bn
                                 + r * s_br).to(tl.float32)
                    if HAS_BETA:
                        Bh = Bh + tl.load(Beta_ptr + h * N * R + offs_n * R + r
                                          ).to(tl.float32)[None, :]
                    dBh = tl.zeros([TS, N], dtype=tl.float32)
                    for oi in tl.range(O, loop_unroll_factor=1):
                        dd = (n_h * tl.load(sd_base + l * s_sl
                                             + oi * s_so + r * s_sr))
                        Ch = tl.load(ch_base + offs_n[None, :] * s_cn
                                     + oi * s_co).to(tl.float32)
                        if HAS_GAMMA:
                            Ch = Ch + tl.load(
                                Gamma_ptr + h * N * O + offs_n * O + oi
                            ).to(tl.float32)[None, :]
                        if HAS_THH:
                            Chx = tl.load(ch_base + (offs_n ^ 1)[None, :] * s_cn
                                          + oi * s_co).to(tl.float32)
                            if HAS_GAMMA:
                                Chx = Chx + tl.load(
                                    Gamma_ptr + h * N * O
                                    + (offs_n ^ 1) * O + oi
                                ).to(tl.float32)[None, :]
                            dBh += dd[:, None] * (cth[None, :] * Ch
                                                  + sth[None, :] * Chx)
                            # d d_t / d nu_h: sgn_n B~[n, r] C~x[n, o]
                            dth_tile += (dd[:, None] * Bh
                                         * (cth[None, :] * Chx
                                            - sth[None, :] * Ch))
                        else:
                            dBh += dd[:, None] * Ch
                    if HAS_BETA:
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r,
                                      tl.sum(dBh, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dB_ptr + b * s_dbb + c * s_dbc
                                  + (h // hpg_b) * s_dbh
                                  + offs_t[:, None] * s_dbt
                                  + offs_n[None, :] * s_dbn
                                  + r * s_dbr, dBh, sem=ACC_SEM)
                if HAS_THH:
                    tl.atomic_add(dThH_ptr + h * (N // 2) + offs_n // 2,
                                  tl.sum(dth_tile * sgn[None, :], axis=0),
                                  sem=ACC_SEM)
                # ── pass over o: dC~[:, o], un-rotated by nu_h ────────────
                for o in tl.range(O, loop_unroll_factor=1):
                    dCh = tl.zeros([TS, N], dtype=tl.float32)
                    for ri in tl.range(R, loop_unroll_factor=1):
                        dd = (n_h * tl.load(sd_base + l * s_sl
                                             + o * s_so + ri * s_sr))
                        Bh = tl.load(bh_base + offs_n[None, :] * s_bn
                                     + ri * s_br).to(tl.float32)
                        if HAS_BETA:
                            Bh = Bh + tl.load(
                                Beta_ptr + h * N * R + offs_n * R + ri
                            ).to(tl.float32)[None, :]
                        if HAS_THH:
                            Bhx = tl.load(bh_base + (offs_n ^ 1)[None, :] * s_bn
                                          + ri * s_br).to(tl.float32)
                            if HAS_BETA:
                                Bhx = Bhx + tl.load(
                                    Beta_ptr + h * N * R
                                    + (offs_n ^ 1) * R + ri
                                ).to(tl.float32)[None, :]
                            dCh += dd[:, None] * (cth[None, :] * Bh
                                                  - sth[None, :] * Bhx)
                        else:
                            dCh += dd[:, None] * Bh
                    if HAS_GAMMA:
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o,
                                      tl.sum(dCh, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dC_ptr + b * s_dcb + c * s_dcc
                                  + (h // hpg_c) * s_dch
                                  + offs_t[:, None] * s_dct
                                  + offs_n[None, :] * s_dcn
                                  + o * s_dco, dCh, sem=ACC_SEM)
        return


    Bt = tl.load(B_ptr + b * s_bb + c * s_bc
                 + offs_t[:, None] * s_bt
                 + offs_n[None, :] * s_bn).to(tl.float32)
    Ct = tl.load(C_ptr + b * s_cb + c * s_cc
                 + offs_t[:, None] * s_ct
                 + offs_n[None, :] * s_cn).to(tl.float32)

    dB_add = tl.zeros([TS, N], dtype=tl.float32)
    dC_add = tl.zeros([TS, N], dtype=tl.float32)
    ddsum = tl.zeros([TS], dtype=tl.float32)

    if GROUPED:
        for h in range(0, H):
            dt1 = tl.load(DW_ptr + b * s_wb + h * s_wh + l * s_wl
                          ).to(tl.float32)
            lam1 = tl.load(Lam_ptr + b * s_lb + h * s_lh + l * s_ll
                           ).to(tl.float32)
            flat1 = (b * H + h) * L + l
            n1 = tl.load(Own_ptr + flat1)
            A1 = tl.load(A_ptr + flat1)
            s1 = tl.load(S_ptr + flat1)
            pm1 = l > 0
            flatp1 = (b * H + h) * L + (l - 1)
            Ap1 = tl.load(A_ptr + flatp1, mask=pm1, other=0.0)
            d1 = tl.load(Dot_ptr + b * s_qb + h * s_qh + l * s_ql
                         ).to(tl.float32)
            Mp1 = Ap1  # TRAPFOLD: M_{t-1} = A_{t-1}; dL/down_t = A_t + d_t s_t
            A1 = A1 + d1 * s1
            tl.store(dDW_ptr + flat1, A1 * lam1 + Mp1 * (1.0 - lam1))
            if NEED_DLAM:
                tl.store(dLam_ptr + flat1, dt1 * (A1 - Mp1))
            dd1 = n1 * s1                                  # [T]
            if B_GROUPED:
                Bh = tl.load(B_ptr + b * s_bb + c * s_bc
                             + (h // hpg_b) * s_bh
                             + offs_t[:, None] * s_bt
                             + offs_n[None, :] * s_bn).to(tl.float32)
            else:
                Bh = Bt
            if C_GROUPED:
                Ch = tl.load(C_ptr + b * s_cb + c * s_cc
                             + (h // hpg_c) * s_ch
                             + offs_t[:, None] * s_ct
                             + offs_n[None, :] * s_cn).to(tl.float32)
            else:
                Ch = Ct
            if HAS_BETA:
                Bh = Bh + tl.load(Beta_ptr + h * N + offs_n
                                  ).to(tl.float32)[None, :]
            if HAS_GAMMA:
                Ch = Ch + tl.load(Gamma_ptr + h * N + offs_n
                                  ).to(tl.float32)[None, :]
            if HAS_THH:
                # pair partner tiles (n ^ 1), same bias fold
                if C_GROUPED:
                    Chx = tl.load(C_ptr + b * s_cb + c * s_cc
                                  + (h // hpg_c) * s_ch
                                  + offs_t[:, None] * s_ct
                                  + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                else:
                    Chx = tl.load(C_ptr + b * s_cb + c * s_cc
                                  + offs_t[:, None] * s_ct
                                  + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                if B_GROUPED:
                    Bhx = tl.load(B_ptr + b * s_bb + c * s_bc
                                  + (h // hpg_b) * s_bh
                                  + offs_t[:, None] * s_bt
                                  + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                else:
                    Bhx = tl.load(B_ptr + b * s_bb + c * s_bc
                                  + offs_t[:, None] * s_bt
                                  + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                if HAS_GAMMA:
                    Chx = Chx + tl.load(Gamma_ptr + h * N + (offs_n ^ 1)
                                        ).to(tl.float32)[None, :]
                if HAS_BETA:
                    Bhx = Bhx + tl.load(Beta_ptr + h * N + (offs_n ^ 1)
                                        ).to(tl.float32)[None, :]
                # fold convention (kernels / _rot_fold): e' = c e + s o,
                # o' = -s e + c o  ->  v' = c v + sign * s * v_xor
                th_h = tl.load(ThH_ptr + h * (N // 2) + offs_n // 2)
                sgn = tl.where((offs_n % 2) == 0, 1.0, -1.0)
                cth = tl.cos(th_h)
                sth = sgn * tl.sin(th_h)
                Crot = cth[None, :] * Ch + sth[None, :] * Chx      # R(nu)^T (C+gamma)
                Crot_x = cth[None, :] * Chx - sth[None, :] * Ch    # its pair partner
                # d d_t / d(B+beta) = Crot ; d d_t / d(C+gamma) = R(nu)(B+beta)
                dBh = dd1[:, None] * Crot
                dCh = dd1[:, None] * (cth[None, :] * Bh - sth[None, :] * Bhx)
                # d d_t / d nu_h[k] = sum_{n in pair k} sgn_n (B+beta)_n Crot_{n^1}
                dth = tl.sum(dd1[:, None] * sgn[None, :] * Bh * Crot_x, axis=0)
                tl.atomic_add(dThH_ptr + h * (N // 2) + offs_n // 2, dth, sem=ACC_SEM)
            else:
                dBh = dd1[:, None] * Ch                            # [T, N]
                dCh = dd1[:, None] * Bh
            if HAS_GAMMA:
                tl.atomic_add(dGamma_ptr + h * N + offs_n, tl.sum(dCh, axis=0), sem=ACC_SEM)
            if HAS_BETA:
                tl.atomic_add(dBeta_ptr + h * N + offs_n, tl.sum(dBh, axis=0), sem=ACC_SEM)
            if B_GROUPED:
                tl.atomic_add(dB_ptr + b * s_dbb + c * s_dbc
                              + (h // hpg_b) * s_dbh
                              + offs_t[:, None] * s_dbt
                              + offs_n[None, :] * s_dbn, dBh, sem=ACC_SEM)
            else:
                dB_add += dBh
            if C_GROUPED:
                tl.atomic_add(dC_ptr + b * s_dcb + c * s_dcc
                              + (h // hpg_c) * s_dch
                              + offs_t[:, None] * s_dct
                              + offs_n[None, :] * s_dcn, dCh, sem=ACC_SEM)
            else:
                dC_add += dCh
        if not B_GROUPED:
            tl.atomic_add(dB_ptr + b * s_dbb + c * s_dbc
                          + offs_t[:, None] * s_dbt
                          + offs_n[None, :] * s_dbn, dB_add, sem=ACC_SEM)
        if not C_GROUPED:
            tl.atomic_add(dC_ptr + b * s_dcb + c * s_dcc
                          + offs_t[:, None] * s_dct
                          + offs_n[None, :] * s_dcn, dC_add, sem=ACC_SEM)
        return


    if HAS_THH:
        # ── vectorised HB path for the theta_h-only case ──
        # B/C are head-shared here (not B_GROUPED / C_GROUPED), so the
        # per-pair rotation `cth, sth` is the only per-head factor and it
        # is DIAGONAL in n: d_t = (B+beta_h) . R(theta_h)^T (C+gamma_h)
        # expands into six [T, N] x [HB, T] contractions (`tl.dot`, one
        # MMA per term) instead of the per-head scalar loop -- the same
        # trick the HB path below already uses for the plain beta/gamma
        # case, extended through the rotation's algebra.  `dGamma` / `dBeta` /
        # `dThH` still atomic-add per (CTA, head-block, n), but now once
        # per HB heads instead of once per head, and into a SLOT plane
        # (`bslot = c % BSLOT`) so the B * nc contention on those tiny
        # [H, N] / [H, N/2] buffers is divided by BSLOT; the host sums
        # the slot axis after the launch.
        Btx = tl.load(B_ptr + b * s_bb + c * s_bc
                     + offs_t[:, None] * s_bt
                     + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
        Ctx = tl.load(C_ptr + b * s_cb + c * s_cc
                     + offs_t[:, None] * s_ct
                     + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
        BtCt = Bt * Ct
        BtCtx = Bt * Ctx
        sgn = tl.where((offs_n % 2) == 0, 1.0, -1.0)
        bslot = c % BSLOT

        for h0 in range(0, H, HB):
            hs = h0 + tl.arange(0, HB)                     # [HB]
            hm = (hs < H)[:, None]
            dt1 = tl.load(DW_ptr + b * s_wb + hs[:, None] * s_wh
                         + l[None, :] * s_wl, mask=hm, other=0.0).to(tl.float32)
            lam1 = tl.load(Lam_ptr + b * s_lb + hs[:, None] * s_lh
                          + l[None, :] * s_ll, mask=hm, other=0.0).to(tl.float32)
            flat = (b * H + hs[:, None]) * L + l[None, :]
            n1 = tl.load(Own_ptr + flat, mask=hm, other=0.0)
            A1 = tl.load(A_ptr + flat, mask=hm, other=0.0)
            s1 = tl.load(S_ptr + flat, mask=hm, other=0.0)
            pm = hm & (l[None, :] > 0)
            flatp = (b * H + hs[:, None]) * L + (l[None, :] - 1)
            Ap1 = tl.load(A_ptr + flatp, mask=pm, other=0.0)
            d1 = tl.load(Dot_ptr + b * s_qb + hs[:, None] * s_qh
                         + l[None, :] * s_ql, mask=hm, other=0.0
                         ).to(tl.float32)
            Mp1 = Ap1  # TRAPFOLD: M_{t-1} = A_{t-1}; dL/down_t = A_t + d_t s_t
            A1 = A1 + d1 * s1
            tl.store(dDW_ptr + flat, A1 * lam1 + Mp1 * (1.0 - lam1), mask=hm)
            if NEED_DLAM:
                tl.store(dLam_ptr + flat, dt1 * (A1 - Mp1), mask=hm)

            dd = n1 * s1                                 # [HB, T]
            ddrow = tl.sum(dd, axis=1)                       # [HB]
            ddT = tl.trans(dd)                               # [T, HB]

            beta_h = tl.zeros([HB, N], dtype=tl.float32)
            betax_h = tl.zeros([HB, N], dtype=tl.float32)
            gamma_h = tl.zeros([HB, N], dtype=tl.float32)
            gammax_h = tl.zeros([HB, N], dtype=tl.float32)
            if HAS_BETA:
                beta_h = tl.load(Beta_ptr + hs[:, None] * N + offs_n[None, :],
                                 mask=hm, other=0.0).to(tl.float32)
                betax_h = tl.load(Beta_ptr + hs[:, None] * N + (offs_n ^ 1)[None, :],
                                  mask=hm, other=0.0).to(tl.float32)
            if HAS_GAMMA:
                gamma_h = tl.load(Gamma_ptr + hs[:, None] * N + offs_n[None, :],
                                  mask=hm, other=0.0).to(tl.float32)
                gammax_h = tl.load(Gamma_ptr + hs[:, None] * N + (offs_n ^ 1)[None, :],
                                   mask=hm, other=0.0).to(tl.float32)
            th_h = tl.load(ThH_ptr + hs[:, None] * (N // 2) + (offs_n // 2)[None, :],
                          mask=hm, other=0.0).to(tl.float32)
            cth = tl.cos(th_h)
            sth = sgn[None, :] * tl.sin(th_h)

            P_B = pdot(dd, Bt, input_precision=PREC)
            P_Bx = pdot(dd, Btx, input_precision=PREC)
            P_C = pdot(dd, Ct, input_precision=PREC)
            P_Cx = pdot(dd, Ctx, input_precision=PREC)
            P_BC = pdot(dd, BtCt, input_precision=PREC)
            P_BCx = pdot(dd, BtCtx, input_precision=PREC)

            dGamma_hb = (cth * (P_B + beta_h * ddrow[:, None])
                        - sth * (P_Bx + betax_h * ddrow[:, None]))
            dBeta_hb = (cth * (P_C + gamma_h * ddrow[:, None])
                       + sth * (P_Cx + gammax_h * ddrow[:, None]))
            dth_full = sgn[None, :] * (
                cth * (P_BCx + gammax_h * P_B + beta_h * P_Cx
                       + beta_h * gammax_h * ddrow[:, None])
                - sth * (P_BC + gamma_h * P_B + beta_h * P_C
                         + beta_h * gamma_h * ddrow[:, None]))

            if HAS_GAMMA:
                tl.atomic_add(dGamma_ptr + bslot * s_dgs
                             + hs[:, None] * N + offs_n[None, :],
                             dGamma_hb, mask=hm, sem=ACC_SEM)
            if HAS_BETA:
                tl.atomic_add(dBeta_ptr + bslot * s_dbes
                             + hs[:, None] * N + offs_n[None, :],
                             dBeta_hb, mask=hm, sem=ACC_SEM)
            tl.atomic_add(dThH_ptr + bslot * s_dths
                         + hs[:, None] * (N // 2) + (offs_n // 2)[None, :],
                         dth_full, mask=hm, sem=ACC_SEM)

            cth_g = cth * gamma_h
            sth_gx = sth * gammax_h
            # dB_add is sum_h dBh (dBh = dd * Crot = dd * (cth*Ch + sth*Chx),
            # Ch/Chx built from C/gamma) -- NOT from Bt, even though the
            # gradient lands on B.  dC_add is sum_h dCh (dCh = dd * (cth*Bh
            # - sth*Bhx), built from B/beta).  Mirrors dBh/dCh above.
            dB_add += (Ct * pdot(ddT, cth, input_precision=PREC)
                      + pdot(ddT, cth_g, input_precision=PREC)
                      + Ctx * pdot(ddT, sth, input_precision=PREC)
                      + pdot(ddT, sth_gx, input_precision=PREC))
            cth_b = cth * beta_h
            sth_bx = sth * betax_h
            dC_add += (Bt * pdot(ddT, cth, input_precision=PREC)
                      + pdot(ddT, cth_b, input_precision=PREC)
                      - Btx * pdot(ddT, sth, input_precision=PREC)
                      - pdot(ddT, sth_bx, input_precision=PREC))

        tl.atomic_add(dB_ptr + b * s_dbb + c * s_dbc
                      + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn,
                      dB_add, sem=ACC_SEM)
        tl.atomic_add(dC_ptr + b * s_dcb + c * s_dcc
                      + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
                      dC_add, sem=ACC_SEM)
        return

    for h0 in range(0, H, HB):
        hs = h0 + tl.arange(0, HB)                        # [HB]
        hm = (hs < H)[:, None]
        dt = tl.load(DW_ptr + b * s_wb + hs[:, None] * s_wh
                     + l[None, :] * s_wl, mask=hm, other=0.0).to(tl.float32)
        lam = tl.load(Lam_ptr + b * s_lb + hs[:, None] * s_lh
                      + l[None, :] * s_ll, mask=hm, other=0.0).to(tl.float32)
        dot = tl.load(Dot_ptr + b * s_qb + hs[:, None] * s_qh
                      + l[None, :] * s_ql, mask=hm, other=0.0).to(tl.float32)
        flat = (b * H + hs[:, None]) * L + l[None, :]
        n = tl.load(Own_ptr + flat, mask=hm, other=0.0)
        A = tl.load(A_ptr + flat, mask=hm, other=0.0)
        s = tl.load(S_ptr + flat, mask=hm, other=0.0)
        # M_{t-1}: A and s are COMPLETE in global memory (Kc's atomics are
        # done), so the one-step shift is a masked load, not a scan -- and
        # the t = 0 tap reaches into the PREVIOUS chunk, which is why the
        # write-scale gradient goes to a SEPARATE buffer, not in place.
        pm = hm & (l[None, :] > 0)
        flatp = (b * H + hs[:, None]) * L + (l[None, :] - 1)
        Ap = tl.load(A_ptr + flatp, mask=pm, other=0.0)
        Mp = Ap  # TRAPFOLD: M_{t-1} = A_{t-1}; dL/down_t = A_t + d_t s_t
        A = A + dot * s
        tl.store(dDW_ptr + flat, A * lam + Mp * (1.0 - lam), mask=hm)
        if NEED_DLAM:
            tl.store(dLam_ptr + flat, dt * (A - Mp), mask=hm)

        dd = n * s                                     # [HB, T]
        ddsum += tl.sum(dd, axis=0)
        ddrow = tl.sum(dd, axis=1)                        # [HB]
        ddT = tl.trans(dd)                                # [T, HB]
        gam = tl.zeros([HB, N], dtype=tl.float32)
        bet = tl.zeros([HB, N], dtype=tl.float32)
        if HAS_GAMMA:
            gam = tl.load(Gamma_ptr + hs[:, None] * N + offs_n[None, :],
                          mask=hm, other=0.0).to(tl.float32)
        if HAS_BETA:
            bet = tl.load(Beta_ptr + hs[:, None] * N + offs_n[None, :],
                          mask=hm, other=0.0).to(tl.float32)
        if SCHED_AC:
            # ── `AC` schedule (default OFF; a static no-op) ──
            # ptxas reports `C7515` here: the two `dg += bet * ddrow` /
            # `db += gam * ddrow` corrections are FFMA writes to a wgmma
            # accumulator in between the four MMAs of the head block.  The
            # four dots are issued together and every non-wgmma accumulator
            # write happens AFTER the last of them (same dots, operands and
            # per-value addition order: bitwise identical).  Triton had
            # already scheduled it this way, so the diagnostic does not come
            # from the source order (see the warp-count note above).
            if HAS_GAMMA:
                dBd = pdot(ddT, gam, input_precision=PREC)
            if HAS_BETA:
                dCd = pdot(ddT, bet, input_precision=PREC)
            if HAS_GAMMA:
                dg = pdot(dd, Bt, input_precision=PREC)
            if HAS_BETA:
                db = pdot(dd, Ct, input_precision=PREC)
            if HAS_GAMMA:
                dB_add += dBd
            if HAS_BETA:
                dC_add += dCd
            if HAS_GAMMA:
                if HAS_BETA:
                    dg += bet * ddrow[:, None]
                tl.atomic_add(dGamma_ptr + hs[:, None] * N + offs_n[None, :],
                              dg, mask=hm, sem=ACC_SEM)
            if HAS_BETA:
                if HAS_GAMMA:
                    db += gam * ddrow[:, None]
                tl.atomic_add(dBeta_ptr + hs[:, None] * N + offs_n[None, :],
                              db, mask=hm, sem=ACC_SEM)
        else:
            if HAS_GAMMA:
                dB_add += pdot(ddT, gam, input_precision=PREC)
            if HAS_BETA:
                dC_add += pdot(ddT, bet, input_precision=PREC)
            if HAS_GAMMA:
                dg = pdot(dd, Bt, input_precision=PREC)
                if HAS_BETA:
                    dg += bet * ddrow[:, None]
                tl.atomic_add(dGamma_ptr + hs[:, None] * N + offs_n[None, :],
                              dg, mask=hm, sem=ACC_SEM)
            if HAS_BETA:
                db = pdot(dd, Ct, input_precision=PREC)
                if HAS_GAMMA:
                    db += gam * ddrow[:, None]
                tl.atomic_add(dBeta_ptr + hs[:, None] * N + offs_n[None, :],
                              db, mask=hm, sem=ACC_SEM)

    dB_add += ddsum[:, None] * Ct
    dC_add += ddsum[:, None] * Bt
    tl.atomic_add(dB_ptr + b * s_dbb + c * s_dbc
                  + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn,
                  dB_add, sem=ACC_SEM)
    tl.atomic_add(dC_ptr + b * s_dcb + c * s_dcc
                  + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn,
                  dC_add, sem=ACC_SEM)


_TRAP_STAGE_CACHE: dict = {}


_THH_MERGE_DEFAULT = 0


def _thh_merge():
    """Which MMA merge the GEN_VEC theta_h fold uses.

    `SSD_MIMO_TRAP_THH_MERGE`, a bitmask: 1 = the `ddT`-side N-merge
    (four right operands concatenated on N, one 4N-wide MMA), 2 = the
    `dd`-side N-merge (two 2N-wide MMAs instead of four), 4 = the
    (h, o) K-merge of the gamma / beta terms.  0 = the one-MMA-per-term
    form.  Bits 1 and 2 are bit-identical to 0; bit 4 re-associates the
    o-sum inside one MMA accumulator.
    """
    return int(_env("TRAP_THH_MERGE", str(_THH_MERGE_DEFAULT)))


def _rot_abl_mimo():
    import os as _o
    return int(_o.environ.get("SSD_MIMO_BWD_ROT_ABL_MIMO", "0"))


def trap_adjoint(Btil, Ctil, beta, gamma, dw, trap_lam, trap_dot,
                 trap_nxt, dDW, dSdot, dBtil, dCtil, dBeta, dGamma,
                 nc, T, N, precision, need_dlam,
                 bc_grouped=False, Gb=1, Gc=1, theta_h=None, dThetaH=None,
                 pair_half=False):
    """The trapezoid adjoint epilogue, in one launch.

    `dDW` comes in as `A_t = dL/dw_t` (Kc's atomic accumulation) and a
    NEW buffer comes back holding `ddt_t`, separate because the t = 0
    tap of each chunk reads the PREVIOUS chunk's `A`.

    `bc_grouped`: B/C (and dB/dC) carry a group axis at dim 2,
    `[B, nc, Gx, T, N]`.
    """
    Bsz, H, L = dw.shape[0], dw.shape[1], nc * T
    f32 = torch.float32
    dDW_out = torch.empty(Bsz, H, L, device=dw.device, dtype=f32)
    dLam_out = (torch.empty(Bsz, H, L, device=dw.device, dtype=f32)
                if need_dlam else dDW_out)
    # The (O, R) tail.  `trap_dot` is [B, H, L] at R = O = 1 and
    # [B, H, L, O, R] above it -- it is the ONE
    # operand that carries both ranks, so it is the shape this reads them
    # off (`fused_fwd_kernel.fused_chunk_fwd`'s `has_or_tail`).
    gen_or = trap_dot.dim() == 5
    O_rank, R_rank = (trap_dot.shape[3], trap_dot.shape[4]) if gen_or else (1, 1)
    if bc_grouped:
        def _g(t, grouped):
            st = list(t.stride())
            return ((st[0], st[1], st[3], st[4]), st[2] if grouped else 0,
                    st[5] if len(st) > 5 else 0)
        b_grouped, c_grouped = Gb > 1, Gc > 1
        sb, s_bh, s_br = _g(Btil, b_grouped)
        sc, s_ch, s_co = _g(Ctil, c_grouped)
        sdb, s_dbh, s_dbr = _g(dBtil, b_grouped)
        sdc, s_dch, s_dco = _g(dCtil, c_grouped)
        hpg_b, hpg_c = max(H // Gb, 1), max(H // Gc, 1)
    else:
        # A trailing length-one R/O axis is a view axis: the first four
        # strides are the whole address at R = O = 1.
        def _u(t):
            st = list(t.stride())
            return tuple(st[:4]), (st[4] if len(st) > 4 else 0)
        sb, s_br = _u(Btil)
        sc, s_co = _u(Ctil)
        sdb, s_dbr = _u(dBtil)
        sdc, s_dco = _u(dCtil)
        b_grouped = c_grouped = False
        s_bh = s_ch = s_dbh = s_dch = 0
        hpg_b = hpg_c = H
    sw, sl, sq = dw.stride(), trap_lam.stride(), trap_dot.stride()
    gen_vec = gen_or and not c_grouped
    HB = 16 if H <= 16 else (32 if H <= 32 else 64)
    # The head-vectorised general-(O, R) branch keeps up to
    # 2(1 + HAS_BETA + HAS_GAMMA) [HB, N] per-head operands live per
    # (o, r) MMA, so the block that is right at R = O = 1 is too wide at
    # R*O = 16.  Same shape of rule as the other paths' (a static table,
    # not a search), overridable by name for a sweep.
    if gen_vec:
        _or = O_rank * R_rank
        _cap = 64 if _or <= 2 else (32 if _or <= 8 else 16)
        if b_grouped:
            # the group walk runs `hpg_b` heads per group, so a block
            # wider than that is masked-off lanes, not work.
            _cap = min(_cap, 1 << max(hpg_b - 1, 0).bit_length())
        # ── the theta_h fold wants the NARROWEST legal block ──
        #
        # `_cap` above is the rule for the TWO-MMA body.  With the static
        # read offset compiled in, this branch emits 9 MMAs per (head
        # block, o, r) in pass 1 and 7 in pass 2 and holds SIX
        # loop-invariant `[TS, N]` operand tiles plus `[HB, N]` fp32
        # dBeta / dGamma / dThH accumulators live across the whole HB
        # loop.  It is ACCUMULATOR- AND LAYOUT-BOUND, not MMA-issue-bound,
        # so HB (which scales the live accumulators linearly and the MMA
        # count inversely) is the lever, and narrower wins (measured: HB
        # 16 vs 32 is 3.5x on this kernel at N = 128, R = O = 2).
        #
        # The non-THH body keeps `_cap`.  `SSD_MIMO_TRAP_HB_OR` still wins
        # over both.
        # The predicate is the kernel's OWN `HAS_THH` argument, spelled
        # out here because `has_thh` is bound below, next to the
        # accumulators it guards -- including the `ROT_ABL_MIMO & 4`
        # ablation, so that arm keeps measuring the non-THH body at the
        # non-THH block width and stays comparable across this change.
        if theta_h is not None and not (_rot_abl_mimo() & 4):
            _cap = 16
        HB = max(16, min(HB, int(_env("TRAP_HB_OR", str(_cap)))))
    # Resource legality, not a Torch fallback: the default 3-stage
    # pipeline can exceed Blackwell shared memory for an N=128 tile.
    # OutOfResources is raised before launch, so none of these additive
    # gradient buffers has been mutated when we retry fewer stages.
    from triton.runtime.errors import OutOfResources
    has_thh = theta_h is not None
    if has_thh and dThetaH is None:
        raise ValueError("trap_adjoint: theta_h needs its dThetaH accumulator")
    # ── THH slot buffers: only the vectorised HAS_THH-only route (not
    # B_GROUPED / C_GROUPED, which stays on the per-head atomic path) is
    # contended enough at [H, N] / [H, N//2] to be worth slotting.
    # GEN_OR takes the per-head atomic path when B/C are
    # GROUPED and never reads a slot plane, so the grouped maps are
    # excluded here; the head-vectorised GEN_VEC branch emits
    # its dBeta/dGamma/dThH once per HEAD BLOCK exactly as the R = O = 1
    # HAS_THH path does, so it slots with it.
    use_slots = has_thh and not c_grouped and (gen_vec or not b_grouped)
    bslot_n = trap_bslots(nc) if use_slots else 1
    has_beta, has_gamma = dBeta is not None, dGamma is not None
    # dBeta / dGamma / dThetaH are the CALLER's accumulators, whose Python
    # shape carries whatever trailing R / O axis the caller's convention
    # uses (`[H, N, R]` / `[H, N, O]`, R = O = 1 collapsing to a
    # contiguous [H, N] the kernel's flat `h * N + n` addressing already
    # assumes) -- the slot plane must match that SHAPE exactly (not just
    # H * N), or the host-side `+=` after the launch broadcasts wrong.
    dBeta_slots = (torch.zeros((bslot_n,) + tuple(dBeta.shape),
                               device=dw.device, dtype=f32)
                  if use_slots and has_beta else dBeta)
    dGamma_slots = (torch.zeros((bslot_n,) + tuple(dGamma.shape),
                                device=dw.device, dtype=f32)
                    if use_slots and has_gamma else dGamma)
    dThH_slots = (torch.zeros((bslot_n,) + tuple(dThetaH.shape),
                              device=dw.device, dtype=f32)
                 if use_slots else dThetaH)
    s_dgs = dGamma_slots.stride(0) if use_slots and has_gamma else 0
    s_dbes = dBeta_slots.stride(0) if use_slots and has_beta else 0
    s_dths = dThH_slots.stride(0) if use_slots else 0
    # dSdot carries the (O, R) tail too (the prologue built it in
    # the saved dot's layout); at R = O = 1 the kernel addresses it FLAT
    # and these five strides are never read.
    ss = dSdot.stride() if gen_or else (0, 0, 0, 0, 0)
    n_warps = trap_warps_or() if gen_or else trap_warps()
    GB_rank = Gb if b_grouped else 1
    key = (dw.device.index, T, N, H, L, HB, precision, Btil.dtype, Ctil.dtype,
           beta is not None, gamma is not None, need_dlam,
           trap_sched(), n_warps, b_grouped, c_grouped, has_thh, bslot_n,
           O_rank, R_rank, hpg_b if gen_vec else 0,
           _thh_merge() if gen_vec else 0)
    if _env("TRAP_DEBUG", "0") == "1" and key not in _TRAP_STAGE_CACHE:
        import sys as _sys
        print(f"[trap_adjoint] H={H} T={T} N={N} O={O_rank} R={R_rank} "
              f"gen_or={gen_or} b_grouped={b_grouped} c_grouped={c_grouped} "
              f"has_thh={has_thh} HB={HB} warps={n_warps} "
              f"gen_vec={gen_vec} GB={GB_rank} hpg_b={hpg_b}",
              file=_sys.stderr, flush=True)
    # T-halving.  `stages` alone does not reach an
    # OutOfResources whose SOURCE is the HAS_THH branch's six [T, N]
    # loop-invariant operand tiles (Bt/Ct/Btx/Ctx/BtCt/BtCtx) -- those are
    # live for the whole HB loop regardless of pipeline depth, so at a
    # wide N tile (a 128-wide plane) even num_stages=1
    # can overflow shared memory.  This kernel has no recurrence WITHIN a
    # chunk (A/S/Nxt/dot are read, not scanned), so retrying with T split
    # into `n_th` disjoint pieces over a third grid axis is exact, not an
    # approximation -- see `_trap_adjoint_kernel`'s TS.  Cached per `key`
    # as (stages, n_th) so a repeat call does not re-pay a failed
    # autotune, mirroring `_KB1_HB_OOR` / `_TRAP_STAGE_CACHE`'s own use.
    cached_stages, cached_nth = _TRAP_STAGE_CACHE.get(key, (3, 1))
    n_th = cached_nth
    last_exc = None
    while True:
        TS = T // n_th
        for stages in range(cached_stages if n_th == cached_nth else 3, 0, -1):
            try:
                _trap_adjoint_kernel[(Bsz, nc, n_th)](
                    Btil, Ctil,
                    beta if beta is not None else Btil,
                    gamma if gamma is not None else Btil,
                    dw, trap_lam, trap_dot, trap_nxt, dDW, dSdot,
                    dDW_out, dLam_out,
                    dBtil, dCtil,
                    dBeta_slots if dBeta_slots is not None else dDW_out,
                    dGamma_slots if dGamma_slots is not None else dDW_out,
                    theta_h if has_thh else dDW_out,
                    dThH_slots if has_thh else dDW_out,
                    H, L,
                    sb[0], sb[1], sb[2], sb[3],
                    sc[0], sc[1], sc[2], sc[3],
                    sdb[0], sdb[1], sdb[2], sdb[3],
                    sdc[0], sdc[1], sdc[2], sdc[3],
                    sw[0], sw[1], sw[2],
                    sl[0], sl[1], sl[2],
                    sq[0], sq[1], sq[2],
                    s_bh, s_ch, s_dbh, s_dch, hpg_b, hpg_c,
                    s_br, s_co, s_dbr, s_dco,
                    sq[3] if gen_or else 0, sq[4] if gen_or else 0,
                    ss[0], ss[1], ss[2], ss[3], ss[4],
                    s_dgs, s_dbes, s_dths, bslot_n,
                    T=T, N=N, HB=HB, TS=TS, PREC=precision,
                    HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None,
                    NEED_DLAM=need_dlam,
                    SCHED=trap_sched(),
                    B_GROUPED=b_grouped, C_GROUPED=c_grouped,
                    # Bit 4 of SSD_MIMO_BWD_ROT_ABL_MIMO compiles the adjoint as
                    # if theta_h were absent -- MEASUREMENT ONLY (wrong
                    # gradient), to price the HAS_THH fold.  0 = off.
                    HAS_THH=bool(has_thh and not (_rot_abl_mimo() & 4)),
                    O=O_rank, R=R_rank, GB=GB_rank,
                    THH_W=_thh_merge(),
                    PAIR_HALF=bool(pair_half),
                    num_warps=n_warps, num_stages=stages,
                )
            except OutOfResources as e:
                last_exc = e
                continue
            else:
                _TRAP_STAGE_CACHE[key] = (stages, n_th)
                break
        else:
            # every stage count OOM'd at this T-split -- halve T again if
            # there is room (TS stays a power of two, >= 16 for tl.dot).
            if TS % 2 == 0 and TS // 2 >= 16:
                n_th *= 2
                continue
            raise last_exc
        break
    if use_slots:
        if has_beta:
            dBeta += dBeta_slots.sum(0).view(dBeta.shape)
        if has_gamma:
            dGamma += dGamma_slots.sum(0).view(dGamma.shape)
        dThetaH += dThH_slots.sum(0).view(dThetaH.shape)
    return dDW_out, (dLam_out if need_dlam else None)



# ══════════════════════════════════════════════════════════════════════
# 5. ROTSPLIT: the rotation as a PROLOGUE + one ADJOINT
# ══════════════════════════════════════════════════════════════════════
#
# Mamba-3's SISO decomposition.  The scan's five backward kernels never
# see an angle: the ROTATED operands are materialised once, the scan
# runs its REAL-mode bodies on them, and ONE small elementwise kernel
# owns every rotation gradient.
#
#   forward   B~[b,l,h,:,r] = R(Theta_{t,h})^T (B[b,l,g_b(h),:,r] + beta[h,:,r])
#             C~[b,l,h,:,o] = R(Theta_{t,h} + nu_h)^T (C[..] + gamma[h,:,o])
#
# -- i.e. the tile the fused forward kernel builds in registers, written
# out instead, at the IO dtype, on a PER KERNEL HEAD axis (the phase is
# per head, so the rotated tile is too).
# The scan is then called with `theta=theta_h=beta=gamma=None` and
# `bias_pre_rot=False` on `bc_grouped=(H, H)` operands, so every kernel
# compiles the branch the REAL (non-rotating) cells compile.
#
#   backward  the scan returns dB~ / dC~ (gradients w.r.t. the ROTATED
#             tiles, including the trapezoid adjoint's contribution,
#             which now runs its real branch because theta_h is folded
#             into C~ already), and `rot_adjoint` turns them into
#
#     dTheta_{t,h,k} = sum_r <dB~^r, J B~^r>_k + sum_o <dC~^o, J C~^o>_k
#     dnu_{h,k}      = sum_{b,t} sum_o <dC~^o, J C~^o>_k
#     d(B+beta)^r    = R(Theta)  dB~^r     (the adjoint of R(Theta)^T)
#     dbeta[h,:,r]   = sum_{b,t} d(B+beta)^r
#     dB[b,l,g,:,r]  = sum_{h in g} d(B+beta)^r
#
# with J the 90-degree pair rotation, J x = (x_odd, -x_even) on the
# interleaved layout and (x_2, -x_1) on the halves.  The derivation is
# one line: with x~ = R(t)^T x, d/dt R(t)^T = J R(t)^T, so
# dL/dt = <dx~, d/dt x~> = <dx~, J x~> -- the partner tile is the
# ROTATED one, which is exactly what the forward saved, so the adjoint
# never needs the raw tile or a second rotation to build its factor.
#
# On the interleaved layout J x is `rot_sign * swap(x)`; under
# `pair_layout="half"` it is the other contiguous half.  Both are
# supported (`PAIR_HALF`); in THIS kernel the half form is a pure win
# (no MMA, so no output width to halve).


@triton.jit
def _rot_pair_split(tile, T: tl.constexpr, N: tl.constexpr,
                    PAIR_HALF: tl.constexpr):
    """[T, N] -> the two pair lanes as [T, N//2] tiles."""
    if PAIR_HALF:
        e, o = tl.split(tl.permute(tl.reshape(tile, (T, 2, N // 2)),
                                   (0, 2, 1)))
    else:
        e, o = tl.split(tl.reshape(tile, (T, N // 2, 2)))
    return e, o


@triton.jit
def _rot_pair_join(e, o, T: tl.constexpr, N: tl.constexpr,
                   PAIR_HALF: tl.constexpr):
    """The inverse of `_rot_pair_split`."""
    if PAIR_HALF:
        return tl.reshape(tl.permute(tl.join(e, o), (0, 2, 1)), (T, N))
    else:
        return tl.reshape(tl.join(e, o), (T, N))


def _rot_flat_ok(st6, rank):
    """Is the (n, r) block of a B/C-shaped tensor ONE contiguous axis?

    `n` stride == `rank` and `r` stride == 1 (the shipped R/O-major
    layout), which is what lets the SLAB branch of the two ROTSPLIT
    kernels address `n*rank + r` as a single coalesced range and take
    the pair partner as `c ^ rank`.
    """
    if rank & (rank - 1):
        return False                      # `c ^ rank` needs a power of two
    if rank == 1:
        # Not taken at R = O = 1 (measured 2x slower): the slab is then
        # the same [T, N] tile the loop form loads, and the partner, which
        # the loop form gets from registers with one `tl.split`, becomes a
        # SECOND global load.  The slab only pays when it merges an actual
        # r / o stack, i.e. above rank 1.
        return False
    return st6[4] == rank and st6[5] == 1


@triton.jit
def _rot_prologue_kernel(
    B_ptr, C_ptr, Beta_ptr, Gamma_ptr, Th_ptr, ThH_ptr,
    Bo_ptr, Co_ptr,
    s_bb, s_bc, s_bh, s_bt, s_bn, s_br,
    s_cb, s_cc, s_ch, s_ct, s_cn, s_co,
    s_qb, s_qc, s_qh, s_qt, s_qn, s_qr,
    s_pb, s_pc, s_ph, s_pt, s_pn, s_po,
    s_thb, s_thc, s_tht, s_thh, s_thk,
    HPG_B: tl.constexpr, HPG_C: tl.constexpr,
    T: tl.constexpr, N: tl.constexpr,
    R: tl.constexpr, O: tl.constexpr,
    HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr,
    HAS_THH: tl.constexpr, PAIR_HALF: tl.constexpr,
    FLAT: tl.constexpr = False,
    NR_B: tl.constexpr = 1, NR_C: tl.constexpr = 1,
):
    """B~ / C~ for one (batch, chunk, kernel head).

    One CTA per (b, c, h); R + O tiles of [T, N] in and out.  The phase
    is the CUMULATED one (`phase_prefix_build`), read on its own pair
    axis exactly as the fused kernels read it.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, N // 2)
    bg = h // HPG_B
    cg = h // HPG_C

    th = tl.load(Th_ptr + b * s_thb + c * s_thc + h * s_thh
                 + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk,
                 ).to(tl.float32)                                # [T, K]
    ph_cos = _cos_approx(th)
    ph_sin = _sin_approx(th)
    if HAS_THH:
        nu = tl.load(ThH_ptr + h * (N // 2) + offs_k).to(tl.float32)
        th_c = th + nu[None, :]
        cr_cos = _cos_approx(th_c)
        cr_sin = _sin_approx(th_c)
    else:
        cr_cos = ph_cos
        cr_sin = ph_sin

    if FLAT:
        # ── the SLAB form ────────
        # (n, r) is ONE contiguous axis of length N*R in every tensor
        # this branch is admitted for, so the whole R (or O) stack is a
        # single fully-coalesced [T, N*R] tile and the r loop disappears.
        # The pair partner of flat column `c = n*R + r` is `c ^ R`
        # (exact for a power-of-two R with r < R), which is a PERMUTATION
        # OF THE SAME 128-byte segments -- another coalesced load, not a
        # cross-lane op.  cos/sin ride a broadcast+reshape (a pure layout
        # op), and the even/odd sign is a compile-time constant vector.
        # The loop form below reads `offs_n * s_bn` with `s_bn = R`, i.e.
        # every load is 1/R-efficient; it serves the layouts this branch
        # refuses.
        offs_f = tl.arange(0, NR_B)
        sgn_b = tl.where((offs_f // R) % 2 == 0, 1.0, -1.0)
        csb = tl.reshape(tl.broadcast_to(ph_cos[:, :, None],
                                         (T, N // 2, 2 * R)), (T, NR_B))
        snb = tl.reshape(tl.broadcast_to(ph_sin[:, :, None],
                                         (T, N // 2, 2 * R)), (T, NR_B))
        base = (B_ptr + b * s_bb + c * s_bc + bg * s_bh
                + offs_t[:, None] * s_bt)
        tile = tl.load(base + offs_f[None, :]).to(tl.float32)
        part = tl.load(base + (offs_f ^ R)[None, :]).to(tl.float32)
        if HAS_BETA:
            be = tl.load(Beta_ptr + h * NR_B + offs_f).to(tl.float32)
            tile = tile + be[None, :]
            part = part + tl.load(
                Beta_ptr + h * NR_B + (offs_f ^ R)).to(tl.float32)[None, :]
        tl.store(Bo_ptr + b * s_qb + c * s_qc + h * s_qh
                 + offs_t[:, None] * s_qt + offs_f[None, :],
                 (csb * tile + sgn_b[None, :] * snb * part
                  ).to(Bo_ptr.dtype.element_ty))

        offs_g = tl.arange(0, NR_C)
        sgn_c = tl.where((offs_g // O) % 2 == 0, 1.0, -1.0)
        csc = tl.reshape(tl.broadcast_to(cr_cos[:, :, None],
                                         (T, N // 2, 2 * O)), (T, NR_C))
        snc = tl.reshape(tl.broadcast_to(cr_sin[:, :, None],
                                         (T, N // 2, 2 * O)), (T, NR_C))
        cbase = (C_ptr + b * s_cb + c * s_cc + cg * s_ch
                 + offs_t[:, None] * s_ct)
        ctile = tl.load(cbase + offs_g[None, :]).to(tl.float32)
        cpart = tl.load(cbase + (offs_g ^ O)[None, :]).to(tl.float32)
        if HAS_GAMMA:
            ga = tl.load(Gamma_ptr + h * NR_C + offs_g).to(tl.float32)
            ctile = ctile + ga[None, :]
            cpart = cpart + tl.load(
                Gamma_ptr + h * NR_C + (offs_g ^ O)).to(tl.float32)[None, :]
        tl.store(Co_ptr + b * s_pb + c * s_pc + h * s_ph
                 + offs_t[:, None] * s_pt + offs_g[None, :],
                 (csc * ctile + sgn_c[None, :] * snc * cpart
                  ).to(Co_ptr.dtype.element_ty))
        return

    for r in tl.range(R, loop_unroll_factor=1):
        tile = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh
                       + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                       + r * s_br).to(tl.float32)
        if HAS_BETA:
            tile = tile + tl.load(
                Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
        e, o = _rot_pair_split(tile, T, N, PAIR_HALF)
        out = _rot_pair_join(ph_cos * e + ph_sin * o,
                             ph_cos * o - ph_sin * e, T, N, PAIR_HALF)
        tl.store(Bo_ptr + b * s_qb + c * s_qc + h * s_qh
                 + offs_t[:, None] * s_qt + offs_n[None, :] * s_qn
                 + r * s_qr,
                 out.to(Bo_ptr.dtype.element_ty))

    for o_i in tl.range(O, loop_unroll_factor=1):
        tile = tl.load(C_ptr + b * s_cb + c * s_cc + cg * s_ch
                       + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn
                       + o_i * s_co).to(tl.float32)
        if HAS_GAMMA:
            tile = tile + tl.load(
                Gamma_ptr + h * N * O + offs_n * O + o_i,
            ).to(tl.float32)[None, :]
        e, o = _rot_pair_split(tile, T, N, PAIR_HALF)
        out = _rot_pair_join(cr_cos * e + cr_sin * o,
                             cr_cos * o - cr_sin * e, T, N, PAIR_HALF)
        tl.store(Co_ptr + b * s_pb + c * s_pc + h * s_ph
                 + offs_t[:, None] * s_pt + offs_n[None, :] * s_pn
                 + o_i * s_po,
                 out.to(Co_ptr.dtype.element_ty))


@triton.jit
def _rot_adjoint_kernel(
    Bt_ptr, Ct_ptr, dBt_ptr, dCt_ptr, Th_ptr, ThH_ptr,
    dB_ptr, dC_ptr, dBeta_ptr, dGamma_ptr, dTh_ptr, dThH_ptr,
    s_qb, s_qc, s_qh, s_qt, s_qn, s_qr,
    s_pb, s_pc, s_ph, s_pt, s_pn, s_po,
    s_ub, s_uc, s_uh, s_ut, s_un, s_ur,
    s_vb, s_vc, s_vh, s_vt, s_vn, s_vo,
    s_bb, s_bc, s_bh, s_bt, s_bn, s_br,
    s_cb, s_cc, s_ch, s_ct, s_cn, s_co,
    s_thb, s_thc, s_tht, s_thh, s_thk,
    s_dtb, s_dtc, s_dtt, s_dth, s_dtk,
    HPG_B: tl.constexpr, HPG_C: tl.constexpr,
    T: tl.constexpr, N: tl.constexpr,
    R: tl.constexpr, O: tl.constexpr,
    HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr,
    HAS_THH: tl.constexpr, NEED_DTHH: tl.constexpr,
    NEED_DTH: tl.constexpr, DTH_ATOMIC: tl.constexpr,
    DB_ATOMIC: tl.constexpr, DC_ATOMIC: tl.constexpr,
    PAIR_HALF: tl.constexpr,
    FLAT: tl.constexpr = False,
    NR_B: tl.constexpr = 1, NR_C: tl.constexpr = 1,
):
    """Every rotation gradient, for one (batch, chunk, kernel head).

    In:  the fwd-saved B~ / C~ and the scan's dB~ / dC~.
    Out: dTheta (per (b, l, h, k), the CUMULATED phase's gradient --
         `phase_prefix_bwd` turns it into the increment's), dnu_h,
         dbeta / dgamma, and dB / dC on the CALLER's (grouped) map,
         summed over the kernel heads that share it.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, N // 2)
    bg = h // HPG_B
    cg = h // HPG_C

    th = tl.load(Th_ptr + b * s_thb + c * s_thc + h * s_thh
                 + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk,
                 ).to(tl.float32)
    ph_cos = _cos_approx(th)
    ph_sin = _sin_approx(th)
    if HAS_THH:
        nu = tl.load(ThH_ptr + h * (N // 2) + offs_k).to(tl.float32)
        th_c = th + nu[None, :]
        cr_cos = _cos_approx(th_c)
        cr_sin = _sin_approx(th_c)
    else:
        cr_cos = ph_cos
        cr_sin = ph_sin

    if FLAT:
        # the SLAB form -- see `_rot_prologue_kernel`'s FLAT branch for
        # why (n, r) is one coalesced axis and why `c ^ R` is the pair
        # partner.  The whole R (and O) stack is one tile, so the two
        # loops, the `tl.split`/`tl.join` pair and the per-r atomics all
        # disappear; dTheta's pair reduction is a reshape + `tl.sum`.
        offs_f = tl.arange(0, NR_B)
        sgn_b = tl.where((offs_f // R) % 2 == 0, 1.0, -1.0)
        csb = tl.reshape(tl.broadcast_to(ph_cos[:, :, None],
                                         (T, N // 2, 2 * R)), (T, NR_B))
        snb = tl.reshape(tl.broadcast_to(ph_sin[:, :, None],
                                         (T, N // 2, 2 * R)), (T, NR_B))
        qb = (Bt_ptr + b * s_qb + c * s_qc + h * s_qh
              + offs_t[:, None] * s_qt)
        ub = (dBt_ptr + b * s_ub + c * s_uc + h * s_uh
              + offs_t[:, None] * s_ut)
        til = tl.load(qb + offs_f[None, :]).to(tl.float32)
        tilx = tl.load(qb + (offs_f ^ R)[None, :]).to(tl.float32)
        gt = tl.load(ub + offs_f[None, :]).to(tl.float32)
        gtx = tl.load(ub + (offs_f ^ R)[None, :]).to(tl.float32)
        if NEED_DTH or NEED_DTHH:
            # <dx~, J x~> summed over the pair's two lanes AND over r
            dth = tl.sum(tl.reshape(sgn_b[None, :] * gt * tilx,
                                    (T, N // 2, 2 * R)), axis=2)
        else:
            dth = tl.zeros([T, N // 2], dtype=tl.float32)
        dpre = csb * gt - sgn_b[None, :] * snb * gtx
        if HAS_BETA:
            tl.atomic_add(dBeta_ptr + h * NR_B + offs_f,
                          tl.sum(dpre, axis=0), sem=ACC_SEM)
        addr = (dB_ptr + b * s_bb + c * s_bc + bg * s_bh
                + offs_t[:, None] * s_bt + offs_f[None, :])
        if DB_ATOMIC:
            tl.atomic_add(addr, dpre, sem=ACC_SEM)
        else:
            tl.store(addr, dpre)

        offs_g = tl.arange(0, NR_C)
        sgn_c = tl.where((offs_g // O) % 2 == 0, 1.0, -1.0)
        csc = tl.reshape(tl.broadcast_to(cr_cos[:, :, None],
                                         (T, N // 2, 2 * O)), (T, NR_C))
        snc = tl.reshape(tl.broadcast_to(cr_sin[:, :, None],
                                         (T, N // 2, 2 * O)), (T, NR_C))
        pc = (Ct_ptr + b * s_pb + c * s_pc + h * s_ph
              + offs_t[:, None] * s_pt)
        vc = (dCt_ptr + b * s_vb + c * s_vc + h * s_vh
              + offs_t[:, None] * s_vt)
        ctil = tl.load(pc + offs_g[None, :]).to(tl.float32)
        ctilx = tl.load(pc + (offs_g ^ O)[None, :]).to(tl.float32)
        cgt = tl.load(vc + offs_g[None, :]).to(tl.float32)
        cgtx = tl.load(vc + (offs_g ^ O)[None, :]).to(tl.float32)
        if NEED_DTH or NEED_DTHH:
            dcth = tl.sum(tl.reshape(sgn_c[None, :] * cgt * ctilx,
                                     (T, N // 2, 2 * O)), axis=2)
            dth += dcth
            if NEED_DTHH:
                tl.atomic_add(dThH_ptr + h * (N // 2) + offs_k,
                              tl.sum(dcth, axis=0), sem=ACC_SEM)
        cpre = csc * cgt - sgn_c[None, :] * snc * cgtx
        if HAS_GAMMA:
            tl.atomic_add(dGamma_ptr + h * NR_C + offs_g,
                          tl.sum(cpre, axis=0), sem=ACC_SEM)
        caddr = (dC_ptr + b * s_cb + c * s_cc + cg * s_ch
                 + offs_t[:, None] * s_ct + offs_g[None, :])
        if DC_ATOMIC:
            tl.atomic_add(caddr, cpre, sem=ACC_SEM)
        else:
            tl.store(caddr, cpre)
        if NEED_DTH:
            daddr = (dTh_ptr + b * s_dtb + c * s_dtc + h * s_dth
                     + offs_t[:, None] * s_dtt + offs_k[None, :] * s_dtk)
            if DTH_ATOMIC:
                tl.atomic_add(daddr, dth, sem=ACC_SEM)
            else:
                tl.store(daddr, dth)
        return

    dth = tl.zeros([T, N // 2], dtype=tl.float32)

    for r in tl.range(R, loop_unroll_factor=1):
        til = tl.load(Bt_ptr + b * s_qb + c * s_qc + h * s_qh
                      + offs_t[:, None] * s_qt + offs_n[None, :] * s_qn
                      + r * s_qr).to(tl.float32)
        gtil = tl.load(dBt_ptr + b * s_ub + c * s_uc + h * s_uh
                       + offs_t[:, None] * s_ut + offs_n[None, :] * s_un
                       + r * s_ur).to(tl.float32)
        xe, xo = _rot_pair_split(til, T, N, PAIR_HALF)
        ge, go = _rot_pair_split(gtil, T, N, PAIR_HALF)
        if NEED_DTH or NEED_DTHH:
            # <dx~, J x~> with J x~ = (x~_odd, -x~_even)
            dth += ge * xo - go * xe
        # the adjoint of x~ = R(theta)^T x is dx = R(theta) dx~
        pe = ph_cos * ge - ph_sin * go
        po = ph_sin * ge + ph_cos * go
        dpre = _rot_pair_join(pe, po, T, N, PAIR_HALF)
        if HAS_BETA:
            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r,
                          tl.sum(dpre, axis=0), sem=ACC_SEM)
        addr = (dB_ptr + b * s_bb + c * s_bc + bg * s_bh
                + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn
                + r * s_br)
        if DB_ATOMIC:
            tl.atomic_add(addr, dpre, sem=ACC_SEM)
        else:
            tl.store(addr, dpre)

    if NEED_DTHH:
        dnu = tl.zeros([N // 2], dtype=tl.float32)
    for o_i in tl.range(O, loop_unroll_factor=1):
        til = tl.load(Ct_ptr + b * s_pb + c * s_pc + h * s_ph
                      + offs_t[:, None] * s_pt + offs_n[None, :] * s_pn
                      + o_i * s_po).to(tl.float32)
        gtil = tl.load(dCt_ptr + b * s_vb + c * s_vc + h * s_vh
                       + offs_t[:, None] * s_vt + offs_n[None, :] * s_vn
                       + o_i * s_vo).to(tl.float32)
        xe, xo = _rot_pair_split(til, T, N, PAIR_HALF)
        ge, go = _rot_pair_split(gtil, T, N, PAIR_HALF)
        if NEED_DTH or NEED_DTHH:
            dcth = ge * xo - go * xe
            dth += dcth
            if NEED_DTHH:
                dnu += tl.sum(dcth, axis=0)
        pe = cr_cos * ge - cr_sin * go
        po = cr_sin * ge + cr_cos * go
        dpre = _rot_pair_join(pe, po, T, N, PAIR_HALF)
        if HAS_GAMMA:
            tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o_i,
                          tl.sum(dpre, axis=0), sem=ACC_SEM)
        addr = (dC_ptr + b * s_cb + c * s_cc + cg * s_ch
                + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn
                + o_i * s_co)
        if DC_ATOMIC:
            tl.atomic_add(addr, dpre, sem=ACC_SEM)
        else:
            tl.store(addr, dpre)

    if NEED_DTHH:
        tl.atomic_add(dThH_ptr + h * (N // 2) + offs_k, dnu, sem=ACC_SEM)
    if NEED_DTH:
        addr = (dTh_ptr + b * s_dtb + c * s_dtc + h * s_dth
                + offs_t[:, None] * s_dtt + offs_k[None, :] * s_dtk)
        if DTH_ATOMIC:
            tl.atomic_add(addr, dth, sem=ACC_SEM)
        else:
            tl.store(addr, dth)


def _rot_six(t, grouped):
    """(b, c, h, t, n, r) strides for a chunked B/C-shaped tensor.

    Grouped tensors carry the group axis at dim 2 and an OPTIONAL
    trailing R/O axis; ungrouped ones have no group axis (stride 0 --
    every kernel head reads the one shared tile).
    """
    st = list(t.stride())
    if grouped:
        return (st[0], st[1], st[2], st[3], st[4],
                st[5] if t.dim() == 6 else 0)
    return (st[0], st[1], 0, st[2], st[3],
            st[4] if t.dim() == 5 else 0)


def _rot_shape(t, grouped):
    """(N, trailing R/O) for a chunked B/C-shaped tensor."""
    if grouped:
        return t.shape[4], (t.shape[5] if t.dim() == 6 else 1)
    return t.shape[3], (t.shape[4] if t.dim() == 5 else 1)


def rot_prologue(Ctil, Btil, theta, theta_h, beta, gamma, H,
                 Gb=1, Gc=1, bc_grouped=False, out_dtype=None,
                 pair_half=False):
    """B~ / C~ per KERNEL HEAD, from the raw tiles and the cumulated phase.

    Returns `(Ctil_rot, Btil_rot)` shaped `[B, nc, H, T, N, O/R]`, i.e.
    the fused kernels' GROUPED spelling at `Gb = Gc = H`.  The caller
    then runs the whole scan -- forward, the STATE_ONLY replay and the
    five backward kernels -- with `theta = theta_h = beta = gamma = None`
    and `bias_pre_rot = False`, on the REAL-mode bodies.
    """
    (Bsz, nc, T, _Hax, K), th_st = _phase_shapes(theta)
    dev = Btil.device
    N, R = _rot_shape(Btil, bc_grouped)
    Nc, O = _rot_shape(Ctil, bc_grouped)
    if Nc != N or 2 * K != N:
        raise ValueError(
            "rot_prologue: state axes disagree (B %d, C %d, phase %d)"
            % (N, Nc, 2 * K))
    if out_dtype is None:
        out_dtype = Btil.dtype
    Bo = torch.empty(Bsz, nc, H, T, N, R, device=dev, dtype=out_dtype)
    Co = torch.empty(Bsz, nc, H, T, N, O, device=dev, dtype=out_dtype)
    sb, sc = _rot_six(Btil, bc_grouped), _rot_six(Ctil, bc_grouped)
    sq, sp = Bo.stride(), Co.stride()
    flat = (not pair_half and _rot_flat_ok(sb, R) and _rot_flat_ok(sc, O)
            and _env("ROT_SPLIT_FLAT", "1") == "1")
    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    _rot_prologue_kernel[(Bsz, nc, H)](
        Btil, Ctil,
        beta if beta is not None else dummy,
        gamma if gamma is not None else dummy,
        theta, theta_h if theta_h is not None else dummy,
        Bo, Co,
        *sb, *sc, *sq, *sp, *th_st,
        HPG_B=max(H // Gb, 1), HPG_C=max(H // Gc, 1),
        T=T, N=N, R=R, O=O,
        HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None,
        HAS_THH=theta_h is not None, PAIR_HALF=bool(pair_half),
        FLAT=flat, NR_B=N * R, NR_C=N * O,
        num_warps=int(_env("ROT_SPLIT_WARPS", "8")),
    )
    return Co, Bo


def rot_adjoint(Ctil_rot, Btil_rot, dCtil_rot, dBtil_rot,
                theta, theta_h, H, Gb=1, Gc=1, bc_grouped=False,
                B_shape=None, C_shape=None, B_stride=None, C_stride=None,
                need_dtheta=True, need_dthh=True,
                need_dbeta=False, need_dgamma=False,
                R_out=1, O_out=1, pair_half=False):
    """The whole rotation adjoint, in one elementwise launch.

    `(dCtil, dBtil, dBeta, dGamma, dTheta, dThetaH)`, with `dTheta` the
    gradient of the CUMULATED phase (the caller runs `phase_prefix_bwd`
    on it exactly as the in-kernel route does) and `dBtil` / `dCtil` on
    the CALLER's map (`B_shape` / `B_stride`), summed over the kernel
    heads that share it.
    """
    (Bsz, nc, T, _Hax, K), th_st = _phase_shapes(theta)
    dev = Btil_rot.device
    f32 = torch.float32
    N = 2 * K
    hpg_b, hpg_c = max(H // Gb, 1), max(H // Gc, 1)
    db_atomic = hpg_b > 1 or not bc_grouped
    dc_atomic = hpg_c > 1 or not bc_grouped
    dB = torch.empty_strided(B_shape, B_stride, device=dev, dtype=f32)
    dC = torch.empty_strided(C_shape, C_stride, device=dev, dtype=f32)
    if db_atomic:
        dB.zero_()
    if dc_atomic:
        dC.zero_()
    dBeta = (torch.zeros(H, N, R_out, device=dev, dtype=f32)
             if need_dbeta else None)
    dGamma = (torch.zeros(H, N, O_out, device=dev, dtype=f32)
              if need_dgamma else None)
    dThH = (torch.zeros(H, N // 2, device=dev, dtype=f32)
            if need_dthh else None)
    # the phase gradient carries the CUMULATED phase's shape; a
    # head-SHARED phase is written by every head, so it is an atomic.
    dth_atomic = th_st[3] == 0
    dTheta = None
    dth_st = [0, 0, 0, 0, 0]
    if need_dtheta:
        dTheta = torch.zeros(theta.shape, device=dev, dtype=f32)
        (_, _, _, _, _), dth_st = _phase_shapes(dTheta)
    flat = (not pair_half
            and _rot_flat_ok(_rot_six(dB, bc_grouped), Btil_rot.shape[5])
            and _rot_flat_ok(_rot_six(dC, bc_grouped), Ctil_rot.shape[5])
            and _env("ROT_SPLIT_FLAT", "1") == "1")
    dummy = torch.empty(1, device=dev, dtype=f32)
    _rot_adjoint_kernel[(Bsz, nc, H)](
        Btil_rot, Ctil_rot, dBtil_rot, dCtil_rot,
        theta, theta_h if theta_h is not None else dummy,
        dB, dC,
        dBeta if dBeta is not None else dummy,
        dGamma if dGamma is not None else dummy,
        dTheta if dTheta is not None else dummy,
        dThH if dThH is not None else dummy,
        *Btil_rot.stride(), *Ctil_rot.stride(),
        *dBtil_rot.stride(), *dCtil_rot.stride(),
        *_rot_six(dB, bc_grouped), *_rot_six(dC, bc_grouped),
        *th_st, *dth_st,
        HPG_B=hpg_b, HPG_C=hpg_c,
        T=T, N=N, R=Btil_rot.shape[5], O=Ctil_rot.shape[5],
        HAS_BETA=need_dbeta, HAS_GAMMA=need_dgamma,
        HAS_THH=theta_h is not None, NEED_DTHH=need_dthh,
        NEED_DTH=need_dtheta, DTH_ATOMIC=dth_atomic,
        DB_ATOMIC=db_atomic, DC_ATOMIC=dc_atomic,
        PAIR_HALF=bool(pair_half),
        FLAT=flat, NR_B=N * Btil_rot.shape[5], NR_C=N * Ctil_rot.shape[5],
        num_warps=int(_env("ROT_SPLIT_WARPS", "8")),
    )
    return dC, dB, dBeta, dGamma, dTheta, dThH


def rot_bwd_mode() -> str:
    """`SSD_MIMO_ROT_BWD`: "split" (the Mamba-3 decomposition) or
    "inkernel" (the shipped per-token dTheta inside Kb1 / Kb2 / Kbc)."""
    v = _env("ROT_BWD", "inkernel").lower()
    if v not in ("split", "inkernel"):
        raise ValueError("SSD_MIMO_ROT_BWD must be 'split' or 'inkernel', "
                         "got %r" % (v,))
    return v
