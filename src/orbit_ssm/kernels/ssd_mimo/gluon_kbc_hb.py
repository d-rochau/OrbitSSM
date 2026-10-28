"""Gluon twin of `_bwd_kbc_hb_kernel`'s rotation-free, bias-free body
(the real-mode R1O1 route), with explicit MMA layouts.

STATUS.  Opt-in only: lever `SSD_MIMO_GLUON_KBC=1`.  `fused_chunk_bwd_5kernel`
calls `refusal()` just before the Triton Kbc_hb launch; inside the envelope
this kernel runs INSTEAD, with the same output buffers and the same
owner / atomic rule; every refusal is by name and falls through to the
Triton kernel unchanged.

WHY.  Every Triton Kbc_hb config sits at the 255-register cap with stack
spills, so two 4-warp CTAs fill an SM.  The cause: at T = 64 the
Triton layout engine gives the [T, NS] accumulators an [8, 1] warp grid, so
with 8 warps every MMA is issued twice (160 vs 80 mma.sync per thread) and
the accumulators are replicated.  Here the accumulator layout is chosen:
WM x WN warps with WN > 1 splitting the state columns, every global tile
loaded coalesced and staged in shared memory, MMA operands (including the
transposes) read from shared memory, and the per-head buffers
time-multiplexed.

MMA OP.  mma_v2 (sm_80 .. sm_120 all execute it).  sm_90 would want a
wgmma body; until one exists the host refuses a device whose reported
capability major is 9 (the MMA-op choice is the only per-arch split).

WHAT IT COMPUTES, per CTA (b, head block, chunk c, G tile, N slice) and head
h of the block -- the Triton body's arithmetic, in its order:
    D', seg   = the segmented chunk prefix (`_SEGPREFIX_NOTE`)
    f         = seg == 0 ? exp(a D') : 0
    dec_mask  = causal & same segment ? exp(a (D'_i - D'_j)) : 0
                (strictly causal under TRAP_DIAG)
    dec_agg   = seg == seg_end ? exp(a (D'_end - D')) : 0
    x_s       = x * dw                              (HAS_DW)
    dC_state  = f * (dY @ S^T)                      [T, NS]
    dSb       = bf16((dY @ bf16(x_s)^T) * dec_mask) [T, T]
    dC_h      = dSb @ B + dC_state
    dB_h      = dSb^T @ C + bf16(x_s * dec_agg) @ dG^T
    dDelta   += revcumsum(a * rowsum(C * dC_state)) (atomic: the read term)
    dC_acc   += dC_h,  dB_acc += dB_h
and the head-block sums leave the CTA once: a store when the CTA owns the
whole (b, c, n-slice) tile, a relaxed atomic add otherwise -- the Triton
owner rule.  Rounding differs from Triton's only in the MMA-internal
summation order.

CHART (the fused chart emitter): the same body, every head
in one CTA, plus the chart's head sums (dCs, dBa, the spanning-pair scores,
the diagonal blocks of alpha S dG^T) from the tiles above; dB~ / dC~ flushed
every HBF heads as the plain launch's HBF-head CTAs would.  Its accumulators
take a second CTA's registers, so it runs one CTA per SM with the cp.async
double buffer (`_chart_pipe`); measured on sm_120 at the 124M cell: 0.98 ms
vs 0.77 ms for the plain body (0.91 ms of it is the one-CTA-per-SM
configuration itself, chart work and its 42 MB of stores the rest).
"""

from __future__ import annotations

import os

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.ampere.async_copy import (
    async_load, commit_group, wait_group)

from .common import ACC_SEM
from .fused_fwd_kernel import _env


def gluon_kbc_wanted(guard: bool = True, T: int | None = None) -> bool:
    """Lever `real_gluon_kbc` (env `SSD_MIMO_GLUON_KBC=1` forces it on, any
    other value off; else the calibrated choice of group real_R1O1).  `guard`:
    the call is a real (rotation-free) R = O = 1 backward at the T = 64 the
    body is laid out for."""
    from . import levers
    return levers.on("real_gluon_kbc", levers.real_group(T), guard=guard)


def _cfg():
    """(num_warps, WM, WN, maxnreg, HB) -- env-overridable for the sweep."""
    w = int(_env("GLUON_KBC_WARPS", "8"))
    # warp grid [2, 4]; maxnreg 128 lets two 8-warp CTAs share an SM
    wn = int(_env("GLUON_KBC_WN", "4"))
    mr = int(_env("GLUON_KBC_MAXNREG", "128"))
    hb = int(_env("GLUON_KBC_HB", "8"))
    return w, w // wn, wn, mr, hb


def pipe_wanted(*, T, NS, GT, device) -> bool:
    """`SSD_MIMO_GLUON_KBC_PIPE`: "auto" (default) | "1" | "0".

    The cp.async double buffer is worth it only where it does not cost the
    second CTA per SM.  "auto" therefore asks the DEVICE: pipe iff two CTAs' shared memory, the double
    buffer included, fit the reported per-SM capacity."""
    v = _env("GLUON_KBC_PIPE", "auto").lower()
    if v in ("0", "off"):
        return False
    if v in ("1", "on"):
        return True
    props = torch.cuda.get_device_properties(device)
    per_sm = int(getattr(props, "shared_memory_per_multiprocessor", 0) or 0)
    # bytes, 16-bit tiles: resident B, C; the double buffer (dY, x, S, dG);
    # the separate dSb slot when GT != T; ~1 KB of conversion scratch
    need = (2 * 2 * T * NS + 2 * 2 * (2 * T * GT + 2 * NS * GT)
            + (0 if GT == T else 2 * T * T) + 1024)
    return per_sm > 0 and 2 * need <= per_sm


def _cp_async_ok(strides, tensors) -> bool:
    """cp.async moves >= 4 B, and 16 B only when the compiler can prove it:
    every staged tile's inner stride must be 1 and every other stride a
    multiple of 8 elements (the JIT's divisibility-16 specialisation), with
    16-byte-aligned bases.  Otherwise the unpipelined body serves (ssd_check's
    small cells hit this)."""
    (_cb, _cc, _ct, _cn, _bb, _bc, _bt, _bn,
     s_xb, s_xh, s_xg, s_xl, _db, _dh, _dc, _dt,
     s_yb, s_yh, s_yc, s_yt, s_yg, s_sb, s_sh, s_sc, s_sn, s_sg,
     s_gb, s_gh, s_gc, s_gn, s_gg) = strides[:31]
    x_in, x_out = (s_xg, s_xl) if s_xg == 1 else (s_xl, s_xg)
    inner = (x_in, s_yg, s_sg, s_gg)
    outer = (s_xb, s_xh, x_out, s_yb, s_yh, s_yc, s_yt, s_sb, s_sh, s_sc, s_sn,
             s_gb, s_gh, s_gc, s_gn)
    ok = (all(v == 1 for v in inner) and all(v % 8 == 0 for v in outer)
          and all(t.data_ptr() % 16 == 0 for t in tensors))
    if not ok and _env("CP_TRACE", "0") == "1":
        print("CP_TRACE kbc_pipe_refused inner=%s outer=%s align=%s" % (
            inner, outer, [t.data_ptr() % 16 for t in tensors]), flush=True)
    return ok


def refusal(*, device, mma_io, rot_in_kernel, rot_per_head, has_theta_h,
            has_rot_bias, has_beta, has_gamma, pre_rot, pair_half, sumds,
            dd_in_kbc, ka_nodalpha, kbc_split, rot_abl, ts_abl, T, N, GT,
            nsplit, trap_diag=False, dcs_ink=False, allow_sm90=False):
    """None when the Gluon body serves this call, else the reason.  `allow_sm90`: the chart experiment lever
    (`fused_chunk_bwd_5kernel`, SSD_MIMO_CHART_FUSED_SM90) admits sm_90's mma.sync."""
    if not mma_io:
        return "fp32 / tf32 operands (the Gluon body is 16-bit MMA only)"
    if (rot_in_kernel or rot_per_head or has_theta_h or has_rot_bias
            or pair_half):
        return "rotation"
    if pre_rot:
        return "bias_pre_rot (the biases rotate with B/C)"
    if sumds or dd_in_kbc or ka_nodalpha or kbc_split or rot_abl or ts_abl:
        return "lever outside the envelope"
    if dcs_ink:
        # off at R = O = 1 by default (autograd `_dcs_ink_on`), so no
        # production arm reaches it and no fp64 arm covers it here
        return "DCS_INK (the in-kernel prefix is off at R = O = 1)"
    if T != 64:
        return f"T={T} (the accumulator warp grid is laid out for T=64)"
    if GT & (GT - 1):
        # gl.allocate_shared_memory needs power-of-two dimensions: a G tile of 48 (G = 48, one tile) compiles to "shape must have
        # power-of-2 ... got 64, 48"
        return f"GT={GT} is not a power of two (Gluon shared-memory tiles)"
    w, wm, wn, _, _ = _cfg()
    NS = N // nsplit
    if NS % (8 * wn) or NS < 16 or GT % 16 or GT < 16 or T % (16 * wm):
        return f"tile NS={NS} GT={GT} vs warp grid [{wm},{wn}]"
    if torch.cuda.get_device_capability(device)[0] == 9 and not allow_sm90:
        return "sm_90: needs the wgmma body (not built)"
    return None


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _segadd(v1, f1, v2, f2):
    return gl.where(f2 != 0, v2, v1 + v2), f1 | f2


@gluon.jit
def _kbc_issue(sY, sX, sS, sG, dY_ptr, X_ptr, SAll_ptr, dG_ptr, b, h, c,
               s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl,
               s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg,
               rt, cg, ct, rg, rn, T: gl.constexpr, X_GC: gl.constexpr):
    async_load(sY, dY_ptr + b * s_yb + h * s_yh + c * s_yc
               + rt[:, None] * s_yt + cg[None, :] * s_yg)
    if X_GC:
        async_load(sX, X_ptr + b * s_xb + h * s_xh + (c * T + rt[:, None]) * s_xl
                   + cg[None, :] * s_xg)
    else:
        async_load(sX, X_ptr + b * s_xb + h * s_xh + (c * T + ct[None, :]) * s_xl
                   + rg[:, None] * s_xg)
    async_load(sS, SAll_ptr + b * s_sb + h * s_sh + c * s_sc
               + rn[:, None] * s_sn + cg[None, :] * s_sg)
    async_load(sG, dG_ptr + b * s_gb + h * s_gh + c * s_gc
               + rn[:, None] * s_gn + cg[None, :] * s_gg)
    commit_group()


@gluon.jit
def _kbc_hb_gl_kernel(
    C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, SAll_ptr, dG_ptr, DW_ptr, dY_ptr,
    dCtil_ptr, dBtil_ptr, dDelta_ptr,
    Beta_ptr, Gamma_ptr, dBeta_ptr, dGamma_ptr,
    H,
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg,
    s_sb, s_sh, s_sc, s_sn, s_sg,
    s_gb, s_gh, s_gc, s_gn, s_gg,
    s_wb, s_wh, s_wl,
    s_dcb, s_dcc, s_dct, s_dcn,
    s_dbb, s_dbc, s_dbt, s_dbn,
    NGT_val,
    NSPLIT: gl.constexpr, T: gl.constexpr, N: gl.constexpr,
    GT: gl.constexpr, HB: gl.constexpr,
    HAS_DW: gl.constexpr, DCS_INK: gl.constexpr, TRAP_DIAG: gl.constexpr,
    IDX64: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr,
    SEG_RESET: gl.constexpr, SEM: gl.constexpr,
    HAS_BETA: gl.constexpr = False, HAS_GAMMA: gl.constexpr = False,
    PIPE: gl.constexpr = False,
    X_GC: gl.constexpr = False,     # x tile G-contiguous (token-major x)
    KBC_DALPHA: gl.constexpr = False, GTB: gl.constexpr = 64,
    dAlpha_ptr=None, s_dab=0, s_dah=0, s_dag=0, s_dac=0,
    # CHART (the local chart cotangent): the CTA runs EVERY head (HB = H, one head
    # block) and also accumulates the head sums the chart emitter's Kch1 formed in a second pass over S_all / dG_all
    # / dY / x: dCs = sum_h f (dY S^T) (= this body's dC_state), dBa = sum_h bf16(x dw e) dG^T (its xw operand),
    # MT[l, j] = sum_h dec(j, l) dY_l . x_j for j < l (its score tile BEFORE the bf16 rounding), and the diagonal
    # [KS, KS] blocks of T1 = sum_h alpha S dG^T.  dB~ / dC~ leave the CTA every HBF heads exactly as the HBF-head
    # CTAs of the plain launch would (same partial sums, an atomic add each), so they are the plain route's tensors.
    CHART: gl.constexpr = False, HBF: gl.constexpr = 8, KS: gl.constexpr = 16,
    dCs_ptr=None, dBa_ptr=None, MT_ptr=None, T1_ptr=None, NCH=1,
):
    NW: gl.constexpr = WM * WN
    NS: gl.constexpr = N // NSPLIT
    KEEP_S: gl.constexpr = KBC_DALPHA or CHART
    MMA: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, WN], instr_shape=[16, 8])
    OPA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    OPB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    ROW: gl.constexpr = gl.SliceLayout(1, MMA)
    COL: gl.constexpr = gl.SliceLayout(0, MMA)
    A_ROW: gl.constexpr = gl.SliceLayout(1, OPA)
    B_COL: gl.constexpr = gl.SliceLayout(0, OPB)
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    # coalesced 16-byte loads along the tile's second (contiguous) axis
    BL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    BL_R: gl.constexpr = gl.SliceLayout(1, BL)
    BL_C: gl.constexpr = gl.SliceLayout(0, BL)
    SH: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                              order=[1, 0])

    b = gl.program_id(0)
    hblk = gl.program_id(1)
    pid2 = gl.program_id(2)
    ns_idx = pid2 % NSPLIT
    rest = pid2 // NSPLIT
    c = rest // NGT_val
    gt_idx = rest % NGT_val
    if IDX64:
        b = b.to(gl.int64)

    # ── head-shared tiles: B, C [T, NS] ─────────────────────────────────
    rt = gl.arange(0, T, layout=BL_R)
    cn = ns_idx * NS + gl.arange(0, NS, layout=BL_C)
    sB = gl.allocate_shared_memory(
        gl.bfloat16, [T, NS], SH,
        gl.load(B_ptr + b * s_bb + c * s_bc + rt[:, None] * s_bt
                + cn[None, :] * s_bn).to(gl.bfloat16))
    sC = gl.allocate_shared_memory(
        gl.bfloat16, [T, NS], SH,
        gl.load(C_ptr + b * s_cb + c * s_cc + rt[:, None] * s_ct
                + cn[None, :] * s_cn).to(gl.bfloat16))
    if PIPE:
        # PIPE (lever SSD_MIMO_GLUON_KBC_PIPE): the NEXT head's dY / x / S /
        # dG are copied with cp.async into the other half of a double buffer
        # while this head computes (16-bit tensors only: no cast on the copy)
        qY = gl.allocate_shared_memory(gl.bfloat16, [2, T, GT], SH)
        if X_GC:
            qX = gl.allocate_shared_memory(gl.bfloat16, [2, T, GT], SH)
        else:
            qX = gl.allocate_shared_memory(gl.bfloat16, [2, GT, T], SH)
        qS = gl.allocate_shared_memory(gl.bfloat16, [2, NS, GT], SH)
        qG = gl.allocate_shared_memory(gl.bfloat16, [2, NS, GT], SH)
        if GT != T:
            sP0 = gl.allocate_shared_memory(gl.bfloat16, [T, T], SH)
    else:
        # per-head slots: dY (then dSb), x as [GT, T], one state slot (S, then dG)
        sY = gl.allocate_shared_memory(gl.bfloat16, [T, GT], SH)
        if X_GC:
            sX = gl.allocate_shared_memory(gl.bfloat16, [T, GT], SH)
        else:
            sX = gl.allocate_shared_memory(gl.bfloat16, [GT, T], SH)
        sSt = gl.allocate_shared_memory(gl.bfloat16, [NS, GT], SH)
        # dSb [T, T] reuses dY's slot when the shapes agree (GT == T)
        if GT == T:
            sP = sY
        else:
            sP = gl.allocate_shared_memory(gl.bfloat16, [T, T], SH)
        if KEEP_S:
            sGd = gl.allocate_shared_memory(gl.bfloat16, [NS, GT], SH)

    rg = gt_idx * GT + gl.arange(0, GT, layout=BL_R)
    ct = gl.arange(0, T, layout=BL_C)
    cg = gt_idx * GT + gl.arange(0, GT, layout=BL_C)
    rn = ns_idx * NS + gl.arange(0, NS, layout=BL_R)
    t1 = gl.arange(0, T, layout=L1)
    t_m = gl.arange(0, T, layout=ROW)
    s_m = gl.arange(0, T, layout=COL)
    dC_acc = gl.zeros([T, NS], gl.float32, MMA)
    dB_acc = gl.zeros([T, NS], gl.float32, MMA)
    h_start = hblk * HB
    hb_count = gl.minimum(HB, H - h_start)
    n_m = ns_idx * NS + gl.arange(0, NS, layout=COL)
    if CHART:
        accCs = gl.zeros([T, NS], gl.float32, MMA)
        accBa = gl.zeros([T, NS], gl.float32, MMA)
        accM = gl.zeros([T, T], gl.float32, MMA)
        accT1 = gl.zeros([NS, NS], gl.float32, MMA)
        db_p0 = (dBtil_ptr + b * s_dbb + c * s_dbc + t_m[:, None] * s_dbt
                 + n_m[None, :] * s_dbn)
        dc_p0 = (dCtil_ptr + b * s_dcb + c * s_dcc + t_m[:, None] * s_dct
                 + n_m[None, :] * s_dcn)
    if PIPE:
        _kbc_issue(qY.index(0), qX.index(0), qS.index(0), qG.index(0),
                   dY_ptr, X_ptr, SAll_ptr, dG_ptr, b, h_start, c,
                   s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl,
                   s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg,
                   rt, cg, ct, rg, rn, T, X_GC)
    for hh in range(hb_count):
        h = h_start + hh
        if IDX64:
            h = h.to(gl.int64)
        if PIPE:
            st = hh % 2
            gl.barrier()             # the other slot's last reader is done
            if hh + 1 < hb_count:
                _kbc_issue(qY.index(1 - st), qX.index(1 - st), qS.index(1 - st),
                           qG.index(1 - st), dY_ptr, X_ptr, SAll_ptr, dG_ptr,
                           b, h + 1, c,
                           s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl,
                           s_sb, s_sh, s_sc, s_sn, s_sg,
                           s_gb, s_gh, s_gc, s_gn, s_gg,
                           rt, cg, ct, rg, rn, T, X_GC)
                wait_group(1)
            else:
                wait_group(0)
            sY = qY.index(st)
            sX = qX.index(st)
            sSt = qS.index(st)
            sGc = qG.index(st)
            if GT == T:
                sP = sY
            else:
                sP = sP0
        a_val = -gl.exp(gl.load(A_ptr + h))

        # ── the segmented prefix, in a 1-D blocked layout ──────────────
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        d = gl.load(dbase + t1 * s_dt)
        if DCS_INK:
            rst = (a_val * d) <= -SEG_RESET
            ri = rst.to(gl.int32)
            D1, _unused = gl.associative_scan((gl.where(rst, 0.0, d), ri), 0,
                                              _segadd)
        else:
            rst = d < 0.0
            ri = rst.to(gl.int32)
            D1 = gl.where(rst, 0.0, d)
        seg1 = gl.associative_scan(ri, 0, _add)
        seg_end = gl.max(seg1, axis=0)
        D_end = gl.sum(gl.where(t1 == T - 1, D1, 0.0), axis=0)
        f1 = gl.where(seg1 == 0, gl.exp(a_val * D1), 0.0)
        agg1 = gl.where(seg1 == seg_end, gl.exp(a_val * (D_end - D1)), 0.0)

        # ── stage dY, x, S (coalesced) ─────────────────────────────────
        gl.barrier()
        if not PIPE:
            sY.store(gl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc
                             + rt[:, None] * s_yt + cg[None, :] * s_yg
                             ).to(gl.bfloat16))
            if X_GC:
                sX.store(gl.load(X_ptr + b * s_xb + h * s_xh
                                 + (c * T + rt[:, None]) * s_xl
                                 + cg[None, :] * s_xg).to(gl.bfloat16))
            else:
                sX.store(gl.load(X_ptr + b * s_xb + h * s_xh
                                 + (c * T + ct[None, :]) * s_xl
                                 + rg[:, None] * s_xg).to(gl.bfloat16))
            sSt.store(gl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc
                              + rn[:, None] * s_sn + cg[None, :] * s_sg
                              ).to(gl.bfloat16))
            gl.barrier()

        # state path first, so dC_state dies before the [T, T] tiles are born
        f = gl.convert_layout(f1, ROW)
        dC_state = mma_v2(sY.load(OPA), sSt.permute((1, 0)).load(OPB),
                          gl.zeros([T, NS], gl.float32, MMA)) * f[:, None]
        # the read term of dL/ddelta (SPANDELTA)
        n_c = ns_idx * NS + gl.arange(0, NS, layout=COL)
        Cm = sC.load(MMA).to(gl.float32)
        if HAS_GAMMA:
            ga = gl.load(Gamma_ptr + h * N + n_c).to(gl.float32)
            Cm = Cm + ga[None, :]                       # C_eff = C + gamma, fp32
        dD = a_val * gl.sum(Cm * dC_state, axis=1)
        dD = gl.associative_scan(gl.convert_layout(dD, L1), 0, _add, reverse=True)
        gl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + t1 * s_dt,
                      dD, sem=SEM)
        # (the Triton body adds dC_state after the score product; fp32
        # addition order only)
        dC_acc = dC_acc + dC_state
        if CHART:
            accCs = accCs + dC_state
        if HAS_GAMMA:
            # dGamma_h = colsum(dC_h); its dC_state part, while it is live
            v_gam = gl.sum(dC_state, axis=0)

        # x^T operand [GT, T]: bf16(x * dw) exactly as the Triton body forms it
        if X_GC:
            xT = sX.permute((1, 0)).load(OPB)
        else:
            xT = sX.load(OPB)
        if HAS_DW:
            dw_b = gl.load(DW_ptr + b * s_wb + h * s_wh
                           + (c * T + gl.arange(0, T, layout=B_COL)) * s_wl)
            xT = (xT.to(gl.float32) * dw_b[None, :]).to(gl.bfloat16)
        dScore = mma_v2(sY.load(OPA), xT, gl.zeros([T, T], gl.float32, MMA))
        D_r = gl.convert_layout(D1, ROW)
        D_c = gl.convert_layout(D1, COL)
        sg_r = gl.convert_layout(seg1, ROW)
        sg_c = gl.convert_layout(seg1, COL)
        # causal & same segment, formed from [T] vectors at its only use
        if TRAP_DIAG:
            causal = t_m[:, None] > s_m[None, :]
        else:
            causal = t_m[:, None] >= s_m[None, :]
        dS32 = dScore * gl.exp(gl.where(causal & (sg_r[:, None] == sg_c[None, :]),
                                        a_val * (D_r[:, None] - D_c[None, :]),
                                        float("-inf")))
        dSb = dS32.to(gl.bfloat16)
        if CHART:
            # the spanning-pair scores (strictly causal: the chart's pairs j < l), fp32, before the rounding
            if TRAP_DIAG:
                accM = accM + dS32
            else:
                accM = accM + gl.where(t_m[:, None] > s_m[None, :], dS32, 0.0)

        gl.barrier()
        sP.store(dSb)
        # sDG: this head's dG slot.  NOT a reassignment of the loop-carried S slot `sSt`: with KEEP_S that made the
        # NEXT head stage its S into the dG slot and then overwrite it with dG (sS_cur = dG: S dG^T -> dG dG^T, the
        # dalpha hand-off -> sum dG^2), every head after the first
        if PIPE:
            sS_cur = sSt
            sDG = sGc
        else:
            if KEEP_S:
                # S stays for dalpha: dG takes its own slot
                sS_cur = sSt
                sGd.store(gl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc
                                  + rn[:, None] * s_gn + cg[None, :] * s_gg
                                  ).to(gl.bfloat16))
                sDG = sGd
            else:
                sDG = sSt
                sDG.store(gl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc
                                  + rn[:, None] * s_gn + cg[None, :] * s_gg
                                  ).to(gl.bfloat16))
        gl.barrier()
        if KBC_DALPHA:
            # dalpha_c = sum(dG_c * S_c) per BACKWARD G tile (width GTB), the
            # partial of this N slice -- what Ka stores when it owns it
            # 16-row slices keep the two fp32 tiles out of the register
            # budget (the whole [NS, GT] pair spilled on the prefetch body)
            col = gl.zeros([GT], gl.float32, BL_C)
            for r0 in gl.static_range(0, NS, 16):
                col += gl.sum(sS_cur.slice(r0, 16, dim=0).load(BL).to(gl.float32)
                              * sDG.slice(r0, 16, dim=0).load(BL).to(gl.float32),
                              axis=0)
            gcl = gl.arange(0, GT, layout=BL_C)
            for j in gl.static_range(GT // GTB):
                gl.atomic_add(dAlpha_ptr + b * s_dab + h * s_dah
                              + (gt_idx * (GT // GTB) + j) * s_dag + c * s_dac,
                              gl.sum(gl.where(gcl // GTB == j, col, 0.0), axis=0),
                              sem=SEM)
        if CHART:
            # T1: the chunk-entry state against the chunk-end adjoint, alpha S dG^T (the diagonal [KS, KS] blocks
            # are stored; the full [NS, NS] tile is one MMA)
            alpha_h = gl.where(seg_end == 0, gl.exp(a_val * D_end), 0.0)
            accT1 = accT1 + alpha_h * mma_v2(sS_cur.load(OPA), sDG.permute((1, 0)).load(OPB),
                                             gl.zeros([NS, NS], gl.float32, MMA))
        dC_acc = mma_v2(sP.load(OPA), sB.load(OPB), dC_acc)
        dB_acc = mma_v2(sP.permute((1, 0)).load(OPA), sC.load(OPB), dB_acc)
        if HAS_BETA or HAS_GAMMA:
            # the per-head biases as EXACT fp32 rank-1 corrections instead of
            # bf16(B + beta) / bf16(C + gamma) operand copies (the Gluon sm_90
            # Kb1 pattern):  dSb (B + beta) = dSb B + rs beta^T,
            # dSb^T (C + gamma) = dSb^T C + cs gamma^T
            dSf = dSb.to(gl.float32)
            rs = gl.sum(dSf, axis=1)                    # [T] ROW
            cs = gl.convert_layout(gl.sum(dSf, axis=0), ROW)
            if HAS_BETA:
                be = gl.load(Beta_ptr + h * N + n_c).to(gl.float32)
                dC_acc = dC_acc + rs[:, None] * be[None, :]
            if HAS_GAMMA:
                dB_acc = dB_acc + cs[:, None] * ga[None, :]
                # dGamma_h = colsum(dSb (B + beta)) + colsum(dC_state)
                Bm = sB.load(MMA).to(gl.float32)
                if HAS_BETA:
                    Bm = Bm + be[None, :]
                v_gam = v_gam + gl.sum(cs[:, None] * Bm, axis=0)
                gl.atomic_add(dGamma_ptr + h * N + n_c, v_gam, sem=SEM)
            if HAS_BETA:
                # dBeta_h = colsum(dSb^T (C + gamma)) + colsum(xw dG^T)
                Cb = sC.load(MMA).to(gl.float32)
                if HAS_GAMMA:
                    Cb = Cb + ga[None, :]
                v_beta = gl.sum(rs[:, None] * Cb, axis=0)
        # x as the LHS [T, GT]: bf16((x * dw) * dec_agg)
        if X_GC:
            xa = sX.load(OPA).to(gl.float32)
        else:
            xa = sX.permute((1, 0)).load(OPA).to(gl.float32)
        agg = gl.convert_layout(agg1, A_ROW)
        if HAS_DW:
            dw_a = gl.load(DW_ptr + b * s_wb + h * s_wh
                           + (c * T + gl.arange(0, T, layout=A_ROW)) * s_wl)
            xa = xa * dw_a[:, None]
        xw = (xa * agg[:, None]).to(gl.bfloat16)
        if CHART:
            dGT = sDG.permute((1, 0)).load(OPB)
            dB_acc = mma_v2(xw, dGT, dB_acc)
            accBa = mma_v2(xw, dGT, accBa)
        else:
            dB_acc = mma_v2(xw, sDG.permute((1, 0)).load(OPB), dB_acc)
        if HAS_BETA:
            # colsum_t(xw dG^T)[n] = sum_g (sum_t xw[t, g]) dG[n, g]
            xws = gl.convert_layout(gl.sum(xw.to(gl.float32), axis=0), COL)
            dGm = sDG.load(MMA).to(gl.float32)                  # [NS, GT]
            v_beta = v_beta + gl.convert_layout(
                gl.sum(dGm * xws[None, :], axis=1), COL)
            gl.atomic_add(dBeta_ptr + h * N + n_c, v_beta, sem=SEM)
        if CHART:
            # dB~ / dC~ leave every HBF heads, as the plain launch's HBF-head CTAs (H > HBF here: atomics)
            if ((hh + 1) % HBF == 0) & (hh + 1 < hb_count):
                gl.atomic_add(db_p0, dB_acc, sem=SEM)
                gl.atomic_add(dc_p0, dC_acc, sem=SEM)
                dB_acc = gl.zeros([T, NS], gl.float32, MMA)
                dC_acc = gl.zeros([T, NS], gl.float32, MMA)

    if CHART:
        sole_owner = (NGT_val == 1) & (H <= HBF)
        # the chart's head sums: one writer per element (deterministic).  Rows of the (b, c) chunk tile: obase
        obase = b * NCH + c
        rows_t = obase * T + t_m
        gl.store(dCs_ptr + rows_t[:, None] * N + n_m[None, :], accCs)
        gl.store(dBa_ptr + rows_t[:, None] * N + n_m[None, :], accBa)
        if ns_idx == 0:
            gl.store(MT_ptr + rows_t[:, None] * T + s_m[None, :], accM)        # MT[l, j]
        n_r = ns_idx * NS + gl.arange(0, NS, layout=ROW)
        # T1 [B, nc, G, N, KS]: row n, column n' - (n's KS-block start), the diagonal KS-blocks only
        gl.store(T1_ptr + (obase * N + n_r)[:, None] * KS + (n_m % KS)[None, :], accT1,
                 mask=(n_r[:, None] // KS) == (n_m[None, :] // KS))
    else:
        sole_owner = (NGT_val == 1) & (h_start == 0) & (hb_count >= H)
    db_p = (dBtil_ptr + b * s_dbb + c * s_dbc + t_m[:, None] * s_dbt
            + n_m[None, :] * s_dbn)
    dc_p = (dCtil_ptr + b * s_dcb + c * s_dcc + t_m[:, None] * s_dct
            + n_m[None, :] * s_dcn)
    if sole_owner:
        gl.store(db_p, dB_acc)
        gl.store(dc_p, dC_acc)
    else:
        gl.atomic_add(db_p, dB_acc, sem=SEM)
        gl.atomic_add(dc_p, dC_acc, sem=SEM)


_LAST = {}


def gluon_tile(GT, NGT, gv="full"):
    """The G tile the Gluon body runs: `gv` "full" (the widest tile that fits: <= 64, dividing the whole G, a multiple of 16) | "bwd" (the
    backward's own) | an int.  Always a POWER OF TWO (gl.allocate_shared_memory refuses other dimensions): G = 48 takes 16 x 3 tiles, not
    one tile of 48."""
    gsz = GT * NGT
    if gv == "bwd":
        gk = GT
    elif gv == "full":
        gk = 0
        for c in (64, 32, 16):
            if c <= gsz and gsz % c == 0:
                gk = c
                break
    else:
        gk = int(gv)
    if gk > 0 and gsz % gk == 0 and gk % 16 == 0 and not (gk & (gk - 1)):
        return gk, gsz // gk
    return GT, NGT


def kbc_hb_gluon(*, Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all,
                 dw_c, dY, dCtil_out, dBtil_out, dDelta_out, H, Bsz, nc, NGT,
                 nsplit, T, N, GT, strides, has_dw, dcs_ink, trap_diag, idx64,
                 beta=None, gamma=None, dBeta=None, dGamma=None,
                 dalpha=None, gt_bwd=None, chart=None):
    """Launch; `strides` is the Triton launch's stride tuple (same order).

    `chart` (the local chart cotangent, the FUSED emitter): dict(dCs, dBa, MT, T1, KS) -- the head-sum buffers
    in the emitter's Kch1 output layout (one head slice).  The launch then runs every head in one CTA per
    (b, chunk, N slice) (`chart_refusal_fused` decided it is served) and flushes dB~ / dC~ every `hb` heads."""
    from .fused_fwd_kernel import SEG_RESET_C
    w, wm, wn, mr, hb = _cfg()
    if chart is not None:
        return _kbc_chart_launch(Ctil=Ctil, Btil=Btil, x_scaled=x_scaled, delta_cumsum=delta_cumsum, a_log=a_log,
                                 S_all=S_all, dG_all=dG_all, dw_c=dw_c, dY=dY, dCtil_out=dCtil_out,
                                 dBtil_out=dBtil_out, dDelta_out=dDelta_out, H=H, Bsz=Bsz, nc=nc, NGT=NGT,
                                 nsplit=nsplit, T=T, N=N, GT=GT, strides=strides, has_dw=has_dw, dcs_ink=dcs_ink,
                                 trap_diag=trap_diag, idx64=idx64, dalpha=dalpha, gt_bwd=gt_bwd, chart=chart)
    # GT: this kernel's outputs do not depend on the G tiling (dC / dB /
    # dDelta are sums over g), and the backward's GT is sized for the
    # TRITON bodies' shared memory (32 on 100 KB parts).  A narrower tile
    # repeats the G-independent score products (dSb B, dSb^T C) and their
    # atomics once per tile, so the Gluon body takes the full G when it fits
    # (`SSD_MIMO_GLUON_KBC_GT`: "full" (default) | "bwd" | an int).
    gsz = GT * NGT
    GT, NGT = gluon_tile(GT, NGT, _env("GLUON_KBC_GT", "full").lower())
    # PIPE needs 16-bit sources (cp.async copies bytes, no cast)
    x_gc = strides[10] == 1           # s_xg: x is G-contiguous (token-major)
    pipe = pipe_wanted(T=T, NS=N // nsplit, GT=GT, device=Ctil.device) and all(
        t.dtype == torch.bfloat16 for t in (dY, x_scaled, S_all, dG_all)) \
        and _cp_async_ok(strides, (dY, x_scaled, S_all, dG_all))
    grid = (Bsz, (H + hb - 1) // hb, nc * NGT * nsplit)
    _kbc_hb_gl_kernel[grid](
        Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all, dw_c, dY,
        dCtil_out, dBtil_out, dDelta_out,
        beta if beta is not None else a_log, gamma if gamma is not None else a_log,
        dBeta if dBeta is not None else dDelta_out,
        dGamma if dGamma is not None else dDelta_out,
        H, *strides, NGT,
        NSPLIT=nsplit, T=T, N=N, GT=GT, HB=hb,
        HAS_DW=has_dw, DCS_INK=dcs_ink, TRAP_DIAG=trap_diag, IDX64=idx64,
        WM=wm, WN=wn, SEG_RESET=float(SEG_RESET_C.value), SEM=ACC_SEM.value,
        HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None,
        PIPE=pipe, X_GC=x_gc,
        **(dict(KBC_DALPHA=True, GTB=gt_bwd, dAlpha_ptr=dalpha,
                s_dab=dalpha.stride(0), s_dah=dalpha.stride(1),
                s_dag=dalpha.stride(2), s_dac=dalpha.stride(3))
           if dalpha is not None else {}),
        num_warps=w, num_stages=1, **({"maxnreg": mr} if mr else {}))
    _LAST.update(route="gluon_pipe" if pipe else "gluon", warps=w, wm=wm,
                 wn=wn, maxnreg=mr, hb=hb)


# ── the local chart cotangent, FUSED into this body (CHART) ─────────────
def chart_refusal_fused(*, GT, NGT, N, nsplit, KS, dalpha) -> str | None:
    """Why the CHART body cannot serve a call the plain Gluon body serves (None = it can).  The caller has already
    established that the plain Gluon Kbc serves (`refusal` is None): bf16 MMA, rotation- and bias-free, T 64."""
    gk, ngk = gluon_tile(GT, NGT, _env("GLUON_KBC_GT", "full").lower())
    if ngk != 1:
        return f"G_SIZE = {GT * NGT} takes {ngk} Gluon G tiles (the chart's head sums need the whole P per CTA)"
    if dalpha is not None:
        return "the GLUON_KBC_DALPHA hand-off lever"
    NS = N // nsplit
    if NS % KS:
        return f"N slice {NS} not a multiple of the chart superblock {KS}"
    return None


def _chart_pipe(T, NS, GT, device, tensors, strides) -> bool:
    """The CHART body runs one CTA per SM (its accumulators take the registers of a second one), so the cp.async
    double buffer is worth it whenever ONE CTA's shared memory (double buffer included) fits.
    `SSD_MIMO_GLUON_KBC_CHART_PIPE` = "auto" (default) | "0" | "1"."""
    v = _env("GLUON_KBC_CHART_PIPE", "auto").lower()
    if v in ("0", "off"):
        return False
    if not (all(t.dtype == torch.bfloat16 for t in tensors) and _cp_async_ok(strides, tensors)):
        return False
    props = torch.cuda.get_device_properties(device)
    per_sm = int(getattr(props, "shared_memory_per_multiprocessor", 0) or 0)
    need = (2 * 2 * T * NS + 2 * 2 * (2 * T * GT + 2 * NS * GT) + (0 if GT == T else 2 * T * T) + 1024)
    return v in ("1", "on") or (per_sm > 0 and need <= per_sm)


def _kbc_chart_launch(*, Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all, dw_c, dY, dCtil_out, dBtil_out,
                      dDelta_out, H, Bsz, nc, NGT, nsplit, T, N, GT, strides, has_dw, dcs_ink, trap_diag, idx64,
                      dalpha, gt_bwd, chart):
    from .fused_fwd_kernel import SEG_RESET_C
    _w, _wm, _wn, _mr, hbf = _cfg()
    w = int(_env("GLUON_KBC_CHART_WARPS", "8"))
    wn = int(_env("GLUON_KBC_CHART_WN", "4"))
    mr = int(_env("GLUON_KBC_CHART_MAXNREG", "0"))
    GT, NGT = gluon_tile(GT, NGT, _env("GLUON_KBC_GT", "full").lower())
    if NGT != 1:
        raise RuntimeError("internal: the CHART Kbc needs one G tile (chart_refusal_fused)")
    x_gc = strides[10] == 1
    pipe = _chart_pipe(T, N // nsplit, GT, Ctil.device, (dY, x_scaled, S_all, dG_all), strides)
    grid = (Bsz, 1, nc * NGT * nsplit)
    _kbc_hb_gl_kernel[grid](
        Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all, dw_c, dY,
        dCtil_out, dBtil_out, dDelta_out,
        a_log, a_log, dDelta_out, dDelta_out,
        H, *strides, NGT,
        NSPLIT=nsplit, T=T, N=N, GT=GT, HB=H,
        HAS_DW=has_dw, DCS_INK=dcs_ink, TRAP_DIAG=trap_diag, IDX64=idx64,
        WM=w // wn, WN=wn, SEG_RESET=float(SEG_RESET_C.value), SEM=ACC_SEM.value,
        HAS_BETA=False, HAS_GAMMA=False, PIPE=pipe, X_GC=x_gc,
        CHART=True, HBF=hbf, KS=int(chart["KS"]),
        dCs_ptr=chart["dCs"], dBa_ptr=chart["dBa"], MT_ptr=chart["MT"], T1_ptr=chart["T1"], NCH=nc,
        num_warps=w, num_stages=1, **({"maxnreg": mr} if mr else {}))
    _LAST.update(route=("gluon_chart_pipe" if pipe else "gluon_chart"), warps=w, wm=w // wn, wn=wn, maxnreg=mr,
                 hb=H, hbf=hbf)
