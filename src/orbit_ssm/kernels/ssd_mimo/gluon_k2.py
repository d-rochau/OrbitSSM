"""Gluon twin of `_bwd_k2_kernel` (Kc) on the real-mode R1O1 chunk-parallel
route: O = R = 1, unrotated, CHUNK_PARALLEL, HAS_DW, FUSE_KB2 with NO_DBAGG
(the head-batched Kbc owns dB_agg), SHARED_CB (the head-independent C B^T
tile), optional per-head biases, D-skip and the trapezoid adjoint.

STATUS.  Opt-in only: lever `SSD_MIMO_GLUON_K2=1`; `fused_chunk_bwd_5kernel`
calls `refusal()` before the Triton Kc launch and every refusal is by name.

WHY.  The Triton Kc sits at 255 registers in every config (the MMA work is
replicated across warps).  Here the [T, GT]
and [T, T] tiles carry an explicit WM x WN warp grid, every global tile is
loaded coalesced into shared memory, and the MMA operands (transposes
included) are read from it.  mma_v2 serves sm_80 .. sm_120; a device reporting
capability major 9 (wgmma) is refused until that body exists.

WHAT IT COMPUTES per CTA (b, h, chunk c, G tile) -- the Triton body's
arithmetic, biases as exact fp32 rank-1 terms:
    BdG    = B dG + 1 (beta^T dG)                        [T, GT]
    S0     = CB + rowsum(C_eff * beta) 1^T + 1 (B gamma)^T,  C_eff = C + gamma
    Score  = S0 * decay mask            (strictly causal under the trapezoid)
    dx     = dec_agg * BdG + bf16(Score)^T dY
    Msp    = Score * (dY bf16(x dw)^T)                   (SPANDELTA pairs)
    q      = rowsum(BdG * x dw);  Msp[T-1, :] += dec_agg * q
    dDW   += rowsum(dx * x);   dx = dx * dw + Dsk * dY (+ nd * dY)
    dDsk  += colsum(dY * x);   dSdot += rowsum(dY * x)   (trapezoid)
    dDelta += a (span_sum(Msp) + alpha * dalpha)
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


def gluon_k2_wanted() -> bool:
    return os.environ.get("SSD_MIMO_GLUON_K2", "0") == "1"


def _cfg():
    w = int(os.environ.get("SSD_MIMO_GLUON_K2_WARPS", "8"))
    wn = int(os.environ.get("SSD_MIMO_GLUON_K2_WN", "4"))
    mr = int(os.environ.get("SSD_MIMO_GLUON_K2_MAXNREG", "128"))
    return w, w // wn, wn, mr


def refusal(*, device, mma_dt_is_bf16, dtypes_bf16, shared_cb, fuse_kb2,
            no_dbagg, has_dw, O, R, use_r_tile, rot_in_kernel, has_theta_h,
            has_rot_bias, pre_rot, pair_half, dd_in_kbc, ts_abl, dcs_ink,
            dskip_slot, use_slots, T, N, GT):
    if not (mma_dt_is_bf16 and dtypes_bf16):
        return "dtypes (bf16 operands only)"
    if not (shared_cb and fuse_kb2 and no_dbagg and has_dw):
        return "route (needs SHARED_CB + FUSE_KB2 + NO_DBAGG + dw)"
    if O != 1 or R != 1 or use_r_tile:
        return "R/O > 1"
    if rot_in_kernel or has_theta_h or has_rot_bias or pre_rot or pair_half:
        return "rotation"
    if dd_in_kbc or ts_abl or dskip_slot or use_slots:
        return "lever outside the envelope"
    if dcs_ink:
        return "DCS_INK (off at R = O = 1)"
    w, wm, wn, _ = _cfg()
    if T != 64 or T % (16 * wm) or GT % (8 * wn) or N % 16:
        return f"tile T={T} GT={GT} N={N} vs warp grid [{wm},{wn}]"
    if torch.cuda.get_device_capability(device)[0] == 9:
        return "sm_90: needs the wgmma body (not built)"
    return None


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _segadd(v1, f1, v2, f2):
    return gl.where(f2 != 0, v2, v1 + v2), f1 | f2


@gluon.jit
def _k2_gl_kernel(
    C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, dG_ptr, dY_ptr, DW_ptr, CB_ptr,
    Beta_ptr, Gamma_ptr, Dsk_ptr, TrapND_ptr, dAlpha_ptr,
    dX_ptr, dDW_ptr, dDelta_ptr, dDsk_ptr, dSdot_ptr,
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg,
    s_dgb, s_dgh, s_dgc, s_dgn, s_dgg,
    s_wb, s_wh, s_wl,
    s_dwb, s_dwh, s_dwl,
    s_cbb, s_cbc,
    s_dab, s_dah, s_dag, s_dac,
    s_bh, s_ch, HPG_B, HPG_C,
    NGT_val, G_SIZE,
    T: gl.constexpr, N: gl.constexpr, GT: gl.constexpr,
    HAS_BETA: gl.constexpr, HAS_GAMMA: gl.constexpr,
    HAS_DSKIP: gl.constexpr, DSKIP_DD: gl.constexpr,
    HAS_TRAP_ADJ: gl.constexpr, IDX64: gl.constexpr,
    WM: gl.constexpr, WN: gl.constexpr, SEM: gl.constexpr,
):
    NW: gl.constexpr = WM * WN
    MMA: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, WN], instr_shape=[16, 8])
    OPA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    OPB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    ROW: gl.constexpr = gl.SliceLayout(1, MMA)
    COL: gl.constexpr = gl.SliceLayout(0, MMA)
    B_COL: gl.constexpr = gl.SliceLayout(0, OPB)
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    BL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    BL_R: gl.constexpr = gl.SliceLayout(1, BL)
    BL_C: gl.constexpr = gl.SliceLayout(0, BL)
    # [T, T] span-sum scan layout (the scan does not lower on MMA layouts)
    BT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NW, 1], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                              order=[1, 0])

    b = gl.program_id(0)
    h = gl.program_id(1)
    c = gl.program_id(2) // NGT_val
    gt_idx = gl.program_id(2) % NGT_val
    if IDX64:
        b = b.to(gl.int64)
        h = h.to(gl.int64)
    a_val = -gl.exp(gl.load(A_ptr + h))
    bg = h // HPG_B
    cgh = h // HPG_C

    # ── the segmented prefix ─────────────────────────────────────────────
    t1 = gl.arange(0, T, layout=L1)
    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
    d = gl.load(dbase + t1 * s_dt)
    rst = d < 0.0
    ri = rst.to(gl.int32)
    D1 = gl.where(rst, 0.0, d)
    seg1 = gl.associative_scan(ri, 0, _add)
    seg_end = gl.max(seg1, axis=0)
    D_end = gl.sum(gl.where(t1 == T - 1, D1, 0.0), axis=0)
    alpha = gl.where(seg_end == 0, gl.exp(a_val * D_end), 0.0)
    agg1 = gl.where(seg1 == seg_end, gl.exp(a_val * (D_end - D1)), 0.0)

    # ── stage B, dG, dY, x (coalesced) ───────────────────────────────────
    rt = gl.arange(0, T, layout=BL_R)
    cn = gl.arange(0, N, layout=BL_C)
    rn = gl.arange(0, N, layout=BL_R)
    cg = gt_idx * GT + gl.arange(0, GT, layout=BL_C)
    rg = gt_idx * GT + gl.arange(0, GT, layout=BL_R)
    ct = gl.arange(0, T, layout=BL_C)
    Bt = gl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + rt[:, None] * s_bt
                 + cn[None, :] * s_bn)
    sB = gl.allocate_shared_memory(gl.bfloat16, [T, N], SH, Bt.to(gl.bfloat16))
    sG = gl.allocate_shared_memory(
        gl.bfloat16, [N, GT], SH,
        gl.load(dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc
                + rn[:, None] * s_dgn + cg[None, :] * s_dgg).to(gl.bfloat16))
    sY = gl.allocate_shared_memory(
        gl.bfloat16, [T, GT], SH,
        gl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc
                + rt[:, None] * s_yt + cg[None, :] * s_yg).to(gl.bfloat16))
    sX = gl.allocate_shared_memory(
        gl.bfloat16, [GT, T], SH,
        gl.load(X_ptr + b * s_xb + h * s_xh + (c * T + ct[None, :]) * s_xl
                + rg[:, None] * s_xg).to(gl.bfloat16))
    sS = gl.allocate_shared_memory(gl.bfloat16, [T, T], SH)

    # bias rank-1 vectors, from the coalesced tiles (fp32, exact)
    if HAS_GAMMA:
        ga_bl = gl.load(Gamma_ptr + h * N + cn).to(gl.float32)
        bga = gl.convert_layout(gl.sum(Bt.to(gl.float32) * ga_bl[None, :], axis=1), COL)
    if HAS_BETA:
        be_bl = gl.load(Beta_ptr + h * N + cn).to(gl.float32)
        Ce = gl.load(C_ptr + b * s_cb + c * s_cc + cgh * s_ch + rt[:, None] * s_ct
                     + cn[None, :] * s_cn).to(gl.float32)
        if HAS_GAMMA:
            Ce = Ce + ga_bl[None, :]
        cbe = gl.convert_layout(gl.sum(Ce * be_bl[None, :], axis=1), ROW)
    gl.barrier()

    # BdG = B dG (+ 1 (beta^T dG))
    BdG = mma_v2(sB.load(OPA), sG.load(OPB), gl.zeros([T, GT], gl.float32, MMA))
    if HAS_BETA:
        be_r = gl.load(Beta_ptr + h * N + gl.arange(0, N, layout=ROW)).to(gl.float32)
        bdg = gl.sum(sG.load(MMA).to(gl.float32) * be_r[:, None], axis=0)   # [GT]
        BdG = BdG + bdg[None, :]

    t_m = gl.arange(0, T, layout=ROW)
    s_m = gl.arange(0, T, layout=COL)
    D_r = gl.convert_layout(D1, ROW)
    D_c = gl.convert_layout(D1, COL)
    sg_r = gl.convert_layout(seg1, ROW)
    sg_c = gl.convert_layout(seg1, COL)
    if HAS_TRAP_ADJ:
        causal = t_m[:, None] > s_m[None, :]
    else:
        causal = t_m[:, None] >= s_m[None, :]
    S0 = gl.load(CB_ptr + b * s_cbb + c * s_cbc + t_m[:, None] * T + s_m[None, :])
    if HAS_BETA:
        S0 = S0 + cbe[:, None]
    if HAS_GAMMA:
        S0 = S0 + bga[None, :]
    Score = S0 * gl.exp(gl.where(causal & (sg_r[:, None] == sg_c[None, :]),
                                 a_val * (D_r[:, None] - D_c[None, :]),
                                 float("-inf")))
    sS.store(Score.to(gl.bfloat16))
    gl.barrier()
    agg_r = gl.convert_layout(agg1, ROW)
    dx = mma_v2(sS.permute((1, 0)).load(OPA), sY.load(OPB), agg_r[:, None] * BdG)

    # Msp = Score * (dY bf16(x dw)^T)
    dw_b = gl.load(DW_ptr + b * s_wb + h * s_wh
                   + (c * T + gl.arange(0, T, layout=B_COL)) * s_wl)
    xsT = (sX.load(OPB).to(gl.float32) * dw_b[None, :]).to(gl.bfloat16)
    Msp = Score * mma_v2(sY.load(OPA), xsT, gl.zeros([T, T], gl.float32, MMA))

    x_m = sX.permute((1, 0)).load(MMA).to(gl.float32)        # raw x [T, GT]
    dw_r = gl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + t_m) * s_wl)
    q = gl.sum(BdG * (x_m * dw_r[:, None]), axis=1)          # [T] ROW
    gl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + t_m) * s_dwl,
                  gl.sum(dx * x_m, axis=1), sem=SEM)
    dx = dx * dw_r[:, None]
    g_m = gt_idx * GT + gl.arange(0, GT, layout=COL)
    if HAS_DSKIP or HAS_TRAP_ADJ:
        # the elementwise terms read dY at its STORED precision (fp32 unless
        # GATE_DY_IO), exactly as the Triton body does -- not the bf16 copy
        dY_m = gl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc
                       + t_m[:, None] * s_yt + g_m[None, :] * s_yg).to(gl.float32)
    if HAS_DSKIP:
        dx = dx + gl.load(Dsk_ptr + h * G_SIZE + g_m)[None, :] * dY_m
        if DSKIP_DD:
            gl.atomic_add(dDsk_ptr + h * G_SIZE + g_m, gl.sum(dY_m * x_m, axis=0),
                          sem=SEM)
    if HAS_TRAP_ADJ:
        nd = gl.load(TrapND_ptr + b * s_wb + h * s_wh + (c * T + t_m) * s_wl)
        dx = dx + nd[:, None] * dY_m
        gl.atomic_add(dSdot_ptr + b * s_wb + h * s_wh + (c * T + t_m) * s_wl,
                      gl.sum(dY_m * x_m, axis=1), sem=SEM)
    gl.store(dX_ptr + b * s_xb + h * s_xh + g_m[None, :] * s_xg
             + (c * T + t_m[:, None]) * s_xl, dx)

    # SPANDELTA: score pairs + the write row (j, T-1) + the carry
    aq = gl.convert_layout(agg_r * q, COL)
    Msp = gl.where((t_m == T - 1)[:, None], Msp + aq[None, :], Msp)
    Mb = gl.convert_layout(Msp, BT)
    Rrev = gl.associative_scan(Mb, 0, _add, reverse=True)
    tb = gl.arange(0, T, layout=gl.SliceLayout(1, BT))
    sb = gl.arange(0, T, layout=gl.SliceLayout(0, BT))
    span = gl.sum(gl.where(sb[None, :] < tb[:, None], Rrev, 0.0), axis=1)
    dalpha_c = gl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag
                       + c * s_dac)
    dD = a_val * (span + alpha * dalpha_c)
    gl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + tb * s_dt, dD,
                  sem=SEM)


_LAST = {}


def k2_gluon(*, Ctil, Btil, x_scaled, delta_cumsum, a_log, dG_all, dY, dw_c,
             cb, beta, gamma, d_skip, trap_nd, dalpha_all, dX, dDW_out,
             dDelta_out, dDsk_out, dSdot, Bsz, H, nc, NGT, T, N, GT, G_SIZE,
             strides, idx64, dskip_dd):
    w, wm, wn, mr = _cfg()
    dummy = a_log
    grid = (Bsz, H, nc * NGT)
    _k2_gl_kernel[grid](
        Ctil, Btil, x_scaled, delta_cumsum, a_log, dG_all, dY, dw_c, cb,
        beta if beta is not None else dummy,
        gamma if gamma is not None else dummy,
        d_skip if d_skip is not None else dummy,
        trap_nd if trap_nd is not None else dummy,
        dalpha_all,
        dX, dDW_out, dDelta_out, dDsk_out, dSdot,
        *strides, NGT, G_SIZE,
        T=T, N=N, GT=GT,
        HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None,
        HAS_DSKIP=d_skip is not None, DSKIP_DD=bool(d_skip is not None and dskip_dd),
        HAS_TRAP_ADJ=trap_nd is not None, IDX64=idx64,
        WM=wm, WN=wn, SEM=ACC_SEM.value,
        num_warps=w, num_stages=1, **({"maxnreg": mr} if mr else {}))
    _LAST.update(route="gluon", warps=w, wm=wm, wn=wn, maxnreg=mr)


# ═══ PIPELINED, HEAD-BATCHED Kc (lever SSD_MIMO_GLUON_K2_PIPE=1) ═════════
# The one-shot body above is latency-bound (one load burst per CTA, nothing
# overlapping it).  Here a CTA owns
# HB heads of one (b, c, G tile): B (and C under the biases) stay resident
# in shared memory, and the NEXT head's dG / dY / x tiles are copied with
# cp.async into the other half of a double buffer while this head computes.
# The per-head arithmetic is the one-shot body's, unchanged.

@gluon.jit
def _k2_issue(sG, sY, sX, dG_ptr, dY_ptr, X_ptr, b, h, c,
              s_dgb, s_dgh, s_dgc, s_dgn, s_dgg,
              s_yb, s_yh, s_yc, s_yt, s_yg,
              s_xb, s_xh, s_xg, s_xl,
              rn, cg, rt, rg, ct, T: gl.constexpr):
    async_load(sG, dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc
               + rn[:, None] * s_dgn + cg[None, :] * s_dgg)
    async_load(sY, dY_ptr + b * s_yb + h * s_yh + c * s_yc
               + rt[:, None] * s_yt + cg[None, :] * s_yg)
    async_load(sX, X_ptr + b * s_xb + h * s_xh + (c * T + ct[None, :]) * s_xl
               + rg[:, None] * s_xg)
    commit_group()


@gluon.jit
def _k2_pipe_kernel(
    C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, dG_ptr, dY_ptr, DW_ptr, CB_ptr,
    Beta_ptr, Gamma_ptr, Dsk_ptr, TrapND_ptr, dAlpha_ptr,
    dX_ptr, dDW_ptr, dDelta_ptr, dDsk_ptr, dSdot_ptr,
    s_cb, s_cc, s_ct, s_cn,
    s_bb, s_bc, s_bt, s_bn,
    s_xb, s_xh, s_xg, s_xl,
    s_db, s_dh, s_dc, s_dt,
    s_yb, s_yh, s_yc, s_yt, s_yg,
    s_dgb, s_dgh, s_dgc, s_dgn, s_dgg,
    s_wb, s_wh, s_wl,
    s_dwb, s_dwh, s_dwl,
    s_cbb, s_cbc,
    s_dab, s_dah, s_dag, s_dac,
    s_bh, s_ch, HPG_B, HPG_C,
    NGT_val, G_SIZE, H,
    T: gl.constexpr, N: gl.constexpr, GT: gl.constexpr, HB: gl.constexpr,
    HAS_BETA: gl.constexpr, HAS_GAMMA: gl.constexpr,
    HAS_DSKIP: gl.constexpr, DSKIP_DD: gl.constexpr,
    HAS_TRAP_ADJ: gl.constexpr, IDX64: gl.constexpr,
    WM: gl.constexpr, WN: gl.constexpr, SEM: gl.constexpr,
):
    NW: gl.constexpr = WM * WN
    MMA: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, WN], instr_shape=[16, 8])
    OPA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    OPB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    ROW: gl.constexpr = gl.SliceLayout(1, MMA)
    COL: gl.constexpr = gl.SliceLayout(0, MMA)
    B_COL: gl.constexpr = gl.SliceLayout(0, OPB)
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    BL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    BL_R: gl.constexpr = gl.SliceLayout(1, BL)
    BL_C: gl.constexpr = gl.SliceLayout(0, BL)
    BT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NW, 1], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                              order=[1, 0])
    SH3: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                               order=[2, 1, 0])

    b = gl.program_id(0)
    hblk = gl.program_id(1)
    c = gl.program_id(2) // NGT_val
    gt_idx = gl.program_id(2) % NGT_val
    if IDX64:
        b = b.to(gl.int64)
    h0 = hblk * HB
    hbc = gl.minimum(HB, H - h0)
    bg = h0 // HPG_B
    cgh = h0 // HPG_C

    rt = gl.arange(0, T, layout=BL_R)
    cn = gl.arange(0, N, layout=BL_C)
    rn = gl.arange(0, N, layout=BL_R)
    cg = gt_idx * GT + gl.arange(0, GT, layout=BL_C)
    rg = gt_idx * GT + gl.arange(0, GT, layout=BL_R)
    ct = gl.arange(0, T, layout=BL_C)

    # head-shared, resident: B (and C under beta)
    sB = gl.allocate_shared_memory(B_ptr.dtype.element_ty, [T, N], SH)
    async_load(sB, B_ptr + b * s_bb + c * s_bc + bg * s_bh + rt[:, None] * s_bt
               + cn[None, :] * s_bn)
    if HAS_BETA:
        sC = gl.allocate_shared_memory(C_ptr.dtype.element_ty, [T, N], SH)
        async_load(sC, C_ptr + b * s_cb + c * s_cc + cgh * s_ch
                   + rt[:, None] * s_ct + cn[None, :] * s_cn)
    commit_group()
    # per-head double buffer
    sG = gl.allocate_shared_memory(dG_ptr.dtype.element_ty, [2, N, GT], SH)
    sY = gl.allocate_shared_memory(dY_ptr.dtype.element_ty, [2, T, GT], SH)
    sX = gl.allocate_shared_memory(X_ptr.dtype.element_ty, [2, GT, T], SH)
    sS = gl.allocate_shared_memory(gl.bfloat16, [T, T], SH)
    _k2_issue(sG.index(0), sY.index(0), sX.index(0), dG_ptr, dY_ptr, X_ptr,
              b, h0, c, s_dgb, s_dgh, s_dgc, s_dgn, s_dgg,
              s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl,
              rn, cg, rt, rg, ct, T)

    t1 = gl.arange(0, T, layout=L1)
    t_m = gl.arange(0, T, layout=ROW)
    s_m = gl.arange(0, T, layout=COL)
    g_m = gt_idx * GT + gl.arange(0, GT, layout=COL)
    n_r = gl.arange(0, N, layout=ROW)
    n_c = gl.arange(0, N, layout=COL)
    tb = gl.arange(0, T, layout=gl.SliceLayout(1, BT))
    sb = gl.arange(0, T, layout=gl.SliceLayout(0, BT))
    for hh in range(hbc):
        st = hh % 2
        h = h0 + hh
        if IDX64:
            h = h.to(gl.int64)
        gl.barrier()                 # the other slot's last reader is done
        if hh + 1 < hbc:
            _k2_issue(sG.index(1 - st), sY.index(1 - st), sX.index(1 - st),
                      dG_ptr, dY_ptr, X_ptr, b, h + 1, c,
                      s_dgb, s_dgh, s_dgc, s_dgn, s_dgg,
                      s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl,
                      rn, cg, rt, rg, ct, T)
            wait_group(1)
        else:
            wait_group(0)
        gl.barrier()
        gG = sG.index(st)
        gY = sY.index(st)
        gX = sX.index(st)

        a_val = -gl.exp(gl.load(A_ptr + h))
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        d = gl.load(dbase + t1 * s_dt)
        rst = d < 0.0
        ri = rst.to(gl.int32)
        D1 = gl.where(rst, 0.0, d)
        seg1 = gl.associative_scan(ri, 0, _add)
        seg_end = gl.max(seg1, axis=0)
        D_end = gl.sum(gl.where(t1 == T - 1, D1, 0.0), axis=0)
        alpha = gl.where(seg_end == 0, gl.exp(a_val * D_end), 0.0)
        agg1 = gl.where(seg1 == seg_end, gl.exp(a_val * (D_end - D1)), 0.0)

        BdG = mma_v2(sB.load(OPA).to(gl.bfloat16), gG.load(OPB).to(gl.bfloat16),
                     gl.zeros([T, GT], gl.float32, MMA))
        S0 = gl.load(CB_ptr + b * s_cbb + c * s_cbc + t_m[:, None] * T + s_m[None, :])
        if HAS_GAMMA:
            ga_c = gl.load(Gamma_ptr + h * N + n_c).to(gl.float32)
            bga = gl.sum(sB.load(MMA).to(gl.float32) * ga_c[None, :], axis=1)
            S0 = S0 + gl.convert_layout(bga, COL)[None, :]
        if HAS_BETA:
            be_c = gl.load(Beta_ptr + h * N + n_c).to(gl.float32)
            Ce = sC.load(MMA).to(gl.float32)
            if HAS_GAMMA:
                Ce = Ce + ga_c[None, :]
            S0 = S0 + gl.sum(Ce * be_c[None, :], axis=1)[:, None]
            be_r = gl.load(Beta_ptr + h * N + n_r).to(gl.float32)
            bdg = gl.sum(gG.load(MMA).to(gl.float32) * be_r[:, None], axis=0)
            BdG = BdG + bdg[None, :]

        D_r = gl.convert_layout(D1, ROW)
        D_c = gl.convert_layout(D1, COL)
        sg_r = gl.convert_layout(seg1, ROW)
        sg_c = gl.convert_layout(seg1, COL)
        if HAS_TRAP_ADJ:
            causal = t_m[:, None] > s_m[None, :]
        else:
            causal = t_m[:, None] >= s_m[None, :]
        Score = S0 * gl.exp(gl.where(causal & (sg_r[:, None] == sg_c[None, :]),
                                     a_val * (D_r[:, None] - D_c[None, :]),
                                     float("-inf")))
        sS.store(Score.to(gl.bfloat16))
        gl.barrier()
        agg_r = gl.convert_layout(agg1, ROW)
        dx = mma_v2(sS.permute((1, 0)).load(OPA), gY.load(OPB).to(gl.bfloat16),
                    agg_r[:, None] * BdG)
        dw_b = gl.load(DW_ptr + b * s_wb + h * s_wh
                       + (c * T + gl.arange(0, T, layout=B_COL)) * s_wl)
        xsT = (gX.load(OPB).to(gl.float32) * dw_b[None, :]).to(gl.bfloat16)
        Msp = Score * mma_v2(gY.load(OPA).to(gl.bfloat16), xsT,
                             gl.zeros([T, T], gl.float32, MMA))
        x_m = gX.permute((1, 0)).load(MMA).to(gl.float32)
        dw_r = gl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + t_m) * s_wl)
        q = gl.sum(BdG * (x_m * dw_r[:, None]), axis=1)
        gl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + t_m) * s_dwl,
                      gl.sum(dx * x_m, axis=1), sem=SEM)
        dx = dx * dw_r[:, None]
        if HAS_DSKIP or HAS_TRAP_ADJ:
            dY_m = gY.load(MMA).to(gl.float32)       # the STORED precision
        if HAS_DSKIP:
            dx = dx + gl.load(Dsk_ptr + h * G_SIZE + g_m)[None, :] * dY_m
            if DSKIP_DD:
                gl.atomic_add(dDsk_ptr + h * G_SIZE + g_m,
                              gl.sum(dY_m * x_m, axis=0), sem=SEM)
        if HAS_TRAP_ADJ:
            nd = gl.load(TrapND_ptr + b * s_wb + h * s_wh + (c * T + t_m) * s_wl)
            dx = dx + nd[:, None] * dY_m
            gl.atomic_add(dSdot_ptr + b * s_wb + h * s_wh + (c * T + t_m) * s_wl,
                          gl.sum(dY_m * x_m, axis=1), sem=SEM)
        gl.store(dX_ptr + b * s_xb + h * s_xh + g_m[None, :] * s_xg
                 + (c * T + t_m[:, None]) * s_xl, dx)

        aq = gl.convert_layout(agg_r * q, COL)
        Msp = gl.where((t_m == T - 1)[:, None], Msp + aq[None, :], Msp)
        Rrev = gl.associative_scan(gl.convert_layout(Msp, BT), 0, _add, reverse=True)
        span = gl.sum(gl.where(sb[None, :] < tb[:, None], Rrev, 0.0), axis=1)
        dalpha_c = gl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag
                           + c * s_dac)
        gl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + tb * s_dt,
                      a_val * (span + alpha * dalpha_c), sem=SEM)


def k2_pipe_wanted() -> bool:
    return os.environ.get("SSD_MIMO_GLUON_K2_PIPE", "0") == "1"


def k2_gluon_pipe(*, Ctil, Btil, x_scaled, delta_cumsum, a_log, dG_all, dY,
                  dw_c, cb, beta, gamma, d_skip, trap_nd, dalpha_all, dX,
                  dDW_out, dDelta_out, dDsk_out, dSdot, Bsz, H, nc, NGT, T, N,
                  GT, G_SIZE, strides, idx64, dskip_dd, hpg_b, hpg_c):
    w, wm, wn, mr = _cfg()
    hb = int(os.environ.get("SSD_MIMO_GLUON_K2_HB", "4"))
    # a head block must not straddle a B / C group (the resident tiles)
    import math
    # HPG = heads per B / C group: with ungrouped B / C (the route's
    # bc_per_head=False view) every head shares the one tile, HPG = H
    hb = max(1, math.gcd(hb, math.gcd(hpg_b, hpg_c)))
    dummy = a_log
    grid = (Bsz, (H + hb - 1) // hb, nc * NGT)
    _k2_pipe_kernel[grid](
        Ctil, Btil, x_scaled, delta_cumsum, a_log, dG_all, dY, dw_c, cb,
        beta if beta is not None else dummy,
        gamma if gamma is not None else dummy,
        d_skip if d_skip is not None else dummy,
        trap_nd if trap_nd is not None else dummy,
        dalpha_all,
        dX, dDW_out, dDelta_out, dDsk_out, dSdot,
        *strides, NGT, G_SIZE, H,
        T=T, N=N, GT=GT, HB=hb,
        HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None,
        HAS_DSKIP=d_skip is not None, DSKIP_DD=bool(d_skip is not None and dskip_dd),
        HAS_TRAP_ADJ=trap_nd is not None, IDX64=idx64,
        WM=wm, WN=wn, SEM=ACC_SEM.value,
        num_warps=w, num_stages=1, **({"maxnreg": mr} if mr else {}))
    _LAST.update(route="gluon_pipe", warps=w, wm=wm, wn=wn, maxnreg=mr, hb=hb)
