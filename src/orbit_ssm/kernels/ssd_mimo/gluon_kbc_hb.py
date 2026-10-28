
from __future__ import annotations
import os
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.ampere.async_copy import async_load, commit_group, wait_group
from .common import ACC_SEM
from .fused_fwd_kernel import _env

def gluon_kbc_wanted(guard: bool=True, T: int | None=None) -> bool:
    from . import levers
    return levers.on('real_gluon_kbc', levers.real_group(T), guard=guard)

def _cfg():
    w = int(_env('GLUON_KBC_WARPS', '8'))
    wn = int(_env('GLUON_KBC_WN', '4'))
    mr = int(_env('GLUON_KBC_MAXNREG', '128'))
    hb = int(_env('GLUON_KBC_HB', '8'))
    return (w, w // wn, wn, mr, hb)

def pipe_wanted(*, T, NS, GT, device) -> bool:
    v = _env('GLUON_KBC_PIPE', 'auto').lower()
    if v in ('0', 'off'):
        return False
    if v in ('1', 'on'):
        return True
    props = torch.cuda.get_device_properties(device)
    per_sm = int(getattr(props, 'shared_memory_per_multiprocessor', 0) or 0)
    need = 2 * 2 * T * NS + 2 * 2 * (2 * T * GT + 2 * NS * GT) + (0 if GT == T else 2 * T * T) + 1024
    return per_sm > 0 and 2 * need <= per_sm

def _cp_async_ok(strides, tensors) -> bool:
    _cb, _cc, _ct, _cn, _bb, _bc, _bt, _bn, s_xb, s_xh, s_xg, s_xl, _db, _dh, _dc, _dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg = strides[:31]
    x_in, x_out = (s_xg, s_xl) if s_xg == 1 else (s_xl, s_xg)
    inner = (x_in, s_yg, s_sg, s_gg)
    outer = (s_xb, s_xh, x_out, s_yb, s_yh, s_yc, s_yt, s_sb, s_sh, s_sc, s_sn, s_gb, s_gh, s_gc, s_gn)
    ok = all((v == 1 for v in inner)) and all((v % 8 == 0 for v in outer)) and all((t.data_ptr() % 16 == 0 for t in tensors))
    if not ok and _env('CP_TRACE', '0') == '1':
        print('CP_TRACE kbc_pipe_refused inner=%s outer=%s align=%s' % (inner, outer, [t.data_ptr() % 16 for t in tensors]), flush=True)
    return ok

def refusal(*, device, mma_io, rot_in_kernel, rot_per_head, has_theta_h, has_rot_bias, has_beta, has_gamma, pre_rot, pair_half, sumds, dd_in_kbc, ka_nodalpha, kbc_split, rot_abl, ts_abl, T, N, GT, nsplit, trap_diag=False, dcs_ink=False):
    if not mma_io:
        return 'fp32 / tf32 operands (the Gluon body is 16-bit MMA only)'
    if rot_in_kernel or rot_per_head or has_theta_h or has_rot_bias or pair_half:
        return 'rotation'
    if pre_rot:
        return 'bias_pre_rot (the biases rotate with B/C)'
    if sumds or dd_in_kbc or ka_nodalpha or kbc_split or rot_abl or ts_abl:
        return 'lever outside the envelope'
    if dcs_ink:
        return 'DCS_INK (the in-kernel prefix is off at R = O = 1)'
    if T != 64:
        return f'T={T} (the accumulator warp grid is laid out for T=64)'
    if GT & GT - 1:
        return f'GT={GT} is not a power of two (Gluon shared-memory tiles)'
    w, wm, wn, _, _ = _cfg()
    NS = N // nsplit
    if NS % (8 * wn) or NS < 16 or GT % 16 or (GT < 16) or T % (16 * wm):
        return f'tile NS={NS} GT={GT} vs warp grid [{wm},{wn}]'
    if torch.cuda.get_device_capability(device)[0] == 9:
        return 'sm_90: needs the wgmma body (not built)'
    return None

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _segadd(v1, f1, v2, f2):
    return (gl.where(f2 != 0, v2, v1 + v2), f1 | f2)

@gluon.jit
def _kbc_issue(sY, sX, sS, sG, dY_ptr, X_ptr, SAll_ptr, dG_ptr, b, h, c, s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg, rt, cg, ct, rg, rn, T: gl.constexpr, X_GC: gl.constexpr):
    async_load(sY, dY_ptr + b * s_yb + h * s_yh + c * s_yc + rt[:, None] * s_yt + cg[None, :] * s_yg)
    if X_GC:
        async_load(sX, X_ptr + b * s_xb + h * s_xh + (c * T + rt[:, None]) * s_xl + cg[None, :] * s_xg)
    else:
        async_load(sX, X_ptr + b * s_xb + h * s_xh + (c * T + ct[None, :]) * s_xl + rg[:, None] * s_xg)
    async_load(sS, SAll_ptr + b * s_sb + h * s_sh + c * s_sc + rn[:, None] * s_sn + cg[None, :] * s_sg)
    async_load(sG, dG_ptr + b * s_gb + h * s_gh + c * s_gc + rn[:, None] * s_gn + cg[None, :] * s_gg)
    commit_group()

@gluon.jit
def _kbc_hb_gl_kernel(C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, SAll_ptr, dG_ptr, DW_ptr, dY_ptr, dCtil_ptr, dBtil_ptr, dDelta_ptr, Beta_ptr, Gamma_ptr, dBeta_ptr, dGamma_ptr, H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg, s_wb, s_wh, s_wl, s_dcb, s_dcc, s_dct, s_dcn, s_dbb, s_dbc, s_dbt, s_dbn, NGT_val, NSPLIT: gl.constexpr, T: gl.constexpr, N: gl.constexpr, GT: gl.constexpr, HB: gl.constexpr, HAS_DW: gl.constexpr, DCS_INK: gl.constexpr, TRAP_DIAG: gl.constexpr, IDX64: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr, SEG_RESET: gl.constexpr, SEM: gl.constexpr, HAS_BETA: gl.constexpr=False, HAS_GAMMA: gl.constexpr=False, PIPE: gl.constexpr=False, X_GC: gl.constexpr=False, KBC_DALPHA: gl.constexpr=False, GTB: gl.constexpr=64, dAlpha_ptr=None, s_dab=0, s_dah=0, s_dag=0, s_dac=0):
    NW: gl.constexpr = WM * WN
    NS: gl.constexpr = N // NSPLIT
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WM, WN], instr_shape=[16, 8])
    OPA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    OPB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    ROW: gl.constexpr = gl.SliceLayout(1, MMA)
    COL: gl.constexpr = gl.SliceLayout(0, MMA)
    A_ROW: gl.constexpr = gl.SliceLayout(1, OPA)
    B_COL: gl.constexpr = gl.SliceLayout(0, OPB)
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    BL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    BL_R: gl.constexpr = gl.SliceLayout(1, BL)
    BL_C: gl.constexpr = gl.SliceLayout(0, BL)
    SH: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    b = gl.program_id(0)
    hblk = gl.program_id(1)
    pid2 = gl.program_id(2)
    ns_idx = pid2 % NSPLIT
    rest = pid2 // NSPLIT
    c = rest // NGT_val
    gt_idx = rest % NGT_val
    if IDX64:
        b = b.to(gl.int64)
    rt = gl.arange(0, T, layout=BL_R)
    cn = ns_idx * NS + gl.arange(0, NS, layout=BL_C)
    sB = gl.allocate_shared_memory(gl.bfloat16, [T, NS], SH, gl.load(B_ptr + b * s_bb + c * s_bc + rt[:, None] * s_bt + cn[None, :] * s_bn).to(gl.bfloat16))
    sC = gl.allocate_shared_memory(gl.bfloat16, [T, NS], SH, gl.load(C_ptr + b * s_cb + c * s_cc + rt[:, None] * s_ct + cn[None, :] * s_cn).to(gl.bfloat16))
    if PIPE:
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
        sY = gl.allocate_shared_memory(gl.bfloat16, [T, GT], SH)
        if X_GC:
            sX = gl.allocate_shared_memory(gl.bfloat16, [T, GT], SH)
        else:
            sX = gl.allocate_shared_memory(gl.bfloat16, [GT, T], SH)
        sSt = gl.allocate_shared_memory(gl.bfloat16, [NS, GT], SH)
        if GT == T:
            sP = sY
        else:
            sP = gl.allocate_shared_memory(gl.bfloat16, [T, T], SH)
        if KBC_DALPHA:
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
    if PIPE:
        _kbc_issue(qY.index(0), qX.index(0), qS.index(0), qG.index(0), dY_ptr, X_ptr, SAll_ptr, dG_ptr, b, h_start, c, s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg, rt, cg, ct, rg, rn, T, X_GC)
    for hh in range(hb_count):
        h = h_start + hh
        if IDX64:
            h = h.to(gl.int64)
        if PIPE:
            st = hh % 2
            gl.barrier()
            if hh + 1 < hb_count:
                _kbc_issue(qY.index(1 - st), qX.index(1 - st), qS.index(1 - st), qG.index(1 - st), dY_ptr, X_ptr, SAll_ptr, dG_ptr, b, h + 1, c, s_yb, s_yh, s_yc, s_yt, s_yg, s_xb, s_xh, s_xg, s_xl, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg, rt, cg, ct, rg, rn, T, X_GC)
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
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        d = gl.load(dbase + t1 * s_dt)
        if DCS_INK:
            rst = a_val * d <= -SEG_RESET
            ri = rst.to(gl.int32)
            D1, _unused = gl.associative_scan((gl.where(rst, 0.0, d), ri), 0, _segadd)
        else:
            rst = d < 0.0
            ri = rst.to(gl.int32)
            D1 = gl.where(rst, 0.0, d)
        seg1 = gl.associative_scan(ri, 0, _add)
        seg_end = gl.max(seg1, axis=0)
        D_end = gl.sum(gl.where(t1 == T - 1, D1, 0.0), axis=0)
        f1 = gl.where(seg1 == 0, gl.exp(a_val * D1), 0.0)
        agg1 = gl.where(seg1 == seg_end, gl.exp(a_val * (D_end - D1)), 0.0)
        gl.barrier()
        if not PIPE:
            sY.store(gl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + rt[:, None] * s_yt + cg[None, :] * s_yg).to(gl.bfloat16))
            if X_GC:
                sX.store(gl.load(X_ptr + b * s_xb + h * s_xh + (c * T + rt[:, None]) * s_xl + cg[None, :] * s_xg).to(gl.bfloat16))
            else:
                sX.store(gl.load(X_ptr + b * s_xb + h * s_xh + (c * T + ct[None, :]) * s_xl + rg[:, None] * s_xg).to(gl.bfloat16))
            sSt.store(gl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + rn[:, None] * s_sn + cg[None, :] * s_sg).to(gl.bfloat16))
            gl.barrier()
        f = gl.convert_layout(f1, ROW)
        dC_state = mma_v2(sY.load(OPA), sSt.permute((1, 0)).load(OPB), gl.zeros([T, NS], gl.float32, MMA)) * f[:, None]
        n_c = ns_idx * NS + gl.arange(0, NS, layout=COL)
        Cm = sC.load(MMA).to(gl.float32)
        if HAS_GAMMA:
            ga = gl.load(Gamma_ptr + h * N + n_c).to(gl.float32)
            Cm = Cm + ga[None, :]
        dD = a_val * gl.sum(Cm * dC_state, axis=1)
        dD = gl.associative_scan(gl.convert_layout(dD, L1), 0, _add, reverse=True)
        gl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + t1 * s_dt, dD, sem=SEM)
        dC_acc = dC_acc + dC_state
        if HAS_GAMMA:
            v_gam = gl.sum(dC_state, axis=0)
        if X_GC:
            xT = sX.permute((1, 0)).load(OPB)
        else:
            xT = sX.load(OPB)
        if HAS_DW:
            dw_b = gl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + gl.arange(0, T, layout=B_COL)) * s_wl)
            xT = (xT.to(gl.float32) * dw_b[None, :]).to(gl.bfloat16)
        dScore = mma_v2(sY.load(OPA), xT, gl.zeros([T, T], gl.float32, MMA))
        D_r = gl.convert_layout(D1, ROW)
        D_c = gl.convert_layout(D1, COL)
        sg_r = gl.convert_layout(seg1, ROW)
        sg_c = gl.convert_layout(seg1, COL)
        if TRAP_DIAG:
            causal = t_m[:, None] > s_m[None, :]
        else:
            causal = t_m[:, None] >= s_m[None, :]
        dSb = (dScore * gl.exp(gl.where(causal & (sg_r[:, None] == sg_c[None, :]), a_val * (D_r[:, None] - D_c[None, :]), float('-inf')))).to(gl.bfloat16)
        gl.barrier()
        sP.store(dSb)
        if PIPE:
            sS_cur = sSt
            sSt = sGc
        elif KBC_DALPHA:
            sS_cur = sSt
            sGd.store(gl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc + rn[:, None] * s_gn + cg[None, :] * s_gg).to(gl.bfloat16))
            sSt = sGd
        else:
            sSt.store(gl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc + rn[:, None] * s_gn + cg[None, :] * s_gg).to(gl.bfloat16))
        gl.barrier()
        if KBC_DALPHA:
            col = gl.zeros([GT], gl.float32, BL_C)
            for r0 in gl.static_range(0, NS, 16):
                col += gl.sum(sS_cur.slice(r0, 16, dim=0).load(BL).to(gl.float32) * sSt.slice(r0, 16, dim=0).load(BL).to(gl.float32), axis=0)
            gcl = gl.arange(0, GT, layout=BL_C)
            for j in gl.static_range(GT // GTB):
                gl.atomic_add(dAlpha_ptr + b * s_dab + h * s_dah + (gt_idx * (GT // GTB) + j) * s_dag + c * s_dac, gl.sum(gl.where(gcl // GTB == j, col, 0.0), axis=0), sem=SEM)
        dC_acc = mma_v2(sP.load(OPA), sB.load(OPB), dC_acc)
        dB_acc = mma_v2(sP.permute((1, 0)).load(OPA), sC.load(OPB), dB_acc)
        if HAS_BETA or HAS_GAMMA:
            dSf = dSb.to(gl.float32)
            rs = gl.sum(dSf, axis=1)
            cs = gl.convert_layout(gl.sum(dSf, axis=0), ROW)
            if HAS_BETA:
                be = gl.load(Beta_ptr + h * N + n_c).to(gl.float32)
                dC_acc = dC_acc + rs[:, None] * be[None, :]
            if HAS_GAMMA:
                dB_acc = dB_acc + cs[:, None] * ga[None, :]
                Bm = sB.load(MMA).to(gl.float32)
                if HAS_BETA:
                    Bm = Bm + be[None, :]
                v_gam = v_gam + gl.sum(cs[:, None] * Bm, axis=0)
                gl.atomic_add(dGamma_ptr + h * N + n_c, v_gam, sem=SEM)
            if HAS_BETA:
                Cb = sC.load(MMA).to(gl.float32)
                if HAS_GAMMA:
                    Cb = Cb + ga[None, :]
                v_beta = gl.sum(rs[:, None] * Cb, axis=0)
        if X_GC:
            xa = sX.load(OPA).to(gl.float32)
        else:
            xa = sX.permute((1, 0)).load(OPA).to(gl.float32)
        agg = gl.convert_layout(agg1, A_ROW)
        if HAS_DW:
            dw_a = gl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + gl.arange(0, T, layout=A_ROW)) * s_wl)
            xa = xa * dw_a[:, None]
        xw = (xa * agg[:, None]).to(gl.bfloat16)
        dB_acc = mma_v2(xw, sSt.permute((1, 0)).load(OPB), dB_acc)
        if HAS_BETA:
            xws = gl.convert_layout(gl.sum(xw.to(gl.float32), axis=0), COL)
            dGm = sSt.load(MMA).to(gl.float32)
            v_beta = v_beta + gl.convert_layout(gl.sum(dGm * xws[None, :], axis=1), COL)
            gl.atomic_add(dBeta_ptr + h * N + n_c, v_beta, sem=SEM)
    n_m = ns_idx * NS + gl.arange(0, NS, layout=COL)
    sole_owner = (NGT_val == 1) & (h_start == 0) & (hb_count >= H)
    db_p = dBtil_ptr + b * s_dbb + c * s_dbc + t_m[:, None] * s_dbt + n_m[None, :] * s_dbn
    dc_p = dCtil_ptr + b * s_dcb + c * s_dcc + t_m[:, None] * s_dct + n_m[None, :] * s_dcn
    if sole_owner:
        gl.store(db_p, dB_acc)
        gl.store(dc_p, dC_acc)
    else:
        gl.atomic_add(db_p, dB_acc, sem=SEM)
        gl.atomic_add(dc_p, dC_acc, sem=SEM)
_LAST = {}

def gluon_tile(GT, NGT, gv='full'):
    gsz = GT * NGT
    if gv == 'bwd':
        gk = GT
    elif gv == 'full':
        gk = 0
        for c in (64, 32, 16):
            if c <= gsz and gsz % c == 0:
                gk = c
                break
    else:
        gk = int(gv)
    if gk > 0 and gsz % gk == 0 and (gk % 16 == 0) and (not gk & gk - 1):
        return (gk, gsz // gk)
    return (GT, NGT)

def kbc_hb_gluon(*, Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all, dw_c, dY, dCtil_out, dBtil_out, dDelta_out, H, Bsz, nc, NGT, nsplit, T, N, GT, strides, has_dw, dcs_ink, trap_diag, idx64, beta=None, gamma=None, dBeta=None, dGamma=None, dalpha=None, gt_bwd=None):
    from .fused_fwd_kernel import SEG_RESET_C
    w, wm, wn, mr, hb = _cfg()
    gsz = GT * NGT
    GT, NGT = gluon_tile(GT, NGT, _env('GLUON_KBC_GT', 'full').lower())
    x_gc = strides[10] == 1
    pipe = pipe_wanted(T=T, NS=N // nsplit, GT=GT, device=Ctil.device) and all((t.dtype == torch.bfloat16 for t in (dY, x_scaled, S_all, dG_all))) and _cp_async_ok(strides, (dY, x_scaled, S_all, dG_all))
    grid = (Bsz, (H + hb - 1) // hb, nc * NGT * nsplit)
    _kbc_hb_gl_kernel[grid](Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all, dw_c, dY, dCtil_out, dBtil_out, dDelta_out, beta if beta is not None else a_log, gamma if gamma is not None else a_log, dBeta if dBeta is not None else dDelta_out, dGamma if dGamma is not None else dDelta_out, H, *strides, NGT, NSPLIT=nsplit, T=T, N=N, GT=GT, HB=hb, HAS_DW=has_dw, DCS_INK=dcs_ink, TRAP_DIAG=trap_diag, IDX64=idx64, WM=wm, WN=wn, SEG_RESET=float(SEG_RESET_C.value), SEM=ACC_SEM.value, HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None, PIPE=pipe, X_GC=x_gc, **dict(KBC_DALPHA=True, GTB=gt_bwd, dAlpha_ptr=dalpha, s_dab=dalpha.stride(0), s_dah=dalpha.stride(1), s_dag=dalpha.stride(2), s_dac=dalpha.stride(3)) if dalpha is not None else {}, num_warps=w, num_stages=1, **{'maxnreg': mr} if mr else {})
    _LAST.update(route='gluon_pipe' if pipe else 'gluon', warps=w, wm=wm, wn=wn, maxnreg=mr, hb=hb)
