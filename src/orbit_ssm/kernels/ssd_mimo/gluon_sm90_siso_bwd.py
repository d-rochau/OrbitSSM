
from __future__ import annotations
import os
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
try:
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
except Exception:
    TensorDescriptor = None

@gluon.jit
def _add_combine(a, b):
    return a + b

@gluon.jit
def _wrap2pi_g(x):
    k = gl.floor(x * 0.15915494309189535 + 0.5)
    return x - k * 6.28125 - k * 0.0019353071795864768

@gluon.jit
def _cos_ptx(x):
    return gl.inline_asm_elementwise('cos.approx.f32 $0, $1;', '=f,f', [_wrap2pi_g(x)], dtype=gl.float32, is_pure=True, pack=1)

@gluon.jit
def _sin_ptx(x):
    return gl.inline_asm_elementwise('sin.approx.f32 $0, $1;', '=f,f', [_wrap2pi_g(x)], dtype=gl.float32, is_pure=True, pack=1)

@gluon.jit
def _rot_w(V, c, s, M: gl.constexpr, N: gl.constexpr):
    e, o = gl.split(gl.reshape(V, [M, N // 2, 2]))
    return gl.convert_layout(gl.reshape(gl.join(c * e + s * o, c * o - s * e), [M, N]), V.type.layout)

@gluon.jit
def _pair_cross(g, X, M: gl.constexpr, N: gl.constexpr):
    ge, go = gl.split(gl.reshape(g, [M, N // 2, 2]))
    xe, xo = gl.split(gl.reshape(X, [M, N // 2, 2]))
    return ge * xo - go * xe

@gluon.jit
def _adj_half(braws, craws, bsl, csl, vecs, Theta_ptr, dTheta_ptr, W_ptr, Gam_ptr, dW_ptr, dG_ptr, beta_v, gam_v, bo, ho, c, s_thb, s_thc, s_tht, s_thh, s_thk, s_pb, s_ph, s_pl, s_wb, s_wh, s_wl, R0: gl.constexpr, T: gl.constexpr, TH: gl.constexpr, N: gl.constexpr, BLK_E: gl.constexpr, HAS_BETA: gl.constexpr, HAS_GAMMA: gl.constexpr, HAS_TRAP: gl.constexpr):
    KP: gl.constexpr = N // 2
    ze, zo = gl.split(gl.reshape(gl.zeros([TH, N], gl.float32, BLK_E), [TH, KP, 2]))
    L_WK: gl.constexpr = ze.type.layout
    t_e = gl.arange(0, TH, layout=gl.SliceLayout(1, BLK_E))
    t_k = gl.arange(0, TH, layout=gl.SliceLayout(1, L_WK))
    k_k = gl.arange(0, KP, layout=gl.SliceLayout(0, L_WK))
    tt = c * T + R0 + t_e
    uB = braws.slice(R0, TH).load(BLK_E).to(gl.float32)
    uC = craws.slice(R0, TH).load(BLK_E).to(gl.float32)
    if HAS_BETA:
        uB = uB + beta_v[None, :]
    if HAS_GAMMA:
        uC = uC + gam_v[None, :]
    tho = bo * s_thb + c * s_thc + (R0 + t_k)[:, None] * s_tht + ho * s_thh + k_k[None, :] * s_thk
    th = gl.load(Theta_ptr + tho).to(gl.float32)
    cw = _cos_ptx(th)
    sw = _sin_ptx(th)
    Bt = _rot_w(uB, cw, sw, TH, N)
    Ct = _rot_w(uC, cw, sw, TH, N)
    gB = bsl.slice(R0, TH).load(BLK_E).to(gl.float32)
    gC = csl.slice(R0, TH).load(BLK_E).to(gl.float32)
    pw = bo * s_pb + ho * s_ph + tt * s_pl
    wo = bo * s_wb + ho * s_wh + tt * s_wl
    gl.store(dW_ptr + pw, gl.sum(gB * Bt, axis=1))
    gB = gB * gl.load(W_ptr + wo)[:, None]
    if HAS_TRAP:
        dqk = vecs.index(4).slice(R0, TH).load(gl.SliceLayout(1, BLK_E))
        gl.store(dG_ptr + pw, dqk * gl.sum(Bt * Ct, axis=1))
        ddot = dqk * gl.load(Gam_ptr + wo)
        gB = gB + ddot[:, None] * Ct
        gC = gC + ddot[:, None] * Bt
    dBr = _rot_w(gB, cw, -sw, TH, N)
    dCr = _rot_w(gC, cw, -sw, TH, N)
    gl.store(dTheta_ptr + tho, _pair_cross(dBr, uB, TH, N) + _pair_cross(dCr, uC, TH, N))
    bsl.slice(R0, TH).store(dBr.to(gl.bfloat16))
    csl.slice(R0, TH).store(dCr.to(gl.bfloat16))
    return (gl.sum(dBr, axis=0), gl.sum(dCr, axis=0))

@gluon.jit
def _gl90_siso_bwd_kernel(bs_desc, cs_desc, dbs_desc, dcs_desc, x_desc, dx_desc, s_desc, dY_ptr, Delta_ptr, A_ptr, DSF_ptr, Dsk_ptr, QK_ptr, dDelta_ptr, dQK_ptr, dX0_ptr, H, s_gb, s_gh, s_gc, s_gt, s_gg, s_db, s_dh, s_dc, s_dt, s_wb, s_wh, s_wl, s_fb, s_fh, s_fg, s_fn, NC, braw_desc, craw_desc, Theta_ptr, dTheta_ptr, W_ptr, Gam_ptr, Beta_ptr, Gamma_ptr, dW_ptr, dG_ptr, dBeta_ptr, dGamma_ptr, s_thb, s_thc, s_tht, s_thh, s_thk, s_pb, s_ph, s_pl, Dth_ptr, s_hb, s_hc, s_ht, s_hh, s_hk, T: gl.constexpr, N: gl.constexpr, G: gl.constexpr, HAS_TRAP: gl.constexpr, HAS_DSKIP: gl.constexpr, HAS_DSF: gl.constexpr, HAS_X0: gl.constexpr, IDX64: gl.constexpr, FUSE_ADJ: gl.constexpr=False, HAS_BETA: gl.constexpr=False, HAS_GAMMA: gl.constexpr=False, X_PACKED: gl.constexpr=False, DY_TMAJOR: gl.constexpr=False, NARROW_DTH: gl.constexpr=True, SPLIT2: gl.constexpr=False, WN: gl.constexpr=2, STAGES: gl.constexpr=2, SYNC: gl.constexpr=False, PRICE: gl.constexpr=0, SPAN_MMA: gl.constexpr=0, DTH_ROT: gl.constexpr=0, DTH_STORE_VEC: gl.constexpr=4, ROT_KA: gl.constexpr=0, NO_DD: gl.constexpr=False):
    NW: gl.constexpr = 4 * WN
    gl.static_assert(gl.num_warps() == NW, 'WN warpgroups, N-split')
    gl.static_assert(T == 64, 'one wgmma M tile per chunk (T = 64)')
    P_SEG: gl.constexpr = PRICE & 1 == 0
    P_SPAN: gl.constexpr = PRICE & 2 == 0
    DD: gl.constexpr = not NO_DD
    P_TRAPF: gl.constexpr = HAS_TRAP and PRICE & 4 == 0
    P_DSK: gl.constexpr = HAS_DSKIP and PRICE & 8 == 0
    MMA_TT: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, T // WN, 16])
    MMA_TN: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, N // WN, 16])
    MMA_TG: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, G // WN, 16])
    V1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    if DY_TMAJOR:
        BLK_Y: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, NW], [0, 1])
    else:
        BLK_Y: gl.constexpr = gl.BlockedLayout([1, 4], [32 // (G // 4), G // 4], [NW, 1], [1, 0])
    if WN == 2:
        BLK_SC: gl.constexpr = gl.BlockedLayout([8, 2], [4, 8], [2, 4], [0, 1])
    else:
        BLK_SC: gl.constexpr = gl.BlockedLayout([8, 4], [4, 8], [2, 2], [0, 1])
    LAY_BN: gl.constexpr = bs_desc.layout
    LAY_X: gl.constexpr = x_desc.layout
    LAY_S: gl.constexpr = s_desc.layout
    LAY_TG: gl.constexpr = gl.NVMMASharedLayout.get_default_for([T, G], gl.bfloat16)
    LAY_TT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([T, T], gl.bfloat16)
    FLAT1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    TH: gl.constexpr = 32
    KP: gl.constexpr = N // 2
    b = gl.program_id(0)
    h = gl.program_id(1)
    if IDX64:
        bo = b.to(gl.int64)
        ho = h.to(gl.int64)
    else:
        bo = b
        ho = h
    a_val = -gl.exp(gl.load(A_ptr + h))
    bs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, T, N], LAY_BN)
    cs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, T, N], LAY_BN)
    if X_PACKED:
        xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, T, G], LAY_X)
    else:
        xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, G, T], LAY_X)
    ss = gl.allocate_shared_memory(gl.bfloat16, [STAGES, N, G], LAY_S)
    dys = gl.allocate_shared_memory(gl.bfloat16, [T, G], LAY_TG)
    dyfs = gl.allocate_shared_memory(gl.bfloat16, [T, G], LAY_TG)
    dss = gl.allocate_shared_memory(gl.bfloat16, [N, G], LAY_S)
    mms = gl.allocate_shared_memory(gl.bfloat16, [T, T], LAY_TT)
    scs = gl.allocate_shared_memory(gl.bfloat16, [T, T], LAY_TT)
    if X_PACKED:
        dxs = gl.allocate_shared_memory(gl.bfloat16, [T, G], LAY_X)
    else:
        dxs = gl.allocate_shared_memory(gl.bfloat16, [G, T], LAY_X)
    if SPAN_MMA > 0 and DD:
        lges = gl.allocate_shared_memory(gl.bfloat16, [T, T], LAY_TT)
        pps = gl.allocate_shared_memory(gl.bfloat16, [SPAN_MMA, T, T], LAY_TT)
    vecs = gl.allocate_shared_memory(gl.float32, [5, T], FLAT1)
    if FUSE_ADJ:
        braws = gl.allocate_shared_memory(gl.bfloat16, [T, N], LAY_BN)
        craws = gl.allocate_shared_memory(gl.bfloat16, [T, N], LAY_BN)
        BLK_E: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [NW, 1], [1, 0])
        n_e = gl.arange(0, N, layout=gl.SliceLayout(0, BLK_E))
        if HAS_BETA:
            beta_v = gl.load(Beta_ptr + h * N + n_e)
            dbeta_acc = gl.zeros([N], gl.float32, gl.SliceLayout(0, BLK_E))
        if HAS_GAMMA:
            gam_v = gl.load(Gamma_ptr + h * N + n_e)
            dgam_acc = gl.zeros([N], gl.float32, gl.SliceLayout(0, BLK_E))
    bars = gl.allocate_shared_memory(gl.int64, [STAGES + 1, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES + 1):
        mbarrier.init(bars.index(i), count=1)
    fence_async_shared()
    CBYTES: gl.constexpr = 2 * T * N * 2 + G * T * 2 + N * G * 2
    L_tot = NC * T
    xrow = (b * H + h) * G
    srow = (b * H + h) * NC
    for st in gl.static_range(STAGES - 1):
        c0 = gl.maximum(NC - 1 - st, 0)
        pv = st < NC
        mbarrier.expect(bars.index(st), CBYTES, pred=pv)
        tma.async_load(bs_desc, [b * L_tot + c0 * T, h * N], bars.index(st), bs.index(st), pred=pv)
        tma.async_load(cs_desc, [b * L_tot + c0 * T, h * N], bars.index(st), cs.index(st), pred=pv)
        if X_PACKED:
            tma.async_load(x_desc, [b * L_tot + c0 * T, h * G], bars.index(st), xs.index(st), pred=pv)
        else:
            tma.async_load(x_desc, [xrow, c0 * T], bars.index(st), xs.index(st), pred=pv)
        tma.async_load(s_desc, [(srow + c0) * N, 0], bars.index(st), ss.index(st), pred=pv)
    offs_t = gl.arange(0, T, layout=V1)
    i_tt = gl.arange(0, T, layout=gl.SliceLayout(1, MMA_TT))
    j_tt = gl.arange(0, T, layout=gl.SliceLayout(0, MMA_TT))
    t_y = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_Y))
    g_y = gl.arange(0, G, layout=gl.SliceLayout(0, BLK_Y))
    t_g = gl.arange(0, T, layout=gl.SliceLayout(1, MMA_TG))
    g_g = gl.arange(0, G, layout=gl.SliceLayout(0, MMA_TG))
    n_g = gl.arange(0, N, layout=gl.SliceLayout(1, MMA_TG))
    s_sc = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_SC))
    k_sc = gl.arange(0, T, layout=gl.SliceLayout(0, BLK_SC))
    below = k_sc[None, :] < s_sc[:, None]
    if SPAN_MMA > 0 and DD:
        lges.store(gl.where(j_tt[None, :] >= i_tt[:, None], 1.0, 0.0).to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
    if HAS_DSF:
        dS = gl.load(DSF_ptr + bo * s_fb + ho * s_fh + g_g[None, :] * s_fg + n_g[:, None] * s_fn).to(gl.float32)
    else:
        dS = gl.zeros([N, G], gl.float32, MMA_TG)
    if P_DSK:
        dv = gl.load(Dsk_ptr + h * G + g_g)
    for cr in range(NC):
        c = NC - 1 - cr
        slot = cr % STAGES
        phase = cr // STAGES & 1
        tma.store_wait(0)
        gl.barrier()
        more = cr + STAGES - 1 < NC
        nsl = (cr + STAGES - 1) % STAGES
        nb = bars.index(nsl)
        cn = gl.maximum(c - (STAGES - 1), 0)
        mbarrier.expect(nb, CBYTES, pred=more)
        tma.async_load(bs_desc, [b * L_tot + cn * T, h * N], nb, bs.index(nsl), pred=more)
        tma.async_load(cs_desc, [b * L_tot + cn * T, h * N], nb, cs.index(nsl), pred=more)
        if X_PACKED:
            tma.async_load(x_desc, [b * L_tot + cn * T, h * G], nb, xs.index(nsl), pred=more)
        else:
            tma.async_load(x_desc, [xrow, cn * T], nb, xs.index(nsl), pred=more)
        tma.async_load(s_desc, [(srow + cn) * N, 0], nb, ss.index(nsl), pred=more)
        if FUSE_ADJ:
            rb = bars.index(STAGES)
            mbarrier.expect(rb, 2 * T * N * 2)
            tma.async_load(braw_desc, [b * L_tot + c * T, 0], rb, braws)
            tma.async_load(craw_desc, [b * L_tot + c * T, 0], rb, craws)
        d_raw = gl.load(Delta_ptr + bo * s_db + ho * s_dh + c * s_dc + offs_t * s_dt)
        if P_SEG:
            rst = d_raw < 0.0
            ri = gl.where(rst, 1.0, 0.0)
            D_h = gl.where(rst, 0.0, d_raw)
            seg_h = gl.associative_scan(ri, 0, _add_combine)
            seg_end = gl.max(seg_h, axis=0)
            D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
            alpha = gl.where(seg_end == 0.0, gl.exp(a_val * D_end), 0.0)
            f_v = gl.where(seg_h == 0.0, gl.exp(a_val * D_h), 0.0)
            dagg_v = gl.where(seg_h == seg_end, gl.exp(a_val * (D_end - D_h)), 0.0)
        else:
            D_h = d_raw
            seg_h = gl.zeros([T], gl.float32, V1)
            D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
            alpha = gl.exp(a_val * D_end)
            f_v = gl.exp(a_val * D_h)
            dagg_v = gl.exp(a_val * (D_end - D_h))
        vecs.index(0).store(D_h)
        vecs.index(1).store(seg_h)
        vecs.index(2).store(f_v)
        vecs.index(3).store(dagg_v)
        gl.barrier()
        dy_b = gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_y[:, None] * s_gt + g_y[None, :] * s_gg).to(gl.float32)
        f_y = vecs.index(2).load(gl.SliceLayout(1, BLK_Y))
        dys.store(dy_b.to(gl.bfloat16))
        dyfs.store((dy_b * f_y[:, None]).to(gl.bfloat16))
        dss.store(dS.to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
        mbarrier.wait(bars.index(slot), phase)
        bsl = bs.index(slot)
        csl = cs.index(slot)
        if X_PACKED:
            xsl = xs.index(slot).permute((1, 0))
        else:
            xsl = xs.index(slot)
        ssl = ss.index(slot)
        dSc = warpgroup_mma(dys, xsl, gl.zeros([T, T], gl.float32, MMA_TT), use_acc=False, is_async=True)
        if SYNC:
            dSc = warpgroup_mma_wait(0, deps=[dSc])
        SB = warpgroup_mma(csl, bsl.permute((1, 0)), gl.zeros([T, T], gl.float32, MMA_TT), use_acc=False, is_async=True)
        if SYNC:
            SB = warpgroup_mma_wait(0, deps=[SB])
        dYS = warpgroup_mma(dys, ssl.permute((1, 0)), gl.zeros([T, N], gl.float32, MMA_TN), use_acc=False, is_async=True)
        if SYNC:
            dYS = warpgroup_mma_wait(0, deps=[dYS])
        xdS = warpgroup_mma(xsl.permute((1, 0)), dss.permute((1, 0)), gl.zeros([T, N], gl.float32, MMA_TN), use_acc=False, is_async=True)
        if SYNC:
            xdS = warpgroup_mma_wait(0, deps=[xdS])
            BdS = warpgroup_mma(bsl, dss, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            BdS = warpgroup_mma_wait(0, deps=[BdS])
            QS = warpgroup_mma(csl, ssl, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            QS = warpgroup_mma_wait(0, deps=[QS])
        elif SPLIT2:
            dSc, SB = warpgroup_mma_wait(2, deps=[dSc, SB])
        else:
            BdS = warpgroup_mma(bsl, dss, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            QS = warpgroup_mma(csl, ssl, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            dSc, SB = warpgroup_mma_wait(4, deps=[dSc, SB])
        D_i = vecs.index(0).load(gl.SliceLayout(1, MMA_TT))
        D_j = vecs.index(0).load(gl.SliceLayout(0, MMA_TT))
        sg_i = vecs.index(1).load(gl.SliceLayout(1, MMA_TT))
        sg_j = vecs.index(1).load(gl.SliceLayout(0, MMA_TT))
        if P_TRAPF:
            cm = i_tt[:, None] > j_tt[None, :]
        else:
            cm = i_tt[:, None] >= j_tt[None, :]
        if P_SEG:
            cm = cm & (sg_i[:, None] == sg_j[None, :])
        dec = gl.where(cm, gl.exp(a_val * (D_i[:, None] - D_j[None, :])), 0.0)
        Mm = dSc * dec
        mms.store(Mm.to(gl.bfloat16))
        scs.store((SB * dec).to(gl.bfloat16))
        if not DD:
            pass
        elif P_SPAN and SPAN_MMA > 0:
            Pf = Mm * SB
            p1 = Pf.to(gl.bfloat16)
            r1 = Pf - p1.to(gl.float32)
            pps.index(0).store(p1)
            if SPAN_MMA >= 2:
                p2 = r1.to(gl.bfloat16)
                pps.index(1).store(p2)
            if SPAN_MMA >= 3:
                pps.index(2).store((r1 - p2.to(gl.float32)).to(gl.bfloat16))
            fence_async_shared()
            gl.barrier()
            Wsp = gl.zeros([T, T], gl.float32, MMA_TT)
            for kk in gl.static_range(SPAN_MMA):
                Wsp = warpgroup_mma(lges, pps.index(SPAN_MMA - 1 - kk), Wsp, is_async=True)
        elif P_SPAN:
            P = gl.convert_layout(Mm * SB, BLK_SC)
            Rrev = gl.associative_scan(P, 0, _add_combine, reverse=True)
            E_intra = gl.sum(gl.where(below, Rrev, 0.0), axis=1)
        else:
            PT = Mm * SB
            rq = gl.convert_layout(gl.sum(PT, axis=1), V1) - gl.convert_layout(gl.sum(PT, axis=0), V1)
            E_intra = gl.associative_scan(rq, 0, _add_combine, reverse=True)
        if SPLIT2 and (not SYNC):
            BdS = warpgroup_mma(bsl, dss, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
            QS = warpgroup_mma(csl, ssl, gl.zeros([T, G], gl.float32, MMA_TG), use_acc=False, is_async=True)
        if not SYNC:
            dYS, xdS, BdS, QS = warpgroup_mma_wait(0, deps=[dYS, xdS, BdS, QS])
        if DD and P_SPAN and (SPAN_MMA > 0):
            Wsp = warpgroup_mma_wait(0, deps=[Wsp])
            E_intra = gl.sum(gl.where(j_tt[None, :] < i_tt[:, None], Wsp, 0.0), axis=1)
        f_i = vecs.index(2).load(gl.SliceLayout(1, MMA_TN))
        dagg_n = vecs.index(3).load(gl.SliceLayout(1, MMA_TN))
        dagg_g = vecs.index(3).load(gl.SliceLayout(1, MMA_TG))
        f_g = vecs.index(2).load(gl.SliceLayout(1, MMA_TG))
        if DY_TMAJOR:
            dy = gl.convert_layout(gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_y[:, None] * s_gt + g_y[None, :] * s_gg).to(gl.float32), MMA_TG)
        else:
            dy = gl.load(dY_ptr + bo * s_gb + ho * s_gh + c * s_gc + t_g[:, None] * s_gt + g_g[None, :] * s_gg).to(gl.float32)
        xt = xsl.permute((1, 0)).load(MMA_TG).to(gl.float32)
        if DD:
            t1 = f_g * gl.sum(QS * dy, axis=1)
            vj = dagg_g * gl.sum(BdS * xt, axis=1)
            tail = alpha * gl.sum(gl.sum(ssl.load(MMA_TG).to(gl.float32) * dS, axis=1), axis=0)
        tbase = bo * s_wb + ho * s_wh + (c * T + t_g) * s_wl
        if P_TRAPF:
            dqk_t = gl.sum(dy * xt, axis=1)
            gl.store(dQK_ptr + tbase, dqk_t)
            if FUSE_ADJ:
                vecs.index(4).store(gl.convert_layout(dqk_t, V1))
        acc_c = dYS * f_i[:, None]
        acc_b = xdS * dagg_n[:, None]
        acc_x = BdS * dagg_g[:, None]
        acc_s = dS * alpha
        fence_async_shared()
        gl.barrier()
        dCt = warpgroup_mma(mms, bsl, acc_c, is_async=True)
        if SYNC:
            dCt = warpgroup_mma_wait(0, deps=[dCt])
        dBt = warpgroup_mma(mms.permute((1, 0)), csl, acc_b, is_async=True)
        if SYNC:
            dBt = warpgroup_mma_wait(0, deps=[dBt])
        dx = warpgroup_mma(scs.permute((1, 0)), dys, acc_x, is_async=True)
        if SYNC:
            dx = warpgroup_mma_wait(0, deps=[dx])
        dSn = warpgroup_mma(csl.permute((1, 0)), dyfs, acc_s, is_async=True)
        if SYNC:
            dSn = warpgroup_mma_wait(0, deps=[dSn])
        if DD:
            t1_s = gl.convert_layout(t1, V1)
            vj_s = gl.convert_layout(vj, gl.SliceLayout(0, BLK_SC))
            if P_SPAN:
                vx = gl.convert_layout(gl.sum(gl.where(below, vj_s[None, :], 0.0), axis=1), V1)
            else:
                vv = gl.convert_layout(vj, V1)
                vx = gl.associative_scan(vv, 0, _add_combine) - vv
            E = gl.convert_layout(E_intra, V1) + gl.associative_scan(t1_s, 0, _add_combine, reverse=True) + vx + tail
            gl.store(dDelta_ptr + bo * s_db + ho * s_dh + c * s_dc + offs_t * s_dt, a_val * E)
        if not SYNC:
            dCt, dBt, dx, dSn = warpgroup_mma_wait(0, deps=[dCt, dBt, dx, dSn])
        if DTH_ROT > 0:
            if ROT_KA == N // 4 and NARROW_DTH:
                BLK_R: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
                gcr = gl.convert_layout(dCt, BLK_R)
                gbr = gl.convert_layout(dBt, BLK_R)
                Cre = csl.load(BLK_R).to(gl.float32)
                Bre = bsl.load(BLK_R).to(gl.float32)
                gc, _ = gl.split(gl.permute(gl.reshape(gcr, [T, 2, N // 2]), (0, 2, 1)))
                gb, _ = gl.split(gl.permute(gl.reshape(gbr, [T, 2, N // 2]), (0, 2, 1)))
                xc, _ = gl.split(gl.permute(gl.reshape(Cre, [T, 2, N // 2]), (0, 2, 1)))
                xb, _ = gl.split(gl.permute(gl.reshape(Bre, [T, 2, N // 2]), (0, 2, 1)))
                active = _pair_cross(gc, xc, T, N // 2) + _pair_cross(gb, xb, T, N // 2)
                zero = gl.full([T, N // 4], 0, gl.float32, active.type.layout)
                dth = gl.reshape(gl.permute(gl.join(active, zero), (0, 2, 1)), [T, N // 2])
            else:
                Cre = csl.load(MMA_TN).to(gl.float32)
                Bre = bsl.load(MMA_TN).to(gl.float32)
                dth = _pair_cross(dCt, Cre, T, N) + _pair_cross(dBt, Bre, T, N)
            if DTH_ROT == 3:
                gl.store(Dth_ptr + bo * s_hb + c * s_hc + ho * s_hh, gl.sum(gl.sum(dth, axis=1), axis=0))
            else:
                BLK_D: gl.constexpr = gl.BlockedLayout([1, DTH_STORE_VEC], [32 // (KP // DTH_STORE_VEC), KP // DTH_STORE_VEC], [NW, 1], [1, 0])
                dthv = gl.convert_layout(dth, BLK_D)
                L_DT: gl.constexpr = dthv.type.layout
                t_d = gl.arange(0, T, layout=gl.SliceLayout(1, L_DT))
                k_d = gl.arange(0, KP, layout=gl.SliceLayout(0, L_DT))
                if ROT_KA < N // 2 and (not NARROW_DTH):
                    dthv = gl.where(k_d[None, :] < ROT_KA, dthv, 0.0)
                gl.store(Dth_ptr + bo * s_hb + c * s_hc + ho * s_hh + t_d[:, None] * s_ht + k_d[None, :] * s_hk, dthv)
            gl.barrier()
        if P_TRAPF:
            qk = gl.load(QK_ptr + tbase)
            dx = dx + qk[:, None] * dy
        if P_DSK:
            dx = dx + dy * dv[None, :]
        if X_PACKED:
            dxs.store(dx.to(gl.bfloat16))
        else:
            dxs.permute((1, 0)).store(dx.to(gl.bfloat16))
        csl.store(dCt.to(gl.bfloat16))
        bsl.store(dBt.to(gl.bfloat16))
        if FUSE_ADJ:
            gl.barrier()
            mbarrier.wait(bars.index(STAGES), cr & 1)
            dBr0, dCr0 = _adj_half(braws, craws, bsl, csl, vecs, Theta_ptr, dTheta_ptr, W_ptr, Gam_ptr, dW_ptr, dG_ptr, beta_v if HAS_BETA else 0.0, gam_v if HAS_GAMMA else 0.0, bo, ho, c, s_thb, s_thc, s_tht, s_thh, s_thk, s_pb, s_ph, s_pl, s_wb, s_wh, s_wl, 0, T, TH, N, BLK_E, HAS_BETA, HAS_GAMMA, HAS_TRAP)
            dBr1, dCr1 = _adj_half(braws, craws, bsl, csl, vecs, Theta_ptr, dTheta_ptr, W_ptr, Gam_ptr, dW_ptr, dG_ptr, beta_v if HAS_BETA else 0.0, gam_v if HAS_GAMMA else 0.0, bo, ho, c, s_thb, s_thc, s_tht, s_thh, s_thk, s_pb, s_ph, s_pl, s_wb, s_wh, s_wl, TH, T, TH, N, BLK_E, HAS_BETA, HAS_GAMMA, HAS_TRAP)
            if HAS_BETA:
                dbeta_acc = dbeta_acc + dBr0 + dBr1
            if HAS_GAMMA:
                dgam_acc = dgam_acc + dCr0 + dCr1
        fence_async_shared()
        gl.barrier()
        if X_PACKED:
            tma.async_store(dx_desc, [b * L_tot + c * T, h * G], dxs)
        else:
            tma.async_store(dx_desc, [xrow, c * T], dxs)
        tma.async_store(dcs_desc, [b * L_tot + c * T, h * N], csl)
        tma.async_store(dbs_desc, [b * L_tot + c * T, h * N], bsl)
        dS = dSn
    tma.store_wait(0)
    for i in gl.static_range(STAGES + 1):
        mbarrier.invalidate(bars.index(i))
    if FUSE_ADJ:
        if HAS_BETA:
            gl.atomic_add(dBeta_ptr + h * N + n_e, dbeta_acc, sem='relaxed')
        if HAS_GAMMA:
            gl.atomic_add(dGamma_ptr + h * N + n_e, dgam_acc, sem='relaxed')
    if HAS_X0:
        gl.store(dX0_ptr + bo * s_fb + ho * s_fh + g_g[None, :] * s_fg + n_g[:, None] * s_fn, dS)
_LAST_ROUTE: dict = {'used': False, 'why': 'never consulted'}
from . import levers as _lv
_GRP = 'complex_R1O1'

def last_route() -> dict:
    return dict(_LAST_ROUTE)

def mode() -> str:
    m = os.environ.get('SSD_MIMO_GLUON90_SISO', 'auto').strip().lower()
    if m in ('1', 'on', 'true'):
        return '1'
    if m in ('0', 'off', 'false'):
        return '0'
    return 'auto'
_DEFAULT_OK: dict = {}

def default_serves(device) -> bool:
    if device is None or getattr(device, 'type', 'cuda') != 'cuda' or (not torch.cuda.is_available()):
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _DEFAULT_OK:
        ok = TensorDescriptor is not None and torch.cuda.get_device_capability(idx) == (9, 0)
        if ok:
            try:
                ok = ptxas_ok() is None
            except Exception:
                ok = False
        _DEFAULT_OK[idx] = ok
    return _DEFAULT_OK[idx]

def enabled(device=None) -> bool:
    m = mode()
    if m == '1':
        return True
    if m == '0':
        return False
    return default_serves(device)

def ptxas_ok() -> str | None:
    from .gluon_sm90_fwd import ptxas_refusal
    return ptxas_refusal()

def packed_x(t):
    if t.dim() != 5 or t.shape[3] != 1:
        return False
    b, h, g, _, length = t.shape
    ok = (t.stride(0), t.stride(1), t.stride(2), t.stride(4)) == (length * h * g, g, 1, h * g)
    from . import levers as _lv
    return ok and _lv.on('gl_packed_x', _GRP, guard=ok)

def refusal(*, Bs, Cs, dBs, dCs, x, dX, S_all, dY, T, N, G, io_tag, m_st, dskip_dd, xtm, abl, dS_final, x0) -> str | None:
    if TensorDescriptor is None:
        return 'no gluon hopper TensorDescriptor in this Triton'
    dev = x.device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        return 'not sm_90'
    why = ptxas_ok()
    if why:
        return why
    if io_tag != 'bf16':
        return f'needs bf16 IO (got {io_tag})'
    if T != 64:
        return f'chunk T={T}: the Gluon SISO body is T = 64 (chunk_size=64 or SSD_MIMO_CHUNK=64)'
    if N not in (64, 128):
        return f'N={N} not in (64, 128)'
    if G not in (32, 64):
        return f'G={G} not in (32, 64)'
    if dskip_dd:
        return 'the in-kernel D-skip reduction (DSKIP_DD) is not served'
    wn, stages, _, _ = levers()
    ring = stages * (2 * T * N * 2 + G * T * 2 + N * G * 2)
    fixed = 2 * T * G * 2 + N * G * 2 + 2 * T * T * 2 + G * T * 2 + 4096
    if adj_enabled():
        fixed += 2 * T * N * 2
    sm = span_mma()
    fixed += (1 + sm) * T * T * 2 if sm else 0
    if ring + fixed > 232448:
        return f'STAGES={stages}: {ring + fixed} B of shared memory > 227 KB at this (N, G)'
    if abl:
        return 'attribution ablation (SSD_MIMO_SISO_BWD_ABL) not served'
    if xtm != '0':
        return 'token-major x (SSD_MIMO_SISO_XTM) not served; family layout only'
    for nm, t in (('Bs', Bs), ('Cs', Cs), ('dBs', dBs), ('dCs', dCs), ('x', x), ('dX', dX), ('S_all', S_all)):
        if t.dtype != torch.bfloat16:
            return f'{nm} must be bf16 (got {t.dtype})'
        if t.data_ptr() % 16:
            return f'{nm}: TMA base not 16-byte aligned'
    Bsz, nc, T_, H, N_ = Bs.shape
    want = [nc * T_ * H * N_, T_ * H * N_, H * N_, N_, 1]
    if list(m_st) != want or list(Bs.stride()) != want or list(Cs.stride()) != want or (list(dBs.stride()) != want) or (list(dCs.stride()) != want):
        return 'maps must be contiguous [B, nc, T, H, N] (token-major rows)'
    Bx, Hx, Gx, Rx, Lx = x.shape
    if not packed_x(x) and (Rx != 1 or x.stride() != (Hx * Gx * Lx, Gx * Lx, Lx, Lx, 1)) or dX.shape != x.shape or dX.stride() != x.stride():
        return 'x / dX must be the contiguous family layout [B, H, G, 1, L]'
    if S_all.shape != (Bsz, H, nc, N, G) or not S_all.is_contiguous():
        return 'S_all must be contiguous [B, H, nc, N, G]'
    if dY.dtype not in (torch.float32, torch.bfloat16):
        return f'dY dtype {dY.dtype} not served'
    if dS_final is not None and dS_final.dtype != torch.float32:
        return 'dS_final must be fp32'
    if x0 is not None:
        pass
    return None

def levers():
    from . import levers as _lv
    _c = lambda n, d: (lambda v: d if v is None else v)(_lv.cfg_value(n, _GRP))
    wn = int(os.environ.get('SSD_MIMO_GLUON90_SISO_WN', str(_c('wn', 2))))
    stages = int(os.environ.get('SSD_MIMO_GLUON90_SISO_STAGES', str(_c('stages', 2))))
    sync = os.environ.get('SSD_MIMO_GLUON90_SISO_SYNC', '0') == '1'
    price = int(os.environ.get('SSD_MIMO_GLUON90_SISO_PRICE', '0'), 0)
    if wn not in (1, 2):
        raise ValueError(f'SSD_MIMO_GLUON90_SISO_WN={wn}: 1 or 2')
    if stages not in (2, 3):
        raise ValueError(f'SSD_MIMO_GLUON90_SISO_STAGES={stages}: 2 or 3')
    if price and os.environ.get('SSD_MIMO_BENCH_PRICING', '0') != '1':
        raise RuntimeError('SSD_MIMO_GLUON90_SISO_PRICE is PRICING ONLY (wrong gradients); it runs only under SSD_MIMO_BENCH_PRICING=1')
    return (wn, stages, sync, price)

def dth_mode() -> int:
    m = int(os.environ.get('SSD_MIMO_GLUON90_SISO_DTH_MODE', '1'))
    if m not in (1, 3):
        raise ValueError(f'SSD_MIMO_GLUON90_SISO_DTH_MODE={m}: 1 or 3')
    if m == 3 and os.environ.get('SSD_MIMO_BENCH_PRICING', '0') != '1':
        raise RuntimeError('SSD_MIMO_GLUON90_SISO_DTH_MODE=3 is PRICING ONLY (wrong gradients); needs SSD_MIMO_BENCH_PRICING=1')
    return m

def span_mma() -> int:
    return int(os.environ.get('SSD_MIMO_SPANDELTA_MMA', '2'))

def adj_enabled() -> bool:
    return os.environ.get('SSD_MIMO_GLUON90_SISO_ADJ', '0').lower() in ('1', 'on', 'true')

def adj_refusal(*, theta, theta_h, rot_bias, bias_pre_rot, beta, gamma, dw, raw_common, need_dtheta, Btil, Ctil, aux, qk_f, has_trap, N) -> str | None:
    if theta is None or theta.dim() != 5:
        return 'needs the per-head phase [B, nc, T, H, N/2]'
    if theta.shape[-1] != N // 2 or not theta.is_contiguous() or theta.dtype not in (torch.float32, torch.bfloat16):
        return 'phase must be contiguous fp32/bf16 with N/2 pairs'
    if theta_h is not None or rot_bias is not None:
        return 'theta_h / rot_bias stages not served'
    if (beta is not None or gamma is not None) and (not bias_pre_rot):
        return 'post-rotation bias order not served'
    if dw is None:
        return 'needs the write scale (dw)'
    if raw_common:
        return 'SSD_MIMO_SISO_ADJ_RAW_COMMON form not served'
    if not need_dtheta:
        return 'dTheta not needed (not served)'
    for nm, t in (('Btil', Btil), ('Ctil', Ctil)):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16:
            return f'{nm} must be contiguous bf16 [B, nc, T, N]'
    w = aux['scale']
    if w.stride() != qk_f.stride() if has_trap else False:
        return 'w / qk strides differ'
    if not w.is_contiguous() or (has_trap and (not aux['gam'].is_contiguous())):
        return 'w / g must be contiguous [B, H, L]'
    return None

def launch(**kw):
    from triton.runtime.errors import OutOfResources
    try:
        return _launch_impl(**kw)
    except OutOfResources:
        cal_on = [u for u in _lv.GLUON_UNITS if _lv.explicit(u) is None and _lv.on(u, _GRP)]
        rk = kw.get('rot_ka')
        N = kw['Bs'].shape[-1]
        cal_ka = rk is not None and rk < N // 2 and (_lv.explicit('rot_ka_bwd') is None)
        if not (cal_on or cal_ka) or not _lv.RETRY_ON_OOR:
            raise
        try:
            with _lv.suppressed(cal_on):
                r = _launch_impl(**kw)
            _LAST_ROUTE['lever_retry'] = cal_on
            return r
        except OutOfResources:
            if not cal_ka:
                raise
        with _lv.suppressed(cal_on):
            r = _launch_impl(**dict(kw, rot_ka=None))
        _LAST_ROUTE['lever_retry'] = cal_on + ['rot_ka_bwd']
        return r

def _launch_impl(*, Bs, Cs, dBs, dCs, x, dX, S_all, dY, delta_cumsum, a_log, dsf, dsk_c, qk_f, dDelta, dQK, dX0, has_trap, has_dskip, has_dsf, has_x0, f_st, w_st, idx64, adj=None, dth=None, no_dd=False, rot_ka=None):
    Bsz, nc, T, H, N = Bs.shape
    rot_ka = N // 2 if rot_ka is None else rot_ka
    G = x.shape[2]
    L = nc * T
    lay_bn = gl.NVMMASharedLayout.get_default_for([T, N], gl.bfloat16)
    x_packed = packed_x(x)
    from . import levers as _lv
    dy_tmajor = dY.stride(3) == 1 and _lv.on('gl_dy_tmajor', _GRP, guard=dY.stride(3) == 1)
    lay_x = gl.NVMMASharedLayout.get_default_for([T, G] if x_packed else [G, T], gl.bfloat16)
    lay_s = gl.NVMMASharedLayout.get_default_for([N, G], gl.bfloat16)

    def mdesc(t):
        v = t.view(Bsz * L, H * N)
        return TensorDescriptor(v, [Bsz * L, H * N], [H * N, 1], [T, N], lay_bn)

    def xdesc(t):
        if x_packed:
            v = t.permute(0, 4, 1, 2, 3).view(Bsz * L, H * G)
            return TensorDescriptor(v, [Bsz * L, H * G], [H * G, 1], [T, G], lay_x)
        v = t.view(Bsz * H * G, L)
        return TensorDescriptor(v, [Bsz * H * G, L], [L, 1], [G, T], lay_x)
    sv = S_all.view(Bsz * H * nc * N, G)
    s_desc = TensorDescriptor(sv, [Bsz * H * nc * N, G], [G, 1], [N, G], lay_s)
    wn, stages, sync, price = levers()
    _e = _lv.explicit('gl_dth_store_vec16')
    dth_store_vec = int(_e) if _e is not None else 16 if _lv.on('gl_dth_store_vec16', _GRP) else 4
    if dth_store_vec not in (2, 4, 8, 16):
        raise ValueError('DTH_STORE_VEC must be 2, 4, 8 or 16')
    _LAST_ROUTE.update(used=True, why=None, lever_retry=None, T=T, N=N, G=G, wn=wn, stages=stages, sync=sync, price=price, fuse_adj=adj is not None, dth_rot=dth is not None, no_dd=bool(no_dd), dth_store_vec=dth_store_vec, rot_ka=rot_ka, x_packed=x_packed, dy_tmajor=dy_tmajor)
    narrow_dth = not _lv.on('gl_narrow_dth_off', _GRP)
    _LAST_ROUTE['narrow_dth'] = narrow_dth
    dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    if adj is not None:
        rd = lambda t: TensorDescriptor(t.view(Bsz * L, N), [Bsz * L, N], [N, 1], [T, N], lay_bn)
        th = adj['theta']
        a_args = (rd(adj['Btil']), rd(adj['Ctil']), th, adj['dTheta'], adj['w'], adj.get('gam', dummy), adj.get('beta', dummy), adj.get('gamma', dummy), adj['dW_p'], adj.get('dG_p', dummy), adj.get('dBeta', dummy), adj.get('dGamma', dummy), th.stride(0), th.stride(1), th.stride(2), th.stride(3), th.stride(4), adj['dW_p'].stride(0), adj['dW_p'].stride(1), adj['dW_p'].stride(3))
        a_kw = dict(FUSE_ADJ=True, HAS_BETA='beta' in adj, HAS_GAMMA='gamma' in adj)
        if w_st == (0, 0, 0):
            w_st = adj['w'].stride()
    else:
        a_args = (s_desc, s_desc, *[dummy] * 10, *[0] * 8)
        a_kw = {}
    ws_ok = T == 64 and N == 128 and (G == 64) and (wn == 2) and (stages == 2) and (adj is None) and (not no_dd) and (span_mma() in (1, 2))
    ws_requested = _lv.on('gl_ws_producer', _GRP, guard=ws_ok)
    ws = ws_requested and ws_ok
    _LAST_ROUTE.update(ws_producer=ws, ws_producer_requested=ws_requested, ws_producer_decline='outside initial T64/N128/G64 full-gradient envelope' if ws_requested and (not ws_ok) else None)
    if ws:
        from .gluon_sm90_siso_ws import _gl90_siso_bwd_ws_kernel
        kernel = _gl90_siso_bwd_ws_kernel
    else:
        kernel = _gl90_siso_bwd_kernel
    ws_prep = ws and _lv.on('gl_ws_prep', _GRP)
    split2 = _lv.on('gl_split2', _GRP)
    _LAST_ROUTE['split2'] = split2
    ws_preg = int(os.environ.get('SSD_MIMO_GLUON90_SISO_WS_PREG', str(_lv.cfg_value('preg_prep', _GRP) or 48 if ws_prep else _lv.cfg_value('preg', _GRP) or 24))) if ws else None
    if ws and ws_preg not in (24, 32, 40, 48, 56, 64):
        raise ValueError('WS_PREG must be 24, 32, 40, 48, 56 or 64')
    _LAST_ROUTE.update(ws_prep=ws_prep, ws_preg=ws_preg)
    maxnreg = int(os.environ.get('SSD_MIMO_GLUON90_SISO_MAXNREG', str(_lv.cfg_value('maxnreg', _GRP) or 0)))
    _LAST_ROUTE['maxnreg'] = maxnreg
    kernel[Bsz, H](mdesc(Bs), mdesc(Cs), mdesc(dBs), mdesc(dCs), xdesc(x), xdesc(dX), s_desc, dY, delta_cumsum, a_log, dsf, dsk_c, qk_f, dDelta, dQK, dX0, H, dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3), dY.stride(5), *delta_cumsum.stride(), *w_st, *f_st, nc, *a_args, dth if dth is not None else dummy, *(dth.stride() if dth is not None else (0,) * 5), T=T, N=N, G=G, HAS_TRAP=bool(has_trap), HAS_DSKIP=bool(has_dskip), HAS_DSF=bool(has_dsf), HAS_X0=bool(has_x0), IDX64=bool(idx64), SPLIT2=split2, WN=wn, STAGES=stages, SYNC=sync, PRICE=price, SPAN_MMA=span_mma(), DTH_ROT=0 if dth is None else dth_mode(), DTH_STORE_VEC=dth_store_vec, ROT_KA=rot_ka, X_PACKED=x_packed, DY_TMAJOR=dy_tmajor, NARROW_DTH=narrow_dth, NO_DD=bool(no_dd), num_warps=4 * wn, **a_kw, **{'maxnreg': maxnreg} if maxnreg else {}, **dict(PREP=ws_prep, PREG=ws_preg) if ws else {})
