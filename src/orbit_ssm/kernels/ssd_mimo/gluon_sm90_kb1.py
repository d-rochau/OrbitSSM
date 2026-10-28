
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
def _segadd_combine(v1, f1, v2, f2):
    return (gl.where(f2 != 0.0, v2, v1 + v2), gl.maximum(f1, f2))

@gluon.jit
def _span_ddelta(rd, wr, a_val, carry, s_i, k_i):
    tile = gl.where(k_i[None, :] >= s_i[:, None], rd[None, :], 0.0) + gl.where(k_i[None, :] < s_i[:, None], wr[None, :], 0.0)
    return a_val * (gl.sum(tile, axis=1) + carry)

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
_OPR = gl.constexpr(os.environ.get('SSD_MIMO_DTHETA_OPROUND', '1') != '0')

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
def _gl90_kb1_kernel(b_desc, c_desc, x_desc, s_desc, g_desc, Delta_ptr, A_ptr, Beta_ptr, Gamma_ptr, DW_ptr, dY_ptr, dG_ptr, dAlpha_ptr, dCtil_ptr, dBtil_ptr, dDelta_ptr, dBeta_ptr, dGamma_ptr, H, rb_b, rb_c, rb_h, rb_r, rc_b, rc_c, rc_h, rc_o, s_xr, rs_b, rs_h, rs_c, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_gb, s_gh, s_gc, s_gn, s_gg, s_dab, s_dah, s_dag, s_dac, s_wb, s_wh, s_wl, s_dcb, s_dcc, s_dch, s_dct, s_dcn, s_dco, s_dbb, s_dbc, s_dbh, s_dbt, s_dbn, s_dbr, NC, Theta_ptr, ThetaH_ptr, dTheta_ptr, dThetaH_ptr, s_thb, s_thc, s_tht, s_thh, s_thk, s_dtb, s_dtc, s_dtt, s_dth, s_dtk, T: gl.constexpr, N: gl.constexpr, G_SIZE: gl.constexpr, GT: gl.constexpr, R: gl.constexpr, O: gl.constexpr, HPG_B: gl.constexpr, HPG_C: gl.constexpr, HBLK: gl.constexpr, HAS_BETA: gl.constexpr, HAS_GAMMA: gl.constexpr, DCS_INK: gl.constexpr, DB_STORE: gl.constexpr, DC_STORE: gl.constexpr, DY_TFAST: gl.constexpr, DG_TMA: gl.constexpr=False, WN: gl.constexpr=2, PF: gl.constexpr=False, ROT: gl.constexpr=False, HAS_THETA_H: gl.constexpr=False, NEED_DTHH: gl.constexpr=False, DTH_STORE: gl.constexpr=False, TRAP_DIAG: gl.constexpr=False, IDX64: gl.constexpr=False):
    NW: gl.constexpr = 4 * WN
    gl.static_assert(gl.num_warps() == NW, 'WN warpgroups per CTA, split along N')
    TO: gl.constexpr = O * T
    TR: gl.constexpr = R * T
    gl.static_assert(TO == 64 and TR == 64, 'stacked M = 64 (T = 32, R = O = 2)')
    gl.static_assert(GT == G_SIZE, 'one G tile per CTA (NGT = 1)')
    MMA_W: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, N // WN, 16])
    MMA_S: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN], instr_shape=[16, T // WN, 16])
    V1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    BLK_X: gl.constexpr = gl.BlockedLayout([1, 4], [32 // (T // 4), T // 4], [NW, 1], [1, 0])
    BLK_G: gl.constexpr = gl.BlockedLayout([1, 4], [32 // (GT // 4), GT // 4], [NW, 1], [1, 0])
    if DY_TFAST:
        BLK_Y: gl.constexpr = gl.BlockedLayout([2, 1], [32, 1], [1, NW], [0, 1])
    else:
        BLK_Y: gl.constexpr = gl.BlockedLayout([1, 4], [32 // (GT // 4), GT // 4], [NW, 1], [1, 0])
    LAY_BN: gl.constexpr = b_desc.layout
    LAY_X: gl.constexpr = x_desc.layout
    LAY_S: gl.constexpr = s_desc.layout
    LAY_MG: gl.constexpr = gl.NVMMASharedLayout.get_default_for([TO, GT], gl.bfloat16)
    LAY_P: gl.constexpr = gl.NVMMASharedLayout.get_default_for([TO, TR], gl.bfloat16)
    FLAT1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    FLAT2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    BLK_TT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NW, 1], [1, 0])
    b = gl.program_id(0)
    hblk = gl.program_id(1)
    c = gl.program_id(2)
    if IDX64:
        bo = b.to(gl.int64)
    else:
        bo = b
    h_start = hblk * HBLK
    bg = h_start // HPG_B
    cg = h_start // HPG_C
    bst = gl.allocate_shared_memory(gl.bfloat16, [TR, N], LAY_BN)
    cst = gl.allocate_shared_memory(gl.bfloat16, [TO, N], LAY_BN)
    xs = gl.allocate_shared_memory(gl.bfloat16, [2 * R, GT, T], LAY_X)
    ss = gl.allocate_shared_memory(gl.bfloat16, [2, N, GT], LAY_S)
    dgs = gl.allocate_shared_memory(gl.bfloat16, [N, GT], LAY_S)
    if DG_TMA:
        dgst = gl.allocate_shared_memory(gl.float32, [2, N, GT], g_desc.layout)
    dys = gl.allocate_shared_memory(gl.bfloat16, [TO, GT], LAY_MG)
    xws = gl.allocate_shared_memory(gl.bfloat16, [TR, GT], LAY_MG)
    psm = gl.allocate_shared_memory(gl.bfloat16, [TO, TR], LAY_P)
    vecs = gl.allocate_shared_memory(gl.float32, [4, T], FLAT1)
    csv = gl.allocate_shared_memory(gl.float32, [O, TR], FLAT1)
    red = gl.allocate_shared_memory(gl.float32, [2, 64], FLAT1)
    KP: gl.constexpr = N // 2
    if ROT:
        bts = gl.allocate_shared_memory(gl.bfloat16, [TR, N], LAY_BN)
        cts = gl.allocate_shared_memory(gl.bfloat16, [TO, N], LAY_BN)
        dthsm = gl.allocate_shared_memory(gl.float32, [128, KP], FLAT2)
        csm = gl.allocate_shared_memory(gl.float32, [128, KP], FLAT2)
        zw = gl.zeros([64, N], gl.float32, MMA_W)
        ze, zo = gl.split(gl.reshape(zw, [64, KP, 2]))
        L_WK: gl.constexpr = ze.type.layout
        k_w = gl.arange(0, KP, layout=gl.SliceLayout(0, L_WK))
        m_k = gl.arange(0, 64, layout=gl.SliceLayout(1, L_WK))
        th_off = (m_k % T)[:, None] * s_tht + k_w[None, :] * s_thk
        BLK_K: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [NW, 1], [1, 0])
        t_o = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_K))
        k_o = gl.arange(0, KP, layout=gl.SliceLayout(0, BLK_K))
    bars = gl.allocate_shared_memory(gl.int64, [3, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(3):
        mbarrier.init(bars.index(i), count=1)
    fence_async_shared()
    bar_bc = bars.index(2)
    mbarrier.expect(bar_bc, (TR + TO) * N * 2)
    for r in gl.static_range(R):
        tma.async_load(b_desc, [b * rb_b + c * rb_c + bg * rb_h + r * rb_r, 0], bar_bc, bst.slice(r * T, T))
    for o in gl.static_range(O):
        tma.async_load(c_desc, [b * rc_b + c * rc_c + cg * rc_h + o * rc_o, 0], bar_bc, cst.slice(o * T, T))
    HBYTES: gl.constexpr = R * GT * T * 2 + N * GT * 2 + (N * GT * 4 if DG_TMA else 0)
    h0 = gl.minimum(h_start, H - 1)
    bar0 = bars.index(0)
    mbarrier.expect(bar0, HBYTES)
    for r in gl.static_range(R):
        tma.async_load(x_desc, [(b * H + h0) * G_SIZE, r * s_xr + c * T], bar0, xs.index(r))
    tma.async_load(s_desc, [b * rs_b + h0 * rs_h + c * rs_c, 0], bar0, ss.index(0))
    if DG_TMA:
        tma.async_load(g_desc, [b * rs_b + h0 * rs_h + c * rs_c, 0], bar0, dgst.index(0))
    offs_t = gl.arange(0, T, layout=V1)
    m_s = gl.arange(0, TO, layout=gl.SliceLayout(1, MMA_S))
    tm_s = m_s % T
    om_s = m_s // T
    k_s = gl.arange(0, T, layout=gl.SliceLayout(0, MMA_S))
    m_w = gl.arange(0, 64, layout=gl.SliceLayout(1, MMA_W))
    tm_w = m_w % T
    blk_w = m_w // T
    n_w = gl.arange(0, N, layout=gl.SliceLayout(0, MMA_W))
    gx = gl.arange(0, GT, layout=gl.SliceLayout(1, BLK_X))
    tx = gl.arange(0, T, layout=gl.SliceLayout(0, BLK_X))
    ng = gl.arange(0, N, layout=gl.SliceLayout(1, BLK_G))
    gg = gl.arange(0, GT, layout=gl.SliceLayout(0, BLK_G))
    my = gl.arange(0, TO, layout=gl.SliceLayout(1, BLK_Y))
    gy = gl.arange(0, GT, layout=gl.SliceLayout(0, BLK_Y))
    s_tt = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_TT))
    k_tt = gl.arange(0, T, layout=gl.SliceLayout(0, BLK_TT))
    L_TK: gl.constexpr = gl.SliceLayout(0, BLK_TT)
    dC_tot = gl.zeros([TO, N], gl.float32, MMA_W)
    dB_tot = gl.zeros([TR, N], gl.float32, MMA_W)
    y_off = (my // T)[:, None] * s_yo + (my % T)[:, None] * s_yt + gy[None, :] * s_yg
    if PF:
        nx_dy = gl.load(dY_ptr + bo * s_yb + h0 * s_yh + c * s_yc + y_off).to(gl.bfloat16)
        nx_d = gl.load(Delta_ptr + bo * s_db + h0 * s_dh + c * s_dc + offs_t * s_dt)
        nx_w = gl.load(DW_ptr + bo * s_wb + h0 * s_wh + (c * T + offs_t) * s_wl)
        nx_a = gl.load(A_ptr + h0)
        nx_al = gl.load(dAlpha_ptr + bo * s_dab + h0 * s_dah + c * s_dac)
        if ROT:
            nx_th = gl.load(Theta_ptr + bo * s_thb + c * s_thc + h0 * s_thh + th_off)
    mbarrier.wait(bar_bc, 0)
    for hh in range(HBLK):
        slot = hh % 2
        phase = hh // 2 & 1
        h_raw = h_start + hh
        hv = h_raw < H
        h = gl.minimum(h_raw, H - 1)
        nslot = 1 - slot
        more = h_raw + 1 < gl.minimum(h_start + HBLK, H)
        hn = gl.minimum(h_raw + 1, H - 1)
        nbar = bars.index(nslot)
        mbarrier.expect(nbar, HBYTES, pred=more)
        for r in gl.static_range(R):
            tma.async_load(x_desc, [(b * H + hn) * G_SIZE, r * s_xr + c * T], nbar, xs.index(nslot * R + r), pred=more)
        tma.async_load(s_desc, [b * rs_b + hn * rs_h + c * rs_c, 0], nbar, ss.index(nslot), pred=more)
        if DG_TMA:
            tma.async_load(g_desc, [b * rs_b + hn * rs_h + c * rs_c, 0], nbar, dgst.index(nslot), pred=more)
        if PF:
            dy = nx_dy
            d_raw = nx_d
            dw = nx_w
            a_val = -gl.exp(nx_a)
            dalpha = nx_al
            pm = (offs_t >= 0) & more
            nx_dy = gl.load(dY_ptr + bo * s_yb + hn * s_yh + c * s_yc + y_off, mask=(y_off >= 0) & more, other=0.0).to(gl.bfloat16)
            nx_d = gl.load(Delta_ptr + bo * s_db + hn * s_dh + c * s_dc + offs_t * s_dt, mask=pm, other=0.0)
            nx_w = gl.load(DW_ptr + bo * s_wb + hn * s_wh + (c * T + offs_t) * s_wl, mask=pm, other=0.0)
            nx_a = gl.load(A_ptr + hn)
            nx_al = gl.load(dAlpha_ptr + bo * s_dab + hn * s_dah + c * s_dac)
            if ROT:
                cur_th = nx_th
                nx_th = gl.load(Theta_ptr + bo * s_thb + c * s_thc + hn * s_thh + th_off, mask=(th_off >= 0) & more, other=0.0)
        else:
            dy = gl.load(dY_ptr + bo * s_yb + h * s_yh + c * s_yc + y_off).to(gl.bfloat16)
            d_raw = gl.load(Delta_ptr + bo * s_db + h * s_dh + c * s_dc + offs_t * s_dt)
            dw = gl.load(DW_ptr + bo * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            a_val = -gl.exp(gl.load(A_ptr + h))
            dalpha = gl.load(dAlpha_ptr + bo * s_dab + h * s_dah + c * s_dac)
            if ROT:
                cur_th = gl.load(Theta_ptr + bo * s_thb + c * s_thc + h * s_thh + th_off)
        if DCS_INK:
            rst = a_val * d_raw <= -104.0
            ri = gl.where(rst, 1.0, 0.0)
            D_h, _rf = gl.associative_scan((gl.where(rst, 0.0, d_raw), ri), 0, _segadd_combine)
        else:
            rst = d_raw < 0.0
            ri = gl.where(rst, 1.0, 0.0)
            D_h = gl.where(rst, 0.0, d_raw)
        seg_h = gl.associative_scan(ri, 0, _add_combine)
        seg_end = gl.max(seg_h, axis=0)
        D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        alpha = gl.where(seg_end == 0.0, gl.exp(a_val * D_end), 0.0)
        vecs.index(0).store(D_h)
        vecs.index(1).store(dw)
        vecs.index(2).store(gl.where(seg_h == 0.0, gl.exp(a_val * D_h), 0.0))
        vecs.index(3).store(seg_h)
        dys.store(dy)
        if not DG_TMA:
            dg = gl.load(dG_ptr + bo * s_gb + h * s_gh + c * s_gc + ng[:, None] * s_gn + gg[None, :] * s_gg)
            dgs.store(dg.to(gl.bfloat16))
        mbarrier.wait(bars.index(slot), phase)
        if DG_TMA:
            dgs.store(dgst.index(slot).load(BLK_G).to(gl.bfloat16))
        D_x = vecs.index(0).load(gl.SliceLayout(0, BLK_X))
        w_x = vecs.index(1).load(gl.SliceLayout(0, BLK_X))
        seg_x = vecs.index(3).load(gl.SliceLayout(0, BLK_X))
        sc_x = w_x * gl.where(seg_x == seg_end, gl.exp(a_val * (D_end - D_x)), 0.0)
        for r in gl.static_range(R):
            x_r = xs.index(slot * R + r).load(BLK_X).to(gl.float32)
            xws.slice(r * T, T).permute((1, 0)).store((x_r * sc_x[None, :]).to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
        if not ROT:
            dS_0 = gl.zeros([TO, T], gl.float32, MMA_S)
            dS_0 = warpgroup_mma(dys, xs.index(slot * R), dS_0, use_acc=False, is_async=True)
            dS_1 = gl.zeros([TO, T], gl.float32, MMA_S)
            dS_1 = warpgroup_mma(dys, xs.index(slot * R + 1), dS_1, use_acc=False, is_async=True)
            dC = gl.zeros([TO, N], gl.float32, MMA_W)
            dC = warpgroup_mma(dys, ss.index(slot).permute((1, 0)), dC, use_acc=False, is_async=True)
            dB = gl.zeros([TR, N], gl.float32, MMA_W)
            dB = warpgroup_mma(xws, dgs.permute((1, 0)), dB, use_acc=False, is_async=True)
            dS_0, dS_1, dC, dB = warpgroup_mma_wait(0, deps=[dS_0, dS_1, dC, dB])
            D_row = vecs.index(0).gather(tm_s, 0)
            D_col = vecs.index(0).load(gl.SliceLayout(0, MMA_S))
            w_col = vecs.index(1).load(gl.SliceLayout(0, MMA_S))
            seg_row = vecs.index(3).gather(tm_s, 0)
            seg_col = vecs.index(3).load(gl.SliceLayout(0, MMA_S))
            if TRAP_DIAG:
                cm = tm_s[:, None] > k_s[None, :]
            else:
                cm = tm_s[:, None] >= k_s[None, :]
            cm = cm & (seg_row[:, None] == seg_col[None, :])
            dec = gl.where(cm, gl.exp(a_val * (D_row[:, None] - D_col[None, :])), 0.0) * w_col[None, :]
            P_0 = (dS_0 * dec).to(gl.bfloat16)
            P_1 = (dS_1 * dec).to(gl.bfloat16)
            psm.slice(0, T, dim=1).store(P_0)
            psm.slice(T, T, dim=1).store(P_1)
            Pf_0 = P_0.to(gl.float32)
            Pf_1 = P_1.to(gl.float32)
            rs_0 = gl.convert_layout(gl.sum(Pf_0, axis=1), gl.SliceLayout(1, MMA_W))
            rs_1 = gl.convert_layout(gl.sum(Pf_1, axis=1), gl.SliceLayout(1, MMA_W))
            if HAS_GAMMA:
                for o in gl.static_range(O):
                    csv.index(o).slice(0, T).store(gl.sum(gl.where((om_s == o)[:, None], Pf_0, 0.0), axis=0))
                    csv.index(o).slice(T, T).store(gl.sum(gl.where((om_s == o)[:, None], Pf_1, 0.0), axis=0))
            f_row = vecs.index(2).gather(tm_w, 0)
            dC = dC * f_row[:, None]
            Cf = cst.load(MMA_W).to(gl.float32)
            if HAS_GAMMA:
                Cf = Cf + gl.load(Gamma_ptr + h * N * O + n_w[None, :] * O + blk_w[:, None])
            red.index(0).store(gl.sum(Cf * dC, axis=1))
            Bf = bst.load(MMA_W).to(gl.float32)
            if HAS_BETA:
                Bf = Bf + gl.load(Beta_ptr + h * N * R + n_w[None, :] * R + blk_w[:, None])
            red.index(1).store(gl.sum(Bf * dB, axis=1))
            fence_async_shared()
            gl.barrier()
            dC = warpgroup_mma(psm.slice(0, T, dim=1), bst.slice(0, T), dC, is_async=True)
            dC = warpgroup_mma(psm.slice(T, T, dim=1), bst.slice(T, T), dC, is_async=True)
            dB = warpgroup_mma(psm.permute((1, 0)), cst, dB, is_async=True)
            dC, dB = warpgroup_mma_wait(0, deps=[dC, dB])
            if HAS_BETA:
                dC = dC + rs_0[:, None] * gl.load(Beta_ptr + h * N * R + n_w * R)[None, :]
                dC = dC + rs_1[:, None] * gl.load(Beta_ptr + h * N * R + n_w * R + 1)[None, :]
            if HAS_GAMMA:
                for o in gl.static_range(O):
                    cs_o = csv.index(o).load(gl.SliceLayout(1, MMA_W))
                    dB = dB + cs_o[:, None] * gl.load(Gamma_ptr + h * N * O + n_w * O + o)[None, :]
            if HAS_GAMMA:
                for o in gl.static_range(O):
                    gl.atomic_add(dGamma_ptr + h * N * O + n_w * O + o, gl.sum(gl.where((blk_w == o)[:, None], dC, 0.0), axis=0), mask=(n_w >= 0) & hv, sem='relaxed')
            dC_tot = dC_tot + gl.where(hv, dC, 0.0)
            if HAS_BETA:
                for r in gl.static_range(R):
                    gl.atomic_add(dBeta_ptr + h * N * R + n_w * R + r, gl.sum(gl.where((blk_w == r)[:, None], dB, 0.0), axis=0), mask=(n_w >= 0) & hv, sem='relaxed')
            dB_tot = dB_tot + gl.where(hv, dB, 0.0)
            rd = red.index(0).slice(0, T).load(L_TK) + red.index(0).slice(T, T).load(L_TK)
            wr = red.index(1).slice(0, T).load(L_TK) + red.index(1).slice(T, T).load(L_TK)
            dD = _span_ddelta(rd, wr, a_val, alpha * dalpha, s_tt, k_tt)
            gl.atomic_add(dDelta_ptr + bo * s_db + h * s_dh + c * s_dc + s_tt * s_dt, dD, mask=(s_tt >= 0) & hv, sem='relaxed')
        else:
            thf = cur_th.to(gl.float32)
            c_w = _cos_ptx(thf)
            s_w = _sin_ptx(thf)
            csm.slice(0, 64).store(c_w)
            csm.slice(64, 64).store(s_w)
            if HAS_THETA_H:
                nu = gl.load(ThetaH_ptr + h * KP + k_w)
                ch_w = _cos_ptx(nu)[None, :]
                sh_w = _sin_ptx(nu)[None, :]
            Bp = bst.load(MMA_W).to(gl.float32)
            if HAS_BETA:
                Bp = Bp + gl.load(Beta_ptr + h * N * R + n_w[None, :] * R + blk_w[:, None])
            bts.store(_rot_w(Bp, c_w, s_w, 64, N).to(gl.bfloat16))
            Cp = cst.load(MMA_W).to(gl.float32)
            if HAS_GAMMA:
                Cp = Cp + gl.load(Gamma_ptr + h * N * O + n_w[None, :] * O + blk_w[:, None])
            Ch = _rot_w(Cp, c_w, s_w, 64, N)
            if HAS_THETA_H:
                Ch = _rot_w(Ch, ch_w, sh_w, 64, N)
            cts.store(Ch.to(gl.bfloat16))
            fence_async_shared()
            gl.barrier()
            dS_0 = gl.zeros([TO, T], gl.float32, MMA_S)
            dS_0 = warpgroup_mma(dys, xs.index(slot * R), dS_0, use_acc=False, is_async=True)
            dS_1 = gl.zeros([TO, T], gl.float32, MMA_S)
            dS_1 = warpgroup_mma(dys, xs.index(slot * R + 1), dS_1, use_acc=False, is_async=True)
            dC = gl.zeros([TO, N], gl.float32, MMA_W)
            dC = warpgroup_mma(dys, ss.index(slot).permute((1, 0)), dC, use_acc=False, is_async=True)
            dB = gl.zeros([TR, N], gl.float32, MMA_W)
            dB = warpgroup_mma(xws, dgs.permute((1, 0)), dB, use_acc=False, is_async=True)
            dS_0, dS_1, dC, dB = warpgroup_mma_wait(0, deps=[dS_0, dS_1, dC, dB])
            D_row = vecs.index(0).gather(tm_s, 0)
            D_col = vecs.index(0).load(gl.SliceLayout(0, MMA_S))
            w_col = vecs.index(1).load(gl.SliceLayout(0, MMA_S))
            seg_row = vecs.index(3).gather(tm_s, 0)
            seg_col = vecs.index(3).load(gl.SliceLayout(0, MMA_S))
            if TRAP_DIAG:
                cm = tm_s[:, None] > k_s[None, :]
            else:
                cm = tm_s[:, None] >= k_s[None, :]
            cm = cm & (seg_row[:, None] == seg_col[None, :])
            dec = gl.where(cm, gl.exp(a_val * (D_row[:, None] - D_col[None, :])), 0.0) * w_col[None, :]
            psm.slice(0, T, dim=1).store((dS_0 * dec).to(gl.bfloat16))
            psm.slice(T, T, dim=1).store((dS_1 * dec).to(gl.bfloat16))
            f_row = vecs.index(2).gather(tm_w, 0)
            dC = dC * f_row[:, None]
            red.index(1).store(gl.sum(bts.load(MMA_W).to(gl.float32) * dB, axis=1))
            red.index(0).store(gl.sum(cts.load(MMA_W).to(gl.float32) * dC, axis=1))
            fence_async_shared()
            gl.barrier()
            dC = warpgroup_mma(psm.slice(0, T, dim=1), bts.slice(0, T), dC, is_async=True)
            dC = warpgroup_mma(psm.slice(T, T, dim=1), bts.slice(T, T), dC, is_async=True)
            dB = warpgroup_mma(psm.permute((1, 0)), cts, dB, is_async=True)
            dC, dB = warpgroup_mma_wait(0, deps=[dC, dB])
            c_w = csm.slice(0, 64).load(L_WK)
            s_w = csm.slice(64, 64).load(L_WK)
            Cp = cst.load(MMA_W).to(gl.float32)
            if HAS_GAMMA:
                Cp = Cp + gl.load(Gamma_ptr + h * N * O + n_w[None, :] * O + blk_w[:, None])
            if _OPR:
                Ch = cts.load(MMA_W).to(gl.float32)
            else:
                Ch = _rot_w(Cp, c_w, s_w, 64, N)
                if HAS_THETA_H:
                    Ch = _rot_w(Ch, ch_w, sh_w, 64, N)
            if HAS_THETA_H:
                if NEED_DTHH:
                    gl.atomic_add(dThetaH_ptr + h * KP + k_w, gl.sum(_pair_cross(dC, Ch, 64, N), axis=0), mask=(k_w >= 0) & hv, sem='relaxed')
                g1 = _rot_w(dC, ch_w, -sh_w, 64, N)
                if _OPR:
                    Ct = _rot_w(Ch, ch_w, -sh_w, 64, N)
                else:
                    Cp = cst.load(MMA_W).to(gl.float32)
                    if HAS_GAMMA:
                        Cp = Cp + gl.load(Gamma_ptr + h * N * O + n_w[None, :] * O + blk_w[:, None])
                    Ct = _rot_w(Cp, c_w, s_w, 64, N)
            else:
                g1 = dC
                Ct = Ch
            dthsm.slice(0, 64).store(_pair_cross(g1, Ct, 64, N))
            g2 = _rot_w(g1, c_w, -s_w, 64, N)
            if HAS_GAMMA:
                for o in gl.static_range(O):
                    gl.atomic_add(dGamma_ptr + h * N * O + n_w * O + o, gl.sum(gl.where((blk_w == o)[:, None], g2, 0.0), axis=0), mask=(n_w >= 0) & hv, sem='relaxed')
            dC_tot = dC_tot + gl.where(hv, g2, 0.0)
            c_w = csm.slice(0, 64).load(L_WK)
            s_w = csm.slice(64, 64).load(L_WK)
            Bp = bst.load(MMA_W).to(gl.float32)
            if HAS_BETA:
                Bp = Bp + gl.load(Beta_ptr + h * N * R + n_w[None, :] * R + blk_w[:, None])
            if _OPR:
                Bt = bts.load(MMA_W).to(gl.float32)
            else:
                Bt = _rot_w(Bp, c_w, s_w, 64, N)
            dthsm.slice(64, 64).store(_pair_cross(dB, Bt, 64, N))
            gB = _rot_w(dB, c_w, -s_w, 64, N)
            if HAS_BETA:
                for r in gl.static_range(R):
                    gl.atomic_add(dBeta_ptr + h * N * R + n_w * R + r, gl.sum(gl.where((blk_w == r)[:, None], gB, 0.0), axis=0), mask=(n_w >= 0) & hv, sem='relaxed')
            dB_tot = dB_tot + gl.where(hv, gB, 0.0)
            gl.barrier()
            dth = dthsm.slice(0, T).load(BLK_K) + dthsm.slice(T, T).load(BLK_K) + dthsm.slice(64, T).load(BLK_K) + dthsm.slice(64 + T, T).load(BLK_K)
            dth_p = dTheta_ptr + bo * s_dtb + c * s_dtc + h * s_dth + t_o[:, None] * s_dtt + k_o[None, :] * s_dtk
            if DTH_STORE:
                gl.store(dth_p, dth, mask=(t_o[:, None] >= 0) & hv)
            else:
                gl.atomic_add(dth_p, dth, mask=(t_o[:, None] >= 0) & hv, sem='relaxed')
            rd = red.index(0).slice(0, T).load(L_TK) + red.index(0).slice(T, T).load(L_TK)
            wr = red.index(1).slice(0, T).load(L_TK) + red.index(1).slice(T, T).load(L_TK)
            dD = _span_ddelta(rd, wr, a_val, alpha * dalpha, s_tt, k_tt)
            gl.atomic_add(dDelta_ptr + bo * s_db + h * s_dh + c * s_dc + s_tt * s_dt, dD, mask=(s_tt >= 0) & hv, sem='relaxed')
    for i in gl.static_range(3):
        mbarrier.invalidate(bars.index(i))
    db_p = dBtil_ptr + bo * s_dbb + c * s_dbc + bg * s_dbh + tm_w[:, None] * s_dbt + n_w[None, :] * s_dbn + blk_w[:, None] * s_dbr
    if DB_STORE:
        gl.store(db_p, dB_tot)
    else:
        gl.atomic_add(db_p, dB_tot, sem='relaxed')
    dc_p = dCtil_ptr + bo * s_dcb + c * s_dcc + cg * s_dch + tm_w[:, None] * s_dct + n_w[None, :] * s_dcn + blk_w[:, None] * s_dco
    if DC_STORE:
        gl.store(dc_p, dC_tot)
    else:
        gl.atomic_add(dc_p, dC_tot, sem='relaxed')
_LAST_ROUTE: dict = {'used': False, 'why': 'never consulted'}

def last_route() -> dict:
    return dict(_LAST_ROUTE)

def _row_view(t, n):
    maxoff = sum(((d - 1) * s for d, s in zip(t.shape, t.stride()) if d > 0))
    return torch.as_strided(t, (maxoff // n + 1, n), (n, 1))

def refusal(*, device, T, N, G_SIZE, GT, NGT, R, O, mma_io, io_dtype, hblk, nsplit, kb2_fold, has_theta_h, rot_in_kernel, has_rot_bias, pair_half, deferred_rot, use_slots, idx64, has_dw, pre_rot, rot_abl, Btil, Ctil, x, S_all, dY, s_bt, s_bn, s_ct, s_cn, has_beta=False, has_gamma=False, phase_cumsum=False, theta=None):
    if TensorDescriptor is None:
        return 'no gluon hopper TensorDescriptor in this Triton'
    if torch.cuda.get_device_capability(device) != (9, 0):
        return 'not sm_90'
    if not mma_io or io_dtype != torch.bfloat16:
        return 'needs bf16 IO with 16-bit MMA operands'
    from .gluon_sm90_fwd import ptxas_refusal, bc_stride_refusal, x_refusal
    why = ptxas_refusal()
    if why:
        return why
    if rot_in_kernel or has_theta_h or has_rot_bias or pair_half or deferred_rot or rot_abl:
        if os.environ.get('SSD_MIMO_GLUON90_ROT', '1') == '0':
            return 'rotating arm disabled (SSD_MIMO_GLUON90_ROT=0)'
        if has_rot_bias:
            return 'rot_bias (the retired write-side angle) not served'
        if pair_half:
            return 'half pair layout not served (interleaved only)'
        if deferred_rot or not rot_in_kernel:
            return 'rotation must be the in-kernel per-head phase'
        if rot_abl:
            return 'rotation ablation ladder not served'
        if (has_beta or has_gamma) and (not pre_rot):
            return 'rotation with the post-rotation bias order not served'
        if phase_cumsum:
            return 'phase offsets (in-kernel phase cumsum) not served'
        if theta is None or theta.dim() not in (4, 5) or theta.dtype not in (torch.float32, torch.bfloat16):
            return 'phase must be fp32/bf16 [B, nc, T, (H,) N/2]'
    elif pre_rot and (has_beta or has_gamma):
        return 'bias_pre_rot without a rotation'
    if not hblk or not kb2_fold:
        return 'needs the owner-accumulate head block with the Kb2 fold'
    if nsplit != 1 or NGT != 1 or use_slots or (not has_dw):
        return 'envelope: NSPLIT 1, NGT 1, no slots, raw-x dw'
    if T != 32 or R * T != 64 or O * T != 64 or (GT != 32) or (G_SIZE != 32):
        return f'tile envelope T=32, R=O=2, G=GT=32 (T={T} R={R} O={O} G={G_SIZE})'
    if N % 64 or N > 256:
        return f'N must be a multiple of 64, <= 256 (got {N})'
    if s_bn != 1 or s_bt != N or s_cn != 1 or (s_ct != N):
        return 'B/C tiles must be [T, N] row-contiguous'
    why = bc_stride_refusal(Btil, Ctil, N=N)
    if why:
        return why
    if Btil.dtype != torch.bfloat16 or Ctil.dtype != torch.bfloat16:
        return 'B/C must be bf16'
    if x.dtype != torch.bfloat16 or x.stride(4) != 1:
        return 'x must be bf16 with L contiguous'
    _, H_, G_, _, _ = x.shape
    s = x.stride()
    if s[1] != G_ * s[2] or s[0] != H_ * s[1] or s[2] * 2 % 16:
        return 'x (b, h, g) rows must collapse to one uniform row stride'
    why = x_refusal(x, R, T, x.shape[4] // T)
    if why:
        return why
    if H_ % hblk:
        return f'H ({H_}) must be a multiple of the head block ({hblk})'
    if S_all.dtype != torch.bfloat16 or S_all.stride(4) != 1:
        return 'S_all must be bf16 with G contiguous'
    ss = S_all.stride()
    Bsz, H2, nc, N2, G2 = S_all.shape
    if ss[3] != G2 or ss[2] != N2 * ss[3] or ss[1] != nc * ss[2] or (ss[0] != H2 * ss[1]):
        return 'S_all must be contiguous'
    if dY.stride(3) != 1 and dY.stride(5) != 1:
        return 'dY must have T or G contiguous'
    for t in (Btil, Ctil, x, S_all):
        if t.data_ptr() % 16:
            return 'TMA base not 16-byte aligned'
    return None

def launch(*, Ctil, Btil, x, delta, a_log, S_all, dY, dG_all, dalpha_all, beta_c, gamma_c, has_beta, has_gamma, dw, dCtil, dBtil, dDelta, dBeta, dGamma, H, T, N, G_SIZE, GT, R, O, hpg_b, hpg_c, hblk, nc, db_store, dc_store, dcs_ink, b_strides, c_strides, db_strides, dc_strides, theta=None, th_strides=None, theta_h=None, dTheta=None, dth_strides=None, dThetaH=None, dth_store=False, trap_diag=False, idx64=False):
    s_bb, s_bc, s_bh, s_bt, s_bn, s_br = b_strides
    s_cb, s_cc, s_ch, s_ct, s_cn, s_co = c_strides
    lay_bn = gl.NVMMASharedLayout.get_default_for([T, N], gl.bfloat16)
    lay_x = gl.NVMMASharedLayout.get_default_for([GT, T], gl.bfloat16)
    lay_s = gl.NVMMASharedLayout.get_default_for([N, GT], gl.bfloat16)
    bv = _row_view(Btil, N)
    cv = _row_view(Ctil, N)
    b_desc = TensorDescriptor(bv, list(bv.shape), [N, 1], [T, N], lay_bn)
    c_desc = TensorDescriptor(cv, list(cv.shape), [N, 1], [T, N], lay_bn)
    Bsz, H_, G_, R_, Lp = x.shape
    xs_g = x.stride(2)
    xv = torch.as_strided(x, (Bsz * H_ * G_, xs_g), (xs_g, 1))
    x_desc = TensorDescriptor(xv, [Bsz * H_ * G_, xs_g], [xs_g, 1], [GT, T], lay_x)
    _, _, ncs, Ns, Gs = S_all.shape
    sv = torch.as_strided(S_all, (Bsz * H_ * ncs * Ns, Gs), (Gs, 1))
    s_desc = TensorDescriptor(sv, [Bsz * H_ * ncs * Ns, Gs], [Gs, 1], [N, GT], lay_s)
    dg_tma = os.environ.get('SSD_MIMO_GLUON90_KB1_DGTMA', '1') == '1'
    lay_g = gl.NVMMASharedLayout.get_default_for([N, GT], gl.float32)
    if dg_tma:
        gs = dG_all.stride()
        if dG_all.shape != S_all.shape or gs[4] != 1 or gs[3] != Gs or (gs[2] != Ns * Gs) or (gs[1] != ncs * Ns * Gs):
            dg_tma = False
    if dg_tma:
        gv = torch.as_strided(dG_all, (Bsz * H_ * ncs * Ns, Gs), (Gs, 1))
        g_desc = TensorDescriptor(gv, [Bsz * H_ * ncs * Ns, Gs], [Gs, 1], [N, GT], lay_g)
    else:
        g_desc = s_desc
    rows = lambda s: 0 if s == 1 else s // N
    dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    grid = (Bsz, (H + hblk - 1) // hblk, nc)
    _LAST_ROUTE['dg_tma'] = bool(dg_tma)
    wn = int(os.environ.get('SSD_MIMO_GLUON90_KB1_WN', '2'))
    pf = os.environ.get('SSD_MIMO_GLUON90_KB1_PF', '0' if theta is not None else '1') == '1'
    _LAST_ROUTE['wn'] = wn
    _LAST_ROUTE['pf'] = pf
    rot = theta is not None
    has_thh = theta_h is not None
    if rot and dTheta is None:
        raise ValueError('the rotating Kb1 needs its dTheta buffer')
    _LAST_ROUTE['rot'] = rot
    _LAST_ROUTE['theta_h'] = has_thh
    _LAST_ROUTE['dth_store'] = bool(rot and dth_store)
    _gl90_kb1_kernel[grid](b_desc, c_desc, x_desc, s_desc, g_desc, delta, a_log, beta_c if has_beta else dummy, gamma_c if has_gamma else dummy, dw, dY, dG_all, dalpha_all, dCtil, dBtil, dDelta, dBeta if has_beta else dummy, dGamma if has_gamma else dummy, H, rows(s_bb), rows(s_bc), rows(s_bh), rows(s_br), rows(s_cb), rows(s_cc), rows(s_ch), rows(s_co), x.stride(3), H_ * ncs * Ns, ncs * Ns, Ns, *delta.stride(), *dY.stride()[:4], dY.stride(5), dY.stride(4), *dG_all.stride(), dalpha_all.stride(0), dalpha_all.stride(1), dalpha_all.stride(2), dalpha_all.stride(3), *dw.stride(), *dc_strides, *db_strides, nc, theta if rot else dummy, theta_h if has_thh else dummy, dTheta if rot else dummy, dThetaH if has_thh and dThetaH is not None else dummy, *(th_strides if rot else (0, 0, 0, 0, 0)), *(dth_strides if rot else (0, 0, 0, 0, 0)), T=T, N=N, G_SIZE=G_SIZE, GT=GT, R=R, O=O, HPG_B=hpg_b, HPG_C=hpg_c, HBLK=hblk, HAS_BETA=has_beta, HAS_GAMMA=has_gamma, DCS_INK=bool(dcs_ink), DB_STORE=bool(db_store), DC_STORE=bool(dc_store), DY_TFAST=dY.stride(3) == 1, DG_TMA=bool(dg_tma), WN=wn, PF=pf, ROT=rot, HAS_THETA_H=has_thh, NEED_DTHH=bool(has_thh and dThetaH is not None), DTH_STORE=bool(rot and dth_store), TRAP_DIAG=bool(trap_diag), IDX64=bool(idx64), num_warps=4 * wn)
