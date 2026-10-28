
from __future__ import annotations
import os
import torch
import triton
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
def _pair_rot(V, c, s, T: gl.constexpr, N: gl.constexpr):
    e, o = gl.split(gl.reshape(V, [T, N // 2, 2]))
    return gl.reshape(gl.join(c * e + s * o, c * o - s * e), [T, N])

@gluon.jit
def _gl90_chunk_fwd_kernel(b_desc, c_desc, x_desc, Delta_ptr, A_ptr, X0_ptr, Beta_ptr, Gamma_ptr, Y_ptr, SAll_ptr, SF_ptr, DW_ptr, Lam_ptr, Dot_ptr, H, rb_b, rb_c, rb_h, rb_r, rc_b, rc_c, rc_h, rc_o, s_xr, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_sb, s_sh, s_sc, s_sn, s_sg, s_fb, s_fh, s_fg, s_fn, s_x0b, s_x0h, s_x0g, s_x0n, s_wb, s_wh, s_wl, s_dqb, s_dqh, s_dql, s_do, s_dr, NC, NC_SEG, s_x0s, s_fss, Theta_ptr, ThetaH_ptr, s_thb, s_thc, s_tht, s_thh, s_thk, T: gl.constexpr, N: gl.constexpr, G_SIZE: gl.constexpr, GT: gl.constexpr, R: gl.constexpr, O: gl.constexpr, HPG_B: gl.constexpr, HPG_C: gl.constexpr, HAS_X0: gl.constexpr, HAS_BETA: gl.constexpr, HAS_GAMMA: gl.constexpr, HAS_TRAP: gl.constexpr, STORE_DOT: gl.constexpr, STORE_Y: gl.constexpr, STORE_SALL: gl.constexpr, STATE_ONLY: gl.constexpr, DCS_INK: gl.constexpr, Y_TMAJ: gl.constexpr, PIPE: gl.constexpr=1, PREP: gl.constexpr=1, HAS_ROT: gl.constexpr=False, HAS_THETA_H: gl.constexpr=False, ROT_PF: gl.constexpr=1, IDX64: gl.constexpr=False):
    gl.static_assert(gl.num_warps() == 4, 'one warpgroup per CTA')
    TO: gl.constexpr = T * O
    gl.static_assert(STATE_ONLY or TO == 64, 'wgmma M = O*T must be 64')
    gl.static_assert(N % 64 == 0, 'N is a multiple of the 128B swizzle')
    MMA_T: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, T, 16])
    MMA_Y: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, GT, 16])
    A_Y: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA_Y, k_width=2)
    BLK_BN: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    BLK3: gl.constexpr = gl.BlockedLayout([1, 4, 2], [2, 16, 1], [4, 1, 1], [2, 1, 0])
    L_K: gl.constexpr = gl.SliceLayout(2, BLK3)
    BLK_X: gl.constexpr = gl.BlockedLayout([1, 8], [32 // (T // 8), T // 8], [4, 1], [1, 0])
    V1: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    if Y_TMAJ:
        BLK_Y: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    else:
        BLK_Y: gl.constexpr = gl.BlockedLayout([2, 1], [32, 1], [1, 4], [0, 1])
    LAY_BN: gl.constexpr = b_desc.layout
    LAY_X: gl.constexpr = x_desc.layout
    LAY_S: gl.constexpr = gl.NVMMASharedLayout.get_default_for([N, GT], gl.bfloat16)
    FLAT1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    FLAT2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    b = gl.program_id(0)
    h = gl.program_id(1)
    if IDX64:
        bo = b.to(gl.int64)
    else:
        bo = b
    NGT_C: gl.constexpr = G_SIZE // GT
    seg = gl.program_id(2) // NGT_C
    gt_idx = gl.program_id(2) % NGT_C
    c_lo = seg * NC_SEG
    c_hi = gl.minimum(NC, c_lo + NC_SEG)
    bg = h // HPG_B
    cg = h // HPG_C
    a_val = -gl.exp(gl.load(A_ptr + h))
    bst = gl.allocate_shared_memory(gl.bfloat16, [R * T, N], LAY_BN)
    beff = gl.allocate_shared_memory(gl.bfloat16, [R * T, N], LAY_BN)
    xs = gl.allocate_shared_memory(gl.bfloat16, [2 * R, GT, T], LAY_X)
    xw = gl.allocate_shared_memory(gl.bfloat16, [R, GT, T], LAY_X)
    vecs = gl.allocate_shared_memory(gl.float32, [5, T], FLAT1)
    if not STATE_ONLY:
        cst = gl.allocate_shared_memory(gl.bfloat16, [TO, N], LAY_BN)
        ceff = gl.allocate_shared_memory(gl.bfloat16, [TO, N], LAY_BN)
        s_sm = gl.allocate_shared_memory(gl.bfloat16, [N, GT], LAY_S)
    bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(bars.index(0), count=1)
    mbarrier.init(bars.index(1), count=1)
    fence_async_shared()
    XBYTES: gl.constexpr = R * GT * T * 2
    BBYTES: gl.constexpr = R * T * N * 2
    CBYTES: gl.constexpr = 0 if STATE_ONLY else TO * N * 2
    x_row = (b * H + h) * G_SIZE + gt_idx * GT
    bar0 = bars.index(0)
    mbarrier.expect(bar0, XBYTES + BBYTES + CBYTES)
    for r in gl.static_range(R):
        tma.async_load(b_desc, [b * rb_b + c_lo * rb_c + bg * rb_h + r * rb_r, 0], bar0, bst.slice(r * T, T))
        tma.async_load(x_desc, [x_row, r * s_xr + c_lo * T], bar0, xs.index(r))
    if not STATE_ONLY:
        for o in gl.static_range(O):
            tma.async_load(c_desc, [b * rc_b + c_lo * rc_c + cg * rc_h + o * rc_o, 0], bar0, cst.slice(o * T, T))
    offs_n_y = gl.arange(0, N, layout=gl.SliceLayout(1, MMA_Y))
    offs_g_y = gt_idx * GT + gl.arange(0, GT, layout=gl.SliceLayout(0, MMA_Y))
    if HAS_X0:
        S = gl.load(X0_ptr + bo * s_x0b + h * s_x0h + seg * s_x0s + offs_g_y[None, :] * s_x0g + offs_n_y[:, None] * s_x0n).to(gl.float32)
    else:
        S = gl.zeros([N, GT], gl.float32, MMA_Y)
    offs_t = gl.arange(0, T, layout=V1)
    offs_n_bn = gl.arange(0, N, layout=gl.SliceLayout(0, BLK_BN))
    offs_t_bn = gl.arange(0, T, layout=gl.SliceLayout(1, BLK_BN))
    offs_m_t = gl.arange(0, TO, layout=gl.SliceLayout(1, MMA_T))
    tm_t = offs_m_t % T
    offs_k_t = gl.arange(0, T, layout=gl.SliceLayout(0, MMA_T))
    offs_m_y = gl.arange(0, TO, layout=gl.SliceLayout(1, MMA_Y))
    tm_y = offs_m_y % T
    if not STATE_ONLY:
        s_sm.store(S.to(gl.bfloat16))
    wbase = bo * s_wb + h * s_wh
    dbase0 = Delta_ptr + bo * s_db + h * s_dh
    nx_d = gl.load(dbase0 + c_lo * s_dc + offs_t * s_dt)
    nx_w = gl.load(DW_ptr + wbase + (c_lo * T + offs_t) * s_wl)
    if HAS_TRAP:
        nx_l = gl.load(Lam_ptr + wbase + (c_lo * T + offs_t) * s_wl)
        nidx = c_lo * T + offs_t + 1
        nx_dn = gl.load(DW_ptr + wbase + nidx * s_wl, mask=nidx < NC * T, other=0.0)
        nx_ln = gl.load(Lam_ptr + wbase + nidx * s_wl, mask=nidx < NC * T, other=0.0)
    if HAS_ROT:
        k_k = gl.arange(0, N // 2, layout=gl.SliceLayout(0, L_K))
        t_k = gl.arange(0, T, layout=gl.SliceLayout(1, L_K))
        th_base = Theta_ptr + bo * s_thb + h * s_thh
        th_off = t_k[:, None] * s_tht + k_k[None, :] * s_thk
        if ROT_PF:
            nx_th = gl.load(th_base + c_lo * s_thc + th_off)
    if HAS_THETA_H:
        nu = gl.load(ThetaH_ptr + h * (N // 2) + k_k)
        ch_v = _cos_ptx(nu)[None, :]
        sh_v = _sin_ptx(nu)[None, :]
    for c in range(c_lo, c_hi):
        kk = c - c_lo
        slot = kk % 2
        phase = kk // 2 & 1
        cur_d = nx_d
        cur_w = nx_w
        if HAS_TRAP:
            cur_l = nx_l
            cur_dn = nx_dn
            cur_ln = nx_ln
        pm = (offs_t >= 0) & (c + 1 < c_hi)
        nx_d = gl.load(dbase0 + (c + 1) * s_dc + offs_t * s_dt, mask=pm, other=0.0)
        nx_w = gl.load(DW_ptr + wbase + ((c + 1) * T + offs_t) * s_wl, mask=pm, other=0.0)
        if HAS_TRAP:
            nx_l = gl.load(Lam_ptr + wbase + ((c + 1) * T + offs_t) * s_wl, mask=pm, other=0.0)
            nidx = (c + 1) * T + offs_t + 1
            pn = pm & (nidx < NC * T)
            nx_dn = gl.load(DW_ptr + wbase + nidx * s_wl, mask=pn, other=0.0)
            nx_ln = gl.load(Lam_ptr + wbase + nidx * s_wl, mask=pn, other=0.0)
        if HAS_ROT:
            if ROT_PF:
                cur_th = nx_th
                nx_th = gl.load(th_base + (c + 1) * s_thc + th_off, mask=(th_off >= 0) & (c + 1 < c_hi), other=0.0)
            else:
                cur_th = gl.load(th_base + c * s_thc + th_off)
        if DCS_INK:
            rst = a_val * cur_d <= -104.0
            ri = gl.where(rst, 1.0, 0.0)
            D_h, _rf = gl.associative_scan((gl.where(rst, 0.0, cur_d), ri), 0, _segadd_combine)
        else:
            rst = cur_d < 0.0
            ri = gl.where(rst, 1.0, 0.0)
            D_h = gl.where(rst, 0.0, cur_d)
        seg_h = gl.associative_scan(ri, 0, _add_combine)
        seg_end = gl.max(seg_h, axis=0)
        D_end = gl.sum(gl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        dw = cur_w
        if HAS_TRAP:
            trap_nxt = cur_dn * (1.0 - cur_ln)
            trap_own = cur_l * dw
            dw = cur_l * dw + trap_nxt
        alpha = gl.where(seg_end == 0.0, gl.exp(a_val * D_end), 0.0)
        vecs.index(4).store(seg_h)
        vecs.index(0).store(D_h)
        vecs.index(1).store(dw)
        if HAS_TRAP:
            vecs.index(2).store(trap_own)
        vecs.index(3).store(gl.where(seg_h == 0.0, gl.exp(a_val * D_h), 0.0))
        if STORE_SALL:
            gl.store(SAll_ptr + bo * s_sb + h * s_sh + c * s_sc + offs_n_y[:, None] * s_sn + offs_g_y[None, :] * s_sg, S.to(SAll_ptr.dtype.element_ty))
        mbarrier.wait(bars.index(slot), phase)
        if HAS_ROT:
            thf = cur_th.to(gl.float32)
            ph_c = _cos_ptx(thf)
            ph_s = _sin_ptx(thf)
        if not STATE_ONLY and PREP == 1:
            gl.static_assert(O == 2, 'PREP=1 holds the two C tiles')
            C_0 = cst.slice(0, T).load(BLK_BN).to(gl.float32)
            C_1 = cst.slice(T, T).load(BLK_BN).to(gl.float32)
            if HAS_GAMMA:
                C_0 = C_0 + gl.load(Gamma_ptr + h * N * O + offs_n_bn * O)[None, :]
                C_1 = C_1 + gl.load(Gamma_ptr + h * N * O + offs_n_bn * O + 1)[None, :]
            if HAS_ROT:
                C_0 = _pair_rot(C_0, ph_c, ph_s, T, N)
                C_1 = _pair_rot(C_1, ph_c, ph_s, T, N)
                if HAS_THETA_H:
                    C_0 = _pair_rot(C_0, ch_v, sh_v, T, N)
                    C_1 = _pair_rot(C_1, ch_v, sh_v, T, N)
            ceff.slice(0, T).store(C_0.to(gl.bfloat16))
            ceff.slice(T, T).store(C_1.to(gl.bfloat16))
            for r in gl.static_range(R):
                B_r = bst.slice(r * T, T).load(BLK_BN).to(gl.float32)
                if HAS_BETA:
                    B_r = B_r + gl.load(Beta_ptr + h * N * R + offs_n_bn * R + r)[None, :]
                if HAS_ROT:
                    B_r = _pair_rot(B_r, ph_c, ph_s, T, N)
                beff.slice(r * T, T).store(B_r.to(gl.bfloat16))
                if STORE_DOT:
                    td0 = gl.sum(C_0 * B_r, axis=1)
                    td1 = gl.sum(C_1 * B_r, axis=1)
                    dq = Dot_ptr + bo * s_dqb + h * s_dqh + (c * T + offs_t_bn) * s_dql + r * s_dr
                    dm = (offs_t_bn >= 0) & (gt_idx == 0)
                    gl.store(dq, td0, mask=dm)
                    gl.store(dq + s_do, td1, mask=dm)
        elif not STATE_ONLY:
            for o in gl.static_range(O):
                C_o = cst.slice(o * T, T).load(BLK_BN).to(gl.float32)
                if HAS_GAMMA:
                    C_o = C_o + gl.load(Gamma_ptr + h * N * O + offs_n_bn * O + o)[None, :]
                ceff.slice(o * T, T).store(C_o.to(gl.bfloat16))
                for r in gl.static_range(R):
                    B_r = bst.slice(r * T, T).load(BLK_BN).to(gl.float32)
                    if HAS_BETA:
                        B_r = B_r + gl.load(Beta_ptr + h * N * R + offs_n_bn * R + r)[None, :]
                    if o == 0:
                        beff.slice(r * T, T).store(B_r.to(gl.bfloat16))
                    if STORE_DOT:
                        td = gl.sum(C_o * B_r, axis=1)
                        gl.store(Dot_ptr + bo * s_dqb + h * s_dqh + (c * T + offs_t_bn) * s_dql + o * s_do + r * s_dr, td, mask=(offs_t_bn >= 0) & (gt_idx == 0))
        else:
            for r in gl.static_range(R):
                B_r = bst.slice(r * T, T).load(BLK_BN).to(gl.float32)
                if HAS_BETA:
                    B_r = B_r + gl.load(Beta_ptr + h * N * R + offs_n_bn * R + r)[None, :]
                if HAS_ROT:
                    B_r = _pair_rot(B_r, ph_c, ph_s, T, N)
                beff.slice(r * T, T).store(B_r.to(gl.bfloat16))
        D_x = vecs.index(0).load(gl.SliceLayout(0, BLK_X))
        w_x = vecs.index(1).load(gl.SliceLayout(0, BLK_X))
        seg_x = vecs.index(4).load(gl.SliceLayout(0, BLK_X))
        dec_agg = gl.where(seg_x == seg_end, gl.exp(a_val * (D_end - D_x)), 0.0) * w_x
        for r in gl.static_range(R):
            x_r = xs.index(slot * R + r).load(BLK_X).to(gl.float32)
            xw.index(r).store((x_r * dec_agg[None, :]).to(gl.bfloat16))
        fence_async_shared()
        gl.barrier()
        nslot = 1 - slot
        nbar = bars.index(nslot)
        more = c + 1 < c_hi
        mbarrier.expect(nbar, XBYTES + BBYTES + CBYTES, pred=more)
        for r in gl.static_range(R):
            tma.async_load(b_desc, [b * rb_b + (c + 1) * rb_c + bg * rb_h + r * rb_r, 0], nbar, bst.slice(r * T, T), pred=more)
            tma.async_load(x_desc, [x_row, r * s_xr + (c + 1) * T], nbar, xs.index(nslot * R + r), pred=more)
        if not STATE_ONLY:
            for o in gl.static_range(O):
                tma.async_load(c_desc, [b * rc_b + (c + 1) * rc_c + cg * rc_h + o * rc_o, 0], nbar, cst.slice(o * T, T), pred=more)
        g_acc = gl.zeros([N, GT], gl.float32, MMA_Y)
        for r in gl.static_range(R):
            g_acc = warpgroup_mma(beff.slice(r * T, T).permute((1, 0)), xw.index(r).permute((1, 0)), g_acc, use_acc=r > 0, is_async=True)
        if STATE_ONLY:
            g_acc = warpgroup_mma_wait(0, deps=[g_acc])
            S = alpha * S + g_acc
        else:
            y_acc = gl.zeros([TO, GT], gl.float32, MMA_Y)
            y_acc = warpgroup_mma(ceff, s_sm, y_acc, use_acc=False, is_async=True)
            if PIPE == 1:
                s_0 = gl.zeros([TO, T], gl.float32, MMA_T)
                s_0 = warpgroup_mma(ceff, beff.slice(0, T).permute((1, 0)), s_0, use_acc=False, is_async=True)
                if R == 2:
                    s_1 = gl.zeros([TO, T], gl.float32, MMA_T)
                    s_1 = warpgroup_mma(ceff, beff.slice(T, T).permute((1, 0)), s_1, use_acc=False, is_async=True)
                    g_acc, y_acc, s_0, s_1 = warpgroup_mma_wait(0, deps=[g_acc, y_acc, s_0, s_1])
                else:
                    g_acc, y_acc, s_0 = warpgroup_mma_wait(0, deps=[g_acc, y_acc, s_0])
            else:
                g_acc, y_acc = warpgroup_mma_wait(0, deps=[g_acc, y_acc])
            S = alpha * S + g_acc
            f_row = vecs.index(3).gather(tm_y, 0)
            y_acc = y_acc * f_row[:, None]
            D_row = vecs.index(0).gather(tm_t, 0)
            D_col = vecs.index(0).load(gl.SliceLayout(0, MMA_T))
            w_col = vecs.index(1).load(gl.SliceLayout(0, MMA_T))
            seg_row = vecs.index(4).gather(tm_t, 0)
            seg_col = vecs.index(4).load(gl.SliceLayout(0, MMA_T))
            causal = (tm_t[:, None] >= offs_k_t[None, :]) & (seg_row[:, None] == seg_col[None, :])
            dec = gl.where(causal, gl.exp(a_val * (D_row[:, None] - D_col[None, :])), 0.0) * w_col[None, :]
            if HAS_TRAP:
                own_col = vecs.index(2).load(gl.SliceLayout(0, MMA_T))
                dec = gl.where(tm_t[:, None] == offs_k_t[None, :], own_col[None, :], dec)
            if PIPE == 1:
                gl.static_assert(R <= 2, 'PIPE=1 batches at most two scores')
                sc = s_0 * dec
                p = gl.convert_layout(sc.to(gl.bfloat16), A_Y)
                y_acc = warpgroup_mma(p, xs.index(slot * R).permute((1, 0)), y_acc, is_async=True)
                if R == 2:
                    sc1 = s_1 * dec
                    p1 = gl.convert_layout(sc1.to(gl.bfloat16), A_Y)
                    y_acc = warpgroup_mma(p1, xs.index(slot * R + 1).permute((1, 0)), y_acc, is_async=True)
                y_acc = warpgroup_mma_wait(0, deps=[y_acc])
            else:
                for r in gl.static_range(R):
                    s_acc = gl.zeros([TO, T], gl.float32, MMA_T)
                    s_acc = warpgroup_mma(ceff, beff.slice(r * T, T).permute((1, 0)), s_acc, use_acc=False)
                    sc = s_acc * dec
                    p = gl.convert_layout(sc.to(gl.bfloat16), A_Y)
                    y_acc = warpgroup_mma(p, xs.index(slot * R + r).permute((1, 0)), y_acc, is_async=True)
                    y_acc = warpgroup_mma_wait(0, deps=[y_acc])
            s_sm.store(S.to(gl.bfloat16))
            if STORE_Y:
                y_out = gl.convert_layout(y_acc, BLK_Y)
                offs_m = gl.arange(0, TO, layout=gl.SliceLayout(1, BLK_Y))
                offs_g = gt_idx * GT + gl.arange(0, GT, layout=gl.SliceLayout(0, BLK_Y))
                gl.store(Y_ptr + bo * s_yb + h * s_yh + c * s_yc + (offs_m // T)[:, None] * s_yo + (offs_m % T)[:, None] * s_yt + offs_g[None, :] * s_yg, y_out.to(Y_ptr.dtype.element_ty))
    mbarrier.invalidate(bars.index(0))
    mbarrier.invalidate(bars.index(1))
    do_sf = (s_fss > 0) | (c_hi == NC)
    if do_sf:
        gl.store(SF_ptr + bo * s_fb + h * s_fh + seg * s_fss + offs_g_y[None, :] * s_fg + offs_n_y[:, None] * s_fn, S)
_LAST_ROUTE: dict = {'used': False, 'why': 'never consulted'}

def last_route() -> dict:
    return dict(_LAST_ROUTE)

def _rows(t, n, strides_elems):
    out = []
    for s in strides_elems:
        if s == 1:
            out.append(0)
            continue
        if s % n:
            return None
        out.append(s // n)
    return out
_FAILED: dict = {}

def ptxas_refusal():
    from .fused_fwd_kernel import ro_stack_wgmma_fault, sm90_ptxas_version
    if ro_stack_wgmma_fault((9, 0), True):
        return f'ptxas < 12.9.86 (SM90_WGMMA chained bf16 wgmma miscompile; found {sm90_ptxas_version()})'
    return None

def failed_refusal(key):
    why = _FAILED.get(key)
    return None if why is None else f'gluon launch failed earlier: {why}'

def note_failure(key, exc):
    why = f"{type(exc).__name__}: {(str(exc).splitlines()[0][:300] if str(exc) else '')}"
    if key not in _FAILED:
        import warnings
        warnings.warn(f'GLUON_SM90 {key}: launch failed, falling back to the Triton route for the rest of the process ({why})', RuntimeWarning, stacklevel=3)
    _FAILED[key] = why
    return f'gluon launch failed: {why}'

def bc_stride_refusal(*tensors, N):
    for t in tensors:
        unit = 0
        for st, d in zip(t.stride(), t.shape):
            if d == 1 or st == 0:
                continue
            if st == 1:
                unit += 1
            elif st % N:
                return 'B/C strides must be whole rows of N (TMA row bases)'
        if unit != 1:
            return 'B/C must have exactly one unit-stride axis (the state axis)'
    return None

def x_refusal(x, R, T, nc):
    s = x.stride()
    Lp = x.shape[4]
    if x.shape[3] != R:
        return f'x R axis {x.shape[3]} != R {R}'
    if s[4] != 1:
        return 'x must have L contiguous'
    if Lp < nc * T:
        return 'x L axis shorter than nc * T'
    if R > 1 and s[3] < Lp:
        return 'x R slices overlap (R stride < L)'
    if (R - 1) * s[3] + Lp > s[2]:
        return 'x R slices must lie inside one (b, h, g) row (TMA width)'
    return None

def _row_view(t, n):
    maxoff = sum(((d - 1) * s for d, s in zip(t.shape, t.stride()) if d > 0))
    rows = maxoff // n + 1
    return torch.as_strided(t, (rows, n), (n, 1))

def rot_refusal(*, has_rot, has_theta_h, has_rot_bias, pair_half, bias_pre_rot, has_beta, has_gamma, phase_cumsum, theta=None):
    if not (has_rot or has_theta_h or has_rot_bias or pair_half):
        return None
    if os.environ.get('SSD_MIMO_GLUON90_ROT', '1') == '0':
        return 'rotating arm disabled (SSD_MIMO_GLUON90_ROT=0)'
    if has_rot_bias:
        return 'rot_bias (the retired write-side angle) not served'
    if pair_half:
        return 'half pair layout not served (interleaved only)'
    if has_theta_h and (not has_rot):
        return 'theta_h without the per-token phase not served'
    if (has_beta or has_gamma) and (not bias_pre_rot):
        return 'rotation with the post-rotation bias order not served'
    if phase_cumsum:
        return 'phase offsets (in-kernel phase cumsum) not served'
    if theta is not None and (theta.dim() not in (4, 5) or theta.dtype not in (torch.float32, torch.bfloat16)):
        return 'phase must be fp32/bf16 [B, nc, T, (H,) N/2]'
    return None

def refusal(*, device, T, N, G_SIZE, GT, R, O, mma_io, io_dtype, has_theta_h, has_rot, has_rot_bias, pair_half, has_gate, has_dskip, nseg, idx64, Btil, Ctil, x, s_bt, s_bn, s_ct, s_cn, state_only, bias_pre_rot, has_beta=False, has_gamma=False, phase_cumsum=False, theta=None):
    if TensorDescriptor is None:
        return 'no gluon hopper TensorDescriptor in this Triton'
    if torch.cuda.get_device_capability(device) != (9, 0):
        return 'not sm_90'
    if not mma_io or io_dtype != torch.bfloat16:
        return "needs bf16 IO with 16-bit MMA operands (precision='io')"
    why = ptxas_refusal()
    if why:
        return why
    why = rot_refusal(has_rot=has_rot, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pair_half=pair_half, bias_pre_rot=bias_pre_rot, has_beta=has_beta, has_gamma=has_gamma, phase_cumsum=phase_cumsum, theta=theta)
    if why:
        return why
    if has_gate or has_dskip:
        return 'no in-kernel gate / D-skip'
    if T != 32 or GT != 32 or G_SIZE % 32:
        return f'tile envelope T=32, GT=32 (got T={T}, GT={GT}, G={G_SIZE})'
    if R != 2 or (not state_only and O != 2):
        return f'R = O = 2 only (got R={R}, O={O})'
    if N % 64 or N > 256:
        return f'N must be a multiple of 64 and <= 256 (got {N})'
    if nseg != 1:
        return 'segmented forward (phase A/B) not served'
    if s_bn != 1 or s_bt != N or (not state_only and (s_cn != 1 or s_ct != N)):
        return 'B/C tiles must be [T, N] row-contiguous'
    why = bc_stride_refusal(Btil, *(() if state_only else (Ctil,)), N=N)
    if why:
        return why
    if Btil.dtype != torch.bfloat16 or (not state_only and Ctil.dtype != torch.bfloat16):
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
    for t in (Btil, x) + (() if state_only else (Ctil,)):
        if t.data_ptr() % 16:
            return 'TMA base not 16-byte aligned'
    return None

def launch(*, Ctil, Btil, x, delta, a_log, x0_c, x0_st, x0_seg_stride, beta_c, gamma_c, has_x0, has_beta, has_gamma, Y, S_all, S_final, y_strides, sall_strides, sf_strides, dw_kw, trap_kw, H, T, N, G_SIZE, GT, R, O, hpg_b, hpg_c, nc, nc_seg, s_fss, grid, store_sall, state_only, store_y, dcs_ink, b_strides, c_strides, theta=None, th_strides=None, theta_h=None, idx64=False):
    s_bb, s_bc, s_bh, s_bt, s_bn, s_br = b_strides
    s_cb, s_cc, s_ch, s_ct, s_cn, s_co = c_strides
    lay_bn = gl.NVMMASharedLayout.get_default_for([T, N], gl.bfloat16)
    lay_x = gl.NVMMASharedLayout.get_default_for([GT, T], gl.bfloat16)
    rb = _rows(Btil, N, (s_bb, s_bc, s_bh, s_br))
    if rb is None:
        raise ValueError('B strides are not whole rows of N')
    bv = _row_view(Btil, N)
    b_desc = TensorDescriptor(bv, list(bv.shape), list(bv.stride()), [T, N], lay_bn)
    if state_only:
        c_desc = b_desc
        rc = [0, 0, 0, 0]
    else:
        rc = _rows(Ctil, N, (s_cb, s_cc, s_ch, s_co))
        if rc is None:
            raise ValueError('C strides are not whole rows of N')
        cv = _row_view(Ctil, N)
        c_desc = TensorDescriptor(cv, list(cv.shape), list(cv.stride()), [T, N], lay_bn)
    Bsz, H_, G_, R_, Lp = x.shape
    xs_g = x.stride(2)
    xv = torch.as_strided(x, (Bsz * H_ * G_, xs_g), (xs_g, 1))
    x_desc = TensorDescriptor(xv, [Bsz * H_ * G_, xs_g], [xs_g, 1], [GT, T], lay_x)
    dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    DW = dw_kw['DW_ptr']
    s_wb, s_wh, s_wl = (dw_kw['s_wb'], dw_kw['s_wh'], dw_kw['s_wl'])
    has_trap = bool(trap_kw.get('HAS_TRAP', False))
    store_dot = bool(trap_kw.get('STORE_DOT', False))
    Lam = trap_kw.get('Lam_ptr', dummy)
    Dot = trap_kw.get('Dot_ptr', dummy)
    y_tmaj = y_strides[4] == 1
    has_rot = theta is not None
    has_thh = theta_h is not None and (not state_only)
    if theta_h is not None and (not has_rot):
        raise ValueError('theta_h needs the per-token phase on this route')
    _gl90_chunk_fwd_kernel[grid](b_desc, c_desc, x_desc, delta, a_log, x0_c if has_x0 else dummy, beta_c if has_beta else dummy, gamma_c if has_gamma else dummy, Y, S_all, S_final, DW, Lam, Dot, H, *rb, *rc, x.stride(3), *delta.stride(), *y_strides, *sall_strides, *sf_strides, *x0_st, s_wb, s_wh, s_wl, trap_kw.get('s_dqb', 0), trap_kw.get('s_dqh', 0), trap_kw.get('s_dql', 0), trap_kw.get('s_do', 0), trap_kw.get('s_dr', 0), nc, nc_seg, x0_seg_stride, s_fss, theta if has_rot else dummy, theta_h if has_thh else dummy, *(th_strides if has_rot else (0, 0, 0, 0, 0)), T=T, N=N, G_SIZE=G_SIZE, GT=GT, R=R, O=O, HPG_B=hpg_b, HPG_C=hpg_c, HAS_X0=has_x0, HAS_BETA=has_beta, HAS_GAMMA=has_gamma, HAS_TRAP=has_trap, STORE_DOT=store_dot, STORE_Y=store_y, STORE_SALL=store_sall, STATE_ONLY=state_only, DCS_INK=bool(dcs_ink), Y_TMAJ=y_tmaj, PIPE=int(os.environ.get('SSD_MIMO_GLUON90_PIPE', '1')), PREP=1 if has_rot else int(os.environ.get('SSD_MIMO_GLUON90_PREP', '1')), HAS_ROT=has_rot, HAS_THETA_H=has_thh, ROT_PF=int(os.environ.get('SSD_MIMO_GLUON90_ROT_PF', '1')), IDX64=bool(idx64), num_warps=4)
    _LAST_ROUTE['rot'] = bool(has_rot)
    _LAST_ROUTE['theta_h'] = bool(has_thh)
