
import os
import torch
import triton
from triton.runtime.errors import OutOfResources as _OOR
from .fused_fwd_kernel import occ_bucket, len_bucket
import triton.language as tl
from .autotune_policy import siso_autotune_tier
from .common import pdot
from .fused_fwd_kernel import _dcs_seg, _same_seg, _env, _needs_i64, _pair_bcast_2d, _pair_swap_1d, _pair_swap_2d, _mmav3_eight_warp_ok, _sm_count
from .common import _cos_approx, _sin_approx
from .rotation_active import rotation_ka, rotate_prefix

def _siso_configs():
    from .fused_fwd_kernel import autotune_off
    mode = siso_autotune_tier(_env('SISO_AUTOTUNE', '0' if autotune_off() else '1'))
    _light = lambda: [triton.Config({}, num_stages=s, num_warps=w) for s in (1, 2, 3) for w in (4, 8)]
    _full = lambda: [triton.Config({}, num_stages=s, num_warps=w, maxnreg=r) for s in (1, 2, 3) for w in (2, 4, 8) for r in (None, 128, 256)]
    from .calibration_ssd import grid
    return grid('siso_fwd', mode, full=_full, light=_light, generic_pin=lambda: [triton.Config({}, num_stages=2, num_warps=8)])

def _prune_configs(configs, named_args, **kwargs):
    gt = kwargs.get('G_SIZE', named_args.get('G_SIZE', 64))
    try:
        cc = torch.cuda.get_device_capability()
    except Exception:
        cc = (9, 0)
    if cc == (9, 0) and (not _mmav3_eight_warp_ok(gt)):
        configs = [c for c in configs if c.num_warps < 8]
    from .calibration_ssd import prune
    return prune(configs, 'siso_fwd')
_SISO_KEY = ['T', 'N', 'G_SIZE', 'STORED', 'HAS_TRAP', 'HAS_DSKIP', 'HAS_BETA', 'HAS_GAMMA', 'HAS_THETA_H', 'HAS_ROT_BIAS', 'BIAS_PRE_ROT', 'STORE_SALL', 'STATE_ONLY', 'HAS_X0', 'HAS_DW', 'IO_DT', 'PREC', 'INK_ROT', 'STORE_DOT_RAW', 'ROT_KA', 'OCC', 'LB']

@triton.jit
def _to_opd(t, IO_DT: tl.constexpr):
    if IO_DT == 'bf16':
        return t.to(tl.bfloat16)
    elif IO_DT == 'fp16':
        return t.to(tl.float16)
    else:
        return t.to(tl.float32)

@triton.autotune(configs=_siso_configs(), key=_SISO_KEY, prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _siso_fwd_kernel(C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, Theta_ptr, ThetaH_ptr, RotBias_ptr, Beta_ptr, Gamma_ptr, DW_ptr, Lam_ptr, X0_ptr, Dsk_ptr, Y_ptr, SAll_ptr, SF_ptr, Bs_ptr, Cs_ptr, Dot_ptr, Scale_ptr, Gam_ptr, DotR_ptr, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_thb, s_thc, s_tht, s_thh, s_thk, s_wb, s_wh, s_wl, s_x0b, s_x0h, s_x0g, s_x0n, s_yb, s_yh, s_yc, s_yt, s_yg, s_sb, s_sh, s_sc, s_sn, s_sg, s_fb, s_fh, s_fg, s_fn, s_mb, s_mc, s_mt, s_mh, s_mn, NC, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr, STORED: tl.constexpr, HAS_ROT: tl.constexpr, HAS_THETA_H: tl.constexpr, HAS_ROT_BIAS: tl.constexpr, HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr, BIAS_PRE_ROT: tl.constexpr, HAS_DW: tl.constexpr, HAS_TRAP: tl.constexpr, HAS_X0: tl.constexpr, HAS_DSKIP: tl.constexpr, STORE_SALL: tl.constexpr, STORE_BWD: tl.constexpr, ROT_KA: tl.constexpr, INK_ROT: tl.constexpr, STORE_DOT_RAW: tl.constexpr, IDX64: tl.constexpr, OCC=0, LB=0):
    tl.static_assert(not STORED or HAS_ROT, 'the stored-map schedule exists for the rotation')
    tl.static_assert(not HAS_TRAP or HAS_DW, 'the two-tap trapezoid IS the write scale')
    tl.static_assert(not INK_ROT or (HAS_ROT and (not STORED)), 'SISOFUSE c is the rotation WITHOUT phase 1')
    tl.static_assert(not STORE_DOT_RAW or (STORED and HAS_TRAP), 'the raw dot exists because the STORED one is scaled')
    b = tl.program_id(0)
    h = tl.program_id(1)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.arange(0, G_SIZE)
    offs_k = tl.arange(0, ROT_KA)
    pair_idx = offs_n // 2
    rot_sign = tl.where(offs_n % 2 == 0, 1.0, -1.0)
    if HAS_THETA_H:
        thh = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
        ch_full = _cos_approx(thh)
        sh_full = rot_sign * _sin_approx(thh)
    if HAS_ROT_BIAS:
        rb = tl.load(RotBias_ptr + h * (N // 2) + pair_idx)
        crb = _cos_approx(rb)
        srb = rot_sign * _sin_approx(rb)
    if HAS_BETA:
        beta_v = tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)
    if HAS_GAMMA:
        gam_v = tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)
        if HAS_THETA_H and (not BIAS_PRE_ROT):
            gam_v = ch_full * gam_v + sh_full * _pair_swap_1d(gam_v, N)
    if STORED:
        for c in range(NC):
            tbase = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
            if HAS_TRAP:
                dt_t = tl.load(DW_ptr + tbase)
                lam_t = tl.load(Lam_ptr + tbase)
                idx_n = c * T + offs_t + 1
                m_n = idx_n < NC * T
                dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                g_t = lam_t * dt_t
                w_t = g_t + dt_n * (1.0 - lam_n)
            elif HAS_DW:
                w_t = tl.load(DW_ptr + tbase)
            th = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk).to(tl.float32)
            if ROT_KA < N // 2:
                cs_t = _cos_approx(th)
                sn_t = _sin_approx(th)
            else:
                cs_t = _pair_bcast_2d(_cos_approx(th), T, N)
                sn_t = rot_sign[None, :] * _pair_bcast_2d(_sin_approx(th), T, N)
            Bt = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
            Ct = tl.load(C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
            if BIAS_PRE_ROT:
                if HAS_BETA:
                    Bt = Bt + beta_v[None, :]
                if HAS_GAMMA:
                    Ct = Ct + gam_v[None, :]
            if ROT_KA < N // 2:
                Bt = rotate_prefix(Bt, cs_t, sn_t, T, N, ROT_KA)
            else:
                Bt = cs_t * Bt + sn_t * _pair_swap_2d(Bt, T, N)
            if ROT_KA < N // 2:
                Ct = rotate_prefix(Ct, cs_t, sn_t, T, N, ROT_KA)
            else:
                Ct = cs_t * Ct + sn_t * _pair_swap_2d(Ct, T, N)
            if HAS_ROT_BIAS:
                Bt = crb[None, :] * Bt + srb[None, :] * _pair_swap_2d(Bt, T, N)
            if HAS_THETA_H:
                Ct = ch_full[None, :] * Ct + sh_full[None, :] * _pair_swap_2d(Ct, T, N)
            if not BIAS_PRE_ROT:
                if HAS_BETA:
                    Bt = Bt + beta_v[None, :]
                if HAS_GAMMA:
                    Ct = Ct + gam_v[None, :]
            if HAS_TRAP:
                qk_raw = tl.sum(Bt * Ct, axis=1)
                tl.store(Dot_ptr + tbase, qk_raw * g_t)
                if STORE_DOT_RAW:
                    tl.store(DotR_ptr + tbase, qk_raw)
                if STORE_BWD:
                    tl.store(Scale_ptr + tbase, w_t)
                    tl.store(Gam_ptr + tbase, g_t)
            if HAS_DW:
                Bt = Bt * w_t[:, None]
            mo = b * s_mb + c * s_mc + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
            tl.store(Bs_ptr + mo, Bt.to(Bs_ptr.dtype.element_ty))
            tl.store(Cs_ptr + mo, Ct.to(Cs_ptr.dtype.element_ty))
    if HAS_X0:
        S = tl.load(X0_ptr + b * s_x0b + h * s_x0h + offs_g[None, :] * s_x0g + offs_n[:, None] * s_x0n).to(tl.float32)
    else:
        S = tl.zeros([N, G_SIZE], dtype=tl.float32)
    causal = offs_t[:, None] >= offs_t[None, :]
    strict = offs_t[:, None] > offs_t[None, :]
    neg_inf = float('-inf')
    if HAS_DSKIP:
        dv = tl.load(Dsk_ptr + h * G_SIZE + offs_g)
    for c in range(NC):
        if STORE_SALL:
            tl.store(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg, S.to(SAll_ptr.dtype.element_ty))
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
        seg_end = tl.max(seg_h, axis=0)
        D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end - D_h)), 0.0)
        dec_mask = tl.exp(tl.where((strict if STORED and HAS_TRAP else causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
        tbase = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
        if not STORED:
            if HAS_TRAP:
                dt_t = tl.load(DW_ptr + tbase)
                lam_t = tl.load(Lam_ptr + tbase)
                idx_n = c * T + offs_t + 1
                m_n = idx_n < NC * T
                dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                trap_nxt = dt_n * (1.0 - lam_n)
                w_t = lam_t * dt_t + trap_nxt
            elif HAS_DW:
                w_t = tl.load(DW_ptr + tbase)
            if HAS_DW:
                dec_agg = dec_agg * w_t
                dec_mask = dec_mask * w_t[None, :]
            if HAS_TRAP:
                dec_mask = tl.where(offs_t[:, None] == offs_t[None, :], (lam_t * dt_t)[None, :], dec_mask)
        if STORED:
            mo = b * s_mb + c * s_mc + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
            Bop = _to_opd(tl.load(Bs_ptr + mo), IO_DT)
            Cop = _to_opd(tl.load(Cs_ptr + mo), IO_DT)
        else:
            Bt = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
            Ct = tl.load(C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
            if INK_ROT:
                th = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk).to(tl.float32)
                if ROT_KA < N // 2:
                    cs_t = _cos_approx(th)
                    sn_t = _sin_approx(th)
                else:
                    cs_t = _pair_bcast_2d(_cos_approx(th), T, N)
                    sn_t = rot_sign[None, :] * _pair_bcast_2d(_sin_approx(th), T, N)
                if BIAS_PRE_ROT:
                    if HAS_BETA:
                        Bt = Bt + beta_v[None, :]
                    if HAS_GAMMA:
                        Ct = Ct + gam_v[None, :]
                if ROT_KA < N // 2:
                    Bt = rotate_prefix(Bt, cs_t, sn_t, T, N, ROT_KA)
                else:
                    Bt = cs_t * Bt + sn_t * _pair_swap_2d(Bt, T, N)
                if ROT_KA < N // 2:
                    Ct = rotate_prefix(Ct, cs_t, sn_t, T, N, ROT_KA)
                else:
                    Ct = cs_t * Ct + sn_t * _pair_swap_2d(Ct, T, N)
                if HAS_ROT_BIAS:
                    Bt = crb[None, :] * Bt + srb[None, :] * _pair_swap_2d(Bt, T, N)
                if HAS_THETA_H:
                    Ct = ch_full[None, :] * Ct + sh_full[None, :] * _pair_swap_2d(Ct, T, N)
                if not BIAS_PRE_ROT:
                    if HAS_BETA:
                        Bt = Bt + beta_v[None, :]
                    if HAS_GAMMA:
                        Ct = Ct + gam_v[None, :]
            else:
                if HAS_BETA:
                    Bt = Bt + beta_v[None, :]
                if HAS_GAMMA:
                    Ct = Ct + gam_v[None, :]
            Bop = _to_opd(Bt, IO_DT)
            Cop = _to_opd(Ct, IO_DT)
        x_r = tl.load(X_ptr + b * s_xb + h * s_xh + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
        S_in = S
        G = pdot(tl.trans(Bop), _to_opd(x_r * dec_agg[:, None], IO_DT), input_precision=PREC)
        y = f[:, None] * pdot(Cop, _to_opd(S_in, IO_DT), input_precision=PREC)
        Score = pdot(Cop, tl.trans(Bop), input_precision=PREC) * dec_mask
        if HAS_TRAP and (not STORED):
            if STORE_BWD:
                dott = tl.sum(Bt * Ct, axis=1)
                tl.store(Dot_ptr + tbase, dott)
        y += pdot(_to_opd(Score, IO_DT), _to_opd(x_r, IO_DT), input_precision=PREC)
        if HAS_TRAP and STORED:
            y += tl.load(Dot_ptr + tbase)[:, None] * x_r
        if HAS_DSKIP:
            y += x_r * dv[None, :]
        tl.store(Y_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_g[None, :] * s_yg, y.to(Y_ptr.dtype.element_ty))
        S = alpha * S_in + G
    tl.store(SF_ptr + b * s_fb + h * s_fh + offs_g[None, :] * s_fg + offs_n[:, None] * s_fn, S)
_DT = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}

def _io_tag(dtype) -> str:
    if _env('SISO_OPD', 'io') == 'fp32':
        return 'fp32'
    if dtype == torch.bfloat16:
        return 'bf16'
    if dtype == torch.float16:
        return 'fp16'
    return 'fp32'

def siso_native_y(Bsz, H, G_SIZE, nc, T, device, dtype, token_major=False):
    if token_major:
        buf = torch.empty(Bsz, nc * T, H, 1, G_SIZE, device=device, dtype=dtype)
        return buf.view(Bsz, nc, T, H, 1, G_SIZE).permute(0, 3, 1, 2, 4, 5)
    buf = torch.empty(Bsz, H, 1, G_SIZE, nc * T, device=device, dtype=dtype)
    return buf.view(Bsz, H, 1, G_SIZE, nc, T).permute(0, 1, 4, 5, 2, 3)

def siso_chunk_fwd(Ctil, Btil, x_raw, delta_cumsum, a_log, *, theta=None, theta_h=None, rot_bias=None, beta=None, gamma=None, bias_pre_rot=False, dw=None, trap_lam=None, x0=None, d_skip=None, precision='tf32', store_sall=True, needs_bwd=True, y_token_major=False, fuse=None):
    from .fused_fwd_kernel import dot_precision
    precision = dot_precision(precision)
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_raw.shape[2]
    N = Btil.shape[3]
    _ka_ok = x_raw.dtype == torch.bfloat16 and theta_h is None and (rot_bias is None) and (fuse is None)
    rot_ka = rotation_ka(N, 'fwd', _ka_ok)
    rot_ka_bwd = rotation_ka(N, 'bwd', _ka_ok)
    if rot_ka < N // 2 and x_raw.dtype != torch.bfloat16:
        raise ValueError('SISO_ROT_KA prototype requires bf16 IO')
    if rot_ka < N // 2 and (theta_h is not None or rot_bias is not None or fuse is not None):
        raise ValueError('SISO_ROT_KA prototype requires no static rotations or SISOFUSE')
    device = x_raw.device
    f32 = torch.float32
    L_pad = nc * T
    has_rot = theta is not None
    has_trap = trap_lam is not None
    has_dw = dw is not None
    fuse = fuse if fuse in ('b', 'c') else None
    ink_rot = bool(has_rot and fuse == 'c')
    stored = has_rot and (not ink_rot)
    store_bwd = bool(needs_bwd and fuse != 'b')
    store_dot_raw = bool(stored and has_trap and needs_bwd and (fuse == 'b'))
    io_tag = _io_tag(x_raw.dtype)
    from . import calibration_ssd as _pins
    _pins.note_call('siso_fwd', T=T, N=N, io=io_tag)
    _pins.activate(has_rot, Bsz, H, T, io=io_tag)
    dummy = torch.empty(1, device=device, dtype=f32)
    th_c = theta.float().contiguous() if has_rot else dummy
    if not has_rot:
        th_st = [0, 0, 0, 0, 0]
    elif th_c.dim() == 5:
        st = th_c.stride()
        th_st = [st[0], st[1], st[2], st[3], st[4]]
    else:
        st = th_c.stride()
        th_st = [st[0], st[1], st[2], 0, st[3]]
    thh_c = theta_h.float().contiguous() if theta_h is not None else dummy
    rb_c = rot_bias.float().contiguous() if rot_bias is not None else dummy
    beta_c = beta.float().contiguous().reshape(H, N) if beta is not None else dummy
    gam_c = gamma.float().contiguous().reshape(H, N) if gamma is not None else dummy
    dsk_c = d_skip.float().contiguous().reshape(H, G_SIZE) if d_skip is not None else dummy
    x0_c = x0.float().contiguous() if x0 is not None else dummy
    x0_st = list(x0_c.stride()) if x0 is not None else [0, 0, 0, 0]
    map_dt = _DT.get(_env('SISO_MAP_DTYPE', ''), f32 if x_raw.dtype == torch.float64 else x_raw.dtype)
    if stored:
        from .siso_route import siso_map_head_major
        if siso_map_head_major():
            Bs = torch.empty(Bsz, H, nc, T, N, device=device, dtype=map_dt)
            Cs = torch.empty(Bsz, H, nc, T, N, device=device, dtype=map_dt)
            st = Bs.stride()
            m_st = [st[0], st[2], st[3], st[1], st[4]]
        else:
            Bs = torch.empty(Bsz, nc, T, H, N, device=device, dtype=map_dt)
            Cs = torch.empty(Bsz, nc, T, H, N, device=device, dtype=map_dt)
            m_st = list(Bs.stride())
    else:
        Bs = Cs = dummy
        m_st = [0, 0, 0, 0, 0]
    if has_trap:
        dot = torch.empty(Bsz, H, L_pad, device=device, dtype=f32)
        if store_bwd and (not ink_rot):
            scale = torch.empty_like(dot)
            gam_s = torch.empty_like(dot)
        else:
            scale = gam_s = dummy
        dot_raw = torch.empty_like(dot) if store_dot_raw else dummy
    else:
        dot = scale = gam_s = dot_raw = dummy
    Y = siso_native_y(Bsz, H, G_SIZE, nc, T, device, x_raw.dtype, y_token_major)
    if store_sall:
        sall_dt = f32 if os.environ.get('UNIFIED_SSD_SALL_FP32') == '1' or precision == 'ieee' else x_raw.dtype
        S_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device, dtype=sall_dt)
    else:
        S_all = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
    S_final = torch.empty(Bsz, H, G_SIZE, N, device=device, dtype=f32)
    dw_c = dw if has_dw else dummy
    lam_c = trap_lam if has_trap else dummy
    w_st = dw_c.stride() if has_dw else (0, 0, 0)
    from .siso_route import siso_xtm
    xtm = siso_xtm() != '0'
    if xtm:
        x_k = x_raw[:, :, :, 0, :].transpose(2, 3).contiguous()
        x_st = (x_k.stride(0), x_k.stride(1), x_k.stride(3), x_k.stride(2))
    else:
        x_k = x_raw
        x_st = (x_raw.stride(0), x_raw.stride(1), x_raw.stride(2), x_raw.stride(4))
    idx64 = _needs_i64(x_k, Y, S_all, S_final, Btil, Ctil, x0_c, dw_c, th_c, thh_c, beta_c, gam_c, rb_c, lam_c, Bs if stored else None, Cs if stored else None, dot if has_trap else None)
    grid = (Bsz, H)
    for _attempt in (0, 1):
        try:
            _siso_fwd_kernel[grid](Ctil, Btil, x_k, delta_cumsum, a_log, th_c, thh_c, rb_c, beta_c, gam_c, dw_c, lam_c, x0_c, dsk_c, Y, S_all, S_final, Bs, Cs, dot, scale, gam_s, dot_raw, *Ctil.stride()[:4], *Btil.stride()[:4], *x_st, *delta_cumsum.stride(), *th_st, *w_st, *x0_st, *Y.stride()[:4], Y.stride(5), *S_all.stride(), *S_final.stride(), *m_st, nc, T=T, N=N, G_SIZE=G_SIZE, PREC=precision, IO_DT=io_tag, STORED=stored, HAS_ROT=has_rot, HAS_THETA_H=theta_h is not None, HAS_ROT_BIAS=rot_bias is not None, HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None, BIAS_PRE_ROT=bool(bias_pre_rot), HAS_DW=has_dw, HAS_TRAP=has_trap, HAS_X0=x0 is not None, HAS_DSKIP=d_skip is not None, STORE_SALL=store_sall, STORE_BWD=store_bwd, ROT_KA=rot_ka, INK_ROT=ink_rot, STORE_DOT_RAW=store_dot_raw, IDX64=idx64, OCC=occ_bucket(Bsz * H, Ctil.device), LB=len_bucket(nc * T))
            break
        except _OOR:
            if _attempt or not _pins.smem_demote(T):
                raise
            _pins.activate(has_rot, Bsz, H, T, io=io_tag)
    aux = dict(rot_ka=rot_ka, rot_ka_bwd=rot_ka_bwd, Bs=Bs if stored else None, Cs=Cs if stored else None, dot=dot if has_trap else None, dot_raw=dot_raw if store_dot_raw else None, scale=scale if has_trap and store_bwd and (not ink_rot) else None, gam=gam_s if has_trap and store_bwd and (not ink_rot) else None, xt=x_k if xtm and needs_bwd else None, m_st=m_st if stored else None, stored=stored, ink_rot=ink_rot, fuse=fuse, io_tag=io_tag)
    return (Y, S_all, S_final, aux)
