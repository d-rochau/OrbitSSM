
import os
import torch
import triton
import triton.language as tl
from .autotune_policy import autotune_tier
from .common import pdot
from triton.runtime.errors import OutOfResources
from .common import nan_debug_fill
from .common import ACC_SEM, DTH_OPROUND, _cos_approx, _sin_approx, _opv
from .common import _span_sum, _excl_prefix
from .fused_fwd_kernel import occ_bucket as _occ_bucket
from .fused_fwd_kernel import _env, _pair_swap_2d, _pair_swap_1d, _pair_bcast_2d, _dcs_tile, _dcs_end, _dcs_seg, _same_seg, autotune_off, dot_precision, mma_operand_dtype, SM90_PTXAS_FIXED, sm90_ptxas_version
_AUTOTUNE_TIER = autotune_tier(_env('AUTOTUNE', 'light'))
_AUTOTUNE_OFF = autotune_off() or _AUTOTUNE_TIER == 'pinned'
_BWD_PINNED = (8, 1, 0)
_BWD_WARPS = tuple((int(v) for v in _env('BWD_WARPS', {'pinned': '8', 'light': '1,2,4,8', 'full': '1,2,4,8'}[_AUTOTUNE_TIER]).split(',')))
_BWD_STAGES = tuple((int(v) for v in _env('BWD_STAGES', {'pinned': '1', 'light': '1', 'full': '1,3'}[_AUTOTUNE_TIER]).split(',')))
_BWD_MAXNREG = tuple((int(v) for v in _env('BWD_MAXNREG', {'pinned': '0', 'light': '0', 'full': '0,256'}[_AUTOTUNE_TIER]).split(',')))
_CP_LAST_ROUTE: dict = {}

def _bwd_cfg(w, st, mnr):
    kw = dict(num_warps=w, num_stages=st)
    if mnr:
        kw['maxnreg'] = mnr
    return triton.Config({}, **kw)
_BWD_CFGS = [_bwd_cfg(w, st, m) for w in _BWD_WARPS for st in _BWD_STAGES for m in _BWD_MAXNREG]
_TIER_AXES = {'pinned': ('8', '1', '0'), 'light': ('1,2,4,8', '1', '0'), 'full': ('1,2,4,8', '1,3', '0,256')}
_CALIB_TIER = 'pinned' if _AUTOTUNE_OFF else _AUTOTUNE_TIER

def _bwd_grid(tier):
    w, st, m = _TIER_AXES[tier]
    return [_bwd_cfg(a, b, c) for a in (int(v) for v in _env('BWD_WARPS', w).split(',')) for b in (int(v) for v in _env('BWD_STAGES', st).split(',')) for c in (int(v) for v in _env('BWD_MAXNREG', m).split(','))]

def _cprune(kernel):

    def prune(configs, named_args, **kwargs):
        from .calibration_fused import prune as _p
        return _p(configs, kernel)
    return prune

def _calib_grid(kernel, builder):
    from .calibration_fused import grid
    return grid(kernel, _CALIB_TIER, full=lambda: builder('full'), light=lambda: builder('light'), generic_pin=lambda: builder('pinned'))

def rot_io_bwd_t64_fault(capability, mma_io: bool, has_theta_h: bool, R: int, O: int, ptxas_version='resolve') -> bool:
    if not (tuple(capability) == (9, 0) and bool(has_theta_h) and bool(mma_io) and (R == 1) and (O == 1)):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == 'resolve':
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED

def pairhalf_io_t64_fault(capability, mma_io: bool, pair_half: bool, ptxas_version='resolve') -> bool:
    if int(tuple(capability)[0]) != 9 or not bool(mma_io) or (not bool(pair_half)):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == 'resolve':
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED

def pairhalf_r1_t64_fault(capability, tensor_cores: bool, pair_half: bool, R: int, O: int, ptxas_version='resolve') -> bool:
    if int(tuple(capability)[0]) != 9 or not bool(tensor_cores) or (not bool(pair_half)) or (R != 1) or (O != 1):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == 'resolve':
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED

@triton.jit
def _ph_read_consistent(Ch1, Ch2, ch, sh, Ct1, Ct2, MMA_DT: tl.constexpr, MMA_IO: tl.constexpr, HAS_THETA_H: tl.constexpr, PREC: tl.constexpr='ieee'):
    if (MMA_IO or PREC == 'tf32') and DTH_OPROUND:
        if HAS_THETA_H:
            R1 = _opv(Ch1, MMA_DT, MMA_IO, PREC)
            R2 = _opv(Ch2, MMA_DT, MMA_IO, PREC)
            Ct1 = ch[None, :] * R1 - sh[None, :] * R2
            Ct2 = ch[None, :] * R2 + sh[None, :] * R1
        else:
            Ct1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
            Ct2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
    return (Ct1, Ct2)

def ka_wave_bucket(n_ctas: int, device) -> int:
    global _KA_LAST_WB
    _KA_LAST_WB = _occ_bucket(n_ctas, device)
    return _KA_LAST_WB
_KA_LAST_WB = None

def _ka_prune_inner(configs, named_args, **kwargs):
    w = _env('KA_WARPS', '')
    if not w:
        from .. import calibration as _cal
        if str(_cal.active_group('ssd_mimo_fused') or '').startswith('~smem'):
            return configs
        if not int(named_args.get('MMA_IO', kwargs.get('MMA_IO', 1)) or 0):
            return configs
        wb = named_args.get('KA_WB', kwargs.get('KA_WB', None))
        try:
            from .chunkpar import cp_calibration_key
            from .smem import chunk_cache_lookup
            v = chunk_cache_lookup(cp_calibration_key(torch.cuda.current_device(), 'ka_serial') + f'|warps|wb{wb}')
            w = str(v) if v else ''
        except Exception:
            w = ''
    if w:
        keep = [c for c in configs if c.num_warps == int(w)]
        if keep:
            return keep
    return configs

def _ka_prune(configs, named_args, **kwargs):
    from .calibration_fused import prune as _p
    return _p(_ka_prune_inner(configs, named_args, **kwargs), 'fused_ka')
_KA_STAGES = tuple((int(v) for v in _env('KA_STAGES', '1' if _AUTOTUNE_TIER == 'pinned' else '1,2,3').split(',')))
_KA_CFGS = [_bwd_cfg(w, st, m) for w in _BWD_WARPS for st in _KA_STAGES for m in _BWD_MAXNREG]

def _ka_grid(tier):
    w, _, m = _TIER_AXES[tier]
    st = _env('KA_STAGES', '1' if tier == 'pinned' else '1,2,3')
    return [_bwd_cfg(a_, b_, c_) for a_ in (int(v) for v in _env('BWD_WARPS', w).split(',')) for b_ in (int(v) for v in st.split(',')) for c_ in (int(v) for v in _env('BWD_MAXNREG', m).split(','))]

@triton.autotune(configs=_calib_grid('fused_ka', _ka_grid), key=['T', 'N', 'G_SIZE', 'GT', 'PREC', 'MMA_IO', 'HAS_THETA_H', 'HAS_GAMMA', 'O', 'IDX64', 'SEG_SUMMARY', 'PAIR_HALF', 'NO_DALPHA', 'KA_WB'], prune_configs_by={'early_config_prune': _ka_prune})
@triton.jit
def _bwd_ka_kernel(C_ptr, Delta_ptr, A_ptr, ThetaH_ptr, SAll_ptr, Gamma_ptr, dY_ptr, dSF_ptr, dG_ptr, dAlpha_ptr, dX0_ptr, H, s_cb, s_cc, s_ct, s_cn, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_sb, s_sh, s_sc, s_sn, s_sg, s_fb, s_fh, s_fg, s_fn, s_gb, s_gh, s_gc, s_gn, s_gg, s_dab, s_dah, s_dag, s_dac, s_ch, NC, NC_SEG, s_dsfs, s_x0ss, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, MMA_IO: tl.constexpr=0, HAS_THETA_H: tl.constexpr=False, HAS_DSF: tl.constexpr=False, HAS_GAMMA: tl.constexpr=False, HAS_X0: tl.constexpr=False, SEG_SUMMARY: tl.constexpr=False, s_x0b: tl.constexpr=0, s_x0h: tl.constexpr=0, s_x0g: tl.constexpr=0, s_x0n: tl.constexpr=0, HPG_C: tl.constexpr=1, O: tl.constexpr=1, s_co: tl.constexpr=0, IDX64: tl.constexpr=False, Theta_ptr=None, s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0, HAS_ROT: tl.constexpr=False, BIAS_PRE_ROT: tl.constexpr=False, ThOff_ptr=None, s_ob=0, s_oh=0, s_oc=0, s_ok=0, PHASE_CUMSUM: tl.constexpr=False, PHASE_DEFER: tl.constexpr=False, PAIR_HALF: tl.constexpr=False, DCS_INK: tl.constexpr=False, NO_DALPHA: tl.constexpr=False, KA_WB=0):
    tl.static_assert(not NO_DALPHA or (not PAIR_HALF and (not SEG_SUMMARY)), 'Ka NO_DALPHA: main body, unsegmented only')
    tl.static_assert(not BIAS_PRE_ROT or HAS_ROT, 'Ka: bias_pre_rot needs the in-kernel phase')
    b = tl.program_id(0)
    h = tl.program_id(1)
    NGT_C = G_SIZE // GT
    seg = tl.program_id(2) // NGT_C
    gt_idx = tl.program_id(2) % NGT_C
    c_lo = seg * NC_SEG
    c_hi = tl.minimum(NC, c_lo + NC_SEG)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    cg = h // HPG_C
    if PAIR_HALF:
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        if HAS_THETA_H:
            th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)
            ch_h = _cos_approx(th_h)
            sh_h = _sin_approx(th_h)
        if HAS_DSF:
            f_base = dSF_ptr + b * s_fb + h * s_fh + seg * s_dsfs + offs_gt[None, :] * s_fg
            v1 = tl.load(f_base + offs_h1[:, None] * s_fn).to(tl.float32)
            v2 = tl.load(f_base + offs_h2[:, None] * s_fn).to(tl.float32)
        else:
            v1 = tl.zeros([KH, GT], dtype=tl.float32)
            v2 = tl.zeros([KH, GT], dtype=tl.float32)
        for ci in range(c_hi - c_lo):
            c = c_hi - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            sb = SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_gt[None, :] * s_sg
            S1 = tl.load(sb + offs_h1[:, None] * s_sn).to(tl.float32)
            S2 = tl.load(sb + offs_h2[:, None] * s_sn).to(tl.float32)
            dalpha_c = tl.sum(v1 * S1) + tl.sum(v2 * S2)
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_h1 * s_ok).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
            if not SEG_SUMMARY:
                tl.store(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac, dalpha_c)
                gb = dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_gt[None, :] * s_gg
                tl.store(gb + offs_h1[:, None] * s_gn, v1)
                tl.store(gb + offs_h2[:, None] * s_gn, v2)
            dS1 = tl.zeros([KH, GT], dtype=tl.float32)
            dS2 = tl.zeros([KH, GT], dtype=tl.float32)
            for o in tl.static_range(O):
                c_base = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                C1 = tl.load(c_base + offs_t[:, None] * s_ct + offs_h1[None, :] * s_cn).to(tl.float32)
                C2 = tl.load(c_base + offs_t[:, None] * s_ct + offs_h2[None, :] * s_cn).to(tl.float32)
                if HAS_GAMMA:
                    ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O + o).to(tl.float32)
                    ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O + o).to(tl.float32)
                    if BIAS_PRE_ROT:
                        C1 = C1 + ga1[None, :]
                        C2 = C2 + ga2[None, :]
                if HAS_ROT:
                    Ct1 = ph_cos * C1 + ph_sin * C2
                    Ct2 = ph_cos * C2 - ph_sin * C1
                else:
                    Ct1 = C1
                    Ct2 = C2
                if HAS_THETA_H:
                    Cr1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                    Cr2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                else:
                    Cr1 = Ct1
                    Cr2 = Ct2
                if HAS_GAMMA and (not BIAS_PRE_ROT):
                    if HAS_THETA_H:
                        g1 = ch_h * ga1 + sh_h * ga2
                        g2 = ch_h * ga2 - sh_h * ga1
                    else:
                        g1 = ga1
                        g2 = ga2
                    Cr1 = Cr1 + g1[None, :]
                    Cr2 = Cr2 + g2[None, :]
                dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                f_dY_o = (f[:, None] * dY_o).to(MMA_DT)
                dS1 += pdot(tl.trans(Cr1).to(MMA_DT), f_dY_o, input_precision=PREC)
                dS2 += pdot(tl.trans(Cr2).to(MMA_DT), f_dY_o, input_precision=PREC)
            v1 = alpha * v1 + dS1
            v2 = alpha * v2 + dS2
        if SEG_SUMMARY:
            xb = dX0_ptr + b * s_x0b + h * s_x0h + seg * s_x0ss + offs_gt[None, :] * s_x0g
            tl.store(xb + offs_h1[:, None] * s_x0n, v1)
            tl.store(xb + offs_h2[:, None] * s_x0n, v2)
        elif HAS_X0:
            if c_lo == 0:
                xb = dX0_ptr + b * s_x0b + h * s_x0h + offs_gt[None, :] * s_x0g
                tl.store(xb + offs_h1[:, None] * s_x0n, v1)
                tl.store(xb + offs_h2[:, None] * s_x0n, v2)
    else:
        ROT_ANY: tl.constexpr = HAS_THETA_H or HAS_ROT
        if ROT_ANY:
            pair_idx = offs_n // 2
            fwd_sign = tl.where(offs_n % 2 == 0, 1.0, -1.0)
        if HAS_THETA_H:
            th_full = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
            ch_full = _cos_approx(th_full)
            sh_full = _sin_approx(th_full)
            fwd_coeff = fwd_sign * sh_full
        if HAS_DSF:
            v = tl.load(dSF_ptr + b * s_fb + h * s_fh + seg * s_dsfs + offs_gt[None, :] * s_fg + offs_n[:, None] * s_fn).to(tl.float32)
        else:
            v = tl.zeros([N, GT], dtype=tl.float32)
        for ci in range(c_hi - c_lo):
            c = c_hi - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            if not NO_DALPHA:
                S_c = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg).to(tl.float32)
                dalpha_c = tl.sum(v * S_c)
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + tl.arange(0, N // 2)[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + tl.arange(0, N // 2) * s_ok).to(tl.float32)[None, :]
                cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, N)
                sn_rot = fwd_sign[None, :] * _pair_bcast_2d(_sin_approx(th_rot), T, N)
            if not SEG_SUMMARY:
                if not NO_DALPHA:
                    tl.store(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac, dalpha_c)
                tl.store(dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg, v)
            dS_readout = tl.zeros([N, GT], dtype=tl.float32)
            for o in tl.range(O, loop_unroll_factor=1):
                c_base = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                Co = tl.load(c_base + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                if HAS_ROT:
                    Co = cs_rot * Co + sn_rot * _pair_swap_2d(Co, T, N)
                if HAS_THETA_H:
                    if HAS_ROT:
                        Co_xor = _pair_swap_2d(Co, T, N)
                    else:
                        Co_xor = tl.load(c_base + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                    C_rd_o = ch_full[None, :] * Co + fwd_coeff[None, :] * Co_xor
                else:
                    C_rd_o = Co
                if HAS_GAMMA and (not BIAS_PRE_ROT):
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                    if HAS_THETA_H:
                        gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                        gam = ch_full * gam + fwd_coeff * gam_xor
                    C_rd_o = C_rd_o + gam[None, :]
                dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                f_dY_o = f[:, None] * dY_o
                dS_readout += pdot(tl.trans(C_rd_o).to(MMA_DT), f_dY_o.to(MMA_DT), input_precision=PREC)
            v = alpha * v + dS_readout
        if SEG_SUMMARY:
            tl.store(dX0_ptr + b * s_x0b + h * s_x0h + seg * s_x0ss + offs_gt[None, :] * s_x0g + offs_n[:, None] * s_x0n, v)
        elif HAS_X0:
            if c_lo == 0:
                tl.store(dX0_ptr + b * s_x0b + h * s_x0h + offs_gt[None, :] * s_x0g + offs_n[:, None] * s_x0n, v)

@triton.autotune(configs=_calib_grid('fused_kb1', _bwd_grid), prune_configs_by={'early_config_prune': _cprune('fused_kb1')}, key=['T', 'N', 'G_SIZE', 'GT', 'PREC', 'MMA_IO', 'HAS_THETA_H', 'HAS_BETA', 'HAS_GAMMA', 'HAS_ROT_BIAS', 'HAS_ROT', 'BIAS_PRE_ROT', 'PHASE_CUMSUM', 'R', 'O', 'HAS_DW', 'IDX64', 'HBLK', 'DB_STORE', 'DC_STORE', 'ROT_ABL', 'PAIR_HALF', 'KB2_FOLD', 'DTH_STORE', 'ROT_COMB', 'NSPLIT', 'OCC'], restore_value=['dCtil_ptr', 'dBtil_ptr', 'dDelta_ptr', 'dThetaH_ptr', 'dBeta_ptr', 'dGamma_ptr', 'dRotBias_ptr', 'dTheta_ptr', 'dThOff_ptr', 'Scratch_ptr'])
@triton.jit
def _bwd_kb1_kernel(C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, ThetaH_ptr, SAll_ptr, Beta_ptr, Gamma_ptr, RotBias_ptr, dY_ptr, dCtil_ptr, dBtil_ptr, dDelta_ptr, dThetaH_ptr, dBeta_ptr, dGamma_ptr, dRotBias_ptr, Scratch_ptr, H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xr, s_xl, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_sb, s_sh, s_sc, s_sn, s_sg, s_dcb, s_dcc, s_dct, s_dcn, s_dbb, s_dbc, s_dbt, s_dbn, s_bh, s_ch, NC, NGT_val, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, MMA_IO: tl.constexpr=0, HAS_THETA_H: tl.constexpr=False, NEED_DTHH: tl.constexpr=False, DEFERRED_ROTATION: tl.constexpr=False, HAS_BETA: tl.constexpr=False, HAS_GAMMA: tl.constexpr=False, HAS_ROT_BIAS: tl.constexpr=False, HPG_B: tl.constexpr=1, HPG_C: tl.constexpr=1, R: tl.constexpr=1, O: tl.constexpr=1, s_br: tl.constexpr=0, s_co: tl.constexpr=0, s_dbr: tl.constexpr=0, s_dco: tl.constexpr=0, CHUNK_PARALLEL: tl.constexpr=False, DW_ptr=None, s_wb=0, s_wh=0, s_wl=0, HAS_DW: tl.constexpr=False, s_slb=0, s_slc=0, SLOTS: tl.constexpr=1, s_dbh=0, s_dch=0, IDX64: tl.constexpr=False, Theta_ptr=None, s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0, dTheta_ptr=None, s_dtb=0, s_dth=0, s_dtc=0, s_dtt=0, s_dtk=0, HAS_ROT: tl.constexpr=False, BIAS_PRE_ROT: tl.constexpr=False, ThOff_ptr=None, s_ob=0, s_oh=0, s_oc=0, s_ok=0, dThOff_ptr=None, s_dob=0, s_doh=0, s_doc=0, s_dok=0, PHASE_CUMSUM: tl.constexpr=False, PHASE_DEFER: tl.constexpr=False, NSPLIT: tl.constexpr=1, HBLK: tl.constexpr=0, ROT_ABL: tl.constexpr=0, DB_STORE: tl.constexpr=False, DC_STORE: tl.constexpr=False, DTH_STORE: tl.constexpr=False, ROT_COMB: tl.constexpr=False, PAIR_HALF: tl.constexpr=False, DCS_INK: tl.constexpr=False, KB2_FOLD: tl.constexpr=False, dG_ptr=None, s_gb=0, s_gh=0, s_gc=0, s_gn=0, s_gg=0, dAlpha_ptr=None, s_dab=0, s_dah=0, s_dag=0, s_dac=0, TRAP_DIAG: tl.constexpr=False, OCC=0):
    tl.static_assert(not BIAS_PRE_ROT or (HAS_ROT and (not HAS_ROT_BIAS)), "Kb1: bias_pre_rot is Mamba-3's fold order -- it needs the in-kernel phase and no write-side angle (the READ offset theta_h composes after the phase)")
    tl.static_assert(not HAS_ROT or not DEFERRED_ROTATION, 'Kb1: with the in-kernel phase the theta_h un-rotation is done in registers, not deferred to Kinv (which knows nothing about theta_t)')
    tl.static_assert(not ROT_COMB or (HAS_ROT and HAS_THETA_H and (not HAS_ROT_BIAS) and (ROT_ABL == 0) and (not PAIR_HALF) and (BIAS_PRE_ROT or not HAS_GAMMA)), 'Kb1: ROT_COMB fuses the phase and the static read offset; it needs both, no rot_bias, no ablation and no post-rotation gamma')
    tl.static_assert(not DTH_STORE or (HBLK > 0 and KB2_FOLD and HAS_ROT and (not PAIR_HALF)), 'Kb1: DTH_STORE is the sole-owner dTheta write of the head-blocked interleaved body with Kb2 folded in')
    tl.static_assert(not KB2_FOLD or (HBLK > 0 and (not PAIR_HALF) and (not HAS_ROT_BIAS) and (not DEFERRED_ROTATION) and (ROT_ABL == 0) and (SLOTS == 1)), 'Kb1: KB2_FOLD lives in the head-blocked interleaved body only (real mode and, since M3_STRUCTURE stage 3, the in-kernel phase HAS_ROT); see `_kb2_fold`')
    ABL_NO_DTH: tl.constexpr = ROT_ABL & 1 != 0
    ABL_PREROT: tl.constexpr = ROT_ABL & 2 != 0
    if PAIR_HALF:
        tl.static_assert(not HAS_ROT_BIAS, 'PAIR_HALF does not serve the retired rot_bias')
        tl.static_assert(ROT_ABL == 0, 'PAIR_HALF carries no ablation ladder')
        tl.static_assert(not DEFERRED_ROTATION, 'PAIR_HALF un-rotates theta_h in registers')
        tl.static_assert(N >= 32 * NSPLIT, 'PAIR_HALF needs N/(2*NSPLIT) >= 16 (MMA K floor)')
        PS: tl.constexpr = N // (2 * NSPLIT)
        KH: tl.constexpr = N // 2
        neg_inf = float('-inf')
        offs_t = tl.arange(0, T)
        if HBLK > 0:
            b = tl.program_id(0)
            hblk = tl.program_id(1)
            pid2 = tl.program_id(2)
            ns_idx = pid2 % NSPLIT
            rest2 = pid2 // NSPLIT
            if CHUNK_PARALLEL:
                c = rest2 // NGT_val
                gt_idx = rest2 % NGT_val
            else:
                gt_idx = rest2
            if IDX64:
                b = b.to(tl.int64)
            offs_p = ns_idx * PS + tl.arange(0, PS)
            offs_h1 = offs_p
            offs_h2 = KH + offs_p
            offs_gt = gt_idx * GT + tl.arange(0, GT)
            if TRAP_DIAG:
                causal = offs_t[:, None] > offs_t[None, :]
            else:
                causal = offs_t[:, None] >= offs_t[None, :]
            h_start = hblk * HBLK
            if IDX64:
                h_start = h_start.to(tl.int64)
            bg = h_start // HPG_B
            cg = h_start // HPG_C
            nc_end = 1 if CHUNK_PARALLEL else NC
            for ci in range(nc_end):
                if CHUNK_PARALLEL:
                    pass
                else:
                    c = NC - 1 - ci
                dB1_l = ()
                dB2_l = ()
                for r in tl.static_range(R):
                    dB1_l = dB1_l + (tl.zeros([T, PS], dtype=tl.float32),)
                    dB2_l = dB2_l + (tl.zeros([T, PS], dtype=tl.float32),)
                dC1_l = ()
                dC2_l = ()
                for o in tl.static_range(O):
                    dC1_l = dC1_l + (tl.zeros([T, PS], dtype=tl.float32),)
                    dC2_l = dC2_l + (tl.zeros([T, PS], dtype=tl.float32),)
                for hh in range(HBLK):
                    h = h_start + hh
                    hv = h < H
                    h = tl.minimum(h, H - 1)
                    a_val = -tl.exp(tl.load(A_ptr + h))
                    if HAS_THETA_H:
                        th_h = tl.load(ThetaH_ptr + h * KH + offs_p)
                        ch_h = _cos_approx(th_h)
                        sh_h = _sin_approx(th_h)
                    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                    D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                    seg_end = tl.max(seg_h, axis=0)
                    f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                    if HAS_DW:
                        dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                    if HAS_ROT:
                        th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_p[None, :] * s_thk).to(tl.float32)
                        if PHASE_CUMSUM:
                            th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_p * s_ok).to(tl.float32)[None, :]
                        ph_cos = _cos_approx(th_rot)
                        ph_sin = _sin_approx(th_rot)
                        dth_acc = tl.zeros([T, PS], dtype=tl.float32)
                    dD = tl.zeros([T], dtype=tl.float32)
                    Bt1_l = ()
                    Bt2_l = ()
                    Be1_l = ()
                    Be2_l = ()
                    x_l = ()
                    dBp1 = ()
                    dBp2 = ()
                    for r in tl.static_range(R):
                        b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + r * s_br + offs_t[:, None] * s_bt
                        B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
                        B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
                        if HAS_BETA:
                            be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r).to(tl.float32)
                            be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r).to(tl.float32)
                            if BIAS_PRE_ROT:
                                B1 = B1 + be1[None, :]
                                B2 = B2 + be2[None, :]
                        if HAS_ROT:
                            Bt1 = ph_cos * B1 + ph_sin * B2
                            Bt2 = ph_cos * B2 - ph_sin * B1
                        else:
                            Bt1 = B1
                            Bt2 = B2
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            Bf1 = Bt1 + be1[None, :]
                            Bf2 = Bt2 + be2[None, :]
                        else:
                            Bf1 = Bt1
                            Bf2 = Bt2
                        x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        if HAS_DW:
                            x_r = x_r * dw[:, None]
                        Bt1_l = Bt1_l + (Bt1,)
                        Bt2_l = Bt2_l + (Bt2,)
                        Be1_l = Be1_l + (Bf1,)
                        Be2_l = Be2_l + (Bf2,)
                        x_l = x_l + (tl.trans(x_r).to(MMA_DT),)
                        dBp1 = dBp1 + (tl.zeros([T, PS], dtype=tl.float32),)
                        dBp2 = dBp2 + (tl.zeros([T, PS], dtype=tl.float32),)
                    s_base = SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_gt[None, :] * s_sg
                    S1 = tl.load(s_base + offs_h1[:, None] * s_sn).to(tl.float32)
                    S2 = tl.load(s_base + offs_h2[:, None] * s_sn).to(tl.float32)
                    dCp1 = ()
                    dCp2 = ()
                    for o in tl.static_range(O):
                        c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co + offs_t[:, None] * s_ct
                        C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn).to(tl.float32)
                        C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn).to(tl.float32)
                        if HAS_GAMMA:
                            ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O + o).to(tl.float32)
                            ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O + o).to(tl.float32)
                            if BIAS_PRE_ROT:
                                C1 = C1 + ga1[None, :]
                                C2 = C2 + ga2[None, :]
                        if HAS_ROT:
                            Ct1 = ph_cos * C1 + ph_sin * C2
                            Ct2 = ph_cos * C2 - ph_sin * C1
                        else:
                            Ct1 = C1
                            Ct2 = C2
                        if HAS_THETA_H:
                            Ch1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                            Ch2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                        else:
                            Ch1 = Ct1
                            Ch2 = Ct2
                        Cr1 = Ch1
                        Cr2 = Ch2
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            if HAS_THETA_H:
                                gr1 = ch_h * ga1 + sh_h * ga2
                                gr2 = ch_h * ga2 - sh_h * ga1
                            else:
                                gr1 = ga1
                                gr2 = ga2
                            Cr1 = Cr1 + gr1[None, :]
                            Cr2 = Cr2 + gr2[None, :]
                        dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                        dCs1 = tl.zeros([T, PS], dtype=tl.float32)
                        dCs2 = tl.zeros([T, PS], dtype=tl.float32)
                        dBp1_n = ()
                        dBp2_n = ()
                        for r in tl.static_range(R):
                            dScore_or = pdot(dY_o.to(MMA_DT), x_l[r], input_precision=PREC)
                            dSb_dt = (dScore_or * tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))).to(MMA_DT)
                            dCs1 += pdot(dSb_dt, Be1_l[r].to(MMA_DT), input_precision=PREC)
                            dCs2 += pdot(dSb_dt, Be2_l[r].to(MMA_DT), input_precision=PREC)
                            dSb_tr = tl.trans(dSb_dt)
                            tB1 = pdot(dSb_tr, Cr1.to(MMA_DT), input_precision=PREC)
                            tB2 = pdot(dSb_tr, Cr2.to(MMA_DT), input_precision=PREC)
                            dBp1_n = dBp1_n + (dBp1[r] + tB1,)
                            dBp2_n = dBp2_n + (dBp2[r] + tB2,)
                        dBp1 = dBp1_n
                        dBp2 = dBp2_n
                        dY_dt = dY_o.to(MMA_DT)
                        dCst1 = f[:, None] * pdot(dY_dt, tl.trans(S1).to(MMA_DT), input_precision=PREC)
                        dCst2 = f[:, None] * pdot(dY_dt, tl.trans(S2).to(MMA_DT), input_precision=PREC)
                        dD += a_val * (tl.sum(Cr1 * dCst1, axis=1) + tl.sum(Cr2 * dCst2, axis=1))
                        dC1 = dCs1 + dCst1
                        dC2 = dCs2 + dCst2
                        gC1 = dC1
                        gC2 = dC2
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            sg1 = tl.sum(gC1, axis=0)
                            sg2 = tl.sum(gC2, axis=0)
                            if HAS_THETA_H:
                                if NEED_DTHH:
                                    tl.atomic_add(dThetaH_ptr + h * KH + offs_p, sg1 * gr2 - sg2 * gr1, mask=hv, sem=ACC_SEM)
                                sn1 = ch_h * sg1 - sh_h * sg2
                                sg2 = ch_h * sg2 + sh_h * sg1
                                sg1 = sn1
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_h1 * O + o, sg1, mask=hv, sem=ACC_SEM)
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_h2 * O + o, sg2, mask=hv, sem=ACC_SEM)
                        if HAS_THETA_H:
                            if NEED_DTHH:
                                tl.atomic_add(dThetaH_ptr + h * KH + offs_p, tl.sum(gC1 * _opv(Ch2, MMA_DT, MMA_IO, PREC) - gC2 * _opv(Ch1, MMA_DT, MMA_IO, PREC), axis=0), mask=hv, sem=ACC_SEM)
                            gn1 = ch_h[None, :] * gC1 - sh_h[None, :] * gC2
                            gC2 = ch_h[None, :] * gC2 + sh_h[None, :] * gC1
                            gC1 = gn1
                        if HAS_ROT:
                            if HAS_THETA_H:
                                Cq1, Cq2 = _ph_read_consistent(Ch1, Ch2, ch_h, sh_h, Ct1, Ct2, MMA_DT, MMA_IO, HAS_THETA_H, PREC)
                            else:
                                Cq1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
                                Cq2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
                            dth_acc += gC1 * Cq2 - gC2 * Cq1
                            gn1 = ph_cos * gC1 - ph_sin * gC2
                            gC2 = ph_cos * gC2 + ph_sin * gC1
                            gC1 = gn1
                        if HAS_GAMMA and BIAS_PRE_ROT:
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_h1 * O + o, tl.sum(gC1, axis=0), mask=hv, sem=ACC_SEM)
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_h2 * O + o, tl.sum(gC2, axis=0), mask=hv, sem=ACC_SEM)
                        dCp1 = dCp1 + (gC1,)
                        dCp2 = dCp2 + (gC2,)
                    dBw1 = ()
                    dBw2 = ()
                    for r in tl.static_range(R):
                        gB1 = dBp1[r]
                        gB2 = dBp2[r]
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), mask=hv, sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), mask=hv, sem=ACC_SEM)
                        if HAS_ROT:
                            dth_acc += gB1 * _opv(Bt2_l[r], MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1_l[r], MMA_DT, MMA_IO, PREC)
                            gn1 = ph_cos * gB1 - ph_sin * gB2
                            gB2 = ph_cos * gB2 + ph_sin * gB1
                            gB1 = gn1
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), mask=hv, sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), mask=hv, sem=ACC_SEM)
                        dBw1 = dBw1 + (gB1,)
                        dBw2 = dBw2 + (gB2,)
                    dB1_n = ()
                    dB2_n = ()
                    for r in tl.static_range(R):
                        dB1_n = dB1_n + (dB1_l[r] + tl.where(hv, dBw1[r], 0.0),)
                        dB2_n = dB2_n + (dB2_l[r] + tl.where(hv, dBw2[r], 0.0),)
                    dB1_l = dB1_n
                    dB2_l = dB2_n
                    dC1_n = ()
                    dC2_n = ()
                    for o in tl.static_range(O):
                        dC1_n = dC1_n + (dC1_l[o] + tl.where(hv, dCp1[o], 0.0),)
                        dC2_n = dC2_n + (dC2_l[o] + tl.where(hv, dCp2[o], 0.0),)
                    dC1_l = dC1_n
                    dC2_l = dC2_n
                    if HAS_ROT:
                        dth_ph = dth_acc
                        if PHASE_CUMSUM and (not PHASE_DEFER):
                            dth_off = tl.sum(dth_ph, axis=0)
                            dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                            tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_p * s_dok, dth_off, mask=hv, sem=ACC_SEM)
                        tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_p[None, :] * s_dtk, dth_ph, mask=hv, sem=ACC_SEM)
                    tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, tl.cumsum(dD, axis=0, reverse=True), mask=hv, sem=ACC_SEM)
                for r in tl.static_range(R):
                    db_p = dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                    if DB_STORE:
                        tl.store(db_p + offs_h1[None, :] * s_dbn, dB1_l[r])
                        tl.store(db_p + offs_h2[None, :] * s_dbn, dB2_l[r])
                    else:
                        tl.atomic_add(db_p + offs_h1[None, :] * s_dbn, dB1_l[r], sem=ACC_SEM)
                        tl.atomic_add(db_p + offs_h2[None, :] * s_dbn, dB2_l[r], sem=ACC_SEM)
                for o in tl.static_range(O):
                    dc_p = dCtil_ptr + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco + offs_t[:, None] * s_dct
                    if DC_STORE:
                        tl.store(dc_p + offs_h1[None, :] * s_dcn, dC1_l[o])
                        tl.store(dc_p + offs_h2[None, :] * s_dcn, dC2_l[o])
                    else:
                        tl.atomic_add(dc_p + offs_h1[None, :] * s_dcn, dC1_l[o], sem=ACC_SEM)
                        tl.atomic_add(dc_p + offs_h2[None, :] * s_dcn, dC2_l[o], sem=ACC_SEM)
        else:
            b = tl.program_id(0)
            h = tl.program_id(1)
            pid2 = tl.program_id(2)
            ns_idx = pid2 % NSPLIT
            rest2 = pid2 // NSPLIT
            if CHUNK_PARALLEL:
                c = rest2 // NGT_val
                gt_idx = rest2 % NGT_val
            else:
                gt_idx = rest2
            if IDX64:
                b = b.to(tl.int64)
                h = h.to(tl.int64)
            a_val = -tl.exp(tl.load(A_ptr + h))
            offs_p = ns_idx * PS + tl.arange(0, PS)
            offs_h1 = offs_p
            offs_h2 = KH + offs_p
            offs_gt = gt_idx * GT + tl.arange(0, GT)
            if TRAP_DIAG:
                causal = offs_t[:, None] > offs_t[None, :]
            else:
                causal = offs_t[:, None] >= offs_t[None, :]
            bg = h // HPG_B
            cg = h // HPG_C
            slot_b = h % SLOTS * s_slb
            slot_c = h % SLOTS * s_slc
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * KH + offs_p)
                ch_h = _cos_approx(th_h)
                sh_h = _sin_approx(th_h)
            nc_end = 1 if CHUNK_PARALLEL else NC
            for ci in range(nc_end):
                if CHUNK_PARALLEL:
                    pass
                else:
                    c = NC - 1 - ci
                dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                seg_end = tl.max(seg_h, axis=0)
                f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                if HAS_DW:
                    dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                if HAS_ROT:
                    th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_p[None, :] * s_thk).to(tl.float32)
                    if PHASE_CUMSUM:
                        th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_p * s_ok).to(tl.float32)[None, :]
                    ph_cos = _cos_approx(th_rot)
                    ph_sin = _sin_approx(th_rot)
                    dth_acc = tl.zeros([T, PS], dtype=tl.float32)
                dD = tl.zeros([T], dtype=tl.float32)
                for o in tl.range(O, loop_unroll_factor=1):
                    c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co + offs_t[:, None] * s_ct
                    C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn).to(tl.float32)
                    C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn).to(tl.float32)
                    if HAS_GAMMA:
                        ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O + o).to(tl.float32)
                        ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O + o).to(tl.float32)
                        if BIAS_PRE_ROT:
                            C1 = C1 + ga1[None, :]
                            C2 = C2 + ga2[None, :]
                    if HAS_ROT:
                        Ct1 = ph_cos * C1 + ph_sin * C2
                        Ct2 = ph_cos * C2 - ph_sin * C1
                    else:
                        Ct1 = C1
                        Ct2 = C2
                    if HAS_THETA_H:
                        Ch1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                        Ch2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                    else:
                        Ch1 = Ct1
                        Ch2 = Ct2
                    Cr1 = Ch1
                    Cr2 = Ch2
                    if HAS_GAMMA and (not BIAS_PRE_ROT):
                        if HAS_THETA_H:
                            gr1 = ch_h * ga1 + sh_h * ga2
                            gr2 = ch_h * ga2 - sh_h * ga1
                        else:
                            gr1 = ga1
                            gr2 = ga2
                        Cr1 = Cr1 + gr1[None, :]
                        Cr2 = Cr2 + gr2[None, :]
                    dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                    dCs1 = tl.zeros([T, PS], dtype=tl.float32)
                    dCs2 = tl.zeros([T, PS], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + r * s_br + offs_t[:, None] * s_bt
                        B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
                        B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
                        if HAS_BETA:
                            be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r).to(tl.float32)
                            be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r).to(tl.float32)
                            if BIAS_PRE_ROT:
                                B1 = B1 + be1[None, :]
                                B2 = B2 + be2[None, :]
                        if HAS_ROT:
                            Bt1 = ph_cos * B1 + ph_sin * B2
                            Bt2 = ph_cos * B2 - ph_sin * B1
                        else:
                            Bt1 = B1
                            Bt2 = B2
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            Bf1 = Bt1 + be1[None, :]
                            Bf2 = Bt2 + be2[None, :]
                        else:
                            Bf1 = Bt1
                            Bf2 = Bt2
                        x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        if HAS_DW:
                            x_r = x_r * dw[:, None]
                        dScore_or = pdot(dY_o.to(MMA_DT), tl.trans(x_r).to(MMA_DT), input_precision=PREC)
                        dSb_dt = (dScore_or * tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))).to(MMA_DT)
                        dCs1 += pdot(dSb_dt, Bf1.to(MMA_DT), input_precision=PREC)
                        dCs2 += pdot(dSb_dt, Bf2.to(MMA_DT), input_precision=PREC)
                        dSb_tr = tl.trans(dSb_dt)
                        gB1 = pdot(dSb_tr, Cr1.to(MMA_DT), input_precision=PREC)
                        gB2 = pdot(dSb_tr, Cr2.to(MMA_DT), input_precision=PREC)
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), sem=ACC_SEM)
                        if HAS_ROT:
                            dth_acc += gB1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1, MMA_DT, MMA_IO, PREC)
                            gn1 = ph_cos * gB1 - ph_sin * gB2
                            gB2 = ph_cos * gB2 + ph_sin * gB1
                            gB1 = gn1
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), sem=ACC_SEM)
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), sem=ACC_SEM)
                        db_base_r = dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                        tl.atomic_add(db_base_r + offs_h1[None, :] * s_dbn, gB1, sem=ACC_SEM)
                        tl.atomic_add(db_base_r + offs_h2[None, :] * s_dbn, gB2, sem=ACC_SEM)
                    s_base = SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_gt[None, :] * s_sg
                    S1 = tl.load(s_base + offs_h1[:, None] * s_sn).to(tl.float32)
                    S2 = tl.load(s_base + offs_h2[:, None] * s_sn).to(tl.float32)
                    dY_dt = dY_o.to(MMA_DT)
                    dCst1 = f[:, None] * pdot(dY_dt, tl.trans(S1).to(MMA_DT), input_precision=PREC)
                    dCst2 = f[:, None] * pdot(dY_dt, tl.trans(S2).to(MMA_DT), input_precision=PREC)
                    dD += a_val * (tl.sum(Cr1 * dCst1, axis=1) + tl.sum(Cr2 * dCst2, axis=1))
                    dC1 = dCs1 + dCst1
                    dC2 = dCs2 + dCst2
                    gC1 = dC1
                    gC2 = dC2
                    if HAS_GAMMA and (not BIAS_PRE_ROT):
                        sg1 = tl.sum(gC1, axis=0)
                        sg2 = tl.sum(gC2, axis=0)
                        if HAS_THETA_H:
                            if NEED_DTHH:
                                tl.atomic_add(dThetaH_ptr + h * KH + offs_p, sg1 * gr2 - sg2 * gr1, sem=ACC_SEM)
                            sn1 = ch_h * sg1 - sh_h * sg2
                            sg2 = ch_h * sg2 + sh_h * sg1
                            sg1 = sn1
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h1 * O + o, sg1, sem=ACC_SEM)
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h2 * O + o, sg2, sem=ACC_SEM)
                    if HAS_THETA_H:
                        if NEED_DTHH:
                            tl.atomic_add(dThetaH_ptr + h * KH + offs_p, tl.sum(gC1 * _opv(Ch2, MMA_DT, MMA_IO, PREC) - gC2 * _opv(Ch1, MMA_DT, MMA_IO, PREC), axis=0), sem=ACC_SEM)
                        gn1 = ch_h[None, :] * gC1 - sh_h[None, :] * gC2
                        gC2 = ch_h[None, :] * gC2 + sh_h[None, :] * gC1
                        gC1 = gn1
                    if HAS_ROT:
                        if HAS_THETA_H:
                            Cq1, Cq2 = _ph_read_consistent(Ch1, Ch2, ch_h, sh_h, Ct1, Ct2, MMA_DT, MMA_IO, HAS_THETA_H, PREC)
                        else:
                            Cq1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
                            Cq2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
                        dth_acc += gC1 * Cq2 - gC2 * Cq1
                        gn1 = ph_cos * gC1 - ph_sin * gC2
                        gC2 = ph_cos * gC2 + ph_sin * gC1
                        gC1 = gn1
                    if HAS_GAMMA and BIAS_PRE_ROT:
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h1 * O + o, tl.sum(gC1, axis=0), sem=ACC_SEM)
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_h2 * O + o, tl.sum(gC2, axis=0), sem=ACC_SEM)
                    dc_base_o = dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco + offs_t[:, None] * s_dct
                    tl.atomic_add(dc_base_o + offs_h1[None, :] * s_dcn, gC1, sem=ACC_SEM)
                    tl.atomic_add(dc_base_o + offs_h2[None, :] * s_dcn, gC2, sem=ACC_SEM)
                if HAS_ROT:
                    dth_ph = dth_acc
                    if PHASE_CUMSUM and (not PHASE_DEFER):
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_p * s_dok, dth_off, sem=ACC_SEM)
                    tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_p[None, :] * s_dtk, dth_ph, sem=ACC_SEM)
                tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, tl.cumsum(dD, axis=0, reverse=True), sem=ACC_SEM)
    elif HBLK > 0:
        b = tl.program_id(0)
        hblk = tl.program_id(1)
        NS: tl.constexpr = N // NSPLIT
        pid2 = tl.program_id(2)
        ns_idx = pid2 % NSPLIT
        rest2 = pid2 // NSPLIT
        if CHUNK_PARALLEL:
            c = rest2 // NGT_val
            gt_idx = rest2 % NGT_val
        else:
            gt_idx = rest2
        if IDX64:
            b = b.to(tl.int64)
        offs_t = tl.arange(0, T)
        offs_n = ns_idx * NS + tl.arange(0, NS)
        offs_k = ns_idx * (NS // 2) + tl.arange(0, NS // 2)
        offs_gt = gt_idx * GT + tl.arange(0, GT)
        if TRAP_DIAG:
            causal = offs_t[:, None] > offs_t[None, :]
        else:
            causal = offs_t[:, None] >= offs_t[None, :]
        neg_inf = float('-inf')
        pair_idx_kb1 = offs_n // 2
        fwd_sign_kb1 = tl.where(offs_n % 2 == 0, 1.0, -1.0)
        THH_ACC: tl.constexpr = ROT_COMB and NEED_DTHH
        if KB2_FOLD:
            is_last_kb1 = tl.where(offs_t == T - 1, 1.0, 0.0)
            if NSPLIT == 1:
                da_gate_kb1 = 1.0
            else:
                da_gate_kb1 = tl.where(ns_idx == 0, 1.0, 0.0)
        h_start = hblk * HBLK
        if IDX64:
            h_start = h_start.to(tl.int64)
        bg = h_start // HPG_B
        cg = h_start // HPG_C
        nc_end = 1 if CHUNK_PARALLEL else NC
        for ci in range(nc_end):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci
            dB_l = ()
            for r in tl.static_range(R):
                dB_l = dB_l + (tl.zeros([T, NS], dtype=tl.float32),)
            dC_l = ()
            for o in tl.static_range(O):
                dC_l = dC_l + (tl.zeros([T, NS], dtype=tl.float32),)
            for hh in range(HBLK):
                h = h_start + hh
                hv = h < H
                h = tl.minimum(h, H - 1)
                a_val = -tl.exp(tl.load(A_ptr + h))
                if HAS_THETA_H:
                    th_full_kb1 = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx_kb1)
                    ch_full_kb1 = _cos_approx(th_full_kb1)
                    fwd_coeff_kb1 = fwd_sign_kb1 * _sin_approx(th_full_kb1)
                if HAS_ROT_BIAS:
                    rb_full_kb1 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_kb1)
                    crb_kb1 = _cos_approx(rb_full_kb1)
                    srb_fwd_kb1 = fwd_sign_kb1 * _sin_approx(rb_full_kb1)
                dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                seg_end = tl.max(seg_h, axis=0)
                f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
                if HAS_DW:
                    dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                if KB2_FOLD:
                    D_end_kb1 = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                    dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_kb1 - D_h)), 0.0)
                    dG_agg = tl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg).to(MMA_DT)
                    dend_kb2 = tl.zeros([T], dtype=tl.float32)
                if HAS_ROT:
                    th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk).to(tl.float32)
                    if PHASE_CUMSUM:
                        th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_k * s_ok).to(tl.float32)[None, :]
                    cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, NS)
                    sn_rot = fwd_sign_kb1[None, :] * _pair_bcast_2d(_sin_approx(th_rot), T, NS)
                    dth_acc = tl.zeros([T, NS], dtype=tl.float32)
                dD = tl.zeros([T], dtype=tl.float32)
                B_til_l = ()
                Bc_l = ()
                Bx_l = ()
                x_l = ()
                dBp = ()
                for r in tl.static_range(R):
                    b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + r * s_br
                    Bc_r = tl.load(b_base_r + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    if HAS_ROT and (not ABL_PREROT):
                        B_sw = _pair_swap_2d(Bc_r, T, NS)
                        B_til = cs_rot * Bc_r + sn_rot * B_sw
                        B_tilx = cs_rot * B_sw - sn_rot * Bc_r
                    else:
                        B_til = Bc_r
                        if HAS_ROT:
                            B_tilx = _pair_swap_2d(Bc_r, T, NS)
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            B_unrot_xor = _pair_swap_2d(B_til, T, NS)
                        else:
                            B_unrot_xor = tl.load(b_base_r + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                        Bx_l = Bx_l + (B_unrot_xor,)
                        Bc_r = crb_kb1[None, :] * B_til + srb_fwd_kb1[None, :] * B_unrot_xor
                    else:
                        Bc_r = B_til
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        x_r = x_r * dw[:, None]
                    if HAS_ROT and (not HAS_ROT_BIAS):
                        B_til_l = B_til_l + (B_tilx,)
                    else:
                        B_til_l = B_til_l + (B_til,)
                    Bc_l = Bc_l + (Bc_r,)
                    x_l = x_l + (tl.trans(x_r).to(MMA_DT),)
                    if KB2_FOLD:
                        xw_r = (x_r * dec_agg[:, None]).to(MMA_DT)
                        dB_agg_r = pdot(xw_r, tl.trans(dG_agg), input_precision=PREC)
                        dend_kb2 += tl.sum(Bc_r * dB_agg_r, axis=1)
                        dBp = dBp + (dB_agg_r,)
                    else:
                        dBp = dBp + (tl.zeros([T, NS], dtype=tl.float32),)
                S_c = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg).to(tl.float32)
                dCp = ()
                if THH_ACC:
                    dthh_c = tl.zeros([NS], dtype=tl.float32)
                for o in tl.static_range(O):
                    c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                    Co = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_GAMMA:
                        Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                    if ROT_COMB:
                        C_sw = _pair_swap_2d(Co, T, NS)
                        cph = cs_rot * ch_full_kb1[None, :] - sn_rot * fwd_coeff_kb1[None, :]
                        sph = sn_rot * ch_full_kb1[None, :] + cs_rot * fwd_coeff_kb1[None, :]
                        C_h = cph * Co + sph * C_sw
                        C_hx = cph * C_sw - sph * Co
                    else:
                        if HAS_ROT and (not ABL_PREROT):
                            C_sw = _pair_swap_2d(Co, T, NS)
                            C_til = cs_rot * Co + sn_rot * C_sw
                            C_tilx = cs_rot * C_sw - sn_rot * Co
                        else:
                            C_til = Co
                            if HAS_ROT:
                                C_tilx = _pair_swap_2d(Co, T, NS)
                        if HAS_THETA_H:
                            if HAS_ROT:
                                Co_xor = C_tilx
                            else:
                                Co_xor = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                            C_h = ch_full_kb1[None, :] * C_til + fwd_coeff_kb1[None, :] * Co_xor
                        else:
                            C_h = C_til
                    C_o_rd = C_h
                    if HAS_GAMMA and (not BIAS_PRE_ROT):
                        gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                        if HAS_THETA_H:
                            gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                            gam = ch_full_kb1 * gam + fwd_coeff_kb1 * gam_xor
                        C_o_rd = C_o_rd + gam[None, :]
                    dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                    dC_score_o = tl.zeros([T, NS], dtype=tl.float32)
                    dY_o_dt = dY_o.to(MMA_DT)
                    C_o_dt = C_o_rd.to(MMA_DT)
                    dBp_next = ()
                    for r in tl.static_range(R):
                        dScore_or = pdot(dY_o_dt, x_l[r], input_precision=PREC)
                        dSb_or_dt = (dScore_or * dec_mask).to(MMA_DT)
                        dC_score_o += pdot(dSb_or_dt, Bc_l[r].to(MMA_DT), input_precision=PREC)
                        dB_or = pdot(tl.trans(dSb_or_dt), C_o_dt, input_precision=PREC)
                        dBp_next = dBp_next + (dBp[r] + dB_or,)
                    dBp = dBp_next
                    dC_state_o = f[:, None] * pdot(dY_o_dt, tl.trans(S_c).to(MMA_DT), input_precision=PREC)
                    dC_o_rd = dC_score_o + dC_state_o
                    dD += a_val * tl.sum(C_o_rd * dC_state_o, axis=1)
                    if HAS_GAMMA and (not HAS_ROT):
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, tl.sum(dC_o_rd, axis=0), mask=hv, sem=ACC_SEM)
                    if ROT_COMB:
                        gC_ = dC_o_rd
                        prod_c = gC_ * _opv(C_hx, MMA_DT, MMA_IO, PREC)
                        if NEED_DTHH:
                            dthh_c += tl.sum(fwd_sign_kb1[None, :] * prod_c, axis=0)
                        dth_acc += prod_c
                        cph = cs_rot * ch_full_kb1[None, :] - sn_rot * fwd_coeff_kb1[None, :]
                        sph = sn_rot * ch_full_kb1[None, :] + cs_rot * fwd_coeff_kb1[None, :]
                        gC_ = cph * gC_ - sph * _pair_swap_2d(gC_, T, NS)
                        if HAS_GAMMA:
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, tl.sum(gC_, axis=0), mask=hv, sem=ACC_SEM)
                        dCp = dCp + (gC_,)
                    elif HAS_ROT:
                        gC_ = dC_o_rd
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            sg = tl.sum(gC_, axis=0)
                            if HAS_THETA_H:
                                gam_l = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                                gam_x = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                gam_rot = ch_full_kb1 * gam_l + fwd_coeff_kb1 * gam_x
                                if NEED_DTHH:
                                    tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, fwd_sign_kb1 * sg * _pair_swap_1d(gam_rot, NS), mask=hv, sem=ACC_SEM)
                                sg = ch_full_kb1 * sg - fwd_coeff_kb1 * _pair_swap_1d(sg, NS)
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, sg, mask=hv, sem=ACC_SEM)
                        if HAS_THETA_H:
                            if NEED_DTHH:
                                tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, tl.sum(fwd_sign_kb1[None, :] * gC_ * _opv(ch_full_kb1[None, :] * C_tilx - fwd_coeff_kb1[None, :] * C_til, MMA_DT, MMA_IO, PREC), axis=0), mask=hv, sem=ACC_SEM)
                            gC_ = ch_full_kb1[None, :] * gC_ - fwd_coeff_kb1[None, :] * _pair_swap_2d(gC_, T, NS)
                        if not ABL_NO_DTH:
                            if (MMA_IO or PREC == 'tf32') and DTH_OPROUND and HAS_THETA_H:
                                C_hr = _opv(C_h, MMA_DT, MMA_IO, PREC)
                                C_hxr = _opv(ch_full_kb1[None, :] * C_tilx - fwd_coeff_kb1[None, :] * C_til, MMA_DT, MMA_IO, PREC)
                                dth_acc += gC_ * (ch_full_kb1[None, :] * C_hxr + fwd_coeff_kb1[None, :] * C_hr)
                            else:
                                dth_acc += gC_ * _opv(C_tilx, MMA_DT, MMA_IO, PREC)
                        gC_ = cs_rot * gC_ - sn_rot * _pair_swap_2d(gC_, T, NS)
                        if HAS_GAMMA and BIAS_PRE_ROT:
                            tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, tl.sum(gC_, axis=0), mask=hv, sem=ACC_SEM)
                        dCp = dCp + (gC_,)
                    elif HAS_THETA_H:
                        if NEED_DTHH:
                            Co2_xor_o = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                            Co2_o = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                            Crd_p_o = ch_full_kb1[None, :] * Co2_xor_o - fwd_coeff_kb1[None, :] * Co2_o
                            tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, tl.sum(fwd_sign_kb1[None, :] * dC_o_rd * Crd_p_o, axis=0), mask=hv, sem=ACC_SEM)
                        dCp = dCp + (ch_full_kb1[None, :] * dC_o_rd + _pair_swap_2d(fwd_coeff_kb1[None, :] * dC_o_rd, T, NS),)
                    else:
                        dCp = dCp + (dC_o_rd,)
                if THH_ACC:
                    thh_e, thh_o = tl.split(tl.reshape(dthh_c, (NS // 2, 2)))
                    tl.atomic_add(dThetaH_ptr + h * (N // 2) + offs_k, thh_e + thh_o, mask=hv, sem=ACC_SEM)
                dBw = ()
                for r in tl.static_range(R):
                    dB_r_part = dBp[r]
                    B_til = B_til_l[r]
                    if HAS_ROT:
                        gB_ = dB_r_part
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), mask=hv, sem=ACC_SEM)
                        if HAS_ROT_BIAS:
                            Brd_p = crb_kb1[None, :] * Bx_l[r] - srb_fwd_kb1[None, :] * B_til
                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb1, tl.sum(fwd_sign_kb1[None, :] * gB_ * Brd_p, axis=0), mask=hv, sem=ACC_SEM)
                            gB_ = crb_kb1[None, :] * gB_ - srb_fwd_kb1[None, :] * _pair_swap_2d(gB_, T, NS)
                        if not ABL_NO_DTH:
                            dth_acc += gB_ * _opv(_pair_swap_2d(B_til, T, NS) if HAS_ROT_BIAS else B_til, MMA_DT, MMA_IO, PREC)
                        gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, NS)
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), mask=hv, sem=ACC_SEM)
                        dBw = dBw + (gB_,)
                    elif HAS_ROT_BIAS:
                        Brd_p = crb_kb1[None, :] * Bx_l[r] - srb_fwd_kb1[None, :] * B_til
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb1, tl.sum(fwd_sign_kb1[None, :] * dB_r_part * Brd_p, axis=0), mask=hv, sem=ACC_SEM)
                        dBw = dBw + (crb_kb1[None, :] * dB_r_part + _pair_swap_2d(srb_fwd_kb1[None, :] * dB_r_part, T, NS),)
                    else:
                        dBw = dBw + (dB_r_part,)
                    if HAS_BETA and (not HAS_ROT):
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(dB_r_part, axis=0), mask=hv, sem=ACC_SEM)
                dB_next = ()
                for r in tl.static_range(R):
                    dB_next = dB_next + (dB_l[r] + tl.where(hv, dBw[r], 0.0),)
                dB_l = dB_next
                dC_next = ()
                for o in tl.static_range(O):
                    dC_next = dC_next + (dC_l[o] + tl.where(hv, dCp[o], 0.0),)
                dC_l = dC_next
                if HAS_ROT and (not ABL_NO_DTH):
                    dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, NS // 2, 2)))
                    dth_ph = dth_e - dth_o
                    if PHASE_CUMSUM and (not PHASE_DEFER):
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_k * s_dok, dth_off, mask=hv, sem=ACC_SEM)
                    dth_p = dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_k[None, :] * s_dtk
                    if DTH_STORE:
                        tl.store(dth_p, dth_ph, mask=hv)
                    else:
                        tl.atomic_add(dth_p, dth_ph, mask=hv, sem=ACC_SEM)
                dD = tl.cumsum(dD, axis=0, reverse=True)
                if KB2_FOLD:
                    dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
                    dD += a_val * (_excl_prefix(dend_kb2, offs_t) + tl.where(seg_end == 0, tl.exp(a_val * D_end_kb1), 0.0) * dalpha_c * da_gate_kb1)
                tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, mask=hv, sem=ACC_SEM)
            for r in tl.static_range(R):
                db_p = dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr
                if DB_STORE:
                    tl.store(db_p, dB_l[r])
                else:
                    tl.atomic_add(db_p, dB_l[r], sem=ACC_SEM)
            for o in tl.static_range(O):
                dc_p = dCtil_ptr + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn
                if DC_STORE:
                    tl.store(dc_p, dC_l[o])
                else:
                    tl.atomic_add(dc_p, dC_l[o], sem=ACC_SEM)
    else:
        b = tl.program_id(0)
        h = tl.program_id(1)
        NS: tl.constexpr = N // NSPLIT
        pid2 = tl.program_id(2)
        ns_idx = pid2 % NSPLIT
        rest2 = pid2 // NSPLIT
        if CHUNK_PARALLEL:
            c = rest2 // NGT_val
            gt_idx = rest2 % NGT_val
        else:
            gt_idx = rest2
        if IDX64:
            b = b.to(tl.int64)
            h = h.to(tl.int64)
        a_val = -tl.exp(tl.load(A_ptr + h))
        offs_t = tl.arange(0, T)
        offs_n = ns_idx * NS + tl.arange(0, NS)
        offs_gt = gt_idx * GT + tl.arange(0, GT)
        if TRAP_DIAG:
            causal = offs_t[:, None] > offs_t[None, :]
        else:
            causal = offs_t[:, None] >= offs_t[None, :]
        neg_inf = float('-inf')
        bg = h // HPG_B
        cg = h // HPG_C
        slot_b = h % SLOTS * s_slb
        slot_c = h % SLOTS * s_slc
        pair_idx_kb1 = offs_n // 2
        fwd_sign_kb1 = tl.where(offs_n % 2 == 0, 1.0, -1.0)
        if HAS_THETA_H:
            th_full_kb1 = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx_kb1)
            ch_full_kb1 = _cos_approx(th_full_kb1)
            sh_full_kb1 = _sin_approx(th_full_kb1)
            fwd_coeff_kb1 = fwd_sign_kb1 * sh_full_kb1
        if HAS_ROT_BIAS:
            rb_full_kb1 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_kb1)
            crb_kb1 = _cos_approx(rb_full_kb1)
            srb_fwd_kb1 = fwd_sign_kb1 * _sin_approx(rb_full_kb1)
        nc_end = 1 if CHUNK_PARALLEL else NC
        for ci in range(nc_end):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + (ns_idx * (NS // 2) + tl.arange(0, NS // 2))[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + (ns_idx * (NS // 2) + tl.arange(0, NS // 2)) * s_ok).to(tl.float32)[None, :]
                cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, NS)
                sn_rot = fwd_sign_kb1[None, :] * _pair_bcast_2d(_sin_approx(th_rot), T, NS)
                dth_acc = tl.zeros([T, NS], dtype=tl.float32)
            dD = tl.zeros([T], dtype=tl.float32)
            for o in tl.range(O, loop_unroll_factor=1):
                c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                Co = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                if HAS_ROT:
                    C_til = cs_rot * Co + sn_rot * _pair_swap_2d(Co, T, NS)
                else:
                    C_til = Co
                if HAS_THETA_H:
                    if HAS_ROT:
                        Co_xor = _pair_swap_2d(C_til, T, NS)
                    else:
                        Co_xor = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                    C_h = ch_full_kb1[None, :] * C_til + fwd_coeff_kb1[None, :] * Co_xor
                else:
                    C_h = C_til
                C_o_rd = C_h
                if HAS_GAMMA and (not BIAS_PRE_ROT):
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                    if HAS_THETA_H:
                        gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                        gam = ch_full_kb1 * gam + fwd_coeff_kb1 * gam_xor
                    C_o_rd = C_o_rd + gam[None, :]
                dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                dC_score_o = tl.zeros([T, NS], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r * s_br).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    if HAS_ROT:
                        B_til = cs_rot * Bc_r + sn_rot * _pair_swap_2d(Bc_r, T, NS)
                    else:
                        B_til = Bc_r
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(B_til, T, NS)
                        else:
                            Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r * s_br).to(tl.float32)
                        B_unrot_r = B_til
                        B_unrot_xor = Bc_r_xor_rb
                        Bc_r = crb_kb1[None, :] * B_til + srb_fwd_kb1[None, :] * Bc_r_xor_rb
                    else:
                        Bc_r = B_til
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        x_r = x_r * dw[:, None]
                    dScore_or = pdot(dY_o.to(MMA_DT), tl.trans(x_r).to(MMA_DT), input_precision=PREC)
                    dSb_or = dScore_or * dec_mask
                    dSb_or_dt = dSb_or.to(MMA_DT)
                    dC_score_o += pdot(dSb_or_dt, Bc_r.to(MMA_DT), input_precision=PREC)
                    dB_r_part = pdot(tl.trans(dSb_or_dt), C_o_rd.to(MMA_DT), input_precision=PREC)
                    if HAS_ROT:
                        gB_ = dB_r_part
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), sem=ACC_SEM)
                        if HAS_ROT_BIAS:
                            Brd_p = crb_kb1[None, :] * B_unrot_xor - srb_fwd_kb1[None, :] * B_unrot_r
                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb1, tl.sum(fwd_sign_kb1[None, :] * gB_ * Brd_p, axis=0), sem=ACC_SEM)
                            gB_ = crb_kb1[None, :] * gB_ - srb_fwd_kb1[None, :] * _pair_swap_2d(gB_, T, NS)
                        dth_acc += fwd_sign_kb1[None, :] * gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, NS)
                        gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, NS)
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), sem=ACC_SEM)
                        tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, gB_, sem=ACC_SEM)
                    elif HAS_ROT_BIAS:
                        Brd_p = crb_kb1[None, :] * B_unrot_xor - srb_fwd_kb1[None, :] * B_unrot_r
                        cross_rb = tl.sum(fwd_sign_kb1[None, :] * dB_r_part * Brd_p, axis=0)
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb1, cross_rb, sem=ACC_SEM)
                        db_base_r = dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                        tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn, crb_kb1[None, :] * dB_r_part, sem=ACC_SEM)
                        tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn, srb_fwd_kb1[None, :] * dB_r_part, sem=ACC_SEM)
                    else:
                        tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, dB_r_part, sem=ACC_SEM)
                    if HAS_BETA and (not HAS_ROT):
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(dB_r_part, axis=0), sem=ACC_SEM)
                S_c = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg).to(tl.float32)
                dC_state_o = f[:, None] * pdot(dY_o.to(MMA_DT), tl.trans(S_c).to(MMA_DT), input_precision=PREC)
                dC_o_rd = dC_score_o + dC_state_o
                dD += a_val * tl.sum(C_o_rd * dC_state_o, axis=1)
                if HAS_GAMMA and (not DEFERRED_ROTATION) and (not HAS_ROT):
                    tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, tl.sum(dC_o_rd, axis=0), sem=ACC_SEM)
                if HAS_ROT:
                    gC_ = dC_o_rd
                    if HAS_GAMMA and (not BIAS_PRE_ROT):
                        sg = tl.sum(gC_, axis=0)
                        if HAS_THETA_H:
                            gam_l = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                            gam_x = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                            gam_rot = ch_full_kb1 * gam_l + fwd_coeff_kb1 * gam_x
                            if NEED_DTHH:
                                tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, fwd_sign_kb1 * sg * _pair_swap_1d(gam_rot, NS), sem=ACC_SEM)
                            sg = ch_full_kb1 * sg - fwd_coeff_kb1 * _pair_swap_1d(sg, NS)
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, sg, sem=ACC_SEM)
                    if HAS_THETA_H:
                        if NEED_DTHH:
                            tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, tl.sum(fwd_sign_kb1[None, :] * gC_ * _pair_swap_2d(_opv(C_h, MMA_DT, MMA_IO, PREC), T, NS), axis=0), sem=ACC_SEM)
                        gC_ = ch_full_kb1[None, :] * gC_ - fwd_coeff_kb1[None, :] * _pair_swap_2d(gC_, T, NS)
                    if (MMA_IO or PREC == 'tf32') and DTH_OPROUND and HAS_THETA_H:
                        C_hr = _opv(C_h, MMA_DT, MMA_IO, PREC)
                        dth_acc += fwd_sign_kb1[None, :] * gC_ * (ch_full_kb1[None, :] * _pair_swap_2d(C_hr, T, NS) + fwd_coeff_kb1[None, :] * C_hr)
                    else:
                        dth_acc += fwd_sign_kb1[None, :] * gC_ * _pair_swap_2d(_opv(C_til, MMA_DT, MMA_IO, PREC), T, NS)
                    gC_ = cs_rot * gC_ - sn_rot * _pair_swap_2d(gC_, T, NS)
                    if HAS_GAMMA and BIAS_PRE_ROT:
                        tl.atomic_add(dGamma_ptr + h * N * O + offs_n * O + o, tl.sum(gC_, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn, gC_, sem=ACC_SEM)
                elif HAS_THETA_H and DEFERRED_ROTATION:
                    scratch = Scratch_ptr + (b * H + h) * (NC * O * T * N) + c * (O * T * N) + o * (T * N) + offs_t[:, None] * N + offs_n[None, :]
                    if G_SIZE == GT:
                        tl.store(scratch, dC_o_rd)
                    else:
                        tl.atomic_add(scratch, dC_o_rd, sem=ACC_SEM)
                elif HAS_THETA_H:
                    if NEED_DTHH:
                        Co2_xor_o = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                        Co2_o = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                        Crd_p_o = ch_full_kb1[None, :] * Co2_xor_o - fwd_coeff_kb1[None, :] * Co2_o
                        cross_o = tl.sum(fwd_sign_kb1[None, :] * dC_o_rd * Crd_p_o, axis=0)
                        tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx_kb1, cross_o, sem=ACC_SEM)
                    dc_base_o = dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc + offs_t[:, None] * s_dct + o * s_dco
                    tl.atomic_add(dc_base_o + offs_n[None, :] * s_dcn, ch_full_kb1[None, :] * dC_o_rd, sem=ACC_SEM)
                    tl.atomic_add(dc_base_o + (offs_n ^ 1)[None, :] * s_dcn, fwd_coeff_kb1[None, :] * dC_o_rd, sem=ACC_SEM)
                else:
                    tl.atomic_add(dCtil_ptr + slot_c + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn, dC_o_rd, sem=ACC_SEM)
            if HAS_ROT:
                dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, NS // 2, 2)))
                dth_ph = dth_e + dth_o
                if PHASE_CUMSUM and (not PHASE_DEFER):
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + (ns_idx * (NS // 2) + tl.arange(0, NS // 2)) * s_dok, dth_off, sem=ACC_SEM)
                tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + (ns_idx * (NS // 2) + tl.arange(0, NS // 2))[None, :] * s_dtk, dth_ph, sem=ACC_SEM)
            tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, tl.cumsum(dD, axis=0, reverse=True), sem=ACC_SEM)
_KBC_SCHED_DEFAULT = 8

def kbc_sched(mma_io: bool=False) -> int:
    env = _env('BWD_KBC_SCHED', '')
    if env != '':
        return int(env)
    if mma_io:
        try:
            from .chunkpar import cp_calibration_key
            from .smem import chunk_cache_lookup
            v = chunk_cache_lookup(cp_calibration_key(torch.cuda.current_device(), 'kbc') + '|sched')
            if v is not None:
                return int(v)
        except Exception:
            pass
        return 0
    return _KBC_SCHED_DEFAULT
_HB_VALS = tuple((int(v) for v in _env('BWD_HB_CANDS', '32' if _AUTOTUNE_OFF else '8,32').split(',')))
_HB_MAXNREG = tuple((int(v) for v in _env('BWD_HB_MAXNREG', '0').split(',')))
_SWIZZLE_FLOOR_BYTES = 128

def _hb_ns_floor(mma_io: bool) -> int:
    return _SWIZZLE_FLOOR_BYTES // (2 if mma_io else 4)

def _hb_nsplit(N: int, rot_per_head: bool=False, mma_io: bool=False) -> int:
    ns_floor = _hb_ns_floor(mma_io)
    env = _env('BWD_NSPLIT', '')
    if env:
        ns = int(env)
        if ns > 1 and N // ns < ns_floor:
            raise ValueError("SSD_MIMO_BWD_NSPLIT=%d gives NS=%d at N=%d, below the %d-element (128-byte) wgmma swizzle floor for these MMA operands -- Triton 3.6 mis-lowers the nvmma_shared<swizzlingByteWidth=64> descriptor there (illegal __shared__ read).  See _hb_nsplit's docstring." % (ns, N // ns, N, ns_floor))
    elif rot_per_head:
        ns = max(1, min(4, N // ns_floor))
    else:
        ns = 2 if N >= 128 else 1
    while ns > 1 and (N % ns or N // ns < ns_floor):
        ns //= 2
    return max(1, ns)

def _kb1_nsplit(N: int, R: int, O: int, mma_io: bool=False) -> int:
    ns_floor = _hb_ns_floor(mma_io)
    env = _env('BWD_KB1_NSPLIT', '')
    if env:
        ns = int(env)
        if ns > 1 and N // ns < ns_floor:
            raise ValueError("SSD_MIMO_BWD_KB1_NSPLIT=%d gives NS=%d at N=%d, below the %d-element (128-byte) wgmma swizzle floor for these MMA operands -- Triton 3.6 mis-lowers the nvmma_shared<swizzlingByteWidth=64> descriptor there.  See _hb_nsplit's docstring." % (ns, N // ns, N, ns_floor))
    else:
        ns = 1
    while ns > 1 and (N % ns or N // ns < ns_floor or N // ns % 2):
        ns //= 2
    return max(1, ns)
_KB1_HB_OOR: dict = {}
_KB1_DISPATCH_LOG: list = []
_KB1_DISPATCH_LOG_MAX = 32

def kb1_dispatch_log() -> list:
    return list(_KB1_DISPATCH_LOG)

def _note_kb1_dispatch(record: dict) -> None:
    _KB1_DISPATCH_LOG.append(record)
    del _KB1_DISPATCH_LOG[:-_KB1_DISPATCH_LOG_MAX]
_KB1_CLIFF: dict = {}
_KB1_CLIFF_LEVERS = ('nsplit', 'maxnreg', 'dth0', 'off')
_KB1_CLIFF_REGS = int(_env('BWD_KB1_CLIFF_REGS', '128'))
_KB1_CLIFF_STACK = int(_env('BWD_KB1_CLIFF_STACK', '1024'))

def _kb1_cliff_lever() -> str:
    if _env('BWD_KB1_NSPLIT', ''):
        return 'off'
    v = _env('BWD_KB1_CLIFF', 'nsplit').strip().lower()
    if v not in _KB1_CLIFF_LEVERS:
        raise ValueError('SSD_MIMO_BWD_KB1_CLIFF=%r: expected one of %s' % (v, '/'.join(_KB1_CLIFF_LEVERS)))
    return v

def _kb1_collapsed(n_regs: int, stack_bytes: int) -> bool:
    return n_regs <= _KB1_CLIFF_REGS and stack_bytes >= _KB1_CLIFF_STACK

def _kb1_probe(compile_one, cfg_kwargs: list) -> list:
    from triton.runtime.errors import PTXASError
    try:
        from triton.compiler.errors import CompileTimeAssertionFailure
    except ImportError:
        CompileTimeAssertionFailure = OutOfResources
    out = []
    for kw in cfg_kwargs:
        f = dict(w=kw.get('num_warps'), s=kw.get('num_stages'), mnr=kw.get('maxnreg'))
        try:
            k = compile_one(kw)
            k._init_handles()
            f.update(fit=True, regs=int(k.n_regs), stack=4 * int(k.n_spills))
            f['collapsed'] = _kb1_collapsed(f['regs'], f['stack'])
        except (OutOfResources, PTXASError, CompileTimeAssertionFailure) as e:
            f.update(fit=False, err=type(e).__name__)
        out.append(f)
    return out

def _kb1_all_collapsed(facts: list) -> bool:
    fit = [f for f in facts if f.get('fit')]
    return bool(fit) and all((f['collapsed'] for f in fit))

def _kb1_clears(facts: list) -> bool:
    return any((f.get('fit') and (not f['collapsed']) for f in facts))

def _kb1_spec_key(args, kw) -> tuple:

    def one(v):
        if isinstance(v, torch.Tensor):
            return ('T', str(v.dtype))
        if isinstance(v, (int, float, bool, str, type(None))):
            return v
        return str(v)
    return (tuple((one(v) for v in args)), tuple(sorted(((k, one(v)) for k, v in kw.items()))))
_SM_COUNT = {}

def _sm_count(device) -> int:
    idx = getattr(device, 'index', None)
    if idx is None:
        idx = torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]
_KB1_HB_RO_MAX = int(_env('BWD_KB1_HB_RO_MAX', '16'))
_KB1_HB_RO_MAX_IEEE = int(_env('BWD_KB1_HB_RO_MAX_IEEE', '1'))
_KB1_HB_TN_MAX = int(_env('BWD_KB1_HB_TN_MAX', '8192'))
_ROT_ABL_MIMO = int(_env('BWD_ROT_ABL_MIMO', '0'))

def _kb1_hb(H, hpg_b, hpg_c, Bsz, nc, NGT, nsplit, refused, device):
    if refused:
        return (0, False, False)
    hpg = max(1, min(hpg_b, hpg_c))
    divs = [d for d in range(hpg, 0, -1) if hpg % d == 0]
    env = _env('BWD_KB1_HB', 'auto')
    if env != 'auto':
        hb = int(env)
        if hb <= 0:
            return (0, False, False)
        if hpg % hb:
            raise ValueError('SSD_MIMO_BWD_KB1_HB=%d does not divide the heads per B/C group (%d): a head block that straddles a group would accumulate two different dBtil/dCtil tiles into one register file.' % (hb, hpg))
    else:
        target = 2 * _sm_count(device)
        hb = 1
        for d in divs:
            if Bsz * ((H + d - 1) // d) * nc * NGT * nsplit >= target:
                hb = d
                break
    db_store = bool(hb >= hpg_b and NGT == 1)
    dc_store = bool(hb >= hpg_c and NGT == 1)
    return (hb, db_store, dc_store)

def _hb_cfg(hb, w, st, mnr):
    kw = dict(num_warps=w, num_stages=st)
    if mnr:
        kw['maxnreg'] = mnr
    return triton.Config({'HB': hb}, **kw)
_HB_CFGS = [_hb_cfg(32, 8, 1, 0)] if _AUTOTUNE_OFF else [_hb_cfg(hb, w, st, m) for hb in _HB_VALS for w in _BWD_WARPS for st in _BWD_STAGES for m in _HB_MAXNREG]

def _hb_grid(tier):
    if tier == 'pinned':
        return [_hb_cfg(32, 8, 1, 0)]
    w, st, _ = _TIER_AXES[tier]
    return [_hb_cfg(hb, a, b, m) for hb in (int(v) for v in _env('BWD_HB_CANDS', '8,32').split(',')) for a in (int(v) for v in _env('BWD_WARPS', w).split(',')) for b in (int(v) for v in _env('BWD_STAGES', st).split(',')) for m in _HB_MAXNREG]

@triton.autotune(configs=_calib_grid('fused_kbc_hb', _hb_grid), prune_configs_by={'early_config_prune': _cprune('fused_kbc_hb')}, key=['T', 'N', 'G_SIZE', 'GT', 'PREC', 'MMA_IO', 'HAS_BETA', 'HAS_GAMMA', 'HAS_DW', 'HAS_THETA_H', 'HAS_ROT_BIAS', 'HAS_ROT', 'ROT_PER_HEAD', 'BIAS_PRE_ROT', 'PAIR_HALF', 'SCHED', 'SUMDS', 'KBC_DALPHA', 'DD_PAIRS', 'KBC_PART', 'TS_ABL', 'H', 'NC', 'BSZ', 'IDX64'], restore_value=['dCtil_ptr', 'dBtil_ptr', 'dDelta_ptr', 'dBeta_ptr', 'dGamma_ptr', 'dAlpha_ptr', 'dThetaH_ptr', 'dRotBias_ptr', 'dTheta_ptr', 'dThOff_ptr'])
@triton.jit
def _bwd_kbc_hb_kernel(C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, SAll_ptr, dG_ptr, DW_ptr, Beta_ptr, Gamma_ptr, ThetaH_ptr, RotBias_ptr, Theta_ptr, dY_ptr, dCtil_ptr, dBtil_ptr, dDelta_ptr, dBeta_ptr, dGamma_ptr, dThetaH_ptr, dRotBias_ptr, dTheta_ptr, H, BSZ, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg, s_wb, s_wh, s_wl, s_dcb, s_dcc, s_dct, s_dcn, s_dbb, s_dbc, s_dbt, s_dbn, s_thb, s_thc, s_tht, s_thh, s_thk, s_dtb, s_dtc, s_dtt, s_dth, s_dtk, NC, NGT_val, NSPLIT: tl.constexpr, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, MMA_IO: tl.constexpr=0, HAS_BETA: tl.constexpr=False, HAS_GAMMA: tl.constexpr=False, HAS_DW: tl.constexpr=False, HAS_THETA_H: tl.constexpr=False, HAS_ROT_BIAS: tl.constexpr=False, HAS_ROT: tl.constexpr=False, ROT_PER_HEAD: tl.constexpr=False, BIAS_PRE_ROT: tl.constexpr=False, HB: tl.constexpr=8, IDX64: tl.constexpr=False, ROT_DIET: tl.constexpr=True, ROT_ABL: tl.constexpr=0, BRot_ptr=None, CRot_ptr=None, s_rb=0, s_rc=0, s_rt=0, s_rh=0, s_rn=0, BSLOT: tl.constexpr=1, s_bes=0, s_gas=0, ThOff_ptr=None, dThOff_ptr=None, s_ob=0, s_oh=0, s_oc=0, s_ok=0, s_dob=0, s_doh=0, s_doc=0, s_dok=0, PHASE_CUMSUM: tl.constexpr=False, PAIR_HALF: tl.constexpr=False, PHASE_DEFER: tl.constexpr=False, SCHED: tl.constexpr=0, DCS_INK: tl.constexpr=False, TRAP_DIAG: tl.constexpr=False, SUMDS: tl.constexpr=False, KBC_DALPHA: tl.constexpr=False, dAlpha_ptr=None, s_dab=0, s_dah=0, s_dag=0, s_dac=0, DD_PAIRS: tl.constexpr=False, CB_ptr=None, s_cbb=0, s_cbc=0, KBC_PART: tl.constexpr=0, TS_ABL: tl.constexpr=0):
    tl.static_assert(not BIAS_PRE_ROT or (ROT_PER_HEAD and (not HAS_ROT_BIAS)), 'Kbc-hb: bias_pre_rot needs a per-head in-kernel phase and is refused with rot_bias (theta_h, a per-head READ angle, rides the per-head tile like the phase does)')
    SCHED_DM: tl.constexpr = SCHED.value & 1 != 0
    SCHED_SF: tl.constexpr = SCHED.value & 2 != 0
    SCHED_XR: tl.constexpr = SCHED.value & 8 != 0
    SCHED_AG: tl.constexpr = SCHED.value & 16 != 0
    b = tl.program_id(0)
    hblk = tl.program_id(1)
    NS: tl.constexpr = N // NSPLIT
    pid2 = tl.program_id(2)
    ns_idx = pid2 % NSPLIT
    rest = pid2 // NSPLIT
    c = rest // NGT_val
    gt_idx = rest % NGT_val
    if IDX64:
        b = b.to(tl.int64)
    offs_t = tl.arange(0, T)
    offs_n = ns_idx * NS + tl.arange(0, NS)
    offs_k = ns_idx * (NS // 2) + tl.arange(0, NS // 2)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    if TRAP_DIAG:
        causal = offs_t[:, None] > offs_t[None, :]
    else:
        causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float('-inf')
    if PAIR_HALF:
        tl.static_assert(not HAS_ROT_BIAS, 'PAIR_HALF does not serve the retired rot_bias')
        tl.static_assert(not HAS_ROT or ROT_PER_HEAD, 'PAIR_HALF serves the per-head phase (the SSD rotation law); a head-shared phase keeps the interleaved branch')
        tl.static_assert(ROT_ABL == 0, 'PAIR_HALF carries no ablation ladder')
        tl.static_assert(N >= 32 * NSPLIT, 'PAIR_HALF needs N/(2*NSPLIT) >= 16 (MMA K floor)')
        PS: tl.constexpr = N // (2 * NSPLIT)
        KH: tl.constexpr = N // 2
        offs_p = ns_idx * PS + tl.arange(0, PS)
        offs_h1 = offs_p
        offs_h2 = KH + offs_p
        c_base = C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct
        b_base = B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt
        Co1_base = tl.load(c_base + offs_h1[None, :] * s_cn).to(tl.float32)
        Co2_base = tl.load(c_base + offs_h2[None, :] * s_cn).to(tl.float32)
        Bc1_base = tl.load(b_base + offs_h1[None, :] * s_bn).to(tl.float32)
        Bc2_base = tl.load(b_base + offs_h2[None, :] * s_bn).to(tl.float32)
        dth_sole = NGT_val == 1
        dB1_acc = tl.zeros([T, PS], dtype=tl.float32)
        dB2_acc = tl.zeros([T, PS], dtype=tl.float32)
        dC1_acc = tl.zeros([T, PS], dtype=tl.float32)
        dC2_acc = tl.zeros([T, PS], dtype=tl.float32)
        h_start = hblk * HB
        hb_count = tl.minimum(HB, H - h_start)
        for hh in range(hb_count):
            h = h_start + hh
            if IDX64:
                h = h.to(tl.int64)
            a_val = -tl.exp(tl.load(A_ptr + h))
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            Bs1 = Bc1_base
            Bs2 = Bc2_base
            Cs1 = Co1_base
            Cs2 = Co2_base
            if HAS_BETA:
                be1 = tl.load(Beta_ptr + h * N + offs_h1).to(tl.float32)
                be2 = tl.load(Beta_ptr + h * N + offs_h2).to(tl.float32)
            if HAS_GAMMA:
                ga1 = tl.load(Gamma_ptr + h * N + offs_h1).to(tl.float32)
                ga2 = tl.load(Gamma_ptr + h * N + offs_h2).to(tl.float32)
            if BIAS_PRE_ROT:
                if HAS_BETA:
                    Bs1 = Bs1 + be1[None, :]
                    Bs2 = Bs2 + be2[None, :]
                if HAS_GAMMA:
                    Cs1 = Cs1 + ga1[None, :]
                    Cs2 = Cs2 + ga2[None, :]
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_p[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_p * s_ok).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                Bt1 = ph_cos * Bs1 + ph_sin * Bs2
                Bt2 = ph_cos * Bs2 - ph_sin * Bs1
                Ct1 = ph_cos * Cs1 + ph_sin * Cs2
                Ct2 = ph_cos * Cs2 - ph_sin * Cs1
            else:
                Bt1 = Bs1
                Bt2 = Bs2
                Ct1 = Cs1
                Ct2 = Cs2
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * KH + offs_p)
                ch_h = _cos_approx(th_h)
                sh_h = _sin_approx(th_h)
            if HAS_THETA_H:
                Bop1 = ch_h[None, :] * Bt1 - sh_h[None, :] * Bt2
                Bop2 = ch_h[None, :] * Bt2 + sh_h[None, :] * Bt1
            else:
                Bop1 = Bt1
                Bop2 = Bt2
            Bdd1 = Bt1
            Bdd2 = Bt2
            if HAS_BETA and (not BIAS_PRE_ROT):
                if HAS_THETA_H:
                    Bop1 = Bop1 + (ch_h * be1 - sh_h * be2)[None, :]
                    Bop2 = Bop2 + (ch_h * be2 + sh_h * be1)[None, :]
                else:
                    Bop1 = Bop1 + be1[None, :]
                    Bop2 = Bop2 + be2[None, :]
                Bdd1 = Bdd1 + be1[None, :]
                Bdd2 = Bdd2 + be2[None, :]
            Ceff1 = Ct1
            Ceff2 = Ct2
            if HAS_GAMMA and (not BIAS_PRE_ROT):
                Ceff1 = Ceff1 + ga1[None, :]
                Ceff2 = Ceff2 + ga2[None, :]
            if HAS_THETA_H:
                Cop1 = ch_h[None, :] * Ceff1 + sh_h[None, :] * Ceff2
                Cop2 = ch_h[None, :] * Ceff2 - sh_h[None, :] * Ceff1
            else:
                Cop1 = Ceff1
                Cop2 = Ceff2
            dY_h = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
            s_base = SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_gt[None, :] * s_sg
            S1 = tl.load(s_base + offs_h1[:, None] * s_sn).to(tl.float32)
            S2 = tl.load(s_base + offs_h2[:, None] * s_sn).to(tl.float32)
            if HAS_THETA_H:
                S_op1 = ch_h[:, None] * S1 - sh_h[:, None] * S2
                S_op2 = ch_h[:, None] * S2 + sh_h[:, None] * S1
            else:
                S_op1 = S1
                S_op2 = S2
            dC_state1 = f[:, None] * pdot(dY_h.to(MMA_DT), tl.trans(S_op1).to(MMA_DT), input_precision=PREC)
            dC_state2 = f[:, None] * pdot(dY_h.to(MMA_DT), tl.trans(S_op2).to(MMA_DT), input_precision=PREC)
            x_base = X_ptr + b * s_xb + h * s_xh + c * T * s_xl
            x_s = tl.load(x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                x_s = x_s * dw[:, None]
            dScore = pdot(dY_h.to(MMA_DT), tl.trans(x_s).to(MMA_DT), input_precision=PREC)
            dSb = dScore * dec_mask
            dSb_dt = dSb.to(MMA_DT)
            dSb_tr = tl.trans(dSb_dt)
            dC_h1 = pdot(dSb_dt, Bop1.to(MMA_DT), input_precision=PREC) + dC_state1
            dC_h2 = pdot(dSb_dt, Bop2.to(MMA_DT), input_precision=PREC) + dC_state2
            dB_h1 = pdot(dSb_tr, Cop1.to(MMA_DT), input_precision=PREC)
            dB_h2 = pdot(dSb_tr, Cop2.to(MMA_DT), input_precision=PREC)
            if HAS_THETA_H:
                dD = a_val * (tl.sum(Ceff1 * dC_state1, axis=1) + tl.sum(Ceff2 * dC_state2, axis=1))
            else:
                dD = a_val * (tl.sum(Cop1 * dC_state1, axis=1) + tl.sum(Cop2 * dC_state2, axis=1))
            dD = tl.cumsum(dD, axis=0, reverse=True)
            g_base = dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_gt[None, :] * s_gg
            dG1 = tl.load(g_base + offs_h1[:, None] * s_gn).to(tl.float32)
            dG2 = tl.load(g_base + offs_h2[:, None] * s_gn).to(tl.float32)
            xw = x_s * dec_agg[:, None]
            dB_h1 += pdot(xw.to(MMA_DT), tl.trans(dG1).to(MMA_DT), input_precision=PREC)
            dB_h2 += pdot(xw.to(MMA_DT), tl.trans(dG2).to(MMA_DT), input_precision=PREC)
            if HAS_BETA and (not BIAS_PRE_ROT):
                tl.atomic_add(dBeta_ptr + h * N + offs_h1, tl.sum(dB_h1, axis=0), sem=ACC_SEM)
                tl.atomic_add(dBeta_ptr + h * N + offs_h2, tl.sum(dB_h2, axis=0), sem=ACC_SEM)
            if HAS_GAMMA and (not BIAS_PRE_ROT):
                tl.atomic_add(dGamma_ptr + h * N + offs_h1, tl.sum(dC_h1, axis=0), sem=ACC_SEM)
                tl.atomic_add(dGamma_ptr + h * N + offs_h2, tl.sum(dC_h2, axis=0), sem=ACC_SEM)
            if HAS_THETA_H:
                tl.atomic_add(dThetaH_ptr + h * KH + offs_p, tl.sum(dC_h1 * Ceff2 - dC_h2 * Ceff1, axis=0), sem=ACC_SEM)
            tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
            if HAS_ROT:
                Cq1 = Ct1
                Cq2 = Ct2
                if HAS_THETA_H:
                    Cq1, Cq2 = _ph_read_consistent(ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2, ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1, ch_h, sh_h, Ct1, Ct2, MMA_DT, MMA_IO, HAS_THETA_H, PREC)
                else:
                    Cq1 = _opv(Ct1, MMA_DT, MMA_IO, PREC)
                    Cq2 = _opv(Ct2, MMA_DT, MMA_IO, PREC)
                dth_ph = dB_h1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - dB_h2 * _opv(Bt1, MMA_DT, MMA_IO, PREC) + dC_h1 * Cq2 - dC_h2 * Cq1
                dtp = dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_p[None, :] * s_dtk
                if PHASE_CUMSUM and (not PHASE_DEFER):
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    dop = dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_p * s_dok
                    if dth_sole:
                        tl.store(dop, dth_off)
                    else:
                        tl.atomic_add(dop, dth_off, sem=ACC_SEM)
                if dth_sole:
                    tl.store(dtp, dth_ph)
                else:
                    tl.atomic_add(dtp, dth_ph, sem=ACC_SEM)
                nB1 = ph_cos * dB_h1 - ph_sin * dB_h2
                dB_h2 = ph_cos * dB_h2 + ph_sin * dB_h1
                dB_h1 = nB1
                nC1 = ph_cos * dC_h1 - ph_sin * dC_h2
                dC_h2 = ph_cos * dC_h2 + ph_sin * dC_h1
                dC_h1 = nC1
            if BIAS_PRE_ROT:
                bslot = c % BSLOT
                if HAS_BETA:
                    tl.atomic_add(dBeta_ptr + bslot * s_bes + h * N + offs_h1, tl.sum(dB_h1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + bslot * s_bes + h * N + offs_h2, tl.sum(dB_h2, axis=0), sem=ACC_SEM)
                if HAS_GAMMA:
                    tl.atomic_add(dGamma_ptr + bslot * s_gas + h * N + offs_h1, tl.sum(dC_h1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dGamma_ptr + bslot * s_gas + h * N + offs_h2, tl.sum(dC_h2, axis=0), sem=ACC_SEM)
            dB1_acc += dB_h1
            dB2_acc += dB_h2
            dC1_acc += dC_h1
            dC2_acc += dC_h2
        sole_owner = (NGT_val == 1) & (h_start == 0) & (hb_count >= H)
        db_p = dBtil_ptr + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt
        dc_p = dCtil_ptr + b * s_dcb + c * s_dcc + offs_t[:, None] * s_dct
        if sole_owner:
            tl.store(db_p + offs_h1[None, :] * s_dbn, dB1_acc)
            tl.store(db_p + offs_h2[None, :] * s_dbn, dB2_acc)
            tl.store(dc_p + offs_h1[None, :] * s_dcn, dC1_acc)
            tl.store(dc_p + offs_h2[None, :] * s_dcn, dC2_acc)
        else:
            tl.atomic_add(db_p + offs_h1[None, :] * s_dbn, dB1_acc, sem=ACC_SEM)
            tl.atomic_add(db_p + offs_h2[None, :] * s_dbn, dB2_acc, sem=ACC_SEM)
            tl.atomic_add(dc_p + offs_h1[None, :] * s_dcn, dC1_acc, sem=ACC_SEM)
            tl.atomic_add(dc_p + offs_h2[None, :] * s_dcn, dC2_acc, sem=ACC_SEM)
    else:
        ROT: tl.constexpr = HAS_THETA_H or HAS_ROT_BIAS
        ROT_ANY: tl.constexpr = ROT or HAS_ROT
        if ROT_ANY:
            pair_idx = offs_n // 2
            fwd_sign = tl.where(offs_n % 2 == 0, 1.0, -1.0)
        c_base = C_ptr + b * s_cb + c * s_cc
        b_base = B_ptr + b * s_bb + c * s_bc
        Co_base = tl.load(c_base + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
        Bc_base = tl.load(b_base + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
        if HAS_ROT and (not ROT_PER_HEAD):
            th_rot = tl.load(Theta_ptr + b * s_thb + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk).to(tl.float32)
            if PHASE_CUMSUM:
                th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + c * s_oc + offs_k * s_ok).to(tl.float32)[None, :]
            cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, NS)
            sn_rot = fwd_sign[None, :] * _pair_bcast_2d(_sin_approx(th_rot), T, NS)
            Co_base = cs_rot * Co_base + sn_rot * _pair_swap_2d(Co_base, T, NS)
            Bc_base = cs_rot * Bc_base + sn_rot * _pair_swap_2d(Bc_base, T, NS)
        if ROT and (not ROT_PER_HEAD):
            if HAS_ROT:
                Co_xor = _pair_swap_2d(Co_base, T, NS)
                Bc_xor = _pair_swap_2d(Bc_base, T, NS)
            else:
                Co_xor = tl.load(c_base + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                Bc_xor = tl.load(b_base + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
        dth_sole = NGT_val == 1
        dB_acc = tl.zeros([T, NS], dtype=tl.float32)
        dC_acc = tl.zeros([T, NS], dtype=tl.float32)
        if KBC_PART != 0:
            tl.static_assert(not ROT_ANY and (not ROT_PER_HEAD) and (not SUMDS) and (not BIAS_PRE_ROT) and (not KBC_DALPHA) and (not DD_PAIRS) and (ROT_ABL == 0), 'KBC_PART: the unrotated fused body only')
        if SUMDS:
            tl.static_assert(not ROT_ANY and (not ROT_PER_HEAD) and (not BIAS_PRE_ROT) and (ROT_ABL == 0), 'SUMDS: head-shared unrotated B/C only (the right operand of the score contractions must be the same for every head)')
            dS_sum = tl.zeros([T, T], dtype=tl.float32)
        h_start = hblk * HB
        hb_count = tl.minimum(HB, H - h_start)
        for hh in range(hb_count):
            h = h_start + hh
            if IDX64:
                h = h.to(tl.int64)
            a_val = -tl.exp(tl.load(A_ptr + h))
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            if not SCHED_DM:
                dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            if not SCHED_AG:
                D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if ROT_PER_HEAD and ROT_ABL & 384 != 0:
                rbase = b * s_rb + c * s_rc + h * s_rh + offs_t[:, None] * s_rt + offs_n[None, :] * s_rn
                Co_cur = tl.load(CRot_ptr + rbase).to(tl.float32)
                if ROT_ABL & 256 != 0:
                    Bc_cur = tl.load(BRot_ptr + rbase).to(tl.float32)
            if ROT_PER_HEAD and ROT_ABL & 256 == 0:
                if False:
                    th_rot = tl.zeros([T, NS // 2], dtype=tl.float32)
                else:
                    th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM and ROT_ABL & 1 == 0:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_k * s_ok).to(tl.float32)[None, :]
                if ROT_ABL & 2 != 0:
                    ph_cos = th_rot * 0.0 + 1.0
                    ph_sin = th_rot * 0.0
                else:
                    ph_cos = _cos_approx(th_rot)
                    ph_sin = _sin_approx(th_rot)
                cs_rot = _pair_bcast_2d(ph_cos, T, NS)
                sn_rot = fwd_sign[None, :] * _pair_bcast_2d(ph_sin, T, NS)
                Bc_src = Bc_base
                Co_src = Co_base
                if BIAS_PRE_ROT:
                    if HAS_BETA:
                        Bc_src = Bc_src + tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)[None, :]
                    if HAS_GAMMA:
                        Co_src = Co_src + tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)[None, :]
                if ROT_ABL & 16 != 0:
                    Bc_cur = Bc_src
                    Co_cur = Co_src
                    if ROT or not ROT_DIET:
                        Bc_xor = _pair_swap_2d(Bc_cur, T, NS)
                        Co_xor = _pair_swap_2d(Co_cur, T, NS)
                else:
                    B_sw = _pair_swap_2d(Bc_src, T, NS)
                    Bc_cur = cs_rot * Bc_src + sn_rot * B_sw
                    if ROT or not ROT_DIET:
                        Bc_xor = cs_rot * B_sw - sn_rot * Bc_src
                    if ROT_ABL & 128 == 0:
                        C_sw = _pair_swap_2d(Co_src, T, NS)
                        Co_cur = cs_rot * Co_src + sn_rot * C_sw
                        if ROT or not ROT_DIET:
                            Co_xor = cs_rot * C_sw - sn_rot * Co_src
                    elif ROT or not ROT_DIET:
                        Co_xor = _pair_swap_2d(Co_cur, T, NS)
            else:
                Bc_cur = Bc_base
                Co_cur = Co_base
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
                ch_h = _cos_approx(th_h)
                fc_h = fwd_sign * _sin_approx(th_h)
            if HAS_ROT_BIAS:
                rb_h = tl.load(RotBias_ptr + h * (N // 2) + pair_idx)
                crb_h = _cos_approx(rb_h)
                sfb_h = fwd_sign * _sin_approx(rb_h)
            if HAS_THETA_H and HAS_ROT_BIAS:
                p1 = ch_h * crb_h + fc_h * sfb_h
                p2 = ch_h * sfb_h - fc_h * crb_h
            elif HAS_THETA_H:
                p1 = ch_h
                p2 = -fc_h
            elif HAS_ROT_BIAS:
                p1 = crb_h
                p2 = sfb_h
            if ROT:
                Bop = p1[None, :] * Bc_cur + p2[None, :] * Bc_xor
            else:
                Bop = Bc_cur
            Bdd = Bc_cur
            if HAS_BETA and (not BIAS_PRE_ROT):
                beta_v = tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)
                if ROT:
                    beta_x = tl.load(Beta_ptr + h * N + (offs_n ^ 1)).to(tl.float32)
                if HAS_THETA_H:
                    Bop = Bop + (ch_h * beta_v - fc_h * beta_x)[None, :]
                else:
                    Bop = Bop + beta_v[None, :]
                if HAS_ROT_BIAS:
                    Bdd = Bdd + (crb_h * beta_v - sfb_h * beta_x)[None, :]
                else:
                    Bdd = Bdd + beta_v[None, :]
            Ceff = Co_cur
            if ROT:
                Ceff_x = Co_xor
            if HAS_GAMMA and (not BIAS_PRE_ROT):
                gam = tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)
                Ceff = Ceff + gam[None, :]
                if ROT:
                    gam_x = tl.load(Gamma_ptr + h * N + (offs_n ^ 1)).to(tl.float32)
                    Ceff_x = Ceff_x + gam_x[None, :]
            if ROT:
                Cop = p1[None, :] * Ceff - p2[None, :] * Ceff_x
            else:
                Cop = Ceff
            dY_h = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
            s_base = SAll_ptr + b * s_sb + h * s_sh + c * s_sc
            if TS_ABL & 1:
                S_c = tl.load(s_base + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg, mask=(offs_n[:, None] >= 0) & (c % 2 == 0), other=0.0).to(tl.float32)
            else:
                S_c = tl.load(s_base + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg).to(tl.float32)
            if TS_ABL & 2:
                if c % 2 == 1:
                    x_pv = tl.load(X_ptr + b * s_xb + h * s_xh + (c - 1) * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(MMA_DT)
                    S_c = S_c * 0.5 + pdot(tl.trans(Bop.to(MMA_DT)), x_pv, input_precision=PREC)
            if HAS_THETA_H:
                S_x = tl.load(s_base + (offs_n ^ 1)[:, None] * s_sn + offs_gt[None, :] * s_sg).to(tl.float32)
                S_op = ch_h[:, None] * S_c - fc_h[:, None] * S_x
            else:
                S_op = S_c
            if SCHED_SF:
                dC_state = f[:, None] * pdot(dY_h.to(MMA_DT), tl.trans(S_op).to(MMA_DT), input_precision=PREC)
            x_base = X_ptr + b * s_xb + h * s_xh + c * T * s_xl
            x_s = tl.load(x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                x_s = x_s * dw[:, None]
            if not SCHED_SF:
                dC_state = f[:, None] * pdot(dY_h.to(MMA_DT), tl.trans(S_op).to(MMA_DT), input_precision=PREC)
            dScore = pdot(dY_h.to(MMA_DT), tl.trans(x_s).to(MMA_DT), input_precision=PREC)
            if SCHED_DM:
                dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            dSb = dScore * dec_mask
            if DD_PAIRS:
                tl.static_assert(not ROT and (not ROT_PER_HEAD) and (not HAS_BETA) and (not HAS_GAMMA), 'DD_PAIRS: unrotated, bias-free (S0 = C B^T)')
                S0_dd = tl.load(CB_ptr + b * s_cbb + c * s_cbc + offs_t[:, None] * T + offs_t[None, :])
                dd_pairs = _span_sum(S0_dd * dSb, offs_t) * tl.where(ns_idx == 0, 1.0, 0.0)
            if SUMDS:
                dS_sum += dSb
                dC_h = dC_state
                dB_h = tl.zeros([T, NS], dtype=tl.float32)
                if HAS_BETA or HAS_GAMMA:
                    rs_h = tl.sum(dSb, axis=1)
                    cs_h = tl.sum(dSb, axis=0)
                if HAS_BETA:
                    dC_h += rs_h[:, None] * beta_v[None, :]
                if HAS_GAMMA:
                    dB_h += cs_h[:, None] * gam[None, :]
            else:
                dSb_dt = dSb.to(MMA_DT)
                dC_h = pdot(dSb_dt, Bop.to(MMA_DT), input_precision=PREC) + dC_state
                dB_h = pdot(tl.trans(dSb_dt), Cop.to(MMA_DT), input_precision=PREC)
            if ROT:
                dD = a_val * tl.sum(Ceff * dC_state, axis=1)
            else:
                dD = a_val * tl.sum(Cop * dC_state, axis=1)
            dD = tl.cumsum(dD, axis=0, reverse=True)
            if DD_PAIRS:
                dD += a_val * dd_pairs
            g_base = dG_ptr + b * s_gb + h * s_gh + c * s_gc
            if TS_ABL & 1:
                dG = tl.load(g_base + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg, mask=(offs_n[:, None] >= 0) & (c % 2 == 1), other=0.0).to(tl.float32)
            else:
                dG = tl.load(g_base + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg).to(tl.float32)
            if TS_ABL & 2:
                if c % 2 == 0:
                    dY_nx = tl.load(dY_ptr + b * s_yb + h * s_yh + tl.minimum(c + 1, NC - 1) * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(MMA_DT)
                    dG = dG * 0.5 + pdot(tl.trans(Cop.to(MMA_DT)), dY_nx, input_precision=PREC)
            if KBC_DALPHA:
                tl.static_assert(not ROT_ANY and (not ROT_PER_HEAD), 'KBC_DALPHA: unrotated state tiles only')
                tl.atomic_add(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac, tl.sum(dG * S_c), sem=ACC_SEM)
            if HAS_ROT_BIAS:
                dG_x = tl.load(g_base + (offs_n ^ 1)[:, None] * s_gn + offs_gt[None, :] * s_gg).to(tl.float32)
                dG_op = crb_h[:, None] * dG - sfb_h[:, None] * dG_x
            else:
                dG_op = dG
            if SCHED_AG:
                D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if SCHED_XR:
                x_a = tl.load(x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, volatile=True).to(tl.float32)
                if HAS_DW:
                    x_a = x_a * dw[:, None]
                xw = x_a * dec_agg[:, None]
            else:
                xw = x_s * dec_agg[:, None]
            dB_h += pdot(xw.to(MMA_DT), tl.trans(dG_op).to(MMA_DT), input_precision=PREC)
            if HAS_BETA and (not BIAS_PRE_ROT) and (KBC_PART != 1):
                v_beta = tl.sum(dB_h, axis=0)
                if SUMDS:
                    v_beta += tl.sum(rs_h[:, None] * Co_base, axis=0)
                if HAS_ROT_BIAS:
                    tl.atomic_add(dBeta_ptr + h * N + offs_n, crb_h * v_beta, sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + h * N + (offs_n ^ 1), -sfb_h * v_beta, sem=ACC_SEM)
                else:
                    tl.atomic_add(dBeta_ptr + h * N + offs_n, v_beta, sem=ACC_SEM)
            if HAS_GAMMA and (not BIAS_PRE_ROT) and (KBC_PART != 2):
                v_gam = tl.sum(dC_h, axis=0)
                if SUMDS:
                    v_gam += tl.sum(cs_h[:, None] * Bc_base, axis=0)
                tl.atomic_add(dGamma_ptr + h * N + offs_n, v_gam, sem=ACC_SEM)
            if HAS_THETA_H:
                tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx, tl.sum(fwd_sign[None, :] * dC_h * Ceff_x, axis=0), sem=ACC_SEM)
            if HAS_ROT_BIAS:
                tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx, tl.sum(fwd_sign[None, :] * dB_h * Bc_xor, axis=0), sem=ACC_SEM)
            if KBC_PART != 2:
                tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
            if ROT_PER_HEAD and ROT_ABL & 384 != 0:
                tl.store(CRot_ptr + rbase, dC_h.to(CRot_ptr.dtype.element_ty))
                if ROT_ABL & 256 != 0:
                    tl.store(BRot_ptr + rbase, dB_h.to(BRot_ptr.dtype.element_ty))
            if ROT_PER_HEAD and ROT_ABL & 260 == 0:
                if ROT_DIET and (not ROT):
                    Bc_xor = _pair_swap_2d(Bc_cur, T, NS)
                    Co_xor = _pair_swap_2d(Co_cur, T, NS)
                Cq_xor = Co_xor
                if (MMA_IO or PREC == 'tf32') and DTH_OPROUND:
                    if HAS_THETA_H and (not HAS_ROT_BIAS):
                        Cq_xor = ch_h[None, :] * _opv(ch_h[None, :] * Co_xor - fc_h[None, :] * Co_cur, MMA_DT, MMA_IO, PREC) + fc_h[None, :] * _opv(ch_h[None, :] * Co_cur + fc_h[None, :] * Co_xor, MMA_DT, MMA_IO, PREC)
                    else:
                        Cq_xor = _opv(Co_xor, MMA_DT, MMA_IO, PREC)
                dth_lane = dB_h * _opv(Bc_xor, MMA_DT, MMA_IO, PREC) + dC_h * Cq_xor
                dth_e, dth_o = tl.split(tl.reshape(dth_lane, (T, NS // 2, 2)))
                dth_ph = dth_e - dth_o
                dtp = dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_k[None, :] * s_dtk
                if PHASE_CUMSUM:
                    if not PHASE_DEFER:
                        dth_off = tl.sum(dth_ph, axis=0)
                        if ROT_ABL & 512 == 0:
                            dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        dop = dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_k * s_dok
                        if dth_sole:
                            tl.store(dop, dth_off)
                        else:
                            tl.atomic_add(dop, dth_off, sem=ACC_SEM)
                if ROT_ABL & 64 == 0:
                    if dth_sole:
                        tl.store(dtp, dth_ph)
                    else:
                        tl.atomic_add(dtp, dth_ph, sem=ACC_SEM)
            if ROT_PER_HEAD and ROT_ABL & 256 == 0:
                if ROT_ABL & 16 == 0:
                    if ROT_DIET:
                        cs_rot = _pair_bcast_2d(ph_cos, T, NS)
                        sn_rot = fwd_sign[None, :] * _pair_bcast_2d(ph_sin, T, NS)
                    dB_h = cs_rot * dB_h - sn_rot * _pair_swap_2d(dB_h, T, NS)
                    dC_h = cs_rot * dC_h - sn_rot * _pair_swap_2d(dC_h, T, NS)
                if BIAS_PRE_ROT and ROT_ABL & 8 == 0:
                    bslot = c % BSLOT
                    if HAS_BETA:
                        vb = tl.sum(dB_h, axis=0)
                        tl.atomic_add(dBeta_ptr + bslot * s_bes + h * N + offs_n, vb, sem=ACC_SEM)
                    if HAS_GAMMA:
                        vg = tl.sum(dC_h, axis=0)
                        tl.atomic_add(dGamma_ptr + bslot * s_gas + h * N + offs_n, vg, sem=ACC_SEM)
            if KBC_PART != 1:
                dB_acc += dB_h
            if ROT_ABL & 384 == 0 and KBC_PART != 2:
                dC_acc += dC_h
        if SUMDS:
            dS_dt = dS_sum.to(MMA_DT)
            dC_acc += pdot(dS_dt, Bc_base.to(MMA_DT), input_precision=PREC)
            dB_acc += pdot(tl.trans(dS_dt), Co_base.to(MMA_DT), input_precision=PREC)
        sole_owner = (NGT_val == 1) & (h_start == 0) & (hb_count >= H)
        if HAS_ROT and (not ROT_PER_HEAD):
            if ROT:
                Bx_r = Bc_xor
                Cx_r = Co_xor
            else:
                Bx_r = _pair_swap_2d(Bc_base, T, NS)
                Cx_r = _pair_swap_2d(Co_base, T, NS)
            if not HAS_THETA_H:
                Cx_r = _opv(Cx_r, MMA_DT, MMA_IO, PREC)
            dth_lane = fwd_sign[None, :] * (dB_acc * _opv(Bx_r, MMA_DT, MMA_IO, PREC) + dC_acc * Cx_r)
            dth_e, dth_o = tl.split(tl.reshape(dth_lane, (T, NS // 2, 2)))
            dth = dth_e + dth_o
            dB_acc = cs_rot * dB_acc - sn_rot * _pair_swap_2d(dB_acc, T, NS)
            dC_acc = cs_rot * dC_acc - sn_rot * _pair_swap_2d(dC_acc, T, NS)
            dt_ptr = dTheta_ptr + b * s_dtb + c * s_dtc + offs_t[:, None] * s_dtt + offs_k[None, :] * s_dtk
            if PHASE_CUMSUM:
                if not PHASE_DEFER:
                    dth_off = tl.sum(dth, axis=0)
                    dth = tl.cumsum(dth, axis=0, reverse=True)
                    dop = dThOff_ptr + b * s_dob + c * s_doc + offs_k * s_dok
                    if sole_owner:
                        tl.store(dop, dth_off)
                    else:
                        tl.atomic_add(dop, dth_off, sem=ACC_SEM)
            if sole_owner:
                tl.store(dt_ptr, dth)
            else:
                tl.atomic_add(dt_ptr, dth, sem=ACC_SEM)
        if sole_owner:
            if KBC_PART != 1:
                tl.store(dBtil_ptr + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn, dB_acc)
            if KBC_PART != 2:
                tl.store(dCtil_ptr + b * s_dcb + c * s_dcc + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn, dC_acc)
        else:
            if KBC_PART != 1:
                tl.atomic_add(dBtil_ptr + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn, dB_acc, sem=ACC_SEM)
            if KBC_PART != 2:
                tl.atomic_add(dCtil_ptr + b * s_dcb + c * s_dcc + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn, dC_acc, sem=ACC_SEM)

@triton.autotune(configs=_calib_grid('fused_kb2', _bwd_grid), prune_configs_by={'early_config_prune': _cprune('fused_kb2')}, key=['T', 'N', 'G_SIZE', 'GT', 'PREC', 'MMA_IO', 'HAS_BETA', 'HAS_ROT_BIAS', 'HAS_ROT', 'BIAS_PRE_ROT', 'PHASE_CUMSUM', 'R', 'HAS_DW', 'IDX64', 'ROT_ABL', 'PAIR_HALF', 'OCC'], restore_value=['dBtil_ptr', 'dDelta_ptr', 'dBeta_ptr', 'dRotBias_ptr', 'dTheta_ptr', 'dThOff_ptr'])
@triton.jit
def _bwd_kb2_kernel(B_ptr, X_ptr, Delta_ptr, A_ptr, Beta_ptr, RotBias_ptr, dG_ptr, dAlpha_ptr, dBtil_ptr, dDelta_ptr, dBeta_ptr, dRotBias_ptr, H, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xr, s_xl, s_db, s_dh, s_dc, s_dt, s_gb, s_gh, s_gc, s_gn, s_gg, s_dab, s_dah, s_dag, s_dac, s_dbb, s_dbc, s_dbt, s_dbn, s_bh, NC, NGT_val, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, MMA_IO: tl.constexpr=0, HAS_BETA: tl.constexpr=False, HAS_ROT_BIAS: tl.constexpr=False, HPG_B: tl.constexpr=1, R: tl.constexpr=1, s_br: tl.constexpr=0, s_dbr: tl.constexpr=0, R_TILE: tl.constexpr=False, CHUNK_PARALLEL: tl.constexpr=False, DW_ptr=None, s_wb=0, s_wh=0, s_wl=0, HAS_DW: tl.constexpr=False, s_dbh=0, IDX64: tl.constexpr=False, Theta_ptr=None, s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0, dTheta_ptr=None, s_dtb=0, s_dth=0, s_dtc=0, s_dtt=0, s_dtk=0, HAS_ROT: tl.constexpr=False, BIAS_PRE_ROT: tl.constexpr=False, ThOff_ptr=None, s_ob=0, s_oh=0, s_oc=0, s_ok=0, dThOff_ptr=None, s_dob=0, s_doh=0, s_doc=0, s_dok=0, PHASE_CUMSUM: tl.constexpr=False, PHASE_DEFER: tl.constexpr=False, ROT_ABL: tl.constexpr=0, PAIR_HALF: tl.constexpr=False, DCS_INK: tl.constexpr=False, OCC=0):
    ABL_NO_DTH: tl.constexpr = ROT_ABL & 1 != 0
    ABL_PREROT: tl.constexpr = ROT_ABL & 2 != 0
    b = tl.program_id(0)
    h = tl.program_id(1)
    if CHUNK_PARALLEL:
        if R_TILE:
            c = tl.program_id(2) // (NGT_val * R)
            remainder = tl.program_id(2) % (NGT_val * R)
            r_fixed = remainder // NGT_val
            gt_idx = remainder % NGT_val
        else:
            c = tl.program_id(2) // NGT_val
            gt_idx = tl.program_id(2) % NGT_val
    elif R_TILE:
        r_fixed = tl.program_id(2) // NGT_val
        gt_idx = tl.program_id(2) % NGT_val
    else:
        gt_idx = tl.program_id(2)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    bg = h // HPG_B
    ROT_ANY_KB2: tl.constexpr = HAS_ROT_BIAS or HAS_ROT
    if ROT_ANY_KB2:
        pair_idx_kb2 = offs_n // 2
        rot_sign_kb2 = tl.where(offs_n % 2 == 0, 1.0, -1.0)
    if HAS_ROT_BIAS:
        rb_full_kb2 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_kb2)
        crb_kb2 = _cos_approx(rb_full_kb2)
        srb_fwd_kb2 = rot_sign_kb2 * _sin_approx(rb_full_kb2)
    nc_end_kb2 = 1 if CHUNK_PARALLEL else NC
    if PAIR_HALF:
        tl.static_assert(not HAS_ROT_BIAS, 'PAIR_HALF does not serve the retired rot_bias')
        tl.static_assert(ROT_ABL == 0, 'PAIR_HALF carries no ablation ladder')
        tl.static_assert(N >= 32, 'PAIR_HALF needs N/2 >= 16 (the MMA K floor)')
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        R_ITERS: tl.constexpr = 1 if R_TILE else R
        for ci in range(nc_end_kb2):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_h1 * s_ok).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                dth_acc = tl.zeros([T, KH], dtype=tl.float32)
            dg_base = dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_gt[None, :] * s_gg
            dG1_dt = tl.load(dg_base + offs_h1[:, None] * s_gn).to(tl.float32).to(MMA_DT)
            dG2_dt = tl.load(dg_base + offs_h2[:, None] * s_gn).to(tl.float32).to(MMA_DT)
            q = tl.zeros([T], dtype=tl.float32)
            for rr in tl.range(R_ITERS, loop_unroll_factor=1):
                if R_TILE:
                    r = r_fixed
                else:
                    r = rr
                b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + r * s_br + offs_t[:, None] * s_bt
                B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
                B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
                if HAS_BETA:
                    be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r).to(tl.float32)
                    be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r).to(tl.float32)
                    if BIAS_PRE_ROT:
                        B1 = B1 + be1[None, :]
                        B2 = B2 + be2[None, :]
                if HAS_ROT:
                    Bt1 = ph_cos * B1 + ph_sin * B2
                    Bt2 = ph_cos * B2 - ph_sin * B1
                else:
                    Bt1 = B1
                    Bt2 = B2
                if HAS_BETA and (not BIAS_PRE_ROT):
                    Bf1 = Bt1 + be1[None, :]
                    Bf2 = Bt2 + be2[None, :]
                else:
                    Bf1 = Bt1
                    Bf2 = Bt2
                x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                if HAS_DW:
                    x_r = x_r * dw[:, None]
                xw_dt = (x_r * dec_agg[:, None]).to(MMA_DT)
                gB1 = pdot(xw_dt, tl.trans(dG1_dt), input_precision=PREC)
                gB2 = pdot(xw_dt, tl.trans(dG2_dt), input_precision=PREC)
                if HAS_BETA and (not BIAS_PRE_ROT):
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), sem=ACC_SEM)
                if HAS_ROT:
                    dth_acc += gB1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1, MMA_DT, MMA_IO, PREC)
                    gn1 = ph_cos * gB1 - ph_sin * gB2
                    gB2 = ph_cos * gB2 + ph_sin * gB1
                    gB1 = gn1
                if HAS_BETA and BIAS_PRE_ROT:
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), sem=ACC_SEM)
                db_base_r = dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                tl.atomic_add(db_base_r + offs_h1[None, :] * s_dbn, gB1, sem=ACC_SEM)
                tl.atomic_add(db_base_r + offs_h2[None, :] * s_dbn, gB2, sem=ACC_SEM)
                BdG_r = pdot(Bf1.to(MMA_DT), dG1_dt, input_precision=PREC) + pdot(Bf2.to(MMA_DT), dG2_dt, input_precision=PREC)
                q += tl.sum(BdG_r * x_r, axis=1)
            if HAS_ROT:
                dth_ph = dth_acc
                if PHASE_CUMSUM and (not PHASE_DEFER):
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_h1 * s_dok, dth_off, sem=ACC_SEM)
                tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_h1[None, :] * s_dtk, dth_ph, sem=ACC_SEM)
            dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
            if R_TILE:
                dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
            else:
                dalpha_gate = 1.0
            dD = a_val * (_excl_prefix(dec_agg * q, offs_t) + alpha * dalpha_c * dalpha_gate)
            tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
    else:
        for ci in range(nc_end_kb2):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + tl.arange(0, N // 2)[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + tl.arange(0, N // 2) * s_ok).to(tl.float32)[None, :]
                cs_rot = _pair_bcast_2d(_cos_approx(th_rot), T, N)
                sn_rot = rot_sign_kb2[None, :] * _pair_bcast_2d(_sin_approx(th_rot), T, N)
                dth_acc = tl.zeros([T, N], dtype=tl.float32)
            dG = tl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg).to(tl.float32)
            if R_TILE:
                Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
                if BIAS_PRE_ROT and HAS_BETA:
                    Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)[None, :]
                if HAS_ROT and (not ABL_PREROT):
                    B_til = cs_rot * Bc_r + sn_rot * _pair_swap_2d(Bc_r, T, N)
                else:
                    B_til = Bc_r
                if HAS_ROT_BIAS:
                    if HAS_ROT:
                        Bc_r_xor_rb = _pair_swap_2d(B_til, T, N)
                    else:
                        Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
                    B_unrot_rt = B_til
                    B_unrot_xor_rt = Bc_r_xor_rb
                    Bc_r = crb_kb2[None, :] * B_til + srb_fwd_kb2[None, :] * Bc_r_xor_rb
                else:
                    Bc_r = B_til
                if HAS_BETA and (not BIAS_PRE_ROT):
                    beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)
                    Bc_r = Bc_r + beta_r[None, :]
                x_base_r = b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
                x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                if HAS_DW:
                    x_r = x_r * dw[:, None]
                xw_r = x_r * dec_agg[:, None]
                dG_dt = dG.to(MMA_DT)
                dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
                if HAS_ROT:
                    gB_ = dB_r_agg
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(gB_, axis=0), sem=ACC_SEM)
                    if HAS_ROT_BIAS:
                        Brd_p_rt = crb_kb2[None, :] * B_unrot_xor_rt - srb_fwd_kb2[None, :] * B_unrot_rt
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2, tl.sum(rot_sign_kb2[None, :] * gB_ * Brd_p_rt, axis=0), sem=ACC_SEM)
                        gB_ = crb_kb2[None, :] * gB_ - srb_fwd_kb2[None, :] * _pair_swap_2d(gB_, T, N)
                    if not ABL_NO_DTH:
                        dth_acc += gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N)
                    gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, N)
                    if HAS_BETA and BIAS_PRE_ROT:
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(gB_, axis=0), sem=ACC_SEM)
                    tl.atomic_add(dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r_fixed * s_dbr, gB_, sem=ACC_SEM)
                elif HAS_ROT_BIAS:
                    db_base_r = dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r_fixed * s_dbr
                    tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn, crb_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                    tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn, srb_fwd_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                    Brd_p_rt = crb_kb2[None, :] * B_unrot_xor_rt - srb_fwd_kb2[None, :] * B_unrot_rt
                    cross_rb_rt = tl.sum(rot_sign_kb2[None, :] * dB_r_agg * Brd_p_rt, axis=0)
                    tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2, cross_rb_rt, sem=ACC_SEM)
                else:
                    tl.atomic_add(dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r_fixed * s_dbr, dB_r_agg, sem=ACC_SEM)
                if HAS_BETA and (not HAS_ROT):
                    tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                BdG_r = pdot(Bc_r.to(MMA_DT), dG_dt, input_precision=PREC)
                q = tl.sum(BdG_r * x_r, axis=1)
            else:
                q = tl.zeros([T], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r * s_br).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    if HAS_ROT and (not ABL_PREROT):
                        B_til = cs_rot * Bc_r + sn_rot * _pair_swap_2d(Bc_r, T, N)
                    else:
                        B_til = Bc_r
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(B_til, T, N)
                        else:
                            Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r * s_br).to(tl.float32)
                        B_unrot_rl = B_til
                        B_unrot_xor_rl = Bc_r_xor_rb
                        Bc_r = crb_kb2[None, :] * B_til + srb_fwd_kb2[None, :] * Bc_r_xor_rb
                    else:
                        Bc_r = B_til
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        x_r = x_r * dw[:, None]
                    xw_r = x_r * dec_agg[:, None]
                    dG_dt = dG.to(MMA_DT)
                    dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
                    if HAS_ROT:
                        gB_ = dB_r_agg
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), sem=ACC_SEM)
                        if HAS_ROT_BIAS:
                            Brd_p_rl = crb_kb2[None, :] * B_unrot_xor_rl - srb_fwd_kb2[None, :] * B_unrot_rl
                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2, tl.sum(rot_sign_kb2[None, :] * gB_ * Brd_p_rl, axis=0), sem=ACC_SEM)
                            gB_ = crb_kb2[None, :] * gB_ - srb_fwd_kb2[None, :] * _pair_swap_2d(gB_, T, N)
                        if not ABL_NO_DTH:
                            dth_acc += gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N)
                        gB_ = cs_rot * gB_ - sn_rot * _pair_swap_2d(gB_, T, N)
                        if HAS_BETA and BIAS_PRE_ROT:
                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), sem=ACC_SEM)
                        tl.atomic_add(dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, gB_, sem=ACC_SEM)
                    elif HAS_ROT_BIAS:
                        db_base_r = dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                        tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn, crb_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                        tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn, srb_fwd_kb2[None, :] * dB_r_agg, sem=ACC_SEM)
                        Brd_p_rl = crb_kb2[None, :] * B_unrot_xor_rl - srb_fwd_kb2[None, :] * B_unrot_rl
                        cross_rb_rl = tl.sum(rot_sign_kb2[None, :] * dB_r_agg * Brd_p_rl, axis=0)
                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_kb2, cross_rb_rl, sem=ACC_SEM)
                    else:
                        tl.atomic_add(dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, dB_r_agg, sem=ACC_SEM)
                    if HAS_BETA and (not HAS_ROT):
                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                    BdG_r = pdot(Bc_r.to(MMA_DT), dG_dt, input_precision=PREC)
                    q += tl.sum(BdG_r * x_r, axis=1)
            if HAS_ROT and (not ABL_NO_DTH):
                dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, N // 2, 2)))
                dth_ph = dth_e - dth_o
                if PHASE_CUMSUM and (not PHASE_DEFER):
                    dth_off = tl.sum(dth_ph, axis=0)
                    dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                    tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + tl.arange(0, N // 2) * s_dok, dth_off, sem=ACC_SEM)
                tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + tl.arange(0, N // 2)[None, :] * s_dtk, dth_ph, sem=ACC_SEM)
            dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
            if R_TILE:
                dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
            else:
                dalpha_gate = 1.0
            dD = a_val * (_excl_prefix(dec_agg * q, offs_t) + alpha * dalpha_c * dalpha_gate)
            tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
_KB2_HB_CFGS = [_bwd_cfg(4, 2, 0)] if _AUTOTUNE_OFF else [_bwd_cfg(w, st, m) for w in (1, 2, 4, 8) for st in (1, 2) for m in (0, 128)]

def _kb2_hb_grid(tier):
    if tier == 'pinned':
        return [_bwd_cfg(4, 2, 0)]
    return [_bwd_cfg(w, st, m) for w in (1, 2, 4, 8) for st in (1, 2) for m in (0, 128)]

@triton.autotune(configs=_calib_grid('fused_kb2_hb', _kb2_hb_grid), prune_configs_by={'early_config_prune': _cprune('fused_kb2_hb')}, key=['T', 'N', 'G_SIZE', 'GT', 'PREC', 'MMA_IO', 'HAS_BETA', 'R', 'HAS_DW', 'IDX64', 'HB2', 'OCC'], restore_value=['dBtil_ptr', 'dDelta_ptr', 'dBeta_ptr'])
@triton.jit
def _bwd_kb2_hb_kernel(B_ptr, X_ptr, Delta_ptr, A_ptr, Beta_ptr, dG_ptr, dAlpha_ptr, dBtil_ptr, dDelta_ptr, dBeta_ptr, H, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xr, s_xl, s_db, s_dh, s_dc, s_dt, s_gb, s_gh, s_gc, s_gn, s_gg, s_dab, s_dah, s_dag, s_dac, s_dbb, s_dbc, s_dbt, s_dbn, s_bh, NC, NGT_val, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, MMA_IO: tl.constexpr=0, HAS_BETA: tl.constexpr=False, HPG_B: tl.constexpr=1, R: tl.constexpr=1, s_br: tl.constexpr=0, s_dbr: tl.constexpr=0, DW_ptr=None, s_wb=0, s_wh=0, s_wl=0, HAS_DW: tl.constexpr=False, s_dbh=0, IDX64: tl.constexpr=False, DCS_INK: tl.constexpr=False, HB2: tl.constexpr=1, OCC=0):
    b = tl.program_id(0)
    hb_id = tl.program_id(1)
    c = tl.program_id(2) // (NGT_val * R)
    remainder = tl.program_id(2) % (NGT_val * R)
    r_fixed = remainder // NGT_val
    gt_idx = remainder % NGT_val
    if IDX64:
        b = b.to(tl.int64)
        hb_id = hb_id.to(tl.int64)
    h0 = hb_id * HB2
    bg = h0 // HPG_B
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    is_last = tl.where(offs_t == T - 1, 1.0, 0.0)
    B_raw = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
    dB_acc = tl.zeros([T, N], dtype=tl.float32)
    for hh in tl.range(HB2, loop_unroll_factor=1):
        h = h0 + hh
        a_val = -tl.exp(tl.load(A_ptr + h))
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
        seg_end = tl.max(seg_h, axis=0)
        D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
        dG = tl.load(dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg).to(tl.float32)
        x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
        if HAS_DW:
            x_r = x_r * tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)[:, None]
        xw_r = x_r * dec_agg[:, None]
        dG_dt = dG.to(MMA_DT)
        dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
        dB_acc += dB_r_agg
        if HAS_BETA:
            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
            Bc_r = B_raw + tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)[None, :]
        else:
            Bc_r = B_raw
        BdG_r = pdot(Bc_r.to(MMA_DT), dG_dt, input_precision=PREC)
        q = tl.sum(BdG_r * x_r, axis=1)
        dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
        dD = a_val * (_excl_prefix(dec_agg * q, offs_t) + alpha * dalpha_c * tl.where(r_fixed == 0, 1.0, 0.0))
        tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
    tl.atomic_add(dBtil_ptr + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r_fixed * s_dbr, dB_acc, sem=ACC_SEM)

def _kb2_hb(H, hpg_b, Bsz, nc, NGT, R, refused, device):
    if refused:
        return 0
    hpg = max(1, min(hpg_b, H))
    env = _env('BWD_KB2_HB', 'auto')
    if env != 'auto':
        hb = int(env)
        if hb <= 0:
            return 0
        if hpg % hb:
            raise ValueError('SSD_MIMO_BWD_KB2_HB=%d does not divide the heads per B group (%d): a head block that straddles a group would accumulate two different dBtil tiles into one register file.' % (hb, hpg))
        return hb
    target = 2 * _sm_count(device)
    hb = 1
    for d in (d for d in range(hpg, 0, -1) if hpg % d == 0):
        if Bsz * ((H + d - 1) // d) * nc * NGT * R >= target:
            hb = d
            break
    return hb

def _kb2_fold_rot():
    return _env('BWD_KB2_FOLD_ROT', 'auto') != '0'

def _kb2_fold(kb1_hb, refused):
    env = _env('BWD_KB2_FOLD', 'auto')
    if env == '0':
        return False
    if refused or not kb1_hb:
        if env == '1':
            raise ValueError("SSD_MIMO_BWD_KB2_FOLD=1 but this cell is outside the fold's envelope (rot_bias / PAIR_HALF / slots / deferred rotation / ablation / R*O == 1 / fuse_kb2 / use_hb / HBLK == 0, or the in-kernel phase with SSD_MIMO_BWD_KB2_FOLD_ROT=0); see `_kb2_fold`'s comment block.")
        return False
    return True

@triton.autotune(configs=_calib_grid('fused_kinv', _bwd_grid), prune_configs_by={'early_config_prune': _cprune('fused_kinv')}, key=['T', 'N', 'O', 'OCC'], restore_value=['dCtil_ptr', 'dThetaH_ptr'])
@triton.jit
def _bwd_kinv_kernel(dCrd_buf_ptr, C_ptr, ThetaH_ptr, dCtil_ptr, dThetaH_ptr, H, NC, s_cb, s_cc, s_ct, s_cn, s_dcb, s_dcc, s_dct, s_dcn, s_ch, T: tl.constexpr, N: tl.constexpr, NEED_DTHH: tl.constexpr=True, HPG_C: tl.constexpr=1, O: tl.constexpr=1, s_co: tl.constexpr=0, s_dco: tl.constexpr=0, s_dch=0, IDX64: tl.constexpr=False, OCC=0):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    cg = h // HPG_C
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    pair_idx = offs_n // 2
    th_full = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
    ch_full = _cos_approx(th_full)
    sh_full = _sin_approx(th_full)
    fwd_sign = tl.where(offs_n % 2 == 0, 1.0, -1.0)
    fwd_coeff = fwd_sign * sh_full
    inv_coeff = -fwd_coeff
    for o in tl.range(O, loop_unroll_factor=1):
        dr_off = (b * H + h) * (NC * O * T * N) + c * (O * T * N) + o * (T * N)
        dCrd = tl.load(dCrd_buf_ptr + dr_off + offs_t[:, None] * N + offs_n[None, :]).to(tl.float32)
        dCrd_xor = tl.load(dCrd_buf_ptr + dr_off + offs_t[:, None] * N + (offs_n ^ 1)[None, :]).to(tl.float32)
        dC = ch_full[None, :] * dCrd + inv_coeff[None, :] * dCrd_xor
        tl.atomic_add(dCtil_ptr + cg * s_dch + b * s_dcb + c * s_dcc + o * s_dco + offs_t[:, None] * s_dct + offs_n[None, :] * s_dcn, dC, sem=ACC_SEM)
        if NEED_DTHH:
            c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
            Co_xor = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
            Co = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
            Crd_partner = ch_full[None, :] * Co_xor - fwd_coeff[None, :] * Co
            cross_sum = tl.sum(fwd_sign[None, :] * dCrd * Crd_partner, axis=0)
            tl.atomic_add(dThetaH_ptr + h * (N // 2) + pair_idx, cross_sum, sem=ACC_SEM)

def _rotate_tile_fwd(tile, theta):
    even = tile[..., 0::2]
    odd = tile[..., 1::2]
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    out = torch.empty_like(tile)
    out[..., 0::2] = cos_th * even + sin_th * odd
    out[..., 1::2] = -sin_th * even + cos_th * odd
    return out

def _rotate_tile_inv(tile, theta):
    even = tile[..., 0::2]
    odd = tile[..., 1::2]
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    out = torch.empty_like(tile)
    out[..., 0::2] = cos_th * even - sin_th * odd
    out[..., 1::2] = sin_th * even + cos_th * odd
    return out

def _dtheta_from_rotation(grad_rd, tile_rd, n_dim=-1):
    return grad_rd[..., 0::2] * tile_rd[..., 1::2] - grad_rd[..., 1::2] * tile_rd[..., 0::2]

def kbc_sumds(*, rot_in_kernel, rot_per_head, has_theta_h, has_rot_bias, pre_rot, pair_half, rot_abl):
    if _env('BWD_KBC_SUMDS', '0') != '1':
        return False
    return not (rot_in_kernel or rot_per_head or has_theta_h or has_rot_bias or pre_rot or pair_half or rot_abl)

def kc_shared_cb(*, rot_in_kernel, has_theta_h, has_rot_bias, pre_rot, pair_half, R, O, bc_grouped, use_r_tile, T=None, dtype=None):
    ok = not (rot_in_kernel or has_theta_h or has_rot_bias or pre_rot or pair_half or bc_grouped or use_r_tile or (R != 1) or (O != 1))
    if not ok:
        return False
    from . import levers
    return levers.on('real_kc_shared_cb', levers.real_group(T, dtype), guard=ok)

def _mma_torch_dtype(mma_dt):
    return {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}.get(mma_dt, torch.float32)

def fused_chunk_bwd_5kernel(Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dY, theta_h=None, theta=None, dS_final=None, precision='tf32', bc_grouped=False, Gb=1, Gc=1, beta=None, gamma=None, x0=None, rot_bias=None, bias_pre_rot=False, dw=None, trap_nd=None, theta_off=None, trap_dsdot=None, trap_lam=None, trap_dot=None, trap_nxt=None, trap_dt=None, need_dlam=False, fuse_trap=False, phase_internal=False, d_skip=None, dskip_dd=False, pair_half=False, dcs_ink=False):
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_scaled.shape[2]
    R = x_scaled.shape[3]
    device = x_scaled.device
    f32 = torch.float32
    dcs_kw = dict(DCS_INK=bool(dcs_ink))
    mma_dt = mma_operand_dtype(precision, x_scaled.dtype)
    mma_io = mma_dt is not tl.float32
    dt_bytes = 2 if mma_io else 4
    precision = dot_precision(precision)
    if bc_grouped:
        N = Ctil.shape[4]
        O = Ctil.shape[5] if Ctil.dim() == 6 else 1
    elif Ctil.dim() == 5:
        N = Ctil.shape[3]
        O = Ctil.shape[4]
    else:
        N = Ctil.shape[-1]
        O = 1
    from .calibration_fused import activate as _calib_activate
    _calib_activate(theta is not None or theta_h is not None, R, O, Bsz, H, T, dtype='fp32' if x_scaled.dtype == torch.float32 else 'bf16')
    from .fused_fwd_kernel import _pick_gt, _big_smem
    _gt_cap = 64 if N <= 64 or _big_smem(device) else 32
    _gt_env = _env('BWD_GT_CAP', '')
    if _gt_env:
        _gt_cap = min(_gt_cap, int(_gt_env))
    elif mma_dt is tl.float32 and x_scaled.dtype == torch.float32:
        from .. import calibration as _cal
        _v, _src = _cal.threshold('ssd_mimo_fused', 'fp32|bwd_gt_cap', None)
        if _v:
            _gt_cap = min(_gt_cap, int(_v))
    GT = _pick_gt(G_SIZE, cap=_gt_cap)
    NGT = G_SIZE // GT
    if mma_io and N * dt_bytes < _SWIZZLE_FLOOR_BYTES and (GT * dt_bytes >= _SWIZZLE_FLOOR_BYTES):
        GT = _pick_gt(G_SIZE, cap=max(16, _SWIZZLE_FLOOR_BYTES // dt_bytes // 2))
        NGT = G_SIZE // GT
    if mma_io and N * dt_bytes < _SWIZZLE_FLOOR_BYTES and (GT * dt_bytes >= _SWIZZLE_FLOOR_BYTES):
        raise ValueError("SSD-MIMO backward: N=%d with %d-byte MMA operands gives a %d-byte [T, N] tile against a %d-byte G tile (GT=%d) -- Triton 3.6 strides the nvmma_shared<swizzlingByteWidth=64> descriptor of the narrow one as if it were the 128-byte mode, and _bwd_kbc_hb_kernel / _bwd_kb1_kernel then read past their shared arena.  Use precision='tf32' (the default) at this state size, or a state size of at least %d.  See implementationexperiments/K24_BF16_BWD_FAULTS_20260907.md." % (N, dt_bytes, N * dt_bytes, GT * dt_bytes, GT, _SWIZZLE_FLOOR_BYTES // dt_bytes))
    has_rot = theta is not None
    has_theta_h = theta_h is not None
    need_dthh = has_theta_h
    has_dsf = dS_final is not None
    has_beta = beta is not None
    has_gamma = gamma is not None
    has_x0 = x0 is not None
    has_rot_bias = rot_bias is not None
    has_dw_early = dw is not None
    fuse_kb2_mimo = O > 1 and (not has_rot) and (not has_rot_bias) and (_env('FUSE_KB2_MIMO', '0') == '1')
    fuse_kb2 = has_dw_early and (O == 1 or fuse_kb2_mimo) and (_env('FUSE_KB2', '1') == '1')
    if pair_half and Gc > 1:
        raise NotImplementedError(f"pair_layout='half' does not serve a GROUPED C (read) map (Gc={Gc}): the trapezoid adjoint's head-vectorised branch needs the per-head factors diagonal in the state axis")
    use_hb = fuse_kb2 and O == 1 and (R == 1) and (not bc_grouped) and (_env('BWD_HB', '1') == '1')
    rot_in_kernel = has_rot and _env('BWD_ROT_INKERNEL', '1') == '1'
    pre_rot = bool(bias_pre_rot)
    if pre_rot and (has_rot_bias or (has_rot and (not rot_in_kernel)) or (not has_rot and has_theta_h)):
        raise NotImplementedError('ssd bwd: bias_pre_rot needs the in-kernel per-head phase, or no rotation at all (theta=None, theta_h=None, rot_bias=None) (rot_in_kernel=%r, theta=%r, theta_h=%r, rot_bias=%r)' % (rot_in_kernel, has_rot, has_theta_h, has_rot_bias))
    if has_rot and theta.dim() == 5 and (not rot_in_kernel):
        raise NotImplementedError('ssd bwd: a PER-HEAD phase needs the in-kernel backward (use_hb=%r, SSD_MIMO_BWD_ROT_INKERNEL=%r); the host fold acts on head-shared B/C.' % (use_hb, _env('BWD_ROT_INKERNEL', '1')))
    if has_rot and (not rot_in_kernel):
        theta_f = theta.float()
        if R > 1:
            Bt = Btil.float().transpose(-1, -2).contiguous()
            th_b = theta_f.unsqueeze(3).expand(-1, -1, -1, R, -1)
            Btil = _rotate_tile_fwd(Bt, th_b).transpose(-1, -2).contiguous().to(Btil.dtype)
        else:
            Btil = _rotate_tile_fwd(Btil.float(), theta_f).to(Btil.dtype).contiguous()
        if O > 1:
            Ct = Ctil.float().transpose(-1, -2).contiguous()
            th_c = theta_f.unsqueeze(3).expand(-1, -1, -1, O, -1)
            Ctil = _rotate_tile_fwd(Ct, th_c).transpose(-1, -2).contiguous().to(Ctil.dtype)
        else:
            Ctil = _rotate_tile_fwd(Ctil.float(), theta_f).to(Ctil.dtype).contiguous()
    dX = torch.empty_strided(x_scaled.shape, x_scaled.stride(), device=device, dtype=x_scaled.dtype)
    dDelta_out = torch.zeros_like(delta_cumsum)
    n_slots = int(_env('DB_SLOTS', '1'))
    use_slots = not bc_grouped and fuse_kb2 and (not use_hb) and (n_slots > 1) and (H > n_slots)
    if use_slots:
        dBtil_acc = torch.zeros((n_slots,) + tuple(Btil.shape), device=device, dtype=f32)
        s_slb_v = dBtil_acc.stride(0)
        dBtil_out = dBtil_acc[0]
    else:
        dBtil_acc = dBtil_out = torch.zeros_like(Btil, dtype=f32)
        s_slb_v = 0
    use_slots_c = use_slots and (not has_theta_h)
    if use_slots_c:
        dCtil_acc = torch.zeros((n_slots,) + tuple(Ctil.shape), device=device, dtype=f32)
        s_slc_v = dCtil_acc.stride(0)
        dCtil_out = dCtil_acc[0]
    else:
        dCtil_acc = dCtil_out = torch.zeros_like(Ctil, dtype=f32)
        s_slc_v = 0
    dummy = torch.empty(1, device=device, dtype=f32)
    thh_c = theta_h.float().contiguous() if has_theta_h else dummy
    dsf_c = dS_final.float().contiguous() if has_dsf else dummy
    beta_c = beta.float().contiguous() if has_beta else dummy
    gamma_c = gamma.float().contiguous() if has_gamma else dummy
    rb_c = rot_bias.float().contiguous() if has_rot_bias else dummy
    dThetaH_out = torch.zeros(H, N // 2, device=device, dtype=f32) if need_dthh else dummy
    dBeta_out = torch.zeros(H, N, R, device=device, dtype=f32) if has_beta else dummy
    dGamma_buf = torch.zeros(H, N, O, device=device, dtype=f32) if has_gamma else dummy
    dRotBias_out = torch.zeros(H, N // 2, device=device, dtype=f32) if has_rot_bias else dummy
    th_c_k = (theta if theta.dtype == torch.bfloat16 else theta.float()).contiguous() if rot_in_kernel else dummy
    rot_per_head = rot_in_kernel and th_c_k.dim() == 5
    dth_store_possible = bool(rot_in_kernel and rot_per_head and (not pair_half) and (not fuse_kb2) and (R * O > 1) and (NGT == 1) and (not has_rot_bias) and (_ROT_ABL_MIMO == 0) and (_env('BWD_DTH_STORE', '1') == '1'))
    dth_hb_sole = bool(rot_in_kernel and rot_per_head and use_hb and (NGT == 1) and (not has_rot_bias) and (_ROT_ABL_MIMO == 0) and (int(_env('BWD_ROT_ABL', '0')) == 0) and (_env('BWD_DTH_HB_EMPTY', '1') == '1'))
    dTheta_buf = (torch.empty if dth_store_possible or dth_hb_sole else torch.zeros)(th_c_k.shape, device=device, dtype=f32) if rot_in_kernel else dummy
    if rot_in_kernel and (dth_store_possible or dth_hb_sole):
        nan_debug_fill(dTheta_buf)
    _dth_state = {'zeroed': not (dth_store_possible or dth_hb_sole)}

    def _dth_zero():
        if not _dth_state['zeroed']:
            dTheta_buf.zero_()
            _dth_state['zeroed'] = True

    def _th5(t):
        if not rot_in_kernel:
            return [0, 0, 0, 0, 0]
        st = list(t.stride())
        return st if len(st) == 5 else [st[0], st[1], st[2], 0, st[3]]
    th_st_k = _th5(th_c_k)
    dth_st_k = _th5(dTheta_buf)
    dth_kw = dict(dTheta_ptr=dTheta_buf, s_dtb=dth_st_k[0], s_dtc=dth_st_k[1], s_dtt=dth_st_k[2], s_dth=dth_st_k[3], s_dtk=dth_st_k[4])
    phase_cumsum = rot_in_kernel and theta_off is not None
    if theta_off is not None and (not rot_in_kernel):
        raise NotImplementedError('ssd bwd: theta_off (the in-kernel phase cumsum) needs the in-kernel rotation route (rot_in_kernel=%r)' % (rot_in_kernel,))
    off_c = theta_off.float().contiguous() if phase_cumsum else dummy
    phase_defer = phase_cumsum and _env('BWD_PHASE_DEFER', '1') == '1'
    dThOff_buf = torch.zeros_like(off_c) if phase_cumsum and (not phase_defer) else dummy

    def _off4(t):
        st = list(t.stride())
        return [st[0], 0 if t.shape[1] == 1 else st[1], st[2], st[3]]
    phase_kw = {}
    doff_kw = dict(dThOff_ptr=dThOff_buf)
    if phase_cumsum:
        off_st = _off4(off_c)
        doff_st = [0, 0, 0, 0] if phase_defer else _off4(dThOff_buf)
        phase_kw = dict(ThOff_ptr=off_c, s_ob=off_st[0], s_oh=off_st[1], s_oc=off_st[2], s_ok=off_st[3], PHASE_CUMSUM=True, PHASE_DEFER=phase_defer)
        doff_kw.update(s_dob=doff_st[0], s_doh=doff_st[1], s_doc=doff_st[2], s_dok=doff_st[3])
    dX0_out = torch.empty(Bsz, H, G_SIZE, N, device=device, dtype=f32) if has_x0 else dummy
    x0_st = list(dX0_out.stride()) if has_x0 else [0, 0, 0, 0]
    has_dw = dw is not None
    if has_dw:
        dw_c = dw
        dDW_out = torch.zeros(Bsz, H, nc * T, device=device, dtype=f32)
        dw_st = list(dw_c.stride())
        ddw_st = list(dDW_out.stride())
    else:
        dw_c = dummy
        dDW_out = dummy
        dw_st = [0, 0, 0]
        ddw_st = [0, 0, 0]
    dw_kwargs = dict(DW_ptr=dw_c, s_wb=dw_st[0], s_wh=dw_st[1], s_wl=dw_st[2], HAS_DW=has_dw)
    if bc_grouped:
        s_bb, s_bc, s_bh_v, s_bt, s_bn = Btil.stride()[:5]
        s_cb, s_cc, s_ch_v, s_ct, s_cn = Ctil.stride()[:5]
        s_br_v = Btil.stride(5) if Btil.dim() == 6 else 0
        s_co_v = Ctil.stride(5) if Ctil.dim() == 6 else 0
    else:
        b_has_r = Btil.dim() >= 5
        if b_has_r:
            s_bb, s_bc, s_bt, s_bn, s_br_v = Btil.stride()[:5]
        else:
            s_bb, s_bc, s_bt, s_bn = Btil.stride()[:4]
            s_br_v = 0
        s_bh_v = 0
        c_has_o = Ctil.dim() >= 5
        if c_has_o:
            s_cb, s_cc, s_ct, s_cn, s_co_v = Ctil.stride()[:5]
        else:
            s_cb, s_cc, s_ct, s_cn = Ctil.stride()[:4]
            s_co_v = 0
        s_ch_v = 0
    if bc_grouped:
        s_dbb, s_dbc, s_dbh_v, s_dbt, s_dbn = dBtil_out.stride()[:5]
        s_dbr_v = dBtil_out.stride(5) if dBtil_out.dim() == 6 else 0
        s_dcb, s_dcc, s_dch_v, s_dct, s_dcn = dCtil_out.stride()[:5]
        s_dco_v = dCtil_out.stride(5) if dCtil_out.dim() == 6 else 0
    else:
        s_dbh_v = s_dch_v = 0
        if dBtil_out.dim() >= 5:
            s_dbb, s_dbc, s_dbt, s_dbn, s_dbr_v = dBtil_out.stride()[:5]
        else:
            s_dbb, s_dbc, s_dbt, s_dbn = dBtil_out.stride()[:4]
            s_dbr_v = 0
        if dCtil_out.dim() >= 5:
            s_dcb, s_dcc, s_dct, s_dcn, s_dco_v = dCtil_out.stride()[:5]
        else:
            s_dcb, s_dcc, s_dct, s_dcn = dCtil_out.stride()[:4]
            s_dco_v = 0
    hpg_b = H // Gb
    hpg_c = H // Gc
    dsf_st = list(dsf_c.stride()) if has_dsf else [0, 0, 0, 0]
    grid = (Bsz, H, NGT)
    use_r_tile = R > 1
    dg_dtype = f32
    from . import levers as _lv
    _dg_ok = mma_dt is not tl.float32 and _env('BWD_KA_NODALPHA', '0') != '1' and (not has_rot_bias) and (R == 1) and (O == 1)
    if _dg_ok and _lv.on('real_dg_lo', _lv.real_group(T), guard=not has_theta_h and (not pair_half)):
        dg_dtype = torch.bfloat16 if mma_dt is tl.bfloat16 else torch.float16
    dG_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device, dtype=dg_dtype)
    dalpha_all = torch.empty(Bsz, H, NGT, nc, device=device, dtype=f32)
    deferred_rot = has_theta_h and (not use_hb) and (not rot_in_kernel)
    if pair_half:
        if not rot_in_kernel or deferred_rot:
            raise NotImplementedError(f"pair_layout='half' needs the in-kernel rotation and no deferred theta_h un-rotation (rot_in_kernel={rot_in_kernel}, deferred_rot={deferred_rot})")
        if (O == 1 and R == 1) and (not use_hb):
            raise NotImplementedError("pair_layout='half' at R = O = 1 needs the head-batched Kbc schedule (use_hb=False)")
        if (O > 1 or R > 1) and use_hb:
            raise NotImplementedError("pair_layout='half': the head-batched Kbc schedule is the R = O = 1 route only")
    if deferred_rot:
        alloc_dcrd = torch.empty if NGT == 1 else torch.zeros
        dCrd_buf = alloc_dcrd(Bsz * H * nc * O * T * N, device=device, dtype=f32)
    else:
        dCrd_buf = dummy
    from .fused_fwd_kernel import _needs_i64
    idx64 = _needs_i64(x_scaled, dY, S_all, dG_all, dX, dBtil_out, dCtil_out, dalpha_all, dCrd_buf, Btil, Ctil, dDelta_out, dDW_out, dX0_out, dw_c, delta_cumsum, dsf_c, th_c_k, dTheta_buf, off_c if phase_cumsum else None, dThOff_buf if phase_cumsum else None)
    from .fused_fwd_kernel import _seg_plan
    nseg_ka, nc_seg_ka = _seg_plan(nc, Bsz * H * NGT, device)
    from .chunkpar import chunkpar_wanted, cp_ka_refusal, cp_ka
    cp_ka_why = cp_ka_refusal(rot_in_kernel=rot_in_kernel, pair_half=bool(pair_half), has_theta_h=has_theta_h, O=O, dcs_ink=dcs_ink, seg_ka=False)
    use_cp_ka = cp_ka_why is None and _env('BWD_KA_NODALPHA', '0') != '1' and chunkpar_wanted(Bsz * H, nc, device, part='ka', dtype=x_scaled.dtype)
    _CP_LAST_ROUTE['ka'] = 'chunkpar' if use_cp_ka else cp_ka_why or 'serial'
    if _env('CP_TRACE', '0') == '1':
        print('CP_TRACE ka=' + _CP_LAST_ROUTE['ka'], flush=True)
    if use_cp_ka:
        nseg_ka = 1
    from . import gluon_kbc_hb as _gkbc
    _ka_nod_x = bool(not use_cp_ka and nseg_ka == 1 and (_env('BWD_KA_NODALPHA', '0') == '1') and use_hb and (not pair_half) and (not rot_in_kernel) and (not rot_per_head) and (not has_theta_h) and (not has_rot_bias) and (R == 1) and (O == 1))
    _cb_ok_x = kc_shared_cb(rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, R=R, O=O, bc_grouped=bc_grouped, use_r_tile=use_r_tile, T=T, dtype=x_scaled.dtype)
    _dd_x = bool(_env('BWD_DD_IN_KBC', '0') == '1' and use_hb and _cb_ok_x and fuse_kb2 and (not has_beta) and (not has_gamma) and (not rot_in_kernel) and (not rot_per_head) and (not has_theta_h) and (not has_rot_bias) and (not pair_half))
    _rot_abl_x = int(_env('BWD_ROT_ABL', '0'))
    _sumds_x = kbc_sumds(rot_in_kernel=rot_in_kernel, rot_per_head=rot_per_head, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, rot_abl=_rot_abl_x)
    _split_x = bool(_env('BWD_KBC_SPLIT', '0') == '1' and (not rot_in_kernel) and (not rot_per_head) and (not has_theta_h) and (not has_rot_bias) and (not pre_rot) and (not pair_half) and (not _rot_abl_x) and (not _ka_nod_x) and (not _dd_x) and (not _sumds_x))
    _ns_x = _hb_nsplit(N, rot_per_head, mma_io)
    if pair_half:
        _ns_x = max(1, min(_ns_x, N // 32))
    _gl_why_x = 'lever off'
    if use_hb and _gkbc.gluon_kbc_wanted(guard=R == 1 and O == 1 and (T == 64) and (not has_theta_h) and (not rot_in_kernel) and (not rot_per_head) and (not has_rot_bias) and (not pair_half) and mma_io, T=T):
        _gl_why_x = _gkbc.refusal(device=device, mma_io=mma_io, rot_in_kernel=rot_in_kernel, rot_per_head=rot_per_head, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, has_beta=has_beta, has_gamma=has_gamma, pre_rot=pre_rot, pair_half=pair_half, sumds=_sumds_x, dd_in_kbc=_dd_x, ka_nodalpha=_ka_nod_x, kbc_split=_split_x, rot_abl=_rot_abl_x, ts_abl=int(_env('BWD_TS_ABL', '0')), T=T, N=N, GT=GT, nsplit=_ns_x, trap_diag=bool(fuse_trap), dcs_ink=bool(dcs_ink))
        if _gl_why_x is None and (not (mma_dt is tl.bfloat16 and Ctil.dtype == Btil.dtype == torch.bfloat16 and (x_scaled.dtype == torch.bfloat16))):
            _gl_why_x = f'dtypes (C {Ctil.dtype}, B {Btil.dtype}, x {x_scaled.dtype}, mma {mma_dt}): bf16 only'
    _dal_on = _env('GLUON_KBC_DALPHA', '') == '1'
    _gl_dalpha = bool(_gl_why_x is None and nseg_ka == 1 and _dal_on)
    if _gl_dalpha:
        dalpha_all.zero_()
    from .chunkpar import cp_resource_refused, cp_resource_refuse
    cp_key = (str(device), str(mma_dt), str(dY.dtype), T, N, G_SIZE, GT)
    _cp_ka_done = False
    ka_common = dict(KA_WB=ka_wave_bucket(Bsz * H * NGT * nseg_ka, device), T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, MMA_IO=int(mma_dt is not tl.float32), HAS_THETA_H=has_theta_h, HAS_GAMMA=has_gamma, HPG_C=hpg_c, O=O, s_co=s_co_v, IDX64=idx64, Theta_ptr=th_c_k, s_thb=th_st_k[0], s_thc=th_st_k[1], s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4], HAS_ROT=rot_in_kernel, BIAS_PRE_ROT=pre_rot, PAIR_HALF=bool(pair_half), **phase_kw)
    ka_args = (Ctil, delta_cumsum, a_log, thh_c, S_all, gamma_c, dY)
    ka_strides = (H, s_cb, s_cc, s_ct, s_cn, *delta_cumsum.stride(), *dY.stride()[:4], dY.stride(5), dY.stride(4), *S_all.stride())
    if nseg_ka > 1:
        v_seg = torch.empty(Bsz, H, nseg_ka, G_SIZE, N, device=device, dtype=f32)
        _bwd_ka_kernel[Bsz, H, NGT * nseg_ka](*ka_args, dummy, dG_all, dalpha_all, v_seg, *ka_strides, 0, 0, 0, 0, *dG_all.stride(), *dalpha_all.stride(), s_ch_v, nc, nc_seg_ka, 0, v_seg.stride(2), HAS_DSF=False, HAS_X0=False, SEG_SUMMARY=True, s_x0b=v_seg.stride(0), s_x0h=v_seg.stride(1), s_x0g=v_seg.stride(3), s_x0n=v_seg.stride(4), **ka_common, **dcs_kw)
        from .fused_fwd_kernel import _seg_alpha, _seg_stitch
        alpha_seg = _seg_alpha(a_log, delta_cumsum, nseg_ka, nc_seg_ka, dcs_ink=dcs_ink)
        carry = _seg_stitch(v_seg, alpha_seg, dsf_c.contiguous() if has_dsf else None, reverse=True)
        _bwd_ka_kernel[Bsz, H, NGT * nseg_ka](*ka_args, carry, dG_all, dalpha_all, dX0_out, *ka_strides, carry.stride(0), carry.stride(1), carry.stride(3), carry.stride(4), *dG_all.stride(), *dalpha_all.stride(), s_ch_v, nc, nc_seg_ka, carry.stride(2), 0, HAS_DSF=True, HAS_X0=has_x0, SEG_SUMMARY=False, s_x0b=x0_st[0], s_x0h=x0_st[1], s_x0g=x0_st[2], s_x0n=x0_st[3], **ka_common, **dcs_kw)
        del v_seg, carry
    elif use_cp_ka and (not cp_resource_refused('ka', cp_key)):
        try:
            cp_ka(Ctil=Ctil, s_c=(s_cb, s_cc, s_ct, s_cn), s_ch=s_ch_v, delta_cumsum=delta_cumsum, a_log=a_log, gamma_c=gamma_c, has_gamma=has_gamma, dY=dY, s_y=(*dY.stride()[:4], dY.stride(5)), S_all=S_all, dsf_c=dsf_c, dsf_st=dsf_st, has_dsf=has_dsf, dG_all=dG_all, dalpha_all=dalpha_all, dX0_out=dX0_out, x0_st=x0_st, has_x0=has_x0, H=H, nc=nc, T=T, N=N, G_SIZE=G_SIZE, GT=GT, NGT=NGT, HPG_C=hpg_c, precision=precision, mma_dt=mma_dt, idx64=idx64, no_dalpha=_gl_dalpha)
            _cp_ka_done = True
        except triton.runtime.errors.OutOfResources as _oor:
            cp_resource_refuse('ka', cp_key, _oor)
            _CP_LAST_ROUTE['ka'] = 'serial (chunkpar refused: ' + str(_oor)[:80] + ')'
    if not (nseg_ka > 1 or _cp_ka_done):
        ka_nodalpha = bool(_env('BWD_KA_NODALPHA', '0') == '1' and use_hb and (not pair_half) and (not rot_in_kernel) and (not rot_per_head) and (not has_theta_h) and (not has_rot_bias) and (R == 1) and (O == 1))
        if ka_nodalpha and (not _gl_dalpha):
            dalpha_all.zero_()
        ka_grid, ka_kw = (grid, dict(ka_common))
        _gt_ka = int(_env('BWD_KA_GT', '0') or 0)
        if ka_nodalpha and 16 <= _gt_ka < GT and (G_SIZE % _gt_ka == 0) and (GT % _gt_ka == 0):
            ka_grid = (Bsz, H, G_SIZE // _gt_ka)
            ka_kw['GT'] = _gt_ka
        ka_kw['KA_WB'] = ka_wave_bucket(ka_grid[0] * ka_grid[1] * ka_grid[2], device)
        _bwd_ka_kernel[ka_grid](*ka_args, dsf_c, dG_all, dalpha_all, dX0_out, *ka_strides, *dsf_st, *dG_all.stride(), *dalpha_all.stride(), s_ch_v, nc, nc, 0, 0, HAS_DSF=has_dsf, HAS_X0=has_x0, SEG_SUMMARY=False, s_x0b=x0_st[0], s_x0h=x0_st[1], s_x0g=x0_st[2], s_x0n=x0_st[3], NO_DALPHA=bool(ka_nodalpha or _gl_dalpha), **ka_kw, **dcs_kw)
    ka_nodalpha = bool(locals().get('ka_nodalpha', False))
    kb1_nsplit = _kb1_nsplit(N, R, O, mma_io)
    if pair_half:
        kb1_nsplit = max(1, min(kb1_nsplit, N // 32))
    _kb1_ro, _kb1_tn = (R * O, T * N)
    _kb1_ieee_cap = precision == 'ieee' and _kb1_ro > _KB1_HB_RO_MAX_IEEE
    _kb1_gate_refused = use_hb or deferred_rot or use_slots or use_slots_c or (_kb1_ro > _KB1_HB_RO_MAX) or _kb1_ieee_cap or (_kb1_tn > _KB1_HB_TN_MAX)
    _kb1_reason = next((name for name, val in (('use_hb', use_hb), ('deferred_rot', deferred_rot), ('use_slots', use_slots), ('use_slots_c', use_slots_c), ('R*O>RO_MAX(%d>%d)' % (_kb1_ro, _KB1_HB_RO_MAX), _kb1_ro > _KB1_HB_RO_MAX), ('ieee:R*O>RO_MAX_IEEE(%d>%d)' % (_kb1_ro, _KB1_HB_RO_MAX_IEEE), _kb1_ieee_cap), ('T*N>TN_MAX(%d>%d)' % (_kb1_tn, _KB1_HB_TN_MAX), _kb1_tn > _KB1_HB_TN_MAX)) if val), '')
    kb1_hb, kb1_db_store, kb1_dc_store = _kb1_hb(H, hpg_b, hpg_c, Bsz, nc, NGT, kb1_nsplit, refused=_kb1_gate_refused, device=device)
    _kb1_dispatch_rec = dict(R=R, O=O, T=T, N=N, G_SIZE=G_SIZE, bc_grouped=bc_grouped, fuse_kb2=fuse_kb2, use_hb=use_hb, deferred_rot=deferred_rot, use_slots=use_slots, use_slots_c=use_slots_c, RO=_kb1_ro, TN=_kb1_tn, ro_max=_KB1_HB_RO_MAX, tn_max=_KB1_HB_TN_MAX, gate_refused=_kb1_gate_refused, hb=kb1_hb, oor=False, hb_final=kb1_hb, reason=_kb1_reason, NGT=NGT, hpg_b=hpg_b, hpg_c=hpg_c, db_store=kb1_db_store, dc_store=kb1_dc_store)
    kb2_mult = nc * NGT * R if use_r_tile else nc * NGT
    grid_kb2 = (Bsz, H, kb2_mult)
    _kb2_hb_refused = bool(has_rot_bias or rot_in_kernel or pair_half or (_ROT_ABL_MIMO != 0) or (R * O <= 1) or deferred_rot or use_slots or use_slots_c)
    kb2_hb = _kb2_hb(H, hpg_b, Bsz, nc, NGT, R, refused=_kb2_hb_refused, device=device)
    if kb2_hb and (H % kb2_hb or hpg_b % kb2_hb):
        kb2_hb = 0
    grid_kb2_hb = (Bsz, H // kb2_hb, nc * NGT * R) if kb2_hb else None
    _kb2_fold_rot_refused = bool(rot_in_kernel and (not _kb2_fold_rot()))
    _kb2_fold_refused = bool(has_rot_bias or pair_half or _ROT_ABL_MIMO != 0 or (R * O <= 1) or deferred_rot or use_slots or use_slots_c or fuse_kb2 or use_hb or _kb2_fold_rot_refused)
    kb2_fold = _kb2_fold(kb1_hb, refused=_kb2_fold_refused)
    kb2_fold_kw_on = dict(KB2_FOLD=True, dG_ptr=dG_all, s_gb=dG_all.stride(0), s_gh=dG_all.stride(1), s_gc=dG_all.stride(2), s_gn=dG_all.stride(3), s_gg=dG_all.stride(4), dAlpha_ptr=dalpha_all, s_dab=dalpha_all.stride(0), s_dah=dalpha_all.stride(1), s_dag=dalpha_all.stride(2), s_dac=dalpha_all.stride(3)) if kb2_fold else None

    def _launch_kb2():
        if kb2_hb:
            _bwd_kb2_hb_kernel[grid_kb2_hb](Btil, x_scaled, delta_cumsum, a_log, beta_c, dG_all, dalpha_all, dBtil_out, dDelta_out, dBeta_out, H, s_bb, s_bc, s_bt, s_bn, *x_scaled.stride(), *delta_cumsum.stride(), *dG_all.stride(), *dalpha_all.stride(), s_dbb, s_dbc, s_dbt, s_dbn, s_bh_v, nc, NGT, T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, MMA_IO=int(mma_dt is not tl.float32), HAS_BETA=has_beta, HPG_B=hpg_b, R=R, s_br=s_br_v, s_dbr=s_dbr_v, s_dbh=s_dbh_v, IDX64=idx64, HB2=kb2_hb, **dw_kwargs, **dcs_kw, OCC=_occ_bucket(grid_kb2_hb[0] * grid_kb2_hb[1] * grid_kb2_hb[2], device))
            return
        _bwd_kb2_kernel[grid_kb2](Btil, x_scaled, delta_cumsum, a_log, beta_c, rb_c, dG_all, dalpha_all, dBtil_out, dDelta_out, dBeta_out, dRotBias_out, H, s_bb, s_bc, s_bt, s_bn, *x_scaled.stride(), *delta_cumsum.stride(), *dG_all.stride(), *dalpha_all.stride(), s_dbb, s_dbc, s_dbt, s_dbn, s_bh_v, nc, NGT, T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, MMA_IO=int(mma_dt is not tl.float32), HAS_BETA=has_beta, HAS_ROT_BIAS=has_rot_bias, HPG_B=hpg_b, R=R, s_br=s_br_v, s_dbr=s_dbr_v, R_TILE=use_r_tile, CHUNK_PARALLEL=True, s_dbh=s_dbh_v, IDX64=idx64, Theta_ptr=th_c_k, s_thb=th_st_k[0], s_thc=th_st_k[1], s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4], HAS_ROT=rot_in_kernel, BIAS_PRE_ROT=pre_rot, ROT_ABL=_ROT_ABL_MIMO, PAIR_HALF=bool(pair_half), **dth_kw, **doff_kw, **phase_kw, **dw_kwargs, **dcs_kw, OCC=_occ_bucket(grid_kb2[0] * grid_kb2[1] * grid_kb2[2], device))
    kb1_rot_comb = bool(rot_in_kernel and has_theta_h and (not has_rot_bias) and (not pair_half) and (_ROT_ABL_MIMO == 0) and (pre_rot or not has_gamma) and (_env('BWD_KB1_ROT_COMB', '0') == '1'))
    _kb2_launched = False
    if not fuse_kb2 and (not kb1_db_store) and (not kb2_fold):
        _dth_zero()
        _launch_kb2()
        del dalpha_all
        _kb2_launched = True
    _cb_cache = {}

    def _shared_cb():
        if 'cb' not in _cb_cache:
            _c4 = torch.as_strided(Ctil, (Bsz, nc, T, N), (s_cb, s_cc, s_ct, s_cn))
            _b4 = torch.as_strided(Btil, (Bsz, nc, T, N), (s_bb, s_bc, s_bt, s_bn))
            _mt = _mma_torch_dtype(mma_dt)
            _tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                _cb_cache['cb'] = torch.matmul(_c4.to(_mt).float(), _b4.to(_mt).float().transpose(-1, -2)).contiguous()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = _tf32
        return _cb_cache['cb']
    _cb_ok = kc_shared_cb(rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, R=R, O=O, bc_grouped=bc_grouped, use_r_tile=use_r_tile, T=T, dtype=x_scaled.dtype)
    dd_in_kbc = bool(_env('BWD_DD_IN_KBC', '0') == '1' and use_hb and _cb_ok and fuse_kb2 and (not has_beta) and (not has_gamma) and (not rot_in_kernel) and (not rot_per_head) and (not has_theta_h) and (not has_rot_bias) and (not pair_half))
    dd_kw = {}
    if dd_in_kbc:
        _cbt0 = _shared_cb()
        dd_kw = dict(DD_PAIRS=True, CB_ptr=_cbt0, s_cbb=_cbt0.stride(0), s_cbc=_cbt0.stride(1))
    if use_hb:
        nsplit = _hb_nsplit(N, rot_per_head, mma_io)
        if pair_half:
            nsplit = max(1, min(nsplit, N // 32))
        bslot_n = int(_env('BWD_BIAS_SLOTS', '8')) if pre_rot else 1
        bslot_n = max(1, min(bslot_n, nc))
        hb_dBeta = hb_dGamma = None
        s_bes_v = s_gas_v = 0
        if bslot_n > 1 and has_beta:
            hb_dBeta = torch.zeros(bslot_n, H, N, device=device, dtype=f32)
            s_bes_v = hb_dBeta.stride(0)
        if bslot_n > 1 and has_gamma:
            hb_dGamma = torch.zeros(bslot_n, H, N, device=device, dtype=f32)
            s_gas_v = hb_dGamma.stride(0)
        rot_abl = int(_env('BWD_ROT_ABL', '0'))
        rot_abl_kw = {}
        if rot_abl & 384:
            _rb = torch.zeros(Bsz, nc, T, H, N, device=device, dtype=Btil.dtype)
            _rc = torch.zeros_like(_rb)
            _st = _rb.stride()
            rot_abl_kw = dict(BRot_ptr=_rb, CRot_ptr=_rc, s_rb=_st[0], s_rc=_st[1], s_rt=_st[2], s_rh=_st[3], s_rn=_st[4])
        grid_hb = lambda META: (Bsz, triton.cdiv(H, META['HB']), nc * NGT * nsplit)
        kbc_split = bool(_env('BWD_KBC_SPLIT', '0') == '1' and (not rot_in_kernel) and (not rot_per_head) and (not has_theta_h) and (not has_rot_bias) and (not pre_rot) and (not pair_half) and (not rot_abl) and (not ka_nodalpha) and (not dd_in_kbc) and (not kbc_sumds(rot_in_kernel=rot_in_kernel, rot_per_head=rot_per_head, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, rot_abl=rot_abl)))
        _gl_why = _gl_why_x
        if (nsplit, dd_in_kbc, kbc_split, ka_nodalpha, rot_abl) != (_ns_x, _dd_x, _split_x, _ka_nod_x, _rot_abl_x):
            raise AssertionError('internal: Gluon Kbc envelope inputs changed between the pre-Ka decision and the launch')
        _CP_LAST_ROUTE['kbc'] = 'gluon' if _gl_why is None else 'triton: ' + _gl_why
        if _env('CP_TRACE', '0') == '1':
            print('CP_TRACE kbc=' + _CP_LAST_ROUTE['kbc'], flush=True)
        if _gl_why is None:
            _gkbc.kbc_hb_gluon(Ctil=Ctil, Btil=Btil, x_scaled=x_scaled, delta_cumsum=delta_cumsum, a_log=a_log, S_all=S_all, dG_all=dG_all, dw_c=dw_c, dY=dY, dCtil_out=dCtil_out, dBtil_out=dBtil_out, dDelta_out=dDelta_out, H=H, Bsz=Bsz, nc=nc, NGT=NGT, nsplit=nsplit, T=T, N=N, GT=GT, strides=(s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, x_scaled.stride(0), x_scaled.stride(1), x_scaled.stride(2), x_scaled.stride(4), *delta_cumsum.stride(), *dY.stride()[:4], dY.stride(5), *S_all.stride(), *dG_all.stride(), dw_st[0], dw_st[1], dw_st[2], s_dcb, s_dcc, s_dct, s_dcn, s_dbb, s_dbc, s_dbt, s_dbn), has_dw=has_dw, dcs_ink=bool(dcs_ink), trap_diag=bool(fuse_trap), idx64=idx64, beta=beta_c if has_beta else None, gamma=gamma_c if has_gamma else None, dBeta=dBeta_out if has_beta else None, dGamma=dGamma_buf if has_gamma else None, dalpha=dalpha_all if _gl_dalpha else None, gt_bwd=GT)
            _CP_LAST_ROUTE['kbc'] = _gkbc._LAST.get('route', 'gluon')
            if _env('CP_TRACE', '0') == '1':
                print('CP_TRACE kbc_served=' + _CP_LAST_ROUTE['kbc'], flush=True)
        for _kpart in () if _gl_why is None else (1, 2) if kbc_split else (0,):
            _bwd_kbc_hb_kernel[grid_hb](Ctil, Btil, x_scaled, delta_cumsum, a_log, S_all, dG_all, dw_c, beta_c, gamma_c, thh_c, rb_c, th_c_k, dY, dCtil_out, dBtil_out, dDelta_out, dBeta_out if hb_dBeta is None else hb_dBeta, dGamma_buf if hb_dGamma is None else hb_dGamma, dThetaH_out, dRotBias_out, dTheta_buf, H, Bsz, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, x_scaled.stride(0), x_scaled.stride(1), x_scaled.stride(2), x_scaled.stride(4), *delta_cumsum.stride(), *dY.stride()[:4], dY.stride(5), *S_all.stride(), *dG_all.stride(), dw_st[0], dw_st[1], dw_st[2], s_dcb, s_dcc, s_dct, s_dcn, s_dbb, s_dbc, s_dbt, s_dbn, *th_st_k, *dth_st_k, nc, NGT, NSPLIT=nsplit, T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, MMA_IO=int(mma_dt is not tl.float32), HAS_BETA=has_beta, HAS_GAMMA=has_gamma, HAS_DW=has_dw, HAS_THETA_H=has_theta_h, HAS_ROT_BIAS=has_rot_bias, HAS_ROT=rot_in_kernel, ROT_PER_HEAD=rot_per_head, BIAS_PRE_ROT=pre_rot, IDX64=idx64, ROT_DIET=_env('BWD_ROT_DIET', '1') == '1', ROT_ABL=rot_abl, SCHED=kbc_sched(mma_io), PAIR_HALF=bool(pair_half), TRAP_DIAG=bool(fuse_trap), BSLOT=bslot_n, s_bes=s_bes_v, s_gas=s_gas_v, KBC_DALPHA=ka_nodalpha, dAlpha_ptr=dalpha_all, s_dab=dalpha_all.stride(0), s_dah=dalpha_all.stride(1), s_dag=dalpha_all.stride(2), s_dac=dalpha_all.stride(3), SUMDS=kbc_sumds(rot_in_kernel=rot_in_kernel, rot_per_head=rot_per_head, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, rot_abl=rot_abl), **rot_abl_kw, **doff_kw, **phase_kw, **dcs_kw, **dd_kw, KBC_PART=_kpart, TS_ABL=int(_env('BWD_TS_ABL', '0')))
        if hb_dBeta is not None:
            dBeta_out += hb_dBeta.sum(0).view(dBeta_out.shape)
        if hb_dGamma is not None:
            dGamma_buf += hb_dGamma.sum(0).view(dGamma_buf.shape)
    else:
        _ns_floor_kb1 = _hb_ns_floor(mma_io)
        _ns_ladder = [kb1_nsplit]
        _ns = kb1_nsplit * 2
        while N % _ns == 0 and N // _ns >= _ns_floor_kb1 and (N // _ns % 2 == 0) and (not (pair_half and N // _ns < 32)):
            _ns_ladder.append(_ns)
            _ns = _ns * 2
        _kb1_forced = [None] + [dict(num_warps=w, num_stages=1) for w in (4, 8)]
        hb_now, dbs_now, dcs_now = (kb1_hb, kb1_db_store, kb1_dc_store)
        fold_now = kb2_fold
        ns_now = kb1_nsplit
        forced_now = None
        hb_key = (T, N, G_SIZE, GT, R, O, kb1_hb, kb1_nsplit, precision, str(mma_dt), rot_in_kernel, has_theta_h, has_rot_bias, pre_rot, has_beta, has_gamma, has_dw, bc_grouped, idx64, kb2_fold)
        _oor_cached = _KB1_HB_OOR.get(hb_key)
        if _oor_cached is not None:
            hb_now, dbs_now, dcs_now, fold_now = (0, False, False, False)
            ns_now, forced_now = _oor_cached
        _g90kb1_used = False
        if _env('GLUON90_KB1', '0').lower() in ('1', 'on', 'true') and _oor_cached is None:
            from . import gluon_sm90_kb1 as _g90k
            _why_k = _g90k.refusal(device=device, T=T, N=N, G_SIZE=G_SIZE, GT=GT, NGT=NGT, R=R, O=O, mma_io=mma_io, io_dtype=x_scaled.dtype, hblk=hb_now, nsplit=ns_now, kb2_fold=fold_now, has_theta_h=has_theta_h, rot_in_kernel=rot_in_kernel, has_rot_bias=has_rot_bias, pair_half=bool(pair_half), deferred_rot=deferred_rot, use_slots=bool(use_slots or use_slots_c), idx64=idx64, has_dw=has_dw, pre_rot=pre_rot, rot_abl=_ROT_ABL_MIMO, Btil=Btil, Ctil=Ctil, x=x_scaled, S_all=S_all, dY=dY, s_bt=s_bt, s_bn=s_bn, s_ct=s_ct, s_cn=s_cn, has_beta=has_beta, has_gamma=has_gamma, phase_cumsum=phase_cumsum, theta=th_c_k if rot_in_kernel else None)
            from . import gluon_sm90_fwd as _g90f
            _g90k_key = ('kb1', bool(rot_in_kernel), bool(has_theta_h))
            if _why_k is None:
                _why_k = _g90f.failed_refusal(_g90k_key)
            _g90k._LAST_ROUTE.clear()
            _g90k._LAST_ROUTE.update(used=_why_k is None, why=_why_k or '')
            if _why_k is None:
                try:
                    _g90_dth_store = bool(rot_in_kernel and dth_store_possible and (not _dth_state['zeroed']))
                    if rot_in_kernel and (not _g90_dth_store):
                        _dth_zero()
                    _g90k.launch(Ctil=Ctil, Btil=Btil, x=x_scaled, delta=delta_cumsum, a_log=a_log, S_all=S_all, dY=dY, dG_all=dG_all, dalpha_all=dalpha_all, beta_c=beta_c, gamma_c=gamma_c, has_beta=has_beta, has_gamma=has_gamma, dw=dw_c, dCtil=dCtil_acc, dBtil=dBtil_acc, dDelta=dDelta_out, dBeta=dBeta_out, dGamma=dGamma_buf, H=H, T=T, N=N, G_SIZE=G_SIZE, GT=GT, R=R, O=O, hpg_b=hpg_b, hpg_c=hpg_c, hblk=hb_now, nc=nc, db_store=dbs_now, dc_store=dcs_now, dcs_ink=dcs_ink, b_strides=(s_bb, s_bc, s_bh_v, s_bt, s_bn, s_br_v), c_strides=(s_cb, s_cc, s_ch_v, s_ct, s_cn, s_co_v), db_strides=(s_dbb, s_dbc, s_dbh_v, s_dbt, s_dbn, s_dbr_v), dc_strides=(s_dcb, s_dcc, s_dch_v, s_dct, s_dcn, s_dco_v), theta=th_c_k if rot_in_kernel else None, th_strides=th_st_k, theta_h=thh_c if has_theta_h else None, dTheta=dTheta_buf if rot_in_kernel else None, dth_strides=dth_st_k, dThetaH=dThetaH_out if need_dthh else None, dth_store=_g90_dth_store, trap_diag=bool(fuse_trap), idx64=bool(idx64))
                    _g90kb1_used = True
                    _kb1_dispatch_rec['gluon90'] = True
                except Exception as _e:
                    _g90kb1_used = False
                    _g90k._LAST_ROUTE.update(used=False, why=_g90f.note_failure(_g90k_key, _e))
                    _kb1_dispatch_rec['gluon90'] = False
                    _kb1_dispatch_rec['gluon90_why'] = _g90k._LAST_ROUTE['why']

        def _kb1_grid(hb_, ns_):
            return (Bsz, (H + hb_ - 1) // hb_ if hb_ else H, nc * NGT * ns_)

        def _kb1_call(entry, hb_, dbs_, dcs_, fold_, ns_, dth_, extra):
            return entry(Ctil, Btil, x_scaled, delta_cumsum, a_log, thh_c, S_all, beta_c, gamma_c, rb_c, dY, dCtil_acc, dBtil_acc, dDelta_out, dThetaH_out, dBeta_out, dGamma_buf, dRotBias_out, dCrd_buf, H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, *x_scaled.stride(), *delta_cumsum.stride(), *dY.stride()[:4], dY.stride(5), dY.stride(4), *S_all.stride(), s_dcb, s_dcc, s_dct, s_dcn, s_dbb, s_dbc, s_dbt, s_dbn, s_bh_v, s_ch_v, nc, NGT, T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, MMA_IO=int(mma_dt is not tl.float32), HAS_THETA_H=has_theta_h, NEED_DTHH=bool(need_dthh and (not deferred_rot)), DEFERRED_ROTATION=deferred_rot, HAS_BETA=has_beta, HAS_GAMMA=has_gamma, HAS_ROT_BIAS=has_rot_bias, HPG_B=hpg_b, HPG_C=hpg_c, R=R, O=O, s_br=s_br_v, s_co=s_co_v, s_dbr=s_dbr_v, s_dco=s_dco_v, CHUNK_PARALLEL=True, s_slb=s_slb_v, s_slc=s_slc_v, SLOTS=n_slots if use_slots else 1, s_dbh=s_dbh_v, s_dch=s_dch_v, IDX64=idx64, Theta_ptr=th_c_k, s_thb=th_st_k[0], s_thc=th_st_k[1], s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4], HAS_ROT=rot_in_kernel, BIAS_PRE_ROT=pre_rot, NSPLIT=ns_, HBLK=hb_, ROT_ABL=_ROT_ABL_MIMO, DB_STORE=dbs_, DC_STORE=dcs_, DTH_STORE=dth_, ROT_COMB=bool(hb_ and kb1_rot_comb), PAIR_HALF=bool(pair_half), TRAP_DIAG=bool(fuse_trap), OCC=_occ_bucket(_kb1_grid(hb_, ns_)[0] * _kb1_grid(hb_, ns_)[1] * _kb1_grid(hb_, ns_)[2], device), **dth_kw, **doff_kw, **phase_kw, **dw_kwargs, **dcs_kw, **kb2_fold_kw_on if fold_ else {}, **extra)
        _cliff_lever = _kb1_cliff_lever()
        _cliff_rec = dict(lever=_cliff_lever, remedy='', engaged=False)
        cliff_dth_off = False
        cliff_forced = None
        if _cliff_lever != 'off' and _oor_cached is None and (not _g90kb1_used):
            _dth_first = bool(dth_store_possible and hb_now and fold_now and (not _dth_state['zeroed']))
            _spec = []
            _kb1_call(lambda *a_, **k_: _spec.append(_kb1_spec_key(a_, k_)), hb_now, dbs_now, dcs_now, fold_now, ns_now, _dth_first, {})
            _ckey = (torch.cuda.get_device_capability(device), _cliff_lever, _KB1_CLIFF_REGS, _KB1_CLIFF_STACK, tuple((str(c) for c in _bwd_kb1_kernel.configs)), _spec[0])
            _plan = _KB1_CLIFF.get(_ckey)
            if _plan is None:

                def _facts(ns_, dth_, forced_):
                    g_ = _kb1_grid(hb_now, ns_)

                    def one(kw):
                        return _kb1_call(lambda *a_, **k_: _bwd_kb1_kernel.fn.warmup(*a_, grid=g_, **k_), hb_now, dbs_now, dcs_now, fold_now, ns_, dth_, kw)
                    return _kb1_probe(one, [forced_] if forced_ is not None else [c.all_kwargs() for c in _bwd_kb1_kernel.configs])
                _plan = dict(remedy='', base=_facts(ns_now, _dth_first, None))
                if _kb1_all_collapsed(_plan['base']):
                    _tried = []
                    for _lv in [_cliff_lever] + (['nsplit'] if _cliff_lever != 'nsplit' else []):
                        if _lv == 'nsplit':
                            for _ns in [v for v in _ns_ladder if v > ns_now][:2]:
                                _f = _facts(_ns, _dth_first, None)
                                _tried.append(dict(lever=_lv, ns=_ns, facts=_f))
                                if _kb1_clears(_f):
                                    _plan.update(remedy=_lv, ns=_ns)
                                    break
                        elif _lv == 'maxnreg':
                            _fc = dict(num_warps=_BWD_PINNED[0], num_stages=_BWD_PINNED[1], maxnreg=255)
                            _f = _facts(ns_now, _dth_first, _fc)
                            _tried.append(dict(lever=_lv, facts=_f))
                            if _kb1_clears(_f):
                                _plan.update(remedy=_lv, forced=_fc)
                        elif _lv == 'dth0' and _dth_first:
                            _f = _facts(ns_now, False, None)
                            _tried.append(dict(lever=_lv, facts=_f))
                            if _kb1_clears(_f):
                                _plan.update(remedy=_lv)
                        if _plan['remedy']:
                            break
                    _plan['tried'] = _tried
                    if not _plan['remedy']:
                        _plan['remedy'] = 'none-cleared'
                _KB1_CLIFF[_ckey] = _plan
            _cliff_rec.update(_plan)
            if _plan['remedy'] == 'nsplit':
                ns_now = _plan['ns']
            elif _plan['remedy'] == 'maxnreg':
                cliff_forced = forced_now = dict(_plan['forced'])
            elif _plan['remedy'] == 'dth0':
                cliff_dth_off = True
            _cliff_rec['engaged'] = _plan['remedy'] in ('nsplit', 'maxnreg', 'dth0')
        _kb1_dispatch_rec['cliff'] = _cliff_rec
        for _kb1_try in range(0 if _g90kb1_used else 2 + len(_ns_ladder) + len(_kb1_forced)):
            grid_kb1 = _kb1_grid(hb_now, ns_now)
            _kb1_entry = _bwd_kb1_kernel if forced_now is None else _bwd_kb1_kernel.fn
            dth_store_now = bool(dth_store_possible and hb_now and fold_now and (not _dth_state['zeroed']) and (not cliff_dth_off))
            if not dth_store_now:
                _dth_zero()
            try:
                _kb1_call(_kb1_entry[grid_kb1], hb_now, dbs_now, dcs_now, fold_now, ns_now, dth_store_now, forced_now or {})
            except OutOfResources:
                if cliff_forced is not None and forced_now is cliff_forced:
                    forced_now = None
                    _cliff_rec['oor_fallback'] = True
                    continue
                if hb_now:
                    hb_now, dbs_now, dcs_now, fold_now = (0, False, False, False)
                    _kb1_dispatch_rec['oor'] = True
                    _kb1_dispatch_rec['reason'] = 'oor'
                    _KB1_HB_OOR[hb_key] = (ns_now, None)
                    continue
                _nxt = [v for v in _ns_ladder if v > ns_now]
                if _nxt and forced_now is None:
                    ns_now = _nxt[0]
                    _KB1_HB_OOR[hb_key] = (ns_now, None)
                    _kb1_dispatch_rec['oor'] = True
                    _kb1_dispatch_rec['reason'] = 'oor(nsplit=%d)' % ns_now
                    continue
                _fi = _kb1_forced.index(forced_now) + 1
                if _fi >= len(_kb1_forced):
                    raise
                forced_now = _kb1_forced[_fi]
                _KB1_HB_OOR[hb_key] = (ns_now, forced_now)
                _kb1_dispatch_rec['oor'] = True
                _kb1_dispatch_rec['reason'] = 'oor(nsplit=%d, forced=%r)' % (ns_now, forced_now)
                continue
            break
        _kb1_dispatch_rec['hb_final'] = hb_now
        _kb1_dispatch_rec['db_store'] = dbs_now
        _kb1_dispatch_rec['dc_store'] = dcs_now
        _kb1_dispatch_rec['kb2_fold_final'] = fold_now
        _kb1_dispatch_rec['nsplit'] = kb1_nsplit
        _kb1_dispatch_rec['nsplit_final'] = ns_now
        _kb1_dispatch_rec['forced_cfg'] = forced_now
        kb2_fold = fold_now
        if hb_key in _KB1_HB_OOR and (not _kb1_dispatch_rec['oor']):
            _kb1_dispatch_rec['oor'] = True
            _kb1_dispatch_rec['reason'] = 'oor(cached)'
    _kb1_dispatch_rec['kb2_hb'] = kb2_hb
    _kb1_dispatch_rec['kb2_hb_refused'] = _kb2_hb_refused
    _kb1_dispatch_rec['kb2_fold'] = bool(kb2_fold)
    _kb1_dispatch_rec['kb2_fold_refused'] = _kb2_fold_refused
    _kb1_dispatch_rec['kb2_fold_rot'] = bool(kb2_fold and rot_in_kernel)
    _kb1_dispatch_rec['kb2_fold_rot_refused'] = _kb2_fold_rot_refused
    _note_kb1_dispatch(_kb1_dispatch_rec)
    if not fuse_kb2 and (not _kb2_launched) and (not kb2_fold):
        _launch_kb2()
        del dalpha_all
    dGamma_out = None
    if has_gamma and deferred_rot:
        dCrd_view = dCrd_buf.view(Bsz, H, nc, O, T, N)
        dGamma_rd = dCrd_view.sum(dim=(0, 2, 4))
        thh_f = theta_h.float()
        pair_idx = torch.arange(N, device=device) // 2
        cos_th = torch.cos(thh_f[:, pair_idx])
        sin_th = torch.sin(thh_f[:, pair_idx])
        fwd_sign = torch.ones(N, device=device)
        fwd_sign[1::2] = -1.0
        inv_coeff = -fwd_sign * sin_th
        fwd_coeff = fwd_sign * sin_th
        dGamma_out = torch.empty(H, N, O, device=device, dtype=f32)
        xor_idx = torch.arange(N, device=device) ^ 1
        for o_idx in range(O):
            dc = dGamma_rd[:, o_idx, :]
            dc_xor = dc[:, xor_idx]
            dGamma_out[:, :, o_idx] = cos_th * dc + inv_coeff * dc_xor
        if need_dthh:
            for o_idx in range(O):
                dc = dGamma_rd[:, o_idx, :]
                gam = gamma[:, :, o_idx].float()
                gam_xor = gam[:, xor_idx]
                gamma_partner = cos_th * gam_xor - fwd_coeff * gam
                contrib = fwd_sign * dc * gamma_partner
                dThetaH_out += contrib[:, 0::2] + contrib[:, 1::2]
    if deferred_rot:
        grid_inv = (Bsz, nc, H)
        _bwd_kinv_kernel[grid_inv](dCrd_buf, Ctil, thh_c, dCtil_out, dThetaH_out, H, nc, s_cb, s_cc, s_ct, s_cn, s_dcb, s_dcc, s_dct, s_dcn, s_ch_v, T=T, N=N, NEED_DTHH=need_dthh, HPG_C=hpg_c, O=O, s_co=s_co_v, s_dco=s_dco_v, s_dch=s_dch_v, IDX64=idx64, OCC=_occ_bucket(grid_inv[0] * grid_inv[1] * grid_inv[2], device))
        del dCrd_buf
    if os.environ.get('UNIFIED_SSD_DEBUG') == '1':
        print('KC-LAUNCH', dict(T=T, N=N, GT=GT, G_SIZE=G_SIZE, R=R, O=O, s_br=s_br_v, s_co=s_co_v, s_bh=s_bh_v, s_ch=s_ch_v, HPG_B=hpg_b, HPG_C=hpg_c, fuse=fuse_kb2, slots=use_slots, s_dbr=s_dbr_v, s_dbh=s_dbh_v, bcg=bc_grouped, s_bb=s_bb, s_bc=s_bc, s_bt=s_bt, s_bn=s_bn), flush=True)
    kc_mult = nc * NGT * R if use_r_tile else nc * NGT
    grid_kc = (Bsz, H, kc_mult)
    has_dskip_kc = d_skip is not None
    if has_dskip_kc and (not (O == 1 and has_dw)):
        raise NotImplementedError(f"the in-kernel D-skip adjoint rides Kc's O == 1 raw-x branch (O={O}, dw={has_dw})")
    dskip_slot = _env('DSKIP_SLOT', '0') == '1' and has_dskip_kc and dskip_dd
    if has_dskip_kc and dskip_dd:
        dDsk_out = torch.empty(Bsz * nc * H * G_SIZE * R, device=device, dtype=f32) if dskip_slot else torch.zeros(H * G_SIZE * R, device=device, dtype=f32)
    else:
        dDsk_out = torch.empty(1, device=device, dtype=f32)
    dg_st = list(dG_all.stride())
    has_trap_adj = trap_nd is not None
    if has_trap_adj and (not has_dw_early):
        raise NotImplementedError(f"the trapezoid adjoint rides Kc's raw-x branch (it needs the RAW x tile and `dw`); R={R}, O={O}, dw={has_dw_early}")
    dSdot = trap_dsdot if trap_dsdot is not None else torch.zeros_like(trap_nd) if has_trap_adj else torch.empty(1, device=device, dtype=f32)
    if has_trap_adj and trap_nd.dim() == 5:
        s_nd = trap_nd.stride()
    else:
        s_nd = (ddw_st[0], ddw_st[1], ddw_st[2], 0, 0) if has_trap_adj else (0, 0, 0, 0, 0)
    cb_kw = {}
    if kc_shared_cb(rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, R=R, O=O, bc_grouped=bc_grouped, use_r_tile=use_r_tile, T=T, dtype=x_scaled.dtype):
        _cbt = _shared_cb()
        cb_kw = dict(SHARED_CB=True, CB_ptr=_cbt, s_cbb=_cbt.stride(0), s_cbc=_cbt.stride(1))
    if _env('REALR1_TRACE', '0') == '1':
        print('REALR1_TRACE', dict(T=T, NGT=NGT, use_hb=bool(use_hb), dg_dtype=str(dG_all.dtype), ka_nodalpha=bool(ka_nodalpha), shared_cb=bool(cb_kw), sumds_env=_env('BWD_KBC_SUMDS', '0')), flush=True)
    _k2_why = 'lever off'
    from . import gluon_k2 as _gk2
    if _gk2.gluon_k2_wanted():
        _k2_why = _gk2.refusal(device=device, mma_dt_is_bf16=mma_dt is tl.bfloat16, dtypes_bf16=Ctil.dtype == Btil.dtype == x_scaled.dtype == torch.bfloat16, shared_cb=bool(cb_kw), fuse_kb2=fuse_kb2, no_dbagg=bool(use_hb), has_dw=has_dw, O=O, R=R, use_r_tile=use_r_tile, rot_in_kernel=rot_in_kernel, has_theta_h=has_theta_h, has_rot_bias=has_rot_bias, pre_rot=pre_rot, pair_half=pair_half, dd_in_kbc=dd_in_kbc, ts_abl=int(_env('BWD_TS_ABL', '0')), dcs_ink=bool(dcs_ink), dskip_slot=dskip_slot, use_slots=use_slots, T=T, N=N, GT=GT)
    _CP_LAST_ROUTE['k2'] = 'gluon' if _k2_why is None else 'triton: ' + _k2_why
    if _k2_why is None and _gk2.k2_pipe_wanted():
        _CP_LAST_ROUTE['k2'] = 'gluon_pipe'
    if _env('CP_TRACE', '0') == '1':
        print('CP_TRACE k2=' + _CP_LAST_ROUTE['k2'], flush=True)
    if _k2_why is None:
        _xs = x_scaled.stride()
        _k2_fn = _gk2.k2_gluon
        _k2_extra = {}
        if _gk2.k2_pipe_wanted():
            _k2_fn = _gk2.k2_gluon_pipe
            _k2_extra = dict(hpg_b=hpg_b, hpg_c=hpg_c)
            _CP_LAST_ROUTE['k2'] = 'gluon_pipe'
        _k2_fn(**_k2_extra, Ctil=Ctil, Btil=Btil, x_scaled=x_scaled, delta_cumsum=delta_cumsum, a_log=a_log, dG_all=dG_all, dY=dY, dw_c=dw_c, cb=cb_kw['CB_ptr'], beta=beta_c if has_beta else None, gamma=gamma_c if has_gamma else None, d_skip=d_skip if has_dskip_kc else None, trap_nd=trap_nd if has_trap_adj else None, dalpha_all=dalpha_all, dX=dX, dDW_out=dDW_out, dDelta_out=dDelta_out, dDsk_out=dDsk_out, dSdot=dSdot, Bsz=Bsz, H=H, nc=nc, NGT=NGT, T=T, N=N, GT=GT, G_SIZE=G_SIZE, strides=(s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, _xs[0], _xs[1], _xs[2], _xs[4], *delta_cumsum.stride(), *dY.stride()[:4], dY.stride(5), *dg_st, dw_st[0], dw_st[1], dw_st[2], ddw_st[0], ddw_st[1], ddw_st[2], cb_kw['s_cbb'], cb_kw['s_cbc'], *dalpha_all.stride(), s_bh_v, s_ch_v, hpg_b, hpg_c), idx64=idx64, dskip_dd=bool(dskip_dd))
    k2_warps = int(_env('BWD_K2_WARPS', '0'))
    k2_stages = int(_env('BWD_K2_STAGES', '2'))
    if k2_warps not in (0, 4, 8, 16) or k2_stages not in (1, 2, 3, 4):
        raise ValueError('unsupported measured K2 launch configuration')
    k2_launch_options = {}
    k2_kernel = _bwd_k2_kernel
    if k2_warps:
        k2_kernel = _bwd_k2_kernel.fn
        k2_launch_options = dict(num_warps=k2_warps, num_stages=k2_stages)
    _k2_launch = k2_kernel[grid_kc] if _k2_why is not None else lambda *a, **k: None
    _k2_launch(Ctil, Btil, delta_cumsum, a_log, thh_c, beta_c, gamma_c, rb_c, dY, dsf_c, dG_all, dX, H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, *x_scaled.stride(), *delta_cumsum.stride(), *dY.stride()[:4], dY.stride(5), dY.stride(4), *dsf_st, s_bh_v, s_ch_v, nc, NGT, T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, MMA_IO=int(mma_dt is not tl.float32), HAS_THETA_H=has_theta_h, HAS_DSF=has_dsf, HAS_BETA=has_beta, HAS_GAMMA=has_gamma, HAS_ROT_BIAS=has_rot_bias, HPG_B=hpg_b, HPG_C=hpg_c, R=R, O=O, s_br=s_br_v, s_co=s_co_v, R_TILE=use_r_tile, CHUNK_PARALLEL=True, PAIR_HALF=bool(pair_half), s_dgb=dg_st[0], s_dgh=dg_st[1], s_dgc=dg_st[2], s_dgn=dg_st[3], s_dgg=dg_st[4], X_ptr=x_scaled, dDW_ptr=dDW_out, s_dwb=ddw_st[0], s_dwh=ddw_st[1], s_dwl=ddw_st[2], dBtil_ptr=dBtil_acc, dDelta_ptr=dDelta_out, dBeta_ptr=dBeta_out, dRotBias_ptr=dRotBias_out, dAlpha_ptr=dalpha_all if fuse_kb2 else dummy, s_dbb=s_dbb, s_dbc=s_dbc, s_dbt=s_dbt, s_dbn=s_dbn, s_dab=dalpha_all.stride(0) if fuse_kb2 else 0, s_dah=dalpha_all.stride(1) if fuse_kb2 else 0, s_dag=dalpha_all.stride(2) if fuse_kb2 else 0, s_dac=dalpha_all.stride(3) if fuse_kb2 else 0, s_dbr=s_dbr_v, FUSE_KB2=fuse_kb2, s_slb=s_slb_v, SLOTS=n_slots if use_slots else 1, s_dbh=s_dbh_v, NO_DBAGG=use_hb, NO_DD_PAIRS=dd_in_kbc, TS_ABL=int(_env('BWD_TS_ABL', '0')), IDX64=idx64, Theta_ptr=th_c_k, s_thb=th_st_k[0], s_thc=th_st_k[1], s_tht=th_st_k[2], s_thh=th_st_k[3], s_thk=th_st_k[4], HAS_ROT=rot_in_kernel, BIAS_PRE_ROT=pre_rot, **dth_kw, **doff_kw, TrapND_ptr=trap_nd if has_trap_adj else dummy, dSdot_ptr=dSdot, s_ndb=s_nd[0], s_ndh=s_nd[1], s_ndl=s_nd[2], s_ndo=s_nd[3], s_ndr=s_nd[4], HAS_TRAP_ADJ=has_trap_adj, Dsk_ptr=d_skip if has_dskip_kc else dummy, dDsk_ptr=dDsk_out, HAS_DSKIP=has_dskip_kc, DSKIP_DD=bool(has_dskip_kc and dskip_dd), DSKIP_SLOT=dskip_slot, **cb_kw, **phase_kw, **dw_kwargs, **dcs_kw, **k2_launch_options, OCC=_occ_bucket(grid_kc[0] * grid_kc[1] * grid_kc[2], device))
    if use_slots:
        dBtil_out = dBtil_acc.sum(dim=0)
    if use_slots_c:
        dCtil_out = dCtil_acc.sum(dim=0)
    if fuse_kb2 or kb2_fold:
        del dalpha_all
    del dG_all
    dTheta_out = None
    if rot_in_kernel:
        dTheta_out = dTheta_buf
    elif has_rot:
        theta_f = theta.float()
        if R > 1:
            Bt_rd = Btil.float().transpose(-1, -2).contiguous()
            dBt_rd = dBtil_out.float().transpose(-1, -2).contiguous()
            dTheta_out = _dtheta_from_rotation(dBt_rd, Bt_rd).sum(dim=3)
        else:
            dTheta_out = _dtheta_from_rotation(dBtil_out.float(), Btil.float())
        if O > 1:
            Ct_rd = Ctil.float().transpose(-1, -2).contiguous()
            dCt_rd = dCtil_out.float().transpose(-1, -2).contiguous()
            dTheta_out = dTheta_out + _dtheta_from_rotation(dCt_rd, Ct_rd).sum(dim=3)
        else:
            dTheta_out = dTheta_out + _dtheta_from_rotation(dCtil_out.float(), Ctil.float())
        if R > 1:
            dBt = dBtil_out.float().transpose(-1, -2).contiguous()
            th_b = theta_f.unsqueeze(3).expand(-1, -1, -1, R, -1)
            dBtil_out = _rotate_tile_inv(dBt, th_b).transpose(-1, -2).contiguous()
        else:
            dBtil_out = _rotate_tile_inv(dBtil_out.float(), theta_f).contiguous()
        if O > 1:
            dCt = dCtil_out.float().transpose(-1, -2).contiguous()
            th_c = theta_f.unsqueeze(3).expand(-1, -1, -1, O, -1)
            dCtil_out = _rotate_tile_inv(dCt, th_c).transpose(-1, -2).contiguous()
        else:
            dCtil_out = _rotate_tile_inv(dCtil_out.float(), theta_f).contiguous()
    if not has_beta:
        dBeta_out = None
    if has_gamma and (not deferred_rot):
        dGamma_out = dGamma_buf
    elif not has_gamma:
        dGamma_out = None
    dThOff_ret = dThOff_buf if phase_cumsum else None
    if phase_defer:
        from .m3_epilogue import phase_prefix_bwd
        if phase_internal:
            phase_prefix_bwd(dTheta_out, None, cross=True)
            dThOff_ret = None
        else:
            dThOff_ret = torch.empty_like(off_c)
            phase_prefix_bwd(dTheta_out, dThOff_ret, cross=False)
    elif phase_cumsum and phase_internal:
        from .m3_epilogue import phase_offset_bwd_add
        phase_offset_bwd_add(dThOff_buf, dTheta_out)
        dThOff_ret = None
    dLam_out = None
    if fuse_trap:
        from .m3_epilogue import trap_adjoint
        dDW_new, dLam_out = trap_adjoint(Btil, Ctil, beta_c if has_beta else None, gamma_c if has_gamma else None, trap_dt, trap_lam, trap_dot, trap_nxt, dDW_out, dSdot, dBtil_out, dCtil_out, dBeta_out if has_beta else None, dGamma_buf if has_gamma else None, nc, T, N, precision, bool(need_dlam), bc_grouped=bc_grouped, Gb=Gb, Gc=Gc, theta_h=thh_c if has_theta_h else None, dThetaH=dThetaH_out if has_theta_h else None, pair_half=pair_half)
        dDW_out = dDW_new
    return (dX, dCtil_out, dBtil_out, dDelta_out, dThetaH_out if need_dthh else None, dTheta_out, dBeta_out, dGamma_out, dX0_out if has_x0 else None, dRotBias_out if has_rot_bias else None, dDW_out.view(Bsz, H, nc * T) if has_dw else None, dSdot if trap_nd is not None else None, dThOff_ret, dLam_out, (dDsk_out.view(Bsz * nc, H * G_SIZE * R).sum(0) if dskip_slot else dDsk_out) if has_dskip_kc and dskip_dd else None)

@triton.autotune(configs=_calib_grid('fused_kc', _bwd_grid), prune_configs_by={'early_config_prune': _cprune('fused_kc')}, key=['T', 'N', 'G_SIZE', 'GT', 'PREC', 'MMA_IO', 'HAS_THETA_H', 'HAS_BETA', 'HAS_GAMMA', 'HAS_ROT_BIAS', 'HAS_ROT', 'BIAS_PRE_ROT', 'PHASE_CUMSUM', 'R', 'O', 'HAS_DW', 'FUSE_KB2', 'HAS_TRAP_ADJ', 'NO_DBAGG', 'IDX64', 'HAS_DSKIP', 'PAIR_HALF', 'SHARED_CB', 'NO_DD_PAIRS', 'TS_ABL', 'OCC'], restore_value=['dDW_ptr', 'dBtil_ptr', 'dDelta_ptr', 'dBeta_ptr', 'dRotBias_ptr', 'dSdot_ptr', 'dDsk_ptr', 'dTheta_ptr', 'dThOff_ptr'])
@triton.jit
def _bwd_k2_kernel(C_ptr, B_ptr, Delta_ptr, A_ptr, ThetaH_ptr, Beta_ptr, Gamma_ptr, RotBias_ptr, dY_ptr, dSF_ptr, dG_ptr, dX_ptr, H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xr, s_xl, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_fb, s_fh, s_fg, s_fn, s_bh, s_ch, NC, NGT_val, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, MMA_IO: tl.constexpr=0, HAS_THETA_H: tl.constexpr=False, HAS_DSF: tl.constexpr=False, HAS_BETA: tl.constexpr=False, HAS_GAMMA: tl.constexpr=False, HAS_ROT_BIAS: tl.constexpr=False, HPG_B: tl.constexpr=1, HPG_C: tl.constexpr=1, R: tl.constexpr=1, O: tl.constexpr=1, s_br: tl.constexpr=0, s_co: tl.constexpr=0, R_TILE: tl.constexpr=False, CHUNK_PARALLEL: tl.constexpr=False, PAIR_HALF: tl.constexpr=False, s_dgb: tl.constexpr=0, s_dgh: tl.constexpr=0, s_dgc: tl.constexpr=0, s_dgn: tl.constexpr=0, s_dgg: tl.constexpr=0, X_ptr=None, DW_ptr=None, dDW_ptr=None, s_wb=0, s_wh=0, s_wl=0, s_dwb=0, s_dwh=0, s_dwl=0, HAS_DW: tl.constexpr=False, dBtil_ptr=None, dDelta_ptr=None, dBeta_ptr=None, dRotBias_ptr=None, dAlpha_ptr=None, s_dbb=0, s_dbc=0, s_dbt=0, s_dbn=0, s_dab=0, s_dah=0, s_dag=0, s_dac=0, s_dbr: tl.constexpr=0, FUSE_KB2: tl.constexpr=False, s_slb=0, SLOTS: tl.constexpr=1, s_dbh=0, NO_DBAGG: tl.constexpr=False, IDX64: tl.constexpr=False, Theta_ptr=None, s_thb=0, s_thc=0, s_tht=0, s_thh=0, s_thk=0, HAS_ROT: tl.constexpr=False, BIAS_PRE_ROT: tl.constexpr=False, dTheta_ptr=None, s_dtb=0, s_dth=0, s_dtc=0, s_dtt=0, s_dtk=0, dThOff_ptr=None, s_dob=0, s_doh=0, s_doc=0, s_dok=0, TrapND_ptr=None, dSdot_ptr=None, s_ndb=0, s_ndh=0, s_ndl=0, s_ndo=0, s_ndr=0, HAS_TRAP_ADJ: tl.constexpr=False, Dsk_ptr=None, dDsk_ptr=None, HAS_DSKIP: tl.constexpr=False, DSKIP_DD: tl.constexpr=False, DSKIP_SLOT: tl.constexpr=False, ThOff_ptr=None, s_ob=0, s_oh=0, s_oc=0, s_ok=0, PHASE_CUMSUM: tl.constexpr=False, PHASE_DEFER: tl.constexpr=False, DCS_INK: tl.constexpr=False, SHARED_CB: tl.constexpr=False, NO_DD_PAIRS: tl.constexpr=False, CB_ptr=None, s_cbb=0, s_cbc=0, TS_ABL: tl.constexpr=0, OCC=0):
    tl.static_assert(not HAS_TRAP_ADJ or HAS_DW, 'Kc: the trapezoid adjoint is the raw-x branch (it needs `dw` and the RAW x tile); R, O >= 1 since K28 implementation (2026-09-14)')
    tl.static_assert(not BIAS_PRE_ROT or (HAS_ROT and (not HAS_ROT_BIAS)), 'Kc: bias_pre_rot needs the in-kernel phase and is refused with rot_bias')
    tl.static_assert(not HAS_DSKIP or (O == 1 and HAS_DW), 'Kc: the in-kernel D-skip adjoint is the O == 1 raw-x branch only')
    tl.static_assert(not FUSE_KB2 or (HAS_DW and (O == 1 or (not HAS_ROT and (not HAS_ROT_BIAS) and (not NO_DBAGG)))), 'Kc: MIMO Kb2 fusion requires an unrotated write map and owns dB_agg')
    b = tl.program_id(0)
    h = tl.program_id(1)
    if CHUNK_PARALLEL:
        if R_TILE:
            c = tl.program_id(2) // (NGT_val * R)
            remainder = tl.program_id(2) % (NGT_val * R)
            r_fixed = remainder // NGT_val
            gt_idx = remainder % NGT_val
        else:
            c = tl.program_id(2) // NGT_val
            gt_idx = tl.program_id(2) % NGT_val
    elif R_TILE:
        r_fixed = tl.program_id(2) // NGT_val
        gt_idx = tl.program_id(2) % NGT_val
    else:
        gt_idx = tl.program_id(2)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    if HAS_TRAP_ADJ:
        causal = offs_t[:, None] > offs_t[None, :]
    else:
        causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float('-inf')
    bg = h // HPG_B
    cg = h // HPG_C
    slot_b = h % SLOTS * s_slb
    if PAIR_HALF:
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        KC_DTH: tl.constexpr = HAS_ROT and FUSE_KB2 and (not NO_DBAGG)
        KC_S1: tl.constexpr = R == 1 and O == 1 and CHUNK_PARALLEL and NO_DBAGG and FUSE_KB2 and (not R_TILE)
        tl.static_assert(not HAS_ROT_BIAS, 'PAIR_HALF does not serve the retired rot_bias')
        tl.static_assert(N >= 32, 'PAIR_HALF needs N/2 >= 16 (the MMA K floor)')
        if KC_S1:
            tl.static_assert(R == 1 and O == 1, 'PAIR_HALF stage 1 is the R = O = 1 route')
            tl.static_assert(CHUNK_PARALLEL, 'PAIR_HALF serves the chunk-parallel Kc')
            tl.static_assert(NO_DBAGG and FUSE_KB2, 'PAIR_HALF serves Kc next to the head-batched Kbc (use_hb), which owns dB~/dTheta')
            tl.static_assert(not HAS_ROT_BIAS, 'PAIR_HALF does not serve the retired rot_bias')
            tl.static_assert(not R_TILE, 'PAIR_HALF: R = 1 takes no R tile')
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_h1 * s_ok).to(tl.float32)[None, :]
            if HAS_THETA_H:
                th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)
            c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + offs_t[:, None] * s_ct
            C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn).to(tl.float32)
            C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn).to(tl.float32)
            if HAS_GAMMA:
                ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O).to(tl.float32)
                ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O).to(tl.float32)
                if BIAS_PRE_ROT:
                    C1 = C1 + ga1[None, :]
                    C2 = C2 + ga2[None, :]
            if HAS_ROT:
                if HAS_THETA_H:
                    th_c = th_rot + th_h[None, :]
                else:
                    th_c = th_rot
                cc = _cos_approx(th_c)
                sc = _sin_approx(th_c)
                Cr1 = cc * C1 + sc * C2
                Cr2 = cc * C2 - sc * C1
            elif HAS_THETA_H:
                cc = _cos_approx(th_h)
                sc = _sin_approx(th_h)
                Cr1 = cc[None, :] * C1 + sc[None, :] * C2
                Cr2 = cc[None, :] * C2 - sc[None, :] * C1
            else:
                Cr1 = C1
                Cr2 = C2
            if HAS_GAMMA and (not BIAS_PRE_ROT):
                if HAS_THETA_H:
                    ch_h = _cos_approx(th_h)
                    sh_h = _sin_approx(th_h)
                    Cr1 = Cr1 + (ch_h * ga1 + sh_h * ga2)[None, :]
                    Cr2 = Cr2 + (ch_h * ga2 - sh_h * ga1)[None, :]
                else:
                    Cr1 = Cr1 + ga1[None, :]
                    Cr2 = Cr2 + ga2[None, :]
            Cr1_dt = Cr1.to(MMA_DT)
            Cr2_dt = Cr2.to(MMA_DT)
            b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt
            B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
            B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
            if HAS_BETA:
                be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R).to(tl.float32)
                be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R).to(tl.float32)
                if BIAS_PRE_ROT:
                    B1 = B1 + be1[None, :]
                    B2 = B2 + be2[None, :]
            if HAS_ROT:
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                Bt1 = ph_cos * B1 + ph_sin * B2
                Bt2 = ph_cos * B2 - ph_sin * B1
            else:
                Bt1 = B1
                Bt2 = B2
            if HAS_BETA and (not BIAS_PRE_ROT):
                Bt1 = Bt1 + be1[None, :]
                Bt2 = Bt2 + be2[None, :]
            Bt1_dt = Bt1.to(MMA_DT)
            Bt2_dt = Bt2.to(MMA_DT)
            dg_base = dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc + offs_gt[None, :] * s_dgg
            BdG_r = pdot(Bt1_dt, tl.load(dg_base + offs_h1[:, None] * s_dgn).to(MMA_DT), input_precision=PREC) + pdot(Bt2_dt, tl.load(dg_base + offs_h2[:, None] * s_dgn).to(MMA_DT), input_precision=PREC)
            dx_r = dec_agg[:, None] * BdG_r
            S0_r = pdot(Cr1_dt, tl.trans(Bt1_dt), input_precision=PREC) + pdot(Cr2_dt, tl.trans(Bt2_dt), input_precision=PREC)
            Score_r = S0_r * tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
            dx_r += pdot(tl.trans(Score_r).to(MMA_DT), dY_o.to(MMA_DT), input_precision=PREC)
            x_base_r = b * s_xb + h * s_xh + c * T * s_xl
            q = tl.zeros([T], dtype=tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                q += tl.sum(BdG_r * (x_raw * dw[:, None]), axis=1)
                tl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + offs_t) * s_dwl, tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                dx_r = dx_r * dw[:, None]
                if HAS_DSKIP:
                    dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R + offs_gt * R)
                    dx_r = dx_r + dsk_r[None, :] * dY_o
                    if DSKIP_DD:
                        dd_v = tl.sum(dY_o * x_raw, axis=0)
                        if DSKIP_SLOT:
                            tl.store(dDsk_ptr + ((b * NC + c) * H + h) * G_SIZE * R + offs_gt * R, dd_v)
                        else:
                            tl.atomic_add(dDsk_ptr + h * G_SIZE * R + offs_gt * R, dd_v, sem=ACC_SEM)
                if HAS_TRAP_ADJ:
                    nd = tl.load(TrapND_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                    dx_r = dx_r + nd[:, None] * dY_o
                    tl.atomic_add(dSdot_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl, tl.sum(dY_o * x_raw, axis=1), sem=ACC_SEM)
            tl.store(dX_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, dx_r)
            if HAS_DW:
                xm = x_raw * dw[:, None]
            else:
                xm = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                q += tl.sum(BdG_r * xm, axis=1)
            Msp = Score_r * pdot(dY_o.to(MMA_DT), tl.trans(xm.to(MMA_DT)), input_precision=PREC)
            Msp = tl.where((offs_t == T - 1)[:, None], Msp + (dec_agg * q)[None, :], Msp)
            dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
            dD = a_val * (_span_sum(Msp, offs_t) + alpha * dalpha_c)
            tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
        else:
            if not CHUNK_PARALLEL:
                if HAS_DSF:
                    fb = dSF_ptr + b * s_fb + h * s_fh + offs_gt[None, :] * s_fg
                    v1 = tl.load(fb + offs_h1[:, None] * s_fn).to(tl.float32)
                    v2 = tl.load(fb + offs_h2[:, None] * s_fn).to(tl.float32)
                else:
                    v1 = tl.zeros([KH, GT], dtype=tl.float32)
                    v2 = tl.zeros([KH, GT], dtype=tl.float32)
            R_ITERS: tl.constexpr = 1 if R_TILE else R
            nc_end_kc = 1 if CHUNK_PARALLEL else NC
            for ci in range(nc_end_kc):
                if CHUNK_PARALLEL:
                    pass
                else:
                    c = NC - 1 - ci
                dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
                D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
                seg_end = tl.max(seg_h, axis=0)
                f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
                D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
                alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
                dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
                dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
                if HAS_ROT:
                    th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk).to(tl.float32)
                    if PHASE_CUMSUM:
                        th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_h1 * s_ok).to(tl.float32)[None, :]
                    ph_cos = _cos_approx(th_rot)
                    ph_sin = _sin_approx(th_rot)
                    if KC_DTH:
                        dth_acc = tl.zeros([T, KH], dtype=tl.float32)
                if HAS_THETA_H:
                    th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)
                    ch_h = _cos_approx(th_h)
                    sh_h = _sin_approx(th_h)
                if HAS_DW:
                    dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                if CHUNK_PARALLEL:
                    dgb = dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc + offs_gt[None, :] * s_dgg
                    dG1 = tl.load(dgb + offs_h1[:, None] * s_dgn).to(tl.float32)
                    dG2 = tl.load(dgb + offs_h2[:, None] * s_dgn).to(tl.float32)
                else:
                    v_old1 = v1
                    v_old2 = v2
                    dG1 = v_old1
                    dG2 = v_old2
                dG1_dt = dG1.to(MMA_DT)
                dG2_dt = dG2.to(MMA_DT)
                if not CHUNK_PARALLEL:
                    dS1 = tl.zeros([KH, GT], dtype=tl.float32)
                    dS2 = tl.zeros([KH, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co + offs_t[:, None] * s_ct
                        C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn).to(tl.float32)
                        C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn).to(tl.float32)
                        if HAS_GAMMA:
                            ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O + o).to(tl.float32)
                            ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O + o).to(tl.float32)
                            if BIAS_PRE_ROT:
                                C1 = C1 + ga1[None, :]
                                C2 = C2 + ga2[None, :]
                        if HAS_ROT:
                            Ct1 = ph_cos * C1 + ph_sin * C2
                            Ct2 = ph_cos * C2 - ph_sin * C1
                        else:
                            Ct1 = C1
                            Ct2 = C2
                        if HAS_THETA_H:
                            Cr1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                            Cr2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                        else:
                            Cr1 = Ct1
                            Cr2 = Ct2
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            if HAS_THETA_H:
                                gr1 = ch_h * ga1 + sh_h * ga2
                                gr2 = ch_h * ga2 - sh_h * ga1
                            else:
                                gr1 = ga1
                                gr2 = ga2
                            Cr1 = Cr1 + gr1[None, :]
                            Cr2 = Cr2 + gr2[None, :]
                        dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                        f_dY = (f[:, None] * dY_o).to(MMA_DT)
                        dS1 += pdot(tl.trans(Cr1).to(MMA_DT), f_dY, input_precision=PREC)
                        dS2 += pdot(tl.trans(Cr2).to(MMA_DT), f_dY, input_precision=PREC)
                    v1 = alpha * v_old1 + dS1
                    v2 = alpha * v_old2 + dS2
                if FUSE_KB2:
                    q = tl.zeros([T], dtype=tl.float32)
                Msp = tl.zeros([T, T], dtype=tl.float32)
                for rr in tl.range(R_ITERS, loop_unroll_factor=1):
                    if R_TILE:
                        r = r_fixed
                    else:
                        r = rr
                    b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + r * s_br + offs_t[:, None] * s_bt
                    B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
                    B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
                    if HAS_BETA:
                        be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r).to(tl.float32)
                        be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r).to(tl.float32)
                        if BIAS_PRE_ROT:
                            B1 = B1 + be1[None, :]
                            B2 = B2 + be2[None, :]
                    if HAS_ROT:
                        Bt1 = ph_cos * B1 + ph_sin * B2
                        Bt2 = ph_cos * B2 - ph_sin * B1
                    else:
                        Bt1 = B1
                        Bt2 = B2
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        Bf1 = Bt1 + be1[None, :]
                        Bf2 = Bt2 + be2[None, :]
                    else:
                        Bf1 = Bt1
                        Bf2 = Bt2
                    Bf1_dt = Bf1.to(MMA_DT)
                    Bf2_dt = Bf2.to(MMA_DT)
                    BdG_r = pdot(Bf1_dt, dG1_dt, input_precision=PREC) + pdot(Bf2_dt, dG2_dt, input_precision=PREC)
                    agg_r = dec_agg[:, None] * BdG_r
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        xm_dt = (x_raw * dw[:, None]).to(MMA_DT)
                    else:
                        xm_dt = x_raw.to(MMA_DT)
                    if HAS_TRAP_ADJ:
                        trap_corr = tl.zeros([T, GT], dtype=tl.float32)
                    dx_r = tl.zeros([T, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co + offs_t[:, None] * s_ct
                        C1 = tl.load(c_base_o + offs_h1[None, :] * s_cn).to(tl.float32)
                        C2 = tl.load(c_base_o + offs_h2[None, :] * s_cn).to(tl.float32)
                        if HAS_GAMMA:
                            ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O + o).to(tl.float32)
                            ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O + o).to(tl.float32)
                            if BIAS_PRE_ROT:
                                C1 = C1 + ga1[None, :]
                                C2 = C2 + ga2[None, :]
                        if HAS_ROT:
                            Ct1 = ph_cos * C1 + ph_sin * C2
                            Ct2 = ph_cos * C2 - ph_sin * C1
                        else:
                            Ct1 = C1
                            Ct2 = C2
                        if HAS_THETA_H:
                            Cr1 = ch_h[None, :] * Ct1 + sh_h[None, :] * Ct2
                            Cr2 = ch_h[None, :] * Ct2 - sh_h[None, :] * Ct1
                        else:
                            Cr1 = Ct1
                            Cr2 = Ct2
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            if HAS_THETA_H:
                                gr1 = ch_h * ga1 + sh_h * ga2
                                gr2 = ch_h * ga2 - sh_h * ga1
                            else:
                                gr1 = ga1
                                gr2 = ga2
                            Cr1 = Cr1 + gr1[None, :]
                            Cr2 = Cr2 + gr2[None, :]
                        S0_or = pdot(Cr1.to(MMA_DT), tl.trans(Bf1_dt), input_precision=PREC) + pdot(Cr2.to(MMA_DT), tl.trans(Bf2_dt), input_precision=PREC)
                        Score_or = S0_or * dec_mask
                        dY_o2 = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                        dY_o2_dt = dY_o2.to(MMA_DT)
                        dx_r += pdot(tl.trans(Score_or).to(MMA_DT), dY_o2_dt, input_precision=PREC)
                        Msp += Score_or * pdot(dY_o2_dt, tl.trans(xm_dt), input_precision=PREC)
                        if HAS_TRAP_ADJ:
                            nd_or = tl.load(TrapND_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + o * s_ndo + r * s_ndr)
                            trap_corr += nd_or[:, None] * dY_o2
                            tl.atomic_add(dSdot_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + o * s_ndo + r * s_ndr, tl.sum(dY_o2 * x_raw, axis=1), sem=ACC_SEM)
                        if HAS_DSKIP:
                            dY_dsk = dY_o2
                    dx_r += agg_r
                    if HAS_DW:
                        if FUSE_KB2:
                            x_s = x_raw * dw[:, None]
                            xw_dt = (x_s * dec_agg[:, None]).to(MMA_DT)
                            gB1 = pdot(xw_dt, tl.trans(dG1_dt), input_precision=PREC)
                            gB2 = pdot(xw_dt, tl.trans(dG2_dt), input_precision=PREC)
                            if HAS_BETA and (not BIAS_PRE_ROT):
                                tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), sem=ACC_SEM)
                                tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), sem=ACC_SEM)
                            if KC_DTH:
                                dth_acc += gB1 * _opv(Bt2, MMA_DT, MMA_IO, PREC) - gB2 * _opv(Bt1, MMA_DT, MMA_IO, PREC)
                            if HAS_ROT:
                                gn1 = ph_cos * gB1 - ph_sin * gB2
                                gB2 = ph_cos * gB2 + ph_sin * gB1
                                gB1 = gn1
                            if HAS_BETA and BIAS_PRE_ROT:
                                tl.atomic_add(dBeta_ptr + h * N * R + offs_h1 * R + r, tl.sum(gB1, axis=0), sem=ACC_SEM)
                                tl.atomic_add(dBeta_ptr + h * N * R + offs_h2 * R + r, tl.sum(gB2, axis=0), sem=ACC_SEM)
                            db_base_r = dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                            tl.atomic_add(db_base_r + offs_h1[None, :] * s_dbn, gB1, sem=ACC_SEM)
                            tl.atomic_add(db_base_r + offs_h2[None, :] * s_dbn, gB2, sem=ACC_SEM)
                            q += tl.sum(BdG_r * x_s, axis=1)
                        tl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + offs_t) * s_dwl, tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                        dx_r = dx_r * dw[:, None]
                        if HAS_DSKIP:
                            dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R + offs_gt * R + r)
                            dx_r = dx_r + dsk_r[None, :] * dY_dsk
                            if DSKIP_DD:
                                dd_v = tl.sum(dY_dsk * x_raw, axis=0)
                                if DSKIP_SLOT:
                                    tl.store(dDsk_ptr + ((b * NC + c) * H + h) * G_SIZE * R + offs_gt * R + r, dd_v)
                                else:
                                    tl.atomic_add(dDsk_ptr + h * G_SIZE * R + offs_gt * R + r, dd_v, sem=ACC_SEM)
                        if HAS_TRAP_ADJ:
                            dx_r = dx_r + trap_corr
                    tl.store(dX_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, dx_r)
                if KC_DTH:
                    dth_ph = dth_acc
                    if PHASE_CUMSUM and (not PHASE_DEFER):
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + offs_h1 * s_dok, dth_off, sem=ACC_SEM)
                    tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + offs_h1[None, :] * s_dtk, dth_ph, sem=ACC_SEM)
                if FUSE_KB2:
                    Msp = tl.where((offs_t == T - 1)[:, None], Msp + (dec_agg * q)[None, :], Msp)
                    dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
                    if R_TILE:
                        dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
                    else:
                        dalpha_gate = 1.0
                    dD = a_val * (_span_sum(Msp, offs_t) + alpha * dalpha_c * dalpha_gate)
                else:
                    dD = a_val * _span_sum(Msp, offs_t)
                tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
    else:
        if HAS_THETA_H:
            pair_idx_k2 = offs_n // 2
            th_full_k2 = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx_k2)
            ch_full_k2 = _cos_approx(th_full_k2)
            sh_full_k2 = _sin_approx(th_full_k2)
            fwd_sign_k2 = tl.where(offs_n % 2 == 0, 1.0, -1.0)
            fwd_coeff_k2 = fwd_sign_k2 * sh_full_k2
        KC_DTH: tl.constexpr = HAS_ROT and FUSE_KB2 and (not NO_DBAGG)
        if HAS_ROT:
            rot_sign_k2_r = tl.where(offs_n % 2 == 0, 1.0, -1.0)
            pair_idx_k2_r = offs_n // 2
        if HAS_ROT_BIAS:
            pair_idx_k2_rb = offs_n // 2
            rot_sign_k2 = tl.where(offs_n % 2 == 0, 1.0, -1.0)
            rb_full_k2 = tl.load(RotBias_ptr + h * (N // 2) + pair_idx_k2_rb)
            crb_k2 = _cos_approx(rb_full_k2)
            srb_fwd_k2 = rot_sign_k2 * _sin_approx(rb_full_k2)
        if not CHUNK_PARALLEL:
            if HAS_DSF:
                v = tl.load(dSF_ptr + b * s_fb + h * s_fh + offs_gt[None, :] * s_fg + offs_n[:, None] * s_fn).to(tl.float32)
            else:
                v = tl.zeros([N, GT], dtype=tl.float32)
        nc_end_kc = 1 if CHUNK_PARALLEL else NC
        for ci in range(nc_end_kc):
            if CHUNK_PARALLEL:
                pass
            else:
                c = NC - 1 - ci
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            Msp = tl.zeros([T, T], dtype=tl.float32)
            if HAS_ROT:
                th_rot_k2 = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + tl.arange(0, N // 2)[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot_k2 = tl.cumsum(th_rot_k2, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + tl.arange(0, N // 2) * s_ok).to(tl.float32)[None, :]
                cs_k2 = _pair_bcast_2d(_cos_approx(th_rot_k2), T, N)
                sn_k2 = rot_sign_k2_r[None, :] * _pair_bcast_2d(_sin_approx(th_rot_k2), T, N)
                if KC_DTH:
                    dth_acc = tl.zeros([T, N], dtype=tl.float32)
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            if CHUNK_PARALLEL:
                if TS_ABL & 1:
                    dG = tl.load(dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc + offs_n[:, None] * s_dgn + offs_gt[None, :] * s_dgg, mask=(offs_n[:, None] >= 0) & (c % 2 == 1), other=0.0).to(tl.float32)
                else:
                    dG = tl.load(dG_ptr + b * s_dgb + h * s_dgh + c * s_dgc + offs_n[:, None] * s_dgn + offs_gt[None, :] * s_dgg).to(tl.float32)
                if TS_ABL & 2:
                    if c % 2 == 0:
                        Cn = tl.load(C_ptr + b * s_cb + tl.minimum(c + 1, NC - 1) * s_cc + cg * s_ch + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(MMA_DT)
                        dY_nx = tl.load(dY_ptr + b * s_yb + h * s_yh + tl.minimum(c + 1, NC - 1) * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(MMA_DT)
                        dG = dG * 0.5 + pdot(tl.trans(Cn), dY_nx, input_precision=PREC)
            else:
                v_old = v
                dG = v_old
            if O == 1:
                c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch
                Co = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)[None, :]
                if HAS_ROT:
                    Co = cs_k2 * Co + sn_k2 * _pair_swap_2d(Co, T, N)
                if HAS_THETA_H:
                    if HAS_ROT:
                        Co_xor = _pair_swap_2d(Co, T, N)
                    else:
                        Co_xor = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                    C_rd = ch_full_k2[None, :] * Co + fwd_coeff_k2[None, :] * Co_xor
                else:
                    C_rd = Co
                if HAS_GAMMA and (not BIAS_PRE_ROT):
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)
                    if HAS_THETA_H:
                        gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O).to(tl.float32)
                        gam = ch_full_k2 * gam + fwd_coeff_k2 * gam_xor
                    C_rd = C_rd + gam[None, :]
                dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                C_rd_dt = C_rd.to(MMA_DT)
                if not CHUNK_PARALLEL:
                    f_dY = f[:, None] * dY_o
                    dS_readout = pdot(tl.trans(C_rd_dt), f_dY.to(MMA_DT), input_precision=PREC)
                    v = alpha * v_old + dS_readout
                if R_TILE:
                    Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)[None, :]
                    if HAS_ROT:
                        B_til = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                    else:
                        B_til = Bc_r
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(B_til, T, N)
                        else:
                            Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
                        B_unrot = B_til
                        B_unrot_xor = Bc_r_xor_rb
                        Bc_r = crb_k2[None, :] * B_til + srb_fwd_k2[None, :] * Bc_r_xor_rb
                    else:
                        Bc_r = B_til
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    Bc_dt = Bc_r.to(MMA_DT)
                    dG_dt = dG.to(MMA_DT)
                    BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                    dx_r = dec_agg[:, None] * BdG_r
                    S0_r = pdot(C_rd_dt, tl.trans(Bc_dt), input_precision=PREC)
                    Score_r = S0_r * dec_mask
                    dx_r += pdot(tl.trans(Score_r).to(MMA_DT), dY_o.to(MMA_DT), input_precision=PREC)
                    x_base_r = b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
                    x_sp = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        x_sp = x_sp * dw[:, None]
                    xm_dt = x_sp.to(MMA_DT)
                    if not NO_DD_PAIRS:
                        Msp += Score_r * pdot(dY_o.to(MMA_DT), tl.trans(xm_dt), input_precision=PREC)
                    if HAS_DW:
                        x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        if FUSE_KB2:
                            x_s = x_raw * dw[:, None]
                            xw_r = x_s * dec_agg[:, None]
                            dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
                            if HAS_ROT:
                                gB_ = dB_r_agg
                                if HAS_BETA and (not BIAS_PRE_ROT):
                                    tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(gB_, axis=0), sem=ACC_SEM)
                                if HAS_ROT_BIAS:
                                    Brd_p = crb_k2[None, :] * B_unrot_xor - srb_fwd_k2[None, :] * B_unrot
                                    tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb, tl.sum(rot_sign_k2[None, :] * gB_ * Brd_p, axis=0), sem=ACC_SEM)
                                    gB_ = crb_k2[None, :] * gB_ - srb_fwd_k2[None, :] * _pair_swap_2d(gB_, T, N)
                                dth_acc += rot_sign_k2_r[None, :] * gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N)
                                gB_ = cs_k2 * gB_ - sn_k2 * _pair_swap_2d(gB_, T, N)
                                if HAS_BETA and BIAS_PRE_ROT:
                                    tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(gB_, axis=0), sem=ACC_SEM)
                                tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r_fixed * s_dbr, gB_, sem=ACC_SEM)
                            elif HAS_ROT_BIAS:
                                db_base_r = dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r_fixed * s_dbr
                                tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn, crb_k2[None, :] * dB_r_agg, sem=ACC_SEM)
                                tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn, srb_fwd_k2[None, :] * dB_r_agg, sem=ACC_SEM)
                                Brd_p = crb_k2[None, :] * B_unrot_xor - srb_fwd_k2[None, :] * B_unrot
                                tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb, tl.sum(rot_sign_k2[None, :] * dB_r_agg * Brd_p, axis=0), sem=ACC_SEM)
                            else:
                                tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r_fixed * s_dbr, dB_r_agg, sem=ACC_SEM)
                            if HAS_BETA and (not HAS_ROT):
                                tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                            q = tl.sum(BdG_r * x_s, axis=1)
                        tl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + offs_t) * s_dwl, tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                        dx_r = dx_r * dw[:, None]
                        if HAS_DSKIP:
                            dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R + offs_gt * R + r_fixed)
                            dx_r = dx_r + dsk_r[None, :] * dY_o
                            if DSKIP_DD:
                                dd_v = tl.sum(dY_o * x_raw, axis=0)
                                if DSKIP_SLOT:
                                    tl.store(dDsk_ptr + ((b * NC + c) * H + h) * G_SIZE * R + offs_gt * R + r_fixed, dd_v)
                                else:
                                    tl.atomic_add(dDsk_ptr + h * G_SIZE * R + offs_gt * R + r_fixed, dd_v, sem=ACC_SEM)
                        if HAS_TRAP_ADJ:
                            nd = tl.load(TrapND_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + r_fixed * s_ndr)
                            dx_r = dx_r + nd[:, None] * dY_o
                            tl.atomic_add(dSdot_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + r_fixed * s_ndr, tl.sum(dY_o * x_raw, axis=1), sem=ACC_SEM)
                    tl.store(dX_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, dx_r)
                else:
                    if FUSE_KB2:
                        q = tl.zeros([T], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r * s_br).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_BETA:
                            Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Bc_r = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                        B_til = Bc_r
                        if HAS_ROT_BIAS:
                            if HAS_ROT:
                                Bc_r_xor_rb = _pair_swap_2d(Bc_r, T, N)
                            else:
                                Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r * s_br).to(tl.float32)
                            B_unrot = Bc_r
                            B_unrot_xor = Bc_r_xor_rb
                            Bc_r = crb_k2[None, :] * Bc_r + srb_fwd_k2[None, :] * Bc_r_xor_rb
                        if SHARED_CB:
                            B_raw_cb = Bc_r
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                            Bc_r = Bc_r + beta_r[None, :]
                        Bc_dt = Bc_r.to(MMA_DT)
                        dG_dt = dG.to(MMA_DT)
                        BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                        dx_r = dec_agg[:, None] * BdG_r
                        if SHARED_CB:
                            S0_r = tl.load(CB_ptr + b * s_cbb + c * s_cbc + offs_t[:, None] * T + offs_t[None, :])
                            if HAS_BETA and (not BIAS_PRE_ROT):
                                S0_r += tl.sum(C_rd * beta_r[None, :], axis=1)[:, None]
                            if HAS_GAMMA and (not BIAS_PRE_ROT):
                                S0_r += tl.sum(B_raw_cb * gam[None, :], axis=1)[None, :]
                        else:
                            S0_r = pdot(C_rd_dt, tl.trans(Bc_dt), input_precision=PREC)
                        Score_r = S0_r * dec_mask
                        dx_r += pdot(tl.trans(Score_r).to(MMA_DT), dY_o.to(MMA_DT), input_precision=PREC)
                        x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        x_sp = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        if HAS_DW:
                            x_sp = x_sp * dw[:, None]
                        xm_dt = x_sp.to(MMA_DT)
                        if not NO_DD_PAIRS:
                            Msp += Score_r * pdot(dY_o.to(MMA_DT), tl.trans(xm_dt), input_precision=PREC)
                        if HAS_DW:
                            x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                            if FUSE_KB2:
                                x_s = x_raw * dw[:, None]
                                if not NO_DBAGG:
                                    xw_r = x_s * dec_agg[:, None]
                                    dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
                                    if HAS_ROT:
                                        gB_ = dB_r_agg
                                        if HAS_BETA and (not BIAS_PRE_ROT):
                                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), sem=ACC_SEM)
                                        if HAS_ROT_BIAS:
                                            Brd_p = crb_k2[None, :] * B_unrot_xor - srb_fwd_k2[None, :] * B_unrot
                                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb, tl.sum(rot_sign_k2[None, :] * gB_ * Brd_p, axis=0), sem=ACC_SEM)
                                            gB_ = crb_k2[None, :] * gB_ - srb_fwd_k2[None, :] * _pair_swap_2d(gB_, T, N)
                                        dth_acc += rot_sign_k2_r[None, :] * gB_ * _pair_swap_2d(_opv(B_til, MMA_DT, MMA_IO, PREC), T, N)
                                        gB_ = cs_k2 * gB_ - sn_k2 * _pair_swap_2d(gB_, T, N)
                                        if HAS_BETA and BIAS_PRE_ROT:
                                            tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(gB_, axis=0), sem=ACC_SEM)
                                        tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, gB_, sem=ACC_SEM)
                                    elif HAS_ROT_BIAS:
                                        db_base_r = dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + r * s_dbr
                                        tl.atomic_add(db_base_r + offs_n[None, :] * s_dbn, crb_k2[None, :] * dB_r_agg, sem=ACC_SEM)
                                        tl.atomic_add(db_base_r + (offs_n ^ 1)[None, :] * s_dbn, srb_fwd_k2[None, :] * dB_r_agg, sem=ACC_SEM)
                                        Brd_p = crb_k2[None, :] * B_unrot_xor - srb_fwd_k2[None, :] * B_unrot
                                        tl.atomic_add(dRotBias_ptr + h * (N // 2) + pair_idx_k2_rb, tl.sum(rot_sign_k2[None, :] * dB_r_agg * Brd_p, axis=0), sem=ACC_SEM)
                                    else:
                                        tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, dB_r_agg, sem=ACC_SEM)
                                    if HAS_BETA and (not HAS_ROT):
                                        tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                                q += tl.sum(BdG_r * x_s, axis=1)
                            tl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + offs_t) * s_dwl, tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                            dx_r = dx_r * dw[:, None]
                            if HAS_DSKIP:
                                dsk_r = tl.load(Dsk_ptr + h * G_SIZE * R + offs_gt * R + r)
                                dx_r = dx_r + dsk_r[None, :] * dY_o
                                if DSKIP_DD:
                                    dd_v = tl.sum(dY_o * x_raw, axis=0)
                                    if DSKIP_SLOT:
                                        tl.store(dDsk_ptr + ((b * NC + c) * H + h) * G_SIZE * R + offs_gt * R + r, dd_v)
                                    else:
                                        tl.atomic_add(dDsk_ptr + h * G_SIZE * R + offs_gt * R + r, dd_v, sem=ACC_SEM)
                            if HAS_TRAP_ADJ:
                                nd = tl.load(TrapND_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                                dx_r = dx_r + nd[:, None] * dY_o
                                tl.atomic_add(dSdot_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl, tl.sum(dY_o * x_raw, axis=1), sem=ACC_SEM)
                        tl.store(dX_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, dx_r)
                if KC_DTH:
                    dth_e, dth_o = tl.split(tl.reshape(dth_acc, (T, N // 2, 2)))
                    dth_ph = dth_e + dth_o
                    if PHASE_CUMSUM and (not PHASE_DEFER):
                        dth_off = tl.sum(dth_ph, axis=0)
                        dth_ph = tl.cumsum(dth_ph, axis=0, reverse=True)
                        tl.atomic_add(dThOff_ptr + b * s_dob + h * s_doh + c * s_doc + tl.arange(0, N // 2) * s_dok, dth_off, sem=ACC_SEM)
                    tl.atomic_add(dTheta_ptr + b * s_dtb + h * s_dth + c * s_dtc + offs_t[:, None] * s_dtt + tl.arange(0, N // 2)[None, :] * s_dtk, dth_ph, sem=ACC_SEM)
                if FUSE_KB2:
                    if not NO_DD_PAIRS:
                        Msp = tl.where((offs_t == T - 1)[:, None], Msp + (dec_agg * q)[None, :], Msp)
                    dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
                    if R_TILE:
                        dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
                    else:
                        dalpha_gate = 1.0
                    if NO_DD_PAIRS:
                        dD = a_val * (_excl_prefix(dec_agg * q, offs_t) + alpha * dalpha_c * dalpha_gate)
                    else:
                        dD = a_val * (_span_sum(Msp, offs_t) + alpha * dalpha_c * dalpha_gate)
                else:
                    dD = a_val * _span_sum(Msp, offs_t)
                tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)
            else:
                if not CHUNK_PARALLEL:
                    dS_readout = tl.zeros([N, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                        Co = tl.load(c_base_o + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_GAMMA:
                            Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Co = cs_k2 * Co + sn_k2 * _pair_swap_2d(Co, T, N)
                        if HAS_THETA_H:
                            if HAS_ROT:
                                Co_xor = _pair_swap_2d(Co, T, N)
                            else:
                                Co_xor = tl.load(c_base_o + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                            C_rd_o = ch_full_k2[None, :] * Co + fwd_coeff_k2[None, :] * Co_xor
                        else:
                            C_rd_o = Co
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                            if HAS_THETA_H:
                                gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                gam = ch_full_k2 * gam + fwd_coeff_k2 * gam_xor
                            C_rd_o = C_rd_o + gam[None, :]
                        dY_o = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                        f_dY_o = f[:, None] * dY_o
                        dS_readout += pdot(tl.trans(C_rd_o).to(MMA_DT), f_dY_o.to(MMA_DT), input_precision=PREC)
                    v = alpha * v_old + dS_readout
                if R_TILE:
                    Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)[None, :]
                    if HAS_ROT:
                        Bc_r = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if HAS_ROT:
                            Bc_r_xor_rb = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r_fixed * s_br).to(tl.float32)
                        Bc_r = crb_k2[None, :] * Bc_r + srb_fwd_k2[None, :] * Bc_r_xor_rb
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r_fixed).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    Bc_dt = Bc_r.to(MMA_DT)
                    dG_dt = dG.to(MMA_DT)
                    BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                    agg_r = dec_agg[:, None] * BdG_r
                    x_base_r = b * s_xb + h * s_xh + r_fixed * s_xr + c * T * s_xl
                    x_sp = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DW:
                        x_sp = x_sp * dw[:, None]
                    xm_dt = x_sp.to(MMA_DT)
                    if HAS_TRAP_ADJ:
                        x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        trap_corr = tl.zeros([T, GT], dtype=tl.float32)
                    dx_r = tl.zeros([T, GT], dtype=tl.float32)
                    for o in tl.range(O, loop_unroll_factor=1):
                        c_base_o2 = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                        Co2 = tl.load(c_base_o2 + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_GAMMA:
                            Co2 = Co2 + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Co2 = cs_k2 * Co2 + sn_k2 * _pair_swap_2d(Co2, T, N)
                        if HAS_THETA_H:
                            if HAS_ROT:
                                Co2_xor = _pair_swap_2d(Co2, T, N)
                            else:
                                Co2_xor = tl.load(c_base_o2 + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                            C_rd_o2 = ch_full_k2[None, :] * Co2 + fwd_coeff_k2[None, :] * Co2_xor
                        else:
                            C_rd_o2 = Co2
                        if HAS_GAMMA and (not BIAS_PRE_ROT):
                            gam2 = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                            if HAS_THETA_H:
                                gam2_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                gam2 = ch_full_k2 * gam2 + fwd_coeff_k2 * gam2_xor
                            C_rd_o2 = C_rd_o2 + gam2[None, :]
                        S0_or = pdot(C_rd_o2.to(MMA_DT), tl.trans(Bc_dt), input_precision=PREC)
                        Score_or = S0_or * dec_mask
                        dY_o2 = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                        Msp += Score_or * pdot(dY_o2.to(MMA_DT), tl.trans(xm_dt), input_precision=PREC)
                        dx_r += pdot(tl.trans(Score_or).to(MMA_DT), dY_o2.to(MMA_DT), input_precision=PREC)
                        if HAS_TRAP_ADJ:
                            nd_or = tl.load(TrapND_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + o * s_ndo + r_fixed * s_ndr)
                            trap_corr += nd_or[:, None] * dY_o2
                            tl.atomic_add(dSdot_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + o * s_ndo + r_fixed * s_ndr, tl.sum(dY_o2 * x_raw, axis=1), sem=ACC_SEM)
                    dx_r += agg_r
                    if HAS_DW:
                        if not HAS_TRAP_ADJ:
                            x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        if FUSE_KB2:
                            x_s = x_raw * dw[:, None]
                            xw_r = x_s * dec_agg[:, None]
                            dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
                            tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r_fixed * s_dbr, dB_r_agg, sem=ACC_SEM)
                            if HAS_BETA:
                                tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r_fixed, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                            q = tl.sum(BdG_r * x_s, axis=1)
                        tl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + offs_t) * s_dwl, tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                        dx_r = dx_r * dw[:, None]
                        if HAS_TRAP_ADJ:
                            dx_r = dx_r + trap_corr
                    tl.store(dX_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, dx_r)
                else:
                    if FUSE_KB2:
                        q = tl.zeros([T], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        Bc_r = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn + r * s_br).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_BETA:
                            Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                        if HAS_ROT:
                            Bc_r = cs_k2 * Bc_r + sn_k2 * _pair_swap_2d(Bc_r, T, N)
                        if HAS_ROT_BIAS:
                            if HAS_ROT:
                                Bc_r_xor_rb = _pair_swap_2d(Bc_r, T, N)
                            else:
                                Bc_r_xor_rb = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + (offs_n ^ 1)[None, :] * s_bn + r * s_br).to(tl.float32)
                            Bc_r = crb_k2[None, :] * Bc_r + srb_fwd_k2[None, :] * Bc_r_xor_rb
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                            Bc_r = Bc_r + beta_r[None, :]
                        Bc_dt = Bc_r.to(MMA_DT)
                        dG_dt = dG.to(MMA_DT)
                        BdG_r = pdot(Bc_dt, dG_dt, input_precision=PREC)
                        agg_r = dec_agg[:, None] * BdG_r
                        x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        x_sp = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        if HAS_DW:
                            x_sp = x_sp * dw[:, None]
                        xm_dt = x_sp.to(MMA_DT)
                        if HAS_TRAP_ADJ:
                            x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                            trap_corr = tl.zeros([T, GT], dtype=tl.float32)
                        dx_r = tl.zeros([T, GT], dtype=tl.float32)
                        for o in tl.range(O, loop_unroll_factor=1):
                            c_base_o2 = C_ptr + b * s_cb + c * s_cc + cg * s_ch + o * s_co
                            Co2 = tl.load(c_base_o2 + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
                            if BIAS_PRE_ROT and HAS_GAMMA:
                                Co2 = Co2 + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                            if HAS_ROT:
                                Co2 = cs_k2 * Co2 + sn_k2 * _pair_swap_2d(Co2, T, N)
                            if HAS_THETA_H:
                                if HAS_ROT:
                                    Co2_xor = _pair_swap_2d(Co2, T, N)
                                else:
                                    Co2_xor = tl.load(c_base_o2 + offs_t[:, None] * s_ct + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                                C_rd_o2 = ch_full_k2[None, :] * Co2 + fwd_coeff_k2[None, :] * Co2_xor
                            else:
                                C_rd_o2 = Co2
                            if HAS_GAMMA and (not BIAS_PRE_ROT):
                                gam2 = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                                if HAS_THETA_H:
                                    gam2_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                                    gam2 = ch_full_k2 * gam2 + fwd_coeff_k2 * gam2_xor
                                C_rd_o2 = C_rd_o2 + gam2[None, :]
                            S0_or = pdot(C_rd_o2.to(MMA_DT), tl.trans(Bc_dt), input_precision=PREC)
                            Score_or = S0_or * dec_mask
                            dY_o2 = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
                            Msp += Score_or * pdot(dY_o2.to(MMA_DT), tl.trans(xm_dt), input_precision=PREC)
                            dx_r += pdot(tl.trans(Score_or).to(MMA_DT), dY_o2.to(MMA_DT), input_precision=PREC)
                            if HAS_TRAP_ADJ:
                                nd_or = tl.load(TrapND_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + o * s_ndo + r * s_ndr)
                                trap_corr += nd_or[:, None] * dY_o2
                                tl.atomic_add(dSdot_ptr + b * s_ndb + h * s_ndh + (c * T + offs_t) * s_ndl + o * s_ndo + r * s_ndr, tl.sum(dY_o2 * x_raw, axis=1), sem=ACC_SEM)
                        dx_r += agg_r
                        if HAS_DW:
                            if not HAS_TRAP_ADJ:
                                x_raw = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                            if FUSE_KB2:
                                x_s = x_raw * dw[:, None]
                                xw_r = x_s * dec_agg[:, None]
                                dB_r_agg = pdot(xw_r.to(MMA_DT), tl.trans(dG_dt), input_precision=PREC)
                                tl.atomic_add(dBtil_ptr + slot_b + bg * s_dbh + b * s_dbb + c * s_dbc + offs_t[:, None] * s_dbt + offs_n[None, :] * s_dbn + r * s_dbr, dB_r_agg, sem=ACC_SEM)
                                if HAS_BETA:
                                    tl.atomic_add(dBeta_ptr + h * N * R + offs_n * R + r, tl.sum(dB_r_agg, axis=0), sem=ACC_SEM)
                                q += tl.sum(BdG_r * x_s, axis=1)
                            tl.atomic_add(dDW_ptr + b * s_dwb + h * s_dwh + (c * T + offs_t) * s_dwl, tl.sum(dx_r * x_raw, axis=1), sem=ACC_SEM)
                            dx_r = dx_r * dw[:, None]
                            if HAS_TRAP_ADJ:
                                dx_r = dx_r + trap_corr
                        tl.store(dX_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl, dx_r)
                if FUSE_KB2:
                    Msp = tl.where((offs_t == T - 1)[:, None], Msp + (dec_agg * q)[None, :], Msp)
                    dalpha_c = tl.load(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac)
                    if R_TILE:
                        dalpha_gate = tl.where(r_fixed == 0, 1.0, 0.0)
                    else:
                        dalpha_gate = 1.0
                    dD = a_val * (_span_sum(Msp, offs_t) + alpha * dalpha_c * dalpha_gate)
                else:
                    dD = a_val * _span_sum(Msp, offs_t)
                tl.atomic_add(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, dD, sem=ACC_SEM)

@triton.jit
def _delta_adjoint_kernel(dD_ptr, Delta_ptr, dALog_ptr, n_rows, s_db, s_dh, s_dc, s_dt, H: tl.constexpr, NC: tl.constexpr, T: tl.constexpr, ROWS: tl.constexpr, NEED_DALOG: tl.constexpr):
    h = tl.program_id(0)
    row0 = tl.program_id(1) * ROWS
    rows = row0 + tl.arange(0, ROWS)
    m = rows < n_rows
    bb = rows // NC
    cc = rows % NC
    offs_t = tl.arange(0, T)
    rev = T - 1 - offs_t
    base = dD_ptr + bb[:, None] * s_db + h * s_dh + cc[:, None] * s_dc
    g = tl.load(base + rev[None, :] * s_dt, mask=m[:, None], other=0.0)
    g = tl.cumsum(g, axis=1)
    tl.store(base + rev[None, :] * s_dt, g, mask=m[:, None])
    if NEED_DALOG:
        dbase = Delta_ptr + bb[:, None] * s_db + h * s_dh + cc[:, None] * s_dc
        d = tl.load(dbase + rev[None, :] * s_dt, mask=m[:, None], other=0.0)
        tl.atomic_add(dALog_ptr + h, tl.sum(d * g), sem=ACC_SEM)

def delta_adjoint(dDelta, delta_raw, need_dalog=True):
    Bsz, H, nc, T = dDelta.shape
    if not dDelta.is_contiguous():
        raise ValueError('delta_adjoint needs a contiguous dDelta')
    if delta_raw.stride() != dDelta.stride():
        delta_raw = delta_raw.contiguous()
    n_rows = Bsz * nc
    da_log = torch.zeros(H, device=dDelta.device, dtype=torch.float32) if need_dalog else None
    rows = max(1, min(64, triton.next_power_of_2(n_rows)))
    while rows * T > 4096 and rows > 1:
        rows //= 2
    grid = (H, triton.cdiv(n_rows, rows))
    _delta_adjoint_kernel[grid](dDelta, delta_raw, da_log if need_dalog else dDelta, n_rows, *dDelta.stride(), H=H, NC=nc, T=T, ROWS=rows, NEED_DALOG=bool(need_dalog), num_warps=4)
    return (dDelta, da_log)
