
import os
import torch
import triton
from .fused_fwd_kernel import occ_bucket, len_bucket
import triton.language as tl
from .autotune_policy import siso_autotune_tier
from .common import pdot
from .common import _opv
from .fused_fwd_kernel import _dcs_seg, _same_seg, _env, _mmav3_eight_warp_ok, _needs_i64, _pair_bcast_2d, _pair_swap_1d, _pair_swap_2d
from .common import ACC_SEM, _cos_approx, _sin_approx
from .common import _span_sum, _span_sum_T, _excl_prefix
from .siso_fwd_kernel import _to_opd
from .rotation_active import prefix_half, join_halves, rotate_block
from . import levers as _lv

@triton.jit
def _excl_combine(a1, l1, a2, l2):
    return (a1 + l1 + a2, l2)

@triton.jit
def _excl_prefix_scan(v):
    a, _ = tl.associative_scan((tl.zeros_like(v), v), 0, _excl_combine)
    return a

@triton.jit
def _span_sum_ax1(M, offs_t):
    q, _ = tl.associative_scan((tl.zeros_like(M), M), 1, _excl_combine)
    return tl.sum(tl.where(offs_t[:, None] >= offs_t[None, :], q, 0.0), axis=0)

@triton.jit
def _span_sum_mma(P, Lge, offs_t, NSPLIT: tl.constexpr):
    p1 = P.to(tl.bfloat16)
    r1 = P - p1.to(tl.float32)
    p2 = r1.to(tl.bfloat16)
    if NSPLIT >= 3:
        p3 = (r1 - p2.to(tl.float32)).to(tl.bfloat16)
        W = pdot(Lge, p3)
        W = pdot(Lge, p2, W)
    elif NSPLIT == 2:
        W = pdot(Lge, p2)
    if NSPLIT >= 2:
        W = pdot(Lge, p1, W)
    else:
        W = pdot(Lge, p1)
    return tl.sum(tl.where(offs_t[None, :] < offs_t[:, None], W, 0.0), axis=1)

@triton.jit
def _dth_factor(v, IO_DT: tl.constexpr, PREC: tl.constexpr):
    if IO_DT == 'fp32':
        v = _opv(v, tl.float32, False, PREC)
    return v

def _bwd_configs():
    from .fused_fwd_kernel import autotune_off
    mode = siso_autotune_tier(_env('SISO_AUTOTUNE', '0' if autotune_off() else '1'))
    one = _env('SISO_BWD_ONECFG', '')
    if one:
        w, st, r = one.split(',')
        return [triton.Config({}, num_stages=int(st), num_warps=int(w), maxnreg=None if r in ('', 'None') else int(r))]
    _light = lambda: [triton.Config({}, num_stages=s, num_warps=w) for s in (1, 2, 3) for w in (4, 8)]
    _full = lambda: [triton.Config({}, num_stages=s, num_warps=w, maxnreg=r) for s in (1, 2, 3) for w in (2, 4, 8) for r in (None, 128, 256)]
    from .calibration_ssd import grid
    return grid('siso_bwd', mode, full=_full, light=_light, generic_pin=lambda: [triton.Config({}, num_stages=2, num_warps=8)])
_MMA16 = ('bf16', 'fp16')

def _prune_bwd(configs, named_args, **kwargs):
    gt = kwargs.get('G_SIZE', named_args.get('G_SIZE', 64))
    io = kwargs.get('IO_DT', named_args.get('IO_DT', 'fp32'))
    try:
        cc = torch.cuda.get_device_capability()
    except Exception:
        cc = (9, 0)
    if cc != (9, 0):
        return configs
    if not _mmav3_eight_warp_ok(gt):
        configs = [c for c in configs if c.num_warps < 8]
    if io in _MMA16 and gt * 2 < 128:
        configs = [c for c in configs if not (c.num_warps == 4 and c.num_stages == 1)]
    return configs
_BWD_TUNED: set = set()
_SCHED_DEFAULT = 35
_TMA_ALLOC = False

def _install_tma_allocator():
    global _TMA_ALLOC
    if _TMA_ALLOC:
        return

    def _alloc(size, alignment, stream):
        return torch.empty(size, device='cuda', dtype=torch.int8)
    triton.set_allocator(_alloc)
    _TMA_ALLOC = True

def _tma_refusal(*, tensors, N, G_SIZE):
    if torch.cuda.get_device_capability()[0] < 9:
        return 'TMA needs sm_90 or later'
    for nm, t, row, inner, blk in tensors:
        it = t.element_size()
        if inner != 1:
            return "%s: the descriptor's innermost stride is %d, not 1 (x / dx need K25's token-major layout, SSD_MIMO_SISO_XTM != 0)" % (nm, inner)
        if it > 4:
            return '%s is %d-byte (TMA serves 1, 2 and 4)' % (nm, it)
        if row * it % 16:
            return '%s row stride %d B is not a multiple of 16' % (nm, row * it)
        if blk * it % 16:
            return '%s block row %d B is not a multiple of 16' % (nm, blk * it)
    return None

def _autotune_cache_key(kernel, args, kwargs):
    all_args = {**dict(zip(kernel.arg_names, args)), **kwargs}
    named = {k: v for k, v in all_args.items() if k in kernel.arg_names}
    key = [named[k] for k in kernel.keys if k in named]
    key.extend((str(arg.dtype) for arg in named.values() if hasattr(arg, 'dtype')))
    return tuple(key)

def _bwd_tuning_launch(sig, *, fitted=False) -> bool:
    return not fitted and len(_siso_bwd_scan_kernel.configs) > 1 and (sig not in _siso_bwd_scan_kernel.cache)
_BWD_FIT: dict = {}
_BWD_FIT_TRAIL: dict = {}
_BWD_OOR_FIT: dict = {}

def _bwd_refit_oor(grid, args, kw, device, sig):
    from .smem import device_smem_limit
    limit = device_smem_limit(device)
    cand = _prune_bwd(list(_siso_bwd_scan_kernel.configs), {}, G_SIZE=kw['G_SIZE'], IO_DT=kw['IO_DT'])
    cand = sorted(cand, key=lambda c: (-c.num_warps, -c.num_stages))
    for c in cand:
        opts = dict(num_warps=c.num_warps, num_stages=c.num_stages)
        if c.maxnreg is not None:
            opts['maxnreg'] = c.maxnreg
        try:
            k = _siso_bwd_scan_kernel.fn.warmup(*args, grid=grid, **kw, **opts)
        except Exception:
            continue
        if int(k.metadata.shared) <= limit:
            _BWD_OOR_FIT[sig] = opts
            return opts
    return None

def _bwd_fit_single(grid, args, kw, device, sig):
    cfgs = _siso_bwd_scan_kernel.configs
    if len(cfgs) != 1:
        return None
    c = cfgs[0]
    key = (sig, c.num_warps, c.num_stages, c.maxnreg)
    if key in _BWD_FIT:
        return _BWD_FIT[key]
    from .smem import device_smem_limit, smem_clamp_enabled
    fit, trail = (None, [])
    if smem_clamp_enabled():
        limit = device_smem_limit(device)

        def shared(w, st):
            opts = dict(num_warps=w, num_stages=st)
            if c.maxnreg is not None:
                opts['maxnreg'] = c.maxnreg
            try:
                k = _siso_bwd_scan_kernel.fn.warmup(*args, grid=grid, **kw, **opts)
                r = int(k.metadata.shared)
            except Exception:
                r = None
            trail.append((w, st, r))
            return (r, opts)
        req, _ = shared(c.num_warps, c.num_stages)
        if req is not None and req > limit:
            cand = [triton.Config({}, num_warps=w, num_stages=st) for w in (c.num_warps, c.num_warps // 2, c.num_warps // 4) if w >= 2 for st in range(c.num_stages, 0, -1) if (w, st) != (c.num_warps, c.num_stages)]
            cand = _prune_bwd(cand, {}, G_SIZE=kw['G_SIZE'], IO_DT=kw['IO_DT'])
            for cc in cand:
                r, opts = shared(cc.num_warps, cc.num_stages)
                if r is not None and r <= limit:
                    fit = opts
                    break
    _BWD_FIT[key] = fit
    _BWD_FIT_TRAIL[key] = trail
    return fit

@triton.autotune(configs=_bwd_configs(), key=['T', 'N', 'G_SIZE', 'HAS_TRAP', 'HAS_DSKIP', 'HAS_X0', 'HAS_DSF', 'IO_DT', 'PREC', 'RELOAD', 'NOTRANS', 'GRID_HB', 'TMA_M', 'TMA_X', 'TMA_Y', 'TMA_S', 'SCHED', 'SPAN_MMA', 'DD_SPLIT', 'SPAN_AX1', 'EXCL1D', 'DTH_ROT', 'ROT_KA', 'PFX_OUT', 'NO_DD', 'OCC', 'LB'], prune_configs_by={'early_config_prune': _prune_bwd})
@triton.jit
def _siso_bwd_scan_kernel(Bs_ptr, Cs_ptr, dBs_ptr, dCs_ptr, X_ptr, dY_ptr, SAll_ptr, Delta_ptr, A_ptr, DSF_ptr, Dsk_ptr, QK_ptr, dX_ptr, dDelta_ptr, dQK_ptr, dX0_ptr, dDsk_ptr, Emap_ptr, Dth_ptr, Tail_ptr, s_mb, s_mc, s_mt, s_mh, s_mn, s_xb, s_xh, s_xg, s_xl, s_gb, s_gh, s_gc, s_gt, s_gg, s_sb, s_sh, s_sc, s_sn, s_sg, s_db, s_dh, s_dc, s_dt, s_wb, s_wh, s_wl, s_fb, s_fh, s_fg, s_fn, s_kb, s_kh, s_kg, s_hb, s_hc, s_ht, s_hh, s_hk, NC, H_heads, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr, HAS_TRAP: tl.constexpr, HAS_DSKIP: tl.constexpr, DSKIP_DD: tl.constexpr, HAS_X0: tl.constexpr, HAS_DSF: tl.constexpr, IDX64: tl.constexpr, RELOAD: tl.constexpr=False, NOTRANS: tl.constexpr=False, GRID_HB: tl.constexpr=False, TMA_M: tl.constexpr=False, TMA_X: tl.constexpr=False, TMA_Y: tl.constexpr=False, TMA_S: tl.constexpr=False, SCHED: tl.constexpr=0, SPAN_MMA: tl.constexpr=0, SPAN_AX1: tl.constexpr=False, EXCL1D: tl.constexpr=False, ROT_KA: tl.constexpr=0, DTH_ROT: tl.constexpr=False, PFX_OUT: tl.constexpr=False, NO_DD: tl.constexpr=False, DD_SPLIT: tl.constexpr=False, ABL: tl.constexpr=0, OCC=0, LB=0):
    if GRID_HB:
        h = tl.program_id(0)
        b = tl.program_id(1)
    else:
        b = tl.program_id(0)
        h = tl.program_id(1)
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
    SCHED_ES: tl.constexpr = SCHED.value & 1 != 0
    SCHED_EE: tl.constexpr = SCHED.value & 2 != 0
    SCHED_SR: tl.constexpr = SCHED.value & 4 != 0
    SCHED_RC: tl.constexpr = SCHED.value & 8 != 0
    SCHED_EV: tl.constexpr = SCHED.value & 16 != 0
    SCHED_IL: tl.constexpr = SCHED.value & 32 != 0
    SCHED_RM: tl.constexpr = SCHED.value & 64 != 0
    SCHED_P2: tl.constexpr = SCHED.value & 128 != 0
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.arange(0, G_SIZE)
    causal = offs_t[:, None] >= offs_t[None, :]
    strict = offs_t[:, None] > offs_t[None, :]
    causalT = offs_t[None, :] >= offs_t[:, None]
    strictT = offs_t[None, :] > offs_t[:, None]
    if SPAN_MMA > 0:
        Lge = tl.where(causalT, 1.0, 0.0).to(tl.bfloat16)
    neg_inf = float('-inf')
    if HAS_DSF:
        dS = tl.load(DSF_ptr + b * s_fb + h * s_fh + offs_g[None, :] * s_fg + offs_n[:, None] * s_fn).to(tl.float32)
    else:
        dS = tl.zeros([N, G_SIZE], dtype=tl.float32)
    if HAS_DSKIP:
        dv = tl.load(Dsk_ptr + h * G_SIZE + offs_g)
    if DSKIP_DD:
        dsk_acc = tl.zeros([G_SIZE], dtype=tl.float32)
    if TMA_M or TMA_X or TMA_Y or TMA_S:
        if TMA_M:
            b_desc = tl.make_tensor_descriptor(Bs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N], strides=[s_mt, 1], block_shape=[T, N])
            c_desc = tl.make_tensor_descriptor(Cs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N], strides=[s_mt, 1], block_shape=[T, N])
            db_desc = tl.make_tensor_descriptor(dBs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N], strides=[s_mt, 1], block_shape=[T, N])
            dc_desc = tl.make_tensor_descriptor(dCs_ptr + b * s_mb + h * s_mh, shape=[NC * T, N], strides=[s_mt, 1], block_shape=[T, N])
        if TMA_X:
            x_desc = tl.make_tensor_descriptor(X_ptr + b * s_xb + h * s_xh, shape=[NC * T, G_SIZE], strides=[s_xl, 1], block_shape=[T, G_SIZE])
            dx_desc = tl.make_tensor_descriptor(dX_ptr + b * s_xb + h * s_xh, shape=[NC * T, G_SIZE], strides=[s_xl, 1], block_shape=[T, G_SIZE])
        if TMA_Y:
            dy_desc = tl.make_tensor_descriptor(dY_ptr + b * s_gb + h * s_gh, shape=[NC * T, G_SIZE], strides=[s_gt, 1], block_shape=[T, G_SIZE])
        if TMA_S:
            s_desc = tl.make_tensor_descriptor(SAll_ptr + b * s_sb + h * s_sh, shape=[NC * N, G_SIZE], strides=[s_sn, 1], block_shape=[N, G_SIZE])
    if DTH_ROT:
        rot_sign_n = tl.where(tl.arange(0, N) % 2 == 0, 1.0, -1.0)
    for cr in range(NC):
        c = NC - 1 - cr
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
        seg_end = tl.max(seg_h, axis=0)
        D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end - D_h)), 0.0)
        dec_mask = tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
        if NOTRANS:
            dec_maskT = tl.exp(tl.where((strictT if HAS_TRAP else causalT) & _same_seg(seg_h), a_val * (D_h[None, :] - D_h[:, None]), neg_inf))
        if SCHED_RC:
            dec_mask_2 = tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            if NOTRANS:
                dec_maskT_2 = tl.exp(tl.where((strictT if HAS_TRAP else causalT) & _same_seg(seg_h), a_val * (D_h[None, :] - D_h[:, None]), neg_inf))
        mo = b * s_mb + c * s_mc + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
        mo_ld = b * s_mb + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn if ABL & 512 else mo
        xo = X_ptr + b * s_xb + h * s_xh + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl
        go = dY_ptr + b * s_gb + h * s_gh + c * s_gc + offs_t[:, None] * s_gt + offs_g[None, :] * s_gg
        if TMA_M:
            Bop = _to_opd(b_desc.load([c * T, 0]), IO_DT)
            Cop = _to_opd(c_desc.load([c * T, 0]), IO_DT)
        else:
            Bop = _to_opd(tl.load(Bs_ptr + mo_ld), IO_DT)
            Cop = _to_opd(tl.load(Cs_ptr + mo_ld), IO_DT)
        if TMA_X:
            xop = _to_opd(x_desc.load([c * T, 0]), IO_DT)
        elif RELOAD:
            xop = _to_opd(tl.load(xo), IO_DT)
        if TMA_Y:
            dYop = _to_opd(dy_desc.load([c * T, 0]), IO_DT)
        elif RELOAD:
            dYop = _to_opd(tl.load(go), IO_DT)
        if TMA_X or RELOAD:
            xt = tl.load(xo, volatile=True).to(tl.float32)
        else:
            xt = tl.load(xo).to(tl.float32)
            xop = _to_opd(xt, IO_DT)
        if TMA_Y or RELOAD:
            dY = tl.load(go, volatile=True).to(tl.float32)
        else:
            dY = tl.load(go).to(tl.float32)
            dYop = _to_opd(dY, IO_DT)
        if SCHED_P2:
            pass
        elif ABL & 64:
            S_in = tl.zeros([N, G_SIZE], dtype=SAll_ptr.dtype.element_ty)
        elif TMA_S:
            S_in = s_desc.load([c * N, 0])
        else:
            S_in = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg)
        dSop = _to_opd(dS, IO_DT)
        if ABL & 128:
            dec_mask = tl.where(strict if HAS_TRAP else causal, 1.0, 0.0)
            dec_maskT = tl.where(strictT if HAS_TRAP else causalT, 1.0, 0.0)
        if NOTRANS:
            dScT = pdot(xop, tl.trans(dYop), input_precision=PREC)
            SBT = pdot(Bop, tl.trans(Cop), input_precision=PREC)
            MmT = dScT * dec_maskT
            if SCHED_RC:
                ScoreT = SBT * dec_maskT_2
            else:
                ScoreT = SBT * dec_maskT
            dSc = pdot(dYop, tl.trans(xop), input_precision=PREC)
            Mm = dSc * dec_mask
            Mop = _to_opd(Mm, IO_DT)
        else:
            dSc = pdot(dYop, tl.trans(xop), input_precision=PREC)
            SB = pdot(Cop, tl.trans(Bop), input_precision=PREC)
            Mm = dSc * dec_mask
            Mop = _to_opd(Mm, IO_DT)
            if SCHED_RC:
                Score = SB * dec_mask_2
            else:
                Score = SB * dec_mask
        if SCHED_EE and (not ABL & 1) and (not NO_DD) and (not ABL & 1024) and (not DD_SPLIT):
            if NOTRANS:
                E_intra = _span_sum_T(MmT * SBT, offs_t)
            elif SPAN_MMA > 0:
                E_intra = _span_sum_mma(Mm * SB, Lge, offs_t, SPAN_MMA)
            elif SPAN_AX1:
                E_intra = _span_sum_ax1(Mm * SB, offs_t)
            else:
                E_intra = _span_sum(Mm * SB, offs_t)
        if SCHED_P2:
            S_in = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg)
        Sop = _to_opd(S_in, IO_DT)
        if ABL & 256:
            dYS = tl.zeros([T, N], dtype=tl.float32)
            xdS = tl.zeros([T, N], dtype=tl.float32)
            BdS = tl.zeros([T, G_SIZE], dtype=tl.float32)
        elif not SCHED_IL:
            dYS = pdot(dYop, tl.trans(Sop), input_precision=PREC)
            xdS = pdot(xop, tl.trans(dSop), input_precision=PREC)
            BdS = pdot(Bop, dSop, input_precision=PREC)
        if SCHED_IL and (not ABL & 256):
            dYS = pdot(dYop, tl.trans(Sop), input_precision=PREC)
        dCt = f[:, None] * dYS + pdot(Mop, Bop, input_precision=PREC)
        if DTH_ROT:
            Cre = _dth_factor(tl.load(Cs_ptr + mo, volatile=True).to(tl.float32), IO_DT, PREC)
            if ROT_KA == N // 4:
                ga, _ = prefix_half(dCt, T, N)
                xa, _ = prefix_half(Cre, T, N)
                rs = tl.where(tl.arange(0, N // 2) % 2 == 0, 1.0, -1.0)
                da = _pair_contract(ga, xa, rs, T, N // 2)
                dth_acc = join_halves(da, tl.full((T, N // 4), 0, tl.float32), T, N // 2)
            else:
                dth_acc = _pair_contract(dCt, Cre, rot_sign_n, T, N)
        if SCHED_ES and (not ABL & 4):
            if TMA_M:
                dc_desc.store([c * T, 0], dCt.to(dCs_ptr.dtype.element_ty))
            else:
                tl.store(dCs_ptr + mo, dCt.to(dCs_ptr.dtype.element_ty))
        if SCHED_IL and (not ABL & 256):
            xdS = pdot(xop, tl.trans(dSop), input_precision=PREC)
        if NOTRANS:
            dBt = pdot(_to_opd(MmT, IO_DT), Cop, input_precision=PREC) + dec_agg[:, None] * xdS
        elif SCHED_RM:
            Mop_r = _to_opd(pdot(dYop, tl.trans(xop), input_precision=PREC) * tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf)), IO_DT)
            dBt = pdot(tl.trans(Mop_r), Cop, input_precision=PREC) + dec_agg[:, None] * xdS
        else:
            dBt = pdot(tl.trans(Mop), Cop, input_precision=PREC) + dec_agg[:, None] * xdS
        if DTH_ROT:
            Bre = _dth_factor(tl.load(Bs_ptr + mo, volatile=True).to(tl.float32), IO_DT, PREC)
            if ROT_KA == N // 4:
                ga, _ = prefix_half(dBt, T, N)
                xa, _ = prefix_half(Bre, T, N)
                rs = tl.where(tl.arange(0, N // 2) % 2 == 0, 1.0, -1.0)
                da = _pair_contract(ga, xa, rs, T, N // 2)
                dth_acc += join_halves(da, tl.full((T, N // 4), 0, tl.float32), T, N // 2)
            else:
                dth_acc += _pair_contract(dBt, Bre, rot_sign_n, T, N)
            tl.store(Dth_ptr + b * s_hb + c * s_hc + h * s_hh + offs_t[:, None] * s_ht + tl.arange(0, N // 2)[None, :] * s_hk, dth_acc)
        if SCHED_ES and (not ABL & 2):
            if TMA_M:
                db_desc.store([c * T, 0], dBt.to(dBs_ptr.dtype.element_ty))
            else:
                tl.store(dBs_ptr + mo, dBt.to(dBs_ptr.dtype.element_ty))
        if SCHED_IL and (not ABL & 256):
            BdS = pdot(Bop, dSop, input_precision=PREC)
        if NOTRANS:
            dx = pdot(_to_opd(ScoreT, IO_DT), dYop, input_precision=PREC) + dec_agg[:, None] * BdS
        elif SCHED_RM:
            Sc_r = pdot(Cop, tl.trans(Bop), input_precision=PREC) * tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            dx = pdot(tl.trans(_to_opd(Sc_r, IO_DT)), dYop, input_precision=PREC) + dec_agg[:, None] * BdS
        else:
            dx = pdot(tl.trans(_to_opd(Score, IO_DT)), dYop, input_precision=PREC) + dec_agg[:, None] * BdS
        tbase = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
        if HAS_TRAP:
            dx += tl.load(QK_ptr + tbase)[:, None] * dY
            tl.store(dQK_ptr + tbase, tl.sum(dY * xt, axis=1))
        if DSKIP_DD:
            dsk_acc += tl.sum(dY * xt, axis=0)
        if HAS_DSKIP:
            dx += dY * dv[None, :]
        if not ABL & 16:
            if TMA_X:
                dx_desc.store([c * T, 0], dx.to(dX_ptr.dtype.element_ty))
            else:
                tl.store(dX_ptr + b * s_xb + h * s_xh + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl, dx.to(dX_ptr.dtype.element_ty))
        if SCHED_EV and (not ABL & 1) and (not NO_DD):
            Vj_e = dec_agg * tl.sum(BdS * xt, axis=1)
        if not ABL & 1 and (not NO_DD):
            if DD_SPLIT and PFX_OUT:
                E = tl.zeros([T], dtype=tl.float32)
            elif DD_SPLIT:
                E = tl.load(Emap_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt)
            elif ABL & 1024:
                E = tl.zeros([T], dtype=tl.float32)
            elif SCHED_EE:
                E = E_intra
            elif NOTRANS:
                E = _span_sum_T(MmT * SBT, offs_t)
            elif SPAN_MMA > 0:
                E = _span_sum_mma(Mm * SB, Lge, offs_t, SPAN_MMA)
            elif SPAN_AX1:
                E = _span_sum_ax1(Mm * SB, offs_t)
            else:
                E = _span_sum(Mm * SB, offs_t)
            if not ABL & 2048 and (not DD_SPLIT):
                if ABL & 256:
                    QS = tl.zeros([T, G_SIZE], dtype=tl.float32)
                else:
                    QS = pdot(Cop, Sop, input_precision=PREC)
                E += tl.cumsum(f * tl.sum(QS * dY, axis=1), axis=0, reverse=True)
            if SCHED_EV:
                Vj = Vj_e
            else:
                Vj = dec_agg * tl.sum(BdS * xt, axis=1)
            if PFX_OUT:
                pass
            elif not ABL & 4096:
                if EXCL1D:
                    E += _excl_prefix_scan(Vj)
                else:
                    E += _excl_prefix(Vj, offs_t)
            tail = 0.0
            if not ABL & 8:
                if SCHED_SR:
                    S_re = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg, volatile=True).to(tl.float32)
                    tail += alpha * tl.sum(S_re * dS)
                else:
                    tail += alpha * tl.sum(S_in.to(tl.float32) * dS)
            if PFX_OUT:
                tl.store(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, Vj)
                tl.store(Tail_ptr + (b * H_heads + h) * NC + c, tail)
            else:
                E += tail
                tl.store(dDelta_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, a_val * E)
        if not SCHED_ES and (not ABL & 2):
            if TMA_M:
                db_desc.store([c * T, 0], dBt.to(dBs_ptr.dtype.element_ty))
            else:
                tl.store(dBs_ptr + mo, dBt.to(dBs_ptr.dtype.element_ty))
        if not SCHED_ES and (not ABL & 4):
            if TMA_M:
                dc_desc.store([c * T, 0], dCt.to(dCs_ptr.dtype.element_ty))
            else:
                tl.store(dCs_ptr + mo, dCt.to(dCs_ptr.dtype.element_ty))
        if not ABL & 32:
            dS = alpha * dS + pdot(tl.trans(Cop), _to_opd(dY * f[:, None], IO_DT), input_precision=PREC)
    if HAS_X0:
        tl.store(dX0_ptr + b * s_fb + h * s_fh + offs_g[None, :] * s_fg + offs_n[:, None] * s_fn, dS)
    if DSKIP_DD:
        tl.store(dDsk_ptr + b * s_kb + h * s_kh + offs_g * s_kg, dsk_acc)

@triton.jit
def _siso_ddelta_map_kernel(Bs_ptr, Cs_ptr, X_ptr, dY_ptr, SAll_ptr, Delta_ptr, A_ptr, E_ptr, s_mb, s_mc, s_mt, s_mh, s_mn, s_xb, s_xh, s_xg, s_xl, s_gb, s_gh, s_gc, s_gt, s_gg, s_sb, s_sh, s_sc, s_sn, s_sg, s_db, s_dh, s_dc, s_dt, H, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr, HAS_TRAP: tl.constexpr, IDX64: tl.constexpr, SPAN_MMA: tl.constexpr):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
        c = c.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.arange(0, G_SIZE)
    causal = offs_t[:, None] >= offs_t[None, :]
    strict = offs_t[:, None] > offs_t[None, :]
    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
    D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
    f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
    dec_mask = tl.exp(tl.where((strict if HAS_TRAP else causal) & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), float('-inf')))
    mo = b * s_mb + c * s_mc + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
    Bop = _to_opd(tl.load(Bs_ptr + mo), IO_DT)
    Cop = _to_opd(tl.load(Cs_ptr + mo), IO_DT)
    xop = _to_opd(tl.load(X_ptr + b * s_xb + h * s_xh + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl), IO_DT)
    dY = tl.load(dY_ptr + b * s_gb + h * s_gh + c * s_gc + offs_t[:, None] * s_gt + offs_g[None, :] * s_gg).to(tl.float32)
    dYop = _to_opd(dY, IO_DT)
    S_in = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg)
    Sop = _to_opd(S_in, IO_DT)
    dSc = pdot(dYop, tl.trans(xop), input_precision=PREC)
    SB = pdot(Cop, tl.trans(Bop), input_precision=PREC)
    P = dSc * dec_mask * SB
    if SPAN_MMA > 0:
        Lge = tl.where(offs_t[None, :] >= offs_t[:, None], 1.0, 0.0).to(tl.bfloat16)
        E = _span_sum_mma(P, Lge, offs_t, SPAN_MMA)
    else:
        E = _span_sum(P, offs_t)
    QS = pdot(Cop, Sop, input_precision=PREC)
    E += tl.cumsum(f * tl.sum(QS * dY, axis=1), axis=0, reverse=True)
    tl.store(E_ptr + b * s_db + h * s_dh + c * s_dc + offs_t * s_dt, E)

@triton.jit
def _pair_contract(g, u, rot_sign, T: tl.constexpr, NS: tl.constexpr):
    q = rot_sign[None, :] * g * _pair_swap_2d(u, T, NS)
    e, o = tl.split(tl.reshape(q, (T, NS // 2, 2)))
    return e + o

def _adj_configs():
    mode = siso_autotune_tier(_env('SISO_ADJ_AUTOTUNE', '0'), adjoint=True)
    _generic = lambda: [triton.Config({}, num_stages=int(_env('SISO_ADJ_STAGES', '2')), num_warps=int(_env('SISO_ADJ_WARPS', '8')))]
    _light = lambda: [triton.Config({}, num_stages=st, num_warps=w) for st in (1, 2, 3) for w in (2, 4, 8)]
    _full = lambda: [triton.Config({}, num_stages=st, num_warps=w, maxnreg=r) for st in (1, 2, 3) for w in (2, 4, 8) for r in (None, 128, 256)]
    if mode == 'pinned' and (_env('SISO_ADJ_WARPS', '') or _env('SISO_ADJ_STAGES', '')):
        return _generic()
    from .calibration_ssd import grid
    return grid('siso_adj', mode, full=_full, light=_light, generic_pin=_generic)
_ADJ_TUNED: set = set()

def _adj_tuning_launch(sig) -> bool:
    return len(_siso_rot_adjoint_kernel.configs) > 1 and sig not in _siso_rot_adjoint_kernel.cache

@triton.autotune(configs=_adj_configs(), key=['T', 'N', 'NS', 'HAS_THETA_H', 'HAS_ROT_BIAS', 'HAS_BETA', 'HAS_GAMMA', 'BIAS_PRE_ROT', 'HAS_DW', 'HAS_TRAP', 'THETA_PER_HEAD', 'NEED_DTHETA', 'RAW_COMMON', 'TRAP_DTH_EXCL', 'NEED_DBETA', 'NEED_DGAMMA', 'NEED_DTHH', 'NEED_DRB', 'ROT_KA', 'ROT_STATIC', 'OCC'])
@triton.jit
def _siso_rot_adjoint_kernel(B_ptr, C_ptr, dBs_ptr, dCs_ptr, Theta_ptr, ThetaH_ptr, RotBias_ptr, Beta_ptr, Gamma_ptr, W_ptr, Gam_ptr, dQK_ptr, dB_ptr, dC_ptr, dTheta_ptr, dThetaH_ptr, dRotBias_ptr, dBeta_ptr, dGamma_ptr, dW_ptr, dG_ptr, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_qb, s_qc, s_qt, s_qn, s_mb, s_mc, s_mt, s_mh, s_mn, s_thb, s_thc, s_tht, s_thh, s_thk, s_wb, s_wh, s_wl, s_pb, s_ph, s_pk, s_pl, H, T: tl.constexpr, N: tl.constexpr, NS: tl.constexpr, HAS_THETA_H: tl.constexpr, HAS_ROT_BIAS: tl.constexpr, HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr, BIAS_PRE_ROT: tl.constexpr, HAS_DW: tl.constexpr, HAS_TRAP: tl.constexpr, THETA_PER_HEAD: tl.constexpr, NEED_DTHETA: tl.constexpr, RAW_COMMON: tl.constexpr, IDX64: tl.constexpr, TRAP_DTH_EXCL: tl.constexpr=False, NEED_DBETA: tl.constexpr=True, NEED_DGAMMA: tl.constexpr=True, NEED_DTHH: tl.constexpr=True, NEED_DRB: tl.constexpr=True, ROT_KA: tl.constexpr=0, ROT_STATIC: tl.constexpr=False, OCC=0):
    tl.static_assert(not RAW_COMMON or (BIAS_PRE_ROT and (not HAS_THETA_H) and (not HAS_ROT_BIAS)))
    c = tl.program_id(0)
    b = tl.program_id(1)
    if IDX64:
        b = b.to(tl.int64)
        c = c.to(tl.int64)
    offs_t = tl.arange(0, T)
    NB: tl.constexpr = N // NS
    rot_sign = tl.where(tl.arange(0, NS) % 2 == 0, 1.0, -1.0)
    for nb_static in tl.static_range(NB if ROT_STATIC else 1):
        for nb_loop in range(1 if ROT_STATIC else NB):
            nb = nb_static if ROT_STATIC else nb_loop
            offs_n = nb * NS + tl.arange(0, NS)
            offs_k = nb * (NS // 2) + tl.arange(0, NS // 2)
            pidx = offs_n // 2
            Braw = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
            Craw = tl.load(C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
            dB_acc = tl.zeros([T, NS], dtype=tl.float32)
            dC_acc = tl.zeros([T, NS], dtype=tl.float32)
            if NEED_DTHETA and (not THETA_PER_HEAD):
                dth_sh = tl.zeros([T, NS // 2], dtype=tl.float32)
            for h in range(H):
                active = (ROT_KA == 0) | (nb * NS < 2 * ROT_KA)
                if active:
                    th = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk).to(tl.float32)
                    cs = _pair_bcast_2d(_cos_approx(th), T, NS)
                    sn = rot_sign[None, :] * _pair_bcast_2d(_sin_approx(th), T, NS)
                else:
                    cs = tl.full((T, NS), 1, tl.float32)
                    sn = tl.full((T, NS), 0, tl.float32)
                uB = Braw
                uC = Craw
                if HAS_BETA:
                    beta_v = tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)
                if HAS_GAMMA:
                    gam_v = tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)
                if BIAS_PRE_ROT:
                    if HAS_BETA:
                        uB = uB + beta_v[None, :]
                    if HAS_GAMMA:
                        uC = uC + gam_v[None, :]
                if RAW_COMMON:
                    mo = b * s_mb + c * s_mc + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
                    dBt = tl.load(dBs_ptr + mo).to(tl.float32)
                    dCt = tl.load(dCs_ptr + mo).to(tl.float32)
                    if ROT_KA == N // 4:
                        dBraw = rotate_block(dBt, cs, sn, active, T, NS, True)
                    else:
                        dBraw = cs * dBt - sn * _pair_swap_2d(dBt, T, NS)
                    if ROT_KA == N // 4:
                        dCraw = rotate_block(dCt, cs, sn, active, T, NS, True)
                    else:
                        dCraw = cs * dCt - sn * _pair_swap_2d(dCt, T, NS)
                    tb = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
                    if HAS_DW:
                        tl.store(dW_ptr + b * s_pb + h * s_ph + nb * s_pk + (c * T + offs_t) * s_pl, tl.sum(dBraw * uB, axis=1))
                        dBraw = dBraw * tl.load(W_ptr + tb)[:, None]
                    if NEED_DTHETA:
                        dth = _pair_contract(dBraw, uB, rot_sign, T, NS) + _pair_contract(dCraw, uC, rot_sign, T, NS)
                    if HAS_TRAP:
                        dqk = tl.load(dQK_ptr + tb)
                        rho = cs * cs + sn * sn
                        tl.store(dG_ptr + b * s_pb + h * s_ph + nb * s_pk + (c * T + offs_t) * s_pl, dqk * tl.sum(rho * uB * uC, axis=1))
                        ddot = (dqk * tl.load(Gam_ptr + tb))[:, None] * rho
                        dBraw = dBraw + ddot * uC
                        dCraw = dCraw + ddot * uB
                else:
                    if ROT_KA == N // 4:
                        vB = rotate_block(uB, cs, sn, active, T, NS, False)
                    else:
                        vB = cs * uB + sn * _pair_swap_2d(uB, T, NS)
                    if ROT_KA == N // 4:
                        vC = rotate_block(uC, cs, sn, active, T, NS, False)
                    else:
                        vC = cs * uC + sn * _pair_swap_2d(uC, T, NS)
                    if HAS_ROT_BIAS:
                        rbv = tl.load(RotBias_ptr + h * (N // 2) + pidx)
                        crb = _cos_approx(rbv)
                        srb = rot_sign * _sin_approx(rbv)
                        qB = crb[None, :] * vB + srb[None, :] * _pair_swap_2d(vB, T, NS)
                    else:
                        qB = vB
                    Btil = qB + beta_v[None, :] if HAS_BETA and (not BIAS_PRE_ROT) else qB
                    pC = vC + gam_v[None, :] if HAS_GAMMA and (not BIAS_PRE_ROT) else vC
                    if HAS_THETA_H:
                        thhv = tl.load(ThetaH_ptr + h * (N // 2) + pidx)
                        chh = _cos_approx(thhv)
                        shh = rot_sign * _sin_approx(thhv)
                        Ctil = chh[None, :] * pC + shh[None, :] * _pair_swap_2d(pC, T, NS)
                    else:
                        Ctil = pC
                    mo = b * s_mb + c * s_mc + h * s_mh + offs_t[:, None] * s_mt + offs_n[None, :] * s_mn
                    dBt = tl.load(dBs_ptr + mo).to(tl.float32)
                    dCt = tl.load(dCs_ptr + mo).to(tl.float32)
                    tb = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
                    pbase = dW_ptr + b * s_pb + h * s_ph + nb * s_pk + (c * T + offs_t) * s_pl
                    if HAS_DW:
                        tl.store(pbase, tl.sum(dBt * Btil, axis=1))
                        dBt = dBt * tl.load(W_ptr + tb)[:, None]
                    if TRAP_DTH_EXCL and HAS_TRAP and BIAS_PRE_ROT:
                        dBt0 = dBt
                        dCt0 = dCt
                    if HAS_TRAP:
                        dqk = tl.load(dQK_ptr + tb)
                        tl.store(dG_ptr + b * s_pb + h * s_ph + nb * s_pk + (c * T + offs_t) * s_pl, dqk * tl.sum(Btil * Ctil, axis=1))
                        ddot = (dqk * tl.load(Gam_ptr + tb))[:, None]
                        dBt = dBt + ddot * Ctil
                        dCt = dCt + ddot * Btil
                    if HAS_BETA and (not BIAS_PRE_ROT) and NEED_DBETA:
                        tl.atomic_add(dBeta_ptr + h * N + offs_n, tl.sum(dBt, axis=0), sem=ACC_SEM)
                    if HAS_ROT_BIAS:
                        dvB = crb[None, :] * dBt - srb[None, :] * _pair_swap_2d(dBt, T, NS)
                        if NEED_DRB:
                            tl.atomic_add(dRotBias_ptr + h * (N // 2) + offs_k, tl.sum(_pair_contract(dvB, vB, rot_sign, T, NS), axis=0), sem=ACC_SEM)
                    else:
                        dvB = dBt
                    if HAS_THETA_H:
                        dpC = chh[None, :] * dCt - shh[None, :] * _pair_swap_2d(dCt, T, NS)
                        if NEED_DTHH:
                            tl.atomic_add(dThetaH_ptr + h * (N // 2) + offs_k, tl.sum(_pair_contract(dpC, pC, rot_sign, T, NS), axis=0), sem=ACC_SEM)
                    else:
                        dpC = dCt
                    if HAS_GAMMA and (not BIAS_PRE_ROT) and NEED_DGAMMA:
                        tl.atomic_add(dGamma_ptr + h * N + offs_n, tl.sum(dpC, axis=0), sem=ACC_SEM)
                    dvC = dpC
                    if ROT_KA == N // 4:
                        dBraw = rotate_block(dvB, cs, sn, active, T, NS, True)
                    else:
                        dBraw = cs * dvB - sn * _pair_swap_2d(dvB, T, NS)
                    if ROT_KA == N // 4:
                        dCraw = rotate_block(dvC, cs, sn, active, T, NS, True)
                    else:
                        dCraw = cs * dvC - sn * _pair_swap_2d(dvC, T, NS)
                    if NEED_DTHETA and TRAP_DTH_EXCL and HAS_TRAP and BIAS_PRE_ROT:
                        if HAS_ROT_BIAS:
                            dvB0 = crb[None, :] * dBt0 - srb[None, :] * _pair_swap_2d(dBt0, T, NS)
                        else:
                            dvB0 = dBt0
                        if HAS_THETA_H:
                            dvC0 = chh[None, :] * dCt0 - shh[None, :] * _pair_swap_2d(dCt0, T, NS)
                        else:
                            dvC0 = dCt0
                        if ROT_KA == N // 4:
                            dBraw0 = rotate_block(dvB0, cs, sn, active, T, NS, True)
                        else:
                            dBraw0 = cs * dvB0 - sn * _pair_swap_2d(dvB0, T, NS)
                        if ROT_KA == N // 4:
                            dCraw0 = rotate_block(dvC0, cs, sn, active, T, NS, True)
                        else:
                            dCraw0 = cs * dvC0 - sn * _pair_swap_2d(dvC0, T, NS)
                if NEED_DTHETA:
                    if not RAW_COMMON:
                        if TRAP_DTH_EXCL and HAS_TRAP and BIAS_PRE_ROT:
                            dth = _pair_contract(dBraw0, uB, rot_sign, T, NS) + _pair_contract(dCraw0, uC, rot_sign, T, NS)
                        else:
                            dth = _pair_contract(dBraw, uB, rot_sign, T, NS) + _pair_contract(dCraw, uC, rot_sign, T, NS)
                    if THETA_PER_HEAD:
                        tl.store(dTheta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk, dth)
                    else:
                        dth_sh += dth
                if BIAS_PRE_ROT:
                    if HAS_BETA and NEED_DBETA:
                        tl.atomic_add(dBeta_ptr + h * N + offs_n, tl.sum(dBraw, axis=0), sem=ACC_SEM)
                    if HAS_GAMMA and NEED_DGAMMA:
                        tl.atomic_add(dGamma_ptr + h * N + offs_n, tl.sum(dCraw, axis=0), sem=ACC_SEM)
                dB_acc += dBraw
                dC_acc += dCraw
            if NEED_DTHETA and (not THETA_PER_HEAD):
                tl.store(dTheta_ptr + b * s_thb + c * s_thc + offs_t[:, None] * s_tht + offs_k[None, :] * s_thk, dth_sh)
            tl.store(dB_ptr + b * s_qb + c * s_qc + offs_t[:, None] * s_qt + offs_n[None, :] * s_qn, dB_acc)
            tl.store(dC_ptr + b * s_qb + c * s_qc + offs_t[:, None] * s_qt + offs_n[None, :] * s_qn, dC_acc)

@triton.jit
def _siso_scalar_reduce_kernel(WP, GP, DT, LAM, DDT, DLAM, swb: tl.constexpr, swh: tl.constexpr, swk: tl.constexpr, swt: tl.constexpr, sgb: tl.constexpr, sgh: tl.constexpr, sgk: tl.constexpr, sgt: tl.constexpr, sdb: tl.constexpr, sdh: tl.constexpr, sdt: tl.constexpr, slb: tl.constexpr, slh: tl.constexpr, slt: tl.constexpr, H: tl.constexpr, L: tl.constexpr, NB: tl.constexpr, HAS_TRAP: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(1).to(tl.int64)
    b, h = (bh // H, bh % H)
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = t < L
    w = tl.full((BLOCK,), 0.0, tl.float32)
    if HAS_TRAP:
        g = tl.full((BLOCK,), 0.0, tl.float32)
        previous = tl.full((BLOCK,), 0.0, tl.float32)
    for k in tl.static_range(NB):
        w += tl.load(WP + b * swb + h * swh + k * swk + t * swt, mask, other=0).to(tl.float32)
        if HAS_TRAP:
            g += tl.load(GP + b * sgb + h * sgh + k * sgk + t * sgt, mask, other=0).to(tl.float32)
            previous += tl.load(WP + b * swb + h * swh + k * swk + (t - 1) * swt, mask & (t > 0), other=0).to(tl.float32)
    out = bh * L + t
    if HAS_TRAP:
        dt = tl.load(DT + b * sdb + h * sdh + t * sdt, mask, other=0)
        lam = tl.load(LAM + b * slb + h * slh + t * slt, mask, other=0)
        total = w + g
        tl.store(DDT + out, lam * total + (1.0 - lam) * previous, mask)
        tl.store(DLAM + out, dt * (total - previous), mask)
    else:
        tl.store(DDT + out, w, mask)

def _siso_scalar_reduce(dW_p, dG_p=None, dw=None, trap_lam=None):
    B, H, NB, L = dW_p.shape
    has_trap = trap_lam is not None
    if has_trap and (dG_p is None or dw is None):
        raise ValueError('trapezoid scalar adjoint needs dG_p and dw')
    dDW = torch.empty((B, H, L), dtype=torch.float32, device=dW_p.device)
    dLam = torch.empty_like(dDW) if has_trap else None
    _siso_scalar_reduce_kernel[triton.cdiv(L, 256), B * H](dW_p, dG_p if has_trap else dW_p, dw if has_trap else dW_p, trap_lam if has_trap else dW_p, dDW, dLam if has_trap else dDW, *dW_p.stride(), *(dG_p.stride() if has_trap else (0, 0, 0, 0)), *(dw.stride() if has_trap else (0, 0, 0)), *(trap_lam.stride() if has_trap else (0, 0, 0)), H=H, L=L, NB=NB, HAS_TRAP=has_trap, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return (dDW, dLam)

def siso_chunk_bwd(aux, Ctil, Btil, x_raw, delta_cumsum, a_log, S_all, dY, *, theta, theta_h=None, rot_bias=None, beta=None, gamma=None, bias_pre_rot=False, dw=None, trap_lam=None, x0=None, d_skip=None, dS_final=None, precision='tf32', dskip_dd=False, needs=None):
    from .fused_fwd_kernel import dot_precision
    precision = dot_precision(precision)
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_raw.shape[2]
    N = Btil.shape[3]
    L = nc * T
    dev = x_raw.device
    f32 = torch.float32
    rot_ka = aux.get('rot_ka_bwd', aux.get('rot_ka', N // 2))
    Bs, Cs = (aux['Bs'], aux['Cs'])
    m_st = aux.get('m_st') or (list(Bs.stride()) if Bs is not None else [0, 0, 0, 0, 0])
    io_tag = aux['io_tag']
    has_trap = trap_lam is not None
    from . import calibration_ssd as _pins
    _pins.note_call('siso_bwd', T=T, N=N, io=io_tag)
    _pins.activate(True, Bsz, H, T, io=io_tag)
    _pins.note_call('siso_adj', T=T, N=N, io=io_tag)
    has_dw = dw is not None
    has_dskip = d_skip is not None
    nd = dict(ddelta=True, dtheta=True, dthh=True, drb=True, dbeta=True, dgamma=True, dx0=True, ddw=True, dlam=True)
    if needs is not None:
        nd.update(needs)
    if not nd['dx0']:
        x0 = None
    from .siso_route import siso_xtm
    xtm_mode = siso_xtm()
    x_k = aux.get('xt')
    if xtm_mode == '0' or x_k is None:
        x_k = x_raw
        x_st = (x_raw.stride(0), x_raw.stride(1), x_raw.stride(2), x_raw.stride(4))
        dXt = torch.empty_like(x_raw)
        dX = dXt
    else:
        x_st = (x_k.stride(0), x_k.stride(1), x_k.stride(3), x_k.stride(2))
        dXt = torch.empty_like(x_k)
        dX = None
    dDelta = torch.empty(Bsz, H, nc, T, device=dev, dtype=f32)
    dQK = torch.empty(Bsz, H, L, device=dev, dtype=f32) if has_trap else torch.empty(1, device=dev, dtype=f32)
    dX0 = torch.empty(Bsz, H, G_SIZE, N, device=dev, dtype=f32) if x0 is not None else torch.empty(1, device=dev, dtype=f32)
    dskip_dd = bool(dskip_dd and has_dskip)
    dDsk_p = torch.empty(Bsz, H, G_SIZE, device=dev, dtype=f32) if dskip_dd else torch.empty(1, device=dev, dtype=f32)
    dsk_c = d_skip.float().contiguous().reshape(H, G_SIZE) if has_dskip else torch.empty(1, device=dev, dtype=f32)
    dsf = dS_final if dS_final is not None else torch.empty(1, device=dev, dtype=f32)
    qk_f = aux['dot'] if has_trap else torch.empty(1, device=dev, dtype=f32)
    w_st = dw.stride() if has_dw else (0, 0, 0)
    f_st = dS_final.stride() if dS_final is not None else dX0.stride() if x0 is not None else (0, 0, 0, 0)
    k_st = dDsk_p.stride() if dskip_dd else (0, 0, 0)
    idx64 = _needs_i64(x_k, Bs, Cs, S_all, dY, dXt, dDelta, qk_f if has_trap else None)
    abl = int(_env('SISO_BWD_ABL', '0'))
    sched = int(_env('SISO_BWD_SCHED', str(_SCHED_DEFAULT)), 0)
    reload_xdy = _env('SISO_BWD_RELOAD', '0') == '1'
    notrans = _env('SISO_BWD_NOTRANS', '0') == '1'
    grid_hb = _env('SISO_BWD_GRID_HB', '0') == '1'
    span_mma = int(_env('SPANDELTA_MMA', '0'))
    if span_mma not in (0, 1, 2, 3):
        raise ValueError('SSD_MIMO_SPANDELTA_MMA must be 0, 1, 2 or 3')
    if span_mma and (io_tag not in ('bf16', 'fp16') or notrans):
        span_mma = 0
    dd_split = _env('SISO_BWD_DDSPLIT', '0') == '1' and (not notrans) and nd['ddelta'] and (not abl)
    e_map = torch.empty(Bsz, H, nc, T, device=dev, dtype=f32) if dd_split else dDelta
    inplace = _env('SISO_INPLACE', '1') == '1'
    dmap32 = _env('SISO_DMAP_F32', '0') == '1'
    span_ax1 = _env('SISO_SPAN_AX1', '0') == '1' and (not span_mma)
    excl1d = _env('SISO_EXCL1D', '0') == '1'
    from . import siso_bwd_m3s as _m3s
    m3s_why = _m3s.refusal(Bs=Bs, Cs=Cs, x=x_k, dY=dY, S_all=S_all, delta_cumsum=delta_cumsum, T=T, N=N, G=G_SIZE, io_tag=io_tag, m_st=m_st, xtm=xtm_mode, abl=abl, x0=x0, dskip_dd=dskip_dd) if _m3s.enabled() else 'off'
    _m3s._LAST.update(used=False, why=m3s_why)
    if _env('M3S_TRACE', '0') == '1':
        import sys as _sys
        print('M3S_TRACE route=%s' % ('m3s' if m3s_why is None else m3s_why), file=_sys.stderr, flush=True)
    if m3s_why is None and nd['ddelta']:
        dd_split = True
        e_map = torch.empty(Bsz, H, nc, T, device=dev, dtype=f32)
    pfx_out = dd_split and (_env('SISO_PFX_OUT', '0') == '1' or m3s_why is None)
    m3s_nodd = m3s_why is None and (not nd['ddelta'])
    tail_buf = torch.empty(Bsz, H, nc, device=dev, dtype=f32) if pfx_out else torch.empty(1, device=dev, dtype=f32)
    from .siso_route import dth_rot_can_serve
    dth_ok = dth_rot_can_serve(io_dtype=io_tag, bias_pre_rot=bias_pre_rot, beta=beta, gamma=gamma, theta=theta, N=N)
    dth_rot = bool(nd['dtheta']) and dth_ok
    if dth_rot:
        dth_buf = torch.empty(Bsz, nc, T, H, N // 2, device=dev, dtype=f32)
        dth_st = list(dth_buf.stride())
    else:
        dth_buf = torch.empty(1, device=dev, dtype=f32)
        dth_st = [0, 0, 0, 0, 0]
    from . import gluon_sm90_siso_bwd as _g90s
    g90_served = False
    g90_adj = None
    g90_dth = False
    if _g90s.enabled(dev):
        x_g, dX_g = (x_k, dXt)
        if xtm_mode == '0' and x_k.dim() == 5 and (x_k.shape[3] == 1) and (not x_k.is_contiguous()) and (not _g90s.packed_x(x_k)):
            x_g = x_k.contiguous()
            dX_g = torch.empty_like(x_g)
        g90_why = _g90s.refusal(Bs=Bs, Cs=Cs, dBs=Bs, dCs=Cs, x=x_g, dX=dX_g, S_all=S_all, dY=dY, T=T, N=N, G=G_SIZE, io_tag=io_tag, m_st=m_st, dskip_dd=dskip_dd, xtm=xtm_mode, abl=abl, dS_final=dS_final, x0=x0)
        if g90_why is None and (not inplace):
            dBs_g, dCs_g = (torch.empty_like(Bs), torch.empty_like(Cs))
        else:
            dBs_g, dCs_g = (Bs, Cs)
        _g90s._LAST_ROUTE.update(used=False, why=g90_why)
        if _env('GLUON90_SISO_TRACE', '0') == '1':
            import sys as _sys
            print('G90S_TRACE route=%s' % ('gluon' if g90_why is None else g90_why), file=_sys.stderr, flush=True)
        g90_adj = None
        if g90_why is None and _g90s.adj_enabled():
            raw_common_req = _env('SISO_ADJ_RAW_COMMON', '0') == '1' and bool(bias_pre_rot) and (theta_h is None) and (rot_bias is None)
            need_dth = bool(nd['dtheta'])
            adj_why = _g90s.adj_refusal(theta=theta, theta_h=theta_h, rot_bias=rot_bias, bias_pre_rot=bias_pre_rot, beta=beta, gamma=gamma, dw=dw, raw_common=raw_common_req, need_dtheta=need_dth, Btil=Btil, Ctil=Ctil, aux=aux, qk_f=qk_f, has_trap=has_trap, N=N)
            _g90s._LAST_ROUTE['adj_why'] = adj_why
            if adj_why is None:
                g90_adj = dict(theta=theta, Btil=Btil, Ctil=Ctil, dTheta=torch.empty_like(theta, dtype=f32), w=aux['scale'] if has_trap else dw, dW_p=torch.empty(Bsz, H, 1, L, device=dev, dtype=f32))
                if has_trap:
                    g90_adj['gam'] = aux['gam']
                    g90_adj['dG_p'] = torch.empty(Bsz, H, 1, L, device=dev, dtype=f32)
                if beta is not None:
                    g90_adj['beta'] = beta.float().contiguous().reshape(H, N)
                    g90_adj['dBeta'] = torch.zeros(H, N, device=dev, dtype=f32)
                if gamma is not None:
                    g90_adj['gamma'] = gamma.float().contiguous().reshape(H, N)
                    g90_adj['dGamma'] = torch.zeros(H, N, device=dev, dtype=f32)
        g90_dth = g90_why is None and g90_adj is None and (_env('GLUON90_SISO_DTH_ROT', '1') == '1') and bool(nd['dtheta']) and (bool(bias_pre_rot) or (beta is None and gamma is None)) and (theta.shape[-1] == N // 2)
        if g90_dth and (not dth_rot):
            dth_buf = torch.empty(Bsz, nc, T, H, N // 2, device=dev, dtype=f32)
        if g90_why is None:
            _g90s._LAST_ROUTE['x_copy'] = x_g is not x_k
            _g90s.launch(Bs=Bs, Cs=Cs, dBs=dBs_g, dCs=dCs_g, x=x_g, dX=dX_g, S_all=S_all, dY=dY, delta_cumsum=delta_cumsum, a_log=a_log, dsf=dsf, dsk_c=dsk_c, qk_f=qk_f, dDelta=dDelta, dQK=dQK, dX0=dX0, has_trap=has_trap, has_dskip=has_dskip, has_dsf=dS_final is not None, has_x0=x0 is not None, f_st=f_st, w_st=w_st, idx64=idx64, adj=g90_adj, dth=dth_buf if g90_dth else None, no_dd=not nd['ddelta'], rot_ka=aux.get('rot_ka_bwd', aux.get('rot_ka', N // 2)))
            if dX_g is not dXt:
                dXt.copy_(dX_g)
            dBs, dCs = (dBs_g, dCs_g)
            g90_served = True
    if nd['dtheta'] and (not g90_served) and (not dth_rot) and (theta is not None):
        if io_tag not in ('bf16', 'fp16', 'fp32'):
            why = '%s IO: DTH_ROT serves bf16 / fp16 / fp32 IO' % io_tag
        elif not (bool(bias_pre_rot) or (beta is None and gamma is None)):
            why = 'a post-rotation bias (beta/gamma without bias_pre_rot)'
        else:
            why = 'theta is not the cumulated per-pair phase (theta.shape[-1]=%d != N//2=%d)' % (theta.shape[-1], N // 2)
        raise NotImplementedError('SISO backward refuses this call by name: it needs dL/dtheta, the Hopper Gluon body (gluon_sm90_siso_bwd) does not serve it, and the rotated-frame angle adjoint DTH_ROT cannot serve it either (%s). DTH_ROT is a REQUIREMENT here, not an option (OPUS_implementation20261002.txt task 1) -- re-run this call with SSD_MIMO_SISO_FAST=0 to use the general fused route instead.' % why)
    tma_mode = _env('SISO_BWD_TMA', '0')
    want = 'mxys' if tma_mode in ('1', 'all', 'require') else '' if tma_mode == '0' else tma_mode
    groups = dict(m=[('Bs', Bs, Bs.stride(2), Bs.stride(4), N), ('Cs', Cs, Cs.stride(2), Cs.stride(4), N)], x=[('x', x_k, x_st[3], x_st[2], G_SIZE), ('dx', dXt, x_st[3], x_st[2], G_SIZE)], y=[('dY', dY, dY.stride(3), dY.stride(5), G_SIZE)], s=[('S_all', S_all, S_all.stride(3), S_all.stride(4), G_SIZE)])
    tma_why = {}
    on = {}
    for g, ts in groups.items():
        if g not in want:
            on[g] = False
            tma_why[g] = 'not requested'
            continue
        why = _tma_refusal(N=N, G_SIZE=G_SIZE, tensors=ts)
        on[g] = why is None
        tma_why[g] = why
        if why is not None and tma_mode == 'require':
            raise NotImplementedError('SSD_MIMO_SISO_BWD_TMA=require, group %r: %s' % (g, why))
    tma_m, tma_x, tma_y, tma_s = (on['m'], on['x'], on['y'], on['s'])
    if tma_m or tma_x or tma_y or tma_s:
        _install_tma_allocator()
    sig = (T, N, G_SIZE, has_trap, has_dskip, dskip_dd, x0 is not None, dS_final is not None, io_tag, precision, idx64, abl, Bs.dtype, Cs.dtype, x_raw.dtype, dY.dtype, S_all.dtype, dXt.dtype, dDelta.dtype, dsf.dtype, dsk_c.dtype, qk_f.dtype, dQK.dtype, dX0.dtype, dDsk_p.dtype, delta_cumsum.dtype, a_log.dtype, dev, reload_xdy, notrans, grid_hb, tma_m, tma_x, tma_y, tma_s, sched, span_mma, dd_split, dmap32, span_ax1, excl1d, dth_rot, pfx_out, nd['ddelta'], rot_ka, occ_bucket(Bsz * H, x_k.device), len_bucket(L))
    if not g90_served:
        if dmap32:
            inplace = False
        dBs = Bs if inplace else torch.empty_like(Bs, dtype=f32 if dmap32 else Bs.dtype)
        dCs = Cs if inplace else torch.empty_like(Cs, dtype=f32 if dmap32 else Cs.dtype)
    grid = (H, Bsz) if grid_hb else (Bsz, H)
    args = (Bs, Cs, dBs, dCs, x_k, dY, S_all, delta_cumsum, a_log, dsf, dsk_c, qk_f, dXt, dDelta, dQK, dX0, dDsk_p, e_map, dth_buf, tail_buf, *m_st, *x_st, dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3), dY.stride(5), *S_all.stride(), *delta_cumsum.stride(), *w_st, *f_st, *k_st, dth_st[0], dth_st[1], dth_st[2], dth_st[3], dth_st[4], nc, H)
    kw = dict(OCC=occ_bucket(Bsz * H, x_k.device), LB=len_bucket(L), T=T, N=N, G_SIZE=G_SIZE, PREC=precision, IO_DT=io_tag, HAS_TRAP=has_trap, HAS_DSKIP=has_dskip, DSKIP_DD=dskip_dd, HAS_X0=x0 is not None, HAS_DSF=dS_final is not None, IDX64=idx64, SCHED=sched, SPAN_MMA=span_mma, DD_SPLIT=dd_split, SPAN_AX1=span_ax1, EXCL1D=excl1d, DTH_ROT=dth_rot, ROT_KA=rot_ka, PFX_OUT=pfx_out, NO_DD=not nd['ddelta'], RELOAD=reload_xdy, NOTRANS=notrans, GRID_HB=grid_hb, TMA_M=tma_m, TMA_X=tma_x, TMA_Y=tma_y, TMA_S=tma_s, ABL=abl)
    if not g90_served and m3s_why is not None and inplace and _bwd_tuning_launch(_autotune_cache_key(_siso_bwd_scan_kernel, args, kw), fitted=_BWD_OOR_FIT.get(sig) is not None):
        dBs, dCs = (torch.empty_like(Bs), torch.empty_like(Cs))
        args = (*args[:2], dBs, dCs, *args[4:])
        inplace = False
    if not g90_served and dd_split:
        _siso_ddelta_map_kernel[nc, Bsz * H](Bs, Cs, x_k, dY, S_all, delta_cumsum, a_log, e_map, *m_st, *x_st, dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3), dY.stride(5), *S_all.stride(), *delta_cumsum.stride(), H, T=T, N=N, G_SIZE=G_SIZE, PREC=precision, IO_DT=io_tag, HAS_TRAP=has_trap, IDX64=idx64, SPAN_MMA=span_mma, num_warps=int(_env('SISO_DD_WARPS', '4')), num_stages=1)
    if not g90_served:
        if m3s_why is None:
            x_m, dX_m = (x_k, dXt)
            if not x_k.is_contiguous():
                x_m = x_k.contiguous()
                dX_m = torch.empty_like(x_m)
            _m3s.launch(Bs=Bs, Cs=Cs, dBs=dBs, dCs=dCs, x=x_m, dX=dX_m, dY=dY, S_all=S_all, delta_cumsum=delta_cumsum, a_log=a_log, dsf=dsf, dsk_c=dsk_c, qk_f=qk_f, dQK=dQK, vj=dDelta, tail=tail_buf, dth=dth_buf, dth_st=dth_st, f_st=f_st, w_st=w_st, has_trap=has_trap, has_dskip=has_dskip, has_dsf=dS_final is not None, dth_rot=dth_rot, no_dd=m3s_nodd)
            if dX_m is not dXt:
                dXt.copy_(dX_m)
        else:
            fit = _bwd_fit_single(grid, args, kw, dev, sig)
            if fit is None:
                fit = _BWD_OOR_FIT.get(sig)
            if fit is None:
                try:
                    _siso_bwd_scan_kernel[grid](*args, **kw)
                except triton.runtime.errors.OutOfResources:
                    fit = _bwd_refit_oor(grid, args, kw, dev, sig)
                    if fit is None:
                        raise
                    _siso_bwd_scan_kernel.fn[grid](*args, **kw, **fit)
            else:
                _siso_bwd_scan_kernel.fn[grid](*args, **kw, **fit)
        if pfx_out:
            vj = dDelta
            excl = torch.nn.functional.pad(vj[..., :-1], (1, 0)).cumsum(-1)
            a_h = -torch.exp(a_log.float()).reshape(1, H, 1, 1)
            dDelta = a_h * (e_map + excl + tail_buf[..., None])
    if dX is None:
        dX = dXt.transpose(2, 3).unsqueeze(3)
        if xtm_mode != 'view':
            dX = dX.contiguous()
    NS = min(N, int(_env('SISO_ADJ_NS', '64')))
    if rot_ka < N // 2:
        NS = min(NS, 2 * rot_ka)
    NB = N // NS
    dB = torch.empty(Bsz, nc, T, N, device=dev, dtype=f32)
    dC = torch.empty(Bsz, nc, T, N, device=dev, dtype=f32)
    th_c = theta
    per_head = th_c.dim() == 5
    need_dtheta = bool(nd['dtheta'])
    dTheta = torch.empty_like(th_c, dtype=f32) if need_dtheta else torch.empty(1, device=dev, dtype=f32)
    if need_dtheta and (not per_head):
        dTheta.zero_()
    st = th_c.stride()
    th_st = [st[0], st[1], st[2], st[3], st[4]] if per_head else [st[0], st[1], st[2], 0, st[3]]
    dth_st = th_st
    dth_from_scan = dth_rot and (not g90_served) or g90_dth
    need_dth_adj = need_dtheta and (not dth_from_scan)
    thh_c = theta_h if theta_h is not None else torch.empty(1, device=dev, dtype=f32)
    rb_c = rot_bias if rot_bias is not None else torch.empty(1, device=dev, dtype=f32)
    beta_c = beta.float().contiguous().reshape(H, N) if beta is not None else torch.empty(1, device=dev, dtype=f32)
    gam_c = gamma.float().contiguous().reshape(H, N) if gamma is not None else torch.empty(1, device=dev, dtype=f32)
    dThH = torch.zeros(H, N // 2, device=dev, dtype=f32) if theta_h is not None else torch.empty(1, device=dev, dtype=f32)
    dRB = torch.zeros(H, N // 2, device=dev, dtype=f32) if rot_bias is not None else torch.empty(1, device=dev, dtype=f32)
    dBeta = torch.zeros(H, N, device=dev, dtype=f32) if beta is not None else torch.empty(1, device=dev, dtype=f32)
    dGamma = torch.zeros(H, N, device=dev, dtype=f32) if gamma is not None else torch.empty(1, device=dev, dtype=f32)
    dW_p = torch.empty(Bsz, H, NB, L, device=dev, dtype=f32) if has_dw else torch.empty(1, device=dev, dtype=f32)
    dG_p = torch.empty(Bsz, H, NB, L, device=dev, dtype=f32) if has_trap else torch.empty(1, device=dev, dtype=f32)
    p_st = dW_p.stride() if has_dw else dG_p.stride() if has_trap else (0, 0, 0, 0)
    raw_common = _env('SISO_ADJ_RAW_COMMON', '0') == '1' and bool(bias_pre_rot) and (theta_h is None) and (rot_bias is None)
    adj_args = (Btil, Ctil, dBs, dCs, th_c, thh_c, rb_c, beta_c, gam_c, aux['scale'] if has_trap else dw if has_dw else dW_p, aux['gam'] if has_trap else dG_p, dQK, dB, dC, dTheta, dThH, dRB, dBeta, dGamma, dW_p, dG_p, *Ctil.stride()[:4], *Btil.stride()[:4], *dB.stride(), *m_st, *dth_st, *w_st, *p_st, H)
    adj_kw = dict(T=T, N=N, NS=NS, ROT_KA=rot_ka if rot_ka < N // 2 else 0, ROT_STATIC=rot_ka < N // 2 and _lv.on('adj_static_real', 'complex_R1O1'), HAS_THETA_H=theta_h is not None, HAS_ROT_BIAS=rot_bias is not None, HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None, BIAS_PRE_ROT=bool(bias_pre_rot), HAS_DW=has_dw, HAS_TRAP=has_trap, THETA_PER_HEAD=per_head, NEED_DTHETA=need_dth_adj, RAW_COMMON=raw_common, IDX64=idx64, TRAP_DTH_EXCL=_env('SISO_TRAP_DTH_EXCL', '0') == '1', NEED_DBETA=bool(nd['dbeta']), NEED_DGAMMA=bool(nd['dgamma']), NEED_DTHH=bool(nd['dthh']), NEED_DRB=bool(nd['drb']), OCC=occ_bucket(nc * Bsz, dev))

    def _adj_launch(dThH_, dRB_, dBeta_, dGamma_):
        launch_args = (*adj_args[:15], dThH_, dRB_, dBeta_, dGamma_, *adj_args[19:])
        _siso_rot_adjoint_kernel[nc, Bsz](*launch_args, **adj_kw)
    if g90_adj is not None:
        dB = dBs.float().sum(3)
        dC = dCs.float().sum(3)
        dTheta = g90_adj['dTheta']
        dW_p = g90_adj['dW_p']
        if has_trap:
            dG_p = g90_adj['dG_p']
        if beta is not None:
            dBeta = g90_adj['dBeta']
        if gamma is not None:
            dGamma = g90_adj['dGamma']
    elif _adj_tuning_launch(_autotune_cache_key(_siso_rot_adjoint_kernel, adj_args, adj_kw)):
        _adj_launch(torch.zeros_like(dThH), torch.zeros_like(dRB), torch.zeros_like(dBeta), torch.zeros_like(dGamma))
    if g90_adj is None:
        _adj_launch(dThH, dRB, dBeta, dGamma)
    if need_dtheta and dth_from_scan:
        dTheta = dth_buf.reshape(th_c.shape) if per_head else dth_buf.sum(3)
    out = dict(dX=dX, dDelta=dDelta if nd['ddelta'] else None, dB=dB, dC=dC, dTheta=dTheta if need_dtheta else None, dThetaH=dThH if theta_h is not None and nd['dthh'] else None, dRotBias=dRB if rot_bias is not None and nd['drb'] else None, dBeta=dBeta if beta is not None and nd['dbeta'] else None, dGamma=dGamma if gamma is not None and nd['dgamma'] else None, dX0=dX0 if x0 is not None else None, dDsk=dDsk_p.sum(0) if dskip_dd else None, dDW=None, dLam=None)
    if not (nd['ddw'] or nd['dlam']):
        pass
    elif has_dw and _env('SISO_SCALAR_FUSED', '0') == '1':
        out['dDW'], out['dLam'] = _siso_scalar_reduce(dW_p, dG_p if has_trap else None, dw, trap_lam)
    elif has_dw:
        dwv = dW_p.sum(2)
        if has_trap:
            A = dwv + dG_p.sum(2)
            P = torch.nn.functional.pad(dwv[..., :-1], (1, 0))
            out['dDW'] = trap_lam * A + (1.0 - trap_lam) * P
            out['dLam'] = dw * (A - P)
        else:
            out['dDW'] = dwv
    return out
