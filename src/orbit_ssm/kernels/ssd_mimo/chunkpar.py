
from __future__ import annotations
import os
import torch
import triton
from .fused_fwd_kernel import occ_bucket
import triton.language as tl
from .fused_fwd_kernel import _dcs_seg, _same_seg, _sm_count, _env
from .common import pdot
_LAST_FWD_ROUTE: dict = {}

def chunkpar_mode() -> str:
    return _env('CHUNKPAR', 'auto')
_cp_token_memo = [0.0, None]
_cpw_memo: dict = {}

def _calibration_token():
    import time as _t
    now = _t.monotonic()
    if now - _cp_token_memo[0] < 2.0 and _cp_token_memo[1] is not None:
        return _cp_token_memo[1]
    from .. import calibration as _cal
    from .calibration_chunkpar import FAMILY as _FAM
    from .smem import _CHUNK_CACHE
    data = _cal.load(_FAM)[0] if _cal.enabled(_FAM) else None
    tok = (id(data), len(_CHUNK_CACHE), os.environ.get('SSD_MIMO_CALIBRATION'))
    _cp_token_memo[0], _cp_token_memo[1] = (now, tok)
    return tok

def chunkpar_wanted(n_serial_ctas: int, nc: int, device, part='fwd', dtype=None) -> bool:
    mode = chunkpar_mode()
    if part == 'ka':
        mode = _env('CHUNKPAR_KA', mode)
    if mode == '0':
        return False
    if mode == '1':
        return True
    e = os.environ
    key = (n_serial_ctas, nc, device, part, dtype, e.get('SSD_MIMO_CHUNKPAR'), e.get('SSD_MIMO_CHUNKPAR_KA'), e.get('SSD_MIMO_CHUNKPAR_WAVES'), e.get('SSD_MIMO_CHUNKPAR_NC_PER_WAVE'), _calibration_token())
    hit = _cpw_memo.get(key)
    if hit is None:
        hit = _cpw_memo[key] = _chunkpar_wanted(n_serial_ctas, nc, device, part, dtype)
    return hit

def _chunkpar_wanted(n_serial_ctas: int, nc: int, device, part='fwd', dtype=None) -> bool:
    mode = chunkpar_mode()
    if part == 'ka':
        mode = _env('CHUNKPAR_KA', mode)
    if mode == '0':
        return False
    if mode == '1':
        return True
    waves_env = _env('CHUNKPAR_WAVES', '')
    if waves_env:
        waves = float(waves_env)
        ncpw = float(_env('CHUNKPAR_NC_PER_WAVE', '0'))
    else:
        v = None
        if dtype == torch.float32:
            v = cp_calibration_lookup(device, part + '_fp32')
            if v is None and part == 'fwd_nograd':
                v = cp_calibration_lookup(device, 'fwd_fp32')
        if v is None:
            v = cp_calibration_lookup(device, part)
        if v is None and part == 'fwd_nograd':
            v = cp_calibration_lookup(device, 'fwd')
        if v is None:
            return False
        waves, ncpw = v
    sm = _sm_count(device)
    w = n_serial_ctas / sm
    if not cp_rule(w, nc, waves, ncpw):
        return False
    if part == 'fwd_nograd' and (not waves_env):
        wm, ncm = _fwd_guard(device, dtype)
        if wm is not None and n_serial_ctas * nc < wm and (nc < ncm):
            return False
    return True
_guard_memo: dict = {}
_guard_ttl = 2.0

def _fwd_guard(device, dtype=None):
    import time as _t
    now = _t.monotonic()
    k = (str(device), dtype, _env('CALIBRATION', ''))
    hit = _guard_memo.get(k)
    if hit is not None and now - hit[0] < _guard_ttl:
        return hit[1]
    from .smem import chunk_cache_lookup
    key = cp_calibration_key(device, 'fwd_nograd_fp32' if dtype == torch.float32 else 'fwd_nograd')
    wm = chunk_cache_lookup(key + '|wmin')
    ncm = chunk_cache_lookup(key + '|ncmin')
    val = (None, 0) if wm is None else (int(wm), 1 << 30 if ncm is None else int(ncm))
    _guard_memo[k] = (now, val)
    return val
_PLAN_FITS: dict = {}
_SCAN_CHUNK: object = None
_RCD: dict = {}

def plan_flags(*, x0=None, beta=None, gamma=None, dw=None, trap_lam=None, d_skip=None, gate_z=None, in_kernel=False):
    return (x0 is not None, beta is not None, gamma is not None, dw is not None, trap_lam is not None, d_skip is not None, gate_z is not None and bool(in_kernel))

def general_plan_fits(T, N, G_SIZE, x_dtype, precision, device, flags=(False,) * 7) -> bool:
    import torch
    from .fused_fwd_kernel import fused_chunk_fwd
    from .smem import SsdSmemOverflow
    from .smem import device_smem_limit, smem_clamp_enabled
    key = (device, x_dtype, precision, T, N, G_SIZE, flags, device_smem_limit(device), smem_clamp_enabled())
    hit = _PLAN_FITS.get(key)
    if hit is not None:
        return hit
    flags = tuple((bool(f) for f in flags))
    has_x0, has_beta, has_gamma, has_dw, has_trap, has_dskip, has_gate = flags
    dev = torch.device(device)
    f32 = torch.float32
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            C = torch.zeros(1, 1, T, N, device=dev, dtype=x_dtype)
            kw = {}
            if has_x0:
                kw['x0'] = torch.zeros(1, 1, G_SIZE, N, device=dev, dtype=f32)
            if has_beta:
                kw['beta'] = torch.zeros(1, N, 1, device=dev, dtype=f32)
            if has_gamma:
                kw['gamma'] = torch.zeros(1, N, 1, device=dev, dtype=f32)
            if has_dw or has_trap:
                kw['dw'] = torch.zeros(1, 1, T, device=dev, dtype=f32)
            if has_trap:
                kw['trap_lam'] = torch.zeros(1, 1, T, device=dev, dtype=f32)
                kw['dot_out'] = torch.empty(1, 1, T, device=dev, dtype=f32)
            if has_dskip:
                kw['d_skip'] = torch.zeros(1, G_SIZE, 1, device=dev, dtype=f32)
            if has_gate:
                kw['gate_z'] = torch.zeros(1, T, G_SIZE, device=dev, dtype=x_dtype)
                kw['ypre_out'] = torch.empty(1, 1, 1, G_SIZE, 1, T, device=dev, dtype=x_dtype).permute(0, 1, 4, 5, 2, 3)
                kw['gate_out_dtype'] = x_dtype
            fused_chunk_fwd(C, C.clone(), torch.zeros(1, 1, G_SIZE, 1, T, device=dev, dtype=x_dtype), torch.zeros(1, 1, 1, T, device=dev, dtype=f32), torch.zeros(1, device=dev, dtype=f32), precision=precision, store_sall=True, **kw)
        ok = True
    except SsdSmemOverflow:
        ok = False
    except Exception:
        ok = False
    _PLAN_FITS[key] = ok
    return ok

def real_cp_decision(*, n_serial_ctas, nc, T, N, G_SIZE, x_dtype, precision, device, needs_bwd, flags=(False,) * 7) -> bool:
    _m = _env('CHUNKPAR', 'auto')
    if _m == '0':
        return False
    if _m == '1':
        return True
    e = os.environ
    key = (n_serial_ctas, nc, T, N, G_SIZE, x_dtype, precision, device, needs_bwd, flags, e.get('SSD_MIMO_CHUNKPAR'), e.get('SSD_MIMO_CHUNKPAR_WAVES'), e.get('SSD_MIMO_CHUNKPAR_NC_PER_WAVE'), e.get('SSD_MIMO_CHUNK'), e.get('SSD_MIMO_SMEM_LIMIT'), e.get('SSD_MIMO_SMEM_CLAMP'), e.get('SSD_MIMO_CALIBRATION'), _calibration_token())
    hit = _RCD.get(key)
    if hit is not None:
        return hit
    hit = _real_cp_decision(n_serial_ctas, nc, T, N, G_SIZE, x_dtype, precision, device, needs_bwd, flags)
    _RCD[key] = hit
    return hit

def _real_cp_decision(n_serial_ctas, nc, T, N, G_SIZE, x_dtype, precision, device, needs_bwd, flags) -> bool:
    from .fused_fwd_kernel import _env
    if not chunkpar_wanted(n_serial_ctas, nc, device, part='fwd' if needs_bwd else 'fwd_nograd', dtype=x_dtype):
        return False
    if _env('CHUNKPAR', 'auto') == '1':
        return True
    global _SCAN_CHUNK
    if _SCAN_CHUNK is None:
        from ...ssm.ssd_mimo import _scan_chunk as _SCAN_CHUNK
    if T != _SCAN_CHUNK(1, G_SIZE, 1, needs_bwd, real=True, L=nc * T, work=n_serial_ctas * nc * T, dtype=x_dtype):
        return False
    return general_plan_fits(T, N, G_SIZE, x_dtype, precision, device, flags)

def cp_rule(w, nc, waves, ncpw):
    return w < waves and nc / max(w, 1.0) >= ncpw
_DEVNAME: dict = {}

def _device_name(device):
    k = str(device)
    n = _DEVNAME.get(k)
    if n is None:
        try:
            n = torch.cuda.get_device_name(device)
        except Exception:
            n = 'unknown'
        _DEVNAME[k] = n
    return n

def cp_calibration_key(device, part):
    name = _device_name(device)
    return f'chunk:cp:{name}|{part}'

def cp_calibration_lookup(device, part):
    from .smem import chunk_cache_lookup
    k = cp_calibration_key(device, part)
    w = chunk_cache_lookup(k)
    m = chunk_cache_lookup(k + '|ncpw')
    if w is None:
        return None
    return (float(w) / 100.0, float(m or 0) / 100.0)

def cp_calibration_store(device, part, waves, ncpw, timings=None):
    from .smem import chunk_cache_store
    k = cp_calibration_key(device, part)
    chunk_cache_store(k, int(round(waves * 100)), timings)
    chunk_cache_store(k + '|ncpw', int(round(ncpw * 100)))

def _cp_configs():
    if _env('CP_AUTOTUNE', '1') == '0':
        return [triton.Config({}, num_warps=4, num_stages=2)]
    return [triton.Config({}, num_warps=w, num_stages=st) for w in (4, 8) for st in (1, 2, 3)]
_CP_KEY = ['T', 'N', 'G_SIZE', 'K_SUB', 'IO_DT', 'PREC', 'OCC']

@triton.jit
def _to_io(t, IO_DT: tl.constexpr):
    if IO_DT == 'bf16':
        return t.to(tl.bfloat16)
    elif IO_DT == 'fp16':
        return t.to(tl.float16)
    else:
        return t.to(tl.float32)

@triton.jit
def _cp_pass_kernel(Psi_ptr, Alpha_ptr, Out_ptr, Init_ptr, Fin_ptr, s_pb, s_ph, s_pc, s_ib, s_ih, s_ig, s_in, s_fb, s_fh, s_fg, s_fn, H, NCB, N: tl.constexpr, G_SIZE: tl.constexpr, BLK: tl.constexpr, REVERSE: tl.constexpr, HAS_INIT: tl.constexpr, HAS_FIN: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    off = tl.program_id(2) * BLK + tl.arange(0, BLK)
    n_i = off // G_SIZE
    g_i = off % G_SIZE
    if HAS_INIT:
        acc = tl.load(Init_ptr + b * s_ib + h * s_ih + g_i * s_ig + n_i * s_in).to(tl.float32)
    else:
        acc = tl.zeros([BLK], dtype=tl.float32)
    pbase = Psi_ptr + b * s_pb + h * s_ph + off
    obase = Out_ptr + b * s_pb + h * s_ph + off
    abase = Alpha_ptr + (b * H + h) * NCB
    for i in range(NCB):
        if REVERSE:
            cb = NCB - 1 - i
        else:
            cb = i
        tl.store(obase + cb * s_pc, acc)
        a = tl.load(abase + cb)
        v = tl.load(pbase + cb * s_pc).to(tl.float32)
        acc = a * acc + v
    if HAS_FIN:
        tl.store(Fin_ptr + b * s_fb + h * s_fh + g_i * s_fg + n_i * s_fn, acc)

def _launch_pass(psi, alpha, out, init, init_st, fin, fin_st, reverse, N, G_SIZE):
    Bsz, H, ncb = psi.shape[:3]
    blk = min(int(_env('CP_PASS_BLK', '512')), N * G_SIZE)
    blk = 1 << blk.bit_length() - 1
    while N * G_SIZE % blk:
        blk //= 2
    dummy = psi
    _cp_pass_kernel[Bsz, H, N * G_SIZE // blk](psi, alpha, out, init if init is not None else dummy, fin if fin is not None else dummy, psi.stride(0), psi.stride(1), psi.stride(2), *init_st, *fin_st, H, ncb, N=N, G_SIZE=G_SIZE, BLK=blk, REVERSE=reverse, HAS_INIT=init is not None, HAS_FIN=fin is not None, num_warps=4)

def _pick_ksub(nc, part='fwd'):
    k = _env('CP_KSUB_' + part.upper(), _env('CP_KSUB', ''))
    measured = False
    if not k:
        from .smem import chunk_cache_lookup
        try:
            dev = torch.cuda.current_device()
        except Exception:
            dev = None
        v = chunk_cache_lookup(cp_calibration_key(dev, part) + '|ksub')
        k = v if v else 1 if part == 'fwd' else 16
        measured = bool(v)
    k = int(k)
    if not _env('CP_KSUB_' + part.upper(), '') and (not measured):
        k = max(1, min(k, nc // 8))
    while k > 1 and nc % k:
        k //= 2
    return max(1, k)

@triton.jit
def _cp_write_scale(DW_ptr, Lam_ptr, b, h, c, offs_t, s_wb, s_wh, s_wl, NC, T: tl.constexpr, HAS_DW: tl.constexpr, HAS_TRAP: tl.constexpr):
    tbase = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
    if HAS_TRAP:
        dt_t = tl.load(DW_ptr + tbase)
        lam_t = tl.load(Lam_ptr + tbase)
        idx_n = c * T + offs_t + 1
        m_n = idx_n < NC * T
        dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
        lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
        own = lam_t * dt_t
        w_t = own + dt_n * (1.0 - lam_n)
    elif HAS_DW:
        w_t = tl.load(DW_ptr + tbase)
        own = w_t
    else:
        w_t = tl.zeros([T], dtype=tl.float32) + 1.0
        own = w_t
    return (w_t, own)

@triton.jit
def _cp_chunk_state(B_ptr, X_ptr, Delta_ptr, Beta_ptr, DW_ptr, Lam_ptr, b, h, c, a_val, offs_t, offs_n, offs_g, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_wb, s_wh, s_wl, NC, T: tl.constexpr, N: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr, HAS_BETA: tl.constexpr, HAS_DW: tl.constexpr, HAS_TRAP: tl.constexpr):
    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
    D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
    seg_end = tl.max(seg_h, axis=0)
    D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
    alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
    dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end - D_h)), 0.0)
    if HAS_DW:
        w_t, _own = _cp_write_scale(DW_ptr, Lam_ptr, b, h, c, offs_t, s_wb, s_wh, s_wl, NC, T, HAS_DW, HAS_TRAP)
        dec_agg = dec_agg * w_t
    Bt = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
    if HAS_BETA:
        Bt = Bt + tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)[None, :]
    x_r = tl.load(X_ptr + b * s_xb + h * s_xh + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
    G = pdot(tl.trans(_to_io(Bt, IO_DT)), _to_io(x_r * dec_agg[:, None], IO_DT), input_precision=PREC)
    return (G, alpha)

@triton.autotune(configs=_cp_configs(), key=_CP_KEY + ['GT', 'HAS_BETA', 'HAS_DW', 'HAS_TRAP'])
@triton.jit
def _cp_local_kernel(B_ptr, X_ptr, Delta_ptr, A_ptr, Beta_ptr, DW_ptr, Lam_ptr, G_ptr, Alpha_ptr, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_wb, s_wh, s_wl, s_gb, s_gh, s_gc, s_gn, s_gg, H, NC, NCB, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, K_SUB: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr, HAS_BETA: tl.constexpr, HAS_DW: tl.constexpr, HAS_TRAP: tl.constexpr, IDX64: tl.constexpr, GT: tl.constexpr, OCC=0):
    bh = tl.program_id(0)
    cb = tl.program_id(1)
    b = bh // H
    h = bh % H
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
        cb = cb.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.program_id(2) * GT + tl.arange(0, GT)
    S = tl.zeros([N, GT], dtype=tl.float32)
    a_tot = 1.0
    for s in range(K_SUB):
        c = cb * K_SUB + s
        Gc, alpha = _cp_chunk_state(B_ptr, X_ptr, Delta_ptr, Beta_ptr, DW_ptr, Lam_ptr, b, h, c, a_val, offs_t, offs_n, offs_g, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_wb, s_wh, s_wl, NC, T, N, PREC, IO_DT, HAS_BETA, HAS_DW, HAS_TRAP)
        S = alpha * S + Gc
        a_tot = a_tot * alpha
    tl.store(G_ptr + b * s_gb + h * s_gh + cb * s_gc + offs_n[:, None] * s_gn + offs_g[None, :] * s_gg, S.to(G_ptr.dtype.element_ty))
    if tl.program_id(2) == 0:
        tl.store(Alpha_ptr + (b * H + h) * NCB + cb, a_tot)

@triton.autotune(configs=_cp_configs(), key=['T', 'N', 'PREC', 'IO_DT'])
@triton.jit
def _cp_shared_cb_kernel(C_ptr, B_ptr, CB_ptr, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, NC: tl.constexpr, T: tl.constexpr, N: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.arange(0, T)
    n = tl.arange(0, N)
    C = tl.load(C_ptr + b * s_cb + c * s_cc + t[:, None] * s_ct + n[None, :] * s_cn)
    B = tl.load(B_ptr + b * s_bb + c * s_bc + t[:, None] * s_bt + n[None, :] * s_bn)
    cb = pdot(_to_io(C, IO_DT), tl.trans(_to_io(B, IO_DT)), input_precision=PREC)
    tl.store(CB_ptr + ((b * NC + c) * T + t[:, None]) * T + t[None, :], cb)

@triton.autotune(configs=_cp_configs(), key=_CP_KEY + ['GT', 'SHARED_CB', 'HAS_BETA', 'HAS_GAMMA', 'HAS_DW', 'HAS_TRAP', 'HAS_DSKIP', 'STORE_SALL'])
@triton.jit
def _cp_out_kernel(C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, Beta_ptr, Gamma_ptr, DW_ptr, Lam_ptr, Dsk_ptr, SIn_ptr, SAll_ptr, Y_ptr, Dot_ptr, CB_ptr, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xl, s_db, s_dh, s_dc, s_dt, s_wb, s_wh, s_wl, s_ib, s_ih, s_ic, s_in, s_ig, s_sb, s_sh, s_sc, s_sn, s_sg, s_yb, s_yh, s_yc, s_yt, s_yg, H, NC, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, K_SUB: tl.constexpr, PREC: tl.constexpr, IO_DT: tl.constexpr, HAS_BETA: tl.constexpr, HAS_GAMMA: tl.constexpr, HAS_DW: tl.constexpr, HAS_TRAP: tl.constexpr, HAS_DSKIP: tl.constexpr, STORE_DOT: tl.constexpr, STORE_SALL: tl.constexpr, IDX64: tl.constexpr, GT: tl.constexpr, SHARED_CB: tl.constexpr, OCC=0):
    bh = tl.program_id(0)
    cb = tl.program_id(1)
    b = bh // H
    h = bh % H
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
        cb = cb.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_g = tl.program_id(2) * GT + tl.arange(0, GT)
    causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float('-inf')
    if HAS_BETA:
        beta_v = tl.load(Beta_ptr + h * N + offs_n).to(tl.float32)
    if HAS_GAMMA:
        gam_v = tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)
    if HAS_DSKIP:
        dv = tl.load(Dsk_ptr + h * G_SIZE + offs_g)
    S = tl.load(SIn_ptr + b * s_ib + h * s_ih + cb * s_ic + offs_n[:, None] * s_in + offs_g[None, :] * s_ig)
    for s in range(K_SUB):
        c = cb * K_SUB + s
        if STORE_SALL:
            tl.store(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_g[None, :] * s_sg, S.to(SAll_ptr.dtype.element_ty))
        dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
        D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
        seg_end = tl.max(seg_h, axis=0)
        D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
        f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
        alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
        dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end - D_h)), 0.0)
        dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
        if HAS_DW:
            w_t, own = _cp_write_scale(DW_ptr, Lam_ptr, b, h, c, offs_t, s_wb, s_wh, s_wl, NC, T, HAS_DW, HAS_TRAP)
            dec_agg = dec_agg * w_t
            dec_mask = dec_mask * w_t[None, :]
            if HAS_TRAP:
                dec_mask = tl.where(offs_t[:, None] == offs_t[None, :], own[None, :], dec_mask)
        Bt = tl.load(B_ptr + b * s_bb + c * s_bc + offs_t[:, None] * s_bt + offs_n[None, :] * s_bn).to(tl.float32)
        Ct = tl.load(C_ptr + b * s_cb + c * s_cc + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
        if HAS_BETA:
            Bt = Bt + beta_v[None, :]
        if HAS_GAMMA:
            Ct = Ct + gam_v[None, :]
        Bop = _to_io(Bt, IO_DT)
        Cop = _to_io(Ct, IO_DT)
        x_r = tl.load(X_ptr + b * s_xb + h * s_xh + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
        y = f[:, None] * pdot(Cop, _to_io(S, IO_DT), input_precision=PREC)
        if SHARED_CB:
            cb_raw = tl.load(CB_ptr + ((b * NC + c) * T + offs_t[:, None]) * T + offs_t[None, :])
        else:
            cb_raw = pdot(Cop, tl.trans(Bop), input_precision=PREC)
        Score = cb_raw * dec_mask
        if STORE_DOT and tl.program_id(2) == 0:
            tbase = b * s_wb + h * s_wh + (c * T + offs_t) * s_wl
            tl.store(Dot_ptr + tbase, tl.sum(Bt * Ct, axis=1))
        y += pdot(_to_io(Score, IO_DT), _to_io(x_r, IO_DT), input_precision=PREC)
        if HAS_DSKIP:
            y += x_r * dv[None, :]
        tl.store(Y_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_g[None, :] * s_yg, y.to(Y_ptr.dtype.element_ty))
        if K_SUB > 1:
            Gc = pdot(tl.trans(Bop), _to_io(x_r * dec_agg[:, None], IO_DT), input_precision=PREC)
            S = alpha * S + Gc

def _cfg(env_key, default_w, default_s):
    w = int(_env(env_key + '_WARPS', str(default_w)))
    s = int(_env(env_key + '_STAGES', str(default_s)))
    return (w, s)
_RES_REFUSED: dict = {}

def cp_resource_refused(part, key) -> bool:
    return (part, key) in _RES_REFUSED

def cp_resource_refuse(part, key, err) -> None:
    _RES_REFUSED[part, key] = str(err)[:200]

def _CP_FWD_GT_ENV():
    e = os.environ
    return e.get('SSD_MIMO_CP_FWD_GT') or e.get('UNIFIED_SSD_CP_FWD_GT') or ''

def cp_refusal(*, theta, theta_h, rot_bias, x0, fuse, dcs_ink, T, N, G_SIZE):
    if theta is not None or theta_h is not None or rot_bias is not None:
        return 'rotation (the chunk-parallel schedule is rotation-free)'
    if fuse is not None:
        return 'SISOFUSE mode'
    if dcs_ink:
        return 'DCS_INK (in-kernel prefix) not wired'
    if G_SIZE % 16 or N % 16 or T % 16:
        return 'tile envelope (T, N, G multiples of 16)'
    gt_env = _CP_FWD_GT_ENV()
    gt = G_SIZE if not gt_env else int(gt_env)
    if gt < 16 or gt & gt - 1 or G_SIZE % gt:
        return 'G tile %d (CP_FWD_GT, default the whole head width G = %d) is not a power of two >= 16 that divides G' % (gt, G_SIZE)
    return None

def cp_replay_ok(nc) -> bool:
    return _pick_ksub(nc, 'fwd') == 1

def cp_chunk_fwd(Ctil, Btil, x_raw, delta_cumsum, a_log, *, beta=None, gamma=None, dw=None, trap_lam=None, x0=None, d_skip=None, precision='tf32', store_sall=True, needs_bwd=True, y_token_major=False, states_only=False):
    from .fused_fwd_kernel import dot_precision, _needs_i64
    from .siso_fwd_kernel import siso_native_y, _io_tag
    from .siso_route import siso_bc
    precision_in = precision
    precision = dot_precision(precision)
    Ctil = siso_bc(Ctil)
    Btil = siso_bc(Btil)
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_raw.shape[2]
    N = Btil.shape[3]
    gt = int(_env('CP_FWD_GT', str(G_SIZE)))
    if gt < 16 or gt & gt - 1 or G_SIZE % gt:
        raise ValueError('CP_FWD_GT must be a power of two >=16 and divide G_SIZE')
    ngt = G_SIZE // gt
    device = x_raw.device
    f32 = torch.float32
    L_pad = nc * T
    has_trap = trap_lam is not None
    has_dw = dw is not None
    io_tag = _io_tag(x_raw.dtype)
    ksub = _pick_ksub(nc, 'fwd')
    ncb = nc // ksub
    dummy = torch.empty(1, device=device, dtype=f32)
    beta_c = beta.float().contiguous().reshape(H, N) if beta is not None else dummy
    gam_c = gamma.float().contiguous().reshape(H, N) if gamma is not None else dummy
    dsk_c = d_skip.float().contiguous().reshape(H, G_SIZE) if d_skip is not None else dummy
    x0_c = x0.float().contiguous() if x0 is not None else None
    x0_st = list(x0_c.stride()) if x0 is not None else [0, 0, 0, 0]
    dw_c = dw if has_dw else dummy
    lam_c = trap_lam if has_trap else dummy
    w_st = dw_c.stride() if has_dw else (0, 0, 0)
    store_dot = bool(has_trap and needs_bwd)
    dot = torch.empty(Bsz, H, L_pad, device=device, dtype=f32) if store_dot else dummy
    Y = dummy if states_only else siso_native_y(Bsz, H, G_SIZE, nc, T, device, x_raw.dtype, y_token_major)
    sall_dt = f32 if os.environ.get('UNIFIED_SSD_SALL_FP32') == '1' or precision_in == 'ieee' else x_raw.dtype
    if store_sall:
        S_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device, dtype=sall_dt)
    else:
        S_all = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
    S_final = torch.empty(Bsz, H, G_SIZE, N, device=device, dtype=f32)
    from . import levers as _lv
    _glo_ok = x_raw.dtype in (torch.bfloat16, torch.float16) and precision_in != 'ieee'
    g_dt = x_raw.dtype if _glo_ok and _lv.on('real_cp_g_lo', _lv.real_group(T), guard=_glo_ok) else f32
    Gbuf = torch.empty(Bsz, H, ncb, N, G_SIZE, device=device, dtype=g_dt)
    if ksub == 1:
        if not store_sall:
            S_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device, dtype=sall_dt)
        Sin = S_all
    else:
        Sin = torch.empty(Bsz, H, ncb, N, G_SIZE, device=device, dtype=f32)
    alpha = torch.empty(Bsz, H, ncb, device=device, dtype=f32)
    x_st = (x_raw.stride(0), x_raw.stride(1), x_raw.stride(2), x_raw.stride(4))
    idx64 = _needs_i64(x_raw, Y, S_all, Gbuf, Btil, Ctil, dw_c, lam_c)
    w, s = _cfg('CP_LOCAL', 4, 2)
    _cp_local_kernel[Bsz * H, ncb, ngt](Btil, x_raw, delta_cumsum, a_log, beta_c, dw_c, lam_c, Gbuf, alpha, *Btil.stride()[:4], *x_st, *delta_cumsum.stride(), *w_st, *Gbuf.stride(), H, nc, ncb, T=T, N=N, G_SIZE=G_SIZE, GT=gt, K_SUB=ksub, PREC=precision, IO_DT=io_tag, HAS_BETA=beta is not None, HAS_DW=has_dw, HAS_TRAP=has_trap, OCC=occ_bucket(Bsz * H * ncb * ngt, device), IDX64=idx64)
    _launch_pass(Gbuf, alpha, Sin, x0_c, x0_st, S_final, list(S_final.stride()), False, N, G_SIZE)
    del Gbuf
    if states_only:
        assert ksub == 1, 'states_only needs K_SUB == 1'
        return (None, S_all, S_final, None)
    store_in_out = bool(store_sall and ksub > 1)
    sall_st = S_all.stride() if store_in_out else (0, 0, 0, 0, 0)
    shared_cb = _env('CP_SHARED_CB', '0') == '1' and beta is None and (gamma is None)
    cb_buf = dummy
    if shared_cb:
        cb_buf = torch.empty((Bsz, nc, T, T), device=device, dtype=f32)
        _cp_shared_cb_kernel[Bsz, nc](Ctil, Btil, cb_buf, *Ctil.stride()[:4], *Btil.stride()[:4], NC=nc, T=T, N=N, PREC=precision, IO_DT=io_tag)
    w, s = _cfg('CP_OUT', 4, 2)
    _cp_out_kernel[Bsz * H, ncb, ngt](Ctil, Btil, x_raw, delta_cumsum, a_log, beta_c, gam_c, dw_c, lam_c, dsk_c, Sin, S_all, Y, dot, cb_buf, *Ctil.stride()[:4], *Btil.stride()[:4], *x_st, *delta_cumsum.stride(), *w_st, *Sin.stride(), *sall_st, *Y.stride()[:4], Y.stride(5), H, nc, T=T, N=N, G_SIZE=G_SIZE, GT=gt, K_SUB=ksub, PREC=precision, IO_DT=io_tag, HAS_BETA=beta is not None, HAS_GAMMA=gamma is not None, HAS_DW=has_dw, HAS_TRAP=has_trap, HAS_DSKIP=d_skip is not None, STORE_DOT=store_dot, STORE_SALL=store_in_out, IDX64=idx64, SHARED_CB=shared_cb, OCC=occ_bucket(Bsz * H * ncb * ngt, device))
    if not store_sall and ksub == 1:
        S_all = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
    aux = dict(Bs=None, Cs=None, dot=dot if store_dot else None, dot_raw=None, scale=None, gam=None, xt=None, m_st=None, stored=False, ink_rot=False, fuse=None, io_tag=io_tag, chunkpar=True, ksub=ksub)
    return (Y, S_all, S_final, aux)

@triton.jit
def _cp_readout(C_ptr, Delta_ptr, Gamma_ptr, dY_ptr, b, h, c, cg, a_val, offs_t, offs_n, offs_gt, s_cb, s_cc, s_ct, s_cn, s_ch, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, T: tl.constexpr, N: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr, HAS_GAMMA: tl.constexpr):
    dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
    D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, False)
    seg_end = tl.max(seg_h, axis=0)
    D_end = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
    f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
    alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end), 0.0)
    Co = tl.load(C_ptr + b * s_cb + c * s_cc + cg * s_ch + offs_t[:, None] * s_ct + offs_n[None, :] * s_cn).to(tl.float32)
    if HAS_GAMMA:
        Co = Co + tl.load(Gamma_ptr + h * N + offs_n).to(tl.float32)[None, :]
    dY = tl.load(dY_ptr + b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg).to(tl.float32)
    r = pdot(tl.trans(Co).to(MMA_DT), (f[:, None] * dY).to(MMA_DT), input_precision=PREC)
    return (r, alpha)

@triton.autotune(configs=_cp_configs(), key=_CP_KEY + ['GT', 'HAS_GAMMA'])
@triton.jit
def _cp_ka_local_kernel(C_ptr, Delta_ptr, A_ptr, Gamma_ptr, dY_ptr, R_ptr, Alpha_ptr, s_cb, s_cc, s_ct, s_cn, s_ch, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_rb, s_rh, s_rc, s_rn, s_rg, H, NCB, HPG_C, T: tl.constexpr, N: tl.constexpr, GT: tl.constexpr, NGT: tl.constexpr, K_SUB: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr, HAS_GAMMA: tl.constexpr, IDX64: tl.constexpr, OCC=0):
    pid = tl.program_id(0)
    cb = tl.program_id(1)
    bh = pid // NGT
    gt_idx = pid % NGT
    b = bh // H
    h = bh % H
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
        cb = cb.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    cg = h // HPG_C
    v = tl.zeros([N, GT], dtype=tl.float32)
    a_tot = 1.0
    for si in range(K_SUB):
        c = cb * K_SUB + (K_SUB - 1 - si)
        r, alpha = _cp_readout(C_ptr, Delta_ptr, Gamma_ptr, dY_ptr, b, h, c, cg, a_val, offs_t, offs_n, offs_gt, s_cb, s_cc, s_ct, s_cn, s_ch, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, T, N, PREC, MMA_DT, HAS_GAMMA)
        v = alpha * v + r
        a_tot = a_tot * alpha
    tl.store(R_ptr + b * s_rb + h * s_rh + cb * s_rc + offs_n[:, None] * s_rn + offs_gt[None, :] * s_rg, v)
    if gt_idx == 0:
        tl.store(Alpha_ptr + (b * H + h) * NCB + cb, a_tot)

@triton.autotune(configs=_cp_configs(), key=_CP_KEY + ['GT', 'HAS_GAMMA', 'NO_DALPHA'])
@triton.jit
def _cp_ka_out_kernel(C_ptr, Delta_ptr, A_ptr, Gamma_ptr, dY_ptr, VIn_ptr, SAll_ptr, dG_ptr, dAlpha_ptr, s_cb, s_cc, s_ct, s_cn, s_ch, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_vb, s_vh, s_vc, s_vn, s_vg, s_sb, s_sh, s_sc, s_sn, s_sg, s_gb, s_gh, s_gc, s_gn, s_gg, s_dab, s_dah, s_dag, s_dac, H, HPG_C, T: tl.constexpr, N: tl.constexpr, GT: tl.constexpr, NGT: tl.constexpr, K_SUB: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr, HAS_GAMMA: tl.constexpr, IDX64: tl.constexpr, OCC=0, NO_DALPHA: tl.constexpr=False):
    pid = tl.program_id(0)
    cb = tl.program_id(1)
    bh = pid // NGT
    gt_idx = pid % NGT
    b = bh // H
    h = bh % H
    if IDX64:
        b = b.to(tl.int64)
        h = h.to(tl.int64)
        cb = cb.to(tl.int64)
    a_val = -tl.exp(tl.load(A_ptr + h))
    offs_t = tl.arange(0, T)
    offs_n = tl.arange(0, N)
    offs_gt = gt_idx * GT + tl.arange(0, GT)
    cg = h // HPG_C
    v = tl.load(VIn_ptr + b * s_vb + h * s_vh + cb * s_vc + offs_n[:, None] * s_vn + offs_gt[None, :] * s_vg)
    for si in range(K_SUB):
        c = cb * K_SUB + (K_SUB - 1 - si)
        tl.store(dG_ptr + b * s_gb + h * s_gh + c * s_gc + offs_n[:, None] * s_gn + offs_gt[None, :] * s_gg, v.to(dG_ptr.dtype.element_ty))
        if not NO_DALPHA:
            S_c = tl.load(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg).to(tl.float32)
            tl.store(dAlpha_ptr + b * s_dab + h * s_dah + gt_idx * s_dag + c * s_dac, tl.sum(v * S_c))
        r, alpha = _cp_readout(C_ptr, Delta_ptr, Gamma_ptr, dY_ptr, b, h, c, cg, a_val, offs_t, offs_n, offs_gt, s_cb, s_cc, s_ct, s_cn, s_ch, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, T, N, PREC, MMA_DT, HAS_GAMMA)
        v = alpha * v + r

def cp_ka_refusal(*, rot_in_kernel, pair_half, has_theta_h, O, dcs_ink, seg_ka):
    if rot_in_kernel or has_theta_h:
        return 'rotation'
    if pair_half:
        return 'pair_half'
    if O != 1:
        return 'O > 1'
    if dcs_ink:
        return 'DCS_INK'
    if seg_ka:
        return 'segmented Ka (the existing long-L plan) chosen'
    return None

def cp_ka(*, Ctil, s_c, s_ch, delta_cumsum, a_log, gamma_c, has_gamma, dY, s_y, S_all, dsf_c, dsf_st, has_dsf, dG_all, dalpha_all, dX0_out, x0_st, has_x0, H, nc, T, N, G_SIZE, GT, NGT, HPG_C, precision, mma_dt, idx64, no_dalpha=False):
    Bsz = dY.shape[0]
    device = dY.device
    f32 = torch.float32
    ksub = _pick_ksub(nc, 'ka')
    ncb = nc // ksub
    R = torch.empty(Bsz, H, ncb, N, G_SIZE, device=device, dtype=f32)
    Vin = torch.empty_like(R)
    alpha = torch.empty(Bsz, H, ncb, device=device, dtype=f32)
    w, s = _cfg('CP_KA_LOCAL', 4, 2)
    _cp_ka_local_kernel[Bsz * H * NGT, ncb](Ctil, delta_cumsum, a_log, gamma_c, dY, R, alpha, *s_c, s_ch, *delta_cumsum.stride(), *s_y, *R.stride(), H, ncb, HPG_C, T=T, N=N, GT=GT, NGT=NGT, K_SUB=ksub, PREC=precision, MMA_DT=mma_dt, OCC=occ_bucket(Bsz * H * NGT * ncb, device), HAS_GAMMA=has_gamma, IDX64=idx64)
    _launch_pass(R, alpha, Vin, dsf_c if has_dsf else None, dsf_st if has_dsf else [0, 0, 0, 0], dX0_out if has_x0 else None, x0_st if has_x0 else [0, 0, 0, 0], True, N, G_SIZE)
    del R
    w, s = _cfg('CP_KA_OUT', 4, 2)
    _cp_ka_out_kernel[Bsz * H * NGT, ncb](Ctil, delta_cumsum, a_log, gamma_c, dY, Vin, S_all, dG_all, dalpha_all, *s_c, s_ch, *delta_cumsum.stride(), *s_y, *Vin.stride(), *S_all.stride(), *dG_all.stride(), *dalpha_all.stride(), H, HPG_C, T=T, N=N, GT=GT, NGT=NGT, K_SUB=ksub, PREC=precision, MMA_DT=mma_dt, OCC=occ_bucket(Bsz * H * NGT * ncb, device), HAS_GAMMA=has_gamma, IDX64=idx64, NO_DALPHA=bool(no_dalpha))
