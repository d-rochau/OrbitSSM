
import functools
import os
import re
import subprocess
import warnings
import torch
import triton
import triton.language as tl
from .common import pdot
from triton.language.extra import libdevice
from ... import kernel_options as _kernel_options
from .autotune_policy import autotune_tier
from .common import _cos_approx, _sin_approx

def _env(suffix: str, default: str) -> str:
    ko = _kernel_options.env_override('ssd_mimo', suffix)
    if ko is not None:
        return ko
    keys = ('SSD_MIMO_' + suffix, 'UNIFIED_SSD_' + suffix)
    environ = os.environ
    v = environ.get(keys[0])
    if v is not None:
        return v
    v = environ.get(keys[1])
    if v is not None:
        return v
    from .measured_plan import override as _measured_override
    measured = _measured_override(suffix)
    return default if measured is None else measured
_ENV_KEYS: dict = {}
_MEASURED_OVERRIDE = None

def autotune_off() -> bool:
    return autotune_tier(_env('AUTOTUNE', 'light')) == 'pinned'
_MMA_IO_DT = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

def mma_io_enabled(precision: str) -> bool:
    if precision == 'io':
        return True
    if precision == 'ieee':
        return False
    return _env('MMA_IO', '0') == '1'

def mma_operand_dtype(precision: str, io_dtype):
    if mma_io_enabled(precision):
        return _MMA_IO_DT.get(io_dtype, tl.float32)
    return tl.float32

def dot_precision(precision: str) -> str:
    if precision == 'ieee':
        return 'ieee'
    from .common import FP32_DOT_MODE
    if FP32_DOT_MODE in ('tf32x3', 'bf16x3', 'ieee'):
        return FP32_DOT_MODE
    return 'tf32'
_SM_COUNT_CACHE: dict = {}

def _sm_count(device) -> int:
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx not in _SM_COUNT_CACHE:
        _SM_COUNT_CACHE[idx] = torch.cuda.get_device_properties(idx if idx is not None else 0).multi_processor_count
    return _SM_COUNT_CACHE[idx]

def len_bucket(L: int) -> int:
    import math
    return max(0, int(math.floor(math.log2(max(L, 1) / 1024.0))))

def occ_bucket(n_ctas: int, device=None) -> int:
    import math
    r = n_ctas / max(_sm_count(device), 1)
    return max(0, int(math.floor(math.log2(max(r, 1e-06)))) + 4)
_DEV_CAPS: dict = {}

def _dev_caps(device) -> dict:
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    if idx not in _DEV_CAPS:
        props = torch.cuda.get_device_properties(idx)
        regs = int(getattr(props, 'regs_per_multiprocessor', 0) or 65536)
        thr = int(getattr(props, 'max_threads_per_multi_processor', 0) or 2048)
        from .smem import device_smem_limit
        _DEV_CAPS[idx] = dict(sm=int(props.multi_processor_count), warp=int(getattr(props, 'warp_size', 0) or 32), regs_per_sm=regs, threads_per_sm=thr, regs_per_thread=max(1, regs // max(1, thr)), smem=int(device_smem_limit(idx)), maxnreg_ok=not bool(getattr(torch.version, 'hip', None)))
    return _DEV_CAPS[idx]

def _fwd_maxnreg() -> dict:
    v = _env('FWD_MAXNREG', '')
    return {'maxnreg': int(v)} if v and int(v) else {}

def _mmav3_eight_warp_ok(gt: int, mma_io: bool=False) -> bool:
    return gt >= (64 if mma_io else 32)
_FWD_WARP_CANDIDATES = (4, 8)
_FWD_STAGE_CANDIDATES = (1, 2, 3)
_FWD_MAXNREG_CANDIDATES = (0, 128, 168, 224)
_FWD_LIVE_REG_SHARE = 2.5
_FWD_EST_T_DEFAULT = 64

def _fwd_heuristic_cfg(device, n_ctas: int=0, n_state: int=0, state_only: bool=False, gt: int=0, mma_io: bool=False, pair_half: bool=False, T: int=0) -> tuple[int, int, int]:
    caps = _dev_caps(device)
    lanes = max(1, caps['warp'])
    budget = caps['regs_per_thread']
    warps = 4
    stages = 3 if n_ctas and n_ctas <= caps['sm'] else 1
    live_regs = n_state * (max(int(gt), 1) + (int(T) or _FWD_EST_T_DEFAULT)) / float(warps * lanes)
    if not state_only and n_state > 0 and (live_regs > _FWD_LIVE_REG_SHARE * budget) and _mmav3_eight_warp_ok(gt, mma_io):
        warps = 8
        stages = 3 if mma_io else 1
    if pair_half:
        stages = max(stages, 3)
    return (warps, stages, 0)

def _fwd_fit_stages(args, kw, grid, device):
    from .smem import device_smem_limit, smem_clamp_enabled
    if not smem_clamp_enabled():
        return kw
    limit = device_smem_limit(device)
    start = int(kw.get('num_stages', 1) or 1)
    if start <= 1:
        return kw
    req = _fwd_smem_required(args, kw, grid)
    if req is None or req <= limit:
        return kw
    for st in range(start - 1, 0, -1):
        k2 = dict(kw, num_stages=st)
        r = _fwd_smem_required(args, k2, grid)
        if r is not None and r <= limit:
            return k2
    return kw

def _fwd_cfg_candidates(device, n_ctas: int=0, n_state: int=0, state_only: bool=False, gt: int=0, mma_io: bool=False, pair_half: bool=False, T: int=0):
    caps = _dev_caps(device)
    head = _fwd_heuristic_cfg(device, n_ctas, n_state, state_only, gt, mma_io, pair_half, T)
    primary = [head]
    if gt > 0 and (not _mmav3_eight_warp_ok(gt, mma_io)):
        return (primary, (0,))
    for w in _FWD_WARP_CANDIDATES:
        if w > 4 and (not _mmav3_eight_warp_ok(gt, mma_io)):
            continue
        if w * caps['warp'] > caps['threads_per_sm']:
            continue
        for st in _FWD_STAGE_CANDIDATES:
            cfg = (w, st, 0)
            if cfg not in primary:
                primary.append(cfg)
    if not caps['maxnreg_ok']:
        return (primary, (0,))
    mnr = [m for m in _FWD_MAXNREG_CANDIDATES if m == 0 or m * 4 * caps['warp'] <= caps['regs_per_sm']]
    return (primary, tuple(mnr))

def _fwd_cell_key(n_ctas, n_state, T, gt, g_size, R, O, state_only, mma_io, pair_half, dtype, precision) -> str:
    return 'ctas%d|N%d|T%d|GT%d|G%d|R%d|O%d|%s|%s|%s|%s|%s' % (int(n_ctas), int(n_state), int(T), int(gt), int(g_size), int(R), int(O), 'so1' if state_only else 'so0', 'io1' if mma_io else 'io0', 'ph1' if pair_half else 'ph0', str(dtype).replace('torch.', ''), precision)

def _fwd_launch_cfg(device, n_ctas: int=0, n_state: int=0, state_only: bool=False, gt: int=0, mma_io: bool=False, pair_half: bool=False, cell: str | None=None, T: int=0) -> tuple[int, int]:
    w, s, _mnr, _src = _fwd_launch_cfg_full(device, n_ctas, n_state, state_only, gt, mma_io, pair_half, cell, T)
    return (w, s)

def _fwd_launch_cfg_full(device, n_ctas: int=0, n_state: int=0, state_only: bool=False, gt: int=0, mma_io: bool=False, pair_half: bool=False, cell: str | None=None, T: int=0):
    from .smem import fwd_cfg_key, fwd_cfg_lookup, fwd_cfg_mode
    w = _env('FWD_WARPS', '')
    s = _env('FWD_STAGES', '')
    mnr_env = _env('FWD_MAXNREG', '')
    tw, ts, tm = _fwd_heuristic_cfg(device, n_ctas, n_state, state_only, gt, mma_io, pair_half, T)
    source = 'heuristic'
    if cell is not None and fwd_cfg_mode() != 'off':
        cached = fwd_cfg_lookup(fwd_cfg_key(cell, device))
        if cached is not None:
            tw, ts, tm = cached
            source = 'measured'
    warps = int(w) if w else tw
    stages = int(s) if s else ts
    maxnreg = int(mnr_env) if mnr_env else tm
    if w or s or mnr_env:
        source = 'env'
    if warps >= 8 and gt > 0 and (not _mmav3_eight_warp_ok(gt, mma_io)):
        raise ValueError('SSD-MIMO forward: num_warps=%d cannot be lowered at GT=%d with %s MMA operands -- Triton 3.6 emits wgmma.m64n8k8 (fp32 operands, K20) or mis-strides the nvmma_shared<swizzlingByteWidth=64> descriptor (16-bit operands, K24) there, which returns a ~100%%-wrong y or an illegal memory access depending on N, dtype and route.  Use a G tile of at least %d (SSD_MIMO_FWD_GT) or leave SSD_MIMO_FWD_WARPS unset.  See _mmav3_eight_warp_ok, implementationexperiments/K20_FWD_MMA_FIX_20260906.md and implementationexperiments/K24_BF16_BWD_FAULTS_20260907.md.' % (warps, gt, 'IO-dtype' if mma_io else 'fp32', 64 if mma_io else 32))
    return (warps, stages, maxnreg, source)
_BIG_SMEM_BYTES = 160 * 1024

def _big_smem(device) -> bool:
    from .smem import device_smem_limit
    return device_smem_limit(device) >= _BIG_SMEM_BYTES
_I32_LIMIT = 2 ** 31

def _max_elem_offset(t) -> int:
    off = 1
    for sz, st in zip(t.shape, t.stride()):
        if sz > 1:
            off += (sz - 1) * abs(st)
    return off

def _needs_i64(*tensors) -> bool:
    return any((t is not None and torch.is_tensor(t) and (_max_elem_offset(t) >= _I32_LIMIT) for t in tensors))

def _seg_plan(nc: int, n_ctas: int, device) -> tuple[int, int]:
    if _env('SEG', '1') != '1':
        return (1, nc)
    if torch.cuda.get_device_capability(device) != (9, 0) and _env('SEG_ANY_ARCH', '0') != '1':
        return (1, nc)
    sm = _sm_count(device)
    min_seg = max(1, int(_env('SEG_MIN_CHUNKS', '32')))
    min_nc = int(_env('SEG_MIN_NC', '256'))
    if n_ctas >= sm or nc < max(min_nc, 2 * min_seg):
        return (1, nc)
    nseg_env = _env('SEG_N', '')
    if nseg_env:
        nseg = max(1, int(nseg_env))
    else:
        nseg = min(nc // min_seg, -(-(6 * sm) // max(n_ctas, 1)), 8)
    if nseg <= 1:
        return (1, nc)
    nc_seg = -(-nc // nseg)
    nseg = -(-nc // nc_seg)
    return (nseg, nc_seg)

@triton.jit
def _seg_stitch_kernel(Psi_ptr, Alpha_ptr, Out_ptr, Init_ptr, H, NSEG, GN, s_pb, s_ph, s_ps, s_pg, s_pn, s_ab, s_ah, s_as, s_ib, s_ih, s_ig, s_in, BLK: tl.constexpr, REVERSE: tl.constexpr, HAS_INIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    off = tl.program_id(2) * BLK + tl.arange(0, BLK)
    m = off < GN
    base_p = Psi_ptr + b * s_pb + h * s_ph
    base_o = Out_ptr + b * s_pb + h * s_ph
    if HAS_INIT:
        acc = tl.load(Init_ptr + b * s_ib + h * s_ih + off, mask=m, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros([BLK], dtype=tl.float32)
    for i in range(NSEG):
        if REVERSE:
            sidx = NSEG - 1 - i
        else:
            sidx = i
        tl.store(base_o + sidx * s_ps + off, acc, mask=m)
        a = tl.load(Alpha_ptr + b * s_ab + h * s_ah + sidx * s_as)
        v = tl.load(base_p + sidx * s_ps + off, mask=m, other=0.0).to(tl.float32)
        acc = a * acc + v

@triton.jit
def _dcs_tile(dbase, offs_t, s_dt, DCS_INK: tl.constexpr):
    if DCS_INK:
        d_raw = tl.load(dbase + offs_t * s_dt)
        D_h = tl.cumsum(d_raw, axis=0)
    else:
        D_h = tl.load(dbase + offs_t * s_dt)
    return D_h

@triton.jit
def _dcs_end(dbase, offs_t, s_dt, D_h, T: tl.constexpr, DCS_INK: tl.constexpr):
    D_end_val = tl.sum(tl.where(offs_t == T - 1, D_h, 0.0), axis=0)
    return D_end_val
SEG_RESET_C = tl.constexpr(104.0)
_SEG_RESET_C_HOST = 104.0

def seg_prefix_host(delta_raw, a_log):
    if delta_raw.is_cuda and _env('SEG_PREFIX_FUSED', '1') == '1' and (delta_raw.ndim in (4, 5)) and (a_log.ndim == 1) and (a_log.numel() == delta_raw.shape[1]) and (a_log.device == delta_raw.device) and (min(delta_raw.shape) > 0):
        return seg_prefix_fused(delta_raw, a_log)
    with torch.no_grad():
        shape = [1] * delta_raw.dim()
        shape[1] = -1
        c = delta_raw.double() * torch.exp(a_log.double()).view(shape)
        rst = c >= _SEG_RESET_C_HOST
        cs = torch.where(rst, 0.0, delta_raw.double()).cumsum(-1)
        T = delta_raw.shape[-1]
        pos = torch.arange(T, device=delta_raw.device).expand_as(rst)
        last = torch.where(rst, pos, -1).cummax(-1).values
        base = torch.where(last >= 0, cs.gather(-1, last.clamp(min=0)), 0.0)
        out = (cs - base).float()
        return torch.where(rst, -1.0, out)

@triton.jit
def _segadd(v1, f1, v2, f2):
    return (tl.where(f2 != 0, v2, v1 + v2), f1 | f2)

@triton.jit
def _seg_prefix_kernel(D, A, Out, SHAPE: tl.constexpr, STRIDES: tl.constexpr, A_STRIDE: tl.constexpr, ROWS_PER_HEAD: tl.constexpr, ROWS: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK: tl.constexpr):
    row = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    index = row
    base = 0
    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
        base += index % SHAPE[axis] * STRIDES[axis]
        index //= SHAPE[axis]
    t = tl.arange(0, BLOCK)
    valid = (row[:, None] < ROWS) & (t[None, :] < SHAPE[-1])
    d = tl.load(D + base[:, None] + t[None, :] * STRIDES[-1], valid, 0).to(tl.float64)
    head = row // ROWS_PER_HEAD % SHAPE[1]
    a = tl.load(A + head * A_STRIDE, row < ROWS, 0).to(tl.float64)
    reset = valid & (d * libdevice.exp(a)[:, None] >= 104.0)
    value, _ = tl.associative_scan((tl.where(reset, 0.0, d), reset.to(tl.int32)), 1, _segadd)
    tl.store(Out + row[:, None] * SHAPE[-1] + t[None, :], tl.where(reset, -1.0, value).to(tl.float32), valid)

def seg_prefix_fused(delta_raw, a_log, *, row_tile=None):
    import math
    if delta_raw.ndim not in (4, 5) or a_log.ndim != 1 or a_log.numel() != delta_raw.shape[1]:
        raise ValueError('segmented prefix requires delta[B,H,(G,)nc,T] and a_log[H]')
    if not delta_raw.is_cuda or a_log.device != delta_raw.device:
        raise ValueError('segmented prefix requires one CUDA device')
    if min(delta_raw.shape) <= 0:
        raise ValueError('segmented prefix requires nonempty dimensions')
    row_tile = int(_env('SEG_PREFIX_ROWS', '16')) if row_tile is None else row_tile
    if row_tile not in (1, 4, 16, 32):
        raise ValueError('segmented prefix row_tile must be 1, 4, 16 or 32')
    with torch.no_grad():
        out = torch.empty(delta_raw.shape, device=delta_raw.device, dtype=torch.float32)
        rows = delta_raw.numel() // delta_raw.shape[-1]
        _seg_prefix_kernel[triton.cdiv(rows, row_tile),](delta_raw, a_log, out, tuple(delta_raw.shape), tuple(delta_raw.stride()), a_log.stride(0), math.prod(delta_raw.shape[2:-1]), ROWS=rows, BLOCK_R=row_tile, BLOCK=triton.next_power_of_2(delta_raw.shape[-1]), num_warps=4)
        return out

@triton.jit
def _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK: tl.constexpr):
    d = tl.load(dbase + offs_t * s_dt)
    if DCS_INK:
        rst = a_val * d <= -SEG_RESET_C
        v = tl.where(rst, 0.0, d)
        ri = rst.to(tl.int32)
        D_h, _ = tl.associative_scan((v, ri), 0, _segadd)
    else:
        rst = d < 0.0
        ri = rst.to(tl.int32)
        D_h = tl.where(rst, 0.0, d)
    seg = tl.cumsum(ri, axis=0)
    return (D_h, seg)

@triton.jit
def _same_seg(seg):
    return seg[:, None] == seg[None, :]

def _seg_alpha(a_log, delta_cumsum, nseg, nc_seg, dcs_ink=False):
    Bsz, H, nc, _ = delta_cumsum.shape
    a_val = -torch.exp(a_log.float())
    if dcs_ink:
        d_end = delta_cumsum.float().sum(-1)
    else:
        dc = delta_cumsum.float()
        d_end = torch.where((dc < 0).any(-1), torch.full_like(dc[..., -1], float('inf')), dc[..., -1])
    n_pad = nseg * nc_seg - nc
    if n_pad:
        d_end = torch.nn.functional.pad(d_end, (0, n_pad))
    seg_sum = d_end.view(Bsz, H, nseg, nc_seg).sum(-1)
    return torch.exp(a_val.view(1, -1, 1) * seg_sum)

def _seg_stitch(psi, alpha_seg, init, reverse):
    Bsz, H, nseg, G, N = psi.shape
    out = torch.empty_like(psi)
    dummy = psi
    has_init = init is not None
    ini = init if has_init else dummy
    ini_st = (ini.stride(0), ini.stride(1), ini.stride(2), ini.stride(3)) if has_init else (0, 0, 0, 0)
    blk = min(1024, triton.next_power_of_2(G * N))
    grid = (Bsz, H, (G * N + blk - 1) // blk)
    _seg_stitch_kernel[grid](psi, alpha_seg, out, ini, H, nseg, G * N, *psi.stride(), *alpha_seg.stride(), *ini_st, BLK=blk, REVERSE=reverse, HAS_INIT=has_init, num_warps=4)
    return out

def _pick_gt(g_size: int, cap: int=64) -> int:
    for gt in (cap, 32, 16):
        if gt <= cap and g_size % gt == 0:
            return gt
    raise ValueError(f'G={g_size} is not a multiple of 16 (tensor-core tile envelope)')

@triton.jit
def _pair_swap_2d(tile, T: tl.constexpr, N: tl.constexpr):
    even, odd = tl.split(tl.reshape(tile, (T, N // 2, 2)))
    return tl.reshape(tl.join(odd, even), (T, N))

@triton.jit
def _pair_swap_1d(vec, N: tl.constexpr):
    even, odd = tl.split(tl.reshape(vec, (N // 2, 2)))
    return tl.reshape(tl.join(odd, even), (N,))

@triton.jit
def _pair_bcast_2d(v, T: tl.constexpr, N: tl.constexpr):
    return tl.reshape(tl.join(v, v), (T, N))

@triton.jit
def _half_swap_2d(tile, T: tl.constexpr, N: tl.constexpr):
    h1, h2 = tl.split(tl.permute(tl.reshape(tile, (T, 2, N // 2)), (0, 2, 1)))
    return tl.reshape(tl.permute(tl.join(h2, h1), (0, 2, 1)), (T, N))

@triton.jit
def _half_swap_1d(vec, N: tl.constexpr):
    h1, h2 = tl.split(tl.permute(tl.reshape(vec, (2, N // 2)), (1, 0)))
    return tl.reshape(tl.permute(tl.join(h2, h1), (1, 0)), (N,))

@triton.jit
def _half_bcast_2d(v, T: tl.constexpr, N: tl.constexpr):
    return tl.reshape(tl.permute(tl.join(v, v), (0, 2, 1)), (T, N))

@triton.jit
def _lay_swap_2d(tile, T: tl.constexpr, N: tl.constexpr, PAIR_HALF: tl.constexpr):
    if PAIR_HALF:
        return _half_swap_2d(tile, T, N)
    else:
        return _pair_swap_2d(tile, T, N)

@triton.jit
def _lay_swap_1d(vec, N: tl.constexpr, PAIR_HALF: tl.constexpr):
    if PAIR_HALF:
        return _half_swap_1d(vec, N)
    else:
        return _pair_swap_1d(vec, N)

@triton.jit
def _lay_bcast_2d(v, T: tl.constexpr, N: tl.constexpr, PAIR_HALF: tl.constexpr):
    if PAIR_HALF:
        return _half_bcast_2d(v, T, N)
    else:
        return _pair_bcast_2d(v, T, N)

@triton.jit
def _rot_apply_2d(V, ph_cos, ph_sin, rot_sign, T: tl.constexpr, N: tl.constexpr):
    cs = _pair_bcast_2d(ph_cos, T, N)
    sn = rot_sign[None, :] * _pair_bcast_2d(ph_sin, T, N)
    return cs * V + sn * _pair_swap_2d(V, T, N)
FWD_SCHED_DEFAULT = 3

def fwd_sched(mma_io: bool=False) -> int:
    env = _env('FWD_SCHED', '')
    if env != '':
        return int(env)
    return 0 if mma_io else FWD_SCHED_DEFAULT

@triton.jit
def _fused_chunk_fwd_kernel(C_ptr, B_ptr, X_ptr, Delta_ptr, A_ptr, ThetaH_ptr, Theta_ptr, X0_ptr, Beta_ptr, Gamma_ptr, RotBias_ptr, Y_ptr, SAll_ptr, SF_ptr, H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, s_xb, s_xh, s_xg, s_xr, s_xl, s_db, s_dh, s_dc, s_dt, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_sb, s_sh, s_sc, s_sn, s_sg, s_fb, s_fh, s_fg, s_fn, s_x0b, s_x0h, s_x0g, s_x0n, s_thb, s_thc, s_tht, s_thh, s_thk, s_bh, s_ch, s_br, s_co, NC, NC_SEG, s_x0s, s_fss, T: tl.constexpr, N: tl.constexpr, G_SIZE: tl.constexpr, GT: tl.constexpr, PREC: tl.constexpr, MMA_DT: tl.constexpr=tl.float32, HAS_THETA_H: tl.constexpr=False, HAS_ROT: tl.constexpr=False, HAS_X0: tl.constexpr=False, HPG_B: tl.constexpr=1, HPG_C: tl.constexpr=1, R: tl.constexpr=1, O: tl.constexpr=1, HAS_BETA: tl.constexpr=False, HAS_GAMMA: tl.constexpr=False, HAS_ROT_BIAS: tl.constexpr=False, BIAS_PRE_ROT: tl.constexpr=False, PAIR_SWAP: tl.constexpr=False, PAIR_HALF: tl.constexpr=False, DW_ptr=None, s_wb=0, s_wh=0, s_wl=0, HAS_DW: tl.constexpr=False, STORE_SALL: tl.constexpr=True, STATE_ONLY: tl.constexpr=False, IDX64: tl.constexpr=False, SCHED: tl.constexpr=0, Yg_ptr=None, Z_ptr=None, Dsk_ptr=None, s_zb=0, s_zh=0, s_zt=0, s_zg=0, s_gb=0, s_gh=0, s_gc=0, s_gt=0, s_gg=0, s_go=0, HAS_GATE: tl.constexpr=False, HAS_DSKIP: tl.constexpr=False, STORE_Y: tl.constexpr=True, Lam_ptr=None, Dot_ptr=None, s_do=0, s_dr=0, s_dqb=0, s_dqh=0, s_dql=0, HAS_TRAP: tl.constexpr=False, STORE_DOT: tl.constexpr=False, ThOff_ptr=None, s_ob=0, s_oh=0, s_oc=0, s_ok=0, PHASE_CUMSUM: tl.constexpr=False, DCS_INK: tl.constexpr=False, RO_STACK: tl.constexpr=False, OS: tl.constexpr=1, RS: tl.constexpr=1):
    tl.static_assert(not HAS_DSKIP or HAS_DW, "fused D-skip needs the raw-x (HAS_DW) load: without it X_ptr holds x_scaled, not the skip's operand")
    tl.static_assert(not HAS_DSKIP or O == 1, 'fused D-skip is the O == 1 branch (its weights carry no output axis)')
    tl.static_assert(not (HAS_GATE or HAS_DSKIP) or not STATE_ONLY, 'the epilogue has no place in the STATE_ONLY replay')
    tl.static_assert(not HAS_TRAP or HAS_DW, 'the in-kernel trapezoid IS the write scale: it needs the raw-x (HAS_DW) form')
    tl.static_assert(not HAS_TRAP or not HAS_ROT_BIAS, "the in-kernel trapezoid is Mamba-3's mode at any R, O >= 1: the phase (plus the static READ offset theta_h, whose rotation the dot below and the adjoint kernel carry) with no write-side rot_bias")
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
    causal = offs_t[:, None] >= offs_t[None, :]
    neg_inf = float('-inf')
    bg = h // HPG_B
    cg = h // HPG_C
    pair_idx = offs_n // 2
    rot_sign = tl.where(offs_n % 2 == 0, 1.0, -1.0)
    if HAS_THETA_H and (not PAIR_HALF):
        th_h_full = tl.load(ThetaH_ptr + h * (N // 2) + pair_idx)
        ch_full = _cos_approx(th_h_full)
        sh_full = _sin_approx(th_h_full)
        fwd_coeff = rot_sign * sh_full
    if HAS_ROT_BIAS and (not PAIR_HALF):
        rb_full = tl.load(RotBias_ptr + h * (N // 2) + pair_idx)
        crb = _cos_approx(rb_full)
        srb_fwd = rot_sign * _sin_approx(rb_full)
    if PAIR_HALF:
        R1O1: tl.constexpr = R == 1 and O == 1
        tl.static_assert(not HAS_ROT_BIAS, 'PAIR_HALF does not serve the retired rot_bias')
        tl.static_assert(N >= 32, 'PAIR_HALF needs N/2 >= 16 (the MMA K floor)')
        KH: tl.constexpr = N // 2
        offs_h1 = tl.arange(0, KH)
        offs_h2 = KH + tl.arange(0, KH)
        if HAS_THETA_H:
            th_h = tl.load(ThetaH_ptr + h * KH + offs_h1)
            ch_h = _cos_approx(th_h)
            sh_h = _sin_approx(th_h)
        if HAS_X0:
            x0_base = X0_ptr + b * s_x0b + h * s_x0h + seg * s_x0s + offs_gt[None, :] * s_x0g
            S1 = tl.load(x0_base + offs_h1[:, None] * s_x0n).to(tl.float32)
            S2 = tl.load(x0_base + offs_h2[:, None] * s_x0n).to(tl.float32)
        else:
            S1 = tl.zeros([KH, GT], dtype=tl.float32)
            S2 = tl.zeros([KH, GT], dtype=tl.float32)
        for c in range(c_lo, c_hi):
            if STORE_SALL:
                s_base_st = SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_gt[None, :] * s_sg
                tl.store(s_base_st + offs_h1[:, None] * s_sn, S1)
                tl.store(s_base_st + offs_h2[:, None] * s_sn, S2)
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            if HAS_TRAP:
                lam_t = tl.load(Lam_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                idx_n = c * T + offs_t + 1
                m_n = idx_n < NC * T
                dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                trap_nxt = dt_n * (1.0 - lam_n)
                trap_own = lam_t * dw
                dw = lam_t * dw + trap_nxt
            if HAS_DW:
                dec_agg = dec_agg * dw
                dec_mask = dec_mask * dw[None, :]
            if HAS_TRAP:
                dec_mask = tl.where(offs_t[:, None] == offs_t[None, :], trap_own[None, :], dec_mask)
            S1_ent = S1
            S2_ent = S2
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + offs_h1[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + offs_h1 * s_ok).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
            if R1O1:
                b_base = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt
                B1 = tl.load(b_base + offs_h1[None, :] * s_bn).to(tl.float32)
                B2 = tl.load(b_base + offs_h2[None, :] * s_bn).to(tl.float32)
                if HAS_BETA:
                    be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R).to(tl.float32)[None, :]
                    be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R).to(tl.float32)[None, :]
                    if BIAS_PRE_ROT:
                        B1 = B1 + be1
                        B2 = B2 + be2
                if HAS_ROT:
                    Bt1 = ph_cos * B1 + ph_sin * B2
                    Bt2 = ph_cos * B2 - ph_sin * B1
                else:
                    Bt1 = B1
                    Bt2 = B2
                if HAS_BETA and (not BIAS_PRE_ROT):
                    Bt1 = Bt1 + be1
                    Bt2 = Bt2 + be2
                x_base = b * s_xb + h * s_xh + c * T * s_xl
                x_r = tl.load(X_ptr + x_base + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                xw_r = x_r * dec_agg[:, None]
                G1 = pdot(tl.trans(Bt1).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                G2 = pdot(tl.trans(Bt2).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                if not STATE_ONLY:
                    c_base = C_ptr + b * s_cb + c * s_cc + cg * s_ch + offs_t[:, None] * s_ct
                    C1 = tl.load(c_base + offs_h1[None, :] * s_cn).to(tl.float32)
                    C2 = tl.load(c_base + offs_h2[None, :] * s_cn).to(tl.float32)
                    if HAS_GAMMA:
                        ga1 = tl.load(Gamma_ptr + h * N * O + offs_h1 * O).to(tl.float32)
                        ga2 = tl.load(Gamma_ptr + h * N * O + offs_h2 * O).to(tl.float32)
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
                    if STORE_DOT:
                        trap_dot = tl.sum(Bt1 * Cr1, axis=1) + tl.sum(Bt2 * Cr2, axis=1)
                    S0 = pdot(Cr1.to(MMA_DT), tl.trans(Bt1).to(MMA_DT), input_precision=PREC) + pdot(Cr2.to(MMA_DT), tl.trans(Bt2).to(MMA_DT), input_precision=PREC)
                    Score = S0 * dec_mask
                    if STORE_DOT:
                        if gt_idx == 0:
                            tl.store(Dot_ptr + b * s_dqb + h * s_dqh + (c * T + offs_t) * s_dql, trap_dot)
                    y_intra = pdot(Score.to(MMA_DT), x_r.to(MMA_DT), input_precision=PREC)
                    y_cross = f[:, None] * (pdot(Cr1.to(MMA_DT), S1_ent.to(MMA_DT), input_precision=PREC) + pdot(Cr2.to(MMA_DT), S2_ent.to(MMA_DT), input_precision=PREC))
                    Y_out = y_intra + y_cross
                    if HAS_DSKIP:
                        dv = tl.load(Dsk_ptr + h * (G_SIZE * R) + offs_gt * R)
                        Y_out = Y_out + x_r * dv[None, :]
                    y_base = b * s_yb + h * s_yh + c * s_yc
                    if STORE_Y:
                        tl.store(Y_ptr + y_base + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg, Y_out)
                    if HAS_GATE:
                        zt = tl.load(Z_ptr + b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_gt[None, :] * s_zg).to(tl.float32)
                        tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc + offs_t[:, None] * s_gt + offs_gt[None, :] * s_gg, (Y_out * (zt * tl.sigmoid(zt))).to(Yg_ptr.dtype.element_ty))
            else:
                G1 = tl.zeros([KH, GT], dtype=tl.float32)
                G2 = tl.zeros([KH, GT], dtype=tl.float32)
                for r in tl.static_range(R):
                    b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + r * s_br
                    B1 = tl.load(b_base_r + offs_h1[None, :] * s_bn).to(tl.float32)
                    B2 = tl.load(b_base_r + offs_h2[None, :] * s_bn).to(tl.float32)
                    if HAS_BETA:
                        be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r).to(tl.float32)[None, :]
                        be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r).to(tl.float32)[None, :]
                        if BIAS_PRE_ROT:
                            B1 = B1 + be1
                            B2 = B2 + be2
                    if HAS_ROT:
                        Bt1 = ph_cos * B1 + ph_sin * B2
                        Bt2 = ph_cos * B2 - ph_sin * B1
                    else:
                        Bt1 = B1
                        Bt2 = B2
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        Bt1 = Bt1 + be1
                        Bt2 = Bt2 + be2
                    x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    xw_r = x_r * dec_agg[:, None]
                    G1 += pdot(tl.trans(Bt1).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                    G2 += pdot(tl.trans(Bt2).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                if not STATE_ONLY:
                    if HAS_GATE:
                        zt = tl.load(Z_ptr + b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_gt[None, :] * s_zg).to(tl.float32)
                        silu_z = zt * tl.sigmoid(zt)
                    for o in tl.static_range(O):
                        c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + offs_t[:, None] * s_ct + o * s_co
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
                                g1 = ch_h * ga1 + sh_h * ga2
                                g2 = ch_h * ga2 - sh_h * ga1
                            else:
                                g1 = ga1
                                g2 = ga2
                            Cr1 = Cr1 + g1[None, :]
                            Cr2 = Cr2 + g2[None, :]
                        y_cross = f[:, None] * (pdot(Cr1.to(MMA_DT), S1_ent.to(MMA_DT), input_precision=PREC) + pdot(Cr2.to(MMA_DT), S2_ent.to(MMA_DT), input_precision=PREC))
                        y_intra = tl.zeros([T, GT], dtype=tl.float32)
                        if HAS_DSKIP:
                            dsk_acc = tl.zeros([T, GT], dtype=tl.float32)
                        for r in tl.static_range(R):
                            b_base_r2 = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + r * s_br
                            B1 = tl.load(b_base_r2 + offs_h1[None, :] * s_bn).to(tl.float32)
                            B2 = tl.load(b_base_r2 + offs_h2[None, :] * s_bn).to(tl.float32)
                            if HAS_BETA:
                                be1 = tl.load(Beta_ptr + h * N * R + offs_h1 * R + r).to(tl.float32)[None, :]
                                be2 = tl.load(Beta_ptr + h * N * R + offs_h2 * R + r).to(tl.float32)[None, :]
                                if BIAS_PRE_ROT:
                                    B1 = B1 + be1
                                    B2 = B2 + be2
                            if HAS_ROT:
                                Bt1 = ph_cos * B1 + ph_sin * B2
                                Bt2 = ph_cos * B2 - ph_sin * B1
                            else:
                                Bt1 = B1
                                Bt2 = B2
                            if HAS_BETA and (not BIAS_PRE_ROT):
                                Bt1 = Bt1 + be1
                                Bt2 = Bt2 + be2
                            if STORE_DOT:
                                trap_dot = tl.sum(Bt1 * Cr1, axis=1) + tl.sum(Bt2 * Cr2, axis=1)
                            S0 = pdot(Cr1.to(MMA_DT), tl.trans(Bt1).to(MMA_DT), input_precision=PREC) + pdot(Cr2.to(MMA_DT), tl.trans(Bt2).to(MMA_DT), input_precision=PREC)
                            Score = S0 * dec_mask
                            if STORE_DOT:
                                if gt_idx == 0:
                                    tl.store(Dot_ptr + b * s_dqb + h * s_dqh + (c * T + offs_t) * s_dql + o * s_do + r * s_dr, trap_dot)
                            x_r = tl.load(X_ptr + b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                            if HAS_DSKIP:
                                dv = tl.load(Dsk_ptr + h * (G_SIZE * R) + offs_gt * R + r)
                                dsk_acc += x_r * dv[None, :]
                            y_intra += pdot(Score.to(MMA_DT), x_r.to(MMA_DT), input_precision=PREC)
                        Y_out = y_intra + y_cross
                        if HAS_DSKIP:
                            Y_out = Y_out + dsk_acc
                        if STORE_Y:
                            tl.store(Y_ptr + b * s_yb + h * s_yh + c * s_yc + o * s_yo + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg, Y_out)
                        if HAS_GATE:
                            tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc + o * s_go + offs_t[:, None] * s_gt + offs_gt[None, :] * s_gg, (Y_out * silu_z).to(Yg_ptr.dtype.element_ty))
            S1 = alpha * S1_ent + G1
            S2 = alpha * S2_ent + G2
        do_sf = (s_fss > 0) | (c_hi == NC)
        if do_sf:
            sf_base = SF_ptr + b * s_fb + h * s_fh + seg * s_fss + offs_gt[None, :] * s_fg
            tl.store(sf_base + offs_h1[:, None] * s_fn, S1)
            tl.store(sf_base + offs_h2[:, None] * s_fn, S2)
    else:
        if HAS_X0:
            S = tl.load(X0_ptr + b * s_x0b + h * s_x0h + seg * s_x0s + offs_gt[None, :] * s_x0g + offs_n[:, None] * s_x0n).to(tl.float32)
        else:
            S = tl.zeros([N, GT], dtype=tl.float32)
        for c in range(c_lo, c_hi):
            if STORE_SALL:
                tl.store(SAll_ptr + b * s_sb + h * s_sh + c * s_sc + offs_n[:, None] * s_sn + offs_gt[None, :] * s_sg, S)
            dbase = Delta_ptr + b * s_db + h * s_dh + c * s_dc
            D_h, seg_h = _dcs_seg(dbase, offs_t, s_dt, a_val, DCS_INK)
            seg_end = tl.max(seg_h, axis=0)
            D_end_val = _dcs_end(dbase, offs_t, s_dt, D_h, T, DCS_INK)
            f = tl.where(seg_h == 0, tl.exp(a_val * D_h), 0.0)
            alpha = tl.where(seg_end == 0, tl.exp(a_val * D_end_val), 0.0)
            dec_agg = tl.where(seg_h == seg_end, tl.exp(a_val * (D_end_val - D_h)), 0.0)
            if SCHED & 4 == 0:
                dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
            if HAS_DW:
                dw = tl.load(DW_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
            if HAS_TRAP:
                lam_t = tl.load(Lam_ptr + b * s_wb + h * s_wh + (c * T + offs_t) * s_wl)
                idx_n = c * T + offs_t + 1
                m_n = idx_n < NC * T
                dt_n = tl.load(DW_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                lam_n = tl.load(Lam_ptr + b * s_wb + h * s_wh + idx_n * s_wl, mask=m_n, other=0.0)
                trap_nxt = dt_n * (1.0 - lam_n)
                trap_own = lam_t * dw
                dw = lam_t * dw + trap_nxt
            if HAS_DW:
                dec_agg = dec_agg * dw
                if SCHED & 4 == 0:
                    dec_mask = dec_mask * dw[None, :]
            if HAS_TRAP:
                if SCHED & 4 == 0:
                    dec_mask = tl.where(offs_t[:, None] == offs_t[None, :], trap_own[None, :], dec_mask)
            S_entering = S
            if HAS_ROT:
                th_rot = tl.load(Theta_ptr + b * s_thb + h * s_thh + c * s_thc + offs_t[:, None] * s_tht + tl.arange(0, N // 2)[None, :] * s_thk).to(tl.float32)
                if PHASE_CUMSUM:
                    th_rot = tl.cumsum(th_rot, axis=0) + tl.load(ThOff_ptr + b * s_ob + h * s_oh + c * s_oc + tl.arange(0, N // 2) * s_ok).to(tl.float32)[None, :]
                ph_cos = _cos_approx(th_rot)
                ph_sin = _sin_approx(th_rot)
                if SCHED & 2 == 0:
                    cs_t = _pair_bcast_2d(ph_cos, T, N)
                    sn_t = rot_sign[None, :] * _pair_bcast_2d(ph_sin, T, N)
            if STATE_ONLY:
                G = tl.zeros([N, GT], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + r * s_br
                    Bc_r = tl.load(b_base_r + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if SCHED & 2 != 0:
                            Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin, rot_sign, T, N)
                        else:
                            Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if PAIR_SWAP or HAS_ROT:
                            Bc_r_xor = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor = tl.load(b_base_r + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                        Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    xw_r = x_r * dec_agg[:, None]
                    G += pdot(tl.trans(Bc_r).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                S = alpha * S_entering + G
            elif O == 1:
                c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + offs_t[:, None] * s_ct
                Co = tl.load(c_base_o + offs_n[None, :] * s_cn).to(tl.float32)
                if BIAS_PRE_ROT and HAS_GAMMA:
                    Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)[None, :]
                if HAS_ROT:
                    if SCHED & 2 != 0:
                        Co = _rot_apply_2d(Co, ph_cos, ph_sin, rot_sign, T, N)
                    else:
                        Co = cs_t * Co + sn_t * _pair_swap_2d(Co, T, N)
                if HAS_THETA_H:
                    if PAIR_SWAP or HAS_ROT:
                        Co_xor = _pair_swap_2d(Co, T, N)
                    else:
                        Co_xor = tl.load(c_base_o + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                    C_rd = ch_full[None, :] * Co + fwd_coeff[None, :] * Co_xor
                else:
                    C_rd = Co
                if HAS_GAMMA and (not BIAS_PRE_ROT):
                    gam = tl.load(Gamma_ptr + h * N * O + offs_n * O).to(tl.float32)
                    if HAS_THETA_H:
                        if PAIR_SWAP:
                            gam_xor = _pair_swap_1d(gam, N)
                        else:
                            gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O).to(tl.float32)
                        gam = ch_full * gam + fwd_coeff * gam_xor
                    C_rd = C_rd + gam[None, :]
                if SCHED & 1 == 0:
                    y_cross = f[:, None] * pdot(C_rd.to(MMA_DT), S_entering.to(MMA_DT), input_precision=PREC)
                G = tl.zeros([N, GT], dtype=tl.float32)
                y_intra = tl.zeros([T, GT], dtype=tl.float32)
                if HAS_DSKIP:
                    dsk_acc = tl.zeros([T, GT], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + r * s_br
                    Bc_r = tl.load(b_base_r + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if SCHED & 2 != 0:
                            Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin, rot_sign, T, N)
                        else:
                            Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if PAIR_SWAP or HAS_ROT:
                            Bc_r_xor = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor = tl.load(b_base_r + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                        Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    if HAS_DSKIP:
                        dv = tl.load(Dsk_ptr + h * (G_SIZE * R) + offs_gt * R + r)
                        dsk_acc += x_r * dv[None, :]
                    if STORE_DOT:
                        trap_dot = tl.sum(Bc_r * C_rd, axis=1)
                    xw_r = x_r * dec_agg[:, None]
                    G += pdot(tl.trans(Bc_r).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                    S0 = pdot(C_rd.to(MMA_DT), tl.trans(Bc_r).to(MMA_DT), input_precision=PREC)
                    if SCHED & 4 != 0:
                        dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
                        if HAS_DW:
                            dec_mask = dec_mask * dw[None, :]
                        if HAS_TRAP:
                            dec_mask = tl.where(offs_t[:, None] == offs_t[None, :], trap_own[None, :], dec_mask)
                    Score = S0 * dec_mask
                    if STORE_DOT:
                        if gt_idx == 0:
                            tl.store(Dot_ptr + b * s_dqb + h * s_dqh + (c * T + offs_t) * s_dql + r * s_dr, trap_dot)
                    y_intra += pdot(Score.to(MMA_DT), x_r.to(MMA_DT), input_precision=PREC)
                if SCHED & 1 != 0:
                    y_cross = f[:, None] * pdot(C_rd.to(MMA_DT), S_entering.to(MMA_DT), input_precision=PREC)
                S = alpha * S_entering + G
                Y_out = y_intra + y_cross
                if HAS_DSKIP:
                    Y_out = Y_out + dsk_acc
                y_base = b * s_yb + h * s_yh + c * s_yc
                if STORE_Y:
                    tl.store(Y_ptr + y_base + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg, Y_out)
                if HAS_GATE:
                    zt = tl.load(Z_ptr + b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_gt[None, :] * s_zg).to(tl.float32)
                    tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc + offs_t[:, None] * s_gt + offs_gt[None, :] * s_gg, (Y_out * (zt * tl.sigmoid(zt))).to(Yg_ptr.dtype.element_ty))
            elif RO_STACK:
                TO: tl.constexpr = T * OS
                TR: tl.constexpr = T * RS
                offs_m = tl.arange(0, TO)
                t_m = offs_m % T
                o_m = offs_m // T
                offs_k = tl.arange(0, TR)
                t_k = offs_k % T
                r_k = offs_k // T
                dec_agg_k = tl.reshape(tl.broadcast_to(tl.reshape(dec_agg, (1, T)), (RS, T)), (TR,))
                f_m = tl.reshape(tl.broadcast_to(tl.reshape(f, (1, T)), (OS, T)), (TO,))
                dec_row = tl.reshape(tl.broadcast_to(tl.reshape(dec_mask, (T, 1, T)), (T, RS, T)), (T, TR))
                dec_st = tl.reshape(tl.broadcast_to(tl.reshape(dec_row, (1, T, TR)), (OS, T, TR)), (TO, TR))
                G = tl.zeros([N, GT], dtype=tl.float32)
                for rb in tl.static_range(R // RS):
                    r_st = rb * RS + r_k
                    B_st = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + t_k[:, None] * s_bt + r_st[:, None] * s_br + offs_n[None, :] * s_bn).to(tl.float32)
                    if HAS_BETA:
                        B_st = B_st + tl.load(Beta_ptr + h * N * R + offs_n[None, :] * R + r_st[:, None]).to(tl.float32)
                    x_st = tl.load(X_ptr + b * s_xb + h * s_xh + r_st[:, None] * s_xr + (c * T + t_k)[:, None] * s_xl + offs_gt[None, :] * s_xg).to(tl.float32)
                    G += pdot(tl.trans(B_st).to(MMA_DT), (x_st * dec_agg_k[:, None]).to(MMA_DT), input_precision=PREC)
                S = alpha * S_entering + G
                if HAS_GATE:
                    zt = tl.load(Z_ptr + b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_gt[None, :] * s_zg).to(tl.float32)
                    silu_st = tl.reshape(tl.broadcast_to(tl.reshape(zt * tl.sigmoid(zt), (1, T, GT)), (OS, T, GT)), (TO, GT))
                for ob in tl.static_range(O // OS):
                    o_st = ob * OS + o_m
                    C_st = tl.load(C_ptr + b * s_cb + c * s_cc + cg * s_ch + t_m[:, None] * s_ct + o_st[:, None] * s_co + offs_n[None, :] * s_cn).to(tl.float32)
                    if HAS_GAMMA:
                        C_st = C_st + tl.load(Gamma_ptr + h * N * O + offs_n[None, :] * O + o_st[:, None]).to(tl.float32)
                    Y_out = f_m[:, None] * pdot(C_st.to(MMA_DT), S_entering.to(MMA_DT), input_precision=PREC)
                    for rb in tl.static_range(R // RS):
                        r_st = rb * RS + r_k
                        B_st = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + t_k[:, None] * s_bt + r_st[:, None] * s_br + offs_n[None, :] * s_bn).to(tl.float32)
                        if HAS_BETA:
                            B_st = B_st + tl.load(Beta_ptr + h * N * R + offs_n[None, :] * R + r_st[:, None]).to(tl.float32)
                        Score = pdot(C_st.to(MMA_DT), tl.trans(B_st).to(MMA_DT), input_precision=PREC) * dec_st
                        if STORE_DOT:
                            for rr in tl.static_range(RS):
                                B_rep = tl.load(B_ptr + b * s_bb + c * s_bc + bg * s_bh + t_m[:, None] * s_bt + (rb * RS + rr) * s_br + offs_n[None, :] * s_bn).to(tl.float32)
                                if HAS_BETA:
                                    B_rep = B_rep + tl.load(Beta_ptr + h * N * R + offs_n * R + (rb * RS + rr)).to(tl.float32)[None, :]
                                td = tl.sum(C_st * B_rep, axis=1)
                                if gt_idx == 0:
                                    tl.store(Dot_ptr + b * s_dqb + h * s_dqh + (c * T + t_m) * s_dql + o_st * s_do + (rb * RS + rr) * s_dr, td)
                        x_st = tl.load(X_ptr + b * s_xb + h * s_xh + r_st[:, None] * s_xr + (c * T + t_k)[:, None] * s_xl + offs_gt[None, :] * s_xg).to(tl.float32)
                        Y_out += pdot(Score.to(MMA_DT), x_st.to(MMA_DT), input_precision=PREC)
                    if STORE_Y:
                        tl.store(Y_ptr + b * s_yb + h * s_yh + c * s_yc + o_st[:, None] * s_yo + t_m[:, None] * s_yt + offs_gt[None, :] * s_yg, Y_out)
                    if HAS_GATE:
                        tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc + o_st[:, None] * s_go + t_m[:, None] * s_gt + offs_gt[None, :] * s_gg, (Y_out * silu_st).to(Yg_ptr.dtype.element_ty))
            else:
                G = tl.zeros([N, GT], dtype=tl.float32)
                for r in tl.range(R, loop_unroll_factor=1):
                    b_base_r = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + r * s_br
                    Bc_r = tl.load(b_base_r + offs_n[None, :] * s_bn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_BETA:
                        Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if SCHED & 2 != 0:
                            Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin, rot_sign, T, N)
                        else:
                            Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                    if HAS_ROT_BIAS:
                        if PAIR_SWAP or HAS_ROT:
                            Bc_r_xor = _pair_swap_2d(Bc_r, T, N)
                        else:
                            Bc_r_xor = tl.load(b_base_r + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                        Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor
                    if HAS_BETA and (not BIAS_PRE_ROT):
                        beta_r = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                        Bc_r = Bc_r + beta_r[None, :]
                    x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                    x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                    xw_r = x_r * dec_agg[:, None]
                    G += pdot(tl.trans(Bc_r).to(MMA_DT), xw_r.to(MMA_DT), input_precision=PREC)
                S = alpha * S_entering + G
                if HAS_GATE:
                    zt = tl.load(Z_ptr + b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_gt[None, :] * s_zg).to(tl.float32)
                    silu_z = zt * tl.sigmoid(zt)
                for o in tl.range(O, loop_unroll_factor=1):
                    c_base_o = C_ptr + b * s_cb + c * s_cc + cg * s_ch + offs_t[:, None] * s_ct + o * s_co
                    Co = tl.load(c_base_o + offs_n[None, :] * s_cn).to(tl.float32)
                    if BIAS_PRE_ROT and HAS_GAMMA:
                        Co = Co + tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)[None, :]
                    if HAS_ROT:
                        if SCHED & 2 != 0:
                            Co = _rot_apply_2d(Co, ph_cos, ph_sin, rot_sign, T, N)
                        else:
                            Co = cs_t * Co + sn_t * _pair_swap_2d(Co, T, N)
                    if HAS_THETA_H:
                        if HAS_ROT:
                            Co_xor = _pair_swap_2d(Co, T, N)
                        else:
                            Co_xor = tl.load(c_base_o + (offs_n ^ 1)[None, :] * s_cn).to(tl.float32)
                        C_rd = ch_full[None, :] * Co + fwd_coeff[None, :] * Co_xor
                    else:
                        C_rd = Co
                    if HAS_GAMMA and (not BIAS_PRE_ROT):
                        gam = tl.load(Gamma_ptr + h * N * O + offs_n * O + o).to(tl.float32)
                        if HAS_THETA_H:
                            gam_xor = tl.load(Gamma_ptr + h * N * O + (offs_n ^ 1) * O + o).to(tl.float32)
                            gam = ch_full * gam + fwd_coeff * gam_xor
                        C_rd = C_rd + gam[None, :]
                    y_cross = f[:, None] * pdot(C_rd.to(MMA_DT), S_entering.to(MMA_DT), input_precision=PREC)
                    y_intra = tl.zeros([T, GT], dtype=tl.float32)
                    for r in tl.range(R, loop_unroll_factor=1):
                        b_base_r2 = B_ptr + b * s_bb + c * s_bc + bg * s_bh + offs_t[:, None] * s_bt + r * s_br
                        Bc_r = tl.load(b_base_r2 + offs_n[None, :] * s_bn).to(tl.float32)
                        if BIAS_PRE_ROT and HAS_BETA:
                            Bc_r = Bc_r + tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)[None, :]
                        if HAS_ROT:
                            if SCHED & 2 != 0:
                                Bc_r = _rot_apply_2d(Bc_r, ph_cos, ph_sin, rot_sign, T, N)
                            else:
                                Bc_r = cs_t * Bc_r + sn_t * _pair_swap_2d(Bc_r, T, N)
                        if HAS_ROT_BIAS:
                            if HAS_ROT:
                                Bc_r_xor2 = _pair_swap_2d(Bc_r, T, N)
                            else:
                                Bc_r_xor2 = tl.load(b_base_r2 + (offs_n ^ 1)[None, :] * s_bn).to(tl.float32)
                            Bc_r = crb[None, :] * Bc_r + srb_fwd[None, :] * Bc_r_xor2
                        if HAS_BETA and (not BIAS_PRE_ROT):
                            beta_r2 = tl.load(Beta_ptr + h * N * R + offs_n * R + r).to(tl.float32)
                            Bc_r = Bc_r + beta_r2[None, :]
                        if STORE_DOT:
                            trap_dot = tl.sum(Bc_r * C_rd, axis=1)
                        S0 = pdot(C_rd.to(MMA_DT), tl.trans(Bc_r).to(MMA_DT), input_precision=PREC)
                        if SCHED & 4 != 0:
                            dec_mask = tl.exp(tl.where(causal & _same_seg(seg_h), a_val * (D_h[:, None] - D_h[None, :]), neg_inf))
                            if HAS_DW:
                                dec_mask = dec_mask * dw[None, :]
                            if HAS_TRAP:
                                dec_mask = tl.where(offs_t[:, None] == offs_t[None, :], trap_own[None, :], dec_mask)
                        Score = S0 * dec_mask
                        if STORE_DOT:
                            if gt_idx == 0:
                                tl.store(Dot_ptr + b * s_dqb + h * s_dqh + (c * T + offs_t) * s_dql + o * s_do + r * s_dr, trap_dot)
                        x_base_r = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl
                        x_r = tl.load(X_ptr + x_base_r + offs_gt[None, :] * s_xg + offs_t[:, None] * s_xl).to(tl.float32)
                        y_intra += pdot(Score.to(MMA_DT), x_r.to(MMA_DT), input_precision=PREC)
                    Y_out = y_intra + y_cross
                    y_base = b * s_yb + h * s_yh + c * s_yc + o * s_yo
                    if STORE_Y:
                        tl.store(Y_ptr + y_base + offs_t[:, None] * s_yt + offs_gt[None, :] * s_yg, Y_out)
                    if HAS_GATE:
                        tl.store(Yg_ptr + b * s_gb + h * s_gh + c * s_gc + o * s_go + offs_t[:, None] * s_gt + offs_gt[None, :] * s_gg, (Y_out * silu_z).to(Yg_ptr.dtype.element_ty))
        do_sf = (s_fss > 0) | (c_hi == NC)
        if do_sf:
            tl.store(SF_ptr + b * s_fb + h * s_fh + seg * s_fss + offs_gt[None, :] * s_fg + offs_n[:, None] * s_fn, S)
_LAST_RO_STACK = (False, 1, 1)

def _ro_stack(T: int, R: int, O: int, refused: bool):
    env = _env('FWD_RO_STACK', 'auto')
    if env == '0' or refused or O <= 1:
        return (False, 1, 1)
    if ',' in env:
        os_, rs_ = (int(v) for v in env.split(','))
    else:
        cap = max(1, 64 // int(T))
        os_, rs_ = (min(O, cap), min(R, cap))
    if os_ < 1 or rs_ < 1 or O % os_ or R % rs_:
        return (False, 1, 1)
    if os_ * rs_ <= 1 and env != '1':
        return (False, 1, 1)
    if ',' not in env and env != '1' and (os_ != O or rs_ != R):
        return (False, 1, 1)
    return (True, os_, rs_)
_LAST_STACK_GUARD = ''
SM90_PTXAS_FIXED = (12, 9, 86)

@functools.lru_cache(maxsize=None)
def _ptxas_version_at(path: str):
    try:
        out = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT).decode()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search('\\bV(\\d+)\\.(\\d+)\\.(\\d+)', out)
    return tuple((int(v) for v in m.groups())) if m else None

def sm90_ptxas_version():
    try:
        path = triton.knobs.nvidia.ptxas.path
    except Exception:
        return None
    return _ptxas_version_at(path)

def ro_stack_wgmma_fault(capability, mma_io: bool, ptxas_version='resolve') -> bool:
    if int(tuple(capability)[0]) != 9 or not bool(mma_io):
        return False
    if isinstance(ptxas_version, str) and ptxas_version == 'resolve':
        ptxas_version = sm90_ptxas_version()
    return ptxas_version is None or tuple(ptxas_version) < SM90_PTXAS_FIXED

def fused_chunk_fwd(Ctil, Btil, x_scaled, delta_cumsum, a_log, theta_h=None, theta=None, x0=None, beta=None, gamma=None, rot_bias=None, bias_pre_rot=True, precision='tf32', bc_grouped=False, Gb=1, Gc=1, dw=None, store_sall=True, state_only=False, gate_z=None, d_skip=None, ypre_out=None, gate_out_dtype=None, trap_lam=None, dot_out=None, theta_off=None, pair_half=False, dcs_ink=False, y_token_major=False):
    Bsz, H, nc, T = delta_cumsum.shape
    G_SIZE = x_scaled.shape[2]
    R = x_scaled.shape[3]
    device = x_scaled.device
    f32 = torch.float32
    mma_dt = mma_operand_dtype(precision, x_scaled.dtype)
    mma_io = mma_dt is not tl.float32
    precision = dot_precision(precision)
    if bc_grouped:
        N = Btil.shape[4]
        b_has_r = Btil.dim() >= 6
        c_has_o = Ctil.dim() >= 6
        O = Ctil.shape[5] if c_has_o else 1
    else:
        N = Btil.shape[3]
        b_has_r = Btil.dim() >= 5
        c_has_o = Ctil.dim() >= 5
        O = Ctil.shape[4] if c_has_o else 1
    gt_env = _env('FWD_GT', '')
    GT_base = min(G_SIZE, int(gt_env)) if gt_env else _pick_gt(G_SIZE)
    nseg, nc_seg = _seg_plan(nc, Bsz * H * (G_SIZE // GT_base), device)
    GT = GT_base
    if not gt_env and nseg == 1:
        if torch.cuda.get_device_capability(device) == (9, 0):
            sm = _sm_count(device)
            gt_floor = 32 if mma_io and G_SIZE % 32 == 0 and (GT >= 32) and (_env('GLUON90', 'off').lower() in ('1', 'on', 'true')) else 16
            while GT > gt_floor and G_SIZE % (GT // 2) == 0 and (Bsz * H * (G_SIZE // GT) * 2 <= sm):
                GT //= 2
    NGT = G_SIZE // GT
    has_gate = gate_z is not None
    has_dskip = d_skip is not None
    if has_dskip and O != 1:
        raise NotImplementedError(f'fused D-skip is the O == 1 branch: its weights are [H, G, R] with no output axis, and this call has O={O}')
    if has_dskip and dw is None:
        raise NotImplementedError('fused D-skip needs the raw-x (dw) form: without it x_scaled is already delta-scaled and the skip would add the wrong tensor')
    if state_only and (has_gate or has_dskip):
        raise AssertionError('the fused epilogue has no place in the STATE_ONLY replay (it rebuilds S_all, which the epilogue does not touch)')

    def _native_y(dtype, token_major=False):
        if token_major:
            buf = torch.empty(Bsz, nc * T, H, O, G_SIZE, device=device, dtype=dtype)
            return buf.view(Bsz, nc, T, H, O, G_SIZE).permute(0, 3, 1, 2, 4, 5)
        buf = torch.empty(Bsz, H, O, G_SIZE, nc * T, device=device, dtype=dtype)
        return buf.view(Bsz, H, O, G_SIZE, nc, T).permute(0, 1, 4, 5, 2, 3)
    if state_only:
        Y = torch.empty(1, 1, 1, 1, 1, 1, device=device, dtype=x_scaled.dtype)
    elif has_gate:
        Yg = _native_y(gate_out_dtype or f32, y_token_major)
        Y = ypre_out if ypre_out is not None else torch.empty(1, 1, 1, 1, 1, 1, device=device, dtype=x_scaled.dtype)
    else:
        Y = _native_y(x_scaled.dtype, y_token_major)
    if store_sall:
        sall_dtype = f32 if os.environ.get('UNIFIED_SSD_SALL_FP32') == '1' or not mma_io else x_scaled.dtype
        S_all = torch.empty(Bsz, H, nc, N, G_SIZE, device=device, dtype=sall_dtype)
    else:
        S_all = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
    S_final = torch.empty(Bsz, H, G_SIZE, N, device=device, dtype=f32)
    if bc_grouped:
        s_bb, s_bc, s_bh, s_bt, s_bn = Btil.stride()[:5]
        s_cb, s_cc, s_ch, s_ct, s_cn = Ctil.stride()[:5]
    else:
        s_bb, s_bc, s_bt, s_bn = Btil.stride()[:4]
        s_bh = 0
        s_cb, s_cc, s_ct, s_cn = Ctil.stride()[:4]
        s_ch = 0
    s_br = Btil.stride()[-1] if b_has_r else 0
    s_co = Ctil.stride()[-1] if c_has_o else 0
    hpg_b = H // Gb
    hpg_c = H // Gc
    has_theta_h = theta_h is not None
    has_rot = theta is not None
    has_x0 = x0 is not None
    has_beta = beta is not None
    has_gamma = gamma is not None
    dummy = torch.empty(1, device=device, dtype=f32)
    thh_c = theta_h.float().contiguous() if has_theta_h else dummy
    th_c = (theta if theta.dtype == torch.bfloat16 else theta.float()).contiguous() if has_rot else dummy
    if not has_rot:
        th_st = [0, 0, 0, 0, 0]
    elif th_c.dim() == 5:
        st = th_c.stride()
        th_st = [st[0], st[1], st[2], st[3], st[4]]
    else:
        st = th_c.stride()
        th_st = [st[0], st[1], st[2], 0, st[3]]
    phase_cumsum = theta_off is not None
    if phase_cumsum and (not has_rot):
        raise ValueError('theta_off needs theta (the increments it offsets)')
    off_c = theta_off.float().contiguous() if phase_cumsum else dummy
    phase_kw = {}
    if phase_cumsum:
        st = off_c.stride()
        phase_kw = dict(ThOff_ptr=off_c, s_ob=st[0], s_oh=0 if off_c.shape[1] == 1 else st[1], s_oc=st[2], s_ok=st[3], PHASE_CUMSUM=True)
    x0_c = x0.float().contiguous() if has_x0 else dummy
    x0_st = list(x0_c.stride()) if has_x0 else [0, 0, 0, 0]
    beta_c = beta.float().contiguous() if has_beta else dummy
    gamma_c = gamma.float().contiguous() if has_gamma else dummy
    has_rot_bias = rot_bias is not None
    rb_c = rot_bias.float().contiguous() if has_rot_bias else dummy
    if pair_half and N < 32:
        raise ValueError(f"pair_layout='half' needs N >= 32 (the MMA K floor); got N={N}")
    if pair_half and rot_bias is not None:
        raise ValueError("pair_layout='half' does not serve the retired rot_bias")
    pair_swap = not pair_half and R == 1 and (O == 1) and (N >= 64) and (has_rot or has_theta_h or has_rot_bias) and (_env('FWD_PAIRSWAP', '1') == '1')
    if has_gate:
        if gate_z.shape[-1] != H * G_SIZE:
            raise ValueError(f'gate_z last dim must be H*G_SIZE={H * G_SIZE}, got {gate_z.shape[-1]}')
        if not gate_z.is_contiguous():
            raise ValueError('the fused gate needs a contiguous gate_z')
    z_c = gate_z if has_gate else dummy
    dsk_c = d_skip if has_dskip else dummy
    store_y = not has_gate or ypre_out is not None
    has_trap = trap_lam is not None
    if has_trap:
        if dw is None:
            raise NotImplementedError('the in-kernel trapezoid IS the write scale: pass dw (the raw dt) alongside trap_lam')
        if has_rot_bias:
            raise NotImplementedError('the in-kernel trapezoid is refused with rot_bias (retired write-side angle); the READ offset theta_h is served: the dot k_t . q_t is taken on the rotated tiles and the adjoint kernel rotates by theta_h per head (2026-09-13)')
        if dot_out is not None and dot_out.dim() not in (3, 5):
            raise ValueError(f'dot_out must be [B, H, L_padded] (R = O = 1, the legacy shape) or [B, H, L_padded, O, R] (2026-09-14 generalisation), got shape {tuple(dot_out.shape)}')
    has_or_tail = dot_out is not None and dot_out.dim() == 5
    trap_kw = dict(Lam_ptr=trap_lam if has_trap else dummy, Dot_ptr=dot_out if dot_out is not None else dummy, s_do=dot_out.stride(3) if has_or_tail else 0, s_dr=dot_out.stride(4) if has_or_tail else 0, s_dqb=dot_out.stride(0) if dot_out is not None else 0, s_dqh=dot_out.stride(1) if dot_out is not None else 0, s_dql=dot_out.stride(2) if dot_out is not None else 0, HAS_TRAP=has_trap, STORE_DOT=has_trap and dot_out is not None) if has_trap else {}
    if has_gate:
        zs = z_c.stride()
        ep_kw = dict(Yg_ptr=Yg, Z_ptr=z_c, Dsk_ptr=dsk_c, s_zb=zs[0], s_zh=G_SIZE * zs[2], s_zt=zs[1], s_zg=zs[2], s_gb=Yg.stride(0), s_gh=Yg.stride(1), s_gc=Yg.stride(2), s_gt=Yg.stride(3), s_gg=Yg.stride(5), s_go=Yg.stride(4), HAS_GATE=True, HAS_DSKIP=has_dskip, STORE_Y=store_y)
    elif has_dskip:
        ep_kw = dict(Yg_ptr=dummy, Z_ptr=dummy, Dsk_ptr=dsk_c, HAS_GATE=False, HAS_DSKIP=True, STORE_Y=True)
    else:
        ep_kw = {}
    idx64 = _needs_i64(x_scaled, Y, S_all, S_final, Btil, Ctil, x0_c, dw, th_c, thh_c, beta_c, gamma_c, rb_c, Yg if has_gate else None, z_c, trap_lam, dot_out, off_c if phase_cumsum else None)
    dw_kw = dict(DW_ptr=dw if dw is not None else dummy, s_wb=dw.stride(0) if dw is not None else 0, s_wh=dw.stride(1) if dw is not None else 0, s_wl=dw.stride(2) if dw is not None else 0, HAS_DW=dw is not None)
    _ro_refused = has_rot or has_rot_bias or has_theta_h or has_dskip or bool(pair_half) or (fwd_sched(mma_io) & 4 != 0)
    _ro_stack_on, _ro_os, _ro_rs = _ro_stack(T, R, O, refused=_ro_refused)
    global _LAST_RO_STACK, _LAST_STACK_GUARD
    _LAST_STACK_GUARD = ''
    if _ro_stack_on and ro_stack_wgmma_fault(torch.cuda.get_device_capability(device), mma_io):
        _ro_stack_on, _ro_os, _ro_rs = (False, 1, 1)
        _LAST_STACK_GUARD = 'sm90_io_ro_stack'
    _LAST_RO_STACK = (_ro_stack_on, _ro_os, _ro_rs)
    common_kw = dict(T=T, N=N, G_SIZE=G_SIZE, GT=GT, PREC=precision, MMA_DT=mma_dt, HAS_THETA_H=has_theta_h, HAS_ROT=has_rot, HAS_BETA=has_beta, HAS_GAMMA=has_gamma, HAS_ROT_BIAS=has_rot_bias, BIAS_PRE_ROT=bool(bias_pre_rot), PAIR_SWAP=pair_swap, PAIR_HALF=bool(pair_half), HPG_B=hpg_b, HPG_C=hpg_c, R=R, O=O, IDX64=idx64, SCHED=fwd_sched(mma_io), DCS_INK=bool(dcs_ink), RO_STACK=_ro_stack_on, OS=_ro_os, RS=_ro_rs, **phase_kw, **dw_kw, **trap_kw)
    head_args = (Ctil, Btil, x_scaled, delta_cumsum, a_log, thh_c, th_c)
    mid_args = (beta_c, gamma_c, rb_c)
    stride_args = (H, s_cb, s_cc, s_ct, s_cn, s_bb, s_bc, s_bt, s_bn, *x_scaled.stride(), *delta_cumsum.stride())
    tail_strides = (*th_st, s_bh, s_ch, s_br, s_co)
    if nseg > 1:
        psi = torch.empty(Bsz, H, nseg, G_SIZE, N, device=device, dtype=f32)
        dummy_y = torch.empty(1, 1, 1, 1, 1, 1, device=device, dtype=f32)
        dummy_s = torch.empty(1, 1, 1, 1, 1, device=device, dtype=f32)
        grid_a = (Bsz, H, NGT * nseg)
        wa, sa = _fwd_launch_cfg(device, Bsz * H * NGT * nseg, N, state_only=True, gt=GT, mma_io=mma_io, pair_half=pair_half, cell=_fwd_cell_key(Bsz * H * NGT * nseg, N, T, GT, G_SIZE, R, O, True, mma_io, pair_half, x_scaled.dtype, precision), T=T)
        _fused_chunk_fwd_kernel[grid_a](*head_args, dummy, *mid_args, dummy_y, dummy_s, psi, *stride_args, *dummy_y.stride()[:4], dummy_y.stride(5), dummy_y.stride(4), *dummy_s.stride(), psi.stride(0), psi.stride(1), psi.stride(3), psi.stride(4), 0, 0, 0, 0, *tail_strides, nc, nc_seg, 0, psi.stride(2), HAS_X0=False, STORE_SALL=False, STATE_ONLY=True, num_warps=wa, num_stages=sa, **common_kw)
        alpha_seg = _seg_alpha(a_log, delta_cumsum, nseg, nc_seg, dcs_ink=dcs_ink)
        s_entry = _seg_stitch(psi, alpha_seg, x0_c.contiguous() if has_x0 else None, reverse=False)
        x0_c = s_entry
        x0_st = [s_entry.stride(0), s_entry.stride(1), s_entry.stride(3), s_entry.stride(4)]
        x0_seg_stride = s_entry.stride(2)
        has_x0 = True
    else:
        x0_seg_stride = 0
    grid = (Bsz, H, NGT * nseg)
    _fwd_cell = _fwd_cell_key(Bsz * H * NGT * nseg, N, T, GT, G_SIZE, R, O, state_only, mma_io, pair_half, x_scaled.dtype, precision)
    wc, sc = _fwd_launch_cfg(device, Bsz * H * NGT * nseg, N, state_only=state_only, gt=GT, mma_io=mma_io, pair_half=pair_half, cell=_fwd_cell, T=T)
    _, _, mnrc, _cfg_src = _fwd_launch_cfg_full(device, Bsz * H * NGT * nseg, N, state_only=state_only, gt=GT, mma_io=mma_io, pair_half=pair_half, cell=_fwd_cell, T=T)
    main_args = (*head_args, x0_c, *mid_args, Y, S_all, S_final, *stride_args, *Y.stride()[:4], Y.stride(5), Y.stride(4), *S_all.stride(), *S_final.stride(), *x0_st, *tail_strides, nc, nc_seg, x0_seg_stride, 0)
    main_kw = dict(HAS_X0=has_x0, STORE_SALL=store_sall, STATE_ONLY=state_only, num_warps=wc, num_stages=sc, **{'maxnreg': mnrc} if mnrc else {}, **ep_kw, **common_kw)
    if _env('GLUON90', 'off').lower() in ('1', 'on', 'true'):
        from . import gluon_sm90_fwd as _g90
        _why = _g90.refusal(device=device, T=T, N=N, G_SIZE=G_SIZE, GT=GT, R=R, O=O, mma_io=mma_io, io_dtype=x_scaled.dtype, has_theta_h=has_theta_h, has_rot=has_rot, has_rot_bias=has_rot_bias, pair_half=bool(pair_half), has_gate=has_gate, has_dskip=has_dskip, nseg=nseg, idx64=idx64, Btil=Btil, Ctil=Ctil, x=x_scaled, s_bt=s_bt, s_bn=s_bn, s_ct=s_ct, s_cn=s_cn, state_only=state_only, bias_pre_rot=bias_pre_rot, has_beta=has_beta, has_gamma=has_gamma, phase_cumsum=phase_cumsum, theta=th_c if has_rot else None)
        if _why is None and dw is None:
            _why = 'needs the raw-x write scale (dw)'
        _g90_key = ('fwd', bool(state_only), has_rot, has_theta_h)
        if _why is None:
            _why = _g90.failed_refusal(_g90_key)
        _g90._LAST_ROUTE.clear()
        _g90._LAST_ROUTE.update(used=_why is None, why=_why or '', state_only=bool(state_only))
        if _why is None:
            try:
                _g90.launch(Ctil=Ctil, Btil=Btil, x=x_scaled, delta=delta_cumsum, a_log=a_log, x0_c=x0_c, x0_st=x0_st, x0_seg_stride=x0_seg_stride, beta_c=beta_c, gamma_c=gamma_c, has_x0=has_x0, has_beta=has_beta, has_gamma=has_gamma, Y=Y, S_all=S_all, S_final=S_final, y_strides=(*Y.stride()[:4], Y.stride(5), Y.stride(4)), sall_strides=tuple(S_all.stride()), sf_strides=tuple(S_final.stride()), dw_kw=dw_kw, trap_kw=trap_kw, H=H, T=T, N=N, G_SIZE=G_SIZE, GT=GT, R=R, O=O, hpg_b=hpg_b, hpg_c=hpg_c, nc=nc, nc_seg=nc_seg, s_fss=0, grid=grid, store_sall=bool(store_sall), state_only=bool(state_only), store_y=bool(store_y), dcs_ink=bool(dcs_ink), b_strides=(s_bb, s_bc, s_bh, s_bt, s_bn, s_br), c_strides=(s_cb, s_cc, s_ch, s_ct, s_cn, s_co), theta=th_c if has_rot else None, th_strides=th_st, theta_h=thh_c if has_theta_h else None, idx64=bool(idx64))
                return (Yg if has_gate else Y, S_all, S_final)
            except Exception as _e:
                _g90._LAST_ROUTE.update(used=False, why=_g90.note_failure(_g90_key, _e))
    main_kw = _fwd_fit_stages(main_args, main_kw, grid, device)
    main_kw = _fwd_fix_stack(main_args, main_kw, grid, device)
    main_kw = _fwd_fit_stages(main_args, main_kw, grid, device)
    _check_fwd_smem(main_args, main_kw, grid, T, N, device)
    main_kw = _fwd_fix_spills(main_args, main_kw, grid, device, io_fp32=x_scaled.dtype == torch.float32)
    main_kw = _fwd_autotune(main_args, main_kw, grid, device, _fwd_cell, *_fwd_cfg_candidates(device, Bsz * H * NGT * nseg, N, state_only=state_only, gt=GT, mma_io=mma_io, pair_half=pair_half, T=T), source=_cfg_src, outs=(Y, S_final, Yg if has_gate else None))
    _fused_chunk_fwd_kernel[grid](*main_args, **main_kw)
    return (Yg if has_gate else Y, S_all, S_final)
_LAST_FWD_CFG: dict = {}

def last_fwd_cfg() -> dict:
    return dict(_LAST_FWD_CFG)

def _fwd_out_checksum(outs):
    out = []
    for t in outs:
        if t is None or not torch.is_tensor(t) or t.numel() == 0:
            out.append((0.0, 0.0, 0))
            continue
        f = t.detach().to(torch.float64)
        out.append((float(f.sum()), float(f.abs().max()), int((~torch.isfinite(f)).sum())))
    return tuple(out)

def _fwd_autotune(args, kw, grid, device, cell, primary, mnr_axis, source='heuristic', outs=()):
    from .smem import _time_call, device_smem_limit, fwd_cfg_key, fwd_cfg_lookup, fwd_cfg_mode, fwd_cfg_store, measuring_possible

    def _note(cfg, src):
        global _LAST_FWD_CFG
        _LAST_FWD_CFG = dict(warps=int(cfg[0]), stages=int(cfg[1]), maxnreg=int(cfg[2] or 0), source=src, cell=cell, spill_fix=_LAST_SPILL_FIX, stack_fix=_LAST_STACK_FIX, stack_guard=_LAST_STACK_GUARD)
    running = (int(kw['num_warps']), int(kw['num_stages']), int(kw.get('maxnreg', 0) or 0))
    _note(running, source)
    if cell is None or fwd_cfg_mode() != 'measure':
        return kw
    key = fwd_cfg_key(cell, device)
    if fwd_cfg_lookup(key) is not None or not measuring_possible():
        return kw
    limit = device_smem_limit(device)
    iters = max(1, int(_env('FWD_CFG_ITERS', '3')))
    ref = [None]

    def _trial(cfg):
        k2 = dict(kw)
        k2['num_warps'], k2['num_stages'] = (int(cfg[0]), int(cfg[1]))
        k2.pop('maxnreg', None)
        if cfg[2]:
            k2['maxnreg'] = int(cfg[2])
        try:
            req = _fwd_smem_required(args, k2, grid)
            if req is not None and req > limit:
                return (None, k2)
            ms = _time_call(lambda: _fused_chunk_fwd_kernel[grid](*args, **k2), iters)
            if outs:
                sig = _fwd_out_checksum(outs)
                if ref[0] is None:
                    ref[0] = sig
                elif sig != ref[0]:
                    warnings.warn('SSD-MIMO forward autotune: config (warps=%d, stages=%d, maxnreg=%d) changed the output fingerprint and was DISQUALIFIED; the shipped config stands.  This is a Triton lowering fault, not a tuning result -- please report it with the cell key %r.' % (cfg[0], cfg[1], cfg[2] or 0, cell), RuntimeWarning, stacklevel=2)
                    return (None, k2)
            return (ms, k2)
        except Exception:
            return (None, k2)
    timings, kws = ({}, {})
    for cfg in primary:
        t, k2 = _trial(cfg)
        if t is not None:
            timings[cfg], kws[cfg] = (t, k2)
    if not timings:
        return kw
    best = min(timings, key=timings.get)
    for m in mnr_axis:
        cfg = (best[0], best[1], int(m))
        if cfg in timings:
            continue
        t, k2 = _trial(cfg)
        if t is not None:
            timings[cfg], kws[cfg] = (t, k2)
    best = min(timings, key=timings.get)
    fwd_cfg_store(key, best, {'%d,%d,%d' % c: v for c, v in timings.items()})
    _note(best, 'measured')
    return kws[best]
_FWD_SPILL_FLOOR = 256
_FWD_SPILL_GAIN = 4.0
_FWD_STACK_SPILL_GAIN = float(_env('FWD_STACK_SPILL_GAIN', '2.0'))
_FWD_PROFILE: dict = {}
_LAST_SPILL_FIX = None

def _fwd_profile_key(kw, args):
    return (tuple(sorted(((k, str(v)) for k, v in kw.items() if isinstance(v, (int, bool, str, tl.dtype))))), tuple((a.dtype for a in args[:4] if torch.is_tensor(a))))

def _fwd_kernel_profile(args, kw, grid):
    key = _fwd_profile_key(kw, args)
    if key not in _FWD_PROFILE:
        try:
            compiled = _fused_chunk_fwd_kernel.warmup(*args, grid=grid, **kw)
            compiled._init_handles()
            _FWD_PROFILE[key] = (int(compiled.n_regs), int(compiled.n_spills))
        except Exception:
            _FWD_PROFILE[key] = None
    return _FWD_PROFILE[key]
_LAST_STACK_FIX = None

def _fwd_fix_stack(args, kw, grid, device):
    from .smem import smem_clamp_enabled
    global _LAST_STACK_FIX, _LAST_RO_STACK
    _LAST_STACK_FIX = None
    if not kw.get('RO_STACK'):
        return kw
    if not smem_clamp_enabled() or _env('FWD_SPILL_FIX', '1') == '0' or _env('FWD_RO_STACK', 'auto') != 'auto':
        return kw
    base = _fwd_kernel_profile(args, kw, grid)
    if base is None or base[1] < _FWD_SPILL_FLOOR:
        return kw
    k2 = dict(kw, RO_STACK=False, OS=1, RS=1)
    prof = _fwd_kernel_profile(args, k2, grid)
    if prof is None or prof[1] * _FWD_STACK_SPILL_GAIN > base[1]:
        return kw
    _LAST_STACK_FIX = (base[1], prof[1])
    _LAST_RO_STACK = (False, 1, 1)
    return k2

def _fwd_fix_spills(args, kw, grid, device, io_fp32=False):
    from .smem import device_smem_limit, smem_clamp_enabled
    global _LAST_SPILL_FIX
    _LAST_SPILL_FIX = None
    if not smem_clamp_enabled() or _env('FWD_SPILL_FIX', '1') == '0' or _env('FWD_STAGES', ''):
        return kw
    start = int(kw.get('num_stages', 1) or 1)
    fp32_ops = bool(io_fp32) and kw.get('MMA_DT') == tl.float32
    if start >= max(_FWD_STAGE_CANDIDATES) and (not fp32_ops):
        return kw
    base = _fwd_kernel_profile(args, kw, grid)
    if base is None or base[1] < _FWD_SPILL_FLOOR:
        return kw
    limit = device_smem_limit(device)
    best_kw, best = (kw, base)
    for st in range(start + 1, max(_FWD_STAGE_CANDIDATES) + 1):
        k2 = dict(kw, num_stages=st)
        req = _fwd_smem_required(args, k2, grid)
        if req is None or req > limit:
            break
        prof = _fwd_kernel_profile(args, k2, grid)
        if prof is None:
            continue
        if prof[1] * _FWD_SPILL_GAIN <= base[1] and prof[1] < best[1]:
            best_kw, best = (k2, prof)
            break
    if best_kw is kw and fp32_ops and _fwd_collapsed(base) and (not _env('FWD_WARPS', '')):
        best_kw, best = _fwd_fix_collapse(args, kw, grid, limit, base, device)
    if best_kw is kw:
        return kw
    _LAST_SPILL_FIX = (start, base[1], best[1])
    return best_kw
_FWD_COLLAPSE_REGS = 128

def _fwd_collapsed(prof) -> bool:
    return prof is not None and prof[1] >= _FWD_SPILL_FLOOR and (prof[0] < _FWD_COLLAPSE_REGS)

def _fwd_fix_collapse(args, kw, grid, limit, base, device):
    gt, mma_io = (int(kw.get('GT', 0) or 0), kw.get('MMA_DT') in (tl.bfloat16, tl.float16))
    cands = [w for w in (4, 8) if w < 8 or gt == 0 or _mmav3_eight_warp_ok(gt, mma_io)]
    caps = (None, 255) if _dev_caps(device)['maxnreg_ok'] else (None,)
    best_kw, best = (kw, None)
    for mnr in caps:
        for w in cands:
            for st in _FWD_STAGE_CANDIDATES:
                k2 = dict(kw, num_warps=w, num_stages=st)
                if mnr:
                    k2['maxnreg'] = mnr
                req = _fwd_smem_required(args, k2, grid)
                if req is None or req > limit:
                    break
                prof = _fwd_kernel_profile(args, k2, grid)
                if prof is None or _fwd_collapsed(prof):
                    continue
                if best is None or prof[1] < best[1]:
                    best_kw, best = (k2, prof)
        if best is not None:
            break
    return (best_kw, best) if best is not None else (kw, base)
_SMEM_REQUIRED: dict = {}

def _fwd_smem_required(args, kw, grid) -> int | None:
    key = (tuple(sorted(((k, str(v)) for k, v in kw.items() if isinstance(v, (int, bool, str, tl.dtype))))), tuple((a.dtype for a in args[:4] if torch.is_tensor(a))))
    if key not in _SMEM_REQUIRED:
        try:
            compiled = _fused_chunk_fwd_kernel.warmup(*args, grid=grid, **kw)
            _SMEM_REQUIRED[key] = int(compiled.metadata.shared)
        except Exception:
            _SMEM_REQUIRED[key] = None
    return _SMEM_REQUIRED[key]

def _check_fwd_smem(args, kw, grid, T, N, device) -> None:
    from .smem import SsdSmemOverflow, device_smem_limit, smem_clamp_enabled
    if not smem_clamp_enabled():
        return
    required = _fwd_smem_required(args, kw, grid)
    if required is None:
        return
    limit = device_smem_limit(device)
    if required > limit:
        raise SsdSmemOverflow(required, limit, N, T)
