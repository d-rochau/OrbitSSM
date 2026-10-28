
import os
import triton
import triton.language as tl

def nan_debug_fill(t):
    if os.environ.get('SSD_MIMO_DEBUG_NANFILL', '0') == '1' and t.is_floating_point():
        t.fill_(float('nan'))
    return t

def _acc_sem() -> str:
    v = os.environ.get('SSD_MIMO_ATOMIC_SEM', 'relaxed').strip().lower()
    if v not in ('relaxed', 'acq_rel'):
        raise ValueError(f"SSD_MIMO_ATOMIC_SEM={v!r}: expected 'relaxed' or 'acq_rel'")
    return v
ACC_SEM = tl.constexpr(_acc_sem())

def _ptx_trig_backend() -> bool:
    try:
        from triton.runtime import driver
        return driver.active.get_current_target().backend == 'cuda'
    except Exception:
        return False
_USE_PTX_TRIG_DEFAULT = _ptx_trig_backend() and os.environ.get('SSD_MIMO_PTX_TRIG', '1') != '0'

@triton.jit
def _wrap2pi(x):
    k = tl.floor(x * 0.15915494309189535 + 0.5)
    return x - k * 6.28125 - k * 0.0019353071795864768

@triton.jit
def _cos_approx(x, USE_PTX: tl.constexpr=_USE_PTX_TRIG_DEFAULT):
    if USE_PTX:
        return tl.inline_asm_elementwise('cos.approx.f32 $0, $1;', constraints='=f,f', args=[_wrap2pi(x)], dtype=tl.float32, is_pure=True, pack=1)
    return tl.cos(x)
_DTH_OPROUND_DEFAULT = os.environ.get('SSD_MIMO_DTHETA_OPROUND', '1') != '0'
DTH_OPROUND = tl.constexpr(_DTH_OPROUND_DEFAULT)
FP32_DOT_MODE = os.environ.get('SSD_MIMO_FP32_DOT', 'rna').strip().lower()
if FP32_DOT_MODE not in ('tf32', 'rna', 'tf32x3', 'bf16x3', 'ieee'):
    raise ValueError('SSD_MIMO_FP32_DOT=%r: tf32 | rna | tf32x3 | bf16x3 | ieee' % FP32_DOT_MODE)
TF32_RNA = tl.constexpr(FP32_DOT_MODE == 'rna')
_DTH_TF32 = os.environ.get('SSD_MIMO_DTHETA_TF32', 'rna' if FP32_DOT_MODE == 'rna' else 'rz')
DTH_TF32 = tl.constexpr(_DTH_TF32)

@triton.jit
def _opv(v, MMA_DT: tl.constexpr, MMA_IO: tl.constexpr, PREC: tl.constexpr='ieee', ON: tl.constexpr=_DTH_OPROUND_DEFAULT, TF32: tl.constexpr=_DTH_TF32):
    if ON:
        if MMA_IO:
            v = v.to(MMA_DT).to(tl.float32)
        elif PREC == 'tf32':
            if TF32 == 'rna':
                v = _rna_tf32(v)
            elif TF32 == 'rz':
                v = (v.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
    return v

@triton.jit
def _sin_approx(x, USE_PTX: tl.constexpr=_USE_PTX_TRIG_DEFAULT):
    if USE_PTX:
        return tl.inline_asm_elementwise('sin.approx.f32 $0, $1;', constraints='=f,f', args=[_wrap2pi(x)], dtype=tl.float32, is_pure=True, pack=1)
    return tl.sin(x)
import os
from typing import Optional
_STABLE_DEFAULT = os.environ.get('UNIFIED_SSD_STABLE_SCORE', '0') == '1'
_ASCALAR_R_MAX = int(os.environ.get('UNIFIED_SSD_SCALAR_RMAX', '8'))

def _stable_bc(T: int, c_per_o: bool=False) -> int:
    bc = int(os.environ.get('UNIFIED_SSD_STABLE_BC', '0')) or T // 2
    if T % bc != 0 or T // bc not in (2, 4) or bc < 16:
        bc = T // 2
    if c_per_o and os.environ.get('UNIFIED_SSD_CO_BC32', '1') != '1':
        bc = min(bc, T // 4)
    return bc
_DOT_TILE_MIN = 16

def _resolve_stable(stable_score: Optional[bool], per_block_decay: bool, has_rh: bool, T: int, R: int=1, g_size: int=1, a_scalar: bool=False, c_per_o: bool=False) -> bool:
    if stable_score is None:
        stable_score = _STABLE_DEFAULT
    r_max = _ASCALAR_R_MAX if a_scalar else 4
    if has_rh or R > r_max or (not a_scalar and g_size > 16):
        return False
    if not per_block_decay and (not a_scalar):
        return False
    if a_scalar:
        if T < _DOT_TILE_MIN:
            return False
    elif _stable_bc(T, c_per_o) < _DOT_TILE_MIN:
        return False
    return stable_score

def _mimo_axes(B_tok, C_tok):
    b_per_r = B_tok.dim() == 4
    c_per_o = C_tok.dim() == 4
    O = C_tok.shape[-1] if c_per_o else 1
    return (b_per_r, c_per_o, O)

def _check_mimo_support(b_per_r, c_per_o, O, R, use_stable, Q, r_heads, otile=False, a_scalar=False):
    if not (b_per_r or c_per_o):
        return
    if not use_stable:
        raise NotImplementedError('dynamic per-stream B / per-output C require the stable Score path: a_scalar decay needs T >= 16 (full [T,N]x[N,T] dot, no strip tile), the STRIP kernels (per-block decay) need _stable_bc(T, c_per_o) >= 16 (T >= 32 by default, T >= 64 under the legacy per-output-C BC=T//4 fallback), and either family also needs uniform rank and G_SIZE <= 16 (STRIP only); pass stable_score=True or fix the config')
    if Q is not None:
        raise NotImplementedError('dynamic B/C with rotations (Q) not wired yet — run Q=None (complex/real modes)')
    if r_heads is not None:
        raise NotImplementedError('dynamic B/C with per-head ranks not wired')
    r_max = _ASCALAR_R_MAX if a_scalar else 2
    if b_per_r and R > r_max:
        raise NotImplementedError(f'dynamic per-stream B supports R <= {r_max} on this path (a_scalar={a_scalar}); the strip kernels keep R <= 2 for their named bwd per-stream slots, the A-scalar pair takes UNIFIED_SSD_SCALAR_RMAX (D-F7 put the slot in the grid)')
    if c_per_o and O > 2 and (not otile):
        raise NotImplementedError('dynamic per-output C supports O <= 2 (bwd per-output accumulator slots); the differentiable unified_ssd o-tiles above 2')

@triton.jit
def _span_sum(M, offs_t):
    Rrev = tl.cumsum(M, axis=0, reverse=True)
    return tl.sum(tl.where(offs_t[None, :] < offs_t[:, None], Rrev, 0.0), axis=1)

@triton.jit
def _excl_prefix(v, offs_t):
    return tl.sum(tl.where(offs_t[None, :] < offs_t[:, None], v[None, :], 0.0), axis=1)

@triton.jit
def _span_sum_T(MT, offs_t):
    Rr = tl.cumsum(MT, axis=1, reverse=True)
    return tl.sum(tl.where(offs_t[:, None] < offs_t[None, :], Rr, 0.0), axis=0)

@triton.jit
def _rna_tf32(x):
    return tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=r,f', args=[x], dtype=tl.int32, is_pure=True, pack=1).to(tl.float32, bitcast=True)

@triton.jit
def pdot(a, b, acc=None, input_precision: tl.constexpr='tf32'):
    if TF32_RNA and input_precision == 'tf32':
        if a.dtype == tl.float32:
            a = _rna_tf32(a)
        if b.dtype == tl.float32:
            b = _rna_tf32(b)
    return tl.dot(a, b, acc, input_precision=input_precision)
