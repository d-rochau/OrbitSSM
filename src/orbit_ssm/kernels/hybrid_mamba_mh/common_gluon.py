
from __future__ import annotations
import torch
from torch.utils.weak import WeakTensorKeyDictionary as _WeakTensorKeyDictionary
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice
from .layout import DYN_EXP, DYN_RAW, DYN_SOFTPLUS, LAW_A_DYN, LAW_MU
LOG2E = gl.constexpr(1.4426950408889634)
TWO_PI = gl.constexpr(6.283185307179586)
INV_TWO_PI = gl.constexpr(0.15915494309189535)
K_DYN = gl.constexpr(LAW_A_DYN)
K_MU = gl.constexpr(LAW_MU)

@gluon.constexpr_function
def _row1(tile):
    return gl.BlockedLayout([1, tile.size_per_thread[1]], tile.threads_per_warp, tile.warps_per_cta, tile.order)

@gluon.jit
def _ssm_combine(a_l, b_l, a_r, b_r):
    return (a_l * a_r, a_r * b_l + b_r)

@gluon.jit
def _ssm_combine_q(a_l, b_l, q_l, a_r, b_r, q_r):
    return (a_l * a_r, a_r * b_l + b_r, a_r * b_l + q_r)

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _cmul(c_l, s_l, c_r, s_r):
    return (c_l * c_r - s_l * s_r, c_l * s_r + s_l * c_r)

@gluon.jit
def _softplus(v):
    return gl.where(v > 20.0, v, libdevice.log1p(gl.exp(v)))

@gluon.jit
def _silu_fast(v):
    return v / (1.0 + libdevice.exp2(-v * LOG2E))

@gluon.jit
def _silu_fast_d(v, sg):
    return sg * (1.0 + v * (1.0 - sg))

@gluon.jit
def _gate(v, GATE_ACT: gl.constexpr):
    if GATE_ACT == 0:
        return v
    elif GATE_ACT == 1:
        return 1.0 / (1.0 + gl.exp(-v))
    elif GATE_ACT == 2:
        return gl.exp(v)
    elif GATE_ACT == 3:
        return libdevice.tanh(v)
    elif GATE_ACT == 5:
        return 1.0 + v
    else:
        return _softplus(v)

@gluon.jit
def _gate_dv(v, GATE_ACT: gl.constexpr):
    if GATE_ACT == 0:
        return 1.0 + 0.0 * v
    elif GATE_ACT == 1:
        e = gl.exp(-gl.abs(v))
        return e / ((1.0 + e) * (1.0 + e))
    elif GATE_ACT == 2:
        return gl.exp(v)
    elif GATE_ACT == 5:
        return 1.0 + 0.0 * v
    elif GATE_ACT == 3:
        e = gl.exp(-2.0 * gl.abs(v))
        return 4.0 * e / ((1.0 + e) * (1.0 + e))
    else:
        e = gl.exp(-gl.abs(v))
        return gl.where(v >= 0, 1.0, e) / (1.0 + e)

@gluon.jit
def _rate(v, DYN_ACT: gl.constexpr, DYN_EPS: gl.constexpr):
    if DYN_ACT == 0:
        r = v
    elif DYN_ACT == 2:
        r = gl.exp(v)
    else:
        r = _softplus(v)
    return r + DYN_EPS

def _divisor_le(P: int, target: int) -> int:
    bp = max(min(target, P), 1)
    while P % bp:
        bp -= 1
    return bp

def _i32(t):
    return None if t is None else t.to(torch.int32)
_SEL_CACHE = _WeakTensorKeyDictionary()

def selector_window(t, n_off, n_rows):
    if t is None:
        return None
    base, off = (t, 0)
    b = t._base
    if b is not None and t.dim() == 1 and (b.dim() == 1) and (t.stride(0) == 1) and (b.stride(0) == 1) and b.is_contiguous():
        base, off = (b, t.storage_offset() - b.storage_offset())
    hit = _SEL_CACHE.get(base)
    ver = base._version
    if hit is not None and hit[0] == ver:
        got = hit[1]
    else:
        got = base.reshape(-1).tolist()
        _SEL_CACHE[base] = (ver, got)
    if base is not t:
        got = got[off:off + t.numel()]
    return got[n_off:n_off + n_rows]

def _f32(t):
    return None if t is None else t.to(torch.float32).contiguous()

def _dact(v, act):
    if act == DYN_RAW:
        return torch.ones_like(v)
    if act == DYN_EXP:
        return torch.exp(v)
    if act == DYN_SOFTPLUS:
        return torch.sigmoid(v)
    raise ValueError(f'unknown dyn act {act}')
