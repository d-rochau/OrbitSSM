"""Helpers shared by the HybridMamba-MH forward and backward kernels.

Nothing here is a kernel. It is the small set of device-side scalar helpers
(the scan combiner, the gate/rate activations and their derivatives) and the
host-side casting/divisor utilities that BOTH `fwd_gluon.py` and
`bwd_gluon.py` need, hoisted so neither has to import the other.

Triton refuses plain module globals inside `@jit`, so every scalar constant is
a `gl.constexpr` and the law codes are re-declared as `K_*` constexprs of the
`layout.LAW_*` integers.
"""
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


# ---------------------------------------------------------------------------
# scan / activation helpers (device side)
# ---------------------------------------------------------------------------

@gluon.constexpr_function
def _row1(tile):
    """`tile` narrowed to ONE row: size_per_thread[0] = 1, everything else kept.

    A [1, BLOCK_L] tensor in this layout has the p lanes/warps of `tile` as
    pure broadcast, so its SliceLayout(0) is the same linear layout as
    `tile`'s and `convert_layout` between the two is free. It exists for the
    p-free prefix sums (the rotation phase and its adjoint): a scan along a
    SliceLayout axis does not lower; this avoids a full p-replicated
    [BLOCK_P, BLOCK_L] tile plus a reduce over p.
    """
    return gl.BlockedLayout([1, tile.size_per_thread[1]], tile.threads_per_warp,
                            tile.warps_per_cta, tile.order)


@gluon.jit
def _ssm_combine(a_l, b_l, a_r, b_r):
    """(a,b) o (a',b') = (a a', a' b + b'), i.e. h_t = a_t h_{t-1} + b_t."""
    return a_l * a_r, a_r * b_l + b_r


@gluon.jit
def _ssm_combine_q(a_l, b_l, q_l, a_r, b_r, q_r):
    """`_ssm_combine` plus q = a_t h_{t-1}, the PRE-WRITE state of the
    segment's last step: (a,b,q) o (a',b',q') = (a a', a' b + b', a' b + q').

    (a, b) are `_ssm_combine`'s, textually: `a_r * b_l + b_r` is what keeps
    FMA contraction -- and with it h -- bitwise unchanged, so the shared
    product `a_r * b_l` is NOT hoisted. q_l is never read, so every
    cross-lane / cross-warp copy of it is dead and the scan costs no more
    shuffles or barriers than the 2-operand one (pinned by
    a static test of the combine operand count)."""
    return a_l * a_r, a_r * b_l + b_r, a_r * b_l + q_r


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _cmul(c_l, s_l, c_r, s_r):
    """Product of two unit phasors -- the combine of the INCREMENTAL PHASOR.

    An affine phase Theta_t = carry + (t+1)*nu has a phasor that is a
    geometric sequence, so `associative_scan` over a constant leaf
    (cos nu, sin nu) -- seeded at column 0 with the true sincos of
    carry + nu -- produces (cos Theta_t, sin Theta_t) for the whole chunk
    from ONE scalar sincos instead of one per element (K16).

    It is a rotation composition, so it never leaves the unit circle by
    more than rounding: the scan depth inside a chunk is the per-thread run
    plus log2 of the warp width, ~13 steps, and the phase is re-seeded from
    a true sincos at EVERY chunk boundary.  Nothing here is clamped and
    nothing here is an exponent -- stability by construction.
    """
    return c_l * c_r - s_l * s_r, c_l * s_r + s_l * c_r


@gluon.jit
def _softplus(v):
    return gl.where(v > 20.0, v, libdevice.log1p(gl.exp(v)))


@gluon.jit
def _silu_fast(v):
    """`v * sigmoid(v)` with the HARDWARE exponential (H3).

    `gl.exp` on a full [BLOCK_P, BLOCK_L] tile is a software `expf`, and
    at the store of a kernel that runs ONE warp per CTA at 255 registers
    that is far from a rounding-error-sized cost.  `exp2` is
    `ex2.approx.f32`, i.e. the MUFU the decay's own `libdevice.exp2`
    already rides in the hot loop, and 2 ULP against an output that is
    rounded to bf16 -- the same trade `FAST_TRIG` makes for the rotation,
    and a far less sensitive one than the decay's exponent.

    `tails._silu` (the separate-pass gate this fuses) keeps the accurate
    form: it is bandwidth-bound, so accuracy there is free.  The two
    therefore agree to ~1 ULP of fp32 and not to the bit, which is what
    `test_h3_z_gate_matches_the_separate_gate` allows for.
    """
    return v / (1.0 + libdevice.exp2(-v * LOG2E))


@gluon.jit
def _silu_fast_d(v, sg):
    """d silu/dv from v and sigma = sigmoid(v), both already live."""
    return sg * (1.0 + v * (1.0 - sg))


@gluon.jit
def _gate(v, GATE_ACT: gl.constexpr):
    if GATE_ACT == 0:            # raw
        return v
    elif GATE_ACT == 1:          # sigmoid
        return 1.0 / (1.0 + gl.exp(-v))
    elif GATE_ACT == 2:          # exp
        return gl.exp(v)
    elif GATE_ACT == 3:          # tanh
        return libdevice.tanh(v)
    elif GATE_ACT == 5:          # one_plus: SSD's 1 + g
        return 1.0 + v
    else:                        # softplus
        return _softplus(v)


@gluon.jit
def _gate_dv(v, GATE_ACT: gl.constexpr):
    """d act/dv from the PRE-activation, exact in relative terms everywhere.

    The forms through g = act(v) alone cancel
    where the gate saturates: sigmoid's g (1 - g) at v >> 0 is 1 - fl(g) ~ u against a true e^-v (relative
    error u e^v, 1e-3 at v = 10 in fp32 and all of it past v = 17); tanh's 1 - g^2
    the same at |v| >> 0; softplus's 1 - exp(-g) at v << 0.  With e = exp(-|v|)
    (never overflows) each is a quotient of positive terms.
    """
    if GATE_ACT == 0:            # raw
        return 1.0 + 0.0 * v
    elif GATE_ACT == 1:          # sigmoid: e / (1 + e)^2
        e = gl.exp(-gl.abs(v))
        return e / ((1.0 + e) * (1.0 + e))
    elif GATE_ACT == 2:          # exp
        return gl.exp(v)
    elif GATE_ACT == 5:          # one_plus
        return 1.0 + 0.0 * v
    elif GATE_ACT == 3:          # tanh: 4 e / (1 + e)^2, e = exp(-2|v|)
        e = gl.exp(-2.0 * gl.abs(v))
        return 4.0 * e / ((1.0 + e) * (1.0 + e))
    else:                        # softplus: sigmoid(v)
        e = gl.exp(-gl.abs(v))
        return gl.where(v >= 0, 1.0, e) / (1.0 + e)


@gluon.jit
def _rate(v, DYN_ACT: gl.constexpr, DYN_EPS: gl.constexpr):
    """rate = act(A_dyn) + eps.  Bound by construction, never clamped: a clamp
    that binds has zero gradient there."""
    if DYN_ACT == 0:
        r = v
    elif DYN_ACT == 2:
        r = gl.exp(v)
    else:
        r = _softplus(v)
    return r + DYN_EPS


# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------

def _divisor_le(P: int, target: int) -> int:
    bp = max(min(target, P), 1)
    while P % bp:
        bp -= 1
    return bp


def _i32(t):
    return None if t is None else t.to(torch.int32)


#: `row_law`/`row_rot` windows, memoised per TENSOR OBJECT.
#:
#: Both launchers have to know which laws and which rotation bits the launched
#: window holds before they can pick a kernel, and reading that off a device
#: tensor is a device->host SYNC -- the CPU waits for everything already
#: queued, which in the backward means waiting for the forward.  It never
#: shows in GPU self-time but costs WALL time per fwd+bwd on a busy card.  Both tensors
#: are non-differentiable module BUFFERS that never change after construction,
#: so one read per tensor is all anyone needs.
#:
#: The cache is keyed on the tensor object (weakly, so it dies with it) and
#: validated against `_version`, which torch bumps on every in-place write --
#: a caller that mutates `row_law` between calls gets a fresh read, and one
#: that frees it gets a fresh entry rather than a stale address's answer.
_SEL_CACHE = _WeakTensorKeyDictionary()


def selector_window(t, n_off, n_rows):
    """The launched window of a row selector, as a python list.

    Returns None for None, so the callers' `is None` branches survive.
    """
    if t is None:
        return None
    # A contiguous 1-D slice of a 1-D buffer (the row-window split slices
    # row_law/row_rot per call) is a NEW tensor object every call, so a memo
    # keyed on it would miss inside a CUDA-graph capture, where a host read
    # is illegal.  Key such a view on its base and offset into the base list.
    base, off = t, 0
    b = t._base
    if (b is not None and t.dim() == 1 and b.dim() == 1 and t.stride(0) == 1
            and b.stride(0) == 1 and b.is_contiguous()):
        base, off = b, t.storage_offset() - b.storage_offset()
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
    """d rate/dv on the host, the adjoint partner of `_rate`."""
    if act == DYN_RAW:
        return torch.ones_like(v)
    if act == DYN_EXP:
        return torch.exp(v)
    if act == DYN_SOFTPLUS:
        return torch.sigmoid(v)
    raise ValueError(f"unknown dyn act {act}")
