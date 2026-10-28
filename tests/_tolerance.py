"""The test suite's ONE tolerance module (rule A7: the tolerance follows the arithmetic).

Every numerical comparison in a kernel/SSM test goes through one of three
forms.  Which one is fixed by the arithmetic between the two arms, never by
the observed error (A7).

1. ``assert_close_terms(actual, ref, terms, u, c, per=...)`` -- the
   TERMS-SCALED form, elementwise by default:

       |actual - ref|  <=  c * u * sum|terms|

   ``sum|terms|`` is the sum of the absolute summands that form each value
   (``terms_abs`` computes it by running the fp64 reference in absolute
   arithmetic).  This is the standard forward-error bound of a sum of
   products, ``|fl(s) - s| <= gamma_n * sum|terms|``, so it is scale-free
   AND well defined where the reference is ~0: a gradient that cancels to
   1e-16 in fp64 and to an exact 0 in fp32 is compared against the size of
   the terms that cancelled, not against 1e-16.  It is the only form that
   may be used where a reference element can be ~0.

2. ``assert_rel_close(actual, ref, bound, terms=, u=)`` -- normwise relative
   ``||actual - ref|| / ||ref|| < bound`` (per tensor, or per slice with
   ``per=``).  GUARDED: it raises ``NearZeroReference`` when any compared
   unit has ``||ref|| < k * u * ||sum|terms|||`` -- there the relative error
   measures the reference's own cancellation, not the arm under test, and
   the caller must use form 1.  A relative check can therefore never
   silently compare against ~0.

3. ``assert_dtype_close`` -- the 16-bit policy on top of form 2:
   ``tol16 = clamp(u * sqrt(depth) * sqrt(arms), [0.64, 6.4] * eps)``,
   >= 5 input draws (``draw_seeds``), every draw inside the bound, and the
   mean signed error pooled over draws below bound/4 unless that excess is
   insignificant (|t| <= 3) or carried by outliers (``DrawPool``, ``bias_fails``;
   escalation to 30 draws).

Choosing ``u`` and ``c`` (write the derivation next to the call):

* ``u`` is the unit roundoff of the arithmetic that rounds DIFFERENTLY
  between the arms (``unit_roundoff``): fp32 for two fp32 routes that only
  reassociate, the MMA operand format (tf32/bf16/fp16) for a tensor-core
  dot against an fp64 oracle, fp64 for two fp64 formulations.
* ``terms``: the reference's sum|terms| when only the arm under test
  rounds.  When BOTH arms are computed formulations that round (two fp64
  routes, a chunked route vs a token loop), each carries its own error,
  |a - r| <= gamma_a T_a + gamma_r T_r: pass ``c * (T_arm + T_ref)`` terms
  (``terms_abs`` on each arm).  A reference whose terms are 0 (an exact 0 by
  structure) does not make the other arm's cancelling arithmetic exact.
* nonlinear functions: ``terms_abs`` counts ``|f(x)|`` plus the
  cancellation of the argument through ``f'`` (``T(x) - |x|``); the
  argument's relative rounding (``|f'(x) x|``: exp(x) of a computed x
  adds ``|x|``) belongs in ``c``.
* dense linear algebra (``matrix_exp``, solves) counts each output entry
  as one term; its conditioning (and that of its adjoint) belongs in ``c``
  or in hand-built terms (e.g. |J|^T of the adjoint).
* ``c`` is the rounding depth: ``n + m`` for a sum of n products of m
  factors in the worst case (gamma_{n+m} / u), ``sqrt(n + m)`` for the
  typical random-walk scale; times ``sqrt(arms)`` when both arms round
  independently; plus the amplification of any nonlinear function whose
  ARGUMENT carries rounding (``exp(x)`` with computed ``x`` adds ``|x|``,
  a difference of cumulative sums adds the cumsum's magnitude).  The terms
  bound already carries a sqrt(n)-fold pessimism against the typical error
  (``sum|t|`` vs the rms of the partial sums), so ``sqrt(n + m)`` is not a
  lucky bound.

Bit identity (``rtol=0, atol=0``) is not a tolerance and stays literal where
A7 allows it (same operations in the same order by construction).
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, NamedTuple

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

__all__ = [
    "UNIT_ROUNDOFF", "unit_roundoff", "mma_format", "route_bound",
    "is_sixteen_bit", "tol16", "draw_seeds", "DrawPool", "escalation_seeds", "bias_stats", "bias_fails",
    "NearZeroReference", "terms_abs", "TermsResult",
    "assert_close_terms", "assert_rel_close", "assert_dtype_close",
    "relative", "fd_atol", "gradcheck_fd",
]

# ---------------------------------------------------------------------------
# Unit roundoff of every format the suite compares in
# ---------------------------------------------------------------------------
#: u = eps / 2 (round to nearest).
UNIT_ROUNDOFF = {"fp64": 2.0 ** -53, "fp32": 2.0 ** -24, "tf32": 2.0 ** -11,
                 "fp16": 2.0 ** -11, "bf16": 2.0 ** -8}

_DTYPE_FMT = {torch.float64: "fp64", torch.float32: "fp32",
              torch.float16: "fp16", torch.bfloat16: "bf16"}


def unit_roundoff(fmt) -> float:
    """``u`` of a format name ("fp64", "fp32", "tf32", "bf16", "fp16"), a
    torch dtype, or a float (returned unchanged)."""
    if isinstance(fmt, float):
        return fmt
    if isinstance(fmt, torch.dtype):
        fmt = _DTYPE_FMT[fmt]
    return UNIT_ROUNDOFF[fmt]


def mma_format(precision, io_dtype=torch.float32):
    """The format an SSD-MIMO dot rounds its operands to; None = no MMA.

    `ieee`: fp32 FMA, no tensor cores.  `tf32` (the kernels' default, and
    what the retired alias "bf16" resolves to): tf32 operands.  `io`:
    operands in the IO dtype, fp32 IO -> tf32."""
    if precision == "ieee":
        return None
    if precision == "io":
        return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(
            io_dtype, "tf32")
    return "tf32"


def route_bound(fmt, depth, arms=2):
    """A7 bound for two routes that do NOT perform identical arithmetic.

    Two routes may be asserted BIT-IDENTICAL only where both perform the
    same floating-point operations in the same order by construction (a
    pure data permutation, an address change, a flag that selects the same
    launch).  Everywhere else -- at every precision, `ieee` included -- the
    claim is `u(fmt) * sqrt(depth) * sqrt(arms)`, normwise relative:

    fmt "fp32"  class (a): both routes feed IDENTICAL operand values to the
                same arithmetic and only the fp32 accumulation order
                differs (a split, an atomic order, a reduction tree, an
                FMA dot re-tiled, whole MMA results combined differently).
    fmt "tf32" / "bf16" / "fp16"  a tensor-core dot whose LOWERING differs
                between the routes (another operand layout, another
                instruction tiling): operand conversion and the MMA's
                internal accumulation (not IEEE-rounded) are compiler- and
                hardware-owned, so the routes differ at the operand
                format's scale.  Clamped to A7's band [0.64, 6.4] * eps.

    `depth` is the length of the longest sum feeding the compared output.
    """
    u = UNIT_ROUNDOFF[fmt]
    bound = u * math.sqrt(depth) * math.sqrt(arms)
    if fmt != "fp32":
        eps = 2.0 * u
        bound = min(max(bound, 0.64 * eps), 6.4 * eps)
    return bound


def relative(actual, expected):
    """Relative Frobenius error, computed in fp64 (unguarded: a MEASUREMENT
    for messages and reports; assertions use ``assert_rel_close``)."""
    return ((actual.double() - expected.double()).norm()
            / expected.double().norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------
# sum|terms| by running the fp64 reference in absolute arithmetic
# ---------------------------------------------------------------------------
class TermsResult(NamedTuple):
    """``out`` / ``grads`` / ``param_grads``: the fp64 reference values;
    ``terms`` / ``grad_terms`` / ``param_grad_terms``: sum|terms| of each,
    same structure.  ``grads`` follows the order of the requires-grad
    tensor arguments; ``param_grads`` is keyed by parameter name."""
    out: Any
    terms: Any
    grads: Any
    grad_terms: Any
    param_grads: Any = None
    param_grad_terms: Any = None


_aten = torch.ops.aten


def _is_float(t):
    return isinstance(t, torch.Tensor) and t.is_floating_point()


class _Terms(torch.Tensor):
    """A value together with the sum of the absolute summands forming it.

    Absolute arithmetic: ``a -/+ b -> |a| + |b|``; any multilinear op with
    non-negative structure (mul, matmul, einsum, conv, sum, cumsum, gather,
    views, their backward ops) is applied to the magnitudes; a nonlinear
    function's output counts as ONE term plus the cancellation its argument
    carries, pushed through the derivative: ``|f(x)| + |f'(x)| (T(x) -
    |x|)`` (a silu/exp/cos of a cancelled sum inherits that sum's error
    scale; the argument's RELATIVE rounding, ``|f'(x) x|``, is the caller's
    amplification term in ``c``).  Scale ops (sqrt, rsqrt, reciprocal,
    division) stay one term: see ``_POINTWISE``.  Plain tensors and
    Python scalars that meet a _Terms value are exact constants, |v|.
    """

    @staticmethod
    def __new__(cls, val, mag, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(
            cls, val.size(), strides=val.stride(),
            storage_offset=val.storage_offset(), dtype=val.dtype,
            device=val.device, layout=val.layout,
            requires_grad=requires_grad)

    def __init__(self, val, mag, requires_grad=False):
        self.val = val
        self.mag = mag

    def __repr__(self):
        return f"_Terms(val={self.val!r}, mag={self.mag!r})"

    def __tensor_flatten__(self):
        return ["val", "mag"], None

    @staticmethod
    def __tensor_unflatten__(inner, meta, outer_size, outer_stride):
        return _Terms(inner["val"], inner["mag"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = _dispatch(func, args, kwargs)
        return return_and_correct_aliasing(func, args, kwargs, out)


def _vals(tree):
    return pytree.tree_map(
        lambda x: x.val if isinstance(x, _Terms) else x, tree)


def _mags(tree):
    def m(x):
        if isinstance(x, _Terms):
            return x.mag
        if _is_float(x):
            return x.abs()
        if isinstance(x, float):
            return abs(x)
        if isinstance(x, int) and not isinstance(x, bool):
            return x      # ints are shapes/dims/indices, never summands
        return x
    return pytree.tree_map(m, tree)


def _amag(x):
    """Magnitude of an ARITHMETIC operand (add/sub/mul/div/min/max): every
    number and every non-bool tensor, integer ones included, is |x| (an
    integer tensor multiplying a value is a signed constant, not an
    index)."""
    if isinstance(x, _Terms):
        return x.mag
    if isinstance(x, torch.Tensor):
        return x if x.dtype == torch.bool else x.abs()
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return abs(x)
    return x


def _dense(t):
    """Non-overlapping and dense: the strides are a permutation of a
    contiguous layout."""
    dims = sorted((st, sz) for st, sz in zip(t.stride(), t.shape) if sz != 1)
    expect = 1
    for st, sz in dims:
        if st != expect:
            return False
        expect *= sz
    return True


def _match_strides(v, m):
    """The magnitude in the value's memory layout: view ops on a _Terms
    value are legal for its strides, so they must be for the magnitude."""
    if (not isinstance(m, torch.Tensor) or m.shape != v.shape
            or m.stride() == v.stride()):
        return m
    if not _dense(v):
        return m
    out = torch.empty_strided(v.shape, v.stride(), dtype=m.dtype,
                              device=m.device)
    return out.copy_(m)


def _wrap(v, m):
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        return _Terms(v, _match_strides(v, m))
    return v


def _wrap_tree(vals, mags):
    vf, spec = pytree.tree_flatten(vals)
    mf = pytree.tree_leaves(mags) if mags is not None else [None] * len(vf)
    return pytree.tree_unflatten(
        [_wrap(v, m) for v, m in zip(vf, mf)], spec)


def _mag_of_value(v):
    return pytree.tree_map(lambda x: x.abs() if _is_float(x) else x, v)


def _p(name):
    return getattr(_aten, name)


# data movement and non-negative (multi)linear ops: the same op on mags
_STRUCT = {
    "view", "_unsafe_view", "reshape", "permute", "transpose", "t",
    "expand", "squeeze", "unsqueeze", "slice", "select", "index_select",
    "gather", "index", "_unsafe_index", "cat", "stack", "clone", "detach",
    "contiguous", "type_as", "reshape_as", "flatten", "unflatten", "swapaxes",
    "moveaxis", "narrow_copy", "_autocast_to_reduced_precision",
    "_autocast_to_full_precision",
    "alias", "lift_fresh", "split", "split_with_sizes", "unbind", "narrow",
    "flip", "roll", "repeat", "_to_copy", "to", "diag_embed", "diagonal", "tril",
    "triu", "constant_pad_nd", "unfold", "sum", "mean", "cumsum", "cumprod",
    "prod", "repeat_interleave", "expand_copy", "view_copy",
    "slice_scatter", "select_scatter", "diagonal_scatter", "as_strided",
    "as_strided_scatter", "masked_fill", "index_put", "index_add",
    "scatter", "scatter_add", "scatter_reduce", "embedding",
    "embedding_dense_backward", "mm", "bmm", "matmul", "mv", "dot",
    "addmm", "baddbmm", "addbmm", "addmv", "addcmul", "convolution",
    "convolution_backward", "mul", "_reshape_alias", "movedim",
    "squeeze_copy", "unsqueeze_copy", "permute_copy", "transpose_copy",
    "t_copy", "split_copy", "unbind_copy", "select_backward",
    "slice_backward", "index_select_backward", "unfold_backward",
    "diagonal_backward", "expand_as", "view_as", "sum_to_size",
    "new_empty_strided", "empty_like",
    "linalg_vector_norm", "abs", "block_diag", "full_like_placeholder",
    "reflection_pad1d", "replication_pad1d", "channel_shuffle",
    "pixel_shuffle", "_conj", "conj",
    "resolve_conj", "resolve_neg", "positive", "masked_scatter",
    "index_fill", "where",
}
# `where(cond, a, b)`: cond is bool (passes through both paths unchanged)

# a -/+ b: magnitudes add
_ADDSUB = {"add", "sub", "rsub"}
# comparisons / integer-valued / boolean: the value only
_PLAIN = {
    "eq", "ne", "lt", "gt", "le", "ge", "isfinite", "isnan", "isinf",
    "argmax", "argmin", "any", "all", "logical_and", "logical_or",
    "logical_not", "logical_xor", "_local_scalar_dense", "equal",
    "is_nonzero", "nonzero", "bitwise_and", "bitwise_or",
    "bitwise_not", "floor", "ceil", "round", "trunc", "floor_divide",
    "signbit", "sym_size", "sym_stride", "sym_numel", "_to_copy_int",
    "zeros_like", "ones_like", "full_like", "new_zeros", "new_ones",
    "new_full", "new_empty", "arange", "empty", "zeros", "ones", "full",
    "rand_like", "randn_like", "bernoulli", "sign", "sgn", "heaviside",
}
# nonlinear: the output counts as one term
_NONLINEAR = {
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sigmoid",
    "tanh", "softplus", "silu", "sin", "cos", "tan", "sqrt", "rsqrt",
    "reciprocal", "erf", "erfc", "erfinv", "atan", "atan2", "asin", "acos",
    "sinh", "cosh", "asinh", "acosh", "atanh", "gelu", "elu", "logit",
    "lgamma", "digamma", "logaddexp", "remainder", "fmod", "xlogy", "special_xlog1py",
    "hypot", "log_sigmoid_forward", "_softmax", "_log_softmax",
    "logsumexp", "logcumsumexp", "mish", "hardswish", "hardsigmoid",
    # dense linear algebra: each output entry is one term; the matrix's
    # conditioning (and its adjoint's) belongs in the caller's c.  A
    # componentwise rule, |A^-1|(|B| + |A||X|), was tried and
    # compounds through chained orthogonal factors (a chain of 16 orthogonal
    # factors -> sum|terms| / |ref| ~ 1e56), i.e. a vacuous bound.
    "_linalg_solve_ex", "linalg_solve", "linalg_solve_ex", "linalg_inv_ex",
    "linalg_lu_factor_ex", "linalg_lu_solve", "lu_unpack", "_linalg_det",
    "linalg_matrix_exp", "matrix_exp", "triangular_solve",
    "linalg_solve_triangular", "_linalg_eigh", "linalg_cholesky_ex",
    "cholesky_solve", "_linalg_svd",
}


def _pow(func, args, kwargs):
    base, expo = args[0], args[1]
    vo = func(*_vals(args), **_vals(kwargs))
    if (not isinstance(expo, torch.Tensor) and float(expo) == int(expo)
            and int(expo) >= 1 and isinstance(base, _Terms)):
        mo = base.mag ** int(expo)          # x^k = x*...*x: a product
    else:
        mo = vo.abs()
    return _wrap(vo, mo)


def _div(func, args, kwargs):
    a, b = args[0], args[1]
    vo = func(*_vals(args), **_vals(kwargs))
    ma = _amag(a)
    vb = _vals(b)
    if isinstance(vb, torch.Tensor):
        mo = ma / vb.abs()
    else:
        mo = ma / abs(vb)
    if not isinstance(mo, torch.Tensor):
        mo = torch.as_tensor(mo, dtype=vo.dtype, device=vo.device)
    return _wrap(vo, mo.expand_as(vo).clone() if mo.shape != vo.shape else mo)


def _mul_guard(ma, mb):
    out = ma * mb
    if isinstance(out, torch.Tensor):
        zero = (ma == 0) | (mb == 0) if isinstance(ma, torch.Tensor) and \
            isinstance(mb, torch.Tensor) else None
        if zero is not None:
            out = torch.where(zero, torch.zeros_like(out), out)
    return out


def _clamp(func, args, kwargs):
    x = args[0]
    vo = func(*_vals(args), **_vals(kwargs))
    vx = _vals(x)
    moved = vo != vx
    mo = torch.where(moved, vo.abs(), _mags(x))
    return _wrap(vo, mo)


def _minmax(func, args, kwargs):
    vo = func(*_vals(args), **_vals(kwargs))
    ma, mb = _amag(args[0]), _amag(args[1])
    mo = torch.maximum(torch.as_tensor(ma, dtype=vo.dtype, device=vo.device),
                       torch.as_tensor(mb, dtype=vo.dtype, device=vo.device))
    return _wrap(vo, mo.expand_as(vo).clone())


def _reduce_extreme(func, args, kwargs):
    out = func(*_vals(args), **_vals(kwargs))
    mag_all = func(*_mags(args), **_mags(kwargs))
    if isinstance(out, (tuple, list)):       # (values, indices)
        return (_wrap(out[0], mag_all[0]), out[1])
    return _wrap(out, mag_all)


def _relu(func, args, kwargs):
    vo = func(*_vals(args), **_vals(kwargs))
    mo = torch.where(vo != 0, _mags(args[0]), torch.zeros_like(vo))
    return _wrap(vo, mo)


def _excess(t):
    """sum|terms| of a value beyond its own size, i.e. the cancellation it
    carries (None when there is none: an exact or single-signed value)."""
    if not isinstance(t, _Terms):
        return None
    ex = (t.mag - t.val.abs()).clamp_min(0)
    return ex if bool((ex > 0).any()) else None


_SQRT_PI = math.sqrt(math.pi)


def _dfun(name, v, o, args, kwargs):
    """f'(v) of a unary pointwise op, from its value v and output o."""
    if name == "exp":
        return o
    if name == "exp2":
        return o * math.log(2.0)
    if name == "expm1":
        return o + 1
    if name == "log":
        return 1 / v
    if name == "log2":
        return 1 / (v * math.log(2.0))
    if name == "log10":
        return 1 / (v * math.log(10.0))
    if name == "log1p":
        return 1 / (1 + v)
    if name == "sigmoid":
        return o * (1 - o)
    if name == "tanh":
        return 1 - o * o
    if name == "softplus":
        beta = args[1] if len(args) > 1 else kwargs.get("beta", 1)
        return torch.sigmoid(beta * v)
    if name == "silu":
        sg = torch.sigmoid(v)
        return sg * (1 + v * (1 - sg))
    if name == "sin":
        return torch.cos(v)
    if name == "cos":
        return torch.sin(v)
    if name == "tan":
        return 1 + o * o
    if name == "sqrt":
        return 0.5 / o
    if name == "rsqrt":
        return 0.5 * o ** 3
    if name == "reciprocal":
        return o * o
    if name in ("erf", "erfc"):
        return 2 / _SQRT_PI * torch.exp(-v * v)
    if name == "atan":
        return 1 / (1 + v * v)
    if name in ("asin", "acos"):
        return 1 / torch.sqrt(1 - v * v)
    if name == "sinh":
        return torch.cosh(v)
    if name == "cosh":
        return torch.sinh(v)
    if name == "asinh":
        return 1 / torch.sqrt(v * v + 1)
    if name == "acosh":
        return 1 / torch.sqrt(v * v - 1)
    if name == "atanh":
        return 1 / (1 - v * v)
    if name == "logit":
        return 1 / (v * (1 - v))
    # central difference for the rest (gelu, elu, mish, ...): a magnitude
    # factor, 1e-10 relative is plenty
    h = 1e-6 * v.abs().clamp_min(1.0)
    rest = _vals(args[1:])
    return (func_cache[name](v + h, *rest, **_vals(kwargs))
            - func_cache[name](v - h, *rest, **_vals(kwargs))) / (2 * h)


func_cache = {}


def _propagated_excess(func, name, args, kwargs, out):
    """|f'(x)| * excess(x) for every cancelling tensor argument x of a
    pointwise nonlinear f: the argument's error scale beyond its own size
    (c * u * excess) passes through f' (first order).  The argument's
    RELATIVE rounding (c * u * |x|) is the caller's amplification term."""
    extra = torch.zeros_like(out)
    for i, a in enumerate(args):
        ex = _excess(a)
        if ex is None:
            continue
        v = a.val
        if name == "atan2":
            y, x = _vals(args[0]), _vals(args[1])
            r2 = x * x + y * y
            d = (x if i == 0 else y) / r2
        elif name == "logaddexp":
            other = _vals(args[1 - i])
            d = torch.sigmoid(v - other)
        elif name == "hypot":
            d = v / out
        elif name in ("xlogy", "special_xlog1py"):
            x, y = _vals(args[0]), _vals(args[1])
            if name == "xlogy":
                d = torch.log(y) if i == 0 else x / y
            else:
                d = torch.log1p(y) if i == 0 else x / (1 + y)
        elif i == 0:
            func_cache.setdefault(name, func)
            d = _dfun(name, v, out, args, kwargs)
        else:
            continue
        extra = extra + torch.nan_to_num(d.abs() * ex, nan=0.0, posinf=0.0)
    return extra


# pointwise members of _NONLINEAR whose argument's cancellation propagates
# through f'.  NOT the scale ops (sqrt, rsqrt, reciprocal, non-integer pow,
# a division's denominator): they normalise (x * rsqrt(mean x^2), x / |x|),
# where the numerator's terms already carry x's error and the denominator's
# error is the same error (scale invariance: rel(x/|x|) <= 2 rel(x)); adding
# it again cubes the terms-to-value ratio per layer (measured:
# LM logits sum|terms| overflowed to inf).
_POINTWISE = {
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sigmoid",
    "tanh", "softplus", "silu", "sin", "cos", "tan", "erf", "erfc", "erfinv", "atan", "atan2", "asin", "acos",
    "sinh", "cosh", "asinh", "acosh", "atanh", "gelu", "elu", "logit",
    "lgamma", "digamma", "logaddexp", "xlogy", "special_xlog1py", "hypot",
    "mish", "hardswish", "hardsigmoid",
}


_SPECIAL = {
    "pow": _pow, "div": _div, "clamp": _clamp, "clamp_min": _clamp,
    "clamp_max": _clamp, "hardtanh": _clamp, "maximum": _minmax,
    "minimum": _minmax, "fmax": _minmax, "fmin": _minmax,
    "amax": _reduce_extreme, "amin": _reduce_extreme, "max": _reduce_extreme,
    "min": _reduce_extreme, "relu": _relu,
}


def _out_of_place(func):
    """aten.add_.Tensor -> aten.add.Tensor (None if there is none)."""
    name = func._schema.name.split("::")[-1]
    if not name.endswith("_") or name.startswith("_"):
        return None
    base = getattr(_aten, name[:-1], None)
    if base is None:
        return None
    return getattr(base, func._overloadname, None)


def _dispatch(func, args, kwargs):
    name = func._schema.name.split("::")[-1]
    mutable = name.endswith("_") and not name.startswith("_")
    if name == "copy_":
        dst, src = args[0], args[1]
        if not isinstance(dst, _Terms):
            raise ValueError(
                "terms_abs: an in-place write of a terms value into a plain "
                "tensor; allocate the buffer from an input (x.new_zeros(...))")
        dst.val.copy_(_vals(src))
        m = _mags(src)
        dst.mag.copy_(m if isinstance(m, torch.Tensor) else abs(m))
        return dst
    if mutable:
        dst = args[0]
        oop = _out_of_place(func)
        if oop is None:
            raise NotImplementedError(f"terms_abs: no rule for {func}")
        if not isinstance(dst, _Terms):
            raise ValueError(
                f"terms_abs: {func} writes a terms value into a plain "
                "tensor; allocate the buffer from an input (x.new_zeros)")
        res = _dispatch(oop, args, kwargs)
        dst.val.copy_(_vals(res))
        dst.mag.copy_(_mags(res))
        return dst

    if name in _PLAIN:
        out = func(*_vals(args), **_vals(kwargs))
        if name in ("zeros_like", "ones_like", "full_like", "new_zeros",
                    "new_ones", "new_full", "new_empty", "empty_like",
                    "sign", "sgn", "heaviside"):
            return _wrap_tree(out, _mag_of_value(out))
        return out
    if name in _SPECIAL:
        return _SPECIAL[name](func, args, kwargs)
    if name in _ADDSUB:
        vo = func(*_vals(args), **_vals(kwargs))
        ma, mb = _amag(args[0]), _amag(args[1])
        alpha = abs(kwargs.get("alpha", 1))
        mo = ma + alpha * mb
        if not isinstance(mo, torch.Tensor):
            mo = torch.as_tensor(mo, dtype=vo.dtype, device=vo.device)
        if mo.shape != vo.shape:
            mo = mo.expand_as(vo).clone()
        return _wrap(vo, mo)
    if name == "neg":
        return _wrap(_vals(args[0]).neg(), _amag(args[0]).clone())
    if name == "mul":
        vo = func(*_vals(args), **_vals(kwargs))
        mo = _mul_guard(_amag(args[0]), _amag(args[1]))
        if not isinstance(mo, torch.Tensor):
            mo = torch.as_tensor(mo, dtype=vo.dtype, device=vo.device)
        if mo.shape != vo.shape:
            mo = mo.expand_as(vo).clone()
        return _wrap(vo, mo)
    if name in _NONLINEAR:
        out = func(*_vals(args), **_vals(kwargs))
        mo = _mag_of_value(out)
        if name in _POINTWISE and isinstance(out, torch.Tensor):
            mo = mo + _propagated_excess(func, name, args, kwargs, out)
        return _wrap_tree(out, mo)
    if name in _STRUCT:
        vo = func(*_vals(args), **_vals(kwargs))
        mo = func(*_mags(args), **_mags(kwargs))
        return _wrap_tree(vo, mo)
    from torch._decomp import decomposition_table
    decomp = decomposition_table.get(func)
    if decomp is not None:
        return decomp(*args, **kwargs)
    raise NotImplementedError(
        f"terms_abs: no absolute-arithmetic rule for {func}; add it to "
        "tests/_tolerance.py (_STRUCT for non-negative multilinear/data "
        "movement, _NONLINEAR for a function whose output is one term) or "
        "supply sum|terms| by hand")


def terms_abs(fn: Callable, *args, grad_outputs=None, params=None,
              **kwargs) -> TermsResult:
    """Run the reference ``fn(*args, **kwargs)`` in fp64 absolute arithmetic.

    Returns the reference values and, for each value, ``sum|terms|`` -- the
    sum of the absolute summands that form it (see ``_Terms``).  Floating
    tensor arguments are promoted to fp64; those with ``requires_grad``
    also get their gradients (and the gradients' terms) back, through the
    reference's own autograd graph, when the output is a scalar or
    ``grad_outputs`` (same structure as the output, ``None`` for outputs
    outside the loss) is given.

    ``params``: an ``nn.Module`` the reference calls.  Its parameters and
    floating buffers become terms values for the call (the module must
    already be fp64); parameters with ``requires_grad`` get gradients in
    ``param_grads`` / ``param_grad_terms``.  Without ``params`` a module's
    weights are exact constants (|w|) and get no gradient.

    Everything else the reference builds from plain constants
    (``torch.zeros``, ``torch.arange``) is an exact constant; buffers that
    are written in place must be allocated from an input (``x.new_zeros``).
    """
    import contextlib

    wrt = []

    def wrap_in(x, track=True):
        if not _is_float(x):
            return x
        v = x.detach().double()
        rg = bool(x.requires_grad) and track
        t = _Terms(v, v.abs(), requires_grad=rg)
        if rg:
            wrt.append(t)
        return t

    wargs = pytree.tree_map(wrap_in, args)
    wkw = pytree.tree_map(wrap_in, kwargs)
    n_args = len(wrt)
    names, ctx = [], contextlib.nullcontext()
    if params is not None:
        from torch.nn.utils.stateless import _reparametrize_module
        repl, seen = {}, {}
        # tied parameters (lm_head.weight is embedding.weight) must be ONE
        # terms leaf under every alias, or the alias is a constant
        for name, p in params.named_parameters(remove_duplicate=False):
            if id(p) not in seen:
                seen[id(p)] = wrap_in(p)
                if p.requires_grad:
                    names.append(name)
            repl[name] = seen[id(p)]
        for name, bf in params.named_buffers(remove_duplicate=False):
            if _is_float(bf):
                if id(bf) not in seen:
                    seen[id(bf)] = wrap_in(bf, track=False)
                repl[name] = seen[id(bf)]
        ctx = _reparametrize_module(params, repl)
    # the terms run is fp64 arithmetic: never under a caller's autocast
    with torch.enable_grad(), ctx, \
            torch.autocast("cuda", enabled=False), \
            torch.autocast("cpu", enabled=False):
        out = fn(*wargs, **wkw)
        grads = grad_terms = pgrads = pterms = None
        outs = [o for o in pytree.tree_leaves(out) if isinstance(o, _Terms)]
        scalar = len(outs) == 1 and outs[0].numel() == 1
        if wrt and (grad_outputs is not None or scalar):
            if grad_outputs is None:
                gos = [None]
            else:
                gl = pytree.tree_leaves(grad_outputs)
                allo = [o for o in pytree.tree_leaves(out)
                        if isinstance(o, torch.Tensor)]
                if len(gl) != len(allo):
                    raise ValueError("terms_abs: grad_outputs must match the "
                                     "output's tensor leaves (None to skip)")
                keep = [(o, wrap_in(g.detach(), track=False))
                        for o, g in zip(allo, gl)
                        if g is not None and isinstance(o, _Terms)]
                outs, gos = [k[0] for k in keep], [k[1] for k in keep]
            gs = torch.autograd.grad(outs, wrt, grad_outputs=gos,
                                     allow_unused=True)
            gv = [g.val.detach() if isinstance(g, _Terms) else g for g in gs]
            gm = [g.mag.detach() if isinstance(g, _Terms) else g for g in gs]
            grads, grad_terms = gv[:n_args], gm[:n_args]
            if params is not None:
                pgrads = dict(zip(names, gv[n_args:]))
                pterms = dict(zip(names, gm[n_args:]))
    vals = pytree.tree_map(
        lambda x: x.val.detach() if isinstance(x, _Terms) else x, out)
    mags = pytree.tree_map(
        lambda x: x.mag.detach() if isinstance(x, _Terms) else
        (x.abs() if _is_float(x) else x), out)
    return TermsResult(vals, mags, grads, grad_terms, pgrads, pterms)


# ---------------------------------------------------------------------------
# The three assertion forms
# ---------------------------------------------------------------------------
class NearZeroReference(AssertionError):
    """A relative check was asked to compare against a ~0 reference."""


def _units(t, per):
    """[n_units, unit_size] view of ``t``: per="element" -> every element,
    per=() / None -> the whole tensor, per=dims -> one unit per index of
    the kept dims."""
    if per == "element":
        return t.reshape(-1, 1)
    if per is None or per == ():
        return t.reshape(1, -1)
    dims = (per,) if isinstance(per, int) else tuple(per)
    dims = tuple(d % t.dim() for d in dims)
    rest = [d for d in range(t.dim()) if d not in dims]
    return t.permute(*dims, *rest).reshape(
        math.prod(t.shape[d] for d in dims), -1)


def _prep(actual, ref, terms, name):
    a = actual.detach().double()
    r = ref.detach().double().to(a.device)
    s = torch.as_tensor(terms).detach().double().to(a.device)
    if a.shape != r.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} vs reference "
                             f"{tuple(r.shape)}")
    s = s.expand_as(r)
    if (s < 0).any() or not torch.isfinite(s).all():
        raise ValueError(f"{name}: sum|terms| must be finite and >= 0")
    fin = torch.isfinite(r)
    if not fin.all():
        same = (a == r) | (torch.isnan(a) & torch.isnan(r))
        bad = (~fin) & (~same)
        assert not bad.any(), (f"{name}: non-finite reference entries differ "
                               f"at {bad.nonzero()[:4].tolist()}")
    assert torch.isfinite(a[fin]).all(), (
        f"{name}: non-finite values where the reference is finite at "
        f"{((~torch.isfinite(a)) & fin).nonzero()[:4].tolist()}")
    return a.where(fin, 0.0), r.where(fin, 0.0), s.where(fin, 0.0)


def _is16(u):
    return u >= UNIT_ROUNDOFF["fp16"]


def assert_close_terms(actual, ref, terms, u, c, *, per="element", name="",
                       pool=None, bias="auto"):
    """``|actual - ref| <= c * u * sum|terms|`` (A7 terms-scaled form).

    ``terms``: sum|terms| of each reference value (``terms_abs(...)`` or the
    caller's closed form), broadcastable to ``ref``.  ``u``: unit roundoff
    (a format name, dtype or float, ``unit_roundoff``).  ``c``: the derived
    rounding depth (module docstring).  ``per="element"`` (default) checks
    every element; ``per=dims`` checks the Frobenius norm of the error of
    every slice indexed by ``dims`` against the norm of its terms (a
    per-head metric); ``per=()`` the whole tensor.  Elements whose terms
    are exactly 0 are exact on both arms and must be equal.

    Bias (16-bit u, or ``bias=True``): the signed error normalised by its
    bound, ``z = (a - r) / (c u s)``, averages to 0 under rounding; its mean
    over >= 256 elements must stay below 1/4 (a systematic lean is a
    defect; |z| <= 1 so the mean's sd is <= 1/16).  With ``pool`` the mean
    is also pooled over draws (``pool.assert_unbiased()`` at the end).

    Returns the worst ratio error / bound.
    """
    uu = unit_roundoff(u)
    a, r, s = _prep(actual, ref, terms, name)
    err = (a - r)
    bound = c * uu * s
    if per == "element":
        e_u, b_u = err.abs().reshape(-1), bound.reshape(-1)
    else:
        e_u = _units(err, per).norm(dim=1)
        b_u = _units(bound, per).norm(dim=1)
    exact = b_u == 0
    ratio = torch.where(exact, torch.where(e_u == 0, 0.0, math.inf),
                        e_u / b_u.clamp_min(1e-300))
    worst = ratio.max().item() if ratio.numel() else 0.0
    if worst > 1.0:
        k = int(ratio.argmax())
        nbad = int((ratio > 1).sum())
        if per == "element":
            idx = list(torch.unravel_index(torch.tensor(k), r.shape))
            idx = [int(i) for i in idx]
            where = (f"at {idx}: actual {a.reshape(-1)[k].item():+.6e} ref "
                     f"{r.reshape(-1)[k].item():+.6e} sum|terms| "
                     f"{s.reshape(-1)[k].item():.3e}")
        else:
            where = f"unit {k} of per={per}"
        raise AssertionError(
            f"{name}: |actual - ref| > c*u*sum|terms| in {nbad}/{ratio.numel()}"
            f" units (c={c:.3g}, u={uu:.3e}); worst ratio {worst:.3g} {where}")
    do_bias = _is16(uu) if bias == "auto" else bool(bias)
    live = bound > 0
    if do_bias and int(live.sum()) >= _BIAS_MIN_NUMEL:
        z = (err[live] / bound[live]).mean().item()
        assert abs(z) < 0.25, (
            f"{name}: systematic signed error, mean (a-r)/(c u s) = {z:+.3f} "
            "(limit 1/4)")
        if pool is not None:
            pool.add(name, 1.0, z, 0.0)
    return worst


def _guard(r, s, uu, k, per, name):
    ru = _units(r, per).norm(dim=1)
    su = _units(s, per).norm(dim=1)
    low = ru < k * uu * su
    if low.any():
        j = int(low.nonzero()[0])
        raise NearZeroReference(
            f"{name}: relative check refused -- {int(low.sum())}/{low.numel()}"
            f" unit(s) (per={per!r}) have ||ref|| < k*u*||sum|terms||| "
            f"(k={k}, u={uu:.3e}); first: ||ref|| {ru[j].item():.3e} vs "
            f"||sum|terms||| {su[j].item():.3e}.  The reference is ~0 against "
            "its own summands, so a relative error measures its cancellation; "
            "use assert_close_terms (tests/_tolerance.py, "
            "A7).")


def assert_rel_close(actual, ref, bound, *, terms, u, k=1.0, per=(), name=""):
    """Normwise relative ``||a - r|| / ||r|| < bound`` per unit (``per`` as
    in ``assert_close_terms``, default the whole tensor), GUARDED: raises
    ``NearZeroReference`` if any unit has ``||r|| < k * u * ||terms||``.

    ``terms`` is required: the guard is what makes the relative form safe.
    ``u`` is the unit roundoff of the arithmetic under test.  Returns the
    worst relative error."""
    uu = unit_roundoff(u)
    a, r, s = _prep(actual, ref, terms, name)
    _guard(r, s, uu, k, per, name)
    ru = _units(r, per).norm(dim=1)
    eu = _units(a - r, per).norm(dim=1)
    rel = eu / ru
    worst = rel.max().item()
    assert worst < bound, (f"{name}: relative error {worst:.3e} >= "
                           f"{bound:.3e} (per={per!r})")
    return worst


# ---------------------------------------------------------------------------
# 16-bit policy
# ---------------------------------------------------------------------------
# bf16 / fp16 cells get ONE derived, feasible bound -- neither so large that it
# hides a defect nor so small that it fails on a single rounding (or passes
# only because both arms rounded to the same grid) -- and correctness is shown
# by MANY input draws all staying inside it plus a pooled bias check.  One
# draw inside the bound could be luck; five with no common sign should not be.
#
# THE BOUND FOLLOWS THE ARITHMETIC BETWEEN THE ARMS, NOT THE DTYPE LABEL:
#   class (a) route-vs-route on IDENTICAL rounded inputs with fp32
#     accumulation: the input rounding cancels, so the cell tests route
#     equivalence at an fp32-scale bound even when it is bf16-parametrised.
#     Such cells keep their bound -- never loosen them onto tol16.
#   class (b) kernel vs the fp64 oracle, or vs a torch reference that a
#     standing test pins to the fp64 oracle AND whose kernel arm is proven to
#     have run the kernel (`last_route() is None`) and that is fed the SAME
#     rounded inputs in fp32/fp64: the kernel's own 16-bit rounding does not
#     cancel -> ``tol16`` + multi-draw + pooled bias, via assert_dtype_close.

_SIXTEEN_BIT = (torch.bfloat16, torch.float16)

# Feasible band, in units of eps(dtype).  bf16: [5.0e-3, 5.0e-2].
#   floor 0.64*eps (= 1.28 u): below one unit roundoff of the OUTPUT the bound
#     either fails on a single-ulp flip or passes only when both arms are
#     bit-identical, i.e. it measures the grid, not the math.
#   ceiling 6.4*eps: the Mamba-3 parity defect was 1.3e-1 at bf16
#     and passed a 1.5e-1 bound; the ceiling keeps every bf16 bound >= 2.6x
#     below a defect of that size.  A cell that needs more than this is a
#     finding, not a tolerance.
_BAND_EPS = (0.64, 6.4)

# Draw count for 16-bit cells.  HYBRID_SSM_TEST_DRAWS overrides (e.g. 1 for a
# quick local run, 20 for a soak).  fp32 cells always run their single,
# original draw.
_DEFAULT_DRAWS = 5
_BIAS_MIN_NUMEL = 256


def is_sixteen_bit(dtype):
    return dtype in _SIXTEEN_BIT


def tol16(dtype, depth, *, arms=1):
    """The derived 16-bit bound on the relative Frobenius error.

        bound = clamp( u * sqrt(depth) * sqrt(arms),  band(dtype) )
        u     = eps(dtype) / 2          (unit roundoff, round-to-nearest)

    Derivation: an output that accumulates ``depth`` terms, each carrying
    one independent round-to-nearest error of at most u relative to the
    term, has a random-walk error of RMS u*sqrt(depth) terms.  The metric is
    the relative FROBENIUS error, itself an RMS over the tensor, so no
    max-over-elements tail factor applies; the pessimism is in normalising
    by ONE term rather than by the sum.  This is the TYPICAL scale, not the
    worst case: the multi-draw loop and the pooled signed-bias check carry
    the "not luck" argument.  ``arms=1`` against an fp64 / fp32 reference
    (class (b)); ``arms=2`` only when BOTH sides round independently in 16
    bits along DIFFERENT arithmetic.

    ``depth`` is the number of 16-bit-rounded terms accumulated into one
    output element in the route under test -- the chunk length for the
    intra-chunk MMA, the state size N for a state contraction, the token
    count for a whole-sequence parameter reduction.  Callers state it; it is
    never fitted to an observed error.
    """
    if not is_sixteen_bit(dtype):
        raise ValueError(f"tol16 is the 16-bit policy; got {dtype}")
    eps = torch.finfo(dtype).eps
    raw = 0.5 * eps * math.sqrt(max(int(depth), 1)) * math.sqrt(arms)
    lo, hi = (_BAND_EPS[0] * eps, _BAND_EPS[1] * eps)
    return min(max(raw, lo), hi)


def draw_seeds(dtype, base_seed=0, n=None):
    """Seeds for a cell's input draws: ``[base_seed]`` for fp32/fp64 (the
    original single draw, unchanged), ``n`` (default 5) distinct seeds for
    bf16/fp16, starting with ``base_seed`` so draw 0 IS the original cell."""
    if not is_sixteen_bit(dtype):
        return [base_seed]
    if n is None:
        n = _DEFAULT_DRAWS
    n = int(os.environ.get("HYBRID_SSM_TEST_DRAWS", n))
    base_seed += int(os.environ.get("HYBRID_SSM_TEST_SEED_BASE", 0))     # false-positive-rate runs: other draws
    return [base_seed + 7919 * k for k in range(max(n, 1))]


def _rel_frob(actual, expected):
    a, e = actual.detach().double(), expected.detach().double()
    return ((a - e).norm() / e.norm().clamp_min(1e-30)).item()


def _signed_bias(actual, expected):
    """(additive, gain) signed relative biases, both in the metric's units.

    additive: mean(a - e) / rms(e)            -- a constant offset
    gain:     <a - e, e> / <e, e>             -- a scale error (a = (1+g) e)
    Round-to-nearest errors scatter around zero, so both shrink like
    1/sqrt(numel) under rounding and stay put under a real defect.
    """
    a, e = actual.detach().double().flatten(), expected.detach().double().flatten()
    d = a - e
    ee = (e * e).sum().clamp_min(1e-300)
    rms = (ee / e.numel()).sqrt()
    return (d.mean() / rms).item(), ((d * e).sum() / ee).item()


#: significance of a pooled bias: |t| = |mean| / (sd / sqrt(n)) must exceed this for a failure
BIAS_T_CRIT = 3.0
#: fraction trimmed from EACH end of the per-draw values for the trimmed mean (floor(0.1 n) values per end)
BIAS_TRIM = 0.10
#: total draws of an escalated cell (HYBRID_SSM_BIAS_DRAWS overrides)
_ESCALATED_DRAWS = 30


def escalation_seeds(dtype, base_seed, n_have=None):
    """The seeds of the extra draws of an escalated cell: the continuation of ``draw_seeds``' sequence
    (``base + 7919 k``) from the ``n_have`` draws already run up to HYBRID_SSM_BIAS_DRAWS (default 30) in total.
    Empty for fp32/fp64 (a single draw) and when the cell already ran that many."""
    if not is_sixteen_bit(dtype):
        return []
    have = len(draw_seeds(dtype, base_seed)) if n_have is None else n_have
    total = int(os.environ.get("HYBRID_SSM_BIAS_DRAWS", _ESCALATED_DRAWS))
    b0 = draw_seeds(dtype, base_seed, 1)[0]
    return [b0 + 7919 * k for k in range(have, total)]


def error_tail_fails(ratios, *, screen=False):
    """The typical-scale rule on per-draw error ratios r_i = error_i / bound_i of a 16-bit GRADIENT: the median must be
    <= 1 and the tail no more than 2.  ``screen=True`` is the five-draw screen (the maximum <= 2: with five draws the
    maximum is the 80 % quantile and the only tail there is); the verdict on an escalated set (about 30 draws) takes the
    90th percentile (nearest rank) instead of the single maximum, so one heavy-tailed draw of a well-behaved tensor does not
    decide, while an error that is too large in more than one draw in ten, or in the median draw, still fails.  Returns
    (fails, median, tail)."""
    r = sorted(float(x) for x in ratios)
    n = len(r)
    med = r[n // 2]
    tail = r[-1] if screen or n < 10 else r[min(n - 1, math.ceil(0.9 * n) - 1)]
    return (med > 1.0 or tail > 2.0), med, tail


def bias_stats(vals):
    """Pooled statistics of the per-draw signed biases ``vals``: n, mean, sample sd, t = mean / (sd / sqrt n)
    (inf for a nonzero mean with sd 0 or n < 2, 0 for a zero mean), median and the 10 %-trimmed mean."""
    v = sorted(float(x) for x in vals)
    n = len(v)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n >= 2 else float("nan")
    if mean == 0.0:
        t = 0.0
    elif n < 2 or sd == 0.0:
        t = math.inf
    else:
        t = mean / (sd / math.sqrt(n))
    k = int(BIAS_TRIM * n)
    core = v[k:n - k]
    med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    return dict(n=n, mean=mean, sd=sd, t=t, median=med, trimmed=sum(core) / len(core))


def bias_fails(vals, bound):
    """The pooled-bias rule.  FAIL iff all three hold: |mean| >= bound/4 (the size criterion), |t| > 3 (the
    mean differs from 0 beyond its own sampling noise) and |10 %-trimmed mean| >= bound/4 (one outlier draw does
    not decide).  Returns (fails, size_exceeded, stats)."""
    st = bias_stats(vals)
    lim = bound / 4
    size = abs(st["mean"]) >= lim
    return size and abs(st["t"]) > BIAS_T_CRIT and abs(st["trimmed"]) >= lim, size, st


class DrawPool:
    """Pools the signed biases of every draw of a cell, per named tensor.

    ``assert_unbiased()`` judges each tensor's additive and gain biases over the draws with ``bias_fails``:
    |mean| >= bound/4 AND |t| > 3 AND |trimmed mean| >= bound/4.  Per-draw rounding noise averages away, a
    systematic defect does not; a tensor whose per-draw bias has heavy tails (one token with near-zero rms in front
    of a narrow RMSNorm dominates a gradient, CONTRACT_SUITE 9.1) can exceed bound/4 in its five-draw mean without
    the mean differing from 0 beyond its noise.

    Escalation: ``assert_unbiased(rerun=fn, seeds=[...])``.  When a tensor's mean exceeds bound/4 at the cell's
    default draw count, ``fn(seed) -> {name: (actual, expected, bound)}`` is called for every extra seed
    (``escalation_seeds``; 30 draws in total) and the verdict is taken on the full set.  A defect stays
    significant, a fluke does not.  Without ``rerun`` (a call site that cannot redraw) the size criterion alone
    decides, as before: the pool never passes a mean above bound/4 it cannot escalate.
    """

    def __init__(self):
        self._acc = {}

    def add(self, name, bound, additive, gain):
        self._acc.setdefault(name, [bound, [], []])
        self._acc[name][1].append(additive)
        self._acc[name][2].append(gain)

    def _judge(self, full_rule):
        """The (tensor, metric) pairs that fail: the size criterion alone, or the full rule (size, t, trimmed)."""
        bad = []
        for name, (bound, adds, gains) in self._acc.items():
            for what, vals in (("additive", adds), ("gain", gains)):
                fails, size, st = bias_fails(vals, bound)
                if fails if full_rule else size:
                    bad.append((name, what, bound, st, fails))
        return bad

    def assert_unbiased(self, rerun=None, seeds=()):
        escalated = False
        bad = self._judge(full_rule=False)
        if bad and rerun is not None and seeds:
            for seed in seeds:
                for name, (a, e, bound) in rerun(seed).items():
                    self.add(name, bound, *_signed_bias(a, e))
            escalated = True
            bad = self._judge(full_rule=True)
        if not bad:
            return
        lines = []
        for name, what, bound, st, fails in bad:
            lines.append(f"{name}: {what} pooled over {st['n']} draws mean {st['mean']:+.2e} "
                         f"(t {st['t']:+.1f}, sd {st['sd']:.2e}) median {st['median']:+.2e} trimmed {st['trimmed']:+.2e} "
                         f"limit {bound / 4:.2e} = bound/4")
        hint = ("\n  A rounding bias shrinks with the unit roundoff u (bf16 3.9e-3, fp16 4.9e-4, fp32 6e-8), a defect "
                "does not: rerun the cell in fp16 / fp32 -- a bias of the same size and sign there is a defect.")
        assert False, ("systematic 16-bit bias" + (" (stands after escalation to more draws)" if escalated else
                       " (no redraw available at this call site: size criterion only)") + ":\n  "
                       + "\n  ".join(lines) + (hint if escalated else ""))


def assert_dtype_close(actual, expected, dtype, *, depth, terms,
                       fp32_tol=None, arms=1, draws_axis=None, name="",
                       pool=None, u=None, k=1.0, per=()):
    """Score ``actual`` against ``expected`` under the dtype's policy.

    ``terms``: sum|terms| of ``expected`` (``terms_abs``); the relative
    metric is GUARDED (``assert_rel_close``): a unit whose reference is
    below ``k * u * ||terms||`` raises ``NearZeroReference``.  ``u``
    defaults to the unit roundoff of ``dtype`` (pass the MMA format's u
    for a tf32 fp32 cell).

    fp32/fp64: ``relative(actual, expected) < fp32_tol`` -- the caller's
    existing bound (``fp32_tol=None`` runs only the guard, for callers that
    keep their own fp32 assertion).

    bf16/fp16: relative Frobenius error < ``tol16(dtype, depth, arms=arms)``
    for the tensor (or for EVERY slice along ``draws_axis`` when the draws
    are stacked on one axis), AND each tensor's signed bias (additive and
    gain, see ``_signed_bias``) < bound/4 when it has at least 256 elements.
    With ``pool`` the biases are also pooled over the cell's draws (call
    ``pool.assert_unbiased()`` after the last draw).

    Returns the measured relative error (worst slice).
    """
    uu = unit_roundoff(dtype if u is None else u)
    terms = torch.as_tensor(terms).to(expected.device).expand_as(expected)
    if not is_sixteen_bit(dtype):
        a, r, s = _prep(actual, expected, terms, name)
        _guard(r, s, uu, k, per, name)
        err = _rel_frob(actual, expected)
        if fp32_tol is not None:
            assert err < fp32_tol, f"{name}: {err:.2e} >= {fp32_tol:.1e}"
        return err
    bound = tol16(dtype, depth, arms=arms)
    if draws_axis is None:
        slices = [(actual, expected, terms)]
    else:
        slices = list(zip(actual.unbind(draws_axis),
                          expected.unbind(draws_axis),
                          terms.unbind(draws_axis)))
    worst = 0.0
    for j, (a, e, s) in enumerate(slices):
        tag = name if draws_axis is None else f"{name}[draw {j}]"
        assert torch.isfinite(a.detach().float()).all(), f"{tag}: non-finite"
        _, r_, s_ = _prep(a, e, s, tag)
        _guard(r_, s_, uu, k, per, tag)
        err = _rel_frob(a, e)
        worst = max(worst, err)
        add, gain = _signed_bias(a, e)
        msg = (f"{tag}: {dtype} rel {err:.2e} vs bound {bound:.2e} "
               f"(u*sqrt({depth})*sqrt({arms}) clamped); bias additive "
               f"{add:+.2e} gain {gain:+.2e}")
        assert err < bound, msg
        if a.numel() >= _BIAS_MIN_NUMEL:
            assert abs(add) < bound / 4 and abs(gain) < bound / 4, (
                "systematic bias (limit bound/4): " + msg)
            if pool is not None:
                pool.add(name, bound, add, gain)
    return worst


def assert_sixteen_bit_close(actual, ref, terms, dtype, *, depth, c_terms,
                             arms=1, per=(), pool=None, name=""):
    """A 16-bit comparison under A7's 16-bit policy: `assert_dtype_close`
    (tol16(dtype, depth, arms), normwise relative, signed bias), GUARDED per
    `per` unit.  Where the guard refuses -- a unit whose reference is ~0
    against its own sum|terms| -- the prescription is the
    terms-scaled form, with the caller's derived `c_terms`, instead."""
    try:
        return assert_dtype_close(actual, ref, dtype, depth=depth, terms=terms,
                                  arms=arms, per=per, pool=pool, name=name)
    except NearZeroReference:
        return assert_close_terms(actual, ref, terms, dtype, c_terms, per=per,
                                  pool=pool, name=f"{name} [terms: ref ~ 0]")


# ---------------------------------------------------------------------------
# THE TWO-PART RULE (lifted from tools/ssd_check.py, one implementation for
# every family).  Per tensor it is the strictest of the repo's rules, so a family adopting it never loosens:
#   (i)  PER-TOKEN gate at the contraction-depth bound  u sqrt(depth) sqrt(arms)  (16-bit: clamped to [0.64, 6.4] eps);
#   (ii) SUM gate = per-token bound x kappa1 (kappa1 from the INDEPENDENT fp64 reference only), kappa1 > 10 marked
#        uninformative (held by the per-token gate and the signed-bias rules);
#   (iii) fp32 PREFIX ALLOWANCE 2 u32 (sqrt(T) kappa_D + sqrt(L) kappa_Theta) added at fp32 ieee;
#   (iv) INDEPENDENT-ARM FLOOR: a 16-bit / tf32 tensor above its bound passes iff the median over the draws of
#        kernel error / independent-arm error <= 1.25, never waived otherwise;
#   (v)  SIGNED BIAS: per-token pooled signed error at bound/4, sum-level t-test over >= 8 draws for kappa1 > 2
#        (or the pooled bound/4 rule), judged against the independent same-precision arm where one exists.
# `Scorer.draw(...)` takes the per-draw cells; the family supplies the reference, kappas, per-token errors, prefix
# conditioning and arm floors (SSD: tools/ssd_check `_SlabCapture` / `_PrefixCapture`).


# ---------------------------------------------------------------------------
#: Conditioning above which the SUM over tokens of a parameter gradient is uninformative.  kappa = sum|per-token terms| / |sum|:
#: the sum is a difference of terms kappa times larger than itself, so its relative error and its signed mean measure the
#: rounding of the cancelled terms (a bound scaled by kappa admits a coherent worst case no draw count can test), not the
#: arm under test.  Above it neither the normwise sum check nor the sum-level signed t-test gates; the per-token error and the
#: per-token signed bias (which see the terms before they cancel) are the gate.
KAPPA_UNINFORMATIVE = 10.0
#: the sum-level signed t-test (and only that) needs the sum to be informative: kappa at most KAPPA_UNINFORMATIVE
KAPPA_SUM_TTEST_MIN = 2.0
U32 = 2.0 ** -24
FLOOR_FACTOR = 1.25
#: fewest draws on which the median kernel/arm error ratio may FAIL a tensor.  The median of fewer draws is dominated by a
#: single heavy-tailed draw (B_bias at tf32: per-draw ratios 0.5 to 5.9, one draw red, the median of eight 1.3 to 1.5); a cell
#: with fewer draws escalates (`SsdPool.extra_seeds`) and an undecided verdict is reported, never failed.
FLOOR_MIN_DRAWS = 8


def prefix_allowance(kappa_d, kappa_theta, T, L):
    """2 u32 (sqrt(T) kappa_D + sqrt(L) kappa_Theta): see `_PrefixCapture`."""
    return 2 * U32 * (math.sqrt(T) * kappa_d + math.sqrt(L) * kappa_theta)



def accuracy_bound(u, depth, bf16, arms=2):
    """The per-element accuracy bound of the arithmetic (module docstring): 16-bit u sqrt(depth) sqrt(arms)
    clamped to [0.64, 6.4] eps (eps = 2u); fp32 IO u sqrt(depth) * 4 (floor 1e-6)."""
    eps = 2 * u
    return (min(max(u * math.sqrt(depth) * math.sqrt(arms), 0.64 * eps), 6.4 * eps)
            if bf16 else max(u * math.sqrt(depth) * 4, 1e-6))


class Scorer:
    """The two-part tolerance rule (module docstring) over a set of draws:
    `draw` scores one draw's tensors, `finish` applies the pooled signed-bias
    checks and prints the per-tensor table.  Shared with the contract suite
    (tests/_contract/adapters/ssd.py), so there is ONE implementation of the
    rule.  `fails` is the shared list of failure messages."""

    def __init__(self, B, L, bf16, no_grad_only=False, fails=None, pool_bias=True):
        self.BL, self.bf16, self.no_grad_only = B * L, bf16, no_grad_only
        self.pool_bias = pool_bias        # False: the draws are independent judgements (route-pair arms): no pooled signed-bias check
        self.floor_short = set()          # tensors whose floor verdict is undecided: median ratio > FLOOR_FACTOR on < FLOOR_MIN_DRAWS draws
        self._derived = []                # the failure messages `finish` added (re-derived by every `finish`)
        self.worst, self.signed, self.evidence = {}, {}, []
        self.fails = [] if fails is None else fails
        self.tols, self.failed_names, self.uninformative = {}, set(), set()
        self.kappas_by, self.tok_signed_by, self.tok_tol_by = {}, {}, {}
        self.pooled = {}         # name -> dict(norm=[ratios], tok=[ratios], over=[messages], arm=str): the floor verdict is pooled over the draws

    def draw(self, d, cells, *, T, u, kappas, tok_errs, prefix=None, floors=None):
        """cells: [(name, got, ref, depth)]; kappas / tok_errs from `_SlabCapture`.

        prefix = (kappa_D, kappa_Theta, L) from `_PrefixCapture`: at fp32 IEEE (u = u32) the allowance
        `prefix_allowance` is ADDED to the per-element bound of every tensor (the prefix conditioning is a property of
        the chunked form's fp32 prefix sums, not of the operand format).
        floors = a callable returning {name: (normwise error, per-token error or None), "__arm__": str} of an INDEPENDENT
        same-dtype arm against the same fp64 reference -- our general route where the cell is served by another route, else
        the fp32-internal torch backend (the record names which) -- called once per draw for the 16-bit / tf32 arms.  The
        verdict is POOLED over the draws (`finish`): a tensor that exceeded its bound in any draw passes iff the MEDIAN of the
        per-draw ratios kernel error / arm error is <= FLOOR_FACTOR; the max ratio is reported as a diagnostic; above that it is
        a DEFECT CANDIDATE and fails, never waived.

        Signed bias: the floor tuple may carry the arm's signed errors `(norm, tok, signed, tok_signed)`.  Against an
        INDEPENDENT SAME-PRECISION arm (official Mamba-3 at io, our general route where the cell is served by SISO / chunk-parallel) a pooled bias
        check that would fail passes iff |kernel pooled signed error| <= max(bound/4, 1.25 x |arm pooled signed error|) AND the signs agree (both
        numbers recorded).  Against the fp32-internal torch backend the bias checks are unchanged: a tensor flagged only there stays a defect
        candidate, never waived."""
        import torch
        bf16, fails = self.bf16, self.fails
        worst, signed, evidence = self.worst, self.signed, self.evidence
        tols, failed_names, uninformative = self.tols, self.failed_names, self.uninformative
        kappas_by, tok_signed_by, tok_tol_by = self.kappas_by, self.tok_signed_by, self.tok_tol_by
        allow = 0.0
        if prefix is not None and not bf16 and u == U32:
            allow = prefix_allowance(prefix[0], prefix[1], T, prefix[2])
        fl_all = floors() if (floors is not None and (bf16 or u != U32)) else None
        arm = fl_all.get("__arm__", "same-dtype arm") if fl_all else None

        def floor_of(nm):
            return None if fl_all is None else fl_all.get(nm)
        for name, ga, gb, depth in cells:
            is_param = (not self.no_grad_only) and depth == self.BL and name not in ("y", "y_nograd")
            kap = kappas.get(name, 1.0) if is_param else 1.0
            ga, gb = ga.double(), gb.double()
            if not torch.isfinite(ga).all():
                fails.append(f"draw {d}: {name} non-finite")
                continue
            err = float((ga - gb).norm() / gb.norm().clamp_min(1e-300))
            sgn = float(((ga - gb) * gb).sum() / (gb * gb).sum().clamp_min(1e-300))
            arms = 2 if bf16 else 1
            eps = 2 * u

            def bound(dep):
                return accuracy_bound(u, dep, bf16, arms)
            tol = bound(depth) + allow               # activations; params: the old bound (WARN only)
            warn = False
            row = dict(draw=d, tensor=name, err=err, signed=sgn, depth=depth)
            if is_param:
                tol_tok = bound(T) + allow           # the per-element bound of the arithmetic (+ the fp32 prefix allowance)
                tol_gate = tol_tok * kap             # coherent worst case of the sum
                te, tsg = tok_errs.get(name, (None, None))
                uninf = kap > KAPPA_UNINFORMATIVE
                row.update(tol_old=tol, tol_tok=tol_tok, kappa=kap, tol_sum=tol_gate, tok_err=te, tok_signed=tsg,
                           sum_check=("uninformative (cancellation-dominated)" if uninf else "gated"))
                if uninf:
                    uninformative.add(name)
                warn = err >= tol
                fl = floor_of(name)
                if fl is not None:
                    pl = self.pooled.setdefault(name, dict(norm=[], tok=[], over=[], arm=arm))
                    if te is not None and fl[1]:
                        pl["tok"].append(te / fl[1])
                    if len(fl) > 3 and fl[3] is not None and tsg is not None:
                        pl.setdefault("tsg_k", []).append(tsg)
                        pl.setdefault("tsg_a", []).append(fl[3])
                    if fl[1]:                                     # the arm's own per-token error against the per-token bound
                        pl.setdefault("arm_tok_out", []).append(fl[1] >= tol_tok)
                if te is not None and te >= tol_tok:
                    msg = (f"draw {d}: {name} PER-TOKEN gradient error {te:.3e} >= {tol_tok:.3e} "
                           f"(the arithmetic's per-element bound, T={T})")
                    if fl is not None and fl[1] and fl[1] > 0:
                        self.pooled[name]["over"].append(msg)       # judged pooled over the draws in `finish`
                    else:
                        failed_names.add(name)
                        fails.append(msg)
                tol_use = float("inf") if uninf else tol_gate     # per-token + bias check hold it
            else:
                row.update(tol=tol, kappa=1.0, tol_sum=tol)
                tol_use = tol
            row["warn_old_bound"] = bool(warn)
            evidence.append(row)
            prev = worst.get(name, (0.0, tol_use))
            worst[name] = (max(prev[0], err), max(prev[1], tol_use))
            signed.setdefault(name, []).append(sgn)
            kappas_by.setdefault(name, []).append(kap)
            if is_param and tsg is not None:
                tok_signed_by.setdefault(name, []).append(tsg)
                tok_tol_by[name] = tol_tok
            tols.setdefault(name, []).append(tol)    # the original (unscaled) bound
            fl = floor_of(name)
            if fl is not None and fl[0] > 0:
                pl = self.pooled.setdefault(name, dict(norm=[], tok=[], over=[], arm=arm))
                pl["norm"].append(err / fl[0])
                pl.setdefault("arm_norm_out", []).append(fl[0] >= tol_use)          # (inf for an uninformative sum: never out)
                if len(fl) > 2 and fl[2] is not None:
                    pl.setdefault("sgn_k", []).append(sgn)
                    pl.setdefault("sgn_a", []).append(fl[2])
            if err >= tol_use:
                msg = (f"draw {d}: {name} normwise {err:.3e} >= {tol_use:.3e} "
                       f"(depth {depth}" + (f", kappa {kap:.2f}" if kap > 1 else "") + ")")
                if fl is not None and fl[0] > 0:
                    self.pooled[name]["over"].append(msg)           # judged pooled over the draws in `finish`
                else:
                    failed_names.add(name)
                    fails.append(msg)

    def finish(self, say=print):
        """The pooled checks and the table; returns `fails`.  Idempotent: the messages of an earlier call are replaced, so a cell
        that escalates (more draws, `FLOOR_MIN_DRAWS`) is judged again on the full set."""
        bf16, fails = self.bf16, self.fails
        gone = set(map(id, self._derived))
        fails[:] = [f for f in fails if id(f) not in gone]
        n_before = len(fails)
        self.floor_short = set()
        worst, signed, evidence = self.worst, self.signed, self.evidence
        tols, uninformative = self.tols, self.uninformative
        failed_names = set(self.failed_names)                        # the draw-time failures; `finish` adds its own to the copy
        kappas_by, tok_signed_by, tok_tol_by = self.kappas_by, self.tok_signed_by, self.tok_tol_by
        for name, pl in sorted(self.pooled.items()):                # the pooled floor verdict (see `draw`)
            pl.pop("verdict", None)
            if not pl["over"]:
                continue
            ratios = sorted(pl["norm"] if pl["norm"] else pl["tok"])
            if len(pl["tok"]) and any("PER-TOKEN" in m for m in pl["over"]) and not any("normwise" in m for m in pl["over"]):
                ratios = sorted(pl["tok"])
            med = ratios[len(ratios) // 2]
            note = (f"pooled over {len(ratios)} draws: median kernel/arm ratio {med:.2f}, max {ratios[-1]:.2f} "
                    f"against {pl['arm']}")
            pl["verdict"] = note
            if med > FLOOR_FACTOR and len(ratios) < FLOOR_MIN_DRAWS:
                self.floor_short.add(name)                          # undecided: too few draws to fail on (rule 3)
                pl["verdict"] = note + f"; UNDECIDED, fewer than {FLOOR_MIN_DRAWS} draws"
            elif med > FLOOR_FACTOR:
                failed_names.add(name)
                fails.append(f"{name}: DEFECT CANDIDATE: {note} (> {FLOOR_FACTOR}); first exceedance: {pl['over'][0]}")

        def arm_bias_ok(name, kk, ka, bound4):
            """The pooled signed error of the kernel against an INDEPENDENT same-precision arm (see `draw`): (passes, note)."""
            pl = self.pooled.get(name)
            if not pl or not pl["arm"].startswith(("official", "the general route")):
                return False, ""
            k, a_ = pl.get(kk), pl.get(ka)
            if not k or not a_ or len(k) != len(a_):
                return False, ""
            mk, ma = sum(k) / len(k), sum(a_) / len(a_)
            ok = abs(mk) <= max(bound4, 1.25 * abs(ma)) and mk * ma >= 0
            return ok, f"kernel pooled signed {mk:+.3e} vs {pl['arm']} {ma:+.3e} (bound/4 {bound4:.2e}; {'agree' if ok else 'DISAGREE'})"

        def out_of_regime_ok(name, kk, ka, flags, bound4):
            """Rule 1.  The same-dtype arm's OWN error is above the tensor's bound in most draws: the fp64 oracle is outside its
            linear regime there (the input rounding enters at second order and shifts the oracle's own value), so a signed
            error against it is shared by every arm that rounds the inputs.  The kernel's bias is then the pooled signed
            kernel-minus-arm difference, held to bound/4.  (passes, note); (False, "") when the rule does not apply."""
            pl = self.pooled.get(name)
            if not pl:
                return False, ""
            fl_, k, a_ = pl.get(flags), pl.get(kk), pl.get(ka)
            if not fl_ or not k or not a_ or len(k) != len(a_) or 2 * sum(fl_) <= len(fl_):
                return False, ""
            diff = sum(x - y for x, y in zip(k, a_)) / len(k)
            ok = abs(diff) < bound4
            return ok, (f"arm's own error above the bound in {sum(fl_)} of {len(fl_)} draws (oracle outside its linear regime): "
                        f"kernel-minus-arm pooled signed {diff:+.3e} vs {pl['arm']} (bound/4 {bound4:.2e}; {'ok' if ok else 'EXCEEDS'})")
        for name, (err, tol) in sorted(worst.items()):
            mean = sum(signed[name]) / len(signed[name])
            tol_mean = sum(tols[name]) / len(tols[name])    # the original (unscaled) bound
            n_d = len(signed[name]) if self.pool_bias else 1
            old_flag = n_d > 1 and abs(mean) >= tol_mean / 4          # the old pooled rule (printed WARN)
            kmean = sum(kappas_by[name]) / len(kappas_by[name])
            ts = tok_signed_by.get(name)
            tok_mean = sum(ts) / len(ts) if ts else None
            bias_bad, bias_note = False, ""
            if ts and n_d > 1 and abs(tok_mean) >= tok_tol_by[name] / 4:     # (1) per-token signed bias
                okb, note_b = arm_bias_ok(name, "tsg_k", "tsg_a", tok_tol_by[name] / 4)
                if not okb:
                    ok2, n2 = out_of_regime_ok(name, "tsg_k", "tsg_a", "arm_tok_out", tok_tol_by[name] / 4)
                    okb, note_b = ok2, (n2 or note_b)
                if okb:
                    bias_note += "  | per-token bias matches the arm: " + note_b
                else:
                    bias_bad = True
                    fails.append(f"{name}: PER-TOKEN pooled signed error {tok_mean:.3e} >= bound/4 "
                                 f"{tok_tol_by[name] / 4:.3e} (a systematic bias of the per-token gradient)" + (f"; {note_b}" if note_b else ""))
            sum_uninformative = kmean > KAPPA_UNINFORMATIVE          # rule 2: the per-token check above is the gate
            t_test = kmean > KAPPA_SUM_TTEST_MIN and not sum_uninformative and n_d >= 8
            if sum_uninformative and (old_flag or n_d >= 8):
                bias_note += f"  sum-level signed check does not gate (kappa {kmean:.1f} > {KAPPA_UNINFORMATIVE:g})"
            if t_test:                                                 # (2) sum level: t-test
                sd = (sum((x - mean) ** 2 for x in signed[name]) / (n_d - 1)) ** 0.5
                se = sd / n_d ** 0.5
                t = abs(mean) / se if se > 0 else float("inf")
                pred = abs(tok_mean) * kmean if tok_mean is not None else tol_mean / 4
                sum_bad = abs(mean) > 3 * se and abs(mean) > pred
                bias_note = f"  t={t:.2f} (kappa {kmean:.1f}, per-token-predicted {pred:.1e})"
                if sum_bad:
                    okb, note_b = arm_bias_ok(name, "sgn_k", "sgn_a", max(pred, tol_mean / 4))
                    if not okb:
                        ok2, n2 = out_of_regime_ok(name, "sgn_k", "sgn_a", "arm_norm_out", tol_mean / 4)
                        okb, note_b = ok2, (n2 or note_b)
                    if okb:
                        bias_note += "  | sum bias matches the arm: " + note_b
                    else:
                        bias_bad = True
                        fails.append(f"{name}: sum-level signed error {mean:.3e} is {t:.1f} SE from 0 and above the "
                                     f"per-token-predicted {pred:.3e}" + (f"; {note_b}" if note_b else ""))
            elif old_flag and not sum_uninformative:                   # kappa <= 2, or too few draws
                okb, note_b = arm_bias_ok(name, "sgn_k", "sgn_a", tol_mean / 4)
                if not okb:
                    ok2, n2 = out_of_regime_ok(name, "sgn_k", "sgn_a", "arm_norm_out", tol_mean / 4)
                    okb, note_b = ok2, (n2 or note_b)
                if okb:
                    bias_note += "  | pooled bias matches the arm: " + note_b
                else:
                    bias_bad = True
                    fails.append(f"{name}: pooled signed error {mean:.3e} >= bound/4 {tol_mean / 4:.3e}" + (f"; {note_b}" if note_b else ""))
            if old_flag and (t_test or sum_uninformative):
                bias_note += f"  WARN: old pooled |signed| {abs(mean):.2e} >= bound/4 {tol_mean / 4:.2e} (not gating)"
            if name in self.pooled and self.pooled[name].get("verdict"):
                bias_note += "  | floor: " + self.pooled[name]["verdict"]
            flag = "FAIL" if name in failed_names or bias_bad else ("UNDECIDED" if name in self.floor_short else "ok")
            rows = [r for r in evidence if r["tensor"] == name]
            extra = ""
            if "tol_tok" in rows[0]:
                toks = [r["tok_err"] for r in rows if r.get("tok_err") is not None]
                extra = (f"  | per-token {max(toks):.2e} vs {rows[0]['tol_tok']:.2e}" if toks
                         else "  | per-token: not captured")
                extra += f"  kappa max {max(r['kappa'] for r in rows):.2f}"
                if name in uninformative:
                    extra += "  SUM CHECK UNINFORMATIVE (kappa > 10): per-token + bias check only"
                if any(r["warn_old_bound"] for r in rows):
                    extra += "  WARN: above the old sqrt(depth) bound"
            if tok_mean is not None:
                extra += f"  tok-signed {tok_mean:+.1e}"
            say(f"  {name:36s} err {err:9.3e}  bound {tol:8.2e}  signed {mean:+9.2e}  {flag}{extra}{bias_note}")
        self._derived = fails[n_before:]
        self.failed_names_final = failed_names
        return fails



# ---------------------------------------------------------------------------
# central finite differences against an fp64 Jacobian (gradcheck)

def fd_atol(depth, t_max, eps=1e-6):
    """Absolute bound of a central difference in fp64 against the analytic Jacobian (A7, the policy's fp64 entry).

    Each of the two evaluations of an output carries at most ``depth * u64 * |f|`` of rounding (``depth``: the length of the
    longest accumulation chain of the function: tokens of the recurrence + state / feature dot + chart products), the
    difference is divided by ``2 eps``: ``sqrt(2) depth u64 T / (2 eps)`` with ``T = max |f|``; the truncation of the central
    difference, ``eps^2 / 6`` times the third derivative, is carried as ``eps^2 T`` (O(1) inputs).  ``T`` is floored at 1 so a
    small output does not get a bound below the rounding of its O(1) inputs.  ``rtol`` is 0: the bound is absolute and
    already scales with the output."""
    return (math.sqrt(2.0) * depth * UNIT_ROUNDOFF["fp64"] / (2.0 * eps) + eps ** 2) * max(float(t_max), 1.0)


def gradcheck_fd(fn, args, depth, *, eps=1e-6):
    """``torch.autograd.gradcheck(fn, args, eps, atol=fd_atol(depth, max |fn(args)|), rtol=0)``; returns the atol used."""
    with torch.no_grad():
        out = fn(*args)
    outs = [o for o in pytree.tree_leaves(out) if torch.is_tensor(o) and o.numel()]
    t_max = max(float(o.abs().max()) for o in outs)
    atol = fd_atol(depth, t_max, eps)
    assert torch.autograd.gradcheck(fn, args, eps=eps, atol=atol, rtol=0), (depth, t_max, atol)
    return atol
