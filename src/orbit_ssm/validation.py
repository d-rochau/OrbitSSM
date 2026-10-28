"""Validators shared by the layer families (no Torch import); one definition per rule."""
from __future__ import annotations

import math
from numbers import Integral

from . import families as _families


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def nonneg_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def check_dt_range(dt_min, dt_max, dt_init_floor=None) -> None:
    """``0 < dt_min < dt_max < inf`` and, when given, finite positive ``dt_init_floor``
    (dt is drawn log-uniformly in the range, ``layers.init.init_dt_bias``)."""
    for name, v in (("dt_min", dt_min), ("dt_max", dt_max), ("dt_init_floor", dt_init_floor)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise TypeError(f"{name} must be a number, got {v!r}")
    if not (0 < dt_min < dt_max) or not math.isfinite(dt_max):
        raise ValueError(f"require 0 < dt_min < dt_max < infinity, got dt_min={dt_min!r}, dt_max={dt_max!r}")
    if dt_init_floor is not None and (not math.isfinite(dt_init_floor) or dt_init_floor <= 0):
        raise ValueError("dt_init_floor must be finite and positive")


#: the MIMO ranks the kernels tile (R: write rank, O: read rank)
RANKS = (1, 2, 4, 8)


def check_rank(value, name: str) -> int:
    """A MIMO rank (``R`` or ``O``): a positive integer in `RANKS`."""
    value = positive_int(value, name)
    if value not in RANKS:
        raise ValueError(f"{name} must be one of {','.join(map(str, RANKS))}, got {value}")
    return value


#: Scan operand modes per family: (allowed values, meaning of None).  "io": operands in the IO
#: dtype (bf16 under autocast / for a bf16 module; for fp32 it is "tf32"); "tf32" / "ieee":
#: fp32 operands (TF32 tensor cores / plain fp32 FMA); "bf16_ops": opt-in bf16-operand timing
#: mode of the SSD scan.  HybridMamba has no precision knob.
PRECISIONS = {
    "ssd_mimo": (("ieee", "tf32", "io", "bf16_ops"), None),
    # the kernel entry `ssd_mimo_scan`; scan modes registered by kernel extensions (`families.register_scan_precision`)
    # are appended to its allowed values, a registered family's `precisions` add its own entry
    "ssd_mimo_scan": (("ieee", "tf32", "io", "bf16_ops"), None),
}


def _on_family(fam) -> None:
    if fam.precisions is not None:
        PRECISIONS[fam.name] = fam.precisions


def _on_scan_precision(sp) -> None:
    allowed, none_means = PRECISIONS["ssd_mimo_scan"]
    PRECISIONS["ssd_mimo_scan"] = (allowed + (sp.name,), none_means)


_families.on_register(_on_family)
_families.on_scan_precision(_on_scan_precision)


def check_precision(value, family: str, who: str | None = None) -> str:
    """Validate against `PRECISIONS[family]`; None resolves to the family default where it has one."""
    allowed, none_means = PRECISIONS[family]
    if value is None and none_means is not None:
        return none_means
    if value not in allowed:
        raise ValueError(f"{who or family}: precision must be one of {allowed}, got {value!r}")
    return value


def check_divides(name: str, value, what: str, of, error=ValueError) -> int:
    """``value`` is a positive divisor of ``of`` (``what`` names it)."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1 or of % value:
        raise error(f"{name}={value!r} must be a positive divisor of {what}={of}")
    return int(value)


#: the fused kernels' chunk floor (smem.py `chunk_admissible`)
CHUNK_FLOOR = 16


def valid_chunk_size(v, floor: int = CHUNK_FLOOR, pow2: bool = True) -> bool:
    """'auto' | 'measure' | a chunk length >= ``floor`` that is a power of two when ``pow2`` (defaults: the
    fused kernels' rule)."""
    if isinstance(v, str):
        return v in ("auto", "measure")
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return v >= floor and (not pow2 or (v & (v - 1)) == 0)
    return False


def check_chunk_size(v, who: str, *, measure: bool = True, floor: int = CHUNK_FLOOR, pow2: bool = True):
    """The one ``chunk_size`` rule of the scan families: 'auto', a power of two >= ``floor`` and
    (``measure=True``) 'measure'.  A family without the first-call measurement passes ``measure=False``
    ('measure' is then refused by name); a family whose torch route takes any chunk length passes
    ``floor=1, pow2=False`` (the fused kernels still need a power of two >= 16)."""
    if v == "measure" and not measure:
        raise ValueError(f"{who}: chunk_size='measure' (timing candidate chunks on the first call) is not "
                         f"supported by this family; use 'auto' or an integer chunk length")
    if not valid_chunk_size(v, floor, pow2):
        allowed = "'auto', 'measure' or" if measure else "'auto' or"
        what = f"a power-of-two >= {floor}" if pow2 else f"an integer >= {floor}"
        raise ValueError(f"{who}: chunk_size must be {allowed} {what}, got {v!r}")
    return v


# ── strict records: `from_dict` ──
def _type_ok(value, hint) -> bool:
    """Does ``value`` fit annotation ``hint`` (unions, None, Literal, builtin/generic containers;
    int fits float, bool fits only bool)?"""
    import types
    import typing
    if hint is typing.Any:
        return True
    if hint is None or hint is type(None):
        return value is None
    origin = typing.get_origin(hint)
    if origin is typing.Literal:
        return value in typing.get_args(hint)
    if origin is typing.Union or origin is getattr(types, "UnionType", ()):
        return any(_type_ok(value, a) for a in typing.get_args(hint))
    if origin is not None:
        hint = origin
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, Integral) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, hint)


def check_bool_fields(obj) -> None:
    """Every flag field (annotated ``bool`` or ``bool | None``) of dataclass instance ``obj`` is a bool or
    None, else ``TypeError`` naming the field (``0``, ``1`` and strings are not flags)."""
    import dataclasses
    import typing
    hints = typing.get_type_hints(type(obj))
    for f in dataclasses.fields(obj):
        hint = hints[f.name]
        if hint is bool or set(typing.get_args(hint)) == {bool, type(None)}:
            value = getattr(obj, f.name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{type(obj).__name__}: {f.name}={value!r} is a {type(value).__name__}, "
                                f"expected a bool")


def strict_from_dict(cls, d):
    """``cls(**d)`` for a dataclass, strictly: unknown keys and wrongly typed values raise
    ``TypeError`` naming the offender."""
    import dataclasses
    import typing
    if not isinstance(d, dict):
        raise TypeError(f"{cls.__name__}.from_dict: expected a dict, got {type(d).__name__}")
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(str(k) for k in d if k not in names)
    if unknown:
        raise TypeError(f"{cls.__name__}.from_dict: unknown key(s) {unknown}; "
                        f"valid keys: {', '.join(sorted(names))}")
    hints = typing.get_type_hints(cls)
    for key, value in d.items():
        if not _type_ok(value, hints[key]):
            raise TypeError(f"{cls.__name__}.from_dict: {key}={value!r} is a {type(value).__name__}, "
                            f"expected {hints[key]}")
    return cls(**d)


# Backends: "cuda" production kernels; "torch" parallel/chunked PyTorch; "sequential"
# token-by-token PyTorch recurrence; "auto" picks "cuda"/"torch" from the input's device.
BACKENDS = ("cuda", "torch", "sequential", "auto")

# Write rules: the quadrature of the forcing integral.  All share the exact transition
# exp(delta_t A_t) and differ only in the forcing:
#   "euler"      current endpoint only, Delta*B (phi truncated to I): Mamba-1/2.
#   "trapezoid"  Mamba-3's two-endpoint lambda-weighted rule.
#   "zoh"        the EXACT hold, Delta*phi(Delta A)*b(x_t), phi(Z) = Z^-1 (e^Z - I).
#                Not implemented: accepted by name and refused, so a request for exact ZOH
#                never silently resolves to the Euler truncation (the exact-hold theorem of Paper 1, Section 2).
WRITE_RULES = ("euler", "trapezoid", "zoh")
WRITE_RULES_IMPLEMENTED = ("euler", "trapezoid")
WRITE_RULE_DEFAULT = "trapezoid"

def resolve_write_rule(write_rule: str, who: str,
                       supported=WRITE_RULES_IMPLEMENTED,
                       field: str = "write_rule") -> str:
    """Validate a write-rule selector; "zoh" raises `NotImplementedError`, other unknowns `ValueError`."""
    if write_rule in supported:
        return write_rule
    if write_rule == "zoh":
        raise NotImplementedError(
            f"{who}: {field}='zoh' denotes the EXACT zero-order hold, "
            "forcing Delta*phi(Delta A)*b(x_t) with phi(Z) = Z^-1 (e^Z - I), "
            "and is NOT IMPLEMENTED.  It is accepted by name so that a "
            "request for exact ZOH is REFUSED instead of silently resolving "
            "to the truncation phi ~ I -- which is the Euler write, "
            f"{field}='euler'.  Implemented rules: {', '.join(supported)}")
    raise ValueError(
        f"{who}: unknown {field} {write_rule!r}; supported: "
        f"{', '.join(supported)} (plus 'zoh', which is named and refused "
        "because the exact hold is not implemented)")


def check_backend(backend: str, who: str, backends=BACKENDS) -> str:
    """Validate ``backend`` against a family's allowed set."""
    if backend not in backends:
        raise ValueError(
            f"{who}: backend must be one of: {', '.join(backends)}; "
            f"got {backend!r}")
    return backend
