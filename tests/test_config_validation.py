"""Pins: range and type refusals shared by every family config and `Execution`, by the constructor and by
`from_dict` alike: d_conv >= 0 int; flag fields bool; positive ints for the dimension fields; 0 < dt_min <
dt_max; `Execution.kernel_options` a dict with str keys.  Each refusal is named, once per family.
"""

import pytest
from orbit_ssm import Execution, HybridMambaConfig, SSDMimoConfig


pytestmark = pytest.mark.paper1


_FAMILIES = [SSDMimoConfig, HybridMambaConfig]


_ids = [c.__name__ for c in _FAMILIES]


def _make(cls, how, **kw):
    base = dict(d_model=32)
    base.update(kw)
    return cls(**base) if how == "ctor" else cls.from_dict(base)


@pytest.mark.parametrize("how", ["ctor", "from_dict"])
@pytest.mark.parametrize("cls", _FAMILIES, ids=_ids)
def test_the_defaults_are_accepted(cls, how):
    """contract: the shared checks accept the family defaults (so every refusal below is the field's)."""
    _make(cls, how)


@pytest.mark.parametrize("how", ["ctor", "from_dict"])
@pytest.mark.parametrize("cls", _FAMILIES, ids=_ids)
@pytest.mark.parametrize("kw,exc,match", [
    (dict(d_conv=-1), ValueError, "d_conv must be a non-negative integer"),
    (dict(d_conv=2.5), (TypeError, ValueError), "d_conv"),
    (dict(d_conv=True), (TypeError, ValueError), "d_conv"),
    (dict(d_conv="4"), (TypeError, ValueError), "d_conv"),
    (dict(skip=1), TypeError, "skip"),
    (dict(out_norm="yes"), TypeError, "out_norm"),
    (dict(bc_conv=0), TypeError, "bc_conv"),
    (dict(d_state=0), ValueError, "d_state must be a positive integer"),
    (dict(d_state=-4), ValueError, "d_state must be a positive integer"),
    (dict(d_state=16.0), (TypeError, ValueError), "d_state"),
    (dict(d_inner=0), ValueError, "d_inner must be a positive integer"),
    (dict(n_heads=0), ValueError, "n_heads must be a positive integer"),
    (dict(d_head=-8), ValueError, "d_head must be a positive integer"),
    (dict(n_heads=2.0), (TypeError, ValueError), "n_heads"),
    (dict(expand=0.0), ValueError, "expand must be positive"),
    (dict(dt_min=0.0), ValueError, "0 < dt_min < dt_max"),
    (dict(dt_min=-1e-3), ValueError, "0 < dt_min < dt_max"),
    (dict(dt_min=1e-1, dt_max=1e-3), ValueError, "0 < dt_min < dt_max"),
    (dict(dt_min=1e-2, dt_max=1e-2), ValueError, "0 < dt_min < dt_max"),
    (dict(dt_max=float("inf")), ValueError, "0 < dt_min < dt_max"),
    (dict(dt_min="1e-3"), (TypeError, ValueError), "dt_min"),
])
def test_a_config_refuses_out_of_range_and_mistyped_fields(cls, how, kw, exc, match):
    """contract: constructor and `from_dict` refuse the same values, each naming the field: a flag that is
    not a bool is a TypeError; an integer field that is not a (positive / non-negative) integer is a
    ValueError from the constructor's range check and a TypeError from `from_dict`'s typed read."""
    with pytest.raises(exc, match=match):
        _make(cls, how, **kw)


@pytest.mark.parametrize("cls", _FAMILIES, ids=_ids)
def test_a_config_accepts_the_edges(cls):
    """contract: d_conv=0 (no conv) and an int dt bound are legal."""
    cfg = cls(d_model=32, d_conv=0, dt_min=1e-3, dt_max=1)
    assert cfg.d_conv == 0


@pytest.mark.parametrize("how", ["ctor", "from_dict"])
@pytest.mark.parametrize("kw,match", [
    (dict(kernel_options={1: True}), "kernel_options must be a dict with str keys"),
    (dict(kernel_options={"a": 1, 2: 3}), "kernel_options must be a dict with str keys"),
    (dict(checkpoint=1), "checkpoint"),
])
def test_execution_refuses_mistyped_fields(how, kw, match):
    """contract: `Execution` checks `checkpoint` (a bool) and the keys of `kernel_options` (str) in the
    constructor as `from_dict` does; other values are the families' named refusals."""
    with pytest.raises(TypeError, match=match):
        Execution(**kw) if how == "ctor" else Execution.from_dict(kw)


def test_execution_accepts_a_str_keyed_kernel_options_dict():
    """contract: the valid forms stay valid."""
    ex = Execution(backend="torch", chunk_size=16, checkpoint=False, kernel_options={"a": 1})
    assert ex.fields()["kernel_options"] == {"a": 1}
