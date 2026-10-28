"""Pins: `scan` / `scan_step` / `step` take their shared arguments under the same names, in the same order and
keyword-only in every family; family extras come after them, keyword-only.
"""

import inspect
import pytest
from orbit_ssm import HybridMambaSSM, SSDMimoSSM


pytestmark = pytest.mark.paper1


_FAMILIES = [SSDMimoSSM, HybridMambaSSM]


#: the canonical order of the arguments a family has (a family omits those it has no meaning for)
_SHARED = {
    "scan": ("gate_z", "sel_x", "tok_raw", "state", "return_state", "out_norm"),
    "scan_step": ("gate_z", "sel_x", "tok_raw"),
    "step": (),
}


_EXTRAS = {"theta_scale"}


@pytest.mark.parametrize("method", sorted(_SHARED))
@pytest.mark.parametrize("cls", _FAMILIES, ids=lambda c: c.__name__)
def test_shared_arguments_are_uniform(cls, method):
    """contract: the leading positional is the input (``x`` for scan, ``x_t`` for the per-token calls) then,
    for the per-token calls, ``state``; every other argument is keyword-only; shared ones appear in the
    canonical order; family extras follow them."""
    params = list(inspect.signature(getattr(cls, method)).parameters.values())[1:]      # drop self
    lead = ("x",) if method == "scan" else ("x_t", "state")
    assert tuple(p.name for p in params[:len(lead)]) == lead
    for p in params[:len(lead)]:
        assert p.kind is p.POSITIONAL_OR_KEYWORD
    rest = params[len(lead):]
    assert all(p.kind is p.KEYWORD_ONLY for p in rest), [p.name for p in rest]
    names = [p.name for p in rest]
    shared = [n for n in names if n in _SHARED[method]]
    assert shared == [n for n in _SHARED[method] if n in names]
    assert names == shared + [n for n in names if n not in _SHARED[method]]
    assert set(names) - set(_SHARED[method]) <= _EXTRAS
