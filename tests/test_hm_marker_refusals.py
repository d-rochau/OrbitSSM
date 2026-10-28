"""Pins: a HybridMamba load never changes the module -- the checkpoint's `proj_source_code` / `a_act_code` /
`dyn_act_code` markers must equal the module's resolved values (explicit or default), else ValueError naming
both values and the field to rebuild with; matching markers load silently; a dict with the weights but
without a marker is an older layout (the strict-format ValueError naming the checkpoint-upgrade tooling); a
partial dict and unknown codes behave as stated.  Old dicts load through
the upgrade tool's own tests.
"""

import warnings
import pytest
import torch
from orbit_ssm.ssm.hybrid_mamba import HybridMambaConfig, HybridMambaSSM


pytestmark = pytest.mark.paper1


_ROWS = [dict(rows=8, decay="coord_static"), dict(rows=8, decay="coord_dynamic"),
         dict(rows=16, decay="head_dynamic", complex=True)]


_KW = dict(d_model=16, d_state=32, d_inner=32, n_heads=4, row_blocks=_ROWS,
           learn_theta_freq=False, use_theta_h=False)


def _core(**kw):
    return HybridMambaSSM(HybridMambaConfig(**dict(_KW, **kw)), execution={"backend": "torch"})


def test_proj_source_none_resolves_to_the_documented_default():
    """contract: `proj_source=None` resolves to the documented default and is recorded resolved."""
    m = _core()
    assert m.proj_source == "layer_input"
    assert m.to_recipe()["model"]["proj_source"] == "layer_input"


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default"])
def test_proj_source_mismatch_raises_and_names_both_values(explicit):
    """contract: a declared source differing from the resolved one raises whether the module's value was
    explicit or the default, naming both and `proj_source=<checkpoint's>`; nothing is rebuilt."""
    other = "main_branch"
    ckpt = _core(proj_source=other).state_dict()
    dst = _core(proj_source="layer_input") if explicit else _core()
    params = [id(p) for p in dst.parameters()]
    with pytest.raises(ValueError, match=r"proj_source='main_branch'.*resolves proj_source='layer_input'"
                                         r".*rebuild it with proj_source='main_branch'"):
        dst.load_state_dict(ckpt)
    assert [id(p) for p in dst.parameters()] == params and dst.proj_source == "layer_input"


@pytest.mark.parametrize("field,value,default", [("a_act", "softplus", "heavy_tail"),
                                                 ("dyn_act", "exp", "heavy_tail")])
@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default"])
def test_act_marker_mismatch_raises_and_names_both_values(field, value, default, explicit):
    """contract: the same for the |A| activation markers: a differing declared activation raises naming
    both and `<field>=<checkpoint's>`, explicit or default."""
    ckpt = _core(proj_source="main_branch", **{field: value}).state_dict()
    dst = _core(proj_source="main_branch", **({field: default} if explicit else {}))
    assert getattr(dst, field) == default
    with pytest.raises(ValueError, match=rf"{field}='{value}'.*resolves {field}='{default}'.*"
                                         rf"rebuild it with {field}='{value}'"):
        dst.load_state_dict(ckpt)
    assert getattr(dst, field) == default


def test_matching_markers_load_silently_and_round_trip():
    """contract: matching markers load strictly without a warning and the module computes the source's
    function bit for bit (same code path, same weights; A7)."""
    kw = dict(proj_source="main_branch", a_act="softplus", dyn_act="exp")
    torch.manual_seed(0)
    src = _core(**kw)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    dst = _core(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dst.load_state_dict(src.state_dict(), strict=True)
    x = torch.randn(2, 9, 32, generator=torch.Generator().manual_seed(1))
    assert torch.equal(src.scan(x), dst.scan(x))


def test_unknown_codes_are_refused_and_a_partial_dict_changes_nothing():
    """contract: an unknown marker code raises by name; a dict with neither the markers nor the weights
    they describe loads under strict=False and reports the markers missing."""
    sd = _core().state_dict()
    for key, code in (("proj_source_code", 7), ("a_act_code", 9), ("dyn_act_code", 7)):
        with pytest.raises(ValueError, match=f"{key}={code} is not a known"):
            _core().load_state_dict(dict(sd, **{key: torch.tensor(code)}))
    m = _core()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = m.load_state_dict({"D": m.D.detach().clone()}, strict=False)
    assert {"proj_source_code", "a_act_code", "dyn_act_code"} <= set(res.missing_keys)
    assert m.proj_source == "layer_input"
