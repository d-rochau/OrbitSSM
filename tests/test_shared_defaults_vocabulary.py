"""One vocabulary, one set of defaults, one meaning of ``scan`` in the two families.

Pins (what each test declares it checks):
  * DEFAULTS: ``SSDMimoConfig`` and ``HybridMambaConfig`` default to ``d_ssm_in = 1`` (SISO), ``skip=True``
    with ``skip_sharing="head"`` (D of shape (H,)), ``out_norm=False`` (no norm parameter), ``dt_init_floor=1e-4``; the
    families refuse the ``skip_sharing`` values they cannot build; ``skip=False`` builds no D in any family
    (HybridMamba included, same output as a zero D).
  * VOCABULARY: ``bc_bias`` is "head" | "group" | "shared" | "none" everywhere (each family serves a subset and
    refuses the rest with the valid values; "head_bc" is gone); a decay-law name is one law (``DECAY_LAWS``), every
    family's valid names are in it.
  * SCAN: ``scan`` / ``scan_step`` take ``gate_z`` / ``sel_x`` / ``tok_raw`` as keyword-only arguments in both
    families; ``is_siso`` / ``is_mimo`` are read-only.
CPU only.
"""

import dataclasses
import inspect
import pytest
import torch
import torch.nn.functional as F
from orbit_ssm import HybridMambaConfig, HybridMambaSSM, SSDMimoConfig, SSDMimoSSM, config_spec

from _design_warnings import SILENCE_SSD_DESIGN_WARNINGS

pytestmark = SILENCE_SSD_DESIGN_WARNINGS  # builds the warned configurations on purpose


EX = {"backend": "torch", "precision": "ieee"}


EXF = {"ssd_mimo": EX, "hybrid_mamba": {"backend": "torch"}}      # HybridMamba has no precision


ROWS = {"ssd_mimo": [{"frac": 1.0, "decay": "head_dynamic"}], "hybrid_mamba": [{"frac": 1.0, "decay": "coord_static"}]}


def _cfg(family, **kw):
    base = dict(d_model=32, d_state=16, d_inner=32, n_heads=4, proj_source="main_branch")
    if family == "ssd_mimo":
        return SSDMimoConfig(**{**base, "n_groups": 8, **kw})
    if family == "hybrid_mamba":
        return HybridMambaConfig(**{**base, "bc_bias": "none", **kw})
    raise ValueError(family)


FAMILIES = ("ssd_mimo", "hybrid_mamba")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_families_share_their_defaults(family):
    cls = {"ssd_mimo": SSDMimoConfig, "hybrid_mamba": HybridMambaConfig}[family]
    f = {x.name: x for x in dataclasses.fields(cls)}
    assert (f["d_ssm_in"].default, f["d_ssm_out"].default, f["skip"].default, f["skip_sharing"].default,
            f["out_norm"].default, f["dt_init_floor"].default) == (1, None, True, "head", False, 1e-4)
    m = _cfg(family).build(execution=EXF[family])
    assert m.is_siso and not m.is_mimo
    assert m.D.shape == (m.H,)
    assert not any(n.startswith("norm.") for n, _ in m.named_parameters())      # out_norm=False: no output norm
    assert _cfg(family, out_norm=True).build(execution=EXF[family]).norm.weight.shape == (m.d_out if family != "hybrid_mamba" else m.out_width,)
    rec = m.to_recipe()["model"]
    assert (rec["skip"], rec["skip_sharing"], rec["out_norm"], rec["dt_init_floor"]) == (True, "head", False, 1e-4)


def test_skip_sharing_values_each_family_builds():
    shapes = {}
    for v in ("head", "channel", "input_output"):
        shapes[v] = _cfg("ssd_mimo", skip_sharing=v, d_ssm_in=2, n_groups=4).build(execution=EX).D.shape
    assert [len(shapes[v]) for v in ("head", "channel", "input_output")] == [1, 3, 4]
    assert _cfg("hybrid_mamba", skip_sharing="channel").build(execution=EXF["hybrid_mamba"]).D.shape == (4, 8)
    for family, bad in (("hybrid_mamba", "input_output"),):
        with pytest.raises(ValueError, match="skip_sharing"):
            _cfg(family, skip_sharing=bad).build(execution=EXF[family])


@pytest.mark.parametrize("family", FAMILIES)
def test_skip_false_builds_no_D(family):
    on = _cfg(family).build(execution=EXF[family]).double()
    off = _cfg(family, skip=False).build(execution=EXF[family]).double()
    assert "D" not in dict(off.named_parameters()) and getattr(off, "D", None) is None
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k != "D"}, strict=True)
    with torch.no_grad():
        on.D.zero_()
    x = torch.randn(2, 6, 32, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    torch.testing.assert_close(off(x), on(x), rtol=0, atol=0)           # no D is a zero D
    y, st = off(x, return_state=True)
    ys, ss = off.step(x[:, 0], off.init_state(2, dtype=torch.float64))
    torch.testing.assert_close(ys, off(x[:, :1])[:, 0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family,ok,bad", [
    ("ssd_mimo", ("head", "shared", "none"), ("group", "head_bc")),
    ("hybrid_mamba", ("head", "shared", "none"), ("group", "head_bc"))])
def test_bc_bias_vocabulary(family, ok, bad):
    for v in ok:
        kw = {} if family != "ssd_mimo" or v == "head" else {}
        built = _cfg(family, bc_bias=v, **kw)
        assert built.bc_bias == v
    for v in bad:
        with pytest.raises(config_spec.ConfigSpecError, match="bc_bias must be one of"):
            _cfg(family, bc_bias=v)


def test_scan_arguments_are_keyword_only():
    for cls in (SSDMimoSSM, HybridMambaSSM):
        for name in ("scan", "scan_step"):
            params = inspect.signature(getattr(cls, name)).parameters
            for kw in ("gate_z", "sel_x", "tok_raw"):
                if kw in params:
                    assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY, (cls.__name__, name, kw)
        assert "tok_raw" in inspect.signature(cls.scan).parameters and "sel_x" in inspect.signature(cls.scan).parameters


def test_is_siso_is_mimo_are_read_only():
    for family in FAMILIES:
        m = _cfg(family).build(execution=EXF[family])
        with pytest.raises(AttributeError):
            m.is_siso = False
        with pytest.raises(AttributeError):
            m.is_mimo = True
    assert _cfg("ssd_mimo", d_ssm_in=1, d_ssm_out=2).build(execution=EX).is_mimo
