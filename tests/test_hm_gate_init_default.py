"""Pins the HM gate-init default: gates, when enabled,
are one_plus / pre / kaiming_uniform (gate
form evidence).  What is pinned:

* the constructor and `HybridMambaConfig` default `gate_init="kaiming_uniform"`;
  bc_gate_act / bc_gate_pos resolve to one_plus / pre;
* a default-gated core's gB/gC weights ARE nn.Linear's default draw (same seed,
  same construction order -> bit-identical to an explicit "kaiming_uniform"), not
  zero, and every weight lies in the kaiming bound 1/sqrt(fan_in);
* the resolved recipe records gate_init="kaiming_uniform" (fresh prod: resolved,
  never implicit) and round-trips through from_recipe; an explicit "zero" too;
* SSD-MIMO's gB/gC stay zero-init (deliberate divergence, not changed here).
"""
import json
import math

import torch

from orbit_ssm.ssm.hybrid_mamba import HybridMambaConfig, HybridMambaSSM, _HybridMambaScan
from orbit_ssm import SSDMimoConfig

ROWS = [dict(rows=8, decay="coord_static", complex=False), dict(rows=8, decay="coord_static", complex=True)]


def _core(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(row_blocks=ROWS, proj_source="main_branch", bc_conv=False, b_gate=True, c_gate=True,
                bc_bias="none")
    base.update(kw)
    return HybridMambaSSM(HybridMambaConfig(d_model=16, d_state=16, d_inner=32, n_heads=2,
                          **base), execution={"backend": "torch"})


def test_defaults_are_the_ruled_gate_form():
    import inspect
    assert inspect.signature(_HybridMambaScan.__init__).parameters["gate_init"].default == "kaiming_uniform"
    assert HybridMambaConfig.__dataclass_fields__["gate_init"].default == "kaiming_uniform"
    core = _core()
    assert core.gate_init == "kaiming_uniform"
    model = core.to_recipe()["model"]
    assert model["bc_gate_act"] == "one_plus" and model["bc_gate_pos"] == "pre"


def test_default_gate_weights_are_the_kaiming_draw():
    dflt, expl, zero = _core(seed=7), _core(seed=7, gate_init="kaiming_uniform"), _core(seed=7, gate_init="zero")
    for n in ("gB_proj", "gC_proj"):
        w = getattr(dflt, n).weight
        torch.testing.assert_close(w, getattr(expl, n).weight, rtol=0, atol=0)
        assert (getattr(zero, n).weight == 0).all()
        fan_in = w.shape[1]
        assert w.abs().max() <= 1 / math.sqrt(fan_in) and w.abs().min() >= 0
        assert w.std() > 0.3 / math.sqrt(fan_in)              # a real draw (uniform std = 0.577/sqrt(fan_in))


def test_recipe_records_resolved_gate_init():
    for init in ("kaiming_uniform", "zero"):
        kw = {} if init == "kaiming_uniform" else dict(gate_init=init)
        doc = json.loads(json.dumps(_core(**kw).to_recipe()))
        assert doc["model"]["gate_init"] == init
        assert HybridMambaSSM.from_recipe(doc).gate_init == init
    cfg = HybridMambaConfig(d_inner=32, d_state=16, n_heads=2, row_blocks=tuple(ROWS), b_gate=True, c_gate=True,
                            proj_source="main_branch", bc_bias="none")
    assert cfg.layer_kwargs()["gate_init"] == "kaiming_uniform"


def test_ssd_mimo_gates_stay_zero_init():
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, _SSDMimoScan, SSDMimoConfig
    import inspect
    assert "gate_init" not in inspect.signature(_SSDMimoScan.__init__).parameters
    torch.manual_seed(0)
    m = SSDMimoSSM(SSDMimoConfig(d_model=32, d_state=16, d_inner=64, n_heads=2, d_ssm_in=2, d_ssm_out=2, b_gate=True,
                   c_gate=True, proj_source="main_branch"), execution={"backend": "torch"})
    for n in ("gB_proj", "gC_proj"):
        assert (getattr(m, n).weight == 0).all()
