"""Public geometry and pre-freeze recipe roundtrips, independent of CUDA."""
import copy
import io
import json

import pytest
import torch

from orbit_ssm.configuration import resolve_geometry, resolve_dt_rank
from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig, SSDMimoSSM
from orbit_ssm.ssm.hybrid_mamba import (
    HybridMambaConfig, HybridMambaSSM, HybridMambaConfig,
)


FAMILIES = [(SSDMimoSSM, SSDMimoConfig), (HybridMambaSSM, HybridMambaConfig)]
pytestmark = [pytest.mark.paper1, pytest.mark.contract]


@pytest.mark.parametrize('selection,heads,expected', [
    (2048, 2048, 64), (2048, 32, None), (2048, 66, None), (2048, 67, 64),
    (1024, 68, None), (1024, 69, 64),
])
def test_delta_adaptive_uses_actual_selection_width(selection, heads, expected):
    assert resolve_dt_rank('adaptive', d_sel=selection, heads=heads,
                           d_model=1024) == expected


def test_delta_rank_overrides_and_tie():
    kw = dict(d_sel=16, heads=16, d_model=128)
    assert resolve_dt_rank('adaptive', **kw) is None  # equal multiply counts
    assert resolve_dt_rank('auto', **kw) == 8
    assert resolve_dt_rank(3, **kw) == 3
    assert resolve_dt_rank(None, **kw) is None
    assert resolve_dt_rank('adaptive', d_sel=64, heads=64) == 4
    assert resolve_dt_rank('adaptive', force_dt_dense=True, **kw) is None
    for invalid in (0, -1, True, 2.5, 'unknown'):
        with pytest.raises(ValueError):
            resolve_dt_rank(invalid, **kw)
    with pytest.raises(ValueError, match='conflicts'):
        resolve_dt_rank(3, force_dt_dense=True, **kw)
    with pytest.raises(ValueError, match='requires d_model'):
        resolve_dt_rank('auto', d_sel=16, heads=16)


@pytest.mark.parametrize('source', ['main_branch', 'layer_input'])
@pytest.mark.parametrize('heads,force_dense,expected', [(32, False, 1),
                                                     (1, False, None),
                                                     (32, True, None)])
def test_delta_resolved_architecture_roundtrip_and_gradients(source, heads, force_dense, expected):
    kw = dict(d_inner=32, d_state=8, n_heads=heads,
              proj_source=source, force_dt_dense=force_dense,
              row_blocks=[{'frac': 1., 'decay': 'coord_static'}], bc_bias='none')
    execution = {'backend': 'torch'}
    torch.manual_seed(814)
    module = HybridMambaSSM(HybridMambaConfig(d_model=16, **kw), execution=execution).double()
    assert module.dt_rank == expected
    document = json.loads(json.dumps(module.to_recipe()))
    assert document['model']['dt_rank'] == expected
    restored = HybridMambaSSM.from_recipe(document).double()
    restored.load_state_dict(module.state_dict())
    configured = HybridMambaConfig(**kw).build(16, execution=execution)
    assert configured.to_recipe() == document
    x = torch.randn(1, 3, 32, dtype=torch.float64, requires_grad=True)
    selection = torch.randn(1, 3, 16, dtype=torch.float64, requires_grad=True)
    y = module.scan(x, sel_x=selection if source == 'layer_input' else None)
    yr = restored.scan(x, sel_x=selection if source == 'layer_input' else None)
    torch.testing.assert_close(y, yr, rtol=0, atol=0)
    params = [module.dt_proj.weight]
    if module.dt_low is not None:
        params.append(module.dt_low.weight)
    grads = torch.autograd.grad(y.square().sum(), params)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def layer_kwargs(cls, **kwargs):
    """Keywords of the full layer around the old bare core (width 64, d_model 32)."""
    out = dict(d_model=32, d_inner=64, d_state=32, proj_source="main_branch",
               execution={"backend": "torch"})
    if cls is SSDMimoSSM:
        out.update(d_ssm_in=2)      # these tests are written at the R = 2 geometry (SSD's default was formerly 2)
    if cls is HybridMambaSSM:
        # bc_bias="none": the pre-spec HM default (no B/C bias)
        out.update(row_blocks=[{"frac": 1.0, "decay": "coord_static"}], bc_bias="none")
    return {**out, **kwargs}


def config_kwargs(kw):
    """The architecture part of layer keywords (what a typed config holds)."""
    return {k: v for k, v in kw.items() if k not in ("d_model", "expand", "execution")}


_CONFIG_OF = {SSDMimoSSM: SSDMimoConfig, HybridMambaSSM: HybridMambaConfig}


def build_layer(core, kw):
    """The layer of the keywords `layer_kwargs` builds: its config (the architecture part) plus `execution=`."""
    return core(_CONFIG_OF[core](**{k: v for k, v in kw.items() if k != "execution"}), execution=kw.get("execution"))


def build_config(config, kw):
    return config(**config_kwargs(kw)).build(kw["d_model"], execution=kw.get("execution"))


@pytest.mark.parametrize("core,config", FAMILIES)
@pytest.mark.parametrize("geometry", [
    {"n_heads": 2}, {"d_head": 32}, {"n_heads": 2, "d_head": 32},
])
def test_constructor_dataclass_geometry_and_initialization(core, config, geometry):
    kw = layer_kwargs(core, **geometry)
    torch.manual_seed(99)
    direct = build_layer(core, kw)
    torch.manual_seed(99)
    configured = build_config(config, kw)
    assert (direct.H, direct.P) == (2, 32)
    assert direct.to_recipe() == configured.to_recipe()
    assert direct.state_dict().keys() == configured.state_dict().keys()
    for name, value in direct.state_dict().items():
        torch.testing.assert_close(value, configured.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("core,config", FAMILIES)
@pytest.mark.parametrize("geometry", [
    {"n_heads": 2, "d_head": 16},
    {"n_heads": 3}, {"d_head": 3}, {"n_heads": 0}, {"n_heads": True}, {"d_head": 32.5},
])
def test_conflicting_or_invalid_geometry_fails_on_both_surfaces(core, config, geometry):
    kw = layer_kwargs(core, **geometry)
    with pytest.raises(ValueError):
        build_layer(core, kw)
    with pytest.raises(ValueError):
        config(**config_kwargs(kw))


@pytest.mark.parametrize("core,config", FAMILIES)
@pytest.mark.parametrize("alias", ["heads", "H", "head_subsystems", "P", "head_dim", "headdim"]) 
def test_retired_geometry_spellings_are_unknown_keywords(core, config, alias):
    """One spelling per concept: `n_heads` / `d_head` only."""
    with pytest.raises(TypeError):
        build_layer(core, layer_kwargs(core, **{alias: 2}))
    with pytest.raises(TypeError):
        config(**config_kwargs(layer_kwargs(core, **{alias: 2})))


def test_ssd_groups_and_rank_resolve_from_n_heads():
    m = build_layer(SSDMimoSSM, layer_kwargs(SSDMimoSSM, n_heads=4, d_ssm_in=2))
    assert (m.H, m.P, m.G) == (4, 16, 8)
    for kwargs in ({"n_heads": 4, "n_groups": 16}, {"d_head": 16, "n_groups": 16},
                   {"d_inner": 30, "n_heads": 2, "d_ssm_in": 2}):
        with pytest.raises(ValueError):
            build_layer(SSDMimoSSM, layer_kwargs(SSDMimoSSM, **kwargs))
    # The common helper never imposes SSD's divisibility by R.
    assert resolve_geometry(30, n_heads=2) == (2, 15)


@pytest.mark.parametrize("core,config", FAMILIES)
@pytest.mark.parametrize("rotated", [False, True])
def test_json_recipe_roundtrip_preserves_parameters_outputs_and_gradients(core, config, rotated):
    kw = layer_kwargs(core, n_heads=2)
    if core is SSDMimoSSM:
        kw.update(d_ssm_in=2, d_ssm_out=4, row_blocks=[{"frac": 1.0, "decay": "head_dynamic",
                                   "complex": rotated}])
    elif rotated:
        # a complex flat head_dynamic block is served with the per-head
        # bias and no learned rotation extras only (one global rotation)
        kw.update(row_blocks=[{"rows": 16, "decay": "coord_dynamic", "complex": True},
                        {"rows": 16, "decay": "head_dynamic", "complex": True}],
                  bc_bias="head", learn_theta_freq=False,
                  use_theta_h=False)
    original = build_layer(core, kw).double()
    document = json.loads(json.dumps(original.to_recipe()))
    assert document["semantics_version"] == "unfrozen"
    assert "backend" not in document["model"]
    assert not {"heads", "H", "P", "head_subsystems", "head_dim"} & document["model"].keys()
    assert {"n_heads", "d_head"} <= document["model"].keys()
    restored = core.from_recipe(document).double()
    restored.load_state_dict(original.state_dict(), strict=True)
    assert restored.to_recipe() == document
    x = torch.randn(1, 3, 64, dtype=torch.float64, requires_grad=True)
    y, h = original.scan(x, return_state=True)
    h = h.hidden_state
    y2, h2 = restored.scan(x, return_state=True)
    h2 = h2.hidden_state
    torch.testing.assert_close(y, y2, rtol=0, atol=0)
    torch.testing.assert_close(h, h2, rtol=0, atol=0)
    g = torch.autograd.grad(y.square().sum() + h.square().sum(), x)[0]
    g2 = torch.autograd.grad(y2.square().sum() + h2.square().sum(), x)[0]
    torch.testing.assert_close(g, g2, rtol=0, atol=0)


@pytest.mark.parametrize("core,config", FAMILIES)
def test_config_document_rejects_missing_unknown_or_wrong_family_fields(core, config):
    model = build_layer(core, layer_kwargs(core, n_heads=2))
    document = model.to_recipe()
    cases = []
    for key, value in (("schema", "future"), ("family", "other"),
                       ("semantics_version", "frozen"), ("extra", 3)):
        other = copy.deepcopy(document)
        other[key] = value
        cases.append(other)
    missing = copy.deepcopy(document)
    del missing["model"]["write_rule"]
    cases.append(missing)
    for other in cases:
        with pytest.raises(ValueError):
            core.from_recipe(other)
    # Callers cannot accidentally mutate the model's stored recipe.
    document["model"]["n_heads"] = 999
    assert model.to_recipe()["model"]["n_heads"] == 2


@pytest.mark.parametrize("mixer", [SSDMimoSSM, HybridMambaSSM])
def test_layer_takes_n_heads_and_d_head_without_injecting_conflicting_defaults(mixer):
    model = mixer(_CONFIG_OF[mixer](d_model=32, d_state=32, d_conv=0, n_heads=2, d_head=32),
                  execution={"backend": "torch"})
    assert (model.H, model.P) == (2, 32)
    with pytest.raises(ValueError, match="n_heads\\*d_head"):
        mixer(_CONFIG_OF[mixer](d_model=32, d_state=32, d_conv=0, n_heads=2, d_head=16))


def test_mh_refuses_shared_fields_that_would_otherwise_be_ignored():
    cfg = HybridMambaConfig(d_inner=64, n_heads=2, n_groups=2)
    with pytest.raises(NotImplementedError, match="n_groups"):
        cfg.layer_kwargs()


@pytest.mark.parametrize("field,value", [("theta_per_head", True)])
def test_mh_refuses_inapplicable_spec_fields_by_name(field, value):
    """rows/rotation/maps spec: a rotation field needs a complex block -- it raises
    ConfigSpecError naming the field at construction.  (The retired `decay` /
    `rope_fraction` fields are now plain unknown keywords.)"""
    from orbit_ssm.config_spec import ConfigSpecError
    with pytest.raises(ConfigSpecError, match=field):
        HybridMambaConfig(d_inner=64, n_heads=2, **{field: value}).layer_kwargs()


@pytest.mark.parametrize("field,value", [("decay", "fix"), ("rope_fraction", 0.5)])
def test_mh_config_retired_spec_fields_are_unknown_keywords(field, value):
    with pytest.raises(TypeError):
        HybridMambaConfig(d_inner=64, n_heads=2, **{field: value})


@pytest.mark.parametrize("write_rule,use_trap", [
    (None, True), ("trapezoid", True), ("euler", False),
])
def test_mh_config_maps_exactly_the_implemented_write_rules_onto_use_trap(
        write_rule, use_trap):
    """Pins HybridMambaConfig's write_rule -> use_trap map (None is the shared default "trapezoid"; the two
    implemented rules map to the bool; the kwarg reaches the SSM."""
    cfg = HybridMambaConfig(d_inner=64, n_heads=2, write_rule=write_rule,
                            proj_source="main_branch")
    assert cfg.layer_kwargs()["write_rule"] == write_rule
    assert cfg.build(64).use_trap is use_trap


@pytest.mark.parametrize("write_rule,exc,match", [
    ("zoh", NotImplementedError, "write_rule='zoh'.*EXACT"),
    ("bogus", ValueError, "unknown write_rule 'bogus'"),
])
def test_mh_config_refuses_zoh_by_name_and_every_other_write_rule(
        write_rule, exc, match):
    """Pins the refusal side of W1: "zoh" names the EXACT hold, which is
    not implemented, and must never resolve to the trapezoid (the old
    `write_rule != "euler"` did, for every string); typos raise too, at
    construction.  (The per-spelling "RETIRED" messages are gone.)"""
    with pytest.raises(exc, match=match):
        HybridMambaConfig(d_inner=64, n_heads=2, write_rule=write_rule)


def test_mh_dataclass_exposes_constructor_controls_and_resolves_initialization():
    kw = layer_kwargs(HybridMambaSSM, n_heads=2, d_model=33, dt_rank="auto",
                      dt_init="constant", bc_bias="shared",
                      b_gate=True, c_gate=True, bc_gate_bias=True,
                      bc_gate_act="sigmoid", bc_gate_pos="post", norm_eps=2e-5,
                      learnable_h_init=True,
                      execution={"backend": "torch", "kernel_options": {"mem": {"block_l": 16}}})
    direct = build_layer(HybridMambaSSM, kw)
    configured = build_config(HybridMambaConfig, kw)
    assert direct.to_recipe() == configured.to_recipe()
    assert direct.to_recipe()["model"]["dt_rank"] == 3
    assert direct.to_recipe()["model"]["bc_bias"] == "shared"


@pytest.mark.parametrize("core,config", FAMILIES)
def test_fresh_recipe_weight_and_adamw_resume_matches_uninterrupted_training(core, config):
    """Current-code roundtrip, without a claim about older/future checkpoints."""
    torch.manual_seed(121)
    kw = layer_kwargs(core, n_heads=2)
    if core is SSDMimoSSM:
        kw["r_mix_dyn"] = True
    model = build_layer(core, kw).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, foreach=False)
    inputs = [torch.randn(2, 4, 64, dtype=torch.float64) for _ in range(2)]

    def update(module, opt, x):
        opt.zero_grad(set_to_none=True)
        y, hidden = module.scan(x, return_state=True)
        hidden = hidden.hidden_state
        loss = y.square().mean() + hidden.square().mean()
        loss.backward()
        opt.step()
        return loss.detach()

    update(model, optimizer, inputs[0])
    if core is SSDMimoSSM:
        assert model.r_mix.grad.abs().sum() > 0
        assert model.r_mix_proj.weight.grad.abs().sum() > 0
    buffer = io.BytesIO()
    torch.save({"config": model.to_recipe(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = core.from_recipe(saved["config"]).double()
    restored.load_state_dict(saved["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.002, foreach=False)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    loss = update(model, optimizer, inputs[1])
    resumed_loss = update(restored, resumed_optimizer, inputs[1])
    torch.testing.assert_close(loss, resumed_loss, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    uninterrupted = optimizer.state_dict()
    resumed = resumed_optimizer.state_dict()
    assert uninterrupted["param_groups"] == resumed["param_groups"]
    for param, values in uninterrupted["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, resumed["state"][param][name], rtol=0, atol=0)
