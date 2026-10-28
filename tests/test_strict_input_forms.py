"""Strict input forms of the execution and config entry points.

What these cells pin (CPU, construction and fp32 torch-route forwards on tiny layers):
  contract  `Execution.from_dict` and `<Family>Config.from_dict` raise TypeError on an unknown key (naming the valid
            keys) and on a wrongly typed value; an `Execution` and its dict are the SAME input at every entry point
            (layer `execution=`, `CausalLM(execution=)`, `set_execution`, the context manager); the keyword spellings
            (`set_execution(backend=...)`, `model.execution(backend=...)`) are a TypeError.
  layers    a layer takes ONLY the config (the family config or its strict dict) and `execution=`; every architecture
            keyword is a TypeError, and the config is the single place of the options (keyword-only dataclass).
  equivalence classes (a): an object and its dict build bit-identical state (same resolved recipe, same outputs).
  row_blocks  a list of `RowBlock` and of its strict dicts is the SAME input (same recipe, same weights); an unknown
            key / mistyped value / non-block item is a TypeError; rows and frac together are refused.
  width     one rule in all families: width from `d_inner` alone, `expand` alone (default 2.0) or both in
            agreement (a contradiction raises); the recipe records the resolved `d_inner` and `expand` as given; heads
            from n_heads or d_head or both (both must multiply to the width).  `expand` and `d_inner` are config fields.
  CausalLM  `layers` is one config / dict (then `n_layers` is required) or a sequence of them (any Sequence;
            `n_layers` derived, a mismatch raises); a dict carries `"type"`; `d_model` is optional and taken from the
            configs (all must agree; given twice it must be equal; given nowhere it is an error); `d_mlp` is one int or
            one entry per layer.  There is no `ssm_type`, `config=` or `**mixer_options`.
  targeting `lm.ssm_layers[i]` addresses one layer for `set_execution` / `execution(...)`; `layers=` is a TypeError.
  graphs    a captured graph notices an execution change (`execution_version`, `GraphedStep.stale`).
  removed   the nested `maps=` / `rotation=` inputs and the `bc_norm` shorthand are unknown keywords.
"""

from collections.abc import Sequence
import pytest
import torch
from orbit_ssm import CausalLM, Execution, HybridMambaConfig, HybridMambaSSM, RowBlock, SSDMimoConfig, SSDMimoSSM
from orbit_ssm.execution import execution_version


CONFIGS = [
    (SSDMimoConfig, dict(d_inner=32, d_head=8, d_state=16, d_ssm_in=1), SSDMimoSSM),
    (HybridMambaConfig, dict(d_inner=32, d_head=8, d_state=16), HybridMambaSSM),
]


IDS = ["ssd_mimo", "hybrid_mamba"]


def _x(d=16, seed=0):
    return torch.randn(2, 8, d, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- Execution.from_dict
def test_execution_from_dict_builds_the_object():
    d = {"backend": "torch", "precision": "ieee", "chunk_size": 16, "checkpoint": True,
         "kernel_options": {}}
    assert Execution.from_dict(d) == Execution(**d)
    assert Execution.from_dict({}) == Execution()


def test_execution_from_dict_refuses_unknown_keys_and_names_the_valid_ones():
    with pytest.raises(TypeError, match=r"unknown key\(s\) \['bakend'\].*backend.*chunk_size.*kernel_options"):
        Execution.from_dict({"bakend": "torch"})


@pytest.mark.parametrize("bad", [{"backend": 1}, {"precision": 2.0}, {"chunk_size": 1.5},
                                 {"checkpoint": "yes"}, {"kernel_options": [1]}, {"chunk_size": True}])
def test_execution_from_dict_refuses_wrong_types(bad):
    with pytest.raises(TypeError, match=r"from_dict"):
        Execution.from_dict(bad)


def test_execution_from_dict_refuses_a_non_dict():
    with pytest.raises(TypeError, match="expected a dict"):
        Execution.from_dict(Execution())
    assert not hasattr(Execution, "coerce")


# ---------------------------------------------------------------- <Family>Config.from_dict
@pytest.mark.parametrize("cfg_type, kw, layer", CONFIGS, ids=IDS)
def test_config_from_dict_equals_the_constructor(cfg_type, kw, layer):
    assert cfg_type.from_dict(kw) == cfg_type(**kw)


@pytest.mark.parametrize("cfg_type, kw, layer", CONFIGS, ids=IDS)
def test_config_from_dict_refuses_unknown_keys_and_wrong_types(cfg_type, kw, layer):
    with pytest.raises(TypeError, match=r"unknown key\(s\) \['no_such_field'\].*valid keys: .*d_inner"):
        cfg_type.from_dict(dict(kw, no_such_field=1))
    with pytest.raises(TypeError, match="d_inner"):
        cfg_type.from_dict(dict(kw, d_inner="32"))
    with pytest.raises(TypeError, match="expected a dict"):
        cfg_type.from_dict([("d_inner", 32)])


@pytest.mark.parametrize("cfg_type, kw, layer", CONFIGS, ids=IDS)
def test_config_dict_equals_config_object_in_the_constructor(cfg_type, kw, layer):
    ex = {"backend": "torch"}
    a = layer(config=cfg_type(**kw), execution=ex)
    b = layer(config=dict(kw), execution=ex)
    assert a.to_recipe() == b.to_recipe()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    # same init draws (same seed): the init is a function of the architecture only
    torch.manual_seed(0); a = layer(config=cfg_type(**kw), execution=ex)
    torch.manual_seed(0); b = layer(config=dict(kw), execution=ex)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


def test_causal_lm_config_dict_equals_config_object():
    kw = dict(d_inner=32, d_head=8, d_state=16, d_ssm_in=1, d_model=16)
    torch.manual_seed(0); a = CausalLM(vocab_size=32, layers=SSDMimoConfig(**kw), n_layers=2, execution={"backend": "torch"})
    torch.manual_seed(0); b = CausalLM(vocab_size=32, layers=dict(kw, type="ssd_mimo"), n_layers=2, execution={"backend": "torch"})
    ids = torch.arange(8)[None] % 32
    assert torch.equal(a(ids), b(ids))


# ---------------------------------------------------------------- execution entry points
def _layer():
    torch.manual_seed(0)
    return SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1), execution={"backend": "torch"})


def test_set_execution_object_equals_dict():
    a, b = _layer(), _layer()
    a.set_execution(Execution(backend="sequential", chunk_size=16))
    b.set_execution({"backend": "sequential", "chunk_size": 16})
    assert a.to_recipe() == b.to_recipe()
    x = _x()
    assert torch.equal(a(x), b(x))


def test_context_manager_object_equals_dict_and_restores():
    a = _layer()
    before = a.to_recipe()["execution"]
    x = _x()
    with a.execution(Execution(backend="sequential")):
        ya = a(x)
        inner = a.to_recipe()["execution"]
    with a.execution({"backend": "sequential"}):
        yb = a(x)
    assert torch.equal(ya, yb) and inner["backend"] == "sequential"
    assert a.to_recipe()["execution"] == before


def test_constructor_and_causal_lm_execution_object_equals_dict():
    torch.manual_seed(0); a = SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1), execution=Execution(backend="torch", chunk_size=16))
    torch.manual_seed(0); b = SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1), execution={"backend": "torch", "chunk_size": 16})
    assert a.to_recipe() == b.to_recipe()
    torch.manual_seed(0); m1 = CausalLM(vocab_size=32, layers=dict(type="ssd_mimo", d_state=16, d_head=8, d_ssm_in=1), n_layers=2, d_model=16, execution=Execution(backend="torch"))
    torch.manual_seed(0); m2 = CausalLM(vocab_size=32, layers=dict(type="ssd_mimo", d_state=16, d_head=8, d_ssm_in=1), n_layers=2, d_model=16, execution={"backend": "torch"})
    ids = torch.arange(8)[None] % 32
    assert torch.equal(m1(ids), m2(ids))
    m2.ssm_layers[1].set_execution({"backend": "sequential"})
    assert m2.backbone.layers[0].mixer.to_recipe()["execution"]["backend"] == "torch"
    assert m2.backbone.layers[1].mixer.to_recipe()["execution"]["backend"] == "sequential"


def test_keyword_forms_are_a_type_error():
    m = _layer()
    with pytest.raises(TypeError):
        m.set_execution(backend="torch")
    with pytest.raises(TypeError):
        m.set_execution({"backend": "torch"}, precision="ieee")
    with pytest.raises(TypeError):
        with m.execution(backend="torch"):
            pass
    with pytest.raises(TypeError):
        m.set_execution()
    with pytest.raises(TypeError):
        m.set_execution(["torch"])
    with pytest.raises(TypeError, match="unknown key"):
        m.set_execution({"bakend": "torch"})
    with pytest.raises(TypeError, match="execution must be"):
        SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1), execution="torch")


# ---------------------------------------------------------------- row_blocks: RowBlock or its strict dict
D = "head_dynamic"


_ROWS = [dict(rows=8, decay=D, complex=True), dict(rows=8, decay=D)]


_OBJ = [RowBlock(rows=8, decay=D, complex=True), RowBlock(rows=8, decay=D)]


def _ssd(rows, **kw):
    torch.manual_seed(0)
    return SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1, row_blocks=rows, **kw), execution={"backend": "torch"})


def test_row_blocks_objects_equal_their_dicts():
    a, b = _ssd(_OBJ), _ssd(_ROWS)
    assert a.to_recipe() == b.to_recipe()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert torch.equal(a(_x()), b(_x()))
    # mixed items are the same input too, and the config field takes the same forms
    assert _ssd([_OBJ[0], _ROWS[1]]).to_recipe() == a.to_recipe()
    assert SSDMimoConfig(d_inner=32, d_head=8, d_state=16, d_ssm_in=1, row_blocks=_OBJ).build(16).to_recipe()["model"][
        "row_blocks"] == a.to_recipe()["model"]["row_blocks"]
    assert RowBlock.from_dict(_ROWS[0]) == _OBJ[0]


@pytest.mark.parametrize("bad", [[dict(rows=8, decay=D, bogus=1), dict(rows=8, decay=D)],
                                 [dict(rows="8", decay=D), dict(rows=8, decay=D)],
                                 [dict(rows=8, decay=D, complex=1), dict(rows=8, decay=D)],
                                 [[8, D], dict(rows=8, decay=D)], ["head_dynamic"], [None]])
def test_row_blocks_refuse_unknown_keys_wrong_types_and_other_items(bad):
    with pytest.raises(TypeError):
        _ssd(bad)


def test_row_block_rows_and_frac_are_exclusive():
    with pytest.raises(ValueError, match="'rows' or 'frac', not both"):
        _ssd([dict(rows=16, frac=1.0, decay=D)])
    with pytest.raises(ValueError, match="needs 'rows' or 'frac'"):
        _ssd([dict(decay=D)])
    assert _ssd([dict(frac=1.0, decay=D)]).to_recipe() == _ssd([dict(rows=16, decay=D)]).to_recipe()


def test_row_blocks_of_every_family_take_the_objects():
    torch.manual_seed(0)
    hm = HybridMambaSSM(HybridMambaConfig(d_model=16, d_state=32, d_head=8, row_blocks=[RowBlock(frac=0.5, decay="coord_static"), RowBlock(frac=0.5, decay=D)]), execution={"backend": "torch"})
    assert [b["decay"] for b in hm.to_recipe()["model"]["row_blocks"]] == ["coord_static", D]


def test_the_nested_and_shorthand_inputs_are_gone():
    for bad in (dict(maps={"b_norm": True}), dict(rotation={"use_theta_h": True}), dict(bc_norm=False), dict(rows=_ROWS),
                dict(use_g_b=True), dict(bias_gran="none"), dict(headdim=8), dict(R=1)):
        with pytest.raises(TypeError, match="unexpected keyword"):
            SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1, **bad)
        with pytest.raises(TypeError, match="unknown key"):
            SSDMimoSSM(dict(d_model=16, d_state=16, d_head=8, d_ssm_in=1, **bad))


# ---------------------------------------------------------------- one width rule in all families
FAMILIES = [(SSDMimoSSM, dict(d_ssm_in=1)), (HybridMambaSSM, {})]


FIDS = ["ssd_mimo", "hybrid_mamba"]


_CONFIG_OF = {SSDMimoSSM: SSDMimoConfig, HybridMambaSSM: HybridMambaConfig}


def _family_layer(cls, extra, d_model=16, **kw):
    return cls(_CONFIG_OF[cls](d_model=d_model, d_state=16, **extra, **kw), execution={"backend": "torch"})


@pytest.mark.parametrize("cls, extra", FAMILIES, ids=FIDS)
def test_width_comes_from_expand_by_default_in_every_family(cls, extra):
    m = _family_layer(cls, extra, d_head=8)
    assert (m.d_inner, m.H, m.P) == (32, 4, 8)                      # expand = 2.0
    assert _family_layer(cls, extra, d_head=8, expand=3.0).d_inner == 48


@pytest.mark.parametrize("cls, extra", FAMILIES, ids=FIDS)
def test_width_d_inner_alone_expand_alone_both_agree_both_conflict(cls, extra):
    # d_inner alone: that width, whatever d_model * 2 would be
    m = _family_layer(cls, extra, d_inner=24, d_head=8)
    assert (m.d_inner, m.H) == (24, 3)
    # expand alone, and neither (expand 2.0)
    assert _family_layer(cls, extra, expand=1.5, d_head=8).d_inner == 24
    assert _family_layer(cls, extra, d_head=8).d_inner == 32
    # both given and agreeing
    assert _family_layer(cls, extra, expand=1.5, d_inner=24, d_head=8).d_inner == 24
    assert _family_layer(cls, extra, expand=2.0, d_inner=32, d_head=8).d_inner == 32
    # both given and in conflict
    with pytest.raises(ValueError, match="conflicts with int"):
        _family_layer(cls, extra, expand=2.0, d_inner=24, d_head=8)
    with pytest.raises(ValueError, match="conflicts with int"):
        _family_layer(cls, extra, expand=1.5, d_inner=32, d_head=8)


@pytest.mark.parametrize("cls, extra", FAMILIES, ids=FIDS)
def test_recipe_records_resolved_d_inner_and_expand_as_given(cls, extra):
    for kw, expand in ((dict(), None), (dict(d_inner=24), None), (dict(expand=1.5), 1.5),
                       (dict(expand=1.5, d_inner=24), 1.5)):
        m = _family_layer(cls, extra, d_head=8, **kw)
        rec = m.to_recipe()["model"]
        assert rec["expand"] == expand and rec["d_inner"] == m.d_inner
        rebuilt = cls.from_recipe(m.to_recipe())
        assert rebuilt.d_inner == m.d_inner and rebuilt.to_recipe() == m.to_recipe()


@pytest.mark.parametrize("cls, extra", FAMILIES, ids=FIDS)
def test_heads_from_n_heads_or_d_head_or_both(cls, extra):
    assert (_family_layer(cls, extra, n_heads=4).P, _family_layer(cls, extra, d_head=8).H) == (8, 4)
    assert _family_layer(cls, extra, n_heads=4, d_head=8).H == 4
    with pytest.raises(ValueError, match="n_heads"):
        _family_layer(cls, extra, n_heads=4, d_head=4)                       # 16 != 32
    with pytest.raises(ValueError, match="positive divisor"):
        _family_layer(cls, extra, n_heads=5)
    with pytest.raises(ValueError, match="positive divisor"):
        _family_layer(cls, extra, d_head=5)


# ---------------------------------------------------------------- layers take only config= and execution=
def _cfg(layer):
    return {SSDMimoSSM: SSDMimoConfig(d_model=16, d_state=16, d_head=8, d_ssm_in=1),
            HybridMambaSSM: HybridMambaConfig(d_model=16, d_state=16, d_head=8)}[layer]


@pytest.mark.parametrize("layer", [SSDMimoSSM, HybridMambaSSM], ids=IDS)
def test_layers_take_only_a_config_and_execution(layer):
    cfg = _cfg(layer)
    ex = {"backend": "torch"}
    torch.manual_seed(0); a = layer(cfg, execution=ex)
    torch.manual_seed(0); b = layer(config=cfg, execution=ex)
    torch.manual_seed(0); c = layer(dict(dataclass_fields(cfg)), execution=ex)
    assert a.to_recipe() == b.to_recipe() == c.to_recipe()
    # every architecture keyword (and the removed leading positional d_model) is a TypeError
    for bad in (dict(d_state=16), dict(expand=2.0), dict(d_inner=32), dict(n_heads=2), dict(d_model=16), dict(d_conv=0),
                dict(skip=True), dict(backend="torch")):
        with pytest.raises(TypeError):
            layer(cfg, **bad)
    with pytest.raises(TypeError):
        layer(16, cfg)
    with pytest.raises(TypeError):
        layer(16, d_state=16)
    with pytest.raises(TypeError):
        layer(execution=ex)                                    # the config is required
    with pytest.raises(TypeError, match="expected"):
        layer([1])
    # config.build(d_model) is the same layer as layer(config.replace(d_model=...))
    bare = cfg.replace(d_model=None, d_inner=32)
    assert layer(cfg.replace(d_model=24, d_inner=32)).in_proj.in_features == 24
    assert bare.build(24).in_proj.in_features == 24
    assert bare.build(24, execution=ex).backend == "torch"


def dataclass_fields(cfg):
    import dataclasses
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


@pytest.mark.parametrize("cfg_type, kw, layer", CONFIGS, ids=IDS)
def test_config_has_expand_and_d_inner_fields_and_no_zero_default(cfg_type, kw, layer):
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(cfg_type)}
    assert fields["expand"].default is None and fields["d_inner"].default is None
    assert fields["d_model"].default is None
    assert all(f.kw_only for f in fields.values())            # keyword-only: no positional form
    with pytest.raises(TypeError):
        cfg_type(32)
    # the rule is the layers': contradiction raises as soon as it is knowable
    with pytest.raises(ValueError, match="conflicts with int"):
        cfg_type(**dict(kw, d_model=16, expand=2.0, d_inner=24, d_head=8))
    # a layer needs a width: d_model, or d_inner without expand (the layer is as wide as its inner width)
    with pytest.raises(TypeError, match="d_model is not set"):
        layer(cfg_type(**dict(kw, d_model=None, d_inner=None, expand=2.0, d_head=8)))
    assert layer(cfg_type(**dict(kw, d_model=None))).in_proj.in_features == kw["d_inner"]


def test_config_docstrings_document_every_field():
    import dataclasses, re
    for cfg_type in (SSDMimoConfig, HybridMambaConfig):
        doc = cfg_type.__doc__
        documented = set()
        for m in re.finditer(r"^    ([a-z_0-9, ]+?)(?: :|$)", doc, re.M):
            documented |= {n.strip() for n in m.group(1).split(",")}
        missing = [f.name for f in dataclasses.fields(cfg_type) if f.name not in documented]
        assert not missing, (cfg_type.__name__, missing)


def test_renamed_options_old_spellings_are_unknown():
    for old in (dict(d_skip=False), dict(d_skip_sharing="head")):
        with pytest.raises(TypeError, match="unexpected keyword"):
            SSDMimoConfig(d_model=16, d_state=16, d_head=8, **old)
    with pytest.raises(TypeError, match="unexpected keyword"):
        HybridMambaConfig(d_model=16, d_state=16, d_head=8, d_skip_mode="head")
    # the new names
    s = SSDMimoSSM(SSDMimoConfig(d_model=16, d_state=16, d_head=8, skip=True, skip_sharing="head"))
    assert s.skip and s.skip_sharing == "head" and s.D.shape == (s.H,)
    h = HybridMambaSSM(HybridMambaConfig(d_model=16, d_state=16, d_head=8, skip_sharing="channel"))
    assert h.D.shape == (h.H, h.P)


# ---------------------------------------------------------------- CausalLM: layers, n_layers, d_model, d_mlp
_S = dict(d_state=16, d_head=8, d_ssm_in=1)


_EX = {"backend": "torch"}


def _blocks(lm):
    return [type(b.mixer).__name__ for b in lm.backbone.layers]


def test_layers_single_config_dict_or_sequence_of_them():
    cfg = SSDMimoConfig(d_model=16, **_S)
    # a single config / dict: n_layers is required and every layer is alike
    one = CausalLM(vocab_size=32, layers=cfg, n_layers=3, execution=_EX)
    two = CausalLM(vocab_size=32, layers=dict(type="ssd_mimo", d_model=16, **_S), n_layers=3, execution=_EX)
    assert _blocks(one) == _blocks(two) == ["SSDMimoSSM"] * 3
    assert one.to_recipe() == two.to_recipe()
    with pytest.raises(TypeError, match="n_layers is required"):
        CausalLM(vocab_size=32, layers=cfg)
    # any Sequence, one entry per layer; n_layers derived
    class Seq(Sequence):
        def __init__(self, *a): self.a = a
        def __getitem__(self, i): return self.a[i]
        def __len__(self): return len(self.a)
    entries = [cfg, dict(type="hybrid_mamba", d_state=16, d_head=8)]
    for seq in (entries, tuple(entries), Seq(*entries)):
        lm = CausalLM(vocab_size=32, layers=seq, d_model=16, execution=_EX)
        assert _blocks(lm) == ["SSDMimoSSM", "HybridMambaSSM"] and lm.n_layers == 2
    # the recipe round-trips the heterogeneous stack
    assert CausalLM.from_recipe(lm.to_recipe()).to_recipe() == lm.to_recipe()
    # n_layers given with a sequence must match
    assert CausalLM(vocab_size=32, layers=entries, n_layers=2, d_model=16, execution=_EX).n_layers == 2
    with pytest.raises(ValueError, match="does not match"):
        CausalLM(vocab_size=32, layers=entries, n_layers=3, d_model=16)
    with pytest.raises(ValueError, match="must not be empty"):
        CausalLM(vocab_size=32, layers=[], d_model=16)


def test_layers_dict_needs_a_type_and_is_strict():
    with pytest.raises(TypeError, match="needs a 'type' key"):
        CausalLM(vocab_size=32, layers=dict(d_state=16, d_head=8), n_layers=1, d_model=16)
    with pytest.raises(ValueError, match="unknown type 'nope'"):
        CausalLM(vocab_size=32, layers=dict(type="nope"), n_layers=1, d_model=16)
    with pytest.raises(TypeError, match=r"unknown key\(s\) \['no_such_option'\]"):
        CausalLM(vocab_size=32, layers=dict(type="hybrid_mamba", no_such_option=1), n_layers=1, d_model=16)
    with pytest.raises(TypeError, match="d_state"):
        CausalLM(vocab_size=32, layers=dict(type="hybrid_mamba", d_state="16"), n_layers=1, d_model=16)
    # the removed arguments
    cfg = SSDMimoConfig(d_model=16, **_S)
    for removed in (dict(ssm_type="ssd_mimo"), dict(config=cfg), dict(d_state=16), dict(layer_config=cfg)):
        with pytest.raises(TypeError):
            CausalLM(vocab_size=32, layers=cfg, n_layers=1, **removed)
    with pytest.raises(TypeError):
        CausalLM(32, 16, 1, "ssd_mimo")                        # the old positional (d_model, n_layers, ssm_type) form


def test_registered_foreign_mixer_takes_its_options_from_the_dict():
    seen = {}

    class Toy(torch.nn.Module):
        def __init__(self, d_model, scale=1.0):
            super().__init__()
            seen.update(d_model=d_model, scale=scale)
            self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, state=None, return_state=False):
            return self.out_proj(x)

    CausalLM.register_mixer("toy_cfg_test", Toy)
    lm = CausalLM(vocab_size=32, layers=[dict(type="toy_cfg_test", scale=3.0), dict(type="hybrid_mamba", d_state=16, d_head=8)],
                  d_model=16, execution=_EX)
    assert seen == dict(d_model=16, scale=3.0) and isinstance(lm.backbone.layers[0].mixer, Toy)


def test_d_model_comes_from_the_configs_or_the_argument_and_they_must_agree():
    cfg = HybridMambaConfig(d_model=16, d_state=16, d_head=8)
    assert CausalLM(vocab_size=32, layers=cfg, n_layers=1).d_model == 16                                  # from the configs
    assert CausalLM(vocab_size=32, layers=cfg, n_layers=1, d_model=16).d_model == 16                      # given twice, equal
    assert CausalLM(vocab_size=32, layers=cfg.replace(d_model=None), n_layers=1, d_model=24).d_model == 24   # from the argument, filled into the config
    lm = CausalLM(vocab_size=32, layers=[cfg.replace(d_model=None, d_inner=48), cfg.replace(d_model=None)], d_model=24)
    assert lm.backbone.layers[0].mixer.d_model == 24 and lm.backbone.layers[0].mixer.d_inner == 48
    with pytest.raises(ValueError, match="conflicts"):
        CausalLM(vocab_size=32, layers=cfg, n_layers=1, d_model=24)
    with pytest.raises(ValueError, match="disagree"):
        CausalLM(vocab_size=32, layers=[cfg, cfg.replace(d_model=24)])
    with pytest.raises(ValueError, match="d_model is not set"):
        CausalLM(vocab_size=32, layers=cfg.replace(d_model=None), n_layers=1)
    with pytest.raises(ValueError, match="d_model is not set"):
        CausalLM(vocab_size=32, layers=dict(type="hybrid_mamba", d_state=16, d_head=8), n_layers=1)
    # a dict carries d_model like a config does
    assert CausalLM(vocab_size=32, layers=dict(type="hybrid_mamba", d_model=16, d_state=16, d_head=8), n_layers=1).d_model == 16


def test_d_mlp_is_one_int_or_one_entry_per_layer():
    cfg = HybridMambaConfig(d_model=16, d_state=16, d_head=8)
    mlp = lambda lm: [b.mlp is not None for b in lm.backbone.layers]
    assert mlp(CausalLM(vocab_size=32, layers=cfg, n_layers=3)) == [False] * 3
    assert mlp(CausalLM(vocab_size=32, layers=cfg, n_layers=3, d_mlp=32)) == [True] * 3
    lm = CausalLM(vocab_size=32, layers=cfg, n_layers=3, d_mlp=[0, 32, 24])
    assert mlp(lm) == [False, True, True] and [b.d_mlp for b in lm.backbone.layers] == [0, 32, 24]
    assert lm.backbone.n_residuals == 1 + 2 + 2
    assert mlp(CausalLM(vocab_size=32, layers=[cfg, cfg], d_mlp=(16, 0))) == [True, False]
    with pytest.raises(ValueError, match="2 entries for 3 layers"):
        CausalLM(vocab_size=32, layers=cfg, n_layers=3, d_mlp=[0, 32])
    with pytest.raises(ValueError, match="d_mlp"):
        CausalLM(vocab_size=32, layers=cfg, n_layers=1, d_mlp=-1)
    with pytest.raises(ValueError, match="d_mlp"):
        CausalLM(vocab_size=32, layers=cfg, n_layers=1, d_mlp=[True])
    # the recipe keeps the spec as given and round-trips
    assert lm.to_recipe()["model"]["d_mlp"] == [0, 32, 24]
    assert CausalLM(vocab_size=32, layers=cfg, n_layers=2, d_mlp=8).to_recipe()["model"]["d_mlp"] == 8
    assert CausalLM.from_recipe(lm.to_recipe()).to_recipe() == lm.to_recipe()


# ---------------------------------------------------------------- execution targeting and CUDA-graph staleness
def _lm3():
    torch.manual_seed(0)
    return CausalLM(vocab_size=32, layers=dict(type="ssd_mimo", d_model=16, **_S), n_layers=3, execution=_EX)


def test_ssm_layers_address_one_layer_and_layers_keyword_is_gone():
    lm = _lm3()
    assert isinstance(lm.ssm_layers, tuple) and len(lm.ssm_layers) == 3
    assert all(a is b.mixer for a, b in zip(lm.ssm_layers, lm.backbone.layers))
    assert lm.backbone.ssm_layers == lm.ssm_layers
    with pytest.raises(AttributeError):
        lm.ssm_layers = ()                                      # read-only
    backends = lambda: [m.to_recipe()["execution"]["backend"] for m in lm.ssm_layers]
    lm.ssm_layers[1].set_execution({"backend": "sequential"})
    assert backends() == ["torch", "sequential", "torch"]
    with lm.ssm_layers[2].execution({"backend": "sequential"}):
        assert backends() == ["torch", "sequential", "sequential"]
    assert backends() == ["torch", "sequential", "torch"]
    with lm.execution({"backend": "torch"}):
        assert backends() == ["torch"] * 3
    assert backends() == ["torch", "sequential", "torch"]
    for call in (lambda: lm.set_execution({"backend": "torch"}, layers=[1]),
                 lambda: lm.execution({"backend": "torch"}, layers=[1]),
                 lambda: lm.ssm_layers[0].set_execution({"backend": "torch"}, layers=[0])):
        with pytest.raises(TypeError, match="layers"):
            call()


def test_execution_version_counts_changes_anywhere_in_the_tree():
    lm = _lm3()
    v0 = execution_version(lm)
    lm.set_execution({"backend": "torch"})                        # same value, still a (re)configuration
    v1 = execution_version(lm)
    assert v1 > v0
    lm.ssm_layers[1].set_execution({"backend": "sequential"})      # a change on one layer moves the root's version
    v2 = execution_version(lm)
    assert v2 > v1
    sub = execution_version(lm.ssm_layers[0])
    assert execution_version(lm.ssm_layers[0]) == sub                # untouched layer: stable
    with lm.ssm_layers[0].execution({"precision": "ieee"}):
        inside = execution_version(lm)
        assert inside > v2
    assert execution_version(lm) > inside                           # the restore on exit counts too
    with pytest.raises(ValueError):
        lm.set_execution({"backend": "bogus"})                      # a refused change moves nothing
    after = execution_version(lm)
    with pytest.raises(ValueError):
        lm.set_execution({"backend": "bogus"})
    assert execution_version(lm) == after


def test_graphed_step_detects_an_execution_change_after_capture():
    """The recorded kernels embody the execution at capture time.  CPU test of the detection logic on a handle
    built without a real graph (the replay refuses before it would touch the graph)."""
    from orbit_ssm.graphs import CaptureError, GraphedStep
    layer = SSDMimoSSM(SSDMimoConfig(d_model=16, **_S), execution=_EX)

    class _Graph:
        replayed = 0
        def replay(self): self.replayed += 1

    graph = _Graph()
    step = GraphedStep(module=layer, graph=graph, fwd_bwd=False, static_inputs=(), static_output=torch.zeros(1),
                       static_grad_output=None, leaves=(), static_grads=(), pool=None, stream=None,
                       execution_version=execution_version(layer))
    assert not step.stale
    assert step.replay() is step.static_output and graph.replayed == 1
    layer.set_execution({"precision": "ieee"})
    assert step.stale
    with pytest.raises(CaptureError, match="stale-execution") as err:
        step.replay()
    assert err.value.reason == "stale-execution" and graph.replayed == 1
    # the context manager invalidates too (entering and leaving), also for a graph held on a parent module
    parent = torch.nn.Sequential(layer)
    fresh = GraphedStep(module=parent, graph=graph, fwd_bwd=False, static_inputs=(), static_output=torch.zeros(1),
                        static_grad_output=None, leaves=(), static_grads=(), pool=None, stream=None,
                        execution_version=execution_version(parent))
    with layer.execution({"backend": "sequential"}):
        assert fresh.stale
    assert fresh.stale
    with pytest.raises(CaptureError, match="capture again"):
        fresh.check_fresh()


# ---------------------------------------------------------------- v2 recipes and experiment-file spellings
def _as_v3(recipe):
    """Rewrite a v4 layer recipe into the v3 spellings (what older recipes hold)."""
    import copy
    doc = copy.deepcopy(recipe)
    m = doc["model"]
    doc["schema"] = "orbit_ssm.layer/v3"
    if m.get("bc_bias") == "head":
        m["bc_bias"] = "head_bc"
    if doc["family"] == "hybrid_mamba":
        m.pop("skip")                                           # HybridMamba gained `skip` in v4
    return doc


def _as_v2(recipe):
    """Rewrite a v4 layer recipe into the v2 spellings (what older recipes hold)."""
    doc = _as_v3(recipe)
    m = doc["model"]
    doc["schema"] = "orbit_ssm.layer/v2"
    for new, old in (("skip", "d_skip"), ("skip_sharing", "d_skip_sharing")):
        if doc["family"] == "ssd_mimo" and new in m:
            m[old] = m.pop(new)
    if doc["family"] == "hybrid_mamba":
        m["d_skip_mode"] = m.pop("skip_sharing")
        m["use_trap"] = m.pop("write_rule") == "trapezoid"
    return doc
