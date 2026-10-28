"""CPU tests for the shared row_blocks / rotation / maps config spec.

Pins the shared config spec and the API
unification -- ONE input form, flat keywords
(`row_blocks`, the rotation fields, the B/C map fields), with `RowBlock` or its
strict dict as the block form; both resolve to ONE canonical plain dict; removed
decay-law names raise naming the replacement; the saved dict is canonical and
idempotent (parser section, `orbit_ssm.config_spec`, pure Python).  The module
section pins that the production SSMs build from that dict, record exactly it
as flat recipe fields, round trip through `from_recipe`, and treat every
retired spelling as an unknown keyword.  Does NOT pin numerics.
"""

import pytest
from orbit_ssm.config_spec import (ConfigSpecError, RowBlock, flat_spec,
                                   load_saved, resolve)
from orbit_ssm import HybridMambaConfig, SSDMimoConfig


N = 128


def _lm_complex_flat():
    # today's `_ssd_mimo_arm(2, 2, True)` minus the removed spellings
    return dict(theta_per_head=False, learn_theta_freq=True, use_theta_h=True,
                bc_bias="head", b_gate=True, c_gate=True, bc_bias_init="auto")


HALF = [{"frac": 0.5, "decay": "head_dynamic", "complex": True},
        {"frac": 0.5, "decay": "head_dynamic", "complex": False}]


CANON_SSD_R2_COMPLEX = {
    # BLOCKGATES: each block's resolved `gate` is saved
    "row_blocks": [{"rows": 64, "decay": "head_dynamic", "complex": True, "gate": False},
             {"rows": 64, "decay": "head_dynamic", "complex": False, "gate": True}],
    "rotation": {"theta_per_head": False, "learn_theta_freq": True,
                 "use_theta_h": True, "omega_layout": "pair",
                 "theta_init": "kaiming"},
    "maps": {"b_norm": True, "c_norm": True, "bc_bias": "head",
             "b_gate": True, "c_gate": True, "bc_bias_init": "spread"},
}


def test_rowblock_and_dict_spellings_agree_ssd():
    flat = resolve("ssd_mimo", d_state=N, d_ssm_in=2, row_blocks=HALF, **_lm_complex_flat())
    typed = resolve("ssd_mimo", d_state=N, d_ssm_in=2,
                    row_blocks=[RowBlock(frac=0.5, decay="head_dynamic", complex=True),
                                RowBlock(frac=0.5, decay="head_dynamic")],
                    use_theta_h=True, bc_bias="head")
    defaults = resolve("ssd_mimo", d_state=N, d_ssm_in=2, row_blocks=HALF)
    assert flat == typed == defaults == CANON_SSD_R2_COMPLEX


def test_saved_form_is_plain_and_idempotent():
    spec = resolve("ssd_mimo", d_state=N, d_ssm_in=2, row_blocks=HALF)
    assert spec == CANON_SSD_R2_COMPLEX
    # a saved recipe stores the FLAT form; resolve() is idempotent on it
    saved = flat_spec(spec)
    assert "maps" not in saved and saved["bc_bias"] == "head" and saved["omega_layout"] == "pair"
    assert saved["row_blocks"] == spec["row_blocks"]
    assert load_saved("ssd_mimo", saved, d_state=N, d_ssm_in=2) == spec
    assert resolve("ssd_mimo", d_state=N, d_ssm_in=2, **saved) == spec
    hm = resolve("hybrid_mamba", d_state=64,
                 row_blocks=[{"frac": 0.5, "decay": "coord_static", "complex": True},
                             {"rows": 32, "decay": "head_dynamic"}], b_gate=True, c_gate=True)
    assert load_saved("hybrid_mamba", flat_spec(hm), d_state=64) == hm
    # a non-canonical saved spec (a block still spelled by `frac`) is refused
    with pytest.raises(ConfigSpecError, match="not canonical"):
        load_saved("ssd_mimo", {**saved, "row_blocks": HALF}, d_state=N, d_ssm_in=2)


def test_ssd_default_is_one_real_m3_block():
    spec = resolve("ssd_mimo", d_state=64, d_ssm_in=1)
    assert spec == {"row_blocks": [{"rows": 64, "decay": "head_dynamic", "complex": False,
                                    "gate": False}],         # no B/C gate at R = O = 1: the block resolves to what runs
                    "rotation": None,
                    "maps": {"b_norm": True, "c_norm": True, "bc_bias": "head",
                             "b_gate": False, "c_gate": False,
                             "bc_bias_init": "ones"}}


def test_unknown_block_key_is_a_typeerror_listing_the_valid_ones():
    with pytest.raises(TypeError, match=r"unknown key.*'rot'.*valid keys"):
        resolve("hybrid_mamba", d_state=64, row_blocks=[{"rows": 64, "decay": "coord_static", "rot": True}])


def test_rowblock_dicts_are_strict_about_types_and_shape():
    with pytest.raises(TypeError, match="complex"):
        resolve("ssd_mimo", d_state=64, row_blocks=[{"rows": 64, "decay": "head_dynamic", "complex": 1}])
    with pytest.raises(TypeError, match="RowBlock or its dict"):
        resolve("ssd_mimo", d_state=64, row_blocks=[("rows", 64)])
    with pytest.raises(ConfigSpecError, match="not one block"):
        resolve("ssd_mimo", d_state=64, row_blocks={"rows": 64, "decay": "head_dynamic"})
    with pytest.raises(ConfigSpecError, match="not both"):
        resolve("ssd_mimo", d_state=64, row_blocks=[{"rows": 64, "frac": 1.0, "decay": "head_dynamic"}])


def test_bc_bias_none_resolves_its_init_and_norms_are_independent():
    # the explicit spelling of the retired `bc_norm=False` shorthand
    s = resolve("ssd_mimo", d_state=64, d_ssm_in=1, b_norm=False, c_norm=False, bc_bias="none")
    assert s["maps"]["b_norm"] is False and s["maps"]["c_norm"] is False
    assert s["maps"]["bc_bias"] == "none" and s["maps"]["bc_bias_init"] == "none"
    # a bias init next to bc_bias="none" contradicts it
    with pytest.raises(ConfigSpecError, match="bc_bias='none'"):
        resolve("ssd_mimo", d_state=64, d_ssm_in=1, bc_bias="none", bc_bias_init="ones")
    # the diagnostic arm: the write norm off alone
    t = resolve("ssd_mimo", d_state=64, d_ssm_in=2, b_norm=False)
    assert (t["maps"]["b_norm"], t["maps"]["c_norm"], t["maps"]["bc_bias"]) == \
        (False, True, "head")


def test_rotation_needs_a_complex_block():
    with pytest.raises(ConfigSpecError, match="no complex"):
        resolve("ssd_mimo", d_state=64, d_ssm_in=1, use_theta_h=True)
    # explicit "off" is harmless and not saved
    s = resolve("ssd_mimo", d_state=64, d_ssm_in=1, use_theta_h=False, learn_theta_freq=False)
    assert s["rotation"] is None


def test_per_block_rotation_is_refused():
    with pytest.raises(TypeError, match="unknown key"):
        resolve("hybrid_mamba", d_state=64,
                row_blocks=[{"rows": 64, "decay": "coord_static", "complex": True,
                             "use_theta_h": False}])


@pytest.mark.parametrize("rows, msg", [
    ([{"frac": 0.5, "decay": "head_dynamic"}, {"frac": 0.5, "decay": "head_static"}], "one decay law"),
    ([{"frac": 0.5, "decay": "head_dynamic"}, {"frac": 0.5, "decay": "head_dynamic", "complex": True}],
     "complex block comes first"),
    ([{"frac": 0.5, "decay": "head_dynamic"}, {"frac": 0.5, "decay": "head_dynamic"}], "one complex"),
    ([{"rows": 32, "decay": "head_dynamic", "complex": True}] * 2
     + [{"rows": 64, "decay": "head_dynamic"}], "at most two"),
    ([{"rows": 128, "decay": "coord_static"}], "ssd_mimo decay"),
    ([{"rows": 3, "decay": "head_dynamic", "complex": True}, {"rows": 125, "decay": "head_dynamic"}], "odd"),
    ([{"rows": 64, "decay": "head_dynamic"}], "sum to 64"),
])
def test_ssd_rows_validation(rows, msg):
    with pytest.raises(ConfigSpecError, match=msg):
        resolve("ssd_mimo", d_state=N, d_ssm_in=2, row_blocks=rows)


def test_ssd_complex_needs_head_bias():
    with pytest.raises(ConfigSpecError, match="head"):
        resolve("ssd_mimo", d_state=N, d_ssm_in=2, row_blocks=HALF, bc_bias="shared")


def test_family_extensions_are_scoped():
    with pytest.raises(ConfigSpecError, match="hybrid_mamba-only"):
        resolve("ssd_mimo", d_state=64, d_ssm_in=1, bc_gate_act="sigmoid")
    with pytest.raises(ConfigSpecError, match="ssd_mimo-only"):
        resolve("hybrid_mamba", d_state=64, row_blocks=[{"rows": 64, "decay": "coord_static"}],
                bc_bias_init="ones")


def test_hm_manifest_arm_translation():
    # HM_GATES_INSANE-style arm, rewritten to the new vocabulary
    s = resolve("hybrid_mamba", d_state=64,
                row_blocks=[{"frac": 0.5, "decay": "coord_dynamic", "complex": True, "gate": True},
                            {"frac": 0.5, "decay": "head_dynamic", "complex": True}],
                b_gate=True, c_gate=True, bc_gate_act="sigmoid", bc_gate_bias=True,
                b_norm=True, c_norm=True, bc_bias="none",
                learn_theta_freq=True, use_theta_h=True)
    assert s["row_blocks"][0] == {"rows": 32, "decay": "coord_dynamic", "complex": True, "gate": True}
    assert s["row_blocks"][1]["gate"] is False          # BLOCKGATES default (complex)
    assert s["maps"] == {"b_norm": True, "c_norm": True, "bc_bias": "none",
                         "b_gate": True, "c_gate": True, "bc_gate_act": "sigmoid",
                         "bc_gate_bias": True, "bc_gate_pos": "pre"}
    with pytest.raises(ConfigSpecError, match="served on"):
        resolve("hybrid_mamba", d_state=64,
                row_blocks=[{"rows": 64, "decay": "head_static", "complex": True}])
    with pytest.raises(ConfigSpecError, match="required"):
        resolve("hybrid_mamba", d_state=64, d_ssm_in=1)


@pytest.mark.parametrize("old, new", [("m3", "head_dynamic"), ("m2", "head_static")])
def test_model_named_decay_laws_raise_with_replacement(old, new):
    for fam, extra in (("ssd_mimo", {}), ("hybrid_mamba", {})):
        with pytest.raises(ConfigSpecError, match=new):
            resolve(fam, d_state=64, row_blocks=[{"rows": 64, "decay": old}], **extra)


def test_dropped_fixed_decay_law_is_an_unknown_decay():
    """The HM time-constant flat law "fixed" is
    dropped (fresh prod start, no legacy path).  The spec parser of both
    families and the HM module refuse it as an unknown decay law (the
    message lists the valid ones), never aliasing it to another law."""
    from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM, HybridMambaConfig
    for fam in ("ssd_mimo", "hybrid_mamba"):
        with pytest.raises(ConfigSpecError, match=r"decay must be one of.*'fixed'"):
            resolve(fam, d_state=64, row_blocks=[{"rows": 64, "decay": "fixed"}])
    with pytest.raises(ConfigSpecError, match=r"decay must be one of"):
        HybridMambaSSM(HybridMambaConfig(d_model=16, d_state=32, d_inner=32, n_heads=2,
                       
                       row_blocks=[{"rows": 16, "decay": "coord_static"},
                                   {"rows": 16, "decay": "fixed"}]), execution={"backend": "torch"})


def test_hm_default_maps_one_gate_form():
    s = resolve("hybrid_mamba", d_state=64, row_blocks=[{"rows": 64, "decay": "coord_static"}])
    assert s["maps"] == {"b_norm": True, "c_norm": True, "bc_bias": "head",
                         "b_gate": False, "c_gate": False,
                         "bc_gate_act": "one_plus", "bc_gate_bias": False,
                         "bc_gate_pos": "pre"}
    # one_plus is served on the in-scan "post" path too (kernel code 5)
    t = resolve("hybrid_mamba", d_state=64, row_blocks=[{"rows": 64, "decay": "coord_static"}],
                b_gate=True, c_gate=True, bc_gate_pos="post")
    assert (t["maps"]["bc_gate_act"], t["maps"]["bc_gate_pos"]) == ("one_plus", "post")


# ── the modules ──────────────────────────────────────────────────────────
torch = pytest.importorskip("torch")


HALF_DYN = [{"frac": 0.5, "decay": "head_dynamic", "complex": True},
            {"frac": 0.5, "decay": "head_dynamic", "complex": False}]


def _ssd(**kw):
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig
    return SSDMimoSSM(SSDMimoConfig(d_model=128, d_state=64, d_inner=256, **kw))


def _hm(**kw):
    from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM
    return HybridMambaSSM(HybridMambaConfig(d_model=64, d_state=64, d_inner=128, d_head=16, **kw))


def test_ssd_module_records_the_canonical_spec_and_round_trips():
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    m = _ssd(d_ssm_in=2, row_blocks=HALF_DYN, use_theta_h=False, bc_bias="head")
    model = m.to_recipe()["model"]
    spec = resolve("ssd_mimo", d_state=64, d_ssm_in=2, row_blocks=HALF_DYN, use_theta_h=False)
    # the recipe's model section is FLAT: the spec's fields under the constructor's names
    assert {k: v for k, v in model.items() if k in flat_spec(spec)} == flat_spec(spec)
    for gone in ("rows", "rotation", "maps", "rot_mode", "rope_fraction", "a_dyn",
                 "bias_gran", "use_g_b", "use_g_c", "bc_norm", "R", "O", "headdim", "ngroups"):
        assert gone not in model
    again = SSDMimoSSM.from_recipe(m.to_recipe())
    assert again.to_recipe() == m.to_recipe()
    assert m.n_rope_angles == 16 and m.rot_mode == "m3" and m.a_dyn
    assert not m.use_theta_h and m.learn_theta_freq


def test_ssd_module_maps_are_independent_fields():
    m = _ssd(d_ssm_in=1, bc_bias="none")               # norms on, no biases
    assert m.b_norm and m.c_norm and not m.bc_bias
    assert not hasattr(m, "B_bias")
    n = _ssd(d_ssm_in=1, b_norm=False, c_norm=False)   # biases, no norms
    assert n.bc_bias and not hasattr(n, "B_norm")
    s = _ssd(d_ssm_in=1, row_blocks=[{"frac": 1.0, "decay": "head_static"}])
    assert not s.a_dyn and s.A_log is not None
    x = torch.randn(2, 8, 256)
    sel = torch.randn(2, 8, 128)
    for mod in (m, n, s):
        assert torch.isfinite(mod.scan(x, sel_x=sel)).all()


def test_hm_module_records_the_canonical_spec_and_round_trips():
    from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM
    rows = [{"frac": 0.5, "decay": "coord_static", "complex": True},
            {"frac": 0.5, "decay": "coord_dynamic"}]
    with pytest.raises(NotImplementedError, match="separate B/C gates"):
        _hm(row_blocks=rows, b_gate=True)
    m = _hm(row_blocks=rows, b_gate=True, c_gate=True, bc_gate_act="sigmoid",
            bc_bias="shared")
    model = m.to_recipe()["model"]
    for gone in ("fracs", "gates", "bc_bias_heads", "rows", "rotation", "maps",
                 "gate_act", "gate_bias", "gate_pos", "headdim", "use_g_b", "use_g_c"):
        assert gone not in model
    assert model["b_gate"] and model["c_gate"] and model["bc_bias"] == "shared"
    assert model["bc_gate_act"] == "sigmoid"
    assert m.gB_proj is not None and m.gC_proj is not None
    assert tuple(m.beta_b.shape) == (1, 64)
    again = HybridMambaSSM.from_recipe(m.to_recipe())
    assert again.to_recipe() == m.to_recipe()


def test_hm_flat_rows_refuse_unserved_maps_by_name():
    flat = [{"rows": 32, "decay": "coord_static"}, {"rows": 32, "decay": "head_dynamic"}]
    _hm(row_blocks=flat)                                # served defaults
    for kw, name in ((dict(bc_bias="none"), "bias='none'"),
                     (dict(b_norm=False), "b_norm=False"),
                     (dict(c_gate=True), "c_gate")):
        with pytest.raises(NotImplementedError, match=name):
            _hm(row_blocks=flat, **kw)
    cx = [{"rows": 32, "decay": "coord_static"},
          {"rows": 32, "decay": "head_dynamic", "complex": True}]
    with pytest.raises(NotImplementedError, match="learn_theta_freq"):
        _hm(row_blocks=cx)
    _hm(row_blocks=cx, learn_theta_freq=False, use_theta_h=False)


def test_mixer_and_lm_take_the_flat_map_fields():
    """The B/C map fields are flat on every surface (layer, config, LM) and are
    recorded flat; the retired `bc_norm` shorthand is an unknown keyword."""
    from orbit_ssm.model import CausalLM
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    for fam in ("ssd_mimo", "hybrid_mamba"):
        CausalLM(vocab_size=16, layers=dict(type=fam, d_state=64, b_norm=True, c_norm=True, bc_bias="head"), n_layers=1, d_model=64)
    m = SSDMimoSSM(SSDMimoConfig(d_model=64, d_state=64, d_ssm_in=1, b_norm=False, c_norm=False, bc_bias="none"))
    core = m.to_recipe()["model"]
    assert (core["b_norm"], core["c_norm"], core["bc_bias"]) == (False, False, "none")
    assert "bc_norm" not in core and "maps" not in core


def test_chart_key_is_refused_on_these_families():
    """T2: the 'chart' key belongs to another family and is refused here."""
    with pytest.raises(ConfigSpecError, match="unknown keys"):
        resolve("ssd_mimo", d_state=16,
                row_blocks=[{"rows": 16, "decay": "head_dynamic", "chart": True}])


@pytest.mark.parametrize("key", ["rot_mode", "rope_fraction", "a_dyn", "decay_mode",
                                 "fracs", "gates", "full_blocks", "reflect"])
def test_retired_spellings_are_unknown_keywords_listing_the_valid_names(key):
    """No retired-name tables: a spelling the API no longer has is the generic
    unknown-keyword TypeError, which lists the valid names."""
    from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    for cls in (SSDMimoSSM, HybridMambaSSM):
        with pytest.raises(TypeError, match="valid"):
            cls({"d_model": 64, "d_state": 16, key: 1})
