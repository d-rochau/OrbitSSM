"""Row-block rules on SSD-MIMO and HybridMamba.

Pins a CONTRACT CLAUSE (the row-spec rules: block decay, ignored keys), both families, CPU, no kernels:

* a block's ``decay`` may be omitted: it resolves to the family default (SSD-MIMO ``head_dynamic``, HybridMamba
  ``coord_static``) and the resolved recipe records it on every block;
* ``decay="none"`` (a_t = 1) is not a law of either family: both refuse it by name;
* blocks that share ONE scan with ONE decay per head agree (SSD-MIMO: both blocks), else a ``ConfigSpecError``
  (a ``ValueError``) naming the blocks and why; HybridMamba has no such pair (its law is per row block in the
  ``mh_scan`` launch and the group key of a flat scan), pinned by the row plan below;
* a block has no ``write`` key (the write rule is a layer field): a ``TypeError`` from the strict block dict;
* a layer-level lever of a decay law that no block uses is refused instead of silently ignored, at its default it is
  accepted (recipes record defaults), and it is accepted as soon as a block has the law;
* every key that would be ignored is refused and every resolved value is the one that runs: a block ``gate`` (False when
  no B/C gate exists, ``gate=True`` without one raises), HM ``write_rule`` (recorded per law, raises when every row is
  flat), SSD ``trap_proj_bias=False`` under euler, HM flat head_dynamic order (complex first), HM
  ``bc_gate_act`` / ``bc_gate_pos`` / ``gate_init`` without a gate.
Geometry: tiny CPU modules (N 16-32, d_inner 16); nothing here is a numerical claim.
"""
from __future__ import annotations

import warnings

import pytest

from orbit_ssm import config_spec as cs
from orbit_ssm.ssm.hybrid_mamba import HybridMambaConfig, HybridMambaSSM, _HybridMambaScan, compile_rows
from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig, SSDMimoSSM, _SSDMimoScan

GEOM = dict(d_model=12, d_inner=16, n_heads=2)


def _ssd(**kw):
    return SSDMimoConfig(d_state=16, **GEOM, **kw)


def _hm(**kw):
    return HybridMambaConfig(**GEOM, **kw)


def _recipe_rows(layer):
    return layer.to_recipe()["model"]["row_blocks"]


# ── omitted decay resolves and is recorded ───────────────────────────────
def test_ssd_omitted_decay_resolves_to_head_dynamic_on_every_block_and_round_trips():
    """contract: SSD-MIMO blocks without ``decay`` resolve to ``head_dynamic``; the recipe records it on every block and reloads unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = SSDMimoSSM(_ssd(row_blocks=[{"frac": 0.5, "complex": True}, {"frac": 0.5}]))
        rows = _recipe_rows(layer)
        assert [b["decay"] for b in rows] == ["head_dynamic", "head_dynamic"]
        assert [b["rows"] for b in rows] == [8, 8]
        again = SSDMimoSSM.from_recipe(layer.to_recipe())
    assert _recipe_rows(again) == rows


def test_hm_omitted_decay_resolves_to_coord_static_and_other_blocks_keep_theirs():
    """contract: HybridMamba blocks without ``decay`` resolve to ``coord_static`` (the config default law); a named decay is untouched; recorded and reloadable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = HybridMambaSSM(_hm(d_state=16, row_blocks=[{"rows": 8}, {"rows": 8, "decay": "head_static"}]))
        rows = _recipe_rows(layer)
        assert [b["decay"] for b in rows] == ["coord_static", "head_static"]
        again = HybridMambaSSM.from_recipe(layer.to_recipe())
    assert _recipe_rows(again) == rows


@pytest.mark.parametrize("family, default", [("ssd_mimo", "head_dynamic"), ("hybrid_mamba", "coord_static")])
def test_resolve_records_the_default_decay_per_family(family, default):
    """contract: `config_spec.resolve` puts the family's default law (`DEFAULT_DECAY`) on a block that omits ``decay``; ``RowBlock`` objects and dicts agree."""
    assert cs.DEFAULT_DECAY[family] == default
    for block in ({"rows": 16}, cs.RowBlock(rows=16)):
        got = cs.resolve(family, d_state=16, row_blocks=[block])["row_blocks"]
        assert [b["decay"] for b in got] == [default]


# ── "none" is not a law of either family ────────────────────────────────────────────────────
@pytest.mark.parametrize("family, laws", [("ssd_mimo", "head_dynamic"), ("hybrid_mamba", "coord_static")])
def test_decay_none_is_refused_by_name(family, laws):
    """contract: ``decay="none"`` (a_t = 1) is not served by SSD-MIMO or HybridMamba; the error names the block, the value and the laws that are served."""
    with pytest.raises(cs.ConfigSpecError, match=rf"row_blocks\[1\].*'none'.*not served by {family}.*{laws}"):
        cs.resolve(family, d_state=16, row_blocks=[{"rows": 8, "decay": laws}, {"rows": 8, "decay": "none"}])


# ── blocks of one scan agree ─────────────────────────────────────────────
def test_ssd_blocks_with_different_decays_are_refused_naming_blocks_and_scan():
    """contract: SSD-MIMO's two blocks are rows of one scan with one decay per head; different decays raise a ValueError that names both blocks and says why."""
    rows = [{"frac": 0.5, "decay": "head_dynamic", "complex": True}, {"frac": 0.5, "decay": "head_static"}]
    with pytest.raises(ValueError, match=r"ONE scan.*row_blocks\[0\]='head_dynamic', row_blocks\[1\]='head_static'"):
        _ssd(row_blocks=rows)
    with pytest.raises(ValueError, match=r"ONE scan"):
        cs.resolve("ssd_mimo", d_state=16, row_blocks=rows)


def test_ssd_blocks_with_one_omitted_decay_agree_only_with_the_default():
    """contract: an omitted decay resolves BEFORE the agreement check: ``{}`` next to ``head_dynamic`` agrees, next to ``head_static`` is refused."""
    ok = cs.resolve("ssd_mimo", d_state=16, row_blocks=[{"rows": 8, "complex": True}, {"rows": 8, "decay": "head_dynamic"}])
    assert [b["decay"] for b in ok["row_blocks"]] == ["head_dynamic"] * 2
    with pytest.raises(ValueError, match=r"row_blocks\[0\]='head_dynamic', row_blocks\[1\]='head_static'"):
        cs.resolve("ssd_mimo", d_state=16, row_blocks=[{"rows": 8, "complex": True}, {"rows": 8, "decay": "head_static"}])


def test_hm_row_plan_groups_flat_rows_by_law_and_keeps_a_law_per_coord_block():
    """law: HybridMamba's decay granularity: the flat laws are one scan per law (the law is the group key, one scalar per head), the coord laws are one law per row block (the per-row ``row_law`` of mh_scan), so no two blocks can disagree."""
    plan = compile_rows([{"rows": 8, "decay": "coord_static"}, {"rows": 4, "decay": "head_dynamic", "complex": True},
                         {"rows": 8, "decay": "coord_dynamic"}, {"rows": 4, "decay": "head_dynamic"},
                         {"rows": 8, "decay": "head_static"}], 32)
    assert (plan.n_hm, plan.n3, plan.n3_complex, plan.n2) == (16, 8, 4, 8)
    assert [f.rows for f in plan.hm] == [8, 8] and len({f.law for f in plan.hm}) == 2
    layer = HybridMambaSSM(_hm(d_state=32, row_blocks=[
        {"rows": 8, "decay": "coord_static"}, {"rows": 8, "decay": "coord_dynamic"}, {"rows": 16, "decay": "head_dynamic"}],
        learn_theta_freq=False, use_theta_h=False))
    assert layer.row_law.tolist() == [0] * 8 + [1] * 8


# ── write rule is a layer field ──────────────────────────────────────────
@pytest.mark.parametrize("family, law", [("ssd_mimo", "head_dynamic"), ("hybrid_mamba", "coord_static")])
def test_a_block_with_a_write_key_is_refused(family, law):
    """contract: the write rule is layer level; a ``write`` key on a row block is a TypeError (unknown key) in the dict form, through the config and through `resolve`."""
    block = {"rows": 16, "decay": law, "write": "euler"}
    with pytest.raises(TypeError, match="write"):
        cs.resolve(family, d_state=16, row_blocks=[block])
    make = _ssd if family == "ssd_mimo" else (lambda **kw: _hm(d_state=16, **kw))
    with pytest.raises(TypeError, match="write"):
        make(row_blocks=[block])


# ── levers of an unused law ──────────────────────────────────────────────
STATIC = [{"frac": 1.0, "decay": "head_static"}]


@pytest.mark.parametrize("lever", [dict(a_act="softplus"), dict(a_dyn_per_sibling=True), dict(a_proj_bias=False)])
def test_ssd_rate_levers_without_a_head_dynamic_block_are_refused(lever):
    """contract: SSD-MIMO's ``a_act`` / ``a_dyn_per_sibling`` / ``a_proj_bias`` act on the head_dynamic rate only; set to a non-default value with only head_static blocks they are refused by name, in the config and in the layer."""
    (key,) = lever
    with pytest.raises(ValueError, match=rf"{key}=.*head_dynamic.*would be ignored"):
        _ssd(row_blocks=STATIC, **lever)
    with pytest.raises(ValueError, match=rf"SSDMimoSSM: {key}=.*would be ignored"):       # the scan core, below the config
        _SSDMimoScan(d_inner=16, d_state=16, n_heads=2, d_model=12, row_blocks=STATIC, **lever)


def test_ssd_rate_levers_are_accepted_at_default_or_with_a_head_dynamic_block():
    """contract: the refusal is for a value that would be ignored: defaults (what recipes record) pass with head_static, and a non-default passes once a block has head_dynamic."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ssd(row_blocks=STATIC, a_act="heavy_tail", a_dyn_per_sibling=False, a_proj_bias=True)
        _ssd(row_blocks=[{"frac": 1.0, "decay": "head_dynamic"}], a_act="softplus", a_proj_bias=False)
        layer = SSDMimoSSM(_ssd(row_blocks=STATIC))
        SSDMimoSSM.from_recipe(layer.to_recipe())


@pytest.mark.parametrize("rows, lever", [
    ([{"rows": 16, "decay": "coord_static"}], dict(dyn_res="h")),
    ([{"rows": 16, "decay": "coord_static"}], dict(dyn_act="softplus")),
    ([{"rows": 16, "decay": "coord_dynamic"}], dict(mu_res="h")),
    ([{"rows": 16, "decay": "coord_dynamic"}], dict(mu_cap=8.0)),
    ([{"rows": 16, "decay": "coord_potential"}], dict(a_floor=1e-3)),
    ([{"rows": 16, "decay": "coord_potential"}], dict(a_act="softplus")),
])
def test_hm_law_levers_without_their_law_are_refused(rows, lever):
    """contract: HybridMamba's ``dyn_res``/``dyn_act`` (coord_dynamic), ``mu_res``/``mu_cap`` (coord_potential), ``a_floor``/``a_act`` (head_dynamic) are refused when set to a non-default value and no block has their law, in the config and in the layer."""
    (key,) = lever
    with pytest.raises(ValueError, match=rf"{key}=.*would be ignored"):
        _hm(d_state=16, row_blocks=rows, **lever)
    with pytest.raises(ValueError, match=rf"HybridMambaSSM: {key}=.*would be ignored"):   # the scan core, below the config
        _HybridMambaScan(d_inner=16, d_state=16, n_heads=2, d_model=12, row_blocks=rows, **lever)


def test_hm_law_levers_are_accepted_at_default_or_with_their_law():
    """contract: HybridMamba's law levers pass at their defaults on any rows and at any value once a block has the law."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _hm(d_state=16, row_blocks=[{"rows": 16, "decay": "coord_static"}], dyn_res="n", dyn_act="heavy_tail",
            mu_res="n", mu_cap=4.0, a_floor=1e-4, a_act="heavy_tail")
        rows = [{"rows": 8, "decay": "coord_dynamic"}, {"rows": 4, "decay": "coord_potential"}, {"rows": 4, "decay": "head_dynamic"}]
        HybridMambaSSM(_hm(d_state=16, row_blocks=rows, dyn_res="h", dyn_act="softplus", mu_res="h", mu_cap=8.0,
                           a_floor=1e-3, a_act="softplus"))


# ── block gate resolves to what runs ─────────────────────────────────────
def test_block_gate_resolves_false_without_a_gate_and_default_with_one():
    """contract: an omitted block ``gate`` is False when no B/C gate exists (SSD at R = O = 1, HM default) and `BLOCK_GATE_DEFAULT` (real True, complex False) when one does; the resolved value is recorded."""
    rows = [{"rows": 8, "decay": "head_dynamic", "complex": True}, {"rows": 8, "decay": "head_dynamic"}]
    off = cs.resolve("ssd_mimo", d_state=16, row_blocks=rows)
    assert [b["gate"] for b in off["row_blocks"]] == [False, False] and not off["maps"]["b_gate"]
    on = cs.resolve("ssd_mimo", d_state=16, d_ssm_in=2, row_blocks=rows)
    assert [b["gate"] for b in on["row_blocks"]] == [False, True] and on["maps"]["b_gate"]
    hm_off = cs.resolve("hybrid_mamba", d_state=16, row_blocks=[{"rows": 16, "decay": "coord_static"}])
    assert hm_off["row_blocks"][0]["gate"] is False
    hm_on = cs.resolve("hybrid_mamba", d_state=16, row_blocks=[{"rows": 16, "decay": "coord_static"}], b_gate=True, c_gate=True)
    assert hm_on["row_blocks"][0]["gate"] is True
    assert cs.resolve("ssd_mimo", d_state=16, **cs.flat_spec(on), ) == on        # saved resolved is canonical


@pytest.mark.parametrize("family, law", [("ssd_mimo", "head_dynamic"), ("hybrid_mamba", "coord_static")])
def test_explicit_gate_true_without_a_bc_gate_is_refused(family, law):
    """contract: ``gate=True`` on a block while neither b_gate nor c_gate exists would be ignored: ValueError naming the block."""
    with pytest.raises(ValueError, match=r"row_blocks\[0\]: gate=True, but no B/C gate exists"):
        cs.resolve(family, d_state=16, row_blocks=[{"rows": 16, "decay": law, "gate": True}])


# ── write rule ───────────────────────────────────────────────────────────
def test_hm_write_rule_with_only_flat_rows_is_refused_and_not_recorded():
    """contract: HM ``write_rule`` acts on coord rows only; explicit with every row flat raises (config and layer), and an all-flat layer records ``write_rule=None`` and reports the fixed per-law rules."""
    flat = [{"rows": 8, "decay": "head_dynamic"}, {"rows": 8, "decay": "head_static"}]
    for rule in ("trapezoid", "euler"):
        with pytest.raises(ValueError, match="write_rule is set, but every row block is a flat law"):
            _hm(d_state=16, row_blocks=flat, write_rule=rule)
        with pytest.raises(ValueError, match="write_rule is set, but every row block is a flat law"):
            _HybridMambaScan(d_inner=16, d_state=16, n_heads=2, d_model=12, row_blocks=flat, use_trap=rule == "trapezoid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = HybridMambaSSM(_hm(d_state=16, row_blocks=flat))
    assert layer.to_recipe()["model"]["write_rule"] is None
    assert layer.write_rules == {"head_dynamic": "trapezoid", "head_static": "euler"}


def test_hm_write_rule_with_coord_rows_is_recorded_and_flat_rules_stay_fixed():
    """contract: with a coord row the config's rule is recorded (default trapezoid) and applies to the coord rows only; ``write_rules`` names the rule of every present group."""
    rows = [{"rows": 8, "decay": "coord_static"}, {"rows": 8, "decay": "head_dynamic"}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rule, want in ((None, "trapezoid"), ("euler", "euler")):
            layer = HybridMambaSSM(_hm(d_state=16, row_blocks=rows, write_rule=rule))
            assert layer.to_recipe()["model"]["write_rule"] == want
            assert layer.write_rules == {"coord": want, "head_dynamic": "trapezoid"}
            HybridMambaSSM.from_recipe(layer.to_recipe())


def test_ssd_trap_proj_bias_false_under_euler_is_refused():
    """contract: ``trap_proj_bias=False`` with ``write_rule="euler"`` builds no trapezoid projection, so it would be ignored: ValueError in the config and the layer; accepted under trapezoid."""
    with pytest.raises(ValueError, match="trap_proj_bias=False, but write_rule='euler'"):
        _ssd(write_rule="euler", trap_proj_bias=False)
    with pytest.raises(ValueError, match="trap_proj_bias=False, but write_rule='euler'"):
        _SSDMimoScan(d_inner=16, d_state=16, n_heads=2, d_model=12, write_rule="euler", trap_proj_bias=False)
    _ssd(write_rule="trapezoid", trap_proj_bias=False)
    _ssd(write_rule="euler")


# ── HM order and gate options ────────────────────────────────────────────
def test_hm_flat_head_dynamic_complex_block_must_come_first():
    """contract: the head_dynamic rows are one scan whose rotated rows lead (as in SSD); a complex head_dynamic block after a real one raises, naming the block; the other order and coord rows in between are accepted."""
    real, cx = {"rows": 8, "decay": "head_dynamic"}, {"rows": 8, "decay": "head_dynamic", "complex": True}
    with pytest.raises(cs.ConfigSpecError, match=r"row_blocks\[1\]: a complex head_dynamic block follows a real one"):
        cs.resolve("hybrid_mamba", d_state=16, row_blocks=[real, cx])
    cs.resolve("hybrid_mamba", d_state=16, row_blocks=[cx, real])
    cs.resolve("hybrid_mamba", d_state=24, row_blocks=[cx, {"rows": 8, "decay": "coord_static"}, real])


@pytest.mark.parametrize("opts", [dict(bc_gate_act="sigmoid"), dict(bc_gate_pos="post")])
def test_hm_gate_options_without_a_gate_are_refused_at_resolve(opts):
    """contract: ``bc_gate_act`` / ``bc_gate_pos`` at a non-default value with no B/C gate raise; with gates they resolve."""
    rows = [{"rows": 16, "decay": "coord_static"}]
    with pytest.raises(cs.ConfigSpecError, match="no B/C gate exists"):
        cs.resolve("hybrid_mamba", d_state=16, row_blocks=rows, **opts)
    cs.resolve("hybrid_mamba", d_state=16, row_blocks=rows, b_gate=True, c_gate=True, **opts)


def test_hm_gate_init_without_a_gate_is_refused():
    """contract: ``gate_init`` initialises the B/C gate projections; a non-default value with no gate raises in the config and the layer, and is accepted with gates."""
    with pytest.raises(ValueError, match="gate_init='zero' is set, but no B/C gate exists"):
        _hm(d_state=16, gate_init="zero")
    with pytest.raises(ValueError, match="gate_init='zero' is set, but no B/C gate exists"):
        _HybridMambaScan(d_inner=16, d_state=16, n_heads=2, d_model=12, row_blocks=[{"rows": 16, "decay": "coord_static"}], gate_init="zero")
    _hm(d_state=16, gate_init="zero", b_gate=True, c_gate=True)
