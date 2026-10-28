"""Pins the calibration framework's decisions, CPU only (no kernel launch):
derive() picks the min-worst-case fixed config and a greedy light set at the
fixed stage count; the local layer wins over shipped when valid; stale files
(GPU / Triton / code hash) are ignored; verify uses hysteresis; reset removes
only the local file; disabled == old defaults; unknown GPU == generic light
grid + one message; and every SHIPPED file loads against the current kernel
code hash (regenerate with tools/calibrate_kernels.py after kernel edits)."""
import glob
import json
import os

import pytest
from orbit_ssm.kernels import calibration as cal
from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs  # noqa: F401
from orbit_ssm.kernels.ssd_mimo import calibration_fused, calibration_chunkpar, levers, rot_route  # noqa: F401,E501  (register the shipped families)
from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh  # noqa: F401  (the MH family: P=1 per-cell table)

FAM = "tfam"
GPU = "Test GPU"
TV = "9.9"


cal.register_family(
    FAM, env="TFAM_CALIBRATION", code_files=[cal.__file__],
    kernels=("siso_bwd",), shapes={"A": dict(g=1), "B": dict(g=2)},
    group_key=lambda g, **_: "g%d" % g)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("SSD_CALIBRATION_DIR", str(tmp_path / "local"))
    monkeypatch.setattr(cal, "device_info",
                        lambda: dict(gpu=GPU, sms=1, driver="d", triton=TV,
                                     torch="t"))
    monkeypatch.setattr(cal, "shipped_dir",
                        lambda fam: str(tmp_path / "shipped" / fam))
    cal._STALE.clear(); cal._WARNED.clear(); cal._SOURCE.clear()


T = {  # two shapes: w8s3rN best on A, w4s3r256 best on B, w4s1rN loses on B
    "A": {"w8s3rN": 1.00, "w4s3r256": 1.10, "w4s1rN": 1.01, "w2s3rN": 1.5,
          "w8s1rN": 1.30},
    "B": {"w8s3rN": 1.30, "w4s3r256": 1.00, "w4s1rN": 1.05, "w2s3rN": 1.6,
          "w8s1rN": 1.00},
}


def test_derive_fixed_minimises_worst_case_and_light_keeps_stages():
    d = cal.derive(T)
    assert d["fixed"] == "w4s1rN"          # worst 1.05 < 1.30 / 1.10
    assert d["light"][0] == d["fixed"]
    assert all(cal.cfg_parse(k)["num_stages"] == 1 for k in d["light"])
    d2 = cal.derive({"A": {"w8s3rN": 1.0, "w4s3rN": 1.0},
                     "B": {"w8s3rN": 1.5, "w4s3rN": 1.0}})
    assert d2["fixed"] == "w4s3rN" and d2["worst_slowdown"] == 1.0


def test_light_set_covers_shapes_within_tol_at_fixed_stages():
    t = {"A": {"w4s2rN": 1.0, "w8s2rN": 1.2, "w4s2r128": 1.5},
         "B": {"w4s2rN": 1.2, "w8s2rN": 1.0, "w4s2r128": 1.5},
         "C": {"w4s2rN": 1.5, "w8s2rN": 1.5, "w4s2r128": 1.0}}
    d = cal.derive(t, tol=0.02)
    assert len(d["light"]) == 3 and not d["uncovered"]
    assert d["light_worst_slowdown"] == 1.0


def test_missing_config_counts_as_infinitely_slow():
    d = cal.derive({"A": {"w4s1rN": 1.0}, "B": {"w8s1rN": 1.0}})
    assert d["fixed"] in ("w4s1rN", "w8s1rN")     # nothing covers both


def _write(layer, gpu=GPU, tv=TV, hash_=None, fixed="w4s1rN"):
    data = cal.build_file(FAM, {"siso_bwd": T}, 5, list(T))
    data["provenance"].update(gpu=gpu, triton=tv)
    if hash_:
        data["provenance"]["code_hash"] = hash_
    data["kernels"]["siso_bwd"]["default"]["fixed"] = cal.cfg_parse(fixed)
    p = (cal.local_path if layer == "local" else cal.shipped_path)(FAM, gpu)
    cal.write_json(p, data)


def test_local_beats_shipped_and_stale_is_ignored():
    _write("shipped", fixed="w8s3rN"); _write("local", fixed="w4s3r256")
    assert cal.lookup(FAM, "siso_bwd")["source"] == "local"
    _write("local", tv="0.0")                       # stale local -> shipped
    hit = cal.lookup(FAM, "siso_bwd")
    assert hit["source"] == "shipped" and hit["default"]["fixed"]["num_warps"] == 8
    assert any("triton" in r for r in cal._STALE[(FAM, GPU)])
    _write("shipped", hash_="deadbeef")             # both stale -> none
    assert cal.lookup(FAM, "siso_bwd") is None
    _write("shipped", gpu="Other GPU")              # other GPU's file unused
    assert cal.lookup(FAM, "siso_bwd") is None


def test_verify_hysteresis():
    over, sh, fresh = cal.verify_decision("w4s1rN", T)
    assert not over                                  # shipped == best fixed
    over, sh, fresh = cal.verify_decision("w2s3rN", T)
    assert over and sh > fresh * 1.03
    over, _, _ = cal.verify_decision("w9s9rN", T)    # not even measured
    assert over


def test_reset_removes_only_local():
    _write("shipped"); _write("local")
    assert cal.reset_local(FAM) and not os.path.exists(cal.local_path(FAM, GPU))
    assert os.path.exists(cal.shipped_path(FAM, GPU))
    assert not cal.reset_local(FAM)


def _grid(tier):
    L = lambda: ["light"]
    return cal.calibrated_grid(FAM, "siso_bwd", tier, full=lambda: ["full"],
                               light=L, generic_pin=lambda: ["generic"])


def test_disabled_is_old_defaults(monkeypatch):
    monkeypatch.setenv("TFAM_CALIBRATION", "off")
    _write("local")
    assert [_grid(t) for t in ("pinned", "light", "full")] == [
        ["generic"], ["light"], ["full"]]
    assert cal.source(FAM, "siso_bwd") == "default:full"


class _K:                       # stands in for a Triton Autotuner
    cache = None
    configs = []


def test_enabled_groups_union_prune_and_sources(monkeypatch):
    monkeypatch.setenv("TFAM_CALIBRATION", "on")
    _write("local")
    g = _grid("pinned")
    keys = {cal.cfg_key(c.num_warps, c.num_stages, c.maxnreg) for c in g}
    d = cal.derive_grouped(T, {"A": "g1", "B": "g2"}, fam=FAM)
    want = {cal.cfg_dict_key(d["default"]["fixed"]),
            cal.cfg_dict_key(d["groups"]["g1"]["fixed"]),
            cal.cfg_dict_key(d["groups"]["g2"]["fixed"])}
    assert keys == want and cal.source(FAM, "siso_bwd") == "local"
    k = _K()
    for grp, exp in (("g1", d["groups"]["g1"]), ("g2", d["groups"]["g2"]),
                     ("unseen", d["default"])):
        cal.activate(FAM, grp, {"siso_bwd": k})
        kept = cal.prune(list(g), FAM, "siso_bwd")
        assert len(kept) == 1                       # pinned: ONE config
        assert cal.config_key_of(kept[0], FAM) == cal.cfg_dict_key(exp["fixed"])
    # each group tunes into its OWN cache (never shared between groups)
    cal.activate(FAM, "g1", {"siso_bwd": k}); c1 = k.cache
    cal.activate(FAM, "g2", {"siso_bwd": k}); c2 = k.cache
    cal.activate(FAM, "g1", {"siso_bwd": k})
    assert c1 is not c2 and k.cache is c1
    lg = _grid("light")
    assert cal.source(FAM, "siso_bwd") == "light(local)"
    for grp in ("g1", "g2", "unseen"):
        cal.activate(FAM, grp, {"siso_bwd": k})
        kept = cal.prune(list(lg), FAM, "siso_bwd")
        assert 1 <= len(kept) <= 4
        assert len({c.num_stages for c in kept}) == 1   # stages held fixed
    assert _grid("full") == ["full"] and cal.source(FAM, "siso_bwd") == "full"
    cal.activate(FAM, "g1", {"siso_bwd": k})
    assert cal.prune(list(lg), FAM, "siso_bwd") == list(lg)  # full: no narrowing


def test_disabled_activation_is_a_noop(monkeypatch):
    monkeypatch.setenv("TFAM_CALIBRATION", "off")
    k = _K(); k.cache = {"x": 1}
    cal.activate(FAM, "g1", {"siso_bwd": k})
    assert k.cache == {"x": 1}


def test_owner_four_groups_for_ssd_families():
    from orbit_ssm.kernels.ssd_mimo import calibration_fused as cf
    gk = cf.group_key
    assert gk(rot=True, R=1, O=1) == "complex_R1O1"
    assert gk(rot=False, R=1, O=1) == "real_R1O1"
    assert gk(rot=True, R=2, O=2) == "complex_ROgt1"
    assert gk(rot=False, R=4, O=1) == "real_ROgt1"
    assert cs.group_key() == "complex_R1O1"          # SISO default: complex
    # + the structural rule's T_hi group (real_R1O1_hi) and the fp32 dtype axis (`_fp32`)
    extra = {"real_R1O1_hi"} | {"%s_R1O1_fp32_t%d" % (k, t) for k in ("complex", "real") for t in (32, 64)}
    # + the per-rank launch groups (the per-rank calibration): `<route>_R<R>O<R>[_fp32]_t<T>`, R = 2, 4
    extra |= {"%s_R%dO%d%s_t%d" % (k, R, R, f, t) for k in ("complex", "real") for R, ts in cf.RANK_CHUNKS.items() for t in ts
              for f in ("", "_fp32")}
    for fam, shapes in (("ssd_mimo", cs.SHAPES), ("ssd_mimo_fused", cf.SHAPES)):
        groups = {cal.family(fam)["group_key"](**g) for g in shapes.values()}
        assert groups <= {"complex_R1O1", "real_R1O1", "complex_ROgt1",
                          "real_ROgt1"} | extra
    assert {cf.group_key(**g) for g in cf.SHAPES.values()} == {
        "complex_R1O1", "real_R1O1", "complex_ROgt1", "real_ROgt1"} | extra


def test_per_group_fit_beats_one_global_config():
    t = {"A": {"w8s2rN": 1.0, "w4s2rN": 1.3}, "B": {"w8s2rN": 1.3, "w4s2rN": 1.0}}
    d = cal.derive_grouped(t, {"A": "g1", "B": "g2"}, fam=FAM)
    assert d["default"]["worst_slowdown"] == 1.3
    assert d["groups"]["g1"]["worst_slowdown"] == 1.0
    assert d["groups"]["g2"]["fixed"]["num_warps"] == 4


def test_family_declares_its_own_axes_and_codec():
    """A non-Triton adapter (MH): own axes, own key codec, own class held
    fixed in the light set, own grid object."""
    cal.register_family(
        "mhlike", env="MH_CAL", code_files=[cal.__file__], kernels=("k",),
        shapes={"x": dict(P=64), "y": dict(P=64)},
        group_key=lambda P, **_: "P%d" % P, axes=("block_p", "prefetch"),
        cfg_key=lambda c: "bp%d_pf%d" % (c["block_p"], c["prefetch"]),
        cfg_parse=lambda k: dict(zip(("block_p", "prefetch"),
                                     map(int, k[2:].split("_pf")))),
        cfg_class=lambda c: c["prefetch"],
        make_config=lambda c: ("cfg", c["block_p"], c["prefetch"]),
        config_key=lambda o: "bp%d_pf%d" % (o[1], o[2]))
    t = {"x": {"bp32_pf0": 1.0, "bp16_pf0": 1.01, "bp32_pf1": 1.0},
         "y": {"bp32_pf0": 1.2, "bp16_pf0": 1.0, "bp32_pf1": 1.0}}
    d = cal.derive_grouped(t, {"x": "P64", "y": "P64"}, fam="mhlike")
    f = d["default"]["fixed"]
    assert set(f) == {"block_p", "prefetch"}
    held = {c["prefetch"] for c in d["default"]["light"]}
    assert len(held) == 1


def test_unknown_gpu_generic_light_and_one_message(monkeypatch, capsys):
    monkeypatch.setenv("TFAM_CALIBRATION", "on")
    assert _grid("pinned") == ["light"] and _grid("light") == ["light"]

    assert cal.source(FAM, "siso_bwd").startswith("light-generic")
    err = capsys.readouterr().err
    assert err.count("calibrate_kernels.py") == 1


def test_light_search_never_varies_stages():
    d = cal.derive({s: {k: v for k, v in T[s].items()} for s in T})
    st = {cal.cfg_parse(k)["num_stages"] for k in d["light"]}
    assert len(st) == 1




def test_default_is_one_fixed_config_per_kernel_no_groups():
    """ONE fixed config per kernel per GPU (min worst-case
    slowdown over the shapes); grouping exists only as an explicit
    adapter exception.  The SSD family declares none."""
    assert cs.cal.family("ssd_mimo")["group_key"] is not None   # four groups
    cal.register_family("single", env="X_CAL", code_files=[cal.__file__],
                        kernels=("k",), shapes={"A": {}, "B": {}})
    f = cal.build_file("single", {"k": T}, 5, ["A", "B"])
    k = f["kernels"]["k"]
    assert k["groups"] == {}
    assert cal.cfg_dict_key(k["default"]["fixed"]) == "w4s1rN"
    assert k["default"]["per_shape_slowdown"]["B"] == pytest.approx(1.05)


def test_fused_adapter_codec_and_default_grids_unchanged():

    from orbit_ssm.kernels.ssd_mimo import calibration_fused as cf
    from orbit_ssm.kernels.ssd_mimo import fused_bwd_kernel as fb
    c = dict(num_warps=8, num_stages=1, maxnreg=None, HB=32)
    assert cf._parse(cf._key(c)) == c
    assert cf._key(dict(num_warps=4, num_stages=3, maxnreg=256)) == "w4s3r256"
    cfg = cf._make(c)
    assert cfg.kwargs == {"HB": 32} and cfg.num_warps == 8
    assert cf._obj_key(cfg) == cf._key(c)
    assert cf._class(c) == (1, 32)            # stages AND HB held in light
    # calibration is the default: the decorator grids are the old
    # tier lists only with the switch OFF (checked in a fresh process: the
    # grids are built at import)
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import os
        os.environ["SSD_MIMO_CALIBRATION"] = "off"
        from orbit_ssm.kernels.ssd_mimo import fused_bwd_kernel as fb
        n = lambda k: len(k.configs)
        assert n(fb._bwd_ka_kernel) == len(fb._BWD_CFGS)
        assert n(fb._bwd_k2_kernel) == len(fb._BWD_CFGS)
        assert n(fb._bwd_kbc_hb_kernel) == len(fb._HB_CFGS)
        assert n(fb._bwd_kb2_hb_kernel) == len(fb._KB2_HB_CFGS)
        key = lambda cs: [(c.num_warps, c.num_stages, c.maxnreg) for c in cs]
        assert key(fb._bwd_grid(fb._AUTOTUNE_TIER)) == key(fb._BWD_CFGS)
        assert key(fb._hb_grid(fb._AUTOTUNE_TIER)) == key(fb._HB_CFGS) or fb._AUTOTUNE_OFF
        assert len(fb._hb_grid("pinned")) == 1 and len(fb._kb2_hb_grid("pinned")) == 1
    """)
    env = dict(os.environ)       # the suite's own autotune tier applies
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert cal.family("ssd_mimo_fused")["group_key"] is not None


def test_structural_thresholds_share_layering_and_stale_rules(monkeypatch):
    monkeypatch.setenv("TFAM_CALIBRATION", "on")
    assert cal.threshold(FAM, "cp_crossover", 7.0) == (7.0, "default")
    f = cal.build_file(FAM, {"siso_bwd": T}, 5, list(T),
                       thresholds={"cp_crossover": {"value": 3.5,
                                                    "evidence": "job x"}})
    cal.write_json(cal.shipped_path(FAM, GPU), f)
    assert cal.threshold(FAM, "cp_crossover", 7.0) == (3.5, "shipped")
    f["thresholds"]["cp_crossover"]["value"] = 2.0
    cal.write_json(cal.local_path(FAM, GPU), f)
    assert cal.threshold(FAM, "cp_crossover", 7.0) == (2.0, "local")
    f["provenance"]["triton"] = "0.0"
    cal.write_json(cal.local_path(FAM, GPU), f)    # stale local ignored
    assert cal.threshold(FAM, "cp_crossover", 7.0) == (3.5, "shipped")
    monkeypatch.setenv("TFAM_CALIBRATION", "off")
    assert cal.threshold(FAM, "cp_crossover", 7.0) == (7.0, "default")


def _brute_best(t, k, pool=None):
    import itertools
    cfgs = sorted(pool or {c for s in t.values() for c in s})
    b = {s: min(t[s].values()) for s in t}
    return min(max(min(t[s].get(c, float("inf")) / b[s] for c in sel) for s in t)
               for r in range(1, k + 1) for sel in itertools.combinations(cfgs, r))


def test_exhaustive_light_set_beats_greedy_and_is_optimal():
    """Greedy (+swaps) can be suboptimal; the exhaustive subset search is
    exactly optimal.  Synthetic table: stage class held fixed (all s1)."""
    import random
    cfgs = ["w%ds1rN" % w for w in (1, 2, 4, 8)] + ["w%ds1r%d" % (w, r)
                                                   for w in (2, 4, 8) for r in (128, 256)]
    found = None
    for seed in range(400):
        rnd = random.Random(seed)
        t = {"s%d" % i: {c: rnd.uniform(1.0, 1.6) for c in cfgs} for i in range(6)}
        g = cal.derive(t, max_light=3, exhaustive_limit=0)          # greedy path
        e = cal.derive(t, max_light=3)                              # exhaustive
        if e["light_curve"][2] < g["light_curve"][2] - 1e-9:
            found = (t, g, e)
            break
    assert found, "no instance where greedy is suboptimal"
    t, g, e = found
    assert e["light_curve"][2] == pytest.approx(_brute_best(t, 3))
    assert e["light_curve"][0] == pytest.approx(_brute_best(t, 1))
    assert g["light_curve"][2] > e["light_curve"][2]


def test_exhaustive_never_worse_and_smallest_among_equals():
    t = {"A": {"w4s1rN": 1.0, "w8s1rN": 1.0}, "B": {"w4s1rN": 1.0, "w8s1rN": 1.0}}
    assert len(cal.derive(t, max_light=4)["light"]) == 1       # one config suffices
    d = cal.derive(T, max_light=2)
    st = cal.cfg_parse(d["fixed"])["num_stages"]
    pool = [c for c in {c for s in T.values() for c in s}
            if cal.cfg_parse(c)["num_stages"] == st]
    assert d["light_curve"][1] == pytest.approx(_brute_best(T, 2, pool))


def test_light_k_is_the_smallest_reaching_1_02_and_prefers_1_01():
    # one config worst 1.30; two reach 1.015 -> k=2 (<=1.02), 1.01 needs 3 (> k+1? no: k+1=3 -> prefer)
    t = {"A": {"w1s1rN": 1.0, "w2s1rN": 1.30, "w4s1rN": 1.30, "w8s1rN": 1.012},
         "B": {"w1s1rN": 1.30, "w2s1rN": 1.0, "w4s1rN": 1.30, "w8s1rN": 1.30},
         "C": {"w1s1rN": 1.30, "w2s1rN": 1.30, "w4s1rN": 1.0, "w8s1rN": 1.30}}
    d = cal.derive(t)
    assert d["light_curve"][0] > 1.2 and not d["light_flag"]
    assert d["light_k"] == 3 and d["light_curve"][2] <= 1.01 + 1e-9   # 1.02 at k=3 too: same k
    # cap reached without <=1.02 -> flagged (tuning-key problem, not a size problem)
    t2 = {"s%d" % i: {"w%ds1rN" % w: (1.0 if w == i + 1 else 1.5) for w in range(1, 6)} for i in range(5)}
    f = cal.derive(t2, max_light=3)
    assert f["light_flag"] and f["light_k"] == 3 and f["light_curve"][2] > 1.02
    g = cal.derive(t2, max_light=8)
    assert not g["light_flag"] and g["light_k"] == 5


def test_light_size_is_a_family_setting_and_large_pools_fall_back():
    cal.register_family("five", env="F_CAL", code_files=[cal.__file__],
                        kernels=("k",), shapes={"A": {}, "B": {}}, light_size=5)
    assert cal.family("five")["light_size"] == 5
    assert cal.family(FAM)["light_size"] == 8
    big = {"s%d" % i: {"w%ds1r%d" % (w, r): 1 + 0.01 * ((w * 7 + r + i * 13) % 17)
                       for w in range(1, 40) for r in range(1, 40)} for i in range(4)}
    d = cal.derive(big, max_light=4)                           # > 1e6 subsets -> greedy+swap
    assert 1 <= len(d["light"]) <= 4


def _register_occfam():
    """The synthetic occupancy family (idempotent: every cell registers it itself, none depends on another's side effect)."""
    cal.register_family(
        "occfam", env="OCC_CAL", code_files=[cal.__file__], kernels=("k",),
        shapes={"small": dict(B=4, H=8), "mid": dict(B=8, H=8),
                "big": dict(B=64, H=32), "huge": dict(B=128, H=32)},
        group_key=lambda **g: "grp", occupancy=lambda B, H, **_: B * H)


def test_occupancy_rule_two_configs_split_by_blocks_per_sm():
    """A group whose ONE config is >5% off on a shape gets two configs split
    at a blocks-per-SM threshold; the call picks its side."""
    _register_occfam()
    t = {"small": {"w4s1rN": 1.0, "w8s1rN": 1.4},
         "mid": {"w4s1rN": 1.0, "w8s1rN": 1.3},
         "big": {"w4s1rN": 1.4, "w8s1rN": 1.0},
         "huge": {"w4s1rN": 1.5, "w8s1rN": 1.0}}
    sg = {s: "grp" for s in t}
    d = cal.derive_grouped(t, sg, fam="occfam", sms=100)
    g = d["groups"]["grp"]
    assert g["worst_slowdown"] > 1.3 and "occupancy" in g
    o = g["occupancy"]
    assert o["worst_slowdown"] == pytest.approx(1.0)
    assert o["low"]["fixed"]["num_warps"] == 4 and o["high"]["fixed"]["num_warps"] == 8
    # small: 32/100=0.32, mid 0.64, big 20.5, huge 41  -> threshold between
    assert 0.64 < o["threshold"] < 20.48
    assert cal.occ_part(g, 0.5) == "low" and cal.occ_part(g, 30.0) == "high"
    assert cal.occ_part(g, None) is None
    assert cal.pick(d, "grp", "pinned", occ=0.5)[0][0]["num_warps"] == 4
    assert cal.pick(d, "grp", "pinned", occ=30.0)[0][0]["num_warps"] == 8
    # a group within 5% gets no rule
    ok = cal.derive_grouped({"small": {"w4s1rN": 1.0}, "big": {"w4s1rN": 1.02}},
                            {"small": "g", "big": "g"}, fam="occfam", sms=100)
    assert "occupancy" not in ok["groups"]["g"]


def test_occupancy_rule_runtime_selection_and_separate_caches(monkeypatch):
    _register_occfam()
    monkeypatch.setenv("OCC_CAL", "on")
    monkeypatch.setattr(cal, "device_info", lambda: dict(
        gpu=GPU, sms=100, driver="d", triton=TV, torch="t"))
    t = {"small": {"w4s1rN": 1.0, "w8s1rN": 1.4}, "mid": {"w4s1rN": 1.0, "w8s1rN": 1.3},
         "big": {"w4s1rN": 1.4, "w8s1rN": 1.0}, "huge": {"w4s1rN": 1.5, "w8s1rN": 1.0}}
    f = cal.build_file("occfam", {"k": t}, 5, list(t))
    assert "occupancy" in f["kernels"]["k"]["groups"]["grp"]
    cal.write_json(cal.local_path("occfam", GPU), f)
    g = cal.calibrated_grid("occfam", "k", "pinned", full=lambda: [], light=lambda: [],
                            generic_pin=lambda: [])
    assert len(g) >= 2                          # union holds both sides
    k = _K()
    cal.activate("occfam", "grp", {"k": k}, occupancy=0.3)
    low = cal.prune(list(g), "occfam", "k"); c_low = k.cache
    cal.activate("occfam", "grp", {"k": k}, occupancy=40.0)
    high = cal.prune(list(g), "occfam", "k"); c_high = k.cache
    assert [c.num_warps for c in low] == [4] and [c.num_warps for c in high] == [8]
    assert c_low is not c_high


class _Cfg:
    def __init__(self, w, s, r=None):
        self.num_warps, self.num_stages, self.maxnreg = w, s, r


def test_persistent_winners_survive_processes_and_respect_scope(monkeypatch):
    tuner = _K()
    tuner.configs = [_Cfg(4, 1), _Cfg(8, 1)]
    cal.activate  # noqa
    monkeypatch.setenv("TFAM_CALIBRATION", "on")
    cache = cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)
    key = ("T64", "N128")
    assert key not in cache
    tuner.configs_timings = {tuner.configs[0]: [1.00, 0.99, 1.01],
                             tuner.configs[1]: [0.80, 0.79, 0.81]}   # clear margin
    cache[key] = tuner.configs[1]                    # a search finished
    fresh = cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)  # "new process"
    assert key in fresh and fresh[key] is tuner.configs[1]
    other = cal.PersistentWinners(FAM, "siso_bwd", "g2", None, tuner)
    assert key not in other                           # other group: own winner
    tuner.configs = [_Cfg(4, 1)]                     # winner no longer a candidate
    assert key not in cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)
    tuner.configs = [_Cfg(4, 1), _Cfg(8, 1)]
    monkeypatch.setattr(cal, "device_info", lambda: dict(
        gpu=GPU, sms=1, driver="d", triton="other-triton", torch="t"))
    assert key not in cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)


def test_activate_installs_persistent_cache_only_for_searching_kernels(monkeypatch):
    monkeypatch.setenv("TFAM_CALIBRATION", "on")
    one, many = _K(), _K()
    one.configs, many.configs = [_Cfg(4, 1)], [_Cfg(4, 1), _Cfg(8, 1)]
    cal.activate(FAM, "g1", {"a": one, "b": many})
    assert type(one.cache) is dict
    assert isinstance(many.cache, cal.PersistentWinners)


def test_light_is_the_default_tier_under_calibration(monkeypatch):
    for k in ("SSD_MIMO_AUTOTUNE", "SSD_MIMO_SISO_AUTOTUNE", "SSD_MIMO_SISO_ADJ_AUTOTUNE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SSD_MIMO_CALIBRATION", "on")
    mk = lambda tier: cs.grid("siso_bwd", tier, full=lambda: ["full"],
                              light=lambda: ["light"], generic_pin=lambda: ["pin"])
    assert mk("full") == ["light"]                    # unset -> LIGHT (unknown GPU: generic light)
    monkeypatch.setenv("SSD_MIMO_SISO_AUTOTUNE", "full")
    assert mk("full") == ["full"]                     # explicit full wins
    monkeypatch.setenv("SSD_MIMO_SISO_AUTOTUNE", "0")
    assert mk("pinned") != ["full"]                   # explicit pinned: fixed/fallback
    assert cal.DEFAULT_MODE == "on"                   # Calibrated LIGHT is the default


def test_stage_hold_is_decided_automatically_per_kernel_group():
    """Stages are always measured.  Hold them at the fixed config's value
    unless holding costs > 2% worst-case vs the unconstrained optimum."""
    # SISO-forward-like: stage 3 best on the big cells, stage 2 on the small
    t = {"big1": {"w8s3rN": 1.0, "w4s2rN": 1.25}, "big2": {"w8s3rN": 1.0, "w4s2rN": 1.25},
         "small1": {"w8s3rN": 1.27, "w4s2rN": 1.0}, "small2": {"w8s3rN": 1.27, "w4s2rN": 1.0}}
    d = cal.derive(t)
    assert d["light_stages_varied"] is True and d["light_stage_cost"] > 1.2
    assert {cal.cfg_parse(c)["num_stages"] for c in d["light"]} == {2, 3}
    assert d["light_worst_slowdown"] == pytest.approx(1.0) and not d["light_flag"]
    # holding costs < 2%: stages stay held at the fixed config's value
    u = {"A": {"w8s3rN": 1.0, "w4s2rN": 1.005, "w2s3rN": 1.3},
         "B": {"w8s3rN": 1.2, "w4s2rN": 1.0, "w2s3rN": 1.0}}
    e = cal.derive(u)
    st = cal.cfg_parse(e["fixed"])["num_stages"]
    assert {cal.cfg_parse(c)["num_stages"] for c in e["light"]} == {st}
    assert e["light_stages_varied"] is False and e["light_stage_cost"] <= 1.02
    # recorded in the calibration file entry
    f = cal.build_file("single", {"k": t}, 5, ["A", "B"]) if False else None
    ent = cal._entry(d, list(t))
    assert ent["light_stages_varied"] is True and "light_stage_cost" in ent


def test_chunkpar_thresholds_feed_the_strategy_cache_lookup(monkeypatch):
    """`smem.chunk_cache_lookup` answers `chunk:cp:<device>|<suffix>` keys
    from the calibration file's thresholds (only when the switch is on)."""
    from orbit_ssm.kernels.ssd_mimo import calibration_chunkpar as cc
    from orbit_ssm.kernels.ssd_mimo import smem
    cal.register_family  # noqa
    monkeypatch.setattr(cal, "shipped_dir", lambda fam: str(cal.local_dir(fam)) + "_shipped")
    f = cal.build_file(cc.FAMILY, {}, 5, [], thresholds={
        "fwd": {"value": 345}, "fwd|ncpw": {"value": 800}, "ka_serial|warps|wb5": {"value": 4}})
    cal.write_json(cal.shipped_path(cc.FAMILY, GPU), f)
    key = "chunk:cp:%s|fwd" % GPU
    monkeypatch.setenv("SSD_MIMO_CALIBRATION", "off")
    smem._CHUNK_CACHE.pop(key, None)
    assert smem.chunk_cache_lookup(key) is None          # switch off: inert
    monkeypatch.setenv("SSD_MIMO_CALIBRATION", "on")
    assert smem.chunk_cache_lookup(key) == 345
    assert smem.chunk_cache_lookup("chunk:cp:%s|ka_serial|warps|wb5" % GPU) == 4
    assert smem.chunk_cache_lookup("chunk:cp:%s|ka" % GPU) is None   # not calibrated
    # export round trip: strategy-cache verdicts -> the file's thresholds
    path = cc.export_from_cache({"chunk:cp:%s|ksub" % GPU: 8, "chunk:cp:other|fwd": 1}, GPU)
    th = cal._read(path)["thresholds"]
    assert th["ksub"]["value"] == 8 and "fwd" not in th


def test_load_is_cached_and_revalidated_by_file_state(monkeypatch):
    """Host path: `load` runs on every scan call; it must not re-read JSON,
    yet must see a rewritten / removed / newly stale file."""
    reads = []
    real = cal._read
    monkeypatch.setattr(cal, "_read", lambda p: (reads.append(p), real(p))[1])
    _write("shipped", fixed="w8s3rN")
    cal.load(FAM)
    n = len(reads)
    for _ in range(20):
        assert cal.load(FAM)[1] == "shipped"
    assert len(reads) == n                       # served from the cache
    _write("local", fixed="w4s3r256")            # a new local file is seen
    assert cal.load(FAM)[1] == "local"
    cal.reset_local(FAM)
    assert cal.load(FAM)[1] == "shipped"
    _write("shipped", tv="0.0")                  # stale on rewrite
    assert cal.load(FAM) == (None, None)


def test_code_hash_covers_kernel_behaviour_only(tmp_path):
    """A host-only edit (wrapper, comment, docstring, new helper) keeps the
    calibration hash; a kernel-body, autotune-decorator, launch-config or
    constant edit changes it."""
    src = '''
import triton

_KEY = ["T", "N"]
SEG_C = 104.0

def _siso_configs():
    return [(4, 1), (8, 2)]

@triton.autotune(configs=_siso_configs(), key=_KEY)
@triton.jit
def kern(x_ptr, N: tl.constexpr):
    """doc"""
    v = tl.load(x_ptr)      # a comment
    tl.store(x_ptr, v * 2.0)

def host_call(x):
    """dispatch"""
    return kern[(1,)](x, 4)
'''
    f = tmp_path / "k.py"
    f.write_text(src)
    cal.register_family("hashfam", env="H_CAL", code_files=[str(f)], kernels=())
    h0 = cal.code_hash("hashfam")

    def edit(old, new):
        f.write_text(src.replace(old, new))
        return cal.code_hash("hashfam")
    assert edit("return kern[(1,)](x, 4)", "print('x')\n    return kern[(1,)](x, 8)") == h0  # host body
    assert edit("# a comment", "# another comment") == h0
    assert edit('"""dispatch"""', '"""new docs"""') == h0
    assert edit("def host_call(x):", "def other_host(y):\n    return y\n\ndef host_call(x):") == h0
    assert edit("tl.store(x_ptr, v * 2.0)", "tl.store(x_ptr, v * 3.0)") != h0        # kernel body
    assert edit("return [(4, 1), (8, 2)]", "return [(4, 1), (8, 3)]") != h0          # launch configs
    assert edit('_KEY = ["T", "N"]', '_KEY = ["T", "N", "OCC"]') != h0               # tuning key
    assert edit("SEG_C = 104.0", "SEG_C = 105.0") != h0                              # constant
    f.write_text(src)
    assert cal.code_hash("hashfam") == h0


def test_real_families_hash_is_stable_and_nonempty():
    for fam in ("ssd_mimo", "ssd_mimo_fused", "ssd_mimo_chunkpar"):
        from orbit_ssm.kernels.ssd_mimo import calibration_chunkpar, calibration_fused  # noqa
        assert len(cal.code_hash(fam)) == 16
        for p in cal.family(fam)["code_files"]:
            assert len(cal.kernel_source(p)) > 1000, p      # kernels were found


def test_calibrated_configs_replay_only_at_their_chunk_length(monkeypatch):
    """Configs tuned at T=32 are NOT replayed at T=64 (measured: Kc 0.38 ->
    1.40 ms on H100): another T searches the GENERIC light grid instead."""
    import triton
    cal.register_family("chfam", env="CH_CAL", code_files=[cal.__file__],
                        kernels=("k",), shapes={"s": dict(g=1)},
                        group_key=lambda g, **_: "grp", chunk_of=lambda g: 32)
    monkeypatch.setenv("CH_CAL", "on")
    t = {"s": {"w4s1rN": 1.0, "w8s1rN": 1.3, "w2s1rN": 1.6}}
    f = cal.build_file("chfam", {"k": t}, 5, ["s"])
    assert f["chunk"] == {"grp": 32}
    cal.write_json(cal.local_path("chfam", GPU), f)
    generic = [triton.Config({}, num_warps=2, num_stages=1), triton.Config({}, num_warps=8, num_stages=1)]
    g = cal.calibrated_grid("chfam", "k", "pinned", full=lambda: [], light=lambda: list(generic),
                            generic_pin=lambda: [])
    keys = {cal.config_key_of(c, "chfam") for c in g}
    assert {"w4s1rN", "w2s1rN", "w8s1rN"} <= keys            # calibrated + generic ride in one grid
    k = _K()
    cal.activate("chfam", "grp", {"k": k}, T=32)
    assert [cal.config_key_of(c, "chfam") for c in cal.prune(list(g), "chfam", "k")] == ["w4s1rN"]
    c32 = k.cache
    cal.activate("chfam", "grp", {"k": k}, T=64)             # uncalibrated T
    kept = {cal.config_key_of(c, "chfam") for c in cal.prune(list(g), "chfam", "k")}
    assert kept == {"w2s1rN", "w8s1rN"} and k.cache is not c32
    cal.activate("chfam", "grp", {"k": k}, T=None)           # no T given: as before
    assert [cal.config_key_of(c, "chfam") for c in cal.prune(list(g), "chfam", "k")] == ["w4s1rN"]


def test_winner_persists_only_with_a_margin_over_noise_and_the_fixed_config(monkeypatch):
    """A winner picked from one noisy pass must not outlive it (measured on
    Blackwell: a persisted 1-warp Kc winner cost 4.5% of fwd+bwd)."""
    monkeypatch.setenv("TFAM_CALIBRATION", "on")
    a, b, c = _Cfg(4, 1), _Cfg(8, 1), _Cfg(2, 1)
    tuner = _K(); tuner.configs = [a, b, c]
    fixed_key = cal.cfg_dict_key(dict(num_warps=4, num_stages=1, maxnreg=None))
    monkeypatch.setattr(cal.PersistentWinners, "_fixed_key", lambda self: fixed_key)
    persisted = lambda: bool(cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)._read())
    def run(timings):
        tuner.configs_timings = timings
        cache = cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)
        cache["k"] = min(timings, key=lambda x: timings[x][0])
        return cache["k"]
    # 1. winner (b) only 3% faster than the fixed config (a) with 10% noise: use the FIXED, persist nothing
    got = run({a: [1.00, 0.95, 1.05], b: [0.97, 0.92, 1.02], c: [1.20, 1.19, 1.21]})
    assert got is a and not persisted()
    # 2. a winner within the noise of the fixed config (here 0.5% vs 2% spread) is never used / persisted
    got = run({a: [1.00, 0.99, 1.01], b: [0.995, 0.985, 1.005], c: [1.20, 1.19, 1.21]})
    assert got is a and not persisted()
    # 3. clear margin over noise: used and persisted
    got = run({a: [1.00, 0.99, 1.01], b: [0.70, 0.69, 0.71], c: [1.20, 1.19, 1.21]})
    assert got is b and persisted()
    # 4. no calibrated fixed config (uncalibrated T): runner-up margin must exceed noise
    os.remove(cal.PersistentWinners(FAM, "siso_bwd", "g1", None, tuner)._path())
    monkeypatch.setattr(cal.PersistentWinners, "_fixed_key", lambda self: None)
    got = run({a: [1.00, 0.90, 1.10], b: [0.98, 0.88, 1.08], c: [1.20, 1.19, 1.21]})
    assert not persisted()
    got = run({a: [1.00, 0.99, 1.01], b: [0.60, 0.59, 0.61], c: [1.20, 1.19, 1.21]})
    assert persisted()


def test_pinned_mode_is_no_search_and_t_scoped(monkeypatch):
    """SSD_MIMO_CALIBRATION=pinned: the shipped fixed config of the call's
    group at the recorded T (ONE config after the per-call prune = no search);
    any other T, or an unknown GPU, gets the family's single generic pin."""
    import triton
    cal.register_family("pinfam", env="PIN_CAL", code_files=[cal.__file__],
                        kernels=("k",), shapes={"s": dict(g=1)},
                        group_key=lambda g, **_: "grp", chunk_of=lambda g: 32)
    monkeypatch.setenv("PIN_CAL", "pinned")
    assert cal.mode("pinfam") == "pinned" and cal.enabled("pinfam")
    monkeypatch.setenv("PIN_CAL", "on"); assert cal.mode("pinfam") == "on"
    monkeypatch.setenv("PIN_CAL", "off"); assert not cal.enabled("pinfam")
    monkeypatch.setenv("PIN_CAL", "pinned")
    t = {"s": {"w4s1rN": 1.0, "w8s1rN": 1.3, "w2s1rN": 1.6}}
    cal.write_json(cal.local_path("pinfam", GPU), cal.build_file("pinfam", {"k": t}, 5, ["s"]))
    pin = [triton.Config({}, num_warps=2, num_stages=1)]
    light = [triton.Config({}, num_warps=2, num_stages=1), triton.Config({}, num_warps=8, num_stages=1)]
    # an explicit `full`/`light` request does not turn the search back on
    g = cal.calibrated_grid("pinfam", "k", "full", full=lambda: ["FULL"], light=lambda: list(light),
                            generic_pin=lambda: list(pin))
    assert g != ["FULL"]
    k = _K()
    cal.activate("pinfam", "grp", {"k": k}, T=32)
    assert [cal.config_key_of(c, "pinfam") for c in cal.prune(list(g), "pinfam", "k")] == ["w4s1rN"]
    cal.activate("pinfam", "grp", {"k": k}, T=64)            # uncalibrated T -> the single generic pin
    assert [cal.config_key_of(c, "pinfam") for c in cal.prune(list(g), "pinfam", "k")] == ["w2s1rN"]
    os.remove(cal.local_path("pinfam", GPU))                 # unknown GPU: generic pin, no search
    cal._LOAD_CACHE.clear(); cal._FAST.clear()
    g = cal.calibrated_grid("pinfam", "k", "light", full=lambda: ["FULL"], light=lambda: list(light),
                            generic_pin=lambda: list(pin))
    assert [cal.config_key_of(c, "pinfam") for c in g] == ["w2s1rN"]
    assert cal.source("pinfam", "k").startswith("pinned-generic")
