"""P = 1 per-cell calibration of the MH scan (`calibration_mh.p1_*`, shipped table).

Pins (what each test is evidence for):
  * the cell set IS the shared shape set at P = 1 (H = d_inner), N in {16, 128},
    bf16 and fp32, feature classes bare / m1 -- no private shapes;
  * the candidate space CONTAINS everything the runtime tuner can pick (its axes
    over their range, its coupled corners): a calibration grid narrower than the
    tuner would lose to it;
  * the lookup: exact cell -> a fresh pin dict; any other cell, a stale file, an
    explicit `full` search, calibration off -> None (the tuner serves it);
  * the envelope: only the Mamba-1 function is served from the table;
  * every SHIPPED cell is a legal pin for both kernels;
  * (GPU) a launch served from the table runs the calibrated pin, counts a hit and
    computes the tuner's answer (forward output, every gradient) -- the oracle
    suites stay the arbiter of correctness for the kernels themselves.
"""
import glob
import json
import os

import pytest
import torch

from orbit_ssm import kernel_options as ko
from orbit_ssm.kernels import calibration as cal
from orbit_ssm.kernels import shape_sets
from orbit_ssm.kernels.hybrid_mamba_mh import autotune
from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh as cm


# ------------------------------------------------------------- cell set ----
def test_cells_are_the_shared_shape_set_at_p1():
    sh = cm.P1_SHAPES
    assert len(sh) == len(shape_sets.SET) * len(cm.P1_NS) * len(cm.P1_PRECISIONS) * len(cm.P1_FEATS)
    seen = set()
    for g in sh.values():
        assert g["P"] == 1 and g["N"] in (16, 128)
        assert g["precision"] in ("bf16", "fp32") and g["cls"] in ("bare", "m1")
        seen.add((g["B"], g["H"], g["L"]))
    assert seen == {(B, d, L) for B, d, L in shape_sets.SET.values()}      # H = d_inner


def test_feature_classes_are_distinct_keys():
    k = [cm.p1_cell_key(4, 512, 16, 2048, torch.bfloat16, f) for f in cm.P1_FEATS.values()]
    assert len(set(k)) == 2
    assert cm.p1_feat(delta_softplus=True, delta_bias=None, d_skip=None, z=None,
                      h_init=None) == cm.P1_FEATS["bare"]
    assert cm.p1_feat(delta_softplus=True, delta_bias=1, d_skip=1, z=None,
                      h_init=None) == cm.P1_FEATS["m1"]


# --------------------------------------------- candidate space >= tuner's ----
@pytest.mark.parametrize("L", [1024, 2048, 8192, 32768])
@pytest.mark.parametrize("infer", [True, False])
def test_fwd_space_contains_the_tuners(L, infer):
    from orbit_ssm.kernels.hybrid_mamba_mh.bwd_gluon import auto_block_l
    H = 2048
    ckpt_l = 0 if infer else auto_block_l(1, L)
    mine = cm.p1_fwd_axes(H, L, ckpt_l, L >= 4096, infer)
    for base_bl in (128, 256, 512, 1024):
        base = dict(BLOCK_P=1, BLOCK_L=base_bl, NUM_WARPS=1, PF=0)
        theirs = autotune.t_axes(base, H, L, ckpt_l, False, "full", ts_axis=L >= 4096)
        for ax, vals in theirs.items():
            if ax == "TS" and L < 4096:
                continue
            assert set(vals) <= set(mine[ax]), (ax, vals, mine[ax])
        # the coupled corners at BLOCK_P = 1 are points of the product; BLOCK_P > 1 ones
        # are the FPACK corners
        pk = {tuple(sorted(c.items())) for c in cm.p1_fwd_fpack_corners(H, L, ckpt_l)}
        for g in autotune.t_grid(base, H, L, ckpt_l):
            if g["BLOCK_P"] == 1:
                assert g["BLOCK_L"] in mine["BLOCK_L"] and g["PF"] in mine["PF"]
                assert g["NUM_WARPS"] in mine["NUM_WARPS"]
            else:
                assert tuple(sorted(dict(g, FPACK=1).items())) in pk, g


@pytest.mark.parametrize("N", [16, 128])
@pytest.mark.parametrize("H", [128, 2048, 8192])
def test_bwd_space_contains_the_tuners(N, H):
    mine = cm.p1_bwd_axes(H, N, 64)
    ck_bytes = lambda bl: 0
    for tp in (2, 4, 8):
        base = dict(BLOCK_L=64, NS=2, TP_P=tp, NUM_WARPS=1, AGX=1, BCS=1, MAXNREG=None, HP=8)
        theirs = autotune.v3_axes(base, N, False, xs_bytes1=1, ckpt_bytes=ck_bytes,
                                  tune_block_l=False, bc_axis=True, ts_axis=True)
        for ax, vals in theirs.items():
            assert set(vals) <= set(mine[ax]), (ax, vals, mine[ax])
        for c in autotune.v3_grid(base, H, 4096, lambda hp: 64, caps=(128, 168)):
            assert c["HP"] in mine["HP"]
            assert c.get("MAXNREG", 0) in mine["MAXNREG"]
            assert c.get("TP_P", 2) in mine["TP_P"]


def test_every_axis_value_is_a_legal_pin():
    for c in (dict(BLOCK_L=512, PF=1, NUM_WARPS=2, TS=4, CKV=1, FPACK=1, BLOCK_P=2),):
        assert ko.parse_mh_config(cm.cfg_key(c), ko.MH_FWD_CONFIG_KEYS) == c
    c = dict(HP=8, TP_P=4, NUM_WARPS=1, MAXNREG=0, NS=2, TS=8, AGX=1, BCS=16)
    assert ko.parse_mh_config(cm.cfg_key(c), ko.MH_BWD_CONFIG_KEYS) == c


# ------------------------------------------------------------- lookup ------
@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(cal, "enabled", lambda fam: True)
    monkeypatch.setattr(autotune, "tier", lambda: "light")
    monkeypatch.setattr(cal, "load", lambda fam, *a: (None, None))     # no device needed
    cm._P1_INJECT.clear()
    key = cm.p1_cell_key(6, 2048, 16, 2048, torch.bfloat16, cm.P1_FEATS["m1"])
    cm._P1_INJECT[key] = dict(fwd="BLOCK_L=512,PF=1", bwd="HP=8,NS=2,TP_P=4")
    cm.P1_STATS.update(hit=0, miss=0)
    yield key
    cm._P1_INJECT.clear()


def test_lookup_serves_the_exact_cell_only(served):
    args = (6, 2048, 16, 2048, torch.bfloat16, cm.P1_FEATS["m1"])
    c = cm.p1_config("fwd", *args)
    assert c == dict(BLOCK_L=512, PF=1)
    c["BLOCK_L"] = 1                                   # callers mutate: a fresh dict each time
    assert cm.p1_config("fwd", *args) == dict(BLOCK_L=512, PF=1)
    assert cm.p1_config("bwd", *args) == dict(HP=8, NS=2, TP_P=4)
    assert cm.p1_config("fwd_trn", *args) is None      # a side the cell does not hold
    for other in ((7, 2048, 16, 2048, torch.bfloat16, cm.P1_FEATS["m1"]),      # B
                  (6, 2048, 128, 2048, torch.bfloat16, cm.P1_FEATS["m1"]),     # N
                  (6, 2048, 16, 2048, torch.float32, cm.P1_FEATS["m1"]),       # dtype
                  (6, 2048, 16, 2048, torch.bfloat16, cm.P1_FEATS["bare"])):   # features
        assert cm.p1_config("fwd", *other) is None
    assert cm.P1_STATS["hit"] == 3 and cm.P1_STATS["miss"] == 5


def test_lookup_is_off_when_off_or_when_a_full_search_is_asked(served, monkeypatch):
    args = (6, 2048, 16, 2048, torch.bfloat16, cm.P1_FEATS["m1"])
    monkeypatch.setattr(autotune, "tier", lambda: "full")
    assert cm.p1_config("fwd", *args) is None
    monkeypatch.setattr(autotune, "tier", lambda: "light")
    monkeypatch.setattr(cal, "enabled", lambda fam: False)
    assert cm.p1_config("fwd", *args) is None


def test_envelope_is_the_mamba1_function():
    ok = dict(P=1, ROT=False, pmap=False, bc_head=False, LAW_MASK=1, endpoint_own=None,
              n_off=0, nr=16, N=16, absent=(None, None))
    assert cm.p1_envelope(**ok)
    for bad in (dict(P=16), dict(ROT=True), dict(pmap=True), dict(bc_head=True),
                dict(LAW_MASK=3), dict(endpoint_own=object()), dict(n_off=2), dict(nr=8),
                dict(absent=(None, 1))):
        assert not cm.p1_envelope(**dict(ok, **bad)), bad


# ------------------------------------------------------ shipped files ------
def _shipped():
    root = os.path.join(os.path.dirname(cal.__file__), "calibration_data", "mh")
    return sorted(glob.glob(os.path.join(root, "*.json")))




def test_shipped_lm_cell_is_present():
    """The LM cell (shape_sets lm_b6_d2048_l2048, N16 bf16, the m1 call) is calibrated on every
    GPU that ships a table."""
    for f in _shipped():
        d = json.load(open(f))
        k = cm.p1_cell_key(6, 2048, 16, 2048, torch.bfloat16, cm.P1_FEATS["m1"])
        assert k in d["p1_cells"], (f, k)


# -------------------------------------------------------------- GPU ---------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_table_served_launch_computes_the_tuners_answer(monkeypatch):
    """Forward output and every gradient of a launch served from an injected table equal the
    tuner's at the tuner's own agreement bound (`autotune._agrees`), and the table counted hits."""
    from benchmarks.bench_kernels import _mh_inputs
    from orbit_ssm.kernels.hybrid_mamba_mh import mh_scan
    B, H, N, L = 2, 64, 16, 512
    torch.manual_seed(0)
    args, kw, leaves = _mh_inputs(B, H, 1, N, L, dtype=torch.bfloat16)
    kw["delta_bias"] = (torch.randn(H, device="cuda") * 0.2).requires_grad_()
    kw["d_skip"] = torch.randn(H, device="cuda").requires_grad_()
    leaves = leaves + [kw["delta_bias"], kw["d_skip"]]
    dy = None

    def run():
        nonlocal dy
        for t in leaves:
            t.grad = None
        y = mh_scan(*args, **kw)
        dy = torch.ones_like(y) if dy is None else dy
        y.backward(dy)
        return y.detach(), [t.grad.clone() for t in leaves]

    monkeypatch.setenv("HYBRID_MH_CALIBRATION", "off")
    monkeypatch.setenv("HYBRID_MH_TUNE_CACHE", "off")
    y0, g0 = run()
    key = cm.p1_cell_key(B, H, N, L, torch.bfloat16, cm.P1_FEATS["m1"])
    cm._P1_INJECT.clear()
    # deliberately NOT the tuner's winners: a legal, different config on each side
    cm._P1_INJECT[key] = dict(fwd="BLOCK_L=128,NUM_WARPS=1,PF=0,TS=1",
                              fwd_trn="BLOCK_L=128,NUM_WARPS=1,PF=1,TS=1",
                              bwd="HP=4,TP_P=2,NUM_WARPS=1,NS=1,TS=1")
    monkeypatch.setenv("HYBRID_MH_CALIBRATION", "on")
    cm.P1_STATS.update(hit=0, miss=0)
    try:
        y1, g1 = run()
    finally:
        cm._P1_INJECT.clear()
    assert cm.P1_STATS["hit"] >= 2, cm.P1_STATS             # trn forward + backward served
    assert autotune._agrees(autotune._digest(y0), autotune._digest(y1))
    assert autotune._agrees(autotune._digest(tuple(g0)), autotune._digest(tuple(g1)))
    for a, b in zip(g0, g1):
        assert torch.allclose(a.float(), b.float(), rtol=5e-2, atol=5e-2 * a.float().abs().max().item())
