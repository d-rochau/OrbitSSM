"""Record, per benchmark cell, the route that ACTUALLY served our arm and where its tuned launch configs came from
(benchmarks and paper tables compare ONLY the production default route against the baselines).

`served()` is called from the `post_cell` hook right after a cell's timed rounds: it re-runs the cell's closure once with the
last-route registers cleared (a resident/interleaved group leaves "the last route" at whichever arm ran last), then reads them.
`production_problem(rec)` is the emitters' gate: None when the cell ran the production default route with shipped/local
calibrated configs, else the reason it must be refused.  "General route" is the production default for R/O > 1, grouped B/C, fp32,
the half pair layout, refusals and the real backward -- it is not "old"."""
from __future__ import annotations

import os

# harness / benchmark-arm flags that are NOT route-forcing (they build the arm's inputs, not select a kernel route)
_NON_FORCING = frozenset({"SSD_MIMO_M3_EQ_PHASE_LAW", "SSD_MIMO_M3_EQ_THETA_H", "SSD_MIMO_CP_TRACE",
                          "SSD_MIMO_SMEM_STRATEGY_CACHE", "SSD_MIMO_CALIBRATION"})
_PRODUCTION_SOURCES = ("shipped", "local", "light(shipped)", "light(local)")


def forcing_env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith("SSD_MIMO_") and k not in _NON_FORCING}


def _reset():
    from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs, chunkpar, fused_bwd_kernel as fb
    cs._LAST_CALL.clear()
    chunkpar._LAST_FWD_ROUTE.clear()
    fb._CP_LAST_ROUTE.clear()
    try:
        from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g90
        g90._LAST_ROUTE.clear()
        g90._LAST_ROUTE.update(used=False, why="never consulted")
    except Exception:                                    # noqa: BLE001
        pass


def served(closure) -> dict:
    """Route + calibration record of ONE call of `closure` (our arm)."""
    import torch
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import (autograd, calibration_fused as cf, calibration_ssd as cs,
                                             chunkpar, fused_bwd_kernel as fb)
    _reset()
    n0 = len(autograd.dispatch_log())
    nb0 = len(autograd.chunk_bwd_log())
    closure()
    torch.cuda.synchronize()
    log = autograd.dispatch_log()
    disp = dict(log[-1]) if log else {}
    blog = autograd.chunk_bwd_log()
    bwd_chunk = dict(blog[-1]) if len(blog) > nb0 or (blog and len(blog) == 32) else None
    siso_fwd, siso_bwd = "siso_fwd" in cs._LAST_CALL, "siso_bwd" in cs._LAST_CALL
    gluon = False
    try:
        from orbit_ssm.kernels.ssd_mimo import gluon_sm90_siso_bwd as g90
        gluon = bool(g90._LAST_ROUTE.get("used"))
    except Exception:                                    # noqa: BLE001
        pass
    cp_f, cp_b = dict(chunkpar._LAST_FWD_ROUTE), dict(fb._CP_LAST_ROUTE)
    fwd = "siso" if siso_fwd else ("chunk_parallel" if cp_f else "general")
    bwd = "siso_gluon" if gluon else ("siso_triton" if siso_bwd else ("chunk_parallel" if cp_b else "general"))
    sources = {}
    for fam, mod in (("ssd_mimo", cs), ("ssd_mimo_fused", cf)):
        for kern, info in mod.describe_served().items():
            if isinstance(info, dict) and info.get("served") in ("search-chose", "fixed") and "grid_source" in info:
                sources[f"{fam}/{kern}"] = info["grid_source"]
    return dict(route=dict(fwd=fwd, bwd=bwd, chunk_used=disp.get("chunk_used"), chunk_mode=disp.get("chunk_mode"),
                           chunk_bwd=(bwd_chunk or {}).get("T_bwd"), chunk_bwd_record=bwd_chunk,
                           cp_fwd=cp_f or None, cp_bwd=cp_b or None, dispatch_new=len(log) - n0),
                calibration=dict(mode=cal.mode("ssd_mimo"), sources=sources, forcing_env=forcing_env()))


# MH harness/environment variables that do NOT select a route or a launch config
_MH_NON_FORCING = frozenset({"HYBRID_MH_TUNE_CACHE"})


def mh_forcing_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if (k.startswith("HYBRID_MH_") or k.startswith("HYBRID_SSM_MH_") or k.startswith("MH_"))
            and k not in _MH_NON_FORCING}


def served_mh(closure, ssm=None) -> dict:
    """Route + launch-config + calibration record of ONE call of `closure`
    (an HM arm).  `ssm` (the HybridMambaSSM, optional) adds its per-block
    dispatch list (`last_routes()`).  The fwd/bwd launch configs are the
    ones the LAST mh_scan launch used (`fwd_gluon.LAST_CONFIG` /
    `bwd_gluon.LAST_CONFIG`, reset before the call so a stale value cannot
    be reported); a flat (ssd_mimo_scan) block adds the SSD record."""
    import torch
    from orbit_ssm.kernels.hybrid_mamba_mh import autotune, bwd_gluon, fwd_gluon
    from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh as cmh
    from orbit_ssm.kernels.ssd_mimo import autograd
    fwd_gluon.LAST_CONFIG = None
    bwd_gluon.LAST_CONFIG = None
    n0 = len(autograd.dispatch_log())
    _reset()
    closure()
    torch.cuda.synchronize()
    jsonable = lambda d: None if d is None else {k: (list(v) if isinstance(v, tuple) else v)
                                                 for k, v in d.items()}
    routes = list(ssm.last_routes()) if ssm is not None else None
    rec = dict(route=dict(mh=dict(fwd_config=jsonable(fwd_gluon.LAST_CONFIG),
                                  bwd_config=jsonable(bwd_gluon.LAST_CONFIG),
                                  blocks=[list(r) if isinstance(r, tuple) else r for r in routes] if routes is not None else None,
                                  refusal=(ssm.last_refusal() if ssm is not None else None),
                                  autotune_tier=autotune.tier()),
                     ssd_dispatch_new=len(autograd.dispatch_log()) - n0),
               calibration=dict(mode=autotune.tier(),
                                sources={k: v.get("source") for k, v in cmh.describe_served().items()
                                         if isinstance(v, dict) and "source" in v},
                                mh_bwd_ts=cmh.describe_served().get("mh_bwd_ts"),
                                forcing_env=mh_forcing_env()))
    if fwd_gluon.LAST_CONFIG is None:
        rec["route"]["mh"]["fwd_note"] = "no mh_scan fwd launch observed"
    return rec


def production_problem(rec: dict | None) -> str | None:
    """Why this (our-arm) record is NOT the production default route, or None."""
    if not rec or "calibration" not in rec:
        return "no route record"
    c = rec["calibration"]
    if c.get("forcing_env"):
        return "route-forcing environment: " + ",".join(sorted(c["forcing_env"]))
    if c.get("mode") == "off":
        return "calibration off"
    bad = {k: v for k, v in c.get("sources", {}).items() if not v.startswith(_PRODUCTION_SOURCES)}
    if bad:
        return "uncalibrated launch configs: " + ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
    return None
