"""Offline driver of the `mh` calibration family: the measurement CHILD
(`child_run(shape, rounds)`), one process per shape.

Kept out of the installed package: it builds
its calls from the repo-only `benchmarks` package.  The family's
registration, shapes, grids and config keys stay in
`orbit_ssm.kernels.hybrid_mamba_mh.calibration_mh`.  Run from a checkout
(repo root and `src` on sys.path).
"""
from __future__ import annotations

import math
import os
import time

from orbit_ssm.kernels.hybrid_mamba_mh.calibration_mh import (
    SHAPES, TS_CANDIDATES, bwd_grid, cfg_key, cfg_parse, fwd_grid, nblk)


def _events(fn, sync_ms=25.0):
    """CUDA-event ms per call of `fn`, enough calls for ~`sync_ms`."""
    import torch
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); fn(); b.record(); torch.cuda.synchronize()
    n = max(3, min(200, math.ceil(sync_ms / max(a.elapsed_time(b), 1e-3))))
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def child_run(shape: str, rounds: int) -> dict:
    """Runs INSIDE its own process: pinned tier, no verdict cache, every
    config replayed through the MH pins; rounds interleave the configs."""
    os.environ.update(HYBRID_MH_CALIBRATION="off", HYBRID_MH_AUTOTUNE="pinned",
                      HYBRID_MH_TUNE_CACHE="off")
    for k in ("HYBRID_MH_FWD_CONFIG", "HYBRID_MH_BWD_CONFIG", "MH_BWD_MAXNREG"):
        os.environ.pop(k, None)
    import statistics
    import torch
    from benchmarks import measure
    from benchmarks.bench_kernels import build_mh

    g = SHAPES[shape]
    torch.manual_seed(20261003)
    with measure.precision_ctx(g["precision"]):
        call, leaves = build_mh(g["B"], g["H"], g["P"], g["N"], g["L"],
                                dtype=measure.IO_DTYPE[g["precision"]], rot=g["rot"])
        y0 = call()
        y0 = y0[0] if isinstance(y0, tuple) else y0
        dy = torch.randn_like(y0)

        def fwd():
            with torch.no_grad():
                call()

        def fwd_bwd():
            for t in leaves:
                t.grad = None
            y = call()
            (y[0] if isinstance(y, tuple) else y).backward(dy)

        def pinned(side, c, fn):
            os.environ[f"HYBRID_MH_{side.upper()}_CONFIG"] = cfg_key(c)
            try:
                return _events(fn)
            except Exception:           # not lowerable / illegal here: absent
                return None
            finally:
                os.environ.pop(f"HYBRID_MH_{side.upper()}_CONFIG", None)

        t0 = time.perf_counter()
        fwd_bwd(); torch.cuda.synchronize()
        first = time.perf_counter() - t0
        F, Bk, T = {}, {}, {}
        ts_base = None
        for r in range(rounds):
            for c in fwd_grid():
                ms = pinned("fwd", c, fwd)
                if ms is not None:
                    F.setdefault(cfg_key(c), []).append(ms)
            f_ref = _events(fwd)        # the forward the backward rows carry
            for c in bwd_grid():
                ms = pinned("bwd", c, fwd_bwd)
                if ms is not None:
                    Bk.setdefault(cfg_key({k: v for k, v in c.items() if k != "TS"}),
                                  []).append(ms - f_ref)
            if ts_base is None:         # after round 0: the best untimed-split config
                ts_base = min(Bk, key=lambda k: Bk[k][0])
            for ts in TS_CANDIDATES:
                ms = pinned("bwd", dict(cfg_parse(ts_base), TS=ts), fwd_bwd)
                if ms is not None:
                    T.setdefault(str(ts), []).append(ms - f_ref)
    med = lambda d: {k: statistics.median(v) for k, v in d.items()}
    return dict(shape=shape, geometry=g, first_call_s=first, rounds=rounds,
                timings_ms=dict(mh_fwd=med(F), mh_bwd=med(Bk)),
                ts_ms=med(T), ts_base=ts_base, nblk=nblk(g),
                level="mh_scan op, CUDA events per call, median of rounds; "
                      "mh_bwd = fwd+bwd minus same-round fwd")
