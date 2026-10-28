"""Measure this device's CHUNKPAR crossover and store it (REALMODE4: measured,
never a capability literal).

For the rotation-free R1 real-mode call (the fair cell: D-skip, z gate, gated
output norm, bf16, H heads x G, N) it times fwd+bwd of the serial schedule
against the chunk-parallel one, separately for the forward (Ka serial in both
arms) and for Ka (forward serial in both arms), over a ladder of batch sizes
at a fixed long L and a fixed LM L.  The crossover of each part is the
largest serial-CTA wave count (B*H[*NGT] / SM count) at which the
chunk-parallel arm is still faster, geometric-midpoint to the first losing
rung; it is stored in the strategy cache (`smem.strategy_cache_path()`,
key "cp:<device name>|fwd" / "|ka") where `SSD_MIMO_CHUNKPAR=auto` reads it.

    python tools/cp_calibrate.py              # measure + store
    python tools/cp_calibrate.py --dry-run    # measure, print, do not store
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
ROUNDS = int(os.environ.get("CP_CAL_ROUNDS", "5"))


DTYPE = "bf16"          # --dtype: the IO dtype of the timed call (fp32 = fp32 IO, tf32 operands; keys `<part>_fp32`)


def _leaves(Bsz, L, H, P, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    d = dict(x=r(Bsz, H, P, 1, L, sc=0.5), dt=F.softplus(r(Bsz, H, L) - 3.0),
             B=r(Bsz, L, 1, N, 1, sc=0.5), C=r(Bsz, L, 1, N, sc=0.5),
             a_log=torch.log(torch.rand(H, generator=g) * 15 + 1),
             D=torch.ones(H, P, 1), z=r(Bsz, L, H * P, sc=0.7),
             w=torch.ones(H * P))
    out = {}
    for k, v in d.items():
        t = v.cuda()
        if k in ("x", "B", "C", "z") and DTYPE != "fp32":
            t = t.bfloat16()
        out[k] = t.requires_grad_(True)
    return out


def _chunk_for(P, fwd_only=False):
    """The release STRUCTURE: the production chunk of the real R=O=1 route
    (`_scan_chunk(real=True)`: the calibrated T_fwd of a training call, T of a
    forward-only call; the rule when none), or CP_CAL_CHUNK / CP_CAL_CHUNK_FWD to
    override."""
    c = int(os.environ.get("CP_CAL_CHUNK_FWD" if fwd_only else "CP_CAL_CHUNK", "0"))
    if c:
        return c
    from orbit_ssm.ssm.ssd_mimo import _scan_chunk
    return _scan_chunk(1, P, 1, needs_bwd=not fwd_only, real=True,
                       dtype=torch.float32 if DTYPE == "fp32" else None)


def _time(Bsz, L, H, P, N, env, iters=5, fwd_only=False):
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        lv = _leaves(Bsz, L, H, P, N)
        gy = None

        def step():
            nonlocal gy
            for t in lv.values():
                t.grad = None
            if fwd_only:                       # a call WITHOUT backward (inference / no_grad)
                with torch.no_grad():
                    ssd_mimo_scan(lv["x"], lv["dt"], lv["B"], lv["C"],
                                  lv["a_log"], chunk_size=_chunk_for(P, True), y_token_major=True,
                                  precision="io", bc_per_head=True,
                                  gate_z=lv["z"], d_skip=lv["D"],
                                  out_norm_w=lv["w"], out_norm_eps=1e-5)
                return
            y, _ = ssd_mimo_scan(lv["x"], lv["dt"], lv["B"], lv["C"],
                                 lv["a_log"], chunk_size=_chunk_for(P), y_token_major=True,
                                 precision="io", bc_per_head=True,
                                 gate_z=lv["z"], d_skip=lv["D"],
                                 out_norm_w=lv["w"], out_norm_eps=1e-5)
            if gy is None:
                gy = torch.randn_like(y)
            y.backward(gy)
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        # GPU time (sum of kernel self time, the benchmark's own metric), not
        # wall: wall on a shared node carries host gaps that swamp the
        # crossover.  Median of ROUNDS rounds.
        ts = []
        if os.environ.get("CP_CAL_METRIC", "wall" if fwd_only else "gpu") == "wall":
            # DEFAULT for the forward-only crossover: the GPU-time sum is blind to host-bound calls (the chunk-parallel
            # forward has more launches: at lm_b6 it is 0.395 ms of GPU time but 0.66 ms of wall, the general forward 0.46 / 0.59).
            # EAGER steady state (what an un-graphed caller and `do_bench` see): CUDA-event wall over a back-to-back loop, so a path
            # with more launches / host work than its GPU time pays for them.  The GPU-time metric below is the CUDA-graph view.
            for _ in range(ROUNDS):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                e0.record()
                for _ in range(iters * 8):
                    step()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) / (iters * 8))
            ts.sort()
            return ts[len(ts) // 2]
        for _ in range(ROUNDS):
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
                for _ in range(iters):
                    step()
                torch.cuda.synchronize()
            tot = sum(ev.self_device_time_total for ev in pr.key_averages()
                      if ev.device_type == torch.autograd.DeviceType.CUDA)
            ts.append(tot / 1e3 / iters)
        ts.sort()
        return ts[len(ts) // 2]
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fit_rule(rows, sm, H=None, T=64):
    """Grid-fit `chunkpar.cp_rule` (waves < W and nc/max(w,1) >= R) by total
    GPU time over the ladder rows (each row: B, L, serial_ms, cp_ms)."""
    from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_rule
    best = None
    for W in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 1e4):
        for R in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 1e6):
            tot = 0.0
            for r in rows:
                w = r["B"] * r.get("H", H) / sm
                nc = r["L"] // T
                tot += r["cp_ms"] if cp_rule(w, nc, W, R) else r["serial_ms"]
            if best is None or tot < best[2] - 1e-9:
                best = (W, R, tot)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--L-extra", type=int, nargs="*", default=[512])
    ap.add_argument("--ksub-fwd", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--ksub-ka", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--kbc-sched", type=int, nargs="*", default=[0, 8, 24, 27])
    ap.add_argument("--ka-warps", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--parts", nargs="+", default=["kawarps", "fwd", "fwd_nograd", "ka", "kbc"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"),
                    help="fp32: time fp32-IO calls (tf32 operands) and store the crossovers under `<part>_fp32` (fwd, fwd_nograd, "
                         "ka only: the serial-Ka warps verdict and the Kbc-hb schedule are 16-bit statements)")
    ap.add_argument("--write-shipped", action="store_true",
                    help="export the thresholds into the REPO's shipped calibration file "
                         "(default: the local file)")
    a = ap.parse_args()
    global DTYPE
    DTYPE = a.dtype
    if DTYPE == "fp32":
        a.parts = [p for p in a.parts if p in ("fwd", "fwd_nograd", "ka")]
    sfx = "_fp32" if DTYPE == "fp32" else ""
    from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_calibration_store
    from orbit_ssm.kernels.ssd_mimo.fused_fwd_kernel import _sm_count
    dev = torch.cuda.current_device()
    sm = _sm_count(dev)
    base = {"SSD_MIMO_SISO_FAST": "1", "SSD_MIMO_CHUNKPAR_WAVES": ""}
    # "serial" of the forward parts is what production runs when the chunk-parallel forward is NOT chosen:
    # the GENERAL fused forward (the SISO schedule serves a real call only via chunk-parallel,
    # `siso_route.siso_enabled(real_cp)`), not the bare serial SISO scan
    arms = {
        "fwd": ({"SSD_MIMO_CHUNKPAR": "0", "SSD_MIMO_CHUNKPAR_KA": "0", "SSD_MIMO_SISO_FAST": "0"},
                {"SSD_MIMO_CHUNKPAR": "1", "SSD_MIMO_CHUNKPAR_KA": "0"}),
        "fwd_nograd": ({"SSD_MIMO_CHUNKPAR": "0", "SSD_MIMO_CHUNKPAR_KA": "0", "SSD_MIMO_SISO_FAST": "0"},
                       {"SSD_MIMO_CHUNKPAR": "1", "SSD_MIMO_CHUNKPAR_KA": "0"}),
        "ka": ({"SSD_MIMO_CHUNKPAR": "0", "SSD_MIMO_CHUNKPAR_KA": "0"},
               {"SSD_MIMO_CHUNKPAR": "0", "SSD_MIMO_CHUNKPAR_KA": "1"}),
    }
    # the shared shape set (src/orbit_ssm/kernels/shape_sets.py), real R = O = 1,
    # P = 64, N = a.N; a cell that does not fit is dropped by name
    from orbit_ssm.kernels import shape_sets as ss
    CELLS = [(t, ss.geometry(t)) for t in ss.SET]
    dropped = {}

    def cells():
        for t, g in CELLS:
            if t not in dropped:
                yield t, g["B"], g["H"], g["L"]

    def timed(t, *args, **kw):
        try:
            return _time(*args, **kw)
        except torch.cuda.OutOfMemoryError as e:
            dropped[t] = str(e).splitlines()[0][:200]
            torch.cuda.empty_cache()
            return None
    result = {"device": torch.cuda.get_device_name(dev), "sm": sm, "rows": []}
    # the SERIAL Ka launch config (its in-process autotune pick is not
    # reproducible on every device): Ka forced serial, one warps value each.
    # FIRST, so the serial arms of every later fit are themselves stable
    if a.ka_warps and "kawarps" in a.parts:
        # per occupancy bucket (the Ka autotune key's `KA_WB`): the best
        # config moves 2x with the grid, so ONE device-wide value is wrong
        from orbit_ssm.kernels.ssd_mimo import fused_bwd_kernel as _fbk
        tot = {}                                   # wb -> {warps: ms}
        for tag, B, H, L in cells():
            for w in a.ka_warps:
                # the autotune cache is keyed by shapes only: drop it so
                # the prune re-reads SSD_MIMO_KA_WARPS for this arm
                _fbk._bwd_ka_kernel.cache.clear()
                t = timed(tag, B, L, H, a.P, a.N, dict(
                    base, SSD_MIMO_CHUNKPAR="0", SSD_MIMO_CHUNKPAR_KA="0",
                    SSD_MIMO_KA_WARPS=str(w)))
                if t is None:
                    break
                wb = _fbk._KA_LAST_WB
                d = tot.setdefault(wb, {})
                d[w] = d.get(w, 0.0) + t
                print(json.dumps(dict(part="ka_serial", shape=tag, B=B, H=H, L=L,
                                      wb=wb, warps=w, ms=round(t, 4))),
                      flush=True)
        best = {wb: min(d, key=d.get) for wb, d in tot.items()}
        result["ka_serial"] = dict(warps_by_wb=best, totals=tot)
        if not a.dry_run:
            from orbit_ssm.kernels.ssd_mimo.smem import chunk_cache_store
            from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_calibration_key
            for wb, w in best.items():
                chunk_cache_store(cp_calibration_key(dev, "ka_serial")
                                  + f"|warps|wb{wb}", int(w))
        # every later arm times serial Ka at the calibrated config (dry
        # runs keep the in-process autotune, now keyed per bucket)
        _fbk._bwd_ka_kernel.cache.clear()
    for part, (off, on) in arms.items():
        if part not in a.parts:
            continue
        rows = []
        fo = part == "fwd_nograd"
        ks = a.ksub_ka if part == "ka" else a.ksub_fwd
        kkey = "SSD_MIMO_CP_KSUB_" + ("FWD" if fo else part.upper())
        for tag, B, H, L in cells():
            w = B * H / sm       # serial CTAs (NGT = 1 at G = 64) per SM
            t0 = timed(tag, B, L, H, a.P, a.N, dict(base, **off), fwd_only=fo)
            if t0 is None:
                continue
            cps = {k: timed(tag, B, L, H, a.P, a.N, dict(base, **on, **{kkey: str(k)}), fwd_only=fo)
                   for k in ks}
            if any(v is None for v in cps.values()):
                continue
            row = dict(part=part, shape=tag, B=B, H=H, L=L, waves=round(w, 3),
                       work=B * H * (L // 64), serial_ms=round(t0, 4),
                       cp_ms_by_k={str(k): round(v, 4) for k, v in cps.items()})
            rows.append(row)
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
        # the big-chunk length: best total chunk-parallel time over the ladder
        kbest = min(ks, key=lambda k: sum(r["cp_ms_by_k"][str(k)] for r in rows))
        for r in rows:
            r["cp_ms"] = r["cp_ms_by_k"][str(kbest)]
        _W, _R, _ = fit_rule(rows, sm, T=_chunk_for(a.P, fo))
        W, M = _W, _R
        result[part] = dict(waves=W, nc_per_wave=M, ksub=kbest)
        if fo:
            # the WORK GUARD: the general forward below `wmin` chunks (B*H*nc) UNLESS the serial walk is long (nc >= `ncmin`); thresholds are
            # cell values (0 = no guard), fitted on the SCALE-FREE criterion sum(log time) -- a millisecond of a large cell is not worth a
            # millisecond of a small one -- ascending, so a tie keeps the smaller guard
            from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_rule
            Tn = _chunk_for(a.P, fo)
            works = sorted({0} | {r["B"] * r["H"] * (r["L"] // Tn) for r in rows})
            ncs = sorted({0} | {r["L"] // Tn for r in rows})
            best_g = None
            for wm in works:
                for ncm in ncs:
                    tot = 0.0
                    for r in rows:
                        nc_ = r["L"] // Tn
                        use = cp_rule(r["B"] * r["H"] / sm, nc_, W, M) and (r["B"] * r["H"] * nc_ >= wm or nc_ >= ncm)
                        tot += math.log(r["cp_ms"] if use else r["serial_ms"])
                    if best_g is None or tot < best_g[2] - 1e-3:
                        best_g = (wm, ncm, tot)
            result[part]["wmin_chunks"], result[part]["ncmin"] = best_g[0], best_g[1]
            if not a.dry_run:
                from orbit_ssm.kernels.ssd_mimo.smem import chunk_cache_store
                from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_calibration_key
                chunk_cache_store(cp_calibration_key(dev, part + sfx) + "|wmin", int(best_g[0]))
                chunk_cache_store(cp_calibration_key(dev, part + sfx) + "|ncmin", int(best_g[1]))
        if not a.dry_run:
            cp_calibration_store(dev, part + sfx, min(W, 1e4), M, {"rows": rows})
            from orbit_ssm.kernels.ssd_mimo.smem import chunk_cache_store
            from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_calibration_key
            if not fo and not sfx:      # (a forward-only call serves the same cp forward kernels at the `fwd` ksub;
                #                             the ksub verdict stays the 16-bit one)
                chunk_cache_store(cp_calibration_key(dev, part) + "|ksub", int(kbest))
    # the Kbc-hb body schedule (16-bit operands), both passes chunk-parallel
    # at their calibrated lengths, over the same ladder
    if a.kbc_sched and "kbc" in a.parts:
        rows = []
        cp_on = {"SSD_MIMO_CHUNKPAR": "1", "SSD_MIMO_CHUNKPAR_KA": "1"}
        for p_ in ("fwd", "ka"):       # else the cached calibrated length
            if p_ in result:
                cp_on["SSD_MIMO_CP_KSUB_" + p_.upper()] = str(result[p_]["ksub"])
        tot = {}
        for tag, B, H, L in cells():
            for sc in a.kbc_sched:
                t = timed(tag, B, L, H, a.P, a.N, dict(
                    base, **cp_on, SSD_MIMO_BWD_KBC_SCHED=str(sc)))
                if t is None:
                    break
                tot[sc] = tot.get(sc, 0.0) + t
                rows.append(dict(part="kbc", shape=tag, B=B, H=H, L=L, sched=sc,
                                 ms=round(t, 4)))
                print(json.dumps(rows[-1]), flush=True)
        best_sc = min(tot, key=tot.get)
        result["kbc"] = dict(sched=best_sc, totals=tot)
        if not a.dry_run:
            from orbit_ssm.kernels.ssd_mimo.smem import chunk_cache_store
            from orbit_ssm.kernels.ssd_mimo.chunkpar import cp_calibration_key
            chunk_cache_store(cp_calibration_key(dev, "kbc") + "|sched",
                              int(best_sc))
    if not a.dry_run:
        # structural thresholds -> the calibration file (src/orbit_ssm/kernels/calibration.py)
        from orbit_ssm.kernels.ssd_mimo import smem as _smem
        from orbit_ssm.kernels.ssd_mimo.calibration_chunkpar import export_from_cache
        _smem._disk_load_once()
        print("CP_THRESHOLDS_FILE", export_from_cache(
            _smem._CHUNK_CACHE, torch.cuda.get_device_name(dev),
            write_shipped=a.write_shipped, dropped=dropped), flush=True)
    result["dropped_shapes"] = dropped
    print("CP_CALIBRATION " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
