#!/usr/bin/env python
"""The measurements behind the rotating R = O = 1 route table (`tools/calibrate_routes.py`, family `ssd_mimo_route`) on THIS GPU.

    python tools/calibration_drivers/routes.py all  --dtype bf16 --work DIR [--write-shipped]   # timings + eligibility + the table
    python tools/calibration_drivers/routes.py ab   --dtype bf16 --out DIR/routes.jsonl          # timings only
    python tools/calibration_drivers/routes.py elig --dtype bf16 --dir DIR/elig                  # eligibility only

`ab`: per cell of the shared shape set (`orbit_ssm.kernels.shape_sets`), the routes A (SISO), B (rotation split) and C
(in-kernel rotation) at their chunk lengths and the official Mamba-3 module, forward and forward + backward, with and without
the static read offset theta_h; one process per cell, one jsonl record per (arm, theta_h, pass).  fp32 IO (tf32 operands) also
times the SISO route at T = 32.
`elig`: per route, the fp64 stress (reset / large_phase / euler_limit draws of the exact complex R1O1 function against the fp64
Mamba-3 reference) and the C10 check of `tools/ssd_check.py` (8 draws, theta_h off and on); `calibrate_routes.py` reads the logs.
A route that fails either is ineligible: the table never sends a call to it.
`all`: both, then `tools/calibrate_routes.py` (local calibration file unless --write-shipped).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

SPLIT = {"SSD_MIMO_SISO_FAST": "0", "SSD_MIMO_ROT_BWD": "split"}
#: arm -> (env, chunk); the names are the ones `calibrate_routes.py` maps to routes (ARM / ARM32)
ARMS = {"bf16": {"A_T64": ({"SSD_MIMO_SISO_FAST": "1"}, 64), "B_T32": (SPLIT, 32), "B_T64": (SPLIT, 64),
                 "C_T32": ({"SSD_MIMO_SISO_FAST": "0"}, 32), "C_T64": ({"SSD_MIMO_SISO_FAST": "0"}, 64)},
        "fp32": {"A_T32": ({"SSD_MIMO_SISO_FAST": "1"}, 32), "A_T64": ({"SSD_MIMO_SISO_FAST": "1"}, 64),
                 "B_T32": (SPLIT, 32), "B_T64": (SPLIT, 64),
                 "C_T32": ({"SSD_MIMO_SISO_FAST": "0"}, 32), "C_T64": ({"SSD_MIMO_SISO_FAST": "0"}, 64)}}
#: eligibility log name -> (env, chunk): `calibrate_routes.eligibility` / `eligibility32` read stress_<name>.log, c10_<name>_*.log
ELIG = {"bf16": {"A": ({"SSD_MIMO_SISO_FAST": "1"}, 64), "B32": (SPLIT, 32), "B64": (SPLIT, 64),
                 "C32": ({"SSD_MIMO_SISO_FAST": "0"}, 32), "C64": ({"SSD_MIMO_SISO_FAST": "0"}, 64)},
        "fp32": {"A32": ({"SSD_MIMO_SISO_FAST": "1"}, 32), "A64": ({"SSD_MIMO_SISO_FAST": "1"}, 64),
                 "B32": (SPLIT, 32), "B64": (SPLIT, 64),
                 "C32": ({"SSD_MIMO_SISO_FAST": "0"}, 32), "C64": ({"SSD_MIMO_SISO_FAST": "0"}, 64)}}
ARM_KEYS = ("SSD_MIMO_SISO_FAST", "SSD_MIMO_ROT_BWD", "SSD_MIMO_ROT_SPLIT_DTYPE", "SSD_MIMO_M3_EQ_THETA_H", "SSD_MIMO_CHUNK")

#: the C10 cell of `tools/ssd_check.py`: the production rotating R = O = 1 mixer (half the rows rotating, head_dynamic decay,
#: B/C norms, per-head biases, both gates, trapezoid, learned theta_freq + theta_h), at 4 heads of 64
C10_KWARGS = dict(
    d_ssm_in=1, d_ssm_out=1, n_groups=64, n_heads=4, d_head=64,
    row_blocks=[{"frac": 0.5, "decay": "head_dynamic", "complex": True},
                {"frac": 0.5, "decay": "head_dynamic", "complex": False}],
    b_norm=True, c_norm=True, bc_bias="head", b_gate=True, c_gate=True, bc_bias_init="auto",
    r_mix="learned", r_mix_dyn=True, dt_min=1e-3, dt_max=0.1, dt_init_floor=1e-4,
    a_proj_bias=True, a_dyn_per_sibling=False, a_act="heavy_tail", trap_proj_bias=True, write_rule="trapezoid",
    skip=True, skip_sharing="input_output", pair_layout="interleaved", proj_source="layer_input", bc_conv=False,
    out_norm=False, theta_per_head=False, use_theta_h=True, learn_theta_freq=True, omega_layout="pair", theta_init="kaiming",
    execution=dict(backend="auto", precision="io", chunk_size="auto",
                   kernel_options=dict(m3_kernel="native", compile_token_maps=False, allow_chunk_autotune=True)))
C10_ARGS = ["--R", "1", "--O", "1", "--B", "1", "--L", "128", "--d-model", "128", "--d-inner", "256", "--N", "128", "--draws", "8"]


def _clean_env(keep=("SSD_MIMO_CALIBRATION",)):
    for k in [k for k in os.environ if k.startswith("SSD_MIMO_") and k not in keep]:
        del os.environ[k]
    os.environ["SSD_MIMO_M3_EQ_PHASE_LAW"] = "1"


def cells():
    from orbit_ssm.kernels import shape_sets as s
    return [(g["B"], g["L"], g["H"]) for g in map(s.geometry, s.SET)]


def c10_command(dtype, theta_h):
    kw = []
    for k, v in C10_KWARGS.items():
        kw += ["--ssm-kw", "%s=%r" % (k, v)]
    return ([sys.executable, os.path.join(ROOT, "tools", "ssd_check.py")] + C10_ARGS + ["--dtype", dtype] + kw
            + (["--theta-h-std", "0.3"] if theta_h else []))


# ------------------------------------------------------------------ one cell / one route (child processes) --
def _bench(fn, rounds=5, n=20):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    v = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(n):
            fn()
        b.record()
        b.synchronize()
        v.append(a.elapsed_time(b) / n)
    return statistics.median(v)


def child_ab(dtype, B, L, H, out):
    """Timings of one shared-set cell (the former followup/route_ab.py, route_ab32.py)."""
    _clean_env()
    import torch
    from benchmarks import baselines as bl
    from benchmarks.harness import make_pass
    tdt, prec, n = (torch.bfloat16, "io", 20) if dtype == "bf16" else (torch.float32, "tf32", 10)
    meta = dict(gpu=torch.cuda.get_device_name(), B=B, L=L, H=H, dtype=dtype)

    def emit(**r):
        with open(out, "a") as f:
            f.write(json.dumps(dict(meta, **r)) + "\n")

    def setenv(env, thh):
        for k in ARM_KEYS:
            os.environ.pop(k, None)
        os.environ.update(env)
        if thh:
            os.environ["SSD_MIMO_M3_EQ_THETA_H"] = "1"
    torch.manual_seed(20261003)
    try:
        call, leaves = bl.build_m3_eq(B, H, 64, 128, L, arm="m3", chunk=64, angles="m3", trap_off=False, dtype=tdt, precision=prec)
        for p in ("fwd", "fwd_bwd"):
            emit(arm="official_m3", pass_=p, ms=_bench(make_pass(call, leaves, p), n=n))
        del call, leaves
    except Exception as e:                                  # noqa: BLE001  (recorded; the table needs only our arms)
        emit(arm="official_m3", error=type(e).__name__, msg=str(e).splitlines()[0][:200] if str(e) else "")
    torch.cuda.empty_cache()
    for thh in (False, True):
        for name, (env, chunk) in ARMS[dtype].items():
            setenv(env, thh)
            try:
                call, leaves = bl.build_m3_eq(B, H, 64, 128, L, arm="ours", chunk=chunk, angles="m3", trap_off=False,
                                              dtype=tdt, precision=prec)
                for p in ("fwd", "fwd_bwd"):
                    fn = make_pass(call, leaves, p)
                    setenv(env, thh)
                    emit(arm=name, theta_h=thh, pass_=p, ms=_bench(fn, n=n))
                del call, leaves
            except torch.cuda.OutOfMemoryError as e:
                emit(arm=name, theta_h=thh, error="oom", msg=str(e).splitlines()[0][:160])
            except Exception as e:                          # noqa: BLE001  (a route that cannot run this cell has no time)
                emit(arm=name, theta_h=thh, error=type(e).__name__, msg=str(e).splitlines()[0][:300] if str(e) else "")
            torch.cuda.empty_cache()


def _stress_errs(chunk, dtype, prec, pattern, seed):
    import torch
    from benchmarks import baselines as bl
    from orbit_ssm.ssm.hybrid_mamba import _m3_reference
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float32
    call, leaves = bl.build_m3_eq(1, 4, 64, 128, 1024, arm="ours", chunk=chunk, angles="m3", trap_off=False, dtype=tdt,
                                  precision=prec, seed=seed)
    t = dict(zip(bl._M3_EQ_ORDER, leaves))
    with torch.no_grad():
        if pattern == "reset":
            t["ADT"][:, :, 64] = -40
        elif pattern == "large_phase":
            t["DT"].fill_(5)
            t["ADT"].fill_(-.5)
        else:
            t["Trap"].fill_(-12)
    refs = [x.detach().double().requires_grad_() for x in leaves]
    y, r = call(), _m3_reference(*refs)
    dy = torch.randn((y.shape[0], y.shape[2], y.shape[3], y.shape[1]), device=y.device, dtype=y.dtype,
                     generator=torch.Generator("cuda").manual_seed(seed + 71)).permute(0, 3, 1, 2)
    g = torch.autograd.grad(y, leaves, dy)
    rg = torch.autograd.grad(r, refs, dy.double())
    return {nm: (a_.double() - b_).norm().item() / max(b_.norm().item(), 1e-300)
            for nm, a_, b_ in zip(("y", *bl._M3_EQ_ORDER), (y, *g), (r, *rg))}


def child_stress(dtype, name, chunk, env):
    """fp64 stress of one route (the former followup/route_stress.py; fp32: calkit/stress32.py).
    bf16: normwise error < 2^-8 sqrt(64) sqrt(2) on every tensor and draw (5 draws per pattern).
    fp32 (tf32 operands): < 2^-11 sqrt(64) sqrt(2) on every draw (8 draws), and the median ratio to the general in-kernel route
    on the same inputs <= 1.25 for every tensor above a quarter of that bound (the route must not be worse than the general one)."""
    _clean_env()
    os.environ.update(env)
    import torch
    failed, worst = [], {}
    if dtype == "bf16":
        bound = 2 ** -8 * math.sqrt(64) * math.sqrt(2)
        for pattern in ("reset", "large_phase", "euler_limit"):
            for seed in range(5):
                for nm, e in _stress_errs(chunk, "bf16", "io", pattern, seed).items():
                    worst[pattern + ":" + nm] = max(worst.get(pattern + ":" + nm, 0.0), e)
                    if not math.isfinite(e) or e >= bound:
                        failed.append([pattern, seed, nm, e])
        print("ROUTE_STRESS", name, "passed" if not failed else "FAILED", json.dumps(dict(worst=worst, failures=failed[:20])),
              flush=True)
        return 0 if not failed else 1
    bound = 2 ** -11 * math.sqrt(64) * math.sqrt(2)
    draws = int(os.environ.get("DRAWS", "8"))
    general = {"SSD_MIMO_SISO_FAST": "0"}
    for pattern in ("reset", "large_phase", "euler_limit"):
        ratios, over = {}, set()
        for seed in range(draws):
            for k in ("SSD_MIMO_SISO_FAST", "SSD_MIMO_ROT_BWD", "SSD_MIMO_CHUNK"):
                os.environ.pop(k, None)
            os.environ.update(env)
            e = _stress_errs(chunk, "fp32", "tf32", pattern, seed)
            for k in ("SSD_MIMO_SISO_FAST", "SSD_MIMO_ROT_BWD", "SSD_MIMO_CHUNK"):
                os.environ.pop(k, None)
            os.environ.update(general)
            eg = _stress_errs(chunk, "fp32", "tf32", pattern, seed)
            for nm, v in e.items():
                if not math.isfinite(v) or v >= bound:
                    failed.append([pattern, seed, nm, "flat", v, bound])
                if v >= bound / 4:
                    over.add(nm)
                ratios.setdefault(nm, []).append(v / max(eg[nm], 1e-300))
        for nm in over:
            med = statistics.median(ratios[nm])
            if med > 1.25:
                failed.append([pattern, nm, "median ratio vs the general route", med])
    print("STRESS32", name, "passed" if not failed else "FAILED",
          json.dumps(dict(failures=failed[:30], gpu=torch.cuda.get_device_name(), draws=draws, chunk=chunk, env=env)), flush=True)
    return 0 if not failed else 1


# ------------------------------------------------------------------------------------------------- drivers --
def run_ab(dtype, out, only_cells=None):
    for B, L, H in (only_cells or cells()):
        subprocess.run([sys.executable, os.path.abspath(__file__), "_ab", dtype, str(B), str(L), str(H), out], check=False)


def run_elig(dtype, d):
    os.makedirs(d, exist_ok=True)
    for name, (env, chunk) in ELIG[dtype].items():
        with open(os.path.join(d, "stress_%s.log" % name), "w") as f:
            subprocess.run([sys.executable, os.path.abspath(__file__), "_stress", dtype, name, str(chunk)]
                           + ["%s=%s" % kv for kv in env.items()], stdout=f, stderr=subprocess.STDOUT, check=False)
        for tag, thh in (("noth", False), ("thh", True)):
            cmd = c10_command(dtype, thh) + ["--evidence-out", os.path.join(d, "c10_%s_%s.json" % (name, tag))]
            child_env = {k: v for k, v in os.environ.items() if not k.startswith("SSD_MIMO_") or k == "SSD_MIMO_CALIBRATION"}
            child_env.update(env, SSD_MIMO_CHUNK=str(chunk))
            with open(os.path.join(d, "c10_%s_%s.log" % (name, tag)), "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=child_env, check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["all", "ab", "elig", "_ab", "_stress"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--write-shipped", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "_ab":
        dtype, B, L, H, out = a.rest
        return child_ab(dtype, int(B), int(L), int(H), out)
    if a.cmd == "_stress":
        dtype, name, chunk, *env = a.rest
        return child_stress(dtype, name, int(chunk), dict(kv.split("=", 1) for kv in env))
    if a.cmd == "ab":
        return run_ab(a.dtype, a.out)
    if a.cmd == "elig":
        return run_elig(a.dtype, a.dir)
    work = a.work or "."
    os.makedirs(work, exist_ok=True)
    ab, elig = os.path.join(work, "routes_%s.jsonl" % a.dtype), os.path.join(work, "elig_%s" % a.dtype)
    if os.path.exists(ab):
        os.remove(ab)                                   # the table is fitted on this run's timings only
    run_ab(a.dtype, ab)
    run_elig(a.dtype, elig)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "calibrate_routes.py"), "--ab", ab, "--elig", elig, "--dtype", a.dtype]
    if a.write_shipped:
        cmd.append("--write-shipped")
    if a.dtype == "bf16":
        cmd.append("--allow-fail")                      # as the 2026-10 runs: a misrouted cell is reported, the rule still written
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main() or 0)
