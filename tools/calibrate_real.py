#!/usr/bin/env python
"""Calibrate the REAL (Mamba-2 class, rotation-free) R = O = 1 route on THIS GPU: group `real_R1O1`.

Two kinds of per-GPU choice, both stored in the calibration files like the existing thresholds:

  * STRUCTURE -- the chunk length T of the real route, among {32, 64}, separately for
      train_fwd   the forward of a training call,
      train_bwd   the backward of a training call (T_fwd != T_bwd is the (T_fwd, T_bwd) pair of
                  `SSD_MIMO_CHUNK_PAIR` / `chunk_size_bwd`),
      fwd_only    a call without backward (inference / no_grad),
    thresholds `real_r1o1|T|<kind>` of family `ssd_mimo_chunkpar` (read by `_scan_chunk(real=True)`).
  * LEVERS (family `ssd_mimo_levers`, units `real_*`, see `levers.REAL_UNITS`):
      real_gluon_kbc (bwd)  real_gate_dy_io (bwd)  real_kc_shared_cb (bwd)  real_dg_lo (bwd)
      real_cp_g_lo (fwd)
    Decision rule of `calibrate_levers.py`: ON iff the leave-one-out geomean gain on the unit's phase
    is >= 1.015 and no shape loses more than 3 %; a unit that fails the fp64 oracle check on this GPU is
    INELIGIBLE (stored 0, whatever its timing); verification of the chosen set against the all-off base.

STRUCTURAL RULE: `choose` also fits `T = T_hi iff L <= L_max and B*H*L >= W_min` over the screen (worst shape <= 1.10 x its
per-shape best, geomean gain >= TOL_RULE over the best single T), stored as `real_r1o1|Trule|{T_hi, L_max, W_min}`; `ablate` then ablates the levers
at the rule's T_hi on the shapes it sends there (stage `ablate_hi`, group `real_R1O1_hi`) and at the base T on the others.  `screen --extra` adds
the EXTRA shapes around the boundaries.

Stages (every one resumable: its per-shape raw results live in --out):

    python tools/calibrate_real.py elig   --out DIR      # fp64 oracle: base / all-on / solo (ssd_check, ssd_check_saturated)
    python tools/calibrate_real.py screen --out DIR      # (T_fwd, T_bwd) in {32,64}^2 x {base, all-on}, per shape
    python tools/calibrate_real.py choose --out DIR      # pick the structure -> DIR/structure.json (+ --write)
    python tools/calibrate_real.py ablate --out DIR      # leave-one-out lever ablation AT the chosen structure
    python tools/calibrate_real.py decide --out DIR [--write-shipped]   # thresholds into the calibration files

Shapes: `src/orbit_ssm/kernels/shape_sets.py` (12 grid + 6 LM cells, P = 64, N = 128, bf16, the production
FAIR real call of `benchmarks.baselines.build_real_fair`); a cell that does not fit is dropped by name.
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, str(ROOT) + "/src")

GROUP = "real_R1O1"
TOL_ON, TOL_WORST = 1.015, 0.97
TOL_RULE, RULE_WORST = 1.03, 1.10            # the structural T rule: geomean gain over the best single T, worst shape vs its per-shape best
#: extra screen points (B, d_inner, L) around the rule's boundaries (L = 4096 / 6144 / 8192 at the LM and medium widths, one small cell)
EXTRA = {"x_b8_d2048_l4096": (8, 2048, 4096), "x_b4_d2048_l8192": (4, 2048, 8192), "x_b8_d2048_l8192": (8, 2048, 8192),
         "x_b4_d512_l4096": (4, 512, 4096), "x_b8_d2048_l6144": (8, 2048, 6144), "x_b2_d8192_l4096": (2, 8192, 4096)}
GROUP_HI = "real_R1O1_hi"
N_STATE = 128
INCUMBENT = (32, 32)                      # (T_fwd, T_bwd) of the uncalibrated rule at G = 64
COMBOS = [(32, 32), (64, 64), (64, 32), (32, 64)]
UNIT_ENV = {"real_gluon_kbc": "GLUON_KBC", "real_gate_dy_io": "GATE_DY_IO",
            "real_kc_shared_cb": "BWD_KC_SHARED_CB", "real_dg_lo": "BWD_DG_LO",
            "real_cp_g_lo": "CP_G_LO"}
UNITS = list(UNIT_ENV)
PHASE = {u: ("fwd" if u == "real_cp_g_lo" else "bwd") for u in UNITS}
#: THE fp32 AXIS (`--dtype fp32`): the same stages at fp32 IO (tf32 operands, torch's default dtype), stored under the
#: `|fp32` keys of `calibration_chunkpar` (T, rule) and the lever group `real_R1O1_fp32`.  Of the five units only `real_kc_shared_cb`
#: can act on an fp32 call (gate_dy_io / cp_g_lo / dg_lo store a 16-bit buffer, the Gluon Kbc is a 16-bit body: their guards are
#: false at fp32), so the fp32 stages ablate that unit alone.
FP32_UNITS = ("real_kc_shared_cb",)
GROUP_FP32 = "real_R1O1_fp32"


def _is32(a):
    return getattr(a, "dtype", "bf16") == "fp32"


def _units(a):
    return list(FP32_UNITS) if _is32(a) else list(UNITS)


def _prep_env():
    for k in [k for k in os.environ if k.startswith("SSD_MIMO_")]:
        del os.environ[k]          # an explicit setting bypasses the dispatch under test


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def geometry(tag):
    from orbit_ssm.kernels import shape_sets as ss
    if tag in EXTRA:
        B, d, L = EXTRA[tag]
        return dict(B=B, L=L, H=d // ss.P, P=ss.P)
    return ss.geometry(tag)


def _forced(levers, units):
    """The lever choice of one config: `units` = the list forced on (the rest off); None = the PRODUCTION choice (calibrated levers, no override)."""
    import contextlib
    return contextlib.nullcontext() if units is None else levers.forced({u: (u in units) for u in levers.REAL_UNITS})


def cname(tf, tb, tag):
    return "T%d_%d|%s" % (tf, tb, tag)


# ------------------------------------------------------------- timing child --
def _set_fp32_structure(tf, tb):
    """fp32 screen: make (T_fwd, T_bwd) the PRODUCTION fp32 structure of this child process (a job-local calibration layer:
    the shipped chunkpar file + the `|fp32` T keys), so the call at that T takes the route production would take there.
    The real route's structural decisions (the chunk-parallel forward, `chunkpar.real_cp_decision`) are replayed only at the
    production T; without this an fp32 screen would time T = 32 without and T = 64 with them (or the reverse) and choose the T
    on the route difference instead of the T."""
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import calibration_chunkpar as ccp
    gpu = cal.device_info()["gpu"]
    base = cal._read(cal.shipped_path(ccp.FAMILY, gpu))
    if base is None or not os.environ.get("SSD_CALIBRATION_DIR"):
        raise SystemExit("fp32 screen: needs the shipped chunkpar file and a job-local SSD_CALIBRATION_DIR")
    th = dict(base.get("thresholds", {}))
    for kind, t in (("train_fwd", tf), ("train_bwd", tb), ("fwd_only", tf)):
        th[ccp.REAL_T_KEYS[kind] + ccp.FP32_SUFFIX] = dict(value=int(t), evidence="calibrate_real fp32 screen (job-local)")
    for k in ccp.REAL_RULE_KEYS.values():
        th.pop(k + ccp.FP32_SUFFIX, None)
    data = dict(base, thresholds=th)
    cal.write_json(cal.local_path(ccp.FAMILY, gpu), data)


def child_time(a):
    """One shape, interleaved rounds of every config (name -> [Tf, Tb, [units on]])."""
    _prep_env()
    import torch
    from benchmarks import baselines as bl
    from benchmarks.harness import make_pass
    from orbit_ssm.kernels import shape_sets as ss
    from orbit_ssm.kernels.ssd_mimo import autograd as ua
    from orbit_ssm.kernels.ssd_mimo import levers
    g = geometry(a.shape)
    configs = json.loads(a.sets)
    torch.manual_seed(20261003)
    fns, built, infeasible = {}, {}, {}
    t0 = time.perf_counter()
    for name, (tf, tb, units) in list(configs.items()):
        key = (tf, tb)
        if _is32(a):
            _set_fp32_structure(tf, tb)
        try:
            if key not in built:
                call, leaves = bl.build_real_fair(
                    g["B"], g["H"], g["P"], N_STATE, g["L"], arm="ours", chunk=tf,
                    chunk_bwd=None if tb == tf else tb,
                    dtype=torch.float32 if _is32(a) else torch.bfloat16)
                built[key] = {p: make_pass(call, leaves, p) for p in ("fwd", "fwd_bwd")}
            with _forced(levers, units):
                for fn in built[key].values():
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                if tb != tf:
                    rec = [r for r in ua.chunk_bwd_log() if r.get("T_fwd") == tf][-1:]
                    if rec and rec[0].get("T_bwd") != tb:
                        infeasible[name] = "pair refused: %s" % rec[0].get("refused")
                        del configs[name]
                        continue
            fns[name] = built[key]
        except torch.cuda.OutOfMemoryError as e:
            return dict(shape=a.shape, oom=True, error=str(e).splitlines()[0][:200])
        except Exception as e:                           # noqa: BLE001 (shared memory, launch)
            infeasible[name] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
            del configs[name]
    first = time.perf_counter() - t0
    res = {n: {p: [] for p in ("fwd", "fwd_bwd")} for n in configs}
    for r in range(a.rounds):
        order = list(configs) if r % 2 == 0 else list(reversed(list(configs)))
        for name in order:
            _tf, _tb, units = configs[name]
            if _is32(a):
                _set_fp32_structure(_tf, _tb)
            with _forced(levers, units):
                for p, fn in fns[name].items():
                    e0, e1 = (torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True))
                    torch.cuda.synchronize()
                    e0.record()
                    for _ in range(a.iters):
                        fn()
                    e1.record()
                    e1.synchronize()
                    res[name][p].append(e0.elapsed_time(e1) / a.iters)
    med = {n: {p: statistics.median(v) for p, v in d.items()} for n, d in res.items()}
    return dict(shape=a.shape, geometry=g, first_call_s=first, rounds=a.rounds,
                median_ms=med, raw_ms=res, infeasible=infeasible,
                configs={n: c for n, c in configs.items()})


# ----------------------------------------------------------------- driver ----
def run_child(a, stage, shape, configs):
    out = os.path.join(a.out, "raw", "%s_%s.json" % (stage, shape))
    if os.path.exists(out):
        return json.load(open(out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "time", "--out", out,
           "--rounds", str(a.rounds), "--iters", str(a.iters), "--shape", shape,
           "--sets", json.dumps(configs), "--dtype", getattr(a, "dtype", "bf16")]
    t = time.time()
    p = subprocess.run(cmd, env=env, timeout=a.timeout)
    if p.returncode:
        raise SystemExit("child %s %s failed (exit %d)" % (stage, shape, p.returncode))
    print("[real] %s %s done in %.0fs" % (stage, shape, time.time() - t), flush=True)
    return json.load(open(out))


def shape_list(a, load=False):
    """The shapes a stage RUNS (the 18, plus the EXTRA ones with --extra) or LOADS (`load`: every one that has a raw file)."""
    from orbit_ssm.kernels import shape_sets as ss
    if a.shapes:
        return a.shapes.split(",")
    return list(ss.SET) + (list(EXTRA) if (load or getattr(a, "extra", False)) else [])


def eligible_units(a):
    f = os.path.join(a.out, "eligibility.json")
    if os.path.exists(f):
        e = json.load(open(f))
        return [u for u in _units(a) if e["units"][u]["eligible"]]
    return _units(a)


# ------------------------------------------------------------- eligibility ---
_LINE = re.compile(r"err\s+(\S+)\s+(?:bound\s+(\S+)\s+)?signed\s+(\S+)(?:\s+bound\s+(\S+))?")
M2KW = ["--R", "1", "--O", "1", "--rot", "none", "--write-rule", "euler", "--bc-conv",
        "--ssm-kw", 'row_blocks=[{"frac":1.0,"decay":"head_static","complex":False}]',
        "--ssm-kw", "b_norm=False", "--ssm-kw", "c_norm=False", "--ssm-kw", "bc_bias='none'",
        "--ssm-kw", "b_gate=False", "--ssm-kw", "c_gate=False",
        "--ssm-kw", "skip_sharing='head'", "--ssm-kw", "out_norm=True", "--ssm-kw", "r_mix_dyn=False"]


def _score(text):
    worst, sg = 0.0, 0.0
    for m in _LINE.finditer(text):
        try:
            e, s = float(m.group(1)), float(m.group(3))
            b = float(m.group(2) or m.group(4) or "nan")
        except ValueError:
            continue
        if b > 0 and b < float('inf'):
            worst, sg = max(worst, e / b), max(sg, abs(s) / b)
    return worst, sg


def oracle_checks(units_on, T, cp, logdir, tag, fp32=False):
    """The fp64 oracle suite of the real route with exactly `units_on` set (env), at chunk T and
    CHUNKPAR=cp: m2exact-like mixer arm, the C11 arm, saturated plain / reset / large_phase.  Returns
    dict(passed, per-check passed, worst err/bound, worst |signed|/bound)."""
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for k in [k for k in env if k.startswith("SSD_MIMO_")]:
        del env[k]
    lev = ["--env", "SSD_MIMO_SISO_FAST=1", "--env", "SSD_MIMO_CHUNKPAR=%d" % cp,
           "--env", "SSD_MIMO_CHUNK=%d" % T, "--env", "SSD_MIMO_CP_TRACE=1"]
    for u in UNITS:           # explicit on / off: a calibrated choice must not leak into a "base" arm
        lev += ["--env", "SSD_MIMO_%s=%d" % (UNIT_ENV[u], int(u in units_on))]
    # (the calibrated real_gluon_kbc unit serves the Gluon Kbc with the dalpha hand-off OFF -- the
    # `gluon_kbc_dalpha` kernel option's default; the hand-off fails the mixer-level oracle)
    dt = ["--dtype", "fp32"] if fp32 else []
    runs = {"m2": [sys.executable, "tools/ssd_check.py"] + M2KW + ["--B", "2", "--L", "2048", "--draws", "5"] + lev + dt,
            "c11": [sys.executable, "tools/ssd_check.py", "--arm", "ssd_mimo_r1o1_real", "--B", "2",
                    "--L", "2048", "--draws", "5"] + lev + dt}
    for p in (() if fp32 else ("plain", "reset", "large_phase")):       # (the saturated tool is a 16-bit-IO check)
        pa = [] if p == "plain" else ["--perturb", p]
        runs["sat_" + p] = [sys.executable, "tools/ssd_check_saturated.py", "--arm", "ssd_mimo_r1o1_real",
                            "--scan"] + pa + ["--draws", "5", "--precisions", "io"] + lev
    out = dict(passed=True, checks={}, worst_err_over_bound=0.0, worst_signed_over_bound=0.0)
    for name, cmd in runs.items():
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        text = p.stdout + p.stderr
        open(os.path.join(logdir, "%s_%s.log" % (tag, name)), "w").write(text)
        w, s = _score(text)
        served = sorted({m for m in re.findall(r"CP_TRACE (kbc_served=\S+|kbc=\S+ ?\S*|fwd=\S+)", text)})
        out["checks"][name] = dict(passed=p.returncode == 0, exit=p.returncode, err_over_bound=w,
                                   signed_over_bound=s, trace=served[:6])
        out["passed"] &= p.returncode == 0
        out["worst_err_over_bound"] = max(out["worst_err_over_bound"], w)
        out["worst_signed_over_bound"] = max(out["worst_signed_over_bound"], s)
    return out


def cmd_elig(a):
    """Eligibility against the fp64 oracle.  BASE (every unit off) must pass.  Then, per unit, ALONE at
    its own matrix points (the Gluon Kbc exists only at T = 64; CP_G_LO acts only on the chunk-parallel
    forward, CHUNKPAR = 1), then GREEDILY the combination (a unit that passes alone but fails together
    with the already admitted ones is INTERACTION-ineligible: the later calibration needs the
    all-on set to be a numerically valid configuration), then the final set over the whole matrix
    T in {32, 64} x CHUNKPAR in {0, 1}."""
    os.makedirs(a.out, exist_ok=True)
    logdir = os.path.join(a.out, "elig_logs")
    os.makedirs(logdir, exist_ok=True)
    only = [u for u in a.only.split(",") if u]
    prev = os.path.join(a.out, "eligibility.json")
    full = [(64, 0), (64, 1), (32, 0), (32, 1)]
    pts = {u: ([(64, 1), (32, 1)] if u == "real_cp_g_lo" else [(64, 0), (32, 0)]) for u in UNITS}
    f32 = _is32(a)
    oc = lambda units_on, T, cp, logdir, tag: oracle_checks(units_on, T, cp, logdir, tag, fp32=f32)
    pts["real_gluon_kbc"] = [(64, 0), (64, 1)]
    res = dict(base={}, solo={}, greedy=[], final={})
    k_ = lambda T, cp: "T%d_cp%d" % (T, cp)
    if only and os.path.exists(prev):        # incremental: re-test only these units on top of the stored result
        res = json.load(open(prev))
    for T, cp in ([] if (only and os.path.exists(prev)) else full):
        r = res["base"][k_(T, cp)] = oc([], T, cp, logdir, "base_" + k_(T, cp))
        print("[real] elig base %s passed=%s worst err/bound %.3f signed/bound %.3f" % (
            k_(T, cp), r["passed"], r["worst_err_over_bound"], r["worst_signed_over_bound"]), flush=True)
    if not all(r["passed"] for r in res["base"].values()):
        raise SystemExit("BASE (every lever off) fails the fp64 oracle: see %s" % logdir)
    units = res.get("units", {}) if (only and os.path.exists(prev)) else {}
    todo = only if (only and os.path.exists(prev)) else _units(a)
    for u in todo:
        ok = True
        for T, cp in pts[u]:
            r = res["solo"].setdefault(u, {})[k_(T, cp)] = oc([u], T, cp, logdir, "solo_%s_%s" % (u, k_(T, cp)))
            ok &= r["passed"]
            print("[real] elig solo %s %s passed=%s err/bound %.3f signed/bound %.3f" % (
                u, k_(T, cp), r["passed"], r["worst_err_over_bound"], r["worst_signed_over_bound"]), flush=True)
        units[u] = dict(eligible=ok, via="solo" if ok else "fails alone")
    admitted = list(res.get("admitted", [])) if (only and os.path.exists(prev)) else []
    for u in todo:
        if not units[u]["eligible"]:
            continue
        trial = admitted + [u]
        ok, last = True, None
        for T, cp in sorted({p for x in trial for p in pts[x]}):
            r = oc(trial, T, cp, logdir, "greedy_%s_%s" % ("+".join(x[5:] for x in trial), k_(T, cp)))
            last = r
            res["greedy"].append(dict(units=trial, point=k_(T, cp), passed=r["passed"],
                                      err_over_bound=r["worst_err_over_bound"],
                                      signed_over_bound=r["worst_signed_over_bound"]))
            print("[real] elig greedy %s %s passed=%s" % (trial, k_(T, cp), r["passed"]), flush=True)
            if not r["passed"]:
                ok = False
                break
        if ok:
            admitted = trial
        else:
            units[u] = dict(eligible=False, via="passes alone, fails together with %s" % admitted)
    for T, cp in full:
        r = res["final"][k_(T, cp)] = oc(admitted, T, cp, logdir, "final_" + k_(T, cp))
        print("[real] elig FINAL set %s %s passed=%s err/bound %.3f signed/bound %.3f" % (
            admitted, k_(T, cp), r["passed"], r["worst_err_over_bound"], r["worst_signed_over_bound"]), flush=True)
        if not r["passed"]:
            raise SystemExit("the admitted set fails at %s" % k_(T, cp))
    res["units"], res["admitted"] = units, admitted
    json.dump(res, open(os.path.join(a.out, "eligibility.json"), "w"), indent=1)


# ------------------------------------------------------------------ screen ---
def cmd_screen(a):
    el = eligible_units(a)
    cfgs = {}
    for tf, tb in COMBOS:
        cfgs[cname(tf, tb, "base")] = [tf, tb, []]
        cfgs[cname(tf, tb, "allon")] = [tf, tb, el]
    for s in shape_list(a):
        run_child(a, "screen", s, cfgs)


def cmd_screenp(a):
    """END-TO-END production screen: (32, 32) and (64, 64) with the calibrated levers and the calibrated configs of each T's own group (no forcing);
    stage `screenp` (the `screen` stage forced lever sets, and an explicit chunk away from the recorded one ran the generic grid)."""
    cfgs = {cname(32, 32, "prod"): [32, 32, None], cname(64, 64, "prod"): [64, 64, None]}
    for s in shape_list(a):
        run_child(a, "screenp", s, cfgs)


def cmd_fitprod(a):
    data, dropped = _load(a, "screenp")
    shapes = sorted(data)
    print("%-22s %5s %7s %8s | %8s %8s | T32/T64" % ("shape", "B", "L", "work", "T32 ms", "T64 ms"))
    rows = []
    for s_ in sorted(shapes, key=lambda x: (data[x]["geometry"]["L"], data[x]["geometry"]["B"])):
        g, m = data[s_]["geometry"], data[s_]["median_ms"]
        t32, t64 = m[cname(32, 32, "prod")]["fwd_bwd"], m[cname(64, 64, "prod")]["fwd_bwd"]
        rows.append((s_, t32, t64))
        print("%-22s %5d %7d %8d | %8.3f %8.3f | %.3f" % (s_, g["B"], g["L"], g["B"] * g["H"] * g["L"], t32, t64, t32 / t64))
    print("geomean T32/T64 = %.3f ; T64 wins on %d of %d shapes ; worst T64 loss %.3f x, worst T32 loss %.3f x" % (
        geomean([r[1] / r[2] for r in rows]), sum(r[2] < r[1] for r in rows), len(rows),
        max(r[2] / r[1] for r in rows), max(r[1] / r[2] for r in rows)))
    rule = fit_rule(data, shapes)
    print("rule fit over the production screen:", rule and {k: rule[k] for k in ("T_hi", "L_max", "W_min", "geomean_gain", "worst_vs_best", "adopted")})
    json.dump(dict(rows=rows, rule=rule), open(os.path.join(a.out, "fitprod.json"), "w"), indent=1)


def _load(a, stage):
    d = os.path.join(a.out, "raw")
    out, dropped = {}, {}
    for s in shape_list(a, load=True):
        f = os.path.join(d, "%s_%s.json" % (stage, s))
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        if r.get("oom"):
            dropped[s] = r["error"]
        else:
            out[s] = r
    return out, dropped


def _rule_pick(score, incumbent):
    """score: {candidate: {shape: time}} -> choose by the decision rule against the incumbent:
    a candidate beats it iff geomean(incumbent/cand) >= TOL_ON and min >= TOL_WORST; among those the
    best geomean; else the incumbent."""
    inc = score[incumbent]
    shapes = sorted(inc)
    rows = {}
    for c, t in score.items():
        rs = [inc[s] / t[s] for s in shapes]
        rows[c] = dict(geomean_gain=geomean(rs), worst_shape_gain=min(rs),
                       per_shape=dict(zip(shapes, rs)))
    ok = [c for c, r in rows.items() if c != incumbent
          and r["geomean_gain"] >= TOL_ON and r["worst_shape_gain"] >= TOL_WORST]
    best = max(ok, key=lambda c: rows[c]["geomean_gain"]) if ok else incumbent
    return best, rows


def fit_rule(data, shapes):
    """The structural T rule `T = 64 iff L <= L_max and B*H*L >= W_min` (else 32) over the screen (training fwd+bwd, the faster of
    base / all-on at each T: the levers are decided afterwards, at each structure).  Among the (L_max, W_min) whose worst shape is within
    RULE_WORST of its per-shape best, the smallest geomean time; adopted only when it beats the best SINGLE T by TOL_RULE.
    Returns dict(T_hi, L_max, W_min, geomean_gain, worst_vs_best, ...) or None."""
    rows = []
    for s in shapes:
        m, g = data[s]["median_ms"], data[s]["geometry"]
        t = {}
        for T in (32, 64):
            v = [m[c]["fwd_bwd"] for c in m if c.startswith("T%d_%d|" % (T, T))]
            if v:
                t[T] = min(v)
        if len(t) == 2:
            rows.append(dict(shape=s, L=g["L"], work=g["B"] * g["H"] * g["L"], t32=t[32], t64=t[64]))
    if len(rows) < 8:
        return None
    best_single = min(geomean([r["t32"] for r in rows]), geomean([r["t64"] for r in rows]))
    Ls = sorted({r["L"] for r in rows})
    Ws = sorted({r["work"] for r in rows})
    best = None
    for Lm in Ls:
        for Wm in [0] + Ws:
            ts = [r["t64"] if (r["L"] <= Lm and r["work"] >= Wm) else r["t32"] for r in rows]
            worst = max(t_ / min(r["t32"], r["t64"]) for t_, r in zip(ts, rows))
            if worst > RULE_WORST:
                continue
            g_ = geomean(ts)
            # ties: the narrowest rule (smallest region = fewest calls leave the calibrated T)
            key = (round(g_, 9), Lm, -Wm)
            if best is None or key < best[0]:
                best = (key, Lm, Wm, g_, worst)
    if best is None:
        return None
    _, Lm, Wm, g_, worst = best
    gain = best_single / g_
    return dict(T_hi=64, L_max=int(Lm), W_min=int(Wm), geomean_gain=gain, worst_vs_best=worst, shapes=len(rows),
                adopted=bool(gain >= TOL_RULE),
                inside=sorted(r["shape"] for r in rows if r["L"] <= Lm and r["work"] >= Wm))


def in_rule(rule, geo):
    return bool(rule and geo["L"] <= rule["L_max"] and geo["B"] * geo["H"] * geo["L"] >= rule["W_min"])


def cmd_choose(a):
    data, dropped = _load(a, "screen")
    if not data:
        raise SystemExit("no screen results in %s" % a.out)
    shapes = sorted(data)
    res = dict(shapes=shapes, dropped=dropped, tags={})
    def ms(s, tf, tb, tag, ph):
        """tag "best": the faster of base / all-on at this shape (the structure is chosen under the
        lever choice that suits it; the levers themselves are decided afterwards, at that structure)."""
        if tag == "best":
            return min(ms(s, tf, tb, t, ph) for t in ("base", "allon"))
        return data[s]["median_ms"][cname(tf, tb, tag)][ph]
    for tag in ("best", "allon", "base"):
        fb, fw = {}, {}
        for tf, tb in COMBOS:
            if all(cname(tf, tb, "base") in data[s]["median_ms"] and cname(tf, tb, "allon") in data[s]["median_ms"]
                   for s in shapes):
                fb[(tf, tb)] = {s: ms(s, tf, tb, tag, "fwd_bwd") for s in shapes}
        for tf in (32, 64):
            fw[tf] = {s: ms(s, tf, tf, tag, "fwd") for s in shapes}
        best_t, rows_t = _rule_pick({"%d,%d" % k: v for k, v in fb.items()}, "%d,%d" % INCUMBENT)
        best_f, rows_f = _rule_pick({str(k): v for k, v in fw.items()}, str(INCUMBENT[0]))
        res["tags"][tag] = dict(train=dict(best=best_t, rows=rows_t), fwd_only=dict(best=best_f, rows=rows_f))
        print("[real] %s train best (T_fwd,T_bwd)=%s ; fwd-only best T=%s" % (tag, best_t, best_f))
        for k, r in rows_t.items():
            print("   train %-8s gain vs %s: geomean %.3f worst %.3f" % (k, INCUMBENT, r["geomean_gain"], r["worst_shape_gain"]))
        for k, r in rows_f.items():
            print("   fwd-only T=%-3s gain vs %d: geomean %.3f worst %.3f" % (k, INCUMBENT[0], r["geomean_gain"], r["worst_shape_gain"]))
    pick = res["tags"][a.basis]            # the structure is chosen with the levers' eligible set on
    tf, tb = (int(x) for x in pick["train"]["best"].split(","))
    res["structure"] = dict(train_fwd=tf, train_bwd=tb, fwd_only=int(pick["fwd_only"]["best"]), basis=a.basis)
    rule = fit_rule(data, shapes)
    res["rule"] = rule if (rule and rule["adopted"]) else None
    res["rule_candidate"] = rule
    if rule:
        print("[real] T RULE candidate: T=%d iff L <= %d and B*H*L >= %d: geomean gain %.3f over the best single T, worst shape %.3f x its best, "
              "%d shapes inside -> %s" % (rule["T_hi"], rule["L_max"], rule["W_min"], rule["geomean_gain"], rule["worst_vs_best"],
                                         len(rule["inside"]), "ADOPTED" if rule["adopted"] else "not adopted (< %.2f)" % TOL_RULE))
    json.dump(res, open(os.path.join(a.out, "structure.json"), "w"), indent=1)
    print("[real] CHOSEN", res["structure"])


def structure(a):
    f = os.path.join(a.out, "structure.json")
    if not os.path.exists(f):
        raise SystemExit("run `choose` first")
    return json.load(open(f))["structure"]


def rule_of(a):
    """The adopted structural T rule of structure.json (see `fit_rule`) or None."""
    f = os.path.join(a.out, "structure.json")
    return json.load(open(f)).get("rule") if os.path.exists(f) else None


# ------------------------------------------------------------------ ablate ---
def _ablate_cfgs(el, combos):
    cfgs = {}
    for (f_, b_) in combos:
        app = [u for u in el if u != "real_gluon_kbc" or b_ == 64]
        cfgs[cname(f_, b_, "base")] = [f_, b_, []]
        cfgs[cname(f_, b_, "allon")] = [f_, b_, app]
        for u in app:
            cfgs[cname(f_, b_, "loo_" + u)] = [f_, b_, [x for x in app if x != u]]
    return cfgs


def cmd_ablate(a):
    st, rule = structure(a), rule_of(a)
    el = eligible_units(a)
    tf, tb, tfo = st["train_fwd"], st["train_bwd"], st["fwd_only"]
    lo = _ablate_cfgs(el, {(tf, tb), (tfo, tfo)})
    hi = _ablate_cfgs(el, {(rule["T_hi"], rule["T_hi"])}) if rule else None
    for s in shape_list(a):
        if rule and in_rule(rule, geometry(s)):
            run_child(a, "ablate_hi", s, hi)        # the calls the rule sends to T_hi: their own lever decision (group real_R1O1_hi)
        else:
            run_child(a, "ablate", s, lo)


def decide_levers(a, hi=False):
    st = structure(a)
    rule = rule_of(a)
    el_all = json.load(open(os.path.join(a.out, "eligibility.json")))["units"] \
        if os.path.exists(os.path.join(a.out, "eligibility.json")) else {u: dict(eligible=True) for u in _units(a)}
    data, dropped = _load(a, "ablate_hi" if hi else "ablate")
    if rule and not hi:        # the base-T group is decided on the calls that USE the base T
        data = {s_: d_ for s_, d_ in data.items() if not in_rule(rule, d_["geometry"])}
    shapes = sorted(data)
    tf, tb, tfo = st["train_fwd"], st["train_bwd"], st["fwd_only"]
    if hi:
        tf = tb = tfo = rule["T_hi"]
    chosen, rep = {}, []

    def ph_time(shape, cfg, ph):
        m = data[shape]["median_ms"].get(cfg)
        if m is None:
            return None
        return m["fwd"] if ph == "fwd" else m["fwd_bwd"] - m["fwd"]
    for u in _units(a):
        ph = PHASE[u]
        if not el_all[u]["eligible"]:
            chosen[u] = dict(value=0, evidence=dict(eligible=False, phase=ph,
                                                    reason="fp64 oracle check failed"))
            rep.append("  %-18s %s INELIGIBLE (fp64 oracle)" % (u, ph))
            continue
        # cells: the units' phase at the training structure (bwd, fwd) and the fwd-only T (fwd)
        cells = [(tf, tb)] if ph == "bwd" else sorted({(tf, tb), (tfo, tfo)})
        if u == "real_gluon_kbc" and tb != 64:
            chosen[u] = dict(value=0, evidence=dict(eligible=True, phase=ph, applicable=False,
                                                    reason="the Gluon Kbc body exists only at T_bwd = 64"))
            rep.append("  %-18s %s not applicable (T_bwd = %d)" % (u, ph, tb))
            continue
        rs, per = [], {}
        for (f_, b_) in cells:
            for s in shapes:
                t_on = ph_time(s, cname(f_, b_, "allon"), ph)
                t_off = ph_time(s, cname(f_, b_, "loo_" + u), ph)
                if t_on is None or t_off is None:
                    continue
                r = t_off / t_on
                rs.append(r)
                per["%s@T%d_%d" % (s, f_, b_)] = r
        if not rs:
            chosen[u] = dict(value=0, evidence=dict(eligible=True, phase=ph, reason="no measurement"))
            continue
        gm, mn = geomean(rs), min(rs)
        on = gm >= TOL_ON and mn >= TOL_WORST
        chosen[u] = dict(value=int(on), evidence=dict(eligible=True, phase=ph, geomean_gain=gm,
                                                      worst_shape_gain=mn, per_shape=per))
        rep.append("  %-18s %s gain geomean %.3f worst %.3f -> %s" % (u, ph, gm, mn, "ON" if on else "off"))
    final = [u for u in _units(a) if chosen[u]["value"]]
    ver = {}
    f_run = a.final_raw or os.path.join(a.out, "final_verify.json")
    # VERIFY: the chosen set vs base (no extra measurement: sum of the LOO-run's base/allon rows when the
    # chosen set is the all-on set, else the dedicated `final` run of `verify`)
    fin = {}
    if os.path.exists(f_run):
        fin = json.load(open(f_run))
    for ph, cells in (("bwd", [(tf, tb)]), ("fwd", sorted({(tf, tb), (tfo, tfo)}))):
        rr = []
        for (f_, b_) in cells:
            for s in shapes:
                n = cname(f_, b_, "final")
                m, b0 = fin.get(s, {}).get(n), fin.get(s, {}).get(cname(f_, b_, "base"))
                if m is None or b0 is None:       # final and base of the SAME run (same autotune picks)
                    continue
                tt = m["fwd"] if ph == "fwd" else m["fwd_bwd"] - m["fwd"]
                t0 = b0["fwd"] if ph == "fwd" else b0["fwd_bwd"] - b0["fwd"]
                rr.append(tt / t0)
        if rr:
            ver[ph] = dict(final_over_base=geomean(rr), worst_shape=max(rr))
            rep.append("  VERIFY %s: FINAL/BASE geomean %.3f worst %.3f" % (ph, geomean(rr), max(rr)))
            if geomean(rr) > 1.0:
                for u in _units(a):
                    if chosen[u]["evidence"].get("phase") == ph and chosen[u]["value"]:
                        chosen[u]["value"] = 0
                        chosen[u]["evidence"]["turned_off"] = "FINAL slower than BASE on %s" % ph
                rep.append("  -> every %s unit turned off" % ph)
    grp = GROUP_FP32 if _is32(a) else (GROUP_HI if hi else GROUP)
    th = {"lever|%s|%s" % (u, grp): dict(value=c["value"], evidence=dict(c["evidence"], verify=ver, dtype=getattr(a, "dtype", "bf16")))
          for u, c in chosen.items()}
    return th, rep, dropped, final


def cmd_verify(a):
    """Time FINAL (the chosen units) against base at the chosen structure (one child per shape)."""
    st = structure(a)
    th, _, _, _ = decide_levers(a)
    final = [u.split("|")[1] for u, c in th.items() if c["value"]]
    tf, tb, tfo = st["train_fwd"], st["train_bwd"], st["fwd_only"]
    cfgs = {}
    for (f_, b_) in sorted({(tf, tb), (tfo, tfo)}):
        cfgs[cname(f_, b_, "base")] = [f_, b_, []]
        cfgs[cname(f_, b_, "final")] = [f_, b_, [u for u in final if u != "real_gluon_kbc" or b_ == 64]]
    out = {}
    for s in shape_list(a):
        r = run_child(a, "verify", s, cfgs)
        if not r.get("oom"):
            out[s] = r["median_ms"]
    json.dump(out, open(os.path.join(a.out, "final_verify.json"), "w"), indent=1)


def cmd_decide(a):
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import calibration_chunkpar as ccp
    from orbit_ssm.kernels.ssd_mimo import levers
    th, rep, dropped, final = decide_levers(a)
    print("real_R1O1 lever calibration:")
    print("\n".join(rep))
    st = structure(a)
    rule = rule_of(a)
    print("structure:", st, "| T rule:", rule and {k: rule[k] for k in ("T_hi", "L_max", "W_min", "geomean_gain", "worst_vs_best")})
    with open(os.path.join(a.out, "chosen_env.txt"), "w") as f:   # the chosen levers as env (kernel recalibration runs with them ON)
        for u in final:
            f.write("SSD_MIMO_%s=1\n" % UNIT_ENV[u])
    if rule and not _is32(a):                                     # the rule's T_hi calls: their own lever group (16-bit)
        th_hi, rep_hi, _, final_hi = decide_levers(a, hi=True)
        print("real_R1O1_hi lever calibration (T = %d, %d shapes):" % (rule["T_hi"], len(rule["inside"])))
        print("\n".join(rep_hi))
        th.update(th_hi)
        with open(os.path.join(a.out, "chosen_env_hi.txt"), "w") as f:
            for u in final_hi:
                f.write("SSD_MIMO_%s=1\n" % UNIT_ENV[u])
    dev = cal.device_info()
    gpu = dev["gpu"]
    pick = (cal.shipped_path if a.write_shipped else cal.local_path)
    # levers file: keep every other group's thresholds, replace groups real_R1O1 / real_R1O1_hi
    p = pick(levers.FAMILY, gpu)
    old = cal._read(p) or {}
    drop = (GROUP_FP32,) if _is32(a) else (GROUP, GROUP_HI)
    keep = {k: v for k, v in old.get("thresholds", {}).items() if not any(k.endswith("|" + g) for g in drop)}
    data = cal.build_file(levers.FAMILY, {}, a.rounds, old.get("provenance", {}).get("shapes", []),
                          thresholds=dict(keep, **th), dropped=old.get("provenance", {}).get("dropped_shapes"))
    print("wrote", cal.write_json(p, data))
    # chunkpar file: the structural choice (+ the T rule) next to the crossover thresholds
    p2 = pick(ccp.FAMILY, gpu)
    old2 = cal._read(p2) or {}
    suf = ccp.FP32_SUFFIX if _is32(a) else ""
    t2 = {k: v for k, v in old2.get("thresholds", {}).items() if k not in [x + suf for x in ccp.REAL_RULE_KEYS.values()]}
    for kind, key in ccp.REAL_T_KEYS.items():
        t2[key + suf] = dict(value=int(st[kind]), evidence="tools/calibrate_real.py%s (screen: (T_fwd, T_bwd) in {32,64}^2)" % (
            " --dtype fp32" if suf else ""))
    if rule:
        ev = ("tools/calibrate_real.py fit_rule over %d screen shapes: geomean gain %.3f over the best single T, worst shape %.3f x its "
              "per-shape best" % (rule["shapes"], rule["geomean_gain"], rule["worst_vs_best"]))
        for k, key in ccp.REAL_RULE_KEYS.items():
            t2[key + suf] = dict(value=int(rule[k]), evidence=ev)
    data2 = cal.build_file(ccp.FAMILY, {}, a.rounds, old2.get("provenance", {}).get("shapes", []),
                           thresholds=t2, dropped=old2.get("provenance", {}).get("dropped_shapes"))
    print("wrote", cal.write_json(p2, data2))


def cmd_final(a):
    """fp64 oracle on the CHOSEN DEFAULTS: the production route, no SSD_MIMO_* setting at all (CP_TRACE only
    prints the route): m2exact-like and C11 arms at (B, L) = (2, 2048) and (1, 8192), saturated plain /
    reset / large_phase.  Signed bias and normwise error reported against the bound."""
    os.makedirs(a.out, exist_ok=True)
    logdir = os.path.join(a.out, "final_logs")
    os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for k in [k for k in env if k.startswith("SSD_MIMO_")]:
        del env[k]
    tr = ["--env", "SSD_MIMO_CP_TRACE=1"]
    runs = {}
    for B, L in ((2, 2048), (1, 8192)):
        runs["m2_B%dL%d" % (B, L)] = [sys.executable, "tools/ssd_check.py"] + M2KW + ["--B", str(B), "--L", str(L), "--draws", "5"] + tr
        runs["c11_B%dL%d" % (B, L)] = [sys.executable, "tools/ssd_check.py", "--arm", "ssd_mimo_r1o1_real", "--B", str(B),
                                       "--L", str(L), "--draws", "5"] + tr
    for p in ("plain", "reset", "large_phase"):
        pa = [] if p == "plain" else ["--perturb", p]
        runs["sat_" + p] = [sys.executable, "tools/ssd_check_saturated.py", "--arm", "ssd_mimo_r1o1_real", "--scan"] + pa + [
            "--draws", "5", "--precisions", "io"] + tr
    out = dict(passed=True, checks={})
    for name, cmd in runs.items():
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        text = p.stdout + p.stderr
        open(os.path.join(logdir, name + ".log"), "w").write(text)
        w, s = _score(text)
        trace = {}
        for m in re.findall(r"CP_TRACE (\S+)", text):
            trace[m] = trace.get(m, 0) + 1
        out["checks"][name] = dict(passed=p.returncode == 0, err_over_bound=w, signed_over_bound=s, trace=trace)
        out["passed"] &= p.returncode == 0
        print("[real] final %s passed=%s err/bound %.3f signed/bound %.3f %s" % (name, p.returncode == 0, w, s, trace), flush=True)
    json.dump(out, open(os.path.join(a.out, "final_numerics.json"), "w"), indent=1)
    if not out["passed"]:
        raise SystemExit("final numerics FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["elig", "screen", "screenp", "fitprod", "choose", "ablate", "verify", "decide", "final"])
    ap.add_argument("--out", default="")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--basis", default="best", choices=["best", "allon", "base"])
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--final-raw", default="")
    ap.add_argument("--only", default="", help="elig: re-test only these units (comma list) on top of eligibility.json")
    ap.add_argument("--extra", action="store_true", help="screen / ablate: also the EXTRA boundary shapes")
    ap.add_argument("--gpu", default="", help="decide: GPU name of the raw results (derive offline)")
    ap.add_argument("--sms", type=int, default=0, help="decide: SM count of that GPU")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"),
                    help="fp32: the same stages at fp32 IO / tf32 operands; results go to the |fp32 keys and lever group real_R1O1_fp32")
    ap.add_argument("--child", default="")
    ap.add_argument("--shape", default="")
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    if a.child:
        r = globals()["child_" + a.child](a)
        json.dump(r, open(a.out, "w"))
        return
    if a.gpu and a.sms:          # derive another GPU's results offline (no CUDA device needed)
        import triton as _t
        from orbit_ssm.kernels import calibration as _cal
        _cal._DEV.update(gpu=a.gpu, sms=a.sms, driver=None, triton=_t.__version__, torch="offline")
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
