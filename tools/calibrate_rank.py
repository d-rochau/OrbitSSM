#!/usr/bin/env python
"""Per-RANK structural calibration of the SSD-MIMO R = O = 2 and R = O = 4 routes on THIS GPU: the chunk lengths (T_fwd, T_bwd).

`tools/calibrate_real.py` owns R = O = 1 (real) and `tools/calibrate_routes.py` the rotating R = O = 1 route; for R > 1 the dispatch
(`ssm.ssd_mimo._scan_chunk`) used the FIXED rule (rank table + wide-W clause: T = 32 at R = 2 / 4 training, 32 / 16 forward-only).  This tool
measures, per route {real, cx (rotating)} x rank {2, 4} x IO dtype {bf16, fp32}, the (T_fwd, T_bwd) pair over the shared shape set:

    T_fwd, T_bwd in {32, 64} at R = 2 ;  in {16, 32, 64} at R = 4   (16 is legal at R = 4: 4 x 4 x 16 score rows fill one MMA tile)
    fwd-only T (a call without backward) in the same set.

Decision (the rule of `calibrate_real._rule_pick`): a candidate replaces the fixed rule's T iff its geomean gain over the fixed rule is
>= 1.015 and no shape loses more than 3 %.  Stored as thresholds `rank|<route>|R<R>O<O>|T|<kind>[|fp32]` of family `ssd_mimo_chunkpar`
(`calibration_chunkpar.rank_chunk`; R = O = 1 keys untouched).  The verdict is stored also where it equals the fixed rule (the entry then
records "measured, fixed rule kept"); only a (route, rank, dtype) that was never screened has no entry and takes the fixed rule.
A ROTATING call refuses a decoupled backward chunk (theta is chunk-shaped), so its candidates are the diagonal pairs only.

LEVERS.  Every `real_*` lever unit (`levers.REAL_UNITS`) is guarded to R = O = 1 by its own kernel (`kc_shared_cb`: R != 1 or O != 1; `dg_lo`: R == 1
and O == 1; the Gluon Kbc: R == 1 and O == 1 at T = 64; `gate_dy_io`: `_real_r1o1_call`; `cp_g_lo`: the real R = O = 1 chunk-parallel forward), so at
R = 2 / 4 the real lever set is empty by construction; `levers` records that.  The rotating phase-law units are decided per rank by
`tools/calibrate_levers.py` (group `complex_R<R>O<R>`).

The launch configs of the chosen (route, rank, dtype, T_bwd) are measured afterwards as fused groups `<route>_R<R>O<R>[_fp32]_t<T>` by
`tools/calibrate_kernels.py --family ssd_mimo_fused` (cells `<route>_r<R>o<R>[f32]t<T>_<shape>`, `calibration_fused.SHAPES`); `groups` prints
the cell / group lists.

Stages (resumable: raw results are appended to --out/raw_<set>.jsonl):

    python tools/calibrate_rank.py screen  --out DIR [--R 2,4] [--route real,cx] [--dtype bf16,fp32]
    python tools/calibrate_rank.py choose  --out DIR [--write-shipped]      # thresholds + DIR/rank_structure.json
    python tools/calibrate_rank.py groups  --out DIR                         # the fused cells / groups of the chosen T_bwd != 32 entries
    python tools/calibrate_rank.py verify  --out DIR                         # production (calibrated) vs the fixed rule, per shape
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, str(ROOT) + "/src")

TOL_ON, TOL_WORST = 1.015, 0.97
CANDS = {2: (32, 64), 4: (16, 32, 64)}
G, N_STATE = 64, 128
#: the fixed rule (`ssm.ssd_mimo._scan_chunk` / `_CHUNK_RANK_TABLE*`, G = 64): training (T, T), forward-only T
INCUMBENT = {2: dict(train=(32, 32), fwd_only=32), 4: dict(train=(32, 32), fwd_only=16)}
#: R = 4 on the LM cells only (measurement-time bound, as the fused R4O4 cells); R = 2 on the whole shared set
R4_TAGS = None            # None -> shape_sets.LM_TAGS


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def shapes_of(R):
    from orbit_ssm.kernels import shape_sets as ss
    return (list(ss.LM_TAGS) + list(ss.GRID_TAGS)) if R == 2 else list(ss.LM_TAGS)      # the LM cells first: a partial run still prices the application


def cname(tf, tb):
    return "T%d_%d" % (tf, tb)


def set_name(route, R, dtype):
    return "%s_R%dO%d_%s" % (route, R, R, dtype)


# ------------------------------------------------------------------- child --
def child_time(a):
    """One shape: interleaved rounds of every (T_fwd, T_bwd) pair, the fwd phase and the fwd+bwd pass.  Calibration OFF: every T runs the
    kernels' own default launch search, so no T is advantaged by a calibrated group recorded at another T."""
    for k in [k for k in os.environ if k.startswith("SSD_MIMO_")]:
        del os.environ[k]
    os.environ["SSD_MIMO_CALIBRATION"] = "off"
    import torch
    from benchmarks.bench_kernels import build_rank_mixer
    from benchmarks.harness import make_pass
    from orbit_ssm.kernels import shape_sets as ss
    from orbit_ssm.kernels.ssd_mimo import autograd as ua
    g = ss.geometry(a.shape)
    R, rot, fp = a.R, a.route == "cx", a.dtype == "fp32"
    Ts = CANDS[R]
    # a ROTATING call refuses a decoupled backward chunk (theta / rot_bias are chunk-shaped: `chunk_bwd_log` "pair refused: rotation"): its
    # candidates are the diagonal pairs only
    combos = [(tf, tb) for tf in Ts for tb in Ts if (tf == tb or not rot)]
    torch.manual_seed(20261006)
    res = {cname(*c): {p: [] for p in ("fwd", "fwd_bwd")} for c in combos}
    fns, infeasible, first = {}, {}, {}
    t0 = time.perf_counter()
    try:
        for tf, tb in combos:
            name = cname(tf, tb)
            try:
                call, leaves = build_rank_mixer(rot, R, g["H"] * g["P"], g["B"], g["L"], N=N_STATE, T=tf, T_bwd=tb, fp32=fp, seed=20261006)
                f = {p: make_pass(call, leaves, p) for p in ("fwd", "fwd_bwd")}
                for fn in f.values():
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                if tb != tf:
                    rec = [r for r in ua.chunk_bwd_log() if r.get("T_fwd") == tf][-1:]
                    if rec and rec[0].get("T_bwd") != tb:
                        infeasible[name] = "pair refused: %s" % rec[0].get("refused")
                        continue
                fns[name] = f
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:                                  # noqa: BLE001 (shared memory, compile)
                infeasible[name] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
    except torch.cuda.OutOfMemoryError as e:
        return dict(shape=a.shape, oom=True, error=str(e).splitlines()[0][:200])
    first = time.perf_counter() - t0
    for r in range(a.rounds):
        order = list(fns) if r % 2 == 0 else list(reversed(list(fns)))
        for name in order:
            for p, fn in fns[name].items():
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                e0.record()
                for _ in range(a.iters):
                    fn()
                e1.record()
                e1.synchronize()
                res[name][p].append(e0.elapsed_time(e1) / a.iters)
    med = {n: {p: statistics.median(v) for p, v in d.items() if v} for n, d in res.items() if n in fns}
    return dict(shape=a.shape, geometry=g, first_call_s=first, median_ms=med, infeasible=infeasible)


def child_verify(a):
    """One shape: PRODUCTION (calibration on, the rule / calibrated per-rank T and groups) against the OLD behaviour (T pinned to the fixed rule,
    the shared ROgt1 launch groups: `SSD_MIMO_RANK_GROUPS=0`), both through the production mixer, interleaved."""
    for k in [k for k in os.environ if k.startswith("SSD_MIMO_")]:
        del os.environ[k]
    import torch
    from benchmarks.bench_kernels import build_rank_mixer
    from benchmarks.harness import make_pass
    from orbit_ssm.kernels import shape_sets as ss
    g = ss.geometry(a.shape)
    R, rot, fp = a.R, a.route == "cx", a.dtype == "fp32"
    inc = INCUMBENT[R]["train"]
    kw = dict(N=N_STATE, fp32=fp, seed=20261006)
    try:
        nc, nl = build_rank_mixer(rot, R, g["H"] * g["P"], g["B"], g["L"], **kw)
        oc, ol = build_rank_mixer(rot, R, g["H"] * g["P"], g["B"], g["L"], T=inc[0], T_bwd=inc[1], **kw)
        base = {"new": make_pass(nc, nl, "fwd_bwd"), "old": make_pass(oc, ol, "fwd_bwd"),
                "new_fwd": make_pass(nc, nl, "fwd"), "old_fwd": make_pass(oc, ol, "fwd")}

        def arm(name, fn):
            def run():
                if name.startswith("old"):
                    os.environ["SSD_MIMO_RANK_GROUPS"] = "0"
                try:
                    return fn()
                finally:
                    os.environ.pop("SSD_MIMO_RANK_GROUPS", None)
            return run
        arms = {n: arm(n, f) for n, f in base.items()}
        for fn in arms.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        return dict(shape=a.shape, oom=True, error=str(e).splitlines()[0][:200])
    from orbit_ssm.ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
    kernel_width = max(16, ((G // R + 15) // 16) * 16)
    tf = _scan_chunk(R, kernel_width, R, True, real=not rot, dtype=torch.float32 if fp else torch.bfloat16)
    tnew = (tf, _scan_chunk_bwd(R, kernel_width, R, tf, real=not rot, dtype=torch.float32 if fp else torch.bfloat16))
    res = {n: [] for n in arms}
    for r in range(a.rounds):
        order = list(arms) if r % 2 == 0 else list(reversed(list(arms)))
        for name in order:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e0.record()
            for _ in range(a.iters):
                arms[name]()
            e1.record()
            e1.synchronize()
            res[name].append(e0.elapsed_time(e1) / a.iters)
    return dict(shape=a.shape, geometry=g, T_new=list(tnew), T_old=list(inc), median_ms={n: statistics.median(v) for n, v in res.items()})


# ------------------------------------------------------------------ driver --
def run_child(a, stage, route, R, dtype, shape):
    cmd = [sys.executable, os.path.abspath(__file__), "--child", stage, "--route", route, "--R", str(R), "--dtype", dtype,
           "--shape", shape, "--rounds", str(a.rounds), "--iters", str(a.iters), "--out", "/dev/stdout"]
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout)
    line = [x for x in p.stdout.splitlines() if x.startswith("@@RESULT ")]
    if p.returncode or not line:
        return dict(shape=shape, oom=True, error="child failed (exit %d): %s" % (p.returncode, (p.stderr or "").strip().splitlines()[-1:][:1]))
    print("[rank] %s %s %s R%d %s done in %.0fs" % (stage, route, dtype, R, shape, time.time() - t), flush=True)
    return json.loads(line[0][len("@@RESULT "):])


def raw_path(a, stage, sname):
    return os.path.join(a.out, "%s_%s.jsonl" % (stage, sname))


def load_raw(path):
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            r = json.loads(ln)
            out[r["shape"]] = r
    return out


def sets_of(a):
    return [(route, R, dt) for R in [int(x) for x in a.R.split(",")] for route in a.route.split(",") for dt in a.dtype.split(",")]


def cmd_screen(a, stage="screen"):
    os.makedirs(a.out, exist_ok=True)
    for route, R, dt in sets_of(a):
        sname = set_name(route, R, dt)
        path = raw_path(a, stage, sname)
        done = load_raw(path)
        for s in ([x for x in a.shapes.split(",") if x] or shapes_of(R)):
            if s in done:
                continue
            r = run_child(a, "time" if stage == "screen" else "verify", route, R, dt, s)
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")


def cmd_verify(a):
    cmd_screen(a, stage="verify")
    for route, R, dt in sets_of(a):
        d = load_raw(raw_path(a, "verify", set_name(route, R, dt)))
        for s, r in d.items():
            if r.get("oom"):
                print("verify %s %s: DROPPED %s" % (set_name(route, R, dt), s, r["error"]))
                continue
            m = r["median_ms"]
            print("verify %-18s %-24s T new %s old %s | f+b old %.3f new %.3f ms (x%.3f) | fwd old %.3f new %.3f ms" % (
                set_name(route, R, dt), s, r["T_new"], r["T_old"], m["old"], m["new"], m["old"] / m["new"], m["old_fwd"], m["new_fwd"]))


def _pick(score, incumbent):
    inc = score[incumbent]
    shapes = sorted(inc)
    rows = {}
    for c, t in score.items():
        rs = [inc[s] / t[s] for s in shapes]
        rows[c] = dict(geomean_gain=geomean(rs), worst_shape_gain=min(rs), per_shape=dict(zip(shapes, rs)))
    ok = [c for c in rows if c != incumbent and rows[c]["geomean_gain"] >= TOL_ON and rows[c]["worst_shape_gain"] >= TOL_WORST]
    return (max(ok, key=lambda c: rows[c]["geomean_gain"]) if ok else incumbent), rows


def decide_set(a, route, R, dt):
    data = {s: r for s, r in load_raw(raw_path(a, "screen", set_name(route, R, dt))).items() if not r.get("oom")}
    if len(data) < 3:
        return None
    Ts = CANDS[R]
    inc = INCUMBENT[R]
    shapes = sorted(data)
    out = dict(shapes=shapes, dropped={s: r["error"] for s, r in load_raw(raw_path(a, "screen", set_name(route, R, dt))).items() if r.get("oom")})
    fb = {}
    for tf in Ts:
        for tb in Ts:
            n = cname(tf, tb)
            if all(n in data[s]["median_ms"] for s in shapes):
                fb[n] = {s: data[s]["median_ms"][n]["fwd_bwd"] for s in shapes}
    fo = {}
    for T in Ts:
        n = cname(T, T)
        if all(n in data[s]["median_ms"] for s in shapes):
            fo[n] = {s: data[s]["median_ms"][n]["fwd"] for s in shapes}
    inc_t, inc_f = cname(*inc["train"]), cname(inc["fwd_only"], inc["fwd_only"])
    if inc_t not in fb or inc_f not in fo:
        return None
    bt, rows_t = _pick(fb, inc_t)
    bf, rows_f = _pick(fo, inc_f)
    tf_, tb_ = (int(x) for x in bt[1:].split("_"))
    out.update(train=dict(best=bt, rows={k: {x: v[x] for x in ("geomean_gain", "worst_shape_gain")} for k, v in rows_t.items()}),
               fwd_only=dict(best=bf, rows={k: {x: v[x] for x in ("geomean_gain", "worst_shape_gain")} for k, v in rows_f.items()}),
               structure=dict(train_fwd=tf_, train_bwd=tb_, fwd_only=int(bf[1:].split("_")[0])),
               incumbent=dict(train_fwd=inc["train"][0], train_bwd=inc["train"][1], fwd_only=inc["fwd_only"]))
    return out


def cmd_choose(a):
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import calibration_chunkpar as ccp
    res, th = {}, {}
    for route, R, dt in sets_of(a):
        d = decide_set(a, route, R, dt)
        sname = set_name(route, R, dt)
        if d is None:
            print("[rank] %s: no / incomplete screen" % sname)
            continue
        res[sname] = d
        st, inc = d["structure"], d["incumbent"]
        print("[rank] %-18s T_fwd/T_bwd %d/%d (fixed rule %d/%d), fwd-only %d (fixed %d) | train gains: %s | %d shapes, dropped %s" % (
            sname, st["train_fwd"], st["train_bwd"], inc["train_fwd"], inc["train_bwd"], st["fwd_only"], inc["fwd_only"],
            {k: "%.3f/%.3f" % (v["geomean_gain"], v["worst_shape_gain"]) for k, v in d["train"]["rows"].items()}, len(d["shapes"]),
            sorted(d["dropped"])))
        for kind in ccp.RANK_T_KINDS:
            th[ccp.rank_key(route == "real", R, R, kind, dt == "fp32")] = dict(
                value=int(st[kind]), kernel_width=max(16, ((G // R + 15) // 16) * 16), evidence="tools/calibrate_rank.py screen over %d shapes (calibration off), decision rule geomean >= %.3f and "
                "worst shape >= %.2f vs the fixed rule (%d)" % (len(d["shapes"]), TOL_ON, TOL_WORST, inc[kind]))
    json.dump(res, open(os.path.join(a.out, "rank_structure.json"), "w"), indent=1)
    gpu = cal.device_info()["gpu"] if not a.gpu else a.gpu
    p = (cal.shipped_path if a.write_shipped else cal.local_path)(ccp.FAMILY, gpu)
    old = cal._read(p) or {}
    keep = {k: v for k, v in old.get("thresholds", {}).items() if not k.startswith("rank|")}
    # re-decide ONLY the (route, rank, dtype) sets screened now: the others' stored rank keys stay
    done = {set_name(*s) for s in sets_of(a)}
    for k, v in old.get("thresholds", {}).items():
        if k.startswith("rank|"):
            route = k.split("|")[1]
            R = int(k.split("|")[2][1])
            dt = "fp32" if k.endswith("|fp32") else "bf16"
            if set_name(route, R, dt) not in done:
                keep[k] = v
    data = cal.build_file(ccp.FAMILY, {}, 5, old.get("provenance", {}).get("shapes", []), thresholds=dict(keep, **th),
                          dropped=old.get("provenance", {}).get("dropped_shapes"))
    print("wrote", cal.write_json(p, data), "rank keys:", sorted(k for k in dict(keep, **th) if k.startswith("rank|")))


def cmd_groups(a):
    """The fused cells / groups to measure for the chosen T_bwd != 32 entries (T_bwd = 32 is served by the shared `*_ROgt1` groups)."""
    cells, groups = [], []
    from orbit_ssm.kernels import shape_sets as ss
    for route, R, dt in sets_of(a):
        d = decide_set(a, route, R, dt)
        if d is None:
            continue
        for T in sorted({d["structure"]["train_bwd"]} - {32}):
            g = "%s_R%dO%d%s_t%d" % ("complex" if route == "cx" else "real", R, R, "_fp32" if dt == "fp32" else "", T)
            groups.append(g)
            cells += ["%s_r%do%d%st%d_%s" % ("cx" if route == "cx" else "real", R, R, "f32" if dt == "fp32" else "", T, t)
                      for t in (ss.SET if R == 2 else ss.LM_TAGS)]
    print(json.dumps(dict(groups=groups, cells=cells)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["screen", "choose", "groups", "verify"])
    ap.add_argument("--out", default="")
    ap.add_argument("--R", default="2,4")
    ap.add_argument("--route", default="real,cx")
    ap.add_argument("--dtype", default="bf16,fp32")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--child", default="")
    ap.add_argument("--shape", default="")
    a = ap.parse_args()
    if a.child:
        a.R = int(a.R)
        a.route, a.dtype = a.route, a.dtype
        r = globals()["child_" + a.child](a)
        print("@@RESULT " + json.dumps(r))
        return
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
