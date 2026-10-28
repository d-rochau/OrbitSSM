#!/usr/bin/env python
"""Launch calibration of the production kernels on THIS GPU.

    python tools/calibrate_kernels.py --all      # A GPU THE LIBRARY HAS NOT BEEN VALIDATED ON: run this
                                                 # once.  Every table of this GPU, in dependency order
                                                 # (chunk structure, launch configs, route table, levers,
                                                 # MH), into the LOCAL files; resumable (--work DIR).
                                                 # Expect most of a day on one GPU (--list prints the
                                                 # stages with their measured times).
    python tools/calibrate_kernels.py --all --list               # the plan + each tool's CLI check, no GPU
    python tools/calibrate_kernels.py --all --skip levers,mh_p1  # (or --only <stage|group>,...)

    python tools/calibrate_kernels.py            # calibrate into the LOCAL file
                                                 # (no-op when a valid one exists)
    python tools/calibrate_kernels.py --force    # full local recalibration
    python tools/calibrate_kernels.py --verify   # compare SHIPPED vs this GPU's
                                                 # best; write a local override
                                                 # only if shipped is >3% worse
    python tools/calibrate_kernels.py --reset    # delete the local file (shipped
                                                 # defaults apply again)
    python tools/calibrate_kernels.py --write-shipped   # maintainers: write the
                                                 # repo's shipped file for this GPU

Per kernel and representative shape (own process, own Triton cache under
/dev/shm) the FULL grid is searched; per-config time is the median over
--rounds (>= 5) forced re-tunes.  `derive` (src/orbit_ssm/kernels/calibration.py)
then fixes the default config (min worst-case slowdown over the shapes) and a
2-4 config LIGHT set (stages fixed, warps/maxnreg varied).  Runtime use is
opt-in: SSD_MIMO_CALIBRATION=on.

A family plugs in through its adapter module: it calls
`calibration.register_family(...)` (kernels, code files, shapes) and provides
`child_run(shape_name, rounds) -> {kernel: {cfg_key: ms}}` (see
`ssd_mimo` below).  Measurement drivers that build calls from the repo-only
`benchmarks` package live in `tools/calibration_drivers/` (the fused SSD
workload, the MH child), never in the installed `orbit_ssm` package.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, str(ROOT) + "/src")

ADAPTERS = {"ssd_mimo": "orbit_ssm.kernels.ssd_mimo.calibration_ssd",
            "ssd_mimo_fused": "orbit_ssm.kernels.ssd_mimo.calibration_fused"}


def _adapter(fam):
    import importlib
    return importlib.import_module(ADAPTERS[fam])


# --------------------------------------------------------------- the child --
def child_ssd_mimo(shape, rounds):
    """Runs INSIDE its own process (env fixed before any orbit_ssm import)."""
    os.environ.update(SSD_MIMO_CALIBRATION="off", SSD_MIMO_SISO_FAST="1",
                      SSD_MIMO_SISO_AUTOTUNE="full",
                      SSD_MIMO_SISO_ADJ_AUTOTUNE="full")
    import statistics
    import torch
    from benchmarks import baselines as bl
    from benchmarks.harness import make_pass
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import calibration_ssd as cs
    from orbit_ssm.kernels.ssd_mimo import siso_fwd_kernel as sf
    from orbit_ssm.kernels.ssd_mimo import siso_bwd_kernel as sb
    g = cs.SHAPES[shape]
    torch.manual_seed(20261003)
    if g.get("rot", True) and g.get("dtype") == "fp32":
        # the fp32 group: fp32 IO, tf32 operands (torch's default dtype; official Mamba's tl.dot default), at the
        # GPU's fp32 SISO chunk -- the group's recorded chunk (`calibration_ssd._chunk_of`)
        fn = make_pass(*bl.build_m3_eq(g["B"], g["H"], g["P"], g["N"], g["L"],
                                       arm="ours", chunk=cs._chunk_of(cs.group_key(**g)),
                                       angles="m3", trap_off=False,
                                       dtype=torch.float32, precision="tf32"), "fwd_bwd")
    elif g.get("rot", True):
        fn = make_pass(*bl.build_m3_eq(g["B"], g["H"], g["P"], g["N"], g["L"],
                                       arm="ours", chunk=64, angles="m3",
                                       trap_off=False, dtype=torch.bfloat16,
                                       precision="io"), "fwd_bwd")
    else:
        fn = cs.real_workload(g)
    _CFG_NAME["f"] = lambda c: cal.cfg_key(c.num_warps, c.num_stages, c.maxnreg)
    ks = {"siso_fwd": sf._siso_fwd_kernel, "siso_bwd": sb._siso_bwd_scan_kernel,
          "siso_adj": sb._siso_rot_adjoint_kernel}
    t0 = time.perf_counter()
    fn(); torch.cuda.synchronize()
    first = time.perf_counter() - t0
    per = {n: {} for n in ks}
    for _ in range(rounds):
        for k in ks.values():
            k.cache.clear()                   # force a fresh full search
            k.configs_timings = None
        fn(); torch.cuda.synchronize()
        for n, k in ks.items():
            for c, v in (getattr(k, "configs_timings", None) or {}).items():
                t = v[0] if isinstance(v, (list, tuple)) else v
                if t == t and t != float("inf"):
                    per[n].setdefault(cal.cfg_key(c.num_warps, c.num_stages,
                                                  c.maxnreg), []).append(t)
    out = {n: {c: statistics.median(v) for c, v in d.items()}
           for n, d in per.items()}
    return dict(shape=shape, geometry=g, first_call_s=first, rounds=rounds,
                timings_ms=out,
                rounds_seen={n: {c: len(v) for c, v in d.items()}
                             for n, d in per.items()},
                served=cs.describe_served())


def _collect(kernels, run, rounds):
    """Force `rounds` fresh full searches of every kernel in `kernels`
    ({name: Triton Autotuner}) around `run()` and return the per-config
    median of the autotuner's own timing table plus the first-call time."""
    import statistics
    import torch
    from orbit_ssm.kernels import calibration as cal
    t0 = time.perf_counter()
    run(); torch.cuda.synchronize()
    first = time.perf_counter() - t0
    per = {n: {} for n in kernels}
    fam_key = None
    for _ in range(rounds):
        for k in kernels.values():
            k.cache.clear()                   # force a fresh full search
            k.configs_timings = None
        run(); torch.cuda.synchronize()
        for n, k in kernels.items():
            for c, v in (getattr(k, "configs_timings", None) or {}).items():
                t = v[0] if isinstance(v, (list, tuple)) else v
                if t == t and t != float("inf"):
                    per[n].setdefault(_cfg_name(c), []).append(t)
    return first, {n: {c: statistics.median(v) for c, v in d.items()}
                   for n, d in per.items()}, {
        n: {c: len(v) for c, v in d.items()} for n, d in per.items()}


_CFG_NAME = {}


def _cfg_name(c):
    return _CFG_NAME["f"](c)


def child_ssd_mimo_fused(shape, rounds):
    os.environ.update(SSD_MIMO_CALIBRATION="off", SSD_MIMO_AUTOTUNE="full",
                      SSD_MIMO_SISO_FAST="0")
    import torch
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.ssd_mimo import calibration_fused as cf
    from tools.calibration_drivers.ssd_fused import workload   # repo-only (imports benchmarks)
    _CFG_NAME["f"] = lambda c: cf._obj_key(c)
    g = cf.SHAPES[shape]
    torch.manual_seed(20261003)
    first, med, seen = _collect(cf.autotuners(), workload(g), rounds)
    only = g.get("only_kernels")          # fallback cells feed ONLY the kernels they exist to calibrate
    if only:
        med = {k: v for k, v in med.items() if k in only}
        seen = {k: v for k, v in seen.items() if k in only}
    return dict(shape=shape, geometry=g, first_call_s=first, rounds=rounds,
                timings_ms=med, rounds_seen=seen, served=cf.describe_served())


# ------------------------------------------------------------ orchestration --
def measure(fam, shapes, rounds, timeout):
    base = "/dev/shm/calib_%s_%d" % (os.environ.get("SLURM_JOB_ID", "x"),
                                     os.getpid())
    res = {}
    dropped = {}
    # ONE Triton compile cache for the whole run (per-shape caches made the
    # first call of every child a 100-180 s recompile).  Timing is unaffected:
    # every child clears the autotuners' caches and re-measures all configs.
    cdir = os.path.join(base, "triton")
    os.makedirs(cdir, exist_ok=True)
    try:
        for s in shapes:
            out = os.path.join(base, s + ".json")
            env = dict(os.environ, TRITON_CACHE_DIR=cdir,
                       PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, "src")]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
            t = time.time()
            p = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--family", fam, "--child", s, "--rounds",
                                str(rounds), "--out", out], env=env,
                               timeout=timeout)
            if p.returncode:
                raise SystemExit("shape %s failed (exit %d)" % (s, p.returncode))
            r = json.load(open(out))
            if r.get("oom"):
                dropped[s] = r["error"]
                print("[calibrate] %s DROPPED (out of memory): %s" % (s, r["error"]),
                      flush=True)
                continue
            res[s] = r
            print("[calibrate] %s done in %.0fs (first call %.1fs)" % (
                s, time.time() - t, res[s]["first_call_s"]), flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    res["__dropped__"] = dropped
    return res


def timings_by_kernel(res):
    ks = {}
    for s, r in res.items():
        for k, t in r["timings_ms"].items():
            if t:
                ks.setdefault(k, {})[s] = t
    return ks


def report(fam, data):
    fmt = lambda c: "w%d/s%d/r%s" % (c["num_warps"], c["num_stages"], c["maxnreg"])
    for k, v in data["kernels"].items():
        for gname, g in [("default", v["default"])] + sorted(v["groups"].items()):
            print("  %-9s %-10s fixed %-12s worst %.3f | light %s (worst %.3f) "
                  "shapes %s%s" % (
                      k, gname, fmt(g["fixed"]), g["worst_slowdown"],
                      [fmt(c) for c in g["light"]], g["light_worst_slowdown"],
                      g["shapes"],
                      "  UNCOVERED " + str(g["uncovered_shapes"])
                      if g["uncovered_shapes"] else ""))
            if "occupancy" in g:
                o = g["occupancy"]
                print("    OCCUPANCY RULE %s/%s: blocks/SM < %.3f -> %s (worst %.3f, "
                      "shapes %s) else %s (worst %.3f, shapes %s)  => worst-case "
                      "slowdown WITH rule %.3f (was %.3f)" % (
                          k, gname, o["threshold"], fmt(o["low"]["fixed"]),
                          o["low"]["worst_slowdown"], o["low"]["shapes"],
                          fmt(o["high"]["fixed"]), o["high"]["worst_slowdown"],
                          o["high"]["shapes"], o["worst_slowdown"],
                          g["worst_slowdown"]))
            far = {m: v for m, v in g.get("per_shape_slowdown", {}).items()
                   if v > 1.05 and "occupancy" not in g}
            if far:
                print("  REPORT %s/%s: the ONE fixed config is >5%% off the "
                      "per-shape best on %s" % (k, gname, {
                          m: "%.1f%%" % ((v - 1) * 100)
                          for m, v in sorted(far.items())}))


def verify(fam, sdata, tk, fm, cal):
    """Per kernel and shipped group: shipped fixed config vs the fresh best
    on that group's shapes.  True when any is > 3% worse (override)."""
    override = False
    for k, t in tk.items():
        sk = sdata["kernels"].get(k)
        if sk is None:
            print("  %s: not in shipped -> override" % k)
            override = True
            continue
        for gname, g in [("default", sk["default"])] + sorted(sk["groups"].items()):
            sub = {m: t[m] for m in g["shapes"] if m in t}
            if not sub:
                continue
            o, sh, fr = cal.verify_decision(fm["cfg_key"](g["fixed"]), sub)
            print("  %-9s %-10s shipped worst-case %.3f vs fresh best %.3f -> %s" % (
                k, gname, sh, fr, "OVERRIDE (>3%% worse)" if o else "keep shipped"))
            override |= o
    return override


def _pipeline(a):
    """`--all`: the complete calibration of this GPU (tools/calibration_drivers/pipeline.py)."""
    sys.path.insert(0, os.path.join(ROOT, "tools", "calibration_drivers"))
    import pipeline as P
    split = lambda v: tuple(x.strip() for x in v.split(",") if x.strip())
    stages = P.plan(only=split(a.only), skip=split(a.skip))
    work = os.path.abspath(a.work or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                                                  "orbit_ssm", "calibrate_run"))
    if a.list:
        text, bad = P.describe(stages, work, a.force, a.write_shipped, check_cli=True)
        print("calibration plan (work dir %s; %s files):\n%s" % (work, "SHIPPED" if a.write_shipped else "local", text))
        if bad:
            print("CLI check failed for: " + ", ".join(bad))
        return 1 if bad else 0
    return P.run(stages, work, force=a.force, shipped=a.write_shipped, keep_going=a.keep_going)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n")[1])
    ap.add_argument("--all", action="store_true",
                    help="the complete calibration of this GPU, every family in dependency order (see above)")
    ap.add_argument("--list", action="store_true", help="with --all: print the stages, commands and expected times; "
                    "check that every tool's command line parses; run nothing")
    ap.add_argument("--only", default="", help="with --all: comma list of stages or groups to run")
    ap.add_argument("--skip", default="", help="with --all: comma list of stages or groups to leave out")
    ap.add_argument("--work", default="", help="with --all: resumable work dir (default ~/.cache/orbit_ssm/calibrate_run)")
    ap.add_argument("--keep-going", action="store_true", help="with --all: run the later stages after a failed one")
    ap.add_argument("--family", default="ssd_mimo",
                    choices=sorted(ADAPTERS) + ["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--shapes", default="", help="comma list (default: all)")
    ap.add_argument("--part", default="", help="i/n: measure only every n-th shape "
                    "starting at i (split one family over several GPUs of the SAME "
                    "model; merge the --raw files with --from-raw a.json,b.json)")
    ap.add_argument("--raw", default="", help="also dump raw per-shape results")
    ap.add_argument("--gpu", default="", help="with --from-raw: GPU name of the raw file")
    ap.add_argument("--sms", type=int, default=0, help="with --from-raw: SM count of that GPU")
    ap.add_argument("--pool", default="",
                    help="with --from-raw: SECOND-pass raw file(s) (comma list) of the same GPU; shapes "
                         "both passes measured use the MEAN timing per config, so the fixed config "
                         "(minimax) and the light sets are not decided by a near-tie flip of one pass; "
                         "recorded as provenance.fixed_basis = 'pooled(pass1+pass2 on N shapes)'")
    ap.add_argument("--from-raw", default="",
                    help="derive from a saved --raw file (no measuring; GPU "
                         "identity = the current device)")
    ap.add_argument("--merge-groups", default="",
                    help="comma list of shape groups: derive ONLY these groups from the measured / --from-raw cells and MERGE them "
                         "into the existing file of the target layer (--write-shipped: the shipped file, else the local one); "
                         "the file's other groups / defaults are kept (calibration.merge_groups)")
    ap.add_argument("--restamp-from", default="",
                    help="with --merge-groups: accept a target file whose kernel code hash is this OLDER value (assert that "
                         "the source change is bit-identical for its groups); recorded in provenance.restamped")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--child", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.all:
        raise SystemExit(_pipeline(a))
    if a.rounds < 5 and not a.child:
        raise SystemExit("--rounds must be >= 5")
    if a.family == "all":           # every registered family, one process each
        rc = 0
        for fam in sorted(ADAPTERS):
            argv = [x for x in sys.argv[1:]]
            i = argv.index("--family")
            argv[i + 1] = fam
            raw = [j for j, x in enumerate(argv) if x == "--raw"]
            for j in raw:                      # one raw file per family
                argv[j + 1] += "." + fam
            rc |= subprocess.run([sys.executable, os.path.abspath(__file__)]
                                 + argv).returncode
        raise SystemExit(rc)
    if a.child:
        # BEFORE any orbit_ssm import: the child sets the family's env (SSD_MIMO_AUTOTUNE=full, ...) first thing, and the
        # kernel modules fix their decorator grids AT IMPORT.  Importing the adapter here first would import `orbit_ssm.kernels.ssd_mimo` -> autograd -> fused_bwd_kernel with SSD_MIMO_AUTOTUNE unset, so
        # every `ssd_mimo_fused` "full" search would run the LIGHT tier (warps x 1 stage, no maxnreg; Ka its own stages).
        import torch
        try:
            r = globals()["child_" + a.family](a.child, a.rounds)
        except torch.cuda.OutOfMemoryError as e:
            r = dict(shape=a.child, oom=True, error=str(e).splitlines()[0][:200])
        except Exception as e:                  # noqa: BLE001
            # a per-rank cell (`calibration_fused` RANK cells `<route>_r<R>o<R>[f32]t<T>_<shape>`) that does not compile / fit at its T is DROPPED
            # by name (like an out-of-memory cell); every other cell keeps failing the run
            import re
            if not re.match(r"(cx|real)_r[248]o[248](f32)?t\d+_", a.child):
                raise
            r = dict(shape=a.child, oom=True, error="%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else ""))
        json.dump(r, open(a.out, "w"))
        return
    ad = _adapter(a.family)
    from orbit_ssm.kernels import calibration as cal
    if a.gpu and a.sms:          # re-derive another GPU's raw file offline
        import triton as _t
        cal._DEV.update(gpu=a.gpu, sms=a.sms, driver=None,
                        triton=_t.__version__, torch="offline")
    dev = cal.device_info()
    gpu = dev["gpu"]
    if a.reset:
        print("local calibration %s: %s" % (
            cal.local_path(a.family, gpu),
            "deleted" if cal.reset_local(a.family, gpu) else "none"))
        return
    shapes = [s for s in (a.shapes.split(",") if a.shapes else list(ad.SHAPES)) if s]
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        shapes = shapes[i::n]
    data, layer = cal.load(a.family, gpu, dev["triton"])
    if not (a.force or a.verify or a.write_shipped):
        if layer == "local":
            print("valid local calibration exists (%s); --force to redo" %
                  cal.local_path(a.family, gpu))
            return
    if a.from_raw:
        res, dr = {}, {}
        for f in a.from_raw.split(","):
            part = json.load(open(f))
            dr.update(part.pop("__dropped__", {}))
            res.update(part)
        res["__dropped__"] = dr
    else:
        res = measure(a.family, shapes, a.rounds, a.timeout)
    dropped = res.pop("__dropped__", {})
    fixed_basis = None
    if a.pool:
        res2 = {}
        for f in a.pool.split(","):
            part = json.load(open(f))
            part.pop("__dropped__", None)
            res2.update(part)
        shared = [s for s in res if s in res2]
        for s in shared:
            for k, t1 in res[s]["timings_ms"].items():
                t2 = res2[s]["timings_ms"].get(k) or {}
                res[s]["timings_ms"][k] = {c: (v + t2[c]) / 2 if c in t2 else v for c, v in t1.items()}
        fixed_basis = "pooled(pass1+pass2 mean on %d of %d shapes; pass1 alone elsewhere)" % (len(shared), len(res))
    shapes = list(res)
    if a.raw:
        json.dump(dict(res, __dropped__=dropped), open(a.raw, "w"), indent=1)
    if a.part:
        print("part %s measured (%d shapes, %d dropped); merge the raw files with "
              "--from-raw" % (a.part, len(res), len(dropped)))
        return
    tk = timings_by_kernel(res)
    fresh = cal.build_file(a.family, tk, a.rounds, shapes, dropped=dropped, fixed_basis=fixed_basis)
    print("fresh calibration on %s (triton %s, code %s):" % (
        gpu, dev["triton"], fresh["provenance"]["code_hash"]))
    report(a.family, fresh)
    if a.merge_groups:
        target = (cal.shipped_path if a.write_shipped else cal.local_path)(a.family, gpu)
        base = cal._read(target)
        if base is None:
            raise SystemExit("--merge-groups: no file to merge into at %s" % target)
        merged = cal.merge_groups(base, fresh, [g for g in a.merge_groups.split(",") if g],
                                  restamp_from=a.restamp_from or None)
        print("merged groups %s into %s" % (a.merge_groups, target))
        print("wrote", cal.write_json(target, merged))
        return
    if a.write_shipped:
        print("wrote", cal.write_json(cal.shipped_path(a.family, gpu), fresh))
        return
    if a.verify:
        sdata = cal._read(cal.shipped_path(a.family, gpu))
        why = None if sdata is None else cal.check_file(
            sdata, a.family, gpu, dev["triton"])
        if sdata is None or why:
            print("no valid shipped file (%s): writing local" % (why or "none"))
            override = True
        else:
            override = verify(a.family, sdata, tk, cal.family(a.family), cal)
        if not override:
            print("shipped calibration holds; no local file written")
            return
    print("wrote", cal.write_json(cal.local_path(a.family, gpu), fresh))


if __name__ == "__main__":
    main()
