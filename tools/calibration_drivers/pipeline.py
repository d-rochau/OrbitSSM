"""The complete calibration of THIS GPU, in dependency order: `python tools/calibrate_kernels.py --all` (see its --help).

Every stage is one of the existing tools, run as a child process with the same interpreter; each writes the LOCAL
calibration file of its family (`calibration.local_dir`: $SSD_CALIBRATION_DIR, else ~/.cache/orbit_ssm/calibration), which the
loader reads before the shipped file.  `--write-shipped` (maintainers) passes through to every tool instead.

Order (each stage consumes what the earlier ones wrote):
  chunk structure  the chunk-parallel crossover (`cp_calibrate.py`), then -- where this tree ships the tools -- the real R = O = 1
                   chunk structure and its levers (`calibrate_real.py`) and the per-rank chunk lengths (`calibrate_rank.py`).
                   First: the launch-config groups are recorded at the chunk lengths these tables select, and a later change of
                   them sends a group to the generic grid (calibration._calibrated_at).
  launch configs   `calibrate_kernels.py --family ssd_mimo_fused`, then `--family ssd_mimo` (SISO).
  route table      the rotating R = O = 1 route rule: timings + eligibility (`calibration_drivers/routes.py`) -> `calibrate_routes.py`.
  levers           `calibrate_levers.py` (speed-only switches, fp64-stress eligibility).
  MH tables        `calibrate_mh_p1.py` (the P = 1 per-cell table; MH at P > 1 is served by its runtime tuner, nothing to store).
  check            `check_calibration.py`.

A finished stage leaves `<work>/done_<stage>`; a re-run skips it (resume after an interruption), `--force` repeats it.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")


@dataclasses.dataclass
class Stage:
    name: str
    group: str
    tool: str                      # path relative to the repository root: the stage is skipped when the tree lacks it
    steps: list                    # argv lists after the interpreter; "{work}" / "{shipped}" are filled in
    writes: str
    hours: str                     # expected wall time on one GPU (measured where stated)


def _real(dtype):
    dt = [] if dtype == "bf16" else ["--dtype", "fp32"]
    out = ["--out", "{work}/real_%s" % dtype]
    return [["tools/calibrate_real.py", s] + dt + out for s in ("elig", "screen", "choose", "ablate", "verify")] + \
        [["tools/calibrate_real.py", "decide"] + dt + out + ["{shipped}"]]


STAGES = [
    Stage("chunkpar_bf16", "chunk structure", "tools/cp_calibrate.py", [["tools/cp_calibrate.py", "{shipped}"]],
          "ssd_mimo_chunkpar (crossover, Ka warps, Kbc schedule)", "0.1-0.2 (the fp32 run: 7 min Blackwell, 10 min H100)"),
    Stage("chunkpar_fp32", "chunk structure", "tools/cp_calibrate.py",
          [["tools/cp_calibrate.py", "--dtype", "fp32", "{shipped}"]],
          "ssd_mimo_chunkpar (<part>_fp32 crossovers)", "0.1-0.2 (7 min Blackwell, 10 min H100)"),
    Stage("real_bf16", "chunk structure", "tools/calibrate_real.py", _real("bf16"),
          "ssd_mimo_chunkpar real_r1o1 T rule, ssd_mimo_levers real_* units", "1.0-1.6 (58 min Blackwell, 64 min H100, 98 min A100)"),
    Stage("real_fp32", "chunk structure", "tools/calibrate_real.py", _real("fp32"),
          "the same for fp32 IO (|fp32 keys)", "0.6-0.9 (36 min Blackwell, 52 min H100, 55 min A100)"),
    Stage("rank", "chunk structure", "tools/calibrate_rank.py",
          [["tools/calibrate_rank.py", "screen", "--out", "{work}/rank"],
           ["tools/calibrate_rank.py", "choose", "--out", "{work}/rank", "{shipped}"]],
          "ssd_mimo_chunkpar rank|<route>|R<R>O<R> chunk lengths", "not measured as one stage"),
    Stage("launch_fused", "launch configs", "tools/calibrate_kernels.py",
          [["tools/calibrate_kernels.py", "--family", "ssd_mimo_fused", "{force}", "{shipped}", "--raw", "{work}/raw_fused.json"]],
          "ssd_mimo_fused (every fused cell: 16-bit and fp32 groups)", "5-7 (16-bit 3.5 h A100; fp32 2 h Blackwell, 3.5 h H100/A100)"),
    Stage("launch_siso", "launch configs", "tools/calibrate_kernels.py",
          [["tools/calibrate_kernels.py", "--family", "ssd_mimo", "{force}", "{shipped}", "--raw", "{work}/raw_siso.json"]],
          "ssd_mimo (SISO kernels)", "not measured as one stage"),
    Stage("routes_bf16", "route table", "tools/calibrate_routes.py",
          [["tools/calibration_drivers/routes.py", "all", "--dtype", "bf16", "--work", "{work}/routes", "{shipped}"]],
          "ssd_mimo_route (rotating R1O1 route rule, bf16)", "not measured as one stage (18 cells x 5 routes x 2 + eligibility)"),
    Stage("routes_fp32", "route table", "tools/calibrate_routes.py",
          [["tools/calibration_drivers/routes.py", "all", "--dtype", "fp32", "--work", "{work}/routes", "{shipped}"]],
          "ssd_mimo_route (fp32 rule)", "not measured as one stage"),
    Stage("levers", "levers", "tools/calibrate_levers.py",
          [["tools/calibrate_levers.py", "{force}", "{shipped}", "--raw", "{work}/levers_raw.json"]],
          "ssd_mimo_levers (phase-law / complex units)", "3.5 (3 h 25 min A100)"),
    Stage("mh_p1", "MH tables", "tools/calibrate_mh_p1.py",
          [["tools/calibrate_mh_p1.py", "--run", "{work}/mh_p1"],
           ["tools/calibrate_mh_p1.py", "--derive", "{work}/mh_p1", "{shipped_or_local}"]],
          "mh p1_cells", "not measured as one stage (resumable per cell)"),
    Stage("check", "check", "tools/check_calibration.py", [["tools/check_calibration.py"]],
          "nothing (reports every family's file for this GPU)", "0 (< 1 min)"),
]


def _hit(st, sel):
    """A selector names a stage (`routes_bf16`), a stage-name prefix (`routes`, `real`) or a group (`launch configs`)."""
    return any(x in (st.name, st.group) or st.name.startswith(x + "_") for x in sel)


def plan(only=(), skip=()):
    unknown = [x for x in tuple(only) + tuple(skip) if not any(_hit(st, (x,)) for st in STAGES)]
    if unknown:
        raise SystemExit("unknown stage / group: %s (stages: %s)" % (", ".join(unknown), ", ".join(s.name for s in STAGES)))
    return [st for st in STAGES if (not only or _hit(st, only)) and not _hit(st, skip)]


def _argv(step, work, force, shipped):
    sub = {"{work}": work, "{force}": "--force" if force else None, "{shipped}": "--write-shipped" if shipped else None,
           "{shipped_or_local}": "--write-shipped" if shipped else "--local"}
    argv = []
    for x in step:
        for k, v in sub.items():
            if x == k:
                x = v
                break
            x = x.replace(k, v) if v else x
        if x is not None:
            argv.append(x)
    return [sys.executable, os.path.join(ROOT, argv[0])] + argv[1:]


def describe(stages, work, force=False, shipped=False, check_cli=False):
    """The plan as text; `check_cli`: every tool's `--help` must parse (the wiring, without a GPU)."""
    lines, bad = [], []
    for i, st in enumerate(stages, 1):
        present = os.path.exists(os.path.join(ROOT, st.tool))
        done = os.path.exists(os.path.join(work, "done_" + st.name))
        tag = "" if present else "   [not in this tree: skipped, its table keeps the fixed rule]"
        lines.append("%2d. %-14s %-16s %s%s%s" % (i, st.name, "(" + st.group + ")", "hours: " + st.hours,
                                                     "   [done]" if done else "", tag))
        lines.append("      writes: %s" % st.writes)
        for step in st.steps:
            argv = _argv(step, work, force, shipped)
            lines.append("      $ " + " ".join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in argv[1:]))
            if check_cli and present and st.name != "check":      # check_calibration.py has no options: it needs the GPU
                p = subprocess.run([argv[0], argv[1], "--help"], capture_output=True, text=True, cwd=ROOT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "")))
                ok = p.returncode == 0 and "usage" in p.stdout.lower()
                lines.append("        --help: %s" % ("ok" if ok else "FAILED (%s)" % (p.stderr.strip().splitlines() or ["?"])[-1][:160]))
                if not ok:
                    bad.append(st.name)
    return "\n".join(lines), bad


def run(stages, work, force=False, shipped=False, keep_going=False):
    os.makedirs(work, exist_ok=True)
    log = open(os.path.join(work, "pipeline.log"), "a")
    rc = 0
    for st in stages:
        if not os.path.exists(os.path.join(ROOT, st.tool)):
            print("[calibrate] %-14s not in this tree: skipped" % st.name, flush=True)
            continue
        marker = os.path.join(work, "done_" + st.name)
        if os.path.exists(marker) and not force:
            print("[calibrate] %-14s done (%s); --force repeats it" % (st.name, marker), flush=True)
            continue
        t0 = time.time()
        print("[calibrate] %-14s start %s (expected hours: %s)" % (st.name, time.strftime("%H:%M"), st.hours), flush=True)
        failed = False
        for step in st.steps:
            argv = _argv(step, work, force, shipped)
            log.write("$ %s\n" % " ".join(argv))
            log.flush()
            r = subprocess.run(argv, cwd=ROOT).returncode
            if r and st.name != "check":
                failed = True
                print("[calibrate] %-14s FAILED (exit %d): %s" % (st.name, r, " ".join(argv[1:])), flush=True)
                break
            rc |= r
        if failed:
            rc = 1
            if not keep_going:
                print("[calibrate] stopping: the later stages read this one's file (--keep-going runs them anyway)", flush=True)
                return rc
            continue
        open(marker, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        print("[calibrate] %-14s done in %.1f h" % (st.name, (time.time() - t0) / 3600), flush=True)
    return rc
