#!/usr/bin/env python
"""Per-cell calibration of the MH scan at P = 1 (the exact Mamba-1 path) on THIS GPU.

    # measure (resumable: cells already in OUT/cells.jsonl are skipped)
    python tools/calibrate_mh_p1.py --run OUT [--rounds 7] [--part i/n] [--cells substr,...]
    # build the shipped file from one or more measured dirs
    python tools/calibrate_mh_p1.py --derive OUT[,OUT2] --write-shipped
    python tools/calibrate_mh_p1.py --derive OUT --out FILE      # any other path
    python tools/calibrate_mh_p1.py --derive OUT --local         # this GPU's local file (users)

Cells = `calibration_mh.P1_SHAPES`: the shared shape set (`shape_sets.py`, H = d_inner)
x N in {16, 128} x {bf16, fp32} x feature class {bare, m1}.  Per cell, in ONE process:

 1. REFERENCE: the runtime tuner (calibration off, tier light) runs the cell once for
    the inference forward, the checkpoint-writing forward and the backward; its three
    winning configs are captured (`autotune.choose_axes`) and seed the candidate list.
 2. SEARCH: each side's candidate space (`calibration_mh.p1_fwd_axes` / `p1_bwd_axes`:
    the tuner's axes over their whole legal range, in staged products, plus the FPACK
    corners) is timed by replaying every config through the MH pins (CUDA events).
 3. SELECT: the finalists (top by coarse time + the tuner's winners) are re-timed with
    interleaved rounds (median); the best finalist that (a) AGREES with the tuner's
    output at the tuner's own tolerance (`autotune._agrees`), (b) stays finite at the
    extreme decay rate the tuner probes (K5) and (c) peaks at <= 1.10x the tuner
    config's memory is taken: a speed-only search selects wrong or fat answers.
 4. VERIFY: the production path with the table injected vs the production path on the
    tuner, same process, interleaved rounds: accepted only if calibrated <= 1.02 x
    tuner on the inference forward and on forward+backward.  A failing cell falls back
    to the tuner's own winners; if that fails too the cell is NOT written (it keeps
    running on the tuner).

The file: `src/orbit_ssm/kernels/calibration_data/mh/<gpu>.json`, key `p1_cells`; its
provenance (GPU, Triton, kernel code hash of `calibration_mh._CODE`) makes it inert
when any of them changes.
"""
import argparse
import itertools
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, str(ROOT) + "/src")

ACCEPT = 1.02
MEM_FACTOR = 1.10
KEEP = 6                                    # finalists besides the tuner's


def order(names, shapes):
    def k(n):
        g = shapes[n]
        return (g["cls"] != "m1", not n.startswith("lm_"), g["precision"] != "bf16",
                g["N"] != 16, g["B"] * g["H"] * g["L"], n)
    return sorted(names, key=k)


# ----------------------------------------------------------------- timing --
def events(fn, sync_ms=20.0):
    """CUDA-event ms per call, enough calls for ~`sync_ms` (first call warms)."""
    import math
    import torch
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); fn(); b.record(); torch.cuda.synchronize()
    n = max(2, min(100, math.ceil(sync_ms / max(a.elapsed_time(b), 1e-3))))
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def peak_mb(fn):
    import torch
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


class Cell:
    """One built cell: inputs, the three timed closures and the pin plumbing."""

    def __init__(self, g):
        import torch
        from benchmarks.bench_kernels import _mh_inputs
        from orbit_ssm.kernels.hybrid_mamba_mh import mh_scan
        self.torch, self.g = torch, g
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[g["precision"]]
        torch.manual_seed(20261006)
        args, kw, leaves = _mh_inputs(g["B"], g["H"], 1, g["N"], g["L"], dtype=dtype)
        if g["cls"] == "m1":
            kw["delta_bias"] = (torch.randn(g["H"], device="cuda") * 0.2).requires_grad_()
            kw["d_skip"] = torch.randn(g["H"], device="cuda").requires_grad_()
            leaves = leaves + [kw["delta_bias"], kw["d_skip"]]
        self.args, self.kw, self.leaves, self.fn = args, kw, leaves, mh_scan
        with torch.no_grad():
            y = self.call()
        self.dy = torch.randn_like(y)
        del y

    def call(self, args=None):
        return self.fn(*(args or self.args), **self.kw)

    def f_inf(self):
        with self.torch.no_grad():
            self.call()

    def f_trn(self):
        self.call()

    def fb(self):
        for t in self.leaves:
            t.grad = None
        self.call().backward(self.dy)

    def graph(self):
        """A training forward kept alive: `bwd()` replays only the backward."""
        self.y = self.call()

        def bwd():
            for t in self.leaves:
                t.grad = None
            self.y.backward(self.dy, retain_graph=True)
        self.bwd = bwd
        return bwd

    def grads(self):
        return tuple(t.grad for t in self.leaves if t.grad is not None)

    def extreme_ok(self):
        """K5: a forward + backward at the decay rate the tuner probes stays finite."""
        torch = self.torch
        from orbit_ssm.kernels.hybrid_mamba_mh import autotune
        from orbit_ssm.kernels.hybrid_mamba_mh.bwd_gluon import _EXTREME_A_LOG
        a = list(self.args)
        a[2] = torch.full_like(a[2], _EXTREME_A_LOG).requires_grad_()
        for t in self.leaves:
            t.grad = None
        y = self.call(a)
        y.backward(self.dy)
        return autotune._all_finite((y,) + tuple(
            t.grad for t in self.leaves if t.grad is not None)) and a[2].grad is not None \
            and autotune._all_finite(a[2].grad)


def cfg_key(c):
    return ",".join("%s=%d" % (k, int(c[k])) for k in sorted(c))


class Pins:
    """Set `HYBRID_MH_{FWD,BWD}_CONFIG` around a call (the pin the table is served as)."""

    def __init__(self, fwd=None, bwd=None):
        self.f, self.b = fwd, bwd

    def __enter__(self):
        for side, c in (("FWD", self.f), ("BWD", self.b)):
            if c is not None:
                os.environ["HYBRID_MH_%s_CONFIG" % side] = cfg_key(c)
        return self

    def __exit__(self, *e):
        os.environ.pop("HYBRID_MH_FWD_CONFIG", None)
        os.environ.pop("HYBRID_MH_BWD_CONFIG", None)


def project(cfg, keys, drop=()):
    out = {}
    for k in keys:
        if k in drop or k not in cfg:
            continue
        v = cfg[k]
        if k == "MAXNREG":
            v = 0 if v is None else v
        if v is None:
            continue
        out[k] = int(v)
    return out


# ----------------------------------------------------------------- search --
class Side:
    """Candidate bookkeeping of one side: coarse timings, finalists, selection."""

    def __init__(self, name, run, setpin, tuner_cfgs):
        self.name, self.run, self.setpin = name, run, setpin
        self.T = {}                                  # key -> (ms, cfg)
        self.seeds = [c for c in tuner_cfgs if c is not None]
        self.n = 0

    def time(self, cfg, sync_ms=8.0):
        import torch
        k = cfg_key(cfg)
        if k in self.T:
            return self.T[k][0]
        ms = None
        try:
            with self.setpin(cfg):
                ms = events(self.run, sync_ms)
        except Exception:                    # not lowerable / illegal / OOM: absent
            ms = None
            torch.cuda.empty_cache()
        self.T[k] = (ms, dict(cfg))
        self.n += 1
        return ms

    def sweep(self, cands):
        best = None
        for c in cands:
            ms = self.time(c)
            if ms is not None and (best is None or ms < best[0]):
                best = (ms, c)
        return best

    def best(self):
        ok = [(v[0], v[1]) for v in self.T.values() if v[0] is not None]
        return min(ok, key=lambda x: x[0]) if ok else None

    def finalists(self):
        ok = sorted((v for v in self.T.values() if v[0] is not None),
                    key=lambda v: v[0])
        out, seen = [], set()
        for c in [v[1] for v in ok[:KEEP]] + self.seeds:
            if cfg_key(c) not in seen:
                seen.add(cfg_key(c)); out.append(c)
        return out


def vary(base, **axes):
    """Product of `axes` values over `base` (dict of overrides; skips no-ops)."""
    names = list(axes)
    for vals in itertools.product(*(axes[n] for n in names)):
        c = dict(base, **dict(zip(names, vals)))
        if c.get("NS", 2) == 1 and c.get("AGX", 1) == 0:      # AGX needs NS > 1: same kernel
            continue
        yield c


def descend(side, best, axes, passes=1):
    """One-axis-at-a-time re-fit around `best` (cfg dict) over `axes`."""
    for _ in range(passes):
        for name, vals in axes.items():
            r = side.sweep(vary(best, **{name: vals}))
            if r is not None and r[0] <= side.T[cfg_key(best)][0]:
                best = r[1]
    return best


def tune_fwd(side, cm, g, ckpt_l, ts_ok, infer, base):
    """Staged product over the forward's space, then FPACK corners and a re-fit."""
    axes = cm.p1_fwd_axes(g["H"], g["L"], ckpt_l, ts_ok, infer)
    core = {k: axes[k] for k in ("BLOCK_L", "PF", "NUM_WARPS")}
    side.time(dict(base))
    r = side.sweep(vary(dict(base, TS=1), **core))
    best = r[1] if r else dict(base)
    rest = {k: axes[k] for k in ("TS", "CKV") if k in axes}
    if rest:
        r = side.sweep(vary(best, **rest))
        best = r[1] if r and r[0] < side.T[cfg_key(best)][0] else best
    side.sweep(cm.p1_fwd_fpack_corners(g["H"], g["L"], ckpt_l))
    cur = side.best()[1]
    descend(side, cur, axes)
    return side


def tune_bwd(side, cm, g, nblk, base):
    """Staged products over the backward's space (a full product is ~10^4 compiles):
    the coupled (HP, TP_P, MAXNREG) corners first, then warps, then (NS, AGX, TS), then
    BCS, then a one-axis re-fit of every axis around the best."""
    axes = cm.p1_bwd_axes(g["H"], g["N"], nblk)
    side.time(dict(base))
    r = side.sweep(vary(base, **{k: axes[k] for k in ("HP", "TP_P", "MAXNREG")}))
    best = r[1] if r else dict(base)
    r = side.sweep(vary(best, NUM_WARPS=axes["NUM_WARPS"]))
    best = r[1] if r else best
    r = side.sweep(vary(best, NS=axes["NS"], AGX=axes["AGX"], TS=axes["TS"]))
    best = r[1] if r else best
    r = side.sweep(vary(best, BCS=axes["BCS"]))
    best = r[1] if r else best
    descend(side, best, {k: axes[k] for k in ("HP", "TP_P", "MAXNREG", "NUM_WARPS",
                                              "NS", "AGX", "TS", "BCS")})
    return side


def select(side, rounds, check, ref_peak):
    """Interleaved re-time of the finalists; the first (fastest) one passing `check`
    and the memory bound.  Returns (cfg, median ms, {key: ms}, rejected)."""
    fin = side.finalists()
    ts = {cfg_key(c): [] for c in fin}
    for _ in range(rounds):
        for c in fin:
            try:
                with side.setpin(c):
                    ts[cfg_key(c)].append(events(side.run, 25.0))
            except Exception:
                ts[cfg_key(c)].append(float("inf"))
    med = {k: statistics.median(v) for k, v in ts.items()}
    rejected = {}
    for c in sorted(fin, key=lambda c: med[cfg_key(c)]):
        k = cfg_key(c)
        if med[k] == float("inf"):
            continue
        why = check(c)
        if why is None:
            return c, med[k], med, rejected
        rejected[k] = why
    return None, None, med, rejected


# ------------------------------------------------------------ one cell ----
def measure_cell(name, g, rounds):
    import torch
    from orbit_ssm.kernels.hybrid_mamba_mh import autotune, bwd_gluon, calibration_mh as cm
    from orbit_ssm import kernel_options as ko
    t_cell = time.time()
    FK, BK = ko.MH_FWD_CONFIG_KEYS, ko.MH_BWD_CONFIG_KEYS
    for k in ("HYBRID_MH_FWD_CONFIG", "HYBRID_MH_BWD_CONFIG", "MH_BWD_MAXNREG",
              "HYBRID_MH_AUTOTUNE"):
        os.environ.pop(k, None)
    os.environ.update(HYBRID_MH_CALIBRATION="off", HYBRID_MH_TUNE_CACHE="off")
    cell = Cell(g)
    autotune._CACHE.clear()          # the cell's own build already ran the tuner once
    try:
        bwd_gluon._PLANS.clear()
    except Exception:
        pass
    key = cm.p1_cell_key(g["B"], g["H"], g["N"], g["L"], {"bf16": torch.bfloat16,
                         "fp32": torch.float32}[g["precision"]], cm.P1_FEATS[g["cls"]])
    from orbit_ssm.kernels.hybrid_mamba_mh.bwd_gluon import auto_block_l
    ckpt_l = auto_block_l(1, g["L"])
    nblk = -(-g["L"] // ckpt_l)

    # 1. the tuner's winners (production default tier), per side
    cap = []
    orig = autotune.choose_axes

    def wrap(key_, base, axes, run, **kw):
        r = orig(key_, base, axes, run, **kw)
        cap.append((key_[0], dict(r)))
        return r
    autotune.choose_axes = wrap
    tuner = {}
    try:
        cell.f_inf()
        tuner["fwd"] = next((c for k, c in cap if k == "fwd_t"), None)
        del cap[:]
        cell.f_trn()
        tuner["fwd_trn"] = next((c for k, c in cap if k == "fwd_t"), None)
        del cap[:]
        cell.fb(); cell.fb()
        tuner["fwd_trn"] = tuner["fwd_trn"] or next((c for k, c in cap if k == "fwd_t"), None)
        tuner["bwd"] = next((c for k, c in cap if k == "mh_bwd"), None)
    finally:
        autotune.choose_axes = orig
    seeds = dict(fwd=project(tuner["fwd"], FK) if tuner["fwd"] else None,
                 fwd_trn=project(tuner["fwd_trn"], FK) if tuner["fwd_trn"] else None,
                 bwd=project(tuner["bwd"], BK, drop=("BLOCK_L",)) if tuner["bwd"] else None)

    # reference digests at the tuner's own configs (pinned)
    ref = {}
    with Pins(fwd=seeds["fwd"]):
        with torch.no_grad():
            ref["fwd"] = autotune._digest(cell.call())
    bwd = cell.graph()
    with Pins(bwd=seeds["bwd"]):
        bwd(); ref["bwd"] = autotune._digest(cell.grads())
    ref_peak = {}
    with Pins(fwd=seeds["fwd"]):
        ref_peak["fwd"] = peak_mb(cell.f_inf)
    with Pins(bwd=seeds["bwd"]):
        ref_peak["bwd"] = peak_mb(bwd)

    out = dict(cell=name, geometry=g, key=key, tuner=seeds, n_eval={}, rejected={},
               ms={}, peak_mb={})

    def fwd_pin(c):
        return Pins(fwd=c)

    def bwd_pin(c):
        return Pins(bwd=c)

    def chk_fwd(c):
        with Pins(fwd=c):
            with torch.no_grad():
                d = autotune._digest(cell.call())
            if not autotune._agrees(ref["fwd"], d):
                return "disagrees with the tuner's output"
            pk = peak_mb(cell.f_inf)
        if pk > MEM_FACTOR * ref_peak["fwd"] + 8:
            return "peak %.0f MB > %.2f x tuner %.0f MB" % (pk, MEM_FACTOR, ref_peak["fwd"])
        return None

    def chk_bwd(c):
        with Pins(bwd=c):
            bwd()
            if not autotune._agrees(ref["bwd"], autotune._digest(cell.grads())):
                return "disagrees with the tuner's gradients"
            pk = peak_mb(bwd)
        if pk > MEM_FACTOR * ref_peak["bwd"] + 8:
            return "peak %.0f MB > %.2f x tuner %.0f MB" % (pk, MEM_FACTOR, ref_peak["bwd"])
        with Pins(fwd=seeds["fwd_trn"], bwd=c):
            if not cell.extreme_ok():
                return "non-finite at the extreme decay rate"
        return None

    # the forward's time split is served from this length on (`MH_FWD_TS_MIN_L`)
    ts_ok = g["L"] >= int(os.environ.get("MH_FWD_TS_MIN_L", 4096))

    chosen = {}
    for sname, run, infer in (("fwd", cell.f_inf, True), ("fwd_trn", cell.f_trn, False)):
        base = seeds[sname] or seeds["fwd"] or {}
        side = Side(sname, run, fwd_pin, [seeds[sname], seeds["fwd"] if sname == "fwd_trn" else None])
        tune_fwd(side, cm, g, ckpt_l if not infer else 0, ts_ok, infer, base)
        c, ms, med, rej = select(side, rounds, chk_fwd, ref_peak["fwd"])
        out["n_eval"][sname] = side.n
        out["rejected"][sname] = rej
        chosen[sname] = c or seeds[sname]
        out["ms"][sname + "_search"] = dict(chosen=ms, finalists=med)
    side = Side("bwd", bwd, bwd_pin, [seeds["bwd"]])
    tune_bwd(side, cm, g, nblk, seeds["bwd"] or {})
    c, ms, med, rej = select(side, rounds, chk_bwd, ref_peak["bwd"])
    out["n_eval"]["bwd"] = side.n
    out["rejected"]["bwd"] = rej
    chosen["bwd"] = c or seeds["bwd"]
    out["ms"]["bwd_search"] = dict(chosen=ms, finalists=med)
    del bwd

    # 4. VERIFY: production path on the table vs production path on the tuner
    def verify(sel, nround):
        cm._P1_INJECT.clear()
        cm._P1_INJECT[key] = {s: cfg_key(c) for s, c in sel.items() if c is not None}
        T = {m: [] for m in ("fwd", "fwd_trn", "fb")}
        C = {m: [] for m in T}
        hits = []
        for r in range(nround):
            for arm in ((0, 1) if r % 2 == 0 else (1, 0)):
                os.environ["HYBRID_MH_CALIBRATION"] = "on" if arm else "off"
                h0 = cm.P1_STATS["hit"]
                for m, fn in (("fwd", cell.f_inf), ("fwd_trn", cell.f_trn), ("fb", cell.fb)):
                    (C if arm else T)[m].append(events(fn, 25.0))
                hits.append((arm, cm.P1_STATS["hit"] - h0))
        os.environ["HYBRID_MH_CALIBRATION"] = "off"
        cm._P1_INJECT.clear()
        used = all(h > 0 for a, h in hits if a) and all(h == 0 for a, h in hits if not a)
        t = {m: statistics.median(v) for m, v in T.items()}
        c = {m: statistics.median(v) for m, v in C.items()}
        return t, c, {m: c[m] / t[m] for m in t}, used

    cm._P1_INJECT.clear()
    sel = dict(chosen)
    t, c, ratio, used = verify(sel, max(9, rounds))
    attempt = "search"
    if not (used and ratio["fwd"] <= ACCEPT and ratio["fb"] <= ACCEPT):
        # fall back to the tuner's own winners, then judge again
        sel = dict(fwd=seeds["fwd"], fwd_trn=seeds["fwd_trn"], bwd=seeds["bwd"])
        t, c, ratio, used = verify(sel, max(9, rounds))
        attempt = "tuner-winners"
    out.update(sides={s: cfg_key(v) for s, v in sel.items() if v is not None},
               ms_tuner=t, ms_cal=c, ratio=ratio, table_used=used, attempt=attempt,
               accepted=bool(used and ratio["fwd"] <= ACCEPT and ratio["fb"] <= ACCEPT),
               seconds=round(time.time() - t_cell, 1))
    for s in ("fwd", "bwd"):
        out["peak_mb"][s] = dict(tuner=ref_peak[s])
    # free the cell
    del cell
    autotune._CACHE.clear()
    try:
        bwd_gluon._PLANS.clear()
    except Exception:
        pass
    torch.cuda.empty_cache()
    return out


def run(a):
    import torch
    from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh as cm
    os.makedirs(a.run, exist_ok=True)
    path = os.path.join(a.run, "cells.jsonl")
    done = set()
    if os.path.exists(path):
        for ln in open(path):
            try:
                done.add(json.loads(ln)["cell"])
            except Exception:
                pass
    names = order(list(cm.P1_SHAPES), cm.P1_SHAPES)
    if a.cells:
        keys = a.cells.split(",")
        names = [n for n in names if any(k in n for k in keys)]
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        names = names[i::n]
    deadline = time.time() + a.deadline_min * 60 if a.deadline_min else None
    todo = 0
    for n in names:
        if n in done:
            continue
        if a.max_cells and todo >= a.max_cells:
            break                    # the host process grows ~0.3 GB per cell: restart it
        todo += 1
        if deadline and time.time() > deadline:
            print("[p1cal] deadline reached", flush=True)
            break
        try:
            r = measure_cell(n, cm.P1_SHAPES[n], a.rounds)
        except torch.cuda.OutOfMemoryError as e:
            r = dict(cell=n, geometry=cm.P1_SHAPES[n], dropped="oom: " + str(e).splitlines()[0][:160])
            torch.cuda.empty_cache()
        except Exception as e:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = dict(cell=n, geometry=cm.P1_SHAPES[n], dropped="error: %r" % (e,))
            torch.cuda.empty_cache()
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print("[p1cal] %-40s %s" % (n, r.get("dropped") or "fwd %.3f  fb %.3f  acc=%s  %ss  evals=%s" % (
            r["ratio"]["fwd"], r["ratio"]["fb"], r["accepted"], r["seconds"], r["n_eval"])),
            flush=True)
    return todo


def derive(a):
    from orbit_ssm.kernels import calibration as cal
    from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh as cm
    rows = {}
    for d in a.derive.split(","):
        for ln in open(os.path.join(d, "cells.jsonl")):
            r = json.loads(ln)
            rows[r["cell"]] = r
    cells, dropped = {}, {}
    for n, r in rows.items():
        if r.get("dropped"):
            dropped[n] = r["dropped"]
        elif not r["accepted"]:
            dropped[n] = "not accepted: ratio %s" % {k: round(v, 3) for k, v in r["ratio"].items()}
        else:
            cells[r["key"]] = dict(r["sides"], cell=n, ms_tuner=r["ms_tuner"], ms_cal=r["ms_cal"],
                                   ratio=r["ratio"])
    if a.gpu:                      # offline: another GPU's raw cells, no device needed
        import torch
        import triton
        cal._DEV.update(gpu=a.gpu, sms=a.sms, driver=a.driver or None,
                        triton=triton.__version__, torch=torch.__version__)
    dev = cal.device_info()
    prov = cal.provenance("mh", a.rounds, sorted(rows), dropped=dropped,
                          fixed_basis="p1 per-cell table (tools/calibrate_mh_p1.py)")
    data = dict(schema=cal.SCHEMA, family="mh", provenance=prov, kernels={}, thresholds={},
                p1_cells=cells)
    path = a.out or (cal.shipped_path("mh", dev["gpu"]) if a.write_shipped else
                     cal.local_path("mh", dev["gpu"]) if a.local else None)
    if not path:
        raise SystemExit("--derive needs --write-shipped, --local or --out FILE")
    print("cells accepted %d, not written %d (%s)" % (len(cells), len(dropped), dev["gpu"]))
    print("wrote", cal.write_json(path, data))
    worst = max((c["ratio"][m] for c in cells.values() for m in ("fwd", "fb")), default=None)
    print("worst accepted ratio (calibrated / tuner): %s" % worst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="")
    ap.add_argument("--derive", default="")
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--local", action="store_true", help="--derive: write the LOCAL file of this GPU (read before the shipped one)")
    ap.add_argument("--out", default="")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--gpu", default="", help="--derive offline: GPU name of the cells")
    ap.add_argument("--sms", type=int, default=0)
    ap.add_argument("--driver", default="")
    ap.add_argument("--part", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--deadline-min", type=float, default=0)
    ap.add_argument("--max-cells", type=int, default=0,
                    help="measure at most this many cells, then exit 0 (a wrapper restarts the process)")
    a = ap.parse_args()
    if a.run:
        sys.exit(0 if run(a) else 3)       # 3: nothing left to measure
    elif a.derive:
        derive(a)
    else:
        ap.error("--run or --derive")


if __name__ == "__main__":
    main()
