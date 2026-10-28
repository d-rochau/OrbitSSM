"""ONE launch-calibration framework for the production kernels.

A kernel FAMILY (e.g. ``ssd_mimo``; an MH adapter plugs in the same way)
registers its kernels, the source files whose edits can move a launch
optimum, and its representative shapes.  ``tools/calibrate_kernels.py``
measures every config of the FULL grid per shape (one process per shape,
median of >= 5 rounds) and ``derive`` turns the timings into, per kernel,

  * a FIXED config -- minimises the worst-case slowdown over the shapes
    (warps / stages / maxnreg), and
  * a LIGHT set of 2-4 configs covering every shape within ``tol`` of its
    best.  Stages are measured and then FIXED: the light set only varies
    warps / maxnreg at the fixed config's stage count.

Two layers, one JSON file per (family, GPU name):

  SHIPPED  ``src/orbit_ssm/kernels/calibration_data/<family>/<gpu>.json`` (repo)
  LOCAL    ``$SSD_CALIBRATION_DIR`` or ``$XDG_CACHE_HOME/orbit_ssm/
           calibration/<family>/<gpu>.json`` -- wins when valid

A file is used only when its provenance matches: GPU name, Triton version
and the code hash of the family's kernel files.  Stale files are ignored
automatically (the reason is kept for the diagnostic).  Nothing here selects
by compute capability, and no kernel arithmetic depends on this module.

Runtime (``calibrated_grid``), only when the family's env switch is on
(default OFF -- shipped defaults are unchanged):
  pinned -> the FIXED config            (no search)
  light  -> the calibrated LIGHT set    (search 2-4 configs)
  full   -> the family's full grid      (what calibration itself runs)
  no valid calibration for this GPU -> the family's generic small light grid
  and ONE stderr message telling the user to run the calibration tool.
Every grid records its source (shipped / local / light-... / full / default)
for the family's ``describe_served``.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SCHEMA = 1
_FAMILIES: dict = {}
_SOURCE: dict = {}        # (family, kernel) -> source string of the built grid
_STALE: dict = {}         # (family, gpu) -> [reasons files were ignored]
_WARNED: set = set()


# --------------------------------------------------------------- families --
def register_family(name, *, env, code_files, kernels, shapes=None,
                    group_key=None, axes=("num_warps", "num_stages", "maxnreg"),
                    cfg_key=None, cfg_parse=None, cfg_class=None,
                    make_config=None, config_key=None, light_size=8,
                    occupancy=None, chunk_of=None, notes=None):
    """Adapter registration.

    `code_files`: absolute paths hashed into the provenance.
    `shapes`: {shape_name: geometry dict} -- the adapter's representative set.
    `group_key(**geometry) -> str`: OPTIONAL shape grouping, an exception
      only.  Default (None): ONE fixed config per kernel per GPU,
      chosen by best worst-case slowdown over all shapes.  With a group_key,
      configs are additionally stored per group (the `default` entry stays the
      all-shape fit and serves unseen groups).
    `axes`: the adapter's OWN tuning axes -- the keys of a stored config
      dict (Triton kernels: warps/stages/maxnreg; MH: block sizes, ...).
    `light_size`: CAP on the LIGHT set size (default 8); the size k is chosen
      per kernel x group x GPU by `derive` (smallest k reaching <= 1.02).
    `occupancy(**geometry)->float`: the shape's thread-block count proxy
      (SSD: B*H); with it, a kernel+group whose ONE fixed config is >5% off
      on some shape gets the OCCUPANCY RULE: two configs (low/high) split at
      a calibrated per-GPU threshold of blocks per SM (occupancy/SMs), chosen
      per call (`activate(..., occupancy=)`), see `derive_grouped`.
    `chunk_of(group)->int`: the chunk length T the family's shapes are
      measured at, per group.  It is stored in the file (`chunk`) and the
      calibrated configs are replayed ONLY for a call at that T (`activate(...,
      T=)`): a config tuned at T=32 says nothing about T=64, so any other T searches the family's GENERIC
      light grid instead.
    Config codec (defaults = the Triton warps/stages/maxnreg codec):
      `cfg_key(dict)->str` / `cfg_parse(str)->dict` name a config in the
      timing tables; `cfg_class(dict)->hashable` is the part of a config the
      light set must HOLD at the fixed config's value (SSD: num_stages --
      measured, then fixed); `make_config(dict)->object` builds the grid
      entry; `config_key(object)->str` names a grid entry (for pruning)."""
    _FAMILIES[name] = dict(name=name, env=env, code_files=list(code_files),
                           kernels=tuple(kernels), shapes=shapes or {},
                           group_key=group_key,
                           axes=tuple(axes),
                           cfg_key=cfg_key or cfg_dict_key,
                           cfg_parse=cfg_parse or globals()["cfg_parse"],
                           cfg_class=cfg_class or (lambda c: c["num_stages"]),
                           make_config=make_config or _triton_config,
                           config_key=config_key or _triton_config_key,
                           light_size=int(light_size), occupancy=occupancy,
                           chunk_of=chunk_of, notes=notes)
    return _FAMILIES[name]


def family(name):
    return _FAMILIES[name]


#: Mode when the family's env switch is unset.  "off" = today's defaults.
#: The calibrated LIGHT mode is the default; this line is the switch.
DEFAULT_MODE = "on"


def _const(fn):
    """Mark a pure-host resolver (env / file lookup returning a constant) so that
    `torch.compile` evaluates it at trace time instead of tracing its body: tracing
    `time.monotonic` / dict stores / file stats made graph breaks (and, for a global-dict
    store, a failing guard).  Eager behaviour is unchanged; under compile the answer is
    fixed at trace time (a changed env var needs a `torch._dynamo.reset()`)."""
    try:
        import torch
        return torch.compiler.assume_constant_result(fn)
    except Exception:                                       # pragma: no cover
        return fn


def _cuda_ok() -> bool:
    """Is there a device to calibrate against?  Goes through `device_info`
    (cached), so tests that mock the device count as one."""
    try:
        device_info()
        return True
    except Exception:            # no CUDA device (CPU-only node / collection)
        return False


@_const
def mode(fam: str) -> str:
    """"off" | "on" | "pinned" (`SSD_MIMO_CALIBRATION`).  "on" = the calibrated
    LIGHT tier (search the calibrated light set); "pinned" = the shipped
    per-GPU FIXED config of the call's group, scoped by the recorded chunk
    length, NO SEARCH ANYWHERE (an uncalibrated T or an unknown GPU gets the
    family's single generic pin): the legal per-arch pin for gates, which then
    test exactly the config production uses at that T."""
    raw = os.environ.get(_FAMILIES[fam]["env"])
    if not _cuda_ok():
        # no CUDA device (CPU-only test collection, CPU nodes): nothing to
        # calibrate against -- behave as "off" instead of
        # failing in `device_info()` at import time.
        return "off"
    v = (raw if raw is not None else DEFAULT_MODE).strip().lower()
    if v == "pinned":
        return "pinned"
    return "on" if v in ("on", "1", "true", "calibrated") else "off"


@_const
def enabled(fam: str) -> bool:
    return mode(fam) != "off"


_KERNEL_DECOS = {"jit", "autotune", "heuristics", "constexpr_function",
                 "gluon_jit"}
_CONFIG_NAME = re.compile(r"(configs|_grid|prune|_cfg)(_|$)|^_?cfg_")


def _deco_name(d):
    import ast
    if isinstance(d, ast.Call):
        d = d.func
    if isinstance(d, ast.Attribute):
        return d.attr
    return d.id if isinstance(d, ast.Name) else ""


def kernel_source(path: str) -> str:
    """What of a source file can change a kernel's BEHAVIOUR or its LAUNCH
    CONFIG space, normalised (`ast.unparse`).  What does not count: comments,
    whitespace, docstrings, assert messages (the message of any
    `*.static_assert` / `*.device_assert` call is replaced by '' -- its
    condition is kept).

      * every top-level function decorated `@triton.jit`, `@gluon.jit`,
        `@triton.autotune(...)` (its config grid + tuning key are part of
        the decorator), `@triton.heuristics`, `@gluon.constexpr_function`
        -- the kernel bodies and their constexpr-affecting helpers;
      * the launch-config code: top-level functions named like
        `*configs*`, `*_grid`, `*prune*`, `*_cfg*`;
      * module-level ALL_CAPS constants (tuning keys, reset constants...).

    Host-side functions (dispatch, wrappers, diagnostics) are NOT part of
    it: editing them keeps the calibration valid."""
    import ast
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for n in ast.walk(tree):
        b = getattr(n, "body", None)
        if (isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and b
                and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)):
            n.body = b[1:] or [ast.Pass()]
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("static_assert", "device_assert")):
            if len(n.args) >= 2:
                n.args[1] = ast.Constant("")
            for kw in n.keywords:
                if kw.arg == "msg":
                    kw.value = ast.Constant("")
    parts = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (any(_deco_name(d) in _KERNEL_DECOS for d in n.decorator_list)
                    or _CONFIG_NAME.search(n.name)):
                parts.append(ast.unparse(n))
        elif isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if names and all(re.fullmatch(r"_*[A-Z][A-Z0-9_]*", x) for x in names):
                parts.append(ast.unparse(n))
    return "\n".join(parts)


def code_hash(fam: str) -> str:
    h = hashlib.sha256()
    for p in sorted(_FAMILIES[fam]["code_files"]):
        h.update(os.path.basename(p).encode())
        h.update(kernel_source(p).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- device --
_DEV = {}


def device_info() -> dict:
    """Device-reported identity (no capability literals)."""
    if not _DEV:
        import torch
        import triton
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        drv = None
        try:
            drv = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader", "-i", "0"], capture_output=True,
                text=True, timeout=20).stdout.strip().splitlines()[0]
        except Exception:
            pass
        _DEV.update(gpu=p.name, sms=p.multi_processor_count, driver=drv,
                    triton=triton.__version__, torch=torch.__version__)
    return dict(_DEV)


def slug(gpu: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")


# ------------------------------------------------------------------ paths --
def shipped_dir(fam: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calibration_data", fam)


def local_dir(fam: str) -> str:
    d = os.environ.get("SSD_CALIBRATION_DIR")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        d = os.path.join(base, "orbit_ssm", "calibration")
    return os.path.join(d, fam)


def shipped_path(fam, gpu):
    return os.path.join(shipped_dir(fam), slug(gpu) + ".json")


def local_path(fam, gpu):
    return os.path.join(local_dir(fam), slug(gpu) + ".json")


# ------------------------------------------------------------ cfg encoding --
def cfg_key(w, s, r) -> str:
    """Triton launch config key (the SSD adapter's axes)."""
    return "w%ds%dr%s" % (int(w), int(s), "N" if r is None else int(r))


def cfg_parse(key: str) -> dict:
    m = re.fullmatch(r"w(\d+)s(\d+)r(N|\d+)", key)
    if not m:
        raise ValueError("bad config key %r" % key)
    return dict(num_warps=int(m[1]), num_stages=int(m[2]),
                maxnreg=None if m[3] == "N" else int(m[3]))


def cfg_dict_key(c: dict) -> str:
    return cfg_key(c["num_warps"], c["num_stages"], c["maxnreg"])


def _triton_config(c):
    import triton
    return triton.Config({}, num_stages=c["num_stages"],
                         num_warps=c["num_warps"], maxnreg=c["maxnreg"])


def _triton_config_key(cfg):
    return cfg_key(cfg.num_warps, cfg.num_stages, cfg.maxnreg)


# ----------------------------------------------------------------- derive --
def derive(timings: dict, tol: float = 0.02, max_light: int = 8,
           cls=None, exhaustive_limit: int = 5_000_000,
           target: float = 1.02, prefer: float = 1.01) -> dict:
    """timings: {shape: {cfg_key: ms}} for ONE kernel.

    A config missing for a shape (pruned / not lowerable there) counts as
    infinitely slow on it.  Returns the FIXED config (min worst-case
    slowdown over the shapes), the LIGHT set and diagnostics.

    LIGHT set (a shape is served by its best member; by default the configs
    of the fixed config's class -- SSD: its stage count -- unless HOLDING
    that axis costs > 2% worst-case vs the unconstrained optimum, then the
    axis varies: `light_stages_varied`, recorded with the cost): its SIZE k is chosen PER kernel x
    group x GPU, the smallest k <= `max_light` (the family cap, <= 8) whose
    exhaustively optimal set reaches worst-case slowdown <= `target` (1.02);
    when `prefer` (1.01) is reached with the same k or one more config, that
    k is taken.  If even `max_light` misses `target` the result is flagged
    (`light_flag`) -- likely a tuning-key problem, not a set-size one.  The
    optimal set is found EXHAUSTIVELY (numpy-chunked) when the subset count
    is <= `exhaustive_limit`, else by greedy (fixed first) + pairwise-swap
    local improvement; among equal sets the smaller / lower-mean one wins.
    `light_curve[r-1]` = best worst-case slowdown with r configs."""
    import itertools
    import math
    import numpy as np
    shapes = sorted(timings)
    best = {s: min(timings[s].values()) for s in shapes}
    cfgs = sorted({k for s in shapes for k in timings[s]})
    inf = float("inf")

    def sd(k, s):
        t = timings[s].get(k)
        return inf if t is None else t / best[s]

    def score(k):
        v = [sd(k, s) for s in shapes]
        return (max(v), sum(v) / len(v))
    fixed = min(cfgs, key=lambda k: (score(k), k))
    if score(fixed)[0] == inf:      # nothing runs everywhere: most coverage
        fixed = min(cfgs, key=lambda k: (-sum(sd(k, s) < inf for s in shapes),
                                         score(k)[1], k))
    cls = cls or (lambda k: cfg_parse(k)["num_stages"])
    held = [k for k in cfgs if cls(k) == cls(fixed)]

    def light_sets(pool):
        cap = max(1, min(max_light, len(pool)))
        A = np.array([[sd(c, s) for c in pool] for s in shapes])  # shapes x cfgs
        n_subsets = sum(math.comb(len(pool), r) for r in range(1, cap + 1))
        sets, curve = {}, []
        if n_subsets <= exhaustive_limit:
            for r in range(1, cap + 1):
                bw, bm, bsel = inf, inf, None
                it = itertools.combinations(range(len(pool)), r)
                while True:
                    chunk = list(itertools.islice(it, 200_000))
                    if not chunk:
                        break
                    idx = np.array(chunk)                        # N x r
                    m = A[:, idx].min(axis=2)                    # shapes x N
                    w, mu = m.max(axis=0), m.sum(axis=0)
                    j = np.lexsort((mu, w))[0]
                    if bsel is None or w[j] < bw - 1e-12 or (
                            abs(w[j] - bw) <= 1e-12 and mu[j] < bm - 1e-12):
                        bw, bm, bsel = float(w[j]), float(mu[j]), chunk[j]
                sets[r] = [pool[i] for i in bsel]
                curve.append(bw)
        else:                                   # greedy + pairwise swaps
            def worst(sel):
                return float(A[:, [pool.index(c) for c in sel]].min(axis=1).max())
            cur = [fixed] if fixed in pool else [pool[0]]
            for r in range(1, cap + 1):
                if r > 1:
                    rest = [k for k in pool if k not in cur]
                    if not rest:
                        sets[r] = list(cur)
                        curve.append(curve[-1])
                        continue
                    cur = cur + [min(rest, key=lambda k: (worst(cur + [k]), k))]
                improved = True
                while improved:
                    improved = False
                    for i in range(len(cur)):
                        for c in pool:
                            if c in cur:
                                continue
                            cand = cur[:i] + [c] + cur[i + 1:]
                            if worst(cand) < worst(cur) - 1e-12:
                                cur, improved = cand, True
                sets[r] = list(cur)
                curve.append(worst(cur))
        return cap, sets, curve

    # Stages (the held axis) are ALWAYS measured; whether the light set may
    # vary them is decided here: hold them at the fixed config's value
    # unless holding costs > 2% worst-case vs the unconstrained optimum.
    cap, sets, curve = light_sets(held)
    varied, stage_cost = False, 1.0
    if len(held) < len(cfgs):
        capa, setsa, curvea = light_sets(cfgs)
        stage_cost = curve[-1] / curvea[-1] if curvea[-1] > 0 else 1.0
        if curve[-1] > curvea[-1] * 1.02 + 1e-12:
            varied, cap, sets, curve = True, capa, setsa, curvea
    k20 = next((r for r in range(1, cap + 1) if curve[r - 1] <= target + 1e-9), None)
    k10 = next((r for r in range(1, cap + 1) if curve[r - 1] <= prefer + 1e-9), None)
    if k20 is None:
        k, flag = cap, True
    else:
        k, flag = (k10 if (k10 is not None and k10 <= k20 + 1) else k20), False
    light = list(sets[k])
    if fixed in light:                          # fixed config first
        light = [fixed] + [c for c in light if c != fixed]
    cover = {s: min(sd(c, s) for c in light) for s in shapes}
    unc = sorted(s for s in shapes if cover[s] > 1 + tol)
    return dict(fixed=fixed, light=light, light_k=k, light_flag=flag,
                light_stages_varied=varied, light_stage_cost=float(stage_cost),
                light_curve=[float(v) for v in curve],
                worst_slowdown=score(fixed)[0],
                fixed_slowdown={s: sd(fixed, s) for s in shapes},
                light_worst_slowdown=max(cover.values()), light_cover=cover,
                uncovered=unc)


def verify_decision(shipped_fixed: str, timings: dict, hysteresis=0.03):
    """(override?, shipped worst-case slowdown, fresh best fixed slowdown):
    override only when the shipped fixed config is > hysteresis worse than
    what this machine measures as best (missing config counts as worse)."""
    d = derive(timings)
    shapes = list(timings)
    best = {s: min(timings[s].values()) for s in shapes}
    v = [timings[s].get(shipped_fixed, float("inf")) / best[s] for s in shapes]
    sh = max(v)
    return sh > d["worst_slowdown"] * (1 + hysteresis), sh, d["worst_slowdown"]


# ------------------------------------------------------------------ files --
def provenance(fam, rounds, shapes, dropped=None, fixed_basis=None) -> dict:
    d = device_info()
    return dict(gpu=d["gpu"], sms=d["sms"], driver=d["driver"],
                triton=d["triton"], torch=d["torch"],
                code_hash=code_hash(fam),
                date=datetime.date.today().isoformat(),
                rounds=rounds, shapes=list(shapes),
                dropped_shapes=dict(dropped or {}),
                fixed_basis=fixed_basis or "pass1",
                notes=_FAMILIES[fam].get("notes"))


def _entry(d, shapes, parse=None):
    parse = parse or cfg_parse
    return dict(fixed=parse(d["fixed"]),
                light=[parse(x) for x in d["light"]],
                worst_slowdown=d["worst_slowdown"],
                per_shape_slowdown=d["fixed_slowdown"],
                light_k=d["light_k"], light_flag=d["light_flag"],
                light_stages_varied=d["light_stages_varied"],
                light_stage_cost=d["light_stage_cost"],
                light_curve=d["light_curve"],
                light_worst_slowdown=d["light_worst_slowdown"],
                uncovered_shapes=d["uncovered"], shapes=sorted(shapes))


def _occ_rule(timings, members, occ, fm, kw, parse):
    """Best two-config split of `members` at a blocks-per-SM threshold, or
    None.  `occ`: {shape: blocks per SM}.  Thresholds = midpoints between
    distinct occupancies; a part's configs come from `derive` on its shapes."""
    vals = sorted({round(occ[m], 9) for m in members})
    if len(vals) < 2:
        return None
    best = None
    for lo, hi in zip(vals, vals[1:]):
        thr = (lo + hi) / 2
        low = [m for m in members if occ[m] < thr]
        high = [m for m in members if occ[m] >= thr]
        dl = derive({m: timings[m] for m in low}, **kw)
        dh = derive({m: timings[m] for m in high}, **kw)
        # worst-case slowdown vs the GROUP's per-shape best (all shapes)
        w = max(dl["worst_slowdown"], dh["worst_slowdown"])
        if best is None or w < best[0] - 1e-12:
            best = (w, thr, low, high, dl, dh)
    w, thr, low, high, dl, dh = best
    return dict(proxy="blocks per SM = occupancy(geometry) / SM count",
                threshold=thr, worst_slowdown=w,
                low=_entry(dl, low, parse), high=_entry(dh, high, parse))


def derive_grouped(timings: dict, shape_groups: dict, fam=None, sms=None,
                   **kw) -> dict:
    """{'default': entry over all shapes, 'groups': {gkey: entry}} -- a
    group's fixed/light configs are fitted over ITS shapes only.  A group
    whose ONE fixed config is > 5% off on some shape gets, when the family
    declares `occupancy` and `sms` is known, the OCCUPANCY RULE in
    `entry['occupancy']` if it lowers the group's worst-case slowdown."""
    fm = _FAMILIES[fam] if fam else None
    if fm:
        kw.setdefault("cls", lambda k: fm["cfg_class"](fm["cfg_parse"](k)))
        kw.setdefault("max_light", fm["light_size"])
    parse = fm["cfg_parse"] if fm else None
    out = dict(default=_entry(derive(timings, **kw), timings, parse),
               groups={})
    by = {}
    for sh in timings:
        by.setdefault(shape_groups[sh], []).append(sh)
    for g, members in sorted(by.items()):
        out["groups"][g] = _entry(derive({m: timings[m] for m in members},
                                         **kw), members, parse)
        e = out["groups"][g]
        if (fm and fm.get("occupancy") and sms and len(members) >= 2
                and e["worst_slowdown"] > 1.05):
            occ = {m: fm["occupancy"](**fm["shapes"][m]) / sms for m in members}
            r = _occ_rule(timings, members, occ, fm, kw, parse)
            if r and r["worst_slowdown"] < e["worst_slowdown"] - 1e-9:
                e["occupancy"] = r
    return out


def build_file(fam, per_kernel_timings: dict, rounds, shapes,
               thresholds=None, **kw) -> dict:
    fm = _FAMILIES[fam]
    gk = fm["group_key"]
    shape_groups = {s: (gk(**fm["shapes"][s]) if gk else "all") for s in shapes}
    ks = {}
    for k, t in per_kernel_timings.items():
        if not t:
            continue
        e = derive_grouped(t, shape_groups, fam=fam, sms=device_info()["sms"])
        if gk is None:                 # one fixed config per kernel per GPU
            e["groups"] = {}
        ks[k] = dict(e, timings_ms=t,
                     shape_groups={s: shape_groups[s] for s in t})
    out = dict(schema=SCHEMA, family=fam,
               provenance=provenance(fam, rounds, shapes, **kw), kernels=ks,
               thresholds=dict(thresholds or {}))
    if per_kernel_timings:
        # registered kernels that NO calibration cell launched (the workloads go through the production
        # entry point, so these are kernels production does not run on this GPU / at these shapes):
        # recorded explicitly so a missing entry is a statement, not a silent generic-grid fallback
        out["unlaunched_kernels"] = sorted(set(fm["kernels"]) - set(ks))
    if fm.get("chunk_of"):
        groups = sorted(set(shape_groups.values()))
        out["chunk"] = {g: fm["chunk_of"](g) for g in groups}
    return out


def restamp(data: dict, fam: str, from_hash: str, note: str) -> dict:
    """`data` (a file of family `fam` whose provenance carries the OLDER kernel code hash `from_hash`) re-stamped with the
    CURRENT code hash, every entry unchanged.  Only for a source change that is bit-identical for every configuration the file
    holds (e.g. a constexpr branch none of its groups compiles), asserted by the maintainer in `note` (with its evidence) and
    recorded in `provenance.restamped`; any other change needs a recalibration."""
    import copy
    out = copy.deepcopy(data)
    p = out["provenance"]
    if p.get("code_hash") != from_hash:
        raise ValueError("restamp: file hash %s != the asserted old hash %s" % (p.get("code_hash"), from_hash))
    new = code_hash(fam)
    p.setdefault("restamped", []).append(dict(from_hash=from_hash, to_hash=new, date=datetime.date.today().isoformat(),
                                              note=note))
    p["code_hash"] = new
    return out


def merge_groups(base: dict, fresh: dict, groups, *, restamp_from=None) -> dict:
    """`base` (a shipped / local file of family F) with the shape GROUPS `groups` taken from `fresh` (a file built by
    `build_file` from a measurement of those groups' cells only, same GPU / Triton / code hash): per kernel the group entries,
    their timings and shape-group labels, and each group's recorded chunk.  Everything else of `base` -- its `default` entries,
    its other groups, thresholds -- is kept unchanged, so a new axis of groups (e.g. the fp32 groups) ships without re-deriving
    the 16-bit groups.  A kernel `base` has no entry for gets one whose `default` is the fresh (merged groups only) fit and is
    listed in `provenance.merged[-1].new_kernels`.

    `restamp_from`: `base` may carry an OLDER kernel code hash only when it equals this value -- the caller asserts that the
    kernel-source change since then is bit-identical for every group `base` holds (e.g. a constexpr branch only the merged
    groups' dtype compiles); the result carries the current hash and the assertion is recorded (`provenance.restamped`)."""
    import copy
    out = copy.deepcopy(base)
    bp, fp = out["provenance"], fresh["provenance"]
    for k in ("gpu", "triton"):
        if bp.get(k) != fp.get(k):
            raise ValueError("merge: %s %r (base) != %r (fresh)" % (k, bp.get(k), fp.get(k)))
    if bp.get("code_hash") != fp.get("code_hash"):
        if restamp_from is None or bp.get("code_hash") != restamp_from:
            raise ValueError("merge: base code hash %s != fresh %s (pass restamp_from=%s to assert the change is "
                             "bit-identical for the base groups)" % (bp.get("code_hash"), fp.get("code_hash"),
                                                                   bp.get("code_hash")))
        bp.setdefault("restamped", []).append(dict(from_hash=bp["code_hash"], to_hash=fp["code_hash"],
                                                   date=fp.get("date"), groups_kept=sorted(
                                                       {g for e in out["kernels"].values() for g in e["groups"]}),
                                                   note="kernel source changed only in code the kept groups do not "
                                                        "compile (asserted by the merging maintainer)"))
        bp["code_hash"] = fp["code_hash"]
    groups = list(groups)
    new_kernels, merged_kernels = [], {}
    for k, fe in fresh["kernels"].items():
        have = [g for g in groups if g in fe["groups"]]
        if not have:
            continue
        mine = {s for s, g in fe.get("shape_groups", {}).items() if g in have}
        if k not in out["kernels"]:
            out["kernels"][k] = dict(default=fe["default"], groups={}, timings_ms={}, shape_groups={})
            new_kernels.append(k)
        e = out["kernels"][k]
        for g in have:
            e["groups"][g] = fe["groups"][g]
        e.setdefault("timings_ms", {}).update({s: t for s, t in fe.get("timings_ms", {}).items() if s in mine})
        e.setdefault("shape_groups", {}).update({s: fe["shape_groups"][s] for s in mine})
        merged_kernels[k] = have
    if "unlaunched_kernels" in out:
        out["unlaunched_kernels"] = sorted(set(out["unlaunched_kernels"]) - set(out["kernels"]))
    if fresh.get("chunk"):
        out.setdefault("chunk", {})
        for g in groups:
            if g in fresh["chunk"]:
                out["chunk"][g] = fresh["chunk"][g]
    bp.setdefault("merged", []).append(dict(groups=groups, kernels=merged_kernels, new_kernels=new_kernels,
                                            date=fp.get("date"), rounds=fp.get("rounds"),
                                            shapes=fp.get("shapes"), dropped_shapes=fp.get("dropped_shapes"),
                                            fixed_basis=fp.get("fixed_basis"), notes=fp.get("notes")))
    return out


def write_json(path, data):
    _LOAD_CACHE.clear()
    _FAST.clear()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)
    return path


def check_file(data, fam, gpu, triton_version=None):
    """None when usable for (fam, gpu) here, else the reason it is stale."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return "schema"
    p = data.get("provenance", {})
    if p.get("gpu") != gpu:
        return "gpu %r != %r" % (p.get("gpu"), gpu)
    if triton_version is not None and p.get("triton") != triton_version:
        return "triton %s != %s" % (p.get("triton"), triton_version)
    if p.get("code_hash") != code_hash(fam):
        return "kernel code hash %s != %s" % (p.get("code_hash"),
                                              code_hash(fam))
    return None


def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


_LOAD_CACHE: dict = {}
_FAST: dict = {}
#: A cached file signature is re-`stat`ed at most this often (seconds).  On a
#: network file system two `stat`s per `load` cost 0.2-0.4 ms of HOST time per
#: scan call; a file written by THIS process
#: (`write_json`, `reset_local`) is seen at once, another process's write
#: within this window.
REVALIDATE_S = 2.0


def load(fam, gpu=None, triton_version=None):
    """(data, 'local'|'shipped') of the valid file with precedence local >
    shipped, or (None, None).  Stale files are skipped and noted.  The result
    is cached per (family, GPU, Triton) and revalidated by the two files'
    mtimes (two `stat`s per call, not two JSON reads: this runs on every
    scan call's host path)."""
    if gpu is None or triton_version is None:
        d = device_info()
        gpu, triton_version = gpu or d["gpu"], triton_version or d["triton"]
    # fast path: no path building, no stat (the scan-call host path)
    fkey = (fam, gpu, triton_version, os.environ.get("SSD_CALIBRATION_DIR"),
            shipped_dir)
    now = time.monotonic()
    fh = _FAST.get(fkey)
    if fh is not None and now - fh[1] < REVALIDATE_S:
        return fh[0]
    paths = (local_path(fam, gpu), shipped_path(fam, gpu))
    key = (fam, gpu, triton_version, paths)
    hit = _LOAD_CACHE.get(key)

    def mt(p):
        try:
            st = os.stat(p)
            return (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            return None
    sig = (mt(paths[0]), mt(paths[1]))
    if hit is not None and hit[0] == sig:
        _FAST[fkey] = (hit[1], now)
        return hit[1]
    res = _load_uncached(fam, gpu, triton_version, paths)
    _LOAD_CACHE[key] = (sig, res, now)
    _FAST[fkey] = (res, now)
    return res


def _load_uncached(fam, gpu, triton_version, paths):
    _STALE[(fam, gpu)] = []
    for layer, path in (("local", paths[0]), ("shipped", paths[1])):
        data = _read(path)
        if data is None:
            continue
        why = check_file(data, fam, gpu, triton_version)
        if why is None:
            return data, layer
        _STALE[(fam, gpu)].append("%s file %s ignored: %s" % (layer, path, why))
    return None, None


def reset_local(fam, gpu=None) -> bool:
    _LOAD_CACHE.clear()
    _FAST.clear()
    gpu = gpu or device_info()["gpu"]
    p = local_path(fam, gpu)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def lookup(fam, kernel, gpu=None, triton_version=None):
    """The kernel's calibrated entry {'default','groups',...,'source'} or
    None."""
    data, layer = load(fam, gpu, triton_version)
    if data is None or kernel not in data["kernels"]:
        return None
    return dict(data["kernels"][kernel], source=layer)


@_const
def threshold(fam, name, default=None, gpu=None, triton_version=None):
    """(value, source) of a calibrated STRUCTURAL threshold (route / chunk
    crossover, e.g. a chunk-parallel waves crossover): `thresholds[name]` =
    {'value': ..., 'evidence': ...} in the same shipped/local files, with
    the same precedence and stale-ignore.  `default` + 'default' when the
    family's switch is off or nothing valid is stored."""
    if not enabled(fam):
        return default, "default"
    data, layer = load(fam, gpu, triton_version)
    t = (data or {}).get("thresholds", {}).get(name)
    return (t["value"], layer) if t else (default, "default")


def occ_part(entry_group, occ):
    """'low' / 'high' of a group's occupancy rule for `occ` blocks per SM;
    None when the group has no rule or the call gave no occupancy."""
    r = entry_group.get("occupancy") if entry_group else None
    if r is None or occ is None:
        return None
    return "low" if occ < r["threshold"] else "high"


def pick(entry, group=None, tier="pinned", occ=None):
    """(configs, group_used) of `entry` for a shape group: that group's
    fixed ('pinned') / light set (of the occupancy part when the group has
    the rule and `occ` is given), else the DEFAULT group's."""
    g = entry["groups"].get(group) if group is not None else None
    used = group if g is not None else "default"
    g = g or entry["default"]
    part = occ_part(g, occ)
    if part:
        g = g["occupancy"][part]
    return ([g["fixed"]] if tier == "pinned" else list(g["light"])), used


_TIER: dict = {}          # (family, kernel) -> tier the grid was built at
_IDS: dict = {}           # (family, kernel) -> ['default', gkey, ...] by id


def _union(fam, entry, tier):
    seen = {}
    ck = _FAMILIES[fam]["cfg_key"]
    for g in ["default"] + sorted(entry["groups"]):
        ge = entry["default"] if g == "default" else entry["groups"][g]
        cfgs = list(pick(entry, None if g == "default" else g, tier)[0])
        for part in ("low", "high"):          # the occupancy rule's two sides
            if "occupancy" in ge:
                cfgs += pick({"default": ge["occupancy"][part], "groups": {}},
                             None, tier)[0]
        for c in cfgs:
            seen.setdefault(ck(c), c)
    return list(seen.values()), ["default"] + sorted(entry["groups"])


def calibrated_grid(fam, kernel, tier, *, full, light, generic_pin):
    """Decorator-grid for `kernel` at `tier`.  `full()`, `light()` and
    `generic_pin()` build the family's own lists (called lazily).  With
    calibration on, the grid is the UNION over shape groups of the group's
    fixed (pinned) / light config sets; `group_id` + `prune` then narrow it
    to the calling shape's group per call."""
    _TIER[(fam, kernel)] = tier
    _IDS.pop((fam, kernel), None)
    _FALLBACK_GRIDS[(fam, kernel)] = (generic_pin, light, full)
    if not enabled(fam):
        _SOURCE[(fam, kernel)] = "default:" + tier
        return {"pinned": generic_pin, "light": light, "full": full}[tier]()
    if mode(fam) == "pinned":
        tier = "pinned"                      # no search, whatever else asked
        _TIER[(fam, kernel)] = tier
    if tier == "full":
        _SOURCE[(fam, kernel)] = "full"
        return full()
    hit = lookup(fam, kernel)
    if hit is None:
        data, _layer = load(fam)
        if data is not None and kernel in data.get("unlaunched_kernels", ()):
            # a valid file that STATES no calibration cell launches this kernel (production does not run
            # it on this GPU / at these shapes): the generic grid is the explicit answer, not a missing file
            _SOURCE[(fam, kernel)] = "unlaunched (no calibration cell launches it)"
            return generic_pin() if mode(fam) == "pinned" else light()
        _SOURCE[(fam, kernel)] = ("pinned-generic (no valid calibration)"
                                  if mode(fam) == "pinned" else
                                  "light-generic (no valid calibration)")
        gpu = device_info()["gpu"]
        if (fam, gpu) not in _WARNED:
            _WARNED.add((fam, gpu))
            sys.stderr.write(
                "[calibration] no valid %s calibration for %s (%s); using the "
                "generic %s. Run: python tools/calibrate_kernels.py"
                " --family %s\n" % (fam, gpu,
                                    "; ".join(_STALE.get((fam, gpu)) or
                                              ["no file"]),
                                    "single pin (no search)" if mode(fam) == "pinned"
                                    else "small light grid", fam))
        return generic_pin() if mode(fam) == "pinned" else light()
    cfgs, ids = _union(fam, hit, tier)
    mk = _FAMILIES[fam]["make_config"]
    # the GENERIC light grid rides along: it is what a call at a chunk length
    # the file was not measured at searches (see `activate(T=)`)
    extra = []
    if _FAMILIES[fam].get("chunk_of"):
        ck = _FAMILIES[fam]["config_key"]
        gen = generic_pin() if mode(fam) == "pinned" else light()
        _GENERIC[(fam, kernel)] = {ck(g) for g in gen}
        have = {ck(mk(c)) for c in cfgs}
        extra = [g for g in gen if ck(g) not in have]
    _IDS[(fam, kernel)] = (ids, hit)
    _SOURCE[(fam, kernel)] = (hit["source"] if tier == "pinned"
                              else "light(%s)" % hit["source"])
    return [mk(c) for c in cfgs] + extra


_GENERIC: dict = {}       # (family, kernel) -> config keys of the generic light grid
_FALLBACK_GRIDS: dict = {}   # (family, kernel) -> (generic_pin, light, full) builders: the smem-retry's candidates
_CTX: dict = {}           # family -> the ACTIVE shape group of this call
_CACHES: dict = {}        # (family, kernel, group) -> that group's tuning cache


WINNER_RULE = 2


def _noise(t):
    """Relative spread of one benchmark: Triton's `do_bench` returns
    [median, q20, q80] of its repetitions."""
    try:
        med, lo, hi = t
        return (hi - lo) / med if med > 0 else float("inf")
    except (TypeError, ValueError):
        return float("inf")


class PersistentWinners(dict):
    """A kernel's tuning cache that persists per-key SEARCH WINNERS next to the
    local calibration file (`<gpu>.winners.json`), scoped by GPU, Triton
    version and the family's code hash: the benchmark of a tuning key runs
    once per machine.  Drop-in for Triton's `Autotuner.cache` (a dict whose
    membership test and item assignment are the only operations used)."""

    def __init__(self, fam, kernel, group, part, autotuner):
        super().__init__()
        self._fam, self._k, self._g, self._p = fam, kernel, group, part
        self._tuner = autotuner

    def _name(self, key):
        return "%s|%s|%s|%s" % (self._k, self._g, self._p, key)

    def _path(self):
        return os.path.join(local_dir(self._fam),
                            slug(device_info()["gpu"]) + ".winners.json")

    def _scope(self):
        d = device_info()
        # winners are persisted only with a margin over the measured noise
        # (`__setitem__`); `WINNER_RULE` versions that rule
        return dict(gpu=d["gpu"], triton=d["triton"],
                    code_hash=code_hash(self._fam), rule=WINNER_RULE)

    def _read(self):
        data = _read(self._path())
        if not data or data.get("scope") != self._scope():
            return {}
        return data.get("winners", {})

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        w = self._read().get(self._name(key))
        if w is None:
            return False
        fm = _FAMILIES[self._fam]
        for c in self._tuner.configs:
            if fm["config_key"](c) == w:
                dict.__setitem__(self, key, c)
                return True
        return False

    def _fixed_key(self):
        """Config key of the calibrated FIXED config of this (group, part),
        or None (uncalibrated group / T, or no valid file)."""
        if self._g.startswith("~"):
            return None
        ent = _IDS.get((self._fam, self._k))
        if ent is None:
            return None
        hit = ent[1]
        g = hit["groups"].get(self._g) or hit["default"]
        if self._p:
            g = g["occupancy"][self._p]
        return _FAMILIES[self._fam]["cfg_key"](g["fixed"])

    def _judge(self, cfg):
        """(config to USE, persist?, evidence).  A search winner is used and
        persisted only when it beats the calibrated FIXED config -- and, with
        no fixed config, the runner-up -- by more than the measured noise of
        the two benchmarks.  Otherwise the fixed config is used and nothing
        is persisted: a winner picked from one noisy pass (GPU contention)
        must not outlive it."""
        fm = _FAMILIES[self._fam]
        tm = getattr(self._tuner, "configs_timings", None) or {}
        ck = fm["config_key"]
        if cfg not in tm:
            return cfg, False, "no timing table"
        t = lambda c: tm[c][0] if isinstance(tm[c], (list, tuple)) else tm[c]
        fk = self._fixed_key()
        fixed = next((c for c in tm if ck(c) == fk), None) if fk else None
        if fixed is not None and fixed is not cfg:
            margin = (t(fixed) - t(cfg)) / t(fixed)
            noise = max(_noise(tm[cfg]), _noise(tm[fixed]))
            if not (margin > noise):
                return fixed, False, "winner %s not faster than the fixed config %s beyond noise (%.3f <= %.3f)" % (ck(cfg), fk, margin, noise)
            return cfg, True, dict(margin=margin, noise=noise, vs="fixed")
        if fixed is cfg:
            return cfg, True, dict(vs="is the fixed config")
        others = sorted(t(c) for c in tm if c is not cfg)
        if not others:
            return cfg, True, dict(vs="single candidate")
        margin = (others[0] - t(cfg)) / t(cfg)
        noise = max(_noise(tm[cfg]), 0.0)
        return (cfg, margin > noise, dict(margin=margin, noise=noise, vs="runner-up"))

    def __setitem__(self, key, cfg):
        use, persist, why = self._judge(cfg)
        dict.__setitem__(self, key, use)
        if not persist:
            return
        try:
            data = _read(self._path())
            if not data or data.get("scope") != self._scope():
                data = dict(scope=self._scope(), winners={})
            data["winners"][self._name(key)] = _FAMILIES[self._fam][
                "config_key"](use)
            data.setdefault("evidence", {})[self._name(key)] = why
            write_json(self._path(), data)
        except OSError:
            pass                                 # persistence is best effort


def _part(fam, kernel, group, occ):
    ent = _IDS.get((fam, kernel))
    if ent is None:
        return None
    hit = ent[1]
    g = hit["groups"].get(group) or hit["default"]
    return occ_part(g, occ)


def _calibrated_at(fam, group, T):
    """False when the valid file records the chunk length of `group` and it
    is not `T` (no recorded chunk, or no valid file: nothing to contradict)."""
    if T is None:
        return True
    data, _ = load(fam)
    ch = (data or {}).get("chunk")
    if not ch:
        return True
    t = ch.get(group, ch.get("default"))
    return t is None or int(t) == int(T)


def activate(fam, group, kernels, occupancy=None, T=None):
    """Per call, BEFORE the family's kernels launch: make `group` the active
    shape group (`occupancy` = the call's blocks per SM).  `prune` narrows
    each calibrated grid to the group's config set (the low/high side of a
    kernel's occupancy rule), and each kernel's autotuning cache is swapped
    for the group's (and side's) own -- the group is not an argument of the
    kernels, so it cannot be part of Triton's key.  No-op unless the
    family's switch is on.  `kernels`: {name: Autotuner}."""
    if not enabled(fam):
        return
    if _SMEM_DEMOTED and (fam, T) in _SMEM_DEMOTED:
        group, occupancy = "~smem:%s" % T, None       # a calibrated config did not fit here: generic grid, own caches
    elif not _calibrated_at(fam, group, T):
        group, occupancy = "~uncal:%s" % T, None      # generic light grid, own caches
    _CTX[fam] = (group, occupancy)
    _ACTIVE_KERNELS[fam] = kernels
    for name, k in kernels.items():
        part = _part(fam, name, group, occupancy)
        ck = (fam, name, group, part)
        if ck not in _CACHES:
            _CACHES[ck] = (PersistentWinners(fam, name, group, part, k)
                           if len(k.configs) > 1 else {})
        k.cache = _CACHES[ck]


# ---------------------------------------------------------------- smem-safe replay --
#: (family, chunk T) pairs DEMOTED to the family's generic grids after a calibrated config raised `OutOfResources`.
#: A calibrated fixed / light config replayed on a geometry it was not measured on may need more shared memory than the device
#: has; Triton raises at launch (a one-config grid is never benchmarked, so nothing prunes it).  The dispatcher's existing
#: backward repair loop (`autograd`) catches that, calls `smem_demote` and retries the SAME chunk before it
#: halves it: `activate` then selects the generic grid (a search that benchmarks and skips configs that do not fit, or the single
#: generic pin under `pinned`).  Nothing sits on the happy path but one set lookup per call.
_SMEM_DEMOTED: set = set()


_ACTIVE_KERNELS: dict = {}     # family -> the {name: Autotuner} of its last `activate` (what a demotion widens)


def _widen(fam, name, k):
    """Give the autotuner the family's FULL search grid next to its calibrated / pinned configs: under `pinned` its decorator grid is a
    single config, and a single config that does not fit leaves nothing to fall back to.  Triton's search skips the configs that raise
    `OutOfResources`; `prune` hands the demoted group (`~smem`) every config, other groups still get their own set."""
    b = _FALLBACK_GRIDS.get((fam, name))
    if b is None:
        return
    ck = _FAMILIES[fam]["config_key"]
    have = {ck(c) for c in k.configs}
    try:
        extra = [c for c in b[2]() if ck(c) not in have]        # b = (generic_pin, light, full)
    except Exception:                                           # noqa: BLE001
        return
    k.configs = list(k.configs) + extra


def smem_demote(fam, T) -> bool:
    """Demote (family, T) to the generic search; True the first time (the caller retries), False when it already was or the
    family is not calibrated (nothing to fall back to)."""
    if not enabled(fam) or (fam, T) in _SMEM_DEMOTED:
        return False
    _SMEM_DEMOTED.add((fam, T))
    for name, k in _ACTIVE_KERNELS.get(fam, {}).items():
        _widen(fam, name, k)
    sys.stderr.write("[calibration] %s: a calibrated config did not fit shared memory at chunk %s; this chunk length now "
                     "searches the generic grid\n" % (fam, T))
    return True


def active_group(fam):
    c = _CTX.get(fam)
    return None if c is None else c[0]


def prune(configs, fam, kernel):
    """Narrow `configs` to the active group's calibrated set (the `default`
    entry for a group the kernel was not calibrated on); `configs`
    unchanged when nothing applies or narrowing would empty it."""
    ent = _IDS.get((fam, kernel))
    ctx = _CTX.get(fam)
    if ent is None or ctx is None:
        return configs
    group, occ = ctx
    _, hit = ent
    fm = _FAMILIES[fam]
    if group.startswith("~smem"):
        return configs                       # a demoted chunk length: the whole (widened) grid, Triton skips what does not fit
    if group.startswith("~"):
        gen = _GENERIC.get((fam, kernel))
        keep = [c for c in configs if gen and fm["config_key"](c) in gen]
        return keep or configs
    cfgs, _ = pick(hit, group if group in hit["groups"] else None,
                   _TIER[(fam, kernel)], occ)
    allowed = {fm["cfg_key"](c) for c in cfgs}
    keep = [c for c in configs if fm["config_key"](c) in allowed]
    return keep or configs


def config_key_of(obj, fam) -> str:
    return _FAMILIES[fam]["config_key"](obj)


def source(fam, kernel) -> str:
    return _SOURCE.get((fam, kernel), "unrecorded")
