"""Contract-suite core: a family's feature space, its covering set, cells.

ONE contract suite, generic test functions written
once (tests/test_contract.py), one small adapter per kernel family
(tests/_contract/adapters/).  This module owns the family-independent part:

* `Space` -- the family's options as FACTORS with DECLARED constraints.  A
  factor is active only when one of its clauses holds (a clause maps parent
  factors to the values that enable it); an inactive factor takes the value
  "-" and generates no requirement of its own.  Constraints are data, so the
  generator derives feasibility itself; anything the declared form cannot say
  (a value-level rule, a device-only unit) goes in
  `infeasible(cfg, device) -> reason | None`, which the generator honours too;
  a dispatcher FALLBACK is not infeasibility -- it is `route(cfg)`, a derived
  factor the covering set includes and the served-route check verifies.
* `covering_set(space)` -- a deterministic greedy covering array: every
  feasible PAIR of factor values over all factors plus every feasible TRIPLE
  inside each dependency group (`space.groups`).  A requirement the declared
  constraints allow but `infeasible` rejects every time it is forced is reported
  in `Cover.refused`, never silently dropped.
* `Cell` -- one configuration of one family, with a stable id.

The compile class of a cell (dtype / geometry) is an ordinary factor, so the
covering guarantee holds ACROSS compile classes as well, without multiplying
the cell count by them.
"""
from __future__ import annotations

import dataclasses
import itertools
import random

import numpy as np

NA = "-"
#: an `infeasible` reason with this prefix says the STACK lacks a provider the cell needs (an optional package): the
#: cell is not generated and a recipe is skipped by name, never a red
ENV_UNAVAILABLE = "env:"


@dataclasses.dataclass
class Factor:
    domain: tuple
    when: tuple = ()          # clauses: ({parent: frozenset(values)}, ...); () = always active
    # conditional domains: ((clause, subdomain), ...); the first clause that
    # holds narrows the domain (e.g. R>1 -> T in {16, 32}); none -> `domain`
    domains: tuple = ()
    # True: the covering set requires no pair of this factor's values and draws its first value; the other values are
    # reachable only by a recipe (a shipped configuration that names them)
    recipe_only: bool = False


class Space:
    def __init__(self, factors: dict, groups=(), infeasible=None, route=None):
        self.factors = {k: (v if isinstance(v, Factor) else Factor(tuple(v)))
                        for k, v in factors.items()}
        self.names = list(self.factors)
        self.groups = [list(g) for g in groups]
        # infeasible(cfg, device) -> reason | None: no production path at all
        # (excluded and reported); route(cfg) -> label: the route that should
        # serve the cell (a derived factor the covering set includes, and the
        # served-route check compares against telemetry)
        self.infeasible = infeasible
        self.route = route
        self.device = None        # set by `cells()` from the GPU under test
        self.env = None           # set by `cells()` from the adapter
        for n, f in self.factors.items():
            for cl in list(f.when) + [c for c, _ in f.domains]:
                for p in cl:
                    if self.names.index(p) >= self.names.index(n):
                        raise ValueError(f"factor {n!r} reads {p!r}, which is not declared before it")
        for g in self.groups:
            for n in g:
                if n not in self.factors:
                    raise ValueError(f"group member {n!r} is not a factor")
        self.parents = sorted({p for f in self.factors.values()
                               for cl in list(f.when) + [c for c, _ in f.domains] for p in cl},
                              key=self.names.index)

    # -- configurations ---------------------------------------------------
    def active(self, name, cfg):
        f = self.factors[name]
        return not f.when or any(all(cfg.get(p) in vals for p, vals in cl.items()) for cl in f.when)

    def dom(self, name):
        f = self.factors[name]
        vals = list(f.domain)
        for _, sub in f.domains:
            vals += [v for v in sub if v not in vals]
        return tuple(vals) + ((NA,) if f.when else ())

    def domain_in(self, name, cfg):
        """The factor's domain given the (partial) configuration's parents."""
        f = self.factors[name]
        for cl, sub in f.domains:
            if all(cfg.get(p) in vals for p, vals in cl.items()):
                return tuple(sub)
        return f.domain

    def covered(self, name):
        """The values of `name` the covering set must pair: its whole domain, or none when it is `recipe_only` (a random draw
        gives it its first value; whether it is active at all is then up to the other factors and the recipes)."""
        return () if self.factors[name].recipe_only else self.dom(name)

    def chooser(self, rng):
        """pick(name, domain) for random draws of the covering set: a `recipe_only` factor always takes its first value."""
        return lambda n, d: d[0] if self.factors[n].recipe_only else rng.choice(d)

    def fill(self, fixed, pick):
        """A full configuration from `fixed` and pick(name, domain), or None."""
        c = {}
        for n in self.names:
            on = self.active(n, c)
            if n in fixed:
                if (fixed[n] == NA) == on:
                    return None
                if on and fixed[n] not in self.domain_in(n, c):
                    return None
                c[n] = fixed[n]
            else:
                c[n] = pick(n, self.domain_in(n, c)) if on else NA
        return c

    def wide(self, rng):
        """A configuration drawn from the WHOLE declared domains: a factor's conditional narrowing (`domains`) is ignored, so
        the draw can ask for a combination the family refuses (the covering set never does).  An inactive factor stays "-"."""
        c = {}
        for n in self.names:
            c[n] = rng.choice([v for v in self.dom(n) if v != NA]) if self.active(n, c) else NA
        return c

    def _in_env(self, cfg, fn):
        """Evaluate fn(cfg) under the cell's environment (`self.env(cfg)`):
        a dispatcher that reads env levers live decides route/feasibility
        from them ."""
        if self.env is None:
            return fn(cfg)
        import os
        vals = self.env(cfg)
        old = {k: os.environ.get(k) for k in vals}
        os.environ.update(vals)
        try:
            return fn(cfg)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def route_of(self, cfg):
        return self._in_env(cfg, self.route)

    def why_infeasible(self, cfg):
        if cfg is None:
            return "violates the declared constraints"
        if self.infeasible is None:
            return None
        return self._in_env(cfg, lambda c: self.infeasible(c, self.device))

    def valid(self, cfg):
        return self.why_infeasible(cfg) is None


@dataclasses.dataclass
class Cover:
    configs: list
    requirements: int
    refused: list          # requirements the declared constraints allow but `infeasible` always rejected
    routes: dict = dataclasses.field(default_factory=dict)   # expected route -> cell count


def _index(space):
    return {(n, v): i for i, (n, v) in enumerate((n, v) for n in space.names for v in space.dom(n))}


def _structures(space, idx):
    """Per assignment of the parent factors, the value set each factor may take."""
    rows = []
    for vals in itertools.product(*[space.dom(p) for p in space.parents]):
        c = space.fill(dict(zip(space.parents, vals)), lambda n, d: d[0])
        if c is None:
            continue
        row = np.zeros(len(idx), bool)
        for n in space.names:
            vs = [c[n]] if n in space.parents or c[n] == NA else space.domain_in(n, c)
            for v in vs:
                row[idx[(n, v)]] = True
        rows.append(row)
    return np.array(rows)


def requirements(space, triples=True):
    idx = _index(space)
    S = _structures(space, idx)
    combos = list(itertools.combinations(space.names, 2))
    for g in space.groups if triples else ():
        combos += list(itertools.combinations(g, 3)) if len(g) >= 3 else []
    out = set()
    for fs in combos:
        for vs in itertools.product(*[space.covered(f) for f in fs]):
            if S[:, [idx[(f, v)] for f, v in zip(fs, vs)]].all(axis=1).any():
                out.add(tuple(zip(fs, vs)))
    return sorted(out, key=str)


def covering_set(space, seed=20261003, candidates=400, forced_tries=2000, triples=True):
    """Greedy covering array: every feasible pair of factor values, plus (triples=True) every feasible triple inside
    each dependency group."""
    idx = _index(space)
    R = requirements(space, triples)
    t = max(len(r) for r in R)
    cols = np.array([[idx[fv] for fv in r] + [idx[r[-1]]] * (t - len(r)) for r in R])
    rng = random.Random(seed)
    left = np.ones(len(R), bool)
    rows, refused = [], []

    def draw(fixed=None):
        for _ in range(50):
            c = space.fill(fixed or {}, space.chooser(rng))
            if space.valid(c):
                return c
        return None

    def onehot(cfgs):
        M = np.zeros((len(cfgs), len(idx)), bool)
        for k, c in enumerate(cfgs):
            for n in space.names:
                M[k, idx[(n, c[n])]] = True
        return M

    while left.any():
        first = R[int(np.flatnonzero(left)[0])]
        forced = None
        for _ in range(forced_tries // 50):
            forced = draw(dict(first))
            if forced is not None:
                break
        if forced is None:
            refused.append(first)
            left[int(np.flatnonzero(left)[0])] = False
            continue
        pool = [forced] + [c for c in (draw() for _ in range(candidates)) if c is not None]
        H = onehot(pool)[:, cols[left]].all(axis=2)
        k = int(H.sum(axis=1).argmax())
        left[np.flatnonzero(left)[H[k]]] = False
        rows.append(pool[k])
    routes = {}
    if space.route is not None:
        rows = _cover_routes(space, rows, rng)
        for c in rows:
            r = space.route_of(c)
            routes[r] = routes.get(r, 0) + 1
    return Cover(rows, len(R), refused, routes)


def _cover_routes(space, rows, rng, samples=4000):
    """Extend `rows` so every (expected route, factor value) pair that a
    valid configuration can produce is covered: feasibility of a derived
    factor is not declarative, so it is OBSERVED over random valid draws."""
    def pairs(c):
        r = space.route_of(c)
        return {(r, n, c[n]) for n in space.names}
    pool = []
    for _ in range(samples):
        c = space.fill({}, space.chooser(rng))
        if space.valid(c):
            pool.append(c)
    want = set().union(*(pairs(c) for c in pool)) if pool else set()
    have = set().union(*(pairs(c) for c in rows)) if rows else set()
    left = want - have
    while left:
        best = max(pool, key=lambda c: len(pairs(c) & left))
        got = pairs(best) & left
        if not got:
            break
        rows.append(best)
        left -= got
    return rows


@dataclasses.dataclass(frozen=True)
class Cell:
    family: str
    kind: str              # "cover" | "recipe"
    name: str
    cfg: tuple             # sorted (factor, value) items

    @property
    def c(self):
        return dict(self.cfg)

    @property
    def id(self):
        return f"{self.family}:{self.name}"


def cell_name(space, cfg):
    return "-".join(f"{n}={cfg[n]}" for n in space.names if cfg[n] not in (NA, False, "none", "plain"))


def _device():
    try:
        import torch
        return torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    except Exception:                                   # pragma: no cover
        return None


def tiers():
    """The launch-tier values of the adapters' `tier` factor under the active tuning mode (tests/_tuning.py):
    the calibrated mode runs every cell on its ONE config; the light and full modes add the search slice."""
    import _tuning
    return ("pinned",) if _tuning.resolve() == "calibrated" else ("pinned", "light")


def _env_digest(extra=None, adapter=None):
    """sha256 over the torch / triton versions, the adapter's environment (optional providers), this module, the adapter's
    own module and the helper files its space / route read (`adapter.cover_sources`): everything the covering set depends on
    EXCEPT the production sources, which `_functions_digest` tracks at the granularity of the functions the generation ran."""
    import hashlib
    import inspect
    import pathlib
    tests = pathlib.Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    try:
        import torch
        import triton
        h.update(f"torch {torch.__version__} triton {triton.__version__}".encode())
    except Exception:                                   # noqa: BLE001  (CPU-only stacks)
        pass
    h.update(repr(sorted((extra or {}).items())).encode())
    h.update(("tiers " + repr(tiers())).encode())          # the tuning mode picks the launch-tier factor values
    files = [pathlib.Path(__file__)]
    if adapter is not None:
        files.append(pathlib.Path(inspect.getfile(type(adapter))))
        files += [tests.parent / f for f in getattr(adapter, "cover_sources", ())]   # helpers its space / route read
    from . import overlay_files
    files += [tests.parent / f for f in overlay_files()]            # an extension widens the space: its source is part of the key
    for f in files:
        h.update(str(f.relative_to(tests.parent)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:20]


def _src_root():
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[2] / "src" / "orbit_ssm")


def _traced(fn):
    """Run `fn()` and return (its value, {(module, qualname)}): every function under src/orbit_ssm that RAN, as the
    top-level function or method that encloses it (a nested function is part of its enclosing source).  `sys.monitoring`
    (PY_START, disabled per code object after its first call) costs almost nothing; `sys.setprofile` is the fallback when the
    tool id is taken (a profiler is attached)."""
    import os
    import sys
    root = _src_root()
    base = os.path.dirname(root)
    seen = set()

    def note(co, module=None):
        if co.co_filename.startswith(root) and co.co_name != "<module>":
            mod = module or os.path.relpath(co.co_filename, base)[:-3].replace(os.sep, ".")
            seen.add((mod[:-9] if mod.endswith(".__init__") else mod, co.co_qualname.split(".<locals>")[0]))

    mon = getattr(sys, "monitoring", None)
    tool = None
    if mon is not None:
        try:
            mon.use_tool_id(mon.PROFILER_ID, "contract-cover")
            tool = mon.PROFILER_ID
        except ValueError:
            tool = None
    if tool is not None:
        def on_start(code, offset):
            note(code)
            return mon.DISABLE
        mon.register_callback(tool, mon.events.PY_START, on_start)
        mon.set_events(tool, mon.events.PY_START)
        mon.restart_events()                           # a location disabled by an earlier trace reports again
        try:
            value = fn()
        finally:
            mon.set_events(tool, 0)
            mon.register_callback(tool, mon.events.PY_START, None)
            mon.free_tool_id(tool)
        return value, seen

    def prof(frame, event, arg):
        if event == "call":
            note(frame.f_code, frame.f_globals.get("__name__"))
    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        value = fn()
    finally:
        sys.setprofile(old)
    return value, seen


def _literal(v, depth=0):
    """True for a module constant made only of numbers, strings, bytes, None and containers of them (a table, a threshold,
    a name list); a registry that holds classes / functions / modules, or any other object, is import-order state and is not
    part of a function's identity."""
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return True
    if depth >= 4:
        return False
    if isinstance(v, (tuple, list, set, frozenset)):
        return len(v) <= 4096 and all(_literal(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return len(v) <= 4096 and all(_literal(k, depth + 1) and _literal(x, depth + 1) for k, x in v.items())
    return False


def _canon(v, depth=0):
    """A process-independent text of a module constant: sets sorted (string hashing is randomised per process), object
    addresses removed."""
    import re
    if depth < 4 and isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(_canon(x, depth + 1) for x in v)) + "}"
    if depth < 4 and isinstance(v, (list, tuple)):
        return "(" + ",".join(_canon(x, depth + 1) for x in v) + ")"
    if depth < 4 and isinstance(v, dict):
        return "{" + ",".join(f"{_canon(k, depth + 1)}:{_canon(x, depth + 1)}" for k, x in v.items()) + "}"
    return re.sub(r" at 0x[0-9a-f]+", "", repr(v))


def _functions_digest(funcs):
    """sha256 over the CURRENT source of each (module, qualname) and over the value of every plain global it names (module
    constants a rule reads: a table, a threshold).  A class (its body ran at import) hashes as a whole; an object whose
    source cannot be taken (a jit kernel wrapper, a renamed function) hashes its whole module file -- never a silent miss.
    An edit anywhere else in src/orbit_ssm -- the torch routes, the epilogues, the step, the State code -- leaves it
    unchanged, which is the point."""
    import hashlib
    import importlib
    import inspect
    import types
    h = hashlib.sha256()
    for mod, qual in sorted(funcs):
        try:
            module = importlib.import_module(mod)
        except Exception:                               # noqa: BLE001
            return None
        h.update(f"{mod}.{qual}\n".encode())
        try:
            obj = module
            for part in qual.split("."):
                obj = obj.__dict__[part] if isinstance(obj, type) and part in obj.__dict__ else getattr(obj, part)
            obj = getattr(obj, "__func__", None) or getattr(obj, "fget", None) or getattr(obj, "fn", None) or obj
            obj = inspect.unwrap(obj)
            h.update(inspect.getsource(obj).encode())
            code, glb = getattr(obj, "__code__", None), getattr(obj, "__globals__", None)
        except Exception:                               # noqa: BLE001  (not introspectable: the whole file)
            try:
                h.update(open(module.__file__, "rb").read())
            except Exception:                           # noqa: BLE001
                return None
            continue
        if code is None:
            continue
        for name in sorted(code.co_names):
            v = glb.get(name)
            if v is None or isinstance(v, (types.ModuleType, types.FunctionType, type)) or callable(v) or not _literal(v):
                continue
            h.update(f"{name}={_canon(v)[:4000]}".encode())
    return h.hexdigest()[:20]


def _digest_fresh(funcs):
    """`_functions_digest` as a FRESH process sees it.  The generation itself mutates module state the functions name (the
    adapters' env switches, memo tables), so the digest recorded with a cache entry must be taken before any of that, which
    is what every later process that loads the entry will see."""
    import json
    import os
    import subprocess
    import sys
    code = ("import json, sys; sys.path[:0] = json.loads(sys.argv[1]); from _contract import core; "
            "print(core._functions_digest([tuple(f) for f in json.loads(sys.argv[2])]))")
    try:
        out = subprocess.run([sys.executable, "-c", code, json.dumps(sys.path), json.dumps(sorted(funcs))],
                             capture_output=True, text=True, timeout=600, env=dict(os.environ))
    except (OSError, subprocess.SubprocessError):
        return None
    last = out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else "None"
    return None if last == "None" else last


def _generate_fresh(adapter, device, strength):
    """(Cover, {(module, qualname)}) from a FRESH process: nothing memoised by earlier work in this one (the adapters keep
    refusal memos) can hide a function the generation needs, so the recorded function set is complete."""
    import json
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    code = ("import json, pickle, sys; sys.path[:0] = json.loads(sys.argv[1]); import _contract; "
            "from _contract import core; ad = _contract.load_all()[sys.argv[2]]; dev = json.loads(sys.argv[3]); "
            "ad.space.device = None if dev is None else tuple(dev); ad.space.env = getattr(ad, 'env', None); "
            "cov, funcs = core._traced(lambda: core.covering_set(ad.space, seed=getattr(ad, 'seed', 20261003), "
            "triples=int(sys.argv[4]) >= 3)); pickle.dump((cov, sorted(funcs)), open(sys.argv[5], 'wb'))")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cover.pkl")
        subprocess.run([sys.executable, "-c", code, json.dumps(sys.path), adapter.family,
                        json.dumps(None if adapter.space.device is None else list(adapter.space.device)), str(strength), out],
                       check=True, env=dict(os.environ))
        with open(out, "rb") as f:
            cov, funcs = pickle.load(f)
    return cov, set(funcs)


def _cover_cache_file(adapter, device, strength=3):
    import os
    import pathlib
    root = os.environ.get("HYBRIDSSM_COVER_CACHE")
    if root == "off":
        return None
    root = pathlib.Path(root or pathlib.Path.home() / ".cache" / "hybridssm" / "cover")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    dev = "cpu" if device is None else "sm%d%d" % tuple(device)
    seed = getattr(adapter, "seed", 20261003)
    env = adapter.environment() if hasattr(adapter, "environment") else None
    return root / f"{adapter.family}-{dev}-{seed}-t{strength}-{_env_digest(env, adapter)}.pkl"


def file_cached(name, files, make):
    """`make()` cached on disk under `name`, keyed by the digest of `files` (repo-relative paths): for covering sets that
    depend on test files only (the torch-route spaces).  HYBRIDSSM_COVER_CACHE as for `cells`."""
    import hashlib
    import os
    import pathlib
    import pickle
    root = os.environ.get("HYBRIDSSM_COVER_CACHE")
    repo = pathlib.Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    from . import overlay_files
    for f in sorted(set(files) | set(overlay_files()) | {"tests/_contract/core.py"}):
        h.update(f.encode())
        h.update((repo / f).read_bytes())
    if root == "off":
        return make()
    root = pathlib.Path(root or pathlib.Path.home() / ".cache" / "hybridssm" / "cover")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return make()
    path = root / f"{name}-{h.hexdigest()[:20]}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:                                 # noqa: BLE001  (truncated / foreign pickle)
            pass
    value = make()
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(value))
    tmp.replace(path)
    return value


def cells(adapter, device="auto", strength=3):
    """Covering cells + the adapter's shipped recipes, as `Cell`s, for the
    device under test (its capability reaches `infeasible(cfg, device)`).
    strength: 2 = every feasible pair (+ the expected-route pairs), 3 = also every feasible triple inside each
    dependency group, 0 = the recipes only.
    The covering set is a pure function of (adapter, dispatcher sources,
    device, seed) and costs minutes for a big space, so it is cached on disk
    keyed by a digest of those sources (HYBRIDSSM_COVER_CACHE=off disables)."""
    import pickle
    adapter.space.device = _device() if device == "auto" else device
    adapter.space.env = getattr(adapter, "env", None)
    if strength == 0:                                   # recipes only (the smoke profile): no covering set at all
        cov = Cover([], 0, [])
        cache = None
    else:
        cache = _cover_cache_file(adapter, adapter.space.device, strength)
        cov = None
    if cache is not None and cache.exists():
        try:
            rec = pickle.loads(cache.read_bytes())
            # the production functions the generation ran must be unchanged (an edit elsewhere in src keeps the entry)
            if isinstance(rec, dict) and rec.get("digest") == _functions_digest(rec["funcs"]) is not None:
                cov = rec["cover"]
        except Exception:                                 # truncated / foreign pickle: recompute
            cov = None
    if cov is None and strength:
        from . import FAMILIES
        if cache is not None and FAMILIES.get(adapter.family) is adapter:
            cov, funcs = _generate_fresh(adapter, adapter.space.device, strength)
            digest = _digest_fresh(funcs)
            if digest is not None:
                tmp = cache.with_suffix(f".{__import__('os').getpid()}.tmp")
                tmp.write_bytes(pickle.dumps(dict(cover=cov, funcs=sorted(funcs), digest=digest)))
                tmp.replace(cache)
        else:                                             # an unregistered (test) adapter: generated here, never cached
            cov = covering_set(adapter.space, seed=getattr(adapter, "seed", 20261003), triples=strength >= 3)
    out = [Cell(adapter.family, "cover", cell_name(adapter.space, c), tuple(sorted(c.items())))
           for c in cov.configs]
    for name, c in adapter.recipes():
        full = adapter.space.fill(c, lambda n, d: d[0])
        why = adapter.space.why_infeasible(full)
        if why is not None:
            if why.startswith(ENV_UNAVAILABLE):      # a provider this stack does not have: recorded, not collected
                cov.skipped_recipes = getattr(cov, "skipped_recipes", []) + [(name, why)]
                continue
            raise ValueError(f"{adapter.family} recipe {name!r} is not a valid configuration: {why}")
        out.append(Cell(adapter.family, "recipe", "recipe_" + name, tuple(sorted(full.items()))))
    return out, cov
