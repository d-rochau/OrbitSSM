"""The contract suite: generic checks, written once, over every registered
family adapter (tests/_contract/).

Cells per family = its covering set (all feasible pairs of option values, compile class and
precision included as factors) + its shipped recipes, run exactly.  The adapters in this tree declare a base space, a small
set of cells per family, precision and route: what a user of the library meets.  `--family mh,ssd` (env HYBRIDSSM_FAMILIES)
selects families; the `smoke` profile runs the recipes only.

Checks:
  test_oracle        forward + every leaf gradient vs the family's independent fp64 oracle, per the family's tolerance
                     rule (16-bit: >= 5 draws, pooled bias); finiteness first
  test_determinism   forward twice bitwise; backward twice bitwise under the family's deterministic options
  test_continuation  two calls joined by the returned state == one call
  test_inference     the same cell without autograd: the same route, a forward that passes the oracle
  test_retain_graph  a second backward through a retained graph gives the same gradients and modifies no input
  test_compile       recipes under torch.compile == the oracle (families with a `compile_check` run their own:
                     fullgraph, no graph break, forward + every gradient, the decode step)
"""
import contextlib
import os

PROFILE_SELF_SIZED = True        # tests/_profile.py: the module sizes its own cells

import pytest
import torch

import _contract
from _contract import core

pytestmark = [pytest.mark.gpu, pytest.mark.contract_cells]

FAMILIES = _contract.load_all()
_CELLS = {}


#: covering strength per profile: smoke = the recipes only, compact = all pairs, full = pairs + group triples
STRENGTH = {"smoke": 0, "user": 2, "compact": 2, "full": 3}


def _family_cells(name, profile="compact"):
    key = (name, profile)
    if key not in _CELLS:
        cells = core.cells(FAMILIES[name], strength=STRENGTH[profile])[0]
        # HYBRIDSSM_CONTRACT_SHARD=i/n: every n-th covering cell (recipes in
        # shard 0 only), for jobs under a time limit 
        shard = os.environ.get("HYBRIDSSM_CONTRACT_SHARD")
        if shard:
            i, n = (int(x) for x in shard.split("/"))
            cover = [c for c in cells if c.kind == "cover"]
            keep = {id(c) for c in cover[i::n]}
            cells = [c for c in cells
                     if (c.kind == "cover" and id(c) in keep) or (c.kind != "cover" and i == 0)]
        # HYBRIDSSM_CONTRACT_CELLS=<file>: run only the cells named there (one cell name per line; recipes by their
        # "recipe_..." name) -- the re-run of the cells a previous tree failed, e.g. on a fixed tree
        only = os.environ.get("HYBRIDSSM_CONTRACT_CELLS")
        if only:
            want = {ln.strip() for ln in open(only) if ln.strip()}
            cells = [c for c in cells if c.name in want]
        _CELLS[key] = cells
    return _CELLS[key]


@pytest.fixture(scope="session", autouse=True)
def _prepare_families(request):
    """Once per session, before any cell: each selected family's optional
    `prepare()` (e.g. SSD's tools/check_calibration.py, which fails fast)."""
    for fam in _selected(request.config):
        if hasattr(FAMILIES[fam], "prepare"):
            FAMILIES[fam].prepare()


@pytest.fixture(autouse=True)
def _reset_adapter_state(request):
    """After every cell: an adapter that forces module-level pipeline flags in `build` (pipeline selectors such as
    a fold route's switches) restores them, so no cell leaks its pipeline into the next one."""
    yield
    cs = getattr(request.node, "callspec", None)
    cell = None if cs is None else cs.params.get("cell")
    if cell is not None and hasattr(FAMILIES[cell.family], "reset"):
        FAMILIES[cell.family].reset()


def _selected(config):
    v = config.getoption("hybridssm_family", None) or os.environ.get("HYBRIDSSM_FAMILIES")
    names = [x.strip() for x in v.split(",")] if v else sorted(FAMILIES)
    if not v:
        from _profile import SMOKE_FAMILIES, resolve
        if resolve(config) == "smoke":          # the public smoke: the two kernel families (naming one runs it)
            names = [x for x in names if x in SMOKE_FAMILIES]
    unknown = [x for x in names if x not in FAMILIES]
    if unknown:
        raise pytest.UsageError(f"unknown --family {unknown}; registered: {sorted(FAMILIES)}")
    return names


#: checks that run on the recipes + a deterministic slice of the covering cells (every n-th by a stable hash of the cell
#: name), n per profile; every other check runs on every cell.  The oracle check (forward + every gradient against the
#: independent oracle) is the covering guarantee and always runs on every cell.
_SLICE = {
    # basic checks of a cell: determinism, the no-grad forward, state continuation
    "basic": {"tests": {"test_determinism", "test_inference", "test_continuation"},
              "n": {"smoke": 1, "user": 0, "compact": 3, "full": 1}},
    # the retained-graph check runs the cell twice more: the recipes in every profile that runs them, a tenth of the covering cells
    "meta": {"tests": {"test_retain_graph"}, "n": {"smoke": 0, "user": 0, "compact": 10, "full": 3}},
}


_SLICE_TESTS = set().union(*(g["tests"] for g in _SLICE.values()))


def _primary(cell):
    """A shipped production recipe (every check runs on it), as opposed to a REGRESSION recipe (`reg_*`): a pinned
    configuration that reproduces one specific past red, which the oracle / basic checks hold and the heavier checks
    sample like a covering cell."""
    return cell.kind == "recipe" and not cell.name.startswith("recipe_reg_")


def _in_slice(cell, test, profile, table=None):
    """Primary recipes always; any other cell when a stable hash of its name falls in the profile's slice of the test."""
    if _primary(cell):
        return True
    import zlib                      # stable across processes (hash() is salted)
    for g in (table or _SLICE).values():
        if test in g["tests"]:
            n = g["n"][profile]
            return n > 0 and zlib.crc32(cell.name.encode()) % n == 0
    return True


def generate(metafunc, admit=None):
    """The cells of one check: every selected family's cells of the active profile that the check applies to.  `admit(ad, cell,
    check, profile)` is the check's own rule (None: this module's)."""
    if "cell" not in metafunc.fixturenames:
        return
    if not torch.cuda.is_available():
        # every cell of this file is a kernel cell (module-level gpu mark): without CUDA none can run, and generating the
        # covering sets for them costs up to a minute per family (the torch-route cells are tests/test_contract_torch.py)
        metafunc.parametrize("cell", [], ids=[])
        return
    from _profile import resolve
    profile = resolve(metafunc.config)
    smoke = profile == "smoke"
    want = metafunc.function.__name__
    cells = []
    from _profile import smoke_recipes
    for fam in _selected(metafunc.config):
        ad = FAMILIES[fam]
        fam_cells = _family_cells(fam, profile)
        keep = set(smoke_recipes(fam_cells, lambda c: ad.space.route_of(c) if ad.space.route else None)) if smoke else None
        for cell in fam_cells:
            if smoke and cell.name not in keep:
                continue
            if not (admit or _admit)(ad, cell, want, profile):
                continue
            cells.append(cell)
    metafunc.parametrize("cell", cells, ids=[c.id for c in cells])


def _admit(ad, cell, want, profile):
    c = cell.c
    if want == "test_continuation" and not ad.continuation(c):
        return False
    if want == "test_compile" and not ad.compiled(cell):
        return False
    return want not in _SLICE_TESTS or _in_slice(cell, want, profile)


def pytest_generate_tests(metafunc):
    generate(metafunc)


class _Declared(dict):
    """The declared-nondeterministic map; the key "grad:*" declares EVERY gradient (a family whose whole backward is
    one reduction-atomics scan, e.g. fp32 operands)."""

    def __contains__(self, name):
        return dict.__contains__(self, name) or (str(name).startswith("grad:") and dict.__contains__(self, "grad:*"))


def _declared(ad, c):
    """{tensor name: reason} the family declares nondeterministic for this cell (`nondeterministic_for(c)` when the
    set depends on the cell -- e.g. only one route's atomics -- else the class-wide `nondeterministic`)."""
    return _Declared(ad.nondeterministic_for(c) if hasattr(ad, "nondeterministic_for")
                     else getattr(ad, "nondeterministic", {}))


def _split_arm(values):
    """(environment, '@'-keyed adapter attributes) of a route-pair arm."""
    return ({k: v for k, v in values.items() if not k.startswith("@")},
            {k: v for k, v in values.items() if k.startswith("@")})


def _arm(ad, attrs, kw):
    return ad.arm(attrs, kw) if attrs else contextlib.nullcontext()


@contextlib.contextmanager
def _env(values):
    values, _ = _split_arm(values)
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Outs:
    def __init__(self, values):
        self.values = values
        self.weights = [torch.randn_like(v) for v in values]


def _assert_finite(cell, vals, what):
    for i, v in enumerate(vals):
        if v is not None:
            assert torch.isfinite(v).all(), f"{cell.id}: non-finite {what}[{i}]"


def _oom_skips(fn):
    """OOM is a recorded skip naming the cell, never a red (SSD review (f)):
    a small GPU cannot hold every cell.  The record goes to the skip reason
    and, when HYBRIDSSM_OOM_LOG is set, to that file; that every SHIPPED
    recipe runs on the device is a separate check (test_recipes_fit)."""
    import functools

    @functools.wraps(fn)
    def run(*a, cell, **k):
        try:
            return fn(*a, cell=cell, **k)
        except torch.cuda.OutOfMemoryError as e:
            log = os.environ.get("HYBRIDSSM_OOM_LOG")
            if log:
                with open(log, "a") as f:
                    f.write(f"{cell.id}\t{fn.__name__}\t{str(e)[:200]}\n")
            torch.cuda.empty_cache()
            pytest.skip(f"OOM (recorded): {cell.id}")
    return run


def _oracle_run(ad, c, cell, env, call=None, pool=None, check_route=True, transform=None, seeds=None):
    """Build every draw, run (forward + every leaf gradient), finiteness,
    served route, family oracle.  `call` replaces the adapter's call (e.g.
    a torch.compile'd one).  `seeds` replaces the adapter's draws (the extra
    draws of an escalation).  Without a `pool`, an adapter that has pools
    (`new_pool`) judges the draws together (a route-pair arm: `primary=False`)
    and escalates an undecided floor verdict (`_settle`)."""
    own = pool is None and seeds is None and hasattr(ad, "new_pool")
    if own:
        pool = ad.new_pool(c, primary=False)
    _oracle_draws(ad, c, cell, env, call, pool, check_route, transform, ad.draws(c) if seeds is None else seeds)
    if own:
        _settle(ad, c, cell, env, call, pool, check_route, transform)
        pool.assert_unbiased()


def _settle(ad, c, cell, env, call, pool, check_route, transform):
    """The extra draws that settle an undecided verdict (`SsdPool.extra_seeds`: the floor-ratio rule fails only on
    `FLOOR_MIN_DRAWS` draws).  A no-op for pools without the hook."""
    more = getattr(pool, "extra_seeds", lambda: [])()
    if more:
        _oracle_draws(ad, c, cell, env, call, pool, check_route, transform, more)


def _oracle_draws(ad, c, cell, env, call, pool, check_route, transform, seeds):
    call = call or ad.call
    _, attrs = _split_arm(env)
    with _env(env):
        for seed in seeds:
            args, kw = ad.build(c, seed=seed)
            if transform is not None:
                args, kw = transform(args, kw)
            with _arm(ad, attrs, kw):
                outs = _Outs(call(args, kw))
                _assert_finite(cell, outs.values, "output")
                if check_route and ad.space.route is not None:
                    want, got = ad.space.route_of(c), ad.served_route()
                    assert got == want, f"{cell.id}: served route {got!r}, expected {want!r}"
                leaves = ad.leaves(args, kw)
                grads = torch.autograd.grad(outs.values, leaves, outs.weights, allow_unused=True)
                _assert_finite(cell, grads, "gradient")
            ad.oracle(c, args, kw, outs, grads, pool=pool, label=cell.id + ":")


@_oom_skips
def test_oracle(require_cuda, cell):
    from _tolerance import DrawPool
    ad, c = FAMILIES[cell.family], cell.c
    pool = ad.new_pool(c) if hasattr(ad, "new_pool") else DrawPool()
    _oracle_run(ad, c, cell, ad.env(c), pool=pool)
    _settle(ad, c, cell, ad.env(c), None, pool, True, None)
    pool.assert_unbiased()


@_oom_skips
def test_determinism(require_cuda, cell):
    ad, c = FAMILIES[cell.family], cell.c
    with _env(ad.env(c)):
        args, kw = ad.build(c)
        extra = ad.determinism_kw()
        runs, n_out, gnames = [], 0, []
        leaf_names = ad.leaf_names(c, kw) if hasattr(ad, "leaf_names") else None
        for _ in range(2):
            vals = ad.call(args, kw, **extra)
            n_out = len(vals)
            w = [torch.ones_like(v) for v in vals]
            g = torch.autograd.grad(vals, ad.leaves(args, kw), w, allow_unused=True)
            gnames = [(leaf_names[j] if leaf_names else None) for j, x in enumerate(g) if x is not None]
            runs.append([v.detach() for v in vals] + [x for x in g if x is not None])
        # the spread's own conditioning (kappa per tensor, the fp32 prefix allowance), from the fp64 reference
        ctx = ad.determinism_context(c, args, kw) if hasattr(ad, "determinism_context") else None
    for i, (a, b) in enumerate(zip(*runs)):
        # a family without a deterministic backward (SSD) declares it: the forward stays bitwise, each gradient
        # goes through its own kappa-aware bound
        if getattr(ad, "bitwise_backward", True) or i < n_out:
            assert torch.equal(a, b), f"{cell.id}: run-to-run difference in tensor {i}"
        else:
            ad.grad_repeatability(c, f"{cell.id}[grad {i - n_out}]", a, b, ctx=ctx, name=gnames[i - n_out])


@_oom_skips
def test_continuation(require_cuda, cell):
    """Two calls joined by the returned state reproduce one call (scored by
    the family against its oracle's whole-sequence values)."""
    ad, c = FAMILIES[cell.family], cell.c
    with _env(ad.env(c)):
        ad.check_continuation(c, label=cell.id + ":")



@_oom_skips
def test_compile(require_cuda, cell):
    """Recipes under torch.compile, forward + backward, against the family
    oracle under its tolerance (Inductor fuses the glue and may change bits,
    so bitwise equality with eager is not the contract)."""
    ad, c = FAMILIES[cell.family], cell.c
    torch._dynamo.reset()
    if hasattr(ad, "compile_check"):
        with _env(ad.env(c)):
            ad.compile_check(c, label=cell.id + ":")
        return

    def compiled_call(args, kw):
        return torch.compile(lambda *a: ad.call(list(a), kw), dynamic=False)(*args)
    _oracle_run(ad, c, cell, ad.env(c), call=compiled_call, check_route=False)



@_oom_skips
def test_inference(require_cuda, cell):
    """(E, D) The cell WITHOUT autograd: it must not raise (a refusal belongs
    at dispatch, never in a mode-specific path), must take the expected
    route in this mode too (the SISO no-grad misroute), and its output must
    pass the family oracle."""
    ad, c = FAMILIES[cell.family], cell.c
    with _env(ad.env(c)):
        args, kw = ad.build(c)
        with torch.no_grad():
            vals = ad.call(args, kw)
        _assert_finite(cell, vals, "inference output")
        if ad.space.route is not None:
            want = (ad.space._in_env(c, ad.inference_route) if hasattr(ad, "inference_route")
                    else ad.space.route_of(c))
            got = ad.served_route()
            assert got == want, f"{cell.id}: no-grad served route {got!r}, expected {want!r}"
        ad.check_forward(c, args, kw, vals, label=cell.id + "[no_grad]:")


# ---------------------------------------------------------------------------
# a second backward through a retained graph (a user who calls backward twice)
# ---------------------------------------------------------------------------
def _run_full(ad, args, kw, w=None, retain=False):
    vals = ad.call(args, kw)
    w = w or [torch.ones_like(v) for v in vals]
    g = torch.autograd.grad(vals, ad.leaves(args, kw), w, allow_unused=True, retain_graph=retain)
    return [v.detach() for v in vals], list(g), vals, w


def _noise_floor(ad, c, args, kw, names):
    """Relative spread {name: ||g1 - g2|| / ||g2||} of two IDENTICAL runs, for the tensors the family declares
    nondeterministic (fp32 atomics reorder their sums run to run): the floor under which two arms of one computation
    cannot be told apart.  A tensor that is deterministic has floor 0 and is held to its bound exactly."""
    declared = _declared(ad, c)
    if not declared:
        return {}
    v1, g1, _, _ = _run_full(ad, args, kw)
    v2, g2, _, _ = _run_full(ad, args, kw)
    out = {}
    for n, a, b in zip(names, v1 + g1, v2 + g2):
        if n in declared and a is not None and b is not None and float(b.double().norm()) > 0:
            out[n] = float((a.double() - b.double()).norm() / b.double().norm())
    return out


def _compare(ad, c, label, names, got, ref, exact=True, floor=None):
    """Tensors pairwise: bitwise when `exact`, else the family's `close`;
    tensors the family declares nondeterministic go through `close` too -- and, where `close` refuses, are accepted
    within 4x their own run-to-run spread (`_noise_floor`: two arms of one computation differ by at least that)."""
    declared = _declared(ad, c)
    for n, a, b in zip(names, got, ref):
        if a is None and b is None:
            continue
        if exact and n not in declared:
            assert a is not None and b is not None and torch.equal(a, b), f"{label}: {n} differs"
            continue
        try:
            ad.close(c, f"{label}: {n}", a, b)
        except AssertionError:
            f = (floor or {}).get(n)
            if n not in declared or not f:
                raise
            rel = float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))
            assert rel <= 4 * f, f"{label}: {n} differs by {rel:.2e} > 4x its run-to-run spread {f:.2e}"


@_oom_skips
def test_retain_graph(require_cuda, cell):
    """A second backward through a retained graph gives the same gradients,
    and no call modifies its inputs."""
    ad, c = FAMILIES[cell.family], cell.c
    with _env(ad.env(c)):
        args, kw = ad.build(c)
        names = ad.tensor_names(args, kw)
        before = [t.detach().clone() for t in ad.leaves(args, kw)]
        vals = ad.call(args, kw)
        w = [torch.ones_like(v) for v in vals]
        g1 = torch.autograd.grad(vals, ad.leaves(args, kw), w, allow_unused=True, retain_graph=True)
        g2 = torch.autograd.grad(vals, ad.leaves(args, kw), w, allow_unused=True)
        after = [t.detach() for t in ad.leaves(args, kw)]
        floor = _noise_floor(ad, c, args, kw, names)
    n_out = len(vals)
    _compare(ad, c, f"{cell.id}[retain_graph]", names[n_out:], list(g2), list(g1), floor=floor)
    for n, a, b in zip(names[n_out:], after, before):
        assert torch.equal(a, b), f"{cell.id}: input {n} was modified in place"
