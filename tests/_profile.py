"""Test PROFILES: which parametrised cells a run collects.

    user     the default of a tree that carries no deep-suite extension (the exported user suite): one minimal cell per test
             function (the thinning of `smoke`); the contract suites (kernel cells and torch-route cells) run over their base
             spaces (every covering cell the oracle check, the recipes every check).  About a quarter of an hour on one GPU from a cold cache.
    smoke    the DEFAULT, the public first run -- "does it work here":
             CPU tests: ONE minimal cell per test function (the compact cell
             with the smallest R*O, then N) plus every LLM-cell cell;
             minutes.  GPU tests: the contract suite's fp64-oracle check
             (forward and every gradient) on ONE representative recipe per
             kernel route x precision x rotation x rank class of the SSD-MIMO
             and MH families (`smoke_recipes`); ~10-15 min cold on one GPU.
             No other GPU test runs under smoke.
    compact  "a brief general check": the grid below.  Our merge gate
             (calibrated tuning mode, impact selection), <= 1 h warm on one GPU.
    full     every cell, "all cases and shapes" -- nightly, pre-release,
             and whenever a kernel's specialisation space changed.

Selected with ``pytest --profile {smoke,user,compact,full}`` or the environment
variable ``HYBRIDSSM_TEST_PROFILE`` (the option wins).  This is a
COLLECTION-TIME filter: no parametrisation is deleted, ``--profile full``
collects exactly the cells the suite defines, and a dropped cell is
reported as deselected.

What ``compact`` restricts -- three AXES of a cell, read from its own
parametrisation (named parameters first, then its node-id tokens):

    R, O         (R, O) in {(1,1), (2,2), (4,4), (1,4)}, plus ONE small
                 rectangular pair per test function: the first of R2O1,
                 R1O2, R4O1, R4O2, R2O4 ... that the function has (a
                 rectangular cell is what caught the gate_z d_out = H*G*O
                 sizing bug, dc481bc, at R2O1 / R1O2)
    N            N in {32, 128}
    pair_layout  "interleaved"; "half" only where RULINGS keeps it

A cell that is out of profile on an axis is dropped ONLY when a SIBLING --
the same test function, the same values of every other parameter -- is in
profile on that axis.  So a grid loses its extra R/O/N/layout points, while
a cell that exists to test one value (an R8O8 compile-budget guard, an
N96-tiled geometry row, a half-layout-only test) has no in-profile sibling
and stays.  Unknown or unparseable cells stay in.  Parameters named
``seed`` / ``*_seed`` are not part of a sibling's identity (the geometry
grids give every row its own seed).

``RULINGS`` (by name) and ``PATTERN_RULINGS`` (by name pattern) add, per
test function, values that are in profile besides the defaults (a cell
that exists to test a specific N, e.g. an N-tiling boundary, keeps ONE such
value), named cells to keep (the half-layout regression cells), or keep the
whole function (a refusal / warning test: the refused value IS the claim).
Every ruling carries its reason.

Explicitly named node ids are never filtered: ``pytest path::test[id]``
runs that cell under either profile.

This module is imported by ``tests/conftest.py`` AND loaded by
external test runners for trees whose
conftest predates it, so it must stay importable on its own (stdlib only).
"""
import os
import re

PROFILES = ("smoke", "user", "compact", "full")
#: the profiles that keep one minimal cell per test function
THIN = ("smoke", "user")
DEFAULT_PROFILE = "smoke"      # a tree with the deep-suite extension; a base-only tree defaults to "user" (resolve)
ENV = "HYBRIDSSM_TEST_PROFILE"

#: compact keeps, per test function, a covering subset of the cells that survive the axis rules: every pair of
#: parameter values that some cell realises stays realised (`pair_cover`).  HYBRIDSSM_PROFILE_PAIRS=0 disables it.
PAIRWISE = os.environ.get("HYBRIDSSM_PROFILE_PAIRS", "1") not in ("0", "")

CORE_RO = frozenset({(1, 1), (2, 2), (4, 4), (1, 4)})
CORE_N = frozenset({32, 128})
CORE_LAYOUT = frozenset({"interleaved"})

#: GPU tests the smoke profile keeps: file -> test functions.  Everything else that needs a
#: device is out of smoke (the compact profile is the GPU gate).
SMOKE_GPU = {"test_contract.py": frozenset({"test_oracle"})}
#: contract families the smoke profile runs unless the caller names families
SMOKE_FAMILIES = ("mh", "ssd")


def smoke_recipes(cells, route):
    """The smoke recipes of ONE contract family: the first recipe of every
    (route, precision, rotation, rank class), a shipped production recipe
    preferred over a regression recipe, a light-tier duplicate never.
    `cells`: the family's `Cell`s (`.kind`, `.name`, `.c`); `route(c)`: the
    adapter's expected-route function.  Returns the chosen cell names."""
    def key(c):
        r = c.get("R")
        return (str(route(c)), c.get("dtype", c.get("cls")), c.get("cplx", c.get("rot")),
                None if r is None else min(int(r), 2))

    recipes = [c for c in cells if c.kind == "recipe" and dict(c.c).get("tier") != "light"]
    recipes.sort(key=lambda c: c.name.startswith("recipe_reg_"))      # stable: production recipes first
    chosen, seen = [], set()
    for cell in recipes:
        k = key(dict(cell.c))
        if k not in seen:
            seen.add(k)
            chosen.append(cell.name)
    return chosen


#: function name -> ruling.  Keys: "N" / "RO" / "layout" = extra in-profile
#: values; "cells" = node-id parameter strings (callspec ids) kept whatever
#: the axes say; "keep" = True keeps every cell.  Always give the reason.
RULINGS = {
    # -- cells that exist to test one N ("keep one such cell each")
    "test_k9_the_pair_swap_is_bit_identical_where_it_is_on": dict(
        N={64}, why="the launcher takes the register pair swap at N >= 64; "
                    "N64 is the first swap cell, N32 the load-form cell"),
    "test_k9_a_zero_rotation_is_exact_at_ieee_and_rounding_bounded_at_tf32":
        dict(N={64}, why="as above (the renamed cell)"),
    "test_ssd_n_tiled_matches_fp64_recurrence": dict(
        N={192}, why="N-tiling: 192 is the first tiled, non-power-of-two N"),
    "test_ssd_any_state_size_matches_fp64_recurrence": dict(
        N={5}, why="arbitrary N: one sub-floor, odd N besides 32 / 128"),
    "test_ssd_n_tile_split": dict(
        N={144}, why="host tile-split arithmetic: one N past the 128 tile"),
    "test_bign_r2o2_short_chunk_matches_fp64_recurrence": dict(
        N={256}, why="big-N route: 256 is the widest N it serves"),
    "test_bign_rotating_route_serves_the_whole_128_256_range": dict(
        N={256}, why="big-N rotating route: its widest N"),
    "test_ssd_smem_clamp_fires_exactly_on_the_over_budget_cells": dict(
        keep=True, why="'exactly': the N64 rows are the cells the clamp "
                       "must NOT fire on"),
    "test_k29_the_kb1_nsplit_is_exact": dict(
        N={64}, why="the N-split lever: N64 is its smallest split geometry"),
    # -- the rotate-half layout: a small explicit set of half cells
    "test_k13_the_fused_trap_adjoint_serves_general_rank": dict(
        cells={"half-fp32-R2O2-m3-theta_h", "half-bf16-R2O2-m3-theta_h"},
        why="half layout, fp32 and bf16 at one shape (R2O2, full "
            "rotation feature set)"),
    "test_k13_the_fused_trap_adjoint_is_the_same_function": dict(
        cells={"half-fp32-m3-theta_h", "half-bf16-m3-theta_h"},
        why="half layout, fp32 and bf16 at one shape"),
    "test_pairhalf_io_nograd_forward_matches_fp64": dict(
        keep=True, why="sm_90 rotate-half 16-bit-MMA T64 regression"),
    # -- the A13 CORE covering designs ARE the compact gate
    # (no axis filter on them: their R/O = 21 and N = 32
    #    cells are design points, not grid siblings)
    "test_a13_core_ssd_scan_matches_the_sequential_oracle": dict(
        keep=True, why="A13 SSD core covering design = the compact gate"),
    "test_a13_core_mh_scan_matches_the_sequential_oracle": dict(
        keep=True, why="A13 MH core covering design = the compact gate"),
    # -- smoke keeps every LLM-cell cell (the production geometry)
    "test_llm_cell_fused_kernel_matches_fp64": dict(
        smoke="all", why="the LLM mixer cell: production geometry, both "
                         "ranks, both rotation laws"),
}

#: test functions collected ONLY under `--profile full`: the A13 factorial
#: grids (the covering-design CORE of the same claims is the compact gate,
#: the `test_a13_core_*` tests).  Explicitly named node ids still run under any profile.
FULL_ONLY = frozenset({
    "test_a13_ssd_scan_matches_the_sequential_oracle_over_the_feasible_space",
    "test_a13_mh_scan_matches_the_sequential_oracle_over_the_feasible_space",
    # K0 at k = 96 / 128: 313-693 s of cold LLVM lowering per cell
    "test_k0_serves_its_wide_block_range",
})

#: test functions the `smoke` profile drops entirely (minutes-long cells
#: whose claim a smoke cell of the same family already carries).
NOT_SMOKE = frozenset({
    "test_a13_trained_checkpoint_gradients_match_the_sequential_oracle",
})

#: (name regex, ruling) -- applied when no exact-name ruling exists.
PATTERN_RULINGS = (
    # the K7-K13 geometry grids: the "N96-tiled" row is the N-tiling cell
    (re.compile(r"holds_over_the_geometry_grid$"),
     dict(N={96}, why="the N96-tiled row exercises the N-tiling path")),
    # a refusal / warning cell exists for its out-of-envelope value
    (re.compile(r"refus|reject|raise|warn"),
     dict(keep=True, why="envelope edge: the refused value IS the claim")),
)

_SEED = re.compile(r"(^|_)seed$")
_RO_ID = re.compile(r"(?<![A-Za-z])[Rr](\d+)[Oo](\d+)(?![0-9])")
_N_ID = re.compile(r"(?<![A-Za-z])N(\d+)(?![0-9])")
_R_KEYS = ("R", "rank")
_O_KEYS = ("O", "outputs")
_N_KEYS = ("N", "state", "d_state")
_LAYOUT_KEYS = ("pair_layout",)


def _int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    return None


def _repr(v):
    try:
        return repr(v)
    except Exception:                        # noqa: BLE001
        return f"<{type(v).__name__}>"


def _sig(params, drop):
    return tuple(sorted((k, _repr(v)) for k, v in params.items()
                        if k not in drop and not _SEED.search(k)))


def axes(pid, params):
    """{axis: (value, sibling signature)} for the axes this cell carries."""
    out = {}
    pid = pid or ""
    # R, O: named parameters, then the id token (R2O2 / r2o2)
    rk = next((k for k in _R_KEYS if k in params), None)
    ok = next((k for k in _O_KEYS if k in params), None)
    ro = None
    if rk and ok and _int(params[rk]) and _int(params[ok]):
        ro = ((_int(params[rk]), _int(params[ok])), _sig(params, {rk, ok}))
    elif rk and isinstance(params[rk], (tuple, list)) and \
            len(params[rk]) == 2 and all(_int(x) for x in params[rk]):
        ro = (tuple(params[rk]), _sig(params, {rk}))
    else:
        m = _RO_ID.findall(pid)
        if len(m) == 1:
            ro = ((int(m[0][0]), int(m[0][1])),
                  ("#id", _RO_ID.sub("R*O*", pid)))
    if ro is not None:
        out["RO"] = ro
    # N: named parameter, then the id token (N128, P64N128)
    nk = next((k for k in _N_KEYS if k in params and _int(params[k])), None)
    if nk is not None:
        out["N"] = (_int(params[nk]), _sig(params, {nk}))
    else:
        m = _N_ID.findall(pid)
        if len(m) == 1:
            out["N"] = (int(m[0]), ("#id", _N_ID.sub("N*", pid)))
    lk = next((k for k in _LAYOUT_KEYS if k in params), None)
    if lk is not None and params[lk] in ("interleaved", "half"):
        out["layout"] = (params[lk], _sig(params, {lk}))
    return out


def ruling(fn):
    """The ruling for one test function ({} when none)."""
    if fn in RULINGS:
        return RULINGS[fn]
    for rx, rule in PATTERN_RULINGS:
        if rx.search(fn):
            return rule
    return {}


def _rect_pick(ros):
    """The function's one small rectangular pair outside CORE_RO."""
    rect = sorted((r for r in ros if r[0] != r[1] and r not in CORE_RO
                   and max(r) <= 4), key=lambda r: (r[0] * r[1], -r[0]))
    return rect[:1]


def _smoke_key(a, order):
    """The 'minimal' cell of a family: smallest R*O, then N, interleaved."""
    ro = a.get("RO", ((0, 0), None))[0]
    return (ro[0] * ro[1], max(ro), a.get("N", (0, None))[0],
            a.get("layout", ("interleaved", None))[0] != "interleaved", order)


_ADDR = re.compile(r" at 0x[0-9a-f]+")


def cell_cost(pid, params):
    """A static, deterministic proxy of what a cell costs to compile (not to run): fp32 operands, a high
    R*O and a wide state multiply the kernels' register pressure and ptxas time.  Only the ORDER of costs
    matters (it breaks the covering subset toward cheap cells); it never removes a pair."""
    cost = 1.0
    text = (pid or "") + " " + " ".join(_repr(v) for v in params.values())
    if re.search(r"fp32|float32|ieee|tf32", text):
        cost *= 4.0
    a = axes(pid, params)
    ro = a.get("RO", ((1, 1), None))[0]
    cost *= 1.0 + 0.25 * (ro[0] * ro[1] - 1) ** 0.5
    if a.get("N", (0, None))[0] and a["N"][0] >= 128:
        cost *= 1.5
    return cost


def pair_cover(table, keep_always=(), cost=None):
    """Indices of a subset of `table` (a list of equal-length value tuples, one per cell) that realises every
    value of every column and every pair of (column, value) the table realises together in some row: a greedy covering subset, deterministic
    (most uncovered pairs per unit cost first, then the earliest row; `cost` defaults to 1 per row).  Rows in
    `keep_always` are in the subset.  A column
    with one value carries no pair and is ignored."""
    cols = [j for j in range(len(table[0])) if len({r[j] for r in table}) > 1] if table else []
    pairs = [frozenset([((a, table[i][a]),) for a in cols]
                       + [((a, table[i][a]), (b, table[i][b])) for ia, a in enumerate(cols) for b in cols[ia + 1:]])
             for i in range(len(table))]
    need = set().union(*pairs) if pairs else set()
    chosen = list(dict.fromkeys(keep_always))
    for i in chosen:
        need -= pairs[i]
    left = set(range(len(table))) - set(chosen)
    while need:
        best = max(sorted(left), key=lambda i: len(pairs[i] & need) / (1.0 if cost is None else cost[i]))
        if not pairs[best] & need:
            break
        chosen.append(best)
        left.discard(best)
        need -= pairs[best]
    return sorted(chosen)


def _pairwise(idx, cells, keep, why, rule):
    """Compact: of a test function's surviving cells keep a covering subset over all their parameters (every
    pair of parameter values that occurs together in some cell still does).  A full product of k axes
    shrinks to about the product of its two largest axes."""
    alive = [i for i in idx if keep[i]]
    if len(alive) < 6:
        return
    names = sorted({k for i in alive for k in cells[i][3] if not _SEED.search(k)})
    if any(_ADDR.search(_repr(cells[i][3].get(k))) for i in alive for k in names):
        return                                   # a value without a stable identity: keep the function whole
    table = [tuple(_repr(cells[i][3].get(k, "<unset>")) for k in names) for i in alive]
    pinned = [j for j, i in enumerate(alive) if cells[i][2] in rule.get("cells", ())]
    chosen = set(pair_cover(table, pinned, [cell_cost(cells[i][2], cells[i][3]) for i in alive]))
    for j, i in enumerate(alive):
        if j not in chosen:
            keep[i], why[i] = False, "pairs"


def select(cells, profile="compact"):
    """cells: list of (group key, function name, callspec id, params dict);
    the group key tells same-named functions of different files apart.

    Returns (keep, why) -- two lists aligned with `cells`: a bool, and for
    a dropped cell the reason ("RO" / "N" / "layout" / "smoke")."""
    keep = [True] * len(cells)
    why = [""] * len(cells)
    if profile == "full":
        return keep, why
    by_fn = {}
    for i, c in enumerate(cells):
        by_fn.setdefault(c[0], []).append(i)
    ax = [axes(c[2], c[3]) for c in cells]
    for idx in by_fn.values():
        fn = cells[idx[0]][1]
        if fn in FULL_ONLY or (profile in THIN and fn in NOT_SMOKE):
            for i in idx:
                keep[i], why[i] = False, "full-only" if fn in FULL_ONLY else "smoke"
            continue
        rule = ruling(fn)
        if not rule.get("keep"):
            allowed = {
                "RO": set(CORE_RO) | set(rule.get("RO", ())) | set(
                    _rect_pick({ax[i]["RO"][0] for i in idx if "RO" in ax[i]})),
                "N": set(CORE_N) | set(rule.get("N", ())),
                "layout": set(CORE_LAYOUT) | set(rule.get("layout", ())),
            }
            # signatures that have an in-profile member, per axis
            covered = {a: {ax[i][a][1] for i in idx
                           if a in ax[i] and ax[i][a][0] in allowed[a]}
                       for a in allowed}
            cells_kept = rule.get("cells", ())
            for i in idx:
                if cells[i][2] in cells_kept:
                    continue
                for a, (val, sig) in ax[i].items():
                    if val not in allowed[a] and sig in covered[a]:
                        keep[i], why[i] = False, a
                        break
        if profile == "compact" and not rule.get("keep") and PAIRWISE:
            _pairwise(idx, cells, keep, why, rule)
        if profile in THIN and not rule.get("smoke") == "all":
            alive = [i for i in idx if keep[i]] or idx
            best = min(alive, key=lambda i: _smoke_key(ax[i], i))
            for i in idx:
                if i != best and keep[i]:
                    keep[i], why[i] = False, "smoke"
    return keep, why


USER_DEFAULT = "user"


def base_only():
    """True in a tree that has no deep-suite extension (a `tests/_contract_*/` package), or when HYBRIDSSM_NO_OVERLAY is set."""
    import glob
    if os.environ.get("HYBRIDSSM_NO_OVERLAY", "") not in ("", "0"):
        return True
    return not glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_contract_*", "overlay.p*"))


def resolve(config=None):
    """The active profile: --profile, then $HYBRIDSSM_TEST_PROFILE."""
    val = None
    if config is not None:
        try:
            val = config.getoption("hybridssm_profile")
        except (ValueError, AttributeError):
            val = None
    val = (val or os.environ.get(ENV) or (USER_DEFAULT if base_only() else DEFAULT_PROFILE)).strip().lower()
    if val not in PROFILES:
        raise ValueError(f"unknown test profile {val!r}; one of {PROFILES}")
    return val


def _explicit_ids(config):
    """Parametrised node ids named on the command line."""
    out = set()
    for a in getattr(config, "args", ()) or ():
        if "::" in a and "[" in a:
            out.add(a.split("::", 1)[1])
    return out


#: test files whose parametrisation is a family (or a scope), not a cell of a grid: the profile never thins them
#: a test module that sizes itself (covering sets, one test per family) sets `PROFILE_SELF_SIZED = True`; the profile leaves it whole
SELF_SIZED_ATTR = "PROFILE_SELF_SIZED"


def apply_profile(config, items):
    """Deselect the cells the active profile drops.  Returns the count."""
    if getattr(config, "_hybridssm_profile_applied", False):
        return 0
    config._hybridssm_profile_applied = True
    profile = resolve(config)
    config._hybridssm_profile = profile
    if profile == "full" or os.environ.get("FG_SELECT"):
        return 0            # FG_SELECT: the runner's collection applied it
    explicit = _explicit_ids(config)
    if profile == "smoke":
        gone = [it for it in items if it.get_closest_marker("gpu") is not None
                and it.nodeid.split("::", 1)[-1].split("[", 1)[0] not in SMOKE_GPU.get(it.path.name, ())
                and it.nodeid.split("::", 1)[-1] not in explicit]
        if gone:
            drop = {id(it) for it in gone}
            items[:] = [it for it in items if id(it) not in drop]
            config.hook.pytest_deselected(items=gone)
    cells, pos = [], []
    for j, it in enumerate(items):
        cs = getattr(it, "callspec", None)
        if (cs is None or getattr(getattr(it, "module", None), SELF_SIZED_ATTR, False) or it.get_closest_marker("contract_cells") is not None
                or (profile == "user" and it.path.name == "test_contract_torch.py")):
            # the contract suite sizes itself (covering set; smoke = recipes), and its static / refusal checks are
            # one test per family: a family is not a cell to thin out
            continue
        fn = getattr(it, "originalname", None) or it.name.split("[", 1)[0]
        group = it.nodeid.split("[", 1)[0]
        cells.append((group, fn, cs.id, dict(cs.params)))
        pos.append(j)
    keep, _ = select(cells, profile)
    drop_idx = set()
    for j, k in zip(pos, keep):
        if not k:
            it = items[j]
            tail = it.nodeid.split("::", 1)[-1]
            if tail in explicit:
                continue
            drop_idx.add(j)
    if not drop_idx:
        return 0
    dropped = [it for j, it in enumerate(items) if j in drop_idx]
    items[:] = [it for j, it in enumerate(items) if j not in drop_idx]
    config.hook.pytest_deselected(items=dropped)
    return len(dropped)
