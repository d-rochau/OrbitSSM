"""The refusal-path checks of the contract suite: what a family serves, and what it refuses, in both directions.

The covering arrays (tests/test_contract.py) exercise the FEASIBLE space; nothing there notices a feasible call that is
wrongly refused, or an infeasible request that is accepted or crashes without a name.  Five generic properties, over the
hooks each adapter declares (tests/_contract/__init__.py, "Refusal-path hooks"):

  * every feasible cell BUILDS (`construct`): the constructor has no refusal the feasible space does not know;
  * every infeasible request drawn from the WHOLE declared domains (`core.Space.wide`: the narrowing that keeps the
    covering set feasible is ignored) is refused by NAME by the constructor, and a refusal is always a named
    ValueError / NotImplementedError, never a bare crash;
  * every request in a family's declared `refusals()` list is refused by name before anything launches;
  * an INERT option (one with no meaning in the cell, which a caller may still carry) changes nothing: the call with the
    stray value has the outcome of the call without it, through every public entry that takes the option -- bitwise equal
    outputs, or the same refusal text.  This is the property that breaks when a valid call is wrongly refused;
  * the family's fallback guard re-raises a device error (out of memory, accelerator error) instead of letting a fallback
    site cache it as "this variant does not compile", and passes compile / resource verdicts.

CPU where the refusal is decided before the launch; the inert-option check runs on the GPU when there is one.
"""
import random
import warnings

PROFILE_SELF_SIZED = True        # tests/_profile.py: the module sizes its own cells

import pytest
import torch

import _contract
from _contract import core

FAMILIES = _contract.load_all()
NAMED = (ValueError, NotImplementedError)
WIDE_DRAWS = 400


def _profile(config):
    from _profile import resolve
    return resolve(config)


def _cells(name, config):
    """Recipes (smoke) or recipes + the pair-covering cells (compact / full) of a family."""
    strength = {"smoke": 0, "user": 2, "compact": 2, "full": 3}[_profile(config)]
    return core.cells(FAMILIES[name], strength=strength)[0]


def _with(hook):
    return sorted(n for n, ad in FAMILIES.items() if hasattr(ad, hook))


@pytest.mark.parametrize("family", _with("construct"))
def test_every_infeasible_request_is_refused_by_name(family):
    """contract: a request the space declares infeasible at the constructor is refused by the constructor, and every
    constructor refusal, declared or not, is a named ValueError / NotImplementedError (never a bare crash)."""
    ad = FAMILIES[family]
    rng = random.Random(20261018)
    declared = accepted = 0
    unnamed, missed = [], []
    for _ in range(WIDE_DRAWS):
        c = ad.space.wide(rng)
        why = ad.space.why_infeasible(c)
        try:
            ad.construct(c)
            err = None
        except NAMED as e:
            err = e
        except Exception as e:                      # noqa: BLE001
            unnamed.append(f"{type(e).__name__}: {str(e)[:120]}")
            continue
        if why is not None and why.startswith(tuple(ad.constructor_reasons)):
            declared += 1
            if err is None:
                missed.append(why[:140])
        accepted += err is None
    assert not unnamed, "constructor crash(es) that are not a named refusal:\n  " + "\n  ".join(unnamed[:8])
    assert not missed, "infeasible request(s) the constructor accepted:\n  " + "\n  ".join(missed[:8])
    assert declared and accepted, (f"the wide draw never reached both a declared refusal ({declared}) and a served "
                                   f"request ({accepted}): the check would be vacuous")


@pytest.mark.parametrize("family", _with("refusals"))
def test_declared_refusals_are_raised_by_name_before_launch(family):
    """contract: each request in the family's `refusals()` list is refused with the named error, on CPU tensors (the
    refusal is decided before any launch)."""
    ad = FAMILIES[family]
    rows = ad.refusals()
    assert rows
    for label, fixed, mutate, exc, pattern in rows:
        c = ad.space.fill(fixed, lambda n, d: d[0])
        assert c is not None, f"{label}: the fixed factors {fixed} violate the declared constraints"
        args, kw = ad.build(c, device="cpu")
        args, kw = mutate(args, kw)
        with pytest.raises(exc, match=pattern):
            ad.call(args, kw)


def _guards():
    return [(name, g) for name, ad in sorted(FAMILIES.items()) for g in getattr(ad, "fatal_guards", ())]


@pytest.mark.parametrize("family,guard", _guards())
def test_a_fatal_device_error_is_never_swallowed_by_the_fallback_guard(family, guard):
    """contract: the family's fallback guard re-raises an out-of-memory or accelerator error (never cached as "this variant
    does not compile") and lets compile / resource / lowering verdicts pass."""
    import importlib
    mod, fn = guard.split(":")
    reraise = getattr(importlib.import_module(mod), fn)
    for exc in (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
                *([torch.AcceleratorError("illegal memory access")] if hasattr(torch, "AcceleratorError") else [])):
        with pytest.raises(type(exc)):
            reraise(exc)
    from triton.runtime.errors import OutOfResources
    for verdict in (ValueError("not lowerable"), RuntimeError("compilation failed"),
                    OutOfResources(232448, 101376, "shared memory")):
        assert reraise(verdict) is None, f"{guard} re-raised a compile / resource verdict {verdict!r}"


def _outcome(thunk, args, kw):
    try:
        with torch.no_grad():
            return ("ok", [v.detach().clone().cpu() for v in thunk(args, kw)])
    except NAMED as e:
        return ("refused", type(e).__name__, str(e))


@pytest.mark.parametrize("family", _with("inert_options"))
def test_an_inert_option_leaves_the_outcome_unchanged(family, request):
    """contract: an option with no meaning in the cell, carried by the caller anyway, changes nothing: through every
    public entry the outcome is bitwise the same outputs or the same refusal text as without it."""
    ad = FAMILIES[family]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = [cell for cell in _cells(family, request.config) if ad.inert_options(cell.c)]
    assert cells, "no cell of the family carries an inert option: the check would be vacuous"
    cells = [c for c in cells if c.kind == "recipe"] + [c for c in cells if c.kind != "recipe"][:12]
    bad, runs = [], 0
    warnings.simplefilter("ignore", RuntimeWarning)         # e.g. "narrow rotating rows run the slow schedule"
    for cell in cells:
        args, kw = ad.build(cell.c, device=device)
        for label, thunk in (ad.entries(cell.c, args, kw) if hasattr(ad, "entries") else [("call", ad.call)]):
            base = _outcome(thunk, args, kw)
            for name, values in ad.inert_options(cell.c).items():
                for value in values:
                    a2, k2 = ad.with_inert(cell.c, args, kw, name, value)
                    got = _outcome(thunk, a2, k2)
                    runs += 1
                    same = (got[0] == base[0] == "ok" and len(got[1]) == len(base[1])
                            and all(torch.equal(x, y) for x, y in zip(got[1], base[1]))) \
                        or (got[0] == base[0] == "refused" and got[1:] == base[1:])
                    if not same:
                        bad.append(f"{cell.id} via {label}: {name}={value!r}: {got[0]} "
                                   f"{got[2][:100] if got[0] == 'refused' else ''}  (without it: {base[0]} "
                                   f"{base[2][:60] if base[0] == 'refused' else ''})")
    assert runs, "no inert-option run happened"
    assert not bad, f"{len(bad)} inert option(s) changed the outcome:\n  " + "\n  ".join(bad[:10])


@pytest.mark.parametrize("family", _with("construct"))
def test_every_feasible_cell_builds(family, request):
    """contract: the constructor accepts every cell the space declares feasible (no false refusal)."""
    ad = FAMILIES[family]
    bad = []
    for cell in _cells(family, request.config):
        try:
            ad.construct(cell.c)
        except Exception as e:                      # noqa: BLE001 -- the report names the refusal
            bad.append(f"{cell.id}: {type(e).__name__}: {str(e)[:160]}")
    assert not bad, f"{len(bad)} feasible cell(s) refused at construction:\n  " + "\n  ".join(bad[:10])
