"""The contract suite on the torch route (CPU, no GPU needed).

Per family with a torch-route adapter (`torch_cells` / `torch_layer`), over a small pair-covering set of the options
that are visible on `backend="torch"` x dtype in (fp32, fp64, bf16, amp): the layer against the family's independent fp64
oracle (forward + every gradient), split call == whole call with the complete State and the gradients through the
carries, the single-token step loop == the parallel forward, the State dtype rules.  The checks are generic
(tests/_contract/torch_route.py); a family contributes only its cells, its layer and its oracle.
"""
import os

import pytest

import _contract
from _contract import torch_route as TR

FAMILIES = _contract.load_all()


def _selected(config):
    """`--family a,b` (env HYBRIDSSM_FAMILIES) as in tests/test_contract.py; a family without torch-route hooks is skipped."""
    v = config.getoption("hybridssm_family", None) or os.environ.get("HYBRIDSSM_FAMILIES")
    names = [x.strip() for x in v.split(",")] if v else sorted(FAMILIES)
    return [n for n in names if n in FAMILIES and hasattr(FAMILIES[n], "torch_cells")]


def pytest_generate_tests(metafunc):
    if "cell" not in metafunc.fixturenames:
        return
    layer_only = metafunc.function.__name__ != "test_torch_oracle"     # a `scan` cell has no State / step surface
    params = []
    for fam in _selected(metafunc.config):
        for i, c in enumerate(FAMILIES[fam].torch_cells()):
            if layer_only and c.kind == "scan":
                continue
            # short ids (the full option list is in every assertion message through `cell.id`)
            params.append(pytest.param(c, id=f"{fam}:{c.name}" if c.kind == "regression" else f"{fam}:{c.c['dtype']}:{i:02d}"))
    metafunc.parametrize("cell", params)


def _dtype(cell):
    return cell.c["dtype"]


def _known(request, cell, check):
    """A DIAGNOSED product failure, declared by the adapter (`torch_known_failure(cfg, check) -> reason | None`), is a strict
    xfail: it is listed with its reason, and it turns red the day it stops failing (remove the declaration).  A reason
    starting with "borderline" declares a seed-dependent edge of a policy bound instead (non-strict)."""
    why = getattr(FAMILIES[cell.family], "torch_known_failure", lambda c, k: None)(cell.c, check)
    if why:
        # "borderline:..." = a cell at the edge of a policy bound whose verdict flips with the draw seeds (an open question,
        # not a diagnosed defect): a non-strict xfail.  Every other declaration is strict.
        request.applymarker(pytest.mark.xfail(reason=why, strict=not why.startswith("borderline"),
                                              raises=(RuntimeError, AssertionError)))


def test_torch_oracle(request, cell):
    """forward + every gradient against the family's independent fp64 oracle: of the layer, or (a `scan` cell) of the torch
    scan function."""
    _known(request, cell, "oracle")
    ad = FAMILIES[cell.family]
    if cell.kind == "scan":
        ad.torch_scan_check(cell)
    else:
        TR.check_oracle(ad, cell, _dtype(cell))


def test_torch_continuation(request, cell):
    """two calls joined by the returned State == one call: output, every carry, gradients through the carries."""
    _known(request, cell, "continuation")
    TR.check_continuation(FAMILIES[cell.family], cell, _dtype(cell))


def test_torch_step_equals_forward(request, cell):
    """the single-token step loop == the parallel forward, from the default and from an arbitrary State."""
    _known(request, cell, "step")
    TR.check_step(FAMILIES[cell.family], cell, _dtype(cell))


def test_torch_default_carries(cell):
    """init_state() carries are promote(parameter dtype, fp32); a forward returns the same dtypes."""
    TR.check_default_carries(FAMILIES[cell.family], cell, _dtype(cell))
