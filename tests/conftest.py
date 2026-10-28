"""Shared fixtures and helpers for the HybridSSM test suite."""

import os
import re

import pytest
import torch

# `pythonpath = ["src"]` (pyproject) only reaches this process; tests that spawn
# `sys.executable` children need the same import root.  Layout-agnostic: a missing
# directory on PYTHONPATH is harmless (flat layout).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_REPO, os.path.join(_REPO, "src"), os.environ.get("PYTHONPATH", "")) if p)
# a helper directory of the test tree next to this file (when the checkout has one) is importable by top-level tests too; without this they only
# collected when that directory happened to be collected first
import sys as _sys  # noqa: E402

_HELPERS = os.path.join(_REPO, "tests", "research")
if os.path.isdir(_HELPERS):
    _sys.path.insert(0, _HELPERS)

# --production-defaults arm: with HYBRIDSSM_PRODUCTION_DEFAULTS=1,
# NONE of the env pins below is applied, so the suite runs the shipped defaults
# (autotune tiers, launch configs, validation OFF) instead of the gate's pinned
# environment.  `SUITE_PINS` records every pin either way (report header).
PRODUCTION_DEFAULTS = os.environ.get("HYBRIDSSM_PRODUCTION_DEFAULTS", "") not in ("", "0")
SUITE_PINS: dict = {}


def _pin(name, value):
    SUITE_PINS[name] = value
    if not PRODUCTION_DEFAULTS:
        os.environ.setdefault(name, value)


# TUNING MODE (tests/_tuning.py; `--tuning` / HYBRIDSSM_TEST_TUNING): how a launch picks its config.
#   calibrated (default)  ONE config per launch: the GPU's calibrated optimum for the shape class
#                         (SSD_MIMO_CALIBRATION=pinned), else the library's fixed fallback; no search
#   light / full          the production light tier / the full grids: every candidate is exercised
# The kernels build their decorator grids when `orbit_ssm.kernels` is imported, so the mode is applied
# HERE, at conftest import, from sys.argv + the environment (pytest has not parsed its options yet).
# An explicit setting of the variable by the caller wins over the mode's pin, and
# HYBRIDSSM_PRODUCTION_DEFAULTS=1 applies none of it.  The SSD backward's axis variables
# (SSD_MIMO_BWD_WARPS / STAGES / HB_CANDS) are NOT pinned in any mode: a pin there would replace the
# config the library serves by one it does not.
import _tuning  # noqa: E402

TUNING = _tuning.resolve()
for _name, _value in _tuning.pins(TUNING).items():
    _pin(_name, _value)
# an explicit caller tier wins over the mode: the selection tests below restore a search only where this
# file (not the caller) chose the tier
_SUITE_SETS_AUTOTUNE = not PRODUCTION_DEFAULTS and SUITE_PINS.get("SSD_MIMO_AUTOTUNE") == os.environ.get("SSD_MIMO_AUTOTUNE")
_SUITE_SETS_MH_AUTOTUNE = not PRODUCTION_DEFAULTS and SUITE_PINS.get("HYBRID_MH_AUTOTUNE") == os.environ.get("HYBRID_MH_AUTOTUNE")
_SUITE_SETS_CALIBRATION = not PRODUCTION_DEFAULTS and SUITE_PINS.get("SSD_MIMO_CALIBRATION") == os.environ.get("SSD_MIMO_CALIBRATION")

# calibrated mode: ONE config per launch also where the library's calibration pruning leaves several
# (tests/_tuning.py `enforce_single_config`); off when the caller overrides a pin of the mode
_ENFORCE_SINGLE = (TUNING == "calibrated" and not PRODUCTION_DEFAULTS
                   and all(os.environ.get(k) == v for k, v in _tuning.pins(TUNING).items()))
if _ENFORCE_SINGLE:
    _tuning.enforce_single_config(True)

#: name fragments of tests whose assertion is ABOUT selection/measurement
#: (chunk-size trial, smem strategy trial, forward-config measure, the
#: backward autotuners' atomics); they need the timed searches enabled.
_SELECTION_TESTS = ("autotune", "measure", "strategy")


@pytest.fixture(autouse=True)
def _autotune_tier_for_selection(request, monkeypatch):
    # tests whose subject is the tuner keep their own explicit tier, whatever the tuning mode: the
    # calibrated mode (one config, no search) is relaxed to the light search for them; light/full
    # already search
    selection = TUNING == "calibrated" and any(f in request.node.name for f in _SELECTION_TESTS)
    if selection and _ENFORCE_SINGLE:
        _tuning.enforce_single_config(False)
    if selection:
        if _SUITE_SETS_AUTOTUNE:
            monkeypatch.setenv("SSD_MIMO_AUTOTUNE", "light")
        if _SUITE_SETS_CALIBRATION:
            monkeypatch.setenv("SSD_MIMO_CALIBRATION", "on")      # `pinned` would forbid the search under test
        if _SUITE_SETS_MH_AUTOTUNE:
            monkeypatch.setenv("HYBRID_MH_AUTOTUNE", "light")
    yield
    if selection and _ENFORCE_SINGLE:
        _tuning.enforce_single_config(True)


# The mh family's ABI validation pass (`layout.assert_pair_tied`, reached
# through `mh_scan(validate=...)`).  It is OFF in production because it is a
# device->host sync per call; the suite runs with it ON, so an untied
# modulus on a rotating pair fails loudly here instead of silently defining
# a different linear system that both fp64 oracles reproduce.  Override with
# `HYBRID_SSM_MH_VALIDATE=0`; individual cases pass `validate=` explicitly.
_pin("HYBRID_SSM_MH_VALIDATE", "1")
# Same for the ssd_mimo entry point: refuse rates past fp32's exp range
# (A_LOG_MAX) instead of relying on kernel clamps (chunk K2c).
_pin("SSD_MIMO_VALIDATE", "1")

# ---------------------------------------------------------------------------
# CUDA availability
# ---------------------------------------------------------------------------
# `torch.cuda.is_available()` is True on a GPU this torch build has no kernels
# for (the sm_61 login GPU under the cu130 build: sm_75..sm_120): every CUDA
# cell then crashes or, worse, the suite's skip logic believes it ran.  CUDA
# here means: available AND the device's capability is served by an entry of
# `torch.cuda.get_arch_list()` (sm_XY serves capability X.Z for Z >= Y, the
# rule torch's own warning states; compute_XY PTX serves capability >= XY).

def _cuda_usable():
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
    except Exception as exc:                        # noqa: BLE001
        return False, f"CUDA device query failed: {exc!r}"
    return cuda_arch_served(major, minor, archs)


def cuda_arch_served(major, minor, archs):
    """(usable, reason) for capability major.minor under a torch arch list."""
    for a in archs:
        kind, _, num = a.partition("_")
        if not num.isdigit() or len(num) < 2:
            continue
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True, f"sm_{major}{minor} served by {a}"
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True, f"sm_{major}{minor} served by PTX {a}"
    return False, (f"device sm_{major}{minor} is not in this torch build's arch "
                   f"list {archs}")


CUDA, CUDA_REASON = _cuda_usable()
REQUIRE_GPU_ENV = "HYBRIDSSM_REQUIRE_GPU"

#: a skipif whose reason or string condition names one of these is a device
#: requirement: such a test is a `gpu` test (T12), whatever its module flag is
_GPU_SKIP_WORDS = re.compile(
    r"\bcuda\b|\bgpu\b|\bsm_?\d{2,3}\b|hopper|blackwell|ampere|gluon|"
    r"cuda\.is_available", re.I)


def _skipif_needs_gpu(item):
    # R7: with no CUDA device at all, a skipif whose conditions are all bool False does
    # NOT skip here, so it cannot be a device requirement, whatever its reason mentions
    # (test_orbit_prologue's TRITON_INTERPRET skip names "the CUDA route").  With a device
    # (also an unserved one, where torch.cuda.is_available() is True) the words decide.
    no_device = not torch.cuda.is_available()
    for m in item.iter_markers("skipif"):
        conds = [a for a in m.args if not isinstance(a, str)]
        if "condition" in m.kwargs and not isinstance(m.kwargs["condition"], str):
            conds.append(m.kwargs["condition"])
        if no_device and conds and len(conds) == len(m.args) + ("condition" in m.kwargs) \
                and not any(bool(c) for c in conds):
            continue
        texts = [str(m.kwargs.get("reason", ""))]
        texts += [a for a in m.args if isinstance(a, str)]
        cond = m.kwargs.get("condition")
        if isinstance(cond, str):
            texts.append(cond)
        if any(_GPU_SKIP_WORDS.search(t) for t in texts):
            return True
    return False

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ssd_options():
    """Activate ``kernel_options`` for the test: ``ssd_options(backward_states_fp32=True, ...)`` (family
    "ssd_mimo"; ``family="hybrid_mamba"`` for MH).  Popped at teardown, so a failing test cannot leak a scope."""
    from orbit_ssm import kernel_options as ko
    tokens = []

    def apply(family="ssd_mimo", **options):
        tokens.append(ko.push(family, options))
    yield apply
    for t in reversed(tokens):
        ko.pop(t)


@pytest.fixture
def require_cuda():
    """Skip the test when no CUDA device is available."""
    if not CUDA:
        pytest.skip("CUDA required")


@pytest.fixture(params=["cuda"] if CUDA else [], ids=lambda d: d)
def device(request):
    """Yield a torch.device for every available accelerator."""
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# Standard shape / dtype constants for @pytest.mark.parametrize
# ---------------------------------------------------------------------------
POW2_DIMS = [16, 32, 64, 128, 256]
NONPOW2_DIMS = [17, 31, 47, 63, 127, 255]
PRIME_DIMS = [17, 31, 47, 61, 127]
SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048]
SEQ_LENGTHS_NONPOW2 = [65, 129, 257, 513]
DTYPES_FWD = [torch.float32, torch.bfloat16]
DTYPES_BWD = [torch.float32]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Every tolerance lives in tests/_tolerance.py.  The
# names are re-exported here for the cells that import them from conftest.
# ---------------------------------------------------------------------------
from _tolerance import (  # noqa: E402,F401
    UNIT_ROUNDOFF, DrawPool, NearZeroReference, assert_close_terms,
    assert_dtype_close, assert_rel_close, draw_seeds, is_sixteen_bit,
    mma_format, route_bound, terms_abs, tol16, unit_roundoff)


def assert_kernel_served(mod, claim=""):
    """Pin that the LAST forward on `mod` was served by the CUDA kernel.

    A value/gradient comparison against the fp64 oracle does NOT pin the
    kernel: the chunked torch route is one of the duals the oracle is
    validated against, so a call that fell off the kernel still MATCHES
    and the cell still goes green.  `SSDMimoSSM.last_route()` returns the
    NAME of the refusal that moved the last forward off its requested
    route and `None` when nothing was refused, so this is the assertion
    that the cell measured what it claims to measure.

    Use ONLY where the cell's claim is about the kernel.  A cell that
    deliberately exercises a refusal, a non-CUDA backend or a config the
    kernel does not serve must NOT call this -- there the fallback is the
    statement, not a defect (an R1O1 call outside the kernel envelope demotes
    to the torch route instead of raising).
    """
    assert mod.backend in ("cuda", "auto"), (
        f"assert_kernel_served on backend={mod.backend!r}: this cell does "
        "not request the CUDA route, so the route says nothing")
    route = mod.last_route()
    assert route is None, (
        f"{claim or 'the cell claims the CUDA kernel computed this'}, but "
        f"the call fell off the kernel route: {route}")


# ---------------------------------------------------------------------------
# Test profiles: `smoke` / `compact` (default) / `full`
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu", dest="hybridssm_require_gpu", action="store_true",
        default=False,
        help="error (instead of skipping every gpu cell) when no usable CUDA "
             "device is present; env: HYBRIDSSM_REQUIRE_GPU=1")
    parser.addoption(
        "--profile", dest="hybridssm_profile", default=None,
        choices=("smoke", "user", "compact", "full"),
        help="cell profile: 'user' (the default of a tree without the deep-suite extension: 'smoke' thinning, the contract suite over "
             "its base spaces), 'smoke' (one minimal cell per test function), "
             "'compact' (default; R,O in {1,2,4} + (1,4) + one rectangular "
             "pair, N in {32,128}, interleaved pairs) or 'full' (every "
             "parametrised cell; nightly / pre-release).  "
             "Env: HYBRIDSSM_TEST_PROFILE.  Rules: tests/_profile.py")
    parser.addoption(
        "--tuning", dest="hybridssm_tuning", default=None, choices=_tuning.MODES,
        help="launch-config tuning: 'calibrated' (default; one config per launch, the GPU's "
             "calibrated optimum or the fixed fallback), 'light' (every candidate of the light "
             "sets) or 'full' (the full grids).  Env: HYBRIDSSM_TEST_TUNING.  tests/_tuning.py")
    parser.addoption(
        "--family", dest="hybridssm_family", default=None,
        help="comma list of kernel families for the contract suite "
             "(tests/test_contract.py; e.g. 'mh' or 'mh,ssd'); default all "
             "registered adapters.  Env: HYBRIDSSM_FAMILIES.")


    parser.addoption(
        "--kernel-target", dest="hybridssm_kernel_target", default=None, choices=("auto", "portable"),
        help="force the process's kernel target (orbit_ssm.kernels.portable): 'portable' makes every 'auto' layer run the "
             "plain-Triton path (no Gluon, no inline asm, no tf32 literal) on this NVIDIA GPU -- the correctness check "
             "of the path a non-NVIDIA Triton backend takes.  Default: the shipped auto path.")


def _require_gpu(config):
    return bool(config.getoption("hybridssm_require_gpu", False)) or \
        os.environ.get(REQUIRE_GPU_ENV, "") not in ("", "0")


def pytest_configure(config):
    kt = config.getoption("hybridssm_kernel_target", None)
    if kt == "portable":
        from orbit_ssm.kernels import portable
        portable.force_process_target("portable")        # before any kernel compiles (libdevice trig latch)
    if _require_gpu(config) and not CUDA:
        raise pytest.UsageError(
            f"--require-gpu / {REQUIRE_GPU_ENV}=1: no usable CUDA device "
            f"({CUDA_REASON}); the gpu cells would all skip")


def pytest_report_header(config):
    from _profile import resolve
    status, lines = _tuning.calibration_report() if CUDA else ("no-gpu", [])
    return [f"hybridssm test profile: {resolve(config)} (tests/_profile.py)",
            f"hybridssm {_tuning.summary(_tuning.resolve(config), status)}"
            + ("".join("\n" + ln for ln in lines) if status not in ("no-gpu",) else ""),
            f"hybridssm CUDA usable: {CUDA} ({CUDA_REASON})",
            "hybridssm env pins: " + (f"OFF (production defaults; {len(SUITE_PINS)} pins not "
                                      "applied)" if PRODUCTION_DEFAULTS else
                                      f"{len(SUITE_PINS)} applied (unless set by the caller)")]


def pytest_terminal_summary(terminalreporter):
    if _tuning.TRUNCATED:
        terminalreporter.write_line(
            "hybridssm tuning calibrated: kernels that still had several candidates after the calibration "
            "pruning (the first was used, nothing was timed): "
            + ", ".join(f"{k} x{v}" for k, v in sorted(_tuning.TRUNCATED.items())))


# ---------------------------------------------------------------------------
# Auto-marker hook: tag tests that touch CUDA with the "gpu" marker
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Classify actual device requirements, not words in a test's name."""
    paper1_files = {"test_state_lifecycle.py", "test_ssm_contract.py",
                    "test_ssd_endpoint.py", "test_mh_reference.py", "test_mh_endpoint.py",
                    "test_mh_rotation_law.py", "test_mh_p1_rotation.py",
                    "test_physical_law_witness.py", "test_oracle_structure_audit.py"}
    for item in items:
        filename = item.path.name
        params = getattr(getattr(item, 'callspec', None), 'params', {})
        device_case = any(str(params.get(key, '')).split(':')[0] == 'cuda'
                          for key in ('backend', 'device', 'dev'))
        if device_case or "require_cuda" in getattr(item, "fixturenames", ()) \
                or _skipif_needs_gpu(item):
            item.add_marker(pytest.mark.gpu)
        if not CUDA and item.get_closest_marker("gpu") is not None:
            item.add_marker(pytest.mark.skip(reason=f"CUDA required ({CUDA_REASON})"))
        if filename.startswith(('test_bench',)):
            item.add_marker(pytest.mark.benchmark)
        if filename in paper1_files:
            item.add_marker(pytest.mark.paper1)
        if filename in ('test_state_lifecycle.py', 'test_ssm_contract.py'):
            item.add_marker(pytest.mark.contract)
    # Tier markers (tier1/tier2/tier3, `-m tier1` is the essential gate) of the deep suite's legacy kernel files: the
    # classifier ships with those files only; without it nothing is tiered.
    try:
        from _tiers import apply_tier_markers
    except ImportError:
        pass
    else:
        apply_tier_markers(items)
    # The compact/full profile: a deselection by each cell's own
    # parametrisation (R, O, N, pair_layout); `--profile full` keeps all.
    from _profile import apply_profile
    apply_profile(config, items)
    # the calibrated tuning mode runs every launch on its one config; the contract suite's
    # production-search slice (the cell in the DEFAULT launch mode) is what light/full run
    if _tuning.resolve(config) == "calibrated" and not os.environ.get("FG_SELECT"):
        gone = [it for it in items if it.path.name == "test_contract.py"
                and it.originalname == "test_production_mode"]
        if gone:
            items[:] = [it for it in items if it not in gone]
            config.hook.pytest_deselected(items=gone)
    _order_cheap_first(items)


# ---------------------------------------------------------------------------
# Run order: cheap checks and specific regression tests before the heavy oracle cells
# ---------------------------------------------------------------------------
# A run under a time cap (a merge gate with a budget, a bisection, the defect corpus) stops where the clock says, so what
# runs FIRST is what a capped run can still tell.  Pytest's default order is the file order, which put every oracle cell of
# every family (`test_contract.py::test_oracle`: minutes of compilation, the covering guarantee) ahead of the contract
# adapters' specific regressions and the CPU checks.  Stable sort by class of cost:
#   0  the contract suite's own CPU checks: static, refusal, core, the adapters' CPU tests (seconds each)
#   1  device tests of the adapters' own files and of the gate's specific-regression list: the specific regressions
#   2  the torch-route cells (CPU, hundreds of them) and the generic checks of the kernel suite other than the oracle
#      (sliced to a subset of the cells)
#   3  the kernel suite's oracle check, the heavy covering guarantee
#   4  every other test, in its own order
# Inside a class the order is the original one.  HYBRIDSSM_RUN_ORDER=file restores the file order.

def _specific_files():
    """The specific-regression files of the merge gate (the `GATE_*.txt` list next to this file; absent in a tree without
    a gate)."""
    import glob
    out = set()
    for path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GATE_*.txt")):
        with open(path) as fh:
            out |= {os.path.basename(ln.strip()) for ln in fh if ln.strip() and not ln.startswith("#")}
    return out


def _cost_class(item, specific):
    name = item.path.name
    if name == "test_contract.py" or item.get_closest_marker("contract_cells") is not None:
        return 3 if getattr(item, "originalname", item.name) == "test_oracle" else 2
    if name == "test_contract_torch.py":
        return 2                                # the torch-route cells: CPU, but seconds per cell and hundreds of them
    if name.startswith("test_contract_") or name in specific:
        # the adapters' own files (test_contract_<family>*.py), the static / refusal / core checks, the gate's list
        return 1 if item.get_closest_marker("gpu") is not None else 0
    return 4


def _order_cheap_first(items):
    if os.environ.get("HYBRIDSSM_RUN_ORDER", "") == "file":
        return
    specific = _specific_files()
    items.sort(key=lambda it: _cost_class(it, specific))        # stable


# ---------------------------------------------------------------------------
# torch.compile on CPU needs a C++20 compiler: skip those cells (not fail) where it is missing
# ---------------------------------------------------------------------------
from _cxx20 import pytest_runtest_makereport  # noqa: E402,F401  (hook; see tests/_cxx20.py)
