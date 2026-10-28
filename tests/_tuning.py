"""Test TUNING MODES: how many launch configs a kernel call may try.

The tuning mode is orthogonal to the cell profile (`tests/_profile.py`): the
profile decides WHICH cells run, the tuning mode decides how each launch picks
its config.

    calibrated  (DEFAULT) every kernel launch uses exactly ONE config: the
                GPU's calibrated optimum for the call's shape class when a
                valid calibration exists for this GPU / Triton / kernel code,
                otherwise the library's deterministic fixed fallback config.
                No autotune search anywhere, hence no per-candidate compiles.
                On a GPU without a valid calibration the run says so (header
                and gate verdict) and uses the fixed fallbacks.
    light       the production light tier: every candidate of the calibrated
                light sets (generic light grids on an uncalibrated GPU) is
                run, to find config-specific defects.  A larger, rarer run.
    full        the full candidate grids (rare).

Selected with ``pytest --tuning {calibrated,light,full}`` or the environment
variable ``HYBRIDSSM_TEST_TUNING`` (the option wins).  The kernels read their
tier at import (the decorator grids are built then), so the mode must be known
before the first kernel module is imported: `tests/conftest.py` applies it at
its own import, from `sys.argv` and the environment.  An explicit setting of a
kernel variable by the caller (``SSD_MIMO_AUTOTUNE=...``) always wins over the
mode's pin.

Tests whose subject is the tuner itself (autotune selection, replay, candidate
sets, the subprocess-isolated autotune cells) keep their own explicit tiers
whatever the mode is.

This module must stay importable on its own (stdlib only at import): the fast
runner loads it for trees and reports from it.
"""
import os
import sys

MODES = ("calibrated", "light", "full")
DEFAULT_MODE = "calibrated"
ENV = "HYBRIDSSM_TEST_TUNING"

#: kernel variable -> value, per mode.  `*_CALIBRATION=pinned` is the library's
#: "single calibrated fixed config of the call's group, no search anywhere"
#: (src/orbit_ssm/kernels/calibration.py `mode`), with its generic single pin
#: where the GPU has no valid file.  The BWD_* axis variables are NOT pinned
#: in any mode: a pin there would replace the calibrated config by one the
#: library does not serve.
PINS = {
    "calibrated": {
        "SSD_MIMO_CALIBRATION": "pinned",
        "SSD_MIMO_AUTOTUNE": "pinned",
        "HYBRID_MH_AUTOTUNE": "pinned",
        "HYBRID_MH_CALIBRATION": "on",      # the per-cell P = 1 table is served as a pin
    },
    "light": {
        "SSD_MIMO_CALIBRATION": "on",
        "SSD_MIMO_AUTOTUNE": "light",
        "SSD_MIMO_SISO_AUTOTUNE": "light",
        "HYBRID_MH_AUTOTUNE": "light",
        "HYBRID_MH_CALIBRATION": "off",     # the runtime tuner must see every cell (the table would pin P = 1)
    },
    "full": {
        "SSD_MIMO_CALIBRATION": "on",
        "SSD_MIMO_AUTOTUNE": "full",
        "SSD_MIMO_SISO_AUTOTUNE": "full",
        "HYBRID_MH_AUTOTUNE": "full",
        "HYBRID_MH_CALIBRATION": "off",
    },
}

#: the calibration families (kernels/calibration.py registry names) and the
#: modules that register them
FAMILIES = ("ssd_mimo", "ssd_mimo_fused", "ssd_mimo_chunkpar", "ssd_mimo_levers",
            "ssd_mimo_route", "mh")


def _argv_value(argv):
    for i, a in enumerate(argv):
        if a == "--tuning" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--tuning="):
            return a.split("=", 1)[1]
    return None


def resolve(config=None, argv=None):
    """The active mode: --tuning, then $HYBRIDSSM_TEST_TUNING, then the default.
    `config` (a pytest Config) is consulted first when given; `argv` defaults
    to sys.argv (the conftest import happens before pytest parsed options)."""
    val = None
    if config is not None:
        try:
            val = config.getoption("hybridssm_tuning")
        except (ValueError, AttributeError):
            val = None
    if val is None:
        val = _argv_value(sys.argv[1:] if argv is None else argv)
    val = (val or os.environ.get(ENV) or DEFAULT_MODE).strip().lower()
    if val not in MODES:
        raise ValueError(f"unknown test tuning mode {val!r}; one of {MODES}")
    return val


def pins(mode):
    return dict(PINS[mode])


#: calibrated mode: kernel name -> launches whose candidate list still held several configs after the
#: library's calibration pruning (the first was used, nothing was timed)
TRUNCATED = {}
_SINGLE = {"installed": False, "on": False}


def enforce_single_config(on=True):
    """Calibrated mode's last guarantee: ONE config per launch.

    `SSD_MIMO_CALIBRATION=pinned` narrows every calibrated kernel to its group's fixed config, but a kernel
    without a calibrated entry, a group-pruning that is overridden by a per-device verdict, or a kernel that
    carries no prune hook keeps several candidates, and Triton would time them.  While this is on, an
    orbit_ssm kernel's autotuner that is left with more than one candidate after its own pruning uses the
    FIRST (deterministic: the grid order) and times nothing; `TRUNCATED` counts these launches per kernel and the
    run summary lists them.  Autotuners of other packages (the upstream reference kernels) and of the tests
    themselves are untouched.  `on=False` switches it off (the tests whose subject is the tuner)."""
    _SINGLE["on"] = bool(on)
    if _SINGLE["installed"]:
        return
    try:
        import triton.runtime.autotuner as at
    except Exception:                                              # noqa: BLE001  (no triton: nothing to enforce)
        return
    orig = at.Autotuner.prune_configs
    orig_run = at.Autotuner.run

    def _ours(self):
        base = getattr(self, "base_fn", None)
        return (getattr(base, "__module__", "") or "").startswith("orbit_ssm"), base

    def prune_configs(self, kwargs):
        res = orig(self, kwargs)
        if _SINGLE["on"] and len(res) > 1:
            ours, base = _ours(self)
            if ours:
                failed = getattr(self, "_single_failed", [])
                left = [c for c in res if c not in failed] or res
                TRUNCATED[base.__name__] = TRUNCATED.get(base.__name__, 0) + 1
                self._single_alts = left[1:]
                return left[:1]
        return res

    def run(self, *args, **kwargs):
        # a timed search skips a candidate whose launch does not fit (`OutOfResources`: shared memory, registers);
        # the single-config launch must not turn that into a failure the search mode never shows, so it falls to
        # the next candidate of the grid order (only on that error; the working config is cached as usual)
        if not (_SINGLE["on"] and _ours(self)[0]):
            return orig_run(self, *args, **kwargs)
        from triton.runtime.errors import OutOfResources
        self._single_failed = []
        try:
            while True:
                self._single_alts = []
                try:
                    return orig_run(self, *args, **kwargs)
                except OutOfResources:
                    if not self._single_alts:
                        raise
                    self._single_failed.append(self.best_config)
                    for k in [k for k, v in self.cache.items() if v is self.best_config]:
                        del self.cache[k]
        finally:
            self._single_failed = []

    at.Autotuner.prune_configs = prune_configs
    at.Autotuner.run = run
    _SINGLE["installed"] = True


def calibration_report():
    """(status, lines): status is "calibrated" | "partial" | "uncalibrated" |
    "no-gpu" for THIS device, lines say per family which file serves it.
    Needs torch + the package; run it where the kernels run."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "no-gpu", ["no CUDA device: nothing to calibrate against"]
        from orbit_ssm.kernels import calibration as cal
        from orbit_ssm.kernels.hybrid_mamba_mh import calibration_mh      # noqa: F401
        from orbit_ssm.kernels.ssd_mimo import (calibration_chunkpar, calibration_fused,  # noqa: F401
                                                 calibration_ssd, levers, rot_route)
        dev = cal.device_info()
    except Exception as exc:                                  # noqa: BLE001
        return "unknown", [f"calibration status unavailable: {exc!r}"]
    lines, valid = [f"{dev['gpu']} (triton {dev['triton']})"], 0
    for fam in FAMILIES:
        try:
            data, layer = cal.load(fam, dev["gpu"], dev["triton"])
        except Exception as exc:                              # noqa: BLE001
            data, layer = None, f"error {exc!r}"
        if data is None:
            why = "; ".join(cal._STALE.get((fam, dev["gpu"])) or ["no file"])
            lines.append(f"  {fam}: NO valid calibration ({why})")
        else:
            valid += 1
            lines.append(f"  {fam}: {layer} file")
    status = "calibrated" if valid == len(FAMILIES) else "uncalibrated" if not valid else "partial"
    return status, lines


def summary(mode, status):
    """One verdict/header phrase."""
    if mode != "calibrated":
        return f"tuning {mode} (candidate search)" + (
            "" if status in ("calibrated", "no-gpu") else f", GPU calibration {status}")
    if status == "calibrated":
        return "tuning calibrated (one config per launch, GPU calibrated)"
    if status == "no-gpu":
        return "tuning calibrated (no GPU)"
    return (f"tuning calibrated, GPU calibration {status.upper()}: the families without a valid "
            "file run their fixed fallback config")


if __name__ == "__main__":
    # `python tests/_tuning.py`: the status line of this GPU
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [os.path.join(root, "src"), root]
    m = resolve()
    st, ls = calibration_report()
    print(f"TUNING {m} {st}")
    for ln in ls:
        print(ln)
