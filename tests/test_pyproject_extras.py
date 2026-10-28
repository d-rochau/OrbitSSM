"""Pins: pyproject declares a `test` extra that
covers what the suite imports from third parties (pytest, PyYAML, numpy,
pytest-timeout) and a `baselines` extra for the official reference
implementations; the pytest import root includes the repo root.
`test` is the ONE test extra (the old `dev = ["pytest"]` is gone, no alias) and
the installer's runtime set installs all of it.
Version bounds of torch / Triton / Python are a policy choice.  CPU only.
Resolution was checked with `uv pip install --dry-run --no-build-isolation
".[test]"` on the t38 stack (resolved; only pytest-timeout missing).
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _names(reqs):
    import re
    return {re.split(r"[<>=!~\[; ]", r, 1)[0].lower() for r in reqs}


def test_test_extra_covers_the_suite_requirements():
    """contract: the `test` extra names the suite's third-party requirements."""
    extras = PROJECT["project"]["optional-dependencies"]
    assert {"pytest", "pyyaml", "numpy", "pytest-timeout"} <= _names(extras["test"])


def test_baselines_extra_names_the_reference_implementations():
    """contract: the `baselines` extra names the official reference implementations."""
    extras = PROJECT["project"]["optional-dependencies"]
    assert {"mamba-ssm", "causal-conv1d", "lightning", "torchmetrics"} <= _names(extras["baselines"])


def test_pytest_import_root_includes_the_repo_root():
    """contract: the pytest import root includes the repo root."""
    assert PROJECT["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src", "."]


