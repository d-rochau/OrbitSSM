"""Pins: every script in examples/ runs to completion on CPU in a fresh
interpreter against this tree.

Each example asserts its own claims (finite gradients; for lm_roundtrip:
stream == full and step == full within its stated fp32 bound, recipe +
state_dict reload bit-identical, generate shape); this test pins that they
still run as the API moves.  CUDA is hidden: the examples are CPU examples.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
EXPECT = {"quickstart.py": "CPU forward/backward passed",
          "lm_roundtrip.py": "LM round trip passed"}


def test_every_example_has_an_expected_success_line():
    """contract: every examples/*.py script is registered with its success line here."""
    assert {p.name for p in EXAMPLES} == set(EXPECT), "add the new example to EXPECT"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs_on_cpu(script):
    """contract: the example runs to completion on CPU in a fresh interpreter against this tree."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(ROOT / "src")
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert EXPECT[script.name] in r.stdout, r.stdout[-2000:]
