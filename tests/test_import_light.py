"""Pins the import surface of `import orbit_ssm` and the lazy third-party
gated-norm route.  CPU only.  Since the fused gated norm is our own
kernel, mamba_ssm is resolved only for the explicit ``norm_kernel='mamba_ssm'``.

  * `import orbit_ssm` imports neither mamba_ssm nor transformers (they cost
    ~4.5 s of every import before; measured 7.4 s -> 4.9 s wall on the login
    node, the rest is torch itself and torch._dynamo via compile_compat);
  * the mamba_ssm gated RMSNorm is resolved on the first explicit use that can
    take it, and ONLY ImportError means "not installed": an installed but
    broken package raises instead of silently changing the route;
  * the private `_gated_rms_norm_route` names the route (CPU: "formula").
Whether the third-party route stays in production is a policy choice.
"""
import importlib.abc
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from orbit_ssm.layers import norms

ROOT = Path(__file__).resolve().parents[1]


def test_import_orbit_ssm_does_not_import_mamba_ssm_or_transformers():
    """envelope: `import orbit_ssm` imports neither mamba_ssm nor transformers."""
    code = ("import sys, orbit_ssm; "
            "print(sorted(m for m in ('mamba_ssm', 'transformers') if m in sys.modules))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "[]", r.stdout


@pytest.fixture
def unresolved(monkeypatch):
    monkeypatch.setattr(norms, "_rmsnorm_gated_fn", norms._UNRESOLVED)
    yield


class _Raise(importlib.abc.MetaPathFinder):
    def __init__(self, exc):
        self.exc = exc

    def find_spec(self, name, path=None, target=None):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            raise self.exc
        return None


def _hide_mamba(monkeypatch, exc):
    for m in [m for m in sys.modules if m == "mamba_ssm" or m.startswith("mamba_ssm.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, "meta_path", [_Raise(exc)] + sys.meta_path)


def test_not_installed_is_the_torch_route(unresolved, monkeypatch):
    """envelope: an absent mamba_ssm (ImportError) resolves the gated norm to the torch route."""
    _hide_mamba(monkeypatch, ModuleNotFoundError("No module named 'mamba_ssm'"))
    assert norms._resolve_mamba_gated() is None
    assert norms._rmsnorm_gated_fn is None


def test_a_broken_install_raises_instead_of_changing_the_route(unresolved, monkeypatch):
    """envelope: a broken mamba_ssm install raises instead of silently changing the norm route, and is not cached."""
    _hide_mamba(monkeypatch, RuntimeError("triton ABI mismatch"))
    with pytest.raises(RuntimeError, match="triton ABI mismatch"):
        norms._resolve_mamba_gated()
    assert norms._rmsnorm_gated_fn is norms._UNRESOLVED     # not cached as "absent"


def test_cpu_calls_never_resolve_the_package(unresolved):
    """envelope: CPU gated-norm calls take the formula route and never resolve the package."""
    y, z, w = torch.randn(3, 8), torch.randn(3, 8), torch.ones(8)
    assert norms._gated_rms_norm_route(y, w, z) == "formula"
    assert norms._gated_rms_norm_route(y, None, z) == "none"
    out = norms.gated_rms_norm(y, w, 1e-6, z=z)
    g = y * torch.nn.functional.silu(z)
    torch.testing.assert_close(out, g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + 1e-6),
                               rtol=0, atol=0)
    assert norms._rmsnorm_gated_fn is norms._UNRESOLVED
