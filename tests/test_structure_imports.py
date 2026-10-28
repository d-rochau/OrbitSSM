"""Canonical imports and retired-route isolation without running kernels."""

import ast
import importlib
from pathlib import Path
import sys
from orbit_ssm.layers import conv, proj, glue, scalar_scan, tails
from orbit_ssm.ssm import SSMS, MIXERS
from orbit_ssm.layers.state import BaseSSM, BaseMixer


def test_registered_core_and_mixer_roles():
    for core in SSMS.values(): assert issubclass(core, BaseSSM)
    for mixer in MIXERS.values(): assert issubclass(mixer, BaseMixer)


def test_shared_helpers_have_canonical_ownership():
    for helper, module in [(conv.CausalDepthwiseConv, conv), (proj.matmul_channel_major, proj),
                           (glue.glue_on, glue), (scalar_scan.scalar_decay_scan, scalar_scan),
                           (tails.TailCfg, tails)]:
        assert helper.__module__ == module.__name__


def test_production_imports_no_retired_ssd_route():
    root=Path(__file__).parents[1] / 'src' / 'orbit_ssm'
    retired={'fwd_kernel','bwd_kernel','wrapper','state_passing','omega_agg'}
    from orbit_ssm.kernels.ssd_mimo import autograd
    assert not hasattr(autograd,'unified_ssd')
    for name in retired:
        assert not (root/'kernels/ssd_mimo'/f'{name}.py').exists()
        assert 'orbit_ssm.kernels.ssd_mimo.'+name not in sys.modules
    for p in (root/'layers').glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ('common','model','ssm'), (p,node.module)


def test_package_compatibility_shims_are_gone():
    """Pins: the one-release shims were removed."""
    import importlib
    import pytest
    for name in ("triggered_mm", "ssd_mimo", "ssm_hybrid"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("orbit_ssm." + name)


def test_retired_reference_only_surfaces_refuse_by_name():
    import pytest
    from orbit_ssm.kernels.hybrid_mamba_mh.layout import norm_fracs
    with pytest.raises(ValueError,match='not served'):
        norm_fracs([{'rows':4,'decay':'external'}],4)
