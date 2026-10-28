"""Pins: a ``kernel_options['autotune']`` tier that differs
from the tier the import-time fused-backward / SISO grids were built with
warns BY NAME (`AutotuneTierWarning`), once per (requested, imported) pair;
a matching tier, or no explicit tier, is silent.  CPU only: validation needs
no kernel launch.  The suite's conftest imports with SSD_MIMO_AUTOTUNE=pinned,
so the import tier is set explicitly here.
"""
import warnings

import pytest

from orbit_ssm import SSDMimoConfig, SSDMimoSSM
from orbit_ssm import kernel_options as ko
from orbit_ssm.kernels.ssd_mimo import fused_bwd_kernel as fb


@pytest.fixture
def imported_light(monkeypatch):
    monkeypatch.setattr(fb, "_AUTOTUNE_TIER", "light")
    monkeypatch.setattr(ko, "_WARNED_TIERS", set())


def _layer(opts):
    return SSDMimoSSM(SSDMimoConfig(d_model=32, d_state=16, expand=1, d_conv=0, d_ssm_in=1,
                                    d_ssm_out=1, n_groups=16),
                      execution={"backend": "torch", "kernel_options": opts})


def test_a_config_tier_the_grids_do_not_follow_warns_by_name(imported_light):
    """envelope: a config autotune tier the import-time grids do not follow warns by name."""
    with pytest.warns(ko.AutotuneTierWarning,
                      match=r"autotune'\]='pinned'.*built at import with tier 'light'"):
        _layer({"autotune": "pinned"})


def test_the_warning_is_once_per_pair_and_aliases_normalise(imported_light):
    """envelope: the tier warning fires once per (requested, imported) pair; off/auto normalise to pinned/light."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ko.validate("ssd_mimo", {"autotune": "off"})       # off = pinned: warns
        ko.validate("ssd_mimo", {"autotune": "pinned"})    # same pair: silent
        ko.validate("ssd_mimo", {"autotune": "auto"})      # auto = light = import: silent
        ko.validate("ssd_mimo", {"autotune": "light"})
        ko.validate("ssd_mimo", {})                        # nothing explicit: silent
    hits = [w for w in rec if issubclass(w.category, ko.AutotuneTierWarning)]
    assert len(hits) == 1, [str(w.message) for w in hits]


def test_the_tier_is_the_imported_modules_and_an_invalid_env_raises(monkeypatch):
    """envelope: the grid tier is fused_bwd_kernel's own import-time tier (no second alias
    table or env precedence in kernel_options), and an invalid SSD_MIMO_AUTOTUNE raises (R10)."""
    import sys
    assert not hasattr(ko, "_TIER_ALIASES")
    monkeypatch.setattr(fb, "_AUTOTUNE_TIER", "full")
    monkeypatch.setenv("SSD_MIMO_AUTOTUNE", "pinned")         # the module, not the env, decides
    assert ko.imported_autotune_tier() == "full"
    monkeypatch.delitem(sys.modules, "orbit_ssm.kernels.ssd_mimo.fused_bwd_kernel")
    import orbit_ssm.kernels.ssd_mimo as pkg
    monkeypatch.delattr(pkg, "fused_bwd_kernel")
    monkeypatch.setenv("SSD_MIMO_AUTOTUNE", "bogus")
    with pytest.raises(ValueError, match="SSD_MIMO_AUTOTUNE must be"):
        ko.imported_autotune_tier()
