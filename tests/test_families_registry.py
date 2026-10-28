"""Pins: layer families register themselves (orbit_ssm.families); core tables are filled from the registry."""
import pytest

import orbit_ssm
from orbit_ssm import config_spec, families, kernel_options, validation
from orbit_ssm.ssm import CONFIGS, MIXERS, SSMS

pytestmark = pytest.mark.ssm


def test_every_registered_family_is_in_every_core_table():
    """contract: SSMS / MIXERS / CONFIGS, config_spec.FAMILIES and kernel_options.FAMILIES list exactly the registered families."""
    names = set(families.names())
    assert {"ssd_mimo", "hybrid_mamba"} <= names
    assert set(SSMS) == set(MIXERS) == set(CONFIGS) == names
    assert set(config_spec.FAMILIES) == names
    assert set(kernel_options.FAMILIES) == names
    for name in names:
        assert CONFIGS[name] is families.family(name).config and SSMS[name] is families.family(name).layer


def test_registered_classes_are_package_attributes_without_being_named_statically():
    """contract: `from orbit_ssm import <registered class>` resolves through the registry; unknown names raise AttributeError."""
    for fam in map(families.family, families.names()):
        assert getattr(orbit_ssm, fam.layer.__name__) is fam.layer
        assert getattr(orbit_ssm, fam.config.__name__) is fam.config
        assert fam.layer.__name__ in orbit_ssm.__all__
    with pytest.raises(AttributeError):
        orbit_ssm.NoSuchName


def test_double_registration_is_refused():
    """contract: a family name registers once."""
    fam = families.family("ssd_mimo")
    with pytest.raises(ValueError, match="already registered"):
        families.register_family("ssd_mimo", config=fam.config, layer=fam.layer)


def test_registered_scan_precisions_extend_the_allowed_values():
    """contract: a scan precision registered by a kernel extension is an allowed ssd_mimo_scan precision, and only then."""
    allowed = validation.PRECISIONS["ssd_mimo_scan"][0]
    assert set(families.SCAN_PRECISIONS) <= set(allowed)
    assert set(allowed) - set(families.SCAN_PRECISIONS) == {"ieee", "tf32", "io", "bf16_ops"}
