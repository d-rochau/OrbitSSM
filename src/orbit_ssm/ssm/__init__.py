"""The SSM families: configs, scan cores and complete layers.

Every module of this package is imported here (``pkgutil``); a family module registers itself with
`orbit_ssm.families.register_family` when it is imported, and nothing here names the families a tree happens to
contain.  ``SSMS`` / ``MIXERS`` / ``CONFIGS`` map the family name ("hybrid_mamba", "ssd_mimo", ...) to the layer class
and its ``<Family>Config``.  ``SSMS`` and ``MIXERS`` hold the same classes (a layer is both the SSM and the mixer
of a block).  The kernel families behind them are the SSD scan and the MH scan, plus any registered family.
"""
import importlib
import pkgutil

from .. import families as _families
from .hybrid_mamba import HybridMambaSSM, HybridMambaConfig
from .ssd_mimo import SSDMimoSSM, SSDMimoConfig

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")

SSMS = _families.LAYERS
MIXERS = _families.LAYERS
CONFIGS = _families.CONFIGS

_STATIC = ("SSMS", "MIXERS", "CONFIGS", "HybridMambaSSM", "HybridMambaConfig", "SSDMimoSSM", "SSDMimoConfig")


def __getattr__(name):
    """The layer and config classes of the other registered families from the registry."""
    try:
        return _families.EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(_families.EXPORTS))


__all__ = list(_STATIC) + sorted({c.__name__ for c in (*_families.LAYERS.values(), *_families.CONFIGS.values())}
                                 - set(_STATIC))
