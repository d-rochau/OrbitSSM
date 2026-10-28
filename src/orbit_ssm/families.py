"""The family registry: layer families register themselves; core modules consult the registry.

A family module (one file in `orbit_ssm.ssm`) calls `register_family` when it is imported::

    register_family("ssd_mimo", config=SSDMimoConfig, layer=SSDMimoSSM)

``orbit_ssm.ssm`` imports every module of its package (``pkgutil``), so a tree that lacks a family module simply
lacks that family; nothing in core names it.  A family may also contribute

``precisions``       ``(allowed values, meaning of None)`` for `validation.check_precision`
``kernel_options``   its layer switches, a tuple of ``kernel_options`` option records
``spec``             a `SpecHooks` (rows, rotation and B/C-map rules of `config_spec`)
``exports``          further public names of ``orbit_ssm`` (``{name: object}``)

Kernel extensions register scan operand modes with `register_scan_precision`.  Core modules that keep tables
keyed by family (`config_spec`, `kernel_options`, `validation`) subscribe with `on_register`; a subscriber is
called once for every family registered so far and for every later one.  This module imports nothing (no torch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["Family", "SpecHooks", "ScanPrecision", "register_family", "register_scan_precision",
           "on_register", "on_scan_precision", "family", "names", "LAYERS", "CONFIGS",
           "EXPORTS", "SCAN_PRECISIONS", "SPLIT_DOT_MODES"]


@dataclass(frozen=True)
class SpecHooks:
    """The rules a family adds to the shared row_blocks / rotation / B-C-map spec (`config_spec.resolve`).

    ``row_ext``           block keys beyond rows / frac / decay / complex / gate (each a bool, default False)
    ``default_decay``     the decay a block resolves to when it omits ``decay`` (None: ``decay`` is required)
    ``row_objects``       ``{key: (value, index) -> dict}``: block keys whose value is a nested settings object (an object or its dict);
                          the callable validates it strictly and returns its canonical dict.  The key is kept in the resolved block
                          only when set, so blocks without it are unchanged
    ``block_decay``       ``{object key: marker}``: a block holding that ``row_objects`` key owns its decays inside the object; its outer ``decay`` must
                          be unset and resolves to the marker
    ``maps_ext``          B/C-map fields only this family has
    ``decays``            decay laws the family serves (adds it to ``DECAY_LAWS[law]["families"]``)
    ``rows_required``     ``row_blocks`` has no default (the family derives its own rows)
    ``block_gate``        ``(index, gate, complex) -> bool``: the resolved per-block gate
    ``check_rows``        ``(blocks) -> None``: raise on a layout the family does not serve
    ``rotation_defaults`` the rotation fields' defaults (None: the shared ones)
    ``check_rotation``    ``(rotation) -> None``: raise on a rotation the family does not serve
    ``bias_grans``        admissible ``bc_bias`` values (None: the shared ones), ``default_bias`` the default
    ``resolve_maps``      ``(maps, blocks) -> dict``: the family's ``b_gate`` / ``c_gate`` (+ extension) entries
    """
    row_ext: tuple = ()
    row_objects: dict | None = None
    block_decay: dict | None = None
    default_decay: str | None = None
    maps_ext: tuple = ()
    decays: tuple = ()
    rows_required: bool = False
    block_gate: Callable | None = None
    check_rows: Callable | None = None
    rotation_defaults: dict | None = None
    check_rotation: Callable | None = None
    bias_grans: tuple | None = None
    default_bias: str = "head"
    resolve_maps: Callable | None = None


@dataclass(frozen=True)
class Family:
    name: str
    config: type
    layer: type
    precisions: tuple | None = None
    kernel_options: tuple = ()
    spec: SpecHooks | None = None


@dataclass(frozen=True)
class ScanPrecision:
    """A scan operand mode of the kernel entry ``ssd_mimo_scan`` beyond the shared ones.

    ``dispatch``    ``(x_in, delta, B_tok, C_tok, a_log, **rest) -> result | NotImplemented``: called first;
                    a result is returned as the scan's, ``NotImplemented`` lets the shared kernels run (after the
                    hook's own checks)
    ``split_dot``   the name is also a Triton ``input_precision`` spelling the shared kernels hand to ``tl.dot``
                    (fp32 operands only)
    """
    name: str
    dispatch: Callable | None = None
    split_dot: bool = False


#: family name -> layer class / config class (registration order); `orbit_ssm.ssm` exposes these as
#: ``SSMS`` / ``MIXERS`` / ``CONFIGS``
LAYERS: dict[str, type] = {}
CONFIGS: dict[str, type] = {}
#: public names beyond the Paper-1 ones, by name: the layer and config classes of every family and its ``exports``
EXPORTS: dict[str, Any] = {}
SCAN_PRECISIONS: dict[str, ScanPrecision] = {}
#: Triton ``input_precision`` spellings beyond tf32 / ieee that a scan precision may name (``split_dot``)
SPLIT_DOT_MODES: set[str] = set()
_FAMILIES: dict[str, Family] = {}
_listeners: list = []
_scan_listeners: list = []


def register_family(name: str, *, config: type, layer: type, precisions: tuple | None = None,
                    kernel_options: tuple = (), spec: SpecHooks | None = None,
                    exports: dict | None = None) -> Family:
    """Register a layer family (once per name)."""
    if name in _FAMILIES:
        raise ValueError(f"family {name!r} is already registered")
    fam = Family(name, config, layer, precisions, tuple(kernel_options), spec)
    _FAMILIES[name] = fam
    LAYERS[name], CONFIGS[name] = layer, config
    for obj in (config, layer):
        EXPORTS[obj.__name__] = obj
    EXPORTS.update(exports or {})
    for fn in list(_listeners):
        fn(fam)
    return fam


def on_register(fn: Callable[[Family], None]) -> None:
    """Subscribe ``fn`` to family registrations (called for the families already registered, then for each new one)."""
    _listeners.append(fn)
    for fam in list(_FAMILIES.values()):
        fn(fam)


def register_scan_precision(name: str, *, dispatch: Callable | None = None, split_dot: bool = False) -> ScanPrecision:
    """Register a scan operand mode of ``ssd_mimo_scan`` (see `ScanPrecision`)."""
    if name in SCAN_PRECISIONS:
        raise ValueError(f"scan precision {name!r} is already registered")
    sp = ScanPrecision(name, dispatch, split_dot)
    SCAN_PRECISIONS[name] = sp
    if split_dot:
        SPLIT_DOT_MODES.add(name)
    for fn in list(_scan_listeners):
        fn(sp)
    return sp


def on_scan_precision(fn: Callable[[ScanPrecision], None]) -> None:
    """Subscribe ``fn`` to scan-precision registrations (replayed like `on_register`)."""
    _scan_listeners.append(fn)
    for sp in list(SCAN_PRECISIONS.values()):
        fn(sp)


def family(name: str) -> Family | None:
    """The registered family ``name``, or None."""
    return _FAMILIES.get(name)


def names() -> tuple[str, ...]:
    """The registered family names, in registration order."""
    return tuple(_FAMILIES)
