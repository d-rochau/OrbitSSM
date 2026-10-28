"""Shared config base (`_CoreConfigBase`, `SpecFields`) and helpers; the family dataclasses live with the layers.

A ``<Family>Config`` holds architecture only (what defines the function and its initialization);
how to run it is an `Execution`.  Families refuse settings they cannot honor.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
from dataclasses import dataclass, fields
from .validation import (  # noqa: F401  (re-exported: `config.BACKENDS` etc. stay importable)
    BACKENDS, WRITE_RULE_DEFAULT, WRITE_RULES, WRITE_RULES_IMPLEMENTED, check_backend, check_dt_range,
    check_bool_fields, nonneg_int, positive_int, resolve_write_rule)



@functools.cache
def signature_names(*callables) -> set[str]:
    """Keyword names of ``callables`` (``self`` and ``*``/``**`` parameters excluded)."""
    return {n for fn in callables for n, p in inspect.signature(fn).parameters.items()
            if n != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                              inspect.Parameter.VAR_KEYWORD)}


def reject_unknown_kwargs(unexpected, who: str, *, valid) -> None:
    """Raise ``TypeError`` for the ``unexpected`` names, listing the sorted valid ones."""
    if unexpected:
        raise TypeError(f"{who}: unknown keyword(s) {sorted(unexpected)}; "
                        f"valid keywords: {', '.join(sorted(valid))}")


@dataclass(kw_only=True)
class _CoreConfigBase:
    """Architecture fields shared by `SSDMimoConfig`, `HybridMambaConfig` and the configs of any further registered family.

    Private base; users build the family configs, which document every field.  Nothing here says how to
    run the model (backend, precision, chunking, kernel options: see `Execution`).  Configs are plain,
    keyword-only dataclasses: ``replace(**kw)`` copies with changes, ``from_dict`` is the strict dict
    constructor, ``build(d_model=None, execution=None)`` returns the layer, and the layer's
    ``to_recipe()`` records a full recipe.

    Width rule (all families).  ``expand`` (None means 2.0) and ``d_inner`` give the inner width: only
    ``d_inner`` given: that width; only ``expand`` or neither: ``int(d_model * (expand or 2.0))``; both
    given and ``int(d_model * expand) != d_inner``: ``ValueError``.  ``d_model`` is optional: the enclosing
    `CausalLM` supplies it, `build` takes it, and a layer built straight from the config needs it here (or
    ``d_inner`` with no ``expand``, which then is the layer width too).

    Paper symbols of the shared fields (Paper 1, Section 5; names above are explanatory): ``d_state`` = N (state
    rows per subsystem), ``d_inner`` = d = H P (drive width), ``n_heads`` = H, ``d_head`` = P, ``n_groups`` = G
    (channels per head, SSD-MIMO), ``d_ssm_in`` = R (input rank), ``d_ssm_out`` = O (output rank), ``d_conv`` =
    k_conv (convolution window), ``write_rule="trapezoid"`` = the two-endpoint write (weight lambda_t),
    ``"euler"`` = exponential Euler, ``skip`` = the D term (y_t = C_t h_t + D_t u_t), ``proj_source`` = the
    coefficient source s_t ("layer_input": x_t, "main_branch": u_t), ``dt_min`` / ``dt_max`` = the init range of
    the step Delta_t, ``out_norm`` = the output RMS normalization.  The full table is ``docs/CONFIG_GLOSSARY.md``.

    The defaults are the same in every family: ``d_ssm_in = d_ssm_out = 1`` (the SISO layer), ``skip=True``
    with ``skip_sharing="head"`` (one D per head), ``out_norm=False``, ``dt_init_floor=1e-4``.  A family
    refuses the values it cannot build.
    """

    d_model: int | None = None
    d_state: int = 64
    expand: float | None = None
    d_conv: int = 4
    d_inner: int | None = None
    d_ssm_in: int = 1
    d_ssm_out: int | None = None
    n_groups: int | None = None
    n_heads: int | None = None
    d_head: int | None = None
    write_rule: str | None = None
    proj_source: str = "layer_input"
    bc_conv: bool | None = None
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    skip: bool = True
    skip_sharing: str = "head"
    out_norm: bool = False

    def __post_init__(self):
        check_bool_fields(self)           # a flag that is not a bool is a TypeError, as in from_dict
        nonneg_int(self.d_conv, "d_conv")
        for name in ("d_model", "d_state", "d_inner", "d_ssm_in", "d_ssm_out", "n_groups", "n_heads", "d_head"):
            if getattr(self, name) is not None:
                positive_int(getattr(self, name), name)
        if self.expand is not None and not self.expand > 0:
            raise ValueError(f"expand must be positive, got {self.expand!r}")
        check_dt_range(self.dt_min, self.dt_max, self.dt_init_floor)
        self.width()          # a contradicting d_model / expand / d_inner fails here, once they are all known

    def width(self, d_model: int | None = None) -> int | None:
        """The inner width ``d_inner`` this config resolves to at layer width ``d_model`` (default: the
        config's), or None while it cannot be known yet (no ``d_inner`` and no ``d_model``)."""
        from .configuration import resolve_mixer_width
        d_model = self.d_model if d_model is None else d_model
        if d_model is None:
            return self.d_inner
        return resolve_mixer_width(d_model, self.expand, self.d_inner)

    def layer_width(self) -> int:
        """The layer width ``d_model``: the config's, else ``d_inner`` (when no ``expand`` is set)."""
        if self.d_model is not None:
            return self.d_model
        if self.d_inner is not None and self.expand is None:
            return self.d_inner
        raise TypeError(f"{type(self).__name__}: the layer width d_model is not set; give d_model, or build "
                        f"with config.build(d_model), or let CausalLM supply it")

    def with_d_model(self, d_model: int):
        """This config at layer width ``d_model``.  A config that names a different ``d_model`` is refused."""
        if self.d_model is None:
            return dataclasses.replace(self, d_model=d_model)
        if self.d_model != d_model:
            raise ValueError(f"{type(self).__name__}.d_model={self.d_model} conflicts with the requested "
                             f"d_model={d_model}")
        return self

    def replace(self, **kw):
        """A copy of this config with the fields in ``kw`` changed (validated like the constructor)."""
        return dataclasses.replace(self, **kw)

    def layer_kwargs(self) -> dict:
        """The keywords the layer builds from: every field, as given (the layer resolves them)."""
        _refuse(self, type(self).__name__, getattr(self, "_UNSUPPORTED", {}))
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict):
        """The strict dict form of the config: only its field names, each of its type
        (``TypeError`` otherwise); the field values are then validated as in the constructor."""
        from .validation import strict_from_dict
        return strict_from_dict(cls, d)

    def build(self, d_model: int | None = None, execution=None):
        """The layer of this family at layer width ``d_model`` (default: the config's), run as ``execution``."""
        cfg = self if d_model is None else self.with_d_model(d_model)
        return self._layer_class()(cfg, execution=execution)


@dataclass(kw_only=True)
class SpecFields:
    """The row_blocks / rotation / B-C-map spec fields of a family config.

    Mixed into `SSDMimoConfig` and `HybridMambaConfig`: flat fields named as in the
    constructors and recipes; None = unset (family default)."""

    #: list of `RowBlock` or their strict dicts
    row_blocks: list | tuple | None = None
    theta_per_head: bool | None = None
    learn_theta_freq: bool | None = None
    use_theta_h: bool | None = None
    omega_layout: str | None = None
    theta_init: str | None = None
    b_norm: bool | None = None
    c_norm: bool | None = None
    bc_bias: str | None = None
    b_gate: bool | None = None
    c_gate: bool | None = None

    def spec_kwargs(self) -> dict:
        """The spec fields as the keywords `config_spec.resolve` takes (a deep copy; None = unset)."""
        from .config_spec import SPEC_FIELDS, spec_kwargs
        return copy.deepcopy(spec_kwargs({n: getattr(self, n, None) for n in SPEC_FIELDS}))


def _refuse(cfg: _CoreConfigBase, family: str, unsupported: dict) -> None:
    """Raise on any core field the family cannot honor.

    `unsupported` maps field name -> refused test.  Loud refusal beats a
    silently dropped feature: a config is a claim about the model."""
    problems = [
        f"{name}={getattr(cfg, name)!r}"
        for name, refused in unsupported.items() if refused(cfg)
    ]
    if problems:
        raise NotImplementedError(
            f"{family} does not support: {', '.join(problems)}. "
            f"Use a family that does, or drop the field.")


def __getattr__(name):
    # Lazy import: avoids a config/layer import cycle.
    from . import ssm, families
    if name not in {c.__name__ for c in families.CONFIGS.values()}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(ssm, name)
