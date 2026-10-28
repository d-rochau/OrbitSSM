"""Width / head-geometry resolution and constructor recipes (`Configurable`); no Torch import.

Recipes record constructor semantics and initialization only; they do not freeze
kernel/environment choices or promise compatibility across releases.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
from numbers import Real

from .validation import check_divides, positive_int


CONFIG_SCHEMA = "orbit_ssm.layer/v4"
SEMANTICS_VERSION = "unfrozen"


def resolve_dt_rank(dt_rank, *, d_sel: int, heads: int, d_model=None,
                    force_dt_dense: bool = False) -> int | None:
    """MH delta projection rank: None = dense; 'auto' = Mamba-1's ceil(width/16);
    'adaptive' = that rank only if r*(d_sel+heads) < d_sel*heads (multiply count, not latency).
    Without d_model the width is d_sel."""
    if not isinstance(force_dt_dense, bool):
        raise ValueError("force_dt_dense must be a boolean")
    if force_dt_dense:
        if dt_rank is not None and dt_rank != "adaptive":
            raise ValueError("force_dt_dense conflicts with an explicit bottleneck dt_rank")
        return None
    if dt_rank is None:
        return None
    if dt_rank not in ("adaptive", "auto"):
        return positive_int(dt_rank, "dt_rank")
    if dt_rank == "auto" and d_model is None:
        raise ValueError("dt_rank='auto' requires d_model (Mamba-1 uses ceil(d_model/16))")
    width = positive_int(d_sel if d_model is None else d_model, "delta reference width")
    rank = (width + 15) // 16
    if dt_rank == "auto" or rank * (d_sel + heads) < d_sel * heads:
        return rank
    return None


DEFAULT_EXPAND = 2.0


def resolve_mixer_width(d_model, expand=None, d_inner=None):
    """Inner width of a layer: ``d_inner`` if only it is given, else ``int(d_model * expand)``
    (``expand=None`` = `DEFAULT_EXPAND` = 2.0); both given and inconsistent: ValueError."""
    d_model = positive_int(d_model, "d_model")
    width = None if d_inner is None else positive_int(d_inner, "d_inner")
    if expand is None:
        if width is not None:
            return width
        expand = DEFAULT_EXPAND
    elif isinstance(expand, bool) or not isinstance(expand, Real) or not math.isfinite(expand) or expand <= 0:
        raise ValueError("expand must be a positive finite number or None")
    product = d_model * expand
    if not math.isfinite(product):
        raise ValueError("d_model * expand must be finite")
    derived = positive_int(int(product), "expanded d_inner")
    if width is not None and width != derived:
        raise ValueError(f"d_inner={width} conflicts with int(d_model*expand)={derived}; give only one of them")
    return derived


def resolve_geometry(width=None, *, n_heads=None, d_head=None,
                     default_heads=None, default_d_head=None, what="d_inner"):
    """Head geometry ``n_heads * d_head == width`` for every family; returns ``(n_heads, d_head)``.

    One of the two given: the other is derived; both: they must multiply to ``width``.  Neither:
    ``default_d_head`` (width known) or both defaults (``width=None``: width = ``n_heads * d_head``).
    Family rules on top (SSD's d_head = n_groups * d_ssm_in) stay with the family."""
    h = None if n_heads is None else positive_int(n_heads, "n_heads")
    p = None if d_head is None else positive_int(d_head, "d_head")
    if width is None:
        h = h if h is not None else None if default_heads is None else positive_int(default_heads, "n_heads")
        p = p if p is not None else None if default_d_head is None else positive_int(default_d_head, "d_head")
        if h is None or p is None:
            raise ValueError("without a width, specify both n_heads and d_head")
        return h, p
    width = positive_int(width, what)
    if h is None and p is None:
        if default_d_head is None:
            raise ValueError("specify n_heads or d_head")
        p = positive_int(default_d_head, "d_head")
    if h is None:
        check_divides("d_head", p, what, width)
        h = width // p
    elif p is None:
        check_divides("n_heads", h, what, width)
        p = width // h
    if h * p != width:
        raise ValueError(f"n_heads*d_head must equal {what} ({h}*{p} != {width})")
    return h, p


def resolve_rank_geometry(width, d_ssm_in, *, n_heads=None, d_head=None):
    """The head geometry of the families whose drive carries ``d_ssm_in`` columns per kernel row (SSD-MIMO and
    any registered family with MIMO ranks): ``n_heads * d_head == width`` (the DRIVE width), ``d_head = d_subsystems * d_ssm_in``.  Returns
    ``(n_heads, d_head)``; ``d_head`` defaults to ``max(1, 64 // d_ssm_in) * d_ssm_in`` and must be a multiple
    of ``d_ssm_in``."""
    R = positive_int(d_ssm_in, "d_ssm_in")
    h, p = resolve_geometry(width, n_heads=n_heads, d_head=d_head, default_d_head=max(1, 64 // R) * R)
    if p % R:
        raise ValueError(f"d_head={p} must be divisible by d_ssm_in={R} (d_head = d_subsystems * d_ssm_in)")
    return h, p


class Configurable:
    """Mixin: ``to_recipe()`` / ``from_recipe()`` recipes, separate from weights and state.

    A recipe is ``{schema, semantics_version, family, model, execution}``: ``model`` is one flat
    section of constructor keywords (``n_heads``, ``d_head``, the flat spec fields); ``execution``
    holds the live module's `Execution` fields.  Only the current schema loads; an older recipe has to be
    converted first, with the conversion tool of the release that wrote it."""

    config_schema = CONFIG_SCHEMA

    def to_recipe(self) -> dict:
        """A JSON-safe recipe with every constructor default made explicit, plus the module's
        current execution.  Weights (state_dict), runtime dtype, env overrides and optimizer
        state are recorded separately."""
        document = copy.deepcopy(self._resolved_constructor_config)
        document["execution"] = json.loads(json.dumps(self._execution_state(), allow_nan=False))
        return document

    @classmethod
    def from_recipe(cls, recipe: dict, *, execution=None):
        """Rebuild the module from a ``to_recipe()`` document of the current schema.  An older schema, a
        document of another family, or one that does not reproduce itself raises ``ValueError``.

        ``execution`` overrides the recorded execution (load a GPU-trained layer onto ``backend="torch"``):
        an `Execution` or its strict dict.  The override REPLACES the recorded execution as a whole, it is never
        merged field by field: ``execution=Execution(backend="torch")`` runs the layer with that Execution and the
        constructor defaults for every field it leaves None (default ``kernel_options`` included), not the recorded
        ones.  A recipe of a model (e.g. `CausalLM`) records one execution per layer: ``execution`` is then
        one Execution (or dict) applied to every layer, or a list of one Execution per layer (matching length);
        anything else raises.  The model section is always checked against the recipe; the execution check is
        skipped when an override is given."""
        document = recipe
        if not isinstance(document, dict) or document.get("schema") != cls.config_schema:
            got = document.get("schema") if isinstance(document, dict) else type(document).__name__
            raise ValueError(
                f"unsupported config schema {got!r}: this version reads {cls.config_schema!r} only; this file "
                "uses an older format; convert it with the conversion tool of the release it was written with")
        expected = {"schema", "semantics_version", "family", "model", "execution"}
        if set(document) != expected:
            raise ValueError(f"config fields must be exactly {sorted(expected)}")
        if document["schema"] != cls.config_schema or document["semantics_version"] != SEMANTICS_VERSION:
            raise ValueError("unsupported config schema or semantics version")
        if document["family"] != cls.config_family:
            raise ValueError(f"expected family {cls.config_family!r}, got {document['family']!r}")
        model, recorded = document["model"], document["execution"]
        if set(model) & set(recorded):
            raise ValueError("model and execution config fields overlap")
        module = cls._from_recipe_sections(copy.deepcopy(model), copy.deepcopy(recorded), execution)
        rebuilt = module.to_recipe()
        for section in ("model", "execution"):
            if set(document[section]) != set(rebuilt[section]):
                raise ValueError(f"{section} fields do not match the config schema")
        if execution is not None:
            document = dict(document, execution=rebuilt["execution"])
        if rebuilt != document:
            raise ValueError("config must contain canonical resolved values, including nested fields")
        return module

    @classmethod
    def _from_recipe_sections(cls, model, recorded, override=None):
        """Build from the recipe's sections: the recorded execution, or ``override`` (one `Execution` / dict) as a whole."""
        from .ssm import CONFIGS
        from .execution import as_execution
        return cls(CONFIGS[cls.config_family].from_dict(model),
                   execution=recorded if override is None else as_execution(override))


def as_family_config(config, config_type):
    """The ``config=`` conversion: a `config_type` stays, a dict goes through ``from_dict``
    (strict), anything else (another family's config, a subclass) raises ``TypeError``."""
    if isinstance(config, dict):
        return config_type.from_dict(config)
    if type(config) is not config_type:
        raise TypeError(f"expected {config_type.__name__} or a dict, got {type(config).__name__}")
    return config


def mixer_core_options(options, *, d_model, d_inner, d_state, d_conv, **defaults):
    """Scan keywords of an SSD / HybridMamba layer: shared head + family ``defaults`` + ``options``.
    An option naming a different ``d_inner`` / ``d_model`` / ``d_conv`` than the layer raises."""
    kw = dict(d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv, **defaults)
    options = copy.deepcopy(options or {})
    if options.get("d_model") is None:
        options.pop("d_model", None)
    kw.update(options)
    for name, width in (("d_inner", d_inner), ("d_model", d_model), ("d_conv", d_conv)):
        if kw[name] != width:
            raise ValueError(f"core {name}={kw[name]} conflicts with layer {name}={width}")
    return kw


def record_constructor_config(module, values, *, constructor, overrides=None, drop=(), exclude=()):
    """Store the resolved constructor arguments as ``module._resolved_constructor_config``.

    ``values``: parameter name -> resolved value (typically ``locals()``); ``overrides`` replaces
    or adds model fields; ``exclude`` names input-only parameters (not recorded); ``drop`` names
    fields removed after the overrides.  Head geometry is recorded as ``n_heads`` / ``d_head``;
    the execution section is read off the live module by `Configurable.to_recipe`."""
    parameters = inspect.signature(constructor).parameters
    resolved = {name: copy.deepcopy(values[name]) for name, param in parameters.items()
                if name != "self" and name not in exclude and param.kind not in
                (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
    resolved.update(overrides or {})
    resolved.update(n_heads=module.H, d_head=getattr(module, "d_head", module.P))
    for name in drop:
        resolved.pop(name, None)
    document = dict(schema=module.config_schema, semantics_version=SEMANTICS_VERSION,
                    family=module.config_family, model=resolved, execution={})
    # JSON round trip: tuples -> lists, non-serializable values rejected.
    module._resolved_constructor_config = json.loads(json.dumps(document, allow_nan=False))
