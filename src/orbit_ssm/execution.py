"""How a model runs, as opposed to what it computes.

``Execution`` bundles the run-time knobs shared by the SSM layers.  Every
field defaults to ``None``: "leave as is" on a live model, "the constructor's
own default" at construction.  So ``Execution(precision="ieee")`` changes the
precision and nothing else.
``ExecutionMixin`` gives ``set_execution`` and the ``execution(...)`` context
manager to a module-tree root (`BaseMixer`, `BlockStack`, `CausalLM`).  Every
module in the tree that defines ``_apply_execution(ex)`` applies the non-None
fields it understands; ``_execution_state()`` returns its current values
(in ``Execution`` field names) so the context manager can restore them.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class Execution:
    """Run-time knobs of an SSM layer, model or stack: how it runs, not what it computes.

    Every field defaults to None: "leave unchanged" on a live model (`set_execution`),
    "the constructor's own default" at construction (``execution=``).  Pass an
    instance or the equivalent dict (`Execution.from_dict`, strict).

    Parameters
    ----------
    backend : {"cuda", "torch", "sequential", "auto"} or None
        "cuda" production kernels; "torch" parallel PyTorch implementation;
        "sequential" token-by-token PyTorch recurrence; "auto" picks "cuda" or
        "torch" from the input's device.  Families may accept a subset.
    precision : str or None
        Scan operand mode.  SSD: "ieee" | "tf32" | "io" | "bf16_ops"; other families
        list theirs (`validation.PRECISIONS`).  "io" keeps operands in the IO dtype (for fp32
        that is "tf32"); "tf32" / "ieee" use fp32 operands (TF32 tensor cores /
        plain fp32); "bf16_ops" is the opt-in bf16-operand timing mode.
        HybridMamba has no precision knob.
    chunk_size : int, "auto", "measure" or None
        Scan chunk length: a power of two >= 16, "auto" (heuristic table), or
        "measure" (time candidates on the first call; SSD only).  HybridMamba has none.
    checkpoint : bool or None
        Activation checkpointing of the layer (HybridMamba only).

    ``checkpoint`` must be a bool and the keys of ``kernel_options`` str (``TypeError``), also at construction.

    A field the family cannot honour raises ``ValueError`` (precision / non-"auto" chunk_size on HybridMamba,
    ``checkpoint=True`` on SSD), at construction and in ``set_execution``.  A value that IS the family's
    fixed behaviour is accepted (``checkpoint=False`` on SSD, which never checkpoints; ``chunk_size="auto"``
    on HybridMamba, which always chooses its chunking itself); None is "leave unchanged" everywhere.
    kernel_options : dict or None
        Family-specific kernel and layer switches; unknown keys or unsupported
        values raise.  `orbit_ssm.kernel_options.describe` lists the keys and
        defaults of a family.  On a live model the dict replaces the kernel
        options and updates the named layer switches.
    """
    backend: str | None = None
    precision: str | None = None
    chunk_size: int | str | None = None
    checkpoint: bool | None = None
    kernel_options: dict | None = None

    def __post_init__(self):
        from .validation import check_bool_fields
        check_bool_fields(self)
        if isinstance(self.kernel_options, dict):       # any other type is the layer's named refusal
            bad = [k for k in self.kernel_options if not isinstance(k, str)]
            if bad:
                raise TypeError(f"Execution: kernel_options must be a dict with str keys, got key(s) {bad!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "Execution":
        """Build from a dict whose keys are exactly field names with correctly typed values; else ``TypeError``."""
        from .validation import strict_from_dict
        return strict_from_dict(cls, d)

    def fields(self) -> dict[str, Any]:
        """The non-None fields."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if getattr(self, f.name) is not None}


def as_execution(x) -> Execution:
    """Convert an ``execution=`` argument to an `Execution`.

    Parameters
    ----------
    x : Execution, dict or None
        An `Execution` is returned as is; a dict goes through `Execution.from_dict`
        (strict); None gives the empty `Execution` (change nothing).

    Raises
    ------
    TypeError
        For anything else, or a dict with unknown keys / wrongly typed values.
    """
    if x is None:
        return Execution()
    if isinstance(x, Execution):
        return x
    if isinstance(x, dict):
        return Execution.from_dict(x)
    raise TypeError(f"execution must be an Execution or a dict, got {type(x).__name__}")


class ExecutionMixin:
    """``set_execution`` / ``execution(...)`` for a module-tree root (`BlockStack`, `CausalLM`, SSM layers).

    Both apply to the whole tree below the module they are called on; to address one layer call them on
    that layer (``lm.ssm_layers[3].set_execution(...)``).  Every change bumps a version counter on the
    modules it touched (`execution_version`), which is how captured CUDA graphs notice it."""

    def set_execution(self, execution: "Execution | dict") -> None:
        """Apply ``execution`` to this module and every module below it.

        Parameters
        ----------
        execution : Execution or dict
            Only its non-None fields change; the rest stays as is.

        Every target is validated before any is changed, so a refusal leaves the
        model untouched.
        """
        ex = as_execution(execution)
        targets = [m for m in self.modules() if hasattr(m, "_apply_execution")]
        # validate all targets first: a refusal must not leave a partial change
        for m in targets:
            if hasattr(m, "_check_execution"):
                m._check_execution(ex)
        for m in targets:
            m._apply_execution(ex)
            m._execution_version = getattr(m, "_execution_version", 0) + 1

    def get_execution(self) -> "Execution":
        """The current execution of this module as an `Execution` object, symmetric with `set_execution` (every field
        the family supports is set; ``print`` shows them all).  A fresh copy per access: changing it changes nothing,
        apply it with `set_execution`.  It is the ``execution`` section of the layer's ``to_recipe()``.

        On a model (e.g. `CausalLM`) or a stack, the layers must all run the same execution; if they
        differ a ``ValueError`` points at the per-layer call (``lm.ssm_layers[i].get_execution()``)."""
        states = [m._execution_state() for m in self.modules() if hasattr(m, "_execution_state")]
        if not states:
            raise ValueError(f"{type(self).__name__} holds no layer with an execution")
        if any(s != states[0] for s in states[1:]):
            raise ValueError(f"{type(self).__name__}: the layers run different executions; ask one layer, "
                             "e.g. model.ssm_layers[i].get_execution()")
        return Execution(**copy.deepcopy(states[0]))

    @contextlib.contextmanager
    def execution(self, execution: "Execution | dict"):
        """Context manager: ``with model.execution(ex):`` runs the block under ``ex`` (as `set_execution`)
        and restores the previous execution on exit, also on an exception."""
        ex = as_execution(execution)
        targets = [m for m in self.modules() if hasattr(m, "_execution_state")]
        snapshot = [(m, m._execution_state()) for m in targets]
        try:
            self.set_execution(ex)
            yield self
        finally:
            for m, state in snapshot:
                m._apply_execution(Execution(**state))
                m._execution_version = getattr(m, "_execution_version", 0) + 1


def execution_version(module: nn.Module) -> int:
    """A counter that increases whenever the execution of ``module`` or any module below it is changed
    (`ExecutionMixin.set_execution`, the ``execution(...)`` context manager, on entry and on exit).

    Anything that bakes the execution into a recording (a captured CUDA graph, `orbit_ssm.graphs`)
    stores the value at capture time and treats a different value as stale."""
    return sum(getattr(m, "_execution_version", 0) for m in module.modules())


def layer_executions(override, n_layers: int, who: str) -> list:
    """The per-layer `Execution` list of a model recipe's ``execution=`` override: one `Execution` (or its strict
    dict) applies to every layer; a list / tuple holds one per layer and must have ``n_layers`` entries; anything
    else raises."""
    if isinstance(override, (list, tuple)):
        if len(override) != n_layers:
            raise ValueError(f"{who}: a per-layer execution list needs one entry per layer "
                             f"({n_layers}), got {len(override)}")
        return [as_execution(e) if e is not None else Execution() for e in override]
    if isinstance(override, (Execution, dict)):
        return [as_execution(override) for _ in range(n_layers)]
    raise TypeError(f"{who}: execution must be an Execution, its dict, or a list of one per layer, "
                    f"got {type(override).__name__}")


def refuse_unsupported(ex: Execution, who: str, supported, hints: dict | None = None,
                       fixed: dict | None = None) -> None:
    """Raise ``ValueError`` when ``ex`` sets a field outside ``supported`` (a family's ``_execution_fields``):
    an `Execution` field a family cannot honour is refused, never silently stored or ignored.
    ``hints`` maps a field to a pointer at what to use instead; ``fixed`` maps an unsupported field to the one
    value that is the family's fixed behaviour (``checkpoint=False`` where nothing checkpoints), which is accepted."""
    bad = [k for k, v in ex.fields().items()
           if k not in supported and not (fixed and k in fixed and v == fixed[k] and type(v) is type(fixed[k]))]
    if bad:
        why = "; ".join(f"{k}: {hints[k]}" for k in bad if hints and k in hints)
        raise ValueError(f"{who}: Execution field(s) {bad} are not supported by this family "
                         f"(supported: {', '.join(supported)})" + (f" -- {why}" if why else ""))


def layer_option_values(module) -> dict:
    """The layer-level ``kernel_options`` switches ``module`` holds now (via ``module._layer_option_attrs``)."""
    return {key: copy.deepcopy(getattr(module, attr))
            for key, attr in module._layer_option_attrs.items()}


def split_kernel_options(module, family: str, kernel_options):
    """``(kernel_part, layer_values)`` of an execution's ``kernel_options`` (validated against the family schema).

    ``kernel_part`` replaces the module's kernel options (None: leave as is); ``layer_values``
    are all layer-level switches after the change (named ones override, the rest stay)."""
    from . import kernel_options as _ko
    current = layer_option_values(module)
    if kernel_options is None:
        return None, current
    kernel, named = _ko.split(family, kernel_options)
    return kernel, {**current, **named}


def execution_state_options(module) -> dict:
    """``kernel_options`` of ``_execution_state()``: kernel part + layer switches."""
    return {**module.kernel_options, **layer_option_values(module)}
