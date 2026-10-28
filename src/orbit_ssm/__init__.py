"""OrbitSSM: selective state-space layers (SSD-MIMO, HybridMamba) and a language model around them.

Quick start::

    import torch, orbit_ssm
    config = orbit_ssm.SSDMimoConfig(d_model=512, d_state=64, expand=2.0)
    layer = orbit_ssm.SSDMimoSSM(config)                              # a complete layer
    y = layer(torch.randn(2, 128, 512))                               # [B, L, d_model] -> same shape
    lm = orbit_ssm.CausalLM(vocab_size=32000, layers=config, n_layers=8, d_mlp=1536)

Layers
------
One class per family; each owns everything and is what you build.

``SSDMimoSSM``      SSD (Mamba-2/3 lineage) with MIMO ranks; scalar per-head decay.
``HybridMambaSSM``  the hybrid decay-law family (per-coordinate decay laws).

Each is a complete layer ``x[B, L, d_model] -> y`` (input projection, causal conv, coefficient maps, recurrence,
gate, output projection).  ``layer.scan(x, ...)`` is the recurrence on the projected inputs (the same call in every family); ``layer.step(...)`` is one decoding step.
Further families register themselves with `orbit_ssm.families.register_family` when their module is present in
`orbit_ssm.ssm`; their classes are then attributes of this package like the two above.

Configuring a layer
-------------------
A layer takes exactly two arguments, ``Layer(config, execution=None)``.  The architecture is its
``<Family>Config`` (``SSDMimoConfig``, ``HybridMambaConfig``), a keyword-only dataclass whose
fields are documented on the class (``help(orbit_ssm.SSDMimoConfig)``); a layer also accepts the config's strict
dict (``<Family>Config.from_dict``).  ``config.build(d_model)`` returns the layer, and ``layer.to_recipe()``
returns the recipe it was built from.

* Width: ``expand`` (default 2.0, ``d_inner = int(d_model * expand)``) or an explicit ``d_inner``.
* Heads: ``n_heads`` or ``d_head`` or both; they must satisfy ``n_heads * d_head == d_inner``.
* Naming: ``d_`` is a vector dimension, ``n_`` a count.
* ``RowBlock`` describes one block of state rows (``rows | frac``, ``decay``, ``complex``, ``gate``)
  for the ``row_blocks`` field.

Running a layer
---------------
How a layer runs is an ``Execution``: ``backend`` ("auto" | "cuda" | "torch" | "sequential"), ``precision``
(the scan operand mode; the allowed values are per family), ``chunk_size`` (int or "auto"), ``checkpoint`` and
``kernel_options`` (a dict of kernel levers); a field the family does not support raises.  Give it as ``execution=`` at construction; change it on a live
model with ``model.set_execution(execution)`` or ``with model.execution(execution):``; to address one layer of a
model use ``lm.ssm_layers[i]``.  Every entry point takes an ``Execution`` or its dict (``Execution.from_dict``;
unknown or wrongly typed keys raise).  A field left None means "leave as is" (or the default at construction).

Models
------
``ResidualBlock``  pre-norm, residual, optional MLP around a mixer (one of our layers or any ``BaseMixer``).
``BlockStack``     the blocks plus the final norm.
``CausalLM``       ``CausalLM(*, vocab_size, layers, n_layers=None, d_model=None, d_mlp=0)``: embedding,
                   ``BlockStack``, head.  ``layers`` is one config (all layers alike) or a sequence of
                   them; a dict entry names its family with ``"type"``.

Presets and CUDA graphs
-----------------------
``orbit_ssm.presets``  ``mamba1_config`` / ``mamba2_config`` / ``mamba3_config(d_model, ...)``: the family config that
                       reproduces the official Mamba-1/2/3 layer, and ``build_official_like`` to build the layer.
``orbit_ssm.graphs``   ``graphed_step`` captures forward + backward of a module as one CUDA graph; ``capture_safe`` and
                       ``autocast`` set up the capture regime (``GraphedStep``, ``CaptureError``).

Recurrent state is a ``State`` (``StateLifecycle`` is the shared protocol); ``layer(x, state=..., return_state=True)``
threads it across calls.  Help on one class: ``help(orbit_ssm.SSDMimoSSM)``.
"""

from . import families as _families
from .layers.norms import RMSNorm
from .layers.state import BaseMixer, BaseSSM, State, StateLifecycle
from .execution import Execution
from .model import ResidualBlock, BlockStack
from .model import CausalLM
from .ssm import HybridMambaSSM, SSDMimoSSM, HybridMambaConfig, SSDMimoConfig
from .config import BACKENDS
from .config_spec import RowBlock
from . import graphs, presets


_PAPER1 = (
    "SSDMimoSSM", "HybridMambaSSM",
    "SSDMimoConfig", "HybridMambaConfig",
    "CausalLM", "BlockStack", "Execution", "ResidualBlock",
    "BaseMixer", "BaseSSM", "StateLifecycle", "State", "RMSNorm", "RowBlock",
    "presets", "graphs",
)


def __getattr__(name):
    """The public names of the other registered families (their layer and config classes and their exports)."""
    try:
        return _families.EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(_families.EXPORTS))


__all__ = list(_PAPER1) + sorted(set(_families.EXPORTS) - set(_PAPER1))
