"""Language model (`CausalLM`), residual blocks (`ResidualBlock`) and the block stack (`BlockStack`)."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.norms import RMSNorm
from .layers.state import BaseMixer, CurrentStateDict, State, StateLifecycle
from .execution import ExecutionMixin, Execution, as_execution, layer_executions
from .layers.swiglu import SwiGLU
from .layers.init import (init_lm, mark_residual_out, rescale_residual_out,
                          tag_official_mamba_mixer, tag_residual_mixer)  # init policy: layers/init.py
from .configuration import Configurable, SEMANTICS_VERSION


from .ssm.hybrid_mamba import HybridMambaSSM, _HMInProjFn, _hm_in_proj

# Mixer registry: reference arms this package does not own (vendored Mamba-1/2/3, attention)
# enter through `CausalLM.register_mixer`.  A factory is called as
# ``factory(d_model=..., **options)`` (the options of the layer's dict) and returns a
# BaseMixer-compatible module; it must tag its own residual output projection
# (``tag_official_mamba_mixer`` for vendored mamba_ssm mixers, ``mark_residual_out`` otherwise).
_MIXER_REGISTRY: dict[str, "Any"] = {}


from .ssm.ssd_mimo import (
    SSDMimoSSM, _SSDMimoCombinedFn, _combined_apply, _fused_in_proj,
)


def _per_layer_widths(d_mlp, n_layers: int) -> tuple[int, ...]:
    """``d_mlp`` (an int for every layer, or a sequence with one entry per layer) as a tuple of widths."""
    def one(v, what):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{what} must be a non-negative int, got {v!r}")
        return v
    if isinstance(d_mlp, Sequence) and not isinstance(d_mlp, (str, bytes)):
        if len(d_mlp) != n_layers:
            raise ValueError(f"d_mlp has {len(d_mlp)} entries for {n_layers} layers")
        return tuple(one(v, f"d_mlp[{i}]") for i, v in enumerate(d_mlp))
    return (one(d_mlp, "d_mlp"),) * n_layers


class ResidualBlock(StateLifecycle, nn.Module):
    """One model block: ``x + mixer(norm(x))``, then ``+ mlp(norm_mlp(.))`` if ``d_mlp > 0``.

    The mixer holds the SSM and all its projections, convolution, gates and output
    processing; this wrapper adds the pre-norm and the residual path(s).

    Parameters
    ----------
    mixer : nn.Module
        BaseMixer-compatible module, ``forward(x[B, L, D]) -> y[B, L, D]``.
    d_model : int
        Block width (embedding dimension of the stream).
    d_mlp : int, default 0
        Hidden width of the SwiGLU sublayer (``RMSNorm -> SwiGLU -> +res``);
        0 builds no MLP.

    Attributes
    ----------
    n_residuals : int
        Residual branches: 1, or 2 with an MLP.  `BlockStack` uses it for the
        ``1/sqrt(depth)`` residual rescale.

    Notes
    -----
    The block tags the SwiGLU ``down`` and the ``out_proj`` of a built-in mixer
    family for that rescale.  Any other mixer must tag its own residual output
    projection (``mark_residual_out``); an untagged one keeps its own init.
    """

    def __init__(self, mixer: nn.Module, d_model: int,
                 d_mlp: int = 0):
        super().__init__()
        # Fused ATen RMSNorm: fp-tolerance-equivalent to the hand-rolled path, same
        # parameter.
        self.norm = RMSNorm(d_model, fused=True)
        self.mixer = mixer
        # d_mlp = 0 builds no MLP at all: no extra norm, state_dict keys or RNG draws.
        self.d_mlp = int(d_mlp)
        if self.d_mlp:
            self.norm_mlp = RMSNorm(d_model, fused=True)
            self.mlp = SwiGLU(d_model, self.d_mlp)
            mark_residual_out(self.mlp.down.weight)
        else:
            self.norm_mlp = None
            self.mlp = None
        from .ssm import MIXERS
        if isinstance(mixer, tuple(MIXERS.values())):
            tag_residual_mixer(mixer)

    @property
    def n_residuals(self) -> int:
        """Residual branches of this block: the mixer, plus the MLP if any."""
        return 1 if self.mlp is None else 2

    def _mlp_residual(self, x):
        """The block's second residual branch (identity when there is no MLP)."""
        if self.mlp is None:
            return x
        return x + self.mlp(self.norm_mlp(x))

    def forward(self, x, state=None, return_state: bool = False):
        """``x + mixer(norm(x))``, then the MLP residual if any; ``x`` (B, L, d_model).  ``state`` is the
        mixer's state; ``return_state=True`` returns ``(y, State)``."""
        if state is None and not return_state:
            return self._mlp_residual(x + self.mixer(self.norm(x)))
        y, state = self.mixer(self.norm(x), state=state, return_state=True)
        y = self._mlp_residual(x + y)
        return (y, state) if return_state else y

    def step(self, x_t, state=None):
        """One token ``x_t`` (B, d_model): ``(y_t, State)``; None starts from the initial state."""
        if state is None:
            state = self.init_state(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        y, state = self.mixer.step(self.norm(x_t), state)
        return self._mlp_residual(x_t + y), state

    def validate_state(self, state, **kwargs):
        """Raise if ``state`` does not fit the mixer (keys, shapes, ``batch_size=``, ``device=``)."""
        return self.mixer.validate_state(state, **kwargs)

    def init_state(self, batch_size, **kwargs):
        """The mixer's initial `State` for ``batch_size``."""
        return self.mixer.init_state(batch_size, **kwargs)

    def reset_state(self, state, mask=None, keys=None):
        """In place: reset ``state`` for the batch rows ``mask`` selects (default all)."""
        return self.mixer.reset_state(state, mask, keys=keys)

    def set_state(self, state, other):
        """In place: load ``other`` into ``state``."""
        return self.mixer.set_state(state, other)

    def get_state(self):
        """The state of the most recent forward/step call."""
        return self.mixer.get_state()




class BlockStack(CurrentStateDict, ExecutionMixin, nn.Module):
    """Ordered, possibly heterogeneous blocks plus a final RMSNorm (``norm_f``).

    Each block may wrap a different SSM family or dimensions, as long as its
    external width is ``d_model``.

    Parameters
    ----------
    blocks : iterable of nn.Module
        The layers (`ResidualBlock` or compatible), stored as ``layers``.
    d_model : int
        Stream width.
    residual_rescale : bool, default True
        GPT-2 / Megatron residual-depth init, applied once at construction:
        every weight tagged by ``mark_residual_out`` inside the stack (a block's
        mixer ``out_proj`` and SwiGLU ``down``, a registered mixer's self-tagged
        output projection) is redrawn ``kaiming_uniform_(a=sqrt(5))`` and divided
        by ``sqrt(n_residuals)`` (sum of ``block.n_residuals``; 1 if undeclared).
        Nothing else is touched.  Load checkpoints after building the stack.
        ``False`` keeps the init the mixer/MLP gave those weights.

    Execution settings go through ``set_execution`` / ``execution(...)``.
    """

    def __init__(self, blocks, d_model: int, *, residual_rescale: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(d_model, fused=True)  # see ResidualBlock
        self.residual_rescale = bool(residual_rescale)
        if self.residual_rescale and len(self.layers):
            rescale_residual_out(self, n_residuals=self.n_residuals)

    @classmethod
    def from_mixers(cls, mixers, d_model: int, d_mlp=0, *,
                    residual_rescale: bool = True):
        """Build a stack from mixer instances.  ``d_mlp`` is one int for every block or a sequence with
        one entry per mixer; a width ``> 0`` gives that block the pre-norm SwiGLU sublayer, 0 a
        mixer-only block.  ``residual_rescale`` as in the class."""
        mixers = list(mixers)
        widths = _per_layer_widths(d_mlp, len(mixers))
        return cls(
            [ResidualBlock(mixer=mixer, d_model=d_model, d_mlp=width)
             for mixer, width in zip(mixers, widths)],
            d_model=d_model,
            residual_rescale=residual_rescale,
        )

    @property
    def n_residuals(self) -> int:
        """Residual branches adding into the stream, summed over the blocks.

        This is the depth of the ``1/sqrt(depth)`` residual rescale; a block
        without an ``n_residuals`` attribute counts as one branch.
        """
        total = 0
        for index, block in enumerate(self.layers):
            count = getattr(block, "n_residuals", 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"layers[{index}].n_residuals must be a positive int, got {count!r}")
            total += count
        return total

    @property
    def blocks(self):
        """Alias describing that every layer is a residual block."""
        return self.layers

    @property
    def ssm_layers(self) -> tuple:
        """The blocks' mixer modules in order (read-only), e.g. ``stack.ssm_layers[3].set_execution(...)``."""
        return tuple(block.mixer for block in self.layers)

    @property
    def norms(self):
        """Per-layer pre-norms, exposed for architecture introspection."""
        return [block.norm for block in self.layers]

    def _validate_states(self, states, name="state"):
        if not isinstance(states, (list, tuple)) or len(states) != len(self.layers):
            raise ValueError(f"{name} must contain exactly {len(self.layers)} layer states")

    def _validate_carries(self, states, *, batch_size=None, device=None):
        """Validate every layer's state up front, so a bad late one fails before any layer runs."""
        self._validate_states(states)
        for block, state in zip(self.layers, states):
            if state is not None:
                block.validate_state(state, batch_size=batch_size, device=device)

    def forward(self, x, state=None, return_state: bool = False):
        """`state`, when given, is one entry PER LAYER (the same list
        shape :meth:`init_state`/:meth:`step` use); `None` entries fall
        back to that layer's own default (fresh) start."""
        if state is None and not return_state:
            for b in self.layers:
                x = b(x)
            return self.norm_f(x)
        if state is not None:
            self._validate_carries(state, batch_size=x.shape[0], device=x.device)
        states = state if state is not None else [None] * len(self.layers)
        out_states = []
        for b, s in zip(self.layers, states):
            x, s = b(x, state=s, return_state=True)
            out_states.append(s)
        y = self.norm_f(x)
        return (y, out_states) if return_state else y

    def init_state(self, batch_size, *, h_init=None, overrides=None, **kwargs):
        """One initial `State` per layer for ``batch_size``; ``h_init`` / ``overrides`` are per-layer lists."""
        count = len(self.layers)
        if h_init is not None:
            self._validate_states(h_init, "h_init")
        if overrides is not None:
            self._validate_states(overrides, "overrides")
        hidden = h_init if h_init is not None else [None] * count
        custom = overrides if overrides is not None else [None] * count
        return [b.init_state(batch_size, h_init=h, overrides=o, **kwargs)
                for b, h, o in zip(self.layers, hidden, custom)]

    def step(self, x, state=None):
        """One token ``x`` (B, d_model) through every layer and the final norm: ``(y, per-layer states)``;
        None starts from `init_state`."""
        states = state
        if states is None:
            states = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        self._validate_carries(states, batch_size=x.shape[0], device=x.device)
        out = []
        for b, s in zip(self.layers, states):
            x, s = b.step(x, s)
            out.append(s)
        return self.norm_f(x), out

    def reset_state(self, states, mask=None, keys=None):
        """In place: reset the per-layer ``states`` for the batch rows ``mask`` selects (default all), only
        the state entries named in ``keys`` if given."""
        self._validate_states(states)
        # Validate everything first: a malformed later layer must not leave a partial reset.
        selected = None if keys is None else ((keys,) if isinstance(keys, str) else tuple(keys))
        for b, s in zip(self.layers, states):
            b.validate_state(s)
            if selected is not None:
                unknown = set(selected) - s.keys()
                if unknown:
                    raise ValueError(f"unknown state keys: {sorted(unknown)}")
        for b, s in zip(self.layers, states):
            b.reset_state(s, mask, keys=selected)
        return states

    def set_state(self, states, others):
        """In place: load the per-layer ``others`` into ``states``."""
        self._validate_states(states)
        self._validate_states(others, "other state")
        for b, other in zip(self.layers, others):
            b.validate_state(other)
        for b, s, o in zip(self.layers, states, others):
            b.set_state(s, o)
        return states

    def get_state(self):
        """The per-layer states of the most recent forward/step call."""
        return [b.get_state() for b in self.layers]


class CausalLM(CurrentStateDict, ExecutionMixin, Configurable, nn.Module):
    """Causal language model: Embedding -> n x ResidualBlock -> RMSNorm -> LM head.

    ``forward(input_ids) -> logits [B, L, vocab_size]``; ``step`` / ``generate`` decode
    token by token from the per-layer state.  Every block is ``SSM layer (+ optional SwiGLU MLP)``
    (the Mamba convention); the layers are described by ``layers``.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    layers : <Family>Config, dict, or a sequence of them
        The layers.  A single `SSDMimoConfig` / `HybridMambaConfig` (or a registered family's config, or a dict) builds
        ``n_layers`` identical layers; a sequence (any `collections.abc.Sequence`) gives one entry per
        layer, so stacks may be heterogeneous.  An entry is a family config object or a dict.  A dict
        must carry ``"type"``, the family key (``"ssd_mimo"``, ``"hybrid_mamba"``, or another registered family) or a name
        added with `register_mixer`; the other keys are the family config's fields (strict: unknown
        or mistyped keys are a ``TypeError``) or, for a registered foreign mixer, the options its
        factory receives.
    n_layers : int, optional
        Number of layers.  Required with a single ``layers`` entry; with a sequence it is the
        sequence length (when also given it must match).
    d_model : int, optional
        Model width.  Taken from the layer configs when they all agree on it; given in both places it must
        be equal; given nowhere it is an error.  The model fills it into the configs that leave it None
        (their ``d_inner`` / ``expand`` then give the inner width).
    d_mlp : int or sequence of int, default 0
        SwiGLU hidden width of the per-block MLP sublayer: one int for every block, or one entry per
        layer (0 = no MLP in that block; the length must match the layer count).  A block with an MLP is
        ``RMSNorm -> SSM -> +res; RMSNorm -> SwiGLU -> +res`` and counts two residual branches in the init
        depth.
    execution : Execution, dict or None, default None
        How the layers run; None keeps each layer's own defaults.  Applied to every layer.
    tie_word_embeddings : bool, default True
        Tie the LM head weight to the embedding weight.
    pad_vocab_size_multiple : int, default 8
        Pad the internal embedding/head matrix to this multiple for efficient GEMMs.
        Logits are sliced back to ``vocab_size``, so padded ids are never sampled
        or trained.

    Attributes
    ----------
    ssm_layers : tuple of nn.Module
        The layer modules (the blocks' mixers) in order, for addressing one layer:
        ``lm.ssm_layers[3].set_execution(...)``.
    backbone : BlockStack
        The blocks plus the final norm.

    Execution
    ---------
    How a model runs (``backend``, ``precision``, ``chunk_size``, ``checkpoint``, ``kernel_options``)
    is an `Execution`, or its dict, never part of the architecture.

    * At construction: ``CausalLM(..., execution={"backend": "torch"})`` applies it to every layer.
    * At runtime: ``lm.set_execution({"precision": "ieee"})`` changes the live model (only the
      fields given change).
    * Temporarily: ``with lm.execution({"backend": "torch"}): y = lm(x)`` restores the previous
      settings on exit.
    * One layer: ``lm.ssm_layers[3].set_execution(...)`` or ``with lm.ssm_layers[3].execution(...):``.

    A change clears the caches that depend on it; CUDA graphs captured by `orbit_ssm.graphs`
    notice it and refuse to replay (re-capture them).

    Examples
    --------
    >>> from orbit_ssm import CausalLM, SSDMimoConfig
    >>> lm = CausalLM(vocab_size=32000, layers=SSDMimoConfig(d_state=64, d_head=64), n_layers=8, d_model=512, d_mlp=1536)
    """

    config_schema = "orbit_ssm.model/v4"
    config_family = "causal_lm"

    def __init__(
        self,
        *,
        vocab_size: int,
        layers,
        n_layers: int | None = None,
        d_model: int | None = None,
        d_mlp=0,
        execution=None,
        tie_word_embeddings: bool = True,
        pad_vocab_size_multiple: int = 8,
    ):
        super().__init__()
        execution = None if execution is None else as_execution(execution)

        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if pad_vocab_size_multiple < 1:
            raise ValueError(
                "pad_vocab_size_multiple must be positive, got "
                f"{pad_vocab_size_multiple}"
            )
        specs = self._layer_specs(layers, n_layers)
        n_layers = len(specs)
        d_model = self._resolve_d_model(specs, d_model)
        d_mlps = _per_layer_widths(d_mlp, n_layers)

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padded_vocab_size = (
            (vocab_size + pad_vocab_size_multiple - 1)
            // pad_vocab_size_multiple
            * pad_vocab_size_multiple
        )
        self.n_layers = n_layers
        #: as given: one int for every block, or a tuple with one entry per layer
        self.d_mlp = tuple(d_mlp) if isinstance(d_mlp, Sequence) else d_mlp
        self.tie_word_embeddings = tie_word_embeddings

        # ---- embedding ----
        self.embedding = nn.Embedding(self.padded_vocab_size, d_model)

        # ---- backbone: potentially heterogeneous complete blocks ----
        mixers: list[BaseMixer] = [self._build_spec(d_model, kind, spec, execution)
                                   for kind, spec in specs]

        # The stack applies the residual-depth rescale itself; the LM must not repeat it.
        self.backbone = BlockStack.from_mixers(mixers, d_model, d_mlp=d_mlps)

        # ---- LM head ----
        self.lm_head = nn.Linear(d_model, self.padded_vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()
        if execution is not None:
            self.set_execution(execution)
        self._resolved_constructor_config = None
        if all(isinstance(m, Configurable) for m in mixers):
            recipes = [m.to_recipe() for m in mixers]
            self._resolved_constructor_config = dict(
                schema=self.config_schema, semantics_version=SEMANTICS_VERSION,
                family=self.config_family,
                model=dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           d_mlp=list(d_mlp) if isinstance(d_mlp, Sequence) else d_mlp,
                           tie_word_embeddings=tie_word_embeddings,
                           pad_vocab_size_multiple=pad_vocab_size_multiple,
                           initialization="orbit_ssm.lm/v0",
                           layers=[dict(family=r["family"], model=r["model"]) for r in recipes]),
                execution={})

    # ---------------------------------------------------------------- layer specs
    @staticmethod
    def _layer_specs(layers, n_layers) -> list:
        """``layers`` as a list of ``(type, spec)``, one per layer: ``spec`` is a family config
        object (built-in types) or the options dict of a registered foreign mixer."""
        from .ssm import CONFIGS
        by_class = {cls: name for name, cls in CONFIGS.items()}

        def entry(value, where):
            if isinstance(value, tuple(by_class)):
                return by_class[type(value)], value
            if isinstance(value, dict):
                if "type" not in value:
                    raise TypeError(f"{where}: a dict needs a 'type' key (one of "
                                    f"{sorted(CONFIGS) + sorted(_MIXER_REGISTRY)})")
                options = dict(value)
                kind = options.pop("type")
                if kind in _MIXER_REGISTRY:
                    return kind, options
                if kind not in CONFIGS:
                    raise ValueError(f"{where}: unknown type {kind!r}; supported: {sorted(CONFIGS)}"
                                     f" + registered {sorted(_MIXER_REGISTRY)}")
                return kind, CONFIGS[kind].from_dict(options)
            raise TypeError(f"{where} must be a family config or a dict with a 'type' key, "
                            f"got {type(value).__name__}")

        if isinstance(layers, (dict, *CONFIGS.values())):
            if n_layers is None:
                raise TypeError("n_layers is required with a single layers entry")
            if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
                raise ValueError(f"n_layers must be positive, got {n_layers}")
            kind, spec = entry(layers, "layers")
            return [(kind, spec)] * n_layers
        if isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
            if len(layers) < 1:
                raise ValueError("layers must not be empty")
            if n_layers is not None and n_layers != len(layers):
                raise ValueError(f"n_layers={n_layers} does not match the {len(layers)} "
                                 "layers entries")
            return [entry(v, f"layers[{i}]") for i, v in enumerate(layers)]
        raise TypeError("layers must be a family config, a dict, or a sequence of them, "
                        f"got {type(layers).__name__}")

    @staticmethod
    def _declared_d_model(kind, spec):
        return spec.get("d_model") if isinstance(spec, dict) else spec.d_model

    @classmethod
    def _resolve_d_model(cls, specs, d_model) -> int:
        declared = {cls._declared_d_model(kind, spec) for kind, spec in specs} - {None}
        if len(declared) > 1:
            raise ValueError(f"the layer configs disagree on d_model: {sorted(declared)}")
        if d_model is not None and declared and declared != {d_model}:
            raise ValueError(f"d_model={d_model} conflicts with the layer configs' d_model={declared.pop()}")
        d_model = d_model if d_model is not None else (declared.pop() if declared else None)
        if d_model is None:
            raise ValueError("d_model is not set: give it to CausalLM or in the layer configs")
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 1:
            raise ValueError(f"d_model must be a positive int, got {d_model!r}")
        return d_model

    @property
    def layer_configs(self) -> tuple:
        """Read-only: the resolved family config of each layer (a fresh copy per access; ``None`` for a layer
        that is not one of our SSM layers, e.g. attention)."""
        return tuple(getattr(m, "config", None) for m in self.backbone.ssm_layers)

    @property
    def ssm_layers(self) -> tuple:
        """The layer modules (the blocks' mixers) in order (read-only):
        ``lm.ssm_layers[3].set_execution(...)``, ``with lm.ssm_layers[3].execution(...):``."""
        return self.backbone.ssm_layers

    def to_recipe(self):
        """The full recipe of this model as a dict: every layer's resolved architecture and execution.
        `CausalLM.from_recipe` rebuilds the model from it (``execution=`` overrides the recorded per-layer
        executions: one Execution for every layer, or a list of one per layer; it replaces them as a whole)."""
        if self._resolved_constructor_config is None:
            raise NotImplementedError("resolved model recipes currently require our SSM layers")
        document = copy.deepcopy(self._resolved_constructor_config)
        layers = [b.mixer.to_recipe() for b in self.backbone.layers]
        document["model"]["layers"] = [dict(family=r["family"], model=r["model"]) for r in layers]
        document["execution"] = dict(layer_execution=[r["execution"] for r in layers])
        return document

    @classmethod
    def _from_recipe_sections(cls, model, execution, override=None):
        from .ssm import MIXERS, CONFIGS
        by_family = {m.config_family: name for name, m in MIXERS.items()}
        if model.pop("initialization") != "orbit_ssm.lm/v0":
            raise ValueError("unsupported LM initialization recipe")
        layers = model.pop("layers")
        runtime = execution.pop("layer_execution")
        if execution or len(layers) != model["n_layers"] or len(runtime) != len(layers):
            raise ValueError("model recipe must specify exactly one recipe per layer")
        if override is not None:
            runtime = layer_executions(override, len(layers), "CausalLM.from_recipe")
        configs = []
        for layer in layers:
            family = layer["family"]
            if family not in by_family:
                raise ValueError(f"unsupported model recipe layer family {family!r}")
            kwargs = dict(layer["model"])
            if kwargs.get("d_model") != model["d_model"]:
                raise ValueError("layer width conflicts with model d_model")
            configs.append(CONFIGS[by_family[family]].from_dict(kwargs))
        lm = cls(layers=configs, **model)
        for layer, settings in zip(lm.ssm_layers, runtime):
            layer.set_execution(settings)
        return lm

    # --------------------------------------------------------------- registry
    @classmethod
    def register_mixer(cls, name: str, factory) -> None:
        """Register ``name`` as a layer type built by ``factory``.

        A ``layers`` dict ``{"type": name, **options}`` calls
        ``factory(d_model=<model width>, **options)``; it returns a BaseMixer-compatible module
        (``forward(x[B, L, D]) -> y[B, L, D]``) and must tag its own residual
        output projection.  Built-in family names are reserved and raise
        ``ValueError``, so an experiment cannot shadow them.
        """
        from .ssm import CONFIGS
        if not isinstance(name, str) or not name:
            raise ValueError(f"mixer name must be a non-empty string, got {name!r}")
        if name in CONFIGS:
            raise ValueError(
                f"{name!r} is a built-in SSM family and cannot be re-registered")
        if not callable(factory):
            raise TypeError(f"factory for {name!r} must be callable")
        _MIXER_REGISTRY[name] = factory

    @classmethod
    def registered_mixers(cls) -> tuple[str, ...]:
        """Names currently served by the registry (sorted)."""
        return tuple(sorted(_MIXER_REGISTRY))

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build_spec(d_model: int, kind: str, spec, execution) -> nn.Module:
        """One layer: the family layer of config ``spec`` at width ``d_model``, or a registered factory."""
        factory = _MIXER_REGISTRY.get(kind)
        if factory is not None:
            return factory(**{**spec, "d_model": d_model})
        from .ssm import MIXERS
        return MIXERS[kind](spec.with_d_model(d_model), execution=execution)

    @staticmethod
    def _build_mixer(d_model: int, layer_config, execution=None) -> nn.Module:
        """One bare layer as a ``layers`` entry would build it (a family config, or a dict with
        ``"type"``), at width ``d_model``; for the benchmarks and tests that need a single mixer."""
        (kind, spec), = CausalLM._layer_specs(layer_config, 1)
        return CausalLM._build_spec(d_model, kind, spec, execution)

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        """Embedding N(0, 0.02), untied head trunc-normal(0.02), biases zeroed (except ``_no_reinit``).

        SSM-specific init stays as each core set it; residual output projections
        keep the `BlockStack` depth-rescaled init.
        """
        init_lm(self, embedding_std=0.02)
        if not self.tie_word_embeddings:
            nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

    def _project_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Project through the padded head and expose only real token logits."""
        return self.lm_head(x)[..., : self.vocab_size]

    # ------------------------------------------------------------------ forward
    def encode(self, input_ids: torch.Tensor, *, state=None,
              return_state: bool = False) -> torch.Tensor:
        """Final normalized hidden states ``(B, L, d_model)``, before the LM head."""
        x = self.embedding(input_ids)
        if state is None and not return_state:
            return self.backbone(x)
        y, new_state = self.backbone(x, state=state, return_state=True)
        return (y, new_state) if return_state else y

    def forward(
        self, input_ids: torch.Tensor, return_embeddings: bool = False,
        *, state=None, return_state: bool = False,
    ) -> torch.Tensor:
        """Full-sequence forward.

        Parameters
        ----------
        input_ids : (B, L) long tensor
        return_embeddings : bool, default False
            Return the final hidden states ``(B, L, d_model)`` instead of logits.
        state : list[State] or None
            One entry per layer, from :meth:`init_state` or a previous call with
            ``return_state=True``; None starts fresh.
        return_state : bool, default False
            Also return the per-layer state list.

        Returns
        -------
        logits : (B, L, vocab_size)
            Or ``(logits, state)`` when ``return_state``.
        """
        if state is None and not return_state:
            x = self.encode(input_ids)
            return x if return_embeddings else self._project_logits(x)
        x, new_state = self.encode(input_ids, state=state, return_state=True)
        out = x if return_embeddings else self._project_logits(x)
        return (out, new_state) if return_state else out

    # ------------------------------------------------------------------ step
    def step(
        self,
        input_ids_single: torch.Tensor,
        state: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Single-token autoregressive step.

        Parameters
        ----------
        input_ids_single : (B,) or (B, 1)
            Token ids for a single time step.
        state : list[State]
            Per-layer state from :meth:`init_state` or a previous step.
            Omit to start from the documented initial state.

        Returns
        -------
        logits : (B, vocab_size)
        states : list[State]
            Updated per-layer state.
        """
        ids = input_ids_single.squeeze(-1) if input_ids_single.ndim == 2 else input_ids_single
        x = self.embedding(ids)
        x, states = self.backbone.step(x, state)
        return self._project_logits(x), states

    # ------------------------------------------------------------------ state
    def init_state(self, batch_size: int, **kwargs) -> list:
        """Allocate the per-layer states (the documented initial state) for ``batch_size``."""
        return self.backbone.init_state(batch_size, **kwargs)

    def reset_state(self, states: list, mask=None, keys=None) -> list:
        """In place: reset `states` for the batch rows `mask` selects (default all), e.g. at episode boundaries."""
        return self.backbone.reset_state(states, mask, keys=keys)

    def set_state(self, states: list, others: list) -> list:
        """In place: load `others` into `states`."""
        return self.backbone.set_state(states, others)

    def get_state(self) -> list:
        """The per-layer state produced by the most recent forward/step
        call (``return_state=True`` or :meth:`step`)."""
        return self.backbone.get_state()

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation loop.

        The prompt is prefilled in one ``forward(..., return_state=True)`` pass, then decoded token by token
        with :meth:`step`.

        Parameters
        ----------
        input_ids : (B, L)
            Prompt token ids: a non-empty ``torch.long`` tensor with values in ``[0, vocab_size)``.
        max_new_tokens : int
            Number of new tokens to generate (>= 0; 0 returns the prompt).
        temperature : float
            Sampling temperature, finite and >= 0.  0 = greedy (argmax).
        top_k : int or None
            If set, a positive int: restrict sampling to the top-k logits.

        Returns
        -------
        ids : (B, L + max_new_tokens)
            Concatenated prompt + generated token ids.

        Raises
        ------
        TypeError
            For an argument of the wrong type.
        ValueError
            For a value out of range (ids outside the vocabulary, negative ``max_new_tokens`` or
            ``temperature``, non-positive ``top_k``) or an input of the wrong shape.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"generate: input_ids must be a torch.Tensor, got {type(input_ids).__name__}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"generate: input_ids must be a torch.long tensor, got {input_ids.dtype}")
        if input_ids.ndim != 2:
            raise ValueError(f"generate: input_ids must be 2-D (B, L), got shape {tuple(input_ids.shape)}")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError(f"generate: input_ids must hold at least one prompt token per row, "
                             f"got shape {tuple(input_ids.shape)}")
        lo, hi = int(input_ids.min()), int(input_ids.max())
        if lo < 0 or hi >= self.vocab_size:
            raise ValueError(f"generate: input_ids must lie in [0, {self.vocab_size}), got values in [{lo}, {hi}]")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError(f"generate: max_new_tokens must be an int, got {max_new_tokens!r}")
        if max_new_tokens < 0:
            raise ValueError(f"generate: max_new_tokens must be >= 0, got {max_new_tokens}")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError(f"generate: temperature must be a number, got {temperature!r}")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(f"generate: temperature must be finite and >= 0 (0 = greedy), got {temperature}")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError(f"generate: top_k must be an int or None, got {top_k!r}")
            if top_k < 1:
                raise ValueError(f"generate: top_k must be >= 1 or None, got {top_k}")
        if max_new_tokens == 0:
            return input_ids.clone()

        # ---- process prompt ----
        logits, states = self(input_ids, return_state=True)
        logits = logits[:, -1]

        # ---- generate ----
        generated: list[torch.Tensor] = []
        for i in range(max_new_tokens):
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k is not None:
                    k = min(top_k, scaled.size(-1))
                    v, _ = torch.topk(scaled, k)
                    scaled[scaled < v[:, -1:]] = float("-inf")
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, 1)       # [B, 1]
            generated.append(next_token)
            if i + 1 < max_new_tokens:
                logits, states = self.step(next_token, states)

        return torch.cat(
            [input_ids, torch.cat(generated, dim=1)], dim=1,
        )
