"""Unified carry state, lifecycle tags, and abstract layer contracts."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from ..execution import ExecutionMixin


def _state_map(value, fn):
    """Recurse through a State/dict/list/tuple, applying `fn` to every
    tensor leaf; every other leaf (int counters, None, ...) passes
    through unchanged."""
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, State):
        return State({k: _state_map(v, fn) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _state_map(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_state_map(v, fn) for v in value)
    return value


# Named phase carries share a minimum accumulation precision across families.
_PHASE_KEYS = frozenset({"rot_ang", "theta", "c3_phase"})


class State(dict):
    """A recurrent carry: a dict of tensors (plus the occasional int counter or
    nested State) that moves between devices/dtypes as one object.

    It is a plain ``dict`` subclass, so ``state["hidden_state"]``, ``**state``
    and ``dict(state)`` work unchanged and it pickles / ``torch.save``s as is.

    Keys depend on the mixer. ``"hidden_state"`` is the scan state, batched on
    axis 0; other carries (conv window ``conv_state``, rotation phases
    ``rot_ang`` / ``theta`` / ``c3_phase``, ...) are mixer-specific. Phase
    carries are kept at least fp32 by `to` and rejected below fp32 by
    `StateLifecycle.validate_state`. Create states with
    ``mixer.init_state(B, device=, dtype=, h_init=, overrides=)``.
    """

    @property
    def hidden_state(self) -> torch.Tensor:
        """The original graph-bearing hidden tensor, without copying."""
        return self["hidden_state"]

    @classmethod
    def from_hidden(cls, module, hidden, *, batch_size=None, device=None, dtype=None):
        """Build a complete initial state from a hidden seed, filling the other
        carries from `module.init_state`.

        A batched seed ``[B, *hidden_shape]`` infers B; an unbatched seed
        requires `batch_size`. For a multi-layer model use
        ``model.init_state(B, h_init=[...])`` instead.
        """
        if not torch.is_tensor(hidden):
            raise TypeError("hidden must be a tensor")
        if batch_size is None:
            schema = module.init_state(1, device="meta", dtype=hidden.dtype)
            if not isinstance(schema, dict) or "hidden_state" not in schema:
                raise TypeError("use model.init_state(B, h_init=[...]) for a multi-layer model")
            shape = schema["hidden_state"].shape[1:]
            if tuple(hidden.shape) == tuple(shape):
                raise ValueError("an unbatched hidden seed requires batch_size")
            if tuple(hidden.shape[1:]) != tuple(shape):
                raise ValueError(f"hidden must have shape [B, {tuple(shape)}]")
            batch_size = hidden.shape[0]
        return module.init_state(batch_size, h_init=hidden,
                                 device=hidden.device if device is None else device,
                                 dtype=hidden.dtype if dtype is None else dtype)

    def to(self, device=None, dtype=None) -> "State":
        """Move/cast every tensor. `dtype` applies to floating tensors only
        (integer dtypes are kept), and phase carries never drop below fp32."""
        def move(key, value):
            if isinstance(value, dict):
                mapped = {k: move(k, v) for k, v in value.items()}
                return State(mapped) if isinstance(value, State) else mapped
            if isinstance(value, (list, tuple)):
                return type(value)(move(key, v) for v in value)
            def cast(t):
                target = dtype if dtype is not None and t.is_floating_point() else t.dtype
                if key in _PHASE_KEYS and target in (torch.float16, torch.bfloat16):
                    target = torch.float32
                return t.to(device=device if device is not None else t.device, dtype=target)
            return _state_map(value, cast)
        return State({key: move(key, value) for key, value in self.items()})

    def detach(self) -> "State":
        """A copy with every tensor detached from the autograd graph."""
        return State(_state_map(self, lambda t: t.detach()))

    def clone(self) -> "State":
        """A copy with every tensor cloned."""
        return State(_state_map(self, lambda t: t.clone()))

    def index_select(self, indices):
        """Reorder/select the batch of every carry; repeated indices accumulate gradients."""
        if not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a one-dimensional int64 tensor")
        return State(_state_map(self, lambda t: t.index_select(0, indices.to(t.device)) if t.ndim else t))

    def to_dict(self) -> dict:
        """The state as a plain dict (same tensors, no copy)."""
        return dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        """Wrap a plain dict as a State (no copy)."""
        return cls(d)


def _first_tensor(value):
    if torch.is_tensor(value) and value.ndim:
        return value
    if isinstance(value, dict):
        for item in value.values():
            ref = _first_tensor(item)
            if ref is not None:
                return ref
    return None


def _check_tree(value, expected, *, batch_size, device=None, path="state"):
    """Validate structure and shape without copying or inspecting tensor values."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(value):
            raise TypeError(f"{path} must be a tensor")
        shape = ((batch_size, *expected.shape[1:]) if expected.ndim else ())
        if tuple(value.shape) != shape:
            raise ValueError(f"{path}: expected shape {shape}, got {tuple(value.shape)}")
        if value.is_floating_point() != expected.is_floating_point():
            raise TypeError(f"{path}: incompatible dtype {value.dtype}")
        if not expected.is_floating_point() and value.dtype != expected.dtype:
            raise TypeError(f"{path}: expected dtype {expected.dtype}")
        if device is not None and value.device != torch.device(device):
            raise ValueError(f"{path}: expected device {device}, got {value.device}")
    elif isinstance(expected, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be a dict")
        missing, extra = expected.keys() - value.keys(), value.keys() - expected.keys()
        if missing or extra:
            raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
        for key in expected:
            if key in _PHASE_KEYS and torch.is_tensor(value[key]) and value[key].dtype in (torch.float16, torch.bfloat16):
                raise TypeError(f"{path}.{key}: phase carries require at least float32")
            _check_tree(value[key], expected[key], batch_size=batch_size,
                        device=device, path=f"{path}.{key}")
    elif type(value) is not type(expected):
        raise TypeError(f"{path}: expected {type(expected).__name__}")


def _copy_state_inplace(dst, src, mask=None):
    """Replace container entries; never mutate graph-bearing tensor storage."""
    if mask is None:
        replacement = _state_map(src, lambda t: t.clone())
        dst.clear()
        dst.update(replacement)
        return dst
    for key, fresh in src.items():
        current = dst[key]
        if isinstance(fresh, dict):
            dst[key] = _copy_state_inplace(dict(current), fresh, mask)
        elif torch.is_tensor(fresh) and fresh.ndim:
            value = current.clone()
            value[mask] = fresh[mask].to(current.dtype)
            dst[key] = value
        # Scalar/global leaves cannot be reset for selected batch rows.
    return dst


def _default_reset_state(module, state, mask=None, keys=None):
    ref = _first_tensor(state.get("hidden_state", state))
    if ref is None:
        raise ValueError("state has no batch-resolved tensor")
    selected = tuple(state) if keys is None else ((keys,) if isinstance(keys, str) else tuple(keys))
    unknown = set(selected) - state.keys()
    if unknown:
        raise ValueError(f"unknown state keys: {sorted(unknown)}")
    if mask is not None:
        if not torch.is_tensor(mask) or mask.ndim != 1 or mask.dtype not in (torch.bool, torch.long):
            raise ValueError("mask must be a one-dimensional bool mask or int64 batch indices")
        if mask.dtype == torch.bool and mask.numel() != ref.shape[0]:
            raise ValueError("boolean mask length must equal batch size")
        mask = mask.to(ref.device)
    fresh = module.init_state(ref.shape[0], device=ref.device, dtype=ref.dtype)
    selected_fresh = {key: fresh[key] for key in selected}
    if keys is None:
        return _copy_state_inplace(state, selected_fresh, mask)
    updates = _copy_state_inplace({key: state[key] for key in selected}, selected_fresh, mask)
    state.update(updates)
    return state


def _default_set_state(state, other):
    return _copy_state_inplace(state, other)


def _default_get_state(module):
    final = getattr(module, "_final_state", None)
    if final is None:
        raise RuntimeError(f"{type(module).__name__}: no state recorded; call step or forward(return_state=True)")
    return final


class StateLifecycle:
    """Shared state lifecycle (allocate, validate, reset, load, read).

    A family implements only ``_init_state`` (and optionally
    ``_default_state_dtype``); the public methods below are shared.
    """

    def _state_schema(self):
        schema = getattr(self, "_cached_state_schema", None)
        if schema is None:
            with torch.no_grad():
                schema = self._init_state(1, device="meta", dtype=torch.float32)
            self._cached_state_schema = schema
        return schema

    def validate_state(self, state, *, batch_size=None, device=None):
        """Check `state` against the module's schema (keys, shapes, dtype class,
        device, phases at least fp32) without reading tensor values; returns it.
        `batch_size` and `device` default to those of the state itself."""
        ref = _first_tensor(state.get("hidden_state", state)) if isinstance(state, dict) else None
        if ref is None:
            raise ValueError("state must contain a batch-resolved tensor")
        batch_size = ref.shape[0] if batch_size is None else batch_size
        device = ref.device if device is None else device
        _check_tree(state, self._state_schema(), batch_size=batch_size, device=device)
        return state

    def _default_state_dtype(self):
        """Carry dtype when the caller passes none: the parameter dtype, never below fp32 (every family)."""
        p = next(self.parameters(), None)
        return torch.float32 if p is None else torch.promote_types(p.dtype, torch.float32)

    def _conform_carries(self, out, ref=None):
        """The dtype law of returned carries (one rule, every family): each floating carry of `out` has the dtype
        of the same carry of `ref` (the State the call was given), or ``_default_state_dtype()`` when there is no
        `ref` or `ref` lacks the key; integer / bool leaves and non-tensors pass through.  Only the dtype changes
        (``.to`` keeps autograd); a State whose dtypes drifted between calls would recompile ``torch.compile`` and
        break static CUDA-graph buffers.  Returns `out` itself when nothing needs a cast, else a new State."""
        default = self._default_state_dtype()

        def conform(value, want):
            if torch.is_tensor(value):
                if not value.is_floating_point():
                    return value
                if torch.is_tensor(want) and want.is_floating_point():
                    target = want.dtype
                elif torch.is_tensor(want):
                    return value                      # incoming integer leaf: not a carry dtype question
                else:
                    target = default
                return value if value.dtype == target else value.to(target)
            if isinstance(value, dict):
                sub = want if isinstance(want, dict) else {}
                mapped = {k: conform(v, sub.get(k)) for k, v in value.items()}
                if all(mapped[k] is value[k] for k in value):
                    return value
                return State(mapped) if isinstance(value, State) else mapped
            return value

        return conform(out, ref if isinstance(ref, dict) else None)

    def init_state(self, batch_size, device=None, dtype=None, *, h_init=None, overrides=None):
        """Allocate a fresh State for `batch_size` rows.

        Parameters
        ----------
        batch_size : int
            Nonnegative number of batch rows.
        device, dtype : optional
            Carry placement; `dtype` defaults to ``_default_state_dtype()``
            (fp32 unless a family overrides it).
        h_init : Tensor, optional
            Initial ``hidden_state``, either batched ``[B, *shape]`` or a
            single unbatched seed broadcast over the batch. Gradients flow
            into it. Mutually exclusive with ``overrides["hidden_state"]``.
        overrides : dict, optional
            Replacement tensors for other state keys; each is validated
            against the schema and cast to the schema's dtype and device.
        """
        if not isinstance(batch_size, int) or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        if dtype is None:
            dtype = self._default_state_dtype()
        state = self._init_state(batch_size, device=device, dtype=dtype)
        updates = dict(overrides or {})
        if h_init is not None:
            if "hidden_state" in updates:
                raise ValueError("pass h_init or overrides['hidden_state'], not both")
            expected = state.get("hidden_state")
            if expected is None:
                raise ValueError("this module has no hidden_state")
            if torch.is_tensor(h_init) and tuple(h_init.shape) == tuple(expected.shape[1:]):
                h_init = h_init.unsqueeze(0).expand(batch_size, *h_init.shape)
            updates["hidden_state"] = h_init
        unknown = updates.keys() - state.keys()
        if unknown:
            raise ValueError(f"unknown state overrides: {sorted(unknown)}")
        for key, value in updates.items():
            _check_tree(value, state[key], batch_size=batch_size, path=f"state.{key}")
            # Conversion preserves autograd, including learned/custom initial states.
            target = state[key]
            if torch.is_tensor(value):
                value = value.to(device=target.device, dtype=target.dtype)
            state[key] = value
        return State(state)

    def resolve_state(self, state, *, batch_size, device, dtype, h_init=None):
        """Validate a given `state`, or allocate one via `init_state` when None.  An allocated State has the
        default carry dtype (``_default_state_dtype()``), never the activation dtype: `dtype` is accepted for the
        callers' signature only, so a layer's carries are the same with and without a State passed."""
        if state is not None:
            if h_init is not None:
                raise ValueError("pass h_init OR state, not both")
            return self.validate_state(state, batch_size=batch_size, device=device)
        return self.init_state(batch_size, device=device, h_init=h_init)

    def reset_state(self, state, mask=None, keys=None):
        """Reset `state` in place to its fresh value.

        `mask` is a 1-D bool mask (length B) or int64 batch indices selecting
        rows (None = all rows); `keys` a key or keys to reset (None = all).
        Scalar leaves cannot be reset per row. Returns `state`.
        """
        self.validate_state(state)
        return _default_reset_state(self, state, mask, keys)

    def set_state(self, state, other):
        """Load the (validated) state `other` into `state` in place; returns `state`."""
        self.validate_state(other)
        return _default_set_state(state, other)

    def get_state(self):
        """The carry recorded by the last ``step`` or ``forward(return_state=True)``."""
        return _default_get_state(self)


class ScanState(StateLifecycle):
    """State lifecycle of the bare scan (no drive conv), for a layer's `scan` /
    `scan_step`; the layer's own `init_state` / `validate_state` describe the
    full layer carry, which also holds `conv_state`."""

    def __init__(self, owner):
        self._owner = owner

    def _init_state(self, *args, **kwargs):
        return self._owner._scan_init_state(*args, **kwargs)

    def _default_state_dtype(self):
        return self._owner._default_state_dtype()

    def _state_schema(self):
        schema = getattr(self._owner, "_cached_scan_schema", None)
        if schema is None:
            with torch.no_grad():
                schema = self._init_state(1, device="meta", dtype=torch.float32)
            self._owner._cached_scan_schema = schema
        return schema


# Support status registry: production = actively developed; maintained = correct,
# feature-frozen; legacy = reproducibility only.  Never changes numerics.
SSM_STATUS: dict[str, str] = {}


def ssm_status(status: str, **meta):
    """Class decorator tagging a core's lifecycle status.

    `status` is one of "production", "maintained", "legacy", "experimental".
    Optional `**meta` (e.g. the kernel a core dispatches to, and why it has
    this status) is stored on the class as `__ssm_meta__`; the status as
    `__ssm_status__` and in `SSM_STATUS`.
    """
    assert status in ("production", "maintained", "legacy", "experimental")

    def deco(cls):
        SSM_STATUS[cls.__name__] = status
        cls.__ssm_status__ = status
        cls.__ssm_meta__ = dict(meta)
        return cls
    return deco



class BaseMixer(ExecutionMixin, StateLifecycle, nn.Module):
    """Abstract base for all mixer layers.

    Contract:  in_proj → [prologue] → conv / SSM / gate → epilogue → out_proj

    Subclasses own their full projection stack (in_proj through out_proj).
    The backbone treats every mixer through the unified state API:

        y            = mixer(x)                     # full-sequence, no carry
        y, state     = mixer(x, state=s, return_state=True)  # carried / carry-returning
        y_t, state   = mixer.step(x_t, state)        # single-token recurrence; optional state
        state        = mixer.init_state(B, device=, dtype=)  # allocate decode state
        mixer.reset_state(state, mask=None)          # in place; mask [B] resets selected rows
        mixer.set_state(state, other)                # in place: load `other` into `state`
        mixer.get_state()                            # the carry from the last forward/step

    SSDMimoSSM and the other SSM layers implement this contract.
    """

    # step() takes a no-graph, in-place-cache fast path when pinned with
    # set_inference(True) (what CUDA-Graph capture needs) or, unpinned, when
    # gradients are globally disabled. set_inference(False) forces the
    # differentiable path; set_inference(None) restores automatic dispatch.
    _inference_pin: bool | None = None

    def set_inference(self, pin: bool | None = True) -> None:
        """Pin `step` to the inference fast path (True), the differentiable path
        (False), or automatic dispatch on grad mode (None)."""
        self._inference_pin = pin

    def _fast_inference(self) -> bool:
        if self._inference_pin is not None:
            return self._inference_pin
        return not torch.is_grad_enabled()

    def forward(self, x: torch.Tensor, state=None, return_state: bool = False):
        """Map ``x`` (B, L, d_model) to ``y`` (B, L, d_model), or ``(y, State)`` with ``return_state=True``;
        ``state`` continues a previous call.  Subclasses implement it."""
        raise NotImplementedError

    def step(self, x_t: torch.Tensor, state) -> tuple[torch.Tensor, "State"]:
        """One token: ``x_t`` (B, d_model) and the carried ``state`` give ``(y_t, new State)``.
        Subclasses implement it."""
        raise NotImplementedError

    def _init_state(self, batch_size: int, device=None,
                   dtype: torch.dtype = torch.float32, **kwargs) -> "State":
        raise NotImplementedError





def read_alias(attr: str) -> property:
    """Read-only public name for an internal (paper-notation) attribute."""
    return property(lambda self: getattr(self, attr), doc=f"Read-only view of ``.{attr}``.")


class _ScanBase(StateLifecycle, nn.Module):
    """Private base of the scan cores (`_SSDMimoScan`, `_HybridMambaScan`
    and the other families): consumes post-projection activations, returns the scan
    output and carry state.

    Contract:
        y = scan(x)                             # ordinary stateless training
        y, state  = scan(x, state=s, return_state=True)  # unified state API
        y_t, state = scan.step(x_t, state)        # optional state
        scan.out_width                            # readout width attribute
    """

    out_width: int

    def forward(self, x, *, state=None, return_state: bool = False, **kwargs):
        """The scan over ``x``; ``(y, State)`` with ``return_state=True``.  Subclasses implement it."""
        raise NotImplementedError

    def step(self, x_t, state) -> tuple[torch.Tensor, "State"]:
        """One token of the scan: ``(y_t, new State)``.  Subclasses implement it."""
        raise NotImplementedError

    def hidden_state_shape(self) -> tuple[int, ...]:
        """The UNBATCHED hidden-state shape for this core."""
        raise NotImplementedError

    def empty_state(self, batch_size: int) -> torch.Tensor:
        """A zero hidden state with the batched state shape."""
        return next(self.parameters()).new_zeros(batch_size, *self.hidden_state_shape())

    def _base_initial_state(self) -> torch.Tensor | None:
        """The unbatched learned/default base state, if the core has one."""
        return None

    def base_seed(self) -> torch.Tensor:
        """The unbatched default start state, as a fresh tensor.

        A clone (mutating a state cannot corrupt the parameter) but not
        detached, so gradient still reaches a learnable `h_init`.
        """
        base = self._base_initial_state()
        if base is None:
            return next(self.parameters()).new_zeros(*self.hidden_state_shape())
        return base.clone()

    def initial_state(self, batch_size, h_init=None) -> torch.Tensor:
        """Resolve an initial state, batched or not.

        `h_init` may be unbatched (`hidden_state_shape()`) or already batched.
        `batch_size=None` returns the unbatched seed as-is, to be broadcast at
        the first forward/step.
        """
        state_shape = self.hidden_state_shape()
        if h_init is None:
            h_init = self.base_seed()
        if tuple(h_init.shape) == tuple(state_shape):
            if batch_size is None:
                return h_init
            return h_init.unsqueeze(0).expand(
                batch_size, *state_shape).contiguous()
        expected = (batch_size, *state_shape)
        if tuple(h_init.shape) != expected:
            raise ValueError(
                f"{type(self).__name__} h_init must have shape {state_shape} "
                f"or {expected}, got {tuple(h_init.shape)}.")
        return h_init.contiguous()

    def _init_state(self, batch_size: int, device=None,
                   dtype: torch.dtype = torch.float32, **kwargs) -> "State":
        """Default: a State holding only ``"hidden_state"``, seeded per
        :meth:`base_seed`. Cores with more carries override this."""
        h = self.initial_state(batch_size)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device,
                     dtype=dtype if dtype is not None else h.dtype)
        return State(hidden_state=h)


#: state-dict keys that only older checkpoints carry: the ``.ssm.`` module level of the layers (dropped by the API
#: unification) and the retired unbounded ``mu_gain``
_OLD_FORMAT_KEY = re.compile(r"(^|\.)ssm\.|(^|\.)mu_gain$")


class CurrentStateDict:
    """Mixin: ``load_state_dict`` takes the CURRENT key layout only.  A state dict in an older format raises a
    ValueError; old checkpoints have to be converted first, with the conversion tool of the release that wrote them."""

    def load_state_dict(self, state_dict, *args, **kwargs):
        """`torch.nn.Module.load_state_dict` for the current key layout only: a state dict in an older format
        raises ``ValueError`` (see the class docstring)."""
        own = self.state_dict().keys()
        old = [k for k in state_dict if isinstance(k, str) and k not in own and _OLD_FORMAT_KEY.search(k)]
        if old:
            raise_old_format(f"key {old[0]!r}")
        return super().load_state_dict(state_dict, *args, **kwargs)


def raise_old_format(what: str, *, hint: str = "") -> None:
    """The strict-format refusal of `CurrentStateDict`: ``what`` is the evidence (e.g. a key), ``hint`` an
    addition to the advice."""
    raise ValueError(
        f"state dict is in an older format ({what}); `load_state_dict` reads the current "
        "layout only; convert it with the conversion tool of the release it was written with{hint}")


class BaseSSM(CurrentStateDict, BaseMixer):
    """Shared base of the SSM layers (`SSDMimoSSM`, `HybridMambaSSM`
    and any registered family): a complete block-slot mixer (in_proj .. out_proj) that owns
    its scan core and applies `Execution` settings (`_apply_execution`).

        BaseMixer        block-slot interface (forward / step / init_state)
          BaseSSM        our SSM layers
        Llama3Attention  BaseMixer only (and foreign registered mixers)
        ResidualBlock    StateLifecycle + nn.Module (not a BaseMixer)
    """

    @property
    def config(self):
        """Read-only: the layer's RESOLVED family config (defaults filled; equal to
        ``Config.from_dict(layer.to_recipe()["model"])``).

        The family configs are plain mutable dataclasses, so every access returns a FRESH copy built from the
        recipe the layer recorded at construction: changing it changes nothing in the layer (rebuild with
        ``type(layer)(new_config)``).  It adds no state: not in ``state_dict``, and it survives ``torch.save`` /
        ``deepcopy`` because it is derived from the recorded recipe."""
        import copy
        from ..families import CONFIGS
        return CONFIGS[self.config_family].from_dict(copy.deepcopy(self._resolved_constructor_config["model"]))

    @property
    def is_siso(self) -> bool:
        """Read-only: ``d_ssm_in == d_ssm_out == 1`` (a single input and a single output per subsystem)."""
        return self.d_ssm_in == 1 and self.d_ssm_out == 1

    @property
    def is_mimo(self) -> bool:
        """Read-only: not `is_siso` (``d_ssm_in > 1`` or ``d_ssm_out > 1``).

        Includes the asymmetric cases MISO (``d_ssm_in > 1, d_ssm_out == 1``) and
        SIMO (``d_ssm_in == 1, d_ssm_out > 1``); check ``d_ssm_in``/``d_ssm_out``
        to tell them apart."""
        return not self.is_siso

    #: the `Execution` fields this family honours (anything else is refused by `refuse_unsupported`), in
    #: the order `_execution_state` reports them; ``kernel_options`` is kernel part + layer switches
    _execution_fields: tuple = ("backend", "precision", "chunk_size", "checkpoint", "kernel_options")
    #: field -> attribute, where the attribute is named differently
    _execution_attr: dict = {}
    #: field -> what to use instead, for the refusal message
    _execution_hints: dict = {}
    #: unsupported field -> the one value that is this family's fixed behaviour (accepted, not stored)
    _execution_fixed: dict = {}

    def _refuse_unsupported_execution(self, ex):
        from ..execution import refuse_unsupported
        refuse_unsupported(ex, type(self).__name__, self._execution_fields, self._execution_hints,
                           self._execution_fixed)

    def _execution_state(self):
        """The current values of the supported `Execution` fields (the same rule in every family)."""
        from ..execution import execution_state_options
        return {f: (execution_state_options(self) if f == "kernel_options"
                    else getattr(self, self._execution_attr.get(f, f)))
                for f in self._execution_fields}

    def _set_execution_values(self, out):
        """Store a validated ``_check_execution`` result on the attributes named by
        ``_layer_option_attrs`` (default: same name)."""
        for name, value in out.items():
            setattr(self, self._layer_option_attrs.get(name, name), value)

    def _apply_execution(self, ex):
        self._set_execution_values(self._check_execution(ex))
