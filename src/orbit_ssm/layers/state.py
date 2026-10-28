
from __future__ import annotations
import torch
import torch.nn as nn

def _state_map(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, State):
        return State({k: _state_map(v, fn) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _state_map(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)((_state_map(v, fn) for v in value))
    return value
_PHASE_KEYS = frozenset({'rot_ang', 'theta', 'c3_phase'})

class State(dict):

    @property
    def hidden_state(self) -> torch.Tensor:
        return self['hidden_state']

    @classmethod
    def from_hidden(cls, module, hidden, *, batch_size=None, device=None, dtype=None):
        if not torch.is_tensor(hidden):
            raise TypeError('hidden must be a tensor')
        if batch_size is None:
            schema = module.init_state(1, device='meta', dtype=hidden.dtype)
            if not isinstance(schema, dict) or 'hidden_state' not in schema:
                raise TypeError('use model.init_state(B, h_init=[...]) for a multi-layer model')
            shape = schema['hidden_state'].shape[1:]
            if tuple(hidden.shape) == tuple(shape):
                raise ValueError('an unbatched hidden seed requires batch_size')
            if tuple(hidden.shape[1:]) != tuple(shape):
                raise ValueError(f'hidden must have shape [B, {tuple(shape)}]')
            batch_size = hidden.shape[0]
        return module.init_state(batch_size, h_init=hidden, device=hidden.device if device is None else device, dtype=hidden.dtype if dtype is None else dtype)

    def to(self, device=None, dtype=None) -> 'State':

        def move(key, value):
            if isinstance(value, dict):
                mapped = {k: move(k, v) for k, v in value.items()}
                return State(mapped) if isinstance(value, State) else mapped
            if isinstance(value, (list, tuple)):
                return type(value)((move(key, v) for v in value))

            def cast(t):
                target = dtype if dtype is not None and t.is_floating_point() else t.dtype
                if key in _PHASE_KEYS and target in (torch.float16, torch.bfloat16):
                    target = torch.float32
                return t.to(device=device if device is not None else t.device, dtype=target)
            return _state_map(value, cast)
        return State({key: move(key, value) for key, value in self.items()})

    def detach(self) -> 'State':
        return State(_state_map(self, lambda t: t.detach()))

    def clone(self) -> 'State':
        return State(_state_map(self, lambda t: t.clone()))

    def index_select(self, indices):
        if not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError('indices must be a one-dimensional int64 tensor')
        return State(_state_map(self, lambda t: t.index_select(0, indices.to(t.device)) if t.ndim else t))

    def to_dict(self) -> dict:
        return dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'State':
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

def _check_tree(value, expected, *, batch_size, device=None, path='state'):
    if torch.is_tensor(expected):
        if not torch.is_tensor(value):
            raise TypeError(f'{path} must be a tensor')
        shape = (batch_size, *expected.shape[1:]) if expected.ndim else ()
        if tuple(value.shape) != shape:
            raise ValueError(f'{path}: expected shape {shape}, got {tuple(value.shape)}')
        if value.is_floating_point() != expected.is_floating_point():
            raise TypeError(f'{path}: incompatible dtype {value.dtype}')
        if not expected.is_floating_point() and value.dtype != expected.dtype:
            raise TypeError(f'{path}: expected dtype {expected.dtype}')
        if device is not None and value.device != torch.device(device):
            raise ValueError(f'{path}: expected device {device}, got {value.device}')
    elif isinstance(expected, dict):
        if not isinstance(value, dict):
            raise TypeError(f'{path} must be a dict')
        missing, extra = (expected.keys() - value.keys(), value.keys() - expected.keys())
        if missing or extra:
            raise ValueError(f'{path}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}')
        for key in expected:
            if key in _PHASE_KEYS and torch.is_tensor(value[key]) and (value[key].dtype in (torch.float16, torch.bfloat16)):
                raise TypeError(f'{path}.{key}: phase carries require at least float32')
            _check_tree(value[key], expected[key], batch_size=batch_size, device=device, path=f'{path}.{key}')
    elif type(value) is not type(expected):
        raise TypeError(f'{path}: expected {type(expected).__name__}')

def _copy_state_inplace(dst, src, mask=None):
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
    return dst

def _default_reset_state(module, state, mask=None, keys=None):
    ref = _first_tensor(state.get('hidden_state', state))
    if ref is None:
        raise ValueError('state has no batch-resolved tensor')
    selected = tuple(state) if keys is None else (keys,) if isinstance(keys, str) else tuple(keys)
    unknown = set(selected) - state.keys()
    if unknown:
        raise ValueError(f'unknown state keys: {sorted(unknown)}')
    if mask is not None:
        if not torch.is_tensor(mask) or mask.ndim != 1 or mask.dtype not in (torch.bool, torch.long):
            raise ValueError('mask must be a one-dimensional bool mask or int64 batch indices')
        if mask.dtype == torch.bool and mask.numel() != ref.shape[0]:
            raise ValueError('boolean mask length must equal batch size')
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
    final = getattr(module, '_final_state', None)
    if final is None:
        raise RuntimeError(f'{type(module).__name__}: no state recorded; call step or forward(return_state=True)')
    return final

class StateLifecycle:

    def _state_schema(self):
        schema = getattr(self, '_cached_state_schema', None)
        if schema is None:
            with torch.no_grad():
                schema = self._init_state(1, device='meta', dtype=torch.float32)
            self._cached_state_schema = schema
        return schema

    def validate_state(self, state, *, batch_size=None, device=None):
        ref = _first_tensor(state.get('hidden_state', state)) if isinstance(state, dict) else None
        if ref is None:
            raise ValueError('state must contain a batch-resolved tensor')
        batch_size = ref.shape[0] if batch_size is None else batch_size
        device = ref.device if device is None else device
        _check_tree(state, self._state_schema(), batch_size=batch_size, device=device)
        return state

    def _default_state_dtype(self):
        return torch.float32

    def init_state(self, batch_size, device=None, dtype=None, *, h_init=None, overrides=None):
        if not isinstance(batch_size, int) or batch_size < 0:
            raise ValueError('batch_size must be a nonnegative integer')
        if dtype is None:
            dtype = self._default_state_dtype()
        state = self._init_state(batch_size, device=device, dtype=dtype)
        updates = dict(overrides or {})
        if h_init is not None:
            if 'hidden_state' in updates:
                raise ValueError("pass h_init or overrides['hidden_state'], not both")
            expected = state.get('hidden_state')
            if expected is None:
                raise ValueError('this module has no hidden_state')
            if torch.is_tensor(h_init) and tuple(h_init.shape) == tuple(expected.shape[1:]):
                h_init = h_init.unsqueeze(0).expand(batch_size, *h_init.shape)
            updates['hidden_state'] = h_init
        unknown = updates.keys() - state.keys()
        if unknown:
            raise ValueError(f'unknown state overrides: {sorted(unknown)}')
        for key, value in updates.items():
            _check_tree(value, state[key], batch_size=batch_size, path=f'state.{key}')
            target = state[key]
            if torch.is_tensor(value):
                value = value.to(device=target.device, dtype=target.dtype)
            state[key] = value
        return State(state)

    def resolve_state(self, state, *, batch_size, device, dtype, h_init=None):
        if state is not None:
            if h_init is not None:
                raise ValueError('pass h_init OR state, not both')
            return self.validate_state(state, batch_size=batch_size, device=device)
        return self.init_state(batch_size, device=device, dtype=dtype, h_init=h_init)

    def reset_state(self, state, mask=None, keys=None):
        self.validate_state(state)
        return _default_reset_state(self, state, mask, keys)

    def set_state(self, state, other):
        self.validate_state(other)
        return _default_set_state(state, other)

    def get_state(self):
        return _default_get_state(self)
SSM_STATUS: dict[str, str] = {}

def ssm_status(status: str, **meta):
    assert status in ('production', 'maintained', 'legacy', 'experimental')

    def deco(cls):
        SSM_STATUS[cls.__name__] = status
        cls.__ssm_status__ = status
        cls.__ssm_meta__ = dict(meta)
        return cls
    return deco

class BaseMixer(StateLifecycle, nn.Module):
    _inference_pin: bool | None = None

    def set_inference(self, pin: bool | None=True) -> None:
        self._inference_pin = pin

    def _fast_inference(self) -> bool:
        if self._inference_pin is not None:
            return self._inference_pin
        return not torch.is_grad_enabled()

    def forward(self, x: torch.Tensor, state=None, return_state: bool=False):
        raise NotImplementedError

    def step(self, x_t: torch.Tensor, state) -> tuple[torch.Tensor, 'State']:
        raise NotImplementedError

    def _init_state(self, batch_size: int, device=None, dtype: torch.dtype=torch.float32, **kwargs) -> 'State':
        raise NotImplementedError

class BaseSSM(StateLifecycle, nn.Module):
    out_width: int

    def forward(self, x, *, state=None, return_state: bool=False, **kwargs):
        raise NotImplementedError

    def step(self, x_t, state) -> tuple[torch.Tensor, 'State']:
        raise NotImplementedError

    def hidden_state_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def empty_state(self, batch_size: int) -> torch.Tensor:
        return next(self.parameters()).new_zeros(batch_size, *self.hidden_state_shape())

    def _base_initial_state(self) -> torch.Tensor | None:
        return None

    def base_seed(self) -> torch.Tensor:
        base = self._base_initial_state()
        if base is None:
            return next(self.parameters()).new_zeros(*self.hidden_state_shape())
        return base.clone()

    def initial_state(self, batch_size, h_init=None) -> torch.Tensor:
        state_shape = self.hidden_state_shape()
        if h_init is None:
            h_init = self.base_seed()
        if tuple(h_init.shape) == tuple(state_shape):
            if batch_size is None:
                return h_init
            return h_init.unsqueeze(0).expand(batch_size, *state_shape).contiguous()
        expected = (batch_size, *state_shape)
        if tuple(h_init.shape) != expected:
            raise ValueError(f'{type(self).__name__} h_init must have shape {state_shape} or {expected}, got {tuple(h_init.shape)}.')
        return h_init.contiguous()

    def _init_state(self, batch_size: int, device=None, dtype: torch.dtype=torch.float32, **kwargs) -> 'State':
        h = self.initial_state(batch_size)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device, dtype=dtype if dtype is not None else h.dtype)
        return State(hidden_state=h)
