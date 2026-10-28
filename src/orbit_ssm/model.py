
from __future__ import annotations
import copy
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.norms import RMSNorm
from .layers.state import BaseMixer, State
from .layers.swiglu import SwiGLU
from .config import reject_retired_keys
from .configuration import Configurable, MixerConfigurable, SEMANTICS_VERSION
from .ssm.hybrid_mamba import HybridMambaMixer, _HMInProjFn, _hm_in_proj

def mark_residual_out(weight: nn.Parameter) -> None:
    weight._normalize_w_num_layers = True

def tag_official_mamba_mixer(mixer: nn.Module) -> None:
    out_proj = getattr(mixer, 'out_proj', None)
    if isinstance(out_proj, nn.Linear):
        mark_residual_out(out_proj.weight)

def tag_residual_mixer(mixer: nn.Module) -> None:
    out_proj = getattr(mixer, 'out_proj', None)
    if not isinstance(out_proj, nn.Linear):
        raise TypeError(f'{type(mixer).__name__} must expose nn.Linear out_proj so its residual branch can receive depth-aware initialization')
    mark_residual_out(out_proj.weight)
_MIXER_REGISTRY: dict[str, 'Any'] = {}

def _builtin_mixer_types() -> tuple[type, ...]:
    from .ssm import MIXERS
    return tuple(MIXERS.values())

def rescale_residual_out(module: nn.Module, *, n_residuals: int) -> int:
    if isinstance(n_residuals, bool) or not isinstance(n_residuals, int) or n_residuals < 1:
        raise ValueError(f'n_residuals must be a positive int, got {n_residuals!r}')
    count = 0
    for p in module.parameters():
        if getattr(p, '_normalize_w_num_layers', False):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p.div_(math.sqrt(n_residuals))
            count += 1
    return count

def init_lm(model: nn.Module, *, embedding_std: float=0.02) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            if not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=embedding_std)
from .ssm.ssd_mimo import SSDMimoMixer, _SSDMimoCombinedFn, _combined_apply, _fused_in_proj

class ResidualBlock(BaseMixer):

    def __init__(self, mixer: nn.Module, d_model: int, d_intermediate: int=0):
        super().__init__()
        self.norm = RMSNorm(d_model, fused=True)
        self.mixer = mixer
        self.d_intermediate = int(d_intermediate)
        if self.d_intermediate:
            self.norm_mlp = RMSNorm(d_model, fused=True)
            self.mlp = SwiGLU(d_model, self.d_intermediate)
            mark_residual_out(self.mlp.down.weight)
        else:
            self.norm_mlp = None
            self.mlp = None
        if isinstance(mixer, _builtin_mixer_types()):
            tag_residual_mixer(mixer)

    @property
    def n_residuals(self) -> int:
        return 1 if self.mlp is None else 2

    def _mlp_residual(self, x):
        if self.mlp is None:
            return x
        return x + self.mlp(self.norm_mlp(x))

    def forward(self, x, state=None, return_state: bool=False):
        if state is None and (not return_state):
            return self._mlp_residual(x + self.mixer(self.norm(x)))
        y, state = self.mixer(self.norm(x), state=state, return_state=True)
        y = self._mlp_residual(x + y)
        return (y, state) if return_state else y

    def step(self, x_t, state=None):
        if state is None:
            state = self.init_state(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        y, state = self.mixer.step(self.norm(x_t), state)
        return (self._mlp_residual(x_t + y), state)

    def validate_state(self, state, **kwargs):
        return self.mixer.validate_state(state, **kwargs)

    def init_state(self, batch_size, **kwargs):
        return self.mixer.init_state(batch_size, **kwargs)

    def reset_state(self, state, mask=None, keys=None):
        return self.mixer.reset_state(state, mask, keys=keys)

    def set_state(self, state, other):
        return self.mixer.set_state(state, other)

    def get_state(self):
        return self.mixer.get_state()

class MixerStack(nn.Module):

    def __init__(self, blocks, d_model: int, *, residual_rescale: bool=True):
        super().__init__()
        self.layers = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(d_model, fused=True)
        self.residual_rescale = bool(residual_rescale)
        if self.residual_rescale and len(self.layers):
            rescale_residual_out(self, n_residuals=self.n_residuals)

    @classmethod
    def from_mixers(cls, mixers, d_model: int, d_intermediate: int=0, *, residual_rescale: bool=True):
        return cls([ResidualBlock(mixer=mixer, d_model=d_model, d_intermediate=d_intermediate) for mixer in mixers], d_model=d_model, residual_rescale=residual_rescale)

    @property
    def n_residuals(self) -> int:
        total = 0
        for index, block in enumerate(self.layers):
            count = getattr(block, 'n_residuals', 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f'layers[{index}].n_residuals must be a positive int, got {count!r}')
            total += count
        return total

    @property
    def blocks(self):
        return self.layers

    @property
    def norms(self):
        return [block.norm for block in self.layers]

    def _validate_states(self, states, name='state'):
        if not isinstance(states, (list, tuple)) or len(states) != len(self.layers):
            raise ValueError(f'{name} must contain exactly {len(self.layers)} layer states')

    def _validate_carries(self, states, *, batch_size=None, device=None):
        self._validate_states(states)
        for block, state in zip(self.layers, states):
            if state is not None:
                block.validate_state(state, batch_size=batch_size, device=device)

    def forward(self, x, state=None, return_state: bool=False):
        if state is None and (not return_state):
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
        count = len(self.layers)
        if h_init is not None:
            self._validate_states(h_init, 'h_init')
        if overrides is not None:
            self._validate_states(overrides, 'overrides')
        hidden = h_init if h_init is not None else [None] * count
        custom = overrides if overrides is not None else [None] * count
        return [b.init_state(batch_size, h_init=h, overrides=o, **kwargs) for b, h, o in zip(self.layers, hidden, custom)]

    def step(self, x, state=None):
        states = state
        if states is None:
            states = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        self._validate_carries(states, batch_size=x.shape[0], device=x.device)
        out = []
        for b, s in zip(self.layers, states):
            x, s = b.step(x, s)
            out.append(s)
        return (self.norm_f(x), out)

    def reset_state(self, states, mask=None, keys=None):
        self._validate_states(states)
        selected = None if keys is None else (keys,) if isinstance(keys, str) else tuple(keys)
        for b, s in zip(self.layers, states):
            b.validate_state(s)
            if selected is not None:
                unknown = set(selected) - s.keys()
                if unknown:
                    raise ValueError(f'unknown state keys: {sorted(unknown)}')
        for b, s in zip(self.layers, states):
            b.reset_state(s, mask, keys=selected)
        return states

    def set_state(self, states, others):
        self._validate_states(states)
        self._validate_states(others, 'other state')
        for b, other in zip(self.layers, others):
            b.validate_state(other)
        for b, s, o in zip(self.layers, states, others):
            b.set_state(s, o)
        return states

    def get_state(self):
        return [b.get_state() for b in self.layers]

class CausalLM(Configurable, nn.Module):
    _SSM_TYPES = frozenset({'ssd_mimo', 'hybrid_mamba'})
    config_schema = 'orbit_ssm.model/v0'
    config_family = 'causal_lm'

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, ssm_type: str='ssd_mimo', ssm_kwargs: dict | None=None, expand: float | None=2.0, d_conv: int=4, d_state: int=64, gate: str='swiglu', bc_norm: bool | None=None, d_intermediate: int=0, tie_word_embeddings: bool=True, pad_vocab_size_multiple: int=8, layer_configs: list[dict] | None=None, *, d_inner: int | None=None):
        super().__init__()
        if gate != 'swiglu':
            raise ValueError(f"Unsupported gate type: {gate!r}. Only 'swiglu' is supported.")
        if vocab_size < 1:
            raise ValueError(f'vocab_size must be positive, got {vocab_size}')
        if n_layers < 1:
            raise ValueError(f'n_layers must be positive, got {n_layers}')
        if d_intermediate < 0:
            raise ValueError(f'd_intermediate must be non-negative, got {d_intermediate}')
        if pad_vocab_size_multiple < 1:
            raise ValueError(f'pad_vocab_size_multiple must be positive, got {pad_vocab_size_multiple}')
        if layer_configs is not None:
            if not isinstance(layer_configs, (list, tuple)) or len(layer_configs) > n_layers:
                raise ValueError('layer_configs must have at most n_layers entries')
            allowed = {'ssm_type', 'ssm_kwargs', 'expand', 'd_conv', 'd_state', 'gate', 'bc_norm', 'mixer_kwargs', 'd_inner'}
            for index, cfg in enumerate(layer_configs):
                if cfg is not None and (not isinstance(cfg, dict) or set(cfg) - allowed):
                    raise ValueError(f'unsupported layer_configs[{index}]: {cfg!r}')
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padded_vocab_size = (vocab_size + pad_vocab_size_multiple - 1) // pad_vocab_size_multiple * pad_vocab_size_multiple
        self.n_layers = n_layers
        self.d_intermediate = int(d_intermediate)
        self.residuals_per_layer = 2 if self.d_intermediate else 1
        self.tie_word_embeddings = tie_word_embeddings
        self.embedding = nn.Embedding(self.padded_vocab_size, d_model)
        mixers: list[BaseMixer] = []
        for i in range(n_layers):
            cfg: dict = {}
            if layer_configs is not None and i < len(layer_configs):
                cfg = layer_configs[i] or {}
            layer_type = cfg.get('ssm_type', ssm_type)
            mixer = self._build_mixer(d_model=d_model, ssm_type=layer_type, ssm_kwargs=cfg.get('ssm_kwargs', ssm_kwargs), expand=cfg.get('expand', expand), d_inner=cfg.get('d_inner', d_inner), d_conv=cfg.get('d_conv', d_conv), d_state=cfg.get('d_state', d_state), gate=cfg.get('gate', gate), bc_norm=cfg.get('bc_norm', bc_norm), mixer_kwargs=cfg.get('mixer_kwargs'))
            mixers.append(mixer)
        self.backbone = MixerStack.from_mixers(mixers, d_model, d_intermediate=self.d_intermediate)
        self.lm_head = nn.Linear(d_model, self.padded_vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self._init_weights()
        self._resolved_constructor_config = None
        if all((isinstance(m, MixerConfigurable) for m in mixers)):
            recipes = [m.resolved_config() for m in mixers]
            self._resolved_constructor_config = dict(schema=self.config_schema, semantics_version=SEMANTICS_VERSION, family=self.config_family, model=dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_intermediate=self.d_intermediate, tie_word_embeddings=tie_word_embeddings, pad_vocab_size_multiple=pad_vocab_size_multiple, initialization='orbit_ssm.lm/v0', layers=[dict(family=r['family'], model=r['model']) for r in recipes]), execution=dict(layer_execution=[r['execution'] for r in recipes]))

    def resolved_config(self):
        if self._resolved_constructor_config is None:
            raise NotImplementedError('resolved model recipes currently require Paper 1 mixers')
        return super().resolved_config()

    @classmethod
    def _with_legacy_defaults(cls, document):
        from .configuration import fill_legacy_core_fields
        model = document.get('model') if isinstance(document, dict) else None
        if isinstance(model, dict) and isinstance(model.get('layers'), list):
            document = copy.deepcopy(document)
            for layer in document['model']['layers']:
                if isinstance(layer, dict) and isinstance(layer.get('model'), dict) and isinstance(layer['model'].get('core'), dict):
                    fill_legacy_core_fields(layer.get('family'), layer['model']['core'])
        return document

    @classmethod
    def _from_config_sections(cls, model, execution):
        from .ssm import MIXERS
        if model.pop('initialization') != 'orbit_ssm.lm/v0':
            raise ValueError('unsupported LM initialization recipe')
        layers = model.pop('layers')
        runtime = execution.pop('layer_execution')
        if execution or len(layers) != model['n_layers'] or len(runtime) != len(layers):
            raise ValueError('model recipe must specify exactly one recipe per layer')
        configs = []
        for layer, settings in zip(layers, runtime):
            family = layer['family']
            mixer = MIXERS.get(family)
            if mixer is None or not issubclass(mixer, MixerConfigurable):
                raise ValueError(f'unsupported model recipe mixer {family!r}')
            recipe = dict(schema=mixer.config_schema, family=family, model=layer['model'], execution=settings)
            kwargs = mixer.constructor_kwargs(recipe)
            if kwargs.pop('d_model') != model['d_model']:
                raise ValueError('layer width conflicts with model d_model')
            configs.append(dict(ssm_type=family, mixer_kwargs=kwargs))
        return cls(**model, layer_configs=configs)

    @classmethod
    def register_mixer(cls, name: str, factory) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f'mixer name must be a non-empty string, got {name!r}')
        if name in cls._SSM_TYPES:
            raise ValueError(f'{name!r} is a built-in SSM family and cannot be re-registered')
        if not callable(factory):
            raise TypeError(f'factory for {name!r} must be callable')
        _MIXER_REGISTRY[name] = factory

    @classmethod
    def registered_mixers(cls) -> tuple[str, ...]:
        return tuple(sorted(_MIXER_REGISTRY))

    @staticmethod
    def _build_mixer(d_model: int, ssm_type: str, ssm_kwargs: dict | None=None, expand: float | None=2.0, d_conv: int=4, d_state: int=64, gate: str='swiglu', bc_norm: bool | None=None, mixer_kwargs: dict | None=None, d_inner: int | None=None) -> nn.Module:
        if d_inner is not None and ssm_type not in ('ssd_mimo', 'hybrid_mamba'):
            raise ValueError('explicit d_inner currently requires a Paper 1 mixer')
        if gate != 'swiglu':
            raise ValueError(f'Unsupported gate type: {gate!r}')
        if mixer_kwargs and ssm_type not in ('hybrid_mamba', 'ssd_mimo'):
            raise ValueError('mixer_kwargs currently requires a Paper 1 mixer')
        factory = _MIXER_REGISTRY.get(ssm_type)
        if factory is not None:
            return factory(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, ssm_kwargs=ssm_kwargs)
        from .config import SSMCoreConfig, SSDMimoConfig, HybridMambaConfig, FAMILY_CONFIGS, build_ssm
        import dataclasses as _dc
        if ssm_type not in FAMILY_CONFIGS:
            raise ValueError(f'Unknown ssm_type: {ssm_type!r}; supported: {sorted(FAMILY_CONFIGS)}')
        if isinstance(ssm_kwargs, dict):
            reject_retired_keys(ssm_kwargs, f'ssm_kwargs[{ssm_type!r}]')
        if isinstance(ssm_kwargs, SSMCoreConfig):
            cfg = ssm_kwargs
            want = FAMILY_CONFIGS.get(ssm_type)
            if want is not None and type(cfg) not in (want, SSMCoreConfig):
                raise TypeError(f'ssm_type={ssm_type!r} but config is {type(cfg).__name__}')
            if type(cfg) is SSMCoreConfig:
                cfg = want(**cfg.core_fields())
            if ssm_type not in ('ssd_mimo', 'hybrid_mamba'):
                cfg = _dc.replace(cfg, d_inner=d_model)
                return build_ssm(cfg)
            ssm_kwargs = cfg
        if ssm_type in ('ssd_mimo', 'hybrid_mamba'):
            from .ssm import MIXERS
            if bc_norm is not None:
                from .config_spec import ConfigSpecError, MIXER_BC_NORM_REMOVED
                raise ConfigSpecError(f'CausalLM[{ssm_type}]: {MIXER_BC_NORM_REMOVED}')
            kw = dict(d_state=d_state, expand=expand, d_conv=d_conv, ssm_kwargs=ssm_kwargs)
            if d_inner is not None:
                kw['d_inner'] = d_inner
            if mixer_kwargs:
                if 'd_model' in mixer_kwargs:
                    raise ValueError('mixer_kwargs must not override model d_model')
                if d_inner is not None and mixer_kwargs.get('d_inner', d_inner) != d_inner:
                    raise ValueError('mixer_kwargs d_inner conflicts with layer d_inner')
                kw.update(mixer_kwargs)
            return MIXERS[ssm_type](d_model=d_model, **kw)
        raise ValueError(f'Unknown ssm_type: {ssm_type!r}. Supported: {sorted(CausalLM._SSM_TYPES)} + registered {sorted(_MIXER_REGISTRY)}.')

    def _init_weights(self):
        init_lm(self, embedding_std=0.02)
        if not self.tie_word_embeddings:
            nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

    def _project_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)[..., :self.vocab_size]

    def encode(self, input_ids: torch.Tensor, *, state=None, return_state: bool=False) -> torch.Tensor:
        x = self.embedding(input_ids)
        if state is None and (not return_state):
            return self.backbone(x)
        y, new_state = self.backbone(x, state=state, return_state=True)
        return (y, new_state) if return_state else y

    def forward(self, input_ids: torch.Tensor, return_embeddings: bool=False, *, state=None, return_state: bool=False) -> torch.Tensor:
        if state is None and (not return_state):
            x = self.encode(input_ids)
            return x if return_embeddings else self._project_logits(x)
        x, new_state = self.encode(input_ids, state=state, return_state=True)
        out = x if return_embeddings else self._project_logits(x)
        return (out, new_state) if return_state else out

    def step(self, input_ids_single: torch.Tensor, state: list | None=None) -> tuple[torch.Tensor, list]:
        ids = input_ids_single.squeeze(-1) if input_ids_single.ndim == 2 else input_ids_single
        x = self.embedding(ids)
        x, states = self.backbone.step(x, state)
        return (self._project_logits(x), states)

    def init_state(self, batch_size: int, **kwargs) -> list:
        return self.backbone.init_state(batch_size, **kwargs)

    def reset_state(self, states: list, mask=None, keys=None) -> list:
        return self.backbone.reset_state(states, mask, keys=keys)

    def set_state(self, states: list, others: list) -> list:
        return self.backbone.set_state(states, others)

    def get_state(self) -> list:
        return self.backbone.get_state()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float=1.0, top_k: int | None=None) -> torch.Tensor:
        B, L = input_ids.shape
        states = self.init_state(B)
        for t in range(L):
            logits, states = self.step(input_ids[:, t:t + 1], states)
        generated: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k is not None:
                    k = min(top_k, scaled.size(-1))
                    v, _ = torch.topk(scaled, k)
                    scaled[scaled < v[:, -1:]] = float('-inf')
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, 1)
            generated.append(next_token)
            logits, states = self.step(next_token, states)
        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)
