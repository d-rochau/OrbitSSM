
from __future__ import annotations
import torch.nn as nn
from orbit_ssm.layers.attention import Llama3Attention
from orbit_ssm.model import CausalLM, mark_residual_out, tag_official_mamba_mixer
__all__ = ['LM_MIXERS', 'register_lm_mixers']

def _kw(ssm_kwargs) -> dict:
    if ssm_kwargs is None:
        return {}
    if not isinstance(ssm_kwargs, dict):
        raise TypeError(f'LM-contract mixers take a plain dict of constructor overrides, got {type(ssm_kwargs).__name__}')
    return dict(ssm_kwargs)

def _official_mamba1(*, d_model, d_state, expand, d_conv, ssm_kwargs=None) -> nn.Module:
    from mamba_ssm.modules.mamba_simple import Mamba
    kw = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=int(expand))
    kw.update(_kw(ssm_kwargs))
    mixer = Mamba(**kw)
    tag_official_mamba_mixer(mixer)
    return mixer

def _official_mamba2(*, d_model, d_state, expand, d_conv, ssm_kwargs=None) -> nn.Module:
    from mamba_ssm.modules.mamba2 import Mamba2
    kw = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=int(expand), headdim=64)
    kw.update(_kw(ssm_kwargs))
    mixer = Mamba2(**kw)
    tag_official_mamba_mixer(mixer)
    return mixer

def _mamba3(*, d_model, d_state, expand, d_conv, ssm_kwargs, is_mimo: bool) -> nn.Module:
    from mamba_ssm import Mamba3
    kw = dict(d_model=d_model, d_state=d_state, expand=int(expand), headdim=64, ngroups=1, chunk_size=16 if is_mimo else 64, is_mimo=is_mimo)
    if is_mimo:
        kw['mimo_rank'] = 4
    kw.update(_kw(ssm_kwargs))
    mixer = Mamba3(**kw)
    tag_official_mamba_mixer(mixer)
    return mixer

def _official_mamba3(*, d_model, d_state, expand, d_conv, ssm_kwargs=None) -> nn.Module:
    return _mamba3(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, ssm_kwargs=ssm_kwargs, is_mimo=False)

def _official_mamba3_mimo(*, d_model, d_state, expand, d_conv, ssm_kwargs=None) -> nn.Module:
    return _mamba3(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, ssm_kwargs=ssm_kwargs, is_mimo=True)

def _llama3_attention(*, d_model, d_state, expand, d_conv, ssm_kwargs=None) -> nn.Module:
    kw = dict(d_model=d_model, head_dim=64, rope_base=500000.0)
    kw.update(_kw(ssm_kwargs))
    mixer = Llama3Attention(**kw)
    mark_residual_out(mixer.out_proj.weight)
    return mixer
LM_MIXERS = {'official_mamba1': _official_mamba1, 'official_mamba2': _official_mamba2, 'official_mamba3': _official_mamba3, 'official_mamba3_mimo': _official_mamba3_mimo, 'llama3_attention': _llama3_attention}

def register_lm_mixers() -> tuple[str, ...]:
    for name, factory in LM_MIXERS.items():
        CausalLM.register_mixer(name, factory)
    return tuple(LM_MIXERS)
register_lm_mixers()
