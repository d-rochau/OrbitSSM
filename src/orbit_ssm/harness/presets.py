
from __future__ import annotations
import math
_FAMILIES = ('mamba1', 'mamba2', 'mamba3')

def _check_family(name: str) -> None:
    if name not in _FAMILIES:
        raise ValueError(f'unknown preset {name!r}, must be one of {_FAMILIES}')

def required_sizes(name: str) -> tuple[str, ...]:
    _check_family(name)
    return ('d_model',)

def mamba1_kwargs(d_model: int, *, expand: int=2, d_state: int=16, d_conv: int=4, dt_rank: int | str='auto', **overrides) -> dict:
    if dt_rank == 'auto':
        dt_rank = math.ceil(d_model / 16)
    ssm_kwargs = dict(rows=[{'frac': 1.0, 'decay': 'A', 'complex': False}], head_dim=1, dt_rank=int(dt_rank), dt_init='random', dt_scale=1.0, proj_source='main_branch', out_norm=False, b_norm=False, c_norm=False, bias_gran='none', use_trap=False, backend='auto')
    ssm_kwargs.update(overrides)
    return dict(d_model=d_model, d_state=d_state, expand=float(expand), d_conv=d_conv, ssm_kwargs=ssm_kwargs)

def mamba2_kwargs(d_model: int, *, expand: int=2, d_state: int=128, headdim: int=64, ngroups: int=1, d_conv: int=4, **overrides) -> dict:
    if ngroups != 1:
        raise ValueError(f"mamba2_kwargs: ngroups must be 1 (SSD-MIMO shares one B/C across every head; there is no counterpart to official Mamba-2's ngroups={ngroups} grouped-B/C layout), got {ngroups}")
    d_inner = int(expand) * d_model
    if d_inner % headdim:
        raise ValueError(f'mamba2_kwargs: d_inner={d_inner} (expand*d_model) is not a multiple of headdim={headdim}')
    n_heads = d_inner // headdim
    ssm_kwargs = dict(R=1, O=1, ngroups=headdim, n_heads=n_heads, proj_source='layer_input', bc_conv=True, rows=[{'frac': 1.0, 'decay': 'head_static', 'complex': False}], d_skip=True, d_skip_sharing='head', write_rule='euler', out_norm=True, b_norm=False, c_norm=False, bias_gran='none', use_g_b=False, use_g_c=False, r_mix_dyn=False, dt_min=0.001, dt_max=0.1, backend='auto')
    ssm_kwargs.update(overrides)
    return dict(d_model=d_model, d_state=d_state, expand=float(expand), d_conv=d_conv, ssm_kwargs=ssm_kwargs)

def mamba3_kwargs(d_model: int, *, expand: int=2, d_state: int=128, headdim: int=64, d_conv: int=0, **overrides) -> dict:
    d_inner = int(expand) * d_model
    if d_inner % headdim:
        raise ValueError(f'mamba3_kwargs: d_inner={d_inner} (expand*d_model) is not a multiple of headdim={headdim}')
    n_heads = d_inner // headdim
    ssm_kwargs = dict(R=1, O=1, ngroups=headdim, n_heads=n_heads, proj_source='layer_input', write_rule='trapezoid', rows=[{'frac': 0.5, 'decay': 'head_dynamic', 'complex': True}, {'frac': 0.5, 'decay': 'head_dynamic', 'complex': False}], theta_per_head=False, use_theta_h=False, learn_theta_freq=False, b_norm=True, c_norm=True, bias_gran='head_bc', d_skip=True, d_skip_sharing='head', a_proj_bias=False, trap_proj_bias=False, use_g_b=False, use_g_c=False, r_mix_dyn=False, dt_min=0.001, dt_max=0.1, out_norm=False, backend='auto')
    ssm_kwargs.update(overrides)
    return dict(d_model=d_model, d_state=d_state, expand=float(expand), d_conv=d_conv, ssm_kwargs=ssm_kwargs)
_PRESET_FNS = {'mamba1': mamba1_kwargs, 'mamba2': mamba2_kwargs, 'mamba3': mamba3_kwargs}

def build_official_like(name: str, d_model: int, **kwargs):
    _check_family(name)
    from ..model import HybridMambaMixer, SSDMimoMixer
    cls = HybridMambaMixer if name == 'mamba1' else SSDMimoMixer
    return cls(**_PRESET_FNS[name](d_model, **kwargs))
