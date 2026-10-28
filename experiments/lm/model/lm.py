
from __future__ import annotations
import re
from dataclasses import dataclass
import torch
import torch.nn as nn
from orbit_ssm.harness.presets import mamba3_kwargs
from orbit_ssm.model import CausalLM, init_lm
import experiments.models.lm_mixers as _lm_mixers
REGISTERED_MIXERS = tuple(_lm_mixers.LM_MIXERS)
HYBRID_SSM_FAMILIES = tuple(sorted(CausalLM._SSM_TYPES))

def _ssd_mimo_arm(R: int, O: int, rotation: bool, **extra) -> dict:
    law = 'head_dynamic'
    rows = [{'frac': 0.5, 'decay': law, 'complex': True}, {'frac': 0.5, 'decay': law, 'complex': False}] if rotation else [{'frac': 1.0, 'decay': law, 'complex': False}]
    kw = dict(R=R, O=O, ngroups=64 // R, heads=32, head_subsystems=64, rows=rows, b_norm=True, c_norm=True, bias_gran='head_bc', use_g_b=True, use_g_c=True, bc_bias_init='auto', r_mix='learned', r_mix_dyn=True, dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, a_proj_bias=True, a_dyn_per_sibling=False, a_act='heavy_tail', trap_proj_bias=True, write_rule='trapezoid', d_skip=True, d_skip_sharing='input_output', pair_layout='interleaved', proj_source='layer_input', bc_conv=False, out_norm=False, backend='auto', precision='io', m3_kernel='native', compile_token_maps=False, chunk_size='auto', allow_chunk_autotune=True, kernel_options={})
    if rotation:
        kw.update(theta_per_head=False, use_theta_h=True, learn_theta_freq=True, rotation=extra.pop('rot_spec', {'omega_layout': 'pair', 'theta_init': 'kaiming'}))
    kw.update(extra)
    return {'name': 'ssd_mimo', 'kwargs': kw}

def _m3exact_arm(d_model: int) -> dict:
    pre = mamba3_kwargs(d_model, expand=2, d_state=128, headdim=64)
    return {'name': 'ssd_mimo', 'd_conv': pre['d_conv'], 'kwargs': dict(pre['ssm_kwargs'])}

def _m3exact_real_arm(d_model: int) -> dict:
    arm = _m3exact_arm(d_model)
    kw = dict(arm['kwargs'])
    kw['rows'] = [{'frac': 1.0, 'decay': 'head_dynamic', 'complex': False}]
    for k in ('theta_per_head', 'use_theta_h', 'learn_theta_freq'):
        kw.pop(k, None)
    return {**arm, 'kwargs': kw}

def _hm_arm(rows: list, **extra) -> dict:
    kw = dict(head_subsystems=64, rows=rows, b_norm=True, c_norm=True, bias_gran='head_bc', use_g_b=False, use_g_c=False, proj_source='layer_input', bc_conv=False, dt_rank='adaptive', force_dt_dense=False, dt_init='random', dt_scale=1.0, dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, dyn_res='n', dyn_act='heavy_tail', a_floor=0.0001, norm_eps=1e-05, d_skip_mode='head', use_trap=True, learnable_h_init=False, out_norm=False, backend='auto', ckpt=True, m3_kernel='native', kernel_options={})
    kw.update(extra)
    return {'name': 'hybrid_mamba', 'kwargs': kw}
NAMED_SSM: dict[str, dict] = {'ssd_mimo_r2o2_complex': _ssd_mimo_arm(2, 2, True), 'ssd_mimo_r2o2_real': _ssd_mimo_arm(2, 2, False), 'ssd_mimo_r2o2_real_bnorm_off': _ssd_mimo_arm(2, 2, False, b_norm=False), 'ssd_mimo_r2o2_complex_adynsib': _ssd_mimo_arm(2, 2, True, a_dyn_per_sibling=True), 'ssd_mimo_r1o1_complex': _ssd_mimo_arm(1, 1, True), 'ssd_mimo_r2o2_complex_hpair': _ssd_mimo_arm(2, 2, True, rot_spec={'omega_layout': 'head_pair', 'theta_init': 'small'}), 'ssd_mimo_r1o1_complex_hpair': _ssd_mimo_arm(1, 1, True, rot_spec={'omega_layout': 'head_pair', 'theta_init': 'small'}), 'ssd_mimo_r2o2_complex_ohead': _ssd_mimo_arm(2, 2, True, theta_per_head=True, rot_spec={'omega_layout': 'head', 'theta_init': 'small'}), 'ssd_mimo_r1o1_complex_ohead': _ssd_mimo_arm(1, 1, True, theta_per_head=True, rot_spec={'omega_layout': 'head', 'theta_init': 'small'}), 'ssd_mimo_r1o1_real': _ssd_mimo_arm(1, 1, False), 'ssd_mimo_m3exact': _m3exact_arm(1024), 'ssd_mimo_m3exact_real': _m3exact_real_arm(1024), 'ssd_mimo_r4o4_complex': _ssd_mimo_arm(4, 4, True), 'ssd_mimo_r4o4_real': _ssd_mimo_arm(4, 4, False), 'hm_a_dyn_cplx25': _hm_arm([{'frac': 0.25, 'decay': 'dyn', 'complex': True}, {'frac': 0.375, 'decay': 'A', 'complex': False}, {'frac': 0.375, 'decay': 'dyn', 'complex': False}], proj_source='main_branch', theta_per_head=False, use_theta_h=True, learn_theta_freq=True, rotation={'omega_layout': 'pair', 'theta_init': 'kaiming'}), 'hm_a_dyn_cplx75': _hm_arm([{'frac': 0.75, 'decay': 'dyn', 'complex': True}, {'frac': 0.125, 'decay': 'A', 'complex': False}, {'frac': 0.125, 'decay': 'dyn', 'complex': False}], proj_source='main_branch', theta_per_head=False, use_theta_h=True, learn_theta_freq=True, rotation={'omega_layout': 'pair', 'theta_init': 'kaiming'}), 'hm_a_dyn_cplx100': _hm_arm([{'frac': 1.0, 'decay': 'dyn', 'complex': True}], proj_source='main_branch', theta_per_head=False, use_theta_h=True, learn_theta_freq=True, rotation={'omega_layout': 'pair', 'theta_init': 'kaiming'}), 'hm_a_dyn': _hm_arm([{'frac': 0.5, 'decay': 'A', 'complex': False}, {'frac': 0.5, 'decay': 'dyn', 'complex': False}], proj_source='main_branch')}

@dataclass
class MambaLMConfig:
    vocab_size: int
    n_layers: int
    d_model: int
    mlp_dim: int
    expand_factor: int = 2
    d_state: int = 128
    d_conv: int = 4
    ssm: str | dict = 'official_mamba3'
    init_std: float = 0.02
    tie_embedding: bool = True

    def mixer_d_conv(self) -> int:
        if isinstance(self.ssm, dict) and 'd_conv' in self.ssm:
            return int(self.ssm['d_conv'])
        return self.d_conv

    def mixer_spec(self) -> tuple[str, dict | None]:
        spec = self.ssm
        if isinstance(spec, dict):
            if 'name' not in spec:
                raise ValueError(f"ssm dict needs a 'name' key, got {sorted(spec)}")
            name = spec['name']
            kwargs = spec.get('kwargs')
            if kwargs is not None and (not isinstance(kwargs, dict)):
                raise TypeError(f"ssm['kwargs'] must be a dict, got {kwargs!r}")
            extra = set(spec) - {'name', 'kwargs', 'd_conv'}
            if extra:
                raise ValueError(f'ssm dict keys are name/kwargs/d_conv, got extra {sorted(extra)}')
        elif isinstance(spec, str):
            name, kwargs = (spec, None)
        else:
            raise TypeError(f'ssm must be a str or dict, got {type(spec).__name__}')
        known = REGISTERED_MIXERS + HYBRID_SSM_FAMILIES
        if name not in known:
            raise ValueError(f"unknown mixer {name!r}. The LM contract serves: {', '.join(REGISTERED_MIXERS)} (registered reference arms) and {', '.join(HYBRID_SSM_FAMILIES)} (HybridSSM families, passed as a dict with 'name' and 'kwargs'). Arms built on the retired hybrid_mamba package were removed with it and are re-run, not revived.")
        return (name, kwargs)

def _chunked_ce(x: torch.Tensor, W: torch.Tensor, targets: torch.Tensor, chunk: int) -> torch.Tensor:
    valid = (targets != -100).sum().clamp(min=1)

    def seg(xs, W, ts):
        return torch.nn.functional.cross_entropy(xs @ W.T, ts, ignore_index=-100, reduction='sum')
    total = None
    for i in range(0, x.shape[0], chunk):
        l = torch.utils.checkpoint.checkpoint(seg, x[i:i + chunk], W, targets[i:i + chunk], use_reentrant=False)
        total = l if total is None else total + l
    return total / valid
_LEGACY_BLOCK_RE = re.compile('^blocks\\.(\\d+)\\.(norm_mamba|mamba|norm_mlp|mlp)(\\..*)?$')
_LEGACY_SUBMODULE = {'norm_mamba': 'norm', 'mamba': 'mixer', 'norm_mlp': 'norm_mlp', 'mlp': 'mlp'}

def _rename_legacy_keys(state_dict: dict) -> dict:
    if 'embedding.weight' in state_dict or 'backbone.norm_f.weight' in state_dict:
        return state_dict
    out: dict = {}
    for key, value in state_dict.items():
        if key == 'embed.weight':
            out['embedding.weight'] = value
        elif key == 'norm_f.weight':
            out['backbone.norm_f.weight'] = value
        else:
            m = _LEGACY_BLOCK_RE.match(key)
            if m:
                idx, sub, rest = (m.group(1), m.group(2), m.group(3) or '')
                out[f'backbone.layers.{idx}.{_LEGACY_SUBMODULE[sub]}{rest}'] = value
            else:
                out[key] = value
    return out

class MambaLM(CausalLM):
    ckpt_every_blocks: int = 0
    ce_chunk: int = 0

    def __init__(self, cfg: MambaLMConfig):
        name, ssm_kwargs = cfg.mixer_spec()
        super().__init__(vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, ssm_type=name, ssm_kwargs=ssm_kwargs, expand=float(cfg.expand_factor), d_conv=cfg.mixer_d_conv(), d_state=cfg.d_state, d_intermediate=cfg.mlp_dim, tie_word_embeddings=cfg.tie_embedding, pad_vocab_size_multiple=8)
        self.cfg = cfg
        if cfg.init_std != 0.02:
            init_lm(self, embedding_std=cfg.init_std)
            if not cfg.tie_embedding:
                nn.init.trunc_normal_(self.lm_head.weight, std=cfg.init_std)

    def num_params(self, non_embedding: bool=False) -> int:
        n = sum((p.numel() for p in self.parameters()))
        if non_embedding:
            n -= self.embedding.weight.numel()
            if self.lm_head.weight is not self.embedding.weight:
                n -= self.lm_head.weight.numel()
        return n

    def load_state_dict(self, state_dict, strict: bool=True, assign: bool=False):
        return super().load_state_dict(_rename_legacy_keys(dict(state_dict)), strict=strict, assign=assign)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None):
        x = self.embedding(idx)
        ck = self.ckpt_every_blocks
        for i, blk in enumerate(self.backbone.layers):
            if ck and torch.is_grad_enabled() and (i % ck == 0):
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.backbone.norm_f(x)
        if targets is not None and self.ce_chunk:
            W = self.lm_head.weight[:self.vocab_size]
            loss = _chunked_ce(x.reshape(-1, x.size(-1)), W, targets.reshape(-1), self.ce_chunk)
            return {'logits': None, 'loss': loss}
        logits = self._project_logits(x)
        out: dict[str, torch.Tensor] = {'logits': logits}
        if targets is not None:
            out['loss'] = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return out

def build_mixer(cfg: MambaLMConfig) -> nn.Module:
    name, ssm_kwargs = cfg.mixer_spec()
    return CausalLM._build_mixer(d_model=cfg.d_model, ssm_type=name, ssm_kwargs=ssm_kwargs, expand=float(cfg.expand_factor), d_conv=cfg.mixer_d_conv(), d_state=cfg.d_state)
_build_mixer = build_mixer
