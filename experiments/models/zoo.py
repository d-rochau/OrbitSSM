
from __future__ import annotations
import math
import torch.nn as nn
from experiments.models.mamba_official import OfficialMambaLMHeadModel
from experiments.models.mlstm import XLSTMLargeWrapper
from experiments.models.transformers import LlamaWrapper
BASELINE_NAMES = {'llama3', 'attention', 'mlstm_large', 'official_mamba1', 'official_mamba2', 'official_mamba3'}

def _state_mixer_config(state_mixer: str | None):
    from zoology.config import ModuleConfig
    if state_mixer in (None, 'default'):
        return ModuleConfig(name='zoology.mixers.mlp.MLP', kwargs={'hidden_mult': 4})
    if state_mixer == 'identity':
        return ModuleConfig(name='torch.nn.Identity', kwargs={})
    if state_mixer == 'mlp':
        return ModuleConfig(name='zoology.mixers.mlp.MLP', kwargs={'hidden_mult': 4})
    raise ValueError(f'Unsupported state_mixer setting: {state_mixer}')

def _build_zoology_lm(language_model_cls, model_cfg):
    import torch
    import zoology.model as zm
    base = zm.TokenEmbeddings

    class DeviceFollowingTokenEmbeddings(base):

        def __init__(self, *args, **kw):
            kw.setdefault('device', 'cpu')
            super().__init__(*args, **kw)

        def forward(self, input_ids, position_ids=None):
            if position_ids is None and self.max_position_embeddings > 0:
                position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
            return super().forward(input_ids, position_ids=position_ids)
    zm.TokenEmbeddings = DeviceFollowingTokenEmbeddings
    try:
        return language_model_cls(model_cfg)
    finally:
        zm.TokenEmbeddings = base

def build_mqmkvar_model(model_spec, data_spec):
    kwargs = dict(model_spec.kwargs)
    vocab_size = int(data_spec.kwargs.get('vocab_size', 8192))
    d_model = int(kwargs.pop('d_model', 128))
    n_layers = int(kwargs.pop('n_layers', 2))
    if model_spec.name == 'llama3':
        n_heads = kwargs.pop('num_heads', None)
        num_key_value_heads = kwargs.pop('num_key_value_heads', None)
        intermediate_size = kwargs.pop('intermediate_size', None)
        max_position_embeddings = int(kwargs.pop('max_position_embeddings', 8192))
        attention_dropout = float(kwargs.pop('attention_dropout', 0.0))
        rms_norm_eps = float(kwargs.pop('rms_norm_eps', 1e-05))
        tie_word_embeddings = bool(kwargs.pop('tie_word_embeddings', True))
        if kwargs:
            raise ValueError(f'Unsupported llama3 model kwargs: {sorted(kwargs)}')
        return LlamaWrapper(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, num_key_value_heads=num_key_value_heads, intermediate_size=intermediate_size, max_position_embeddings=max_position_embeddings, attention_dropout=attention_dropout, rms_norm_eps=rms_norm_eps, tie_word_embeddings=tie_word_embeddings)
    if model_spec.name == 'attention':
        from zoology.config import ModelConfig, ModuleConfig
        from zoology.model import LanguageModel
        num_heads = int(kwargs.pop('num_heads', 2))
        dropout = float(kwargs.pop('dropout', 0.1))
        max_position_embeddings = int(kwargs.pop('max_position_embeddings', 8192))
        state_mixer = kwargs.pop('state_mixer', 'identity')
        resid_dropout = float(kwargs.pop('resid_dropout', dropout))
        embed_dropout = float(kwargs.pop('embed_dropout', dropout))
        bias = bool(kwargs.pop('bias', True))
        if kwargs:
            raise ValueError(f'Unsupported attention model kwargs: {sorted(kwargs)}')
        model_cfg = ModelConfig(block_type='TransformerBlock', d_model=d_model, n_layers=n_layers, sequence_mixer=ModuleConfig(name='zoology.mixers.attention.MHA', kwargs={'dropout': dropout, 'num_heads': num_heads, 'bias': bias}), state_mixer=_state_mixer_config(state_mixer), max_position_embeddings=max_position_embeddings, vocab_size=vocab_size, name=model_spec.name, resid_dropout=resid_dropout, embed_dropout=embed_dropout)
        return _build_zoology_lm(LanguageModel, model_cfg)
    if model_spec.name == 'mlstm_large':
        num_heads = int(kwargs.pop('num_heads', 4))
        kernel_mode = str(kwargs.pop('kernel_mode', 'fp32'))
        chunk_size = int(kwargs.pop('chunk_size', 64))
        qk_dim_factor = float(kwargs.pop('qk_dim_factor', 0.5))
        v_dim_factor = float(kwargs.pop('v_dim_factor', 1.0))
        ffn_proj_factor = float(kwargs.pop('ffn_proj_factor', 2.6667))
        gate_soft_cap = float(kwargs.pop('gate_soft_cap', 15.0))
        output_logit_soft_cap = float(kwargs.pop('output_logit_soft_cap', 30.0))
        weight_mode = str(kwargs.pop('weight_mode', 'single'))
        if kwargs:
            raise ValueError(f'Unsupported mlstm_large model kwargs: {sorted(kwargs)}')
        return XLSTMLargeWrapper(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, kernel_mode=kernel_mode, chunk_size=chunk_size, qk_dim_factor=qk_dim_factor, v_dim_factor=v_dim_factor, ffn_proj_factor=ffn_proj_factor, gate_soft_cap=gate_soft_cap, output_logit_soft_cap=output_logit_soft_cap, weight_mode=weight_mode)
    if model_spec.name in {'official_mamba1', 'official_mamba2', 'official_mamba3'}:
        d_state = kwargs.pop('d_state', None)
        d_conv = int(kwargs.pop('d_conv', 4))
        expansion = kwargs.pop('expand_factor', 2)
        if model_spec.name == 'official_mamba3':
            expand_factor = float(expansion)
            inner = expand_factor * d_model
            if not math.isfinite(inner) or inner <= 0 or (not inner.is_integer()):
                raise ValueError('Mamba3 expand_factor*d_model must be an exact positive integer')
        else:
            expand_factor = int(expansion)
        headdim = int(kwargs.pop('headdim', 64))
        ngroups = int(kwargs.pop('ngroups', 1))
        chunk_size = kwargs.pop('chunk_size', None)
        tie_word_embeddings = bool(kwargs.pop('tie_word_embeddings', True))
        rms_eps = float(kwargs.pop('rms_eps', 1e-05))
        residual_in_fp32 = bool(kwargs.pop('residual_in_fp32', True))
        if kwargs:
            raise ValueError(f'Unsupported official Mamba model kwargs for {model_spec.name}: {sorted(kwargs)}')
        return OfficialMambaLMHeadModel(variant=model_spec.name, vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_state=None if d_state is None else int(d_state), d_conv=d_conv, expand_factor=expand_factor, headdim=headdim, ngroups=ngroups, chunk_size=None if chunk_size is None else int(chunk_size), tie_word_embeddings=tie_word_embeddings, rms_eps=rms_eps, residual_in_fp32=residual_in_fp32)
    from types import SimpleNamespace
    from orbit_ssm.harness.integration import build_model
    spec_kwargs = dict(model_spec.kwargs)
    spec_kwargs['d_model'] = d_model
    spec_kwargs['n_layers'] = n_layers
    declared = spec_kwargs.get('mixer')
    if declared is not None and declared != model_spec.name:
        raise ValueError(f'model name {model_spec.name!r} disagrees with mixer= {declared!r}; drop the mixer kwarg (the name is the family).')
    spec_kwargs['mixer'] = model_spec.name
    init_override = spec_kwargs.pop('init_override', None)
    model = build_model(SimpleNamespace(name=model_spec.name, kwargs=spec_kwargs), data_spec)
    if init_override is not None:
        if init_override != 'kaiming_proj':
            raise ValueError(f'unknown init_override {init_override!r}')
        hit = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name.rsplit('.', 1)[-1] in ('in_proj', 'B_proj', 'C_proj'):
                nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
                hit.append(name)
        if not hit:
            raise ValueError("init_override='kaiming_proj' matched no projection")
        model._init_override = {'kind': init_override, 'reinit': hit}
    return model

def count_trainable_parameters(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def count_parameter_breakdown(model: nn.Module, **kwargs) -> dict:

    def _count(module):
        return sum((p.numel() for p in module.parameters() if p.requires_grad))
    total = _count(model)
    result: dict = {'total': total}
    if hasattr(model, 'embedding'):
        emb = _count(model.embedding)
        result['embedding'] = emb
        result['non_embedding'] = total - emb
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'embeddings'):
        emb = _count(model.backbone.embeddings)
        result['embedding'] = emb
        result['non_embedding'] = total - emb
        pos = getattr(model.backbone.embeddings, 'position_embeddings', None)
        result['position_embedding'] = 0 if pos is None else _count(pos)
    elif hasattr(model, 'llama') and hasattr(model.llama, 'embed_tokens'):
        emb = _count(model.llama.embed_tokens)
        result['embedding'] = emb
        result['non_embedding'] = total - emb
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'backbone') and hasattr(backbone, 'embedding'):
            inner = backbone
            emb = _count(inner.embedding)
            result['embedding'] = emb
            result['non_embedding'] = total - emb
            if hasattr(inner, 'lm_head'):
                result['lm_head'] = _count(inner.lm_head)
            backbone = inner.backbone
        result['backbone'] = _count(backbone)
        if hasattr(backbone, 'layers'):
            ssm_total = sum((_count(layer.ssm) for layer in backbone.layers if hasattr(layer, 'ssm')))
            result['ssm'] = ssm_total
        if hasattr(backbone, 'embeddings'):
            result['backbone'] -= _count(backbone.embeddings)
    return result
