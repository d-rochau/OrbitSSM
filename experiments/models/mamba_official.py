
from __future__ import annotations
from functools import partial
from typing import Any
import torch.nn as nn

class OfficialMambaLMHeadModel(nn.Module):

    def __init__(self, variant: str, vocab_size: int, d_model: int, n_layers: int, *, d_state: int | None=None, d_conv: int=4, expand_factor: int=2, headdim: int=64, ngroups: int=1, chunk_size: int | None=None, tie_word_embeddings: bool=True, rms_eps: float=1e-05, residual_in_fp32: bool=True, mamba3_hybrid_kwargs: dict[str, Any] | None=None):
        super().__init__()
        from mamba_ssm.models.mixer_seq_simple import _init_weights
        from mamba_ssm.modules.block import Block
        self.variant = variant
        self.embedding = nn.Embedding(vocab_size, d_model)
        norm_cls = partial(nn.RMSNorm, eps=rms_eps)
        self.blocks = nn.ModuleList([Block(dim=d_model, mixer_cls=_mixer_cls(variant, layer_idx=layer_idx, n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand_factor=expand_factor, headdim=headdim, ngroups=ngroups, chunk_size=chunk_size, mamba3_hybrid_kwargs=mamba3_hybrid_kwargs), mlp_cls=nn.Identity, norm_cls=norm_cls, fused_add_norm=False, residual_in_fp32=residual_in_fp32) for layer_idx in range(n_layers)])
        self.norm_f = nn.RMSNorm(d_model, eps=rms_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(partial(_init_weights, n_layer=n_layers, n_residuals_per_layer=1))
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, position_ids=None, state=None, return_embeddings=False):
        hidden_states = self.embedding(input_ids)
        residual = None
        for block in self.blocks:
            hidden_states, residual = block(hidden_states, residual)
        hidden_states = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm_f(hidden_states)
        if return_embeddings:
            return hidden_states
        return self.lm_head(hidden_states)

def _mixer_cls(variant: str, *, layer_idx: int, n_layers: int, d_state: int | None, d_conv: int, expand_factor: int, headdim: int, ngroups: int, chunk_size: int | None, mamba3_hybrid_kwargs: dict[str, Any] | None=None):
    if variant == 'official_mamba1':
        from mamba_ssm.modules.mamba_simple import Mamba
        return partial(Mamba, d_state=16 if d_state is None else d_state, d_conv=d_conv, expand=expand_factor, use_fast_path=True, layer_idx=layer_idx)
    if variant == 'official_mamba2':
        from mamba_ssm import Mamba2
        return partial(Mamba2, d_state=128 if d_state is None else d_state, d_conv=d_conv, expand=expand_factor, headdim=headdim, ngroups=ngroups, chunk_size=256 if chunk_size is None else chunk_size, layer_idx=layer_idx)
    if variant == 'official_mamba3':
        from mamba_ssm import Mamba3
        return partial(Mamba3, d_state=128 if d_state is None else d_state, expand=expand_factor, headdim=headdim, ngroups=ngroups, chunk_size=64 if chunk_size is None else chunk_size, layer_idx=layer_idx, n_layer=n_layers)
    raise ValueError(f'Unsupported official Mamba variant: {variant}')
