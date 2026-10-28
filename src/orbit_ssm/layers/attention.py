
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import BaseMixer

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = (x[..., :half], x[..., half:])
    return torch.cat((-x2, x1), dim=-1)

class Llama3Attention(BaseMixer):

    def __init__(self, d_model: int, head_dim: int=64, rope_base: float=500000.0):
        super().__init__()
        if d_model % head_dim != 0:
            raise ValueError(f'd_model={d_model} must be a multiple of head_dim={head_dim}')
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.rope_base = float(rope_base)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        inv_freq = 1.0 / self.rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _rope(self, q: torch.Tensor, k: torch.Tensor, length: int):
        pos = torch.arange(length, device=q.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(q.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        qf, kf = (q.float(), k.float())
        qo = qf * cos + _rotate_half(qf) * sin
        ko = kf * cos + _rotate_half(kf) * sin
        return (qo.to(q.dtype), ko.to(k.dtype))

    def forward(self, x: torch.Tensor, state=None, return_state: bool=False):
        if state is not None or return_state:
            raise NotImplementedError("Llama3Attention is a training-time reference mixer: it carries no recurrent state (attention's 'state' is the full KV cache). Use it for full-sequence forward/backward only.")
        b, length, _ = x.shape
        h, hd = (self.n_heads, self.head_dim)
        q = self.q_proj(x).view(b, length, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, length, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, length, h, hd).transpose(1, 2)
        q, k = self._rope(q, k, length)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, length, self.d_model)
        return self.out_proj(y)

    def step(self, x_t, state):
        raise NotImplementedError('Llama3Attention has no single-token recurrence (no KV cache).')

    def init_state(self, batch_size: int, device=None, dtype: torch.dtype=torch.float32, **kwargs):
        raise NotImplementedError('Llama3Attention has no recurrent state to allocate.')
