"""Pure-torch causal multi-head attention mixer (the LM contract's reference arm).

The reference row of the LM contract: a Llama-3 style attention mixer that
drops into the shared backbone where an SSM mixer goes. Dependency-free: the
only kernel is ``torch.nn.functional.scaled_dot_product_attention`` with
``is_causal=True``.

Shape contract:
  * ``n_heads = d_model // head_dim`` (head_dim 64), no GQA -- q, k and v all
    carry ``n_heads`` heads, so every projection is ``d_model x d_model`` and
    the mixer costs EXACTLY ``4 * d_model**2`` parameters (q, k, v, out; no
    biases anywhere).
  * RoPE (base 500000, Llama-3's value) is applied to q and k only.

``out_proj`` feeds the block's residual branch and is tagged with
``mark_residual_out`` by the registry factory, so it receives the
``1/sqrt(n_residuals)`` depth rescale when the enclosing ``BlockStack`` is
constructed, like every other mixer's output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state import BaseMixer


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """The Llama/GPT-NeoX RoPE pairing: (x1, x2) -> (-x2, x1) on halves."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class Llama3Attention(BaseMixer):
    """Causal MHA with RoPE, ``forward(x[B, L, D]) -> y[B, L, D]``.

    Parameters
    ----------
    d_model : int
        Model width; must be a multiple of ``head_dim``. ``n_heads =
        d_model // head_dim``.
    head_dim : int, default 64
        Per-head width.
    rope_base : float, default 500000.0
        RoPE frequency base (Llama-3's value).

    Notes
    -----
    No GQA, no biases: q, k, v and out are each ``d_model x d_model``, so the
    mixer has exactly ``4 * d_model**2`` parameters. RoPE is applied to q and
    k only. Training-time reference only: no recurrent state, ``step`` and
    ``init_state`` raise.
    """

    def __init__(self, d_model: int, head_dim: int = 64,
                 rope_base: float = 500000.0):
        super().__init__()
        if d_model % head_dim != 0:
            raise ValueError(
                f"d_model={d_model} must be a multiple of head_dim={head_dim}")
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.rope_base = float(rope_base)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        inv_freq = 1.0 / (self.rope_base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # Non-persistent: a derived constant must not be pinned into checkpoints.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ rope
    def _rope(self, q: torch.Tensor, k: torch.Tensor, length: int):
        """Apply RoPE in fp32 to q, k of shape [B, H, L, head_dim]."""
        pos = torch.arange(length, device=q.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(q.device))       # [L, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)                    # [L, hd]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        qf, kf = q.float(), k.float()
        qo = qf * cos + _rotate_half(qf) * sin
        ko = kf * cos + _rotate_half(kf) * sin
        return qo.to(q.dtype), ko.to(k.dtype)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, state=None, return_state: bool = False):
        if state is not None or return_state:
            raise NotImplementedError(
                "Llama3Attention is a training-time reference mixer: it carries "
                "no recurrent state (attention's 'state' is the full KV cache). "
                "Use it for full-sequence forward/backward only.")
        b, length, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(b, length, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, length, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, length, h, hd).transpose(1, 2)
        q, k = self._rope(q, k, length)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, length, self.d_model)
        return self.out_proj(y)

    # ------------------------------------------------------------------ state
    def step(self, x_t, state):
        raise NotImplementedError(
            "Llama3Attention has no single-token recurrence (no KV cache).")

    def init_state(self, batch_size: int, device=None,
                   dtype: torch.dtype = torch.float32, **kwargs):
        raise NotImplementedError(
            "Llama3Attention has no recurrent state to allocate.")
