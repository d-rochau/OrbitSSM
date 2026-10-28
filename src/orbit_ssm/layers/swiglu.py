"""Llama-3 style gated MLP sublayer (the LM backbone's second residual branch).

The LM contract's backbone is, per block,

    x = x + mixer(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

so arms differ by their mixer alone.  This module is the second half of that
block: the function of ``mamba_ssm.modules.mlp.GatedMLP`` with ``bias=False``,

    down( silu(gate(x)) * up(x) )

as three bias-free ``nn.Linear``, hence exactly ``3 * d_mlp * d_model``
parameters (``GatedMLP`` fuses gate and up into one ``fc1`` matrix).

``down`` feeds a residual branch, so ``ResidualBlock`` tags it with
``mark_residual_out`` and the enclosing ``BlockStack`` rescales it at
construction with depth ``2 * n_layers`` (mamba_ssm's
``n_residuals_per_layer = 2`` convention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))``: three bias-free Linear layers.

    Parameters
    ----------
    d_model : int
        Input and output width.
    d_mlp : int
        Hidden width (positive); the Llama-style choice is about
        ``8/3 * d_model`` rounded to a hardware-friendly multiple.
    """

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        if d_mlp < 1:
            raise ValueError(
                f"d_mlp must be positive, got {d_mlp}")
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.gate = nn.Linear(d_model, d_mlp, bias=False)
        self.up = nn.Linear(d_model, d_mlp, bias=False)
        self.down = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
