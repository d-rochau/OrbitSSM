
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_intermediate: int):
        super().__init__()
        if d_intermediate < 1:
            raise ValueError(f'd_intermediate must be positive, got {d_intermediate}')
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.gate = nn.Linear(d_model, d_intermediate, bias=False)
        self.up = nn.Linear(d_model, d_intermediate, bias=False)
        self.down = nn.Linear(d_intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
