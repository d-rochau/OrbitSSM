
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def rms_norm_cast(x: torch.Tensor, weight: torch.Tensor, eps) -> torch.Tensor:
    return F.rms_norm(x, weight.shape, weight.to(x.dtype), eps)

class RMSNorm(nn.Module):

    def __init__(self, d: int, eps: float=1e-05, fused: bool=False):
        super().__init__()
        self.eps = eps
        self.fused = bool(fused)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        if self.fused:
            return rms_norm_cast(x, self.weight, self.eps)
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def _swiglu_gate_eager(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return y * F.silu(z)
_swiglu_gate_compiled = None

def swiglu_gate(y: torch.Tensor, z: torch.Tensor, use_compile: bool=False) -> torch.Tensor:
    if use_compile:
        global _swiglu_gate_compiled
        if _swiglu_gate_compiled is None:
            _swiglu_gate_compiled = torch.compile(_swiglu_gate_eager)
        return _swiglu_gate_compiled(y, z)
    return _swiglu_gate_eager(y, z)
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as _rmsnorm_gated_fn
except Exception:
    _rmsnorm_gated_fn = None

def gated_rms_norm(y: torch.Tensor, weight: torch.Tensor | None, eps: float, z: torch.Tensor | None=None) -> torch.Tensor:
    if z is not None and (_rmsnorm_gated_fn is None or not y.is_cuda or y.dtype not in (torch.float32, torch.float16, torch.bfloat16) or (weight is None)):
        y = y * F.silu(z)
        z = None
    if weight is None:
        return y
    if z is None:
        if y.is_cuda and y.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return F.rms_norm(y, weight.shape, weight.to(y.dtype), eps)
        return weight * y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    shape = y.shape
    out = _rmsnorm_gated_fn(y.reshape(-1, shape[-1]), weight, None, z=z.reshape(-1, shape[-1]), eps=eps, norm_before_gate=False)
    return out.view(shape)
