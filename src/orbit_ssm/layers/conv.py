
from __future__ import annotations
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
except ImportError:
    _causal_conv1d_fn = None

def _buffer(x):
    if x.stride(1) == 1:
        b, c, length = x.shape
        return torch.empty((b, length, c), device=x.device, dtype=x.dtype).transpose(1, 2)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)

@torch.library.custom_op('orbit_ssm::conv_fwd', mutates_args=())
def conv_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: bool) -> torch.Tensor:
    from causal_conv1d.cpp_functions import _causal_conv1d_fwd_cpp
    out = _buffer(x)
    _causal_conv1d_fwd_cpp(x, weight, bias, None, None, out, None, silu)
    return out

@conv_fwd.register_fake
def _conv_fake(x, weight, bias, silu):
    return _buffer(x)

@torch.library.custom_op('orbit_ssm::conv_bwd', mutates_args=(), schema='(Tensor x, Tensor weight, Tensor? bias, Tensor dy, bool silu) -> (Tensor, Tensor, Tensor?)')
def conv_bwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, dy: torch.Tensor, silu: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
    if dy.stride(1) != 1 and dy.stride(2) != 1:
        dy = dy.contiguous()
    dx, dw, db, _ = causal_conv1d_bwd_function(x, weight, bias, dy, None, None, None, _buffer(x), False, silu)
    return (dx, dw, db)

@conv_bwd.register_fake
def _conv_bwd_fake(x, weight, bias, dy, silu):
    return (_buffer(x), torch.empty_like(weight), None if bias is None else torch.empty_like(bias))

def _setup(ctx, inputs, output):
    x, weight, bias, ctx.silu = inputs
    ctx.save_for_backward(x, weight, bias)

def _backward(ctx, dy):
    return (*conv_bwd(*ctx.saved_tensors, dy, ctx.silu), None)
conv_fwd.register_autograd(_backward, setup_context=_setup)

def causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, d_conv: int, activation: str | None=None, allow_channel_last: bool=False) -> torch.Tensor:
    return causal_depthwise_conv1d_cl(x.transpose(1, 2), weight, bias, d_conv, activation, allow_channel_last=allow_channel_last).transpose(1, 2)

def causal_conv1d_cl_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None=None, out: torch.Tensor | None=None) -> torch.Tensor:
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, _causal_conv1d_fwd_cpp
    xc = x if x.stride(2) == 1 or x.stride(1) == 1 else x.contiguous()
    silu = activation in ('silu', 'swish')
    bias = None if bias is None else bias.contiguous()
    if out is None:
        return causal_conv1d_fwd_function(xc, weight.squeeze(1), bias, None, None, None, silu)
    _causal_conv1d_fwd_cpp(x=xc, weight=weight.squeeze(1), bias=bias, seq_idx=None, initial_states=None, out=out, final_states_out=None, silu_activation=silu)
    return out

def causal_conv1d_cl_bwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, dout: torch.Tensor, activation: str | None=None, dx: torch.Tensor | None=None):
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()
    dx, dw, db, _ = causal_conv1d_bwd_function(x, weight.squeeze(1), None if bias is None else bias.contiguous(), dout, None, None, None, dx, False, activation in ('silu', 'swish'))
    return (dx, dw.view(weight.shape), db)

def causal_depthwise_conv1d_cl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, d_conv: int, activation: str | None=None, allow_channel_last: bool=False) -> torch.Tensor:
    L = x.shape[-1]
    _fast_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if _causal_conv1d_fn is not None and x.is_cuda and (x.dtype in _fast_dtypes):
        xk = x if allow_channel_last and x.stride(1) == 1 else x.contiguous()
        if torch.compiler.is_compiling() and activation in (None, 'silu', 'swish'):
            return conv_fwd(xk, weight.squeeze(1), bias, activation is not None)
        return _causal_conv1d_fn(xk, weight.squeeze(1), bias, activation=activation)
    y = F.conv1d(x, weight, bias, padding=d_conv - 1, groups=x.shape[1])
    y = y[..., :L]
    if activation == 'silu':
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f'unsupported activation {activation!r}')
    return y

def channel_last_conv_view_ok(x: torch.Tensor) -> bool:
    if torch.compiler.is_compiling():
        return False
    return x.is_cuda and x.dim() == 3 and (x.stride(-1) == 1) and (x.dtype in (torch.float32, torch.float16, torch.bfloat16)) and (x.shape[-1] % 8 == 0) and (x.stride(0) % 8 == 0) and (x.stride(1) % 8 == 0) and (x.storage_offset() % 8 == 0)

class _NarrowChannelLastConvFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, activation):
        xcl = x.transpose(1, 2)
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype).transpose(1, 2)
        causal_conv1d_cl_fwd(xcl, weight, bias, activation, out=out)
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        return out.transpose(1, 2)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        if not dy.is_contiguous():
            dy = dy.contiguous()
        dxcl = torch.empty(x.shape, device=x.device, dtype=x.dtype).transpose(1, 2)
        dx, dw, db = causal_conv1d_cl_bwd(x.transpose(1, 2), weight, bias, dy.transpose(1, 2), ctx.activation, dx=dxcl)
        dx = dx.transpose(1, 2)
        return (dx if ctx.needs_input_grad[0] else None, dw if ctx.needs_input_grad[1] else None, db if bias is not None and ctx.needs_input_grad[2] else None, None)

def causal_depthwise_conv1d_narrow_cl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None=None) -> torch.Tensor:
    return _NarrowChannelLastConvFn.apply(x, weight, bias, activation)

class CausalDepthwiseConv(nn.Module):

    def __init__(self, channels: int, d_conv: int=4):
        super().__init__()
        self.channels, self.d_conv = (channels, d_conv)
        self.conv = nn.Conv1d(channels, channels, kernel_size=d_conv, groups=channels, padding=d_conv - 1, bias=True)

    def forward(self, x):
        return causal_depthwise_conv1d(x, self.conv.weight, self.conv.bias, self.d_conv, activation='silu')

    def forward_with_prefix(self, x, prefix=None):
        Bsz, _, C = x.shape
        if prefix is None:
            prefix = x.new_zeros(Bsz, self.d_conv - 1, C)
        xc = torch.cat([prefix, x], dim=1)
        y = torch.nn.functional.silu(torch.nn.functional.conv1d(xc.transpose(1, 2), self.conv.weight.to(x.dtype), self.conv.bias.to(x.dtype), groups=C).transpose(1, 2))
        tail = xc[:, -(self.d_conv - 1):] if self.d_conv > 1 else xc[:, :0]
        return (y, tail if torch.is_grad_enabled() else tail.detach().clone())

    def init_state(self, batch_size, device=None, dtype=torch.float32):
        return torch.zeros(batch_size, self.channels, self.d_conv, device=device, dtype=dtype)

    def step(self, x_t, state):
        state = torch.cat([state[:, :, 1:], x_t.unsqueeze(-1)], dim=-1)
        w = self.conv.weight.squeeze(1).to(x_t.dtype)
        y = (state * w).sum(-1) + self.conv.bias.to(x_t.dtype)
        return (torch.nn.functional.silu(y), state)
