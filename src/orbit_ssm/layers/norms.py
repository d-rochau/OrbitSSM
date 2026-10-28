"""Normalization and output-gating layers shared by HybridSSM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm_cast(x: torch.Tensor, weight: torch.Tensor, eps) -> torch.Tensor:
    """Autocast-safe RMSNorm: cast the (fp32) ``weight`` to ``x``'s dtype, then
    call the fused ``F.rms_norm`` ATen op.

    With a bf16 input and an fp32 weight ATen falls back to an unfused path
    (about 2x slower). ``F.rms_norm`` normalizes in a precision keyed off the
    input, so the cast changes only whether the fused kernel is taken, not
    the arithmetic.
    """
    return F.rms_norm(x, weight.shape, weight.to(x.dtype), eps)


class RMSNorm(nn.Module):
    """RMSNorm over the last axis: ``weight * x * rsqrt(mean(x**2) + eps)``.

    Parameters
    ----------
    d : int
        Normalized width (size of the last axis); ``weight`` has shape ``(d,)``
        and starts at ones.
    eps : float, default 1e-5
        Added to the mean square before ``rsqrt``.
    fused : bool, default False
        ``True`` routes forward through ``rms_norm_cast`` (one fused ATen
        kernel; bf16 in, fp32 accumulation, bf16 out). ``False`` is the
        hand-rolled pow/mean/rsqrt/mul chain, which under bf16 autocast runs
        the whole activation in fp32 and returns fp32. Both agree to fp
        tolerance and share the same parameter, so checkpoints are
        interchangeable.

    Notes
    -----
    Production mixer norms and the backbone norms (``ResidualBlock.norm`` /
    ``.norm_mlp``, ``BlockStack.norm_f``) use
    ``fused=True``; the layers are held to the independent oracle at its dtype's
    tolerance, not to a bit pattern.
    """

    def __init__(self, d: int, eps: float = 1e-5, fused: bool = False):
        super().__init__()
        self.eps = eps
        self.fused = bool(fused)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """``weight * x / rms(x)`` over the last dimension, in the input dtype."""
        if self.fused:
            return rms_norm_cast(x, self.weight, self.eps)
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)



def _swiglu_gate_eager(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return y * F.silu(z)


_swiglu_gate_compiled = None


def swiglu_gate(y: torch.Tensor, z: torch.Tensor,
               use_compile: bool = False) -> torch.Tensor:
    """``y * silu(z)``: the module-side SwiGLU gate for an SSM core without a
    fused gate.

    ``use_compile=True`` (opt-in) routes through a lazily built
    ``torch.compile`` wrapper that fuses ``silu`` and the multiply; the eager
    default is the plain two ops.
    """
    if use_compile:
        global _swiglu_gate_compiled
        if _swiglu_gate_compiled is None:
            _swiglu_gate_compiled = torch.compile(_swiglu_gate_eager)
        return _swiglu_gate_compiled(y, z)
    return _swiglu_gate_eager(y, z)


NORM_KERNELS = ("triton", "torch", "mamba_ssm")

_UNRESOLVED = object()
_rmsnorm_gated_fn = _UNRESOLVED


def _resolve_mamba_gated():
    """mamba_ssm's fused gated-RMSNorm entry, or None when the package is not importable.

    Resolved LAZILY, on the first explicit ``norm_kernel="mamba_ssm"`` use: importing mamba_ssm pulls
    in transformers (~4.5 s).  Only ImportError means "not installed"; any other error of an
    installed-but-broken package propagates."""
    global _rmsnorm_gated_fn
    if _rmsnorm_gated_fn is _UNRESOLVED:
        try:
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
        except ImportError:
            rmsnorm_fn = None
        _rmsnorm_gated_fn = rmsnorm_fn
    return _rmsnorm_gated_fn


def check_norm_kernel(kernel: str) -> str:
    """Validate a ``norm_kernel`` value; ``"mamba_ssm"`` raises, by name, when the package is missing."""
    if kernel not in NORM_KERNELS:
        raise ValueError(f"norm_kernel must be one of {NORM_KERNELS}, got {kernel!r}")
    if kernel == "mamba_ssm" and _resolve_mamba_gated() is None:
        raise ImportError("kernel_options norm_kernel='mamba_ssm' needs the compiled `mamba_ssm` package "
                          "(mamba_ssm.ops.triton.layernorm_gated), which is not importable; "
                          "use norm_kernel='triton' (the default) or 'torch'")
    return kernel


def _gated_rms_norm_route(y: torch.Tensor, weight: torch.Tensor | None,
                          z: torch.Tensor | None = None, kernel: str = "triton") -> str:
    """The route `gated_rms_norm` takes for these operands: ``"triton"`` (our fused kernel),
    ``"mamba_ssm"`` (explicit opt-in), ``"rms_norm"`` (``F.rms_norm``), ``"formula"`` (the plain
    expression; CPU / fp64) or ``"none"`` (``weight=None``: gate only, no norm)."""
    if weight is None:
        return "none"
    if not (y.is_cuda and y.dtype in (torch.float32, torch.float16, torch.bfloat16)):
        return "formula"
    if z is not None and kernel in ("triton", "mamba_ssm"):
        return kernel
    return "rms_norm"


def gated_rms_norm(y: torch.Tensor, weight: torch.Tensor | None, eps: float,
                   z: torch.Tensor | None = None, kernel: str = "triton") -> torch.Tensor:
    """``rms_norm(y * silu(z)) * weight`` -- a mixer's whole output epilogue.

    ``weight=None`` drops the scale, ``z=None`` drops the gate::

        g = y if z is None else y * silu(z)
        out = g * rsqrt(mean(g**2, dim=-1, keepdim=True) + eps) * weight

    ``kernel`` (the ``norm_kernel`` option) picks the CUDA fp16/bf16/fp32 route:

    * ``"triton"`` (default): `kernels.gated_norm`, our fused forward + backward, fp32 inside;
    * ``"torch"``: ``F.rms_norm`` on ``y * silu(z)``;
    * ``"mamba_ssm"``: mamba_ssm's fused kernel; raises when the package is missing.

    CPU and fp64 always evaluate the plain formula.  The routes agree up to rounding; the fused ones
    accumulate the mean in fp32 where the eager chain does so in bf16 under autocast.
    """
    route = _gated_rms_norm_route(y, weight, z, kernel)
    if route == "triton":
        from ..kernels.gated_norm import gated_rms_norm_triton
        return gated_rms_norm_triton(y, weight, eps, z)
    if route == "mamba_ssm":
        fn = _resolve_mamba_gated()
        if fn is None:
            check_norm_kernel("mamba_ssm")
        shape = y.shape
        return fn(y.reshape(-1, shape[-1]), weight, None, z=z.reshape(-1, shape[-1]), eps=eps,
                  norm_before_gate=False).view(shape)
    if z is not None:
        y = y * F.silu(z)
    if weight is None:
        return y
    if route == "rms_norm":
        return F.rms_norm(y, weight.shape, weight.to(y.dtype), eps)
    return weight * y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
