"""Fused gated RMSNorm, ``rms_norm(y * silu(z)) * weight``: forward and backward Triton kernels.

One program per row; the row (width ``N``, padded to a power of two) lives in registers, every
reduction and the whole elementwise chain run in fp32, only the stores round to the IO dtype.
``y`` and ``z`` may be fp16 / bf16 / fp32 (mixed is fine), ``weight`` any float dtype.

Forward saves ``y``, ``z`` (the caller's own tensors) and the per-row ``rstd`` (fp32); the backward
recomputes ``g = y * silu(z)``.  With ``xh = g * rstd`` and ``u = dout * weight``::

    dg = rstd * (u - xh * mean(xh * u))
    dy = dg * silu(z)             dz = dg * y * silu'(z)        dweight = sum_rows dout * xh

``dweight`` is reduced in two stages: each of ``G`` programs strides over its rows into a private
fp32 partial, and the host sums the ``G`` partials (no atomics, deterministic).

Written after ``mamba_ssm.ops.triton.layernorm_gated`` (Apache-2.0) for the layout of the problem
only; nothing is shared with it.  Runs under ``TRITON_INTERPRET=1`` on the CPU.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["gated_rms_norm_triton"]

_MAX_N = 1 << 16          # one row in registers; wider rows need a looped variant


@triton.jit
def _fwd_kernel(Y, Z, W, OUT, RSTD, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    y = tl.load(Y + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    g = y * (z * tl.sigmoid(z))
    rstd = 1.0 / tl.sqrt(tl.sum(g * g, axis=0) / N + eps)
    tl.store(OUT + row * N + cols, (g * rstd * w).to(OUT.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _bwd_kernel(Y, Z, W, DOUT, RSTD, DY, DZ, DWP, M, N, G, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(pid, M, G):
        row = pid.to(tl.int64) + (r - pid)       # int64 offsets (r itself is int32)
        y = tl.load(Y + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(Z + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DOUT + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)
        sg = tl.sigmoid(z)
        sz = z * sg
        xh = y * sz * rstd
        u = do * w
        c1 = tl.sum(xh * u, axis=0) / N
        dg = rstd * (u - xh * c1)
        dw += do * xh
        tl.store(DY + row * N + cols, (dg * sz).to(DY.dtype.element_ty), mask=mask)
        tl.store(DZ + row * N + cols,
                 (dg * y * sg * (1.0 + z * (1.0 - sg))).to(DZ.dtype.element_ty), mask=mask)
    tl.store(DWP + pid * N + cols, dw, mask=mask)


def _programs(x: torch.Tensor, rows: int) -> int:
    if x.is_cuda:
        return max(1, min(rows, 4 * torch.cuda.get_device_properties(x.device).multi_processor_count))
    return max(1, min(rows, 8))


def _launch_params(N: int) -> tuple[int, int]:
    block = triton.next_power_of_2(N)
    if block > _MAX_N:
        raise ValueError(f"gated RMSNorm kernel: width {N} exceeds {_MAX_N}; "
                         "use norm_kernel='torch' for this width")
    return block, (1 if block <= 1024 else 4 if block <= 8192 else 8)


class _GatedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, z, weight, eps):
        M, N = y.shape
        block, warps = _launch_params(N)
        out = torch.empty_like(y)
        rstd = torch.empty(M, device=y.device, dtype=torch.float32)
        if M:
            _fwd_kernel[(M,)](y, z, weight, out, rstd, N, eps, BLOCK_N=block, num_warps=warps)
        ctx.save_for_backward(y, z, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, dout):
        y, z, weight, rstd = ctx.saved_tensors
        M, N = y.shape
        block, warps = _launch_params(N)
        dout = dout.contiguous()
        dy, dz = torch.empty_like(y), torch.empty_like(z)
        G = _programs(y, M)
        dwp = torch.empty(G, N, device=y.device, dtype=torch.float32)
        if M:
            _bwd_kernel[(G,)](y, z, weight, dout, rstd, dy, dz, dwp, M, N, G,
                              BLOCK_N=block, num_warps=warps)
        else:
            dwp.zero_()
        dw = dwp.sum(0).to(weight.dtype) if ctx.needs_input_grad[2] else None
        return dy, dz, dw, None


def gated_rms_norm_triton(y: torch.Tensor, weight: torch.Tensor, eps: float,
                          z: torch.Tensor) -> torch.Tensor:
    """``rms_norm(y * silu(z)) * weight`` over the last axis, one fused forward and one fused backward.

    ``y`` and ``z`` share a shape ``(..., N)``; ``weight`` is ``(N,)``.  The output has ``y``'s
    dtype and shape.  Statistics and the elementwise chain are fp32 whatever the IO dtype.
    """
    if y.shape != z.shape or weight.shape != y.shape[-1:]:
        raise ValueError(f"gated_rms_norm_triton: y {tuple(y.shape)}, z {tuple(z.shape)}, "
                         f"weight {tuple(weight.shape)} do not match")
    shape = y.shape
    out = _GatedRMSNormFn.apply(y.reshape(-1, shape[-1]).contiguous(),
                                z.reshape(-1, shape[-1]).contiguous(),
                                weight.contiguous(), float(eps))
    return out.view(shape)
