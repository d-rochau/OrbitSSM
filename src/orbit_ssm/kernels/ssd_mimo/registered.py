"""Registered scan and endpoint compiler boundaries (internal).

Prepared N128/G64/T64 SISO tensors, static decay, no rotation or carries.
The M2 boundary is opt-in. Endpoint local-map forward/backward are selected
under torch.compile, retaining Gluon without tracing its launcher. The
existing launchers own all kernel arithmetic. Saved tensors are explicit.
"""
from __future__ import annotations
import torch

from ..autograd_guard import no_double_backward as _no_double_backward


def _check_inputs(c, b, x, dw, dc, a, z, skip, norm):
    batch, heads, chunks, chunk = dc.shape
    if (chunk != 64 or b.shape != (batch, chunks, 64, 128) or c.shape != b.shape
            or x.shape != (batch, heads, 64, 1, chunks * 64)
            or dw.shape != (batch, heads, chunks * 64)
            or a.shape != (heads,) or z.shape != (batch, chunks * 64, heads * 64)
            or skip.shape != (heads, 64, 1) or norm.shape != (heads * 64,)):
        raise ValueError("registered M2 requires prepared N128/G64/T64 SISO operands")
    if not skip.is_contiguous():
        raise ValueError("registered M2 D-skip must be contiguous for Kc")
    if x.dtype not in (torch.float32, torch.bfloat16) or any(t.dtype != x.dtype for t in (b, c, z)):
        raise ValueError("registered M2 maps, drive and gate must share FP32/BF16 IO dtype")
    if any(t.dtype != torch.float32 for t in (dw, dc, a, skip, norm)):
        raise ValueError("registered M2 decay, skip and norm operands must be FP32")
    if not x.is_cuda or any(t.device != x.device for t in (c,b,dw,dc,a,z,skip,norm)):
        raise ValueError("registered M2 operands must share one CUDA device")

@torch.library.custom_op("orbit_ssm::m2_fwd", mutates_args=())
def m2_fwd(c: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
           dw: torch.Tensor, dc: torch.Tensor, a: torch.Tensor,
           z: torch.Tensor, skip: torch.Tensor, norm: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_inputs(c, b, x, dw, dc, a, z, skip, norm)
    from orbit_ssm.kernels.ssd_mimo.siso_fwd_kernel import siso_chunk_fwd
    from orbit_ssm.kernels.ssd_mimo.fused_fwd_kernel import seg_prefix_host
    from orbit_ssm.kernels.ssd_mimo.gate_epilogue import gate_norm_fwd
    # SPANDELTA (fused_bwd_kernel `_SPANDELTA_NOTE`): `dc` is the RAW
    # chunked delta; the chunk prefix is formed here and the backward
    # returns dL/ddelta directly.
    ypre, states, final, _ = siso_chunk_fwd(
        c, b, x, seg_prefix_host(dc, a), a, dw=dw, d_skip=skip, precision="io",
        store_sall=True, needs_bwd=True, y_token_major=True)
    y, rstd = gate_norm_fwd(ypre, z, 64, norm, 1e-5, out_dtype=x.dtype)
    return y, final, states, ypre, rstd


@m2_fwd.register_fake
def _fake(c, b, x, dw, dc, a, z, skip, norm):
    _check_inputs(c, b, x, dw, dc, a, z, skip, norm)
    from orbit_ssm.kernels.ssd_mimo.siso_fwd_kernel import siso_native_y
    batch, heads, chunks, chunk = dc.shape
    assert chunk == 64 and b.shape[-1] == 128 and x.shape[2:4] == (64, 1)
    def y():
        return siso_native_y(batch, heads, 64, chunks, 64, x.device, x.dtype, True)
    final = torch.empty((batch, heads, 64, 128), device=x.device, dtype=torch.float32)
    states = torch.empty((batch, heads, chunks, 128, 64), device=x.device, dtype=x.dtype)
    rstd = torch.empty((batch, chunks * 64), device=x.device, dtype=torch.float32)
    return y(), final, states, y(), rstd


@torch.library.custom_op("orbit_ssm::m2_bwd", mutates_args=())
def m2_bwd(c: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
           dw: torch.Tensor, dc: torch.Tensor, a: torch.Tensor,
           z: torch.Tensor, skip: torch.Tensor, norm: torch.Tensor,
           states: torch.Tensor, ypre: torch.Tensor, rstd: torch.Tensor,
           dy: torch.Tensor, dfinal: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                      torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from orbit_ssm.kernels.ssd_mimo.gate_epilogue import gate_norm_bwd
    from orbit_ssm.kernels.ssd_mimo.fused_bwd_kernel import fused_chunk_bwd_5kernel
    from orbit_ssm.kernels.ssd_mimo.fused_fwd_kernel import seg_prefix_host
    dypre, dz, dnorm = gate_norm_bwd(ypre, z, norm, rstd, dy, 64)
    # This is the existing rotation-free SISO forward/general backward
    # schedule, including Kc's ownership of both D-skip gradients.
    grads = fused_chunk_bwd_5kernel(
        c, b, x, seg_prefix_host(dc, a), a, states, dypre, dS_final=dfinal,
        precision="io", Gb=1, Gc=1, bc_grouped=False,
        dw=dw, d_skip=skip, dskip_dd=True)
    dx, dcmap, dbmap, ddc = grads[:4]
    da = (dc * ddc).sum(dim=(0, 2, 3))
    return (dcmap.to(c.dtype), dbmap.to(b.dtype), dx, grads[10], ddc, da,
            dz, grads[14].view(skip.shape).to(skip.dtype), dnorm.to(norm.dtype))


@m2_bwd.register_fake
def _bwd_fake(c, b, x, dw, dc, a, z, skip, norm, states, ypre, rstd, dy, dfinal):
    return tuple(torch.empty_like(t) for t in (c, b, x, dw, dc, a, z, skip, norm))


def _setup(ctx, inputs, output):
    _, _, states, ypre, rstd = output
    ctx.save_for_backward(*inputs, states, ypre, rstd)
    ctx.mark_non_differentiable(states, ypre, rstd)


@_no_double_backward("SSD-MIMO scan")
def _backward(ctx, dy, dfinal, _dstates, _dypre, _drstd):
    return m2_bwd(*ctx.saved_tensors, dy, dfinal)


m2_fwd.register_autograd(_backward, setup_context=_setup)


# Gluon launchers cannot be re-emitted by this Inductor version. Keep the
# endpoint forward and analytic backward opaque, with explicit saved inputs;
# surrounding Torch code remains in the graph. Eager uses the direct wrapper.
@torch.library.custom_op("orbit_ssm::endpoint_local_fwd", mutates_args=())
def endpoint_local_fwd(b: torch.Tensor, c: torch.Tensor,
                       phase: torch.Tensor | None, beta: torch.Tensor | None,
                       gamma: torch.Tensor | None, rb: torch.Tensor | None,
                       th: torch.Tensor | None, heads: int) -> torch.Tensor:
    from .endpoint import _LocalMap
    return _LocalMap.apply(b, c, phase, beta, gamma, rb, th, heads)


@endpoint_local_fwd.register_fake
def _endpoint_fake(b, c, phase, beta, gamma, rb, th, heads):
    return b.new_empty(b.shape[0], b.shape[1], heads, b.shape[-1], c.shape[-1], dtype=torch.float32)


@torch.library.custom_op("orbit_ssm::endpoint_local_bwd", mutates_args=())
def endpoint_local_bwd(dout: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                       phase: torch.Tensor | None, beta: torch.Tensor | None,
                       gamma: torch.Tensor | None, rb: torch.Tensor | None,
                       th: torch.Tensor | None, heads: int) -> list[torch.Tensor]:
    from types import SimpleNamespace
    from .endpoint import _LocalMap, _local_plan
    values, meta, flags, rows, kernel = _local_plan(b, c, phase, beta, gamma, rb, th, heads)
    ctx = SimpleNamespace(saved_tensors=values, meta=meta, flags=flags, rows=rows, kernel=kernel)
    grads = _LocalMap.backward(ctx, dout)[:-1]
    return [g if g is not None else b.new_empty(0) for g in grads]


@endpoint_local_bwd.register_fake
def _endpoint_bwd_fake(dout, b, c, phase, beta, gamma, rb, th, heads):
    return [torch.empty(v.shape, device=v.device, dtype=v.dtype)
            if v is not None else b.new_empty(0)
            for v in (b, c, phase, beta, gamma, rb, th)]


def _endpoint_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:-1])
    ctx.heads = inputs[-1]


@_no_double_backward("SSD-MIMO endpoint")
def _endpoint_backward(ctx, dout):
    values = ctx.saved_tensors
    grads = endpoint_local_bwd(dout, *values, ctx.heads)
    return (*(g if v is not None else None for g, v in zip(grads, values)), None)


endpoint_local_fwd.register_autograd(_endpoint_backward, setup_context=_endpoint_setup)
