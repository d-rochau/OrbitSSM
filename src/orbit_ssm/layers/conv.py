"""Depthwise causal conv1d: the native torch path, the opt-in ``causal_conv1d`` package routes, and the
CausalDepthwiseConv module.

A layer's ``conv_kernel`` option picks the conv: ``"auto"`` (default) and ``"triton"`` run the plain-Triton conv
(`kernels.conv_triton`: any backend, no package), ``"causal_conv1d"`` is the OPT-IN compiled package (raises by name
when it is missing) and ``"torch"`` is ``F.conv1d``.  The Triton conv is within 1.1x of the package at the LM shapes;
all of them compute the same convolution up to rounding.  The
resolved choice is the layers' ``conv_kernel_in_use``; they pass it as ``package=`` to the conv entry points.
"""
from __future__ import annotations
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import portable as _portable

CONV_KERNELS = ("auto", "causal_conv1d", "torch", "triton")
_UNRESOLVED = object()
_causal_conv1d_fn = _UNRESOLVED


def package_conv_fn():
    """The ``causal_conv1d`` package's ``causal_conv1d_fn``, or None when it is not importable (resolved on
    first use; only ImportError means "not installed").  None under kernel_target="portable": the package is a
    compiled CUDA extension, not Triton, so the portable path runs torch's conv."""
    global _causal_conv1d_fn
    if _portable.explicit():
        return None
    if _causal_conv1d_fn is _UNRESOLVED:
        try:
            from causal_conv1d import causal_conv1d_fn
        except ImportError:
            causal_conv1d_fn = None
        _causal_conv1d_fn = causal_conv1d_fn
    return _causal_conv1d_fn


def check_conv_kernel(kernel: str) -> bool:
    """Validate a ``conv_kernel`` value; True when the convolutions run on the package, i.e. only ``"causal_conv1d"``
    (an opt-in: ``"auto"`` never takes the package).  ``"causal_conv1d"`` raises by name when it is missing."""
    if kernel not in CONV_KERNELS:
        raise ValueError(f"conv_kernel must be one of {CONV_KERNELS}, got {kernel!r}")
    if kernel == "causal_conv1d" and package_conv_fn() is None:
        raise ImportError("kernel_options conv_kernel='causal_conv1d' needs the compiled `causal_conv1d` "
                          "package, which is not importable (pip install causal-conv1d); conv_kernel='auto' "
                          "(the default, the plain-Triton conv) or 'torch' run without it" + (
                              "; kernel_target='portable' excludes the package (a compiled CUDA extension, not Triton)"
                              if _portable.explicit() else ""))
    return kernel == "causal_conv1d"


def conv_impl(kernel: str):
    """What ``conv_kernel`` resolves to, as the ``package=`` argument of the conv entry points: ``True`` (the
    ``causal_conv1d`` package, opt-in), ``"triton"`` (the plain-Triton conv, `kernels.conv_triton`: ``"auto"`` and
    ``"triton"``) or ``False`` (torch's ``F.conv1d``)."""
    if check_conv_kernel(kernel):
        return True
    return False if kernel == "torch" else "triton"


def conv_kernel_running(layer) -> str:
    """The conv that ACTUALLY runs in ``layer``: ``"triton"`` (the plain-Triton conv: ``conv_kernel="auto"`` or ``"triton"``, CUDA
    or interpreter tensors), ``"causal_conv1d"`` only when its ``conv_kernel`` resolves to the
    package AND its parameters are CUDA fp32/fp16/bf16 tensors AND its backend is ``"auto"`` / ``"cuda"``
    (``causal_depthwise_conv1d_cl`` never takes the package for a CPU or fp64 tensor, and the torch /
    sequential backends are the pure-torch routes); otherwise ``"torch"``."""
    impl = conv_impl(layer.conv_kernel)
    if not impl or layer.backend not in ("auto", "cuda"):
        return "torch"
    p = next(layer.parameters(), None)
    if impl == "triton":
        from ..kernels.conv_triton import supported
        return "triton" if (p is not None and supported(p)) else "torch"
    ok = p is not None and p.is_cuda and p.dtype in (torch.float32, torch.float16, torch.bfloat16)
    return "causal_conv1d" if ok else "torch"


def _buffer(x):
    # Output buffer in the layout the extension's kernel variant expects:
    # empty_like on a non-dense (narrowed) x would give channel-major storage
    # while the channel-last kernel is selected.
    if x.stride(1) == 1:
        b, c, length = x.shape
        return torch.empty((b, length, c), device=x.device, dtype=x.dtype).transpose(1, 2)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op("orbit_ssm::conv_fwd", mutates_args=())
def conv_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
             silu: bool) -> torch.Tensor:
    from causal_conv1d.cpp_functions import _causal_conv1d_fwd_cpp
    out = _buffer(x)
    _causal_conv1d_fwd_cpp(x, weight, bias, None, None, out, None, silu)
    return out


@conv_fwd.register_fake
def _conv_fake(x, weight, bias, silu):
    return _buffer(x)


@torch.library.custom_op(
    "orbit_ssm::conv_bwd", mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor? bias, Tensor dy, bool silu) -> (Tensor, Tensor, Tensor?)")
def conv_bwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
             dy: torch.Tensor, silu: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward of `conv_fwd`.

    ``x`` and ``dy`` must share one memory order: the extension picks its
    kernel variant from ``x`` and silently copies a mismatched ``dy``. The
    module passes both channel-major; a conv-layout test pins the
    copy count at zero.
    """
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
    if dy.stride(1) != 1 and dy.stride(2) != 1:
        dy = dy.contiguous()
    dx, dw, db, _ = causal_conv1d_bwd_function(
        x, weight, bias, dy, None, None, None, _buffer(x), False, silu)
    return dx, dw, db


@conv_bwd.register_fake
def _conv_bwd_fake(x, weight, bias, dy, silu):
    return _buffer(x), torch.empty_like(weight), None if bias is None else torch.empty_like(bias)


def _setup(ctx, inputs, output):
    x, weight, bias, ctx.silu = inputs
    ctx.save_for_backward(x, weight, bias)


def _backward(ctx, dy):
    return (*conv_bwd(*ctx.saved_tensors, dy, ctx.silu), None)


conv_fwd.register_autograd(_backward, setup_context=_setup)


@torch.library.custom_op("orbit_ssm::conv_tri_fwd", mutates_args=())
def conv_tri_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: bool) -> torch.Tensor:
    """The plain-Triton conv as an opaque op (compiled graphs); x [B, C, L] any strides, y in x's memory order."""
    from ..kernels.conv_triton import conv_fwd_triton
    return conv_fwd_triton(x, weight, bias, silu)


@conv_tri_fwd.register_fake
def _conv_tri_fake(x, weight, bias, silu):
    return _buffer(x)


@torch.library.custom_op(
    "orbit_ssm::conv_tri_bwd", mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor? bias, Tensor dy, bool silu) -> (Tensor, Tensor, Tensor?)")
def conv_tri_bwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                 dy: torch.Tensor, silu: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from ..kernels.conv_triton import conv_bwd_triton
    return conv_bwd_triton(x, weight, bias, dy, silu, (True, True, bias is not None))


@conv_tri_bwd.register_fake
def _conv_tri_bwd_fake(x, weight, bias, dy, silu):
    return _buffer(x), torch.empty_like(weight), None if bias is None else torch.empty_like(bias)


def _tri_setup(ctx, inputs, output):
    x, weight, bias, ctx.silu = inputs
    ctx.save_for_backward(x, weight, bias)


def _tri_backward(ctx, dy):
    return (*conv_tri_bwd(*ctx.saved_tensors, dy, ctx.silu), None)


conv_tri_fwd.register_autograd(_tri_backward, setup_context=_tri_setup)


def conv_op_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: bool, package=True) -> torch.Tensor:
    """The registered (compile-opaque, autograd-wired) conv op of the resolved ``package`` (`conv_impl`): the Triton op for
    ``"triton"``, the ``causal_conv1d`` op otherwise.  x [B, C, L] (any strides for Triton), weight [C, W]."""
    return (conv_tri_fwd if package == "triton" else conv_fwd)(x, weight, bias, silu)




def causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor | None, d_conv: int,
                            activation: str | None = None,
                            allow_channel_last: bool = False,
                            package: bool = False) -> torch.Tensor:
    """Depthwise causal conv1d with optional fused SiLU, channel-last.

    Parameters
    ----------
    x : Tensor (B, L, C)
    weight : Tensor (C, 1, d_conv)
        ``nn.Conv1d``'s depthwise layout.
    bias : Tensor (C,) or None
    d_conv : int
        Kernel width (used by the torch fallback's padding).
    activation : None or "silu"
    allow_channel_last : bool
        See `causal_depthwise_conv1d_cl`.
    package : bool
        ``True`` (the layer's ``conv_kernel="causal_conv1d"``) runs the ``causal_conv1d`` CUDA kernel when
        ``x`` is a CUDA fp32/fp16/bf16 tensor (the kernel rejects fp64); the default is the native
        ``F.conv1d`` plus a slice.  The two paths agree to fp tolerance (mixer test "G2").

    Returns
    -------
    Tensor (B, L, C)
    """
    return causal_depthwise_conv1d_cl(
        x.transpose(1, 2), weight, bias, d_conv, activation,
        allow_channel_last=allow_channel_last, package=package).transpose(1, 2)


def causal_conv1d_cl_fwd(x: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor | None,
                         activation: str | None = None,
                         out: torch.Tensor | None = None,
                         package=True) -> torch.Tensor:
    """The fused depthwise causal conv, forward only, with no autograd node.

    For callers that run under `no_grad` and own their backward (it calls the
    raw entry points of the resolved conv: ``package=True`` the ``causal_conv1d`` package's,
    ``package="triton"`` the plain-Triton conv's, which takes any strides and any ``out``).

    ``out`` must be passed when ``x`` is non-dense (a narrow of a wider slab,
    e.g. the B|C columns of the fused token projection): the package would
    allocate a channel-major buffer while the channel-last kernel it then
    selects assumes ``out.stride(1) == 1``, which gives silently wrong
    results. Pass an ``out`` whose layout matches the variant ``x`` selects.
    """
    if package == "triton":
        from ..kernels.conv_triton import conv_fwd_triton
        return conv_fwd_triton(x, weight, bias, activation in ("silu", "swish"), out=out)
    from causal_conv1d.cpp_functions import (causal_conv1d_fwd_function,
                                             _causal_conv1d_fwd_cpp)
    xc = x if (x.stride(2) == 1 or x.stride(1) == 1) else x.contiguous()
    silu = activation in ("silu", "swish")
    bias = None if bias is None else bias.contiguous()
    if out is None:
        return causal_conv1d_fwd_function(
            xc, weight.squeeze(1), bias, None, None, None, silu)
    _causal_conv1d_fwd_cpp(x=xc, weight=weight.squeeze(1), bias=bias,
                           seq_idx=None, initial_states=None, out=out,
                           final_states_out=None, silu_activation=silu)
    return out


def causal_conv1d_cl_bwd(x: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor | None, dout: torch.Tensor,
                         activation: str | None = None,
                         dx: torch.Tensor | None = None,
                         package=True):
    """Adjoint of `causal_conv1d_cl_fwd`: returns (dx, dweight, dbias), with
    `dweight` shaped like ``weight`` ([C, 1, d_conv] or [C, d_conv]).

    ``dx`` plays the role of ``out`` in the forward: a channel-last buffer
    must be supplied for a strided ``x``, which the C++ entry otherwise
    rejects.
    """
    if package == "triton":
        from ..kernels.conv_triton import conv_bwd_triton
        dx, dw, db = conv_bwd_triton(x, weight, bias, dout, activation in ("silu", "swish"),
                                     (True, True, bias is not None), dx=dx)
        return dx, dw, db
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()
    dx, dw, db, _ = causal_conv1d_bwd_function(
        x, weight.squeeze(1), None if bias is None else bias.contiguous(),
        dout, None, None, None, dx, False,
        activation in ("silu", "swish"))
    return dx, dw.view(weight.shape), db


def causal_depthwise_conv1d_cl(x: torch.Tensor, weight: torch.Tensor,
                               bias: torch.Tensor | None, d_conv: int,
                               activation: str | None = None,
                               allow_channel_last: bool = False,
                               package: bool = False) -> torch.Tensor:
    """The same depthwise causal conv, channel-major in and out.

    x : (B, C, L), the layout both routes want. Returns (B, C, L).
    ``causal_depthwise_conv1d`` is the channel-last wrapper; call this
    directly when the drive is already produced channel-major, to avoid the
    wrapper's transpose copy.

    ``allow_channel_last=True`` lets the CUDA kernel read a channel-last ``x``
    (``x.stride(1) == 1``, e.g. a column slice of a token-major slab) in
    place instead of copying it channel-major. It is a different kernel
    variant, so each call site opts in.  ``package=True`` takes the ``causal_conv1d`` kernel (CUDA
    fp32/fp16/bf16; the package must be importable, `check_conv_kernel`), else the native path.
    """
    L = x.shape[-1]
    _fast_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if isinstance(package, str):                       # "triton": the plain-Triton conv (any float dtype, any strides)
        from ..kernels.conv_triton import _FLOATS, causal_conv_triton, supported
        if activation not in (None, "silu", "swish"):
            raise ValueError(f"unsupported activation {activation!r}")
        compiling = torch.compiler.is_compiling()
        # `supported` asks the backend probe (a Dynamo-skipped function): under compile the CUDA test is the whole question
        if (x.is_cuda and x.dtype in _FLOATS) if compiling else supported(x):
            if compiling:
                # same layout contract as the package's registered op: channel-major unless the caller allows (and has) channel-last
                xk = x if (allow_channel_last and x.stride(1) == 1) else x.contiguous()
                return conv_tri_fwd(xk, weight.reshape(weight.shape[0], -1), bias, activation is not None)
            return causal_conv_triton(x, weight, bias, activation is not None)
        package = False
    if package and x.is_cuda and x.dtype in _fast_dtypes:
        fn = package_conv_fn()
        if fn is None:
            check_conv_kernel("causal_conv1d")
        xk = (x if (allow_channel_last and x.stride(1) == 1)
              else x.contiguous())
        if torch.compiler.is_compiling() and activation in (None, "silu", "swish"):
            return conv_fwd(xk, weight.squeeze(1), bias, activation is not None)
        return fn(xk, weight.squeeze(1), bias, activation=activation)
    y = F.conv1d(x, weight, bias, padding=d_conv - 1, groups=x.shape[1])
    y = y[..., :L]
    if activation == "silu":
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation!r}")
    return y


def channel_last_conv_view_ok(x: torch.Tensor, package=True) -> bool:
    """May the channel-last conv variant read this [B, L, C] view as it stands, without a copy?
    ``package="triton"``: any strided view with a contiguous channel run (the kernels take strides).  Else
    `causal_conv1d`'s variant:

    Restates the C++ entry's preconditions on the [B, L, C] view: contiguous
    channel run, channel count and batch/token strides multiples of 8, and an
    8-element-aligned base offset. Shared by the SSM families so they agree
    on what the kernel accepts.
    """
    if torch.compiler.is_compiling():
        # Dynamo cannot trace `storage_offset()` and the narrow route drives
        # raw launchers; compiled callers take the dense route, which uses the
        # registered `orbit_ssm::conv_fwd`/`conv_bwd` (`conv_tri_fwd`/`conv_tri_bwd`) ops.
        return False
    if package == "triton":
        from ..kernels.conv_triton import supported
        return x.dim() == 3 and x.stride(-1) == 1 and supported(x)
    return (x.is_cuda and x.dim() == 3 and x.stride(-1) == 1
            and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and x.shape[-1] % 8 == 0
            and x.stride(0) % 8 == 0 and x.stride(1) % 8 == 0
            and x.storage_offset() % 8 == 0)


class _NarrowChannelLastConvFn(torch.autograd.Function):
    """Autograd-tracked depthwise causal conv (+ optional SiLU), channel-last
    in and out, safe for a non-dense ``x`` (a column narrow of a wider slab).

    The package's own autograd Function allocates output and dx with
    ``empty_like(x)``, which for a non-dense ``x`` is channel-major while the
    channel-last kernel is still selected: silently wrong results. This
    Function hands the raw entry points correctly laid-out buffers instead.
    Matches the channel-major route bitwise in fp32 and to bf16 accumulation
    tolerance in bf16.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, activation, sink=None, package=True):
        # x: [B, L, C], possibly a narrow view (caller checked the kernel's
        # preconditions).  sink: the SlabGrad of the slab x spans, or None.
        ctx.sink = sink
        ctx.package = package
        xcl = x.transpose(1, 2)                                # [B, C, L] view
        out = torch.empty(x.shape, device=x.device,
                          dtype=x.dtype).transpose(1, 2)        # channel-last
        causal_conv1d_cl_fwd(xcl, weight, bias, activation, out=out, package=package)
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        return out.transpose(1, 2)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        if not dy.is_contiguous():
            dy = dy.contiguous()
        # dx goes straight into the slab's shared gradient buffer (no cat); it
        # has x's strides, so the channel-last preconditions still hold.
        dxo = None if ctx.sink is None or not ctx.needs_input_grad[0] else ctx.sink.region(x)
        dxcl = (torch.empty(x.shape, device=x.device, dtype=x.dtype) if dxo is None
                else dxo).transpose(1, 2)
        dx, dw, db = causal_conv1d_cl_bwd(
            x.transpose(1, 2), weight, bias, dy.transpose(1, 2),
            ctx.activation, dx=dxcl, package=ctx.package)
        dx = dx.transpose(1, 2)
        return (dx if ctx.needs_input_grad[0] else None,
                dw if ctx.needs_input_grad[1] else None,
                db if bias is not None and ctx.needs_input_grad[2] else None,
                None, None, None)


def causal_depthwise_conv1d_narrow_cl(x: torch.Tensor, weight: torch.Tensor,
                                      bias: torch.Tensor | None,
                                      activation: str | None = None,
                                      sink=None, package=True) -> torch.Tensor:
    """Autograd-tracked wrapper around `_NarrowChannelLastConvFn`.

    x : (B, L, C), possibly a narrow view of a wider slab; the caller must
    have checked `channel_last_conv_view_ok(x)`. Returns (B, L, C).
    """
    return _NarrowChannelLastConvFn.apply(x, weight, bias, activation, sink, package)


def causal_conv_step(x_t: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                     package=False, out_dtype=None, silu: bool = True):
    """One decode token of the depthwise causal conv: x_t [B, C], state [B, C, W] (the W newest raw inputs BEFORE this token, oldest first),
    weight [C, 1, W] | [C, W], bias [C] -> (y [B, C], new_state [B, C, W]).

    ``package="triton"`` (the default conv; `conv_impl`) runs `kernels.conv_triton.conv_step_triton`, one launch, when x_t is a CUDA float tensor
    and no autograd graph is being recorded and the graph is not being compiled; every other case runs the torch composite (cat, multiply, sum,
    SiLU) that `step()` always ran.  The ``causal_conv1d`` package's update kernel is not used (it needs its own state layout and carries no
    gradient); the weight is cast to the activation dtype first (as the layers do), the carry keeps its own dtype."""
    act = x_t.dtype
    w = weight.reshape(weight.shape[0], -1).to(act)
    b = None if bias is None else bias.to(act)
    if package == "triton" and not torch.compiler.is_compiling() and x_t.dim() == 2:
        from ..kernels.conv_triton import conv_step_triton, supported
        needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x_t, state, w, b))
        if supported(x_t) and state.is_cuda == x_t.is_cuda and not needs_grad:
            return conv_step_triton(x_t, state, w, b, silu, out_dtype)
    new = torch.cat([state[:, :, 1:], x_t.unsqueeze(-1)], dim=-1)
    y = (new * w).sum(-1)
    if b is not None:
        y = y + b
    y = F.silu(y) if silu else y
    return (y if out_dtype is None else y.to(out_dtype)), new


class CausalDepthwiseConv(nn.Module):
    """Depthwise causal conv1d + SiLU on channel-last tensors, with chunked and
    single-token stepping (the Mamba ``conv1d`` idiom).

    Parameters
    ----------
    channels : int
        Number of channels C (groups equal channels; one filter per channel).
    d_conv : int, default 4
        Kernel width. `forward_with_prefix` carries the last ``d_conv - 1``
        raw inputs; `step` keeps a ``d_conv``-wide window.  ``self.conv`` is an ``nn.Conv1d`` with bias and default init.
    """

    def __init__(self, channels: int, d_conv: int = 4):
        super().__init__()
        self.channels, self.d_conv = channels, d_conv
        self.package = False          # the owning layer sets it from its ``conv_kernel``
        self.conv = nn.Conv1d(channels, channels, kernel_size=d_conv,
                              groups=channels, padding=d_conv - 1, bias=True)

    def forward(self, x):                       # [B, L, C] -> [B, L, C]
        return causal_depthwise_conv1d(
            x, self.conv.weight, self.conv.bias, self.d_conv,
            activation="silu", package=self.package)

    def forward_with_prefix(self, x, prefix=None):
        """Chunked form: ``x`` [B, L, C] preceded by the carried ``prefix``
        [B, d_conv-1, C] of raw inputs (zeros = sequence start). Returns
        (y [B, L, C], new prefix) so multi-chunk stepping is exact."""
        Bsz, _, C = x.shape
        if prefix is None:
            prefix = x.new_zeros(Bsz, self.d_conv - 1, C)
        xc = torch.cat([prefix, x], dim=1)
        if isinstance(self.package, str):                # the plain-Triton conv over prefix + chunk, prefix outputs dropped
            y = causal_depthwise_conv1d(xc, self.conv.weight.to(x.dtype), self.conv.bias.to(x.dtype), self.d_conv,
                                        activation="silu", package=self.package)[:, self.d_conv - 1:]
            tail = xc[:, -(self.d_conv - 1):] if self.d_conv > 1 else xc[:, :0]
            return y, tail if torch.is_grad_enabled() else tail.detach().clone()
        y = torch.nn.functional.silu(torch.nn.functional.conv1d(
            xc.transpose(1, 2), self.conv.weight.to(x.dtype),
            self.conv.bias.to(x.dtype), groups=C).transpose(1, 2))
        tail = xc[:, -(self.d_conv - 1):] if self.d_conv > 1 else xc[:, :0]
        return y, tail if torch.is_grad_enabled() else tail.detach().clone()

    def init_state(self, batch_size, device=None, dtype=torch.float32):
        """Zero window for `step`, shape [B, channels, d_conv]."""
        return torch.zeros(batch_size, self.channels, self.d_conv,
                           device=device, dtype=dtype)

    def step(self, x_t, state):                 # [B, C], [B, C, d_conv]
        y, state = causal_conv_step(x_t, state, self.conv.weight, self.conv.bias, package=self.package)
        return y, state
