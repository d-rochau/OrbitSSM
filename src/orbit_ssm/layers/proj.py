"""Projection helpers: autocast casts, layout-preserving GEMMs, fused in_proj nodes, proj_source resolution."""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .glue import glue_on, hm_glue_on


def autocast_cast(x: torch.Tensor) -> torch.Tensor:
    """Cast ``x`` to the active CUDA autocast dtype once, up front.

    A block input feeding two GEMMs (one on ``x.transpose(1, 2)``) would
    otherwise be cast twice, because autocast's cast cache is keyed on the
    tensor; casting once gives both GEMMs the same contiguous operand.
    Identity outside CUDA autocast. Only fp32/fp16/bf16 are cast; fp64 and
    non-float dtypes pass through, as through ``F.linear``.
    """
    if not x.is_cuda or not torch.is_autocast_enabled("cuda"):
        return x
    dt = torch.get_autocast_dtype("cuda")
    if x.dtype == dt or x.dtype not in AUTOCAST_CASTABLE:
        return x
    return x.to(dt)


#: the dtypes `autocast_cast` / `autocast_io` ever cast (never fp64).
AUTOCAST_CASTABLE = (torch.float32, torch.float16, torch.bfloat16)


def autocast_io(t):
    """`autocast_cast` for an SSM core's activation input (CUDA or CPU).

    Casts an fp32/fp16/bf16 tensor to the autocast dtype; ``None``, fp64,
    non-float tensors and calls outside autocast pass through. Cores call it
    on activation maps only (drive, B/C maps, output gate), so an upstream
    promotion (16-bit activation times fp32 parameter) cannot put a scan on
    its fp32-IO path. Inputs that are fp32 by design (decay steps, rates,
    phases, parameters, state carries) must not be passed here.
    """
    if not isinstance(t, torch.Tensor):
        return t
    dev = t.device.type
    if dev not in ("cuda", "cpu") or not torch.is_autocast_enabled(dev):
        return t
    dt = torch.get_autocast_dtype(dev)
    if t.dtype == dt or t.dtype not in AUTOCAST_CASTABLE:
        return t
    return t.to(dt)


def linear_keep_layout(x: torch.Tensor, w: torch.Tensor,
                       b: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear`` without ATen's fold-copy for channel-major inputs.

    Under autograd ATen folds (B, L) into one row axis, which copies the whole
    activation when ``x`` is a [B, L, C] view of [B, C, L] memory (what the
    depthwise conv returns). A batched ``bmm`` reads the operand's own
    strides instead. Contiguous inputs keep the plain ``F.linear`` route.
    """
    if x.dim() == 3 and x.stride(-1) != 1:
        wt = w.to(x.dtype).t().expand(x.shape[0], w.shape[1], w.shape[0])
        y = torch.bmm(x, wt)
        return y if b is None else y + b.to(y.dtype)
    return F.linear(x, w, b)


class _CachedCatFn(torch.autograd.Function):
    """``torch.cat(tensors, dim)`` memoised by value across calls with the same
    cache key, staying differentiable inside the autograd graph.

    A hit returns the value an earlier call computed; backward is cat's own
    adjoint (a split), so correctness does not depend on the cache.
    ``cache`` is a dict owned by the caller (one per fused slab); ``key``
    says whether the cached value is still valid (see `param_cache_key`).
    """

    @staticmethod
    def forward(ctx, cache, key, dim, *tensors):
        ctx.sizes = [t.shape[dim] for t in tensors]
        ctx.dim = dim
        ctx.needs = [bool(t.requires_grad) for t in tensors]
        if cache.get("key") == key:
            val = cache["value"]
        else:
            val = torch.cat(tensors, dim=dim)
            cache["key"], cache["value"] = key, val.detach()
        return val

    @staticmethod
    def backward(ctx, dval):
        parts = dval.split(ctx.sizes, dim=ctx.dim)
        grads = [p if need else None for p, need in zip(parts, ctx.needs)]
        return (None, None, None, *grads)


def cached_cat(cache: dict, key, tensors, dim: int = 0) -> torch.Tensor:
    """``torch.cat(tensors, dim)`` through `_CachedCatFn`; a plain memoised
    cat outside grad mode or when no tensor needs a gradient."""
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in tensors):
        if cache.get("key") == key:
            return cache["value"]
        val = torch.cat(tensors, dim=dim)
        cache["key"], cache["value"] = key, val.detach()
        return val
    return _CachedCatFn.apply(cache, key, dim, *tensors)


class _ChannelMajorMM(torch.autograd.Function):
    """``W @ x^T`` -> [B, F, L], with the input gradient in a chosen layout.

    The forward is `matmul_channel_major`'s ``bmm``. The layout of ``dx``
    matters because the block input feeds two GEMMs and autograd sums their
    gradients; opposite layouts make that sum uncoalesced.

    ``token_grad=False`` ("channel"): ``dx`` is a [B, L, C] view of [B, C, L]
    memory, right when the other consumer is the channel-major scan.
    ``token_grad=True`` ("token"): ``dx`` is contiguous [B, L, C], right when
    the other consumer is a token-major ``F.linear``.
    """

    @staticmethod
    def forward(ctx, x, w, token_grad):
        # Cast here rather than in `torch.bmm`: backward runs outside the
        # autocast region and would otherwise meet a bf16 `dy` against a
        # saved fp32 `x`.
        dt = (torch.get_autocast_dtype("cuda")
              if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype)
        xc = x if x.dtype == dt else x.to(dt)
        ctx.save_for_backward(xc, w)
        ctx.token_grad = token_grad
        ctx.x_dtype = x.dtype
        wb = w.to(dt).expand(x.shape[0], w.shape[0], w.shape[1])
        return torch.bmm(wb, xc.transpose(1, 2))

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        B, F_, _ = dy.shape
        C = w.shape[1]
        dy = dy if dy.dtype == x.dtype else dy.to(x.dtype)
        wc = w.to(dy.dtype)
        dx = None
        if ctx.needs_input_grad[0]:
            if ctx.token_grad:
                dx = torch.bmm(dy.transpose(1, 2), wc.expand(B, F_, C))
            else:
                dx = torch.bmm(wc.t().expand(B, C, F_), dy).transpose(1, 2)
            # return the gradient in the caller's dtype (the forward cast is an edge)
            if dx.dtype != ctx.x_dtype:
                dx = dx.to(ctx.x_dtype)
        dw = (torch.bmm(dy, x).sum(0).to(w.dtype)
              if ctx.needs_input_grad[1] else None)
        return dx, dw, None


class _HMSelTeeFn(torch.autograd.Function):
    """The post-conv drive and its token-map GEMM as one node.

    Under ``proj_source="main_branch"`` the token maps read the same drive
    the scan does, so the drive has two consumers. This node has one input
    and two outputs (the drive again, and the token slab), so backward sees
    both gradients and joins them inside cuBLAS (``baddbmm(g_drive, W^T,
    g_maps)``) instead of an engine-side add.

    Forward equals `matmul_channel_major`. The backward join accumulates in
    cuBLAS's fp32 accumulator, one rounding step fewer than a bf16 add.
    """

    @staticmethod
    def forward(ctx, x_cf, w_tok):
        with torch.amp.autocast("cuda", enabled=False):
            dt, B = x_cf.dtype, x_cf.shape[0]
            wt = w_tok.to(dt)
            maps = torch.bmm(wt.expand(B, *wt.shape), x_cf)
        ctx.save_for_backward(x_cf, wt)
        ctx.w_dtype = w_tok.dtype
        return x_cf.view_as(x_cf), maps

    @staticmethod
    def backward(ctx, g_drive, g_maps):
        x_cf, wt = ctx.saved_tensors
        B, C, _L = x_cf.shape
        tok = wt.shape[0]
        g_maps = g_maps if g_maps.is_contiguous() else g_maps.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            wtt = wt.t().expand(B, C, tok)
            # Out of place: `g_drive` is the engine's buffer and may not be
            # written into.
            dx = (torch.bmm(wtt, g_maps) if g_drive is None
                  else torch.baddbmm(g_drive, wtt, g_maps))
        dw = None
        if ctx.needs_input_grad[1]:
            dw = torch.bmm(g_maps, x_cf.transpose(1, 2)).sum(0)
            dw = dw.to(ctx.w_dtype)
        return dx, dw


def matmul_channel_major(w: torch.Tensor, x: torch.Tensor,
                         grad_layout: str = "channel") -> torch.Tensor:
    """``W @ x^T`` for x = [B, L, d_in], returned channel-major and contiguous
    as [B, d_out, L], without ATen's fold-copy (see `linear_keep_layout`).

    Uses ``bmm`` on a batch-expanded (stride-0) weight; the weight is cast
    before the expand so B copies are never materialised. ``grad_layout``
    ("channel" or "token") picks the memory order of ``dx``, see
    `_ChannelMajorMM`; "channel" matches plain autograd.
    """
    if grad_layout not in ("channel", "token"):
        raise ValueError(
            f"grad_layout must be 'channel'|'token', got {grad_layout!r}")
    if not (x.requires_grad or w.requires_grad) or not torch.is_grad_enabled():
        wb = w.to(x.dtype).expand(x.shape[0], w.shape[0], w.shape[1])
        return torch.bmm(wb, x.transpose(1, 2))
    return _ChannelMajorMM.apply(x, w, grad_layout == "token")


class _SplitInProjFn(torch.autograd.Function):
    """``in_proj`` (x branch and z/token slab) as one node.

    The block input feeds two GEMMs through two row-slices of
    ``in_proj.weight``. With one activation input and three outputs
    (x branch [B, d_inner, L], z, token slab), backward sees every incoming
    gradient at once: the input gradient accumulates through ``mm`` ->
    ``addmm_`` -> ``baddbmm_`` (``beta = 1``, no separate adds or split
    adjoint cat), both halves of ``d(in_proj.weight)`` land in one buffer,
    and the token bias gradient reduces over its ``tok_width`` columns only.

    Forward numerics equal the unfused ops (the weight is cast to the
    autocast dtype before the concatenation, which is bitwise identical).
    Backward changes accumulation order, so it matches only to GEMM tolerance.
    """

    @staticmethod
    def forward(ctx, xa, w_in, w_tok, b_tok, d_inner):
        dt = (torch.get_autocast_dtype("cuda")
              if xa.is_cuda and torch.is_autocast_enabled("cuda") else xa.dtype)
        xc = xa if xa.dtype == dt else xa.to(dt)
        B, L, K = xc.shape
        n_in = w_in.shape[0]
        d_gate = n_in - d_inner
        # One weight slab in the autocast dtype: in_proj rows and token-map
        # rows are cast straight into one buffer, and all three GEMMs read
        # narrows of it.
        if w_tok is None:
            wc = w_in if w_in.dtype == dt else w_in.to(dt)
        elif w_in.dtype == dt and w_tok.dtype == dt:
            wc = torch.cat([w_in, w_tok], dim=0)
        else:
            wc = torch.empty(n_in + w_tok.shape[0], K, dtype=dt,
                             device=w_in.device)
            wc[:n_in].copy_(w_in)
            wc[n_in:].copy_(w_tok)
        xcl = torch.bmm(wc[:d_inner].expand(B, d_inner, K),
                        xc.transpose(1, 2))                  # [B, d_inner, L]
        bpad = (None if b_tok is None or w_tok is None
                else F.pad(b_tok.to(dt), (d_gate, 0)))
        zt = F.linear(xc, wc[d_inner:], bpad)          # [B, L, d_inner(+tok)]
        if w_tok is None:
            z, tok = zt, None
        else:
            z = zt[..., :d_gate]
            tok = zt[..., d_gate:]
        ctx.save_for_backward(xc, wc)
        ctx.d_inner = d_inner
        ctx.x_dtype = xa.dtype
        ctx.w_dtype = w_in.dtype
        ctx.wt_dtype = None if w_tok is None else w_tok.dtype
        ctx.b_dtype = None if b_tok is None else b_tok.dtype
        ctx.tok_width = 0 if w_tok is None else w_tok.shape[0]
        ctx.n_in = n_in
        return xcl, z, tok

    @staticmethod
    def backward(ctx, dxcl, dz, dtok):
        xc, wc = ctx.saved_tensors
        d_inner, tok = ctx.d_inner, ctx.tok_width
        B, L, K = xc.shape
        n_in = ctx.n_in
        wi, wt = wc[:n_in], (wc[n_in:] if tok else None)
        def _c(g):
            return g if (g is None or g.dtype == xc.dtype) else g.to(xc.dtype)
        dxcl, dz, dtok = _c(dxcl), _c(dz), _c(dtok)
        x2 = xc.reshape(B * L, K)
        dz2 = None if dz is None else dz.reshape(B * L, n_in - d_inner)
        dt2 = None if (dtok is None or not tok) else dtok.reshape(B * L, tok)
        # Under `create_graph=True` grad mode is on; use the out-of-place
        # forms there so the engine never differentiates through an in-place write.
        acc = not torch.is_grad_enabled()

        dxa = None
        if ctx.needs_input_grad[0]:
            # one buffer, three cuBLAS calls, every join a beta = 1 epilogue
            if dz2 is not None:
                dxa = torch.mm(dz2, wi[d_inner:])             # [B*L, K]
            if dt2 is not None:
                dxa = (torch.mm(dt2, wt) if dxa is None else
                       dxa.addmm_(dt2, wt) if acc else
                       torch.addmm(dxa, dt2, wt))
            if dxcl is not None:
                b1 = dxcl.transpose(1, 2)
                b2 = wi[:d_inner].expand(B, d_inner, K)
                if dxa is None:
                    dxa = torch.bmm(b1, b2).reshape(B * L, K)
                elif acc:
                    dxa = dxa.view(B, L, K).baddbmm_(b1, b2).view(B * L, K)
                else:
                    dxa = torch.baddbmm(dxa.view(B, L, K), b1,
                                        b2).reshape(B * L, K)
            dxa = dxa.view(B, L, K)
            if dxa.dtype != ctx.x_dtype:
                dxa = dxa.to(ctx.x_dtype)

        dw_in = None
        if ctx.needs_input_grad[1]:
            # both halves land in ONE buffer: no zero-padding, no add
            top = (torch.bmm(dxcl, xc).sum(0) if dxcl is not None
                   else xc.new_zeros(d_inner, K))
            bot = (torch.mm(dz2.t(), x2) if dz2 is not None
                   else xc.new_zeros(n_in - d_inner, K))
            if acc:
                dw_in = torch.empty(n_in, K, dtype=ctx.w_dtype,
                                    device=xc.device)
                dw_in[:d_inner].copy_(top)
                dw_in[d_inner:].copy_(bot)
            else:
                dw_in = torch.cat([top, bot], 0).to(ctx.w_dtype)

        dw_tok = None
        if ctx.needs_input_grad[2] and dt2 is not None:
            dw_tok = torch.mm(dt2.t(), x2).to(ctx.wt_dtype)

        db_tok = None
        if ctx.needs_input_grad[3] and dt2 is not None:
            # over tok_width columns, not over the whole slab
            db_tok = dt2.sum(0).to(ctx.b_dtype)

        return dxa, dw_in, dw_tok, db_tok, None


def split_in_proj(xa: torch.Tensor, w_in: torch.Tensor,
                  w_tok: torch.Tensor | None, b_tok: torch.Tensor | None,
                  d_inner: int):
    """``(x branch [B, d_inner, L], z [B, L, w_in.shape[0]-d_inner], token slab)``
    from one node, see `_SplitInProjFn`. Falls back to the plain ops outside
    grad mode or with the ``inproj`` glue item off, with the same values."""
    if (glue_on("inproj") and torch.is_grad_enabled()
            and (xa.requires_grad or w_in.requires_grad
                 or (w_tok is not None and w_tok.requires_grad))):
        return _SplitInProjFn.apply(xa, w_in, w_tok, b_tok, d_inner)
    xcl = matmul_channel_major(w_in[:d_inner], xa, grad_layout="token")
    if w_tok is None:
        return xcl, F.linear(xa, w_in[d_inner:]), None
    wcat = torch.cat([w_in[d_inner:], w_tok], dim=0)
    d_gate = w_in.shape[0] - d_inner
    bpad = None if b_tok is None else F.pad(b_tok.to(xa.dtype), (d_gate, 0))
    zt = F.linear(xa, wcat, bpad)
    return xcl, zt[..., :d_gate], zt[..., d_gate:]


def matmul_channel_major_tee(x_cf: torch.Tensor, w: torch.Tensor):
    """``(x_cf, W @ x_cf)`` through one node (see `_HMSelTeeFn`).

    Falls back to `matmul_channel_major` (and the caller's own tensor)
    where the node cannot help: on CPU, outside grad mode, or with the
    `sel` glue item off.
    """
    if (hm_glue_on("sel") and x_cf.is_cuda and torch.is_grad_enabled()
            and (x_cf.requires_grad or w.requires_grad)):
        return _HMSelTeeFn.apply(x_cf, w)
    return x_cf, matmul_channel_major(w, x_cf.transpose(1, 2))


PROJ_SOURCES = ("main_branch", "layer_input")


# proj_source values are named after what they read, not after a Mamba
# version.


def resolve_proj_source(proj_source: str, d_inner: int, d_model, bc_conv,
                        *, who: str, d_conv: int = 4) -> tuple[str, int, bool]:
    """Resolve token-map source, selection width and B/C convolution once.

    Parameters
    ----------
    proj_source : {"layer_input", "main_branch"}
        What the B/C/dt token maps read.
        "layer_input" (Mamba-2/3 style): the residual-stream block input x,
        width ``d_model``; with ``bc_conv`` the raw B/C projections pass a
        depthwise causal conv + SiLU (equivalent to Mamba-2/3's joint
        ``xBC`` conv, since the conv is per-channel).
        "main_branch" (Mamba-1 style): the post-conv drive u, width
        ``d_inner``; no conv on B/C.
    d_inner : int
        Selection width for "main_branch".
    d_model : int or None
        Block width; required for "layer_input".
    bc_conv : bool or None
        Conv on B/C; None means False (no B/C conv by default, Mamba-3's
        ``d_conv=0`` recipe). True is only valid with
        "layer_input" and ``d_conv > 0``.
    who : str
        Name used in error messages.
    d_conv : int, default 4
        Conv width; ``<= 0`` means a conv-free block.

    Returns
    -------
    (proj_source, d_sel, bc_conv)
    """
    if proj_source not in PROJ_SOURCES:
        raise ValueError(f"{who}: proj_source must be one of {PROJ_SOURCES}, "
                         f"got {proj_source!r}")
    if proj_source == "layer_input":
        if d_model is None:
            raise ValueError(
                f"{who}: proj_source='layer_input' reads the block input, so "
                f"d_model (its width) is required; pass d_model=<block "
                f"width> or choose proj_source='main_branch'")
        if d_conv <= 0:                     # conv-free block: no B/C conv either
            if bc_conv:
                raise ValueError(f"{who}: bc_conv=True needs d_conv > 0")
            return proj_source, int(d_model), False
        return proj_source, int(d_model), (False if bc_conv is None else bool(bc_conv))
    if bc_conv:
        raise ValueError(f"{who}: bc_conv is a layer_input-source "
                         f"(Mamba-2/3) feature; proj_source='main_branch' has none")
    return proj_source, int(d_inner), False
