
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .glue import glue_on, hm_glue_on

def autocast_cast(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not torch.is_autocast_enabled('cuda'):
        return x
    dt = torch.get_autocast_dtype('cuda')
    if x.dtype == dt or x.dtype not in AUTOCAST_CASTABLE:
        return x
    return x.to(dt)
AUTOCAST_CASTABLE = (torch.float32, torch.float16, torch.bfloat16)

def autocast_io(t):
    if not isinstance(t, torch.Tensor):
        return t
    dev = t.device.type
    if dev not in ('cuda', 'cpu') or not torch.is_autocast_enabled(dev):
        return t
    dt = torch.get_autocast_dtype(dev)
    if t.dtype == dt or t.dtype not in AUTOCAST_CASTABLE:
        return t
    return t.to(dt)

def linear_keep_layout(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None=None) -> torch.Tensor:
    if x.dim() == 3 and x.stride(-1) != 1:
        wt = w.to(x.dtype).t().expand(x.shape[0], w.shape[1], w.shape[0])
        y = torch.bmm(x, wt)
        return y if b is None else y + b.to(y.dtype)
    return F.linear(x, w, b)

class _CachedCatFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cache, key, dim, *tensors):
        ctx.sizes = [t.shape[dim] for t in tensors]
        ctx.dim = dim
        ctx.needs = [bool(t.requires_grad) for t in tensors]
        if cache.get('key') == key:
            val = cache['value']
        else:
            val = torch.cat(tensors, dim=dim)
            cache['key'], cache['value'] = (key, val.detach())
        return val

    @staticmethod
    def backward(ctx, dval):
        parts = dval.split(ctx.sizes, dim=ctx.dim)
        grads = [p if need else None for p, need in zip(parts, ctx.needs)]
        return (None, None, None, *grads)

def cached_cat(cache: dict, key, tensors, dim: int=0) -> torch.Tensor:
    if not torch.is_grad_enabled() or not any((t.requires_grad for t in tensors)):
        if cache.get('key') == key:
            return cache['value']
        val = torch.cat(tensors, dim=dim)
        cache['key'], cache['value'] = (key, val.detach())
        return val
    return _CachedCatFn.apply(cache, key, dim, *tensors)

class _ChannelMajorMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, token_grad):
        dt = torch.get_autocast_dtype('cuda') if x.is_cuda and torch.is_autocast_enabled('cuda') else x.dtype
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
            if dx.dtype != ctx.x_dtype:
                dx = dx.to(ctx.x_dtype)
        dw = torch.bmm(dy, x).sum(0).to(w.dtype) if ctx.needs_input_grad[1] else None
        return (dx, dw, None)

class _HMSelTeeFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_cf, w_tok):
        with torch.amp.autocast('cuda', enabled=False):
            dt, B = (x_cf.dtype, x_cf.shape[0])
            wt = w_tok.to(dt)
            maps = torch.bmm(wt.expand(B, *wt.shape), x_cf)
        ctx.save_for_backward(x_cf, wt)
        ctx.w_dtype = w_tok.dtype
        return (x_cf.view_as(x_cf), maps)

    @staticmethod
    def backward(ctx, g_drive, g_maps):
        x_cf, wt = ctx.saved_tensors
        B, C, _L = x_cf.shape
        tok = wt.shape[0]
        g_maps = g_maps if g_maps.is_contiguous() else g_maps.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            wtt = wt.t().expand(B, C, tok)
            dx = torch.bmm(wtt, g_maps) if g_drive is None else torch.baddbmm(g_drive, wtt, g_maps)
        dw = None
        if ctx.needs_input_grad[1]:
            dw = torch.bmm(g_maps, x_cf.transpose(1, 2)).sum(0)
            dw = dw.to(ctx.w_dtype)
        return (dx, dw)

def matmul_channel_major(w: torch.Tensor, x: torch.Tensor, grad_layout: str='channel') -> torch.Tensor:
    if grad_layout not in ('channel', 'token'):
        raise ValueError(f"grad_layout must be 'channel'|'token', got {grad_layout!r}")
    if not (x.requires_grad or w.requires_grad) or not torch.is_grad_enabled():
        wb = w.to(x.dtype).expand(x.shape[0], w.shape[0], w.shape[1])
        return torch.bmm(wb, x.transpose(1, 2))
    return _ChannelMajorMM.apply(x, w, grad_layout == 'token')

class _SplitInProjFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xa, w_in, w_tok, b_tok, d_inner):
        dt = torch.get_autocast_dtype('cuda') if xa.is_cuda and torch.is_autocast_enabled('cuda') else xa.dtype
        xc = xa if xa.dtype == dt else xa.to(dt)
        B, L, K = xc.shape
        n_in = w_in.shape[0]
        d_gate = n_in - d_inner
        if w_tok is None:
            wc = w_in if w_in.dtype == dt else w_in.to(dt)
        elif w_in.dtype == dt and w_tok.dtype == dt:
            wc = torch.cat([w_in, w_tok], dim=0)
        else:
            wc = torch.empty(n_in + w_tok.shape[0], K, dtype=dt, device=w_in.device)
            wc[:n_in].copy_(w_in)
            wc[n_in:].copy_(w_tok)
        xcl = torch.bmm(wc[:d_inner].expand(B, d_inner, K), xc.transpose(1, 2))
        bpad = None if b_tok is None or w_tok is None else F.pad(b_tok.to(dt), (d_gate, 0))
        zt = F.linear(xc, wc[d_inner:], bpad)
        if w_tok is None:
            z, tok = (zt, None)
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
        return (xcl, z, tok)

    @staticmethod
    def backward(ctx, dxcl, dz, dtok):
        xc, wc = ctx.saved_tensors
        d_inner, tok = (ctx.d_inner, ctx.tok_width)
        B, L, K = xc.shape
        n_in = ctx.n_in
        wi, wt = (wc[:n_in], wc[n_in:] if tok else None)

        def _c(g):
            return g if g is None or g.dtype == xc.dtype else g.to(xc.dtype)
        dxcl, dz, dtok = (_c(dxcl), _c(dz), _c(dtok))
        x2 = xc.reshape(B * L, K)
        dz2 = None if dz is None else dz.reshape(B * L, n_in - d_inner)
        dt2 = None if dtok is None or not tok else dtok.reshape(B * L, tok)
        acc = not torch.is_grad_enabled()
        dxa = None
        if ctx.needs_input_grad[0]:
            if dz2 is not None:
                dxa = torch.mm(dz2, wi[d_inner:])
            if dt2 is not None:
                dxa = torch.mm(dt2, wt) if dxa is None else dxa.addmm_(dt2, wt) if acc else torch.addmm(dxa, dt2, wt)
            if dxcl is not None:
                b1 = dxcl.transpose(1, 2)
                b2 = wi[:d_inner].expand(B, d_inner, K)
                if dxa is None:
                    dxa = torch.bmm(b1, b2).reshape(B * L, K)
                elif acc:
                    dxa = dxa.view(B, L, K).baddbmm_(b1, b2).view(B * L, K)
                else:
                    dxa = torch.baddbmm(dxa.view(B, L, K), b1, b2).reshape(B * L, K)
            dxa = dxa.view(B, L, K)
            if dxa.dtype != ctx.x_dtype:
                dxa = dxa.to(ctx.x_dtype)
        dw_in = None
        if ctx.needs_input_grad[1]:
            top = torch.bmm(dxcl, xc).sum(0) if dxcl is not None else xc.new_zeros(d_inner, K)
            bot = torch.mm(dz2.t(), x2) if dz2 is not None else xc.new_zeros(n_in - d_inner, K)
            if acc:
                dw_in = torch.empty(n_in, K, dtype=ctx.w_dtype, device=xc.device)
                dw_in[:d_inner].copy_(top)
                dw_in[d_inner:].copy_(bot)
            else:
                dw_in = torch.cat([top, bot], 0).to(ctx.w_dtype)
        dw_tok = None
        if ctx.needs_input_grad[2] and dt2 is not None:
            dw_tok = torch.mm(dt2.t(), x2).to(ctx.wt_dtype)
        db_tok = None
        if ctx.needs_input_grad[3] and dt2 is not None:
            db_tok = dt2.sum(0).to(ctx.b_dtype)
        return (dxa, dw_in, dw_tok, db_tok, None)

def split_in_proj(xa: torch.Tensor, w_in: torch.Tensor, w_tok: torch.Tensor | None, b_tok: torch.Tensor | None, d_inner: int):
    if glue_on('inproj') and torch.is_grad_enabled() and (xa.requires_grad or w_in.requires_grad or (w_tok is not None and w_tok.requires_grad)):
        return _SplitInProjFn.apply(xa, w_in, w_tok, b_tok, d_inner)
    xcl = matmul_channel_major(w_in[:d_inner], xa, grad_layout='token')
    if w_tok is None:
        return (xcl, F.linear(xa, w_in[d_inner:]), None)
    wcat = torch.cat([w_in[d_inner:], w_tok], dim=0)
    d_gate = w_in.shape[0] - d_inner
    bpad = None if b_tok is None else F.pad(b_tok.to(xa.dtype), (d_gate, 0))
    zt = F.linear(xa, wcat, bpad)
    return (xcl, zt[..., :d_gate], zt[..., d_gate:])

def matmul_channel_major_tee(x_cf: torch.Tensor, w: torch.Tensor):
    if hm_glue_on('sel') and x_cf.is_cuda and torch.is_grad_enabled() and (x_cf.requires_grad or w.requires_grad):
        return _HMSelTeeFn.apply(x_cf, w)
    return (x_cf, matmul_channel_major(w, x_cf.transpose(1, 2)))
PROJ_SOURCES = ('main_branch', 'layer_input')
_PROJ_SOURCE_RETIRED = {'m3': 'layer_input', 'm1': 'main_branch', 'residual': 'layer_input', 'drive': 'main_branch'}

def resolve_proj_source(proj_source: str, d_inner: int, d_model, bc_conv, *, who: str, d_conv: int=4) -> tuple[str, int, bool]:
    if proj_source in _PROJ_SOURCE_RETIRED and proj_source not in PROJ_SOURCES:
        raise ValueError(f'{who}: proj_source={proj_source!r} was RETIRED by the 2026-09-20 naming ruling; use proj_source={_PROJ_SOURCE_RETIRED[proj_source]!r}.  The values are named after what they read, not after a Mamba version.')
    if proj_source not in PROJ_SOURCES:
        raise ValueError(f'{who}: proj_source must be one of {PROJ_SOURCES}, got {proj_source!r}')
    if proj_source == 'layer_input':
        if d_model is None:
            raise ValueError(f"{who}: proj_source='layer_input' reads the block input, so d_model (its width) is required; pass d_model=<block width> or choose proj_source='main_branch'")
        if d_conv <= 0:
            if bc_conv:
                raise ValueError(f'{who}: bc_conv=True needs d_conv > 0')
            return (proj_source, int(d_model), False)
        return (proj_source, int(d_model), False if bc_conv is None else bool(bc_conv))
    if bc_conv:
        raise ValueError(f"{who}: bc_conv is a layer_input-source (Mamba-2/3) feature; proj_source='main_branch' has none")
    return (proj_source, int(d_inner), False)
