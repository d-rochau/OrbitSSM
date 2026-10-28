
import os
import torch
import triton
import triton.language as tl
from .common import ACC_SEM

@triton.jit
def _silu_gate_fwd(Y, Z, O, n, BS: tl.constexpr):
    offs = tl.program_id(0) * BS + tl.arange(0, BS)
    m = offs < n
    y = tl.load(Y + offs, mask=m, other=0.0).to(tl.float32)
    z = tl.load(Z + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(z)
    tl.store(O + offs, y * z * s, mask=m)

@triton.jit
def _silu_gate_bwd(Y, Z, dO, dY, dZ, n, BS: tl.constexpr):
    offs = tl.program_id(0) * BS + tl.arange(0, BS)
    m = offs < n
    y = tl.load(Y + offs, mask=m, other=0.0).to(tl.float32)
    z = tl.load(Z + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(dO + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(z)
    silu = z * s
    tl.store(dY + offs, g * silu, mask=m)
    tl.store(dZ + offs, g * y * s * (1.0 + z * (1.0 - s)), mask=m)

class _SiluGateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, z):
        y = y.contiguous()
        z = z.contiguous()
        out = torch.empty_like(y)
        n = y.numel()
        _silu_gate_fwd[triton.cdiv(n, 1024),](y, z, out, n, BS=1024)
        ctx.save_for_backward(y, z)
        return out

    @staticmethod
    def backward(ctx, dout):
        y, z = ctx.saved_tensors
        dout = dout.contiguous()
        dy = torch.empty_like(y)
        dz = torch.empty_like(z)
        n = y.numel()
        _silu_gate_bwd[triton.cdiv(n, 1024),](y, z, dout, dy, dz, n, BS=1024)
        return (dy, dz)

def silu_gate(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    assert y.shape == z.shape
    return _SiluGateFn.apply(y, z)

def _dypre_buffer(Ypre, dy_dtype=torch.float32, real=False):
    from . import levers
    if Ypre.dtype in (torch.bfloat16, torch.float16) and levers.on('real_gate_dy_io', levers.real_group(Ypre.shape[3]), guard=real):
        Bsz, H, nc, T, O, G = Ypre.shape
        buf = torch.empty(Bsz, nc * T, H, O, G, device=Ypre.device, dtype=Ypre.dtype)
        return buf.view(Bsz, nc, T, H, O, G).permute(0, 3, 1, 2, 4, 5)
    return torch.empty(Ypre.shape, device=Ypre.device, dtype=dy_dtype)

@triton.jit
def _ascalar_epilogue_bwd_kernel(Ypre_ptr, Z_ptr, XR_ptr, Dsk_ptr, dOut_ptr, dYpre_ptr, dZ_ptr, dXR_ptr, dDsk_ptr, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_zb, s_zh, s_zt, s_zg, s_xb, s_xh, s_xg, s_xr, s_xl, T: tl.constexpr, G_SIZE: tl.constexpr, R: tl.constexpr, O: tl.constexpr, HAS_DSKIP: tl.constexpr, G_BLK: tl.constexpr=0, STORE_DXR: tl.constexpr=True, s_ob=0, s_oh=0, s_oc=0, s_ot=0, s_og=0, s_oo=0, s_db=0, s_dh=0, s_dc=0, s_dt=0, s_dg=0, s_do=0, s_pb=0, s_ph=0, s_pt=0, s_pg=0, HEAD_FIRST: tl.constexpr=0):
    if HEAD_FIRST:
        h = tl.program_id(0)
        c = tl.program_id(1)
        b = tl.program_id(2)
    else:
        c = tl.program_id(0)
        b = tl.program_id(1)
        h = tl.program_id(2)
    offs_t = tl.arange(0, T)
    GB: tl.constexpr = G_BLK if G_BLK != 0 else G_SIZE
    offs_g = tl.arange(0, GB)
    gm = offs_g < G_SIZE
    gmask = gm[None, :] & (offs_t[:, None] >= 0)
    y_off = b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_g[None, :] * s_yg
    o_off = b * s_ob + h * s_oh + c * s_oc + offs_t[:, None] * s_ot + offs_g[None, :] * s_og
    d_off = b * s_db + h * s_dh + c * s_dc + offs_t[:, None] * s_dt + offs_g[None, :] * s_dg
    z_off = b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_g[None, :] * s_zg
    dz_off = b * s_pb + h * s_ph + (c * T + offs_t)[:, None] * s_pt + offs_g[None, :] * s_pg
    z = tl.load(Z_ptr + z_off, mask=gmask, other=0.0).to(tl.float32)
    sg = tl.sigmoid(z)
    dz_acc = tl.zeros([T, GB], dtype=tl.float32)
    dypre_last = tl.zeros([T, GB], dtype=tl.float32)
    for o in tl.static_range(O):
        yo = y_off + o * s_yo
        ypre = tl.load(Ypre_ptr + yo, mask=gmask, other=0.0).to(tl.float32)
        dout = tl.load(dOut_ptr + o_off + o * s_oo, mask=gmask, other=0.0).to(tl.float32)
        dypre = dout * (z * sg)
        tl.store(dYpre_ptr + d_off + o * s_do, dypre.to(dYpre_ptr.dtype.element_ty), mask=gmask)
        dz_acc += dout * ypre * sg * (1.0 + z * (1.0 - sg))
        dypre_last = dypre
    tl.store(dZ_ptr + dz_off, dz_acc, mask=gmask)
    dypre = dypre_last
    if HAS_DSKIP:
        for r in tl.static_range(R):
            x_off = b * s_xb + h * s_xh + r * s_xr + c * T * s_xl + offs_g[None, :] * s_xg + offs_t[:, None] * s_xl
            xr = tl.load(XR_ptr + x_off, mask=gmask, other=0.0).to(tl.float32)
            if STORE_DXR:
                dv = tl.load(Dsk_ptr + h * G_SIZE * R + offs_g * R + r, mask=gm, other=0.0)
                tl.store(dXR_ptr + x_off, dypre * dv[None, :], mask=gmask)
            tl.atomic_add(dDsk_ptr + h * G_SIZE * R + offs_g * R + r, tl.sum(dypre * xr, axis=0), mask=gm, sem=ACC_SEM)

def ascalar_epilogue_bwd(Ypre, Z, XR, Dsk, dOut, T, dZ_out=None, store_dxr=True, real=False):
    Bsz, H, nc, _T, O, G = Ypre.shape
    assert _T == T
    has_dskip = XR is not None
    dYpre = _dypre_buffer(Ypre, real=real)
    dZ = torch.empty_like(Z) if dZ_out is None else dZ_out
    if tuple(dZ.shape) != tuple(Z.shape):
        raise ValueError(f'dZ_out shape {tuple(dZ.shape)} != z shape {tuple(Z.shape)}')
    R = XR.shape[3] if has_dskip else 1
    store_dxr = bool(store_dxr and has_dskip)
    dXR = torch.empty_like(XR) if store_dxr else Ypre.new_empty(1)
    dDsk = torch.zeros(H * G * R, device=Ypre.device, dtype=torch.float32) if has_dskip else Ypre.new_empty(1)
    xs = XR.stride() if has_dskip else (0, 0, 0, 0, 0)
    head_first = os.environ.get('SSD_MIMO_ASCALAR_BWD_HEAD_FIRST', '1') == '1'
    _ascalar_epilogue_bwd_kernel[(H, nc, Bsz) if head_first else (nc, Bsz, H)](Ypre, Z, XR if has_dskip else Ypre, Dsk if store_dxr else Ypre, dOut, dYpre, dZ, dXR, dDsk, Ypre.stride(0), Ypre.stride(1), Ypre.stride(2), Ypre.stride(3), Ypre.stride(5), Ypre.stride(4), Z.stride(0), G * Z.stride(2), Z.stride(1), Z.stride(2), *xs, T=T, G_SIZE=G, R=R, O=O, HAS_DSKIP=has_dskip, G_BLK=0 if G & G - 1 == 0 else triton.next_power_of_2(G), STORE_DXR=store_dxr, s_pb=dZ.stride(0), s_ph=G * dZ.stride(2), s_pt=dZ.stride(1), s_pg=dZ.stride(2), s_ob=dOut.stride(0), s_oh=dOut.stride(1), s_oc=dOut.stride(2), s_ot=dOut.stride(3), s_og=dOut.stride(5), s_oo=dOut.stride(4), s_db=dYpre.stride(0), s_dh=dYpre.stride(1), s_dc=dYpre.stride(2), s_dt=dYpre.stride(3), s_dg=dYpre.stride(5), s_do=dYpre.stride(4), num_warps=4, HEAD_FIRST=int(head_first))
    return (dYpre, dZ, dXR if store_dxr else None, dDsk if has_dskip else None)

@triton.jit
def _ascalar_epilogue_fwd_kernel(Ypre_ptr, Z_ptr, Out_ptr, s_yb, s_yh, s_yc, s_yt, s_yg, s_yo, s_zb, s_zh, s_zt, s_zg, s_ob, s_oh, s_oc, s_ot, s_og, s_oo, T: tl.constexpr, G_SIZE: tl.constexpr, O: tl.constexpr, G_BLK: tl.constexpr=0):
    c = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tl.arange(0, T)
    GB: tl.constexpr = G_BLK if G_BLK != 0 else G_SIZE
    offs_g = tl.arange(0, GB)
    gmask = (offs_g < G_SIZE)[None, :] & (offs_t[:, None] >= 0)
    y_off = b * s_yb + h * s_yh + c * s_yc + offs_t[:, None] * s_yt + offs_g[None, :] * s_yg
    o_off = b * s_ob + h * s_oh + c * s_oc + offs_t[:, None] * s_ot + offs_g[None, :] * s_og
    z_off = b * s_zb + h * s_zh + (c * T + offs_t)[:, None] * s_zt + offs_g[None, :] * s_zg
    z = tl.load(Z_ptr + z_off, mask=gmask, other=0.0).to(tl.float32)
    silu = z * tl.sigmoid(z)
    for o in tl.static_range(O):
        ypre = tl.load(Ypre_ptr + y_off + o * s_yo, mask=gmask, other=0.0).to(tl.float32)
        tl.store(Out_ptr + o_off + o * s_oo, (ypre * silu).to(Out_ptr.dtype.element_ty), mask=gmask)

def ascalar_epilogue_fwd(Ypre, Z, T, out_dtype=None, token_major=False):
    Bsz, H, nc, _T, O, G = Ypre.shape
    assert _T == T
    assert Z.dim() == 3, f'gate z must be [B, L, H*G], got {tuple(Z.shape)}'
    dt = out_dtype or torch.float32
    if token_major:
        buf = torch.empty(Bsz, nc * T, H, O, G, device=Ypre.device, dtype=dt)
        out = buf.view(Bsz, nc, T, H, O, G).permute(0, 3, 1, 2, 4, 5)
    else:
        buf = torch.empty(Bsz, H, O, G, nc * T, device=Ypre.device, dtype=dt)
        out = buf.view(Bsz, H, O, G, nc, T).permute(0, 1, 4, 5, 2, 3)
    ys, zs, os_ = (Ypre.stride(), Z.stride(), out.stride())
    _ascalar_epilogue_fwd_kernel[nc, Bsz, H](Ypre, Z, out, ys[0], ys[1], ys[2], ys[3], ys[5], ys[4], zs[0], G * zs[2], zs[1], zs[2], os_[0], os_[1], os_[2], os_[3], os_[5], os_[4], T=T, G_SIZE=G, O=O, G_BLK=0 if G & G - 1 == 0 else triton.next_power_of_2(G))
    return out

@triton.jit
def _gate_norm_fwd_kernel(Ypre_ptr, Z_ptr, W_ptr, Out_ptr, Rstd_ptr, U_ptr, D_ptr, s_ub, s_ut, s_uc, s_yb, s_yt, s_yh, s_yg, s_zb, s_zt, s_zh, s_zg, s_ob, s_ot, s_oh, s_og, L, eps, inv_c, G: tl.constexpr, C: tl.constexpr, C_BLK: tl.constexpr, TB: tl.constexpr, HAS_W: tl.constexpr, HAS_D: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = tl.arange(0, C_BLK)
    m = offs < C
    hh = offs // G
    gg = offs - hh * G
    y_c = hh * s_yh + gg * s_yg
    z_c = hh * s_zh + gg * s_zg
    o_c = hh * s_oh + gg * s_og
    if HAS_W:
        w = tl.load(W_ptr + offs, mask=m, other=0.0).to(tl.float32)
    else:
        w = tl.zeros([C_BLK], dtype=tl.float32) + 1.0
    for i in tl.static_range(TB):
        t = pid * TB + i
        y = tl.load(Ypre_ptr + b * s_yb + t * s_yt + y_c, mask=m, other=0.0).to(tl.float32)
        if HAS_D:
            u = tl.load(U_ptr + b * s_ub + t * s_ut + offs * s_uc, mask=m, other=0.0).to(tl.float32)
            ds = tl.load(D_ptr + hh, mask=m, other=0.0).to(tl.float32)
            y = y + ds * u
        z = tl.load(Z_ptr + b * s_zb + t * s_zt + z_c, mask=m, other=0.0).to(tl.float32)
        v = y * (z * tl.sigmoid(z))
        r = 1.0 / tl.sqrt(tl.sum(v * v, axis=0) * inv_c + eps)
        tl.store(Rstd_ptr + b * L + t, r)
        tl.store(Out_ptr + b * s_ob + t * s_ot + o_c, (v * r * w).to(Out_ptr.dtype.element_ty), mask=m)

@triton.jit
def _gate_norm_bwd_kernel(Ypre_ptr, Z_ptr, W_ptr, Rstd_ptr, dOut_ptr, dYpre_ptr, dZ_ptr, dWpart_ptr, U_ptr, D_ptr, dU_ptr, dDpart_ptr, s_ub, s_ut, s_uc, s_yb, s_yt, s_yh, s_yg, s_zb, s_zt, s_zh, s_zg, s_ob, s_ot, s_oh, s_og, s_db, s_dt, s_dh, s_dg, s_pb, s_pt, s_ph, s_pg, L, inv_c, NBLK, G: tl.constexpr, C: tl.constexpr, C_BLK: tl.constexpr, TB: tl.constexpr, HAS_W: tl.constexpr, HAS_D: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = tl.arange(0, C_BLK)
    m = offs < C
    hh = offs // G
    gg = offs - hh * G
    y_c = hh * s_yh + gg * s_yg
    z_c = hh * s_zh + gg * s_zg
    o_c = hh * s_oh + gg * s_og
    d_c = hh * s_dh + gg * s_dg
    p_c = hh * s_ph + gg * s_pg
    if HAS_W:
        w = tl.load(W_ptr + offs, mask=m, other=0.0).to(tl.float32)
    else:
        w = tl.zeros([C_BLK], dtype=tl.float32) + 1.0
    dw_acc = tl.zeros([C_BLK], dtype=tl.float32)
    dd_acc = tl.zeros([C_BLK], dtype=tl.float32)
    for i in tl.static_range(TB):
        t = pid * TB + i
        y = tl.load(Ypre_ptr + b * s_yb + t * s_yt + y_c, mask=m, other=0.0).to(tl.float32)
        if HAS_D:
            u = tl.load(U_ptr + b * s_ub + t * s_ut + offs * s_uc, mask=m, other=0.0).to(tl.float32)
            ds = tl.load(D_ptr + hh, mask=m, other=0.0).to(tl.float32)
            y = y + ds * u
        z = tl.load(Z_ptr + b * s_zb + t * s_zt + z_c, mask=m, other=0.0).to(tl.float32)
        dn = tl.load(dOut_ptr + b * s_ob + t * s_ot + o_c, mask=m, other=0.0).to(tl.float32)
        sg = tl.sigmoid(z)
        silu = z * sg
        v = y * silu
        r = tl.load(Rstd_ptr + b * L + t)
        dw_acc += dn * (v * r)
        gw = dn * w
        s = tl.sum(gw * v, axis=0)
        dv = r * gw - r * r * r * inv_c * s * v
        if HAS_D:
            tl.store(dU_ptr + (b * L + t) * C + offs, dv * silu * ds, mask=m)
            dd_acc += dv * silu * u
        tl.store(dYpre_ptr + b * s_db + t * s_dt + d_c, (dv * silu).to(dYpre_ptr.dtype.element_ty), mask=m)
        tl.store(dZ_ptr + b * s_pb + t * s_pt + p_c, dv * y * sg * (1.0 + z * (1.0 - sg)), mask=m)
    tl.store(dWpart_ptr + (b * NBLK + pid) * C + offs, dw_acc, mask=m)
    if HAS_D:
        tl.store(dDpart_ptr + (b * NBLK + pid) * C + offs, dd_acc, mask=m)

def _token_block(Bsz, L, target=1024):
    tb = 1
    while tb < 64 and Bsz * (L // (tb * 2)) >= target and (L % (tb * 2) == 0):
        tb *= 2
    return tb

def _row_strides(t, H, O, G):
    return (t.stride(0), t.stride(3), t.stride(1), t.stride(5))

def gate_norm_fwd(Ypre, Z, T, weight, eps, out_dtype=None, *, U=None, D=None):
    Bsz, H, nc, _T, O, G = Ypre.shape
    assert _T == T
    if O != 1:
        raise NotImplementedError('the fused output norm is wired at O == 1 (at O > 1 the gate broadcasts over o, so dZ needs a segmented reduction inside the token row)')
    L = nc * T
    C = H * O * G
    dt = out_dtype or torch.float32
    buf = torch.empty(Bsz, L, H, O, G, device=Ypre.device, dtype=dt)
    out = buf.view(Bsz, nc, T, H, O, G).permute(0, 3, 1, 2, 4, 5)
    rstd = torch.empty(Bsz, L, device=Ypre.device, dtype=torch.float32)
    tb = _token_block(Bsz, L)
    has_w = weight is not None
    _gate_norm_fwd_kernel[L // tb, Bsz](Ypre, Z, weight if has_w else Ypre, out, rstd, U if D is not None else Ypre, D if D is not None else Ypre, *(U.stride() if D is not None else (0, 0, 0)), *_row_strides(Ypre, H, O, G), Z.stride(0), Z.stride(1), G * Z.stride(2), Z.stride(2), *_row_strides(out, H, O, G), L, float(eps), 1.0 / C, G=G, C=C, C_BLK=triton.next_power_of_2(C), TB=tb, HAS_W=has_w, HAS_D=D is not None, num_warps=8)
    return (out, rstd)

def gate_norm_bwd(Ypre, Z, weight, rstd, dOut, T, dZ_out=None, *, U=None, D=None, dy_dtype=torch.float32, real=False):
    Bsz, H, nc, _T, O, G = Ypre.shape
    assert _T == T and O == 1
    L = nc * T
    C = H * O * G
    dYpre = _dypre_buffer(Ypre, dy_dtype, real=real)
    dZ = torch.empty_like(Z) if dZ_out is None else dZ_out
    if tuple(dZ.shape) != tuple(Z.shape):
        raise ValueError(f'dZ_out shape {tuple(dZ.shape)} != z shape {tuple(Z.shape)}')
    tb = _token_block(Bsz, L)
    nblk = L // tb
    has_w = weight is not None
    dw_part = torch.empty(Bsz, nblk, C, device=Ypre.device, dtype=torch.float32)
    du = torch.empty(U.shape, device=U.device, dtype=U.dtype) if D is not None else None
    dd_part = torch.empty_like(dw_part) if D is not None else None
    _gate_norm_bwd_kernel[nblk, Bsz](Ypre, Z, weight if has_w else Ypre, rstd, dOut, dYpre, dZ, dw_part, U if D is not None else Ypre, D if D is not None else Ypre, du if D is not None else dYpre, dd_part if D is not None else dw_part, *(U.stride() if D is not None else (0, 0, 0)), *_row_strides(Ypre, H, O, G), Z.stride(0), Z.stride(1), G * Z.stride(2), Z.stride(2), *_row_strides(dOut, H, O, G), *_row_strides(dYpre, H, O, G), dZ.stride(0), dZ.stride(1), G * dZ.stride(2), dZ.stride(2), L, 1.0 / C, nblk, G=G, C=C, C_BLK=triton.next_power_of_2(C), TB=tb, HAS_W=has_w, HAS_D=D is not None, num_warps=8)
    dW = dw_part.sum(dim=(0, 1)) if has_w else None
    if D is not None:
        dD = dd_part.sum(dim=(0, 1)).view(H, G).sum(-1).to(D.dtype)
        return (dYpre, dZ, dW, du, dD)
    return (dYpre, dZ, dW)

class _MixerGateNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, z, u, D, weight, eps, P):
        batch, length, width = z.shape
        heads = width // P
        yc = y.reshape(batch, length, heads, P).permute(0, 2, 1, 3)[:, :, None, :, None, :]
        uc = u.reshape(batch, length, width) if D is not None else None
        out, rstd = gate_norm_fwd(yc, z, length, weight, eps, out_dtype=z.dtype, U=uc, D=D)
        ctx.save_for_backward(yc, z, uc, D, weight, rstd)
        ctx.shapes = (y.shape, u.shape)
        return out.permute(0, 2, 3, 1, 4, 5).reshape(batch, length, width)

    @staticmethod
    def backward(ctx, grad):
        y, z, u, D, weight, rstd = ctx.saved_tensors
        b, length, width = z.shape
        heads, channels = (y.shape[1], y.shape[-1])
        go = grad.reshape(b, length, heads, channels).permute(0, 2, 1, 3)[:, :, None, :, None, :]
        result = gate_norm_bwd(y, z, weight, rstd, go, length, U=u, D=D, dy_dtype=y.dtype)
        dy, dz, dw = result[:3]
        dy = dy.permute(0, 2, 3, 1, 4, 5).reshape(ctx.shapes[0])
        du, dd = result[3:] if D is not None else (None, None)
        return (dy, dz, None if du is None else du.reshape(ctx.shapes[1]), dd, dw, None, None)

def fused_gate_norm(y, z, u, D, weight, eps, P):
    return _MixerGateNorm.apply(y, z, u, D, weight, eps, P)
