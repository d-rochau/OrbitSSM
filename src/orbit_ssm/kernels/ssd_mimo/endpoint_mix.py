
from __future__ import annotations
import torch
import triton
import triton.language as tl
from .common import ACC_SEM
from .rmix_epilogue import _skip_term

def _token_block(length: int, channels: int) -> int:
    budget = max(1, 8192 // max(1, triton.next_power_of_2(channels) * 4))
    size = 1
    cap = min(64, length, budget)
    while size * 2 <= cap:
        size *= 2
    return size

@triton.jit
def _endpoint_mix_fwd(Y, X, M, C, GATE, STATIC, DYN, DSK, OUT, YS0, YS1, YS2, YS3, YS4, XS0, XS1, XS2, XS3, XS4, MS0, MS1, MS2, MS3, MS4, CS0, CS1, CS2, GS0, GS1, GS2, ST0, ST1, DY0, DY1, DY2, DY3, DK0, DK1, DK2, DK3, L: tl.constexpr, H: tl.constexpr, G: tl.constexpr, R: tl.constexpr, O: tl.constexpr, BG: tl.constexpr, TB: tl.constexpr, HAS_GATE: tl.constexpr, HAS_STATIC: tl.constexpr, HAS_DYN: tl.constexpr, HAS_D: tl.constexpr):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tb * TB + tl.arange(0, TB)
    offs_g = tl.arange(0, BG)
    t_mask = offs_t < L
    g_mask = offs_g < G
    acc0 = tl.zeros((TB, BG), dtype=tl.float32)
    acc1 = tl.zeros((TB, BG), dtype=tl.float32)
    out_stride_h = H * G * O
    for sib in tl.static_range(2):
        s = sib * H + h
        coeff = tl.load(C + b * CS0 + s * CS1 + offs_t * CS2, t_mask, other=0).to(tl.float32)
        mix = tl.full((TB,), 0.0 if HAS_STATIC or HAS_DYN else 1.0, tl.float32)
        if HAS_STATIC:
            mix += tl.load(STATIC + h * ST0 + sib * ST1).to(tl.float32)
        if HAS_DYN:
            mix += tl.load(DYN + b * DY0 + offs_t * DY1 + h * DY2 + sib * DY3, t_mask, other=0).to(tl.float32)
        if HAS_GATE:
            z = tl.load(GATE + b * GS0 + offs_t[:, None] * GS1 + (s * G + offs_g[None, :]) * GS2, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
            gate = z * tl.sigmoid(z)
        else:
            gate = tl.full((TB, BG), 1.0, tl.float32)
        x_base = b * XS0 + s * XS1 + offs_t[:, None] * XS4 + offs_g[None, :] * XS2
        if HAS_D:
            xg = tl.load(X + x_base + sib * XS3, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32) * gate
        for o in tl.static_range(2):
            dot = tl.zeros((TB, BG), dtype=tl.float32)
            for r in tl.static_range(2):
                x = tl.load(X + x_base + r * XS3, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
                mv = tl.load(M + b * MS0 + offs_t * MS1 + s * MS2 + r * MS3 + o * MS4, mask=t_mask, other=0.0).to(tl.float32)[:, None]
                dot += x * mv
            correction = coeff[:, None] * dot * gate
            correction = correction.to(Y.dtype.element_ty)
            y = tl.load(Y + b * YS0 + s * YS1 + o * YS2 + offs_g[None, :] * YS3 + offs_t[:, None] * YS4, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
            y_corr = (y - correction.to(tl.float32)).to(Y.dtype.element_ty).to(tl.float32)
            if HAS_D:
                y_corr += _skip_term(xg, DSK, DK0, DK1, DK2, DK3, h, sib, o, offs_g, g_mask)
            if o == 0:
                acc0 += y_corr * mix[:, None]
            else:
                acc1 += y_corr * mix[:, None]
    base = (b * L + offs_t)[:, None] * out_stride_h + h * G * O
    o0 = base + offs_g[None, :] * O + 0
    o1 = base + offs_g[None, :] * O + 1
    mask = t_mask[:, None] & g_mask[None, :]
    tl.store(OUT + o0, acc0, mask=mask)
    tl.store(OUT + o1, acc1, mask=mask)

@triton.jit
def _endpoint_mix_bwd(Y, X, M, C, GATE, STATIC, DYN, DSK, DY_IN, DY, DX, DM, DC, DZ, DS, DR, DD, YS0, YS1, YS2, YS3, YS4, XS0, XS1, XS2, XS3, XS4, MS0, MS1, MS2, MS3, MS4, CS0, CS1, CS2, GS0, GS1, GS2, ST0, ST1, DY0, DY1, DY2, DY3, DK0, DK1, DK2, DK3, L: tl.constexpr, H: tl.constexpr, RH: tl.constexpr, G: tl.constexpr, R: tl.constexpr, O: tl.constexpr, BG: tl.constexpr, TB: tl.constexpr, HAS_GATE: tl.constexpr, HAS_STATIC: tl.constexpr, HAS_DYN: tl.constexpr, HAS_D: tl.constexpr):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    offs_t = tb * TB + tl.arange(0, TB)
    offs_g = tl.arange(0, BG)
    t_mask = offs_t < L
    g_mask = offs_g < G
    for sib in tl.static_range(2):
        s = sib * H + h
        coeff = tl.load(C + b * CS0 + s * CS1 + offs_t * CS2, t_mask, other=0).to(tl.float32)
        mix = tl.full((TB,), 0.0 if HAS_STATIC or HAS_DYN else 1.0, tl.float32)
        if HAS_STATIC:
            mix += tl.load(STATIC + h * ST0 + sib * ST1).to(tl.float32)
        if HAS_DYN:
            mix += tl.load(DYN + b * DY0 + offs_t * DY1 + h * DY2 + sib * DY3, t_mask, other=0).to(tl.float32)
        if HAS_GATE:
            z = tl.load(GATE + b * GS0 + offs_t[:, None] * GS1 + (s * G + offs_g[None, :]) * GS2, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
            sig = tl.sigmoid(z)
            gate = z * sig
            gate_prime = sig * (1.0 + z * (1.0 - sig))
        else:
            gate = tl.full((TB, BG), 1.0, tl.float32)
            gate_prime = tl.zeros((TB, BG), tl.float32)
        dcoeff = tl.zeros((TB,), tl.float32)
        mix_grad = tl.zeros((TB,), tl.float32)
        dz_term = tl.zeros((TB, BG), tl.float32)
        dxi0 = tl.zeros((TB, BG), tl.float32)
        dxi1 = tl.zeros((TB, BG), tl.float32)
        x_base = b * XS0 + s * XS1 + offs_t[:, None] * XS4 + offs_g[None, :] * XS2
        if HAS_D:
            xs = tl.load(X + x_base + sib * XS3, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
            xg = xs * gate
            dskip = tl.zeros((TB, BG), tl.float32)
        for o in tl.static_range(2):
            dot = tl.zeros((TB, BG), tl.float32)
            for r in tl.static_range(2):
                x = tl.load(X + x_base + r * XS3, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
                mv = tl.load(M + b * MS0 + offs_t * MS1 + s * MS2 + r * MS3 + o * MS4, mask=t_mask, other=0.0).to(tl.float32)[:, None]
                dot += x * mv
            correction = (coeff[:, None] * dot * gate).to(Y.dtype.element_ty)
            y = tl.load(Y + b * YS0 + s * YS1 + o * YS2 + offs_g[None, :] * YS3 + offs_t[:, None] * YS4, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
            y_corr = (y - correction.to(tl.float32)).to(Y.dtype.element_ty).to(tl.float32)
            if HAS_D:
                y_corr += _skip_term(xg, DSK, DK0, DK1, DK2, DK3, h, sib, o, offs_g, g_mask)
            dout = tl.load(DY_IN + (b * L + offs_t)[:, None] * (H * G * O) + h * G * O + offs_g[None, :] * O + o, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
            dy = (dout * mix[:, None]).to(Y.dtype.element_ty).to(tl.float32)
            tl.store(DY + (((b * RH + s) * O + o) * G + offs_g[None, :]) * L + offs_t[:, None], dy.to(Y.dtype.element_ty), mask=t_mask[:, None] & g_mask[None, :])
            dcoeff += -tl.sum(dy * dot * gate, axis=1)
            if HAS_STATIC or HAS_DYN:
                mix_grad += tl.sum(y_corr * dout, axis=1)
            dz_term += dy * dot
            if HAS_D:
                dv = tl.load(DSK + h * DK0 + offs_g * DK1 + sib * DK2 + o * DK3, mask=g_mask, other=0.0).to(tl.float32)
                dskip += dy * dv[None, :]
                tl.atomic_add(DD + h * DK0 + offs_g * DK1 + sib * DK2 + o * DK3, tl.sum(tl.where(t_mask[:, None] & g_mask[None, :], dy * xg, 0.0), axis=0), mask=g_mask, sem=ACC_SEM)
            for r in tl.static_range(2):
                x = tl.load(X + x_base + r * XS3, mask=t_mask[:, None] & g_mask[None, :], other=0.0).to(tl.float32)
                mv = tl.load(M + b * MS0 + offs_t * MS1 + s * MS2 + r * MS3 + o * MS4, mask=t_mask, other=0.0).to(tl.float32)[:, None]
                adj = -dy * coeff[:, None] * gate
                dmp = tl.sum(x * adj, axis=1)
                dmp_ptr = (((b * L + offs_t) * RH + s) * R + r) * O + o
                tl.store(DM + dmp_ptr, dmp, t_mask)
                if r == 0:
                    dxi0 += adj * mv
                else:
                    dxi1 += adj * mv
        if HAS_D:
            if sib == 0:
                dxi0 += dskip * gate
            else:
                dxi1 += dskip * gate
        tl.store(DC + (b * RH + s) * L + offs_t, dcoeff, t_mask)
        if HAS_DYN:
            tl.store(DR + ((b * L + offs_t) * H + h) * R + sib, mix_grad, t_mask)
        if HAS_STATIC:
            tl.atomic_add(DS + h * R + sib, tl.sum(mix_grad, axis=0), sem=ACC_SEM)
        if HAS_GATE:
            tz = -coeff[:, None] * dz_term
            if HAS_D:
                tz += dskip * xs
            tz = tz * gate_prime
            tl.store(DZ + (b * L + offs_t)[:, None] * (RH * G) + s * G + offs_g[None, :], tz.to(DZ.dtype.element_ty), mask=t_mask[:, None] & g_mask[None, :])
        dxi0_ptr = (((b * RH + s) * G + offs_g[None, :]) * R + 0) * L + offs_t[:, None]
        dxi1_ptr = (((b * RH + s) * G + offs_g[None, :]) * R + 1) * L + offs_t[:, None]
        mask = t_mask[:, None] & g_mask[None, :]
        tl.store(DX + dxi0_ptr, dxi0, mask=mask)
        tl.store(DX + dxi1_ptr, dxi1, mask=mask)

class _EndpointMix(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, x, m, coefficient, gate, c_static, rmix, H, R, O, G, out_dtype, d):
        if R != 2 or O != 2:
            raise ValueError('endpoint_mix supports only R=O=2')
        batch, siblings, o_dim, ch, length = y.shape
        if siblings != H * R:
            raise ValueError('endpoint_mix: y second dim must be R*H')
        if o_dim != O or ch != G:
            raise ValueError('endpoint_mix: y must be [B, R*H, O, G, L]')
        if x.shape != (batch, siblings, G, R, length):
            raise ValueError('endpoint_mix: x must be [B, R*H, G, R, L]')
        if m.shape != (batch, length, siblings, R, O):
            raise ValueError('endpoint_mix: m must be [B, L, R*H, R, O]')
        if coefficient.shape != (batch, siblings, length):
            raise ValueError('endpoint_mix: coefficient must be [B, R*H, L]')
        if gate is not None and gate.shape != (batch, length, siblings * G):
            raise ValueError('endpoint_mix: gate must be [B, L, R*H*G]')
        if c_static is not None and c_static.shape != (H, R):
            raise ValueError('endpoint_mix: c_static must be [H, R]')
        if rmix is not None and rmix.shape != (batch, length, H, R):
            raise ValueError('endpoint_mix: rmix must be [B, L, H, R]')
        if d is not None and tuple(d.shape) != (H, G, R, O):
            raise ValueError(f'endpoint_mix: d must be the normalised [H, G, R, O] gain, got {tuple(d.shape)} (H={H} G={G} R={R} O={O})')
        if d is not None and (not d.is_contiguous()):
            raise ValueError(f'endpoint_mix: d must be contiguous (the dD adjoint is addressed with its stride set, got {tuple(d.stride())})')
        tb = _token_block(length, G)
        out = torch.empty((batch, length, H, G, O), device=y.device, dtype=out_dtype)
        ys = y.stride()
        xs = x.stride()
        ms = m.stride()
        cs = coefficient.stride()
        gs = gate.stride() if gate is not None else (0, 0, 0)
        ds = c_static.stride() if c_static is not None else (0, 0)
        rs = rmix.stride() if rmix is not None else (0, 0, 0, 0)
        ks = d.stride() if d is not None else (0, 0, 0, 0)
        dummy = y.new_empty(0, dtype=out_dtype)
        bg = triton.next_power_of_2(G)
        _endpoint_mix_fwd[triton.cdiv(length, tb), batch, H](y, x, m, coefficient, gate if gate is not None else dummy, c_static if c_static is not None else dummy, rmix if rmix is not None else dummy, d if d is not None else dummy, out, ys[0], ys[1], ys[2], ys[3], ys[4], xs[0], xs[1], xs[2], xs[3], xs[4], ms[0], ms[1], ms[2], ms[3], ms[4], cs[0], cs[1], cs[2], gs[0], gs[1], gs[2], ds[0], ds[1], rs[0], rs[1], rs[2], rs[3], ks[0], ks[1], ks[2], ks[3], L=length, H=H, G=G, R=R, O=O, BG=bg, TB=tb, HAS_GATE=gate is not None, HAS_STATIC=c_static is not None, HAS_DYN=rmix is not None, HAS_D=d is not None, num_warps=4)
        ctx.save_for_backward(y, x, m, coefficient, gate if gate is not None else dummy, c_static if c_static is not None else dummy, rmix if rmix is not None else dummy, d)
        ctx.meta = (batch, length, H, G, bg, tb, gate is not None, c_static is not None, rmix is not None)
        return out.reshape(batch, length, H * G * O)

    @staticmethod
    def backward(ctx, dout):
        y, x, m, coefficient, gate, c_static, rmix, d = ctx.saved_tensors
        batch, length, H, G, bg, tb, has_gate, has_static, has_dyn = ctx.meta
        has_d = d is not None
        dout = dout.contiguous().reshape(batch, length, H, G, 2)
        RH = H * 2
        dY = torch.empty((batch, RH, 2, G, length), device=y.device, dtype=y.dtype)
        dX = torch.empty((batch, RH, G, 2, length), device=x.device, dtype=x.dtype)
        dM = torch.empty((batch, length, RH, 2, 2), device=m.device, dtype=torch.float32)
        dC = torch.empty((batch, RH, length), device=coefficient.device, dtype=torch.float32)
        dZ = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype) if has_gate else torch.empty(0, device=y.device, dtype=y.dtype)
        dS = torch.zeros(c_static.shape, device=c_static.device, dtype=torch.float32) if has_static else torch.empty(0, device=y.device, dtype=y.dtype)
        dR = torch.empty(rmix.shape, device=rmix.device, dtype=rmix.dtype) if has_dyn else torch.empty(0, device=y.device, dtype=y.dtype)
        dD = torch.zeros(d.shape, device=d.device, dtype=torch.float32) if has_d else torch.empty(0, device=y.device, dtype=y.dtype)
        ys = y.stride()
        xs = x.stride()
        ms = m.stride()
        cs = coefficient.stride()
        gs = gate.stride() if has_gate else (0, 0, 0)
        ds = c_static.stride() if has_static else (0, 0)
        rs = rmix.stride() if has_dyn else (0, 0, 0, 0)
        ks = d.stride() if has_d else (0, 0, 0, 0)
        dummy = y.new_empty(0, dtype=y.dtype)
        _endpoint_mix_bwd[triton.cdiv(length, tb), batch, H](y, x, m, coefficient, gate if has_gate else dummy, c_static if has_static else dummy, rmix if has_dyn else dummy, d if has_d else dummy, dout, dY, dX, dM, dC, dZ, dS, dR, dD, ys[0], ys[1], ys[2], ys[3], ys[4], xs[0], xs[1], xs[2], xs[3], xs[4], ms[0], ms[1], ms[2], ms[3], ms[4], cs[0], cs[1], cs[2], gs[0], gs[1], gs[2], ds[0], ds[1], rs[0], rs[1], rs[2], rs[3], ks[0], ks[1], ks[2], ks[3], L=length, H=H, RH=RH, G=G, R=2, O=2, BG=bg, TB=tb, HAS_GATE=has_gate, HAS_STATIC=has_static, HAS_DYN=has_dyn, HAS_D=has_d, num_warps=4)
        return (dY, dX, dM.to(m.dtype), dC.to(coefficient.dtype), dZ if has_gate else None, dS.to(c_static.dtype) if has_static else None, dR.to(rmix.dtype) if has_dyn else None, None, None, None, None, None, dD.to(d.dtype) if has_d else None)

def endpoint_mix(y, x, m, coefficient, gate, c_static, rmix, H: int, R: int, O: int, G: int, out_dtype: torch.dtype, d=None):
    return _EndpointMix.apply(y, x, m, coefficient, gate, c_static, rmix, H, R, O, G, out_dtype, d)
