
import torch
import triton
import triton.language as tl
from .common import ACC_SEM
import math
from .registered import endpoint_local_fwd

@triton.jit
def _local(B, C, P, BB, CB, RB, TH, M, DM, DB, DC, DP, DBB, DCB, DRB, DTH, BS: tl.constexpr, CS: tl.constexpr, PS: tl.constexpr, L: tl.constexpr, H: tl.constexpr, GB: tl.constexpr, GC: tl.constexpr, N: tl.constexpr, R: tl.constexpr, O: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, BO: tl.constexpr, BT: tl.constexpr, PHASE: tl.constexpr, BIAS_B: tl.constexpr, BIAS_C: tl.constexpr, OFFSET_B: tl.constexpr, OFFSET_C: tl.constexpr, BACKWARD: tl.constexpr):
    block = tl.program_id(0)
    blocks: tl.constexpr = triton.cdiv(L, BT)
    h, b = (block % H, block // (H * blocks))
    t = block // H % blocks * BT + tl.arange(0, BT)
    row = (b * L + t) * H + h
    k, r, o = (tl.arange(0, BN), tl.arange(0, BR), tl.arange(0, BO))
    e, odd = (2 * k, 2 * k + 1)
    bp = b * BS[0] + t * BS[1] + h // (H // GB) * BS[2]
    cp = b * CS[0] + t * CS[1] + h // (H // GC) * CS[2]
    bm = (r[None, :, None] < R) & (t[None, None, :] < L)
    cm = (o[None, :, None] < O) & (t[None, None, :] < L)
    be = tl.load(B + bp[None, None, :] + e[:, None, None] * BS[3] + r[None, :, None] * BS[4], (e[:, None, None] < N) & bm, 0).to(tl.float32)
    bo = tl.load(B + bp[None, None, :] + odd[:, None, None] * BS[3] + r[None, :, None] * BS[4], (odd[:, None, None] < N) & bm, 0).to(tl.float32)
    ce = tl.load(C + cp[None, None, :] + e[:, None, None] * CS[3] + o[None, :, None] * CS[4], (e[:, None, None] < N) & cm, 0).to(tl.float32)
    co = tl.load(C + cp[None, None, :] + odd[:, None, None] * CS[3] + o[None, :, None] * CS[4], (odd[:, None, None] < N) & cm, 0).to(tl.float32)
    phase = tl.full((BN, BT), 0.0, tl.float32)
    rb, th = (tl.full((BN,), 0.0, tl.float32), tl.full((BN,), 0.0, tl.float32))
    if PHASE:
        phase = tl.load(P + b * PS[0] + t[None, :] * PS[1] + h * PS[2] + k[:, None] * PS[3], (k[:, None] < K) & (t[None, :] < L), 0)
    if OFFSET_B:
        rb = tl.load(RB + h * K + k, k < K, 0)
    if OFFSET_C:
        th = tl.load(TH + h * K + k, k < K, 0)
    cb, sb = (tl.cos(phase + rb[:, None]), tl.sin(phase + rb[:, None]))
    cc, sc = (tl.cos(phase + th[:, None]), tl.sin(phase + th[:, None]))
    ch, sh = (tl.cos(th), tl.sin(th))
    bre, bro = (cb[:, None, :] * be + sb[:, None, :] * bo, cb[:, None, :] * bo - sb[:, None, :] * be)
    cre, cro = (cc[:, None, :] * ce + sc[:, None, :] * co, cc[:, None, :] * co - sc[:, None, :] * ce)
    bwe, bwo = (bre, bro)
    if BIAS_B:
        bwe += tl.load(BB + h * N * R + e[:, None] * R + r[None, :], (e[:, None] < N) & (r[None, :] < R), 0)[:, :, None]
        bwo += tl.load(BB + h * N * R + odd[:, None] * R + r[None, :], (odd[:, None] < N) & (r[None, :] < R), 0)[:, :, None]
    cwe, cwo = (cre, cro)
    gre, gro = (tl.full((BN, BO), 0.0, tl.float32), tl.full((BN, BO), 0.0, tl.float32))
    if BIAS_C:
        ge = tl.load(CB + h * N * O + e[:, None] * O + o[None, :], (e[:, None] < N) & (o[None, :] < O), 0)
        go = tl.load(CB + h * N * O + odd[:, None] * O + o[None, :], (odd[:, None] < N) & (o[None, :] < O), 0)
        gre, gro = (ch[:, None] * ge + sh[:, None] * go, ch[:, None] * go - sh[:, None] * ge)
        cwe += gre[:, :, None]
        cwo += gro[:, :, None]
    mp = row[None, None, :] * R * O + r[:, None, None] * O + o[None, :, None]
    mmask = (r[:, None, None] < R) & (o[None, :, None] < O) & (t[None, None, :] < L)
    if not BACKWARD:
        value = tl.sum(bwe[:, :, None, :] * cwe[:, None, :, :] + bwo[:, :, None, :] * cwo[:, None, :, :], 0)
        tl.store(M + mp, value, mmask)
    else:
        dm = tl.load(DM + mp, mmask, 0)
        dbe = tl.sum(dm[None, :, :, :] * cwe[:, None, :, :], 2)
        dbo = tl.sum(dm[None, :, :, :] * cwo[:, None, :, :], 2)
        dce = tl.sum(dm[None, :, :, :] * bwe[:, :, None, :], 1)
        dco = tl.sum(dm[None, :, :, :] * bwo[:, :, None, :], 1)
        dbp = ((b * L + t) * GB + h // (H // GB)) * N * R
        dcp = ((b * L + t) * GC + h // (H // GC)) * N * O
        tl.atomic_add(DB + dbp[None, None, :] + e[:, None, None] * R + r[None, :, None], cb[:, None, :] * dbe - sb[:, None, :] * dbo, (e[:, None, None] < N) & bm, sem=ACC_SEM)
        tl.atomic_add(DB + dbp[None, None, :] + odd[:, None, None] * R + r[None, :, None], sb[:, None, :] * dbe + cb[:, None, :] * dbo, (odd[:, None, None] < N) & bm, sem=ACC_SEM)
        tl.atomic_add(DC + dcp[None, None, :] + e[:, None, None] * O + o[None, :, None], cc[:, None, :] * dce - sc[:, None, :] * dco, (e[:, None, None] < N) & cm, sem=ACC_SEM)
        tl.atomic_add(DC + dcp[None, None, :] + odd[:, None, None] * O + o[None, :, None], sc[:, None, :] * dce + cc[:, None, :] * dco, (odd[:, None, None] < N) & cm, sem=ACC_SEM)
        if BIAS_B:
            tl.atomic_add(DBB + h * N * R + e[:, None] * R + r[None, :], tl.sum(dbe, 2), (e[:, None] < N) & (r[None, :] < R), sem=ACC_SEM)
            tl.atomic_add(DBB + h * N * R + odd[:, None] * R + r[None, :], tl.sum(dbo, 2), (odd[:, None] < N) & (r[None, :] < R), sem=ACC_SEM)
        if BIAS_C:
            tl.atomic_add(DCB + h * N * O + e[:, None] * O + o[None, :], tl.sum(ch[:, None, None] * dce - sh[:, None, None] * dco, 2), (e[:, None] < N) & (o[None, :] < O), sem=ACC_SEM)
            tl.atomic_add(DCB + h * N * O + odd[:, None] * O + o[None, :], tl.sum(sh[:, None, None] * dce + ch[:, None, None] * dco, 2), (odd[:, None] < N) & (o[None, :] < O), sem=ACC_SEM)
        dbangle = tl.sum(dbe * bro - dbo * bre, 1)
        dcangle = tl.sum(dce * cro - dco * cre, 1)
        if PHASE:
            tl.store(DP + row[None, :] * K + k[:, None], dbangle + dcangle, (k[:, None] < K) & (t[None, :] < L))
        if OFFSET_B:
            tl.atomic_add(DRB + h * K + k, tl.sum(dbangle, 1), k < K, sem=ACC_SEM)
        if OFFSET_C:
            tl.atomic_add(DTH + h * K + k, tl.sum(dcangle + tl.sum(dce * gro[:, :, None] - dco * gre[:, :, None], 1), 1), k < K, sem=ACC_SEM)

def _local_kernel(n, r, o, heads, gb, gc):
    shared = math.gcd(heads // gb, heads // gc)
    if n == 64 and r == o == 2 and (shared >= 2) and (shared % 2 == 0):
        from .endpoint_gluon import local_grouped
        return local_grouped
    return _local

def _local_plan(b, c, phase, beta, gamma, rb, th, heads):
    batch, length, gb, n, r = b.shape
    o, gc = (c.shape[-1], c.shape[2])
    flags = tuple((v is not None for v in (phase, beta, gamma, rb, th)))
    dummy = b.new_empty(0, dtype=torch.float32)
    values = (b, c, *(v if v is not None else dummy for v in (phase, beta, gamma, rb, th)))
    bt = max(1, min(8, 2048 // (triton.next_power_of_2((n + 1) // 2) * triton.next_power_of_2(r) * triton.next_power_of_2(o))))
    rows = batch * heads * triton.cdiv(length, bt)
    meta = dict(BT=bt, BS=b.stride(), CS=c.stride(), PS=phase.stride() if phase is not None else (0, 0, 0, 0), L=length, H=heads, GB=gb, GC=gc, N=n, R=r, O=o, K=n // 2, BN=triton.next_power_of_2((n + 1) // 2), BR=triton.next_power_of_2(r), BO=triton.next_power_of_2(o), PHASE=flags[0], BIAS_B=flags[1], BIAS_C=flags[2], OFFSET_B=flags[3], OFFSET_C=flags[4])
    kernel = _local_kernel(n, r, o, heads, gb, gc)
    return (values, meta, flags, rows, kernel)

class _LocalMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, c, phase, beta, gamma, rb, th, heads):
        values, meta, flags, rows, kernel = _local_plan(b, c, phase, beta, gamma, rb, th, heads)
        out = b.new_empty(b.shape[0], b.shape[1], heads, b.shape[-1], c.shape[-1], dtype=torch.float32)
        dummy = b.new_empty(0, dtype=torch.float32)
        kernel[rows,](*values, out, dummy, *[dummy] * 7, **meta, BACKWARD=False, num_warps=4)
        ctx.save_for_backward(*values)
        ctx.meta, ctx.flags, ctx.rows = (meta, flags, rows)
        ctx.kernel = kernel
        return out

    @staticmethod
    def backward(ctx, dout):
        values = ctx.saved_tensors
        grads = [torch.zeros(v.shape, device=v.device, dtype=torch.float32) for v in values]
        dummy = values[0].new_empty(0, dtype=torch.float32)
        ctx.kernel[ctx.rows,](*values, dummy, dout.contiguous(), *grads, **ctx.meta, BACKWARD=True, num_warps=4)
        result = [g.to(v.dtype) for g, v in zip(grads, values)]
        for i, present in enumerate(ctx.flags, 2):
            if not present:
                result[i] = None
        return (*result, None)

def endpoint_local_map(b, c, phase, beta, gamma, rb, th, heads):
    n = b.shape[-2]
    result = None
    for start in range(0, n, 128):
        end = min(start + 128, n)

        def bias_slice(v):
            return None if v is None else v[:, start:end].contiguous()

        def angle_slice(v):
            return None if v is None else v[..., start // 2:end // 2].contiguous()
        local_op = endpoint_local_fwd if torch.compiler.is_compiling() else _LocalMap.apply
        value = local_op(b[..., start:end, :], c[..., start:end, :], None if phase is None else phase[..., start // 2:end // 2], bias_slice(beta), bias_slice(gamma), angle_slice(rb), angle_slice(th), heads)
        result = value if result is None else result + value
    return result

@triton.jit
def _correct(Y, X, M, C, Z, OUT, DY, DX, DM, DC, DZ, XS: tl.constexpr, YS: tl.constexpr, CS: tl.constexpr, ZS: tl.constexpr, L: tl.constexpr, H: tl.constexpr, G: tl.constexpr, R: tl.constexpr, O: tl.constexpr, BG: tl.constexpr, BR: tl.constexpr, BO: tl.constexpr, BT: tl.constexpr, GATE: tl.constexpr, BACKWARD: tl.constexpr):
    block = tl.program_id(0)
    blocks: tl.constexpr = triton.cdiv(L, BT)
    h, b = (block % H, block // (H * blocks))
    t = block // H % blocks * BT + tl.arange(0, BT)
    row = (b * L + t) * H + h
    g, r, o = (tl.arange(0, BG), tl.arange(0, BR), tl.arange(0, BO))
    xp = b * XS[0] + h * XS[1] + g[:, None, None] * XS[2] + r[None, :, None] * XS[3] + t[None, None, :] * XS[4]
    xv = tl.load(X + xp, (g[:, None, None] < G) & (r[None, :, None] < R) & (t[None, None, :] < L), 0).to(tl.float32)
    mp = row[None, None, :] * R * O + r[:, None, None] * O + o[None, :, None]
    mmask = (r[:, None, None] < R) & (o[None, :, None] < O) & (t[None, None, :] < L)
    mv = tl.load(M + mp, mmask, 0)
    cp = b * CS[0] + h * CS[1] + t * CS[2]
    cv = tl.load(C + cp, t < L, 0)
    dot = tl.sum(xv[:, :, None, :] * mv[None, :, :, :], 1)
    gate = tl.full((BG, BT), 1.0, tl.float32)
    if GATE:
        zp = b * ZS[0] + t[None, :] * ZS[1] + h * G + g[:, None]
        zv = tl.load(Z + zp, (g[:, None] < G) & (t[None, :] < L), 0).to(tl.float32)
        sig = tl.sigmoid(zv)
        gate = zv * sig
    yp = b * YS[0] + h * YS[1] + o[None, :, None] * YS[2] + g[:, None, None] * YS[3] + t[None, None, :] * YS[4]
    op = (((b * H + h) * O + o[None, :, None]) * G + g[:, None, None]) * L + t[None, None, :]
    mask = (g[:, None, None] < G) & (o[None, :, None] < O) & (t[None, None, :] < L)
    if not BACKWARD:
        yv = tl.load(Y + yp, mask, 0).to(tl.float32)
        correction = (cv[None, None, :] * dot * gate[:, None, :]).to(Y.dtype.element_ty)
        tl.store(OUT + op, yv - correction.to(tl.float32), mask)
    else:
        dy = tl.load(DY + op, mask, 0).to(tl.float32)
        adj = -dy * cv[None, None, :] * gate[:, None, :]
        dx = tl.sum(adj[:, None, :, :] * mv[None, :, :, :], 2)
        dxp = (((b * H + h) * G + g[:, None, None]) * R + r[None, :, None]) * L + t[None, None, :]
        tl.store(DX + dxp, dx, (g[:, None, None] < G) & (r[None, :, None] < R) & (t[None, None, :] < L))
        dm = tl.sum(xv[:, :, None, :] * adj[:, None, :, :], 0)
        tl.store(DM + mp, dm, mmask)
        tl.store(DC + (b * H + h) * L + t, -tl.sum(tl.sum(dy * dot * gate[:, None, :], 0), 0), t < L)
        if GATE:
            dz = -cv[None, :] * tl.sum(dy * dot, 1) * sig * (1.0 + zv * (1.0 - sig))
            tl.store(DZ + ((b * L + t[None, :]) * H + h) * G + g[:, None], dz, (g[:, None] < G) & (t[None, :] < L))

class _EndpointCorrect(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, x, local_map, coefficient, gate):
        batch, heads, channels, rank, length = x.shape
        outputs = local_map.shape[-1]
        yy = y.reshape(batch, heads, outputs, channels, length)
        dummy = x.new_empty(0)
        z = dummy if gate is None else gate
        out = torch.empty(yy.shape, device=y.device, dtype=y.dtype)
        bt = max(1, min(64, 8192 // (triton.next_power_of_2(channels) * triton.next_power_of_2(rank) * triton.next_power_of_2(outputs))))
        rows = batch * heads * triton.cdiv(length, bt)
        meta = dict(BT=bt, XS=x.stride(), YS=yy.stride(), CS=coefficient.stride(), ZS=z.stride() if gate is not None else (0, 0, 0), L=length, H=heads, G=channels, R=rank, O=outputs, BG=triton.next_power_of_2(channels), BR=triton.next_power_of_2(rank), BO=triton.next_power_of_2(outputs), GATE=gate is not None)
        _correct[rows,](yy, x, local_map, coefficient, z, out, *[dummy] * 5, **meta, BACKWARD=False)
        ctx.save_for_backward(x, local_map, coefficient, z)
        ctx.meta, ctx.shape, ctx.rows = (meta, y.shape, rows)
        return out.reshape(y.shape)

    @staticmethod
    def backward(ctx, dy):
        x, matrix, coefficient, z = ctx.saved_tensors
        dummy = x.new_empty(0)
        dx, dm, dc = (torch.empty_like(x, memory_format=torch.contiguous_format), torch.empty_like(matrix), torch.empty_like(coefficient, memory_format=torch.contiguous_format))
        dz = torch.empty_like(z, memory_format=torch.contiguous_format) if ctx.meta['GATE'] else dummy
        _correct[ctx.rows,](dummy, x, matrix, coefficient, z, dummy, dy.contiguous(), dx, dm, dc, dz, **ctx.meta, BACKWARD=True)
        return (dy, dx, dm, dc, dz if ctx.meta['GATE'] else None)

def endpoint_correct(y, x, local_map, coefficient, gate=None):
    return _EndpointCorrect.apply(y, x, local_map, coefficient, gate)

@triton.jit
def _weights(D, A, W, C, GW, GC, GD, GA, SIZE: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < SIZE
    dt = tl.load(D + i, mask, 0).to(tl.float32)
    lam = tl.load(A + i, mask, 0).to(tl.float32)
    if not BACKWARD:
        dn = tl.load(D + i + 1, mask & (i % L < L - 1), 0).to(tl.float32)
        ln = tl.load(A + i + 1, mask & (i % L < L - 1), 0).to(tl.float32)
        nxt = dn * (1.0 - ln)
        tl.store(C + i, nxt, mask)
        tl.store(W + i, lam * dt + nxt, mask)
    else:
        gw = tl.load(GW + i, mask, 0)
        previous = tl.load(GW + i - 1, mask & (i % L > 0), 0) + tl.load(GC + i - 1, mask & (i % L > 0), 0)
        tl.store(GD + i, lam * gw + (1.0 - lam) * previous, mask)
        tl.store(GA + i, dt * (gw - previous), mask)

class _EndpointWeights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dt, lam):
        dt, lam = (dt.contiguous(), lam.contiguous())
        write, coefficient = (torch.empty_like(dt), torch.empty_like(dt))
        _weights[triton.cdiv(dt.numel(), 256),](dt, lam, write, coefficient, dt, dt, dt, dt, SIZE=dt.numel(), L=dt.shape[-1], BLOCK=256, BACKWARD=False)
        ctx.save_for_backward(dt, lam)
        return (write, coefficient)

    @staticmethod
    def backward(ctx, gw, gc):
        dt, lam = ctx.saved_tensors
        dd, dl = (torch.empty_like(dt), torch.empty_like(lam))
        _weights[triton.cdiv(dt.numel(), 256),](dt, lam, dt, dt, gw.contiguous(), gc.contiguous(), dd, dl, SIZE=dt.numel(), L=dt.shape[-1], BLOCK=256, BACKWARD=True)
        return (dd, dl)

def endpoint_weights(dt, lam):
    if dt.shape != lam.shape:
        raise ValueError('endpoint dt and lambda must have matching shapes')
    return _EndpointWeights.apply(dt, lam)
