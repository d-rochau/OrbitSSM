
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _gate(x, ACT: tl.constexpr):
    if ACT == 0:
        return x
    elif ACT == 1:
        return tl.sigmoid(x)
    elif ACT == 2:
        return tl.exp(x)
    elif ACT == 3:
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0
    elif ACT == 5:
        return 1.0 + x
    else:
        return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

@triton.jit
def _gate_grad(x, y, ACT: tl.constexpr):
    if ACT == 0:
        return tl.full(x.shape, 1.0, tl.float32)
    elif ACT == 1:
        return y * (1.0 - y)
    elif ACT == 2:
        return y
    elif ACT == 3:
        return 1.0 - y * y
    elif ACT == 5:
        return tl.full(x.shape, 1.0, tl.float32)
    else:
        return tl.sigmoid(x)

@triton.jit
def _maps(X, Bias, Gate, GBias, b, h, k, t, mask, N: tl.constexpr, L: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr, GGROUP: tl.constexpr, HAS_GATE: tl.constexpr, ACT: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr):
    n0 = 2 * k
    base = (b * N + n0[:, None]) * L + t[None, :]
    x0 = tl.load(X + base, mask, 0).to(tl.float32)
    x1 = tl.load(X + base + L, mask, 0).to(tl.float32)
    if GROUP:
        x0 += tl.load(Bias + h // (H // GROUP) * N + n0, k < N // 2, 0)[:, None]
        x1 += tl.load(Bias + h // (H // GROUP) * N + n0 + 1, k < N // 2, 0)[:, None]
    g0 = tl.full((BK, BT), 0.0, tl.float32)
    g1 = tl.full((BK, BT), 0.0, tl.float32)
    f0 = tl.full((BK, BT), 1.0, tl.float32)
    f1 = tl.full((BK, BT), 1.0, tl.float32)
    if HAS_GATE:
        g0 = tl.load(Gate + base, mask, 0).to(tl.float32)
        g1 = tl.load(Gate + base + L, mask, 0).to(tl.float32)
        if GGROUP:
            g0 += tl.load(GBias + h // (H // GGROUP) * N + n0, k < N // 2, 0)[:, None]
            g1 += tl.load(GBias + h // (H // GGROUP) * N + n0 + 1, k < N // 2, 0)[:, None]
        f0, f1 = (_gate(g0, ACT), _gate(g1, ACT))
    return (x0, x1, g0, g1, f0, f1)

@triton.jit
def _fold_forward(B, C, Delta, Theta, Offset, Initial, BB, CB, GB, GC, GBB, GCB, DBias, Rot, BO, CO, Prefix, Last, BS: tl.constexpr, H: tl.constexpr, N: tl.constexpr, L: tl.constexpr, NC: tl.constexpr, BGR: tl.constexpr, CGR: tl.constexpr, GBGR: tl.constexpr, GCGR: tl.constexpr, HAS_GB: tl.constexpr, HAS_GC: tl.constexpr, HAS_OFFSET: tl.constexpr, HAS_INITIAL: tl.constexpr, HAS_DBIAS: tl.constexpr, HAS_ROT: tl.constexpr, THETA_DELTA: tl.constexpr, SOFTPLUS: tl.constexpr, ACT: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr):
    b, h = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64))
    k = tl.program_id(2) * BK + tl.arange(0, BK)
    K: tl.constexpr = N // 2
    rot = tl.full((BK,), True, tl.int1)
    if HAS_ROT:
        rot = tl.load(Rot + 2 * k, k < K, 0) != 0
    carry = tl.full((BK,), 0.0, tl.float32)
    if HAS_INITIAL:
        carry = tl.load(Initial + (b * H + h) * K + k, k < K, 0).to(tl.float32)
    offset = tl.full((BK,), 0.0, tl.float32)
    if HAS_OFFSET:
        offset = tl.load(Offset + h * K + k, k < K, 0).to(tl.float32)
    for j in range(NC):
        t = j * BT + tl.arange(0, BT)
        mask = (k[:, None] < K) & (t[None, :] < L)
        tl.store(Prefix + ((h * BS + b) * K + k) * NC + j, carry, k < K)
        inc = tl.load(Theta + (b * K + k[:, None]) * L + t[None, :], mask, 0).to(tl.float32)
        if THETA_DELTA:
            raw = tl.load(Delta + (b * H + h) * L + t, t < L, 0).to(tl.float32)
            if HAS_DBIAS:
                raw += tl.load(DBias + h).to(tl.float32)
            de = _gate(raw, 4) if SOFTPLUS else raw
            inc *= de[None, :]
        phase = carry[:, None] + tl.cumsum(inc, 1)
        cs, sn = (tl.cos(phase), tl.sin(phase))
        cc, sc = (tl.cos(phase + offset[:, None]), tl.sin(phase + offset[:, None]))
        b0, b1, _, _, fb0, fb1 = _maps(B, BB, GB, GBB, b, h, k, t, mask, N, L, H, BGR, GBGR, HAS_GB, ACT, BK, BT)
        c0, c1, _, _, fc0, fc1 = _maps(C, CB, GC, GCB, b, h, k, t, mask, N, L, H, CGR, GCGR, HAS_GC, ACT, BK, BT)
        be = tl.where(rot[:, None], cs * b0 - sn * b1, b0) * fb0
        bo = tl.where(rot[:, None], sn * b0 + cs * b1, b1) * fb1
        ce = tl.where(rot[:, None], cc * c0 - sc * c1, c0) * fc0
        co = tl.where(rot[:, None], sc * c0 + cc * c1, c1) * fc1
        out = ((h * BS + b) * N + 2 * k[:, None]) * L + t[None, :]
        tl.store(BO + out, be, mask)
        tl.store(BO + out + L, bo, mask)
        tl.store(CO + out, ce, mask)
        tl.store(CO + out + L, co, mask)
        carry += tl.sum(inc, 1)
        carry -= 6.283185307179586 * tl.floor(carry * 0.15915494309189535)
    tl.store(Last + (b * H + h) * K + k, carry, k < K)

@triton.jit
def _map_backward(X, Bias, Gate, GBias, GO, DX, DB, DG, DGB, cs, sn, rot, b, h, k, t, mask, BS: tl.constexpr, H: tl.constexpr, N: tl.constexpr, L: tl.constexpr, GROUP: tl.constexpr, GGROUP: tl.constexpr, HAS_GATE: tl.constexpr, ACT: tl.constexpr, SLOTS: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr):
    K: tl.constexpr = N // 2
    x0, x1, g0, g1, f0, f1 = _maps(X, Bias, Gate, GBias, b, h, k, t, mask, N, L, H, GROUP, GGROUP, HAS_GATE, ACT, BK, BT)
    re = tl.where(rot[:, None], cs * x0 - sn * x1, x0)
    im = tl.where(rot[:, None], sn * x0 + cs * x1, x1)
    out = ((h * BS + b) * N + 2 * k[:, None]) * L + t[None, :]
    ge = tl.load(GO + out, mask, 0).to(tl.float32)
    gi = tl.load(GO + out + L, mask, 0).to(tl.float32)
    dst = ((h % SLOTS * BS + b) * N + 2 * k[:, None]) * L + t[None, :]
    if HAS_GATE:
        dg0 = ge * re * _gate_grad(g0, f0, ACT)
        dg1 = gi * im * _gate_grad(g1, f1, ACT)
        tl.atomic_add(DG + dst, dg0, mask, sem='relaxed')
        tl.atomic_add(DG + dst + L, dg1, mask, sem='relaxed')
        if GGROUP:
            idx = h // (H // GGROUP) * N + 2 * k
            tl.atomic_add(DGB + idx, tl.sum(tl.where(mask, dg0, 0.0), 1), k < K, sem='relaxed')
            tl.atomic_add(DGB + idx + 1, tl.sum(tl.where(mask, dg1, 0.0), 1), k < K, sem='relaxed')
    ge *= f0
    gi *= f1
    dphase = tl.where(rot[:, None], gi * re - ge * im, 0.0)
    dx0 = tl.where(rot[:, None], cs * ge + sn * gi, ge)
    dx1 = tl.where(rot[:, None], cs * gi - sn * ge, gi)
    tl.atomic_add(DX + dst, dx0, mask, sem='relaxed')
    tl.atomic_add(DX + dst + L, dx1, mask, sem='relaxed')
    if GROUP:
        idx = h // (H // GROUP) * N + 2 * k
        tl.atomic_add(DB + idx, tl.sum(tl.where(mask, dx0, 0.0), 1), k < K, sem='relaxed')
        tl.atomic_add(DB + idx + 1, tl.sum(tl.where(mask, dx1, 0.0), 1), k < K, sem='relaxed')
    return dphase

@triton.jit
def _fold_backward(B, C, Delta, Theta, Offset, Initial, BB, CB, GB, GC, GBB, GCB, DBias, Rot, Prefix, GBO, GCO, GLast, DB, DC, DDelta, DTheta, DOffset, DInitial, DBB, DCB, DGB, DGC, DGBB, DGCB, BS: tl.constexpr, H: tl.constexpr, N: tl.constexpr, L: tl.constexpr, NC: tl.constexpr, BGR: tl.constexpr, CGR: tl.constexpr, GBGR: tl.constexpr, GCGR: tl.constexpr, HAS_GB: tl.constexpr, HAS_GC: tl.constexpr, HAS_OFFSET: tl.constexpr, HAS_INITIAL: tl.constexpr, HAS_DBIAS: tl.constexpr, HAS_ROT: tl.constexpr, HAS_GL: tl.constexpr, THETA_DELTA: tl.constexpr, SOFTPLUS: tl.constexpr, ACT: tl.constexpr, SLOTS: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr):
    b, h = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64))
    k = tl.program_id(2) * BK + tl.arange(0, BK)
    K: tl.constexpr = N // 2
    rot = tl.full((BK,), True, tl.int1)
    if HAS_ROT:
        rot = tl.load(Rot + 2 * k, k < K, 0) != 0
    offset = tl.full((BK,), 0.0, tl.float32)
    if HAS_OFFSET:
        offset = tl.load(Offset + h * K + k, k < K, 0).to(tl.float32)
    adj = tl.full((BK,), 0.0, tl.float32)
    if HAS_GL:
        adj = tl.load(GLast + (b * H + h) * K + k, k < K, 0).to(tl.float32)
    off_adj = tl.full((BK,), 0.0, tl.float32)
    for q in range(NC):
        j = NC - 1 - q
        t = j * BT + tl.arange(0, BT)
        mask = (k[:, None] < K) & (t[None, :] < L)
        carry = tl.load(Prefix + ((h * BS + b) * K + k) * NC + j, k < K, 0)
        feat = tl.load(Theta + (b * K + k[:, None]) * L + t[None, :], mask, 0).to(tl.float32)
        de = tl.full((BT,), 1.0, tl.float32)
        deriv = tl.full((BT,), 1.0, tl.float32)
        if THETA_DELTA:
            raw = tl.load(Delta + (b * H + h) * L + t, t < L, 0).to(tl.float32)
            if HAS_DBIAS:
                raw += tl.load(DBias + h).to(tl.float32)
            de = _gate(raw, 4) if SOFTPLUS else raw
            if SOFTPLUS:
                deriv = tl.sigmoid(raw)
        phase = carry[:, None] + tl.cumsum(feat * de[None, :], 1)
        pb = _map_backward(B, BB, GB, GBB, GBO, DB, DBB, DGB, DGBB, tl.cos(phase), tl.sin(phase), rot, b, h, k, t, mask, BS, H, N, L, BGR, GBGR, HAS_GB, ACT, SLOTS, BK, BT)
        pc = _map_backward(C, CB, GC, GCB, GCO, DC, DCB, DGC, DGCB, tl.cos(phase + offset[:, None]), tl.sin(phase + offset[:, None]), rot, b, h, k, t, mask, BS, H, N, L, CGR, GCGR, HAS_GC, ACT, SLOTS, BK, BT)
        gphase = tl.where(mask, pb + pc, 0.0)
        off_adj += tl.sum(tl.where(mask, pc, 0.0), 1)
        ginc = tl.cumsum(gphase, 1, reverse=True) + adj[:, None]
        adj += tl.sum(gphase, 1)
        idx = ((h % SLOTS * BS + b) * K + k[:, None]) * L + t[None, :]
        tl.atomic_add(DTheta + idx, ginc * de[None, :], mask, sem='relaxed')
        if THETA_DELTA:
            dd = tl.sum(tl.where(mask, ginc * feat, 0.0), 0) * deriv
            tl.atomic_add(DDelta + (b * H + h) * L + t, dd, t < L, sem='relaxed')
    if HAS_INITIAL:
        tl.store(DInitial + (b * H + h) * K + k, adj, k < K)
    if HAS_OFFSET:
        tl.atomic_add(DOffset + h * K + k, off_adj, k < K, sem='relaxed')

class _HalfFold(torch.autograd.Function):

    @staticmethod
    def forward(ctx, opts, *args):
        bmap, cmap, delta, theta = args[:4]
        bs, n, length = bmap.shape
        h = delta.shape[1]
        bk, bt = (opts.get('pair_tile', 4), opts.get('time_tile', 128))
        nc = triton.cdiv(length, bt)
        bo = torch.empty(h, bs, n, length, device=bmap.device, dtype=torch.float32)
        co = torch.empty_like(bo)
        prefix = torch.empty(h, bs, n // 2, nc, device=bmap.device, dtype=torch.float32)
        last = torch.empty(bs, h, n // 2, device=bmap.device, dtype=torch.float32)
        offset, initial, bb, cb, gb, gc, gbb, gcb, dbias, rot = args[4:]
        kw = dict(BS=bs, H=h, N=n, L=length, NC=nc, BK=bk, BT=bt, BGR=0 if bb is None else bb.shape[0], CGR=0 if cb is None else cb.shape[0], GBGR=0 if gbb is None else gbb.shape[0], GCGR=0 if gcb is None else gcb.shape[0], HAS_GB=gb is not None, HAS_GC=gc is not None, HAS_OFFSET=offset is not None, HAS_INITIAL=initial is not None, HAS_DBIAS=dbias is not None, HAS_ROT=rot is not None, THETA_DELTA=opts['theta_delta'], SOFTPLUS=opts['delta_softplus'], ACT=opts['gate_act'])
        packed = [a.contiguous() if a is not None else None for a in args]
        ctx.save_for_backward(*packed, prefix)
        ctx.kw = kw
        ctx.set_materialize_grads(False)
        _fold_forward[bs, h, triton.cdiv(n // 2, bk)](*[a if a is not None else bmap for a in packed], bo, co, prefix, last, **kw, num_warps=4)
        return (bo, co, last)

    @staticmethod
    def backward(ctx, gbo, gco, glast):
        *args, prefix = ctx.saved_tensors
        kw = ctx.kw
        bs, h, n, length = (kw[k] for k in ('BS', 'H', 'N', 'L'))
        device = args[0].device
        slots = min(h, 8)

        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=torch.float32)
        dummy = zeros(1)
        grad = [zeros(*a.shape) if a is not None else None for a in args]
        for i in (0, 1, 3, 8, 9):
            if args[i] is not None:
                grad[i] = zeros(slots, *args[i].shape)
        if gbo is None:
            gbo = zeros(h, bs, n, length)
        if gco is None:
            gco = zeros(h, bs, n, length)
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        _fold_backward[bs, h, triton.cdiv(n // 2, kw['BK'])](*[a if a is not None else dummy for a in args], prefix, gbo.contiguous(), gco.contiguous(), glast.contiguous() if glast is not None else dummy, *[grad[i] if grad[i] is not None else dummy for i in order], **kw, HAS_GL=glast is not None, SLOTS=slots, num_warps=4)
        for i in (0, 1, 3, 8, 9):
            if grad[i] is not None:
                grad[i] = grad[i].sum(0) if slots > 1 else grad[i][0]
        if args[12] is not None:
            grad[12] = grad[2].sum((0, 2))
        grad[13] = None
        return (None, *[g.to(a.dtype) if g is not None and ctx.needs_input_grad[i + 1] else None for i, (a, g) in enumerate(zip(args, grad))])

def half_fold(B, C, delta, theta, *, theta_h=None, theta_init=None, beta_b=None, beta_c=None, g_B=None, g_C=None, beta_gB=None, beta_gC=None, delta_bias=None, row_rot=None, theta_delta=False, delta_softplus=True, gate_act=1, pair_tile=4, time_tile=128):
    if B.ndim != 3 or B.shape != C.shape or B.shape[1] % 2:
        raise ValueError('half fold requires equal B/C[B,N,L] with even N')
    if theta.shape != (B.shape[0], B.shape[1] // 2, B.shape[2]):
        raise ValueError('theta must be (B,N/2,L)')
    bs, n, length = B.shape
    if min(bs, n, length) <= 0 or delta.ndim != 3 or delta.shape[0] != bs or (delta.shape[-1] != length):
        raise ValueError('nonempty maps and delta[B,H,L] are required')
    h = delta.shape[1]
    if h <= 0:
        raise ValueError('H must be positive')
    for name, t, shape in (('theta_h', theta_h, (h, n // 2)), ('theta_init', theta_init, (bs, h, n // 2)), ('g_B', g_B, B.shape), ('g_C', g_C, B.shape), ('delta_bias', delta_bias, (h,)), ('row_rot', row_rot, (n,))):
        if t is not None and t.shape != shape:
            raise ValueError(f'{name} must have shape {tuple(shape)}')
    for name, t in (('beta_b', beta_b), ('beta_c', beta_c), ('beta_gB', beta_gB), ('beta_gC', beta_gC)):
        if t is not None and (t.ndim != 2 or t.shape[1] != n or t.shape[0] <= 0 or h % t.shape[0]):
            raise ValueError(f'{name} must be (G,N) with G dividing H')
    if g_B is None and g_C is None:
        gate_act = 0
    if gate_act not in range(6):
        raise ValueError('gate_act must be one of 0,1,2,3,4,5')
    for name, value in (('pair_tile', pair_tile), ('time_tile', time_tile)):
        if value <= 0 or value & value - 1:
            raise ValueError(f'{name} must be a positive power of two')
    tensors = (B, C, delta, theta, theta_h, theta_init, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC, delta_bias, row_rot)
    if not B.is_cuda or any((t is not None and t.device != B.device for t in tensors)):
        raise ValueError('half fold requires all inputs on the same CUDA device')
    if any((t is not None and t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in tensors[:-1])):
        raise ValueError('half fold supports fp16, bf16 and fp32 inputs')
    opts = dict(theta_delta=theta_delta, delta_softplus=delta_softplus, gate_act=gate_act, pair_tile=pair_tile, time_tile=time_tile)
    return _HalfFold.apply(opts, B, C, delta, theta, theta_h, theta_init, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC, delta_bias, row_rot)
