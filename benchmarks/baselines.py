
from __future__ import annotations
import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Adapter:
    name: str
    module: str
    attr: str
    what: str

    def available(self) -> tuple[bool, str]:
        try:
            mod = importlib.import_module(self.module)
        except Exception as exc:
            return (False, f'import {self.module}: {type(exc).__name__}: {exc}')
        if not hasattr(mod, self.attr):
            return (False, f'{self.module} has no attribute {self.attr!r}')
        return (True, 'ok')

    def load(self):
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f'{self.name} unavailable: {why}')
        return getattr(importlib.import_module(self.module), self.attr)

    @property
    def entry_point(self) -> str:
        return f'{self.module}.{self.attr}'
M1_SELECTIVE_SCAN = Adapter('m1_selective_scan', 'mamba_ssm.ops.selective_scan_interface', 'selective_scan_fn', 'Mamba-1 reference selective scan (CUDA)')
M2_CHUNK_SCAN = Adapter('m2_chunk_scan', 'mamba_ssm.ops.triton.ssd_combined', 'mamba_chunk_scan_combined', 'Mamba-2 SSD chunked scan (Triton)')
M3_SISO = Adapter('m3_siso', 'mamba_ssm.ops.triton.mamba3.mamba3_siso_combined', 'mamba3_siso_combined', 'Mamba-3 SISO combined (Triton)')
M3_MIMO = Adapter('m3_mimo', 'mamba_ssm.ops.tilelang.mamba3.mamba3_mimo', 'mamba3_mimo', 'Mamba-3 MIMO (TileLang)')
MAMBA1_LAYER = Adapter('mamba1_layer', 'mamba_ssm.modules.mamba_simple', 'Mamba', 'official Mamba-1 mixer')
MAMBA2_LAYER = Adapter('mamba2_layer', 'mamba_ssm', 'Mamba2', 'official Mamba-2 mixer')
MAMBA3_LAYER = Adapter('mamba3_layer', 'mamba_ssm', 'Mamba3', 'official Mamba-3 mixer')
KERNEL_ADAPTERS = (M1_SELECTIVE_SCAN, M2_CHUNK_SCAN, M3_SISO, M3_MIMO)
MODULE_ADAPTERS = (MAMBA1_LAYER, MAMBA2_LAYER, MAMBA3_LAYER)
ALL_ADAPTERS = KERNEL_ADAPTERS + MODULE_ADAPTERS

def availability_report() -> list[dict]:
    out = []
    for ad in ALL_ADAPTERS:
        ok, why = ad.available()
        out.append(dict(name=ad.name, entry_point=ad.entry_point, available=ok, reason=why, what=ad.what))
    return out

def _rng(seed: int=0) -> torch.Generator:
    return torch.Generator(device='cpu').manual_seed(seed)

def _r(g, *shape, scale=1.0, device='cuda', dtype=torch.float32, requires_grad=False) -> torch.Tensor:
    t = (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)
    return t.contiguous().requires_grad_(requires_grad)

def build_m1(B: int, D: int, N: int, L: int, *, dtype=torch.float32, device='cuda', seed=0) -> tuple[Callable[[], Any], list]:
    fn = M1_SELECTIVE_SCAN.load()
    g = _rng(seed)
    u = _r(g, B, D, L, device=device, dtype=dtype, requires_grad=True)
    delta = _r(g, B, D, L, scale=0.5, device=device, dtype=dtype, requires_grad=True)
    a_log = _r(g, D, N, scale=0.5, device=device)
    A = (-a_log.exp()).detach().requires_grad_(True)
    Bt = _r(g, B, N, L, device=device, dtype=dtype, requires_grad=True)
    Ct = _r(g, B, N, L, device=device, dtype=dtype, requires_grad=True)

    def call():
        return fn(u, delta, A, Bt, Ct, delta_softplus=True)
    return (call, [u, delta, A, Bt, Ct])

def build_m2(B: int, H: int, P: int, N: int, L: int, *, chunk: int=64, ngroups: int=1, dtype=torch.bfloat16, device='cuda', seed=0) -> tuple[Callable[[], Any], list]:
    fn = M2_CHUNK_SCAN.load()
    g = _rng(seed)
    x = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)
    dt = (torch.rand(B, L, H, generator=g).to(device) * 0.1 + 0.01).requires_grad_(True)
    A = (-torch.rand(H, generator=g).to(device)).requires_grad_(True)
    Bm = _r(g, B, L, ngroups, N, device=device, dtype=dtype, requires_grad=True)
    Cm = _r(g, B, L, ngroups, N, device=device, dtype=dtype, requires_grad=True)

    def call():
        return fn(x, dt, A, Bm, Cm, chunk, dt_softplus=False, return_final_states=False)
    return (call, [x, dt, A, Bm, Cm])

def build_real_fair(B: int, H: int, P: int, N: int, L: int, *, arm: str, chunk: int, dtype=torch.bfloat16, device='cuda', seed=0, chunk_bwd: int | None=None) -> tuple[Callable[[], Any], list]:
    import torch.nn.functional as F
    d_in = H * P
    g = _rng(seed)
    rr = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    base = dict(x=rr(B, L, H, P, sc=0.5), dt_raw=rr(B, L, H) - 3.0, dt_bias=rr(H, sc=0.5), a_log=torch.log(torch.rand(H, generator=g) * 15 + 1), Bm=rr(B, L, N, sc=0.5), Cm=rr(B, L, N, sc=0.5), D=torch.ones(H) + rr(H, sc=0.1), z=rr(B, L, H, P, sc=0.7), w=torch.ones(d_in) + rr(d_in, sc=0.1))
    params = {k: base[k].to(device).requires_grad_() for k in ('dt_bias', 'a_log', 'D', 'w')}
    if arm == 'm2':
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
        fn = M2_CHUNK_SCAN.load()
        zxbcdt = torch.empty(B, L, 2 * d_in + 2 * N + H, device=device, dtype=dtype)
        zxbcdt[..., :d_in] = base['z'].reshape(B, L, d_in).to(device, dtype)
        zxbcdt[..., -H:] = base['dt_raw'].to(device, dtype)
        xbc = torch.empty(B, L, d_in + 2 * N, device=device, dtype=dtype)
        xbc[..., :d_in] = base['x'].reshape(B, L, d_in).to(device, dtype)
        xbc[..., d_in:d_in + N] = base['Bm'].to(device, dtype)
        xbc[..., d_in + N:] = base['Cm'].to(device, dtype)
        zxbcdt.requires_grad_(True)
        xbc.requires_grad_(True)

        def call():
            z = zxbcdt[..., :d_in]
            x = xbc[..., :d_in].view(B, L, H, P)
            Bt = xbc[..., d_in:d_in + N].view(B, L, 1, N)
            Ct = xbc[..., d_in + N:].view(B, L, 1, N)
            A = -torch.exp(params['a_log'])
            y = fn(x, zxbcdt[..., -H:], A, Bt, Ct, chunk, D=params['D'], z=None, dt_bias=params['dt_bias'], dt_softplus=True)
            return rmsnorm_fn(y.reshape(B, L, d_in), params['w'], None, z=z, eps=1e-05, norm_before_gate=False)
        return (call, [zxbcdt, xbc, *params.values()])
    if arm != 'ours':
        raise ValueError(f'arm must be m2|ours, got {arm!r}')
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    x = base['x'].permute(0, 2, 3, 1).contiguous().to(device, dtype).unsqueeze(3)
    width = d_in + H + 2 * N
    slab = torch.zeros(B, L, width, device=device, dtype=dtype)
    slab[..., :d_in] = base['z'].reshape(B, L, d_in).to(device, dtype)
    slab[..., d_in:d_in + H] = base['dt_raw'].to(device, dtype)
    slab[..., d_in + H:d_in + H + N] = base['Bm'].to(device, dtype)
    slab[..., d_in + H + N:] = base['Cm'].to(device, dtype)
    x.requires_grad_(True)
    slab.requires_grad_(True)
    from orbit_ssm.layers import tails as _tails
    cfg = _tails.TailCfg(N=N, R=1, O=1, H=H, L=L, bc_norm=False, b_norm=False, c_norm=False, shared_bias=False, fold_gb=False, fold_gc=False, a_dyn=False, a_per_sibling=False, has_dt_bias=True, has_a_bias=False, eps_b=0.0, eps_c=0.0, a_floor=0.0001, a_heavy_tail=False)

    def call():
        z = slab[..., :d_in]
        dt = slab[..., d_in:d_in + H]
        Bt = slab[..., d_in + H:d_in + H + N].unsqueeze(2).unsqueeze(-1)
        Ct = slab[..., d_in + H + N:].unsqueeze(2)
        _, _, delta, _ = _tails.token_tails(cfg, b_src=None, c_src=None, gb_src=None, gc_src=None, dt_src=dt, a_src=None, w_b=None, w_c=None, bias_b=None, bias_c=None, bias_dt=params['dt_bias'], bias_a=None, need_b=False, need_c=False, out_dtype_b=None, out_dtype_c=None)
        dsk = params['D'][:, None, None].expand(H, P, 1)
        y, _ = ssd_mimo_scan(x, delta, Bt, Ct, params['a_log'], chunk_size=chunk, chunk_size_bwd=chunk_bwd, y_token_major=True, precision='io', bc_per_head=True, gate_z=z, d_skip=dsk, out_norm_w=params['w'], out_norm_eps=1e-05)
        return y.permute(0, 3, 1, 2).reshape(B, L, d_in)
    return (call, [x, slab, *params.values()])

def real_fair_check(B, H, P, N, L, *, chunk_ours, chunk_m2=256, dtype=torch.bfloat16, device='cuda', tol_fwd=0.05, tol_grad=0.2) -> dict:
    d_in = H * P
    out = {}
    for arm, ch in (('m2', chunk_m2), ('ours', chunk_ours)):
        call, leaves = build_real_fair(B, H, P, N, L, arm=arm, chunk=ch, dtype=dtype, device=device)
        y = call()
        gy = torch.randn(y.shape, generator=_rng(1), dtype=torch.float32).to(device, y.dtype)
        y.backward(gy)
        if arm == 'm2':
            zxbcdt, xbc, dtb, a_log, D, w = leaves
            g = dict(dx=xbc.grad[..., :d_in], dB=xbc.grad[..., d_in:d_in + N], dC=xbc.grad[..., d_in + N:], dz=zxbcdt.grad[..., :d_in], ddt=zxbcdt.grad[..., -H:])
        else:
            x, slab, dtb, a_log, D, w = leaves
            g = dict(dx=x.grad.squeeze(3).permute(0, 3, 1, 2).reshape(B, L, d_in), dB=slab.grad[..., d_in + H:d_in + H + N], dC=slab.grad[..., d_in + H + N:], dz=slab.grad[..., :d_in], ddt=slab.grad[..., d_in:d_in + H])
        g.update(dD=D.grad, da=a_log.grad, dw=w.grad, ddtb=dtb.grad)
        out[arm] = dict(y=y.detach().float(), **{k: v.detach().float().clone() for k, v in g.items()})
        del call, leaves, y, g
        torch.cuda.empty_cache()
    err = {k: ((out['ours'][k] - out['m2'][k]).norm() / out['m2'][k].norm().clamp_min(1e-30)).item() for k in out['m2'] if k in out['ours']}
    for k, e in err.items():
        lim = tol_fwd if k == 'y' else tol_grad
        if not e < lim:
            raise AssertionError(f'real fair pair disagrees on {k}: rel {e:.3e} >= {lim}')
    return err

def build_m3_siso(B: int, H: int, P: int, N: int, L: int, *, chunk: int=64, angle_pairs: int | None=None, dtype=torch.float32, device='cuda', seed=0) -> tuple[Callable[[], Any], list]:
    fn = M3_SISO.load()
    g = _rng(seed)
    ka = angle_pairs if angle_pairs is not None else max(2, N // 4 & ~1)
    v = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)
    Bt = _r(g, B, L, 1, N, device=device, dtype=dtype, requires_grad=True)
    Ct = _r(g, B, L, 1, N, device=device, dtype=dtype, requires_grad=True)
    dt = F.softplus(_r(g, B, H, L, scale=0.5, device=device) - 3.0).detach().requires_grad_(True)
    ADT = (-dt.detach()).clone().requires_grad_(True)
    trap = _r(g, B, H, L, scale=0.5, device=device, requires_grad=True)
    Qb = _r(g, H, N, scale=0.1, device=device, requires_grad=True)
    Kb = _r(g, H, N, scale=0.1, device=device, requires_grad=True)
    angles = _r(g, B, L, H, ka, scale=0.3, device=device, requires_grad=True)

    def call():
        return fn(Q=Ct, K=Bt, V=v, ADT=ADT, DT=dt, Trap=trap, Q_bias=Qb, K_bias=Kb, Angles=angles, D=None, Z=None, chunk_size=chunk, Input_States=None)
    return (call, [v, Bt, Ct, dt, ADT, trap, Qb, Kb, angles])
_TRAP_OFF_LOGIT = 30.0

def _m3_eq_theta_h(H, N, device):
    g = torch.Generator(device='cpu').manual_seed(4242)
    return (torch.randn(H, N // 2, generator=g) * 0.3).to(device)

def m3_eq_angle_pairs(N: int, convention: str) -> int:
    if convention == 'm3':
        return max(2, N // 4 & ~1)
    if convention == 'half':
        return N // 2
    raise ValueError(f'angle convention must be m3|half, got {convention!r}')

def _rel_eq(a, b) -> float:
    a, b = (a.detach().double(), b.detach().double())
    return float((a - b).norm() / (b.norm() + 1e-30))

def _m3_eq_leaves(B, H, P, N, L, *, angle_pairs, trap_off, dtype, device='cuda', seed=0, bias='perhead'):
    g = _rng(seed)
    f32 = dict(device=device, dtype=torch.float32)
    dt = (torch.rand(B, H, L, generator=g).to(**f32) * 0.05 + 0.01).requires_grad_(True)
    adt = (-F.softplus(_r(g, B, H, L, device=device)) * dt.detach()).clone().requires_grad_(True)
    if trap_off:
        trap = torch.full((B, H, L), _TRAP_OFF_LOGIT, **f32).requires_grad_(True)
    else:
        trap = _r(g, B, H, L, device=device, requires_grad=True)
    return dict(K=_r(g, B, L, 1, N, scale=0.3, device=device, dtype=dtype, requires_grad=True), Q=_r(g, B, L, 1, N, scale=0.3, device=device, dtype=dtype, requires_grad=True), V=_r(g, B, L, H, P, scale=0.3, device=device, dtype=dtype, requires_grad=True), ADT=adt, DT=dt, Trap=trap, K_bias=_r(g, 1 if bias == 'shared' else H, N, scale=0.2, device=device, dtype=dtype, requires_grad=True), Q_bias=_r(g, 1 if bias == 'shared' else H, N, scale=0.2, device=device, dtype=dtype, requires_grad=True), Angles=_r(g, B, L, H, angle_pairs, scale=0.5, device=device, requires_grad=True))
_M3_EQ_ORDER = ('K', 'Q', 'V', 'ADT', 'DT', 'Trap', 'K_bias', 'Q_bias', 'Angles')

def _m3_eq_phase(t, N, cumsum=True):
    dt = t['DT']
    angle_pairs = t['Angles'].shape[-1]
    if not cumsum and os.environ.get('SSD_MIMO_M3_PHASE_FUSED', '1') == '1':
        from orbit_ssm.kernels.ssd_mimo.m3_epilogue import m3_phase_increments
        return m3_phase_increments(t['Angles'], dt, N // 2, sign=-1.0)
    incr = torch.tanh(t['Angles']) * math.pi * dt.permute(0, 2, 1).unsqueeze(-1)
    if cumsum:
        incr = torch.cumsum(incr, dim=1)
    return F.pad(-incr, (0, N // 2 - angle_pairs))

def _m3_eq_write(t, trap_off):
    dt, trap = (t['DT'], t['Trap'])
    if trap_off:
        return (dt, None)
    lam = torch.sigmoid(trap)
    nxt = F.pad((dt * (1 - lam))[..., 1:], (0, 1))
    return (lam * dt + nxt, nxt)

def _m3_eq_ours_fold(t, *, N, chunk, precision, trap_off, rot_where=None):
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    k_tok, q_tok, v = (t['K'], t['Q'], t['V'])
    adt = t['ADT']
    batch, length, heads, headdim = v.shape
    write, nxt = _m3_eq_write(t, trap_off)
    phase = _m3_eq_phase(t, N)
    k = k_tok[:, :, 0].unsqueeze(2) + t['K_bias']
    q = q_tok[:, :, 0].unsqueeze(2) + t['Q_bias']

    def fold(x, tail):
        return x.permute(0, 2, 1, 3).reshape(batch * heads, length, tail).contiguous()
    y, _ = ssd_mimo_scan(v.permute(0, 2, 3, 1).reshape(batch * heads, 1, headdim, 1, length).contiguous(), (-adt).reshape(batch * heads, 1, length).contiguous(), fold(k, N), fold(q, N), torch.zeros(1, device=v.device), chunk_size=chunk, precision=precision, delta_w=write.reshape(batch * heads, 1, length).contiguous(), theta=fold(phase, N // 2), rot_where=rot_where, bias_pre_rot=False)
    y = y.reshape(batch, heads, headdim, length).permute(0, 3, 1, 2)
    if not trap_off:
        dot = (k * q).sum(-1)
        y = y - (nxt.permute(0, 2, 1) * dot).unsqueeze(-1) * v
    return y

def _m3_eq_ours_head(t, *, N, chunk, precision, trap_off, rot_where=None):
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    k_tok, q_tok, v = (t['K'], t['Q'], t['V'])
    adt = t['ADT']
    batch, length, heads, headdim = v.shape
    phase_in_kernel = os.environ.get('SSD_MIMO_PHASE_INKERNEL', '1') == '1'
    if os.environ.get('SSD_MIMO_M3_EQ_PHASE_LAW', '0') == '1':
        from orbit_ssm.kernels.ssd_mimo.m3_epilogue import phase_law
        phase = phase_law(t['Angles'], t['DT'], None, heads, N // 2, sign=-1.0)
        phase_in_kernel = False
    else:
        phase = _m3_eq_phase(t, N, cumsum=not phase_in_kernel)
    kb, qb = (t['K_bias'], t['Q_bias'])
    beta = kb.float().unsqueeze(-1).contiguous()
    gamma = qb.float().unsqueeze(-1).contiguous()
    in_kernel = not trap_off and os.environ.get('SSD_MIMO_TRAP_INKERNEL', '1') == '1'
    lam = None
    if in_kernel:
        write, nxt = (t['DT'], None)
        lam = torch.sigmoid(t['Trap'])
    else:
        write, nxt = _m3_eq_write(t, trap_off)
    strided_v = os.environ.get('SSD_MIMO_STRIDED_V', '1') == '1'
    v_in = v.permute(0, 2, 3, 1).unsqueeze(3)
    if not strided_v:
        v_in = v_in.contiguous()
    y, _ = ssd_mimo_scan(v_in, (-adt).contiguous(), k_tok[:, :, 0].contiguous(), q_tok[:, :, 0].contiguous(), torch.zeros(heads, device=v.device), chunk_size=chunk, precision=precision, delta_w=write.contiguous(), trap_lam=lam.contiguous() if lam is not None else None, theta=phase, theta_increment=phase_in_kernel, beta=beta, gamma=gamma, bias_pre_rot=True, rot_where=rot_where, **dict(theta_h=_m3_eq_theta_h(heads, N, v.device)) if os.environ.get('SSD_MIMO_M3_EQ_THETA_H', '0') == '1' else {})
    y = y.permute(0, 3, 1, 2)
    if not trap_off and (not in_kernel):
        kk, qq = (k_tok[:, :, 0].float(), q_tok[:, :, 0].float())
        kbf, qbf = (kb.float(), qb.float())
        dot = (kk * qq).sum(-1)[:, :, None] + kk @ qbf.t() + qq @ kbf.t() + (kbf * qbf).sum(-1)
        y = y - (nxt.permute(0, 2, 1) * dot).unsqueeze(-1) * v
    return y
_M3_EQ_MAPS = {'fold': _m3_eq_ours_fold, 'head': _m3_eq_ours_head}

def m3_eq_bias_for(themap: str) -> str:
    return 'perhead'

def build_m3_eq(B: int, H: int, P: int, N: int, L: int, *, arm: str, chunk: int=64, angles: str='m3', trap_off: bool=True, dtype=torch.float32, precision: str='tf32', rot_where=None, device='cuda', seed=0, themap: str='head') -> tuple[Callable[[], Any], list]:
    if themap not in _M3_EQ_MAPS:
        raise ValueError(f'themap must be head|fold, got {themap!r}')
    ka = m3_eq_angle_pairs(N, angles)
    t = _m3_eq_leaves(B, H, P, N, L, angle_pairs=ka, trap_off=trap_off, dtype=dtype, device=device, seed=seed, bias=m3_eq_bias_for(themap))
    leaves = [t[k] for k in _M3_EQ_ORDER]
    if arm == 'm3':
        fn = M3_SISO.load()
        kb, qb = (t['K_bias'], t['Q_bias'])
        m3_chunk = 64 if chunk == 'auto' else chunk

        def call():
            return fn(Q=t['Q'], K=t['K'], V=t['V'], ADT=t['ADT'], DT=t['DT'], Trap=t['Trap'], Q_bias=qb, K_bias=kb, Angles=t['Angles'], D=None, Z=None, chunk_size=m3_chunk, Input_States=None)
    elif arm == 'ours':
        ours = _M3_EQ_MAPS[themap]
        from orbit_ssm.kernels.ssd_mimo import levers as _levers
        _decl = ka if 4 * ka == N else None

        def _chunk():
            if chunk != 'auto':
                return chunk
            from orbit_ssm.kernels.ssd_mimo import rot_route as _rr
            if _rr.explicit_siso() or _rr.explicit_rot_bwd() or _rr.explicit_chunk():
                return int(os.environ.get('SSD_MIMO_CHUNK') or 64)
            nb = torch.is_grad_enabled() and any((x.requires_grad for x in leaves))
            if dtype == torch.float32 and precision != 'bf16_ops':
                r = _rr.pick(B * H, L, device, nb, os.environ.get('SSD_MIMO_M3_EQ_THETA_H') == '1', dtype='fp32')
                if r is None:
                    from orbit_ssm.ssm.ssd_mimo import _scan_chunk
                    return _scan_chunk(1, P, 1, nb)
                return r[1]
            r = _rr.pick(B * H, L, device, nb, os.environ.get('SSD_MIMO_M3_EQ_THETA_H') == '1')
            return 64 if r is None else r[1]

        def call():
            with _levers.declare_active_pairs(_decl):
                return ours(t, N=N, chunk=_chunk(), precision=precision, trap_off=trap_off, rot_where=rot_where)
    else:
        raise ValueError(f'arm must be ours|m3, got {arm!r}')
    return (call, leaves)
M3_EQ_TOL = dict()
M3_EQ_TOL[torch.float32] = (0.04, 0.12)
M3_EQ_TOL[torch.bfloat16] = (0.12, 0.25)

def m3_eq_check(B, H, P, N, L, chunk=64, angles='m3', trap_off=True, dtype=torch.float32, precision='tf32', rot_where=None, device='cuda', seed=0, grads=True, themap='head') -> dict:
    call_o, leaves_o = build_m3_eq(B, H, P, N, L, arm='ours', chunk=chunk, angles=angles, trap_off=trap_off, dtype=dtype, precision=precision, rot_where=rot_where, device=device, seed=seed, themap=themap)
    call_m, leaves_m = build_m3_eq(B, H, P, N, L, arm='m3', chunk=chunk, angles=angles, trap_off=trap_off, dtype=dtype, precision=precision, device=device, seed=seed, themap=themap)
    y_o, y_m = (call_o(), call_m())
    errs = dict(y=_rel_eq(y_o, y_m))
    if grads:
        gen = torch.Generator(device=device).manual_seed(7)
        dy = torch.randn(y_m.shape, generator=gen, device=device, dtype=torch.float32)
        (y_o.float() * dy).sum().backward()
        (y_m.float() * dy).sum().backward()
        for name, a, b in zip(_M3_EQ_ORDER, leaves_o, leaves_m):
            if trap_off and name == 'Trap':
                continue
            if a.grad is None or b.grad is None:
                continue
            errs['d' + name] = _rel_eq(a.grad, b.grad)
    tol_fwd, tol_grad = M3_EQ_TOL[dtype]
    bad = sorted(((k, v) for k, v in errs.items() if v > (tol_fwd if k == 'y' else tol_grad)))
    if bad:
        detail = ', '.join(('%s %.2e' % kv for kv in bad))
        raise AssertionError('ssd_parity_m3_eq: the two arms disagree at B%d H%d P%d N%d L%d angles=%s trap_off=%s map=%s %s: %s' % (B, H, P, N, L, angles, trap_off, themap, str(dtype).split('.')[1], detail))
    return errs

def build_m3_mimo(B: int, H: int, P: int, N: int, L: int, *, R: int=4, chunk: int=16, rotary_divisor: int=4, dtype=torch.bfloat16, device='cuda', seed=0) -> tuple[Callable[[], Any], list]:
    fn = M3_MIMO.load()
    g = _rng(seed)
    Q = _r(g, B, L, R, 1, N, device=device, dtype=dtype, requires_grad=True)
    K = _r(g, B, L, R, 1, N, device=device, dtype=dtype, requires_grad=True)
    V = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)
    DT = F.softplus(_r(g, B, H, L, device=device) - 3.0).detach().requires_grad_(True)
    ADT = (-DT.detach() * math.log2(math.e)).clone().requires_grad_(True)
    Trap = (torch.rand(B, H, L, generator=g).to(device=device, dtype=dtype) * 0.5).requires_grad_(True)
    Qb = _r(g, H, R, N, device=device, requires_grad=True)
    Kb = _r(g, H, R, N, device=device, requires_grad=True)
    MV = _r(g, H, R, P, device=device, requires_grad=True)
    MZ = _r(g, H, R, P, scale=1.0 / R, device=device, requires_grad=True)
    MO = _r(g, H, R, P, scale=1.0 / R, device=device, requires_grad=True)
    Ang = _r(g, B, L, H, N // rotary_divisor, scale=0.3, device=device, requires_grad=True)
    Dsk = _r(g, H, device=device, requires_grad=True)
    Z = _r(g, B, L, H, P, device=device, dtype=dtype, requires_grad=True)

    def call():
        return fn(Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap, Q_bias=Qb, K_bias=Kb, MIMO_V=MV, MIMO_Z=MZ, MIMO_Out=MO, Angles=Ang, D=Dsk, Z=Z, chunk_size=chunk, rotary_dim_divisor=rotary_divisor, dtype=dtype, return_state=False)
    return (call, [Q, K, V, ADT, DT, Trap, Qb, Kb, MV, MZ, MO, Ang, Dsk, Z])

def build_mamba_layer(variant: str, d_model: int, *, d_state: int, L: int, B: int, d_conv: int=4, expand: int=2, headdim: int=64, ngroups: int=1, chunk_size: int | None=None, n_layers: int=1, dtype=torch.float32, device='cuda', seed=0) -> tuple[Callable[[], Any], list, torch.nn.Module]:
    if seed is not None:
        torch.manual_seed(seed)
    if variant == 'mamba1':
        cls = MAMBA1_LAYER.load()
        mod = cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True, layer_idx=0)
    elif variant == 'mamba2':
        cls = MAMBA2_LAYER.load()
        mod = cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, ngroups=ngroups, chunk_size=256 if chunk_size is None else chunk_size, layer_idx=0)
    elif variant == 'mamba3':
        cls = MAMBA3_LAYER.load()
        mod = cls(d_model, d_state=d_state, expand=expand, headdim=headdim, ngroups=ngroups, chunk_size=64 if chunk_size is None else chunk_size, layer_idx=0, n_layer=n_layers)
    else:
        raise ValueError(f'unknown official variant {variant!r}')
    mod = mod.to(device=device, dtype=dtype)
    x = module_input(B, L, d_model, device=device, dtype=dtype, seed=0 if seed is None else seed)

    def call():
        return mod(x)
    return (call, [x, *mod.parameters()], mod)

def module_input(B: int, L: int, d_model: int, *, device='cuda', dtype=torch.float32, seed: int=0) -> torch.Tensor:
    g = torch.Generator(device='cpu').manual_seed(int(seed) + 1)
    return torch.randn(B, L, d_model, generator=g).to(device=device, dtype=dtype).requires_grad_(True)
if __name__ == '__main__':
    for row in availability_report():
        mark = 'OK  ' if row['available'] else 'MISS'
        print(f"{mark} {row['name']:<18} {row['entry_point']}")
        if not row['available']:
            print(f"     {row['reason']}")
