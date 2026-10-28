
from __future__ import annotations
import argparse
import os
from typing import Any, Callable
import torch
from . import baselines as bl
from . import shapes as sh
from .harness import IO_DTYPE, KERNEL_PRECISION, Cell, cuda_buckets, group_key, main_for, make_pass, resolve_budget
GRIDS = ('smoke', 'mh_siso', 'mh_features', 'paper_c3', 'paper_c3_nscale', 'paper_c3_lscale', 'ssd_parity26', 'ssd_parity_m3', 'ssd_parity_m3_eq', 'ssd_long_l', 'ssd_mimo_gates', 'ssd_features', 'kernel_shared_set')
MH_ENTRY = 'orbit_ssm.kernels.hybrid_mamba_mh.functions.mh_scan'
SSD_ENTRY = 'orbit_ssm.kernels.ssd_mimo.autograd.ssd_mimo_scan'

def _mh_inputs(B, H, P, N, L, *, dtype, device='cuda', seed=0, rot=False, gates=False, biases=False, delta_w=None, law='A', G=None, theta_delta=False, theta_h=False):
    from orbit_ssm.kernels.hybrid_mamba_mh.layout import DYN_SOFTPLUS, GATE_SIGMOID, LAW_A_DYN, LAW_MU, fracs_to_rows, norm_fracs
    g = torch.Generator(device='cpu').manual_seed(seed)

    def r(*shape, scale=1.0, f32=False, grad=False):
        t = (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=torch.float32 if f32 else dtype).contiguous()
        return t.requires_grad_(grad)
    K = N // 2
    fracs = [{'frac': 1.0, 'decay': law, 'rot': rot}]
    frs = norm_fracs(fracs, N)
    row_law, row_rot = fracs_to_rows(frs, N, device=device)
    laws = {f.law for f in frs}
    x = r(B, H, P, L, grad=True)
    delta = r(B, H, L, scale=0.5, grad=True)
    B_t = r(B, N, L, grad=True)
    C_t = r(B, N, L, grad=True)
    A_log = r(H, P, N, scale=0.5, f32=True)
    if bool(row_rot.any()):
        A_log = A_log.reshape(H, P, K, 2)[..., :1].expand(H, P, K, 2)
        A_log = A_log.reshape(H, P, N).contiguous()
    A_log = A_log.detach().requires_grad_(True)
    kw: dict[str, Any] = dict(row_law=row_law, row_rot=row_rot, dyn_act=DYN_SOFTPLUS, dyn_eps=0.0001, gate_act=GATE_SIGMOID)
    Gg = H if G in (None, 'H') else int(G)
    if LAW_A_DYN in laws:
        kw['a_dyn'] = r(B, N, L, scale=0.5, f32=True, grad=True)
    if LAW_MU in laws:
        kw['mu'] = r(B, N, L, scale=0.3, f32=True, grad=True)
        kw['mu_gain'] = r(H, P, N, scale=0.3, f32=True, grad=True)
    if rot:
        kw['theta'] = r(B, K, L, scale=0.4, f32=True, grad=True)
        kw['theta_delta'] = theta_delta
        if theta_h:
            kw['theta_h'] = r(H, K, scale=0.2, f32=True, grad=True)
    elif theta_delta or theta_h:
        raise ValueError('theta_delta/theta_h require rot=True')
    if gates:
        kw['g_B'] = r(B, N, L, scale=0.7, grad=True)
        kw['g_C'] = r(B, N, L, scale=0.7, grad=True)
    if biases:
        kw['beta_a'] = r(Gg, N, scale=0.2, f32=True, grad=True)
        kw['beta_b'] = r(Gg, N, scale=0.3, f32=True, grad=True)
        kw['beta_c'] = r(Gg, N, scale=0.3, f32=True, grad=True)
        kw['delta_bias'] = r(H, scale=0.2, f32=True, grad=True)
        if gates:
            kw['beta_gB'] = r(Gg, N, scale=0.3, f32=True, grad=True)
            kw['beta_gC'] = r(Gg, N, scale=0.3, f32=True, grad=True)
    if delta_w == 'head':
        kw['delta_w'] = r(B, H, L, scale=0.5, grad=True)
    elif delta_w == 'chan':
        kw['delta_w'] = r(B, H, P, L, scale=0.5, grad=True)
    leaves = [x, delta, A_log, B_t, C_t] + [v for v in kw.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    return ((x, delta, A_log, B_t, C_t), kw, leaves)

def build_mh(B, H, P, N, L, *, dtype, endpoint=False, backend='cuda', **feat):
    from orbit_ssm.kernels.hybrid_mamba_mh import mh_scan
    from functools import partial
    from orbit_ssm.kernels.hybrid_mamba_mh.torch_scan import mh_torch_scan
    entries = {'cuda': mh_scan, 'torch': mh_torch_scan, 'sequential': partial(mh_torch_scan, sequential=True)}
    if backend not in entries:
        raise ValueError(f'unknown MH backend {backend!r}')
    entry = entries[backend]
    args, kw, leaves = _mh_inputs(B, H, P, N, L, dtype=dtype, **feat)
    if endpoint:
        lam = torch.full((B, H, L), 0.5, device=args[0].device, dtype=torch.float32, requires_grad=True)
        kw['trap_lam'] = lam
        leaves.append(lam)

    def call():
        return entry(*args, **kw)
    return (call, leaves)

def _mh_cell(grid, arm, B, H, P, N, L, precision, pass_, *, work, tags=None, **feat):
    dtype = IO_DTYPE[precision]
    return Cell(family='mh_scan', arm=arm, entry_point=MH_ENTRY, shape=dict(B=B, H=H, P=P, N=N, L=L, D=H * P), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), tags=dict(tags or {}), build=lambda: make_pass(*build_mh(B, H, P, N, L, dtype=dtype, **feat), pass_))

def _m1_cell(grid, B, D, N, L, precision, pass_, *, work):
    dtype = IO_DTYPE[precision]
    return Cell(family='mh_scan', arm='m1_selective_scan', entry_point=bl.M1_SELECTIVE_SCAN.entry_point, shape=dict(B=B, H=D, P=1, N=N, L=L, D=D), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), build=lambda: make_pass(*bl.build_m1(B, D, N, L, dtype=dtype), pass_))
_SSD_FEATURE_KW = {'none', 'theta', 'theta_h', 'rot_bias', 'beta', 'gamma', 'gate_z', 'delta_w', 'x0', 'bc_per_head', 'rotation'}

def build_rank_mixer(rot, R, d_inner, B, L, *, N=128, T=None, T_bwd=None, fp32=False, seed=0, device='cuda'):
    from experiments.lm.model.lm import NAMED_SSM
    from orbit_ssm.model import CausalLM
    spec = NAMED_SSM['ssd_mimo_r%do%d_%s' % (R, R, 'complex' if rot else 'real')]
    kw = dict(spec['kwargs'], heads=d_inner // 64)
    if T is not None:
        kw['chunk_size'] = int(T)
    if T_bwd is not None and T_bwd != T:
        kw['kernel_options'] = dict(kw.get('kernel_options', {}), chunk_size_bwd=int(T_bwd))
    if fp32:
        kw['precision'] = 'tf32'
    torch.manual_seed(seed)
    mod = CausalLM._build_mixer(d_model=d_inner // 2, ssm_type='ssd_mimo', ssm_kwargs=kw, expand=2.0, d_conv=4, d_state=N).to(device=device)
    x = torch.randn(B, L, d_inner // 2, device=device, requires_grad=True)

    def call():
        if fp32:
            return mod(x)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return mod(x)
    return (call, [x, *mod.parameters()])

def _ssd_auto_chunk(B, H, G, R, O, L, dtype, feature, nb, device):
    import os as _os
    from orbit_ssm.kernels.ssd_mimo import rot_route as _rr
    from orbit_ssm.ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
    if _os.environ.get('SSD_MIMO_CHUNK'):
        return (int(_os.environ['SSD_MIMO_CHUNK']), None)
    rotating = feature in ('theta', 'theta_h', 'rot_bias', 'rotation')
    if rotating and R == 1 and (O == 1):
        if _rr.explicit_siso() or _rr.explicit_rot_bwd() or _rr.explicit_chunk():
            return (64, None)
        r = _rr.pick(B * H, L, device, nb, feature in ('theta_h', 'rotation'), **{'dtype': 'fp32'} if dtype == torch.float32 else {})
        if r is not None:
            return (r[1], None)
        return (64, None) if dtype != torch.float32 else (_scan_chunk(1, G, 1, nb), None)
    kw = dict(real=not rotating, L=L, work=B * H * L, dtype=dtype)
    T = _scan_chunk(R, G, O, nb, **kw)
    return (T, _scan_chunk_bwd(R, G, O, T, **kw) if nb else None)

def build_ssd(B, H, G, R, O, N, L, *, chunk='auto', dtype, precision='tf32', feature='none', device='cuda', seed=0, backend='cuda', chunk_bwd=None):
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    from orbit_ssm.kernels.ssd_mimo.reference import unified_ssd_chunked, unified_ssd_quadratic
    entries = {'cuda': ssd_mimo_scan, 'torch': unified_ssd_chunked, 'quadratic': unified_ssd_quadratic}
    if backend not in entries:
        raise ValueError(f'SSD kernel backend must be one of {tuple(entries)}; sequential is an SSM route')
    entry = entries[backend]
    if feature not in _SSD_FEATURE_KW:
        raise ValueError(f'unknown ssd feature {feature!r}')
    g = torch.Generator(device='cpu').manual_seed(seed)

    def r(*shape, scale=1.0, f32=False, grad=False):
        t = (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=torch.float32 if f32 else dtype).contiguous()
        return t.requires_grad_(grad)
    bc_per_head = feature == 'bc_per_head'
    x = r(B, H, G, R, L, grad=True)
    delta = (torch.rand(B, H, L, generator=g).to(device) * 0.1 + 0.01).requires_grad_(True)
    a_log = (torch.randn(H, generator=g).to(device) - 1.0).requires_grad_(True)
    b_shape = (B, L, H, N) if bc_per_head else (B, L, N)
    c_shape = (B, L, H, N) if bc_per_head else (B, L, N)
    if R > 1:
        b_shape = b_shape + (R,)
    if O > 1:
        c_shape = c_shape + (O,)
    B_tok = r(*b_shape, grad=True)
    C_tok = r(*c_shape, grad=True)
    kw: dict[str, Any] = dict(chunk_size=chunk, precision=precision, bc_per_head=bc_per_head)
    if feature in ('theta', 'theta_h', 'rot_bias', 'rotation'):
        kw['bias_pre_rot'] = False
        kw['theta'] = r(B, L, N // 2, scale=0.4, f32=True, grad=True)
    if feature in ('theta_h', 'rotation'):
        kw['theta_h'] = r(H, N // 2, scale=0.2, f32=True, grad=True)
    if feature in ('rot_bias', 'rotation'):
        kw['rot_bias'] = r(H, N // 2, scale=0.2, f32=True, grad=True)
    if feature == 'beta':
        kw['beta'] = r(H, N, R, scale=0.2, f32=True, grad=True)
    if feature == 'gamma':
        kw['gamma'] = r(H, N, O, scale=0.2, f32=True, grad=True)
    if feature == 'gate_z':
        kw['gate_z'] = r(B, L, H * G, scale=0.5, grad=True)
    if feature == 'delta_w':
        kw['delta_w'] = r(B, H, L, scale=0.5, f32=True, grad=True)
    if feature == 'x0':
        kw['x0'] = r(B, H, G, N, scale=0.2, f32=True, grad=True)
    args = (x, delta, B_tok, C_tok, a_log)
    leaves = [t for t in (*args, *kw.values()) if isinstance(t, torch.Tensor) and t.requires_grad]
    if chunk_bwd is not None and chunk != 'auto' and (backend == 'cuda'):
        kw['chunk_size_bwd'] = int(chunk_bwd)
    if chunk == 'auto':
        if backend == 'cuda':
            kw['chunk_size'], tb = _ssd_auto_chunk(B, H, G, R, O, L, dtype, feature, torch.is_grad_enabled(), device)
            if tb is not None:
                kw['chunk_size_bwd'] = tb
        else:
            kw['chunk_size'] = 64
    if backend != 'cuda':
        kw.pop('precision')
    if backend == 'quadratic':
        kw.pop('chunk_size')

    def call():
        return entry(*args, **kw)
    return (call, leaves)

def _ssd_cell(grid, B, H, G, R, O, N, L, precision, pass_, *, work, chunk=64, feature='none', arm='ours', tags=None):
    dtype = IO_DTYPE[precision]
    kprec = KERNEL_PRECISION[precision]
    pinned = os.environ.get('PARITY_PIN_CHUNK') == '1'
    tags = dict(tags or {}, chunk_mode='pinned' if pinned else 'auto', chunk_ours='auto' if not pinned else chunk, chunk_official_m2=M2_DEFAULT_CHUNK if not pinned else chunk)
    if not pinned:
        chunk = 'auto'
    return Cell(family='ssd_mimo', arm=arm, entry_point=SSD_ENTRY, shape=dict(B=B, H=H, G=G, R=R, O=O, N=N, L=L, chunk=chunk, d_inner=H * G * R), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), tags=dict(tags or {}, feature=feature), build=lambda: make_pass(*build_ssd(B, H, G, R, O, N, L, chunk=chunk, dtype=dtype, precision=kprec, feature=feature), pass_))
M2_DEFAULT_CHUNK = 256
M3_DEFAULT_CHUNK = 64

def _m2_cell(grid, B, H, P, N, L, precision, pass_, *, work, chunk=64, tags=None):
    dtype = IO_DTYPE[precision]
    pinned = os.environ.get('PARITY_PIN_CHUNK') == '1'
    if not pinned:
        chunk = M2_DEFAULT_CHUNK
    tags = dict(tags or {}, chunk_mode='pinned' if pinned else 'auto', chunk_ours='pinned' if pinned else 'auto', chunk_official_m2=chunk)
    return Cell(family='ssd_mimo', arm='m2_chunk_scan', entry_point=bl.M2_CHUNK_SCAN.entry_point, shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk, d_inner=H * P), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), tags=tags, build=lambda: make_pass(*bl.build_m2(B, H, P, N, L, chunk=chunk, dtype=dtype), pass_))

def _m3_siso_cell(grid, B, H, P, N, L, precision, pass_, *, work, chunk=64, tags=None):
    dtype = IO_DTYPE[precision]
    pinned = os.environ.get('PARITY_PIN_CHUNK') == '1'
    if not pinned:
        chunk = M3_DEFAULT_CHUNK
    tags = dict(tags or {}, chunk_mode='pinned' if pinned else 'auto', chunk_official_m3=chunk)
    return Cell(family='ssd_mimo', arm='m3_siso', entry_point=bl.M3_SISO.entry_point, shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk, d_inner=H * P), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), tags=tags, build=lambda: make_pass(*bl.build_m3_siso(B, H, P, N, L, chunk=chunk, dtype=dtype), pass_))

def _m3_mimo_cell(grid, B, H, P, N, L, precision, pass_, *, work, R=4, chunk=16):
    dtype = IO_DTYPE[precision]
    return Cell(family='ssd_mimo', arm='m3_mimo', entry_point=bl.M3_MIMO.entry_point, shape=dict(B=B, H=H, G=P, R=R, O=R, N=N, L=L, chunk=chunk, d_inner=H * P), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), tags=dict(engine='tilelang'), build=lambda: make_pass(*bl.build_m3_mimo(B, H, P, N, L, R=R, chunk=chunk, dtype=dtype), pass_))

def grid_smoke() -> list[Cell]:
    g = 'smoke'
    cells: list[Cell] = []
    m = sh.SMOKE_MH
    for prec in ('tf32', 'bf16'):
        for p in ('fwd', 'fwd_bwd'):
            mhw, ssdw = ('mh', 'ssd')
            cells.append(_mh_cell(g, 'ours_p1', m['B'], m['D'], 1, m['N'], m['L'], prec, p, work=mhw))
            cells.append(_m1_cell(g, m['B'], m['D'], m['N'], m['L'], prec, p, work=mhw))
            w = sh.SMOKE_MH_MIMO
            cells.append(_mh_cell(g, f"ours_p{w['P']}", w['B'], w['H'], w['P'], w['N'], w['L'], prec, p, work=mhw))
            s = sh.SMOKE_SSD
            cells.append(_ssd_cell(g, s['B'], s['H'], s['G'], s['R'], s['O'], s['N'], s['L'], prec, p, chunk=s['T'], work=ssdw))
            cells.append(_m2_cell(g, s['B'], s['H'], s['G'], s['N'], s['L'], prec, p, chunk=s['T'], work=ssdw))
            cells.append(_m3_siso_cell(g, s['B'], s['H'], s['G'], s['N'], s['L'], prec, p, chunk=s['T'], work=ssdw))
            q = sh.SMOKE_SSD_MIMO
            cells.append(_ssd_cell(g, q['B'], q['H'], q['G'], q['R'], q['O'], q['N'], q['L'], prec, p, chunk=q['T'], work=ssdw))
    s = sh.SMOKE_SSD
    cells.append(_m3_mimo_cell(g, s['B'], s['H'], 64, 128, s['L'], 'bf16', 'fwd', R=4, chunk=16, work='ssd'))
    return cells

def grid_mh_siso() -> list[Cell]:
    g = 'mh_siso'
    cells: list[Cell] = []
    for s in sh.MH_SISO_SHAPES:
        w = f"B{s['B']}D{s['D']}N{s['N']}L{s['L']}"
        for prec in ('ieee', 'tf32', 'bf16'):
            for p in ('fwd', 'fwd_bwd'):
                cells.append(_mh_cell(g, 'ours_p1', s['B'], s['D'], 1, s['N'], s['L'], prec, p, work=w))
                cells.append(_m1_cell(g, s['B'], s['D'], s['N'], s['L'], prec, p, work=w))
    return cells

def grid_mh_features() -> list[Cell]:
    g = 'mh_features'
    cells: list[Cell] = []
    for s in sh.MH_CONTRACT_SHAPES:
        D = s['H'] * s['P']
        w = f"B{s['B']}D{D}N{s['N']}L{s['L']}"
        for hh, pp, suffix in ((s['H'], s['P'], ''), (D, 1, '_p1')):
            for name, feat in sh.MH_FEATURES:
                for prec in ('tf32', 'bf16'):
                    for p in ('fwd', 'fwd_bwd'):
                        kw = dict(feat)
                        if kw.get('G') == 'H':
                            kw['G'] = 'H'
                        cells.append(_mh_cell(g, f'feat:{name}{suffix}', s['B'], hh, pp, s['N'], s['L'], prec, p, work=w, tags=dict(feature=name, P=pp), **kw))
            for law in sh.MH_LAWS:
                for p in ('fwd', 'fwd_bwd'):
                    cells.append(_mh_cell(g, f'law:{law}{suffix}', s['B'], hh, pp, s['N'], s['L'], 'tf32', p, work=w, tags=dict(law=law, P=pp), law=law))
    return cells

def grid_paper_c3() -> list[Cell]:
    g = 'paper_c3'
    cells: list[Cell] = []
    n = sh.PAPER_C3_N
    for s in sh.PAPER_C3_SHAPES:
        B, D, L = (s['B'], s['D'], s['L'])
        w = f'B{B}D{D}L{L}N{n}'
        for prec in ('ieee', 'bf16'):
            for p in ('fwd', 'fwd_bwd'):
                cells.append(_m1_cell(g, B, D, n, L, prec, p, work=w))
                for P in sh.PAPER_C3_HEAD_WIDTHS:
                    if D % P:
                        continue
                    cells.append(_mh_cell(g, f'ours_p{P}', B, D // P, P, n, L, prec, p, work=w, tags=dict(gates=False)))
                    cells.append(_mh_cell(g, f'ours_gated_p{P}', B, D // P, P, n, L, prec, p, work=w, tags=dict(gates=True), gates=True))
    return cells

def grid_paper_c3_nscale() -> list[Cell]:
    g = 'paper_c3_nscale'
    cfg = sh.PAPER_C3_NSWEEP
    cells: list[Cell] = []
    for n in cfg['N']:
        w = f"B{cfg['B']}D{cfg['D']}L{cfg['L']}N{n}"
        for prec in ('ieee', 'bf16'):
            for p in ('fwd', 'fwd_bwd'):
                cells.append(_m1_cell(g, cfg['B'], cfg['D'], n, cfg['L'], prec, p, work=w))
                for P in cfg['head_widths']:
                    cells.append(_mh_cell(g, f'ours_p{P}', cfg['B'], cfg['D'] // P, P, n, cfg['L'], prec, p, work=w))
    return cells

def grid_paper_c3_lscale() -> list[Cell]:
    g = 'paper_c3_lscale'
    cfg = dict(sh.PAPER_C3_LSWEEP)
    if os.environ.get('BENCH_LSCALE_D'):
        cfg['D'] = int(os.environ['BENCH_LSCALE_D'])
    cells: list[Cell] = []
    for L in cfg['L']:
        w = f"B{cfg['B']}D{cfg['D']}N{cfg['N']}L{L}"
        for prec in ('ieee', 'bf16'):
            cells.append(_m1_cell(g, cfg['B'], cfg['D'], cfg['N'], L, prec, 'fwd_bwd', work=w))
            for P in (1, 4):
                cells.append(_mh_cell(g, f'ours_p{P}', cfg['B'], cfg['D'] // P, P, cfg['N'], L, prec, 'fwd_bwd', work=w))
    return cells

def grid_ssd_parity26() -> list[Cell]:
    g = 'ssd_parity26'
    t = sh.SSD_PARITY_T
    cells: list[Cell] = []
    for c in sh.SSD_PARITY26:
        for n in sh.SSD_PARITY_N_VALUES:
            tags = dict(sweep=c['sweep'])
            w = f"{c['sweep']}|B{c['B']}H{c['H']}G{c['G']}L{c['L']}|N{n}"
            for prec in ('ieee', 'bf16'):
                for p in ('fwd', 'fwd_bwd'):
                    cells.append(_ssd_cell(g, c['B'], c['H'], c['G'], 1, 1, n, c['L'], prec, p, chunk=t, work=w, tags=tags))
                    cells.append(_m2_cell(g, c['B'], c['H'], c['G'], n, c['L'], prec, p, chunk=t, work=w, tags=tags))
    return cells

def grid_ssd_parity_m3() -> list[Cell]:
    g = 'ssd_parity_m3'
    t = sh.SSD_PARITY_T
    cells: list[Cell] = []
    for c in sh.SSD_PARITY26:
        for n in sh.SSD_PARITY_N_VALUES:
            tags = dict(sweep=c['sweep'])
            w = f"{c['sweep']}|B{c['B']}H{c['H']}G{c['G']}L{c['L']}|N{n}"
            for prec in ('ieee', 'bf16'):
                for p in ('fwd', 'fwd_bwd'):
                    cells.append(_ssd_cell(g, c['B'], c['H'], c['G'], 1, 1, n, c['L'], prec, p, chunk=t, work=w, feature='rotation', tags=tags))
                    cells.append(_m3_siso_cell(g, c['B'], c['H'], c['G'], n, c['L'], prec, p, chunk=t, work=w, tags=tags))
    return cells
_M3_EQ_VARIANTS = (('m3', False, 'a4_trapon'), ('half', False, 'a2_trapon'), ('m3', True, 'a4_trapoff'), ('half', True, 'a2_trapoff'))

def _m3_eq_cell(grid, B, H, P, N, L, precision, pass_, arm, angles, trap_off, variant, work, chunk=64, tags=None, themap='head'):
    dtype = IO_DTYPE[precision]
    kprec = KERNEL_PRECISION[precision]
    ka = bl.m3_eq_angle_pairs(N, angles)
    entry = SSD_ENTRY if arm == 'ours' else bl.M3_SISO.entry_point
    pinned = os.environ.get('PARITY_PIN_CHUNK', '0') == '1'
    if arm == 'ours' and (not pinned):
        chunk = 'auto'
    tags = dict(tags or {})
    tags['chunk_mode'] = 'pinned' if pinned or arm != 'ours' else 'auto'
    if arm == 'ours':
        tags['chunk_arm'] = chunk

    def build():
        bl.m3_eq_check(B, H, P, N, L, chunk=chunk, angles=angles, trap_off=trap_off, dtype=dtype, precision=kprec, grads=pass_ == 'fwd_bwd', themap=themap)
        return make_pass(*bl.build_m3_eq(B, H, P, N, L, arm=arm, chunk=chunk, angles=angles, trap_off=trap_off, dtype=dtype, precision=kprec, themap=themap), pass_)
    return Cell(family='ssd_mimo', arm='ours_eq' if arm == 'ours' else 'm3_siso_eq', entry_point=entry, shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk, d_inner=H * P), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), tags=dict(tags, variant=variant, angle_pairs=ka, trap_off=trap_off, feature='m3_equivalent', themap=themap, bias=bl.m3_eq_bias_for(themap)), build=build)

def grid_ssd_parity_m3_eq() -> list[Cell]:
    g = 'ssd_parity_m3_eq'
    t = sh.SSD_PARITY_T
    themap = os.environ.get('K10_EQ_MAP', 'head')
    want = os.environ.get('K9_EQ_VARIANTS', '')
    keep = [v.strip() for v in want.split(',') if v.strip()]
    variants = [v for v in _M3_EQ_VARIANTS if not keep or v[2] in keep]
    if keep and (not variants):
        raise ValueError('K9_EQ_VARIANTS=%r matched none of %s' % (want, [v[2] for v in _M3_EQ_VARIANTS]))
    cells: list[Cell] = []
    for c in sh.SSD_PARITY26:
        for n in sh.SSD_PARITY_N_VALUES:
            for angles, trap_off, variant in variants:
                tags = dict(sweep=c['sweep'])
                w = variant + '|' + c['sweep'] + '|B' + str(c['B']) + 'H' + str(c['H']) + 'G' + str(c['G']) + 'L' + str(c['L']) + '|N' + str(n)
                for prec in ('tf32', 'bf16'):
                    for p in ('fwd', 'fwd_bwd'):
                        for arm in ('ours', 'm3'):
                            cells.append(_m3_eq_cell(g, c['B'], c['H'], c['G'], n, c['L'], prec, p, arm, angles, trap_off, variant, w, chunk=t, tags=tags, themap=themap))
    return cells

def grid_ssd_long_l() -> list[Cell]:
    g = 'ssd_long_l'
    n, t = (sh.SSD_PARITY_N, sh.SSD_PARITY_T)
    cells: list[Cell] = []
    for c in sh.SSD_LONG_L:
        w = f"B{c['B']}H{c['H']}G{c['G']}L{c['L']}"
        for p in ('fwd', 'fwd_bwd'):
            cells.append(_ssd_cell(g, c['B'], c['H'], c['G'], 1, 1, n, c['L'], 'bf16', p, chunk=t, work=w, tags=dict(sweep='long_l')))
            cells.append(_m2_cell(g, c['B'], c['H'], c['G'], n, c['L'], 'bf16', p, chunk=t, work=w, tags=dict(sweep='long_l')))
    return cells

def grid_ssd_mimo_gates() -> list[Cell]:
    g = 'ssd_mimo_gates'
    cells: list[Cell] = []

    def work_of(c) -> str:
        return f"B{c['B']}L{c['L']}d{c['d_inner']}N{c['N']}"
    for c in sh.MIMO_GATES:
        H = c['d_inner'] // (c['G'] * c['R'])
        if H < 1:
            continue
        for prec in ('tf32', 'bf16'):
            for p in ('fwd', 'fwd_bwd'):
                for chunk in sh.SSD_CHUNKS:
                    cells.append(_ssd_cell(g, c['B'], H, c['G'], c['R'], c['O'], c['N'], c['L'], prec, p, chunk=chunk, work=work_of(c), tags=dict(gate='mimo')))
    c = sh.SSD_R4O4_GATE
    H = c['d_inner'] // (c['G'] * c['R'])
    for prec in ('ieee', 'tf32', 'bf16'):
        for p in ('fwd', 'fwd_bwd'):
            cells.append(_ssd_cell(g, c['B'], H, c['G'], c['R'], c['O'], c['N'], c['L'], prec, p, chunk=64, work=work_of(c), tags=dict(gate='r4o4_18ms')))
            if prec == 'ieee':
                cells.append(_ssd_cell(g, c['B'], c['d_inner'] // c['G'], c['G'], 1, 1, c['N'], c['L'], prec, p, chunk=64, work=work_of(c), tags=dict(gate='mimo')))
    return cells

def grid_ssd_features() -> list[Cell]:
    g = 'ssd_features'
    base = sh.SSD_FEATURE_BASE
    w = f"B{base['B']}H{base['H']}G{base['G']}R{base['R']}O{base['O']}N{base['N']}L{base['L']}"
    cells: list[Cell] = []
    for feat in sh.SSD_FEATURES:
        for prec in ('tf32', 'bf16'):
            for p in ('fwd', 'fwd_bwd'):
                cells.append(_ssd_cell(g, base['B'], base['H'], base['G'], base['R'], base['O'], base['N'], base['L'], prec, p, chunk=base['T'], feature=feat, work=w, tags=dict(feature=feat)))
    return cells

def _real_fair_cell(grid, B, H, P, N, L, pass_, arm, chunk, work, tags=None, *, chunk_ours, chunk_m2=256, chunk_bwd=None):
    ours = arm == 'ours'

    def build():
        bl.real_fair_check(B, H, P, N, L, chunk_ours=chunk_ours, chunk_m2=chunk_m2)
        return make_pass(*bl.build_real_fair(B, H, P, N, L, arm=arm, chunk=chunk, chunk_bwd=chunk_bwd if ours else None), pass_)
    return Cell(family='ssd_mimo', arm='ours_fair' if ours else 'm2_fair', entry_point=SSD_ENTRY if ours else bl.M2_CHUNK_SCAN.entry_point, shape=dict(B=B, H=H, G=P, R=1, O=1, N=N, L=L, chunk=chunk, d_inner=H * P), precision='bf16', pass_=pass_, grid=grid, group=group_key(grid, work, 'bf16', pass_), tags=dict(tags or {}, feature='m2_block', chunk_arm=chunk, features='dt_bias+softplus+D+gated_rmsnorm'), build=build)

def grid_kernel_shared_set() -> list[Cell]:
    from orbit_ssm.kernels import shape_sets as ss
    g = 'kernel_shared_set'
    n, t = (sh.SSD_PARITY_N128, sh.SSD_PARITY_T)
    cells: list[Cell] = []
    for tag, (B, d_inner, L) in ss.SET.items():
        H, P = (d_inner // ss.P, ss.P)
        w = f'{tag}|B{B}H{H}G{P}L{L}|N{n}'
        tags = dict(sweep=tag, shape=tag)
        for p in ('fwd', 'fwd_bwd'):
            from orbit_ssm.ssm.ssd_mimo import _scan_chunk, _scan_chunk_bwd
            t_ours = _scan_chunk(1, P, 1, needs_bwd=p == 'fwd_bwd', real=True)
            t_bwd = _scan_chunk_bwd(1, P, 1, t_ours, real=True) if p == 'fwd_bwd' else None
            rt = dict(tags, kind='real', chunk_bwd=t_bwd or t_ours)
            cells.append(_real_fair_cell(g, B, H, P, n, L, p, 'ours', t_ours, w + '|real', tags=rt, chunk_ours=t_ours, chunk_bwd=t_bwd))
            cells.append(_real_fair_cell(g, B, H, P, n, L, p, 'm2', 256, w + '|real', tags=rt, chunk_ours=t_ours))
            for arm in ('ours', 'm3'):
                cells.append(_m3_eq_cell(g, B, H, P, n, L, 'bf16', p, arm, 'm3', False, 'a4_trapon', w + '|rot', chunk=t, tags=dict(tags, kind='rot', chunk_arm=t)))
    return cells
_GRID_FNS: dict[str, Callable[[], list[Cell]]] = {'kernel_shared_set': grid_kernel_shared_set, 'smoke': grid_smoke, 'mh_siso': grid_mh_siso, 'mh_features': grid_mh_features, 'paper_c3': grid_paper_c3, 'paper_c3_nscale': grid_paper_c3_nscale, 'paper_c3_lscale': grid_paper_c3_lscale, 'ssd_parity26': grid_ssd_parity26, 'ssd_parity_m3': grid_ssd_parity_m3, 'ssd_parity_m3_eq': grid_ssd_parity_m3_eq, 'ssd_long_l': grid_ssd_long_l, 'ssd_mimo_gates': grid_ssd_mimo_gates, 'ssd_features': grid_ssd_features}

def build_grid(name: str, args: argparse.Namespace) -> list[Cell]:
    return _GRID_FNS[name]()

def _paper_mh_features(c: Cell) -> bool:
    s = sh.MH_CONTRACT_SHAPES_PAPER[0]
    return (c.shape.get('B'), c.shape.get('D'), c.shape.get('N'), c.shape.get('L')) == (s['B'], s['H'] * s['P'], s['N'], s['L'])

def _paper_paper_c3(c: Cell) -> bool:
    if c.precision == 'ieee':
        return False
    s = sh.PAPER_C3_SHAPES_PAPER[0]
    return (c.shape.get('B'), c.shape.get('D'), c.shape.get('L')) == (s['B'], s['D'], s['L'])

def _paper_ssd_parity(c: Cell) -> bool:
    if c.precision == 'ieee':
        return False
    if c.shape.get('N') != sh.SSD_PARITY_N128:
        return False
    return (c.tags.get('sweep'), c.shape.get('B'), c.shape.get('H'), c.shape.get('G'), c.shape.get('L')) in sh.SSD_PARITY26_PAPER_KEYS

def _paper_ssd_parity_m3_eq(c: Cell) -> bool:
    if c.shape.get('N') != sh.SSD_PARITY_N128:
        return False
    return c.tags.get('variant') == 'a4_trapon' and (c.tags.get('sweep'), c.shape.get('B'), c.shape.get('H'), c.shape.get('G'), c.shape.get('L')) in sh.SSD_PARITY26_PAPER_KEYS
_SSD_LONG_L_PAPER_KEYS = frozenset(((c['B'], c['H'], c['G'], c['L']) for c in sh.SSD_LONG_L_PAPER))

def _paper_ssd_long_l(c: Cell) -> bool:
    return (c.shape.get('B'), c.shape.get('H'), c.shape.get('G'), c.shape.get('L')) in _SSD_LONG_L_PAPER_KEYS

def _paper_ssd_mimo_gates(c: Cell) -> bool:
    if c.precision == 'ieee':
        return False
    gate = c.tags.get('gate')
    if gate == 'r4o4_18ms':
        return True
    if gate == 'mimo':
        return c.shape.get('G') == 16 and c.shape.get('N') == 128
    return False

def _paper_accept_all(c: Cell) -> bool:
    return True

def _paper_accept_all_no_ieee(c: Cell) -> bool:
    return c.precision != 'ieee'
PAPER_SETS: dict[str, Callable[[Cell], bool]] = {'mh_siso': _paper_accept_all_no_ieee, 'mh_features': _paper_mh_features, 'paper_c3': _paper_paper_c3, 'paper_c3_nscale': _paper_accept_all_no_ieee, 'paper_c3_lscale': _paper_accept_all_no_ieee, 'ssd_parity26': _paper_ssd_parity, 'ssd_parity_m3': _paper_ssd_parity, 'ssd_parity_m3_eq': _paper_ssd_parity_m3_eq, 'ssd_long_l': _paper_ssd_long_l, 'ssd_mimo_gates': _paper_ssd_mimo_gates, 'ssd_features': _paper_accept_all, 'kernel_shared_set': _paper_accept_all}
_SELF_CUDA_GRIDS = frozenset({'ssd_parity26', 'ssd_parity_m3', 'ssd_parity_m3_eq'})

def _post_cell(args: argparse.Namespace):
    budget = resolve_budget(args)

    def hook(cell: Cell, closure):
        out: dict = {}
        if cell.grid == 'kernel_shared_set' and cell.arm in ('ours_fair', 'ours_eq'):
            from .route_record import served
            try:
                out.update(served(closure))
            except Exception as exc:
                out['route_error'] = f'{type(exc).__name__}: {exc}'
        if cell.grid in _SELF_CUDA_GRIDS and budget.cuda_self:
            try:
                out['ms_cuda_self'] = cuda_buckets(closure, iters=1)['total']
            except Exception as exc:
                out.update(ms_cuda_self=None, ms_cuda_self_error=f'{type(exc).__name__}: {exc}')
        return out or None
    return hook
if __name__ == '__main__':
    main_for(build_grid, GRIDS, 'benchmarks.bench_kernels', post_cell=_post_cell, paper_filter=PAPER_SETS.get)
