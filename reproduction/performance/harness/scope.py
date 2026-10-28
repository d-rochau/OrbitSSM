
from __future__ import annotations
import contextlib
import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
AMP = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
NULL = contextlib.nullcontext
DY_SEED = 24301

class _Stop(Exception):
    pass

def _leaf(t, dtype=None):
    t = t.detach()
    if dtype is not None:
        t = t.to(dtype)
    return t.clone().requires_grad_(True)

def _cleaf(t, dtype=None):
    t = t.detach()
    if dtype is not None:
        t = t.to(dtype)
    return t.contiguous().clone().requires_grad_(True)

class Scope:

    def __init__(self, arm, call, leaves, canon, grads, amp, meta, ssm=None, keep=None):
        self.arm, self.call, self.leaves, self.canon, self._grads = (arm, call, leaves, canon, grads)
        self.amp, self.meta, self.ssm, self.keep = (amp, meta, ssm, keep)

    def grads(self):
        return self._grads()

    def leaf_list(self):
        return [t for t in self.leaves.values() if t is not None and t.requires_grad]

def canonical_dy(B, L, d, device='cuda'):
    g = torch.Generator(device='cpu').manual_seed(DY_SEED)
    return torch.randn(B, L, d, generator=g).to(device)

def make_pass(scope: Scope, pass_: str, dy_c=None):
    amp = scope.amp
    if pass_ == 'fwd':

        def run_fwd():
            with torch.no_grad(), amp():
                return scope.call()
        return run_fwd
    with torch.no_grad(), amp():
        y = scope.call()
    yc = scope.canon(y)
    if dy_c is None:
        dy_c = canonical_dy(*yc.shape, device=y.device)
    dy = torch.empty_like(y)
    v = scope.canon(dy)
    assert v.untyped_storage().data_ptr() == dy.untyped_storage().data_ptr(), 'canon() must be a view'
    v.copy_(dy_c)
    del y, yc, v
    leaves = scope.leaf_list()

    def run_fb():
        for t in leaves:
            t.grad = None
        with amp():
            y = scope.call()
        y.backward(dy)
        return y
    run_fb.dy = dy
    return run_fb

def _mixer(arm, d_model, N, *, kprec=None, seed=0):
    from benchmarks.bench_ssms import build_exact_pair
    import hmarms
    cfg = {'HMM:m1exact': 'm1exact', 'PM:m2exact': 'm2exact', 'PM:m3exact': 'm3exact'}.get(arm)
    if cfg is not None:
        over = {} if cfg == 'm1exact' else dict(precision=kprec or 'io')
        official, ours = build_exact_pair(cfg, d_model, N, 1, 1, seed=seed, **over)
        return (ours, official)
    _, _, mod = hmarms.build(arm, d_model, N, 1, 1, seed=seed, x_in=False)
    if kprec is not None and hasattr(mod.ssm, 'precision'):
        mod.ssm.precision = kprec
    return (mod, None)

def _hm_capture(mod, B, L, d_model, dtype, amp):
    from benchmarks.baselines import module_input
    from orbit_ssm.ssm.hybrid_mamba import matmul_channel_major, matmul_channel_major_tee
    ssm = mod.ssm
    assert ssm.proj_source == 'main_branch', ssm.proj_source
    cap = {}

    def rec(x_cf, h_init, sel_x, want_state, tok_raw=None, z_cf=None, complete_state=None):
        cap.update(x_cf=x_cf.detach(), sel_x=sel_x, tok_raw=tok_raw)
        raise _Stop()
    ssm._scan_cf = rec
    x0 = module_input(B, L, d_model, device='cuda', dtype=dtype, seed=0).detach()
    try:
        with torch.no_grad(), amp():
            mod(x0)
    except _Stop:
        pass
    finally:
        del ssm._scan_cf
    assert cap and cap['tok_raw'] is None, 'expected main_branch token maps'
    with torch.no_grad(), amp():
        x_cf, proj = matmul_channel_major_tee(ssm._sel_input_cf(cap['x_cf'], cap['sel_x']), ssm.tok_proj_weight())
        maps0 = ssm._sel_slice(proj)
        dt_lin = matmul_channel_major(ssm.dt_proj.weight, maps0['dt'].transpose(1, 2)) if ssm.dt_low is not None else None
    return (x_cf.detach(), proj.detach(), None if dt_lin is None else dt_lin.detach())

def _hm_scope(arm, mod, x_cf, proj, dt_lin, amp, d_model):
    import orbit_ssm.ssm.hybrid_mamba as hmmod
    from orbit_ssm.layers import regen as _regen
    ssm = mod.ssm
    for p in mod.parameters():
        p.requires_grad_(False)
    used = {}
    for n, p in ssm.named_parameters():
        if not (n.endswith(('proj.weight', 'proj.bias', 'dt_low.weight')) or n == 'D'):
            p.requires_grad_(True)
            used[n] = p
    ssm._d_skip_in_kernel = lambda x_cf: False
    assert ssm.bc_conv1d is None, 'HM arm with a B/C conv: feed its output (not needed for the shipped arms)'
    x_l, proj_l = (_leaf(x_cf), _leaf(proj))
    dt_l = None if dt_lin is None else _leaf(dt_lin)
    rg = hmmod.hm_held_regen() - {'dtlin'}
    rg_env = ','.join(sorted(rg)) if rg else 'off'
    W = ssm.dt_proj.weight

    def call():
        orig_mm, old = (hmmod.matmul_channel_major, os.environ.get('HM_HELD_REGEN'))

        def mm(w, t, *a, **k):
            return dt_l if w is W and dt_l is not None else orig_mm(w, t, *a, **k)
        hmmod.matmul_channel_major = mm
        os.environ['HM_HELD_REGEN'] = rg_env
        try:
            with _regen.scope(bool(rg)):
                y, _ = ssm._scan_cf(x_l, None, None, False, tok_raw=proj_l.transpose(1, 2), z_cf=None)
        finally:
            hmmod.matmul_channel_major = orig_mm
            if old is None:
                os.environ.pop('HM_HELD_REGEN', None)
            else:
                os.environ['HM_HELD_REGEN'] = old
        return y
    N = ssm.d_state
    leaves = OrderedDict(x=x_l, slab=proj_l)
    if dt_l is not None:
        leaves['dt'] = dt_l
    for n, p in used.items():
        leaves[f'param:{n}'] = p

    def grads():
        maps = ssm._sel_slice(proj_l.grad)
        bc = maps['bc']
        g = dict(x=x_l.grad.transpose(1, 2), B=bc[:, :N].transpose(1, 2), C=bc[:, N:2 * N].transpose(1, 2))
        if dt_l is not None:
            g['dt'] = dt_l.grad.transpose(1, 2)
        if 'dt_bias' in used:
            g['dt_bias'] = used['dt_bias'].grad
        if 'A_log' in used and used['A_log'].grad is not None:
            g['A_log'] = used['A_log'].grad.reshape(-1, N)
        g.update({f'param:{n}': p.grad for n, p in used.items()})
        return g
    meta = dict(H=ssm.H, P=ssm.P, N=N, d_inner=ssm.H * ssm.P, regen=rg_env, family='hybrid_mamba')
    return Scope(arm, call, leaves, lambda y: y.transpose(1, 2), grads, amp, meta, ssm=ssm, keep=mod)

def _official_m1_scope(x_cf, maps_bc, dt_lin, A_log, dt_bias, N, amp):
    from benchmarks import baselines as bl
    fn = bl.M1_SELECTIVE_SCAN.load()
    u, delta = (_cleaf(x_cf), _cleaf(dt_lin))
    Bt, Ct = (_cleaf(maps_bc[:, :N]), _cleaf(maps_bc[:, N:2 * N]))
    a_log = _leaf(A_log.reshape(-1, N), torch.float32)
    dbias = _leaf(dt_bias, torch.float32)

    def call():
        return fn(u, delta, -torch.exp(a_log), Bt, Ct, D=None, z=None, delta_bias=dbias, delta_softplus=True)
    leaves = OrderedDict(x=u, dt=delta, B=Bt, C=Ct, A_log=a_log, dt_bias=dbias)

    def grads():
        return dict(x=u.grad.transpose(1, 2), dt=delta.grad.transpose(1, 2), B=Bt.grad.transpose(1, 2), C=Ct.grad.transpose(1, 2), A_log=a_log.grad, dt_bias=dbias.grad)
    return Scope('official_m1', call, leaves, lambda y: y.transpose(1, 2), grads, amp, dict(H=x_cf.shape[1], P=1, N=N, d_inner=x_cf.shape[1], entry='selective_scan_fn', family='mamba1'))

def _ssd_capture(mix, B, L, d_model, dtype, amp):
    from benchmarks.baselines import module_input
    ssm = mix.ssm
    cap = {}

    def rec(x, **k):
        cap.update(x=x.detach(), k=k)
        raise _Stop()
    ssm.forward = rec
    x0 = module_input(B, L, d_model, device='cuda', dtype=dtype, seed=0).detach()
    try:
        with torch.no_grad(), amp():
            mix._forward_multinode(x0)
    except _Stop:
        pass
    finally:
        del ssm.forward
    assert cap, 'core forward was not reached'
    k = cap['k']
    assert k.get('tok_raw') is not None, 'expected the fused token-map slab'
    sel = k.get('sel_x')
    return (cap['x'], k['tok_raw'].detach(), sel.detach() if sel is not None else None)

def _slab_offsets(ssm):
    off, out = (0, {})
    for k, w in zip(ssm._proj_keys, ssm._proj_widths):
        out[k] = (off, off + w)
        off += w
    return out

def _ssd_scope(arm, mix, x, slab, sel, amp):
    from orbit_ssm.layers import regen as _regen
    ssm = mix.ssm
    offs = _slab_offsets(ssm)
    if ssm.bc_conv1d is not None:
        with torch.no_grad(), amp():
            bpost, cpost = ssm._bc_source(ssm.split_tok_raw(slab))
        slab = slab.clone()
        slab[..., offs['B'][0]:offs['B'][1]] = bpost.to(slab.dtype)
        slab[..., offs['C'][0]:offs['C'][1]] = cpost.to(slab.dtype)
        ssm.bc_conv1d = None
    ssm.d_skip = False
    for p in mix.parameters():
        p.requires_grad_(False)
    used = {}
    for n, p in ssm.named_parameters():
        if not (n.endswith('_proj.weight') or (n.endswith('_proj.bias') and n != 'dt_proj.bias') or n == 'D'):
            p.requires_grad_(True)
            used[n] = p
    x_l, slab_l = (_leaf(x), slab.detach().contiguous().clone().requires_grad_(True))
    sp = getattr(ssm, 'spec', None) or {}
    real = ssm.R == 1 and ssm.O == 1 and (not sp.get('rotation')) and (not any((b.get('complex') for b in sp.get('rows', ()))))
    rg = _regen.mode(real) != ''

    def call():
        with _regen.scope(rg):
            return ssm(x_l, gate_z=None, sel_x=sel, tok_raw=slab_l)
    leaves = OrderedDict(x=x_l, slab=slab_l)
    for n, p in used.items():
        leaves[f'param:{n}'] = p
    N, H = (ssm.N, ssm.H)

    def grads():
        gs = slab_l.grad
        g = dict(x=x_l.grad)
        for k, (a, b) in offs.items():
            g[{'a': 'a', 'theta': 'theta', 'trap': 'trap'}.get(k, k)] = gs[..., a:b]
        if 'dt_proj.bias' in used:
            g['dt_bias'] = used['dt_proj.bias'].grad
        if 'A_log' in used and used['A_log'].grad is not None:
            g['A_log'] = used['A_log'].grad.reshape(H)
        for nm, key in (('B_bias', 'B_bias'), ('C_bias', 'C_bias'), ('B_norm.weight', 'B_norm'), ('C_norm.weight', 'C_norm')):
            if nm in used and used[nm].grad is not None:
                g[key] = used[nm].grad
        g.update({f'param:{n}': p.grad for n, p in used.items()})
        return g
    meta = dict(H=H, P=ssm.P, N=N, R=ssm.R, O=ssm.O, d_inner=ssm.H * ssm.P, regen='on' if rg else 'off', bc_conv_fed=bool(ssm.bc_conv), family='ssd_mimo', precision=getattr(ssm, 'precision', None))
    sc = Scope(arm, call, leaves, lambda y: y, grads, amp, meta, ssm=ssm, keep=mix)
    sc.sel = sel
    return sc

def _official_m2_scope(x, slab, offs, official, N, amp, dtype):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    B_, L_, d = x.shape
    H, P = (official.nheads, official.headdim)
    xbc = torch.empty(B_, L_, d + 2 * N, device=x.device, dtype=dtype)
    with torch.no_grad():
        xbc[..., :d] = x
        xbc[..., d:d + N] = slab[..., offs['B'][0]:offs['B'][1]]
        xbc[..., d + N:] = slab[..., offs['C'][0]:offs['C'][1]]
    xbc.requires_grad_(True)
    dt = _cleaf(slab[..., offs['dt'][0]:offs['dt'][1]], dtype)
    a_log, dtb = (_leaf(official.A_log, torch.float32), _leaf(official.dt_bias, torch.float32))
    chunk = official.chunk_size

    def call():
        return mamba_chunk_scan_combined(xbc[..., :d].view(B_, L_, H, P), dt, -torch.exp(a_log), xbc[..., d:d + N].view(B_, L_, 1, N), xbc[..., d + N:].view(B_, L_, 1, N), chunk_size=chunk, D=None, z=None, dt_bias=dtb, dt_softplus=True)
    leaves = OrderedDict(xBC=xbc, dt=dt, A_log=a_log, dt_bias=dtb)

    def grads():
        g = xbc.grad
        return dict(x=g[..., :d], B=g[..., d:d + N], C=g[..., d + N:], dt=dt.grad, A_log=a_log.grad, dt_bias=dtb.grad)
    return Scope('official_m2', call, leaves, lambda y: y.reshape(B_, L_, H * P), grads, amp, dict(H=H, P=P, N=N, d_inner=d, chunk=chunk, entry='mamba_chunk_scan_combined', family='mamba2'))

def _official_m3_scope(arm, official, raw, amp, mimo=False):
    from einops import rearrange
    mod = official
    ins = OrderedDict(((k, _cleaf(raw[k])) for k in ('x', 'B', 'C', 'dt', 'a', 'trap', 'theta')))
    xl, Bl, Cl, ddt, dda, trl, angl = ins.values()
    if mimo:
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as fn
    else:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined as fn
    prm = OrderedDict(dt_bias=mod.dt_bias, B_bias=mod.B_bias, C_bias=mod.C_bias, B_norm=mod.B_norm.weight, C_norm=mod.C_norm.weight)
    if mimo:
        prm.update(mimo_x=mod.mimo_x, mimo_o=mod.mimo_o)

    def call():
        xx = rearrange(xl, 'b l (h p) -> b l h p', p=mod.headdim)
        Bb = rearrange(Bl, 'b l (r g n) -> b l r g n', r=mod.mimo_rank, g=mod.num_bc_heads)
        Cc = rearrange(Cl, 'b l (r g n) -> b l r g n', r=mod.mimo_rank, g=mod.num_bc_heads)
        tr = rearrange(trl, 'b l h -> b h l')
        _A = torch.clamp(-F.softplus(dda.to(torch.float32)), max=-mod.A_floor)
        DT = F.softplus(ddt + prm['dt_bias'])
        ADT = _A * DT
        DT = rearrange(DT, 'b l n -> b n l')
        ADT = rearrange(ADT, 'b l n -> b n l')
        ang = angl.unsqueeze(-2).expand(-1, -1, mod.nheads, -1).to(torch.float32)
        Bb = mod.B_norm(Bb)
        Cc = mod.C_norm(Cc)
        if mimo:
            return fn(Q=Cc, K=Bb, V=xx, ADT=ADT, DT=DT, Trap=tr, Q_bias=prm['C_bias'], K_bias=prm['B_bias'], MIMO_V=prm['mimo_x'], MIMO_Z=mod.mimo_z.detach(), MIMO_Out=prm['mimo_o'], Angles=ang, D=None, Z=None, chunk_size=mod.chunk_size, rotary_dim_divisor=mod.rotary_dim_divisor, dtype=xx.dtype, return_state=False)
        return fn(Q=Cc.squeeze(2), K=Bb.squeeze(2), V=xx, ADT=ADT, DT=DT, Trap=tr, Q_bias=prm['C_bias'].squeeze(1), K_bias=prm['B_bias'].squeeze(1), Angles=ang, D=None, Z=None, chunk_size=mod.chunk_size, Input_States=None, return_final_states=False)
    leaves = OrderedDict(list(ins.items()) + list(prm.items()))
    B_, L_ = xl.shape[:2]

    def grads():
        g = {k: t.grad for k, t in ins.items()}
        g.update(dt_bias=prm['dt_bias'].grad, B_bias=prm['B_bias'].grad, C_bias=prm['C_bias'].grad, B_norm=prm['B_norm'].grad, C_norm=prm['C_norm'].grad)
        return g
    return Scope(arm, call, leaves, lambda y: y.reshape(B_, L_, -1), grads, amp, dict(H=mod.nheads, P=mod.headdim, N=mod.d_state, R=mod.mimo_rank, d_inner=mod.d_inner, chunk=mod.chunk_size, entry='mamba3_mimo' if mimo else 'mamba3_siso_combined', family='mamba3'))
PAIRS = {'m1': ('HMM:m1exact', 'official_m1'), 'm2': ('PM:m2exact', 'official_m2'), 'm3': ('PM:m3exact', 'official_m3_siso')}
ARM_PAIR = {a: p for p, arms in PAIRS.items() for a in arms}
SCOPE_GATE = {'bf16': (0.03, 0.1), 'tf32': (0.01, 0.05), 'ieee': (0.0001, 0.001)}
NAMED = {'m1': ('x', 'B', 'C', 'dt', 'A_log', 'dt_bias'), 'm2': ('x', 'B', 'C', 'dt', 'A_log', 'dt_bias'), 'm3': ('x', 'B', 'C', 'dt', 'a', 'trap', 'theta', 'dt_bias', 'B_bias', 'C_bias', 'B_norm', 'C_norm')}

def _rel(a, b):
    a, b = (a.detach().float().reshape(-1), b.detach().float().reshape(-1))
    return float((a - b).norm() / b.norm().clamp_min(1e-30))

def _ssd_ref_fp64(so, dy_c):
    mod64, _ = _mixer(so.arm, so.meta['d_inner'] // 2, so.meta['N'], kprec='ieee')
    ssm64 = mod64.ssm.double()
    if so.meta.get('bc_conv_fed'):
        assert so.ssm.bc_conv1d is None
        ssm64.bc_conv1d = None
    ssm64.d_skip = False
    ssm64.backend = 'torch'
    for p in mod64.parameters():
        p.requires_grad_(False)
    x64 = so.leaves['x'].detach()[:1].double().clone().requires_grad_(True)
    slab64 = so.leaves['slab'].detach()[:1].double().clone().requires_grad_(True)
    sel = getattr(so, 'sel', None)
    with torch.autocast('cuda', enabled=False):
        y = ssm64(x64, gate_z=None, sel_x=None if sel is None else sel[:1].double(), tok_raw=slab64)
        y.backward(dy_c[:1].double().reshape(y.shape))
    offs = _slab_offsets(ssm64)
    g = {k: slab64.grad[..., a:b].clone() for k, (a, b) in offs.items() if k in ('theta', 'dt')}
    del ssm64, mod64, x64, slab64, y
    torch.cuda.empty_cache()
    return g

def sum_bound_err(per_tok, ref_tok):
    d = (per_tok.double() - ref_tok.double()).sum(0)
    scale = ref_tok.double().abs().sum(0)
    return float(d.norm() / scale.norm().clamp_min(1e-300))

def build_pair(pair, d_inner, N, B, L, precision='bf16', kprec=None):
    d_model = d_inner // 2
    amp = AMP if precision == 'bf16' else NULL
    ours_arm, off_arm = PAIRS[pair]
    mod, official = _mixer(ours_arm, d_model, N, kprec=kprec)
    if pair == 'm1':
        x_cf, proj, dt_lin = _hm_capture(mod, B, L, d_model, torch.float32, amp)
        ssm = mod.ssm
        bc = ssm._sel_slice(proj)['bc']
        off = _official_m1_scope(x_cf, bc, dt_lin, ssm.A_log.detach(), ssm.dt_bias.detach(), N, amp)
        ours = _hm_scope(ours_arm, mod, x_cf, proj, dt_lin, amp, d_model)
    else:
        x, slab, sel = _ssd_capture(mod, B, L, d_model, torch.float32, amp)
        ssm = mod.ssm
        offs = _slab_offsets(ssm)
        if pair == 'm2':
            with torch.no_grad(), amp():
                bpost, cpost = ssm._bc_source(ssm.split_tok_raw(slab))
            slab_post = slab.clone()
            slab_post[..., offs['B'][0]:offs['B'][1]] = bpost.to(slab.dtype)
            slab_post[..., offs['C'][0]:offs['C'][1]] = cpost.to(slab.dtype)
            off = _official_m2_scope(x, slab_post, offs, official, N, amp, slab.dtype)
        else:
            raw = dict(x=x, **{k: slab[..., a:b] for k, (a, b) in offs.items()})
            off = _official_m3_scope(off_arm, official, raw, amp)
        ours = _ssd_scope(ours_arm, mod, x, slab, sel, amp)
    del official
    return {ours_arm: ours, off_arm: off}

def gate_pair(scopes, pair, precision='bf16'):
    ours_arm, off_arm = PAIRS[pair]
    so, sf = (scopes[ours_arm], scopes[off_arm])
    res, out = ({}, {})
    dy_c = None
    for name, s in ((off_arm, sf), (ours_arm, so)):
        for t in s.leaf_list():
            t.grad = None
        with s.amp():
            y = s.call()
        yc = s.canon(y)
        if dy_c is None:
            dy_c = canonical_dy(*yc.shape, device=y.device)
        dy = torch.empty_like(y)
        s.canon(dy).copy_(dy_c)
        y.backward(dy)
        out[name] = dict(y=yc.detach().float().clone(), g={k: None if v is None else v.detach().float().clone() for k, v in s.grads().items() if not k.startswith('param:')})
        for t in s.leaf_list():
            t.grad = None
        del y, yc, dy
    tol_y, tol_g = SCOPE_GATE['bf16' if pair == 'm3' else precision]
    res['rel_y'] = _rel(out[ours_arm]['y'], out[off_arm]['y'])
    bad = [] if res['rel_y'] < tol_y else [f"y {res['rel_y']:.3e}>={tol_y}"]
    rels, missing, tols = ({}, [], {})
    for k in NAMED[pair]:
        a, b = (out[ours_arm]['g'].get(k), out[off_arm]['g'].get(k))
        if a is None or b is None:
            missing.append(f"{k}({('ours' if a is None else 'official')})")
            continue
        rels[k] = _rel(a, b)
        if pair == 'm3' and k == 'theta':
            continue
        if pair in ('m2', 'm3') and k == 'dt_bias':
            continue
        tol = tol_g
        tols[k] = tol
        if not rels[k] < tol:
            bad.append(f'd{k} {rels[k]:.3e}>={tol:.3g}')
    res.update(rel_grad=rels, missing_grad=missing, tol_y=tol_y, tol_grad=tols, leaves=list(NAMED[pair]), weights='copied', dy='shared canonical (fixed generator)')
    if pair in ('m2', 'm3'):
        refs = _ssd_ref_fp64(so, dy_c)
        Bn = out[off_arm]['y'].shape[0]
        tok = lambda t: t.reshape(Bn, -1, t.shape[-1])[0]
        e_ours = sum_bound_err(tok(out[ours_arm]['g']['dt']), refs['dt'][0])
        e_off = sum_bound_err(tok(out[off_arm]['g']['dt']), refs['dt'][0])
        res['dt_bias_sum_bound'] = dict(ours=e_ours, official=e_off, bound_ours=tol_g, official_status='recorded, not gated', rule='|sum_t (g_t - g_t^fp64)| / sum_t |g_t^fp64| per head (L2 over heads), batch element 0; dt_bias grad = sum_t dL/d(dt pre-activation)_t exactly')
        if not e_ours < tol_g:
            bad.append(f'ddt_bias(ours vs fp64, sum bound) {e_ours:.3e}>={tol_g}')
    if pair == 'm3':
        ref = refs['theta'].reshape(-1)
        b0 = lambda t: t.reshape(out[off_arm]['y'].shape[0], -1)[0]
        r_ours, r_off = (_rel(b0(out[ours_arm]['g']['theta']), ref), _rel(b0(out[off_arm]['g']['theta']), ref))
        res['theta_vs_fp64'] = dict(ours=r_ours, official=r_off, bound_ours=tol_g, official_status='recorded, not gated', reference='our scope core, float64, chunked torch route, batch element 0, same values + dY')
        if not r_ours < tol_g:
            bad.append(f'dtheta(ours vs fp64) {r_ours:.3e}>={tol_g}')
    if missing:
        bad.append('no gradient: ' + ','.join(missing))
    if bad:
        raise AssertionError(f'scope pair {pair} gate failed: ' + '; '.join(bad))
    return res
SSD_ARMS = {f'SSD-R{r}O{r}-{k}': f"ssd_mimo_r{r}o{r}_{('complex' if k == 'cx' else 'real')}" for r in (1, 2, 4) for k in ('re', 'cx')}
HM_ARMS = ('HM-0', 'HM-75', 'HM-100', 'HM-0@P32', 'HM-75@P32', 'HM-100@P32')

def build_solo(arm, d_inner, N, B, L, precision='bf16', kprec=None):
    d_model = d_inner // 2
    amp = AMP if precision == 'bf16' else NULL
    if arm == 'official_m3_mimo':
        from mamba_ssm.modules.mamba3 import Mamba3
        from benchmarks.baselines import module_input
        torch.manual_seed(0)
        mod = Mamba3(d_model=d_model, d_state=N, headdim=64, expand=2, is_mimo=True, mimo_rank=4, chunk_size=16, device='cuda', dtype=torch.float32)
        x0 = module_input(B, L, d_model, device='cuda', seed=0).detach()
        with torch.no_grad(), amp():
            z, x, Bp, Cp, dd_dt, dd_A, trap, angles = torch.split(mod.in_proj(x0), [mod.d_inner, mod.d_inner, mod.d_state * mod.num_bc_heads * mod.mimo_rank, mod.d_state * mod.num_bc_heads * mod.mimo_rank, mod.nheads, mod.nheads, mod.nheads, mod.num_rope_angles], dim=-1)
        raw = dict(x=x, B=Bp, C=Cp, dt=dd_dt, a=dd_A, trap=trap, theta=angles)
        s = _official_m3_scope(arm, mod, raw, amp, mimo=True)
        s.keep = mod
        return s
    mod, _ = _mixer(arm, d_model, N, kprec=kprec)
    if arm in HM_ARMS:
        x_cf, proj, dt_lin = _hm_capture(mod, B, L, d_model, torch.float32, amp)
        return _hm_scope(arm, mod, x_cf, proj, dt_lin, amp, d_model)
    x, slab, sel = _ssd_capture(mod, B, L, d_model, torch.float32, amp)
    return _ssd_scope(arm, mod, x, slab, sel, amp)

def check_solo_grads(s: Scope):
    for t in s.leaf_list():
        t.grad = None
    with s.amp():
        y = s.call()
    s.canon(y)
    y.float().square().mean().backward()
    have = {k: t.grad is not None for k, t in s.leaves.items()}
    for t in s.leaf_list():
        t.grad = None
    need = [k for k in s.leaves if not k.startswith('param:')]
    miss = [k for k in need if not have[k]]
    if miss:
        raise AssertionError(f'{s.arm}: no gradient for scope inputs {miss}')
    return dict(grads=f'{sum(have.values())}/{len(have)}', params_without_grad=sorted((k for k, v in have.items() if not v)))
SCOPE_ARMS = tuple((a for arms in PAIRS.values() for a in arms)) + ('official_m3_mimo',) + HM_ARMS + tuple(SSD_ARMS)
_PAIR_CACHE: dict = {}

def build_scope(arm, d_inner, N, B, L, precision='bf16', kprec=None, keep_other=True):
    pair = ARM_PAIR.get(arm)
    if pair is None:
        from benchmarks.measure import precision_ctx
        with precision_ctx(precision):
            s = build_solo(arm, d_inner, N, B, L, precision, kprec)
            return (s, dict(kind='solo', **check_solo_grads(s)))
    key = (pair, d_inner, N, B, L, precision, kprec)
    ent = _PAIR_CACHE.get(key)
    if ent is None or arm not in ent[0]:
        from benchmarks.measure import precision_ctx
        with precision_ctx(precision):
            scopes = build_pair(pair, d_inner, N, B, L, precision, kprec)
            gate = gate_pair(scopes, pair, precision)
        ent = _PAIR_CACHE[key] = (scopes, gate)
    scopes, gate = ent
    s = scopes.pop(arm)
    if not scopes or not keep_other:
        _PAIR_CACHE.pop(key, None)
    return (s, dict(kind='pair', pair=pair, **gate))
