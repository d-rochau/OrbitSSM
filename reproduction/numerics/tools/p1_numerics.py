
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/src')

def build_m3(d_model, N, headdim, seed, ours_kw=None):
    from mamba_ssm.modules.mamba3 import Mamba3
    from orbit_ssm.harness import presets
    from orbit_ssm.harness.compat import copy_official_mamba3_into_ssd_mimo_mixer, mamba3_reference_a_act
    from orbit_ssm.model import SSDMimoMixer
    torch.manual_seed(seed)
    off = Mamba3(d_model=d_model, d_state=N, expand=2, headdim=headdim, ngroups=1).cuda().float()
    kw = presets.mamba3_kwargs(d_model, d_state=N, headdim=headdim, expand=2, backend='cuda', a_act=mamba3_reference_a_act(off), **ours_kw or {})
    ours = SSDMimoMixer(**kw).cuda().float()
    assert copy_official_mamba3_into_ssd_mimo_mixer(off, ours) == []
    return (off, ours)

def grads_m3(off, ours):
    di, N, H = (ours.d_inner, ours.ssm.N, ours.ssm.H)
    w = off.in_proj.weight.grad
    z, x = (w[:di], w[di:2 * di])
    o = 2 * di
    b, c = (w[o:o + N], w[o + N:o + 2 * N])
    o += 2 * N
    dt, a, trap = (w[o:o + H], w[o + H:o + 2 * H], w[o + 2 * H:o + 3 * H])
    ang = w[o + 3 * H:]
    ssm = ours.ssm
    return {'in_proj.weight': torch.cat([x, z], 0), 'ssm.B_proj.weight': b, 'ssm.C_proj.weight': c, 'ssm.dt_proj.weight': dt, 'ssm.dt_proj.bias': off.dt_bias.grad, 'ssm.a_dyn_proj.weight': a, 'ssm.trap_proj.weight': trap, 'ssm.theta_proj.weight': ang, 'ssm.D': off.D.grad.reshape_as(ssm.D), 'ssm.B_bias': off.B_bias.grad.transpose(1, 2).unsqueeze(1).reshape_as(ssm.B_bias), 'ssm.C_bias': off.C_bias.grad.transpose(1, 2).unsqueeze(1).reshape_as(ssm.C_bias), 'ssm.B_norm.weight': off.B_norm.weight.grad, 'ssm.C_norm.weight': off.C_norm.weight.grad, 'out_proj.weight': off.out_proj.weight.grad}

def build_m2(d_model, N, headdim, seed, ours_kw=None):
    from benchmarks.baselines import build_mamba_layer
    from orbit_ssm.harness import presets
    from orbit_ssm.harness.compat import copy_official_mamba2_into_ssd_mimo_mixer
    from orbit_ssm.model import SSDMimoMixer
    _, _, off = build_mamba_layer('mamba2', d_model, d_state=N, L=8, B=1, d_conv=4, expand=2, headdim=headdim, ngroups=1, chunk_size=256, dtype=torch.float32, device='cuda', seed=seed)
    kw = presets.mamba2_kwargs(d_model, d_state=N, headdim=headdim, expand=2, backend='cuda')
    kw['combined'] = True
    kw['ssm_kwargs'].update(ours_kw or {})
    kw.pop('backend', None)
    ours = SSDMimoMixer(**kw).cuda().float()
    assert copy_official_mamba2_into_ssd_mimo_mixer(off, ours) == []
    return (off, ours)

def grads_m2(off, ours):
    di, N, H = (ours.d_inner, ours.ssm.N, ours.ssm.H)
    w = off.in_proj.weight.grad
    z, x = (w[:di], w[di:2 * di])
    b, c = (w[2 * di:2 * di + N], w[2 * di + N:2 * di + 2 * N])
    dt = w[2 * di + 2 * N:2 * di + 2 * N + H]
    cw, cb = (off.conv1d.weight.grad, off.conv1d.bias.grad)
    ssm = ours.ssm
    return {'in_proj.weight': torch.cat([x, z], 0), 'conv1d.weight': cw[:di], 'conv1d.bias': cb[:di], 'ssm.bc_conv1d.conv.weight': cw[di:di + 2 * N], 'ssm.bc_conv1d.conv.bias': cb[di:di + 2 * N], 'ssm.B_proj.weight': b, 'ssm.C_proj.weight': c, 'ssm.dt_proj.weight': dt, 'ssm.dt_proj.bias': off.dt_bias.grad, 'ssm.A_log': off.A_log.grad.reshape_as(ssm.A_log), 'ssm.D': off.D.grad.reshape_as(ssm.D), 'norm.weight': off.norm.weight.grad, 'out_proj.weight': off.out_proj.weight.grad}

def build_m1(d_model, N, headdim, seed, ours_kw=None):
    from mamba_ssm.modules.mamba_simple import Mamba
    from orbit_ssm.harness import presets
    from orbit_ssm.model import HybridMambaMixer
    torch.manual_seed(seed)
    off = Mamba(d_model=d_model, d_state=N, d_conv=4, expand=2, dt_rank='auto', use_fast_path=True).cuda().float()
    kw = presets.mamba1_kwargs(d_model, d_state=N, dt_rank=off.dt_rank, backend='cuda', **ours_kw or {})
    ours = HybridMambaMixer(**kw).cuda().float()
    r = off.dt_rank
    with torch.no_grad():
        ours.in_proj.weight.copy_(off.in_proj.weight)
        ours.conv1d.weight.copy_(off.conv1d.weight)
        ours.conv1d.bias.copy_(off.conv1d.bias)
        ours.ssm.dt_low.weight.copy_(off.x_proj.weight[:r])
        ours.ssm.bc_proj.weight.copy_(off.x_proj.weight[r:])
        ours.ssm.dt_proj.weight.copy_(off.dt_proj.weight)
        ours.ssm.dt_bias.copy_(off.dt_proj.bias)
        ours.ssm.A_log.copy_(off.A_log.unsqueeze(1))
        ours.ssm.D.copy_(off.D)
        ours.out_proj.weight.copy_(off.out_proj.weight)
    return (off, ours)

def grads_m1(off, ours):
    r = off.dt_rank
    return {'in_proj.weight': off.in_proj.weight.grad, 'conv1d.weight': off.conv1d.weight.grad, 'conv1d.bias': off.conv1d.bias.grad, 'ssm.dt_low.weight': off.x_proj.weight.grad[:r], 'ssm.bc_proj.weight': off.x_proj.weight.grad[r:], 'ssm.dt_proj.weight': off.dt_proj.weight.grad, 'ssm.dt_bias': off.dt_proj.bias.grad, 'ssm.A_log': off.A_log.grad.reshape_as(ours.ssm.A_log), 'ssm.D': off.D.grad.reshape_as(ours.ssm.D), 'out_proj.weight': off.out_proj.weight.grad}
PAIRS = {'m1': (build_m1, grads_m1, 16, 'torch'), 'm2': (build_m2, grads_m2, 128, 'torch'), 'm3': (build_m3, grads_m3, 128, 'torch')}

def score(a, b):
    a, b = (a.double(), b.double())
    if not torch.isfinite(a).all():
        return dict(err=float('inf'), signed=float('nan'))
    d = a - b
    return dict(err=float(d.norm() / b.norm().clamp_min(1e-300)), signed=float((d * b).sum() / (b * b).sum().clamp_min(1e-300)))

def bound(dtype, depth):
    if dtype == 'bf16':
        u = 2.0 ** (-8)
        return min(max(u * math.sqrt(depth) * math.sqrt(2), 1.28 * u), 12.8 * u)
    u = 2.0 ** (-11)
    return max(u * math.sqrt(depth) * 4, 1e-06)

def run_cell(pair, dtype, seed, a, ref_backend):
    build, grads, N0, _ = PAIRS[pair]
    N = a.N or N0
    s = seed
    off, ours = build(a.d_model, N, a.headdim, s)
    arms = {'ours': ours}
    if dtype == 'fp32' and pair in ('m2', 'm3'):
        _, ours_ieee = build(a.d_model, N, a.headdim, s, {'precision': 'ieee'})
        arms['ours_ieee'] = ours_ieee
    bf16 = dtype == 'bf16'
    g = torch.Generator().manual_seed(s + 11)
    x = (torch.randn(a.B, a.L, a.d_model, generator=g) * 0.5).cuda()
    if bf16:
        x = x.to(torch.bfloat16).float()
    ref_m = copy.deepcopy(ours).double()
    ref_m.ssm.backend = ref_backend
    xo = x.double().requires_grad_(True)
    yo = ref_m(xo)
    w = torch.randn(yo.shape, generator=torch.Generator().manual_seed(s + 12)).to(yo.device, torch.float64)
    gr = torch.autograd.grad(yo, [xo, *ref_m.parameters()], w)
    ref = {'y': yo.detach(), 'x': gr[0]}
    ref.update({n: g_ for (n, _), g_ in zip(ref_m.named_parameters(), gr[1:])})
    del ref_m, yo, gr, xo
    torch.cuda.empty_cache()
    got = {}
    ac = torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16)
    for arm, mod in [('official', off)] + list(arms.items()):
        mod.zero_grad(set_to_none=True)
        xa = x.clone().requires_grad_(True)
        with ac:
            y = mod(xa)
        y.backward(w.to(y.dtype))
        torch.cuda.synchronize()
        d = {'y': y.detach().double(), 'x': xa.grad.double()}
        if arm == 'official':
            d.update({k: v.double() for k, v in grads(off, ours).items()})
        else:
            d.update({n: p.grad.double() for n, p in mod.named_parameters()})
        got[arm] = d
    assert set(got['official']) == set(ref), set(got['official']) ^ set(ref)
    route = {k: m.ssm.last_route() if hasattr(m.ssm, 'last_route') else None for k, m in arms.items()}
    cells = {}
    for name in ref:
        depth = 64 if name in ('y', 'x') else a.B * a.L
        c = {'bound': bound(dtype, depth), 'ref_norm': float(ref[name].norm())}
        c['off_ref'] = score(got['official'][name], ref[name])
        for arm in arms:
            c[f'{arm}_ref'] = score(got[arm][name], ref[name])
            c[f'{arm}_off'] = score(got[arm][name], got['official'][name])
        cells[name] = c
    return dict(pair=pair, dtype=dtype, seed=seed, B=a.B, L=a.L, d_model=a.d_model, N=N, headdim=a.headdim, ref_backend=ref_backend, route=route, cells=cells)

def validate_ref(a):
    out = []
    for pair in a.pairs.split(','):
        build, grads, N0, _ = PAIRS[pair]
        off, ours = build(a.d_model, a.N or N0, a.headdim, 7)
        x = (torch.randn(2, 256, a.d_model, device='cuda', generator=None) * 0.5).double()
        res = {}
        for be in ('torch', 'sequential'):
            m = copy.deepcopy(ours).double()
            m.ssm.backend = be
            xo = x.clone().requires_grad_(True)
            y = m(xo)
            w = torch.randn(y.shape, generator=torch.Generator().manual_seed(1)).to(y.device, torch.float64)
            gr = torch.autograd.grad(y, [xo, *m.parameters()], w)
            res[be] = {'y': y.detach(), 'x': gr[0], **{n: g_ for (n, _), g_ in zip(m.named_parameters(), gr[1:])}}
        r = {k: score(res['torch'][k], res['sequential'][k])['err'] for k in res['torch']}
        out.append(dict(pair=pair, validate_ref=r))
        print('VALIDATE_REF', pair, 'max rel err torch-vs-sequential fp64:', max(r.values()), flush=True)
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', default='m1,m2,m3')
    p.add_argument('--dtypes', default='bf16,fp32')
    p.add_argument('--B', type=int, default=6)
    p.add_argument('--L', type=int, default=2048)
    p.add_argument('--d-model', type=int, default=1024)
    p.add_argument('--N', type=int, default=0, help='0 = official default per pair (m1 16, m2/m3 128)')
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--seed0', type=int, default=3100)
    p.add_argument('--ref-backend', default='')
    p.add_argument('--json-out', default='')
    p.add_argument('--validate-ref', action='store_true')
    a = p.parse_args()
    import mamba_ssm
    dev = torch.cuda.get_device_properties(0)
    print('P1NUM_CONFIG', json.dumps(dict(gpu=dev.name, cc=list(torch.cuda.get_device_capability(0)), torch=torch.__version__, mamba_ssm=mamba_ssm.__version__, args=vars(a), env={k: v for k, v in os.environ.items() if k.startswith(('SSD_MIMO', 'TRITON_F32', 'MH_'))})), flush=True)
    out = open(a.json_out, 'a') if a.json_out else None
    if a.validate_ref:
        for r in validate_ref(a):
            if out:
                out.write(json.dumps(r) + '\n')
        return 0
    for pair in a.pairs.split(','):
        for dtype in a.dtypes.split(','):
            for d in range(a.seeds):
                t0 = time.time()
                try:
                    r = run_cell(pair, dtype, a.seed0 + 1000 * d, a, a.ref_backend or PAIRS[pair][3])
                except torch.OutOfMemoryError as e:
                    r = dict(pair=pair, dtype=dtype, seed=a.seed0 + 1000 * d, oom=str(e)[:200])
                r['seconds'] = round(time.time() - t0, 1)
                if out:
                    out.write(json.dumps(r) + '\n')
                    out.flush()
                yy = r.get('cells', {}).get('y')
                print(f"  done {pair} {dtype} seed {d} {r['seconds']}s route={r.get('route')} " + (f"y ours/off {yy['ours_off']['err']:.2e} off/ref {yy['off_ref']['err']:.2e} ours/ref {yy['ours_ref']['err']:.2e}" if yy else 'OOM'), flush=True)
                torch.cuda.empty_cache()
    return 0
if __name__ == '__main__':
    sys.exit(main())
