
import os
import sys
MOD = sys.argv.pop(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from benchmarks import harness
from benchmarks import route_record as _rr
from benchmarks import shapes as sh
from benchmarks.harness import Cell, group_key
from benchmarks.measure import KERNEL_PRECISION, amp_ctx
import hmarms
import scope as S
from orbit_ssm.kernels import shape_sets as ss
spec = os.environ.get('HMB_TAGS', 'all')
tags = set()
for t in filter(None, spec.split(',')):
    if t == 'main':
        tags |= {k for k in ss.SET if k.startswith('large_') or k in ss.LM_TAGS}
    elif t == 'scope':
        tags |= {k for k in ss.SET if k.startswith('large_')} | {'lm_b6_d2048_l2048'}
    elif t == 'rest':
        tags |= {k for k in ss.SET if k.startswith(('small_', 'medium_'))}
    elif t == 'all':
        tags |= set(ss.SET)
    else:
        tags.add(t)
PAIR = os.environ.get('FIN_PAIR') or None
COMPILE = os.environ.get('FIN_COMPILE', '0')
PREC = os.environ.get('FIN_PREC', 'bf16')
KPREC = os.environ.get('FIN_KPREC') or None
if KPREC:
    KERNEL_PRECISION[PREC] = KPREC
import hashlib
_here = os.path.dirname(os.path.abspath(__file__))
SCRIPT_SHA = hashlib.sha256(b''.join((open(os.path.join(_here, f), 'rb').read() for f in ('fin_run.py', 'scope.py', 'hmarms.py')))).hexdigest()[:16]
ARMF = set(filter(None, os.environ.get('FIN_ARMS', '').split(',')))
S_ARMS = dict(m1=('HMM:m1exact', 'official_mamba1'), m2=('SM-R1:m2exact', 'official_mamba2'), m3=('SM-R1:m3exact', 'official_mamba3'))
PAIR_N = dict(m1=sh.EXACT_D_STATE_M1, m2=sh.EXACT_D_STATE_M23, m3=sh.EXACT_D_STATE_M23)
TABLE2_SCOPE = tuple((a for a in S.SCOPE_ARMS if a != 'official_m3_mimo'))
OURS = set(hmarms.OURS_ARMS) | {'SM-R1:m2exact', 'SM-R1:m3exact'}

def _scope_cell(g, arm, tag, B, d, L, N, p, group, others):
    rt = {}
    other = [a for a in others if S.ARM_PAIR.get(a) and S.ARM_PAIR.get(a) == S.ARM_PAIR.get(arm) and (a != arm)]
    kprec = KPREC or KERNEL_PRECISION[PREC]

    def build():
        s, v = S.build_scope(arm, d, N, B, L, precision=PREC, kprec=kprec, keep_other=bool(other))
        rt['validation'] = v
        rt['geometry'] = {**dict(B=B, L=L, d_inner=d, d_model=d // 2, R=1, O=1), **s.meta}
        fn = S.make_pass(s, p)
        fn.scope = s
        return fn
    return Cell(family='scope', arm=arm, entry_point=f'scope.{arm}', shape=dict(B=B, L=L, d_inner=d, N=N, P=64 if 'P32' not in arm else 32), precision=PREC, pass_=p, grid=g, group=group, tags=dict(sweep=tag, shape=tag, kind='scope', pair=S.ARM_PAIR.get(arm)), build=build, runtime=rt)
if MOD == 'bench_kernels':
    from benchmarks import bench_kernels as b

    def kernel_pairs():
        g = 'kernel_pairs'
        cells = []
        for tag, (B, d, L) in ss.SET.items():
            for pair in (PAIR,) if PAIR else ('m1', 'm2', 'm3'):
                n = PAIR_N[pair]
                arms = S.PAIRS[pair]
                for p in ('fwd', 'fwd_bwd'):
                    grp = group_key(g, f'{tag}|B{B}D{d}L{L}|N{n}|{pair}', PREC, p)
                    cells += [_scope_cell(g, a, tag, B, d, L, n, p, grp, arms) for a in arms]
        return cells

    def scope_grid():
        g, n = ('kernel_scope_n128', 128)
        cells = []
        arms = [a for a in S.SCOPE_ARMS if a in ARMF] if ARMF else list(TABLE2_SCOPE)
        for tag, (B, d, L) in ss.SET.items():
            for p in ('fwd', 'fwd_bwd'):
                grp = group_key(g, f'{tag}|B{B}D{d}L{L}|N{n}', PREC, p)
                cells += [_scope_cell(g, a, tag, B, d, L, n, p, grp, arms) for a in arms]
        return cells

    def mimo_grid():
        g, n = ('kernel_scope_mimo', 128)
        cells = []
        for tag, (B, d, L) in ss.SET.items():
            for p in ('fwd', 'fwd_bwd'):
                grp = group_key(g, f'{tag}|B{B}D{d}L{L}|N{n}', PREC, p)
                cells += [_scope_cell(g, a, tag, B, d, L, n, p, grp, ()) for a in ('official_m3_mimo', 'official_m3_siso')]
        return cells
    for name, fn in (('kernel_pairs', kernel_pairs), ('kernel_scope_n128', scope_grid), ('kernel_scope_mimo', mimo_grid)):
        b._GRID_FNS[name] = fn
        b.GRIDS = tuple(b.GRIDS) + (name,)
        b.PAPER_SETS[name] = b._paper_accept_all
else:
    from benchmarks import bench_ssms as b
    if PREC != 'bf16':
        _e, _o = (b._exact_mixer_cell, b._official_cell)

        def _ex(*a, **k):
            a = list(a)
            assert a[7] == 'bf16'
            a[7] = PREC
            return _e(*a, **k)

        def _of(*a, **k):
            a = list(a)
            assert a[6] == 'bf16'
            a[6] = PREC
            return _o(*a, **k)
        b._exact_mixer_cell, b._official_cell = (_ex, _of)

    def m1_grid(graphed=False):
        g, n = ('ssm_shared_set_m1', sh.EXACT_D_STATE_M1)
        cells = []
        for tag, (B, di, L) in ss.SET.items():
            dm = di // 2
            w = f'{tag}:B{B}L{L}d{dm}N{n}'
            for p in ('fwd', 'fwd_bwd'):
                for graph in (False, True):
                    cells.append(b._exact_mixer_cell(g, 'HMM:m1exact', 'm1exact', dm, n, B, L, 'bf16', p, work=w, tags=dict(config='m1exact', shape=tag), graphed=graph))
                    cells.append(b._official_cell(g, 'mamba1', dm, n, B, L, 'bf16', p, work=w, tags=dict(config='official', shape=tag), graphed=graph))
        return cells
    COST_ARMS = ('HMM:m1exact', 'official_mamba1', 'HM-0', 'HM-75', 'HM-100', 'HM-0@P32', 'HM-75@P32', 'HM-100@P32', 'official_mamba2', 'official_mamba3', *S.SSD_ARMS)
    if 'official_mamba3_mimo' in ARMF:
        COST_ARMS = COST_ARMS + ('official_mamba3_mimo',)

    def cost_grid(graphed=False):
        g, n = ('ssm_cost_n128', 128)
        cells = []
        for tag, (B, di, L) in ss.SET.items():
            dm = di // 2
            w = f'{tag}:B{B}L{L}d{dm}N{n}'
            for p in ('fwd', 'fwd_bwd'):
                for graph in (False, True):
                    for arm in [a for a in COST_ARMS if not ARMF or a in ARMF]:
                        kw = dict(work=w, tags=dict(config=arm, shape=tag), graphed=graph)
                        if arm == 'HMM:m1exact':
                            cells.append(b._exact_mixer_cell(g, arm, 'm1exact', dm, n, B, L, 'bf16', p, **kw))
                            continue
                        if arm in hmarms.OFFICIAL and arm != 'official_mamba3_mimo':
                            cells.append(b._official_cell(g, hmarms.OFFICIAL[arm], dm, n, B, L, 'bf16', p, headdim=64, **kw))
                            continue
                        name = arm + ('@graph' if graph else '')

                        def build(arm=arm, dm=dm, B=B, L=L, p=p, graph=graph):
                            call, leaves, mod = hmarms.build(arm, dm, n, B, L)
                            b._PARAMS[f'{arm}/{dm}'] = sum((q.numel() for q in mod.parameters()))
                            return b._with_module(mod, leaves, amp_ctx('bf16'), p, b._graphed_closure(mod, leaves, p, 'bf16') if graph else b.make_pass(call, leaves, p, amp=amp_ctx('bf16')))
                        cells.append(Cell(family='ssm', arm=name, entry_point=f'hmarms.build({arm})', shape=dict(B=B, L=L, d_model=dm, d_inner=di, N=n, headdim=64), precision='bf16', pass_=p, grid=g, group=group_key(g, w, 'bf16', p), autocast=True, tags=dict(config=arm, shape=tag, mixer_level='full', **{'capture': 'cuda_graph'} if graph else {}), build=build))
        return cells
    b._GRID_FNS['ssm_shared_set_m1'] = m1_grid
    b._GRID_FNS['ssm_cost_n128'] = cost_grid
    b.GRIDS = tuple(b.GRIDS) + ('ssm_shared_set_m1', 'ssm_cost_n128')
    b.PAPER_SETS['ssm_shared_set_m1'] = lambda c: True
    b.PAPER_SETS['ssm_cost_n128'] = lambda c: True
_GF = os.environ.get('FIN_GROUPS_FILE')
ONLY_GROUPS = {line.rstrip('\n') for line in open(_GF) if line.strip()} if _GF else None

def keep(c):
    if c.tags.get('shape') not in tags:
        return False
    if ONLY_GROUPS is not None and c.resolved_group not in ONLY_GROUPS:
        return False
    base = c.arm.split('@')[0]
    if c.grid == 'ssm_shared_set':
        if PAIR and base not in S_ARMS[PAIR]:
            return False
        is_c = c.arm.endswith('@compile')
        return is_c if COMPILE == 'only' else is_c <= (COMPILE == '1')
    if c.grid == 'ssm_shared_set_m1' and PREC != 'bf16':
        return not c.arm.endswith('@compile')
    return True

def _geometry_of_module(mod):
    s = getattr(mod, 'ssm', None)
    if s is not None:
        return dict(d_model=getattr(mod, 'd_model', None), d_inner=getattr(mod, 'd_inner', None), H=getattr(s, 'H', None), P=getattr(s, 'P', None), N=getattr(s, 'N', getattr(s, 'd_state', None)), R=getattr(s, 'R', 1), O=getattr(s, 'O', 1))
    return dict(d_model=getattr(mod, 'd_model', None), d_inner=getattr(mod, 'd_inner', None), H=getattr(mod, 'nheads', getattr(mod, 'd_inner', None)), P=getattr(mod, 'headdim', 1), N=getattr(mod, 'd_state', None), R=getattr(mod, 'mimo_rank', 1), chunk=getattr(mod, 'chunk_size', None))

def _ours_route(ssm, closure):
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    if isinstance(ssm, SSDMimoSSM):
        r = _rr.served(closure)
        r['production_problem'] = _rr.production_problem(r)
        return r
    return _rr.served_mh(closure, ssm)
GEOM = ('B', 'L', 'd_model', 'd_inner', 'H', 'P', 'N', 'R', 'O')

def _required(prov, ours):
    miss = [k for k in ('geometry', 'route', 'chunk', 'calibration') if prov.get(k) in (None, {}, '')]
    g = prov.get('geometry') or {}
    miss += [f'geometry.{k}' for k in GEOM if g.get(k) is None]
    if prov.get('production_problem'):
        miss.append(f"production_problem: {prov['production_problem']}")
    if ours:
        rt = prov.get('route') or {}
        if 'mh' in rt:
            if not (rt['mh'].get('fwd_config') and (rt['mh'].get('bwd_config') or prov.get('pass_') == 'fwd')):
                miss.append('route.mh launch configs')
        elif not (rt.get('fwd') and rt.get('bwd')):
            miss.append('route.fwd/bwd')
    return miss
_pc = b._post_cell

def post_cell(args):
    inner = _pc(args)

    def f(cell, closure):
        out = dict((inner(cell, closure) if inner else None) or {})
        base = cell.arm.split('@')[0]
        ours = base in OURS or cell.arm in OURS
        prov = dict(pass_=cell.pass_)
        try:
            sc = getattr(closure, 'scope', None)
            if sc is not None:
                prov['geometry'] = dict(cell.runtime.get('geometry', {}))
                if ours:
                    r = _ours_route(sc.ssm, closure)
                    prov['route'], prov['calibration'] = (r.get('route'), r.get('calibration'))
                    rt = r.get('route') or {}
                    prov['chunk'] = dict(fwd=rt.get('chunk_used'), bwd=rt.get('chunk_bwd'), mode=rt.get('chunk_mode')) if 'mh' not in rt else dict(fwd_BLOCK_L=(rt['mh'].get('fwd_config') or {}).get('BLOCK_L'), bwd_BLOCK_L=(rt['mh'].get('bwd_config') or {}).get('BLOCK_L'))
                    if 'production_problem' in r:
                        prov['production_problem'] = r['production_problem']
                else:
                    prov['route'] = dict(entry=sc.meta.get('entry'))
                    prov['chunk'] = dict(official=sc.meta.get('chunk', 'n/a (selective scan)'))
                    prov['calibration'] = 'official defaults (mamba_ssm 2.3.2.post1)'
            else:
                mod = getattr(closure, 'module', None)
                prov['geometry'] = dict(B=cell.shape.get('B'), L=cell.shape.get('L'), **_geometry_of_module(mod)) if mod is not None else None
                if prov['geometry'] is not None:
                    gm = prov['geometry']
                    for k, v in (('N', cell.shape.get('N')), ('d_model', cell.shape.get('d_model')), ('d_inner', cell.shape.get('d_inner')), ('R', 1), ('O', 1)):
                        if gm.get(k) is None:
                            gm[k] = v
                if ours and mod is not None:
                    r = _ours_route(mod.ssm, closure.eager)
                    prov['route'], prov['calibration'] = (r.get('route'), r.get('calibration'))
                    rt = r.get('route') or {}
                    prov['chunk'] = dict(fwd=rt.get('chunk_used'), bwd=rt.get('chunk_bwd'), mode=rt.get('chunk_mode')) if 'mh' not in rt else dict(fwd_BLOCK_L=(rt['mh'].get('fwd_config') or {}).get('BLOCK_L'), bwd_BLOCK_L=(rt['mh'].get('bwd_config') or {}).get('BLOCK_L'))
                    if 'production_problem' in r:
                        prov['production_problem'] = r['production_problem']
                    prov['route_source'] = 'eager proxy (same module/input; a replay/compiled call runs no host dispatch)' if '@' in cell.arm else 'the timed closure'
                elif mod is not None:
                    prov['route'] = dict(entry=cell.entry_point)
                    prov['chunk'] = dict(official=getattr(mod, 'chunk_size', 'n/a (selective scan)'))
                    prov['calibration'] = 'official defaults (mamba_ssm 2.3.2.post1)'
        except Exception as exc:
            prov['error'] = f'{type(exc).__name__}: {exc}'[:400]
        miss = _required(prov, ours)
        if prov.get('error') or miss:
            out['error'] = f"PROVENANCE MISSING ({', '.join(miss) or prov.get('error')}) -- record refused"
        prov['script_sha'] = SCRIPT_SHA
        out['provenance'] = prov
        out['kprec'] = KPREC or KERNEL_PRECISION.get(PREC)
        if os.environ.get('FIN_CUDA_SELF') == '1':
            try:
                out['ms_cuda_self'] = harness.cuda_buckets(closure, iters=3)['total']
            except Exception as exc:
                out['ms_cuda_self_error'] = f'{type(exc).__name__}: {exc}'[:300]
        return out
    return f
_bg = b.build_grid

def build_grid(name, args):
    return [c for c in _bg(name, args) if keep(c)]
harness.main_for(build_grid, b.GRIDS, f'benchmarks.{MOD}', add_args=getattr(b, '_add_args', None), extra_meta=getattr(b, '_extra_meta', None), post_cell=post_cell, paper_filter=b.PAPER_SETS.get)
