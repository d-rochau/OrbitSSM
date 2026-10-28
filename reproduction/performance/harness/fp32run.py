
import os, sys
P = os.environ['PB_PREC']
MOD = sys.argv.pop(1)
PAIR = os.environ.get('FIN_PAIR')
HMB = os.environ.get('HMB_TAGS', 'main')
from orbit_ssm.kernels import shape_sets as ss
TAGS = set()
for t in filter(None, HMB.split(',')):
    if t == 'main':
        TAGS |= {k for k in ss.SET if k.startswith('large_') or k in ss.LM_TAGS}
    elif t == 'rest':
        TAGS |= {k for k in ss.SET if k.startswith(('small_', 'medium_'))}
    else:
        TAGS.add(t)
K_ARMS = dict(m2=('ours_fair', 'm2_fair'), m3=('ours_eq', 'm3_siso_eq'))
S_ARMS = dict(m2=('SM-R1:m2exact', 'official_mamba2'), m3=('SM-R1:m3exact', 'official_mamba3'))
import torch
from benchmarks import harness
from benchmarks.measure import IO_DTYPE, KERNEL_PRECISION
KERNEL_PRECISION[P] = os.environ.get('PB_KPREC', KERNEL_PRECISION[P])
from benchmarks.harness import Cell, group_key
if MOD == 'bench_kernels':
    from benchmarks import bench_kernels as b, baselines as bl
    make_pass = b.make_pass
    _m3 = b._m3_eq_cell

    def m3(*a, **k):
        a = list(a)
        assert a[6] == 'bf16'
        a[6] = P
        return _m3(*a, **k)
    b._m3_eq_cell = m3

    def real_fair(grid, B, H, P_, N, L, pass_, arm, chunk, work, tags=None, *, chunk_ours, chunk_m2=256, chunk_bwd=None):
        ours = arm == 'ours'
        dt = IO_DTYPE[P]

        def build():
            bl.real_fair_check(B, H, P_, N, L, chunk_ours=chunk_ours, chunk_m2=chunk_m2, dtype=dt)
            return make_pass(*bl.build_real_fair(B, H, P_, N, L, arm=arm, chunk=chunk, dtype=dt, chunk_bwd=chunk_bwd if ours else None), pass_)
        return Cell(family='ssd_mimo', arm='ours_fair' if ours else 'm2_fair', entry_point=b.SSD_ENTRY if ours else bl.M2_CHUNK_SCAN.entry_point, shape=dict(B=B, H=H, G=P_, R=1, O=1, N=N, L=L, chunk=chunk, d_inner=H * P_), precision=P, pass_=pass_, grid=grid, group=group_key(grid, work, P, pass_), tags=dict(tags or {}, feature='m2_block', chunk_arm=chunk, features='dt_bias+softplus+D+gated_rmsnorm'), build=build)
    b._real_fair_cell = real_fair
    _bgk = b.build_grid

    def build_grid(name, args):
        return [c for c in _bgk(name, args) if c.tags.get('shape') in TAGS and (not PAIR or c.arm in K_ARMS[PAIR])]
    harness.main_for(build_grid, b.GRIDS, 'benchmarks.bench_kernels', add_args=getattr(b, '_add_args', None), extra_meta=getattr(b, '_extra_meta', None), post_cell=b._post_cell, paper_filter=b.PAPER_SETS.get)
else:
    from benchmarks import bench_ssms as b
    _e, _o = (b._exact_mixer_cell, b._official_cell)

    def ex(*a, **k):
        a = list(a)
        assert a[7] == 'bf16'
        a[7] = P
        return _e(*a, **k)

    def of(*a, **k):
        a = list(a)
        assert a[6] == 'bf16'
        a[6] = P
        return _o(*a, **k)
    b._exact_mixer_cell, b._official_cell = (ex, of)
    _bg = b.build_grid

    def build_grid(name, args):
        return [c for c in _bg(name, args) if not c.arm.endswith('@compile') and c.tags.get('shape') in TAGS and (not PAIR or c.arm.split('@')[0] in S_ARMS[PAIR])]
    harness.main_for(build_grid, b.GRIDS, 'benchmarks.bench_ssms', add_args=b._add_args, extra_meta=b._extra_meta, post_cell=b._post_cell, paper_filter=b.PAPER_SETS.get)
