
from __future__ import annotations
from dataclasses import dataclass

def ssd_rows(rotate: bool, law: str, complex_frac: float=1.0) -> list[dict]:
    if not rotate:
        return [{'frac': 1.0, 'decay': law, 'complex': False}]
    rows = [{'frac': complex_frac, 'decay': law, 'complex': True, 'gate': True}]
    if complex_frac < 1.0:
        rows.append({'frac': 1.0 - complex_frac, 'decay': law, 'complex': False})
    return rows
PAPER_C3_SHAPES = (dict(B=8, D=2048, L=2048), dict(B=2, D=4096, L=4096), dict(B=4, D=512, L=16384))
PAPER_C3_N = 64
PAPER_C3_SHAPES_PAPER = (PAPER_C3_SHAPES[0],)
PAPER_C3_HEAD_WIDTHS = (1, 2, 4, 8, 16)
PAPER_C3_NSWEEP = dict(B=6, D=2048, L=2048, N=(16, 32, 64, 128, 256), head_widths=(1, 4, 16))
PAPER_C3_LSWEEP = dict(B=1, D=2048, N=64, L=(2048, 4096, 8192, 16384, 32768))
PAPER_C3_LSWEEP_D512 = dict(B=1, D=512, N=64, L=(2048, 4096, 8192, 16384, 32768))
SSD_PARITY_N = 64
SSD_PARITY_N128 = 128
SSD_PARITY_N_VALUES = (SSD_PARITY_N, SSD_PARITY_N128)
SSD_PARITY_T = 64
SSD_PARITY_REF = dict(H=32, G=64)
SSD_PARITY_PROD = dict(H=13, G=128)
SSD_PARITY26 = tuple([dict(sweep='batch_ref', B=b, **SSD_PARITY_REF, L=2048) for b in (1, 2, 4, 8, 16)] + [dict(sweep='batch_prod', B=b, **SSD_PARITY_PROD, L=2048) for b in (1, 2, 4, 8, 16)] + [dict(sweep='len_ref', B=4, **SSD_PARITY_REF, L=l) for l in (512, 2048, 8192)] + [dict(sweep='len_prod', B=4, **SSD_PARITY_PROD, L=l) for l in (512, 2048, 8192)] + [dict(sweep='hg_grid', B=4, H=h, G=g, L=2048) for h, g in ((8, 64), (16, 128), (32, 32), (32, 64), (32, 128), (48, 64), (64, 64), (64, 128), (8, 256), (13, 128))])
assert len(SSD_PARITY26) == 26, 'the 26-cell grid must have 26 cells'
SSD_PARITY26_PAPER_KEYS = frozenset((('batch_ref', 1, 32, 64, 2048), ('batch_ref', 16, 32, 64, 2048), ('len_ref', 4, 32, 64, 512), ('len_ref', 4, 32, 64, 8192), ('hg_grid', 4, 32, 32, 2048), ('hg_grid', 4, 8, 256, 2048)))
SSD_PARITY26_PAPER = tuple((c for c in SSD_PARITY26 if (c['sweep'], c['B'], c['H'], c['G'], c['L']) in SSD_PARITY26_PAPER_KEYS))
assert len(SSD_PARITY26_PAPER) == 6, 'the paper subset of ssd_parity26 must have exactly 6 cells'
SSD_LONG_L = tuple((dict(sweep='long_l', B=b, **SSD_PARITY_REF, L=l) for b in (1, 2, 4, 8) for l in (8192, 32768, 65536)))
SSD_LONG_L_PAPER = tuple((dict(sweep='long_l', B=1, **SSD_PARITY_REF, L=l) for l in (32768, 65536)))
MIMO_GATES = tuple((dict(B=2, L=2048, d_inner=2048, R=r, O=o, G=g, N=n) for r, o in ((1, 1), (2, 2), (4, 4)) for g in (16, 32, 64, 128) for n in (64, 128)))
SSD_R4O4_GATE = dict(B=2, L=2048, d_inner=2048, R=4, O=4, G=128, N=128)
SSD_FEATURE_BASE = dict(B=4, H=32, G=64, R=1, O=1, N=64, L=2048, T=64)
SSD_FEATURES = ('none', 'theta', 'theta_h', 'rot_bias', 'beta', 'gamma', 'gate_z', 'delta_w', 'x0', 'bc_per_head')
SSD_CHUNKS = (64, 128)
MH_CONTRACT_SHAPES = (dict(name='B8 H64 P32 N128 L2048', B=8, H=64, P=32, N=128, L=2048), dict(name='B8 H16 P32 N64 L2048', B=8, H=16, P=32, N=64, L=2048), dict(name='B4 H32 P32 N128 L1024', B=4, H=32, P=32, N=128, L=1024), dict(name='B2 H32 P64 N64 L4096', B=2, H=32, P=64, N=64, L=4096))
MH_CONTRACT_SHAPES_PAPER = (MH_CONTRACT_SHAPES[0],)
MH_SISO_SHAPES = (dict(B=8, D=2048, N=64, L=2048), dict(B=8, D=2048, N=16, L=2048), dict(B=4, D=1024, N=16, L=2048), dict(B=2, D=2048, N=64, L=4096))
MH_FEATURES = (('real', dict()), ('gates', dict(gates=True)), ('bias_g1', dict(biases=True, G=1)), ('bias_gh', dict(biases=True, G='H')), ('delta_w_head', dict(delta_w='head')), ('delta_w_chan', dict(delta_w='chan')), ('rot', dict(rot=True)), ('gates_bias', dict(gates=True, biases=True, G=1)), ('all', dict(gates=True, biases=True, G=1, delta_w='chan')), ('all_rot', dict(gates=True, biases=True, G=1, delta_w='chan', rot=True)))
MH_LAWS = ('A', 'dyn', 'mu')
MQMKVAR_SHAPE = dict(B=32, L=2048, d_model=256, N_mhr=64)
LM_SPINE = dict(B=8, D=2048, L=2048, N=64)
EXACT_D_STATE_M23 = 128
EXACT_D_STATE_M23_N64 = 64
EXACT_D_STATE_M1 = 16
LM_440M = dict(vocab_size=50288, d_model=1024, n_layers=24, B=8, L=2048, d_state=64)
MQMKVAR_S_ARMS = (dict(id='S1', width=264, d_inner=528, R=1, O=1, ngroups=16, rows=ssd_rows(False, 'head_static')), dict(id='S2', width=200, d_inner=400, R=2, O=2, ngroups=8, rows=ssd_rows(True, 'head_static')), dict(id='S3', width=200, d_inner=400, R=2, O=2, ngroups=8, rows=ssd_rows(False, 'head_static')), dict(id='S4', width=152, d_inner=304, R=4, O=4, ngroups=4, rows=ssd_rows(True, 'head_static')), dict(id='S5', width=200, d_inner=400, R=2, O=2, ngroups=8, rows=ssd_rows(True, 'head_dynamic')))
MQMKVAR_OTHER_ARMS = (dict(id='HM', family='hybrid_mamba', d_model=256, d_inner=512, head_dim=16, width_convention='officials'),)

def _check_mqmkvar_arms() -> None:
    for arm in MQMKVAR_S_ARMS:
        if arm['d_inner'] != 2 * arm['width']:
            raise ValueError(f"{arm['id']}: d_inner must be 2*width (expand=2), got {arm}")
        P = arm['ngroups'] * arm['R']
        if P != 16:
            raise ValueError(f"{arm['id']}: G*R = {P}, the spec's head-dim-16 rule says 16")
        if arm['d_inner'] % P:
            raise ValueError(f"{arm['id']}: d_inner {arm['d_inner']} is not a multiple of P=G*R={P} — the SSD core refuses it")
    official = 2 * MQMKVAR_SHAPE['d_model']
    for arm in MQMKVAR_OTHER_ARMS:
        if arm['d_inner'] != 2 * arm['d_model']:
            raise ValueError(f"{arm['id']}: d_inner must be 2*d_model (expand=2), got {arm}")
        if arm['d_inner'] != official:
            raise ValueError(f"{arm['id']}: d_inner {arm['d_inner']} is not the officials' {official} — the ssm_mqmkvar group compares these arms against official_mamba1/2/3 built from d_model={MQMKVAR_SHAPE['d_model']} through expand=2")
        hd = arm.get('head_dim')
        if hd is not None and hd & hd - 1:
            raise ValueError(f"{arm['id']}: head width {hd} is not a power of two — the MH CUDA backward refuses it")
_check_mqmkvar_arms()
SMOKE_MH = dict(B=2, D=256, N=16, L=512)
SMOKE_MH_MIMO = dict(B=2, H=4, P=8, N=16, L=512)
SMOKE_SSD = dict(B=2, H=4, G=64, R=1, O=1, N=64, L=512, T=64)
SMOKE_SSD_MIMO = dict(B=2, H=4, G=16, R=2, O=2, N=64, L=512, T=64)
SMOKE_SSM = dict(B=2, L=256, d_model=128, d_state=32)
SMOKE_MODEL = dict(vocab_size=1024, d_model=128, n_layers=2, B=2, L=256, d_state=32)

@dataclass(frozen=True)
class GridInfo:
    name: str
    module: str
    source: str
from orbit_ssm.kernels import shape_sets as _shape_sets
SHARED_GRID, SHARED_LM, SHARED_SET = (_shape_sets.GRID, _shape_sets.LM, _shape_sets.SET)
GRID_SOURCES = (GridInfo('kernel_shared_set', 'bench_kernels', 'src/orbit_ssm/kernels/shape_sets.py: kernel-level ours vs Mamba-2 chunk scan / Mamba-3 SISO (a4_trapon) on the one shared set'), GridInfo('ssm_shared_set', 'bench_ssms', 'src/orbit_ssm/kernels/shape_sets.py: the one set every calibration and benchmark uses'), GridInfo('ssm_shared_small', 'bench_ssms', 'src/orbit_ssm/kernels/shape_sets.py: the small cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)'), GridInfo('ssm_shared_medium', 'bench_ssms', 'src/orbit_ssm/kernels/shape_sets.py: the medium cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)'), GridInfo('ssm_shared_large', 'bench_ssms', 'src/orbit_ssm/kernels/shape_sets.py: the large cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)'), GridInfo('ssm_shared_lm', 'bench_ssms', 'src/orbit_ssm/kernels/shape_sets.py: the lm cells of ssm_shared_set (a shard-planner unit; its cells carry grid = ssm_shared_set)'), GridInfo('smoke', 'bench_kernels', 'this file (plumbing gate only)'), GridInfo('mh_siso', 'bench_kernels', 'the retired mh SISO bench (never run)'), GridInfo('mh_features', 'bench_kernels', 'the retired mh feature + decay-law benches at the mh-vs-A shapes; fills paper/hybrid_ssm.tex:1023'), GridInfo('paper_c3', 'bench_kernels', 'anonymous-revision-12:paper/tables/timings_v2_h100_all_packed.tex:3 (Table 10; removed 2026-09-23)'), GridInfo('paper_c3_nscale', 'bench_kernels', 'anonymous-revision-12:paper/tables/timings_h100_nscale.tex:2 (Table 11; removed 2026-09-23)'), GridInfo('paper_c3_lscale', 'bench_kernels', 'paper/appendix_kernels.tex:262-264 (Figure 4) — caption vs generator shape discrepancy, see PAPER_C3_LSWEEP'), GridInfo('ssd_parity26', 'bench_kernels', 'implementation.md:391-397 (Section 6 results)'), GridInfo('ssd_long_l', 'bench_kernels', 'implementation.md:574-586'), GridInfo('ssd_mimo_gates', 'bench_kernels', 'contract Section 1.1/1.7 envelope; R4O4 gate at :346-352'), GridInfo('ssd_features', 'bench_kernels', 'contract Section 1.7 feature list'), GridInfo('ssm_mqmkvar', 'bench_ssms', 'SPEC_MQMKVAR.md:60-100 + archive/implementationAGENT_SSDMIMO_FAMILY_20260902.md:33-34'), GridInfo('ssm_lm_spine', 'bench_ssms', 'the LM-width spine cells'), GridInfo('model_440m', 'bench_models', 'the FineWeb-Edu 440M spine + the param-match standing rule'))
