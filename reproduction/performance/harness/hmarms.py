
import torch
LM_ARM = {'HM-0': 'hm_a_dyn', 'HM-75': 'hm_a_dyn_cplx75', 'HM-100': 'hm_a_dyn_cplx100'}
for _r in (1, 2, 4):
    for _k, _n in (('re', 'real'), ('cx', 'complex')):
        LM_ARM[f'SSD-R{_r}O{_r}-{_k}'] = f'ssd_mimo_r{_r}o{_r}_{_n}'
PAIR_CFG = {'HMM:m1exact': 'm1exact', 'PM:m2exact': 'm2exact', 'PM:m3exact': 'm3exact'}
OURS_ARMS = ('HMM:m1exact', 'HM-0', 'HM-75', 'HM-100', 'HM-0@P32', 'HM-75@P32', 'HM-100@P32', 'PM:m2exact', 'PM:m3exact') + tuple((k for k in LM_ARM if k.startswith('SSD-')))
OFFICIAL = {'official_mamba1': 'mamba1', 'official_mamba2': 'mamba2', 'official_mamba3': 'mamba3', 'official_mamba3_mimo': 'mamba3'}

def build(arm, d_model, N, B, L, dtype=torch.float32, seed=0, x_in=True, **over):
    from benchmarks import baselines as bl
    P = 64
    if '@P' in arm:
        arm, P = arm.split('@P')
        P = int(P)
    if arm in PAIR_CFG:
        from benchmarks.bench_ssms import build_exact_pair
        cfg = PAIR_CFG[arm]
        if cfg != 'm1exact':
            over.setdefault('precision', 'io')
        _, mod = build_exact_pair(cfg, d_model, N, B, L, dtype=dtype, seed=seed, **over)
    elif arm == 'official_mamba3_mimo':
        from mamba_ssm.modules.mamba3 import Mamba3
        torch.manual_seed(seed)
        mod = Mamba3(d_model=d_model, d_state=N, headdim=64, expand=2, is_mimo=True, mimo_rank=4, chunk_size=16, device='cuda', dtype=dtype)
    elif arm in OFFICIAL:
        kw = {} if arm == 'official_mamba1' else dict(headdim=64, ngroups=1, expand=2)
        _, _, mod = bl.build_mamba_layer(OFFICIAL[arm], d_model, d_state=N, L=1, B=1, dtype=dtype, seed=seed, **kw)
    elif arm in LM_ARM:
        from experiments.lm.model.lm import NAMED_SSM
        from orbit_ssm.model import CausalLM
        spec = NAMED_SSM[LM_ARM[arm]]
        torch.manual_seed(seed)
        kw = dict(spec['kwargs'])
        if spec['name'] == 'hybrid_mamba' and P != 64:
            kw['head_subsystems'] = P
        if spec['name'] == 'ssd_mimo':
            kw['heads'] = 2 * d_model // kw['head_subsystems']
        kw.update(over)
        mod = CausalLM._build_mixer(d_model=d_model, ssm_type=spec['name'], ssm_kwargs=kw, expand=2.0, d_conv=4, d_state=N)
        mod = mod.to(device='cuda', dtype=dtype)
    else:
        raise ValueError(arm)
    if not x_in:
        return (None, None, mod)
    x = bl.module_input(B, L, d_model, device='cuda', dtype=dtype, seed=seed)
    return (lambda: mod(x), [x, *mod.parameters()], mod)
