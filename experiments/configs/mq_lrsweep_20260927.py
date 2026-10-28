
from __future__ import annotations
import copy
from pathlib import Path
from experiments.configs.mq_attn_ref_20260925 import ATTN_KWARGS, ATTN_REF, GUARD as ATTN_GUARD, GUARD_EVIDENCE as ATTN_GUARD_EVIDENCE, UPSTREAM_ATTN
from experiments.configs.mq_p1final_20260927 import HM_ARMS as P1FINAL_HM, SSD_ARMS as P1FINAL_SSD
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, HM_MIXER, HM_TOP, SSD_AXES, SSD_MIXER, SSD_REAL, SSD_TOP
from experiments.mqmkvar.campaign import OFFICIAL_MAMBA, Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'
LRS = (0.0005, 0.002)
LOW_LR = 0.0001
SEEDS = (101, 202, 303)

def lr_tag(lr: float) -> str:
    return {0.0001: '1e-4', 0.0005: '5e-4', 0.002: '2e-3'}[lr]

def _at_lr(definition: dict, lr: float, budget_s: int, origin: str) -> dict:
    d = copy.deepcopy(definition)
    d['paper_arm'] = d['arm']
    d['arm'] = f"{d['arm']}-lr{lr_tag(lr)}"
    d['learning_rate'] = lr
    d['budget_s'] = budget_s
    d['note'] = f"paper arm {d['paper_arm']} ({origin}) unchanged, peak LR {lr_tag(lr)} instead of 1e-3; frozen recipe otherwise" + (' (flat after warmup: eta_min = peak)' if lr == LOW_LR else '')
    return d

def _arm(arms, name):
    return next((a for a in arms if a['arm'] == name))
BUDGET = {'R2O2-real-d248-i448': 5400, 'R2O2-complex-realgates-d248-i512': 6600, 'R4O4-real-d176-i384': 11700, 'R4O4-complex-realgates-d208-i384': 18200, 'A_dyn': 6200, 'zoology-attn-d392': 4700, 'official_mamba1': 6900, 'official_mamba2': 3600, 'official_mamba3': 4000}
SSD_PAPER = ((_arm(SSD_REAL, 'R2O2-real-d248-i448'), 'ssd-real-20260925'), (_arm(P1FINAL_SSD, 'R2O2-complex-realgates-d248-i512'), 'blockgates-insane-20260926'), (_arm(P1FINAL_SSD, 'R4O4-real-d176-i384'), 'r4o4-insane-20260926'), (_arm(P1FINAL_SSD, 'R4O4-complex-realgates-d208-i384'), 'p1final-ssd-20260927'))
SSD_ARMS = tuple((_at_lr(d, lr, BUDGET[d['arm']], origin) for d, origin in (SSD_PAPER[3], SSD_PAPER[2], SSD_PAPER[1], SSD_PAPER[0]) for lr in LRS))
HM_ARMS = tuple((_at_lr(_arm(P1FINAL_HM, 'A_dyn'), lr, BUDGET['A_dyn'], 'hm-screen2-20260925') for lr in LRS))
ATTN_ARMS = tuple((_at_lr(ATTN_REF[0], lr, BUDGET['zoology-attn-d392'], 'attn-ref-20260925') for lr in (LOW_LR,) + LRS))
M_KWARGS = {'official_mamba1': dict(d_model=256, n_layers=2, d_state=128, expand_factor=2, headdim=64), 'official_mamba2': dict(d_model=320, n_layers=2, d_state=128, expand_factor=2, headdim=64, ngroups=1), 'official_mamba3': dict(d_model=320, n_layers=2, d_state=128, expand_factor=2, headdim=64, ngroups=1)}
M_ORIGIN = {'official_mamba1': ('mq-baselines-matched-20260920', dict(total=3317504, embedding=2097152, non_embedding=1220352)), 'official_mamba2': ('mq-stock-above-20260922', dict(total=4031740, embedding=2621440, non_embedding=1410300)), 'official_mamba3': ('mq-m3-upstream-20260924', dict(total=4060392, embedding=2621440, non_embedding=1438952))}

def _official(family: str) -> dict:
    root, counts = M_ORIGIN[family]
    return dict(arm=family, family=family, kwargs=dict(M_KWARGS[family]), budget_s=BUDGET[family], supersedes=dict(root=root, **counts), note=f'official {family} paper arm of {root}')
M12_ARMS = tuple((_at_lr(_official(f), lr, BUDGET[f], M_ORIGIN[f][0]) for f in ('official_mamba1', 'official_mamba2') for lr in LRS))
M3_ARMS = tuple((_at_lr(_official('official_mamba3'), lr, BUDGET['official_mamba3'], M_ORIGIN['official_mamba3'][0]) for lr in (LOW_LR,) + LRS))
GUARD_NONFINITE_ONLY = dict(GUARD, delta_max=1e+30, grad_ratio=1e+30, detail_stats=False)
GUARD_NONFINITE_EVIDENCE = 'official baselines ran unguarded (mq-baselines-matched-20260920, mq-stock-above-20260922, mq-m3-upstream-20260924); only the always-on non-finite abort is active here'
M12_UPSTREAM = ('mamba_ssm.modules.mamba_simple', 'mamba_ssm.modules.mamba2')
M3_UPSTREAM = ('mamba_ssm.modules.mamba3',)
_STATUS = 'owner decision 2026-09-27 (TASK_P1_MQMKVAR_FINAL_20260927.md implementation): insane-tier peak-LR robustness sweep of the nine main-text paper arms, arms unchanged, only the peak LR differs from the frozen recipe'
_H = dict(status=_STATUS, lrs_swept=[0.0001, 0.0005, 0.002], lr_reference='1e-3 = the paper-table runs', flat_schedule_note='peak 1e-4 = eta_min 1e-4: constant LR after the 2 warmup epochs')
CAMPAIGNS = {'lrsweep-ssd-20260927': Campaign(id='lrsweep-ssd-20260927', manifest=DOCS / 'MQ_LRSWEEP_SSD_20260927.json', family='ssd_mimo', prefix='lrsweep_ssd', tiers=('insane',), seeds=SEEDS, allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER, arm_axes=SSD_AXES, module_defaults=True, definitions=SSD_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=_H), 'lrsweep-hm-20260927': Campaign(id='lrsweep-hm-20260927', manifest=DOCS / 'MQ_LRSWEEP_HM_20260927.json', family='hybrid_mamba', prefix='lrsweep_hm', tiers=('insane',), seeds=SEEDS, allowed_top=HM_TOP, allowed_mixer=HM_MIXER, module_defaults=False, definitions=HM_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=_H), 'lrsweep-attn-20260927': Campaign(id='lrsweep-attn-20260927', manifest=DOCS / 'MQ_LRSWEEP_ATTN_20260927.json', family='attention', prefix='lrsweep_attn', tiers=('insane',), seeds=SEEDS, allowed_top=frozenset(ATTN_KWARGS), module_defaults=False, definitions=ATTN_ARMS, guard=ATTN_GUARD, guard_evidence=ATTN_GUARD_EVIDENCE, header=_H, size_rule='floor', upstream_modules=UPSTREAM_ATTN), 'lrsweep-m12-20260927': Campaign(id='lrsweep-m12-20260927', manifest=DOCS / 'MQ_LRSWEEP_M12_20260927.json', family=OFFICIAL_MAMBA, prefix='lrsweep_m12', tiers=('insane',), seeds=SEEDS, allowed_top=frozenset(M_KWARGS['official_mamba2']), module_defaults=False, definitions=M12_ARMS, guard=GUARD_NONFINITE_ONLY, guard_evidence=GUARD_NONFINITE_EVIDENCE, header=_H, size_rule='none', upstream_modules=M12_UPSTREAM), 'lrsweep-m3-20260927': Campaign(id='lrsweep-m3-20260927', manifest=DOCS / 'MQ_LRSWEEP_M3_20260927.json', family=OFFICIAL_MAMBA, prefix='lrsweep_m3', tiers=('insane',), seeds=SEEDS, allowed_top=frozenset(M_KWARGS['official_mamba3']), module_defaults=False, definitions=M3_ARMS, guard=GUARD_NONFINITE_ONLY, guard_evidence=GUARD_NONFINITE_EVIDENCE, header=_H, size_rule='none', upstream_modules=M3_UPSTREAM)}
