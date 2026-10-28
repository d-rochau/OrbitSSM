
from __future__ import annotations
import copy
from pathlib import Path
from experiments.configs.mq_blockgates_insane_20260926 import ARMS as BLOCKGATES_ARMS, REALGATE_ROWS
from experiments.configs.mq_r4o4_insane_20260926 import ARMS as R4O4_ARMS
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, HM_MIXER, HM_SCREEN2, HM_TOP, SSD_AXES, SSD_MIXER, SSD_R1O1_REAL, SSD_TOP, _ssd
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'

def _reuse(arms, name, budget_s=None, origin=''):
    d = copy.deepcopy(next((a for a in arms if a['arm'] == name)))
    if budget_s is not None:
        d['budget_s'] = budget_s
    d['note'] = f"{d['note']} [reused unchanged from {origin}]"
    return d
R4O4_REAL_BUDGET_S = 10500
R4O4_COMPLEX_BUDGET_S = 15500
SSD_ARMS = (_reuse(BLOCKGATES_ARMS, 'R2O2-complex-realgates-d248-i512', origin='blockgates-insane-20260926'), _reuse(R4O4_ARMS, 'R4O4-real-d176-i384', R4O4_REAL_BUDGET_S, origin='r4o4-insane-20260926'), _ssd('R4O4-complex-realgates-d208-i384', 208, 384, 4, REALGATE_ROWS, R4O4_COMPLEX_BUDGET_S, None, 'complex head_dynamic half with gate=False + real head_dynamic half with gate=True (the new default, BLOCKGATES 2026-09-26), R = O = 4; d_inner 384 (6 heads) as the r4o4 arms, re-widthed d176 -> d208 into the band; module defaults otherwise; new arm, replaces R4O4-complex-d176-i384 (per-coordinate complex gates, deprecated)'), _reuse(SSD_R1O1_REAL, 'R1O1-real-d296-i576', 4500, origin='ssd-r1o1real-20260925'))
HM_ARMS = (_reuse(HM_SCREEN2, 'A_dyn', origin='hm-screen2-20260925'),)
_STATUS = 'owner plan 2026-09-27 (TASK_P1_MQMKVAR_FINAL_20260927.md): missing cells of the Paper-1 MQMKVAR table; arms reused unchanged from their earlier campaigns except the new-default R4O4 complex arm; cells chosen by the launcher INDICES'
CAMPAIGNS = {'p1final-ssd-20260927': Campaign(id='p1final-ssd-20260927', manifest=DOCS / 'MQ_P1FINAL_SSD_20260927.json', family='ssd_mimo', prefix='p1final_ssd', tiers=('hard', 'easy', 'insane'), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER, arm_axes=SSD_AXES, module_defaults=True, definitions=SSD_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status=_STATUS)), 'p1final-hm-20260927': Campaign(id='p1final-hm-20260927', manifest=DOCS / 'MQ_P1FINAL_HM_20260927.json', family='hybrid_mamba', prefix='p1final_hm', tiers=('hard', 'easy'), allowed_top=HM_TOP, allowed_mixer=HM_MIXER, module_defaults=False, definitions=HM_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status=_STATUS))}
