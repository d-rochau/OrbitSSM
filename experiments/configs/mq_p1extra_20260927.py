
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_blockgates_insane_20260926 import REALGATE_ROWS
from experiments.configs.mq_gates_off_insane_20260926 import ARMS as GATES_OFF_ARMS, GATE_KEYS
from experiments.configs.mq_hm_mixc_insane_20260926 import ARMS as MIXC_ARMS
from experiments.configs.mq_p1final_20260927 import _reuse
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, HM_GATES, HM_MIXER, HM_TOP, SSD_AXES, SSD_MIXER, SSD_TOP, _ssd
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'
ROTATION_KEYS = frozenset({'rotation'})
PER_HEAD_OMEGA = {'omega_layout': 'head', 'theta_init': 'small'}

def _per_head_omega():
    d = _ssd('R2O2-complex-realgates-head-d248-i512', 248, 512, 2, REALGATE_ROWS, 6000, None, 'R2O2-complex-realgates-d248-i512 (the new default: complex head_dynamic half gate=False + real head_dynamic half gate=True) with rotation omega_layout "head" (one complex eigenvalue per head: theta_proj emits H values, theta_freq (H,)) and theta_init "small" (W_theta kaiming x 0.1 sqrt(3)); same width, module defaults otherwise; new arm, replaces R2O2-complex-head-d224-i512 of quad-insane-20260926 (per-coordinate complex gates, deprecated)')
    d['kwargs']['mixer_kwargs']['rotation'] = dict(PER_HEAD_OMEGA)
    return d
SSD_ARMS = (_per_head_omega(), _reuse(GATES_OFF_ARMS, 'R2O2-real-nogates-d312-i448', origin='gates-off-insane-20260926'), _reuse(GATES_OFF_ARMS, 'R2O2-complex-nogates-d280-i512', origin='gates-off-insane-20260926'))
HM_ARMS = (_reuse(MIXC_ARMS, 'A_dyn_c25a', origin='hm-mixc-insane-20260926'), _reuse(MIXC_ARMS, 'A_dyn_c25d', origin='hm-mixc-insane-20260926'), _reuse(HM_GATES, 'real_off', origin='hm-gates-20260925 (mq-hm-20260925b)'), _reuse(HM_GATES, 'rot_off', origin='hm-gates-20260925 (mq-hm-rot-20260925b)'))
_STATUS = 'owner plan 2026-09-27 (TASK_P1_MQMKVAR_FINAL_20260927.md implementation-implementation): missing hard / easy cells of the further paper configs (gates-off ablation, HM row-mix variants, per-head omega at the new default); arms reused unchanged from their earlier campaigns except the new per-head-omega arm; cells chosen by the launcher INDICES'
CAMPAIGNS = {'p1extra-ssd-20260927': Campaign(id='p1extra-ssd-20260927', manifest=DOCS / 'MQ_P1EXTRA_SSD_20260927.json', family='ssd_mimo', prefix='p1extra_ssd', tiers=('hard', 'easy', 'insane'), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER | GATE_KEYS | ROTATION_KEYS, arm_axes=SSD_AXES | GATE_KEYS | ROTATION_KEYS, module_defaults=True, definitions=SSD_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status=_STATUS)), 'p1extra-hm-20260927': Campaign(id='p1extra-hm-20260927', manifest=DOCS / 'MQ_P1EXTRA_HM_20260927.json', family='hybrid_mamba', prefix='p1extra_hm', tiers=('hard', 'easy'), allowed_top=HM_TOP, allowed_mixer=HM_MIXER, module_defaults=False, definitions=HM_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status=_STATUS))}
