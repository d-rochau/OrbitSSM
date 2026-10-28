
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_gates_off_insane_20260926 import GATE_KEYS
from experiments.configs.mq_lrsweep_20260927 import LRS, SEEDS, _H, _at_lr, _arm
from experiments.configs.mq_p1extra_20260927 import HM_ARMS as P1EXTRA_HM, ROTATION_KEYS, SSD_ARMS as P1EXTRA_SSD
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, HM_MIXER, HM_TOP, SSD_AXES, SSD_COMPLEX, SSD_MIXER, SSD_R1O1_REAL, SSD_TOP
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'
BUDGET = {'R2O2-complex-realgates-head-d248-i512': 7100, 'R2O2-real-nogates-d312-i448': 5700, 'R2O2-complex-nogates-d280-i512': 7100, 'R1O1-real-d296-i576': 4900, 'R1O1-complex-d296-i576': 4500, 'A_dyn_c25a': 10800, 'A_dyn_c25d': 10800, 'real_off': 6600, 'rot_off': 8900}
SSD_PAPER = ((_arm(P1EXTRA_SSD, 'R2O2-complex-realgates-head-d248-i512'), 'p1extra-ssd-20260927'), (_arm(P1EXTRA_SSD, 'R2O2-complex-nogates-d280-i512'), 'gates-off-insane-20260926'), (_arm(P1EXTRA_SSD, 'R2O2-real-nogates-d312-i448'), 'gates-off-insane-20260926'), (_arm(SSD_COMPLEX, 'R1O1-complex-d296-i576'), 'ssd-complex-20260925 (mq-ssd-complex-20260925b)'), (_arm(SSD_R1O1_REAL, 'R1O1-real-d296-i576'), 'ssd-r1o1real-20260925'))
HM_PAPER = ((_arm(P1EXTRA_HM, 'A_dyn_c25a'), 'hm-mixc-insane-20260926'), (_arm(P1EXTRA_HM, 'A_dyn_c25d'), 'hm-mixc-insane-20260926'), (_arm(P1EXTRA_HM, 'rot_off'), 'hm-gates-20260925 (mq-hm-rot-20260925b)'), (_arm(P1EXTRA_HM, 'real_off'), 'hm-gates-20260925 (mq-hm-20260925b)'))
SSD_ARMS = tuple((_at_lr(d, lr, BUDGET[d['arm']], origin) for d, origin in SSD_PAPER for lr in LRS))
HM_ARMS = tuple((_at_lr(d, lr, BUDGET[d['arm']], origin) for d, origin in HM_PAPER for lr in LRS))
_STATUS = 'owner decision 2026-09-28 (TASK_P1_MQMKVAR_FINAL_20260927.md implementation addendum): the insane-tier peak-LR sweep extended to the nine not-yet-swept paper configurations, arms unchanged, only the peak LR differs from the frozen recipe'
_H2 = dict(_H, status=_STATUS, lrs_swept=[0.0005, 0.002])
CAMPAIGNS = {'lrsweep2-ssd-20260928': Campaign(id='lrsweep2-ssd-20260928', manifest=DOCS / 'MQ_LRSWEEP2_SSD_20260928.json', family='ssd_mimo', prefix='lrsweep2_ssd', tiers=('insane',), seeds=SEEDS, allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER | GATE_KEYS | ROTATION_KEYS, arm_axes=SSD_AXES | GATE_KEYS | ROTATION_KEYS, module_defaults=True, definitions=SSD_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=_H2), 'lrsweep2-hm-20260928': Campaign(id='lrsweep2-hm-20260928', manifest=DOCS / 'MQ_LRSWEEP2_HM_20260928.json', family='hybrid_mamba', prefix='lrsweep2_hm', tiers=('insane',), seeds=SEEDS, allowed_top=HM_TOP, allowed_mixer=HM_MIXER, module_defaults=False, definitions=HM_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=_H2)}
