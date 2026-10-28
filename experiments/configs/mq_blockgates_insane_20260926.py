
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, SSD_AXES, SSD_MIXER, SSD_TOP, _ssd
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'
REALGATE_ROWS = [{'frac': 0.5, 'decay': 'head_dynamic', 'complex': True, 'gate': False}, {'frac': 0.5, 'decay': 'head_dynamic', 'complex': False, 'gate': True}]
ARMS = (_ssd('R2O2-complex-realgates-d248-i512', 248, 512, 2, REALGATE_ROWS, 6000, None, 'complex head_dynamic half with gate=False + real head_dynamic half with gate=True (BLOCKGATES, owner 2026-09-26), gates on (R = O = 2 default); d_inner 512 (8 heads) as R2O2-complex-d224-i512, re-widthed d224 -> d248 into the band; module defaults otherwise; new arm'),)
CAMPAIGNS = {'blockgates-insane-20260926': Campaign(id='blockgates-insane-20260926', manifest=DOCS / 'MQ_BLOCKGATES_INSANE_20260926.json', family='ssd_mimo', prefix='blockgates', tiers=('insane',), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER, arm_axes=SSD_AXES, module_defaults=True, definitions=ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status='owner 2026-09-26 (BLOCKGATES): SSD-MIMO R2O2 complex with the B/C gates on the real rows only (complex block gate=False, real block gate=True), module defaults otherwise, re-widthed into the +-2 % band of M1 (d_inner kept); tier insane, 3 seeds; compared with R2O2 complex gated / ungated and R2O2 real'))}
