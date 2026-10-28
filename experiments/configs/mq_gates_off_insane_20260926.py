
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_recollect_20260925 import COMPLEX_ROWS, GUARD, GUARD_EVIDENCE, REAL_ROWS, SSD_AXES, SSD_MIXER, SSD_TOP, _ssd
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'
GATE_KEYS = frozenset({'use_g_b', 'use_g_c'})

def _nogates(arm, d_model, d_inner, rows, note):
    d = _ssd(arm, d_model, d_inner, 2, rows, 6000, None, note)
    d['kwargs']['mixer_kwargs'].update(use_g_b=False, use_g_c=False)
    return d
ARMS = (_nogates('R2O2-real-nogates-d312-i448', 312, 448, REAL_ROWS, 'R2O2-real-d248-i448 (mq-ssd-real-20260925) with use_g_b = use_g_c = False, re-widthed (d_inner kept, 7 heads); module defaults otherwise; new arm'), _nogates('R2O2-complex-nogates-d280-i512', 280, 512, COMPLEX_ROWS, 'R2O2-complex-d224-i512 (mq-ssd-complex-20260925b) with use_g_b = use_g_c = False, re-widthed (d_inner kept, 8 heads); module defaults otherwise; new arm'))
CAMPAIGNS = {'gates-off-insane-20260926': Campaign(id='gates-off-insane-20260926', manifest=DOCS / 'MQ_GATES_OFF_INSANE_20260926.json', family='ssd_mimo', prefix='gates_off', tiers=('insane',), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER | GATE_KEYS, arm_axes=SSD_AXES | GATE_KEYS, module_defaults=True, definitions=ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status='owner 2026-09-26: SSD-MIMO R2O2 real / complex without B/C gates (use_g_b = use_g_c = False), module defaults otherwise, re-widthed into the +-2 % band of M1 (d_inner kept); tier insane; compared with the gated R2O2 real (mq-ssd-real-20260925) / complex (mq-ssd-complex-20260925b)'))}
