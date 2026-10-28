
from __future__ import annotations
import copy
from pathlib import Path
from experiments.configs.mq_blockgates_insane_20260926 import ARMS as BLOCKGATES_ARMS
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, HM_MIXER, HM_SCREEN2, HM_TOP, SSD_AXES, SSD_MIXER, SSD_REAL, SSD_TOP
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'

def _euler(arms, name, new_name, origin, hm=False):
    d = copy.deepcopy(next((a for a in arms if a['arm'] == name)))
    mk = d['kwargs']['mixer_kwargs']
    if hm:
        assert mk['use_trap'] is True
        mk['use_trap'] = False
    else:
        mk['write_rule'] = 'euler'
    d.update(arm=new_name, supersedes=None, note=f"{name} of {origin} with write_rule='euler' (Mamba exponential-Euler forcing) instead of the trapezoid default; width unchanged (in band); everything else as the baseline")
    return d
SSD_ARMS = (_euler(SSD_REAL, 'R2O2-real-d248-i448', 'R2O2-real-euler-d248-i448', 'ssd-real-20260925'), _euler(BLOCKGATES_ARMS, 'R2O2-complex-realgates-d248-i512', 'R2O2-complex-realgates-euler-d248-i512', 'blockgates-insane-20260926'))
HM_ARMS = (_euler(HM_SCREEN2, 'A_dyn', 'A_dyn_euler', 'hm-screen2-20260925', hm=True),)
_STATUS = 'owner 2026-09-27: write rule euler vs the trapezoid default on current code; each arm = its trapezoid baseline (read-only: mq-ssd-real-20260925, mq-blockgates-insane-20260926, mq-hm-screen2-20260925 A_dyn) with only the write rule switched; widths unchanged (in band); tier insane, 3 seeds'
CAMPAIGNS = {'euler-insane-ssd-20260927': Campaign(id='euler-insane-ssd-20260927', manifest=DOCS / 'MQ_EULER_INSANE_SSD_20260927.json', family='ssd_mimo', prefix='euler_ssd', tiers=('insane',), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER | {'write_rule'}, arm_axes=SSD_AXES | {'write_rule'}, module_defaults=True, definitions=SSD_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status=_STATUS)), 'euler-insane-hm-20260927': Campaign(id='euler-insane-hm-20260927', manifest=DOCS / 'MQ_EULER_INSANE_HM_20260927.json', family='hybrid_mamba', prefix='euler_hm', tiers=('insane',), allowed_top=HM_TOP, allowed_mixer=HM_MIXER, module_defaults=False, definitions=HM_ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status=_STATUS))}
