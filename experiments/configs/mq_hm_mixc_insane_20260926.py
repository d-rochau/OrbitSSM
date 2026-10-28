
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_recollect_20260925 import GUARD, GUARD_EVIDENCE, HM_MIXER, HM_TOP, _a_mix, _hm_screen2
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'

def _mix3(third):
    return [{'rows': 48, 'decay': 'A', 'complex': False}, {'rows': 48, 'decay': 'dyn', 'complex': False}, third]

def _nogates(arm, d_inner, rows, note):
    d = _hm_screen2(arm, d_inner, rows, note=note)
    d['kwargs']['mixer_kwargs'].pop('gate_init')
    return d
WIDTH = dict(A_dyn_gates=416, A_dyn_c25a=512, A_dyn_c25d=512)
ARMS = (_hm_screen2('A_dyn_gates', WIDTH['A_dyn_gates'], _a_mix({'rows': 64, 'decay': 'dyn', 'complex': False}), ('one_plus', 'pre'), 'hm-screen2 A_dyn rows (64 A + 64 dyn, real) with use_g_b = use_g_c = True at the ruled HM default gate form (one_plus, pre, kaiming_uniform), re-widthed; new arm'), _nogates('A_dyn_c25a', WIDTH['A_dyn_c25a'], _mix3({'rows': 32, 'decay': 'A', 'complex': True}), 'rows 48 A (real) + 48 dyn (real) + 32 A (complex), no gates; new arm'), _nogates('A_dyn_c25d', WIDTH['A_dyn_c25d'], _mix3({'rows': 32, 'decay': 'dyn', 'complex': True}), 'rows 48 A (real) + 48 dyn (real) + 32 dyn (complex), no gates; new arm'))
CAMPAIGNS = {'hm-mixc-insane-20260926': Campaign(id='hm-mixc-insane-20260926', manifest=DOCS / 'MQ_HM_MIXC_INSANE_20260926.json', family='hybrid_mamba', prefix='hm_mixc', tiers=('insane',), allowed_top=HM_TOP, allowed_mixer=HM_MIXER, module_defaults=False, definitions=ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status='owner 2026-09-26 (insane only): HM decay-law mixtures with a complex block on the hm-screen2 A_dyn base (head_subsystems 16, d_state 128, no gates): A_dyn_c25a (48 A + 48 dyn real + 32 A complex), A_dyn_c25d (32 dyn complex), plus A_dyn_gates (A_dyn rows, gates on at one_plus/pre/kaiming_uniform); widths in-band vs M1 by d_inner (P = 16); compared with hm-screen2 A_dyn / A_halfrot, hm-gates real_off / rot_off, SSD R2O2 complex / real'))}
