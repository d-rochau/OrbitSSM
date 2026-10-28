
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_recollect_20260925 import COMPLEX_ROWS, GUARD, GUARD_EVIDENCE, REAL_ROWS, SSD_AXES, SSD_MIXER, SSD_TOP
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'

def _arm(arm, d_model, d_inner, rows, rotation, note):
    mixer = dict(R=2, O=2, rows=rows, expand=d_inner / d_model)
    if rotation is not None:
        mixer['rotation'] = rotation
    return dict(arm=arm, family='ssd_mimo', budget_s=6000, note=note, supersedes=None, kwargs=dict(d_model=d_model, n_layers=2, d_state=128, mixer_kwargs=mixer))
ARMS = (_arm('R2O2-complex-pair-d224-i512', 224, 512, COMPLEX_ROWS, {'omega_layout': 'pair', 'theta_init': 'kaiming'}, "ssd-complex-20260925's R2O2 complex arm (omega per pair, kaiming W_theta), spelled explicitly"), _arm('R2O2-complex-hpair-d224-i512', 224, 512, COMPLEX_ROWS, {'omega_layout': 'head_pair', 'theta_init': 'small'}, 'omega per (head, pair): theta_freq (H, K_rot); W_theta kaiming x 0.1 sqrt(3)'), _arm('R2O2-complex-head-d224-i512', 224, 512, COMPLEX_ROWS, {'omega_layout': 'head', 'theta_init': 'small'}, 'ONE complex eigenvalue per head: theta_proj emits H values, theta_freq (H,), all rotated pairs of a head carry one phase; W_theta kaiming x 0.1 sqrt(3)'), _arm('R2O2-real-d248-i448', 248, 448, REAL_ROWS, None, "ssd-real-20260925's R2O2 real arm (module default rows)"))
CAMPAIGNS = {'rot-longgap-20260926': Campaign(id='rot-longgap-20260926', manifest=DOCS / 'MQ_ROT_LONGGAP_20260926.json', family='ssd_mimo', prefix='rot_longgap', tiers=('longgap',), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER | {'rotation'}, arm_axes=SSD_AXES | {'rotation'}, module_defaults=True, definitions=ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status='owner 2026-09-26 (ROT_RECALL): rotation layouts pair / head_pair / head vs real on the long-gap tier (constant filler); gap-binned accuracy by the collect job'))}
