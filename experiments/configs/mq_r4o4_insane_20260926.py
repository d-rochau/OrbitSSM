
from __future__ import annotations
from pathlib import Path
from experiments.configs.mq_recollect_20260925 import COMPLEX_ROWS, GUARD, GUARD_EVIDENCE, REAL_ROWS, SSD_AXES, SSD_MIXER, SSD_TOP, _ssd
from experiments.mqmkvar.campaign import Campaign
DOCS = Path(__file__).resolve().parents[2] / 'docs' / 'experiments'
R4O4_BUDGET_S = 8000
ARMS = (_ssd('R4O4-real-d176-i384', 176, 384, 4, REAL_ROWS, R4O4_BUDGET_S, None, 'module default rows (one real head_dynamic block, spelled explicitly), R = O = 4; module defaults otherwise (the R2O2-real-d248-i448 definition at R = O = 4); new arm'), _ssd('R4O4-complex-d176-i384', 176, 384, 4, COMPLEX_ROWS, R4O4_BUDGET_S, None, 'complex head_dynamic half (frac 0.5) + real head_dynamic half, R = O = 4; module defaults otherwise (the R2O2-complex-d224-i512 definition at R = O = 4); new arm'))
CAMPAIGNS = {'r4o4-insane-20260926': Campaign(id='r4o4-insane-20260926', manifest=DOCS / 'MQ_R4O4_INSANE_20260926.json', family='ssd_mimo', prefix='r4o4', tiers=('insane', 'hard'), allowed_top=SSD_TOP, allowed_mixer=SSD_MIXER, arm_axes=SSD_AXES, module_defaults=True, definitions=ARMS, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status='owner 2026-09-26: SSD-MIMO R4O4 real / complex at module defaults (the 09-25 R2O2 arm definitions at R = O = 4), shared width d176 / i384 in the +-2 % band of M1; tiers insane (priority) + hard; compared with R1O1 / R2O2 complex / real, M1 / M2 / M3 and HM A_dyn'))}
