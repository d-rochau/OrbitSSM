
from dataclasses import replace
import json
from pathlib import Path
from experiments.configs.mqmkvar_baselines_20260920 import build_spec as baseline_spec
from experiments.specs import ModelSpec
MANIFEST = Path(__file__).resolve().parents[2] / 'implementationexperiments/P1_SSD_ABLATION_MATCHED_20260920.json'
SEEDS = (101, 202, 303)
ARMS = tuple((a['label'] for a in json.loads(MANIFEST.read_text())['arms']))
JOBS = tuple((('insane', arm, seed) for arm in ARMS for seed in SEEDS))
SMOKES = tuple((('insane', arm, 101) for arm in ARMS))

def arm_for(label):
    return next((a for a in json.loads(MANIFEST.read_text())['arms'] if a['label'] == label))

def arm_for_spec(spec):
    return arm_for(spec.tags[-1])

def build_spec(tier, arm, seed, root, *, smoke=False, resume_smoke=False):
    if tier != 'insane' or arm not in ARMS:
        raise ValueError('Only the released real R2 insane factorial may run')
    from experiments.configs._frozen import refuse
    refuse('P1_SSD_ABLATION_MATCHED_20260920.json', f'ssd_ablation arm {arm!r}')
    base = baseline_spec(tier, 'official_mamba1', seed, root, smoke=smoke, resume_smoke=resume_smoke)
    name = f'ablation_{arm}_s{seed}' + ('_smoke' if smoke else '')
    train = replace(base.train, kwargs={**base.train.kwargs, 'default_root_dir': str(Path(root) / 'trials' / name)})
    return replace(base, name=name, train=train, model=ModelSpec('ssd_mimo', arm_for(arm)['kwargs']), tags=('paper1', 'ssd-factorial-20260920', tier, arm))
