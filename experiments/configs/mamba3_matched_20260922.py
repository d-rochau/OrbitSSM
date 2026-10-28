
from dataclasses import replace
from pathlib import Path
import json
from experiments.configs.mqmkvar_baselines_20260920 import build_spec as baseline_spec
from experiments.specs import ModelSpec
MANIFEST = Path(__file__).resolve().parents[2] / 'implementationexperiments/MAMBA3_MATCHED_20260922.json'
JOBS = tuple(((t, 'official_mamba3', s) for t in ('insane', 'hard', 'easy') for s in (101, 202, 303)))
SMOKES = tuple((job for job in JOBS if job[2] == 101))

def arm_for_spec(spec):
    return json.loads(MANIFEST.read_text())['arms'][0]

def build_spec(tier, family, seed, root, *, smoke=False, resume_smoke=False):
    if tier not in ('insane', 'hard', 'easy') or family != 'official_mamba3':
        raise ValueError('Only Mamba3 matched replacement tiers are released')
    s = baseline_spec(tier, family, seed, root, smoke=smoke, resume_smoke=resume_smoke)
    name = f'matched_{tier}_mamba3_d298_i576_head64_s{seed}' + ('_smoke' if smoke else '')
    return replace(s, name=name, model=ModelSpec(family, arm_for_spec(s)['kwargs']), train=replace(s.train, kwargs={**s.train.kwargs, 'default_root_dir': str(Path(root) / 'trials' / name)}), tags=('paper1', 'matched-m3-20260922', tier, family))
