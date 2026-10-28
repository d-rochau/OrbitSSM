
from dataclasses import replace
from pathlib import Path
import json
from experiments.configs.mqmkvar_baselines_20260920 import build_spec as baseline_spec
from experiments.specs import ModelSpec
MANIFEST = Path(__file__).resolve().parents[2] / 'implementationexperiments/MQ_STOCK_ABOVE_20260922.json'
JOBS = tuple(((t, f, s) for t in ('insane', 'hard', 'easy') for f in ('official_mamba2', 'official_mamba3') for s in (101, 202, 303)))
SMOKES = tuple((job for job in JOBS if job[2] == 101))

def arm_for_spec(spec):
    return next((a for a in json.loads(MANIFEST.read_text())['arms'] if a['family'] == spec.model.name))

def build_spec(tier, family, seed, root, *, smoke=False, resume_smoke=False):
    if tier not in ('insane', 'hard', 'easy') or family not in ('official_mamba2', 'official_mamba3'):
        raise ValueError('Only stock M2/M3 replacement tiers are released')
    s = baseline_spec(tier, family, seed, root, smoke=smoke, resume_smoke=resume_smoke)
    name = f'stock_above_{tier}_{family}_d320_i640_h64_s{seed}' + ('_smoke' if smoke else '')
    return replace(s, name=name, model=ModelSpec(family, arm_for_spec(s)['kwargs']), train=replace(s.train, kwargs={**s.train.kwargs, 'default_root_dir': str(Path(root) / 'trials' / name)}), tags=('paper1', 'stock-above-20260922', tier, family))
