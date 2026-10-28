
from dataclasses import replace
from pathlib import Path
import json
from experiments.configs.mqmkvar_baselines_20260920 import build_spec as baseline_spec
from experiments.specs import ModelSpec
MANIFEST = Path(__file__).resolve().parents[2] / 'implementationexperiments/MAMBA3_HEAD64_20260921.json'
JOBS = tuple((('insane', 'official_mamba3', s) for s in (101, 202, 303)))
SMOKES = (JOBS[0],)

def arm_for_spec(spec):
    return json.loads(MANIFEST.read_text())['arms'][0]

def build_spec(tier, family, seed, root, *, smoke=False, resume_smoke=False):
    if tier != 'insane' or family != 'official_mamba3':
        raise ValueError('Only insane Mamba3 head64 is released for this diagnostic')
    s = baseline_spec(tier, family, seed, root, smoke=smoke, resume_smoke=resume_smoke)
    name = f'diagnostic_insane_mamba3_head64_s{seed}' + ('_smoke' if smoke else '')
    return replace(s, name=name, model=ModelSpec(family, arm_for_spec(s)['kwargs']), train=replace(s.train, kwargs={**s.train.kwargs, 'default_root_dir': str(Path(root) / 'trials' / name)}), tags=('paper1', 'diagnostic-head64', tier, family))
