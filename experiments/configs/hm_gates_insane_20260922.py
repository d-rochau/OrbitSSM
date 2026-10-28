
from dataclasses import replace
import json
from pathlib import Path
from experiments.configs.mqmkvar_baselines_20260920 import build_spec as baseline_spec
from experiments.specs import ModelSpec
MANIFEST = Path(__file__).resolve().parents[2] / 'implementationexperiments/HM_GATES_INSANE_MANIFEST_20260922.json'
ARMS = tuple((a['arm'] for a in json.loads(MANIFEST.read_text())['arms']))
JOBS = tuple((('insane', a, s) for a in ARMS for s in (101, 202, 303)))
SMOKES = tuple((j for j in JOBS if j[2] == 101))

def arm_for_spec(spec):
    name = next((tag for tag in spec.tags if tag in ARMS))
    a = next((a for a in json.loads(MANIFEST.read_text())['arms'] if a['arm'] == name))
    return dict(a, non_embedding=a['backbone'])

def build_spec(tier, arm, seed, root, *, smoke=False, resume_smoke=False):
    if tier != 'insane' or arm not in ARMS:
        raise ValueError((tier, arm))
    from experiments.configs._frozen import refuse
    refuse('HM_GATES_INSANE_MANIFEST_20260922.json', f'hm_gates_insane arm {arm!r}')
    s = baseline_spec(tier, 'official_mamba1', seed, root, smoke=smoke, resume_smoke=resume_smoke)
    a = next((a for a in json.loads(MANIFEST.read_text())['arms'] if a['arm'] == arm))
    kw = a['kwargs']
    model_kw = dict(d_model=kw['d_model'], n_layers=kw['n_layers'], d_state=kw['d_state'], d_conv=kw['d_conv'], bc_conv=False, bc_norm=True, tie_word_embeddings=True, mixer_kwargs=dict(kw['ssm_kwargs'], expand=None, d_inner=kw['d_inner']))
    name = f'hm_gates_{arm}_s{seed}' + ('_smoke' if smoke else '')
    return replace(s, name=name, model=ModelSpec('hybrid_mamba', model_kw), train=replace(s.train, kwargs={**s.train.kwargs, 'default_root_dir': str(Path(root) / 'trials' / name)}), tags=('paper1', 'hm-gates-20260922', arm))
