
from dataclasses import replace
import json
import os
from pathlib import Path
from experiments.configs.mqmkvar_p1 import _data_spec, _train_spec, _task_spec, _scheduler_spec
from experiments.specs import ExperimentSpec, ModelSpec, SeedSpec, OptimizerSpec, ResourceSpec
from experiments.mqmkvar.presets import mqmkvar_lightning_integration
MANIFEST = Path(__file__).resolve().parents[2] / 'docs/experiments/P1_MQMKVAR_MATCHING_CANDIDATE_20260920.json'
SEEDS = (101, 202, 303)
TIERS = ('insane', 'hard', 'easy')
FAMILIES = ('official_mamba1', 'official_mamba2', 'official_mamba3')
JOBS = tuple(((tier, family, seed) for tier in TIERS for family in FAMILIES for seed in SEEDS))
SMOKES = tuple(((tier, family, 101) for tier in TIERS for family in FAMILIES))

def arm_for(family):
    if family not in FAMILIES:
        raise ValueError(f'{family} is not released for this baseline collection')
    manifest = json.loads(MANIFEST.read_text())
    arm = next((a for a in manifest['arms'] if a['family'] == family))
    assert arm['role'] == 'matched'
    assert arm['kwargs']['d_state'] == 128
    assert arm['kwargs']['n_layers'] == 2
    return arm

def build_spec(tier, family, seed, root, *, smoke=False, resume_smoke=False):
    if tier not in TIERS or seed not in SEEDS:
        raise ValueError((tier, seed))
    for key in ('MQMKVAR_P1_EPOCHS', 'MQMKVAR_P1_LR', 'MQMKVAR_P1_MAX_STEPS', 'MQMKVAR_P1_CACHE_DIR'):
        if os.environ.get(key):
            raise ValueError(f'Frozen recipe forbids override {key}')
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('run root must be absolute')
    name = f'matched_{tier}_{family}_s{seed}' + ('_smoke' if smoke else '')
    data = _data_spec()
    dk = dict(data.kwargs, cache_dir=str(root / 'data_cache' / tier))
    train = _train_spec(root_dir=str(root / 'trials' / name), max_steps=None)
    if tier != 'insane':
        dk.update(train_layout=[(1024, 100000, 32)], test_layout=[(1024, 10000, 32)], min_key_len=2, max_key_len=4, min_list_len=2, max_list_len=6, interleaving_frac=0.4)
        train = replace(train, batch_size=64, epochs=32)
    if tier == 'easy':
        dk.update(generator='mqar', num_kv_queries=0, key_len=1, val_len=1, min_key_len=1, max_key_len=1, min_list_len=1, max_list_len=1, interleaving_frac=0.0)
    if smoke:
        length = dk['train_layout'][0][0]
        dk.update(train_layout=[(length, 256, 32)], test_layout=[(length, 32, 32)], cache_dir=None)
        train = replace(train, epochs=None, max_steps=64 if resume_smoke else 60, lr_warmup_epochs=0)
    train = replace(train, kwargs={**train.kwargs, 'enable_progress_bar': False, 'enable_checkpointing': True})
    return ExperimentSpec(name=name, model=ModelSpec(family, arm_for(family)['kwargs']), data=replace(data, kwargs=dk), train=train, integration=mqmkvar_lightning_integration(), task=_task_spec(), seeds=SeedSpec(model_seed=seed, data_seed=12345), optimizer=OptimizerSpec('AdamW', {'betas': (0.9, 0.95)}), scheduler=_scheduler_spec(), resources=ResourceSpec(gpus=1, cpus=4), tags=('paper1', 'matched-baselines-20260920', tier, family))
