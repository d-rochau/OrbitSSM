
from __future__ import annotations
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT) + '/src')
from experiments.configs import register
from experiments.mqmkvar.presets import mqmkvar_lightning_integration
from experiments.specs import DataSpec, ExperimentSpec, ModelSpec, OptimizerSpec, ResourceSpec, SchedulerSpec, SeedSpec, TaskSpec, TrainSpec
P1_JSON_PATH = REPO_ROOT / 'docs' / 'experiments' / 'P1_CONFIG_PRIORITIES_20260906.json'
DEFAULT_SEEDS = (101, 202, 303)
DATA_SEED = 12345
FINAL_TEST_SEED = 99999
DEFAULT_ROOT = './local-data'
ROWS_IN_SCOPE = ('S-R1-real', 'S-R2-real', 'S-R4-real', 'S-R1-m3', 'S-R2-m3', 'S-R2-m3x', 'H-A', 'H-C', 'H-D', 'H-Mix', 'H-G0', 'H-N0', 'B-M1', 'B-M2', 'B-M3', 'B-Attn', 'B-Attn-clip', 'B-Attn-4L-clip')
DIAG_ROWS = {'B-Attn-clip': {'n_layers': 2}, 'B-Attn-4L-clip': {'n_layers': 4}}
S_R2_M3_GRID_STEP = 8

def _git_sha() -> str:
    return 'anonymous-release'

@lru_cache(maxsize=1)
def _load_p1_json() -> dict[str, Any]:
    with open(P1_JSON_PATH, 'r', encoding='utf-8') as handle:
        return json.load(handle)

@lru_cache(maxsize=1)
def _rows_by_id() -> dict[str, dict[str, Any]]:
    data = _load_p1_json()
    return {row['id']: row for row in data['rows']}

def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    return tuple((item.strip() for item in raw.split(',') if item.strip()))

@lru_cache(maxsize=1)
def scan_s_r2_m3_width() -> dict[str, Any]:
    from experiments.configs._frozen import refuse
    refuse('P1_CONFIG_PRIORITIES_20260906.json', 'mqmkvar_p1 S-R2-m3 width scan')
    from experiments.models.zoo import build_mqmkvar_model, count_parameter_breakdown
    row = _rows_by_id()['S-R2-m3']
    mixer_kwargs = dict(row['mixer_kwargs'])
    data = _load_p1_json()
    gate_upper = int(data['gate_upper'])
    vocab_size = int(data['data_vocab_size'])
    n_layers = int(data['n_layers'])
    data_spec = DataSpec(task='mqmkvar', mode='lightning', kwargs={'vocab_size': vocab_size})
    best = None
    for width in range(400, 63, -S_R2_M3_GRID_STEP):
        model_kwargs = {'d_model': width, 'n_layers': n_layers, 'd_state': 64, 'd_conv': 4, 'mixer_kwargs': mixer_kwargs}
        model_spec = ModelSpec(name='ssd_mimo', kwargs=dict(model_kwargs))
        try:
            model = build_mqmkvar_model(model_spec, data_spec)
        except Exception:
            continue
        counts = count_parameter_breakdown(model)
        non_embedding = int(counts.get('non_embedding', counts['total']))
        if non_embedding <= gate_upper:
            best = {'width': width, 'counts': {'total': int(counts['total']), 'embedding': int(counts.get('embedding', 0)), 'non_embedding': non_embedding}, 'model_kwargs': model_kwargs}
            break
    if best is None:
        raise RuntimeError(f'S-R2-m3 width scan found no legal width <= gate_upper ({gate_upper}) in range [64, 400) step {S_R2_M3_GRID_STEP}')
    return best

def _model_spec_for_row(row_id: str) -> tuple[str, dict[str, Any]]:
    if row_id in DIAG_ROWS:
        family, kwargs = _model_spec_for_row('B-Attn')
        kwargs = dict(kwargs, **DIAG_ROWS[row_id])
        return (family, kwargs)
    row = _rows_by_id()[row_id]
    family = row['family']
    if family in ('ssd_mimo', 'hybrid_mamba'):
        from experiments.configs._frozen import refuse
        refuse('P1_CONFIG_PRIORITIES_20260906.json', f'mqmkvar_p1 row {row_id!r}')
    selected = row.get('selected')
    if not selected or 'model_kwargs' not in selected:
        raise ValueError(f'Row {row_id!r} has no selected.model_kwargs in the P1 JSON; cannot launch it verbatim.')
    return (family, dict(selected['model_kwargs']))

def _data_spec() -> DataSpec:
    return DataSpec(task='mqmkvar', mode='lightning', kwargs={'train_layout': [(2048, 100000, 32)], 'test_layout': [(2048, 10000, 32)], 'vocab_size': 8192, 'min_key_len': 5, 'max_key_len': 12, 'min_list_len': 5, 'max_list_len': 12, 'interleaving_frac': 0.8, 'num_kv_queries': -1, 'shared_token_prob': 0.0, 'query_mode': 'streaming', 'random_non_queries': False, 'power_a': 0.01, 'cache_dir': os.environ.get('MQMKVAR_P1_CACHE_DIR') or None, 'final_test_seed': FINAL_TEST_SEED})

def _train_spec(*, root_dir: str, max_steps: int | None, clip: float | None=None) -> TrainSpec:
    kwargs = {'trainer_backend': 'lightning', 'precision': 'bf16-mixed', 'val_check_interval': None, 'enable_checkpointing': False, 'default_root_dir': root_dir, 'gradient_clip_val': clip}
    if max_steps is not None:
        return TrainSpec(batch_size=32, learning_rate=0.001, weight_decay=0.0, max_steps=int(max_steps), lr_warmup_epochs=0, kwargs=kwargs)
    epochs = int(os.environ.get('MQMKVAR_P1_EPOCHS', '16'))
    lr = float(os.environ.get('MQMKVAR_P1_LR', '1e-3'))
    return TrainSpec(batch_size=32, learning_rate=lr, weight_decay=0.0, epochs=epochs, lr_warmup_epochs=2, kwargs=kwargs)

def _scheduler_spec() -> SchedulerSpec:
    return SchedulerSpec(name='CosineAnnealingLR', interval='epoch', kwargs={'eta_min': 0.0001})

def _task_spec() -> TaskSpec:
    return TaskSpec(name='mqmkvar_p1', loss='ce', metrics=('loss', 'accuracy'), primary_metric='val/accuracy', primary_mode='max', kwargs={'ignore_index': -100, 'slice_keys': ('num_kv_pairs',)})

def build_p1_experiments() -> list[ExperimentSpec]:
    row_ids = _env_list('MQMKVAR_P1_ROWS', ROWS_IN_SCOPE)
    unknown = set(row_ids) - set(ROWS_IN_SCOPE)
    if unknown:
        raise ValueError(f'MQMKVAR_P1_ROWS has rows outside the 15-row scope: {sorted(unknown)}')
    seed_strs = _env_list('MQMKVAR_P1_SEEDS', tuple((str(s) for s in DEFAULT_SEEDS)))
    seeds = tuple((int(s) for s in seed_strs))
    max_steps_raw = os.environ.get('MQMKVAR_P1_MAX_STEPS', '').strip()
    max_steps = int(max_steps_raw) if max_steps_raw else None
    root = os.environ.get('MQMKVAR_P1_ROOT', DEFAULT_ROOT)
    root_path = Path(root)
    if not root_path.is_absolute():
        raise ValueError(f'MQMKVAR_P1_ROOT must be absolute (Ray chdirs workers), got {root!r}')
    integration = mqmkvar_lightning_integration()
    task = _task_spec()
    scheduler = _scheduler_spec()
    data_spec = _data_spec()
    git_sha = _git_sha()
    experiments: list[ExperimentSpec] = []
    for row_id in row_ids:
        family, model_kwargs = _model_spec_for_row(row_id)
        model_spec = ModelSpec(name=family, kwargs=dict(model_kwargs))
        for seed in seeds:
            diag = ''
            if os.environ.get('MQMKVAR_P1_EPOCHS') or os.environ.get('MQMKVAR_P1_LR'):
                diag = f"_ep{os.environ.get('MQMKVAR_P1_EPOCHS', '16')}_lr{os.environ.get('MQMKVAR_P1_LR', '1e-3')}"
            run_root = str(root_path / (row_id + diag) / f's{seed}')
            train = _train_spec(root_dir=run_root, max_steps=max_steps, clip=1.0 if row_id.endswith('-clip') else None)
            experiments.append(ExperimentSpec(name=f'mqmkvar_p1_{row_id}{diag}_s{seed}', model=model_spec, data=data_spec, integration=integration, task=task, train=train, seeds=SeedSpec(model_seed=seed, data_seed=DATA_SEED), optimizer=OptimizerSpec(name='AdamW', kwargs={'betas': (0.9, 0.95)}), scheduler=scheduler, resources=ResourceSpec(gpus=1, cpus=4, nodes=1, ddp=False), tags=('mqmkvar', 'p1', row_id, f's{seed}', f'sha{git_sha[:12]}')))
    return experiments
register('mqmkvar_p1')(build_p1_experiments)
