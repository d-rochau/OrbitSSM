
from __future__ import annotations
import json
import os
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
_CKPT_RE = re.compile('ckpt_step(\\d+)\\.pt$')

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model

def _is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def rng_state() -> dict[str, Any]:
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def save(run_dir: str | os.PathLike, *, step: int, tokens_seen: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, model_cfg, extra: dict[str, Any] | None=None) -> Path | None:
    if dist.is_initialized():
        dist.barrier()
    if not _is_rank0():
        return None
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    final = run / f'ckpt_step{step:08d}.pt'
    tmp = run / f'.{final.name}.tmp'
    blob = {'step': step, 'tokens_seen': tokens_seen, 'model': _unwrap(model).state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'model_cfg': asdict(model_cfg) if hasattr(model_cfg, '__dataclass_fields__') else dict(model_cfg), 'extra': extra or {}}
    torch.save(blob, tmp)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)
    latest = run / 'latest.json'
    latest_tmp = run / '.latest.json.tmp'
    json.dump({'path': final.name, 'step': step, 'tokens_seen': tokens_seen}, open(latest_tmp, 'w'))
    os.replace(latest_tmp, latest)
    keep = int(os.environ.get('KEEP_CKPTS', '2'))
    if keep > 0:
        cks = sorted(run.glob('ckpt_step*.pt'), key=lambda p: int(_CKPT_RE.search(p.name).group(1)))
        for old in cks[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass
    return final

def find_latest(run_dir: str | os.PathLike) -> Path | None:
    run = Path(run_dir)
    pointer = run / 'latest.json'
    if pointer.exists():
        rec = json.load(open(pointer))
        p = run / rec['path']
        if p.exists():
            return p
    cands = []
    if run.exists():
        for p in run.iterdir():
            m = _CKPT_RE.search(p.name)
            if m:
                cands.append((int(m.group(1)), p))
    if not cands:
        return None
    return max(cands)[1]

def load(path: str | os.PathLike, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None, map_location: str | torch.device='cpu') -> dict[str, Any]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    raw = _unwrap(model)
    before = [id(p) for p in raw.parameters()]
    raw.load_state_dict(blob['model'])
    if (optimizer is not None or raw is not model) and [id(p) for p in raw.parameters()] != before:
        raise RuntimeError(f"loading {path} replaced model parameters (a HybridMamba core adopted the checkpoint's proj_source and rebuilt its token maps), but the optimizer / DDP / torch.compile wrapper was built on the old ones.  Pass the checkpoint's proj_source explicitly in the mixer kwargs (see the load warning), or build the optimizer and wrappers after loading.")
    if optimizer is not None and 'optimizer' in blob:
        optimizer.load_state_dict(blob['optimizer'])
    if 'rng' in blob:
        restore_rng_state(blob['rng'])
    return {'step': blob['step'], 'tokens_seen': blob['tokens_seen'], 'model_cfg': blob.get('model_cfg'), 'extra': blob.get('extra', {})}
