
from __future__ import annotations
import math
from dataclasses import dataclass
import torch

def make_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d)):
            w = getattr(m, 'weight', None)
            if w is not None and (not getattr(w, '_no_weight_decay', False)):
                decay_ids.add(id(w))
    decay, no_decay = ([], [])
    seen: set[int] = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if id(p) in decay_ids else no_decay).append(p)
    return [{'params': decay, 'weight_decay': weight_decay, '_name': 'decay'}, {'params': no_decay, 'weight_decay': 0.0, '_name': 'no_decay'}]

def describe_param_groups(model: torch.nn.Module) -> dict[str, list[str]]:
    groups = make_param_groups(model, 0.1)
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {g['_name']: sorted((by_id[id(p)] for p in g['params'])) for g in groups}

@dataclass
class LRSchedule:
    peak_lr: float
    total_tokens: int
    warmup_tokens: int
    min_lr: float = 1e-05
    constant: bool = False
    decay_from: int = 0
    decay_to: int = 0

    def lr_at(self, tokens_seen: int) -> float:
        if tokens_seen < self.warmup_tokens:
            return self.peak_lr * tokens_seen / max(1, self.warmup_tokens)
        if self.decay_from > 0 and self.decay_to > self.decay_from and (tokens_seen >= self.decay_from):
            progress = (tokens_seen - self.decay_from) / (self.decay_to - self.decay_from)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine
        if self.constant:
            return self.peak_lr
        decay_tokens = max(1, self.total_tokens - self.warmup_tokens)
        progress = (tokens_seen - self.warmup_tokens) / decay_tokens
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def apply(self, optimizer: torch.optim.Optimizer, tokens_seen: int) -> float:
        lr = self.lr_at(tokens_seen)
        for g in optimizer.param_groups:
            g['lr'] = lr
        return lr

def make_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float=0.1, betas: tuple[float, float]=(0.9, 0.95)) -> torch.optim.AdamW:
    groups = make_param_groups(model, weight_decay)
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
