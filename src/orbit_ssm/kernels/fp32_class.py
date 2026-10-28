
from __future__ import annotations
import contextlib
import torch
__all__ = ['ieee_matmul', 'fp32_class']

@contextlib.contextmanager
def ieee_matmul():
    saved = []
    for be in (torch.backends.cuda.matmul, getattr(torch.backends.mkldnn, 'matmul', None)):
        if be is None:
            continue
        if hasattr(be, 'fp32_precision'):
            saved.append((be, 'fp32_precision', be.fp32_precision))
            be.fp32_precision = 'ieee'
        elif hasattr(be, 'allow_tf32'):
            saved.append((be, 'allow_tf32', be.allow_tf32))
            be.allow_tf32 = False
    try:
        yield
    finally:
        for be, name, old in reversed(saved):
            setattr(be, name, old)

@contextlib.contextmanager
def fp32_class(device_type: str):
    with torch.autocast(device_type, enabled=False), ieee_matmul():
        yield
