
from __future__ import annotations
import os
from typing import Any, Callable, TypeVar
import torch
_F = TypeVar('_F', bound=Callable[..., Any])
_OPAQUE = os.environ.get('HYBRID_SSM_COMPILE_OPAQUE', '1') == '1'

def opaque_boundary_enabled() -> bool:
    return _OPAQUE

def dynamo_opaque(fn: _F) -> _F:
    if not _OPAQUE:
        return fn
    return torch.compiler.disable(fn)
_TAILS_OPAQUE = os.environ.get('HYBRID_SSM_COMPILE_TAILS_OPAQUE', '0') == '1'

def tails_boundary_enabled() -> bool:
    return _TAILS_OPAQUE and _OPAQUE

def tails_boundary(fn: _F) -> _F:
    return dynamo_opaque(fn) if _TAILS_OPAQUE else fn
