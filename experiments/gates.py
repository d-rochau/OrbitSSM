
from __future__ import annotations
from typing import Callable, Sequence
UNMATCHED_REFERENCES = {'attention', 'llama3', 'mlstm_large'}
DEFAULT_TOL = 0.02

class ParamGateError(RuntimeError):
    pass

def assert_param_gate(name: str, arm_backbone: int, anchor_backbone: int, tol: float=DEFAULT_TOL) -> None:
    if name in UNMATCHED_REFERENCES:
        return
    limit = int(anchor_backbone * (1.0 + tol))
    if arm_backbone > limit:
        raise ParamGateError(f'{name}: backbone {arm_backbone:,} exceeds the Mamba-1 anchor {anchor_backbone:,} (+{tol:.0%} = {limit:,}). Reduce width.')

def maximal_width(backbone_at: Callable[[int], int], anchor_backbone: int, widths: Sequence[int], tol: float=DEFAULT_TOL) -> int:
    limit = int(anchor_backbone * (1.0 + tol))
    ok = [d for d in sorted(widths) if backbone_at(d) <= limit]
    if not ok:
        raise ParamGateError(f'no width in {list(widths)} fits under the anchor limit {limit:,}; apply the granularity exception explicitly.')
    return ok[-1]
