"""`torch.compile` compatibility: opaque (graph-break) boundaries around the Triton launchers.

Dynamo cannot trace the SSD and MH kernel launchers (kwargs-templated launcher arguments for
`_fused_chunk_fwd_kernel`; `@gluon.jit` sources that Inductor's kernel reload cannot re-emit),
so `dynamo_opaque` wraps each family's public entry point in `torch.compiler.disable`.  In eager
it is a pass-through (identical numerics and graph); under `torch.compile` it is a graph break
and the `autograd.Function` runs eagerly.  It does not give Inductor the kernel, so
``fullgraph=True`` and CUDA-graph trees cannot span the break.

The SSD scan has a registered custom-op pair (`kernels/ssd_mimo/registered_scan.py`,
``orbit_ssm::ssd_scan_fwd`` / ``ssd_scan_bwd``) and skips the boundary unless
``HYBRID_SSM_COMPILE_CUSTOMOP=0``.  `token_tails` is traceable and not opaque by default
(`tails_boundary`): an opaque tail would split the mixer into two graphs, and the channel-major
drive view crossing the cut costs two 64 MiB copies.

Env switches: ``HYBRID_SSM_COMPILE_OPAQUE=0`` disables all boundaries (A/B measurement);
``HYBRID_SSM_COMPILE_TAILS_OPAQUE=1`` makes `token_tails` opaque.
"""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

import torch

_F = TypeVar("_F", bound=Callable[..., Any])

#: Read once at import: the decorators run at definition time.
_OPAQUE = os.environ.get("HYBRID_SSM_COMPILE_OPAQUE", "1") == "1"


def dynamo_opaque(fn: _F) -> _F:
    """Mark ``fn`` as a boundary TorchDynamo must not trace through (eager: unchanged).

    Applied where the autograd node begins (the families' entry points, the `SSDMimoSSM`
    combined node), not inside kernel files.
    """
    if not _OPAQUE:
        return fn
    return torch.compiler.disable(fn)          # type: ignore[return-value]


#: `token_tails`' boundary is switched separately and is off by default (see module docstring).
_TAILS_OPAQUE = os.environ.get("HYBRID_SSM_COMPILE_TAILS_OPAQUE", "0") == "1"


def tails_boundary_enabled() -> bool:
    """Whether `token_tails` keeps a `dynamo_opaque` boundary."""
    return _TAILS_OPAQUE and _OPAQUE


def tails_boundary(fn: _F) -> _F:
    """`dynamo_opaque` for the fused tail law, off by default: its `@triton.jit` launcher passes
    every argument positionally, so Dynamo can trace it.  ``HYBRID_SSM_COMPILE_TAILS_OPAQUE=1`` enables it."""
    return dynamo_opaque(fn) if _TAILS_OPAQUE else fn
