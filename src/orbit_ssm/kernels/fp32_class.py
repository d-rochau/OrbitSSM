"""fp32-class arithmetic guard for the prologue's torch code (chart and fold).

Prologue arithmetic is fp32-class -- fp32 data, IEEE fp32 (or
fp32-emulating split-precision) products, never plain TF32 / bf16 -- whatever the caller's
autocast region or process-wide matmul precision says.  ONE copy, shared by the torch
prologue routes that import it; the CUDA kernels choose their dot precision in-kernel
and never read these settings.
"""

from __future__ import annotations

import contextlib

import torch

__all__ = ["ieee_matmul", "fp32_class"]


@contextlib.contextmanager
def ieee_matmul():
    """IEEE fp32 GEMMs (no TF32 / bf16 reduced-precision products) inside the block,
    previous settings restored after.  Uses the torch >= 2.9 `fp32_precision` API
    only (mixing it with the legacy `allow_tf32` setter raises); older torch: the
    legacy flag.

    Under `torch.compile` the block is a no-op: the precision getter/setter is a pybind call
    Dynamo cannot trace (and a `with` block cannot span a graph break), and a compiled graph
    takes its matmul precision from the setting in force when it is COMPILED (Inductor reads it
    at lowering), not from a flag toggled while tracing.  Compile such a region under the
    default `highest` precision to keep the fp32-class guarantee."""
    if torch.compiler.is_compiling():
        yield
        return
    saved = []
    for be in (torch.backends.cuda.matmul, getattr(torch.backends.mkldnn, "matmul", None)):
        if be is None:
            continue
        if hasattr(be, "fp32_precision"):
            saved.append((be, "fp32_precision", be.fp32_precision))
            be.fp32_precision = "ieee"
        elif hasattr(be, "allow_tf32"):
            saved.append((be, "allow_tf32", be.allow_tf32))
            be.allow_tf32 = False
    try:
        yield
    finally:
        for be, name, old in reversed(saved):
            setattr(be, name, old)


@contextlib.contextmanager
def fp32_class(device_type: str):
    """Autocast off for `device_type` AND IEEE fp32 matmuls inside the block."""
    with torch.autocast(device_type, enabled=False), ieee_matmul():
        yield
