"""Keep LLVM's SLP vectorizer ON for the MH kernels on sm_90 (Triton 3.8).

Triton 3.8's `make_llir` (backends/nvidia/compiler.py) turns the SLP vectorizer
OFF on capability 90 whenever the TTGIR contains `bf16` and `tt.reduce` -- a
workaround for llvm/llvm-project#203741, SLP reassociating ordered fadd
reductions.  Every bf16 MH kernel matches that test, and the compile-only
effect: with SLP off, bf16 tiles are scalarised, the forward's row-ahead
prefetch is unpacked in the iteration that issued it, and load-to-use windows
halve.

What the workaround protects is summation ORDER.  The MH kernels' fp32 sums
already have no fixed order across builds (tree reductions, fp32 atomics;
determinism is not required), and their fp32 builds -- which never match the
test -- run with SLP on.

Mechanism: Triton's own stages hook (`knobs.runtime.add_stages_inspection_hook`)
wraps the `llir` stage and, for an MH kernel only, runs it with
`optimize_module(..., disable_slp_vectorizer=False)`.  The hook adds nothing to
the cache key (it returns an empty key, so no other kernel's cache entry
moves); instead the launchers pass `SLP=active()` as a kernel constexpr, so a
binary built with SLP forced on and one built without are different
specialisations and can never be served for each other.

`ENABLED` is a module attribute (measurement and tests set it); the hook is
installed lazily at the first launch that asks.  It stands down -- `active()`
returns False -- when Triton lacks the pieces it wraps or another hook owns
the slot.
"""
from __future__ import annotations

import os

# Default ON.  What the Triton workaround protects is summation ORDER
# (llvm#203741), and MH's fp32 sums have no fixed order anyway.
# HYBRID_MH_SLP=0 turns it off for a process; code may still assign the
# attribute.  Read once at import: the SLP constexpr and the per-compile
# check below both follow the attribute.
ENABLED = os.environ.get("HYBRID_MH_SLP", "1").strip().lower() not in (
    "", "0", "off", "false", "no")

_MH_KERNELS = ("@_mh_fwd_kernel", "@_mh_bwd_kernel")
_STATE = {"installed": None}


class _SLPOn:
    """`triton.backends.nvidia.compiler.llvm` with SLP forced on."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def optimize_module(self, *args, **kwargs):
        kwargs["disable_slp_vectorizer"] = False
        return self._real.optimize_module(*args, **kwargs)


def _install():
    try:
        from triton import knobs
        import triton.backends.nvidia.compiler as nvc
    except Exception:                          # noqa: BLE001 -- no triton
        return False
    if not hasattr(nvc, "llvm") or not hasattr(knobs.runtime,
                                               "add_stages_inspection_hook"):
        return False
    if knobs.runtime.add_stages_inspection_hook is not None:
        return False                           # somebody else's hook

    def hook(*args):
        if not args:                           # the cache-key query
            return "", ""
        _backend, stages, _options, _language, capability = args
        if capability != 90 or "llir" not in stages:
            return
        inner = stages["llir"]

        def llir(src, metadata):
            # Re-checked per COMPILE, not per install: once ENABLED goes off
            # the launchers pass SLP=False, and that build must be the plain
            # one (the hook stays installed and adds nothing to the cache key,
            # so forcing SLP here would file an SLP build under the SLP-off
            # key for every later process).
            if not _enabled():
                return inner(src, metadata)
            text = src.str_nodebug()
            if not any(k in text for k in _MH_KERNELS):
                return inner(src, metadata)
            real = nvc.llvm
            nvc.llvm = _SLPOn(real)
            try:
                return inner(src, metadata)
            finally:
                nvc.llvm = real
        stages["llir"] = llir

    knobs.runtime.add_stages_inspection_hook = hook
    return True


def _enabled():
    return bool(ENABLED) and ENABLED not in ("0", "off", "False")


def active(capability):
    """True when an MH kernel for `capability` (e.g. 90) is compiled with SLP
    forced on; the launchers pass it as the kernel constexpr `SLP`."""
    if capability != 90 or not _enabled():
        return False
    if _STATE["installed"] is None:
        _STATE["installed"] = _install()
    return bool(_STATE["installed"])
