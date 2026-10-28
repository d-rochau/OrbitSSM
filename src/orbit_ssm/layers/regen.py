"""Opt-in held-memory regeneration: save a RECIPE instead of a tensor.

`SSD_MIMO_HELD_REGEN`: unset = "all" on the real R = O = 1 route, "off"/"0"/"" = save everything, "x" | "out" | "all" = that part,
on every route. Trades about 4 % time for roughly a quarter less held memory.

Inside a `scope()` a tensor registered with `register(t, fn)` is not kept alive by
the autograd saves that reference its storage: `pack` swaps it for a token
carrying `fn` and the view geometry, and `unpack` (backward) calls `fn()` -- the SAME
kernel on the same saved inputs, so the rebuilt bytes are identical -- and
re-creates the saved view over the result.

  x   the post-conv x of the SSD mixer (fn = the causal conv on its saved input)
  out the out_proj input (fn = the gated-RMSNorm epilogue on the saved Y_pre / z)

Safety: an entry matches only while the registered tensor object is alive (weakref),
by storage address + dtype + storage size, and the rebuilt storage is checked
against the original size/dtype at unpack.
"""
from __future__ import annotations

import contextlib
import os
import threading
import weakref

import torch

_tls = threading.local()


def mode(real: bool = True) -> str:
    """Effective mode: unset means "all" when `real` (the R = O = 1 route), off
    elsewhere; an explicit value applies everywhere."""
    v = os.environ.get("SSD_MIMO_HELD_REGEN")
    if v is None:
        return "all" if real else ""
    v = v.strip().lower()
    return "" if v in ("off", "0", "false", "no") else v


def wants(kind: str, real: bool = True) -> bool:
    """Is regeneration of `kind` ("x" or "out") active?"""
    m = mode(real)
    return m == "all" or m == kind


def _key(t):
    st = t.untyped_storage()
    return (st.data_ptr(), t.dtype, st.nbytes())


def register(t: torch.Tensor, fn, track: bool = True) -> bool:
    """Register `t` as regenerable by `fn()` inside the current `scope`; False if no scope."""
    reg = getattr(_tls, "reg", None)
    if reg is None or not t.is_cuda:
        return False
    # track=False: the producing object is not kept alive (an op output
    # re-wrapped on the way to its consumer); the entry then fires once, on
    # the first match.
    reg[_key(t)] = (weakref.ref(t) if track else None, fn)
    return True


@contextlib.contextmanager
def scope(enabled: bool):
    """Enable regeneration of registered tensors for the autograd saves made inside."""
    if not enabled or not torch.is_grad_enabled() or torch.compiler.is_compiling():
        # Not traceable under torch.compile (the storage-address key graph-breaks);
        # Inductor owns the saved-activation policy there.
        yield
        return
    prev = getattr(_tls, "reg", None)
    reg = {}
    _tls.reg = reg

    def pack(t):
        if os.environ.get("HELD_REGEN_DEBUG") and t.is_cuda and t.numel() * t.element_size() > 2**20:
            hit = reg.get(_key(t))
            print("pack", tuple(t.shape), t.dtype, _key(t)[0], "hit" if hit else "miss",
                  "", "| reg:", [k[0] for k in reg], flush=True)
        if t.is_cuda and reg:
            hit = reg.get(_key(t))
            if hit is not None and (hit[0] is None or hit[0]() is not None):
                if hit[0] is None:
                    del reg[_key(t)]
                return ("regen", hit[1], t.dtype, t.shape, t.stride(),
                        t.storage_offset(), t.untyped_storage().nbytes(), t.device)
        return t

    def unpack(o):
        if not isinstance(o, tuple):
            return o
        _, fn, dtype, shape, stride, off, nbytes, dev = o
        with torch.no_grad():
            base = fn()
        st = base.untyped_storage()
        if st.nbytes() != nbytes or base.dtype != dtype:
            raise RuntimeError("regen: rebuilt storage differs (%d B %s vs %d B %s)"
                               % (st.nbytes(), base.dtype, nbytes, dtype))
        return torch.empty(0, dtype=dtype, device=dev).set_(st, off, shape, stride)

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield
    finally:
        _tls.reg = prev
