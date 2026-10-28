"""The kernel target: `"auto"` (the shipped NVIDIA kernels) or `"portable"` (plain Triton only).

``Execution(kernel_options={"kernel_target": "portable"})`` (a key of every kernel family) selects the
PORTABLE kernel path of the production mixers: only `triton.language` kernels, so it compiles on any Triton
backend (NVIDIA, AMD HIP, Intel XPU).  It uses

  * no Gluon kernel (``triton.experimental.gluon``: NVIDIA MMA / wgmma / TMA layouts) and no import of one,
  * no inline asm (PTX ``cos.approx`` / ``sin.approx`` / ``cvt.rna.tf32`` / ``rcp.approx`` / ``red.v4``),
  * no ``tf32`` dot literal: the fp32 operand mode is ``input_precision="ieee"`` (``"io"`` at fp32 IO and
    ``"tf32"`` resolve to ``"ieee"``; the Triton split-tf32 / bf16 dot modes are refused),
  * generic launch configs (no ``maxnreg``, no capability-keyed table).

``"auto"`` is the shipped behaviour, bit-identical on NVIDIA.  It resolves to ``"portable"`` by itself when
Triton's active target backend is not ``"cuda"`` (the detection of `ssd_mimo.common._ptx_trig_backend`; a process with
no active driver stays ``"auto"`` unless it runs the Triton interpreter, ``TRITON_INTERPRET=1``), or
when the process was forced (`force_process_target`, used by the portable test runs on NVIDIA).

A production route with no plain-Triton equivalent REFUSES in portable mode, by name (`PortableRefusal`);
the list of them is the gap table.

One process-level fact: the trig helpers (`ssd_mimo.common._cos_approx`) choose PTX or libdevice when the
first kernel that uses them compiles (Triton bakes the module-level constexpr `_TRIG_PTX` into the cache key and
refuses a change afterwards).  Selecting ``"portable"`` therefore latches libdevice trig for the PROCESS, and is
refused by name when a PTX trig kernel has already compiled (`latch_portable`).  Everything else (Gluon routes,
tf32, launch configs) is decided per call.

Imports no torch at module level.
"""
from __future__ import annotations

import contextlib
import functools
import os
import threading

__all__ = ["TARGETS", "PortableRefusal", "backend", "backend_is_cuda", "resolve", "active", "use",
           "force_process_target", "latch_portable", "gluon_refusal", "refuse_gluon", "forced",
           "layer_request", "scoped_forward", "scope_request", "nvidia_capability", "explicit"]

TARGETS = ("auto", "portable")


class PortableRefusal(NotImplementedError):
    """A route that has no plain-Triton equivalent, refused because ``kernel_target`` resolved to ``"portable"``."""


@functools.cache
def backend() -> str | None:
    """Triton's active target backend (``"cuda"``, ``"hip"``, ``"xpu"``, ...); None without an active driver.
    Read once: the backend cannot change mid-process."""
    try:
        from triton.runtime import driver
        return driver.active.get_current_target().backend
    except Exception:                      # noqa: BLE001 -- no active driver (CPU-only import, docs build)
        return None


def backend_is_cuda() -> bool:
    """True on an NVIDIA/CUDA Triton backend: the one test `ssd_mimo.common._ptx_trig_backend` makes."""
    return backend() == "cuda"


_FORCED: str | None = None        # process-level force (tests): "portable" makes "auto" resolve to portable
_LOCAL = threading.local()        # use() stack: the target of the layer / function being run


def forced() -> str | None:
    return _FORCED


def resolve(requested: str = "auto") -> str:
    """``"auto"`` / ``"portable"`` -> the target in force.  ``"auto"`` is ``"portable"`` off CUDA (no active driver
    included: nothing can launch, and the interpreter has no PTX / Gluon) and under `force_process_target`."""
    if requested not in TARGETS:
        raise ValueError(f"kernel_target must be one of {TARGETS}, got {requested!r}")
    if requested == "portable":
        return "portable"
    if _FORCED == "portable":
        return "portable"
    b = backend()
    if b is None:           # no active driver: CPU-only process (nothing launches, behaviour unchanged) -- except the interpreter
        return "portable" if os.environ.get("TRITON_INTERPRET", "0") == "1" else "auto"
    return "auto" if b == "cuda" else "portable"


def _requested(family: str | None) -> str:
    st = getattr(_LOCAL, "stack", None)
    if st:
        return st[-1]
    if family is not None:
        from .. import kernel_options as _ko
        try:
            return _ko.value(family, "kernel_target")
        except KeyError:                  # a family without the key
            return "auto"
    return "auto"


def active(family: str | None = None) -> bool:
    """True when the portable path is in force for the running call: the innermost `use` target, else the
    ``kernel_target`` of the active ``kernel_options`` scope of ``family``, resolved by `resolve`."""
    return resolve(_requested(family)) == "portable"


@contextlib.contextmanager
def use(requested: str):
    """``with use("portable"):`` -- the kernel target of the block (thread-local, nests).  ``"auto"`` inside an
    outer ``"portable"`` block stays auto for the inner block only when the process is not forced."""
    resolve(requested)                      # validates
    if requested == "portable":
        latch_portable()
    st = getattr(_LOCAL, "stack", None)
    if st is None:
        st = _LOCAL.stack = []
    st.append(requested)
    try:
        yield
    finally:
        st.pop()


def scope_request() -> str | None:
    """The request of the innermost `use` block (None outside any)."""
    st = getattr(_LOCAL, "stack", None)
    return st[-1] if st else None


def nvidia_capability(device=None) -> tuple:
    """``torch.cuda.get_device_capability(device)`` on an NVIDIA build, ``(0, 0)`` otherwise.  A ROCm build reports
    the gfx version through the same call (gfx90a is ``(9, 0)``, gfx94x ``(9, 4)``), which the Hopper-keyed tests
    (``== (9, 0)``, ``[0] >= 9``) would mistake for sm_90; they ask this instead."""
    import torch
    if getattr(torch.version, "hip", None) or not getattr(torch.version, "cuda", None):
        return (0, 0)
    return torch.cuda.get_device_capability(device)


def explicit() -> bool:
    """True only when portable was ASKED for (an enclosing `use("portable")` or the process force) -- never from the backend
    detection.  For host-side choices (the conv package) that must not change on a CPU / driverless process."""
    r = scope_request()
    return (r == "portable") if r is not None else _FORCED == "portable"


def force_process_target(target: str) -> None:
    """Make ``"auto"`` resolve to ``target`` for the whole process (test runs on NVIDIA).  Latches libdevice trig
    when ``target`` is ``"portable"``.  Raises `PortableRefusal` when a PTX trig kernel has already compiled."""
    global _FORCED
    if target not in TARGETS:
        raise ValueError(f"kernel_target must be one of {TARGETS}, got {target!r}")
    if target == "portable":
        latch_portable()
        _FORCED = "portable"
    else:
        _FORCED = None


_LATCH_COPIES = ("orbit_ssm.kernels.ssd_mimo.chart_emit",)


def latch_portable() -> None:
    """Make the trig helpers use libdevice (``tl.cos`` / ``tl.sin``) for the rest of the process.

    No-op when already latched, or when the backend is not CUDA (the import-time default is libdevice already).
    Raises `PortableRefusal` when a kernel using the PTX variant has compiled in this process."""
    from .ssd_mimo import common
    if not common._TRIG_PTX.value:
        return
    if common.trig_compiled():
        raise PortableRefusal(
            "kernel_target='portable' needs libdevice trig (no inline asm) but a kernel with the PTX "
            "cos.approx / sin.approx variant has already compiled in this process and Triton bakes that choice "
            "into the compiled kernel: select kernel_target='portable' before the first forward, or start a new process")
    common._TRIG_PTX = common.tl.constexpr(False)
    import sys
    for name in _LATCH_COPIES:                  # modules that took a `from .common import _TRIG_PTX` copy before the latch
        mod = sys.modules.get(name)
        if mod is not None:
            mod._TRIG_PTX = common._TRIG_PTX


def gluon_refusal(family: str | None = None) -> str | None:
    """The reason a Gluon route may not run now (None: it may).  Call it BEFORE importing a Gluon module."""
    if active(family):
        return "kernel_target='portable' (plain Triton only: Gluon kernels are NVIDIA-only by construction)"
    return None


def refuse_gluon(family: str | None, what: str, hint: str = "") -> None:
    """Raise `PortableRefusal` naming ``what`` when the portable path is in force (a Gluon-only route); ``hint``
    says what to use instead."""
    why = gluon_refusal(family)
    if why is not None:
        raise PortableRefusal(f"{what} has no plain-Triton kernel and is refused under {why}." + (f"  {hint}" if hint else ""))


def layer_request(layer) -> str:
    """The ``kernel_target`` a layer asks for: its kernel option (SSD-MIMO, HybridMamba), else its layer switch."""
    ko = getattr(layer, "kernel_options", None) or {}
    return ko.get("kernel_target", getattr(layer, "kernel_target", "auto"))


def scoped_forward(fn):
    """Decorator of a layer method: run it under the layer's ``kernel_target`` (`use`).  A layer at ``"auto"`` calls
    straight through (no scope, no overhead): the shipped behaviour."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        req = layer_request(self)
        if req == "auto":
            return fn(self, *args, **kwargs)
        with use(req):
            return fn(self, *args, **kwargs)
    return wrapper
