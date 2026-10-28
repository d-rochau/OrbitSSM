"""No silent CUDA fallback: what a layer does when the kernels cannot serve a call.

* ``backend="cuda"`` is a request for the kernels.  A configuration they refuse raises a ``ValueError`` that
  names the reason and the two fixes (adapt the config, or ask for ``backend="torch"``).
* ``backend="auto"`` chooses the route.  When it picks a non-kernel route for a CUDA tensor it warns once per
  (layer class, reason) with `NonKernelRouteWarning`; the layer's ``last_route()`` records the reason too.
* ``backend="torch"`` / ``"sequential"`` are requests for those routes; nothing is refused or warned.
"""
from __future__ import annotations

import warnings

__all__ = ["NonKernelRouteWarning", "demote_kernel_route", "kernel_refusal_error"]


class NonKernelRouteWarning(UserWarning):
    """``backend="auto"`` ran a non-kernel route for a CUDA tensor (the reason is in the message and in
    the layer's ``last_route()``)."""


_WARNED: set = set()


def kernel_refusal_error(who: str, reason: str) -> ValueError:
    """The ``backend="cuda"`` refusal: names the reason and the two fixes (also raised at construction and in
    ``set_execution`` for a refusal that depends on the configuration alone)."""
    return ValueError(
        f"{who}: backend='cuda' but the CUDA kernels refuse this configuration: {reason}.  "
        "Adapt the config, or use backend='torch' (backend='auto' would run the non-kernel route "
        "and say so in last_route()).")


def demote_kernel_route(layer, on_cuda: bool, reason: str, *, stacklevel: int = 3) -> None:
    """Called where a layer is about to run a non-kernel route for ``reason``.

    Raises under ``layer.backend == "cuda"``; warns once per (class, reason) under ``"auto"`` with a CUDA
    input (``on_cuda``); returns silently otherwise (an explicitly requested non-kernel backend, or a CPU
    tensor, which has no kernel route to be moved off)."""
    who = type(layer).__name__
    backend = getattr(layer, "backend", "auto")
    if backend == "cuda":
        raise kernel_refusal_error(who, reason)
    if backend == "auto" and on_cuda and (who, reason) not in _WARNED:
        _WARNED.add((who, reason))
        warnings.warn(
            f"{who}: backend='auto' runs a NON-kernel route for a CUDA input: {reason}.  Adapt the config "
            "to stay on the kernels, or set backend='torch' to ask for this route.",
            NonKernelRouteWarning, stacklevel=stacklevel)
