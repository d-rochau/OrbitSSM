"""One gradient buffer for a fused projection output ("slab"): consumers write their gradient in place.

A mixer's fused input projection produces ONE token-major slab ``[B, L, W]`` that is split into column
spans (gate z, drive x, token maps B / C, the small dt / angle rows ...).  With plain autograd the backward of
that split is a ``cat`` of the spans' gradients (one full read and write of the slab-sized gradient) and each big consumer allocates its own gradient first.  Here the split is ONE custom function whose
backward gradient buffer is shared:

* `SlabGrad` is created per forward call from the slab.  It allocates NOTHING until a backward consumer asks.
* A consumer that knows its input is a column span of the slab (the channel-last conv's ``dx``, the gate / norm
  adjoint's ``dZ``, the token-map prep's ``dv``) asks ``region(x)`` for the view of the gradient buffer that has
  exactly the layout of ``x`` (same strides, same offset) and writes its gradient there instead of into a fresh
  tensor.
* `slab_split`'s backward finds those gradients already in place (same pointer, same strides: no copy), copies the
  spans that arrived elsewhere (the small ones, anything a different node produced or accumulated), zero-fills spans
  that received no gradient and the pad columns, and returns the buffer as the slab's gradient.

Correctness does not depend on any consumer cooperating: a span whose gradient is not in place is copied, so the
result equals the ``cat`` it replaces; the cooperating consumers only save the copy.  The buffer is released after
the split's backward (a second backward through a retained graph allocates its own).
"""

from __future__ import annotations

import torch

__all__ = ["SlabGrad", "slab_split"]


class SlabGrad:
    """The (lazily allocated) gradient buffer of one slab; see the module docstring."""

    __slots__ = ("shape", "dtype", "device", "base", "es", "buf")

    def __init__(self, slab: torch.Tensor):
        self.shape = tuple(slab.shape)
        self.dtype, self.device = slab.dtype, slab.device
        self.base, self.es = slab.data_ptr(), slab.element_size()
        self.buf = None

    def _offset(self, like: torch.Tensor):
        """Element offset of the column span `like` inside the slab, or None when `like` is not a
        full-token column span of it (a different tensor, a copy, a row slice, another dtype)."""
        if like.dim() != 3 or like.dtype != self.dtype or like.device != self.device:
            return None
        if tuple(like.shape[:2]) != self.shape[:2] or like.stride(2) != 1:
            return None
        if like.stride(1) != self.shape[2] or like.stride(0) != self.shape[1] * self.shape[2]:
            return None
        delta = like.data_ptr() - self.base
        if delta < 0 or delta % self.es:
            return None
        col = delta // self.es
        if col + like.shape[2] > self.shape[2]:
            return None
        return col

    def buffer(self) -> torch.Tensor:
        """The shared gradient buffer, allocated on first use (uninitialised)."""
        if self.buf is None:
            self.buf = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        return self.buf

    def region(self, like: torch.Tensor):
        """The view of the gradient buffer laid out exactly like `like` (a column span of the slab), or None
        when `like` is not one (the caller then allocates its own gradient, and the split copies it)."""
        col = self._offset(like)
        if col is None:
            return None
        return self.buffer().as_strided(like.shape, like.stride(), col)


class _SlabSplit(torch.autograd.Function):
    """Split a slab into column spans; backward assembles the slab gradient in `SlabGrad`'s buffer."""

    @staticmethod
    def forward(ctx, slab, gs, spans):
        ctx.gs, ctx.spans = gs, spans
        return tuple(slab[..., a:a + w] for a, w in spans)

    @staticmethod
    def backward(ctx, *grads):
        gs, spans = ctx.gs, ctx.spans
        buf = gs.buffer()
        end = 0
        for (a, w), g in zip(spans, grads):
            reg = buf[..., a:a + w]
            if a > end:
                buf[..., end:a].zero_()
            end = a + w
            if g is None:
                reg.zero_()
            elif not (g.data_ptr() == reg.data_ptr() and g.shape == reg.shape and g.stride() == reg.stride()):
                reg.copy_(g)
        if end < gs.shape[2]:
            buf[..., end:].zero_()
        gs.buf = None
        return buf, None, None


def slab_split(slab: torch.Tensor, spans) -> tuple["SlabGrad", tuple]:
    """(gs, parts): `parts[i]` is the view ``slab[..., a:a+w]`` of ``spans[i] = (a, w)`` (ascending, disjoint)
    and `gs` the gradient buffer the consumers of the parts may write into (`SlabGrad.region`)."""
    gs = SlabGrad(slab)
    return gs, _SlabSplit.apply(slab, gs, tuple((int(a), int(w)) for a, w in spans))
