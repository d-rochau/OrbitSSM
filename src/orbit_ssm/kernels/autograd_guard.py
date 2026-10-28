"""One guard for kernel backwards that are not themselves differentiable (the CUDA kernels' autograd Functions).

A CUDA kernel's backward records no autograd graph.  Under `create_graph=True` autograd would still run it, return
gradients that depend on nothing, and a second derivative would silently keep only the torch-op part of the layer
(measured: rel 0.57-0.86 against finite differences).  `no_double_backward(layer)` decorates a backward
`(ctx, *grads)`; under `create_graph` it runs the kernel under `no_grad` and ties every gradient to the
differentiable tensors it depends on (the Function's saved tensors, `ctx.saved` if present, and the incoming
gradients) by a node whose backward raises a RuntimeError naming the layer and pointing to `backend='torch'` (`use`: a family whose torch route has hand-written backwards too passes its own message).

Cost: with `create_graph=False` (every training step) autograd runs backward with grad mode off, so the wrapper
is one `is_grad_enabled()` test and a call.  `once_differentiable` alone does not suffice: it refuses only when the
incoming GRADIENT requires grad, not when the saved inputs do.  The MH kernels keep their own `_NoDoubleBackward`
(hybrid_mamba_mh/functions.py; deliberately not shared, its sources key the MH calibration hash).
"""
import functools

import torch


USE_TORCH = "Build the layer with execution backend='torch' for higher-order gradients."


class NoDoubleBackward(torch.autograd.Function):
    """Identity on a first-order gradient; differentiating it raises."""

    @staticmethod
    def forward(ctx, layer, use, g, *deps):
        ctx.layer, ctx.use = layer, use
        return g.view_as(g)

    @staticmethod
    def backward(ctx, *_):
        raise RuntimeError(
            f"{ctx.layer}: double backward (create_graph=True) is not supported -- the CUDA kernels' backward is "
            "not differentiable, so a second derivative would silently miss the kernel's part.  " + ctx.use)


def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif obj is None:
        return
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tensors(o)


def _is_compiling():
    c = getattr(torch, "compiler", None)
    return c is not None and c.is_compiling()


def tie(layer, use, out, ctx, grads):
    """Tie the tuple/tensor `out` of a no_grad backward to the differentiable tensors it depends on."""
    deps = [t for t in (*_saved(ctx), *_tensors(grads)) if t is not None and t.requires_grad]
    if not deps:
        return out
    if torch.is_tensor(out):
        return NoDoubleBackward.apply(layer, use, out, *deps)
    return tuple(o if o is None or not torch.is_tensor(o) else NoDoubleBackward.apply(layer, use, o, *deps)
                 for o in out)


def _saved(ctx):
    try:
        yield from ctx.saved_tensors
    except RuntimeError:                      # released by a consuming backward
        pass
    yield from _tensors(getattr(ctx, "saved", None))


def no_double_backward(layer, use=USE_TORCH):
    """Decorator for a backward `(ctx, *grads)` of a kernel autograd Function or `register_autograd` backward."""
    def deco(bwd):
        @functools.wraps(bwd)
        def wrapper(ctx, *grads):
            if not torch.is_grad_enabled() or _is_compiling():
                return bwd(ctx, *grads)       # plain / retain_graph backward: untouched
            with torch.no_grad():
                out = bwd(ctx, *grads)
            return tie(layer, use, out, ctx, grads)
        wrapper.double_backward_guard = layer      # wiring marker (read by the double-backward tests)
        return wrapper
    return deco
