"""Lazy, explicitly selected upstream Mamba-3 kernels.

Raw angles and raw trap logits enter the upstream API (its docstrings are
ambiguous, but angle_dt.py applies tanh*pi and the scan applies sigmoid).
No parameters or persistent buffers are introduced by this adapter.
"""
from importlib import import_module
from functools import lru_cache
from types import FunctionType
from threading import Lock


def _clone_function(fn, overrides):
    """Scope launcher substitutions to our provider, never installed modules."""
    cloned = FunctionType(fn.__code__, {**fn.__globals__, **overrides},
                          fn.__name__, fn.__defaults__, fn.__closure__)
    cloned.__kwdefaults__ = fn.__kwdefaults__
    return cloned


def _cached_factory(factory):
    # Upstream launchers make NEW T.dynamic(B,S,H,G) Vars at every call.
    # TileLang 0.1.8 treats their identities as new JIT keys, regenerating
    # IR before hitting its compiled-kernel cache (~hundreds of ms/step).
    # Key these four symbolic dimensions by name instead. All other factory
    # arguments and the device remain in the key. Cache CODE, never tensors.
    import torch
    cache, lock = {}, Lock()
    def get(*args, **kwargs):
        key = (torch.cuda.current_device(), tuple(str(v) for v in args[:4]),
               args[4:], tuple(sorted(kwargs.items())))
        with lock:
            if key not in cache:
                if len(cache) >= 64:
                    cache.pop(next(iter(cache)))
                cache[key] = factory(*args, **kwargs)
            return cache[key]
    return get


@lru_cache(maxsize=1)
def _mimo_entry():
    """Reuse upstream autograd and kernel bodies with scoped factory caches."""
    import torch
    base = "mamba_ssm.ops.tilelang.mamba3."
    up = import_module(base + "mamba3_mimo")
    fw = import_module(base + "mamba3_mimo_fwd")
    bw = import_module(base + "mamba3_mimo_bwd")
    forward = _clone_function(fw.mamba_mimo_forward, {
        "mamba_mimo_fwd": _cached_factory(fw.mamba_mimo_fwd)})
    backward = _clone_function(bw.mamba_mimo_bwd_combined, {
        "mamba_mimo_bwd_fwd": _cached_factory(bw.mamba_mimo_bwd_fwd),
        "mamba_mimo_bwd_bwd": _cached_factory(bw.mamba_mimo_bwd_bwd)})
    class CachedMamba3(torch.autograd.Function):
        pass
    CachedMamba3.forward = staticmethod(_clone_function(
        up._Mamba3Function.forward, {"mamba_mimo_forward": forward}))
    CachedMamba3.backward = staticmethod(_clone_function(
        up._Mamba3Function.backward, {"mamba_mimo_bwd_combined": backward}))
    return _clone_function(up.mamba3_mimo, {"_Mamba3Function": CachedMamba3})


def check_provider(value, backend):
    if value not in ("native", "upstream"):
        raise ValueError("kernel_options m3_kernel must be 'native' or 'upstream'")
    if value == "upstream" and backend not in ("auto", "cuda"):
        raise ValueError("kernel_options m3_kernel='upstream' requires execution backend 'auto' or 'cuda'")


def siso(**kwargs):
    if not kwargs["V"].is_cuda:
        raise ValueError("kernel_options m3_kernel='upstream' requires CUDA input")
    try:
        fn = import_module("mamba_ssm.ops.triton.mamba3.mamba3_siso_combined").mamba3_siso_combined
    except ImportError as exc:
        raise ImportError("kernel_options m3_kernel='upstream' requires an installation of mamba_ssm with Mamba-3 Triton kernels") from exc
    return fn(**kwargs)


def mimo(**kwargs):
    if not kwargs["V"].is_cuda:
        raise ValueError("kernel_options m3_kernel='upstream' requires CUDA input")
    try:
        fn = _mimo_entry()
    except ImportError as exc:
        raise ImportError("MIMO kernel_options m3_kernel='upstream' requires mamba_ssm Mamba-3 and TileLang") from exc
    # The installed non-varlen backward requires complete chunks. Neutral
    # padding (DT=ADT=V=0) preserves every real output and the final hidden
    # state; the real last token's lookahead coefficient remains zero.
    import torch.nn.functional as F
    import torch
    props = torch.cuda.get_device_properties(kwargs["V"].device)
    if props.shared_memory_per_block_optin <= 100 * 1024:
        # Devices with ~99 KiB of opt-in shared memory cannot fit the large R2 backward tile.
        # Smaller chunks change the schedule, not the recurrence.
        rank = kwargs["Q"].shape[2]
        kwargs["chunk_size"] = min(kwargs["chunk_size"], max(8, 32 // rank))
    length = kwargs["V"].shape[1]
    pad = (-length) % kwargs["chunk_size"]
    if pad:
        for key in ("Q", "K", "V", "Angles"):
            value = kwargs[key]
            kwargs[key] = F.pad(value, (0, 0) * (value.ndim - 2) + (0, pad))
        for key in ("ADT", "DT", "Trap"):
            kwargs[key] = F.pad(kwargs[key], (0, pad))
    result = fn(**kwargs)
    if isinstance(result, tuple):
        return (result[0][:, :length], *result[1:])
    return result[:, :length]
