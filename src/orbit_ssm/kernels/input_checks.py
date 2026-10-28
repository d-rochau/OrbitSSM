"""Named refusals for kernel-entry inputs.

A bare ``assert`` at a kernel entry is no input check: it vanishes under
``python -O`` and, when it fires, says nothing (``AssertionError`` with an
empty message), so a CPU tensor handed to a CUDA route surfaced as an
anonymous crash.  `check_cuda_input` raises a ``ValueError`` that names the
entry, the argument, its device / dtype, and the route that serves CPU
tensors.
"""
from __future__ import annotations

import functools
import warnings

import torch

#: the oldest architecture the kernels run on (Ampere, sm_80) and the ones they are validated and calibrated on
#: (A100 sm_80, H100 sm_90, RTX PRO 5000 Blackwell sm_120)
MIN_CAPABILITY = (8, 0)
TESTED_CAPABILITIES = frozenset({(8, 0), (9, 0), (12, 0)})


@functools.cache
def check_cuda_arch(index: int) -> None:
    """Refuse, by name, a device below `MIN_CAPABILITY`; warn once per device outside `TESTED_CAPABILITIES`.
    NVIDIA-only: on a ROCm build ``get_device_capability`` is the gfx version, not an sm number, so the check
    does not apply (the portable path runs there, untuned)."""
    if getattr(torch.version, "hip", None):
        return
    cap = torch.cuda.get_device_capability(index)
    name = torch.cuda.get_device_name(index)
    if cap < MIN_CAPABILITY:
        raise ValueError(
            f"{name} (sm_{cap[0]}{cap[1]}) is below the supported minimum sm_{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]}; "
            f"run the layer with execution={{'backend': 'torch'}}")
    if cap not in TESTED_CAPABILITIES:
        tested = ", ".join(f"sm_{a}{b}" for a, b in sorted(TESTED_CAPABILITIES))
        warnings.warn(f"{name} (sm_{cap[0]}{cap[1]}) is not among the validated architectures ({tested}); "
                      "the kernels run with untuned launch grids", RuntimeWarning, stacklevel=3)


def check_cuda_input(who: str, tensors: dict, dtypes=None) -> None:
    """Refuse, by name, inputs the CUDA route of `who` cannot take.

    ``tensors`` maps argument names to tensors (``None`` entries are skipped).
    Every given tensor must be a CUDA tensor, all on ONE device of a supported architecture
    (`check_cuda_arch`), and -- when ``dtypes`` is given -- of one of those dtypes.
    """
    devices = {}
    for name, t in tensors.items():
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{who}: {name} must be a torch.Tensor, got {type(t).__name__}")
        if not t.is_cuda:
            raise ValueError(
                f"{who}: {name} is on {t.device}, but this is the CUDA kernel route; "
                f"move the inputs to a CUDA device or run the layer with "
                f"execution={{'backend': 'torch'}} (the CPU route)")
        if dtypes is not None and t.dtype not in dtypes:
            raise ValueError(
                f"{who}: {name} has dtype {t.dtype}; this route takes "
                f"{', '.join(str(d) for d in dtypes)}")
        devices.setdefault(t.device, name)
    if len(devices) > 1:
        where = ", ".join(f"{n} on {d}" for d, n in devices.items())
        raise ValueError(f"{who}: inputs must share one CUDA device ({where})")
    if torch.compiler.is_compiling():
        return          # the arch query (`get_device_name`) is a graph break under Dynamo; the check runs on the first eager call
    for device in devices:
        check_cuda_arch(device.index if device.index is not None else torch.cuda.current_device())
