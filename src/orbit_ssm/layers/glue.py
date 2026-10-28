"""Environment switches for the SSD-MIMO and HM fused-projection glue nodes, and the parameter-cache key."""
from __future__ import annotations
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


GLUE_ITEMS = ("bias", "bc", "wcat", "addmm", "dwin", "inproj")


GLUE_OPT_IN = ("wcat",)


def dt_fp32_on(kernel_options=None) -> bool:
    """Is the dt bias added in fp32 inside the tail's softplus? Default on;
    the ``dt_fp32`` kernel option (``False``) folds a bf16 bias
    into the token GEMM instead.  ``kernel_options`` is the layer's option dict
    (default: the active `kernel_options.push` scope).  Read per call.

    The bf16 fold rounds a pre-activation near -9 with absolute error ~2^-5,
    a 2-3 % signed error in Delta (~1e-4).
    """
    from .. import kernel_options as _ko
    return bool(_ko.value("ssd_mimo", "dt_fp32", kernel_options))


def glue_on(item: str) -> bool:
    """Is the named SSD-MIMO glue item live for this call?

    ``SSD_MIMO_GLUE`` is ``"1"`` (default: every item on except the opt-in
    ones in :data:`GLUE_OPT_IN`), ``"all"`` (every item on), ``"0"`` (every
    item off) or a comma-separated list of items to turn off, e.g.
    ``"bias,bc"`` (opt-in items stay off unless prefixed ``"all:"``). Read
    per call so it can be flipped inside one process.
    """
    v = os.environ.get("SSD_MIMO_GLUE", "1")
    if v == "1" or v == "":
        return item not in GLUE_OPT_IN
    if v == "all":
        return True
    if v == "0":
        return False
    if v.startswith("all:"):
        return item not in v[4:].split(",")
    return item not in v.split(",") and item not in GLUE_OPT_IN


HM_GLUE_ITEMS = ("inproj", "sel", "dtk", "zk")


# HM glue items. dtk (softplus + delta_bias gradient inside the scan) is on by
# default; sel (x_proj/conv-drive tee node) and zk (output gate inside the
# scan's store) are opt-in because they do not pay.
# HM_GLUE=0 / a comma list turn items off; HM_GLUE=all (or all:<list>) turns
# every item on, opt-ins included.
HM_GLUE_OPT_IN = ("sel", "zk")


def hm_glue_on(item: str) -> bool:
    """Is the named HM glue item live for this call?

    ``HM_GLUE`` follows :func:`glue_on`'s grammar: ``"1"``/unset (every item
    on but the opt-in ones), ``"all"``, ``"0"`` (all off), a comma list of
    items to turn off, or ``"all:<list>"``. Read per call.
    """
    v = os.environ.get("HM_GLUE", "1")
    if v == "1" or v == "":
        return item not in HM_GLUE_OPT_IN
    if v == "all":
        return True
    if v == "0":
        return False
    if v.startswith("all:"):
        return item not in v[4:].split(",")
    return item not in v.split(",") and item not in HM_GLUE_OPT_IN


def param_cache_key(params, *extra):
    """Invalidation key for a cached `torch.cat`/cast of parameters.

    Per source parameter: ``data_ptr()`` (moves on `.to(device)` or
    `load_state_dict(assign=True)`), ``_version`` (bumped by in-place writes
    autograd sees: ordinary optimizer steps and `load_state_dict`) and
    ``requires_grad``.

    Known hole: ``torch.optim.AdamW(..., fused=True)`` updates parameters
    without bumping ``_version``, so a cache keyed this way serves stale
    weights. Runs with a fused optimizer must set ``SSD_MIMO_GLUE=wcat`` (or
    ``0``) to turn the cache off.

    The cached tensor is a copy made with grad mode off, so it aliases no
    parameter storage.
    """
    return tuple(extra) + tuple(
        (p.data_ptr(), p._version, p.requires_grad) for p in params)
