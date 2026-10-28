"""SSD-MIMO core, configuration and mixer.

Each head has scalar decay over state coordinates; R sibling systems share
rank-R input maps and produce O output channels per channel group. Optional
M3 rotation and transported two-endpoint writes are defined in
the SSD-MIMO contract. The mixer owns the outer
projections, convolution, gate and normalization.
"""

from __future__ import annotations

import math
import os as _os
import warnings

import torch
from ..kernels.autograd_guard import no_double_backward as _no_double_backward
import torch.nn as nn
import torch.nn.functional as F

from ..layers.conv import (CausalDepthwiseConv, causal_conv_step, check_conv_kernel)
from ..layers.routes import demote_kernel_route, kernel_refusal_error
from ..layers.glue import (glue_on, param_cache_key, dt_fp32_on)
from ..layers.proj import (linear_keep_layout, matmul_channel_major,
                           resolve_proj_source, cached_cat, split_in_proj)
from ..layers.norms import rms_norm_cast
from ..layers.state import read_alias, BaseSSM, _ScanBase, ScanState, State, ssm_status
from .. import kernel_options as _kernel_options
from ..kernels import portable as _portable
from ..families import register_family
from ..config import reject_unknown_kwargs, signature_names
from ..config_spec import refuse_per_coordinate_gates
from ..layers.init import SMALL_SCALE, freq_param, init_dt_bias, skip_param
from ..validation import (BACKENDS, RANKS, WRITE_RULE_DEFAULT, check_backend, check_chunk_size, check_dt_range, check_precision,
                          check_rank, positive_int, resolve_write_rule)
from .. import config_spec as _config_spec
from ..execution import Execution, as_execution, split_kernel_options
from ..configuration import (Configurable, resolve_rank_geometry,
                             record_constructor_config,
                             mixer_core_options, as_family_config)


def _ref_f(t: torch.Tensor) -> torch.Tensor:
    """fp32 working precision, but PRESERVE fp64 for the reference oracle."""
    return t.to(torch.float64 if t.dtype == torch.float64 else torch.float32)


# CUDA output layout (SSD_MIMO_Y_LAYOUT); token-major avoids a full activation transpose.
_Y_LAYOUTS = ("token", "contig", "channel")


def _ua_gate_epilogue_off() -> bool:
    """True when the gate is not on its dedicated epilogue kernel (where the fused output norm lives)."""
    from ..kernels.ssd_mimo import autograd as _ua
    return not _ua._gate_epilogue_reads_strided()


def _y_layout() -> str:
    """Which output layout the CUDA route asks the scan for."""
    mode = _os.environ.get("SSD_MIMO_Y_LAYOUT", "token")
    if mode not in _Y_LAYOUTS:
        raise ValueError(
            f"SSD_MIMO_Y_LAYOUT must be one of {_Y_LAYOUTS}, got {mode!r}")
    return mode


def _y_is_token_major(y: torch.Tensor) -> bool:
    """True when `y` = [B, H, (O,) G, L] is a view of token-major memory (checked from strides:
    ops after the kernel store may relayout it)."""
    stride, shape = y.stride(), y.shape
    ch = 1
    for size, st in zip(reversed(shape[1:-1]), reversed(stride[1:-1])):
        if size == 1:
            continue                      # a size-1 axis addresses nothing
        if st != ch:
            return False
        ch *= size
    return stride[-1] == ch


#: Which torch implementation `backend="torch"` and `scan_step` run: "fast" (`ssd_mimo_torch`) or
#: "legacy" (`_forward_torch_legacy` / `_step_legacy`: the shared `_prepare_kernel_inputs` chain, the
#: CUDA route's own host code, which this makes testable on CPU).  A test hook set with
#: ``monkeypatch.setattr``; NOT an environment lever and not a model setting.
_TORCH_IMPL = "fast"


def _torch_impl() -> str:
    if _TORCH_IMPL not in ("fast", "legacy"):
        raise ValueError(f"ssd_mimo._TORCH_IMPL must be fast|legacy, got {_TORCH_IMPL!r}")
    return _TORCH_IMPL


_R_MIX_MODES = ("auto", "learned", "none")


# Token-map tails: uncapped softplus for Delta; |A| = clamp_min(f(x), 1e-4) with f selected by `a_act`.

#: Chunk the scan is asked for by `_forward_cuda`: the combined node's backward views dy as chunks of
#: this size; also the fallback for unknown ranks and for "measure" calls that cannot measure.
_COMBINED_CHUNK = 64

# Per-rank chunk defaults (fwd+bwd); an explicit chunk_size pins T, "measure" benchmarks candidates.
# Chunk selection changes scheduling, never the recurrence.
_CHUNK_RANK_TABLE = {1: 64, 2: 32, 4: 32, 8: 16}

# Admissible chunk lengths for "measure" (kernel floor 16; T=128 never won); mirrors
# `smem._CHUNK_T_CANDIDATES`.
_CHUNK_CANDIDATES = (64, 32, 16)


def _ssd_geometry(d_inner: int, *, d_ssm_in: int, n_groups=None, n_heads=None, d_head=None):
    """SSD adds its G*R rule to the shared head-width equation: d_head = n_groups * d_ssm_in,
    returns ``(n_heads, d_head, n_groups)``."""
    R = positive_int(d_ssm_in, "d_ssm_in")
    if d_head is not None:
        positive_int(d_head, "d_head")
    if n_groups is not None:
        group_width = positive_int(n_groups, "n_groups") * R
        if d_head is not None and d_head != group_width:
            raise ValueError(f"conflicting d_head: {{'d_head': {d_head}, "
                             f"'n_groups_times_d_ssm_in': {group_width}}}")
        d_head = group_width
    h, p = resolve_rank_geometry(d_inner, R, n_heads=n_heads, d_head=d_head)
    return h, p, p // R


#: Width refinement of the rank table: with W = R*O*G_kernel the per-head intra-chunk width, a table
#: T > 32 drops to 32 once W >= 64 (the [T, T] score and [T, T] x [T, G] output work grows like T*(N + W)).
#: For shapes outside the calibration use `chunk_size="measure"`.
_CHUNK_WIDE_W = 64
_CHUNK_WIDE_T = 32

# Forward-only (no backward) defaults.  The width clause here keys on G alone, not R*O*G: the forward
# amortises the per-chunk [N, GT] state update and carry read (scale with G) and does not carry the
# backward's score path R*O times.  R = 8 keeps the fwd+bwd entry.
_CHUNK_RANK_TABLE_FWD = {1: 64, 2: 64, 4: 16, 8: 16}
_CHUNK_WIDE_W_FWD = 64        # compared against G alone
_CHUNK_WIDE_T_FWD = 32


def _call_size(kw: dict) -> dict:
    """{L, work = B * H * L} of the call, for the calibrated structural chunk rule of the real route (`calibration_chunkpar.real_rule`).
    Empty when the call's kernel-layout slab is not in `kw` (the rule then does not apply)."""
    x = kw.get("x_k")
    if isinstance(x, torch.Tensor) and x.dim() == 5:
        B, Hk, _, _, L = x.shape
        # the IO dtype the kernels will see (the autocast dtype of a CUDA call under autocast): the fp32 axis of the rule
        dt = (torch.get_autocast_dtype("cuda") if (x.is_cuda and torch.is_autocast_enabled("cuda")) else x.dtype)
        return dict(L=int(L), work=int(B) * int(Hk) * int(L), dtype=dt)
    return {}


def _needs_bwd(kw: dict) -> bool:
    """Does this call have a backward to pay for (grad enabled and some leaf requires it)?"""
    return bool(torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in kw.values()))


def _gluon_siso_default(kw: dict, with_route: bool = True) -> bool:
    """Is this call on the default SISO route (bf16 IO: the tensor dtype under autocast, else the leaf
    dtype)?  With `with_route` the calibrated per-GPU structural route (`rot_route`) decides between the
    SISO schedule and the general routes by occupancy and length."""
    x = kw.get("x_k")
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    io_bf16 = (torch.get_autocast_dtype("cuda") == torch.bfloat16
               if torch.is_autocast_enabled("cuda")
               else x.dtype == torch.bfloat16)
    # the SISO schedule is the default for a rotating R = O = 1 bf16 call
    # unless the calibrated route picks a general route
    from ..kernels.ssd_mimo.siso_route import siso_enabled
    if with_route:
        return bool(siso_enabled(x.device, rotating=True, io_bf16=io_bf16, T=64,
                                 bh=x.shape[0] * x.shape[1], L=x.shape[-1],
                                 needs_bwd=_needs_bwd(kw), thh=kw.get("thh_k") is not None))
    return bool(siso_enabled(x.device, rotating=True, io_bf16=io_bf16, T=64))


def _calibrated_rot_route(kw: dict, fp32_only: bool = False):
    """(kind, T) the calibrated structural route (`rot_route`) picks for a rotating R = O = 1 bf16 call, or None
    (explicit SISO_FAST / ROT_BWD lever, not bf16 / CUDA, no valid calibration)."""
    x = kw.get("x_k")
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return None
    from ..kernels.ssd_mimo import rot_route as rr
    if rr.explicit_siso() or rr.explicit_rot_bwd():
        return None
    io = (torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype)
    if io == torch.float32:
        # fp32 IO: its own calibrated rule (rot_route's `*_fp32` phases), None when this GPU has none
        return rr.pick(x.shape[0] * x.shape[1], x.shape[-1], x.device, _needs_bwd(kw),
                       kw.get("thh_k") is not None, dtype="fp32")
    if io != torch.bfloat16 or fp32_only:
        return None
    return rr.pick(x.shape[0] * x.shape[1], x.shape[-1], x.device, _needs_bwd(kw),
                   kw.get("thh_k") is not None)


def _scan_chunk(R: int = 1, G: int | None = None,
                O: int | None = None, needs_bwd: bool = True,
                real: bool | None = None, L: int | None = None,
                work: int | None = None, dtype=None) -> int:
    """Rule default for `chunk_size="auto"` at rank `R`, width `G`.

    `real=None`: the caller does not say which route it serves, so no per-rank calibration is read.
    `SSD_MIMO_CHUNK` overrides; `needs_bwd=False` takes the forward-only table.
    """
    import os as _os
    env = _os.environ.get("SSD_MIMO_CHUNK")
    if env:
        return int(env)
    if real and R == 1 and (O is None or O == 1):
        # real R = O = 1 route: per-GPU calibrated structural choice (calibration_chunkpar.py)
        from ..kernels.ssd_mimo.calibration_chunkpar import real_chunk
        t = real_chunk("train_fwd" if needs_bwd else "fwd_only", G, L, work, dtype)
        if t is not None:
            return t
    elif real is not None and R > 1 and (O is None or O == R):
        # R = O > 1: per-rank calibrated chunk (the per-rank calibration); absent -> the fixed rule below
        from ..kernels.ssd_mimo.calibration_chunkpar import rank_chunk
        t = rank_chunk(bool(real), R, R, "train_fwd" if needs_bwd else "fwd_only", G, dtype)
        if t is not None:
            return t
    table = _CHUNK_RANK_TABLE if needs_bwd else _CHUNK_RANK_TABLE_FWD
    wide_w = _CHUNK_WIDE_W if needs_bwd else _CHUNK_WIDE_W_FWD
    wide_t = _CHUNK_WIDE_T if needs_bwd else _CHUNK_WIDE_T_FWD
    T = table.get(R, _COMBINED_CHUNK)
    if G is not None:
        # width clause: R*O*G for fwd+bwd, G alone for the forward
        W = int(G) * (R * (R if O is None else O) if needs_bwd else 1)
        if T > wide_t and W >= wide_w:
            T = wide_t
    return T


# `SSD_MIMO_CHUNK_PAIR` ("0" default | "1"): a training step runs its forward at the forward-only
# table's T and its backward at the fwd+bwd table's T (the two passes prefer different lengths).
def _chunk_pair_on() -> bool:
    import os as _os
    return _os.environ.get("SSD_MIMO_CHUNK_PAIR", "0").lower() in (
        "1", "true", "yes", "on")


def _chunk_pair_search_on() -> bool:
    """Does `chunk_size="measure"` time (T_fwd, T_bwd) pairs?  `SSD_MIMO_CHUNK_PAIR_SEARCH=0` restores
    the diagonal-only search."""
    import os as _os
    return _os.environ.get("SSD_MIMO_CHUNK_PAIR_SEARCH", "1").lower() \
        not in ("0", "false", "no", "off")


# Backward chunk table: None = the forward's T (identity); `chunk_size="measure"` searches the pair.
_CHUNK_BWD_TABLE: dict = {}      # (R, O, G_kernel) -> T_bwd; empty = identity


def _scan_chunk_bwd(R: int = 1, G: int | None = None,
                    O: int | None = None, T_fwd: int = 64,
                    real: bool | None = None, L: int | None = None,
                    work: int | None = None, dtype=None) -> int | None:
    """Rule default for the backward's chunk; `None` = T_fwd.  `SSD_MIMO_CHUNK_BWD` overrides it
    (also read by `autograd._chunk_bwd_env`)."""
    import os as _os
    env = _os.environ.get("SSD_MIMO_CHUNK_BWD", "").strip().lower()
    if env and env not in ("auto", "0", ""):
        return int(env)
    tb = _CHUNK_BWD_TABLE.get((R, R if O is None else O, G))
    if real and R == 1 and (O is None or O == 1) \
            and not _os.environ.get("SSD_MIMO_CHUNK"):
        from ..kernels.ssd_mimo.calibration_chunkpar import real_chunk
        tc = real_chunk("train_bwd", G, L, work, dtype)
        if tc is not None:
            tb = tc
    elif real is not None and R > 1 and (O is None or O == R) and not _os.environ.get("SSD_MIMO_CHUNK"):
        from ..kernels.ssd_mimo.calibration_chunkpar import rank_chunk
        tc = rank_chunk(bool(real), R, R, "train_bwd", G, dtype)
        if tc is not None:
            tb = tc
    if tb is None or tb == T_fwd:
        return None
    return tb


def _dt_tail_eager(dt_lin: torch.Tensor) -> torch.Tensor:
    return F.softplus(dt_lin)


# The |A| clamps pass the gradient at the boundary (`grad * (x >= lo)` / `grad * (x <= hi)`), as the
# fused tail kernel does (`layers/tails.py::_rate_grad`).  torch >= 2.14 clamp uses the strict
# inequality, so the rule is spelled out; in bf16 the boundary is not measure-zero.
def _clamp_min_pass(x: torch.Tensor, lo: float) -> torch.Tensor:
    return torch.where(x >= lo, x, x.clamp_min(lo).detach())


def _clamp_max_pass(x: torch.Tensor, hi: float) -> torch.Tensor:
    return torch.where(x <= hi, x, x.clamp_max(hi).detach())


def _a_mag_tail_eager(a_lin: torch.Tensor, floor: float) -> torch.Tensor:
    # matches Mamba-3, including the clamp subgradients at zero
    neg = _clamp_max_pass(a_lin, 0.)
    pos = _clamp_min_pass(a_lin, 0.)
    return _clamp_min_pass(pos + torch.reciprocal(1 - neg), floor)


def _a_mag_softplus_eager(a_lin: torch.Tensor, floor: float) -> torch.Tensor:
    # |A| = clamp_min(softplus(x), floor): upstream's clamp(-softplus(x), max=-floor), adjoint included
    return _clamp_min_pass(F.softplus(a_lin), floor)


#: `a_act` values: the |A| activation of the data-dependent rate (a_dyn).
A_ACTS = ("heavy_tail", "softplus")
#: layer-level levers of the ``head_dynamic`` rate (shared by every block of that law): `config_spec.check_law_levers` refuses a
#: non-default value when no row block has that law (the lever would be silently ignored)
_RATE_LEVERS = [((_config_spec.HEAD_DYNAMIC,), {"a_dyn_per_sibling": False, "a_act": "heavy_tail", "a_proj_bias": True})]

_dt_tail_compiled = None
_a_mag_tail_compiled: dict = {}


# The fused tail implements the stock eager laws; fusion is refused if one is replaced.
_DT_TAIL_ORIG = _dt_tail_eager
_A_MAG_TAIL_ORIG = _a_mag_tail_eager
_A_MAG_SOFTPLUS_ORIG = _a_mag_softplus_eager


def _bc_narrow_conv_ready(joint: torch.Tensor, package=True) -> bool:
    """May the channel-last conv kernel read this [B, L, W] B|C narrow of the fused slab in place?

    For the `causal_conv1d` package it restates the C++ entry's preconditions on the view: contiguous channel run
    (excludes the channel-major slab), channel count and batch/token strides multiples of 8; the plain-Triton conv
    (``package="triton"``) takes any strides.  Either way the token stride must be wider than the narrow (else it is already dense).
    """
    from ..layers.conv import channel_last_conv_view_ok
    return (channel_last_conv_view_ok(joint, package)
            and joint.stride(1) > joint.shape[-1])


def _tail_law_is_stock() -> bool:
    """Are the module's dt / |A| maps still the ones the kernel implements?"""
    return (_dt_tail_eager is _DT_TAIL_ORIG
            and _a_mag_tail_eager is _A_MAG_TAIL_ORIG
            and _a_mag_softplus_eager is _A_MAG_SOFTPLUS_ORIG)


# Phase increments combine tanh features, head sharing, dt and frequency; `_OmegaFoldFn` recomputes
# them in backward to save activation memory.
def _kernel_phase_pre_omega(raw: torch.Tensor, dt_raw_k: torch.Tensor,
                            *, H: int, R: int, K: int, ka: int,
                            per_head: bool, sign: float) -> torch.Tensor:
    from ..kernels.ssd_mimo.m3_epilogue import m3_phase_increments
    B, L = raw.shape[0], raw.shape[1]
    raw_f = _ref_f(raw)
    if per_head:
        raw_h = raw_f.view(B, L, H, ka)
    else:
        raw_h = raw_f.view(B, L, 1, ka).expand(B, L, H, ka).contiguous()
    if R == 1:
        return m3_phase_increments(raw_h, dt_raw_k, K, sign)
    parts = []
    for r in range(R):
        dt_r = dt_raw_k[:, r * H:(r + 1) * H, :].contiguous()
        parts.append(m3_phase_increments(raw_h, dt_r, K, sign))
    return torch.stack(parts, dim=2).reshape(B, L, R * H, K)


class _OmegaFoldFn(torch.autograd.Function):
    """`out = _kernel_phase_pre_omega(raw, dt_raw_k) * omega`, saving only (raw, dt_raw_k, omega)."""

    @staticmethod
    def forward(ctx, raw, dt_raw_k, omega, H, R, K, ka, per_head, sign):
        with torch.no_grad():
            pre = _kernel_phase_pre_omega(raw, dt_raw_k, H=H, R=R, K=K,
                                          ka=ka, per_head=per_head, sign=sign)
            out = pre * omega if omega is not None else pre
        ctx.save_for_backward(
            raw, dt_raw_k, omega if omega is not None
            else raw.new_empty(0))
        ctx.H, ctx.R, ctx.K, ctx.ka = H, R, K, ka
        ctx.per_head, ctx.sign = per_head, sign
        ctx.has_omega = omega is not None
        ctx.need_raw = raw.requires_grad
        ctx.need_dt = dt_raw_k.requires_grad
        ctx.need_omega = ctx.has_omega and omega.requires_grad
        return out

    @staticmethod
    @_no_double_backward("SSD-MIMO phase fold")
    def backward(ctx, dout):
        raw, dt_raw_k, omega_s = ctx.saved_tensors
        omega = omega_s if ctx.has_omega else None
        need_raw, need_dt, need_omega = (ctx.need_raw, ctx.need_dt,
                                         ctx.need_omega)
        if not (need_raw or need_dt or need_omega):
            return None, None, None, None, None, None, None, None, None
        dout = dout.contiguous()
        with torch.enable_grad():
            raw_l = raw.detach().requires_grad_(need_raw or need_dt)
            dt_l = dt_raw_k.detach().requires_grad_(need_dt)
            pre = _kernel_phase_pre_omega(
                raw_l, dt_l, H=ctx.H, R=ctx.R, K=ctx.K, ka=ctx.ka,
                per_head=ctx.per_head, sign=ctx.sign)
        domega = (dout * pre).sum(dim=(0, 1, 2)) if need_omega else None
        dpre = dout * omega if omega is not None else dout
        draw = ddt = None
        if need_raw or need_dt:
            leaves = ([raw_l] if need_raw else []) + ([dt_l] if need_dt else [])
            grads = torch.autograd.grad(pre, leaves, grad_outputs=dpre,
                                        allow_unused=True)
            it = iter(grads)
            draw = next(it) if need_raw else None
            ddt = next(it) if need_dt else None
        return draw, ddt, domega, None, None, None, None, None, None


def _phase_chain_torch(raw: torch.Tensor, dt_raw_k: torch.Tensor, omega,
                       *, H: int, R: int, K: int, ka: int, per_head: bool,
                       sign: float, omega_per_head: bool = False
                       ) -> torch.Tensor:
    """Plain torch phase chain (non-CUDA / fp64-oracle fallback).  `omega_per_head`: `raw` holds one angle per
    head, broadcast over the `ka` rotated pairs, and `omega` is [H]."""
    B, L = raw.shape[0], raw.shape[1]
    Hk = R * H
    raw_f = _ref_f(raw)
    if omega_per_head:
        ang = raw_f.view(B, L, H, 1).expand(B, L, H, ka)
        ang = F.pad(torch.tanh(ang) * math.pi, (0, K - ka))    # [B,L,H,K]
        if omega is not None:
            ang = ang * omega.view(H, 1)
        if R > 1:
            ang = (ang.unsqueeze(2).expand(B, L, R, H, K)
                   .reshape(B, L, Hk, K))
        return sign * (ang * dt_raw_k.permute(0, 2, 1)[..., None])
    ang = raw_f.view(B, L, H, ka) if per_head else raw_f.view(B, L, ka)
    if ka < K:
        ang = F.pad(ang, (0, K - ka))
    ang = torch.tanh(ang) * math.pi                        # [B,L,(H,)K]
    if not per_head:
        ang = ang.unsqueeze(2).expand(B, L, H, K)
    if R > 1:
        ang = ang.unsqueeze(2).expand(B, L, R, H, K).reshape(B, L, Hk, K)
    incr = sign * (ang * dt_raw_k.permute(0, 2, 1)[..., None])
    if omega is not None:
        # [K] (pair) or [H, K] (head_pair: r-major kernel heads)
        incr = incr * (omega if omega.dim() == 1 else omega.repeat(R, 1))
    return incr


def _wrap2pi_t(x: torch.Tensor) -> torch.Tensor:
    """x - 2 pi round(x / 2 pi): reduce the running phase to [-pi, pi].

    Consumers use only cos / sin, so this is exact math with identity derivative; it keeps a large trained
    phase at fp32's fine resolution.  fp64 inputs are reduced in fp64."""
    two_pi = 2.0 * math.pi
    return x - two_pi * torch.round(x.detach() / two_pi)


def _theta_increments(raw: torch.Tensor, dt_raw_k: torch.Tensor, omega,
                      *, H: int, R: int, K: int, ka: int, per_head: bool,
                      use_kernel: bool, sign: float,
                      omega_per_head: bool = False) -> torch.Tensor:
    if omega_per_head:
        # per-head eigenvalue: fused `phase_law` on CUDA, this torch chain elsewhere
        return _phase_chain_torch(raw, dt_raw_k, omega, H=H, R=R, K=K,
                                  ka=ka, per_head=per_head, sign=sign,
                                  omega_per_head=True)
    if use_kernel and R > 1:
        # recompute in backward: trades a relaunch for less saved memory
        return _OmegaFoldFn.apply(raw, dt_raw_k, omega, H, R, K, ka,
                                  per_head, sign)
    if use_kernel:
        incr = _kernel_phase_pre_omega(raw, dt_raw_k, H=H, R=R, K=K, ka=ka,
                                       per_head=per_head, sign=sign)
        if omega is not None:
            incr = incr * omega
        return incr
    return _phase_chain_torch(raw, dt_raw_k, omega, H=H, R=R, K=K, ka=ka,
                              per_head=per_head, sign=sign)


def _dt_tail(dt_lin: torch.Tensor, use_compile: bool) -> torch.Tensor:
    if use_compile:
        global _dt_tail_compiled
        if _dt_tail_compiled is None:
            _dt_tail_compiled = torch.compile(_dt_tail_eager)
        return _dt_tail_compiled(dt_lin)
    return _dt_tail_eager(dt_lin)


def _a_mag_tail(a_lin: torch.Tensor, floor: float,
                use_compile: bool, act: str = "heavy_tail") -> torch.Tensor:
    fn = _a_mag_tail_eager if act == "heavy_tail" else _a_mag_softplus_eager
    if use_compile:
        if act not in _a_mag_tail_compiled:
            _a_mag_tail_compiled[act] = torch.compile(fn)
        return _a_mag_tail_compiled[act](a_lin, floor)
    return fn(a_lin, floor)


def _bc_bias_init(shape, mode: str, R: int, N: int) -> torch.Tensor:
    """Static B/C bias init: "ones" is Mamba-3's constant; "spread" gives sibling r the direction
    `r_off[r] * dct[r]` (DCT-II rows, orthogonal), so sibling 0 equals "ones" and R == 1 reduces to it.

    Initialisation conditioning only: the operator's block rank does not change.
    """
    base = torch.ones(shape)
    if mode == "ones" or len(shape) < 4 or R < 2:
        return base
    # shape is (H, R, N, R') for B / (H, R, N, O) for C; axis 1 is the
    # sibling (kernel-head) axis and axis 2 is the state axis.
    r_off = torch.linspace(0.0, math.log(2.0), R)              # [R]
    k = torch.arange(R).float()[:, None]                       # [R,1]
    n = torch.arange(N).float()[None, :]                       # [1,N]
    dct = torch.cos(math.pi * k * (n + 0.5) / N)               # [R,N]
    return base + (r_off[:, None] * dct)[None, :, :, None]


class _SSDMimoScan(Configurable, _ScanBase):
    """Internal base of `SSDMimoSSM`: token-to-SSM projections, SSM parameters and the recurrence.

    Never built on its own; its constructor keywords are documented on `SSDMimoSSM`.
    """

    config_family = "ssd_mimo"

    def __init__(
        self,
        d_inner: int,
        d_state: int = 64,
        *,
        d_ssm_in: int = 1,
        d_ssm_out: int | None = None,
        n_groups: int | None = None,
        n_heads: int | None = None,
        d_head: int | None = None,
        row_blocks: list | None = None,
        r_mix: str = "auto",
        r_mix_dyn: bool | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        a_proj_bias: bool = True,
        trap_proj_bias: bool = True,
        write_rule: str = WRITE_RULE_DEFAULT,
        skip: bool = True,
        skip_sharing: str = "head",
        pair_layout: str = "interleaved",
        proj_source: str = "layer_input",
        d_model: int | None = None,
        bc_conv: bool | None = None,
        d_conv: int = 4,
        a_dyn_per_sibling: bool | None = None,
        a_act: Literal["heavy_tail", "softplus"] = "heavy_tail",
        # rotation and B/C maps: resolved by config_spec, recorded under these names
        theta_per_head: bool | None = None,
        learn_theta_freq: bool | None = None,
        use_theta_h: bool | None = None,
        omega_layout: str | None = None,
        theta_init: str | None = None,
        b_norm: bool | None = None,
        c_norm: bool | None = None,
        bc_bias: str | None = None,
        b_gate: bool | None = None,
        c_gate: bool | None = None,
        bc_bias_init: str | None = None,
    ):
        if "_parameters" not in self.__dict__:      # SSDMimoSSM initializes the Module first
            super().__init__()
        # Execution defaults; the caller's values arrive via `_apply_execution`.
        self.kernel_options = {}
        self.backend, self.precision, self.chunk_size = "auto", "io", "auto"
        self.m3_kernel, self._compile_token_maps, self.allow_chunk_autotune = "native", False, True
        self.conv_kernel = "auto"
        d_inner = positive_int(d_inner, "d_inner")
        d_state = positive_int(d_state, "d_state")
        check_dt_range(dt_min, dt_max, dt_init_floor)
        # selection source: where the token maps read from
        self.proj_source, d_sel, self.bc_conv = resolve_proj_source(
            proj_source, d_inner, d_model, bc_conv, who="SSDMimoSSM", d_conv=d_conv)
        self.d_sel = d_sel
        write_rule = resolve_write_rule(write_rule, "SSDMimoSSM")

        # geometry: O defaults to R (width-preserving)
        R = d_ssm_in = check_rank(d_ssm_in, "d_ssm_in")
        N = d_state
        n_heads, P, G = _ssd_geometry(
            d_inner, d_ssm_in=R, n_groups=n_groups, n_heads=n_heads, d_head=d_head)
        # fused-kernel channel tile
        self.G_kernel = -(-G // 16) * 16
        O = d_ssm_out = check_rank(R if d_ssm_out is None else d_ssm_out, "d_ssm_out")
        self.O = O
        upstream_refusals = []
        if R != O or N not in (16, 32, 64, 128) or G * R not in (32, 64, 128):
            upstream_refusals.append("upstream M3 requires d_ssm_in == d_ssm_out, d_state in {16,32,64,128}, n_groups * d_ssm_in in {32,64,128}")
        self.d_out = n_heads * G * O
        self.out_width = self.d_out
        self.H, self.P, self.G, self.R, self.N = n_heads, P, G, R, N
        self.d_inner = d_inner

        # row_blocks / rotation / maps spec: resolved once; below reads the
        # canonical dict (`a_dyn`, `rot_mode`, ... are internal views, not constructor names)
        spec = _config_spec.resolve(
            "ssd_mimo", d_state=N, d_ssm_in=R, d_ssm_out=O,
            **_config_spec.spec_kwargs(locals()))
        self.spec = spec
        _config_spec.check_law_levers("SSDMimoSSM", spec["row_blocks"], _RATE_LEVERS, dict(
            a_dyn_per_sibling=a_dyn_per_sibling, a_act=a_act, a_proj_bias=a_proj_bias))
        n_complex = sum(b["rows"] for b in spec["row_blocks"] if b["complex"])
        a_dyn = spec["row_blocks"][0]["decay"] == _config_spec.HEAD_DYNAMIC
        rot = spec["rotation"] or {}
        rot_mode = "m3" if n_complex else None
        use_theta_h = bool(rot.get("use_theta_h", False))
        learn_theta_freq = bool(rot.get("learn_theta_freq", False))
        theta_per_head = bool(rot.get("theta_per_head", False))
        omega_layout = rot.get("omega_layout", "pair")
        theta_init = rot.get("theta_init", "kaiming")
        mp = spec["maps"]
        use_g_b, use_g_c = mp["b_gate"], mp["c_gate"]
        bias_gran = mp["bc_bias"]
        if not n_complex or use_theta_h or learn_theta_freq:
            upstream_refusals.append("upstream M3 requires a complex row block with "
                                     "use_theta_h=False and learn_theta_freq=False")
        if omega_layout != "pair":
            upstream_refusals.append("upstream M3 runs Mamba-3's per-pair angles; "
                                     f"omega_layout={omega_layout!r} is the "
                                     "native route only")
        #: why `kernel_options["m3_kernel"] = "upstream"` is refused here (None: it is not)
        self._upstream_refusal = "; ".join(upstream_refusals) or None

        # Paper 1, Section 4 (crosstalk theorem): per-subsystem rank ceiling min(N, R, O);
        # the layer aggregates R subsystems through zeta, so it is not itself bounded.
        if (R == 1) != (self.O == 1):
            warnings.warn(
                f"SSDMimoSSM: d_ssm_in={R}, d_ssm_out={self.O} -- exactly one of them is 1, "
                "so each SUBSYSTEM's local temporal block C_t Phi(t,s) B_s "
                "has rank at most min(d_state, d_ssm_in, d_ssm_out) = 1, the same ceiling as "
                "SISO, while the larger rank is still paid for in state, "
                "parameters and work (the crosstalk theorem of Paper 1, Section 4). "
                "The LAYER is not bounded by this: it aggregates R "
                "subsystems through the zeta sibling mix, so the "
                "configuration still buys state and parameters and may buy "
                "layer-level rank.  Use d_ssm_in == d_ssm_out to lift the per-subsystem "
                "ceiling itself.", RuntimeWarning, stacklevel=2)

        self.skip = bool(skip)
        if skip_sharing not in ("input_output", "channel", "head"):
            raise ValueError("skip_sharing must be input_output, channel, or head")
        self.skip_sharing = skip_sharing
        if self.skip:
            shape = {"input_output": (n_heads, G, R, self.O),
                     "channel": (n_heads, G, R), "head": (n_heads,)}[skip_sharing]
            self.D = skip_param(shape)

        # sibling mix zeta_{h,r}: init 1/R so the reduction starts as a plain mean
        if r_mix not in _R_MIX_MODES:
            raise ValueError(
                f"SSDMimoSSM: r_mix must be one of {_R_MIX_MODES}, "
                f"got {r_mix!r}")
        self.r_mix_mode = ("none" if R == 1 else "learned") \
            if r_mix == "auto" else r_mix
        if r_mix_dyn is None:
            r_mix_dyn = R > 1 and self.r_mix_mode != "none"
        if self.r_mix_mode == "none":
            if r_mix_dyn:
                # refuse rather than silently drop an explicit request
                raise ValueError(
                    "SSDMimoSSM(r_mix_dyn=True) needs an active sibling "
                    f"mix, but r_mix resolved to 'none' (d_ssm_in={R}); pass "
                    "r_mix='learned' to keep it")
            self.r_mix = None
            self.r_mix_dyn = False
        else:
            self.r_mix = nn.Parameter(torch.full((n_heads, R), 1.0 / R))
            self.r_mix_dyn = bool(r_mix_dyn)
            if r_mix_dyn:
                self.r_mix_proj = nn.Linear(d_sel, n_heads * R, bias=False)
                nn.init.zeros_(self.r_mix_proj.weight)
        self._register_load_state_dict_pre_hook(self._validate_pair_layout)

        # B/C projections (always token-dependent)
        self.B_proj = nn.Linear(d_sel, N * R, bias=False)
        self.C_proj = nn.Linear(d_sel, N * O, bias=False)
        # optional depthwise causal conv + SiLU on the raw (B, C) projections
        self.bc_conv1d = (CausalDepthwiseConv(N * R + N * O, d_conv)
                          if self.bc_conv else None)

        # gB: per-r structure in the write; gC: reweights the per-o read
        self.use_g_b = bool(use_g_b)
        self.use_g_c = bool(use_g_c)
        if R > 1 and not self.use_g_b:
            warnings.warn(
                f"SSDMimoSSM(d_ssm_in={R}, b_gate=False): sibling-specific write gating "
                "is disabled; biases or different clocks may still distinguish siblings.",
                RuntimeWarning, stacklevel=2)
        if O > 1 and not self.use_g_c:
            warnings.warn(
                f"SSDMimoSSM(d_ssm_out={O} > 1, c_gate=False): output-specific read gating "
                "is disabled; the read projection still has distinct output columns.",
                RuntimeWarning, stacklevel=2)
        # Gate weights are compact: the K gated complex pairs (one factor per pair, shared by both
        # coordinates), then the gated real rows.  `_proj_w` gathers them to the N-row layout
        # (ungated rows read an appended zero row, so their factor is exactly 1).
        gate_cx = any(b["gate"] for b in spec["row_blocks"] if b["complex"])
        gate_re = any(b["gate"] for b in spec["row_blocks"] if not b["complex"])
        self.gate_blocks = (gate_cx, gate_re)
        self.gate_pair_tied = n_complex > 0 and gate_cx
        self.gate_masked = (n_complex > 0 and not gate_cx) or (
            N > n_complex and not gate_re)
        n_gate = (n_complex // 2 if gate_cx else 0) + (
            N - n_complex if gate_re else 0)
        self.n_gate_rows = n_gate
        if self.use_g_b:
            self.gB_proj = nn.Linear(d_sel, n_gate * R, bias=False)
            nn.init.zeros_(self.gB_proj.weight)
        if self.use_g_c:
            self.gC_proj = nn.Linear(d_sel, n_gate * O, bias=False)
            nn.init.zeros_(self.gC_proj.weight)

        # delta projection (per head, broadcast over G)
        dt_out = n_heads * R
        self.dt_proj = nn.Linear(d_sel, dt_out, bias=True)
        with torch.no_grad():
            self.dt_proj.bias.copy_(init_dt_bias(dt_out, dt_min, dt_max, dt_init_floor))
        self.dt_proj.bias._no_reinit = True

        self.a_dyn = bool(a_dyn)
        self.a_dyn_per_sibling = bool(a_dyn_per_sibling)
        if a_act not in A_ACTS:
            raise ValueError(f"a_act must be one of {A_ACTS}, got {a_act!r}")
        self.a_act = a_act
        if self.a_dyn:
            self.A_log = None
            rate_width = n_heads * R if self.a_dyn_per_sibling else n_heads
            self.a_dyn_proj = nn.Linear(d_sel, rate_width, bias=a_proj_bias)
            if self.a_dyn_proj.bias is not None:
                nn.init.zeros_(self.a_dyn_proj.bias)
            # the weight is decayed like any Linear (as Mamba-3's dd_A rows); the bias is exempt
            if self.a_dyn_proj.bias is not None:
                self.a_dyn_proj.bias._no_weight_decay = True
        else:
            # static law: one decay rate per (head, r), constant over N and G
            r_off = torch.linspace(0.0, math.log(2.0), R)
            hbase = torch.exp(torch.linspace(math.log(0.05), math.log(2.0), n_heads))
            A = hbase[:, None] * torch.exp(r_off)[None, :]  # [H, R]
            self.A_log = nn.Parameter(A.log())
            self.A_log._no_weight_decay = True

        # B/C norm + biases; consumers test `bc_bias` / `head_bias`, never the granularity alone
        self.bias_gran = bias_gran
        self.bc_bias = bias_gran != "none"
        self.head_bias = bias_gran == "head"
        self.b_norm = bool(mp["b_norm"])
        self.c_norm = bool(mp["c_norm"])
        # resolved by the spec, never "auto": a module must be able to say what it built
        self.bc_bias_init = mp["bc_bias_init"]
        if self.b_norm:
            self.B_norm = nn.RMSNorm(N, eps=1e-5)
        if self.c_norm:
            self.C_norm = nn.RMSNorm(N, eps=1e-5)
        if self.bc_bias:
            # static biases are per kernel head (h, r): the cheapest place siblings can differ
            bshape = {"shared": (N,),
                      "head": (n_heads, R, N, R)}[bias_gran]
            cshape = {"shared": (N,),
                      "head": (n_heads, R, N, O)}[bias_gran]
            self.B_bias = nn.Parameter(_bc_bias_init(
                bshape, self.bc_bias_init, R, N))
            self.C_bias = nn.Parameter(_bc_bias_init(
                cshape, self.bc_bias_init, R, N))

        # theta_h: static per-head read-phase offset, C~ = R(Theta + theta_h)^T (C + gamma); it ADDS
        # to the physical phase Theta (shared convention with HybridMambaSSM, pinned by
        # a theta_h convention test).  One entry per rotated pair; the kernels see all N/2 pairs
        # with the real tail zero-padded host-side (`_theta_h_pairs`).
        self.use_theta_h = bool(use_theta_h) and n_complex > 0
        if self.use_theta_h:
            self.theta_h = nn.Parameter(
                torch.zeros(n_heads, n_complex // 2))
            self.theta_h._no_weight_decay = True

        # rot_mode is an internal view: "m3" (Delta-scaled paired rotation) iff a complex block exists
        self.rot_mode = rot_mode
        if pair_layout not in ("interleaved", "half"):
            raise ValueError(
                "pair_layout must be 'interleaved' or 'half', got "
                f"{pair_layout!r}")
        self.pair_layout = pair_layout
        if pair_layout == "half":
            # m3 route only; anything outside the kernel envelope is refused by name in the launchers
            if self.rot_mode != "m3":
                raise NotImplementedError(
                    "pair_layout='half' is the rotation law's layout: it "
                    "needs a complex row block in `row_blocks`")
            if N % 32:
                raise NotImplementedError(
                    f"pair_layout='half' needs N a multiple of 32, got {N}")
        self.register_buffer(
            "pair_layout_code",
            torch.tensor(1 if pair_layout == "half" else 0,
                         dtype=torch.int64),
            persistent=True)
        # row n of the per-coordinate gate layout reads compact gate row gate_tie_idx[n]; None = untied
        self.register_buffer(
            "gate_tie_idx",
            self._gate_tie_index(N, n_complex, pair_layout, self.gate_blocks)
            if (self.gate_pair_tied or self.gate_masked) else None,
            persistent=False)
        self._register_load_state_dict_pre_hook(
            self._refuse_per_coordinate_complex_gates)
        self.theta_per_head = theta_per_head
        # omega = pi tanh(theta_freq); layouts "pair" (K_rot,), "head_pair" (H, K_rot), "head" (H,)
        # ("head": one phase per (head, sibling) shared by all its pairs)
        self.omega_layout = omega_layout if n_complex > 0 else "pair"
        self.omega_per_head = self.omega_layout == "head"
        self.theta_init = theta_init
        # theta_per_head widens the angle projection; otherwise heads share features
        if self.rot_mode == "m3":
            # the complex block is even (spec) and leads the state
            self.n_rope_angles = n_complex // 2
            #: angle features per angle head: K_rot, or 1 (per-head eigenvalue)
            self.n_theta_feat = 1 if self.omega_per_head else self.n_rope_angles
            theta_width = ((n_heads if self.theta_per_head else 1)
                          * self.n_theta_feat)
            self.theta_proj = nn.Linear(d_sel, theta_width, bias=False)
            if theta_init == "zero":
                # no rotation at init (tanh(0) = 0)
                nn.init.zeros_(self.theta_proj.weight)
            elif theta_init == "small":
                # kaiming draw x 0.1 sqrt(3): std(q) ~ 0.1 on unit-RMS inputs
                with torch.no_grad():
                    self.theta_proj.weight.mul_(SMALL_SCALE)
            # theta_freq: omega_k = pi * tanh(theta_freq_k) stays in Mamba-3's [-pi, pi] band;
            # init 0.999 pi.  learn_theta_freq=False keeps the constant pi (no parameter).
            self.learn_theta_freq = bool(learn_theta_freq)
            if self.learn_theta_freq:
                # shape by `omega_layout`
                shape = {"pair": (self.n_rope_angles,),
                         "head_pair": (n_heads, self.n_rope_angles),
                         "head": (n_heads,)}[self.omega_layout]
                self.theta_freq = freq_param(shape)
        else:
            self.learn_theta_freq = False
        # biases fold before the rotation (Mamba-3 order)
        self.bias_pre_rot = self.rot_mode == "m3"

        # endpoint weight lambda_t = sigmoid(trap_proj(sel_t)), per token and head
        self.write_rule = write_rule
        if not self.use_trap and not trap_proj_bias:
            raise ValueError(
                f"SSDMimoSSM: trap_proj_bias=False, but write_rule={write_rule!r} builds no trapezoid weight projection, so it would "
                "be ignored; drop trap_proj_bias or use write_rule='trapezoid'")
        if self.use_trap:
            self.trap_proj = nn.Linear(d_sel, n_heads, bias=trap_proj_bias)
            if self.trap_proj.bias is not None:
                nn.init.zeros_(self.trap_proj.bias)

        # Token-map projections run as one GEMM; _proj_attrs and widths define the packed row order.
        self._fuse_proj = True
        specs = [("B", "B_proj", N * R), ("C", "C_proj", N * self.O)]
        if self.use_g_b:
            specs.append(("gB", "gB_proj", N * R))
        if self.use_g_c:
            specs.append(("gC", "gC_proj", N * self.O))
        specs.append(("dt", "dt_proj", dt_out))
        if self.a_dyn:
            specs.append(("a", "a_dyn_proj", self.a_dyn_proj.out_features))
        if self.use_trap:
            specs.append(("trap", "trap_proj", n_heads))
        if self.rot_mode == "m3":
            specs.append(("theta", "theta_proj", self.theta_proj.out_features))
        if self.r_mix_dyn:
            specs.append(("rmix", "r_mix_proj", n_heads * R))
        self._proj_keys = tuple(s[0] for s in specs)
        self._proj_attrs = tuple(s[1] for s in specs)
        self._proj_widths = tuple(s[2] for s in specs)
        # channel-major token GEMM needs the fused tail to consume every slice (eager maps `.view()`
        # theirs, which a channel-major slab does not support)
        self._tail_owns_all = set(s[0] for s in specs) <= {
            "B", "C", "gB", "gC", "dt", "a"}
        self.tok_width = sum(self._proj_widths)
        # B and C adjacent at the slab head: enables their joint narrow and the in-place B/C conv
        self._bc_adjacent = self._proj_keys[:2] == ("B", "C")
        self._bc_width = sum(self._proj_widths[:2])
        self._proj_any_bias = any(
            getattr(self, a).bias is not None for a in self._proj_attrs)
        # zeros for the bias-free sections of the fused bias vector
        self.register_buffer("_proj_zero", torch.zeros(self.tok_width),
                             persistent=False)
        # `tok_proj_weight()`'s cat cache: a plain dict (not a buffer), rebuilt when the key changes
        self._tok_proj_w_cache: dict = {}
        record_constructor_config(
            self, locals(), constructor=_SSDMimoScan.__init__,
            overrides=dict(kernel_options={}, n_groups=G, d_ssm_in=R, d_ssm_out=O,
                           backend="auto", precision="io", chunk_size="auto", m3_kernel="native",
                           compile_token_maps=False, allow_chunk_autotune=True, r_mix=self.r_mix_mode,
                           write_rule=self.write_rule,
                           skip=self.skip, r_mix_dyn=self.r_mix_dyn,
                           a_dyn_per_sibling=self.a_dyn_per_sibling,
                           a_act=self.a_act,
                           bc_conv=self.bc_conv, proj_source=self.proj_source,
                           **_config_spec.flat_spec(spec)),
            drop=("kernel_options", "backend", "precision", "m3_kernel", "compile_token_maps",
                  "chunk_size", "allow_chunk_autotune"))

    @property
    def use_trap(self) -> bool:
        """Read-only view of `write_rule`: is the two-endpoint (trapezoid) rule selected?"""
        return self.write_rule == "trapezoid"

    # pair_layout names the pair of rotation pair k: (2k, 2k+1) "interleaved" or (k, k + N/2) "half".
    # They differ by a fixed permutation of the state axis only: per-pair quantities (theta, theta_h,
    # theta_freq) do not move; per-coordinate ones (B/C rows, B/C norm weights, biases) do.  A checkpoint
    # therefore crosses layouts exactly (a gather); `pair_layout_code` records the layout it was written in.
    def _refuse_per_coordinate_complex_gates(self, state_dict, prefix, *_args):
        refuse_per_coordinate_gates(
            self, state_dict, prefix,
            lambda name: self.N * (self.R if name.startswith("gB") else self.O))

    @classmethod
    def _gate_tie_index(cls, N: int, n_complex: int, pair_layout: str,
                        gated=(True, True)):
        """(N,) long: the compact gate row each state coordinate reads.

        Complex pair k (coordinates 2k, 2k+1 interleaved) reads compact row k, real row n reads K + (n -
        n_complex); `gated` = (complex block gated, real block gated).  An ungated block's rows are not
        stored and read the sentinel row n_gate (the appended zero of `_proj_w`).  Under "half" the index is
        composed with the state permutation."""
        gate_cx, gate_re = gated
        with torch.device("cpu"):
            K = n_complex // 2 if gate_cx else 0
            n_gate = K + ((N - n_complex) if gate_re else 0)
            n = torch.arange(N)
            cx = (n // 2 if gate_cx else torch.full_like(n, n_gate))
            re = (K + n - n_complex if gate_re else torch.full_like(n, n_gate))
            inter = torch.where(n < n_complex, cx, re)
            if pair_layout == "half":
                inter = inter[cls._pair_perm(N, to_half=True)]
        return inter.long()

    def _proj_w(self, attr: str) -> torch.Tensor:
        """Token-map weight of `attr` in the per-coordinate row layout; identity except the compact gate
        weights, which are gathered from (n_gate * blk, d_sel) to (N * blk, d_sel)."""
        w = getattr(self, attr).weight
        idx = getattr(self, "gate_tie_idx", None)
        if idx is None or attr not in ("gB_proj", "gC_proj"):
            return w
        blk = self.R if attr == "gB_proj" else self.O
        w = w.view(self.n_gate_rows, blk, w.shape[-1])
        if self.gate_masked:            # BLOCKGATES: the exact-zero sentinel row
            w = torch.cat([w, w.new_zeros(1, blk, w.shape[-1])], 0)
        return w.index_select(0, idx).reshape(self.N * blk, w.shape[-1])

    def _proj_w_grad(self, attr: str, g: torch.Tensor) -> torch.Tensor:
        """Adjoint of `_proj_w`: reduce a per-coordinate weight gradient onto the compact rows."""
        idx = getattr(self, "gate_tie_idx", None)
        if idx is None or attr not in ("gB_proj", "gC_proj"):
            return g
        blk = self.R if attr == "gB_proj" else self.O
        d = g.shape[-1]
        n_out = self.n_gate_rows + int(self.gate_masked)
        out = g.new_zeros(n_out, blk, d)
        out.index_add_(0, idx, g.reshape(self.N, blk, d))
        # the sentinel row is not a parameter: its gradient is dropped
        return out[:self.n_gate_rows].reshape(-1, d)

    @staticmethod
    def _pair_perm(N: int, to_half: bool, device=None):
        """State-axis permutation: `to_half=True` gives `idx` with `half[i] = interleaved[idx[i]]`; else its inverse."""
        fwd = torch.cat([torch.arange(0, N, 2, device=device),
                         torch.arange(1, N, 2, device=device)])
        if to_half:
            return fwd
        inv = torch.empty_like(fwd)
        inv[fwd] = torch.arange(N, device=device)
        return inv

    def _pair_layout_targets(self):
        """(attribute path, N-axis, inner block) for every per-N tensor."""
        N, R, O = self.N, self.R, self.O
        out = [("B_proj.weight", 0, R), ("C_proj.weight", 0, O)]
        # tied / block-masked gates are stored per pair / gated real row
        # (`_gate_tie_index`): layout-independent, nothing to permute
        tied = getattr(self, "gate_tie_idx", None) is not None
        if getattr(self, "use_g_b", False) and not tied:
            out.append(("gB_proj.weight", 0, R))
        if getattr(self, "use_g_c", False) and not tied:
            out.append(("gC_proj.weight", 0, O))
        if getattr(self, "b_norm", False):
            out.append(("B_norm.weight", 0, 1))
        if getattr(self, "c_norm", False):
            out.append(("C_norm.weight", 0, 1))
        if getattr(self, "bc_bias", False):
            if getattr(self, "head_bias", False):
                out.append(("B_bias", 2, 1))
                out.append(("C_bias", 2, 1))
            else:
                out.append(("B_bias", 0, 1))
                out.append(("C_bias", 0, 1))
        return out

    def _validate_pair_layout(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Refuse a mismatched or absent `pair_layout_code`: loading would silently install a permuted state basis."""
        key = prefix + "pair_layout_code"
        code = state_dict.get(key)
        if code is None:
            raise ValueError(
                f"{key} is absent: this state dict does not declare its pair-layout "
                f"code, so the state-coordinate basis of its B/C rows is "
                f"unknown and loading it into pair_layout="
                f"{self.pair_layout!r} could silently reinterpret the state "
                f"axis. Set {key!r} to the layout it was written in "
                f"(0 = interleaved, 1 = half) and pass it through "
                f"convert_pair_layout before loading.")
        if int(code) != int(self.pair_layout_code):
            raise ValueError("pair_layout differs: explicitly convert_pair_layout before loading")

    def convert_pair_layout(self, state_dict):
        """Return a core state dict converted to this core's pair layout (only the state axis changes)."""
        code = int(state_dict["pair_layout_code"])
        if code not in (0, 1):
            raise ValueError("invalid pair_layout_code")
        state_dict = dict(state_dict)
        dst_half = self.pair_layout == "half"
        if code == int(dst_half):
            return state_dict
        state_dict["pair_layout_code"] = self.pair_layout_code.clone()
        N = self.N
        idx = self._pair_perm(N, to_half=dst_half)
        for name, axis, blk in self._pair_layout_targets():
            t = state_dict[name]
            if t.shape[axis] != N * blk:
                raise ValueError(f"{name}: expected state axis of width {N * blk}")
            if blk == 1:
                state_dict[name] = t.index_select(axis, idx.to(t.device)).contiguous()
            else:
                v = t.unflatten(axis, (N, blk)).index_select(axis, idx.to(t.device))
                state_dict[name] = v.flatten(axis, axis + 1).contiguous()
        if self.bc_conv1d is not None:
            # Convolution channels follow the concatenated raw B/C rows.
            b_idx = (idx[:, None] * self.R + torch.arange(self.R)).flatten()
            c_idx = (idx[:, None] * self.O + torch.arange(self.O)).flatten() + N * self.R
            conv_idx = torch.cat((b_idx, c_idx))
            for name in ("bc_conv1d.conv.weight", "bc_conv1d.conv.bias"):
                t = state_dict[name]
                state_dict[name] = t.index_select(0, conv_idx.to(t.device)).contiguous()
        return state_dict

    def tok_proj_weight(self):
        """Packed token-projection weight and optional bias (a cat of the per-projection Linears).

        Public so an enclosing mixer can cat it onto its own in_proj rows (single GEMM under
        proj_source="layer_input") and pass the result back as ``tok_raw``.  With the "wcat" glue item on
        (``SSD_MIMO_GLUE``) the cats are memoised per parameter version in ``_tok_proj_w_cache``.
        """
        mods = [getattr(self, a) for a in self._proj_attrs]
        ws = [self._proj_w(a) for a in self._proj_attrs]
        if not glue_on("wcat"):
            w = torch.cat(ws, dim=0)
            b = None
            if self._proj_any_bias:
                dtz = dt_fp32_on(self.kernel_options)      # dt bias rides the tail in fp32
                b = torch.cat([
                    m.bias if (m.bias is not None
                               and not (dtz and a == "dt_proj"))
                    else self._proj_zero[:n]
                    for a, m, n in zip(self._proj_attrs, mods,
                                       self._proj_widths)])
            return w, b
        cache = self._tok_proj_w_cache
        w_key = param_cache_key([m.weight for m in mods])
        w = cached_cat(cache.setdefault("w", {}), w_key, ws, dim=0)
        b = None
        if self._proj_any_bias:
            dtz = dt_fp32_on(self.kernel_options)
            b_tensors = [m.bias if (m.bias is not None
                                    and not (dtz and a == "dt_proj"))
                         else self._proj_zero[:n]
                         for a, m, n in zip(self._proj_attrs, mods,
                                            self._proj_widths)]
            b_key = param_cache_key(
                [m.bias for m in mods if m.bias is not None], dtz)
            b = cached_cat(cache.setdefault("b", {}), b_key, b_tensors, dim=0)
        return w, b

    def tok_proj_params(self):
        """The parameters `tok_proj_weight()` cats, in its order (its cache key)."""
        out = []
        for a in self._proj_attrs:
            mod = getattr(self, a)
            out.append(mod.weight)
            if mod.bias is not None:
                out.append(mod.bias)
        return out

    @property
    def _conv_pkg(self) -> bool:
        """Do the convolutions run on the ``causal_conv1d`` package (the opt-in ``conv_kernel="causal_conv1d"``)?"""
        return check_conv_kernel(self.conv_kernel)

    @property
    def _conv_raw(self) -> bool:
        """Do the convolutions have raw (autograd-free) forward / adjoint entry points the one-node routes can drive:
        the package or the plain-Triton conv (not torch's ``F.conv1d``)?"""
        return bool(self._conv_impl)

    @property
    def _conv_impl(self):
        """``package=`` of the conv entry points: True (package), ``"triton"`` or False (torch); `layers.conv.conv_impl`."""
        from ..layers.conv import conv_impl
        return conv_impl(self.conv_kernel)

    @property
    def conv_kernel_in_use(self) -> str:
        """The conv that actually runs in this layer: ``"triton"`` (the default plain-Triton conv; CUDA parameters, backend
        ``"auto"`` / ``"cuda"``), ``"causal_conv1d"`` (the opt-in package; additionally fp32/fp16/bf16 only) or ``"torch"``
        (``conv_kernel="torch"``, a CPU module, ``backend="torch"``, or the package on an fp64 module)."""
        from ..layers.conv import conv_kernel_running
        return conv_kernel_running(self)

    def split_tok_raw(self, raw: torch.Tensor) -> dict:
        """Split a caller-computed [B, L, tok_width] projection output into the slices `_proj_split` returns."""
        if raw.shape[-1] != self.tok_width:
            raise ValueError(
                f"tok_raw last dim {raw.shape[-1]} != tok_width="
                f"{self.tok_width} (the fused token-map projection's "
                f"width; use `tok_proj_weight()` to build it)")
        out = dict(zip(self._proj_keys,
                       raw.split(self._proj_widths, dim=-1)))
        if dt_fp32_on(self.kernel_options) and self.dt_proj.bias is not None:
            # the slab's dt columns are bias-free (see `tok_proj_weight`);
            # consumers add this in fp32 before the softplus
            out["_dtb"] = self.dt_proj.bias
        if self._bc_adjacent:
            # joint B|C narrow of the slab (lets the B/C conv read it in place); must be `narrow`, not an
            # as_strided widening, which drops gradients past the B slice
            out["_bc"] = raw.narrow(-1, 0, self._bc_width)
        return out

    def _proj_split(self, u: torch.Tensor, channel_major: bool = False) -> dict:
        """One fused GEMM for all token-map projections; returns raw slices keyed by ``self._proj_keys``
        (views of one [B, L, tok_width] buffer; equal to separate Linears up to GEMM tolerance)."""
        if not self._fuse_proj:
            dtz = dt_fp32_on(self.kernel_options) and self.dt_proj.bias is not None
            out = {k: F.linear(u, self._proj_w(a),
                               None if (dtz and a == "dt_proj")
                               else getattr(self, a).bias)
                   for k, a in zip(self._proj_keys, self._proj_attrs)}
            if dtz:     # same arithmetic as the fused slab: bias-free dt + fp32 bias
                out["_dtb"] = self.dt_proj.bias
            return out
        return self.split_tok_raw(self.proj_slab(u, channel_major))

    def proj_slab(self, u: torch.Tensor,
                  channel_major: bool = False) -> torch.Tensor:
        """The [B, L, tok_width] fused token-projection slab (public: the combined node needs the whole buffer)."""
        w, b = self.tok_proj_weight()
        if channel_major:
            # channel-major GEMM (`W @ u^T`): puts the adjoint in the drive's own layout so autograd's sum
            # with the scan's channel-major gradient streams.  Only the fused tail may read this slab.
            outc = matmul_channel_major(w, u)               # [B, tok, L]
            if b is not None:
                outc = outc + b.to(outc.dtype)[:, None]
            out = outc.transpose(1, 2)                      # [B, L, tok]
        else:
            # `u` may be a channel-major view; `linear_keep_layout` avoids a clone of the drive
            out = linear_keep_layout(u, w, b)
        return out


    def _bc_source(self, parts, prefix=None):
        """The flat (B, L, N*R) / (B, L, N*O) B / C slabs after the optional B/C conv.

        `prefix` [B, d_conv-1, N*R + N*O]: raw B/C rows before this chunk (the `bc_conv_state` carry);
        convolving `cat([prefix, joint])` and dropping the first d_conv-1 outputs continues the sequence
        exactly.  The only place conv history enters; the scan never sees it."""
        Bt, Ct = parts["B"], parts["C"]
        if self.bc_conv1d is None:
            if prefix is not None:
                raise ValueError(
                    "state carry 'bc_conv_state' on a core built with "
                    "bc_conv=False: there is no B/C convolution to prefix")
            return Bt, Ct
        if prefix is not None:
            if getattr(self, "_bc_capture", None) is not None:
                # assertion: `combined_refusal` already excludes state carries
                raise NotImplementedError(
                    "state carry 'bc_conv_state' inside the combined "
                    "SSD-MIMO node: the node captures the convolution's "
                    "own input and `combined_refusal` excludes carries "
                    "-- ask it first")
            k1 = self.bc_conv1d.d_conv - 1
            cur = parts.get("_bc")
            if cur is None:
                cur = torch.cat([Bt, Ct], dim=-1)
            joint = torch.cat([prefix.to(cur.dtype), cur], dim=1)
            # a continuation takes the plain fused conv and trims the prefix back off
            from ..layers.conv import causal_depthwise_conv1d
            bc = causal_depthwise_conv1d(
                joint.contiguous(), self.bc_conv1d.conv.weight,
                self.bc_conv1d.conv.bias, self.bc_conv1d.d_conv,
                activation="silu", allow_channel_last=True, package=self._conv_impl)[:, k1:]
            return bc.split([self.N * self.R, self.N * self.O], dim=-1)
        # B and C are the first two slab slices, so `_bc` is a view and the conv can run channel-last
        # on the slab in place (as Mamba-2's single xBC conv does)
        joint = parts.get("_bc")
        if joint is None:
            joint = torch.cat([Bt, Ct], dim=-1)
        cap = getattr(self, "_bc_capture", None)
        conv = self.bc_conv1d.conv
        view_ok = self._conv_raw and glue_on("bc") and _bc_narrow_conv_ready(joint, self._conv_impl)
        if cap is None:
            # keep the narrow view when the conv accepts its strides (a copy scales with the slab)
            if view_ok:
                from ..layers.conv import causal_depthwise_conv1d_narrow_cl
                bc = causal_depthwise_conv1d_narrow_cl(
                    joint, conv.weight, conv.bias, activation="silu", package=self._conv_impl)
            else:
                from ..layers.conv import (causal_depthwise_conv1d)
                bc = causal_depthwise_conv1d(
                    joint.contiguous(), conv.weight, conv.bias,
                    self.bc_conv1d.d_conv, activation="silu",
                    allow_channel_last=True, package=self._conv_impl)
        else:
            # the combined node's backward needs the conv's input: capture it here and skip the
            # wrapper's autograd node
            from ..layers.conv import (causal_conv1d_cl_fwd)
            xcl = joint.transpose(1, 2)
            cap.append(xcl)
            out = (torch.empty(joint.shape, device=joint.device,
                               dtype=joint.dtype).transpose(1, 2)
                   if view_ok else None)
            bc = causal_conv1d_cl_fwd(xcl, conv.weight, conv.bias,
                                      "silu", out=out, package=self._conv_impl).transpose(1, 2)
        return bc.split([self.N * self.R, self.N * self.O], dim=-1)

    def _tail_will_fuse(self, u: torch.Tensor) -> bool:
        """Will `_fused_token_tails` run for this input?  Asked before `parts` exists (it sets the
        token GEMM layout); config and dtype only, `tails.can_fuse` re-checks the slabs."""
        from ..layers import tails as _tails
        return bool(self._fuse_proj and _tails.tails_enabled()
                    and _tail_law_is_stock()
                    and u.is_cuda and u.dtype != torch.float64)

    def _fused_token_tails(self, parts, L: int, bc_prefix=None):
        """Fused B/C norm, gates and clock preparation in one kernel; None outside its envelope
        (the caller then runs the same ops in torch)."""
        from ..layers import tails as _tails
        if not (self._fuse_proj and _tails.tails_enabled()
                and _tail_law_is_stock()):
            return None
        N, R, O, H = self.N, self.R, self.O, self.H
        b_src, c_src = self._bc_source(parts, prefix=bc_prefix)
        gb_src = parts.get("gB")
        gc_src = parts.get("gC")
        dt_src = parts["dt"]
        dtb = parts.get("_dtb")
        a_src = parts.get("a")
        if not _tails.can_fuse(b_src, c_src, gb_src, gc_src, dt_src, a_src):
            return None

        shared_bias = bool(self.bc_bias) and not self.head_bias
        fold_gb = bool(self.use_g_b) and R == 1
        fold_gc = bool(self.use_g_c)
        # without norm, shared bias or gate, B / C stay views of the slab (no copy)
        need_b = bool(self.b_norm or shared_bias or fold_gb)
        need_c = bool(self.c_norm or shared_bias or fold_gc)

        def _eps(norm):
            # eps=None means fp32 epsilon in F.rms_norm whatever the input dtype (finfo(bf16).eps
            # would be a 0.28-relative error under bf16)
            e = norm.eps
            return float(torch.finfo(torch.float32).eps if e is None else e)

        cfg = _tails.TailCfg(
            N=N, R=R, O=O, H=H, L=L,
            bc_norm=bool(self.b_norm or self.c_norm),
            b_norm=bool(self.b_norm), c_norm=bool(self.c_norm),
            shared_bias=shared_bias,
            fold_gb=fold_gb, fold_gc=fold_gc, a_dyn=bool(self.a_dyn),
            a_per_sibling=bool(self.a_dyn_per_sibling),
            has_dt_bias=dtb is not None, has_a_bias=False,
            eps_b=_eps(self.B_norm) if self.b_norm else 0.0,
            eps_c=_eps(self.C_norm) if self.c_norm else 0.0,
            a_floor=1e-4, a_heavy_tail=(self.a_act == "heavy_tail"))

        # the dtype the eager chain would have produced (autocast keeps
        # fp32 parameters, and a bf16 map plus an fp32 bias promotes)
        def _odt(base, bias, gate):
            dt = base.dtype
            if bias is not None:
                dt = torch.promote_types(dt, bias.dtype)
            if gate is not None:
                dt = torch.promote_types(dt, gate.dtype)
            return dt

        Bt, Ct, dt_k, dec_k = _tails.token_tails(
            cfg,
            b_src=b_src if need_b else None,
            c_src=c_src if need_c else None,
            gb_src=gb_src if fold_gb else None,
            gc_src=gc_src if fold_gc else None,
            dt_src=dt_src, a_src=a_src if self.a_dyn else None,
            w_b=self.B_norm.weight if self.b_norm else None,
            w_c=self.C_norm.weight if self.c_norm else None,
            bias_b=self.B_bias if shared_bias else None,
            bias_c=self.C_bias if shared_bias else None,
            bias_dt=dtb, bias_a=None,
            need_b=need_b, need_c=need_c,
            out_dtype_b=_odt(b_src, self.B_bias if shared_bias else None,
                             gb_src if fold_gb else None),
            out_dtype_c=_odt(c_src, self.C_bias if shared_bias else None,
                             gc_src if fold_gc else None))
        B = dt_src.shape[0]
        if not need_b:
            Bt = b_src.view(B, L, N, R)
        if not need_c:
            Ct = c_src.view(B, L, N, O)
        gB = (None if fold_gb or not self.use_g_b
              else 1.0 + gb_src.view(B, L, N, R))
        return Bt, Ct, gB, dt_k, dec_k

    def _token_maps(self, x: torch.Tensor, bc_conv_state=None, parts=None,
                    wide: bool = True, bc_prefix=None):
        """Project the selection input u (see :meth:`_select`) to B, C, gB, gC, dt, dt_dec.

        ``bc_conv_state`` (step path only): the (B, C) conv state; given, the conv runs as a one-token
        step and the new state is returned as a 7th element.  ``wide`` returns dt/dt_dec broadcast over G
        to the head width P (what the step recurrence indexes); ``wide=False`` keeps the natural
        (B, L, H, R) width, avoiding a full-size zero-filled gradient in backward.

        Returns
        -------
        Bt : (B, L, N, R)
        Ct : (B, L, N, O)
        gB : (B, L, N, R) or None
        gC : (B, L, N, O) or None
        dt : (B, L, H, P) if ``wide`` else (B, L, H, R)
        dt_dec : same shape as dt, or None when a_dyn is off
        """
        B, L, _ = x.shape
        H, P, G, R, N = self.H, self.P, self.G, self.R, self.N

        # `parts`: slices of the fused token-map GEMM, computed once by the caller; None on the step path
        if parts is None:
            parts = self._proj_split(x)
        Bt = parts["B"]
        Ct = parts["C"]
        new_conv_state = None
        if self.bc_conv1d is not None:
            if bc_conv_state is None:
                Bt, Ct = self._bc_source(parts, prefix=bc_prefix)
            else:
                bc = torch.cat([Bt, Ct], dim=-1)
                bc, new_conv_state = self.bc_conv1d.step(bc.squeeze(1), bc_conv_state)
                bc = bc.unsqueeze(1)
                Bt, Ct = bc.split([N * R, N * self.O], dim=-1)
        Bt = Bt.view(B, L, N, R)
        Ct = Ct.view(B, L, N, self.O)

        # rms_norm_cast: autocast-safe, bit-identical to self.B_norm / self.C_norm
        if self.b_norm:
            Bt = rms_norm_cast(
                Bt.transpose(-1, -2), self.B_norm.weight, self.B_norm.eps
            ).transpose(-1, -2)
        if self.c_norm:
            Ct = rms_norm_cast(
                Ct.transpose(-1, -2), self.C_norm.weight, self.C_norm.eps
            ).transpose(-1, -2)
        if self.bc_bias:
            if self.head_bias:
                # per-head biases are applied by the kernel
                pass
            else:
                Bt = Bt + self.B_bias[:, None]
                Ct = Ct + self.C_bias[:, None]

        gB = (1.0 + parts["gB"].view(B, L, N, R)) if self.use_g_b else None
        gC = (1.0 + parts["gC"].view(B, L, N, self.O)) if self.use_g_c else None

        # dt_raw / a_mag at their natural (unexpanded) width (B,L,H*R) / (B,L,H)
        if parts.get("_dtb") is not None:
            # fp32 for 16-bit slabs; fp64 stays fp64 (never narrow the oracle)
            _w = torch.promote_types(parts["dt"].dtype, torch.float32)
            # no rounding back to the slab dtype: the fused tail and the combined node keep the fp32 softplus output
            dt_raw = _dt_tail(parts["dt"].to(_w) + parts["_dtb"].to(_w),
                              self._compile_token_maps)
        else:
            dt_raw = _dt_tail(parts["dt"], self._compile_token_maps)

        if self.a_dyn:
            # |A_t| = clamp(act(f(x)), min=floor) per head (shared over G, and over R unless
            # a_dyn_per_sibling: then kernel head k = r*H + h has its own)
            a_mag = _a_mag_tail(
                parts["a"], 1e-4, self._compile_token_maps,
                self.a_act)  # (B,L,H) or (B,L,H*R)
            # multiply before the G-broadcast (neither factor varies with G; bit-identical, G times cheaper)
            if self.a_dyn_per_sibling:
                # a_mag is per (head, r) in dt_raw's H*R layout (column h*R + r)
                dt_dec_raw = dt_raw.view(B, L, H, R) * a_mag.view(B, L, H, R)
            else:
                dt_dec_raw = dt_raw.view(B, L, H, R) * a_mag.unsqueeze(-1)
            dt_dec = (dt_dec_raw.view(B, L, H, 1, R)
                      .expand(B, L, H, G, R).reshape(B, L, H, P)
                      if wide else dt_dec_raw)
        else:
            dt_dec = None
        # per (head, r), broadcast across the G groups of the head
        dt = (dt_raw.view(B, L, H, 1, R).expand(B, L, H, G, R)
              .reshape(B, L, H, P) if wide else dt_raw.view(B, L, H, R))
        if bc_conv_state is not None:
            return Bt, Ct, gB, gC, dt, dt_dec, new_conv_state
        return Bt, Ct, gB, gC, dt, dt_dec

    def _select(self, x: torch.Tensor, sel_x: torch.Tensor | None) -> torch.Tensor:
        """Pick the selection input u per proj_source, loudly."""
        if self.proj_source == "layer_input":
            if sel_x is None:
                raise ValueError(
                    "SSDMimoSSM(proj_source='layer_input') needs sel_x (the "
                    "block input, width d_model) in forward/step; the layer "
                    "passes it — or build the core with proj_source='main_branch'")
            u = sel_x
        else:
            u = x
        if u.shape[-1] != self.d_sel:
            raise ValueError(
                f"selection input width {u.shape[-1]} != d_sel={self.d_sel} "
                f"(proj_source={self.proj_source!r})")
        return u

    def scan(
        self,
        x: torch.Tensor,
        *,
        gate_z: torch.Tensor | None = None,
        sel_x: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
        state: State | dict | None = None,
        return_state: bool = False,
        out_norm=None,
    ) -> torch.Tensor | tuple[torch.Tensor, State]:
        """Run the SSD-MIMO recurrence over a full sequence.

        Parameters
        ----------
        x : (B, L, d_inner)
            Pre-projected input from the enclosing block.
        gate_z : (B, L, d_out) or None
            Output-coordinate SwiGLU gate from the mixer, applied after sibling reduction and skip
            (SISO keeps its fused kernel epilogue).
        sel_x : (B, L, d_model) or None
            Block input the token maps read from under proj_source="layer_input"; ignored under "main_branch".
        tok_raw : (B, L, tok_width) or None
            Token-map projection already applied by the caller (via `tok_proj_weight()`); None projects here.
        state : State or None
            Carry ``{"hidden_state", "rot_ang", "write_prev", "bc_conv_state"}`` (the keys this config
            has).  The fast routes thread every carry; a carry a route cannot take is refused by name and
            falls one step down (cuda -> chunked torch -> step), never dropped; see :meth:`last_route`.
        out_norm : (weight, eps) or None
            The enclosing module's output RMSNorm, fused into the CUDA gate epilogue.  Ask
            `out_norm_refusal` first and skip the module's own norm exactly when it returns None.
        return_state : bool
            Also return a complete State; its carries keep their graph only when gradients are enabled.

        Returns
        -------
        y : (B, L, d_out)
            Output, where d_out = H * G * O.
        state_final : State, only when return_state=True
            Complete carry; state_final.hidden_state retains its autograd graph.
        """
        # MIMO gates the final O features: keep the scan ungated, gate after mixing/skip
        output_gate = gate_z if self.R > 1 or self.O > 1 else None
        if gate_z is not None and gate_z.shape != (*x.shape[:-1], self.d_out):
            raise ValueError(f"gate_z must have shape {(*x.shape[:-1], self.d_out)}")
        if output_gate is not None:
            if out_norm is not None:
                raise NotImplementedError("MIMO output norm must follow the output gate in the layer")
            gate_z = None
        h_init = None
        if state is not None:
            state = ScanState(self).resolve_state(
                state, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
            h_init = state.hidden_state if isinstance(state, State) else state["hidden_state"]
        # dtype of the returned hidden carry: the supplied State's, else the IO dtype
        self._carry_dtype = h_init.dtype if h_init is not None else None
        extra_carry = ({k: v for k, v in state.items() if k != "hidden_state"}
                      if state is not None else {})

        backend = self.backend
        if backend == "auto":
            backend = "cuda" if x.is_cuda else "torch"
        if self.m3_kernel == "upstream" and not x.is_cuda:
            raise ValueError("kernel_options m3_kernel='upstream' batched forward requires CUDA")
        u = self._select(x, sel_x)

        if out_norm is not None and (backend != "cuda" or extra_carry or return_state):
            raise NotImplementedError(
                "out_norm is fused into the CUDA gate epilogue; this "
                f"call resolved to execution backend {backend!r}, extra_carry="
                f"{bool(extra_carry)}.  Ask `out_norm_refusal` first and "
                "apply the norm module-side when it names a clause.")
        # returned carries keep their graph only for a differentiable State request
        self._keep_state_graph = bool(return_state
                                      and torch.is_grad_enabled())
        # carries are threaded by the shared kernel-input build; what remains is a per-route refusal
        # (upstream M3 adapter), which falls back one step: cuda -> chunked torch -> step
        chunk_why = self._carry_refusal_chunked(extra_carry)
        cuda_why = (self._carry_refusal_cuda(extra_carry)
                    if chunk_why is None else None)
        if backend == "cuda" and cuda_why is None and chunk_why is None:
            # refusals decided BEFORE any kernel work, from the call's static properties; cached per signature
            cuda_why = self._cuda_route_refusal(x, gate_z=gate_z, h_init=h_init, out_norm=out_norm,
                                                carry=extra_carry)
        if backend == "cuda" and cuda_why is not None:
            demote_kernel_route(self, x.is_cuda, cuda_why)       # raises under backend='cuda'
            backend = "torch"
        # only the CUDA gate epilogue applies `out_norm`; every other route returns un-normalised y and
        # the norm is applied after it, so a demotion never drops the caller's norm
        norm_pending = out_norm is not None
        if x.shape[1] == 0 or (extra_carry and chunk_why is not None):
            # empty chunk, or a carry no fast route threads: the O(L) step loop (named in `last_route()`)
            if x.shape[1] and backend == "cuda":
                demote_kernel_route(self, x.is_cuda, chunk_why)  # raises under backend='cuda'
            self._route_note = chunk_why
            y = self._forward_sequential(x, u, h_init, gate_z,
                                         init_extra=extra_carry,
                                         tok_raw=tok_raw)
            full_state = self._seq_final_state
        elif backend == "cuda":
            self._route_note = None
            full_state = None
            fell_back = False
            try:
                y = self._forward_cuda(x, u, h_init, gate_z, tok_raw,
                                       out_norm=out_norm, carry=extra_carry)
            except NotImplementedError as why:
                # a config envelope the kernel refuses by name (e.g. 4-dim per-head phase with d_skip and no
                # gate_z at O == 1: the fused D-skip rides the gate epilogue, so without a gate its adjoint
                # has no y_pre tile).  SSDMimoSSM always passes gate_z, so only bare-core callers reach this.
                # backend='cuda' raises; backend='auto' falls one step, to the chunked torch route, warning
                # once and recording the reason for `last_route()`.
                # last-resort net: the pre-check (`_cuda_route_refusal`) names every refusal known today; a
                # new one is recorded here so the signature is not retried on later calls
                self._remember_cuda_refusal(x, gate_z, h_init, out_norm, extra_carry, f"cuda route envelope: {why}")
                demote_kernel_route(self, x.is_cuda, f"cuda route envelope: {why}")
                self._route_note = f"cuda route envelope: {why}"
                y, full_state = self._forward_torch(
                    x, u, h_init, gate_z, tok_raw, carry=extra_carry,
                    return_state=return_state)
                fell_back = True
            if not fell_back:
                norm_pending = False             # the fused epilogue normalised it
            if not fell_back and return_state:
                full_state = self._collect_fast_state(x, u, tok_raw,
                                                      carry=extra_carry)
        elif backend == "torch":
            self._route_note = cuda_why
            y, full_state = self._forward_torch(
                x, u, h_init, gate_z, tok_raw, carry=extra_carry,
                return_state=return_state)
        elif backend == "sequential":
            self._route_note = None
            y = self._forward_sequential(x, u, h_init, gate_z,
                                         init_extra=extra_carry,
                                         tok_raw=tok_raw)
            full_state = self._seq_final_state
        else:
            raise ValueError(f"SSDMimoSSM: unknown execution backend {backend!r}")

        if norm_pending:
            w_n, eps_n = out_norm
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + eps_n) * w_n.float()).to(y.dtype)
        if output_gate is not None:
            y = y * F.silu(output_gate)
        if not return_state:
            return y
        full_state = ScanState(self)._conform_carries(full_state, state)
        self._final_state = full_state
        return y, full_state

    #: carries threaded by the shared fast-route input build: `rot_ang` offsets the cumulated phase,
    #: `write_prev` folds into `x0`, `bc_conv_state` prefixes the module-side B/C conv.
    FAST_CARRY_KEYS = ("rot_ang", "write_prev", "bc_conv_state")

    def _carry_refusal_chunked(self, carry) -> str | None:
        """Why no fast (chunked/kernel) route can thread this carry, by name; None = all keys threaded."""
        if not carry:
            return None
        unknown = sorted(set(carry) - set(self.FAST_CARRY_KEYS))
        if unknown:
            return (f"state carries {unknown} are not threaded by the "
                    f"chunked/kernel input build (threaded: "
                    f"{list(self.FAST_CARRY_KEYS)})")
        if "rot_ang" in carry and self.rot_mode != "m3":
            return ("state carry 'rot_ang' on a layer without the m3 rotation law "
                    f"(rotation law {self.rot_mode!r}): the scan has no phase to offset")
        if "write_prev" in carry and not self.use_trap:
            return ("state carry 'write_prev' with write_rule='euler': the "
                    "scan's one-tap write has no trapezoid lag to fold")
        if "bc_conv_state" in carry and self.bc_conv1d is None:
            return ("state carry 'bc_conv_state' with bc_conv=False: "
                    "there is no B/C convolution to prefix")
        return None

    def _carry_refusal_cuda(self, carry) -> str | None:
        """Why the CUDA route specifically cannot thread this carry (costs one step down to chunked torch)."""
        if not carry:
            return None
        if self.m3_kernel == "upstream":
            return ("kernel_options m3_kernel='upstream': the upstream M3 adapter takes "
                    "no initial state at all, so state carries "
                    f"{sorted(carry)} cannot ride it")
        return None

    # ── CUDA refusals decided before any kernel work ────────────────────────────────────────────────────
    # The kernels refuse some calls by name (NotImplementedError inside `ssd_mimo_scan`).  Every refusal
    # known today is a function of static properties, so it is named here first, per call signature, and a
    # refused signature goes straight to the torch route (no CUDA attempt, no exception) on later calls.
    #   config only (also raised at construction / set_execution under backend='cuda'):
    #     * in-kernel rotation (rot_mode='m3') needs the per-head phase kernel: d_state a power of two <= 128
    #   with the environment (rotation site, per call signature):
    #     * SSD_MIMO_ROT_WHERE=host: a per-head phase has no host form
    #   call properties:
    #     * upstream M3 kernel: h_init, a fused out_norm or any state carry (the adapter has no initial state)
    #     * fused D-skip (in-scan) needs O == 1 and a gate_z (its adjoint reads the gate epilogue's y_pre tile)
    #     * fused out_norm: the clauses of `out_norm_refusal` (gate, O == 1, no G pad, token-major y, ...)
    #   carries: `_carry_refusal_cuda` (upstream adapter) and `_carry_refusal_chunked` (no fast route).
    # The caught NotImplementedError in `scan` stays as the net for what is device-dependent (N-tile plans).

    def _cuda_config_refusal(self, *, env: bool = False) -> str | None:
        """Why the CUDA kernels refuse this CONFIGURATION whatever the call, or None.  ``env=True`` also reads the
        environment levers (at call time); ``env=False`` is the construction / `set_execution` view."""
        if self.m3_kernel == "upstream":
            return None
        N = self.N
        in_kernel_rot = (self.rot_mode == "m3" and self.R in RANKS and self.O in RANKS
                         and N >= 16 and N % 16 == 0 and self.G_kernel % 16 == 0
                         and (not env or _os.environ.get("SSD_MIMO_ROT_INKERNEL", "1") == "1"))
        if in_kernel_rot:
            if N & (N - 1) or N > 128:
                return (f"the in-kernel rotation (per-head phase) needs d_state a power of two in [16, 128], "
                        f"got d_state={N}")
            if env:
                from ..kernels.ssd_mimo.autograd import _rot_where
                if _rot_where(None) != "kernel":
                    return ("SSD_MIMO_ROT_WHERE='host': a per-head phase has no host form, so the "
                            "rotation law needs the in-kernel phase")
        return None

    def _cuda_call_refusal(self, x, *, gate_z, h_init, out_norm, carry) -> str | None:
        """Why the CUDA kernels refuse THIS call (config, ranks, gate, carries, dtype), or None."""
        if self.m3_kernel == "upstream":
            if h_init is not None or out_norm is not None or carry:
                return ("upstream M3 batched route does not support h_init or fused out_norm; "
                        "use the state/step API for continuation")
            return None
        why = self._cuda_config_refusal(env=True)
        if why is not None:
            return why
        if self.skip and self._dskip_route(x, gate_z) == "in_scan":
            if self.O != 1:
                return f"the fused D-skip needs O == 1 (its weights carry no output axis), got O={self.O}"
            if gate_z is None:
                return ("the fused D-skip rides the gate epilogue (its adjoint reads the same y_pre tile): "
                        "this call has no gate_z")
        if out_norm is not None and x.is_cuda:
            why = self.out_norm_refusal(x, gate_z=gate_z)
            if why is not None:
                return f"fused out_norm: {why}"
        return None

    def _cuda_signature(self, x, gate_z, h_init, out_norm, carry) -> tuple:
        get = _os.environ.get
        env = (get("SSD_MIMO_ROT_WHERE"), get("SSD_MIMO_ROT_INKERNEL"), get("SSD_MIMO_Y_LAYOUT"),
               get("SSD_MIMO_DSKIP_EPILOGUE"), get("SSD_MIMO_OUT_NORM_FUSED"))
        return (getattr(self, "_execution_version", 0), x.dtype, bool(x.is_cuda), gate_z is not None,
                h_init is not None, out_norm is not None, tuple(sorted(carry or ())), env)

    def _cuda_route_refusal(self, x, *, gate_z, h_init, out_norm, carry) -> str | None:
        """`_cuda_call_refusal`, decided once per call signature (`_cuda_signature`) and remembered on the layer."""
        cache = self.__dict__.setdefault("_cuda_refusal_cache", {})
        sig = self._cuda_signature(x, gate_z, h_init, out_norm, carry)
        if sig not in cache:
            cache[sig] = self._cuda_call_refusal(x, gate_z=gate_z, h_init=h_init, out_norm=out_norm, carry=carry)
        return cache[sig]

    def _remember_cuda_refusal(self, x, gate_z, h_init, out_norm, carry, reason: str) -> None:
        self.__dict__.setdefault("_cuda_refusal_cache", {})[
            self._cuda_signature(x, gate_z, h_init, out_norm, carry)] = reason

    def last_route(self) -> str | None:
        """The refusal (if any) that moved the last forward off its requested route; None if none."""
        return getattr(self, "_route_note", None)

    def state_grad_refusal(self, key: str) -> str | None:
        """Why a gradient w.r.t. state carry `key` is unavailable, by name (a carry this config lacks).

        The kernel takes one state argument `x0` and returns its adjoint; every other carry enters through
        a host fold, so its adjoint is an ordinary torch graph.
        """
        have = set(ScanState(self)._state_schema().keys())
        if key not in have:
            return (f"state carry {key!r} is not carried by this config "
                    f"(carried: {sorted(have)}), so it has no gradient; "
                    "asking for one is a configuration error, not a zero")
        return None

    def _collect_fast_state(self, x: torch.Tensor, u: torch.Tensor,
                            tok_raw: torch.Tensor | None = None,
                            carry: dict | None = None) -> State:
        """The non-``h`` carries of :meth:`step`'s state for the fast forward, from quantities the fold
        already computed (no scan re-run).  `carry` is the incoming carry: a chunk shorter than the B/C
        conv window takes the missing tail from its `bc_conv_state`."""
        carry = carry or {}
        st = State(hidden_state=self.h_final)
        if self.rot_mode == "m3":
            st["rot_ang"] = self._rot_ang_final
        if self.use_trap:
            key = "write_prev"
            st[key] = self._trap_w_final(x, u, tok_raw, carry=carry)
        if self.bc_conv1d is not None:
            # step() convention (B, channels, d_conv): the last d_conv raw inputs ending at the current
            # token; raw B/C are a per-token function of u, so the tail alone reproduces it
            kc = self.bc_conv1d.d_conv
            if tok_raw is None:
                tail = u[:, -kc:]
                parts_t = self._proj_split(tail)
            else:
                # off the caller's fused GEMM: bit-for-bit the B/C the forward convolved
                parts_t = self.split_tok_raw(tok_raw[:, -kc:])
            bc_tail = torch.cat([parts_t["B"], parts_t["C"]], dim=-1)
            if bc_tail.shape[1] < kc:
                bc_tail = torch.cat(
                    [self._bc_window_pad(carry.get("bc_conv_state"), kc,
                                         kc - bc_tail.shape[1], bc_tail),
                     bc_tail], dim=1)
            st["bc_conv_state"] = self._state_leaf(bc_tail.transpose(1, 2))
        return st

    @staticmethod
    def _bc_window_pad(prev, kc: int, need: int, like: torch.Tensor):
        """The `need` most recent raw B/C rows before this chunk, from `step`'s (B, C, d_conv) carry
        (`None` pads with zeros, as the conv's left padding does)."""
        if prev is None:
            return like.new_zeros(like.shape[0], need, like.shape[-1])
        return prev[:, :, kc - need:].transpose(1, 2).to(like.dtype)

    def _state_leaf(self, t: torch.Tensor) -> torch.Tensor:
        """A carry handed back by the fast route: differentiable only for a State request under grad, else detached."""
        return t if getattr(self, "_keep_state_graph", False) else t.detach()

    def _trap_w_final(self, x: torch.Tensor, u: torch.Tensor,
                   tok_raw: torch.Tensor | None = None,
                   carry: dict | None = None):
        """`step`'s unscaled previous-write carry at the end of a fast chunk.

        A pure function of the last token's maps and the cumulative phase (no dependence on ``h``), so it
        is rebuilt by replaying that token through the step code from a zero ``h``; the replay needs the
        phase one token back (``_rot_ang_prev``) and the raw B/C of the last ``d_conv + 1`` tokens.
        """
        carry = carry or {}
        L = u.shape[1]
        # the replay carries are the contracted state dtype (parameter dtype, never below fp32), as in
        # `step`: a 16-bit IO dtype must not make them 16-bit; only its write_prev is read
        cdt = self._default_state_dtype()
        prev = State(hidden_state=torch.zeros_like(self.h_final, dtype=cdt))
        if self.rot_mode == "m3":
            prev["rot_ang"] = self._rot_ang_prev
        key = "write_prev"
        prev[key] = torch.zeros_like(self.h_final, dtype=cdt)
        if self.bc_conv1d is not None:
            kc = self.bc_conv1d.d_conv
            if tok_raw is None:
                parts_t = self._proj_split(u[:, -kc - 1:-1]) if L > 1 else None
            else:
                parts_t = (self.split_tok_raw(tok_raw[:, -kc - 1:-1])
                           if L > 1 else None)
            if parts_t is None:
                bc_tail = None
            else:
                bc_tail = torch.cat([parts_t["B"], parts_t["C"]], dim=-1)
            width = (self.N * self.R) + (self.N * self.O)
            if bc_tail is None or bc_tail.shape[1] < kc:
                have = 0 if bc_tail is None else bc_tail.shape[1]
                like = (bc_tail if bc_tail is not None
                        else u.new_zeros(u.shape[0], 0, width))
                # a window running off the front of this chunk continues into the incoming carry
                pad = self._bc_window_pad(carry.get("bc_conv_state"), kc,
                                          kc - have, like)
                bc_tail = (pad if bc_tail is None
                           else torch.cat([pad, bc_tail], dim=1))
            prev["bc_conv_state"] = self._state_leaf(bc_tail.transpose(1, 2))
        saved_h, saved_state = self.h_final, getattr(self, "_final_state", None)
        # the replay keeps its graph only for a differentiable State request
        keep = getattr(self, "_keep_state_graph", False)
        # `_step_legacy`, not `step`: the CUDA route keeps its established numerics
        with torch.enable_grad() if keep else torch.no_grad():
            _, st_last = self._step_legacy(
                x[:, -1], prev,
                sel_x=u[:, -1] if self.proj_source == "layer_input" else None,
                tok_raw=tok_raw[:, -1] if tok_raw is not None else None)
        self.h_final, self._final_state = saved_h, saved_state
        return self._state_leaf(st_last[key])

    def _forward_torch(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        h_init: torch.Tensor | None = None,
        gate_z: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
        carry: dict | None = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, State | None]:
        """backend='torch' (and the CUDA route's fallback): chunked SSD dual `ssd_mimo_torch.ssd_chunked`.

        Returns ``(y, State or None)``; the State is complete and keeps its graph under autograd.  Sets
        ``self.h_final`` (differentiable).
        """
        if _torch_impl() == "legacy":
            y = self._forward_torch_legacy(x, u, h_init, gate_z, tok_raw,
                                           carry=carry)
            st = (self._collect_fast_state(x, u, tok_raw, carry=carry)
                  if return_state else None)
            return y, st
        from .ssd_mimo_torch import ssd_chunked, ssd_scan_seq, torch_route
        if torch_route(x.shape[1]) == "scan":
            y, st, hidden = ssd_scan_seq(
                self, x, u, h0=h_init, gate_z=gate_z, tok_raw=tok_raw,
                carry=carry, want_state=return_state)
        else:
            chunk = (self.chunk_size if isinstance(self.chunk_size, int)
                     and not isinstance(self.chunk_size, bool) else None)
            y, st, hidden = ssd_chunked(
                self, x, u, h0=h_init, gate_z=gate_z, tok_raw=tok_raw,
                carry=carry, want_state=return_state, chunk_size=chunk)
        self.h_final = hidden
        return y, st

    def _forward_torch_legacy(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        h_init: torch.Tensor | None = None,
        gate_z: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
        carry: dict | None = None,
    ) -> torch.Tensor:
        """Older backend='torch' route (the ``_TORCH_IMPL = "legacy"`` test hook): the shared input build plus
        `unified_ssd_chunked` (chunk-local cumsums, quadratic intra-chunk form, parallel inter-chunk scan).
        CPU-capable, differentiable."""

        from ..kernels.ssd_mimo.reference import unified_ssd_chunked

        kw = self._prepare_kernel_inputs(x, u, h_init, gate_z, tok_raw,
                                         carry=carry)
        y, s_final = unified_ssd_chunked(
            kw["x_k"], kw["delta_k"], kw["B_k"], kw["C_k"], kw["a_k"],
            beta=kw["beta_k"], gamma=kw["gamma_k"],
            theta=kw["theta_k"], theta_h=kw["thh_k"],
            rot_bias=kw["rb_k"], bias_pre_rot=kw["bias_pre_rot"],
            delta_w=kw["delta_wk"],
            x0=kw["x0"], gate_z=kw["gz_k"], d_skip=kw["dsk_k"],
            bc_per_head=True, pair_layout=self.pair_layout)
        y = self._apply_m3_correction(y, kw)
        return self._finish_forward(x, u, y, s_final, kw.get("rmix_k"), gate_z=gate_z)

    def _forward_sequential(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        h_init: torch.Tensor | None = None,
        gate_z: torch.Tensor | None = None,
        init_extra: dict | None = None,
        tok_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """backend='sequential': token-by-token recurrence via step(), pure torch (O(L) python loop), exact
        for every config.  For the fp64 oracle cast the module and inputs to float64.

        `init_extra`: carries beyond `h` (rot_ang / write_prev / bc_conv_state / pos) to seed the state;
        the final state is stashed on `self._seq_final_state`.  `tok_raw` (B, L, tok_width): the caller's
        pre-projected token slab, sliced per token into `step`.
        """
        B, L, _ = x.shape
        state = ScanState(self).init_state(
            B, device=x.device, dtype=x.dtype, h_init=h_init, overrides=init_extra)
        ys = []
        for t in range(L):
            y_t, state = self.scan_step(
                x[:, t], state,
                gate_z=gate_z[:, t] if gate_z is not None else None,
                sel_x=u[:, t] if self.proj_source == "layer_input" else None,
                tok_raw=tok_raw[:, t] if tok_raw is not None else None)
            ys.append(y_t)
        self._seq_final_state = state
        self.h_final = state["hidden_state"]
        return torch.stack(ys, dim=1) if ys else x.new_empty(B, 0, self.d_out)

    def _prepare_kernel_inputs(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        h_init: torch.Tensor | None = None,
        gate_z: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
        in_kernel_trap: bool = False,
        in_kernel_phase: bool = False,
        m3_host_fold: bool = False,
        rot_host_fold: bool | None = None,
        carry: dict | None = None,
    ) -> dict:
        """Build the kernel inputs from x, shared by backend='cuda' and backend='torch'.

        Subsystems ride the kernel head axis r-blocked: decay unit (h, r), kernel heads (r, h),
        H_k = R*H, G_SIZE = G.

        `rot_host_fold`: where the per-(kernel head, token) phase is applied.  None = the law's default
        (kernel `theta` slot for m3); True = fold on the host (`SSD_MIMO_ROT_INKERNEL=0`, and the
        `_forward_cuda` fallback outside the kernel envelope); False = force the theta slot.
        `m3_host_fold=True` is an older spelling of `rot_host_fold=True`: biases join the raw maps and the
        pair rotation is folded on the host, so the scan gets beta = gamma = theta = None; it exists
        because the kernel's per-head phase is wired at R = O = 1 only.  `backend="torch"` never uses it
        (its reference folds the phase itself, keeping it an independent oracle).

        `carry`: continuation carries beyond ``hidden_state``; none is a new scan argument.  ``rot_ang``
        is a base offset on the cumulated phase, ``write_prev`` folds into `x0` (g_{-1} = S_{-1} + dt_0
        (1-lam_0) W_{-1}, the same substitution as `delta_w` / `trap_nxt`), ``bc_conv_state`` prefixes the
        module-side B/C conv.  Each fold is an ordinary torch expression, so adjoints are torch graphs.
        """
        B, L, D = x.shape
        H, P, G, R, N = self.H, self.P, self.G, self.R, self.N
        carry = carry or {}
        host_fold = (bool(m3_host_fold) if rot_host_fold is None
                     else bool(rot_host_fold) or bool(m3_host_fold))
        # carried B/C conv window: the d_conv-1 raw inputs before the first output of this chunk
        bc_prefix = None
        if carry.get("bc_conv_state") is not None and self.bc_conv1d is not None:
            kc = self.bc_conv1d.d_conv
            if kc > 1:
                bc_prefix = carry["bc_conv_state"][:, :, 1:].transpose(1, 2)

        # one fused GEMM for every token map (see `_proj_split`); channel-major when the drive is
        # channel-major and the fused tail reads the slab, so the two drive gradients share a layout
        parts = (self._proj_split(
                     u, channel_major=(self._tail_owns_all
                                       and u.dim() == 3 and u.stride(-1) != 1
                                       and self._tail_will_fuse(u)))
                 if tok_raw is None else self.split_tok_raw(tok_raw))
        # kernel heads = (r, h), H_k = R*H
        Hk = R * H

        xi = x.view(B, L, H, G, R).permute(0, 2, 3, 4, 1)  # [B,H,G,R',L]

        def _dk(v):
            # v : (B, L, H, R) -> (B, Hk, L), r-major kernel heads
            return (v.permute(0, 3, 2, 1).reshape(B, Hk, L)
                    .to(torch.float64 if v.dtype == torch.float64 else torch.float32).contiguous())

        # fused tail returns dt_k / dec_k already in `_dk`'s (B, Hk, L) fp32 form; None = outside its
        # envelope (or SSD_MIMO_TAILS=0), then the eager chain runs
        fused = self._fused_token_tails(parts, L, bc_prefix=bc_prefix)
        if fused is not None:
            Bt, Ct, gB, dt_k, dec_k = fused
            gC = None                       # folded into Ct by the kernel
            delta_k = dec_k if self.a_dyn else dt_k
            delta_wk = dt_k if self.a_dyn else None
            dt = dt_dec = None              # only `_dk`'d forms are used
        else:
            # wide=False: the kernel wants one decay value per (head, r), not the G-broadcast
            Bt, Ct, gB, gC, dt, dt_dec = self._token_maps(
                u, parts=parts, wide=False, bc_prefix=bc_prefix)
            dt_k = _dk(dt)
            delta_k = dt_k if dt_dec is None else _dk(dt_dec)
            delta_wk = None if dt_dec is None else dt_k

        endpoint = None
        trap_nxt = None
        trap_lam_k = None
        lam_k = None
        if self.use_trap:
            dt_w = dt_k                                          # [B, Hk, L]
            # lambda_t = sigmoid(trap_proj(x_t)) per (token, head), r-major expanded to kernel heads.
            # fp32 sigmoid for 16-bit slabs: a bf16 sigmoid rounds lambda near 1 and the adjoint
            # lambda (1 - lambda) loses (1 - lambda) entirely for lambda > 0.99.  An fp64 slab stays fp64.
            _tr = parts["trap"]
            lam = torch.sigmoid(_tr.float() if _tr.dtype in (torch.bfloat16, torch.float16) else _tr)      # [B, L, H]
            lam_k = (lam.permute(0, 2, 1).unsqueeze(1)
                     .expand(B, R, H, L).reshape(B, Hk, L)
                     .to(dt_w.dtype))
            if in_kernel_trap:
                # the whole two-tap trapezoid (collapsed write scale and rank-1 output correction) runs in
                # the scan kernel; the host hands over `lam` and the raw dt
                trap_lam_k = lam_k
                delta_wk = dt_w
            else:
                # two-tap trapezoid: the recurrence also writes the previous token's outer product,
                #   S_t = e^{ADT} S_{t-1}
                #         + e^{ADT} dt_t (1-lam_t) v_{t-1} k_{t-1}^T
                #         + lam_t dt_t v_t k_t^T
                # Set c_t = dt_{t+1}(1-lam_{t+1}), c_{L-1}=0, and
                # g_t = S_t + c_t v_t k_t^T. This collapses the
                # recurrence onto the kernel's ONE-tap write with
                #   w_t  = lam_t dt_t + dt_{t+1} (1 - lam_{t+1})
                # and leaves a LOCAL rank-1 output correction
                #   y_t -= dt_{t+1}(1-lam_{t+1}) (k_t . q_t) v_t
                # (the rotation cancels in k_t . q_t: same t).  The final
                # state is unchanged by the substitution.
                if x.is_cuda and x.dtype != torch.float64:
                    from ..kernels.ssd_mimo.endpoint import endpoint_weights
                    delta_wk, trap_nxt = endpoint_weights(dt_w, lam_k)
                else:
                    trap_nxt = F.pad((dt_w * (1 - lam_k))[..., 1:], (0, 1))
                    delta_wk = lam_k * dt_w + trap_nxt

        # x/B/C keep the IO dtype (kernels upcast tiles internally); delta/a_log/state stay fp32
        x_k = (xi.unsqueeze(1).expand(B, R, H, G, R, L)
               .reshape(B, Hk, G, R, L))          # [B, Hk, G, R', L]

        # grouped write map, Gb = R
        if gB is None:
            B_k = Bt.unsqueeze(2)                              # [B,L,1,N,R']
        else:
            B_k = (gB.permute(0, 1, 3, 2).unsqueeze(-1)
                   * Bt.unsqueeze(2))                          # [B,L,R,N,R']

        # read map, single group (Gc = 1): it does not vary with the write rank r
        if self.O == 1:
            gated_c = Ct if gC is None else Ct * gC            # [B,L,N,1]
            C_k = gated_c.permute(0, 1, 3, 2)                  # [B,L,1,N]
        else:
            gated_c = Ct if gC is None else gC * Ct            # [B,L,N,O]
            C_k = gated_c[:, :, None]                          # [B,L,1,N,O]

        # per-kernel-head scalar a_log, r-blocked (a_dyn form: a_log = 0, the decay lives in dt_dec)
        if self.A_log is None:
            a_k = _ref_f(torch.zeros(R * H, device=x.device, dtype=x.dtype))
        else:
            a_k = _ref_f(
                self.A_log.permute(1, 0).reshape(R * H))

        def _to_kernel_state(v):
            # (B, H, P=G*R, N) module layout -> [B, Hk, G, N] r-major
            return _ref_f(v.view(B, H, G, R, N)
                          .permute(0, 3, 1, 2, 4).reshape(B, R * H, G, N))

        x0 = None
        if h_init is not None:
            x0 = _to_kernel_state(h_init)                     # [B,Hk,G,N]
        # the trapezoid lag folds into the initial state: the collapsed one-tap scan runs on
        # g_t = S_t + dt_{t+1}(1-lam_{t+1}) W_t, so g_{-1} = S_{-1} + dt_0(1-lam_0) W_{-1}, with W_{-1} the
        # carried unscaled write (matches `step`'s two-tap law; the adjoint arrives as part of dX0)
        w_prev = carry.get("write_prev")
        if w_prev is not None:
            if lam_k is None:
                raise ValueError(
                    "state carry 'write_prev' with write_rule='euler': the "
                    "scan's one-tap write has no trapezoid lag to fold")
            c0 = (dt_k[:, :, 0] * (1.0 - lam_k[:, :, 0]))     # [B,Hk]
            lag = _to_kernel_state(w_prev) * c0[:, :, None, None]
            x0 = lag if x0 is None else x0 + lag

        # per-head biases (beta, gamma), r-major kernel heads
        beta_k = None
        gamma_k = None
        if self.head_bias:
            beta_k = (self.B_bias.permute(1, 0, 2, 3)
                      .reshape(R * H, N, R).to(a_k.dtype).contiguous())
            gamma_k = (self.C_bias.permute(1, 0, 2, 3)
                       .reshape(R * H, N, self.O).to(a_k.dtype).contiguous())

        # SwiGLU gate for kernel fusion
        gz_k = None
        if gate_z is not None:
            if R == 1:
                # R=1: the (H,G,R)->(R,H,G) regroup is the identity; the gate epilogue reads z through
                # full strides, so a slice of the fused in_proj slab is consumed in place
                from ..kernels.ssd_mimo.autograd import (
                    _gate_epilogue_reads_strided)
                gz_k = (gate_z if _gate_epilogue_reads_strided()
                        else gate_z.contiguous())
            else:
                gz_k = (gate_z.view(B, L, H, G, R)
                        .permute(0, 1, 4, 2, 3)
                        .reshape(B, L, R * H * G).contiguous())

        # SISO uses the native fused scan skip. General R/O uses the
        # equivalent scalar skip reduction in _finish_forward.
        dsk_k = None
        if self.skip and self._dskip_route(x, gate_z) == "in_scan":
            dsk_k = self._skip_gains().to(a_k.dtype).reshape(H, G, 1)

        # theta_h (static read offset): in-kernel (C~ = R(theta_h) R(Theta)^T (C + gamma)); host-folded
        # into a per-kernel-head read map only where the rotation itself is host-folded
        thh_k = None
        rb_k = None

        def _expand_heads(v):
            # group axis Gx in {1, R} -> Hk = R*H kernel heads (r-major)
            if v.shape[2] == Hk:
                return v
            if v.shape[2] == 1:
                return v.expand(B, L, Hk, *v.shape[3:])
            return (v.unsqueeze(3).expand(B, L, R, H, *v.shape[3:])
                    .reshape(B, L, Hk, *v.shape[3:]))

        def _rot_fold(v, th):
            # R(th)^T on the N pairs (dim 3), kernel convention: e' = c*e + s*o ; o' = -s*e + c*o
            c, s_ = torch.cos(th), torch.sin(th)
            if v.dim() == 5:
                c, s_ = c[..., None], s_[..., None]
            if self.pair_layout == "half":
                # pair k is (k, k + N/2): contiguous halves, plain concat
                Kh = N // 2
                e = v[:, :, :, :Kh]
                o = v[:, :, :, Kh:]
                return torch.cat([c * e + s_ * o, -s_ * e + c * o], dim=3)
            e = v[:, :, :, 0::2]
            o = v[:, :, :, 1::2]
            # stack at dim 4, then merge (K, 2) -> N to re-interleave even/odd
            out = torch.stack([c * e + s_ * o, -s_ * e + c * o], dim=4)
            return out.reshape(v.shape[:3] + (N,) + v.shape[4:])

        # host-folded theta_h: C~ = R(Theta + nu)^T (C + gamma) = R(Theta)^T (R(nu)^T C + R(nu)^T gamma),
        # so the read map becomes per kernel head (Gc = Hk)
        if self.use_theta_h:
            nu_k = (self._theta_h_pairs().unsqueeze(0).expand(R, H, N // 2)
                    .reshape(Hk, N // 2))                     # [Hk, K]
        if self.use_theta_h and not host_fold:
            # in-kernel at every (R, O)
            thh_k = nu_k.contiguous()
        elif self.use_theta_h:
            C5 = _expand_heads(C_k if C_k.dim() == 5 else C_k.unsqueeze(-1))
            C_k = _rot_fold(_ref_f(C5), nu_k[None, None]).to(x.dtype)
            if self.O == 1:
                C_k = C_k[..., 0]                               # [B,L,Hk,N]
            if gamma_k is not None:
                g5 = gamma_k.unsqueeze(0).unsqueeze(0)          # [1,1,Hk,N,O]
                gamma_k = (_rot_fold(_ref_f(g5), nu_k[None, None])[0, 0]
                           .to(gamma_k.dtype).contiguous())

        theta_k = None
        theta_lo_k = None      # theta_k's bf16 twin (data only)
        theta_incr = False
        m3_pre = None          # (B_pre, C_pre) per kernel head, unrotated
        if self.rot_mode == "m3":
            # Theta_t[h,k] = cumsum_s(tanh(Ang_s[h,k]) * pi * dt[h,s]) on the write clock (Mamba-3's
            # angle), read off the fused token-map GEMM.  Sign: the kernel folds R(theta)^T onto B and C
            # while Mamba-3 applies R(+Theta), so theta = -Theta.  Pairs beyond `n_rope_angles` are
            # zero-padded before the tanh and fold as the identity (cos 0 = 1, sin 0 = 0).
            ka = self.n_rope_angles
            dt_raw_k = dt_k                                    # [B,Hk,L]
            use_phase_kernel = (
                x.is_cuda
                # the kernel accumulates in fp32; fp64 oracle routes keep the torch chain
                and parts["theta"].dtype != torch.float64
                and _os.environ.get("SSD_MIMO_M3_PHASE_KERNEL", "1") == "1")
            omega = None
            if self.learn_theta_freq and self.omega_layout == "head":
                # omega_h / pi, one per head
                omega = torch.tanh(_ref_f(self.theta_freq))       # [H]
            elif self.learn_theta_freq and self.omega_layout == "head_pair":
                # omega_{h,k} / pi, zero on the un-rotated tail
                omega = F.pad(torch.tanh(_ref_f(self.theta_freq)),
                              (0, N // 2 - ka))                  # [H, K]
            elif self.learn_theta_freq:
                # omega_k / pi on the rotated pairs; zero on the un-rotated tail
                omega = F.pad(torch.tanh(_ref_f(self.theta_freq)),
                              (0, N // 2 - ka))                  # omega_k / pi
            raw_theta = parts["theta"]
            # fused phase law: one launch builds the cumulated per-kernel-head phase from (angles, dt,
            # omega) (m3_epilogue.py section 1c); the scan takes the host-cumsum contract
            # (theta_incr=False).  SSD_MIMO_PHASE_FUSED=0 restores the chain below.
            phase_fused = (
                use_phase_kernel
                and _os.environ.get("SSD_MIMO_PHASE_FUSED", "1") == "1")
            if phase_fused:
                from ..kernels.ssd_mimo.m3_epilogue import phase_law
                incr = None
                # bf16 twin: on 16-bit IO routes the law's launch also stores the phase rounded to bf16
                # for the scan's backward.  The scan ignores it where the saved phase must stay fp32, so
                # it is produced only where those exclusions cannot fire.  SSD_MIMO_PHASE_LO=0 disables.
                phase_lo = (
                    torch.bfloat16
                    if (x_k.dtype in (torch.bfloat16, torch.float16)
                        and self.precision != "ieee"
                        and _os.environ.get("SSD_MIMO_PHASE_LO", "1") == "1"
                        # the scan keeps the fp32 phase unless kernel_options theta_bwd_fp32=False
                        and not _kernel_options.value("ssd_mimo", "theta_bwd_fp32",
                                                      self.kernel_options))
                    else None)
                if (self.omega_per_head and _os.environ.get(
                        "SSD_MIMO_OMEGA_HEAD_KERNEL", "1") == "0"):
                    # A/B route: per-head eigenvalue by host broadcast (angle expanded over pairs,
                    # omega_h folded into the clock); same function as the default kernel route
                    ang_b = raw_theta.reshape(B, L, H, 1).expand(B, L, H, ka)
                    dt_b = (dt_raw_k if omega is None else
                            dt_raw_k * omega.repeat(R).to(dt_raw_k.dtype)
                            [None, :, None])
                    theta_k = phase_law(ang_b, dt_b, None, H, N // 2,
                                        sign=-1.0, lo_dtype=phase_lo)
                else:
                    theta_k = phase_law(
                        raw_theta.reshape(B, L,
                                          H if self.theta_per_head else 1,
                                          self.n_theta_feat),
                        dt_raw_k, omega, H, N // 2, sign=-1.0,
                        lo_dtype=phase_lo,
                        ka=ka if self.omega_per_head else 0,
                        omega_layout=self.omega_layout)
                if phase_lo is not None:
                    theta_k, theta_lo_k = theta_k
            else:
                incr = _theta_increments(
                    raw_theta, dt_raw_k, omega,
                    H=H, R=R, K=N // 2, ka=ka, per_head=self.theta_per_head,
                    use_kernel=(use_phase_kernel
                                and self.omega_layout == "pair"),
                    sign=-1.0, omega_per_head=self.omega_per_head)
            # increment form: the kernels take per-token increments plus a per-chunk offset and cumsum
            # in-tile (SSD_MIMO_PHASE_INKERNEL=0 restores the host cumsum)
            theta_incr = (in_kernel_phase and not host_fold
                          and not phase_fused
                          and _os.environ.get(
                              "SSD_MIMO_PHASE_INKERNEL", "1") == "1")
            if not phase_fused:
                theta_k = (incr if theta_incr
                           else _wrap2pi_t(incr.cumsum(dim=1)))
            # carried rotation clock: Theta_t = Theta_{-1} + sum_{s<=t} incr_s, i.e. a base offset on the
            # cumulated phase; the carry is (B, H, R, K), kernel heads are r-major
            rot_carry = carry.get("rot_ang")
            if rot_carry is not None:
                # the carry holds the rotated pairs only; the real tail enters as zeros
                off = F.pad(_ref_f(rot_carry).permute(0, 2, 1, 3)
                            .reshape(B, Hk, ka),
                            (0, N // 2 - ka))                  # [B,Hk,K]
                if theta_incr:
                    # the increment form has no slot for an absolute base phase: take the host cumsum
                    theta_k = incr.cumsum(dim=1)
                    theta_incr = False
                theta_k = _wrap2pi_t(theta_k + off[:, None].to(theta_k.dtype))
                if theta_lo_k is not None:
                    # rebuild the twin from the offset phase so the two agree
                    theta_lo_k = theta_k.to(theta_lo_k.dtype)
            # cumulative phase at the last two tokens in the carry layout: `_rot_ang_final` is the
            # returned carry, `_rot_ang_prev` seeds the replay in `_trap_w_final`
            def _ang_carry(v):
                # v [B, Hk, K] -> carry layout (B, H, R, ka), rotated pairs only
                v = v[..., :ka]
                if R == 1:
                    return self._state_leaf(v.unsqueeze(2)).clone()
                return self._state_leaf(
                    v.reshape(B, R, H, ka).permute(0, 2, 1, 3)).clone()

            # at L == 1 the previous phase is the incoming clock (zeros at a sequence start)
            def _ang_prev_carry():
                if rot_carry is None:
                    return torch.zeros_like(self._rot_ang_final)
                return self._state_leaf(_ref_f(rot_carry)).clone()

            if theta_incr:
                tot = incr.sum(dim=1)                          # [B,Hk,K]
                self._rot_ang_final = _ang_carry(_wrap2pi_t(tot))
                self._rot_ang_prev = (
                    _ang_prev_carry() if L == 1
                    else _ang_carry(_wrap2pi_t(tot - incr[:, -1])))
            else:
                self._rot_ang_final = _ang_carry(theta_k[:, -1])
                self._rot_ang_prev = (
                    _ang_prev_carry() if L == 1
                    else _ang_carry(theta_k[:, -2]))
        # host-fold route: one effective map per kernel head; biases before rotation, gB/gC placement as in step
        bias_pre_rot = self.bias_pre_rot
        if self.rot_mode == "m3" and host_fold:
            B_pre = _ref_f(_expand_heads(B_k))                # [B,L,Hk,N,R]
            C_pre = _ref_f(_expand_heads(C_k))                # [B,L,Hk,N(,O)]
            if beta_k is not None:
                B_pre = B_pre + beta_k.to(B_pre.dtype)[None, None]
            if gamma_k is not None:
                gam = gamma_k if C_pre.dim() == 5 else gamma_k[..., 0]
                C_pre = C_pre + gam.to(C_pre.dtype)[None, None]
            m3_pre = (B_pre, C_pre)
            B_k = _rot_fold(B_pre, theta_k).to(x.dtype)
            C_k = _rot_fold(C_pre, theta_k).to(x.dtype)
            # biases and phase are inside the folded operands: the scan runs the plain law
            beta_k = gamma_k = theta_k = None
            bias_pre_rot = False

        # trapezoid rank-1 output correction (m3): y_t -= dt_{t+1}(1-lam_{t+1}) (k_t . q_t) v_t with k / q
        # the biased maps (the rotation cancels at equal t); expanded as four small contractions so the
        # per-head map is never materialised: (B+bb).(C+cb) = B.C + B@cb + C@bb + bb.cb
        m3_corr = None
        if (trap_nxt is not None and self.rot_mode == "m3"
                and x.is_cuda and x.dtype != torch.float64):
            from ..kernels.ssd_mimo.endpoint import endpoint_local_map
            if m3_pre is None:
                # in-kernel read offset: the endpoint kernel forms R(nu)(C + gamma) itself
                eb, ec, ebias, cbias = B_k, C_k, beta_k, gamma_k
            else:
                eb, ec = m3_pre
                ebias = cbias = None
            matrix = endpoint_local_map(
                eb if eb.dim() == 5 else eb.unsqueeze(-1),
                ec if ec.dim() == 5 else ec.unsqueeze(-1),
                None, ebias, cbias, None, thh_k, Hk)
            endpoint = (matrix, trap_nxt)
        elif (trap_nxt is not None and self.rot_mode == "m3"
                and R == 1 and self.O == 1 and m3_pre is None
                and C_k.shape[2] == 1):
            Bv = B_k[:, :, 0, :, 0] if B_k.dim() == 5 else B_k[:, :, 0]
            Cv = C_k[:, :, 0, :, 0] if C_k.dim() == 5 else C_k[:, :, 0]
            Bv, Cv = _ref_f(Bv), _ref_f(Cv)                   # [B,L,N]
            bb = beta_k[:, :, 0]                              # [Hk,N]
            cb = gamma_k[:, :, 0]                             # [Hk,N]
            if thh_k is not None:
                # read offset in-kernel: d_t = (B+bb_h) . R(nu_h)(C+cb_h),
                # so the read side is per kernel head here (oracle route)
                Cr = _rot_fold(Cv[:, :, None, :, None].expand(B, L, Hk, N, 1),
                               thh_k[None, None])[..., 0]     # [B,L,Hk,N]
                cr = _rot_fold(cb[None, None, :, :, None],
                               thh_k[None, None])[0, 0, :, :, 0]  # [Hk,N]
                dot = (torch.einsum("bln,blhn->blh", Bv, Cr)
                       + Bv @ cr.t() + torch.einsum("hn,blhn->blh", bb, Cr)
                       + (bb * cr).sum(-1))                   # [B,L,Hk]
            else:
                dot = ((Bv * Cv).sum(-1)[:, :, None]          # [B,L,1]
                       + Bv @ cb.t() + Cv @ bb.t()            # [B,L,Hk]
                       + (bb * cb).sum(-1))                   # [Hk]
            m3_corr = (trap_nxt[:, :, None, :]                # [B,Hk,1,L]
                       * dot.permute(0, 2, 1)[:, :, None, :]
                       * _ref_f(x_k[:, :, :, 0, :]))          # [B,Hk,G,L]
        elif trap_nxt is not None and self.rot_mode == "m3":
            # the same correction at general (R, O), kept on the host at R > 1 / O > 1 (the fused two-tap
            # write is the R = O = 1 branch).  Substituting g_t = S_t + dt_{t+1}(1-lam_{t+1}) W_t leaves
            #   y_t -= dt_{t+1}(1-lam_{t+1}) <C~_t, W_t>,  W_t[g,n] = sum_r' B~_t[n,r'] x_t[g,r'],
            # and the rotation cancels (same t, same kernel-head frame), so it is built from the
            # unrotated pre-maps:
            #   M[t,hk,r',o] = sum_n (B+beta)[n,r'] (C+gamma)[n,o]
            #   corr[hk,g,o,t] = trap_nxt[hk,t] sum_r' x[hk,g,r',t] M[..]
            # trap_nxt is zero at the last token, so the final state is unchanged.
            Oo = self.O
            if m3_pre is None:
                # raw maps per kernel head plus biases (CPU / fp64 route; CUDA uses `endpoint_local_map`)
                Bg = _ref_f(_expand_heads(B_k if B_k.dim() == 5
                                          else B_k.unsqueeze(-1)))
                Cg = _ref_f(_expand_heads(C_k if C_k.dim() == 5
                                          else C_k.unsqueeze(-1)))
                if beta_k is not None:
                    Bg = Bg + beta_k.to(Bg.dtype)[None, None]
                if gamma_k is not None:
                    Cg = Cg + gamma_k.to(Cg.dtype)[None, None]
                if thh_k is not None:
                    # read offset in-kernel: the read map is R(nu_h)(C+gamma)
                    Cg = _rot_fold(Cg, thh_k[None, None])
                M = torch.einsum("blknr,blkno->blkro", Bg, Cg)
            else:
                B_pre, C_pre = m3_pre
                C_pre5 = C_pre if C_pre.dim() == 5 else C_pre.unsqueeze(-1)
                M = torch.einsum("blknr,blkno->blkro", B_pre, C_pre5)
            corr = torch.einsum("bkgrl,blkro->bkogl",
                                _ref_f(x_k), M)               # [B,Hk,O,G,L]
            corr = corr * trap_nxt[:, :, None, None, :]
            m3_corr = corr[:, :, 0] if Oo == 1 else corr

        if trap_nxt is not None and self.rot_mode != "m3":
            # general endpoint correction in the scan's folded coordinates: rotation does not cancel
            # through post-rotation biases or distinct write/read offsets, so build the maps in order
            phase = theta_k.cumsum(1) if theta_incr else theta_k
            if x.is_cuda and x.dtype != torch.float64:
                from ..kernels.ssd_mimo.endpoint import endpoint_local_map
                local_map = endpoint_local_map(
                    B_k, C_k if C_k.dim() == 5 else C_k.unsqueeze(-1),
                    phase, beta_k, gamma_k, rb_k, thh_k, Hk)
            else:
                be = _ref_f(_expand_heads(B_k))
                ce = _ref_f(_expand_heads(C_k))
                if ce.dim() == 4:
                    ce = ce.unsqueeze(-1)
                if phase is not None:
                    be, ce = _rot_fold(be, phase), _rot_fold(ce, phase)
                if rb_k is not None:
                    be = _rot_fold(be, rb_k[None, None])
                if beta_k is not None:
                    be = be + beta_k[None, None]
                if gamma_k is not None:
                    ce = ce + gamma_k[None, None]
                if thh_k is not None:
                    ce = _rot_fold(ce, thh_k[None, None])
                local_map = torch.einsum("blknr,blkno->blkro", be, ce)
            if x.is_cuda and x.dtype != torch.float64:
                endpoint = (local_map, trap_nxt)
            else:
                corr = torch.einsum("bkgrl,blkro->bkogl", _ref_f(x_k), local_map)
                corr = corr * trap_nxt[:, :, None, None, :]
                m3_corr = corr[:, :, 0] if self.O == 1 else corr

        return dict(x_k=x_k, delta_k=delta_k, delta_wk=delta_wk,
                    B_k=B_k, C_k=C_k, a_k=a_k, x0=x0,
                    beta_k=beta_k, gamma_k=gamma_k, gz_k=gz_k,
                    dsk_k=dsk_k, trap_lam_k=trap_lam_k,
                    thh_k=thh_k, rb_k=rb_k, theta_k=theta_k,
                    theta_lo_k=theta_lo_k,
                    theta_incr=theta_incr, bias_pre_rot=bias_pre_rot,
                    rmix_k=parts.get("rmix"), m3_corr=m3_corr, endpoint=endpoint)

    def _forward_upstream_m3(self, x, u, gate_z, tok_raw):
        """Upstream Mamba-3 adapter (explicit comparison route, not an optimisation).

        Upstream has v'_j[p] = Psi[j,p] v[p]: pack p=(g,i), set Psi[j,(g,i)] = [j=i], sum i after the
        readout; this uses R times as many value/state channels as the native scan.
        """
        from ..kernels.m3_adapter import siso, mimo
        B, L, _ = x.shape
        H, R, O, G, N = self.H, self.R, self.O, self.G, self.N
        Hk = H * R
        parts = self._proj_split(u) if tok_raw is None else self.split_tok_raw(tok_raw)
        fused = self._fused_token_tails(parts, L)
        if fused is not None:
            bt, ct, gb, dt_k, dec_k = fused
            gc = None
            adt = -dec_k if self.a_dyn else -dt_k * self.A_log.T.reshape(1, Hk, 1).float().exp()
        else:
            # clocks are fp32 inputs to upstream: keep their nonlinear tail in fp32
            clock_parts = {**parts, "dt": parts["dt"].float()}
            if "a" in parts:
                clock_parts["a"] = parts["a"].float()
            bt, ct, gb, gc, dt, dec = self._token_maps(u, parts=clock_parts, wide=False)
            dk = lambda t: t.permute(0, 3, 2, 1).reshape(B, Hk, L).float().contiguous()
            dt_k = dk(dt)
            adt = (-dk(dec) if dec is not None else
                   -dt_k * self.A_log.T.reshape(1, Hk, 1).float().exp())
        trap = parts["trap"].transpose(1, 2)[:, None].expand(B, R, H, L).reshape(B, Hk, L).float().contiguous()
        ka = self.n_rope_angles
        angles = parts["theta"].reshape(B, L, H if self.theta_per_head else 1, ka)
        angles = angles.expand(B, L, H, ka)[:, :, None].expand(B, L, R, H, ka).reshape(B, L, Hk, ka).float()
        # zero-padded raw angles give the partial rotation
        angles = F.pad(angles, (0, N // 2 - ka)).contiguous()
        # native cache convention: negative accumulated phase, r-major
        with torch.no_grad():
            inc = -angles.tanh() * math.pi * dt_k.transpose(1, 2)[..., None]
            total = inc.sum(1)
            carry = lambda a: a[..., :ka].reshape(B, R, H, ka).permute(0, 2, 1, 3).clone()
            self._rot_ang_final = carry(_wrap2pi_t(total))
            self._rot_ang_prev = carry(_wrap2pi_t(total - inc[:, -1]))
        bk = bt[:, :, None] if gb is None else gb.permute(0, 1, 3, 2)[..., None] * bt[:, :, None]
        ck = (ct if gc is None else ct * gc)[:, :, None]
        # upstream needs equal Q/K group counts: replicate over sibling groups
        bk = bk.expand(B, L, R, N, R)
        ck = ck.expand(B, L, R, N, O)
        # biases are [H, R, N, R'] / [H, R, N, O] (bias_gran="head"); kernel head k = r*H + h, r-major
        beta = (self.B_bias.permute(1, 0, 2, 3).reshape(Hk, N, R)
                if self.head_bias else x.new_zeros(Hk, N, R))
        gamma = (self.C_bias.permute(1, 0, 2, 3).reshape(Hk, N, O)
                 if self.head_bias else x.new_zeros(Hk, N, O))
        if R == 1:
            v = x.reshape(B, L, H, G).contiguous()
            z = None if gate_z is None else gate_z.reshape(B, L, H, G)
            result = siso(Q=ck[..., 0], K=bk[..., 0], V=v,
                          ADT=adt, DT=dt_k, Trap=trap,
                          Q_bias=gamma[..., 0].contiguous(), K_bias=beta[..., 0].contiguous(),
                          Angles=angles, D=self.D if self.skip and self.skip_sharing == "head" else None,
                          Z=z if not self.skip or self.skip_sharing == "head" else None, chunk_size=64, return_final_states=True)
            y, _, final, _, _ = result
            if self.skip and self.skip_sharing != "head":
                yt = y.float() + v.float() * self._skip_gains().reshape(H, G).float()
                if z is not None:
                    yt = yt * F.silu(z.float())
                y = yt.to(y.dtype)
            y = y.permute(0, 2, 3, 1)
        else:
            if x.dtype != torch.bfloat16:
                raise ValueError("upstream MIMO currently requires BF16 value inputs (use the layer's BF16 autocast)")
            index = torch.arange(G * R, device=x.device) % R
            psi = (torch.arange(R, device=x.device)[:, None] == index[None]).float()
            psi = psi[None].expand(H, R, G * R).contiguous()
            # upstream backward supports Q/K groups 1 or all heads only: put sibling groups in batch;
            # TileLang rotates split halves, checkpoints use adjacent pairs (hence `order`)
            order = torch.cat((torch.arange(0, N, 2, device=x.device),
                               torch.arange(1, N, 2, device=x.device)))
            bk, ck = bk.index_select(3, order), ck.index_select(3, order)
            beta, gamma = beta[:H].index_select(1, order), gamma[:H].index_select(1, order)
            def maps_to_batch(t):
                return t.permute(0, 2, 1, 4, 3).reshape(B * R, L, R, 1, N).contiguous().to(x.dtype)
            def tokens_to_batch(t):
                return t.reshape(B, L, R, H, -1).permute(0, 2, 1, 3, 4).reshape(B * R, L, H, -1).contiguous()
            result = mimo(Q=maps_to_batch(ck), K=maps_to_batch(bk),
                          V=x.reshape(B, L, H, G * R)[:, None].expand(B, R, L, H, G * R).reshape(B * R, L, H, G * R).contiguous(),
                          ADT=adt.reshape(B * R, H, L),
                          DT=dt_k.reshape(B * R, H, L),
                          Trap=trap.reshape(B * R, H, L).to(x.dtype),
                          Q_bias=gamma.transpose(1, 2).contiguous().float(),
                          K_bias=beta.transpose(1, 2).contiguous().float(),
                          MIMO_V=psi, MIMO_Z=None, MIMO_Out=None,
                          Angles=tokens_to_batch(angles), D=None, Z=None,
                          chunk_size=64 // R, rotary_dim_divisor=2,
                          dtype=x.dtype, return_state=True)
            y, _, final, _, _ = result
            # sum the input-rank contributions before the nonlinear gate
            y = y.reshape(B, R, L, O, H, G, R).float().sum(-1).to(x.dtype)
            y = y.permute(0, 1, 4, 3, 5, 2).reshape(B, Hk, O, G, L).contiguous()
            final = final.reshape(B, Hk, G, R, N).sum(3)
            final = final.reshape(B, Hk, G, 2, N // 2).transpose(-1, -2).reshape(B, Hk, G, N)
            if gate_z is not None:
                z = gate_z.reshape(B, L, H, G, R).permute(0, 4, 2, 3, 1).reshape(B, Hk, G, L)
                y = (y.float() * F.silu(z.float())[:, :, None]).to(x.dtype)
        return self._finish_forward(x, u, y, final, parts.get("rmix"), gate_z=gate_z)

    def _resolve_chunk_size(self, kw: dict) -> tuple[int, str]:
        if getattr(self, "precision", None) == "bf16_ops" and not torch.is_autocast_enabled("cuda"):
            # the opt-in bf16-operand mode runs the bf16 path: resolve chunk and route as a bf16 call would
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._resolve_chunk_size_impl(kw)
        return self._resolve_chunk_size_impl(kw)

    def _resolve_chunk_size_impl(self, kw: dict) -> tuple[int, str]:
        """This call's fused-scan chunk T and how it was chosen.

        `SSD_MIMO_CHUNK` wins; else `self.chunk_size` is an int, "auto" (per-rank rule `_scan_chunk`, every
        call) or "measure" (timed once per strategy cell and cached, see `_measure_chunk_size`; refused
        when `allow_chunk_autotune` is False).  Controls timing trials, not gradient determinism.
        """
        import os as _os
        env = _os.environ.get("SSD_MIMO_CHUNK")
        if env:
            return int(env), "env"
        cs = self.chunk_size
        if isinstance(cs, int):
            return cs, "explicit"
        nb = _needs_bwd(kw)
        if cs == "auto":
            # grad and no-grad alike: the forward-only table would send an inference call to the
            # general route and lose the SISO forward; the SISO schedule is a T = 64 body
            if self.R == 1 and self.O == 1 \
                    and self.rot_mode is not None \
                    and _gluon_siso_default(kw, with_route=False):
                # SISO default route (siso_route.siso_enabled) is a T = 64 schedule; the calibrated
                # structural route (rot_route) may pick a general route instead, at its own chunk
                r = _calibrated_rot_route(kw)
                if r is None or r[0] == "siso":
                    return 64, "auto-gluon90-siso"
                return r[1], "auto-route-" + r[0]
            if (self.R == 1 and self.O == 1 and self.rot_mode is not None
                    and getattr(self, "precision", "tf32") != "ieee"):
                # fp32 IO: the GPU's calibrated fp32 route; None keeps the chunk rule below
                r = _calibrated_rot_route(kw, fp32_only=True)
                if r is not None:
                    return r[1], "auto-route32-" + r[0]
            return _scan_chunk(self.R, self.G_kernel, self.O, nb,
                               real=self.rot_mode is None, **_call_size(kw)), "auto"
        if cs == "measure":
            # the cache keys on the pass, so an inference call measures the forward it runs
            if not self.allow_chunk_autotune:
                return (_scan_chunk(self.R, self.G_kernel, self.O, nb, real=self.rot_mode is None, **_call_size(kw)),
                        "auto-fallback")
            return self._measure_chunk_size(kw), "measure"
        raise ValueError(
            f"chunk_size must be an int, 'auto' or 'measure'; got {cs!r}")

    def _resolve_chunk_size_bwd(self, kw: dict, chunk_T: int,
                                chunk_mode: str):
        """The chunk the backward runs at, or None = the forward's.

        "measure" mode takes the pair found by `_measure_chunk_size`; otherwise `_scan_chunk_bwd`.
        Precedence: `kernel_options`, then `SSD_MIMO_CHUNK_BWD`, then the rule.
        """
        ko = self.kernel_options.get("chunk_size_bwd", "auto")
        if ko != "auto":               # config beats env
            return int(ko)
        import os as _os
        env = _os.environ.get("SSD_MIMO_CHUNK_BWD", "").strip().lower()
        if env and env not in ("", "auto", "0"):
            return int(env)
        if not _needs_bwd(kw):
            return None
        if chunk_mode == "measure":
            tb = getattr(self, "_chunk_bwd_measured", None)
            return None if (tb is None or tb == chunk_T) else tb
        return _scan_chunk_bwd(self.R, self.G_kernel, self.O, chunk_T,
                               real=self.rot_mode is None, **_call_size(kw))

    def _measure_chunk_size(self, kw: dict) -> int:
        """Time each admissible T's fwd(+bwd) once and cache the winner.

        Runs the real `ssd_mimo_scan` on detached leaves mirroring the caller's tensors (shapes, dtypes,
        strides, requires_grad), warms each candidate, times 3 reps with CUDA events (median).  Cached
        in-process and on disk (`smem.py` chunk_cache_*, under a `chunk:` key).
        """
        from ..kernels.ssd_mimo import autograd as _ua
        from ..kernels.ssd_mimo.smem import (
            chunk_admissible, chunk_cache_key, chunk_cache_lookup,
            chunk_cache_store)

        x = kw["x_k"]
        needs_bwd = _needs_bwd(kw)
        key = chunk_cache_key(
            self.R, self.O, self.N, x.dtype, self.precision,
            "fwd_bwd" if needs_bwd else "fwd", tuple(x.shape),
            device=x.device)
        cached = chunk_cache_lookup(key)
        if cached is not None:
            return cached

        leaf_names = ("x_k", "delta_k", "B_k", "C_k", "a_k", "x0",
                     "delta_wk", "theta_k", "thh_k", "gz_k", "dsk_k",
                     "trap_lam_k", "beta_k", "gamma_k", "rb_k")

        def _detached(name):
            t = kw.get(name)
            if not isinstance(t, torch.Tensor):
                return t
            return t.detach().requires_grad_(bool(needs_bwd and t.requires_grad))

        det = {name: _detached(name) for name in leaf_names}
        call_kwargs = dict(
            x0=det["x0"], precision=self.precision, bc_per_head=True,
            delta_w=det["delta_wk"], theta=det["theta_k"],
            theta_lo=kw.get("theta_lo_k"),
            theta_h=det["thh_k"], theta_increment=kw["theta_incr"],
            gate_z=det["gz_k"], d_skip=det["dsk_k"],
            trap_lam=det["trap_lam_k"], beta=det["beta_k"],
            gamma=det["gamma_k"], rot_bias=det["rb_k"],
            bias_pre_rot=kw["bias_pre_rot"], pair_layout=self.pair_layout,
            kernel_options=self.kernel_options,
        )
        leaves = [det[n] for n in
                 ("x_k", "delta_k", "B_k", "C_k", "a_k")]

        def trial(T, T_bwd=None):
            extra = {} if T_bwd in (None, T) else dict(chunk_size_bwd=T_bwd)

            def _run():
                if not needs_bwd:
                    with torch.no_grad():
                        _ua.ssd_mimo_scan(*leaves, chunk_size=T,
                                          chunk_mode="measure", **call_kwargs)
                    return
                for t in det.values():
                    if isinstance(t, torch.Tensor):
                        t.grad = None
                y, s = _ua.ssd_mimo_scan(*leaves, chunk_size=T,
                                         chunk_mode="measure", **extra,
                                         **call_kwargs)
                (y.float().sum() + s.float().sum()).backward()
            return _run

        def time_it(run):
            try:
                run()                      # warm-up: compile + autotune
                torch.cuda.synchronize()
            except Exception:              # noqa: BLE001 -- disqualify
                torch.cuda.empty_cache()
                return None
            samples = []
            for _ in range(3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    run()
                except Exception:          # noqa: BLE001 -- disqualify
                    return None
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            if not samples:
                return None
            samples.sort()
            return samples[len(samples) // 2]

        # Search over the (T_fwd, T_bwd) pair: the diagonal is timed in full, then the off-diagonal only
        # from the best T_fwd (|T| + |T| - 1 trials, not |T|^2).  A pair the backward cannot re-chunk
        # times as its own diagonal and loses.
        timings = {}
        for T in chunk_admissible():
            t = time_it(trial(T))
            if t is not None:
                timings[T] = t
        if not timings:
            self._chunk_bwd_measured = None
            return _scan_chunk(self.R, self.G_kernel, self.O, needs_bwd)
        best = min(timings, key=timings.get)
        best_bwd, pair_timings = None, {}
        if needs_bwd and _chunk_pair_search_on():
            for Tb in chunk_admissible():
                if Tb == best:
                    continue
                t = time_it(trial(best, Tb))
                if t is not None:
                    pair_timings[Tb] = t
            if pair_timings:
                cand = min(pair_timings, key=pair_timings.get)
                if pair_timings[cand] < timings[best]:
                    best_bwd = cand
        self._chunk_bwd_measured = best_bwd
        chunk_cache_store(key, best,
                          dict(diag=timings, pair=pair_timings,
                               T_bwd=best_bwd))
        return best

    def _forward_cuda(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        h_init: torch.Tensor | None = None,
        gate_z: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
        out_norm=None,
        carry: dict | None = None,
    ) -> torch.Tensor:
        """backend='cuda': exact recurrence on the unified_ssd CUDA kernel."""
        if self.m3_kernel == "upstream":
            if h_init is not None or out_norm is not None or carry:
                raise NotImplementedError("upstream M3 batched route does not support h_init or fused out_norm; use the state/step API for continuation")
            return self._forward_upstream_m3(x, u, gate_z, tok_raw)
        from ..kernels.ssd_mimo import autograd as _ua
        from ..kernels.ssd_mimo.common import _ASCALAR_R_MAX

        if self.R > _ASCALAR_R_MAX:
            raise NotImplementedError(
                f"SSDMimoSSM: execution backend 'cuda' supports d_ssm_in <= {_ASCALAR_R_MAX}, "
                f"got {self.R}.")

        # in-kernel phase accumulation when the native layout supports it
        rot_kernel_ok = (
            self.rot_mode == "m3"
            and self.R in RANKS and self.O in RANKS
            and self.N >= 16 and self.N % 16 == 0
            and self.G_kernel % 16 == 0
            and _os.environ.get("SSD_MIMO_ROT_INKERNEL", "1") == "1")
        rot_host_fold = (self.rot_mode == "m3" and not rot_kernel_ok)
        # fuse the two-endpoint rule when the kernel envelope permits (same formula as the host route;
        # carries and final state stay in physical coordinates)
        trap_geom_ok = (
            self.R in RANKS and self.O in RANKS
            and self.N >= 16 and self.N % 16 == 0
            and self.G_kernel % 16 == 0)
        trap_kernel_ok = (rot_kernel_ok if self.rot_mode == "m3"
                          else (self.rot_mode is None and trap_geom_ok))
        in_kernel_trap = (
            self.use_trap and trap_kernel_ok
            # in-scan trapezoid is the default at every R, O (faster and smaller); the endpoint pass
            # is the fallback, or SSD_MIMO_TRAP_INKERNEL_RO=0
            and (self.R == 1 and self.O == 1
                 or _os.environ.get("SSD_MIMO_TRAP_INKERNEL_RO", "1") == "1")
            and _os.environ.get("SSD_MIMO_TRAP_INKERNEL", "1") == "1")
        # the phase cumsum runs in the scan kernels inside the in-kernel rotation envelope
        in_kernel_phase = rot_kernel_ok
        kw = self._prepare_kernel_inputs(x, u, h_init, gate_z, tok_raw,
                                         in_kernel_trap=in_kernel_trap,
                                         in_kernel_phase=in_kernel_phase,
                                         rot_host_fold=rot_host_fold,
                                         carry=carry)
        G, Gp = self.G, self.G_kernel
        Hk = self.R * self.H
        Bsz, L = x.shape[0], x.shape[1]
        if Gp != G:
            # G off the kernels' 16-lane tile: zero-pad the channel axis (exact: zero channels carry zero
            # state/output, their gradients are sliced away); per-(head, R/O) maps need no padding
            kw["x_k"] = F.pad(kw["x_k"], (0, 0, 0, 0, 0, Gp - G))     # [B,Hk,Gp,R,L]
            if kw["x0"] is not None:
                kw["x0"] = F.pad(kw["x0"], (0, 0, 0, Gp - G))          # [B,Hk,Gp,N]
            if kw["gz_k"] is not None:                                 # [B,L,Hk*G]
                kw["gz_k"] = F.pad(kw["gz_k"].view(Bsz, L, Hk, G),
                                   (0, Gp - G)).reshape(Bsz, L, Hk * Gp)
            if kw["dsk_k"] is not None:                                # [Hk,G,R]
                # padded channels carry zero x: a zero weight is the identity there
                kw["dsk_k"] = F.pad(kw["dsk_k"], (0, 0, 0, Gp - G))
        # token-major output where supported avoids an epilogue transpose
        layout = _y_layout()
        want_token = (layout == "token" and self.O == 1 and Gp == G
                      and kw.get("m3_corr") is None and kw.get("endpoint") is None)
        if out_norm is not None and not want_token:
            # the per-call clause (host trapezoid correction after the readout)
            raise NotImplementedError(
                "the fused output norm needs the token-major store and a "
                "readout with nothing after it (d_ssm_out == 1, no group pad, no host "
                "m3 correction); run with SSD_MIMO_OUT_NORM_FUSED=0")
        chunk_T, chunk_mode = self._resolve_chunk_size(kw)
        # decoupled backward chunk: the passes take their own tables; the entry point refuses a pair it
        # cannot re-chunk by name and falls back to `chunk_T` for both
        chunk_T_bwd = None
        if (chunk_mode == "auto" and _chunk_pair_on()
                and _needs_bwd(kw)):
            chunk_T = _scan_chunk(self.R, self.G_kernel, self.O, False)
            chunk_T_bwd = _scan_chunk(self.R, self.G_kernel, self.O, True)
        else:
            # "measure" files its searched pair; otherwise `_scan_chunk_bwd`'s rule; the env var wins
            chunk_T_bwd = self._resolve_chunk_size_bwd(
                kw, chunk_T, chunk_mode)
        if chunk_T_bwd == chunk_T:
            chunk_T_bwd = None
        # declare the structurally real phase pairs: with N/4 rotated pairs (interleaved layout) the
        # zero-padded angle tail is real
        from ..kernels.ssd_mimo import levers as _levers
        _nka = getattr(self, "n_rope_angles", 0)
        _decl = (_nka if (_nka and 4 * _nka == self.N and kw["theta_k"] is not None
                          and self.pair_layout == "interleaved") else None)
        with _levers.declare_active_pairs(_decl):
            y, s_final = _ua.ssd_mimo_scan(
                kw["x_k"], kw["delta_k"], kw["B_k"], kw["C_k"], kw["a_k"],
                chunk_size=chunk_T, chunk_size_bwd=chunk_T_bwd,
                chunk_mode=chunk_mode, kernel_options=self.kernel_options,
                x0=kw["x0"], y_token_major=want_token,
                precision=self.precision, bc_per_head=True,
                delta_w=kw["delta_wk"],
                theta=kw["theta_k"], theta_h=kw["thh_k"],
                theta_lo=kw.get("theta_lo_k"),
                theta_increment=kw["theta_incr"],
                gate_z=kw["gz_k"], d_skip=kw["dsk_k"],
                trap_lam=kw["trap_lam_k"],
                beta=kw["beta_k"], gamma=kw["gamma_k"],
                rot_bias=kw["rb_k"], bias_pre_rot=kw["bias_pre_rot"],
                pair_layout=self.pair_layout,
                # the enclosing module's output RMSNorm, fused into the epilogue; `out_norm_refusal` has
                # checked that `_finish_forward` is a pure view here
                out_norm_w=(None if out_norm is None else out_norm[0]),
                out_norm_eps=(1e-5 if out_norm is None else out_norm[1]))
        if Gp != G:
            if self.O == 1:
                y = y.reshape(Bsz, Hk, Gp, L)[:, :, :G]
            else:
                y = y.reshape(Bsz, Hk, self.O, Gp, L)[:, :, :, :G]
            s_final = s_final.reshape(Bsz, Hk, Gp, -1)[:, :, :G]
        endpoint_kw = kw if self._can_fuse_endpoint(y, kw) else None
        if endpoint_kw is None:
            y = self._apply_m3_correction(y, kw)
        return self._finish_forward(x, u, y, s_final, kw.get("rmix_k"),
                                    gate_z=gate_z, y_contig=(layout == "contig"), endpoint_kw=endpoint_kw)

    def _can_fuse_endpoint(self, y, kw):
        return (kw.get("endpoint") is not None and self.R == self.O == 2
                and y.is_cuda and y.dtype in (torch.float32, torch.bfloat16, torch.float16)
                and _os.environ.get("SSD_ENDPOINT_MIX_FUSED", "1") == "1"
                and _os.environ.get("SSD_MIMO_RMIX_EPILOGUE", "1") == "1")

    def _endpoint_operands(self, y, kw):
        matrix, coefficient = kw["endpoint"]
        channels = y.shape[-2]
        xk = kw["x_k"][:, :, :channels]
        gate = kw.get("gz_k")
        if gate is not None:
            gate = gate.view(gate.shape[0], gate.shape[1], xk.shape[1], -1)
            gate = gate[..., :channels].reshape(gate.shape[0], gate.shape[1], -1).contiguous()
        return xk, matrix, coefficient, gate

    def _apply_m3_correction(self, y, kw):
        """Subtract the transported-endpoint local readout correction from the kernel readout y.

        The SwiGLU gate is applied inside the scan epilogue, so `y` is already gated; the correction
        belongs to the pre-gate readout, hence it is scaled by the same gate ((y_raw - corr) g = y - corr g).
        """
        if kw.get("endpoint") is not None:
            from ..kernels.ssd_mimo.endpoint import endpoint_correct
            return endpoint_correct(y, *self._endpoint_operands(y, kw))
        corr = kw.get("m3_corr")
        if corr is None:
            return y
        corr = corr.reshape(y.shape)
        gz = kw.get("gz_k")
        if gz is not None:
            # gz_k has the padded kernel width while y is sliced back to G: trim the trailing axis; at
            # O > 1 the gate is per subsystem and broadcasts over o
            Bsz, L = gz.shape[0], gz.shape[1]
            g = F.silu(_ref_f(gz)).view(Bsz, L, y.shape[1], -1)
            g = g[..., :y.shape[-2]].permute(0, 2, 3, 1)      # [B,Hk,G,L]
            corr = corr * (g[:, :, None] if y.dim() == 5 else g)
        return y - corr.to(y.dtype)

    def _skip_gains(self):
        """Logical [H,G,R,O] gains; broadcast views preserve parameter tying."""
        if self.skip_sharing == "head":
            return self.D[:, None, None, None].expand(self.H, self.G, self.R, self.O)
        if self.skip_sharing == "channel":
            return self.D[..., None].expand(self.H, self.G, self.R, self.O)
        return self.D

    def _skip_gains_kernel(self, dtype):
        return self._skip_gains().to(dtype).contiguous()

    def _dskip_route(self, ref, gate_z=None):
        """Where the D skip is applied: "in_scan" (SISO fused form), "epilogue" (rmix_reduce), "host" or "off".

        At R = O = 1 the in-scan skip is gate-aware: its adjoint reads the `y_pre` tile the gate epilogue
        writes (`rot_kernel` clause in `kernels/ssd_mimo/autograd.py`), so a bare core called without
        gate_z takes the host fold (same function, `test_dskip_epilogue_matches_the_host_form`) instead
        of making the CUDA route refuse and fall to chunked torch.  `SSDMimoSSM` always passes gate_z.
        """
        mode = _os.environ.get("SSD_MIMO_DSKIP_EPILOGUE", "auto").lower()
        if mode not in ("auto", "on", "off", "host", "0", "1"):
            raise ValueError("SSD_MIMO_DSKIP_EPILOGUE must be auto/on/off/host/0/1")
        if not self.skip:
            return "off"
        siso = self.R == self.O == 1
        # the upstream SISO provider owns its skip
        if siso and self.m3_kernel == "upstream":
            return "in_scan"
        if mode == "host":
            return "host"
        # only the in-scan arm rides the gate epilogue (`epilogue` applies the gains in `rmix_reduce`)
        fusable = gate_z is not None
        if siso and mode not in ("on", "1"):
            return "in_scan" if fusable else "host"
        if (mode not in ("off", "0") and ref.is_cuda
                and self.backend in ("cuda", "auto")
                and ref.dtype in (torch.float32, torch.float16, torch.bfloat16)):
            return "epilogue"
        return "in_scan" if (siso and fusable) else "host"

    def _dskip_in_epilogue(self, ref, gate_z=None):
        return self._dskip_route(ref, gate_z) == "epilogue"

    def _input_skip(self, drive, gate_z, rmix):
        """Weight each input's O-vector by the same sibling coefficient as q."""
        lead = drive.shape[:-1]
        values = _ref_f(drive).reshape(*lead, self.H, self.G, self.R)
        if gate_z is not None:
            values = values * F.silu(_ref_f(gate_z)).reshape_as(values)
        if self.r_mix_mode != "none":
            weight = _ref_f(self.r_mix)
            if rmix is not None and self.r_mix_dyn:
                weight = weight + _ref_f(rmix).reshape(*lead, self.H, self.R)
            values = values * weight.unsqueeze(-2)
        # contraction avoids materializing a B*L*H*G*R*O skip tensor
        return torch.einsum("...hgr,hgro->...hgo", values,
                            _ref_f(self._skip_gains())).reshape(*lead, self.d_out)

    def _h_final_dtype(self, x: torch.Tensor) -> torch.dtype:
        """dtype of the returned hidden carry: the supplied State's on a continuation, else the IO dtype."""
        dt = getattr(self, "_carry_dtype", None)
        return x.dtype if dt is None else dt

    def _finish_forward(self, x, u, y, s_final, rmix=None, *,
                        gate_z=None, y_contig=False, endpoint_kw=None):
        if rmix is None and self.r_mix_dyn:
            rmix = self.r_mix_proj(u)
        route = self._dskip_route(x, gate_z)
        if route == "epilogue":
            B, L = x.shape[:2]
            H, G, R, O = self.H, self.G, self.R, self.O
            c = None if self.r_mix_mode == "none" else self.r_mix
            rm = (rmix.reshape(B, L, H, R).to(y.dtype)
                  if self.r_mix_dyn and c is not None else None)
            d = self._skip_gains_kernel(y.dtype)
            if endpoint_kw is not None:
                from ..kernels.ssd_mimo.endpoint_mix import endpoint_mix
                out = endpoint_mix(y, *self._endpoint_operands(y, endpoint_kw),
                                   c, rm, H, R, O, G, x.dtype, d=d)
            else:
                from ..kernels.ssd_mimo.rmix_epilogue import rmix_reduce
                drive = x.reshape(B, L, H, G, R).contiguous()
                gate = (None if gate_z is None else
                        gate_z.reshape(B, L, H, G, R).contiguous())
                out = rmix_reduce(y, c, rm, H, R, O, G, x.dtype,
                                  d=d, drive=drive, gate=gate)
            self.h_final = (s_final.reshape(B, R, H, G, self.N)
                           .permute(0, 2, 3, 1, 4)
                           .reshape(B, H, self.P, self.N).to(self._h_final_dtype(x)).clone())
            return out
        out = self._reduce_siblings(x, u, y, s_final, rmix,
                                   y_contig=y_contig, endpoint_kw=endpoint_kw)
        if route == "host":
            out = out + self._input_skip(x, gate_z, rmix).to(out.dtype)
        return out

    def _reduce_siblings(self, x, u, y, s_final, rmix=None,
                        *, y_contig: bool = False, endpoint_kw=None) -> torch.Tensor:
        """Reduce sibling readouts and set the differentiable `h_final`.

        y may be token-major or channel-major (strides are inspected, no layout copy); rmix is the optional
        projected mixing weight."""
        if rmix is None and self.r_mix_dyn:
            rmix = self.r_mix_proj(u)
        B, L, _ = x.shape
        H, P, G, R, N = self.H, self.P, self.G, self.R, self.N

        if endpoint_kw is not None:
            from ..kernels.ssd_mimo.endpoint_mix import endpoint_mix
            c_st = None if self.r_mix_mode == "none" else self.r_mix
            rm = (rmix.view(B, L, H, R).to(y.dtype)
                  if self.r_mix_dyn and self.r_mix_mode != "none" else None)
            out = endpoint_mix(y, *self._endpoint_operands(y, endpoint_kw),
                               c_st, rm, H, R, self.O, G, x.dtype)
            self.h_final = (s_final.reshape(B, R, H, G, N)
                           .permute(0, 2, 3, 1, 4).reshape(B, H, P, N)
                           .to(self._h_final_dtype(x)).clone())
            return out

        if self.O == 1 and _y_is_token_major(y):
            # token-major fast path: `y` is already [B, L, H_k, G] memory, so the reduction needs no
            # transpose; at R == 1 with the mix off the epilogue is a single view
            yt = y.permute(0, 3, 1, 2)                     # [B,L,H_k,G]
            yv = yt.reshape(B, L, R, H, G) if R > 1 else yt
            if self.r_mix_mode == "none":
                y_out = (yv.sum(dim=2) if R > 1 else yv)
            elif R == 1:
                c = self.r_mix.to(yv.dtype).reshape(H)         # [H]
                if self.r_mix_dyn:
                    c = c[None, None, :] + rmix.view(B, L, H).to(yv.dtype)
                    y_out = yv * c[..., None]
                else:
                    y_out = yv * c[None, None, :, None]
            else:
                ck = self.r_mix.to(yv.dtype).permute(1, 0)[None, None,
                                                          :, :, None]
                if self.r_mix_dyn:
                    ck = ck + (rmix.view(B, L, H, R).permute(0, 1, 3, 2)
                               .to(yv.dtype)[..., None])
                y_out = (yv * ck).sum(dim=2)
            y_out = y_out.reshape(B, L, H * G).to(x.dtype)
            self.h_final = (s_final.reshape(B, R, H, G, N)
                           .permute(0, 2, 3, 1, 4).reshape(B, H, P, N)
                           .to(self._h_final_dtype(x)).clone()) if R > 1 else \
                s_final.reshape(B, H, G, N).to(self._h_final_dtype(x)).clone()
            return y_out

        if R == 1 and self.O == 1:
            # SISO: no sibling reduction, merge the stride-compatible H/G axes
            yk = y.reshape(B, H, G, L)                         # [B,H,G,L]
            if self.r_mix_mode != "none":
                c = self.r_mix.to(yk.dtype).reshape(H)         # [H] (R=1)
                if self.r_mix_dyn:
                    cd = (c[None, :, None] + rmix.view(B, L, H).to(yk.dtype)
                          .permute(0, 2, 1))                   # [B,H,L]
                    yk = yk * cd[:, :, None, :]
                else:
                    yk = yk * c[None, :, None, None]
            y_out = (yk.permute(0, 3, 1, 2).reshape(B, L, H * G)
                     .to(x.dtype))
            if y_contig:
                y_out = y_out.contiguous()
            self.h_final = s_final.reshape(B, H, G, N).to(self._h_final_dtype(x)).clone()
            return y_out

        # fused sibling reduction when supported; the torch path below defines the same weighted sum
        if (_os.environ.get("SSD_MIMO_RMIX_EPILOGUE", "1") == "1"
                and self.backend != "torch" and y.is_cuda
                and y.dtype in (torch.float32, torch.bfloat16,
                                torch.float16)):
            from ..kernels.ssd_mimo.rmix_epilogue import rmix_reduce
            c_st = None if self.r_mix_mode == "none" else self.r_mix
            rm = (rmix.view(B, L, H, R).to(y.dtype)
                  if (self.r_mix_dyn and self.r_mix_mode != "none")
                  else None)
            y_out = rmix_reduce(y, c_st, rm, H, R, self.O, G, x.dtype)
            self.h_final = (s_final.reshape(B, R, H, G, N)
                           .permute(0, 2, 3, 1, 4)
                           .reshape(B, H, P, N)
                           .to(self._h_final_dtype(x)).clone())
            return y_out
        # kernel Y is [B, H_k, G, L] (O==1) or [B, H_k, O, G, L] (O>1), H_k = R*H r-major: scale and
        # reduce channel-major (r outermost) and transpose once at the end
        O = self.O
        off = self.r_mix_mode == "none"
        c = (None if off
             else self.r_mix.to(y.dtype).permute(1, 0))            # [R, H]
        if self.O == 1:
            yk = y.reshape(B, R, H, G, L)
            if off:
                y_out = yk.sum(dim=1)                              # [B,H,G,L]
            else:
                ck = c[None, :, :, None, None]                     # [1,R,H,1,1]
                if self.r_mix_dyn:
                    ck = ck + (rmix.view(B, L, H, R).permute(0, 3, 2, 1)
                               .to(y.dtype)[:, :, :, None, :])     # [B,R,H,1,L]
                y_out = (yk * ck).sum(dim=1)                       # [B,H,G,L]
            y_out = y_out.permute(0, 3, 1, 2)                      # [B,L,H,G]
        else:
            yk = y.reshape(B, R, H, O, G, L)
            if off:
                y_out = yk.sum(dim=1)                              # [B,H,O,G,L]
            else:
                ck = c[None, :, :, None, None, None]               # [1,R,H,1,1,1]
                if self.r_mix_dyn:
                    ck = ck + (rmix.view(B, L, H, R).permute(0, 3, 2, 1)
                               .to(y.dtype)[:, :, :, None, None, :])
                y_out = (yk * ck).sum(dim=1)                       # [B,H,O,G,L]
            y_out = y_out.permute(0, 4, 1, 3, 2)                   # [B,L,H,G,O]
        y_out = y_out.reshape(B, L, H * G * self.O).to(x.dtype)
        if y_contig:
            y_out = y_out.contiguous()

        self.h_final = (s_final.reshape(B, R, H, G, N)
                       .permute(0, 2, 3, 1, 4)
                       .reshape(B, H, P, N)
                       .to(self._h_final_dtype(x)).clone())

        return y_out

    # Combined single-node path (core half): `SSDMimoSSM` can run the whole mixer (in_proj, conv, token
    # tail, scan, gate, norm, out_proj) as one autograd node.  This half runs the ordinary `_forward_cuda`
    # under no_grad with the scan and tail Functions captured (`capture_scan`, `capture_tails`) and
    # hand-writes only the adjoint of the layout prep plus the conv and GEMM backward calls.  Outside the
    # envelope `combined_refusal` names the failing clause and the multi-node path runs.

    def combined_refusal(self, x, *, gate_z=None, h_init=None,
                         extra_carry=None, return_state=False):
        """Why this call cannot take the combined single-node path, or None.  Each clause is a place the
        hand-written adjoint would need a branch, not a kernel limit."""
        from ..layers import tails as _tails
        if self.R != 1 or self.O != 1:
            return f"combined is wired at d_ssm_in = d_ssm_out = 1 (d_ssm_in={self.R}, d_ssm_out={self.O})"
        backend = self.backend
        if backend == "auto":
            backend = "cuda" if x.is_cuda else "torch"
        if backend != "cuda":
            return f"backend={backend!r} (the combined node is the CUDA route)"
        if not x.is_cuda or x.dtype == torch.float64:
            return f"CUDA, non-fp64 only (dtype={x.dtype}, cuda={x.is_cuda})"
        # gate first: without it `_dskip_route` leaves the in-scan arm and would blame the skip
        if gate_z is None:
            return "no gate: the fused epilogue owns the combined output store"
        if self.skip and self._dskip_route(x, gate_z) != "in_scan":
            return "D skip outside the scan requires its ordinary autograd path"
        if h_init is not None or extra_carry or return_state:
            return "a state carry (h_init / continuation / return_state)"
        if self.G_kernel != self.G:
            return f"G={self.G} is zero-padded to {self.G_kernel}"
        if self.r_mix_mode != "none":
            return f"r_mix={self.r_mix_mode!r} (the sibling mix)"
        if self.rot_mode is not None:
            return (f"rot_mode={self.rot_mode!r}: its angle map is host-side "
                    "differentiable work the combined adjoint does not cover")
        if self.use_trap:
            return "the trapezoid"
        if _y_layout() != "token":
            return f"SSD_MIMO_Y_LAYOUT={_y_layout()!r} (token-major only)"
        if not (self._fuse_proj and _tails.tails_enabled()
                and _tail_law_is_stock()):
            return "the eager token-map tail (SSD_MIMO_TAILS=0 or a dial law)"
        if (self.conv1d is not None or self.bc_conv1d is not None) and not self._conv_raw:
            return ("the combined node runs the conv's raw kernels (plain Triton or the causal_conv1d package), "
                    "which are not in use (kernel_options conv_kernel='torch')")
        if self.N & (self.N - 1) or not (16 <= self.N <= 128):
            return f"N={self.N} takes the tiled / padded scan repair"
        if x.dim() == 3 and x.shape[1] % _COMBINED_CHUNK:
            # the scan pads L to whole chunks and the combined backward views dy as chunks, so padded L
            # would need zero-extension and slicing; those (short-sequence) calls take the multi-node path
            return (f"L={x.shape[1]} is not a multiple of the scan chunk "
                    f"({_COMBINED_CHUNK}), so the scan pads it")
        return None

    def out_norm_refusal(self, x, *, gate_z=None, state=None,
                         return_state=False):
        """Why this call cannot fuse the enclosing module's output RMSNorm into the gate epilogue kernel,
        or None.  Each clause is a place where the module's epilogue after the scan is not a pure view
        (so a normalised `y` would not be its output) or the epilogue kernel does not run."""
        if self.m3_kernel == "upstream":
            return "upstream M3: output norm remains in the enclosing layer"
        from ..layers import tails as _tails            # noqa: F401
        if _os.environ.get("SSD_MIMO_OUT_NORM_FUSED", "1") != "1":
            return "SSD_MIMO_OUT_NORM_FUSED=0"
        backend = self.backend
        if backend == "auto":
            backend = "cuda" if x.is_cuda else "torch"
        if backend != "cuda" or not x.is_cuda or x.dtype == torch.float64:
            return (f"backend={backend!r} / dtype={x.dtype}: the fused "
                    "output norm is the CUDA epilogue kernel")
        # gate first, as in `combined_refusal`
        if gate_z is None:
            return "no gate: the fused norm rides the gate epilogue kernel"
        if self.skip and self._dskip_route(x, gate_z) != "in_scan":
            return "D skip outside the scan requires its ordinary autograd path"
        if state is not None or return_state:
            return "a state carry (the step / continuation route)"
        if self.R != 1 or self.O != 1:
            return (f"d_ssm_in={self.R}, d_ssm_out={self.O}: the fused norm is wired at "
                    "d_ssm_in = d_ssm_out = 1 (at d_ssm_out > 1 the gate broadcasts over o, so "
                    "dZ needs a segmented reduction inside the token row)")
        if self.r_mix_mode != "none":
            return (f"r_mix={self.r_mix_mode!r}: the sibling mix runs "
                    "AFTER the scan, so the norm would see the wrong row")
        if self.G_kernel != self.G:
            return f"G={self.G} is zero-padded to {self.G_kernel}"
        if _y_layout() != "token":
            return f"SSD_MIMO_Y_LAYOUT={_y_layout()!r} (token-major only)"
        if self.use_trap and self.rot_mode != "m3":
            return ("the HOST two-tap trapezoid correction runs after "
                    "the scan's readout")
        if self.rot_mode == "m3" and self.use_trap and not (
                _os.environ.get("SSD_MIMO_TRAP_INKERNEL", "1") == "1"
                and not self.use_theta_h):
            return "the host trapezoid correction runs after the readout"
        if _ua_gate_epilogue_off():
            return ("SSD_MIMO_GATE_INKERNEL=1 / SSD_MIMO_GATE_FUSED=0: "
                    "the fused norm rides the DEDICATED epilogue kernel")
        if self.N & (self.N - 1) or not (16 <= self.N <= 128):
            return (f"N={self.N} takes the tiled / padded scan repair, "
                    "and a per-N-tile norm is not the row norm")
        return None

    def _combined_core_fwd(self, x, u, gate_z, tok_raw, out_norm=None):
        """`_forward_cuda` under no_grad with the scan and tail Functions captured; returns ``(y_out, tape)``."""
        from ..layers import tails as _tails
        from ..kernels.ssd_mimo import autograd as _ua
        own_slab = tok_raw is None
        with torch.no_grad():
            cm = False
            slab = tok_raw
            if own_slab:
                cm = (self._tail_owns_all and u.dim() == 3
                      and u.stride(-1) != 1 and self._tail_will_fuse(u))
                slab = self.proj_slab(u, channel_major=cm)
            self._bc_capture = bcap = []
            try:
                with _tails.capture_tails() as tslot, \
                        _ua.capture_scan() as sslot:
                    y_out = self._forward_cuda(x, u, None, gate_z, slab,
                                               out_norm=out_norm)
            finally:
                self._bc_capture = None
        if len(sslot) != 1 or len(tslot) != 1:
            raise RuntimeError(
                "combined SSD-MIMO node: expected exactly one captured scan "
                f"and one captured tail, got {len(sslot)} / {len(tslot)} -- "
                "the call took a repaired (N-tiled / chunk-halved) route "
                "that `combined_refusal` should have excluded")
        tape = dict(scan=sslot[0], tail=tslot[0], slab=slab,
                    own_slab=own_slab, slab_cm=cm, u=u,
                    bc_in=bcap[0] if bcap else None)
        return y_out, tape

    def _combined_core_bwd(self, tape, dy_out, dz_out=None, dslab_out=None):
        """Adjoint of `_combined_core_fwd`.

        `dy_out` is the core output gradient [B, L, H*G] token-major.  `dz_out` / `dslab_out`, when given,
        are buffers for the gate and token-slab gradients (the mixer points both at its one in_proj-slab
        gradient, removing the split's backward `cat`).  Returns ``(dx_branch, du, d_tok_raw, grads)``;
        `grads` maps parameter names relative to this core to gradients."""
        scan, tail = tape["scan"], tape["tail"]
        slab, u = tape["slab"], tape["u"]
        H, G, N, R, O = self.H, self.G, self.N, self.R, self.O
        Bsz, L = slab.shape[0], slab.shape[1]
        T = scan.meta["T"]
        nc = L // T
        if scan.meta["L_padded"] != L or nc * T != L:
            raise AssertionError(
                f"combined SSD-MIMO node: L={L} is not nc*T={nc}*{T} (the "
                f"scan padded it to {scan.meta['L_padded']}); "
                "`combined_refusal` should have excluded this call")
        grads = {}

        # 1. the scan, driven by hand: the Function's Y gradient is a view of the token-major `dy_out`
        if not dy_out.is_contiguous():
            dy_out = dy_out.contiguous()
        dY = dy_out.view(Bsz, nc, T, H, 1, G).permute(0, 3, 1, 2, 4, 5)
        scan.ctx.dz_out = dz_out
        g = scan.backward(dY, None)
        # a hand-written backward must free saved tensors itself
        del dY, dy_out
        scan.ctx.saved_tensors = ()
        tape["scan"] = None

        # 2. `_prepare_kernel_inputs` in reverse (x_k is a view of the conv'd drive at R = 1)
        dx_branch = (g["x_in"].reshape(Bsz, H, G, L).permute(0, 3, 1, 2)
                     .reshape(Bsz, L, H * G))
        # the scan drops the size-1 group axis, so the adjoint of the module's size-1 reshapes is a reshape
        d_bt = g["B_tok"].reshape(Bsz, L, N, R)
        d_ct = g["C_tok"].reshape(Bsz, L, N, O)
        if self.a_dyn:
            d_dec_k, d_dt_k = g["delta"], g["delta_w"]
        else:
            d_dec_k, d_dt_k = None, g["delta"]
        if self.A_log is not None and g["a_log"] is not None:
            d_a = g["a_log"].view(R, H).permute(1, 0).to(self.A_log.dtype)
            grads["A_log"] = d_a
        if self.head_bias:
            if g["beta"] is not None:
                grads["B_bias"] = (g["beta"].view(R, H, N, R).sum(0)
                                   .to(self.B_bias.dtype))
            if g["gamma"] is not None:
                grads["C_bias"] = (g["gamma"].view(R, H, N, O).sum(0)
                                   .to(self.C_bias.dtype))
        if self.skip and g["d_skip"] is not None:
            grads["D"] = (g["d_skip"].sum(dim=(1, 2))
                          if self.skip_sharing == "head" else g["d_skip"]).reshape_as(self.D).to(self.D.dtype)
        if self.use_theta_h and g.get("theta_h") is not None:
            # in-kernel read offset: [Hk, K] r-major -> [H, K]; only the rotated pairs are parameters
            grads["theta_h"] = (g["theta_h"].view(R, H, N // 2).sum(0)
                                [:, :self.theta_h.shape[-1]]
                                .to(self.theta_h.dtype))
        # the enclosing mixer's output-norm weight (fused into the gate epilogue), keyed outside this core
        if g.get("out_norm_w") is not None:
            grads["@out_norm_w"] = g["out_norm_w"]

        # 3. the fused token tail in reverse
        shared_bias = bool(self.bc_bias) and not self.head_bias
        need_b = bool(self.b_norm or shared_bias
                      or (self.use_g_b and R == 1))
        need_c = bool(self.c_norm or shared_bias or self.use_g_c)
        td = tail.backward(d_bt.reshape(Bsz, L, N, R) if need_b else None,
                           d_ct.reshape(Bsz, L, N, O) if need_c else None,
                           d_dt_k, d_dec_k)
        d_b_src = td["b_src"] if need_b else d_bt.reshape(Bsz, L, N * R)
        d_c_src = td["c_src"] if need_c else d_ct.reshape(Bsz, L, N * O)
        g["B_tok"] = g["C_tok"] = g["delta"] = g["delta_w"] = None
        d_dt_k = d_dec_k = None
        tail.ctx.saved_tensors = ()
        tape["tail"] = None
        if self.b_norm and td["w_b"] is not None:
            grads["B_norm.weight"] = td["w_b"]
        if self.c_norm and td["w_c"] is not None:
            grads["C_norm.weight"] = td["w_c"]
        if shared_bias:
            if td["bias_b"] is not None:
                grads["B_bias"] = td["bias_b"]
            if td["bias_c"] is not None:
                grads["C_bias"] = td["bias_c"]

        # 4. the B/C conv in reverse
        d_parts = {"dt": td["dt_src"]}
        if self.a_dyn:
            d_parts["a"] = td["a_src"]
        if self.use_g_b:
            d_parts["gB"] = td["gb_src"]
        if self.use_g_c:
            d_parts["gC"] = td["gc_src"]
        if self.bc_conv1d is None:
            d_parts["B"], d_parts["C"] = d_b_src, d_c_src
        else:
            from ..layers.conv import (causal_conv1d_cl_bwd)
            d_bc = torch.cat([d_b_src.reshape(Bsz, L, N * R),
                              d_c_src.reshape(Bsz, L, N * O)], dim=-1)
            conv = self.bc_conv1d.conv
            bc_in = tape["bc_in"]
            # a non-dense `bc_in` (narrow read in place) needs a caller-provided channel-last dx; the
            # package's C++ entry rejects a channel-major one
            dx_cl = (None if bc_in.stride(2) == bc_in.shape[1] else
                     torch.empty(bc_in.shape[0], bc_in.shape[2],
                                 bc_in.shape[1], device=bc_in.device,
                                 dtype=bc_in.dtype).transpose(1, 2))
            d_in, d_w, d_b = causal_conv1d_cl_bwd(
                bc_in, conv.weight, conv.bias,
                d_bc.transpose(1, 2), "silu", dx=dx_cl, package=self._conv_impl)
            d_in = d_in.transpose(1, 2)
            grads["bc_conv1d.conv.weight"] = d_w
            grads["bc_conv1d.conv.bias"] = d_b
            d_parts["B"], d_parts["C"] = d_in.split([N * R, N * O], dim=-1)
            if self._bc_adjacent:
                # B|C are the slab's first columns: one copy back
                d_parts["_bc"] = d_in
            del d_bc
            tape["bc_in"] = None
        # dt bias gradient: the tail kernel's fp32 block-partial reduction (as the multi-node route)
        d_dtb = td.get("bias_dt")
        del d_b_src, d_c_src, d_bt, d_ct, td

        # 4b. token-projection bias gradients: sum each contiguous part (a slab slice reduces uncoalesced)
        for key, attr, width in zip(self._proj_keys, self._proj_attrs,
                                    self._proj_widths):
            mod = getattr(self, attr)
            gp = d_parts.get(key)
            if mod.bias is not None and gp is not None:
                if attr == "dt_proj" and d_dtb is not None:
                    grads[f"{attr}.bias"] = d_dtb.to(mod.bias.dtype)
                    continue
                grads[f"{attr}.bias"] = (gp.reshape(-1, width).sum(0)
                                         .to(mod.bias.dtype))

        # 5. the token slab gradient in one buffer
        d_slab = (torch.zeros(Bsz, L, self.tok_width, device=slab.device,
                              dtype=slab.dtype) if dslab_out is None
                  else dslab_out)
        off = 0
        joint = d_parts.pop("_bc", None)
        if joint is not None:
            d_slab[..., :self._bc_width].copy_(
                joint.reshape(Bsz, L, self._bc_width))
            off = self._bc_width
        for key, width in zip(self._proj_keys, self._proj_widths):
            if joint is not None and key in ("B", "C"):
                continue
            gp = d_parts.get(key)
            if gp is not None:
                d_slab[..., off:off + width].copy_(gp.reshape(Bsz, L, width))
            elif dslab_out is not None:
                d_slab[..., off:off + width].zero_()
            off += width

        # 6. the token GEMM, when this core owns it (main_branch)
        du = None
        if tape["own_slab"]:
            du, dw_tok = self._proj_slab_bwd(u, d_slab, tape["slab_cm"])
            grads.update(self.scatter_tok_grads(dw_tok))
            d_slab = None
        return dx_branch, du, d_slab, grads

    def scatter_tok_grads(self, dw_tok) -> dict:
        """Split the fused token-projection weight gradient onto the per-projection Linears (shared by the
        core's own GEMM and the enclosing mixer's folded in_proj; biases are handled in `_combined_core_bwd`)."""
        grads: dict = {}
        off = 0
        for attr, width in zip(self._proj_attrs, self._proj_widths):
            mod = getattr(self, attr)
            grads[f"{attr}.weight"] = self._proj_w_grad(
                attr, dw_tok[off:off + width]).to(mod.weight.dtype)
            off += width
        return grads

    def _proj_slab_bwd(self, u, d_out, channel_major):
        """Adjoint of `proj_slab`: (du, dW) with dW in the fused [tok_width, d_sel] layout."""
        w, _ = self.tok_proj_weight()
        Bsz = u.shape[0]
        if channel_major:
            d_outc = d_out.transpose(1, 2)                 # [B, tok, L]
            dw = torch.bmm(d_outc, u).sum(0)
            du = torch.bmm(w.to(d_outc.dtype).t()
                           .expand(Bsz, w.shape[1], w.shape[0]),
                           d_outc).transpose(1, 2)
        else:
            f = d_out.reshape(-1, d_out.shape[-1])
            dw = f.t() @ u.reshape(-1, u.shape[-1])
            du = (f @ w.to(f.dtype)).view(*u.shape)
        return du, dw

    def scan_step(
        self,
        x_t: torch.Tensor,
        state: State | dict | None = None,
        *,
        gate_z: torch.Tensor | None = None,
        sel_x: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, State]:
        """Single-token autoregressive decode step.

        O(1) step via `ssd_mimo_torch.ssd_step` (all heads and siblings batched, same State).

        Parameters
        ----------
        x_t : (B, d_inner)
            Single input token.
        state : State
            From :meth:`init_state` or a previous step.
        gate_z : (B, d_out) or None
            Output gate from the block's z branch.
        sel_x : (B, d_model) or None
            Block input token (residual source), see :meth:`forward`.
        tok_raw : (B, tok_width) or None
            The fused token-map projection already applied by the caller.

        Returns
        -------
        y_t : (B, d_out)
        state : State with every carry of this config.
        """
        if _torch_impl() == "legacy":
            return self._step_legacy(x_t, state, gate_z=gate_z, sel_x=sel_x,
                                     tok_raw=tok_raw)
        from .ssd_mimo_torch import ssd_step
        return ssd_step(self, x_t, state, gate_z=gate_z, sel_x=sel_x,
                        tok_raw=tok_raw)

    def _step_legacy(
        self,
        x_t: torch.Tensor,
        state: State | dict | None = None,
        gate_z: torch.Tensor | None = None,
        sel_x: torch.Tensor | None = None,
        tok_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, State]:
        """Older single-token step (the ``_TORCH_IMPL = "legacy"`` test hook); also the CUDA route's state replay.
        Arguments and return as in :meth:`scan_step` (`state` is required)."""
        output_gate = gate_z if self.R > 1 or self.O > 1 else None
        if gate_z is not None and gate_z.shape != (*x_t.shape[:-1], self.d_out):
            raise ValueError(f"gate_z must have shape {(*x_t.shape[:-1], self.d_out)}")
        if output_gate is not None:
            gate_z = None
        Bsz = x_t.shape[0]
        H, P, G, R, N, O = self.H, self.P, self.G, self.R, self.N, self.O

        state = ScanState(self).resolve_state(
            state, batch_size=Bsz, device=x_t.device, dtype=x_t.dtype)
        cache = state
        h_old = cache["hidden_state"]
        u_t = self._select(x_t, sel_x)

        # token maps off the same fused GEMM as the forward (add L=1, squeeze back)
        u_seq = u_t.unsqueeze(1)                              # [B, 1, d_sel]
        parts = (self._proj_split(u_seq) if tok_raw is None
                 else self.split_tok_raw(tok_raw.unsqueeze(1)))
        if self.bc_conv1d is not None:
            Bt, Ct, gB, gC, dt, dt_dec, bc_state = self._token_maps(
                u_seq, bc_conv_state=cache["bc_conv_state"], parts=parts)
            cache = {**cache, "bc_conv_state": bc_state}
        else:
            Bt, Ct, gB, gC, dt, dt_dec = self._token_maps(u_seq, parts=parts)
        Bt = Bt.squeeze(1)                                    # [B, N, R]
        Ct = Ct.squeeze(1)                                    # [B, N, O]
        if gB is not None:
            gB = gB.squeeze(1)                                # [B, N, R]
        if gC is not None:
            gC = gC.squeeze(1)                                # [B, N, O]
        dt = dt.squeeze(1)                                    # [B, H, P]
        if dt_dec is not None:
            dt_dec = dt_dec.squeeze(1)                        # [B, H, P]

        # trapezoid: current-clock endpoint weights (decay keeps the plain d_dec; trap modifies delta_w only)
        lam_trap = None
        if self.use_trap:
            if "dt_prev" in cache:
                raise ValueError("SSDMimoSSM: a 'dt_prev' cache entry is not accepted (the trapezoid "
                                 "uses endpoint weights); restart from the sequence prefix")
            lam_trap = torch.sigmoid(parts["trap"].squeeze(1))[:, :, None]
            # build the unscaled write; the current dt scales both endpoints in the recurrence below
            dt_w = torch.ones_like(dt)
        else:
            dt_w = dt

        # decay a_t = exp(-rate * d_dec), rate = exp(A_log); a_dyn form: rate = 1, decay lives in dt_dec
        if self.A_log is None:
            rates = torch.ones(H, P, device=dt.device, dtype=dt.dtype)
        else:
            rates = self.A_log.exp()          # [H, R]
            rates = (rates[:, None, :].expand(H, G, R)
                     .reshape(H, P))                          # [H, P]
        d_dec = dt if dt_dec is None else dt_dec              # [B, H, P]
        a_t = torch.exp(
            -d_dec.unsqueeze(-1) * rates.unsqueeze(-1))       # [B, H, P, 1]

        # rotation angle (fold convention: out_e = cos*e + sin*o ; out_o = -sin*e + cos*o)
        theta_t = None
        # theta_hr: per-(head, rank) angle [B|1, H|1, R|1, K] (broadcast)
        theta_hr = None
        if self.rot_mode == "m3":
            # cumulative delta-scaled angle carried in the cache, advanced before use (inclusive cumsum)
            if self.rot_mode == "m3":
                # Mamba-3's angle: tanh(Ang_t[h,k]) * pi * dt[h,t], on the write step
                dw_hr = _ref_f(dt.view(Bsz, H, G, R)[:, :, 0, :])  # [B,H,R]
                # the clock lives on the rotated pairs only; the real pairs enter the fold as zeros
                ka = self.n_rope_angles
                raw = _ref_f(parts["theta"].squeeze(1))
                if self.omega_per_head:
                    # one angle per head, on every rotated pair
                    raw = raw.view(Bsz, H, 1, 1).expand(Bsz, H, 1, ka)
                elif self.theta_per_head:
                    raw = raw.view(Bsz, H, 1, ka)               # [B,H,1,ka]
                else:
                    raw = raw.view(Bsz, 1, 1, ka).expand(Bsz, H, 1, ka)
                ang = torch.tanh(raw)
                if self.learn_theta_freq and self.omega_layout != "pair":
                    # [H,1,1] (head) or [H,1,ka] (head_pair)
                    omega = (math.pi * torch.tanh(_ref_f(self.theta_freq))
                             ).view(H, 1, -1)
                elif self.learn_theta_freq:
                    omega = math.pi * torch.tanh(_ref_f(self.theta_freq))
                else:
                    omega = math.pi
                incr = -dw_hr[..., None] * ang * omega
            rot_ang = _wrap2pi_t(cache["rot_ang"] + incr)     # [B,H,R,ka]
            cache = {**cache, "rot_ang": rot_ang}
            theta_hr = (F.pad(rot_ang, (0, N // 2 - ka)) if ka < N // 2
                        else rot_ang)                         # [B,H,R,K]

        xi = x_t.view(Bsz, H, G, R)
        if self.bias_pre_rot and R == 1:
            # Mamba-3 fold order: the (ungated) write bias joins the raw map and rotates with it
            Bt_eff = (Bt if gB is None else gB * Bt)[:, None]  # [B,1,N,R]
            Bt_eff = Bt_eff + self.B_bias[:, 0].to(Bt.dtype)[None]  # [B,H,N,R]
            w = torch.einsum("bhnr,bhgr->bhgn", Bt_eff, xi)
            if lam_trap is not None:
                # the two-tap write needs the unscaled W_t (its second tap carries this step's dt), so the dt
                # scale is applied at the recurrence below, after the rotation
                drive = w[:, :, :, None, :].expand(Bsz, H, G, R, N)
            else:
                drive = (dt_w.view(Bsz, H, G, R)[..., None]
                         * w[:, :, :, None, :])                # [B,H,G,R,N]
        elif self.bias_pre_rot:
            # R > 1: gB is indexed by the decay unit r, the write map is contracted over the write rank r':
            #   Beff[h,r][n,r'] = gB[n,r] * B[n,r'] + beta[h,n,r']   (bias ungated, before the rotation)
            Bb = self.B_bias.to(Bt.dtype)                      # [H,R,N,R']
            if gB is None:
                Beff = Bt[:, None, None] + Bb[None]            # [B,H,R,N,R']
                w = torch.einsum("bhrnq,bhgq->bhgrn", Beff, xi)
            else:
                gB_r = gB.permute(0, 2, 1)                     # [B,R,N]
                Beff = (gB_r[:, None, :, :, None] * Bt[:, None, None]
                        + Bb[None])                            # [B,H,R,N,R']
                w = torch.einsum("bhrnq,bhgq->bhgrn", Beff, xi)
            if lam_trap is not None:
                drive = w
            else:
                drive = dt_w.view(Bsz, H, G, R)[..., None] * w
        else:
            w = torch.einsum("bnr,bhgr->bhgn", Bt, xi)       # [B, H, G, N]
            drive = (dt_w.view(Bsz, H, G, R)[..., None]
                     * w[:, :, :, None, :])                    # [B,H,G,R,N]
            if gB is not None:
                drive = drive * gB.permute(0, 2, 1).view(Bsz, 1, 1, R, N)

        def rotate_pairs(v, cs, sn, axis=-1):
            axis %= v.ndim
            if self.pair_layout == "half":
                e, o = v.chunk(2, dim=axis)
                return torch.cat((cs * e + sn * o, -sn * e + cs * o), dim=axis)
            even = [slice(None)] * v.ndim
            odd = even.copy()
            even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
            e, o = v[tuple(even)], v[tuple(odd)]
            return torch.stack((cs * e + sn * o, -sn * e + cs * o),
                               dim=axis + 1).flatten(axis, axis + 1)

        # write-side rotation on state-coordinate pairs of the drive.  It commutes with scalar dt but not
        # with an arbitrary diagonal gB, hence the gate-before-rotation order above.
        if theta_hr is not None:
            ang_b = theta_hr                                  # [B,H,R,K]
            cs_b = torch.cos(ang_b)[:, :, None, :, :]         # [B,H,1,R,K]
            sn_b = torch.sin(ang_b)[:, :, None, :, :]
            drive = rotate_pairs(drive, cs_b, sn_b)
        drive = drive.reshape(Bsz, H, G, R, N)
        drive = drive.reshape(Bsz, H, P, N)

        # per-head write bias (head); under bias_pre_rot it is already inside `drive`
        if self.head_bias and not self.bias_pre_rot:
            bd = torch.einsum("hrnq,bhgq->bhgrn",
                              self.B_bias.to(xi.dtype), xi)   # [B, H, G, R, N]
            bd = dt_w.view(Bsz, H, G, R)[..., None] * bd
            drive = drive + bd.reshape(Bsz, H, P, N)

        if lam_trap is not None:
            # two-tap write, causal form:
            #   S_t = a_t S_{t-1} + a_t dt_t (1-lam_t) W_{t-1} + lam_t dt_t W_t
            # with W_s the rotated, bias-folded outer product (`drive` = unscaled W_t).  The forward uses
            # the collapsed one-tap form plus a rank-1 output correction; same recurrence and final state.
            lam_p = lam_trap.reshape(Bsz, H, 1, 1)
            dtp = dt.reshape(Bsz, H, P, 1)      # THIS step's write scale
            key = "write_prev"
            w_prev = cache.get(key)
            drive_now = lam_p * dtp * drive
            if w_prev is not None:
                drive_now = drive_now + a_t * (1.0 - lam_p) * dtp * w_prev
            cache = {**cache, key: drive}
            drive = drive_now
        h_new = a_t * h_old + drive                           # [B, H, P, N]

        # the carry is fp32 under 16-bit IO (state contract): read in the carry's dtype (exact upcast)
        gated = (Ct if gC is None else gC * Ct).to(h_new.dtype)   # [B, N, O]
        h_view = h_new.view(Bsz, H, G, R, N)

        if self.bias_pre_rot and R == 1 and O == 1:
            # C_eff = R(Theta_t) (C + C_bias_h): the read bias joins the raw map and rotates with it
            K = N // 2
            ce = gated[:, None] + self.C_bias[:, 0].to(gated.dtype)[None]
            ce = ce[..., 0]                                    # [B,H,N]
            ang_c = theta_hr[:, :, 0, :]                       # [B,H,K]
            if self.use_theta_h:
                ang_c = ang_c + self._theta_h_pairs()[None]
            cs = torch.cos(ang_c)
            sn = torch.sin(ang_c)
            read = rotate_pairs(ce, cs, sn)
            yr = ((h_view[:, :, :, 0, :] * read[:, :, None, :])
                  .sum(-1)[:, :, :, None, None])               # [B,H,G,1,1]
        elif self.bias_pre_rot:
            # R > 1 or O > 1: the read is taken in the frame of kernel head (r, h), which is what makes
            # the phase cancel in the trapezoid's rank-1 correction
            K = N // 2
            ce = gated[:, None, None] + self.C_bias.to(gated.dtype)[None]
            ang_c = theta_hr                                   # [B,H,R,K]
            if self.use_theta_h:
                ang_c = ang_c + self._theta_h_pairs()[None, :, None, :]
            cs = torch.cos(ang_c)[..., None]                   # [B,H,R,K,1]
            sn = torch.sin(ang_c)[..., None]
            read = rotate_pairs(ce, cs, sn, axis=-2)
            yr = torch.einsum("bhgrn,bhrno->bhgro", h_view, read)
        elif self.use_theta_h or theta_hr is not None:
            K = N // 2
            # read-side angle theta_hr + theta_h, per (head, rank)
            ang_c = None
            if self.use_theta_h:
                ang_c = self._theta_h_pairs()[None, :, None, :]  # [1,H,1,K]
            if theta_hr is not None:
                ang_c = theta_hr if ang_c is None else ang_c + theta_hr
            cs = torch.cos(ang_c)[..., None]              # [B|1,H|1,R|1,K,1]
            sn = torch.sin(ang_c)[..., None]
            read = rotate_pairs(gated[:, None, None], cs, sn, axis=-2)
            read = read.expand(Bsz, H, R, N, O)
            yr = torch.einsum("bhgrn,bhrno->bhgro", h_view, read)
        else:
            read = gated.unsqueeze(1).expand(Bsz, R, N, O)
            yr = torch.einsum("bhgrn,brno->bhgro", h_view, read)

        # per-head read bias (head); under bias_pre_rot it is already inside `read`
        if self.head_bias and not self.bias_pre_rot:
            c_bias = self.C_bias.to(h_new.dtype)              # [H, R, N, O]
            y_gam = torch.einsum("hrno,bhgrn->bhgro", c_bias, h_view)
            yr = yr + y_gam

        # branch-local skip, before the gate and sibling reduction
        if self.skip:
            yr = yr + (x_t.reshape(Bsz, H, G, R).to(yr.dtype)[..., None]
                       * self._skip_gains().to(yr.dtype)[None])

        # SwiGLU gate before reduction (kernel placement)
        if gate_z is not None:
            gz = F.silu(gate_z).view(Bsz, H, G, R)[..., None]
            yr = yr * gz

        if self.r_mix_mode == "none":
            # one term at R == 1; at R > 1 (explicit r_mix="none") the plain unweighted sum
            y = yr.sum(dim=3)                                 # [B, H, G, O]
        else:
            c = self.r_mix.to(yr.dtype)[None, :, None, :]     # [1, H, 1, R]
            if self.r_mix_dyn:
                c = c + parts["rmix"].view(Bsz, H, 1, R).to(yr.dtype)
            y = (yr * c[..., None]).sum(dim=3)                # [B, H, G, O]

        self.h_final = h_new
        new_state = State(hidden_state=h_new)
        if self.rot_mode == "m3":
            new_state["rot_ang"] = cache["rot_ang"]  # already advanced
        if self.use_trap:
            key = "write_prev"
            new_state[key] = cache[key]
        if self.bc_conv1d is not None:
            new_state["bc_conv_state"] = cache["bc_conv_state"]
        new_state = ScanState(self)._conform_carries(new_state, state)
        self._final_state = new_state
        y = y.reshape(Bsz, H * G * O)
        if output_gate is not None:
            y = y * F.silu(output_gate)
        return y, new_state

    def theta_omega(self) -> torch.Tensor:
        """Angular frequency omega = pi * tanh(theta_freq): [K_rot] per pair, or [H] under
        `omega_per_head` (constant pi when not learned)."""
        if getattr(self, "learn_theta_freq", False):
            return math.pi * torch.tanh(self.theta_freq)
        if getattr(self, "omega_layout", "pair") == "head":
            return torch.full((self.H,), math.pi,
                              device=next(self.parameters()).device)
        return torch.full((getattr(self, "n_rope_angles", self.N // 2),), math.pi,
                          device=next(self.parameters()).device)

    def _theta_h_pairs(self) -> torch.Tensor | None:
        """theta_h on all N/2 pairs in working precision: learned on the rotated pairs, exact zeros on
        the real ones (the per-pair width the kernels take)."""
        if not self.use_theta_h:
            return None
        th = _ref_f(self.theta_h)
        pad = self.N // 2 - th.shape[-1]
        return F.pad(th, (0, pad)) if pad else th

    def _scan_init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> State:
        """Zero state for stepping: ``"hidden_state"`` (B, H, P, N) plus the rotation clock, trapezoid lag
        and B/C conv window this config carries.  ``dtype=None``: parameter dtype, at least fp32."""
        dev = device or next(self.parameters()).device
        if dtype is None:
            dtype = self._default_state_dtype()
        state = State(hidden_state=torch.zeros(batch_size, self.H, self.P, self.N,
                                    device=dev, dtype=dtype))
        if self.rot_mode == "m3":
            # cumulative delta-scaled angle per (head, rank), one entry per rotated pair
            ang_dtype = (torch.float64 if dtype == torch.float64
                         else torch.float32)
            state["rot_ang"] = torch.zeros(
                batch_size, self.H, self.R, self.n_rope_angles,
                device=dev, dtype=ang_dtype)
        if self.use_trap:
            # the previous rotated outer product (tensor zeros also let reset_state(mask=...) clear it)
            state["write_prev"] = torch.zeros_like(state["hidden_state"])
        if self.bc_conv1d is not None:
            state["bc_conv_state"] = self.bc_conv1d.init_state(
                batch_size, device=dev, dtype=dtype)
        return state

    def hidden_state_shape(self) -> tuple[int, ...]:
        """Shape of the hidden state per batch element: (H, P, N)."""
        return (self.H, self.P, self.N)


from dataclasses import dataclass
from typing import Literal
from ..config import _CoreConfigBase, SpecFields, _refuse

@dataclass(kw_only=True)
class SSDMimoConfig(SpecFields, _CoreConfigBase):
    """Architecture of an `SSDMimoSSM` layer as one typed, serializable object.

    A keyword-only dataclass: ``replace(**kw)`` copies with changes, ``from_dict`` is the strict dict
    constructor (unknown keys and wrongly typed values are a ``TypeError``), ``build(d_model=None,
    execution=None)`` returns the layer, and ``layer.to_recipe()`` records the full recipe.  How the
    layer runs (backend, precision, chunking, kernel options) is an `Execution`, not part of the config.
    These fields are the layer's only construction options.

    Field names are explanatory; each entry below also gives the paper's symbol where there is one
    (Paper 1, Section 5; the glossary table in ``docs/CONFIG_GLOSSARY.md`` lists them all).

    Parameters
    ----------
    d_model : int, optional
        Layer width: the input and output width of the layer.  None: `build(d_model)` or the enclosing
        `CausalLM` supplies it; a layer built from the config alone then uses ``d_inner`` (when ``expand``
        is None).
    d_state : int, default 64
        N in the paper: state rows per subsystem.
    expand : float, optional
        Inner width factor, ``d_inner = int(d_model * expand)``; None means 2.0 unless ``d_inner`` is given.
        Giving both requires them to agree.
    d_conv : int, default 4
        k_conv in the paper: width of the depthwise causal conv on the drive (and of the optional
        B/C conv), which sets the window (l = k_conv + 1 with the two-endpoint write).
        0 removes the conv *and* its SiLU, so a conv-free arm is
        "conv plus activation off". ``bc_conv=True`` requires ``d_conv > 0``.
    d_inner : int, optional
        d = H P in the paper, the drive width (= ``n_heads * d_head``).  Alone it is the width; with
        ``expand`` it must equal ``int(d_model * expand)``; neither: ``int(d_model * 2.0)``.
    d_ssm_in : int, default 1
        R in the paper, the input rank: input siblings per group, the MIMO rank of the write
        map B_t in (N, R). R = 1 is SISO.
    d_ssm_out : int or None, default None (= d_ssm_in)
        O in the paper, the output rank: outputs per subsystem (C_t in (N, O)). Exactly one of
        R, O equal to 1 triggers a RuntimeWarning (per-subsystem rank ceiling
        min(N, R, O) = 1, Paper 1 crosstalk theorem).
    n_groups : int or None, default None
        G in the paper: channels per head; default 64 // d_ssm_in. Any G >= 1 is legal, but
        the CUDA kernels pad G to a multiple of 16 (``self.G_kernel``), which
        costs time and activation memory for G < 16.
    n_heads, d_head : int or None
        H heads (eta) and head width P = G*R (``d_head``) in the paper. Give either or both
        (n_heads * d_head == d_inner); with neither, d_head = 64.  A scan head (eta, r) shares
        one scalar decay.
    row_blocks : list of RowBlock (or strict dicts) or None
        At most two blocks ``{rows|frac, decay, complex, gate}`` partitioning
        the N state rows. ``decay`` is "head_dynamic" (the paper's "SSD dynamic": selected per-head
        rate, Mamba-3) or "head_static" ("SSD static": static per-(head, sibling) rate, Mamba-2); a
        leading ``complex=True`` block is the rotated part of the state (K_rot rotating pairs).
        Default: one real "head_dynamic" block (no rotation).  Both blocks are rows of one scan with one decay per
        head, so their decays must agree (``ValueError`` otherwise); an omitted ``decay`` resolves to "head_dynamic"
        and the recorded config carries it on every block.  ``a_act``, ``a_dyn_per_sibling`` and ``a_proj_bias``
        belong to the "head_dynamic" rate: a non-default value with only "head_static" blocks is refused.
    theta_per_head : bool, default False
        One angle feature q_{k,t} per head instead of shared across heads (needs a
        complex block; forced True by ``omega_layout="head"``).
    learn_theta_freq : bool, default True
        Learn the pair frequencies omega_k = pi * tanh(phi_k) (``theta_freq`` is phi_k; init 0.999 pi,
        i.e. Mamba-3 at init); False keeps the constant pi. Needs a complex block.
    use_theta_h : bool, default True
        nu in the paper: static per-head read-phase offset (``theta_h``), one entry per rotated pair,
        carried on the read side. Needs a complex block.
    omega_layout : {"pair", "head_pair", "head"}, default "pair"
        Shape of the frequency omega: (K_rot,), (H, K_rot), or (H,) = one complex
        eigenvalue per head with one angle feature per head.
    theta_init : {"kaiming", "small", "zero"}, default "kaiming"
        Init of the angle projection (the projection of q_{k,t}): nn.Linear default, that draw scaled by
        0.1*sqrt(3) (near-zero rotation), or zeros (no rotation at init).
    b_norm, c_norm : bool, default True
        RMSNorm over N on the base maps B-hat / C-hat (the paper's default state-axis RMS normalization).
    bc_bias : {"head", "shared", "none"}, default "head"
        Granularity of the static additive B/C biases beta, gamma: per kernel head and sibling, one shared
        vector, or none. A complex block needs "head".
    bc_bias_init : {"auto", "ones", "spread"} or None, default None (= "auto")
        "ones" is Mamba-3's constant init; "spread" starts the R siblings in
        independent directions; "auto" picks "ones" at R == 1 or shared bias,
        else "spread". The recorded config stores the resolved value.
    b_gate, c_gate : bool or None, default None
        The write / read gates phi_B, phi_C of the paper (per sibling / per output). None means on iff
        R > 1 / O > 1 and some row block is gated. On a complex row pair the gate
        is one factor shared by both coordinates (`_gate_tie_index`).
    r_mix : {"auto", "learned", "none"}, default "auto"
        Sibling mix zeta in R^{H x R} combining the R readouts (a coefficient of the read map). "auto" is "none"
        at R == 1 and "learned" at R > 1; "none" at R > 1 is a plain unweighted
        sum (an ablation). Checkpoints must match the resolved choice.
    r_mix_dyn : bool or None, default None
        Selected sibling weights (a bias-free, zero-initialised d_sel -> H*R
        projection added to the static zeta). None means on at R > 1 with an
        active mix; True requires an active ``r_mix``.
    a_dyn_per_sibling : bool or None, default None (= False)
        With a "head_dynamic" block: one selected rate per (head, sibling)
        instead of per head.
    a_act : {"heavy_tail", "softplus"}, default "heavy_tail"
        act in the paper's SSD dynamic law a_t = -max(act(W_a s_t + b_a), 1e-4): ``ht(x) = 1 + x`` for
        x >= 0 and ``1 / (1 - x)`` below (upstream Mamba-3), or ``softplus(x)``.
    a_proj_bias : bool, default True
        b_a in the paper: bias on the selected-rate projection (zero-initialised).
    dt_min, dt_max, dt_init_floor : float, default 1e-3, 1e-1, 1e-4
        Range of the init of the step Delta_t = softplus(. + dt_bias) (``dt_min <= dt_max``) and its floor.
    write_rule : {"trapezoid", "euler"}, default "trapezoid"
        Quadrature of the forcing integral; the transition exp(delta A) is
        exact in both. "trapezoid" is the paper's two-endpoint write (Mamba-3's rule, weight lambda_t),
        "euler" the exponential-Euler Delta*B write (lambda = 1). "zoh" is accepted by name and raises
        (not implemented).
    trap_proj_bias : bool, default True
        b_lambda in the paper: bias of the two-endpoint weight projection (trapezoid only).
    skip : bool, default True
        The D term of the paper (y_t = C_t h_t + D_t u_t; "D" in Mamba): branch-local input gains
        (inside sibling reduction, before the gate).
    skip_sharing : {"head", "channel", "input_output"}, default "head"
        Shape of D: (H,), (H, G, R) or (H, G, R, O).
    pair_layout : {"interleaved", "half"}, default "interleaved"
        State pairs (2k, 2k+1) or (k, k+N/2); "half" needs a complex block and
        N % 32 == 0. A saved model changes layout only via ``convert_pair_layout``.
    proj_source : {"layer_input", "main_branch"}, default "layer_input"
        The coefficient source s_t of the paper: where the token maps (B, C, dt, gates, rate, theta,
        trap, zeta) read: the block input x_t of width ``d_model`` (Mamba-2/3) or the post-conv drive
        u_t (Mamba-1).
    bc_conv : bool or None, default None (= False)
        Depthwise causal conv + SiLU on the raw (B, C) projections; needs
        "layer_input" and ``d_conv > 0``.
    out_norm : bool, default False
        The paper's output RMS normalization: RMSNorm over the H*G*O scan output before ``out_proj``
        (off by default, on in the Mamba-2 preset).
    """

    pair_layout: str = "interleaved"
    a_dyn_per_sibling: bool = False
    a_act: Literal["heavy_tail", "softplus"] = "heavy_tail"
    r_mix: str = "auto"
    r_mix_dyn: bool | None = None
    bc_bias_init: str | None = None
    a_proj_bias: bool = True
    trap_proj_bias: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.width() is not None:
            self._geometry()
        # validate the spec now (the layer resolves it again, identically)
        spec = _config_spec.resolve("ssd_mimo", d_state=self.d_state, d_ssm_in=self.d_ssm_in,
                                    d_ssm_out=self.d_ssm_out, **self.spec_kwargs())
        if self.trap_proj_bias is False and self.write_rule not in (None, "trapezoid"):
            raise ValueError(
                f"SSDMimoConfig: trap_proj_bias=False, but write_rule={self.write_rule!r} builds no trapezoid weight projection, "
                "so it would be ignored; drop trap_proj_bias or use write_rule='trapezoid'")
        _config_spec.check_law_levers("SSDMimoConfig", spec["row_blocks"], _RATE_LEVERS, dict(
            a_dyn_per_sibling=self.a_dyn_per_sibling, a_act=self.a_act, a_proj_bias=self.a_proj_bias))

    def _geometry(self):
        return _ssd_geometry(
            self.width(), d_ssm_in=self.d_ssm_in, n_groups=self.n_groups, n_heads=self.n_heads,
            d_head=self.d_head)

    @staticmethod
    def _layer_class():
        return SSDMimoSSM


from ..compile_compat import dynamo_opaque
from typing import Any
from ..layers.norms import RMSNorm
from ..layers.proj import (autocast_cast, matmul_channel_major)
from ..layers.conv import (causal_depthwise_conv1d, causal_depthwise_conv1d_cl)
from ..layers.glue import (glue_on, param_cache_key, dt_fp32_on)

def _fused_in_proj(m, w, dt):
    """Apply the packed mixer/core projection (transient packed weights) preserving gradient layout;
    the channel-major drive and token-major gate share one cast input."""
    fold = glue_on("bias")
    cached = glue_on("wcat")
    key = None
    if cached:
        key = param_cache_key([w] + m.tok_proj_params(), dt, fold,
                               dt_fp32_on(m.kernel_options))
        hit = getattr(m, "_in_proj_cache", None)
        if hit is not None and hit[0] == key:
            return hit[1]
    wt, bt = m.tok_proj_weight()
    wcat = torch.cat([w[m.d_inner:], wt], dim=0).to(dt)
    bcat = None
    if bt is not None and fold:
        bcat, bt = F.pad(bt.to(dt), (m.d_inner, 0)), None
    val = (wcat, bcat, bt)
    if cached:
        m._in_proj_cache = (key, val)
    return val


class _SSDMimoCombinedFn(torch.autograd.Function):
    """The whole `SSDMimoSSM` forward as one autograd node (`mixer` is a non-tensor argument; every
    parameter that can take a gradient is passed positionally so autograd routes to it)."""

    @staticmethod
    def forward(ctx, mixer, x, *params):
        from ..layers.conv import (causal_conv1d_cl_fwd)
        m = mixer
        Bsz, L, _ = x.shape
        dt = (torch.get_autocast_dtype("cuda")
              if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype)
        with torch.amp.autocast("cuda", enabled=False):
            xa = x if x.dtype == dt else x.to(dt)
            w = m.in_proj.weight
            # x branch channel-major (what conv and scan want), as `matmul_channel_major`
            wx = w[:m.d_inner].to(dt)
            xcl = torch.bmm(wx.expand(Bsz, *wx.shape), xa.transpose(1, 2))
            tok_raw = None
            wcat = bt = None
            if m.fuse_in_proj:
                wcat, bcat, bt = _fused_in_proj(m, w, dt)
                zt = F.linear(xa, wcat, bcat)
                z, tok_raw = zt.split([m.d_inner, m.tok_width], -1)
                if bt is not None:
                    tok_raw = tok_raw + bt.to(tok_raw.dtype)
            else:
                wcat = w[m.d_inner:].to(dt)
                z = F.linear(xa, wcat)
            xcl_pre = xcl
            if m.conv1d is not None:
                xcl = causal_conv1d_cl_fwd(xcl, m.conv1d.weight,
                                           m.conv1d.bias, "silu", package=m._conv_impl)
            x_branch = xcl.transpose(1, 2)
            sel = xa if m.proj_source == "layer_input" else None
            u = m._select(x_branch, sel)
            # the output RMSNorm rides the core's gate epilogue when the core says it can
            fused_norm = getattr(m, "_fused_out_norm", None)
            y_core, tape = m._combined_core_fwd(
                x_branch, u, z, tok_raw, out_norm=fused_norm)
            rstd = nw = None
            yn = y_core
            if fused_norm is None and not isinstance(m.norm, nn.Identity):
                nw = m.norm.weight.to(y_core.dtype)
                yn, rstd = torch.ops.aten._fused_rms_norm(
                    y_core, [y_core.shape[-1]], nw, m.norm.eps)
            wo = m.out_proj.weight.to(dt)
            out = F.linear(yn, wo)
        ctx.mixer = m
        ctx.tape = tape
        ctx.amp_dtype = dt
        ctx.x_dtype = x.dtype
        ctx.saved = (xa, xcl_pre, y_core, yn, rstd, nw, wx, wcat, wo)
        ctx.needs = tuple(p.requires_grad for p in params)
        ctx.x_needs = x.requires_grad
        return out

    @staticmethod
    @_no_double_backward("SSD-MIMO")
    def backward(ctx, dout):
        from ..layers.conv import (causal_conv1d_cl_bwd)
        m = ctx.mixer
        if ctx.saved is None:
            raise RuntimeError(
                "the combined SSD-MIMO node releases its saved "
                "tensors as it consumes them -- that is what keeps its "
                "peak at the multi-node path's -- so it cannot serve a "
                "SECOND backward through the same graph.  Build the "
                "layer with kernel_options {'mode': 'modular'} "
                "if you need retain_graph=True.")
        (xa, xcl_pre, y_core, yn, rstd, nw, wx, wcat, wo) = ctx.saved
        dt, tape = ctx.amp_dtype, ctx.tape
        Bsz, L, dm = xa.shape
        di, tw = m.d_inner, m.tok_width
        grads: dict = {}

        dout = dout if dout.is_contiguous() else dout.contiguous()
        f_out = dout.reshape(-1, dout.shape[-1])
        dyn = (f_out @ wo).view(Bsz, L, wo.shape[1])
        grads["out_proj.weight"] = (f_out.t() @ yn.reshape(-1, wo.shape[1])
                                    ).to(m.out_proj.weight.dtype)
        if rstd is None:
            # no norm, or the epilogue kernel owns it (its weight gradient comes back with the core's)
            dy_core = dyn
        else:
            dy_core, dwn = torch.ops.aten._fused_rms_norm_backward(
                dyn, y_core, [y_core.shape[-1]], rstd, nw, [True, True])
            grads["norm.weight"] = dwn.to(m.norm.weight.dtype)
        # a hand-written backward must free what autograd would drop as each node finishes
        del dout, f_out, dyn, yn
        ctx.saved = None

        # the in_proj slab gradient is allocated once: the gate and token-map backward kernels write
        # into their columns directly, avoiding the split's backward `cat`
        if m.fuse_in_proj:
            dslab = torch.empty(Bsz, L, di + tw, device=xa.device, dtype=dt)
            dz_out, dtok_out = dslab[..., :di], dslab[..., di:]
        else:
            dslab = dz_out = torch.empty(Bsz, L, di, device=xa.device,
                                         dtype=dt)
            dtok_out = None
        dx_branch, du, _dtok, cgrads = m._combined_core_bwd(
            tape, dy_core, dz_out=dz_out, dslab_out=dtok_out)
        del dy_core, y_core, dz_out, dtok_out
        ctx.tape = None
        for k, v in cgrads.items():
            if k == "@out_norm_w":
                grads["norm.weight"] = v.to(m.norm.weight.dtype)
            else:
                grads[k] = v
        if du is not None:
            # main_branch: the selection input is the conv'd drive, so its GEMM gradient joins the scan's
            dx_branch = dx_branch + du.reshape(dx_branch.shape)

        dxcl = dx_branch.transpose(1, 2)                     # [B, C, L]
        if m.conv1d is not None:
            dxcl, dcw, dcb = causal_conv1d_cl_bwd(
                xcl_pre, m.conv1d.weight, m.conv1d.bias, dxcl, "silu", package=m._conv_impl)
            grads["conv1d.weight"] = dcw.to(m.conv1d.weight.dtype)
            grads["conv1d.bias"] = dcb.to(m.conv1d.bias.dtype)
        del dx_branch, xcl_pre
        dw_in = torch.empty_like(m.in_proj.weight)
        # not `torch.sum(out=dw_in[:di])`: `out=` in the parameter dtype would keep the batch reduce in
        # fp32 where `.sum(0)` rounds to bf16 first, changing in_proj.weight's gradient vs the multi-node
        # path by 1.5e-3.  `copy_` only changes where the result is written (bf16 -> fp32 is exact).
        if glue_on("dwin"):
            dw_in[:di].copy_(torch.bmm(dxcl, xa).sum(0))
        else:
            dw_in[:di] = torch.bmm(dxcl, xa).sum(0).to(dw_in.dtype)
        # the block input feeds two GEMMs: this gradient comes back token-major like the slab's so the sum streams
        dxa = torch.bmm(dxcl.transpose(1, 2), wx.expand(Bsz, *wx.shape))
        f_slab = dslab.reshape(-1, dslab.shape[-1])
        # in-place `addmm_` avoids cuBLAS copying `self` into the output; same operands and order
        dxa = dxa.reshape(-1, dm)
        if glue_on("addmm"):
            dxa.addmm_(f_slab, wcat)
        else:
            dxa = torch.addmm(dxa, f_slab, wcat)
        dxa = dxa.view(Bsz, L, dm)
        dw_cat = (f_slab.t() @ xa.reshape(-1, dm))
        dw_src = dw_cat[:di] if m.fuse_in_proj else dw_cat
        if glue_on("dwin"):
            dw_in[di:].copy_(dw_src)
        else:
            dw_in[di:] = dw_src.to(dw_in.dtype)
        grads["in_proj.weight"] = dw_in
        if m.fuse_in_proj:
            for k, v in m.scatter_tok_grads(dw_cat[di:]).items():
                grads[k] = v

        dx = dxa.to(ctx.x_dtype) if ctx.x_needs else None
        out = [None, dx]
        for (name, p), need in zip(m._combined_named_params(), ctx.needs):
            g = grads.get(name) if need else None
            if g is not None and tuple(g.shape) != tuple(p.shape):
                # per-kernel-head biases: the R = O = 1 scan hands back
                # [H, N, 1]; the parameter is [H, 1, N, 1]
                g = g.reshape(p.shape)
            out.append(g)
        return tuple(out)


# opaque to Dynamo: tracing through the captured Triton tape yields no valid graph
@dynamo_opaque
def _combined_apply(mixer, x, params):
    return _SSDMimoCombinedFn.apply(mixer, x, *params)


_LAYER_OWN = frozenset({"in_proj", "conv1d", "norm", "out_proj"})


@ssm_status("production")
class SSDMimoSSM(BaseSSM, _SSDMimoScan):
    """SSD-MIMO layer (structured state-space duality with MIMO subsystems).

    Input projections, causal drive conv, the scan and a SwiGLU output gate::

        in_proj : d_model -> d_inner + d_gate   (x branch + output gate z)
        conv1d  : depthwise causal conv on the x branch, then SiLU (d_conv > 0)
        scan    : `scan(...)` with gate_z = z (fused SwiGLU)
        norm    : optional RMSNorm on the SSM output (``out_norm``)
        out_proj: d_gate -> d_model

    The gate has H*G*O channels. MIMO applies it after sibling mixing; SISO
    keeps the fused gate epilogue. The recurrence is reached through
    ``layer.scan(...)`` / ``layer.scan_step(...)``.

    Parameters
    ----------
    config : SSDMimoConfig or dict
        The architecture.  Every field, its meaning, valid values and default are documented on
        `SSDMimoConfig`; a dict goes through `SSDMimoConfig.from_dict` (strict).  The layer width is the config's
        ``d_model`` (``d_inner`` when the config names no ``expand``); `SSDMimoConfig.build(d_model)` and
        `CausalLM` supply it otherwise.
    execution : Execution or dict, optional
        How the layer runs; see "Execution" below.  None keeps the defaults.

    Execution
    ---------
    How the layer runs is an `Execution` (or its dict), never part of the architecture: ``backend``,
    ``precision``, ``chunk_size`` and ``kernel_options`` (the keys this layer reads are
    listed at the end); ``checkpoint=True`` is refused (``False`` is what the layer does).

    * At construction: ``SSDMimoSSM(config, execution={"backend": "torch"})``.
    * At runtime: ``layer.set_execution({"precision": "ieee"})`` (only the fields given change).
    * Temporarily: ``with layer.execution({"backend": "torch"}): y = layer(x)`` (restored on exit).

    Inside a `CausalLM` or `BlockStack`, address one layer through ``lm.ssm_layers[i]``.

"""

    config_family = "ssd_mimo"

    # Execution (execution.py): `kernel_options` carries the kernel levers and the layer-level switches
    # (kernel_options.LAYER_KEYS["ssd_mimo"]: m3_kernel, compile_token_maps, allow_chunk_autotune, mode,
    # modular_checkpoint).  Changing one clears the caches that depend on it (`_tok_proj_w_cache`,
    # `_prologue_wcat_cache`, `_in_proj_cache`); the chunk autotune cache is keyed on precision/geometry.
    _layer_option_attrs = {"m3_kernel": "m3_kernel", "conv_kernel": "conv_kernel",
                           "compile_token_maps": "_compile_token_maps",
                           "allow_chunk_autotune": "allow_chunk_autotune",
                           "mode": "mode", "modular_checkpoint": "modular_checkpoint",
                           "registered_m2": "_registered_m2"}

    _execution_fields = ("backend", "precision", "chunk_size", "kernel_options")
    _execution_fixed = {"checkpoint": False}           # False is what it does; only checkpoint=True is refused
    _execution_hints = {"checkpoint": "SSD has no activation checkpointing of the layer; "
                                      "kernel_options={'modular_checkpoint': True} checkpoints the modular graph"}

    def _check_execution(self, ex):
        self._refuse_unsupported_execution(ex)
        f, out = ex.fields(), {}
        if "backend" in f:
            out["backend"] = check_backend(f["backend"], "SSDMimoSSM", BACKENDS)
        if "precision" in f:
            out["precision"] = check_precision(f["precision"], "ssd_mimo", "SSDMimoSSM")
        if "chunk_size" in f:
            out["chunk_size"] = check_chunk_size(f["chunk_size"], "SSDMimoSSM")
        kernel, lay = split_kernel_options(self, "ssd_mimo", f.get("kernel_options"))
        if kernel is not None:
            out["kernel_options"] = kernel
        if lay["m3_kernel"] == "upstream" and self._upstream_refusal:
            raise ValueError(self._upstream_refusal)
        from ..kernels.m3_adapter import check_provider
        check_provider(lay["m3_kernel"], out.get("backend", self.backend))
        if out.get("backend", self.backend) == "cuda":
            why = self._cuda_config_refusal()
            if why is not None:
                raise kernel_refusal_error("SSDMimoSSM", why)
        check_conv_kernel(lay["conv_kernel"])
        if lay["modular_checkpoint"] and lay["mode"] != "modular":
            raise ValueError("kernel_options modular_checkpoint is only meaningful with mode='modular'")
        out.update(lay)
        return out

    def _apply_execution(self, ex):
        out = self._check_execution(ex)
        changed = {k: v for k, v in out.items()
                   if k in self._layer_option_attrs and getattr(self, self._layer_option_attrs[k]) != v
                   or k not in self._layer_option_attrs and getattr(self, k) != v}
        self._set_execution_values(out)
        if self.bc_conv1d is not None:
            self.bc_conv1d.package = self._conv_impl
        if self._registered_m2:
            from ..layers import conv as _registered_conv           # noqa: F401  (registers the ops)
            from ..kernels.ssd_mimo import registered as _registered_scan   # noqa: F401
        if changed:
            self._tok_proj_w_cache.clear()
            self._prologue_wcat_cache.clear()
            self._in_proj_cache = None
            self.combined = self.mode == "fused" and (
                _os.environ.get("SSD_MIMO_COMBINED", "1") == "1")
            self._combined_refusal = "off"

    # Public names for the paper-notation attributes (read-only; kernel code keeps .H .P .N ...).
    d_ssm_in, d_ssm_out = read_alias("R"), read_alias("O")
    d_head, n_heads = read_alias("P"), read_alias("H")
    d_state, n_groups = read_alias("N"), read_alias("G")

    def __init__(self, config: "SSDMimoConfig | dict", execution=None):
        """See the class docstring: the architecture is the `SSDMimoConfig` (or its strict dict), how the
        layer runs is ``execution`` (`Execution` or dict)."""
        config = as_family_config(config, SSDMimoConfig)
        kw = config.layer_kwargs()
        kw["d_model"] = config.layer_width()
        self._build(execution=execution, **kw)

    def _build(self, d_model, d_state=64, expand=None, d_conv=4, *, d_inner=None, execution=None,
               **core_options):
        """The constructor body: the config's keywords, resolved and validated by the scan."""
        nn.Module.__init__(self)
        _ex = as_execution(execution)

        valid = signature_names(SSDMimoSSM._build, _SSDMimoScan.__init__) | {"out_norm"}
        reject_unknown_kwargs([k for k in core_options if k not in valid],
                              "SSDMimoSSM", valid=valid)
        if core_options.get("write_rule") is None:
            core_options["write_rule"] = WRITE_RULE_DEFAULT

        from ..configuration import resolve_mixer_width
        d_inner = resolve_mixer_width(d_model, expand, d_inner)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv

        # input projection (drive + output gate); default nn.Linear init, as Mamba-3's in_proj
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # depthwise causal conv on the x branch
        if d_conv > 0:
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, kernel_size=d_conv,
                groups=d_inner, padding=d_conv - 1, bias=True,
            )
        else:
            self.conv1d = None

        kw = mixer_core_options(
            core_options,
            d_inner=d_inner, d_state=d_state,
            d_model=d_model, d_conv=d_conv,
        )
        # the norm width follows the core output H*G*O
        out_norm = bool(kw.pop("out_norm", False))
        _SSDMimoScan.__init__(self, **kw)
        self.d_gate = self.d_out
        if self.d_gate != d_inner:
            self.in_proj = nn.Linear(d_model, d_inner + self.d_gate, bias=False)

        # fused=True: one ATen rms_norm launch instead of the autocast fp32 pow/mean/rsqrt chain
        self.norm = (RMSNorm(self.d_out, fused=True) if out_norm
                    else nn.Identity())
        # default nn.Linear init; inside a BlockStack the residual-out tag (`model.tag_residual_mixer`)
        # re-initialises it Kaiming-uniform / sqrt(n_residuals)
        self.out_proj = nn.Linear(self.d_out, d_model, bias=False)

        # one fused input GEMM (Mamba-2's in_proj): only under proj_source="layer_input", where the token
        # maps read the block input, so the z branch and the token stack ride one d_model -> (d_inner +
        # tok_width) GEMM.  The x branch stays separate (it wants channel-major layout); under
        # "main_branch" the maps read the post-conv drive, which does not exist yet at in_proj time.
        self.fuse_in_proj = self.proj_source == "layer_input"
        # the modular route's own "wcat" cache (in_proj's z rows catted onto `tok_proj_weight()`); keyed
        # separately from `_tok_proj_w_cache` because it also depends on `in_proj.weight`
        self._prologue_wcat_cache: dict = {}

        # `mode` / `modular_checkpoint` are execution switches (kernel_options) applied below by
        # `_apply_execution`; the combined node ('fused') falls back to the modular path outside its envelope
        self.mode = "modular"
        self.modular_checkpoint = False
        self.combined = False
        #: why the last forward did not take the combined node (None when it did, "off" when disabled)
        self._combined_refusal = "off"
        #: the same for the output norm fusion into the gate epilogue
        self._out_norm_refusal = "not run"
        self._fused_out_norm = None

        # internal opt-in (kernel_options registered_m2); the ops are registered when the option is
        # applied (`_apply_execution`), before Dynamo traces forward, never during capture
        self._registered_m2 = False
        self._registered_m2_refusal = "not run"
        # the recorded recipe: the scan's options plus the shared head
        record_constructor_config(
            self, locals(), constructor=SSDMimoSSM._build, exclude=("execution",),
            overrides={**self._resolved_constructor_config["model"], "out_norm": out_norm})
        self.set_execution(_ex)

    def _combined_named_params(self):
        """Parameters `_SSDMimoCombinedFn` takes positionally, in fixed order (the layer's, then the scan's;
        names as returned by `_combined_core_bwd`)."""
        out = [("in_proj.weight", self.in_proj.weight)]
        if self.conv1d is not None:
            out.append(("conv1d.weight", self.conv1d.weight))
            out.append(("conv1d.bias", self.conv1d.bias))
        if not isinstance(self.norm, nn.Identity):
            out.append(("norm.weight", self.norm.weight))
        out.append(("out_proj.weight", self.out_proj.weight))
        out.extend((n, p) for n, p in self.named_parameters()
                   if n.split(".")[0] not in _LAYER_OWN)
        return out

    def _resolve_out_norm(self, x_branch, *, gate_z, state, return_state):
        """`(weight, eps)` when the core's gate epilogue can compute this mixer's output RMSNorm, else None
        (then `self.norm` runs separately).  `out_norm_refusal` names the failing clause; the answer is
        recorded in `_out_norm_refusal`."""
        if isinstance(self.norm, nn.Identity):
            self._out_norm_refusal = "no output norm"
            self._fused_out_norm = None
            return None
        why = self.out_norm_refusal(
            x_branch, gate_z=gate_z, state=state, return_state=return_state)
        self._out_norm_refusal = why
        self._fused_out_norm = (None if why is not None
                                else (self.norm.weight, self.norm.eps))
        return self._fused_out_norm

    @_portable.scoped_forward
    def forward(self, x: torch.Tensor, *, state=None,
               return_state: bool = False):
        """Apply the layer to ``x`` of shape (B, L, d_model).

        ``state`` continues from a previous call (State or dict from `init_state` / `return_state`);
        ``return_state=True`` also returns the final State.  The route (registered M2, combined node,
        modular graph) is chosen per call from the execution settings; refusals are recorded in
        ``_combined_refusal`` / ``_out_norm_refusal``.
        """
        if state is not None:
            state = self.resolve_state(state, batch_size=x.shape[0], device=x.device,
                                       dtype=x.dtype)
        if getattr(self, "_registered_m2", False):
            why = self._registered_m2_why(x, state, return_state)
            self._registered_m2_refusal = why
            if why is None:
                return self._forward_registered_m2(x)
        if self.combined:
            # `gate_z=x` only says that a gate exists; the predicate reads device/dtype off the input
            why = self.combined_refusal(
                x, gate_z=x, extra_carry=(state is not None),
                return_state=return_state)
            self._combined_refusal = why
            if why is None:
                # resolved here so the node's forward only reads the answer
                self._resolve_out_norm(x, gate_z=x, state=None,
                                       return_state=False)
                return _combined_apply(
                    self, x, [p for _, p in self._combined_named_params()])
        else:
            self._combined_refusal = "off"
        return self._forward_multinode(x, state=state,
                                       return_state=return_state)

    def _registered_m2_why(self, x, state, return_state):
        """Why the registered M2 route cannot take this call (names the unsupported axis), or None."""
        from ..kernels.ssd_mimo.fused_fwd_kernel import _env
        s = self
        if self.mode != "modular" or self.modular_checkpoint:
            return "requires non-checkpointed modular mode"
        if state is not None or return_state:
            return "stateful call"
        if not x.is_cuda or s.backend not in ("cuda", "auto"):
            return "requires CUDA backend and input"
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        if dtype not in (torch.float32, torch.bfloat16):
            return "requires FP32 or BF16 IO"
        if any(p.dtype != torch.float32 for p in self.parameters()):
            return "requires FP32 parameters"
        if x.ndim != 3 or x.shape[1] < 64 or x.shape[1] % 64:
            return "requires full T64 chunks"
        if (s.N, s.G, s.R, s.O) != (128, 64, 1, 1):
            return "requires d_state=128, G=64, d_ssm_in=d_ssm_out=1"
        if (s.a_dyn or s.rot_mode is not None or s.use_trap or s.use_g_b or s.use_g_c
                or s.use_theta_h):
            return "requires static unrotated ungated maps"
        if s.bc_bias or s.b_norm or s.c_norm or not s.skip or s.proj_source != "layer_input" or not self.fuse_in_proj:
            return "requires M2 projection, skip and map configuration"
        if (self.d_conv != 4 or s.bc_conv1d is None or s.bc_conv1d.d_conv != 4
                or not self._conv_raw):
            return ("requires both width-four convolutions on a registered conv op (plain Triton or the causal_conv1d "
                    "package; not kernel_options conv_kernel='torch')")
        if isinstance(self.norm, nn.Identity) or self.norm.eps != 1e-5:
            return "requires output RMSNorm eps=1e-5"
        if s.precision != "io" or not _tail_law_is_stock():
            return "requires stock tail law and precision='io'"
        if not glue_on("bias"):
            return "requires projection bias epilogue"
        if _kernel_options.value("ssd_mimo", "backward_states_fp32", s.kernel_options):
            return "unsupported setting: kernel_options backward_states_fp32"
        for name, default, required in (
            ("SISO_FAST", "0", "1"),
            ("GATE_INKERNEL", "0", "0"), ("GATE_FUSED", "1", "1"),
            ("GATE_OUT_IO", "1", "1"),
        ):
            if _env(name, default) != required:
                return "unsupported setting: " + name
        return None

    def _forward_registered_m2(self, x):
        """Torch projections/tail surrounding one fixed scan+epilogue operation."""
        from ..layers.conv import conv_op_fwd
        from ..kernels.ssd_mimo.registered import m2_fwd
        s = self
        batch, length, _ = x.shape
        xa = autocast_cast(x)
        xcl = matmul_channel_major(self.in_proj.weight[:self.d_inner], xa,
                                   grad_layout="token")
        wt, bt = s.tok_proj_weight()
        wz = self.in_proj.weight[self.d_inner:]
        if glue_on("wcat"):
            wcat_key = param_cache_key([self.in_proj.weight] +
                                       s.tok_proj_params())
            wcat = cached_cat(self._prologue_wcat_cache, wcat_key,
                              [wz, wt], dim=0)
        else:
            wcat = torch.cat((wz, wt))
        zt = F.linear(xa, wcat, F.pad(bt.to(xa.dtype), (self.d_inner, 0)))
        z, raw = zt.split((self.d_inner, s.tok_width), -1)
        parts = s.split_tok_raw(raw)
        xcl = conv_op_fwd(xcl, self.conv1d.weight.squeeze(1), self.conv1d.bias, True, self._conv_impl)
        bcconv = s.bc_conv1d.conv
        bc = conv_op_fwd(parts["_bc"].transpose(1, 2), bcconv.weight.squeeze(1),
                         bcconv.bias, True, self._conv_impl).transpose(1, 2)
        b, c = bc.split(s.N, -1)
        # uncapped softplus law in fp32
        dt = _dt_tail(parts["dt"].float() + (parts["_dtb"].float() if parts.get("_dtb") is not None else 0.0),
                      False).transpose(1, 2).contiguous()
        # the op takes the raw chunked delta (it forms the prefix and returns dL/ddelta)
        dc = dt.reshape(batch, s.H, length // 64, 64)
        a = s.A_log.reshape(s.H).float()
        # Kc takes a contiguous [H,G,1] skip operand (an expanded view is not its ABI)
        skip = s._skip_gains().float().reshape(s.H, 64, 1).contiguous()
        y, _, _, _, _ = m2_fwd(
            c.reshape(batch, length // 64, 64, 128),
            b.reshape(batch, length // 64, 64, 128),
            xcl.reshape(batch, s.H, 64, 1, length), dt, dc, a, z, skip, self.norm.weight)
        self._combined_refusal = "off"
        self._out_norm_refusal = None
        return self.out_proj(y.permute(0, 2, 3, 1, 4, 5).reshape(batch, length, self.d_inner))

    def _forward_multinode(self, x: torch.Tensor, *, state=None,
                           return_state: bool = False):
        want_state = return_state or state is not None
        if (self.modular_checkpoint and torch.is_grad_enabled()
                and not want_state):
            from torch.utils.checkpoint import checkpoint
            y = checkpoint(self._checkpointed_modular_body, x,
                           use_reentrant=False)
            return self.out_proj(y)

        from ..layers import regen as _regen
        _sp = getattr(self, "spec", None) or {}
        _real = (self.R == 1 and self.O == 1 and not _sp.get("rotation")
                 and not any(b.get("complex") for b in _sp.get("row_blocks", ())))
        self._regen_real = _real
        _rg = (not want_state) and _regen.mode(_real) != ""
        with _regen.scope(_rg):
            return self._forward_multinode_body(x, state, want_state,
                                                return_state, _rg)

    def _forward_multinode_body(self, x, state, want_state, return_state, _rg):
        from ..layers import regen as _regen
        self._x_regen_fn = None
        xa, x_branch, z, tok_raw, conv_tail = self._forward_prologue(
            x, state=state, want_state=want_state)
        if _rg and _regen.wants("x", self._regen_real) and self._x_regen_fn is not None:
            _regen.register(x_branch, self._x_regen_fn)
        self._x_regen_fn = None

        sel = xa if self.proj_source == "layer_input" else None
        # flat schema (as init_state/step): scan State keys sit directly on the mixer state, with "conv_state"
        ssm_state = (State({k: v for k, v in state.items() if k != "conv_state"})
                    if state is not None else None)
        # output RMSNorm fused into the core's gate epilogue (see `out_norm_refusal`), on both graph shapes
        fused_norm = self._resolve_out_norm(
            x_branch, gate_z=z, state=ssm_state, return_state=want_state)
        core_result = self.scan(
            x_branch, gate_z=z, sel_x=sel, tok_raw=tok_raw,
            state=ssm_state, return_state=want_state,
            out_norm=fused_norm)
        y, out_ssm_state = core_result if want_state else (core_result, None)
        # `y` arrives token-major and contiguous, so the fused norm streams its normalized axis
        y = self.out_proj(y if fused_norm is not None else self.norm(y))
        if not return_state:
            return y
        new_state = State(out_ssm_state)
        if conv_tail is not None:
            new_state["conv_state"] = conv_tail
        new_state = self._conform_carries(new_state, state)
        self._final_state = new_state
        return y, new_state

    def _forward_prologue(self, x: torch.Tensor, *, state=None,
                          want_state: bool = False):
        """Plain-torch projection/conv prologue of modular mode (also the stateful path).  The conv
        carry keeps its gradient history under grad; detach the returned State for truncated BPTT."""
        w = self.in_proj.weight
        xa = autocast_cast(x)
        # `split_in_proj` runs the two input GEMMs as one autograd node, so their gradients (xa, the z
        # and token slab, the weight slices) are joined inside cuBLAS; forward is unchanged
        tok_raw = None
        if self.fuse_in_proj:
            # z | B | C | dt | a_dyn | ... in one GEMM off the block input
            wt, bt = self.tok_proj_weight()
            b_ep = bt if bt is not None and glue_on("bias") else None
            xcl, z, tok_raw = split_in_proj(xa, w, wt, b_ep, self.d_inner)
            if bt is not None and b_ep is None:
                tok_raw = tok_raw + bt.to(tok_raw.dtype)
        else:
            xcl, z, _ = split_in_proj(xa, w, None, None, self.d_inner)

        conv_tail = None
        if self.conv1d is not None:
            kc = self.d_conv
            # `conv_state` is step()'s (B, C, d_conv) layout: its non-oldest slots are the d_conv-1 raw
            # inputs preceding this chunk, already channel-major
            prefix = (state["conv_state"][:, :, 1:].to(xcl.dtype)
                     if state is not None else None)
            x_in = xcl if prefix is None else torch.cat([prefix, xcl], dim=-1)
            xcl = causal_depthwise_conv1d_cl(
                x_in, self.conv1d.weight, self.conv1d.bias,
                kc, activation="silu", package=self._conv_impl)
            if prefix is None and getattr(self, "_x_regen_fn", 0) is None:
                # SSD_MIMO_HELD_REGEN=x: recipe that rebuilds this conv output in the backward
                _acd = (torch.is_autocast_enabled("cuda"),
                        torch.get_autocast_dtype("cuda"))
                _cw, _cb, _pkg = self.conv1d.weight, self.conv1d.bias, self._conv_impl

                def _x_regen(x_in=x_in, _cw=_cw, _cb=_cb, kc=kc, _acd=_acd, _pkg=_pkg):
                    with torch.autocast("cuda", dtype=_acd[1], enabled=_acd[0]):
                        return causal_depthwise_conv1d_cl(
                            x_in, _cw, _cb, kc, activation="silu", package=_pkg)
                self._x_regen_fn = _x_regen
            if prefix is not None:
                # drop the prefix's outputs: the rest is the continuation (the conv is causal)
                xcl = xcl[..., prefix.shape[-1]:].contiguous()
            if want_state:
                tail = x_in[..., -kc:]
                if tail.shape[-1] < kc:          # chunk shorter than d_conv
                    tail = torch.cat(
                        [tail.new_zeros(*tail.shape[:-1],
                                        kc - tail.shape[-1]), tail], dim=-1)
                conv_tail = tail if torch.is_grad_enabled() else tail.detach().clone()
        x_branch = xcl.transpose(1, 2)                           # view
        return xa, x_branch, z, tok_raw, conv_tail

    def _checkpointed_modular_body(self, x: torch.Tensor):
        """Checkpointed region (projection/conv + scan/epilogue).  Non-reentrant checkpointing may also
        replay the scan forward; only the prologue recompute is promised."""
        xa, x_branch, z, tok_raw, _ = self._forward_prologue(x)
        sel = xa if self.proj_source == "layer_input" else None
        fused_norm = self._resolve_out_norm(
            x_branch, gate_z=z, state=None, return_state=False)
        y = self.scan(
            x_branch, gate_z=z, sel_x=sel, tok_raw=tok_raw,
            state=None, return_state=False, out_norm=fused_norm)
        return y if fused_norm is not None else self.norm(y)

    def _init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> State:
        dev = device or next(self.parameters()).device
        if dtype is None:
            dtype = self._default_state_dtype()
        state = State(self._scan_init_state(batch_size, device=dev, dtype=dtype))
        if self.conv1d is not None:
            state["conv_state"] = torch.zeros(
                batch_size, self.d_inner, self.d_conv,
                device=dev, dtype=dtype,
            )
        return state

    #: fast-inference step only: write the new carries into the given State's tensors (fixed addresses,
    #: e.g. CUDA-graph capture) and return it.  Off by default: `step` never mutates a passed State.
    step_inplace: bool = False

    @_portable.scoped_forward
    def step(
        self, x_t: torch.Tensor, state=None,
    ) -> tuple[torch.Tensor, State]:
        """Single-token step.  x_t: ``[B, d_model]``.  `state`: REQUIRED.

        Under fast inference (``torch.no_grad()`` or ``set_inference(True)``) this is
        `ssd_mimo_torch.mixer_step_fast` (one GEMM for drive, gate and token maps); with gradients, the
        differentiable composition below.
        """
        if self._fast_inference() and _torch_impl() == "fast":
            from .ssd_mimo_torch import mixer_step_fast
            return mixer_step_fast(self, x_t, state)
        state = self.resolve_state(state, batch_size=x_t.shape[0], device=x_t.device,
                                   dtype=x_t.dtype)
        ref = state                               # the returned carries keep the incoming dtypes
        xz = self.in_proj(x_t)
        x_branch, z = xz.split([self.d_inner, self.d_gate], dim=-1)

        if self.conv1d is not None:
            # the carry may be wider than the activations (fp32 under a bf16 layer): the conv output returns to
            # the activation dtype, as in the forward; the carry keeps its own
            x_branch, conv_state = causal_conv_step(x_branch, state["conv_state"], self.conv1d.weight, self.conv1d.bias,
                                                    package=self._conv_impl, out_dtype=xz.dtype)
            state = {**state, "conv_state": conv_state}

        sel = x_t if self.proj_source == "layer_input" else None
        y_t, ssm_state = self.scan_step(x_branch, {k: v for k, v in state.items() if k != "conv_state"}, gate_z=z, sel_x=sel)
        state = self._conform_carries(State({**state, **ssm_state}), ref)
        self._final_state = state
        return self.out_proj(self.norm(y_t.to(xz.dtype))), state


register_family("ssd_mimo", config=SSDMimoConfig, layer=SSDMimoSSM)

if SSDMimoSSM.__doc__:      # the allowed ``kernel_options`` keys, generated from the schema
    SSDMimoSSM.__doc__ += "\n" + "\n".join("    " + line for line in _kernel_options.describe("ssd_mimo").splitlines()) + "\n"
