"""HybridMambaSSM: routes each row block to mh_scan or ssd_mimo_scan and sums the outputs."""

# Independent state-row blocks share the drive and clock; their readouts sum.
# A/dyn/mu rows use mh_scan; m3/m2 scalar-per-head rows use SSD SISO.
# HM rows are packed first, then complex M3, real M3 and M2 rows.
# Apply the shared D skip once after summing block outputs.
from __future__ import annotations

import functools
import copy
import dataclasses
import math
import os as _os
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.hybrid_mamba_mh.layout import (
    DYN_EXP,
    DYN_RAW,
    DYN_SOFTPLUS,
    GATE_NAMES,
    LAW_A_DYN,
    LAW_A_FIX,
    LAW_MU,
    law_of,
    Frac,
    dyn_rate,
    fracs_to_rows,
    gate_act,
    sigmoid_exact,
    group_index,
    norm_fracs,
)
from ..layers.conv import (CausalDepthwiseConv, causal_conv_step, causal_depthwise_conv1d_cl, check_conv_kernel)
from ..layers.proj import (autocast_cast, matmul_channel_major, matmul_channel_major_tee, resolve_proj_source)
from ..layers.glue import (hm_glue_on)
from ..layers import regen as _regen


def hm_held_regen() -> frozenset:
    """Parse `HM_HELD_REGEN`: which held-for-backward tensors are rebuilt in
    the backward instead of kept.

    Comma list of ``x`` (post-conv drive), ``out`` (out_proj input),
    ``delta`` (softplus clock), ``dtlin`` (raw ``dt_proj(dt_low)`` output) and
    ``amag`` (heavy-tail |A| host chain); ``all`` = every item.  Unset means
    ``all``; ``off`` keeps everything.  Each rebuild
    reruns the same op on the same saved inputs, so gradients are bit-identical
    to the keep-all path.  Read per call."""
    v = _os.environ.get("HM_HELD_REGEN")
    if v is None:
        v = "all"
    v = v.strip().lower()
    if v in ("", "off", "0", "false", "no"):
        return frozenset()
    items = {t.strip() for t in v.split(",") if t.strip()}
    if "all" in items:
        return frozenset(("x", "out", "delta", "dtlin", "amag"))
    bad = items - {"x", "out", "delta", "dtlin", "amag"}
    if bad:
        raise ValueError(f"HM_HELD_REGEN: unknown item(s) {sorted(bad)}")
    return frozenset(items)
from ..layers.routes import demote_kernel_route
from ..layers.norms import RMSNorm, check_norm_kernel, gated_rms_norm, rms_norm_cast, swiglu_gate
from ..layers.state import raise_old_format, read_alias, BaseSSM, _ScanBase, ScanState, State, ssm_status
from .. import kernel_options as _kernel_options
from ..kernels import portable as _portable
from ..families import register_family
from ..execution import split_kernel_options
from .ssd_mimo import A_ACTS, _a_mag_tail
from ..layers.init import SMALL_SCALE, freq_param, init_dt_bias, no_decay, skip_param
from ..kernels.hybrid_mamba_mh.torch_scan import phase_bmm
from ..config import reject_unknown_kwargs, signature_names
from ..config_spec import refuse_per_coordinate_gates
from .. import config_spec as _config_spec
from ..validation import BACKENDS, check_backend, check_dt_range, positive_int
from ..configuration import (Configurable, resolve_geometry, resolve_dt_rank,
                             record_constructor_config,
                             mixer_core_options, as_family_config)

#: `gate_init` values: "kaiming_uniform" (nn.Linear default draw, default),
#: "zero" (B/C gate weights zero, as SSD), "kaiming_small" (the same draw
#: scaled by SMALL_SCALE).
GATE_INITS = ("zero", "kaiming_uniform", "kaiming_small")


def _check_gate_init(who, gate_init, maps):
    """``gate_init`` initialises the B/C gate projections: a non-default value with no gate would be ignored."""
    if gate_init != "kaiming_uniform" and not (maps["b_gate"] or maps["c_gate"]):
        raise ValueError(f"{who}: gate_init={gate_init!r} is set, but no B/C gate exists (b_gate and c_gate are False), so it would be "
                         "ignored; drop it or enable b_gate / c_gate")


def _init_gate_weight(lin, gate_init):
    """B/C gate projection init on a freshly drawn nn.Linear (see GATE_INITS)."""
    if gate_init == "zero":
        nn.init.zeros_(lin.weight)
    elif gate_init == "kaiming_small":
        with torch.no_grad():
            lin.weight.mul_(SMALL_SCALE)


def _flat_phase_tile(batch, heads, pairs):
    # 64 MiB per FP32 phase tensor; never split a below-budget phase stream.
    # A single SSD chunk is the irreducible native scheduling unit.
    return max(64, ((16 << 20) // (batch * heads * pairs) // 64) * 64)


def _expand_group(beta: torch.Tensor | None, H: int) -> torch.Tensor | None:
    """Expand a (G, N) group bias to (H, N) heads (G divides H), for `step()`."""
    if beta is None:
        return None
    G = beta.shape[0]
    return beta[group_index(H, G, device=beta.device)]

# Rate activations of the `dyn` law.  The scans have codes (`layout.DYN_*`)
# for raw/softplus/exp; "heavy_tail" has none: the module applies SSD's
# `_a_mag_tail` on the host and passes the RAW rate to every route.
DYN_ACTS = ("heavy_tail", "softplus", "exp", "raw")
_DYN_NAMES = {"raw": DYN_RAW, "identity": DYN_RAW,
              "softplus": DYN_SOFTPLUS, "exp": DYN_EXP}
_DYN_CODE_NAMES = {DYN_RAW: "raw", DYN_SOFTPLUS: "softplus", DYN_EXP: "exp"}

_HM_LAWS = (LAW_A_FIX, LAW_A_DYN, LAW_MU)

# Gate activations with act(b + v) = c(b) * act(v / c(b)): under gate_pos="pre"
# a gate bias is then a per-row scale.  one_plus_tanh is excluded: its range
# (0, 2) is fixed, so its bias is no row scale.
_SCALE_GATES = ("one_plus", "exp")

# Flat laws have one decay scalar per head and use SSD SISO.
# m3 selects the rate, m2 learns a static rate.
FLAT_LAWS = ("m3", "m2")

_BLOCK_KEYS = set(_config_spec.BLOCK_KEYS)

_MAX_BLOCKS = 8

# Read-offset convention stored in rotating state dicts as the persistent
# buffer `theta_h_convention`: 1 = score C^T R(Theta_t - Theta_s + theta_h) B
# (paper Sec.5, as SSDMimoSSM).  Absent = the older convention (theta_h negated): refused on
# load; the migration tooling converts such a dict.
THETA_H_CONVENTION = 1

# |A|-activation markers: persistent int64 buffers `a_act_code` (iff m3 rows)
# and `dyn_act_code` (iff dyn rows) record the activation the A-producing
# weights were trained with; a load refuses a marker that differs from the
# module's value (`_check_act_markers`).  Frozen checkpoint codes: 0-2 equal
# `layout.DYN_*`, 3 is heavy_tail.
ACT_MARKER_CODES = {"raw": 0, "softplus": 1, "exp": 2, "heavy_tail": 3}
_ACT_MARKER_NAMES = {v: k for k, v in ACT_MARKER_CODES.items()}
# field -> (marker buffer, the A-producing weight it describes, legal names)
_ACT_MARKERS = {"a_act": ("a_act_code", "m3_a_proj.weight", A_ACTS),
                "dyn_act": ("dyn_act_code", "a_dyn_proj.weight", DYN_ACTS)}

# Selection-source marker: persistent int64 buffer `proj_source_code` on every
# core records which input the token maps read; a load refuses a marker that
# differs from the module's source (`_check_proj_source_marker`).  Frozen codes.
PROJ_SOURCE_MARKER_CODES = {"main_branch": 0, "layer_input": 1}
_PROJ_SOURCE_MARKER_NAMES = {v: k for k, v in PROJ_SOURCE_MARKER_CODES.items()}


# public (spec) flat-law names -> the internal kernel-law names
_FLAT_LAW_OF = {_config_spec.HEAD_DYNAMIC: "m3", _config_spec.HEAD_STATIC: "m2"}


def _phase_law(*args, **kwargs):
    """SSD's fused phase law (`m3_epilogue.phase_law`), shared by both families.
    Under `torch.compile` it is the `orbit_ssm::phase_law_fwd`/`_bwd` custom-op
    pair (no graph break); in eager the autograd Function itself."""
    from ..kernels.ssd_mimo.m3_epilogue import phase_law
    return phase_law(*args, **kwargs)


# The mu law's bound.  log a_t gains the telescoping
# potential c (mu_t - mu_{t-1}) with mu = tanh(mu_proj sel) in (-1, 1) and
# c = G sigmoid(v) in (0, G), G = ln(mu_cap)/2: over any span s..t the
# potential's factor exp(c (mu_t - mu_s)) lies in (1/mu_cap, mu_cap), so the
# total decay product of any span is in (0, mu_cap).
MU_CAP_DEFAULT = 4.0
MU_CAP_WARN = 20.0

# Layer-level levers of one decay law (shared by every block of that law): `config_spec.check_law_levers` refuses a non-default
# value when no row block has that law (the lever would be silently ignored).
_LAW_LEVERS = [(("coord_dynamic",), {"dyn_res": "n", "dyn_act": "heavy_tail"}),
               (("coord_potential",), {"mu_res": "n", "mu_cap": MU_CAP_DEFAULT}),
               ((_config_spec.HEAD_DYNAMIC,), {"a_floor": 1e-4, "a_act": "heavy_tail"})]


def check_mu_cap(mu_cap) -> float:
    """Validate `mu_cap`: a real number > 1; warns above `MU_CAP_WARN`."""
    try:
        cap = float(mu_cap)
    except (TypeError, ValueError):
        raise ValueError(f"mu_cap must be a real number > 1, got {mu_cap!r}") from None
    if not (cap > 1.0 and math.isfinite(cap)):
        raise ValueError(
            f"mu_cap must be finite and > 1 (the mu potential's span factor "
            f"lies in (1/mu_cap, mu_cap)), got {mu_cap!r}")
    if cap > MU_CAP_WARN:
        warnings.warn(
            f"mu_cap={cap:g} > {MU_CAP_WARN:g}: the mu law allows large "
            f"transient growth -- a span's decay product (and so the state) "
            f"can amplify by up to {cap:g}x", stacklevel=3)
    return cap


def mu_gain_bound(mu_cap: float) -> float:
    """G = ln(mu_cap)/2, the supremum of the mu law's gain c."""
    return 0.5 * math.log(mu_cap)


@dataclass(frozen=True)
class CompiledFracs:
    """Row plan after packing: HM rows at [0, n_hm), then the flat tail (m3 with complex rows first, then m2)."""
    user: tuple[dict, ...]        # normalised user blocks, resolved rows
    hm: tuple[Frac, ...]          # packed HM blocks (offsets within [0, n_hm))
    n_hm: int
    n3: int                       # m3 rows total (the Mamba-3 flat law)
    n3_complex: int               # leading rotated m3 rows
    n2: int = 0                   # m2 rows (static per-head rate, SSD)
    # one B/C gate flag per `hm` block, and whether any flat block asks for a gate
    hm_gate: tuple[bool, ...] = ()
    flat_gated: bool = False

    @property
    def n_flat(self) -> int:
        return self.n3 + self.n2


def compile_rows(rows: Sequence[dict], d_state: int) -> CompiledFracs:
    """Pack a `rows` spec (canonical or not; the public decay names of
    `config_spec`) into the kernel row plan: packing, pair alignment and the
    flat power-of-two rules.  `plan.user` keeps the PUBLIC decay names."""
    # the blocks are resolved on their own here (the layer resolved them with its maps): keep every gate as given, so a gate
    # the layer built is not refused for lack of the maps, which this function does not see
    gated = any((b.gate if isinstance(b, _config_spec.RowBlock) else b.get("gate")) is not False for b in rows)
    spec = _config_spec.resolve("hybrid_mamba", d_state=d_state, row_blocks=rows, b_gate=True if gated else None)
    fracs = [dict(b, decay=_FLAT_LAW_OF.get(b["decay"], b["decay"]))
             for b in spec["row_blocks"]]
    public = [b["decay"] for b in spec["row_blocks"]]

    user: list[dict] = []
    for i, spec in enumerate(fracs):
        extra = set(spec) - _BLOCK_KEYS
        if extra:
            hint = ""
            if extra & {"rot", "rotate"}:
                hint = " ('rot' and 'rotate' are not keys; use 'complex')"
            raise ValueError(f"row_blocks[{i}]: unknown keys {sorted(extra)}{hint}")
        if "rows" in spec:
            rows = int(spec["rows"])
        elif "frac" in spec:
            rows = int(round(float(spec["frac"]) * d_state))
        else:
            raise ValueError(f"row_blocks[{i}]: needs 'rows' or 'frac'")
        if rows <= 0:
            raise ValueError(f"row_blocks[{i}]: resolved to {rows} rows")
        decay = spec.get("decay", "coord_static")
        cx = bool(spec.get("complex", False))
        if isinstance(decay, str) and decay in FLAT_LAWS:
            law = ("flat", decay)
            if cx and decay != "m3":
                raise ValueError(
                    f"row_blocks[{i}]: complex=True is served on flat rows by "
                    f"the 'head_dynamic' law only (got {public[i]!r})")
        else:
            if decay not in _config_spec.HM_PER_COORD:
                raise ValueError(
                    f"row_blocks[{i}]: decay {decay!r} is not usable here; choose "
                    f"'coord_static'/'coord_dynamic'/'coord_potential' (mh_scan, per-coordinate) or "
                    f"'head_dynamic'/'head_static' (ssd_mimo, flat rows)")
            law = law_of(decay)
            if law not in _HM_LAWS:
                raise ValueError(
                    f"row_blocks[{i}]: decay {decay!r} is not usable here; choose "
                    f"'coord_static'/'coord_dynamic'/'coord_potential' (mh_scan, per-coordinate) or "
                    f"'head_dynamic'/'head_static' (ssd_mimo, flat rows)")
        user.append({"rows": rows, "decay": decay, "complex": cx,
                     "gate": bool(spec["gate"]), "_law": law})

    total = sum(b["rows"] for b in user)
    if total != d_state:
        raise ValueError(f"row_blocks sum to {total}, need exactly "
                         f"d_state={d_state}")

    hm_specs = [b for b in user if not isinstance(b["_law"], tuple)]
    flat_gated = any(b["gate"] for b in user if isinstance(b["_law"], tuple))
    flat = {law: [b for b in user if b["_law"] == ("flat", law)]
            for law in FLAT_LAWS}
    n3_complex = sum(b["rows"] for b in flat["m3"] if b["complex"])
    n3 = sum(b["rows"] for b in flat["m3"])
    n2 = sum(b["rows"] for b in flat["m2"])
    n_hm = d_state - n3 - n2

    if n3_complex % 2:
        raise ValueError(
            f"complex head_dynamic row_blocks must come in pairs of rows, got {n3_complex} rows")
    # No width rule on a flat law's row total: each law is ONE `ssd_mimo_scan` launch whose state width N is that
    # total, and the public entry serves every N exactly (pads below 16, splits into power-of-two tiles above;
    # `kernels/ssd_mimo/autograd.py`).  The upstream M3 kernel's own requirement is refused as an option
    # (`_upstream_refusal`), not as an architecture rule.

    # Re-validate pair alignment AFTER packing: removing flat blocks shifts
    # HM offsets, so a rotating block that was even-aligned in user order can
    # land on an odd offset here.
    try:
        hm = tuple(norm_fracs(
            [{"rows": b["rows"], "decay": b["_law"], "rot": b["complex"]}
             for b in hm_specs], n_hm)) if n_hm > 0 else ()
    except ValueError as e:
        raise ValueError(
            f"row_blocks invalid after packing (flat row blocks move to the tail, so HM "
            f"offsets are counted with flat blocks removed): {e}") from e

    for b, name in zip(user, public):
        del b["_law"]
        b["decay"] = name
    return CompiledFracs(user=tuple(user), hm=hm, n_hm=n_hm,
                         n3=n3, n3_complex=n3_complex, n2=n2,
                         hm_gate=tuple(b["gate"] for b in hm_specs),
                         flat_gated=flat_gated)


def _m3_reference(Bt, Ct, v, ADT, DT, Trap, K_bias, Q_bias, angles, *,
                  S_init=None, kp_init=None, vp_init=None, phase_init=None,
                  return_state=False):
    """fp64 sequential oracle for the head_dynamic (m3) rows: the
    `backend="torch"`/`"sequential"` route and the reference for the CUDA route.

    ``S_init``/``kp_init``/``vp_init``/``phase_init`` give the native two-tap
    carry (S, k_{t-1}, v_{t-1}, cumulative phase); ``return_state=True`` also
    returns it at the end.  The law is Mamba-3's (as upstream's step kernel):

        lam_t   = sigmoid(Trap_t)                    (in-kernel activation)
        alpha_t = exp(ADT_t)
        Theta_t = cumsum_t( tanh(angles)*pi*dt )     (angle_dt.py:94)
        qt, kt  = R(Theta_t) @ (Q_t + Q_bias), R(Theta_t) @ (K_t + K_bias)
                  interleaved pairs (2j, 2j+1); pairs >= Ka
                  unrotated
        S_t = alpha_t S_{t-1} + alpha_t dt_t (1-lam_t) v_{t-1} k_{t-1}^T
                              + lam_t dt_t v_t k_t^T          (trapezoid)
        y_t = S_t q_t

    Shapes: Bt/Ct (B,L,1,n3) post-norm pre-bias, v (B,L,H,P),
    ADT/DT/Trap (B,H,L) with Trap RAW (pre-sigmoid), K/Q_bias (H,n3),
    angles (B,L,H,Ka). Returns (B,L,H,P) fp64.
    """
    Bt, Ct, v = Bt.double(), Ct.double(), v.double()
    ADT, DT = ADT.double(), DT.double(),
    lam = torch.sigmoid(Trap.double())                          # (B,H,L)
    alpha = torch.exp(ADT)
    dt_blh = DT.transpose(1, 2)                                 # (B,L,H)
    phase0 = (phase_init.double().unsqueeze(1) if phase_init is not None
             else 0.0)
    theta = phase0 + torch.cumsum(
        torch.tanh(angles.double()) * math.pi * dt_blh.unsqueeze(-1), dim=1)

    k = Bt + K_bias.double().unsqueeze(0).unsqueeze(0)          # (B,L,H,n3)
    q = Ct + Q_bias.double().unsqueeze(0).unsqueeze(0)
    Ka = angles.shape[-1]
    cos, sin = torch.cos(theta), torch.sin(theta)               # (B,L,H,Ka)

    def rot(t):
        e, o = t[..., 0::2], t[..., 1::2]
        er = torch.cat([e[..., :Ka] * cos - o[..., :Ka] * sin,
                        e[..., Ka:]], dim=-1)
        og = torch.cat([e[..., :Ka] * sin + o[..., :Ka] * cos,
                        o[..., Ka:]], dim=-1)
        return torch.stack([er, og], dim=-1).flatten(-2)

    k, q = rot(k), rot(q)

    Bsz, L, H, P = v.shape
    n3 = k.shape[-1]
    S = v.new_zeros(Bsz, H, P, n3) if S_init is None else S_init.double().clone()
    kp = k.new_zeros(Bsz, H, n3) if kp_init is None else kp_init.double().clone()
    vp = v.new_zeros(Bsz, H, P) if vp_init is None else vp_init.double().clone()
    ys = []
    for t in range(L):
        a = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)          # (B,H,1,1)
        d = DT[:, :, t].unsqueeze(-1).unsqueeze(-1)
        lm = lam[:, :, t].unsqueeze(-1).unsqueeze(-1)
        kt, vt = k[:, t], v[:, t]                               # (B,H,n3)/(B,H,P)
        S = (a * S
             + a * d * (1 - lm) * vp.unsqueeze(-1) * kp.unsqueeze(-2)
             + lm * d * vt.unsqueeze(-1) * kt.unsqueeze(-2))
        kp, vp = kt, vt
        ys.append(torch.einsum("bhpn,bhn->bhp", S, q[:, t]))
    Y = torch.stack(ys, dim=1)                                  # (B,L,H,P)
    if not return_state:
        return Y
    state = dict(S=S, kp=kp, vp=vp, phase=theta[:, -1])
    return Y, state


def _flat_reference(Bt, Ct, v, log_a, write, B_bias, C_bias, *,
                    S_init=None, return_state=False):
    """fp64 sequential oracle for the m2 flat rows, and their
    `backend="torch"`/`"sequential"` route; an oracle for the direct step:
        S_t = exp(log_a_t) S_{t-1} + write_t v_t (B_t + B_bias)^T
        y_t = S_t (C_t + C_bias)
    Shapes: Bt/Ct (B,L,n) post-norm pre-bias, v (B,L,H,P), log_a/write
    (B,H,L), biases (H,n).  Returns (B,L,H,P) fp64 (+ the carry)."""
    Bt, Ct, v = Bt.double(), Ct.double(), v.double()
    a = torch.exp(log_a.double())                               # (B,H,L)
    w = write.double()
    k = Bt.unsqueeze(2) + B_bias.double()                       # (B,L,H,n)
    q = Ct.unsqueeze(2) + C_bias.double()
    Bsz, L, H, P = v.shape
    n = k.shape[-1]
    S = v.new_zeros(Bsz, H, P, n) if S_init is None else S_init.double().clone()
    ys = []
    for t in range(L):
        S = (a[:, :, t, None, None] * S
             + w[:, :, t, None, None] * v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2))
        ys.append(torch.einsum("bhpn,bhn->bhp", S, q[:, t]))
    Y = torch.stack(ys, dim=1)
    return (Y, S) if return_state else Y


# host-buffer value functions are module-level (a lambda would make the layer unpicklable: torch.save)
def _theta_h_convention_code():
    return THETA_H_CONVENTION


def _act_marker_code(layer, field):
    return ACT_MARKER_CODES[getattr(layer, field)]


def _proj_source_code(layer):
    return PROJ_SOURCE_MARKER_CODES[layer.proj_source]


class _HybridMambaScan(Configurable, _ScanBase):
    """Private scan core of `HybridMambaSSM` (reached as ``scan`` / ``scan_step``).

    Owns the row-block dispatch (``mh_scan`` for the A/dyn/mu rows, one
    ``ssd_mimo_scan`` launch per flat law), the shared per-head step size and
    the checkpoint markers.  The public layer adds in_proj, drive conv, gate,
    norm and out_proj; constructor options are documented on `HybridMambaSSM`
    and `HybridMambaConfig`.  The ``mu`` law is
    ``log a_t = -exp(A_log) Delta_t + c (mu_t - mu_{t-1})`` with
    ``mu = tanh(mu_proj(sel))`` and ``c = G sigmoid(mu_gain_logit)``,
    ``G = ln(mu_cap) / 2``.  Backend ``"cuda"`` uses the fused kernels,
    ``"torch"`` bounded affine-prefix chunks, ``"sequential"`` direct steps;
    ``"auto"`` picks CUDA for GPU input.
    """

    supports_fused_scan_gate = False
    external_value_dim = 0
    bc_from_residual = False
    config_family = "hybrid_mamba"

    def __init__(self, d_inner: int, d_state: int, *,
                 n_heads: int | None = None,
                 d_head: int | None = None,
                 # row_blocks / rotation / maps spec; row_blocks is required
                 row_blocks: Sequence | None = None,
                 # None resolves to "layer_input"; the recorded recipe holds the resolved value.
                 proj_source: str | None = None,
                 d_model: int | None = None,
                 bc_conv: bool | None = None,
                 d_conv: int = 4,
                 dt_rank: int | str | None = "adaptive",
                 force_dt_dense: bool = False,
                 dt_init: str | None = "random", dt_scale: float = 1.0,
                 dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4,
                 dyn_res: str = "n",
                 # a `layout.DYN_*` int code is accepted too
                 dyn_act: str = "heavy_tail",
                 mu_res: str = "n",
                 mu_cap: float = MU_CAP_DEFAULT,
                 a_floor: float = 1e-4,
                 a_act: Literal["heavy_tail", "softplus"] = "heavy_tail",
                 norm_eps: float = 1e-5,
                 skip: bool = True,
                 skip_sharing: str = "head",
                 use_trap: bool | None = None,
                 learnable_h_init: bool = False,
                 # init of gB_proj/gC_proj, see GATE_INITS
                 gate_init: str = "kaiming_uniform",
                 # rotation and B/C maps (config_spec; recorded resolved).
                 # theta_per_head: theta_proj emits H*K instead of K angle features.
                 # bc_gate_pos "pre" = R(Theta)[act(g) (.) B^ + beta], "post" = in-scan gate after the rotation.
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
                 bc_gate_act: str | None = None,
                 bc_gate_bias: bool | None = None,
                 bc_gate_pos: str | None = None) -> None:
        if "_parameters" not in self.__dict__:      # HybridMambaSSM initializes the Module first
            super().__init__()
        # How the layer RUNS (`Execution`): the defaults here, the caller's values
        # arrive through `_apply_execution` once the layer is built.
        self.kernel_options = {}
        self.backend, self.ckpt, self.m3_kernel, self.mem = "auto", True, "native", None
        self.conv_kernel = "auto"
        d_inner = positive_int(d_inner, "d_inner")
        d_state = positive_int(d_state, "d_state")
        n_heads, _P = resolve_geometry(d_inner, n_heads=n_heads, d_head=d_head)
        if dyn_res not in ("n", "h") or mu_res not in ("n", "h"):
            raise ValueError(f"dyn_res/mu_res must be 'n' or 'h', got "
                             f"{dyn_res!r}/{mu_res!r}")
        self.mu_cap = check_mu_cap(mu_cap)

        self.d_inner = d_inner
        self.d_state = d_state
        self.H = n_heads
        self.P = d_inner // n_heads
        # width-preserving scan (unlike SSD-MIMO's o-tiled out_width)
        self.out_width = d_inner
        self.a_floor = float(a_floor)
        if gate_init not in GATE_INITS:
            raise ValueError(f"gate_init must be one of {GATE_INITS}, got {gate_init!r}")
        self.gate_init = gate_init
        # both activations are properties: they keep the scans' code, the
        # state-dict marker and the recorded recipe in step
        self.a_act = a_act
        self.dyn_act = dyn_act
        self.dyn_res = dyn_res
        self.mu_res = mu_res
        # ── the row_blocks / rotation / maps spec ───
        spec = _config_spec.resolve(
            "hybrid_mamba", d_state=d_state, **_config_spec.spec_kwargs(locals()))
        self.spec = spec
        _check_gate_init("HybridMambaSSM", gate_init, spec["maps"])
        _config_spec.check_law_levers("HybridMambaSSM", spec["row_blocks"], _LAW_LEVERS, dict(
            dyn_res=dyn_res, dyn_act=dyn_act, mu_res=mu_res, mu_cap=mu_cap, a_floor=a_floor, a_act=a_act))
        rot = spec["rotation"] or {}
        mp = spec["maps"]
        learn_theta_freq = bool(rot.get("learn_theta_freq", False))
        use_theta_h = bool(rot.get("use_theta_h", False))
        theta_per_head = bool(rot.get("theta_per_head", False))
        for key, default in (("omega_layout", "pair"), ("theta_init", "kaiming")):
            if rot.get(key, default) != default:
                # SSD-MIMO options; HM's angle maps keep their init -- refused by name
                raise NotImplementedError(
                    f"HybridMambaSSM: {key}={rot[key]!r} is not "
                    "implemented (SSD-MIMO only)")
        gates_b, gates_c = bool(mp["b_gate"]), bool(mp["c_gate"])
        gate_act, gate_bias, gate_pos = mp["bc_gate_act"], mp["bc_gate_bias"], mp["bc_gate_pos"]
        b_norm, c_norm = bool(mp["b_norm"]), bool(mp["c_norm"])
        self.bias_gran = mp["bc_bias"]
        self._gate_act = GATE_NAMES[gate_act]
        self.gate_pos = gate_pos

        plan = compile_rows(spec["row_blocks"], d_state)
        # internal row plan (flat laws under their kernel names m3/m2)
        self.fracs = plan
        self.d_state_hm = plan.n_hm
        self.d_state_m3 = plan.n3
        #: why `kernel_options["m3_kernel"] = "upstream"` is refused for this layer (None: it is not)
        self._upstream_refusal = (
            None if plan.n3 in (16, 32, 64, 128) and self.P in (16, 32, 64, 128)
            else "upstream M3 requires the head_dynamic d_state and d_head in {16,32,64,128}")
        self.d_state_m2 = plan.n2
        n_hm = plan.n_hm
        laws = {f.law for f in plan.hm}
        has_rot = any(f.rot for f in plan.hm)
        if has_rot and n_hm % 2:
            raise ValueError(f"HM rotation needs even N_hm, got {n_hm}")

        # Host-derived index buffers are built from Python lists (works on meta)
        # and refilled by `_apply`, since to_empty leaves them uninitialised.
        self._host_buffers = {}
        _law, _rot = [], []
        for f in plan.hm:
            _law += [int(f.law)] * f.rows
            _rot += [bool(f.rot)] * f.rows
        assert len(_law) == n_hm
        self._register_host_buffer("row_law", _law, torch.int32)
        self._register_host_buffer("row_rot", _rot, torch.bool)

        # Flat rows serve only SSD's default maps (B/C RMSNorm, head biases,
        # no gates); anything else is refused by name, never silently mapped.
        if plan.n_flat:
            gaps = [name for name, bad in (
                ("b_norm=False", not b_norm), ("c_norm=False", not c_norm),
                (f"bc_bias={self.bias_gran!r}", self.bias_gran != "head"),
                # only a gated flat block needs kernel work
                ("b_gate on a gated flat block", gates_b and plan.flat_gated),
                ("c_gate on a gated flat block", gates_c and plan.flat_gated),
                ("learn_theta_freq", plan.n3_complex and learn_theta_freq),
                ("use_theta_h", plan.n3_complex and use_theta_h),
                ("theta_per_head", plan.n3_complex and theta_per_head)) if bad]
            if gaps:
                raise NotImplementedError(
                    f"HybridMambaSSM: {', '.join(gaps)} on the flat "
                    "head_dynamic/head_static rows is not yet implemented "
                    "(they serve b_norm=c_norm=True, bc_bias='head', no gates -- "
                    "set gate=False on their row blocks --, and "
                    "complex flat rows rotate at the fixed frequency pi with no "
                    "read offset).  The rotation is ONE global setting, so it "
                    "is never dropped silently for the flat rows: pass "
                    "learn_theta_freq=False, use_theta_h=False "
                    "explicitly (this also applies to complex HM rows in the "
                    "same core), or make the flat rows real")
        if n_hm and gate_bias and gate_pos == "pre" and gate_act in _SCALE_GATES:
            # under "pre" a scale-factorising gate's bias is one per-row factor,
            # redundant with the B/C norm weight (or the bc_proj row scale)
            raise ValueError(
                f"HybridMambaSSM: bc_gate_bias=True with bc_gate_pos='pre' and "
                f"bc_gate_act={gate_act!r} is a redundant parametrisation (the "
                f"bias factors into the B/C norm weight); drop bc_gate_bias, or "
                f"use bc_gate_pos='post' or a non-factorising bc_gate_act")
        if n_hm and (gates_b or gates_c) and gate_act == "one_plus_tanh" and gate_pos != "pre":
            # 1 + tanh(g) is a host-side fold of the pre-rotation map only;
            # the in-scan (post) gate kernels have no GATE_ACT for it
            raise NotImplementedError(
                "HybridMambaSSM: bc_gate_act='one_plus_tanh' with bc_gate_pos='post' is "
                "not implemented (host-side pre-rotation gate only); use bc_gate_pos='pre'")
        if n_hm and gates_b != gates_c:
            # the gate_pos paths key the pre-rotation fold on `gB_proj`: one switch
            raise NotImplementedError(
                "HybridMambaSSM: separate B/C gates (b_gate != c_gate) are not "
                "yet implemented on the HM rows; set b_gate == c_gate")
        # which structures exist on the HM (mh_scan) rows
        bc_bias = self.bias_gran != "none" and n_hm > 0
        # gates exist on the HM rows only if some HM block is gated
        hm_gated = n_hm > 0 and any(plan.hm_gate)
        gates_b, gates_c = gates_b and hm_gated, gates_c and hm_gated
        if ((gates_b or gates_c) and gate_pos == "post"
                and not all(plan.hm_gate) and gate_act not in _SCALE_GATES):
            # a gate-off block reads a zero pre-activation in the in-scan gate,
            # so it needs act(0) == 1 ("pre" masks after the act)
            raise NotImplementedError(
                f"HybridMambaSSM: a gate=False row block with bc_gate_pos='post' "
                f"needs bc_gate_act in {_SCALE_GATES} (act(0) == 1), got {gate_act!r}")
        b_norm_hm, c_norm_hm = b_norm and n_hm > 0, c_norm and n_hm > 0

        self.proj_source, d_sel, self.bc_conv = resolve_proj_source(
            "layer_input" if proj_source is None else proj_source,
            d_inner, d_model, bc_conv, who="HybridMambaSSM", d_conv=d_conv)
        self.d_sel = d_sel

        # --- shared per-head delta ------------------------------------------
        # dt_rank=None: Linear(d_sel, H); dt_rank=r: bottleneck d_sel -> r -> H
        # (n_heads = d_inner gives Mamba-1's per-channel dt).
        # dt_bias: softplus(dt_bias) starts log-uniform in [dt_min, dt_max].
        dt_rank = resolve_dt_rank(dt_rank, d_sel=d_sel, heads=self.H,
                                  d_model=d_model, force_dt_dense=force_dt_dense)
        if dt_init not in (None, "random", "constant"):
            raise ValueError("dt_init must be None, 'random', or 'constant'")
        if not math.isfinite(dt_scale) or dt_scale <= 0:
            raise ValueError("dt_scale must be finite and positive")
        if dt_init is None and dt_scale != 1.0:
            raise ValueError("dt_scale requires explicit dt_init='random' or 'constant'")
        check_dt_range(dt_min, dt_max, dt_init_floor)
        self.dt_init, self.dt_scale = dt_init, float(dt_scale)
        self.dt_rank = None if dt_rank is None else int(dt_rank)
        if self.dt_rank is not None:
            if self.dt_rank <= 0:
                raise ValueError(f"dt_rank must be positive, got {dt_rank}")
            self.dt_low = nn.Linear(d_sel, self.dt_rank, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.H, bias=False)
        else:
            # a module slot keeps the constructor's parameter order across source switches
            self.register_module("dt_low", None)
            self.dt_proj = nn.Linear(d_sel, self.H, bias=False)
        if dt_init is not None:
            std = self.dt_proj.in_features ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, std)
            else:
                nn.init.uniform_(self.dt_proj.weight, -std, std)
        self.dt_bias = no_decay(nn.Parameter(init_dt_bias(self.H, dt_min, dt_max, dt_init_floor)))

        # --- shared B/C projection; columns [0,n_hm) HM, [n_hm,N) const -----
        self.bc_proj = nn.Linear(d_sel, 2 * d_state, bias=False)
        # Mamba-2/3's xBC conv (layer_input source only); a slot either way
        self.register_module("bc_conv1d", CausalDepthwiseConv(2 * d_state, d_conv)
                             if self.bc_conv else None)

        # --- HM half ---------------------------------------------------------
        # bias groups: maps.bias "head" = one bias per head (G = H),
        # "shared" = one vector (G = 1).  Gate biases: head-shared under gate_pos="pre" (the only
        # form that folds onto the head-shared maps), per head under "post".
        G = self.H if self.bias_gran == "head" else 1
        G_gate = 1 if gate_pos == "pre" else self.H
        self.b_norm = nn.RMSNorm(n_hm, eps=norm_eps) if b_norm_hm else None
        self.c_norm = nn.RMSNorm(n_hm, eps=norm_eps) if c_norm_hm else None
        self.beta_b = nn.Parameter(torch.zeros(G, n_hm)) if bc_bias else None
        self.beta_c = nn.Parameter(torch.zeros(G, n_hm)) if bc_bias else None

        # ── law-block row stores ──
        # Each decay-side per-row quantity is stored only on the rows of the law
        # that reads it, once per rotating pair (a pair is one complex row, one
        # modulus), and gathered to mh_scan's n_hm row axis on the host
        # (`_rows_to_kernel`).  A row of another law reads an exact zero.
        #   A_log   (H,P,n_A)    laws A and mu (dyn rows: c1 = -1)
        #   a_dyn   (B,n_dyn,L)  a_dyn_proj rows, dyn law (dyn_res="n")
        #   mu      (B,n_mu,L)   mu_proj rows, mu law (mu_res="n")
        #   mu_gain_logit (H,P,n_mu)  the mu law's bounded gain c = G sigmoid(.)
        _rep = [n - 1 if (_rot[n] and n % 2) else n for n in range(n_hm)]

        def _store(law_set):
            rows = [n for n in range(n_hm) if _law[n] in law_set and _rep[n] == n]
            col = {n: i for i, n in enumerate(rows)}
            idx = [col.get(_rep[n], len(rows)) for n in range(n_hm)]
            if idx == list(range(n_hm)):
                return rows, None                  # identity: no gather
            return rows, idx

        self.has_rot = has_rot
        a_rows, a_idx = _store((LAW_A_FIX, LAW_MU))
        self._register_host_buffer("a_log_idx", a_idx, torch.long)

        # A_log: Mamba-1's S4D-real init (A = 1..N per state row), restarting in
        # each A/mu block; a complex block's ladder runs over its pairs (1 .. rows/2).
        ladder, off = [], 0
        for f in plan.hm:
            if f.law in (LAW_A_FIX, LAW_MU):
                ladder += list(range(1, (f.rows // 2 if f.rot else f.rows) + 1))
            off += f.rows
        assert len(ladder) == len(a_rows)
        if a_rows:
            A = torch.tensor(ladder, dtype=torch.float32).repeat(self.H, self.P, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
        else:
            self.A_log = None                      # all-dyn HM rows

        # decay-law inputs (raw projections; the dyn rate's activation is
        # hoisted by mh_scan, or applied by the module under heavy_tail)
        if LAW_A_DYN in laws:
            dyn_rows, dyn_idx = _store((LAW_A_DYN,))
            self.a_dyn_proj = nn.Linear(
                d_sel, len(dyn_rows) if dyn_res == "n" else self.H, bias=False)
        else:
            dyn_idx, self.a_dyn_proj = None, None
        self._register_host_buffer("dyn_idx", dyn_idx if dyn_res == "n" else None,
                                   torch.long)
        # Weight decay applies to the A-producing Linears (a_dyn_proj, mu_proj,
        # m3_a_proj), as in Mamba-3; the static A_log /
        # mu_gain_logit stay exempt.
        if LAW_MU in laws:
            mu_rows, mu_idx = _store((LAW_MU,))
            self.n_mu = len(mu_rows)
            self.mu_proj = nn.Linear(
                d_sel, self.n_mu if mu_res == "n" else self.H, bias=False)
            # bounded potential: mu = tanh(mu_proj sel), c = G sigmoid(mu_gain_logit),
            # one logit per channel and mu row.  It telescopes, so a span
            # contributes c (mu_t - mu_s) in (-2G, 2G).  Zero init: c = G/2.
            self.mu_gain_logit = nn.Parameter(torch.zeros(self.H, self.P, self.n_mu))
            self.mu_gain_logit._no_weight_decay = True
        else:
            mu_idx, self.n_mu = None, 0
            self.mu_proj = None
            self.mu_gain_logit = None
        self._register_host_buffer("mu_idx", mu_idx, torch.long)
        # rotation: SSD's single phase law, as SSDMimoSSM at R = 1:
        #   inc_t[h,k] = Delta_t[h] * omega_k * tanh(q_t[k]),  q = theta_proj(sel),
        #   omega_k = pi * tanh(theta_freq_k) (init 0.999 pi).  Theta is cumulated
        #   in-kernel (`theta_delta`); B is rotated by Theta, C by Theta - theta_h
        #   (static read offset, zero init); see `_theta_h_kernel` for the sign.
        # Every rotation quantity (angle projection, theta_freq, theta_h, the phase
        # carry State["theta"]) has one entry per ROTATED pair, not n_hm/2;
        # `_pairs_to_kernel` scatters them into mh_scan's K = n_hm/2 pair axis.
        self.learn_theta_freq = bool(learn_theta_freq) and has_rot
        self.use_theta_h = bool(use_theta_h) and has_rot
        self.theta_per_head = bool(theta_per_head) and has_rot
        if has_rot:
            # host-side list (no data-dependent op on meta tensors)
            rot_pairs = [k for k, r in enumerate(_rot[0::2]) if r]
            self.n_rot_pairs = len(rot_pairs)
            self._register_host_buffer("rot_pair_idx", rot_pairs, torch.long)
            n_rot = self.n_rot_pairs
            # PyTorch's default init, as SSD's theta_proj
            self.theta_proj = nn.Linear(
                d_sel, (self.H if self.theta_per_head else 1) * n_rot, bias=False)
            if self.learn_theta_freq:
                self.theta_freq = freq_param((n_rot,))
            if self.use_theta_h:
                self.theta_h = nn.Parameter(torch.zeros(self.H, n_rot))
                self.theta_h._no_weight_decay = True
                # state dicts declare the read-offset convention (`_check_theta_h_convention`)
                self._register_host_buffer(
                    "theta_h_convention", _theta_h_convention_code,
                    torch.int64, persistent=True)
                self._register_load_state_dict_pre_hook(
                    self._check_theta_h_convention)
        else:
            self.n_rot_pairs = 0
            self.rot_pair_idx = None
            self.theta_proj = None

        # Trapezoid forcing on the HM rows: lambda_t = sigmoid(trap_proj sel + b)
        # per head (zero bias: lambda ~ 1/2 at init); `step` carries the previous
        # unscaled write (`write_prev`).  HM rows only: the m3 rows have their own
        # `m3_trap_proj`, the m2 rows are exponential-Euler, so all-flat resolves to False.
        if use_trap is not None and n_hm == 0:
            raise ValueError(
                "HybridMambaSSM: write_rule is set, but every row block is a flat law (head_dynamic / head_static), whose write "
                "is fixed by the law (head_dynamic trapezoid, head_static euler), so it would be ignored; drop write_rule "
                "or add coord_* rows")
        self.use_trap = (use_trap is None or bool(use_trap)) and n_hm > 0
        if self.use_trap:
            self.trap_proj = nn.Linear(d_sel, self.H, bias=True)
            nn.init.zeros_(self.trap_proj.bias)
        else:
            self.trap_proj = None

        # B/C gates.  A rotating pair is one complex number, so its gate is a
        # complex scalar (one factor on both coordinates), stored once per pair
        # like the decay stores; real rows keep per-coordinate gates.  Each row
        # block's `gate` field selects the gated rows; gB_proj/gC_proj and the
        # gate biases carry n_gate compact rows, gathered to the n_hm row axis on
        # the host (`_gate_maps`, `_gate_bias`).  A gate-off row reads the
        # sentinel column n_gate (exact zero), masked to factor 1 (`_pre_gate`).
        self.gate_pair_tied = bool(has_rot)
        _gated = []
        for f, on in zip(plan.hm, plan.hm_gate):
            _gated += [on] * f.rows
        gate_rows = [n for n in range(n_hm) if _gated[n] and _rep[n] == n]
        _col = {n: i for i, n in enumerate(gate_rows)}
        n_gate = len(gate_rows)
        _gidx = [_col[_rep[n]] if _gated[n] else n_gate for n in range(n_hm)]
        self.n_gate_rows = n_gate
        self.gate_masked = not all(_gated)
        self._register_host_buffer(
            "gate_tie_idx", None if _gidx == list(range(n_hm)) else _gidx, torch.long)
        self._register_host_buffer(
            "gate_row_on", _gated if self.gate_masked else None, torch.bool)
        self._register_load_state_dict_pre_hook(
            self._refuse_per_coordinate_complex_gates)
        self.gB_proj = nn.Linear(d_sel, n_gate, bias=False) if gates_b else None
        self.gC_proj = nn.Linear(d_sel, n_gate, bias=False) if gates_c else None
        # gate_init: see GATE_INITS ("zero" starts the (1+g) gate at exactly 1)
        for _g in (self.gB_proj, self.gC_proj):
            if _g is not None:
                _init_gate_weight(_g, gate_init)
        self.beta_gB = (nn.Parameter(torch.zeros(G_gate, n_gate))
                        if gates_b and gate_bias else None)
        self.beta_gC = (nn.Parameter(torch.zeros(G_gate, n_gate))
                        if gates_c and gate_bias else None)

        # --- const half (m3 law, ssd_mimo kernel) ----------------------------
        # `m3_` names the law (Mamba-3's); the parameter names are frozen by the LM state-dict manifest.
        n3, n3c = plan.n3, plan.n3_complex
        if n3 > 0:
            # as SSD's `a_dyn_proj` / `trap_proj`: Kaiming-uniform weight, zero bias
            self.m3_a_proj = nn.Linear(d_sel, self.H, bias=True)
            self.m3_trap_proj = nn.Linear(d_sel, self.H, bias=True)
            nn.init.zeros_(self.m3_a_proj.bias)
            nn.init.zeros_(self.m3_trap_proj.bias)
            self.m3_a_proj.bias._no_weight_decay = True
            self.m3_angle_proj = (nn.Linear(d_sel, n3c // 2, bias=False)
                                  if n3c > 0 else None)
            # per-head additive B/C biases (M3's K_bias/Q_bias), ones init; they
            # reach `ssd_mimo_scan` as beta/gamma with bias_pre_rot (before the rotation)
            self.m3_B_bias = nn.Parameter(torch.ones(self.H, n3))
            self.m3_C_bias = nn.Parameter(torch.ones(self.H, n3))
            # the decay is fully data-dependent (`_m3_rate`), so the kernel's static
            # per-head rate is a = -1 and the whole exponent rides `delta`.  Non-persistent.
            self._register_host_buffer("const_a_log", [0.0] * self.H,
                                       torch.float32)
            self.m3_b_norm = nn.RMSNorm(n3, eps=1e-5)
            self.m3_c_norm = nn.RMSNorm(n3, eps=1e-5)
        else:
            self.m3_a_proj = None
            self.m3_trap_proj = None
            self.m3_angle_proj = None
            self.m3_B_bias = None
            self.m3_C_bias = None
            self.m3_b_norm = None
            self.m3_c_norm = None

        # --- m2 flat rows: one scalar rate per head, exponential-Euler write on
        # the shared Delta_t, per-head B/C biases (ones init) behind their own norms.
        #   log a_t = -Delta_t exp(A_log), exp(A_log) log-spaced over heads in [0.05, 2]
        for law, n in (("m2", plan.n2),):
            if n > 0:
                a_log = torch.linspace(math.log(0.05), math.log(2.0), self.H)
                p_a = nn.Parameter(a_log)
                p_a._no_weight_decay = True
                setattr(self, f"{law}_A_log", p_a)
                setattr(self, f"{law}_B_bias", nn.Parameter(torch.ones(self.H, n)))
                setattr(self, f"{law}_C_bias", nn.Parameter(torch.ones(self.H, n)))
                setattr(self, f"{law}_b_norm", nn.RMSNorm(n, eps=1e-5))
                setattr(self, f"{law}_c_norm", nn.RMSNorm(n, eps=1e-5))
            else:
                for nm in ("A_log", "B_bias", "C_bias", "b_norm", "c_norm"):
                    setattr(self, f"{law}_{nm}", None)

        # |A|-activation markers, only where such rows exist (`ACT_MARKER_CODES`)
        markers = False
        for field, (buf, weight, _legal) in _ACT_MARKERS.items():
            if getattr(self, weight.split(".")[0]) is not None:
                self._register_host_buffer(
                    buf, functools.partial(_act_marker_code, self, field),
                    torch.int64, persistent=True)
                markers = True
        if markers:
            self._register_load_state_dict_pre_hook(self._check_act_markers)
        # selection-source marker (`PROJ_SOURCE_MARKER_CODES`), on every core
        self._register_host_buffer(
            "proj_source_code",
            functools.partial(_proj_source_code, self),
            torch.int64, persistent=True)
        self._register_load_state_dict_pre_hook(self._check_proj_source_marker)

        # SHARED, applied once on the summed output.
        if skip_sharing not in ("head", "channel"):
            raise ValueError("skip_sharing must be 'head' or 'channel'")
        self.skip_sharing = skip_sharing
        self.skip = bool(skip)
        self.D = (skip_param((self.H,) if skip_sharing == "head" else (self.H, self.P))
                  if self.skip else None)
        # Per-call fusion flags prevent applying skip or gate twice.
        self._k11_skip_in_kernel = False
        # set by every `_scan_cf`; False until one runs
        self._h3_gate_in_kernel = False

        self.h_init_param = (
            nn.Parameter(torch.zeros(self.H, self.P, self.d_state))
            if learnable_h_init else None)
        record_constructor_config(
            self, locals(), constructor=_HybridMambaScan.__init__,
            overrides=dict(kernel_options={}, backend="auto", ckpt=True, m3_kernel="native", mem=None,
                           bc_conv=self.bc_conv, proj_source=self.proj_source,
                           use_trap=self.use_trap if n_hm > 0 else None, **_config_spec.flat_spec(spec)),
            drop=("kernel_options", "backend", "ckpt", "m3_kernel", "mem"))
        if self.mu_proj is None:
            # `mu_cap` belongs to the mu law: cores without mu rows do not record it
            self._resolved_constructor_config["model"].pop("mu_cap")

    # Pair structure in complex mode: a rotating pair (2k, 2k+1) is ONE complex
    # state row.  Tied per pair (one number): the decay stores (`A_log`,
    # `mu_gain_logit`, the n-resolved `a_dyn`/`mu` streams; `_rows_to_kernel`
    # repeats it on both coordinates), the rotation angle quantities, and the
    # gates `g_B`/`g_C` with biases `beta_gB`/`beta_gC`.  Not tied (B and C are
    # complex vectors): the additive biases `beta_b`/`beta_c` and the flat rows'
    # `m3_B_bias`/`m3_C_bias`.

    def theta_omega(self) -> torch.Tensor:
        """Per-pair angular frequency omega_k = pi * tanh(theta_freq_k)
        (constant pi when the frequency is not learned); the same map as
        `SSDMimoSSM.theta_omega`."""
        if self.learn_theta_freq:
            return math.pi * torch.tanh(self.theta_freq)
        return torch.full((self.n_rot_pairs,), math.pi,
                          device=self.theta_proj.weight.device)

    def _pairs_to_kernel(self, t, dim=-1):
        """Scatter a per-rotated-pair tensor (axis `dim`, width n_rot_pairs) onto
        mh_scan's K = n_hm/2 pair axis, exact zeros on real pairs."""
        if t is None:
            return None
        K = self.d_state_hm // 2
        if self.n_rot_pairs == K:
            return t
        dim = dim % t.dim()
        shape = list(t.shape)
        shape[dim] = K
        return t.new_zeros(shape).index_copy(dim, self.rot_pair_idx, t)

    def _theta_feature(self, q):
        """Angle feature omega_k * tanh(q), q = theta_proj(sel), the kernels scale it by Delta_t[h].

        q is (B,K,L) for the scan, (B,K) for the step, K = n_rot_pairs.  On CUDA
        it reuses SSD's fused `m3_phase_increments` (dt = ones: the Delta factor
        lives in the scan kernel), padding K to a power of two; CPU/fp64 use plain torch."""
        if self.theta_per_head:
            # (B, H*K[, L]) -> the same per-pair map on (B*H, K[, L]), then
            # (B, H, K[, L]): omega is per pair k, shared over heads
            Bq, K = q.shape[0], self.n_rot_pairs
            f = self._theta_feature_shared(q.reshape(Bq * self.H, K, *q.shape[2:]))
            return f.reshape(Bq, self.H, K, *q.shape[2:])
        return self._theta_feature_shared(q)

    def _theta_feature_shared(self, q):
        om = self.theta_omega()
        if q.dim() == 2:                       # `step()`: one token, (B,K)
            return om.to(q.dtype) * torch.tanh(q)
        if not (q.is_cuda and q.dtype != torch.float64):
            return om.to(q.dtype)[None, :, None] * torch.tanh(q)
        from ..kernels.ssd_mimo.m3_epilogue import m3_phase_increments
        Bsz, K, L = q.shape
        Kp = 1 << max(0, K - 1).bit_length()             # next power of two
        ang = m3_phase_increments(
            q.float().transpose(1, 2).unsqueeze(2),      # (B, L, 1, K)
            q.new_ones(Bsz, 1, L, dtype=torch.float32), Kp, 1.0)
        ang = ang[..., :K].squeeze(2).transpose(1, 2)    # (B, K, L)
        # omega = pi * tanh(theta_freq); the kernel already applied the pi.
        w = (om * (1.0 / math.pi)).to(ang.dtype)
        return (ang * w[None, :, None]).to(q.dtype)

    def _theta_h_kernel(self):
        """The read offset in the `mh_scan` frame: ``-theta_h``.

        Same convention as `SSDMimoSSM` (paper Sec.5, eq. hm_rotation_law): with
        the physical phase Theta^phys the score is
        C^T R(Theta^phys_t - Theta^phys_s + theta_h) B.  `mh_scan` folds the
        positive phase Theta = -Theta^phys and rotates C by Theta + (kernel
        theta_h), so the kernel receives the negated parameter (exact)."""
        return (self._pairs_to_kernel(-self.theta_h)
                if self.use_theta_h else None)

    def _check_theta_h_convention(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys,
                                  error_msgs):
        """Load pre-hook: the ``theta_h_convention`` marker must be the current one.

        A dict with ``theta_h`` but no marker holds the older read-offset convention (negated
        ``theta_h``): an older layout, refused like every old format (`raise_old_format`;
        the migration tooling converts it).  An unknown code is refused.
        A partial dict is untouched."""
        key = prefix + "theta_h_convention"
        code = state_dict.get(key)
        if code is None:
            if prefix + "theta_h" in state_dict:
                raise_old_format(f"no {key!r} marker (the older read-offset convention)")
            return
        if int(code) != THETA_H_CONVENTION:
            raise ValueError(
                f"{key}={int(code)} is not a known read-offset convention "
                f"(this version reads {THETA_H_CONVENTION})")

    # ── |A| activations: a_act (m3 rows), dyn_act (dyn rows) ───────────
    # Properties keep the routes' value, the state-dict marker and the recorded
    # recipe in step, whether set by the constructor, a load or an assignment.
    @property
    def a_act(self) -> str:
        """|A| activation of the m3 rows, one of `A_ACTS`."""
        return self._a_act

    @a_act.setter
    def a_act(self, name) -> None:
        if name not in A_ACTS:
            raise ValueError(f"a_act must be one of {A_ACTS}, got {name!r}")
        self._a_act = name
        self._sync_act_record("a_act", name)

    @staticmethod
    def _canonical_dyn_act(value) -> str | None:
        """Canonical `DYN_ACTS` name of a `dyn_act` value (name, alias or
        `layout.DYN_*` code), None if it names none."""
        if isinstance(value, str):
            name = "raw" if value == "identity" else value
            return name if name in DYN_ACTS else None
        if isinstance(value, bool):
            return None
        try:
            return _DYN_CODE_NAMES.get(int(value))
        except (TypeError, ValueError):
            return None

    @property
    def dyn_act(self) -> str:
        """Rate activation of the `dyn` rows, one of `DYN_ACTS`."""
        return self._dyn_act_sel

    @dyn_act.setter
    def dyn_act(self, value) -> None:
        name = self._canonical_dyn_act(value)
        if name is None:
            raise ValueError(f"dyn_act must be one of {DYN_ACTS} (or a "
                             f"layout.DYN_* code), got {value!r}")
        self._dyn_act_sel = name
        # the code the scans receive: heavy_tail is applied by the module
        # and reaches them as the raw rate (`_dyn_scan_input`)
        self._dyn_act = DYN_RAW if name == "heavy_tail" else _DYN_NAMES[name]
        self._sync_act_record("dyn_act", name)

    def _sync_act_record(self, field: str, name: str) -> None:
        """Mirror an activation into its marker buffer and the recipe."""
        buf = self._buffers.get(_ACT_MARKERS[field][0])
        if buf is not None:
            with torch.no_grad():
                buf.fill_(ACT_MARKER_CODES[name])
        rec = self.__dict__.get("_resolved_constructor_config")
        if rec is not None:
            old = rec["model"].get(field)
            same = (old == name if field == "a_act"
                    else self._canonical_dyn_act(old) == name)
            if not same:
                rec["model"][field] = name

    def _check_act_markers(self, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs):
        """Load pre-hook: the |A| activation markers must equal this module's.

        Per marker present on this module (``a_act_code``, ``dyn_act_code``): a
        declared code that differs from the module's value, or is unknown, is
        refused (rebuild with the checkpoint's value); an absent code with its
        weight in the dict is an older layout (strict-format error); a partial
        dict is untouched."""
        for field, (buf, weight, legal) in _ACT_MARKERS.items():
            if buf not in self._buffers:
                continue
            key = prefix + buf
            code = state_dict.get(key)
            current = getattr(self, field)
            if code is None:
                if prefix + weight in state_dict:
                    raise_old_format(f"no {key!r} marker")
                continue
            name = _ACT_MARKER_NAMES.get(int(code))
            if name not in legal:
                raise ValueError(
                    f"{key}={int(code)} is not a known {field} marker (this "
                    f"version reads {[ACT_MARKER_CODES[n] for n in legal]} = "
                    f"{list(legal)})")
            if name != current:
                raise ValueError(
                    f"HybridMambaSSM: the state dict declares {field}={name!r} "
                    f"(the activation its weights were trained with), but this "
                    f"module resolves {field}={current!r}; rebuild it with "
                    f"{field}={name!r}")

    # ── selection source: proj_source ────────
    def _check_proj_source_marker(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys,
                                  error_msgs):
        """Load pre-hook: the ``proj_source_code`` marker must equal this module's source.

        A declared source that differs from the resolved ``proj_source`` (explicit
        or default), or an unknown code, is refused (rebuild with the
        checkpoint's source).  A dict with the token maps (``bc_proj.weight``)
        but no marker is an older layout (strict-format error).  A partial dict
        is untouched."""
        key = prefix + "proj_source_code"
        code = state_dict.get(key)
        if code is None:
            if prefix + "bc_proj.weight" in state_dict:
                raise_old_format(f"no {key!r} marker")
            return
        name = _PROJ_SOURCE_MARKER_NAMES.get(int(code))
        if name is None:
            raise ValueError(
                f"{key}={int(code)} is not a known proj_source marker "
                f"(this version reads {PROJ_SOURCE_MARKER_CODES})")
        if name != self.proj_source:
            raise ValueError(
                f"HybridMambaSSM: the state dict's token maps were trained on "
                f"proj_source={name!r}, but this module resolves "
                f"proj_source={self.proj_source!r}; rebuild it with "
                f"proj_source={name!r}")

    # ── the dynamic-rate law ─────────────────
    # `dyn` rows have no static rate: log a_bar = -Delta_t * act(a_dyn_proj(s_t)),
    # the form of the m3 rows and SSD-MIMO's dynamic law.  The kernel's static
    # coefficient is c1 = -1 there; `_rows_to_kernel` hands it an exact 0 for A_log.
    def _register_host_buffer(self, name, values, dtype, persistent=False):
        """Buffer whose value is host-side: a constant list, or a callable
        returning the current value (the state-dict markers follow the
        config field)."""
        if values is None:
            self.register_buffer(name, None, persistent=persistent)
            return
        self._host_buffers[name] = values if callable(values) else list(values)
        self.register_buffer(name, self._host_value(name, dtype),
                             persistent=persistent)

    def _host_value(self, name, dtype, device=None):
        v = self._host_buffers[name]
        return torch.tensor(v() if callable(v) else v, dtype=dtype,
                            device=device)

    def _apply(self, fn, *args, **kwargs):
        # `to_empty` (the meta-build materialisation) hands back
        # uninitialised storage: restore the host constants on every move.
        out = super()._apply(fn, *args, **kwargs)
        with torch.no_grad():
            for name in getattr(self, "_host_buffers", {}):
                buf = getattr(self, name, None)
                if buf is not None and buf.device.type != "meta":
                    buf.copy_(self._host_value(name, buf.dtype, "cpu"))
        return out

    @staticmethod
    def _rows_to_kernel(t, idx, dim=-1):
        """Gather a law-block store to mh_scan's n_hm row axis: `idx[n]` is row n's
        column, or the store width (an appended exact zero) for rows of another law.
        `idx` None = identity."""
        if t is None or idx is None:
            return t
        dim = dim % t.dim()
        pad = [0, 0] * (t.dim() - 1 - dim) + [0, 1]
        return F.pad(t, pad).index_select(dim, idx)

    def _A_log_kernel(self):
        """(H,P,n_hm) static log-rate for `mh_scan`; exact 0 on dyn rows."""
        if self.A_log is None:
            return self.dt_bias.new_zeros(self.H, self.P, self.d_state_hm)
        return self._rows_to_kernel(self.A_log, self.a_log_idx, -1)

    def _mu_gain_full(self):
        """(H,P,n_mu) bounded mu gain c = G sigmoid(mu_gain_logit) in
        (0, G), G = ln(mu_cap)/2."""
        return mu_gain_bound(self.mu_cap) * torch.sigmoid(self.mu_gain_logit)

    def _bound_mu_map(self, maps):
        """Apply the mu law's bounded driver tanh on the host to the raw `mu` rows (all routes)."""
        if "mu" in maps:
            maps["mu"] = torch.tanh(maps["mu"])
        return maps

    def _refuse_per_coordinate_complex_gates(self, state_dict, prefix, *_args):
        refuse_per_coordinate_gates(self, state_dict, prefix, lambda name: self.d_state_hm)

    def _gate_maps(self, maps):
        """Gather the compact g_B / g_C rows (pair-tied, gated blocks only) to the
        n_hm row axis (axis 1); gate-off rows read an exact zero."""
        idx = self.gate_tie_idx
        if idx is not None:
            for k in ("g_B", "g_C"):
                g = maps.get(k)
                if g is not None:
                    if self.gate_masked:
                        g = torch.cat([g, g.new_zeros(g[:, :1].shape)], 1)
                    maps[k] = g.index_select(1, idx)
        return maps

    def _gate_bias(self, beta):
        """A (G, n_gate) gate bias on the n_hm row axis (exact zero on gate-off rows)."""
        if beta is None or self.gate_tie_idx is None:
            return beta
        if self.gate_masked:
            beta = torch.cat([beta, beta.new_zeros(beta[..., :1].shape)], -1)
        return beta.index_select(-1, self.gate_tie_idx)

    def _mu_gain_kernel(self):
        """(H,P,n_hm) mu gain on the kernel rows, exact 0 off the mu rows."""
        if self.mu_proj is None:
            return None
        return self._rows_to_kernel(self._mu_gain_full(), self.mu_idx, -1)

    # Rate a = f(a_dyn), f = `dyn_act`, no floor.  raw/softplus/exp are applied by
    # the scans; "heavy_tail" is SSD's `_a_mag_tail` (floor 0), applied here and
    # passed on as the raw rate (DYN_RAW) at each route's own precision.
    def _dyn_rate(self, a_dyn: torch.Tensor) -> torch.Tensor:
        """`step`'s rate, in a_dyn's dtype (as `layout.dyn_rate` there)."""
        if self.dyn_act == "heavy_tail":
            return _a_mag_tail(a_dyn, 0.0, False, "heavy_tail")
        return dyn_rate(a_dyn, self._dyn_act, 0.0)

    def _dyn_scan_input(self, a_dyn, eng: str, x_dtype: torch.dtype):
        """``(a_dyn, dyn_act)`` for `mh_scan` / `mh_torch_scan`.

        Unchanged for raw/softplus/exp.  Under heavy_tail the rate is formed here
        in the route's own dtype (fp32 for `mh_scan`; fp64 iff x is for
        `mh_torch_scan`) and passed as DYN_RAW."""
        if a_dyn is None or self.dyn_act != "heavy_tail":
            return a_dyn, self._dyn_act
        wd = (torch.float64 if x_dtype == torch.float64 and eng != "cuda"
              else torch.float32)
        if "amag" in hm_held_regen() and torch.is_grad_enabled() and a_dyn.requires_grad:
            # HM_HELD_REGEN=amag: recompute the clamp/reciprocal chain in the backward (bit-identical)
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda a: _a_mag_tail(a.to(wd), 0.0, False, "heavy_tail"),
                              a_dyn, use_reentrant=False), DYN_RAW
        return _a_mag_tail(a_dyn.to(wd), 0.0, False, "heavy_tail"), DYN_RAW

    # ── route telemetry (A8.3) ─────────────────────
    # Every row-block dispatch records its entry point; a value match against the
    # fp64 oracle cannot tell kernel from reference routes.
    _KERNEL_ROUTES = ("mh_scan", "ssd_mimo_scan", "upstream_siso")

    def _route_begin(self, on_cuda: bool | None = None) -> None:
        self._routes = ()
        self._route_note = None
        self._refusal_note = None
        # the caller asked for a kernel iff backend="cuda", or "auto" on a
        # CUDA input; any other route is then a refusal of that request
        self._wants_kernel = self.backend == "cuda" or (
            self.backend == "auto" and bool(on_cuda))

    def _route(self, block: str, name: str) -> None:
        self._routes = getattr(self, "_routes", ()) + ((block, name),)
        if name not in self._KERNEL_ROUTES:
            note = f"{block}: {name}"
            prev = getattr(self, "_route_note", None)
            self._route_note = note if prev is None else f"{prev}; {note}"
            if getattr(self, "_wants_kernel", False):
                prev = getattr(self, "_refusal_note", None)
                self._refusal_note = note if prev is None \
                    else f"{prev}; {note}"
                # backend='cuda' raises, backend='auto' warns once (layers/routes.py)
                demote_kernel_route(self, True, f"row block {block} would run {name}")

    def last_route(self) -> str | None:
        """`None` iff every row block of the LAST forward ran its CUDA kernel entry
        (`mh_scan` for A/dyn/mu rows, `ssd_mimo_scan` or the upstream SISO adapter
        for flat rows).  Otherwise each non-kernel route, ``"<block>: <route>"``,
        joined by ``"; "``.  Requested non-kernel routes (``backend="torch"``,
        CPU input) are named too.  ``step()`` does not reset it."""
        return getattr(self, "_route_note", None)

    def last_refusal(self) -> str | None:
        """`None` unless the LAST forward was moved off the route the caller asked
        for; then the routes that served instead, as in `last_route()`.  A
        requested non-kernel route is not a refusal."""
        return getattr(self, "_refusal_note", None)

    def last_routes(self) -> tuple:
        """Every ``(block, route)`` dispatch of the LAST forward, in order."""
        return tuple(getattr(self, "_routes", ()))

    def hidden_state_shape(self):
        """One [H,P,N] tensor; N slices are MH, M3 (complex first), M2."""
        return (self.H, self.P, self.d_state)

    def _base_initial_state(self):
        if self.h_init_param is None:
            return None
        return self.h_init_param

    @property
    def _conv_pkg(self) -> bool:
        """Do the convolutions run on the ``causal_conv1d`` package (the opt-in ``conv_kernel="causal_conv1d"``)?"""
        return check_conv_kernel(self.conv_kernel)

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

    def _scan_init_state(self, batch_size: int, device=None,
                         dtype: torch.dtype = torch.float32) -> State:
        dev = device or next(self.parameters()).device
        state = State(hidden_state=self.initial_state(batch_size).to(device=dev, dtype=dtype))
        phase_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        n_hm, n3 = self.d_state_hm, self.d_state_m3
        if n_hm > 0:
            if self.has_rot:
                state["theta"] = torch.zeros(       # rotated pairs only
                    batch_size, self.H, self.n_rot_pairs, device=dev, dtype=phase_dtype)
            if self.mu_proj is not None:
                mu_width = self.mu_proj.out_features  # n_mu | H
                state["mu"] = torch.zeros(batch_size, mu_width, device=dev, dtype=dtype)
            if self.trap_proj is not None:
                state["write_prev"] = torch.zeros(batch_size, self.H, self.P, n_hm,
                                                  device=dev, dtype=dtype)
        if self.bc_conv1d is not None:
            state["bc_conv_state"] = self.bc_conv1d.init_state(
                batch_size, device=dev, dtype=dtype)
        if n3 > 0:
            state["c3_kp"] = torch.zeros(batch_size, self.H, n3, device=dev, dtype=dtype)
            state["c3_vp"] = torch.zeros(batch_size, self.H, self.P, device=dev, dtype=dtype)
            if self.m3_angle_proj is not None:
                # the complex m3 rows' pairs only (n3_complex/2)
                state["c3_phase"] = torch.zeros(
                    batch_size, self.H, self.m3_angle_proj.out_features,
                    device=dev, dtype=phase_dtype)
        return state

    def scan_step(self, x_t: torch.Tensor, state: State | dict | None = None, *,
                  sel_x: torch.Tensor | None = None,
                  tok_raw: torch.Tensor | None = None) -> tuple[torch.Tensor, State]:
        """Single-token decode: a torch-native update of the HM rows (laws A/dyn/mu
        with rotation, gates, biases) plus the flat rows' two-tap recurrence,
        summed as in the forward.  Omitted state starts from `init_state`.

        Token maps come off the same fused GEMM as the forward.  `tok_raw`
        (B, tok_width) passes in a slab the caller already projected."""
        Bsz = x_t.shape[0]
        state = ScanState(self).resolve_state(state, batch_size=Bsz, device=x_t.device,
                                   dtype=x_t.dtype)
        if self.proj_source == "layer_input":
            if sel_x is None:
                raise ValueError(
                    "HybridMambaSSM(proj_source='layer_input') needs sel_x (the "
                    "block input, width d_model) in step(); the layer passes "
                    "it -- or build the core with proj_source='main_branch'")
            u_t = sel_x
        else:
            u_t = x_t
        if u_t.shape[-1] != self.d_sel:
            raise RuntimeError(
                f"selection input is width {u_t.shape[-1]}, expected "
                f"{self.d_sel} (proj_source={self.proj_source!r})")

        n_hm, n3 = self.d_state_hm, self.d_state_m3
        # one GEMM for every token map, then squeeze the L=1 axis
        maps = self._gate_maps(self._bound_mu_map({k: v.squeeze(-1) for k, v in (
            self._sel_slice(tok_raw.unsqueeze(-1)) if tok_raw is not None
            else self._sel_project(u_t.unsqueeze(-1))).items()}))
        # with the dt bottleneck, `dt` holds the rank-r code that `dt_proj` reads
        dt_lin = (maps["dt"] if self.dt_low is None
                  else self.dt_proj(maps["dt"]))
        # the clock in the carries' working dtype (fp32 under 16-bit IO), as the forward forms it: a 16-bit clock
        # would round every token's decay factor, and the fp32 carry would collect one rounding per token
        wd = torch.float64 if dt_lin.dtype == torch.float64 else torch.float32
        dt = F.softplus(dt_lin.to(wd) + self.dt_bias.to(wd))         # (B,H)

        bc = maps["bc"]                                              # (B, 2*d_state)
        new_state = State(state)
        next_hidden = []
        if self.bc_conv1d is not None:
            bc, bc_conv_state = causal_conv_step(bc, state["bc_conv_state"], self.bc_conv1d.conv.weight, self.bc_conv1d.conv.bias,
                                                 package=self._conv_impl)
            new_state["bc_conv_state"] = bc_conv_state
        Bt_all, Ct_all = bc.split([self.d_state, self.d_state], dim=-1)

        y = torch.zeros(Bsz, self.d_inner, device=x_t.device, dtype=x_t.dtype)
        if n_hm > 0:
            y_hm, new_state = self._hm_step(
                maps, x_t, dt, Bt_all[..., :n_hm], Ct_all[..., :n_hm],
                state, new_state)
            next_hidden.append(new_state.pop("_hidden_update"))
            y = y + y_hm.to(y.dtype)
        o3, o2 = self._flat_offsets()
        if n3 > 0:
            y_c3, new_state = self._m3_step(
                maps, x_t, dt, Bt_all[..., o3:o2], Ct_all[..., o3:o2],
                state, new_state)
            next_hidden.append(new_state.pop("_hidden_update"))
            y = y + y_c3.to(y.dtype)
        if self.d_state_m2 > 0:
            y_f, new_state = self._flat_step(
                "m2", x_t, dt, Bt_all[..., o2:], Ct_all[..., o2:],
                state, new_state)
            next_hidden.append(new_state.pop("_hidden_update"))
            y = y + y_f.to(y.dtype)
        if self.skip:
            y = y + x_t * self._skip_channels().to(x_t.dtype)
        new_state["hidden_state"] = (next_hidden[0] if len(next_hidden) == 1
                                     else torch.cat(next_hidden, dim=-1))
        new_state = ScanState(self)._conform_carries(new_state, state)
        self._final_state = new_state
        self.h_final = new_state["hidden_state"]
        return y, new_state

    def _hm_step(self, maps, x_t, dt, Bt, Ct, state, new_state):
        """One token of the HM rows (laws A/dyn/mu, rotation, gates, biases): the
        single-token form of `mh_reference`, in the model's dtype rather than fp64."""
        H, P, n_hm = self.H, self.P, self.d_state_hm
        Bsz = x_t.shape[0]
        if self.b_norm is not None:
            Bt = self.b_norm(Bt)
        if self.c_norm is not None:
            Ct = self.c_norm(Ct)

        A_log = self._A_log_kernel()                                  # (H,P,n_hm)
        c1 = -A_log.to(dt.dtype).exp()                                # (H,P,n_hm), the clock's (working) dtype

        row_law = self.row_law
        is_dyn = (row_law == LAW_A_DYN)
        is_mu = (row_law == LAW_MU)

        s1 = dt[:, :, None].expand(Bsz, H, n_hm).clone()               # (B,H,n_hm)
        if self.a_dyn_proj is not None:
            a_dyn_t = maps["a_dyn"]                                    # (B,n_hm)|(B,H)
            rate = self._dyn_rate(a_dyn_t.to(dt.dtype))
            if self.dyn_res == "n":
                rate_b = self._rows_to_kernel(rate, self.dyn_idx, 1)[:, None, :].expand(Bsz, H, n_hm)
            else:
                rate_b = rate[:, :, None].expand(Bsz, H, n_hm)
            s1 = torch.where(is_dyn.view(1, 1, n_hm), dt[:, :, None] * rate_b, s1)

        c2 = torch.zeros_like(c1)
        s2 = torch.zeros(Bsz, H, n_hm, device=x_t.device, dtype=c1.dtype)
        mu_t = None
        if self.mu_proj is not None:
            mu_gain = self._mu_gain_kernel()                           # (H,P,n_hm)
            c2 = torch.where(is_mu.view(1, 1, n_hm), mu_gain.to(c1.dtype), torch.zeros_like(c1))
            mu_t = maps["mu"]                                          # (B,n_mu)|(B,H)
            mu_prev = state.get("mu")
            if mu_prev is None:
                mu_prev = torch.zeros_like(mu_t)
            if self.mu_res == "n":
                mu_b = self._rows_to_kernel(mu_t, self.mu_idx, 1)[:, None, :].expand(Bsz, H, n_hm)
                prev_b = self._rows_to_kernel(mu_prev, self.mu_idx, 1)[:, None, :].expand(Bsz, H, n_hm)
            else:
                mu_b = mu_t[:, :, None].expand(Bsz, H, n_hm)
                prev_b = mu_prev[:, :, None].expand(Bsz, H, n_hm)
            s2 = torch.where(is_mu.view(1, 1, n_hm), mu_b.to(s2.dtype) - prev_b.to(s2.dtype), s2)

        log_a = c1[None] * s1[:, :, None, :] + c2[None] * s2[:, :, None, :]   # (B,H,P,n_hm)
        a_bar = torch.exp(log_a)

        # ── B/C fold: gate_pos "pre" is gate -> bias -> rotate (SSD's order);
        # "post" is bias -> rotate -> gate ──
        pre = self.gate_pos == "pre"
        if pre and self.gB_proj is not None:
            Bt = self._pre_gate(Bt[..., None], maps["g_B"][..., None], self._gate_bias(self.beta_gB))[..., 0]
            Ct = self._pre_gate(Ct[..., None], maps["g_C"][..., None], self._gate_bias(self.beta_gC))[..., 0]
        Bh = Bt[:, None, :].expand(Bsz, H, n_hm).clone()
        Ch = Ct[:, None, :].expand(Bsz, H, n_hm).clone()
        bb = _expand_group(self.beta_b, H)
        cc = _expand_group(self.beta_c, H)
        if bb is not None:
            Bh = Bh + bb[None]
        if cc is not None:
            Ch = Ch + cc[None]

        theta_t = None
        if self.has_rot:
            # inc = Delta_t[h] * omega_k tanh(q_t[k]); B rotates by Theta, C by
            # Theta + kernel theta_h (see `_theta_h_kernel`).
            feat = self._theta_feature(maps["theta"].to(dt.dtype))       # (B,Kr)
            theta_t = state["theta"] + dt[:, :, None] * (             # (B,H,Kr)
                feat if self.theta_per_head else feat[:, None, :])
            new_state["theta"] = theta_t
            # widen the Kr rotated pairs to all K pairs (zeros on real ones; `row_rot` masks too)
            ang_b = self._pairs_to_kernel(theta_t)                     # (B,H,K)
            ang_c = ang_b if not self.use_theta_h else \
                ang_b + self._theta_h_kernel()[None].to(theta_t.dtype)

            def _rot(v, ang):
                cos_t, sin_t = torch.cos(ang), torch.sin(ang)
                vp = v.view(Bsz, H, n_hm // 2, 2)
                e, o = vp[..., 0], vp[..., 1]
                rotated = torch.stack([e * cos_t - o * sin_t,
                                       e * sin_t + o * cos_t], dim=-1).reshape(Bsz, H, n_hm)
                rot_mask = self.row_rot.view(1, 1, n_hm)
                return torch.where(rot_mask, rotated, v)

            Bh, Ch = _rot(Bh, ang_b), _rot(Ch, ang_c)

        if self.gB_proj is not None and not pre:
            gv = maps["g_B"][:, None, :].expand(Bsz, H, n_hm).clone()
            bgB = _expand_group(self._gate_bias(self.beta_gB), H)
            if bgB is not None:
                gv = gv + bgB[None]
            Bh = Bh * gate_act(gv, self._gate_act)
        if self.gC_proj is not None and not pre:
            gv = maps["g_C"][:, None, :].expand(Bsz, H, n_hm).clone()
            bgC = _expand_group(self._gate_bias(self.beta_gC), H)
            if bgC is not None:
                gv = gv + bgC[None]
            Ch = Ch * gate_act(gv, self._gate_act)

        h_old = state["hidden_state"][..., :self.d_state_hm]                                              # (B,H,P,n_hm)
        x_hp = x_t.view(Bsz, H, P)
        W_t = x_hp.unsqueeze(-1) * Bh[:, :, None, :]                    # unscaled write (B,H,P,n_hm)
        if self.trap_proj is not None:
            # transported endpoint rule in the folded frame:
            #   h_t = a h_{t-1} + dt_t [lam W_t + (1 - lam) a W_{t-1}]
            lam = sigmoid_exact(maps["trap"].to(W_t.dtype))[:, :, None, None]
            W_prev = state.get("write_prev")
            if W_prev is None:
                W_prev = torch.zeros_like(W_t)
            drive = dt[:, :, None, None] * (lam * W_t + (1.0 - lam) * a_bar * W_prev)
            new_state["write_prev"] = W_t
        else:
            drive = dt[:, :, None, None] * W_t
        h_new = a_bar * h_old + drive
        y_t = (Ch[:, :, None, :] * h_new).sum(-1)                        # (B,H,P)

        new_state["_hidden_update"] = h_new
        if mu_t is not None:
            new_state["mu"] = mu_t
        return y_t.reshape(Bsz, self.d_inner), new_state

    def _m3_rate(self, a_lin: torch.Tensor) -> torch.Tensor:
        """The m3 rows' ``|A_t| = max(f(a_lin), a_floor)``, f = ``a_act``.

        Used by every m3 route: SSD's `_a_mag_tail`, eager, so value, adjoint and
        boundary rule match `SSDMimoSSM`."""
        return _a_mag_tail(a_lin, self.a_floor, False, self.a_act)

    def _m3_step(self, maps, x_t, dt, Bt, Ct, state, new_state):
        """M3 endpoint update in folded coordinates, with fp32/fp64 accumulation."""
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        Bsz, H, P, n3 = x_t.shape[0], self.H, self.P, self.d_state_m3
        dt = dt.to(wd)
        rate = self._m3_rate(maps["m3_a"].to(wd))
        alpha = torch.exp(-rate * dt)[..., None, None]
        lam = sigmoid_exact(maps["m3_trap"].to(wd))[..., None, None]
        d = dt[..., None, None]
        k = self.m3_b_norm(Bt).to(wd)[:, None] + self.m3_B_bias.to(wd)
        q = self.m3_c_norm(Ct).to(wd)[:, None] + self.m3_C_bias.to(wd)
        if self.m3_angle_proj is not None:
            increment = torch.tanh(maps["m3_ang"].to(wd))[:, None] * math.pi * dt[..., None]
            phase = state["c3_phase"] + increment             # rotated pairs
            full = F.pad(phase, (0, n3 // 2 - phase.shape[-1]))  # real pairs: 0
            cos, sin = full.cos(), full.sin()
            def rotate(v):
                e, o = v[..., 0::2], v[..., 1::2]
                return torch.stack((e * cos - o * sin, e * sin + o * cos), -1).flatten(-2)
            k, q = rotate(k), rotate(q)
            new_state["c3_phase"] = phase
        v = x_t.reshape(Bsz, H, P).to(wd)
        start = self.d_state_hm
        old = state["hidden_state"][..., start:start + n3].to(wd)
        previous = state["c3_vp"].to(wd)[..., None] * state["c3_kp"].to(wd)[:, :, None]
        hidden = alpha * old + alpha * d * (1 - lam) * previous + lam * d * v[..., None] * k[:, :, None]
        new_state["_hidden_update"] = hidden.to(state["hidden_state"].dtype)
        new_state["c3_kp"] = k.to(state["c3_kp"].dtype)
        new_state["c3_vp"] = v.to(state["c3_vp"].dtype)
        y = (hidden * q[:, :, None]).sum(-1)
        return y.reshape(Bsz, self.d_inner).to(x_t.dtype), new_state

    _SEL_ORDER = ("dt", "bc", "a_dyn", "mu", "theta", "trap", "g_B", "g_C",
                  "m3_a", "m3_trap", "m3_ang")

    def _sel_modules(self):
        """The present token-map projections, in `_SEL_ORDER`.

        With the dt bottleneck `dt` is `dt_low`; `dt_proj` reads its rank-r code,
        so it runs as its own small matmul afterwards."""
        m = {
            "dt": self.dt_low if self.dt_low is not None else self.dt_proj,
            "bc": self.bc_proj, "a_dyn": self.a_dyn_proj, "mu": self.mu_proj,
            "theta": self.theta_proj, "trap": self.trap_proj,
            "g_B": self.gB_proj, "g_C": self.gC_proj,
            "m3_a": self.m3_a_proj, "m3_trap": self.m3_trap_proj,
            "m3_ang": self.m3_angle_proj,
        }
        return [(k, m[k]) for k in self._SEL_ORDER if m[k] is not None]

    def _sel_project(self, u_cf):
        """One GEMM for every token map; ``u_cf`` is (B, d_sel, L).  Returns
        ``{key: (B, rows, L)}`` row slices.  Fusing rounds like any GEMM-shape change."""
        return self._sel_slice(matmul_channel_major(
            self.tok_proj_weight(), u_cf.transpose(1, 2)))

    def tok_proj_weight(self) -> torch.Tensor:
        """The fused (tok_width, d_sel) token-map weight.  With
        ``proj_source="layer_input"`` the layer cats it onto its `in_proj` so all
        projections run as one GEMM (see `HybridMambaSSM.forward`)."""
        mods = self._sel_modules()
        if len(mods) == 1:
            return mods[0][1].weight
        return torch.cat([m.weight for _, m in mods], 0)

    @property
    def tok_width(self) -> int:
        """Total rows of `tok_proj_weight` -- the fused token-map width."""
        return sum(m.weight.shape[0] for _, m in self._sel_modules())

    def _sel_slice(self, proj):
        """Split a (B, tok_width, L) token slab into ``{key: (B, rows, L)}`` in
        `_sel_modules` order (also for a slab folded into `in_proj`, `tok_raw`)."""
        out, off = {}, 0
        for key, m in self._sel_modules():
            rows = m.weight.shape[0]
            sl = proj[:, off:off + rows]
            if m.bias is not None:
                # the slab GEMM is weights only: add the bias of biased projections here
                sl = sl + m.bias.to(sl.dtype).view(1, rows, *([1] * (proj.dim() - 2)))
            out[key] = sl
            off += rows
        return out

    def _sel_input_cf(self, x_cf, sel_x):
        """(B, d_sel, L) selection input, per `resolve_proj_source`'s rule."""
        if self.proj_source == "layer_input":
            if sel_x is None:
                raise ValueError(
                    "HybridMambaSSM(proj_source='layer_input') needs sel_x (the "
                    "block input, width d_model); the layer passes it -- or "
                    "build the core with proj_source='main_branch'")
            u_cf = sel_x.transpose(1, 2)
        else:
            u_cf = x_cf
        if u_cf.shape[1] != self.d_sel:
            raise RuntimeError(
                f"selection input is width {u_cf.shape[1]}, expected "
                f"{self.d_sel} (proj_source={self.proj_source!r})")
        return u_cf

    def _scan_cf(self, x_cf, h_init, sel_x, want_state, tok_raw=None,
                 z_cf=None, complete_state=None):
        """All row blocks, channel-major.  ``x_cf``: (B, d_inner, L), contiguous.
        Returns ((B, d_inner, L), h); the D skip is added only if fused into the scan.

        ``tok_raw`` (B, L, tok_width) is the token slab already projected by the
        caller's fused `in_proj` (``proj_source="layer_input"``); it is the one
        channel-last input."""
        if tok_raw is not None:
            maps = self._sel_slice(tok_raw.transpose(1, 2))
        elif self.proj_source == "main_branch":
            # the maps read the same drive as the scan; the tee joins the two
            # activation-sized gradients inside the GEMM (`matmul_channel_major_tee`)
            x_cf, proj = matmul_channel_major_tee(
                self._sel_input_cf(x_cf, sel_x), self.tok_proj_weight())
            maps = self._sel_slice(proj)
        else:
            maps = self._sel_project(self._sel_input_cf(x_cf, sel_x))
        maps = self._gate_maps(self._bound_mu_map(maps))
        n_hm, n3 = self.d_state_hm, self.d_state_m3
        n_flat = self.fracs.n_flat
        if self.dt_low is None:
            dt_lin = maps["dt"]
        else:
            # bmm writes the bottleneck output channel-major without matmul fold-copy.
            dt_lin = matmul_channel_major(self.dt_proj.weight,
                                          maps["dt"].transpose(1, 2))
            if ("dtlin" in hm_held_regen() and torch.is_grad_enabled()
                    and dt_lin.requires_grad
                    and self._delta_prep_in_kernel(x_cf, n_flat, n_hm)):
                # HM_HELD_REGEN=dtlin: rebuild `dt_lin` from `dt_low` with the same bmm
                _acd = (torch.is_autocast_enabled("cuda"),
                        torch.get_autocast_dtype("cuda"))
                _w, _lo = self.dt_proj.weight, maps["dt"].transpose(1, 2)

                def _dtlin_regen(_w=_w, _lo=_lo, _acd=_acd):
                    with torch.autocast("cuda", dtype=_acd[1], enabled=_acd[0]):
                        return matmul_channel_major(_w, _lo)
                _regen.register(dt_lin, _dtlin_regen)
        bc = maps["bc"]                                         # (B,2N,L)
        bc_raw = bc
        if self.bc_conv1d is not None:
            prefix = (complete_state["bc_conv_state"][..., 1:].to(bc.dtype)
                      if complete_state is not None else None)
            if prefix is not None:
                # valid conv over the explicit differentiable history (the padded
                # kernel gives different boundary adjoints across split calls)
                conv = self.bc_conv1d.conv
                bc = F.silu(F.conv1d(torch.cat((prefix, bc), dim=-1),
                    conv.weight.to(bc.dtype), conv.bias.to(bc.dtype),
                    groups=bc.shape[1]))
            else:
                bc = causal_depthwise_conv1d_cl(
                    bc, self.bc_conv1d.conv.weight, self.bc_conv1d.conv.bias,
                    self.bc_conv1d.d_conv, activation="silu", package=self._conv_impl)
        Bt_all, Ct_all = bc[:, :self.d_state], bc[:, self.d_state:]

        # dt softplus and the two B/C RMS-norms in one launch, only without flat
        # rows (they have a second norm pair).
        dt = bc_normed = None
        # the scan fuses clock preparation only when its ABI supports the law
        dt_in_kernel = self._delta_prep_in_kernel(x_cf, n_flat, n_hm)
        if dt_in_kernel:
            dt = dt_lin
        elif n_flat == 0 and n_hm > 0:
            got = self._fused_hm_tails(dt_lin, Bt_all[:, :n_hm],
                                       Ct_all[:, :n_hm])
            if got is not None:
                dt, Bhm, Chm = got
                Bt_all, Ct_all, bc_normed = Bhm, Chm, True
        if dt is None:
            dt = F.softplus(dt_lin + self.dt_bias.unsqueeze(-1))  # (B,H,L)
        elif ("delta" in hm_held_regen() and not dt_in_kernel and self.b_norm is None
              and dt.is_cuda and dt is not dt_lin):
            # HM_HELD_REGEN=delta: rebuild the softplus clock from the raw `dt_lin`
            from ..layers.tails import dt_softplus_cf as _dtsp
            _acd = (torch.is_autocast_enabled("cuda"),
                    torch.get_autocast_dtype("cuda"))
            _bias = self.dt_bias

            def _dt_regen(dt_lin=dt_lin, _bias=_bias, _acd=_acd):
                with torch.autocast("cuda", dtype=_acd[1], enabled=_acd[0]):
                    return _dtsp(dt_lin, _bias)
            _regen.register(dt, _dt_regen)

        h_final = None
        # The D skip is on the shared drive, so exactly one summand may carry it:
        # the HM kernel adds it at its store; without HM rows the module does.
        in_kernel = (n_hm > 0 and self._d_skip_in_kernel(x_cf))
        self._k11_skip_in_kernel = in_kernel
        # the output gate rides the HM scan, or not at all
        gate_in_kernel = (z_cf is not None
                          and self._z_gate_in_kernel(x_cf, n_flat, n_hm,
                                                     in_kernel, z_cf))
        self._h3_gate_in_kernel = gate_in_kernel
        if n_hm > 0:
            y, h_final = self._hm_half_cf(
                maps, x_cf, dt, Bt_all[:, :n_hm], Ct_all[:, :n_hm], h_init,
                want_state=want_state,
                d_skip=self.D if in_kernel else None,
                bc_normed=bool(bc_normed), dt_raw=dt_in_kernel,
                z=z_cf if gate_in_kernel else None, complete_state=complete_state)
        else:
            y = torch.zeros_like(x_cf)
        o3, o2 = self._flat_offsets()
        flat_finals = []
        if complete_state is not None:
            for law, start, end in (("m3", o3, o2), ("m2", o2, self.d_state)):
                if end > start:
                    yf, hf = self._flat_state_scan(law, maps, x_cf, dt,
                        Bt_all[:, start:end], Ct_all[:, start:end],
                        complete_state, start, end)
                    y = y + yf.to(y.dtype)
                    flat_finals.append(hf)
            pieces = ([h_final] if n_hm else []) + flat_finals
            h_final = torch.cat(pieces, -1)
        if n3 > 0 and complete_state is None:
            y = y + self._m3_half_cf(
                maps, x_cf, dt, Bt_all[:, o3:o2], Ct_all[:, o3:o2]).to(y.dtype)
        if self.d_state_m2 > 0 and complete_state is None:
            y = y + self._flat_half_cf(
                "m2", x_cf, dt, Bt_all[:, o2:], Ct_all[:, o2:]).to(y.dtype)
        if complete_state is not None:
            # Rebuild the terminal auxiliary carries with one token of the step
            # law; the prefixes stay in autograd so boundary losses keep gradients.
            before_last = State(complete_state)
            if "theta" in before_last:
                wd = before_last["theta"].dtype
                clock = F.softplus(dt_lin.to(wd) + self.dt_bias.to(wd)[None, :, None])
                feature = self._theta_feature(maps["theta"].to(wd))
                # contract time directly (no global B,H,K,L phase product)
                before_last["theta"] = before_last["theta"] + (
                    torch.einsum("bhl,bhkl->bhk", clock[..., :-1], feature[..., :-1])
                    if self.theta_per_head else
                    phase_bmm(clock[..., :-1], feature[..., :-1].transpose(1, 2)))
            if "mu" in before_last and x_cf.shape[-1] > 1:
                before_last["mu"] = maps["mu"][..., -2]
            if "c3_phase" in before_last:
                wd = before_last["c3_phase"].dtype
                feature = torch.tanh(maps["m3_ang"].to(wd)) * math.pi
                terminal = phase_bmm(dt.to(wd)[..., :-1],
                                     feature[..., :-1].transpose(1, 2))
                before_last["c3_phase"] = before_last["c3_phase"] + terminal
            if self.bc_conv1d is not None:
                cache = complete_state["bc_conv_state"]
                history = torch.cat((cache, bc_raw.to(cache.dtype)), -1)
                kc = self.bc_conv1d.d_conv
                before_last["bc_conv_state"] = history[..., -(kc + 1):-1].contiguous()
            _, final = self.scan_step(x_cf[..., -1], before_last,
                sel_x=None if sel_x is None else sel_x[:, -1],
                tok_raw=None if tok_raw is None else tok_raw[:, -1])
            final["hidden_state"] = h_final.to(complete_state["hidden_state"].dtype)
            return y, h_final, final
        return y, h_final

    def _z_gate_in_kernel(self, x_cf, n_flat: int, n_hm: int,
                          skip_in_kernel: bool, z_cf) -> bool:
        """Does `mh_scan` apply the mixer's `y * silu(z)` for this call?

        The gate multiplies the whole readout, so it rides the kernel only if that
        kernel produced it all: HM rows and no flat rows, the D skip fused in the
        same kernel (else the skip would be ungated), `z` channel-major with x's
        shape and dtype, the CUDA backend (references apply the gate at the end,
        as an independent oracle), and the `HM_GLUE=zk` lever on."""
        if not (n_hm > 0 and n_flat == 0 and skip_in_kernel):
            return False
        if not hm_glue_on("zk"):
            return False
        if (z_cf.shape != x_cf.shape or z_cf.dtype != x_cf.dtype
                or not z_cf.is_contiguous()):
            return False
        eng = self.backend
        if eng == "auto":
            eng = "cuda" if x_cf.is_cuda else "sequential"
        return eng == "cuda"

    def _delta_prep_in_kernel(self, x_cf, n_flat: int, n_hm: int) -> bool:
        """Does `mh_scan` apply `softplus(delta + dt_bias)` for this call?

        Only with HM rows and no flat rows (they consume the same `dt` module-side),
        no B/C RMSNorm (the fused tail already computes dt with the norms), the CUDA
        backend (references keep the pass, as an independent oracle) and the
        `HM_GLUE=dtk` lever on.  The fused form skips one bf16 rounding of delta."""
        if not (n_hm > 0 and n_flat == 0):
            return False
        if self.b_norm is not None or self.c_norm is not None:
            return False
        if not hm_glue_on("dtk"):
            return False
        eng = self.backend
        if eng == "auto":
            eng = "cuda" if x_cf.is_cuda else "sequential"
        return eng == "cuda"

    def _skip_channels(self):
        return (self.D.repeat_interleave(self.P) if self.skip_sharing == "head"
                else self.D.flatten())

    def _d_skip_in_kernel(self, x_cf) -> bool:
        """Is the D skip served at the scan kernel's store for this call?

        CUDA route with ``skip_sharing="head"``
        only; the references keep the module-side pass.
        ``HYBRID_SSM_MH_DSKIP_INKERNEL=0`` disables the fusion."""
        eng = self.backend
        if eng == "auto":
            eng = "cuda" if x_cf.is_cuda else "sequential"
        return (eng == "cuda" and self.skip and self.skip_sharing == "head"
                and _os.environ.get("HYBRID_SSM_MH_DSKIP_INKERNEL", "1")
                == "1")

    # ── fused token-map tail ───────────────
    def _fused_hm_tails(self, dt_lin, Bt, Ct):
        """`softplus(dt_lin + dt_bias)` and the B/C RMS-norms in one launch (tails kernel, as SSD-MIMO).

        ``dt_lin`` (B, H, L), ``Bt`` / ``Ct`` (B, n_hm, L), already conv'd and sliced
        to the HM rows.  Returns the same layouts, or ``None`` outside the fusable
        envelope or when `SSD_MIMO_TAILS=0`."""
        from ..layers import tails as _t
        if not _t.tails_enabled():
            return None
        # a b_norm/c_norm split would need two `bc_norm` flags: not fused
        both = (self.b_norm is not None) == (self.c_norm is not None)
        if not both:
            return None
        if self.b_norm is None:
            # Without B/C normalization, only the clock needs a fused tail.
            return _t.dt_softplus_cf(dt_lin, self.dt_bias), Bt, Ct
        b_src, c_src = Bt.transpose(1, 2), Ct.transpose(1, 2)
        d_src = dt_lin.transpose(1, 2)
        if not _t.can_fuse(b_src, c_src, d_src):
            return None
        N = Bt.shape[1]
        norm = self.b_norm is not None
        cfg = _t.TailCfg(
            N=N, R=1, O=1, H=self.H, L=Bt.shape[-1],
            bc_norm=norm, shared_bias=False, fold_gb=False, fold_gc=False,
            a_dyn=False, has_dt_bias=True, has_a_bias=False,
            eps_b=float(self.b_norm.eps) if norm else 0.0,
            eps_c=float(self.c_norm.eps) if norm else 0.0,
            a_floor=0.0,
            dt_dtype=torch.float32, bc_channel_major=True)
        Bo, Co, dt, _ = _t.token_tails(
            cfg, b_src=b_src if norm else None,
            c_src=c_src if norm else None,
            gb_src=None, gc_src=None, dt_src=d_src, a_src=None,
            w_b=self.b_norm.weight if norm else None,
            w_c=self.c_norm.weight if norm else None,
            bias_b=None, bias_c=None, bias_dt=self.dt_bias, bias_a=None,
            need_b=norm, need_c=norm,
            out_dtype_b=b_src.dtype, out_dtype_c=c_src.dtype)
        if not norm:
            return dt, Bt, Ct
        return dt, Bo.squeeze(-1).transpose(1, 2), Co.squeeze(-1).transpose(1, 2)

    def _pre_gate(self, M, g, beta):
        """act(g + beta_g) (.) M on the head-shared (B, n, L) map, in at least
        fp32 (fp64 stays fp64) and rounded back to M's dtype."""
        if g is None:
            return M
        wd = torch.promote_types(torch.promote_types(M.dtype, g.dtype),
                                 torch.float32)
        gv = g.to(wd)
        if beta is not None:                       # G == 1 (checked at init)
            gv = gv + beta.to(wd).reshape(1, -1, 1)
        fac = gate_act(gv, self._gate_act)
        on = getattr(self, "gate_row_on", None)
        if on is not None:
            # rows of a gate-off block are ungated exactly
            fac = torch.where(on.reshape(1, -1, 1), fac, torch.ones_like(fac))
        return (fac * M.to(wd)).to(M.dtype)

    def _hm_half_cf(self, maps, x_cf, delta, Bt, Ct, h_init, want_state,
                    d_skip=None, bc_normed=False, dt_raw=False, z=None,
                    complete_state=None):
        """Evaluate HM rows with maps already in the scan layout (B,channels,L)."""
        Bsz, _, L = x_cf.shape
        if self.b_norm is not None and not bc_normed:
            Bt = rms_norm_cast(Bt.transpose(1, 2), self.b_norm.weight,
                               self.b_norm.eps).transpose(1, 2)
        if self.c_norm is not None and not bc_normed:
            Ct = rms_norm_cast(Ct.transpose(1, 2), self.c_norm.weight,
                               self.c_norm.eps).transpose(1, 2)
        x_t = x_cf.reshape(Bsz, self.H, self.P, L)      # free view
        A_log = self._A_log_kernel()

        h0 = None
        if h_init is not None or self.h_init_param is not None:
            h0 = self.initial_state(Bsz, h_init)[..., :self.d_state_hm].contiguous()

        pre = self.gate_pos == "pre" and self.gB_proj is not None

        def _m(key, idx=None):
            t = maps.get(key)
            if t is None:
                return None
            return self._rows_to_kernel(t, idx, 1).contiguous()

        eng = self.backend
        if eng == "auto":
            eng = "cuda" if x_cf.is_cuda else "torch"
        a_dyn, dyn_act = self._dyn_scan_input(
            _m("a_dyn", self.dyn_idx), eng, x_cf.dtype)
        kw = dict(
            # `dt_raw`: delta is the raw `dt_lin`; the kernel applies softplus and bias
            delta_softplus=bool(dt_raw),
            delta_bias=self.dt_bias if dt_raw else None,
            row_law=self.row_law, row_rot=self.row_rot,
            a_dyn=a_dyn, a_dyn_res=self.dyn_res, dyn_act=dyn_act,
            mu=_m("mu", self.mu_idx if self.mu_res == "n" else None),
            mu_gain=self._mu_gain_kernel(), mu_res=self.mu_res,
            theta=(None if maps.get("theta") is None else
                   self._pairs_to_kernel(self._theta_feature(_m("theta").to(
                       torch.float64 if x_cf.dtype == torch.float64 else torch.float32)),
                       dim=-2)),
            theta_delta=maps.get("theta") is not None,
            theta_h=self._theta_h_kernel(),
            g_B=None if pre else _m("g_B"), g_C=None if pre else _m("g_C"),
            gate_act=self._gate_act,
            beta_a=None,
            beta_b=self.beta_b, beta_c=self.beta_c,
            beta_gB=None if pre else self._gate_bias(self.beta_gB),
            beta_gC=None if pre else self._gate_bias(self.beta_gC),
            h_init=h0,
            theta_init=(None if complete_state is None else
                        self._pairs_to_kernel(complete_state.get("theta"))),
            mu_init=(None if complete_state is None else
                     self._rows_to_kernel(complete_state.get("mu"), self.mu_idx, 1)
                     if self.mu_res == "n" else complete_state.get("mu")),
            write_prev=None if complete_state is None else complete_state.get("write_prev"),
            d_skip=d_skip,
            z=None if z is None else z.reshape(Bsz, self.H, self.P, L),
            trap_lam=(sigmoid_exact(_m("trap").to(
                          torch.float64 if x_cf.dtype == torch.float64 else torch.float32))
                      if maps.get("trap") is not None else None),
        )

        if pre:
            # gate_pos="pre": one head-free fold on the (B, n, L) maps; the engines
            # then add the per-head bias and the rotation
            Bt = self._pre_gate(Bt, _m("g_B"), self._gate_bias(self.beta_gB))
            Ct = self._pre_gate(Ct, _m("g_C"), self._gate_bias(self.beta_gC))
        args = (x_t, delta.contiguous(), A_log, Bt.contiguous(),
                Ct.contiguous())
        self._route("hm", "mh_scan" if eng == "cuda" else
                    "mh_torch_scan(sequential)" if eng == "sequential"
                    else "mh_torch_scan")
        if eng == "cuda":
            from ..kernels.hybrid_mamba_mh.functions import mh_scan
            out = mh_scan(*args, ckpt=self.ckpt, mem=self.mem,
                          return_state=want_state,
                          kernel_options=self.kernel_options or None, **kw)
            Y, h_final = (out[0], out[1]) if want_state else (out, None)
        else:
            from ..kernels.hybrid_mamba_mh.torch_scan import mh_torch_scan
            Y, h_final = mh_torch_scan(*args, sequential=eng == "sequential", **kw)
            if not want_state:
                h_final = None
        y = Y.reshape(Bsz, self.d_inner, L)             # free view
        # the hidden carry stays in the scan's working dtype (fp32 under autocast); rounding it to the activation
        # dtype would hand back a bf16-valued carry
        return y.to(x_cf.dtype), h_final

    def _flat_state_scan(self, law, maps, x_cf, dt, Bt, Ct, state, start, end):
        """Bound rotating flat phases by time tiles, with differentiable carries."""
        if law != "m3" or self.m3_angle_proj is None:
            return self._flat_state_scan_chunk(law, maps, x_cf, dt, Bt, Ct, state, start, end)
        tile = _flat_phase_tile(x_cf.shape[0], self.H, (end - start) // 2)
        if x_cf.shape[-1] <= tile:
            return self._flat_state_scan_chunk(law, maps, x_cf, dt, Bt, Ct, state, start, end)
        from torch.utils.checkpoint import checkpoint
        wd = torch.float64 if x_cf.dtype == torch.float64 else torch.float32
        h = state["hidden_state"][..., start:end]
        phase, kp, vp = state["c3_phase"], state["c3_kp"], state["c3_vp"]

        def run(h, phase, kp, vp, x, clock, b, c, rate, trap, angle):
            # Only this family's hidden slice is read by the chunk scan.
            local = State(hidden_state=F.pad(h, (start, self.d_state - end)),
                          c3_phase=phase, c3_kp=kp, c3_vp=vp)
            local_maps = {"m3_a": rate, "m3_trap": trap, "m3_ang": angle}
            y, hf = self._flat_state_scan_chunk(law, local_maps, x, clock, b, c,
                                               local, start, end)
            feature = torch.tanh(angle.to(wd)) * math.pi
            terminal = phase_bmm(clock.to(wd), feature.transpose(1, 2))
            ph = phase.to(wd) + terminal
            key = self.m3_b_norm(b[..., -1]).to(wd)[:, None] + self.m3_B_bias.to(wd)
            e, o = key[..., 0::2], key[..., 1::2]
            phf = F.pad(ph, (0, (end - start) // 2 - ph.shape[-1]))  # real pairs: 0
            key = torch.stack((e*phf.cos()-o*phf.sin(), e*phf.sin()+o*phf.cos()), -1).flatten(-2)
            return y, hf, ph, key, x[..., -1].reshape(x.shape[0], self.H, self.P).to(wd)

        outputs = []
        for t in range(0, x_cf.shape[-1], tile):
            sl = slice(t, t + tile)
            args = (h, phase, kp, vp, x_cf[..., sl], dt[..., sl], Bt[..., sl], Ct[..., sl],
                    maps["m3_a"][..., sl], maps["m3_trap"][..., sl], maps["m3_ang"][..., sl])
            if self.ckpt and torch.is_grad_enabled() and any(a.requires_grad for a in args):
                y, h, phase, kp, vp = checkpoint(run, *args, use_reentrant=False)
            else:
                y, h, phase, kp, vp = run(*args)
            outputs.append(y)
        return torch.cat(outputs, -1), h

    def _flat_state_scan_chunk(self, law, maps, x_cf, dt, Bt, Ct, state, start, end):
        """Flat rows with arbitrary hidden and endpoint carries.  Native SSD folds the
        shared maps in its tiles; the first transported previous-write term is folded
        into the initial hidden state (adjoints stay in autograd)."""
        wd = torch.float64 if x_cf.dtype == torch.float64 else torch.float32
        batch, _, length = x_cf.shape
        n = end - start
        dt = dt.to(wd)
        v = x_cf.reshape(batch, self.H, self.P, length).to(wd)
        k = getattr(self, f"{law}_b_norm")(Bt.transpose(1, 2)).to(wd).transpose(1, 2)[:, None]
        q = getattr(self, f"{law}_c_norm")(Ct.transpose(1, 2)).to(wd).transpose(1, 2)[:, None]
        beta = getattr(self, f"{law}_B_bias").to(wd).unsqueeze(-1)
        gamma = getattr(self, f"{law}_C_bias").to(wd).unsqueeze(-1)
        native = self.backend == "cuda" or (self.backend == "auto" and x_cf.is_cuda)
        phase = theta = None
        lam = previous = None
        if law == "m3":
            log_a = -self._m3_rate(maps["m3_a"].to(wd)) * dt
            lam = sigmoid_exact(maps["m3_trap"].to(wd))
            previous = state["c3_vp"].to(wd)[..., None] * state["c3_kp"].to(wd)[:, :, None]
            if self.m3_angle_proj is not None:
                if native:
                    # SSD's fused phase law (no expanded angle input or cumsum tape)
                    angle = maps["m3_ang"].transpose(1, 2).to(wd).unsqueeze(2)
                    theta = _phase_law(angle, dt, None, self.H, n // 2, -1.)
                    c3 = state["c3_phase"].to(wd)
                    theta = theta - F.pad(c3, (0, n // 2 - c3.shape[-1]))[:, None]
                else:
                    inc = torch.tanh(maps["m3_ang"].to(wd))[:, None] * math.pi * dt[:, :, None]
                    inc = F.pad(inc, (0, 0, 0, n // 2 - inc.shape[-2]))
                    c3 = state["c3_phase"].to(wd)
                    phase = (F.pad(c3, (0, n // 2 - c3.shape[-1]))[..., None]
                             + inc.cumsum(-1))
        else:
            log_a, _, _ = self._flat_decay(law, dt)
        hidden = state["hidden_state"][..., start:end].to(wd)
        self._route(law, "ssd_mimo_scan" if native else "flat_torch_scan")
        if native:
            from ..kernels.ssd_mimo.autograd import ssd_mimo_scan
            # B/C keep their singleton group axis; SSD's rotation sign is opposite
            # to this module's positive folded phase (hence the -1. in the phase law)
            bk = k.permute(0, 3, 1, 2).unsqueeze(-1).to(x_cf.dtype).contiguous()
            cq = q.permute(0, 3, 1, 2).unsqueeze(-1).to(x_cf.dtype).contiguous()
            # The two-tap trapezoid runs in the scan:
            # delta_w is the raw dt and trap_lam the gate; only the carried previous
            # write folds into x0.
            if lam is not None:
                hidden = hidden + (dt[..., 0] * (1 - lam[..., 0]))[..., None, None] * previous
            y, final = ssd_mimo_scan(
                v.to(x_cf.dtype).unsqueeze(3).contiguous(), (-log_a).contiguous(),
                bk, cq, dt.new_zeros(self.H), delta_w=dt.contiguous(),
                trap_lam=None if lam is None else lam.contiguous(),
                beta=beta, gamma=gamma, theta=theta, bias_pre_rot=True,
                x0=hidden.contiguous(), bc_per_head=True, chunk_size=64,
                precision="ieee" if x_cf.dtype == torch.float32 else "io")
            y = y.reshape(batch, self.H, self.P, length)
        else:
            from ..kernels.hybrid_mamba_mh.flat_scan import flat_torch_scan
            y, final = flat_torch_scan(v, k, q, log_a, dt, hidden,
                                      lam=lam, previous=previous, beta=beta,
                                      gamma=gamma, phase=phase, ckpt=self.ckpt)
        return y.reshape(batch, self.d_inner, length).to(x_cf.dtype), final

    def _m3_half_cf(self, maps, x_cf, dt, Bt, Ct):
        """Evaluate M3 rows with maps in channel-major layout."""
        if self.m3_kernel == "upstream" and not x_cf.is_cuda:
            raise ValueError("kernel_options m3_kernel='upstream' batched forward requires CUDA")
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        Bsz, _, L = x_cf.shape
        H, P, n3 = self.H, self.P, self.d_state_m3
        # long rotating sequences take the time-tiled path (bounded phase memory)
        if self.m3_angle_proj is not None and self.m3_kernel == "native":
            tile = _flat_phase_tile(Bsz, H, n3 // 2)
            if L > tile:
                initial = ScanState(self).init_state(Bsz, device=x_cf.device, dtype=x_cf.dtype)
                y, _ = self._flat_state_scan("m3", maps, x_cf, dt, Bt, Ct, initial,
                                              self.d_state_hm, self.d_state_hm + n3)
                return y
        Bt = rms_norm_cast(Bt.transpose(1, 2), self.m3_b_norm.weight,
                           self.m3_b_norm.eps)                  # (B,L,n3)
        Ct = rms_norm_cast(Ct.transpose(1, 2), self.m3_c_norm.weight,
                           self.m3_c_norm.eps)
        A = -self._m3_rate(maps["m3_a"].to(wd))
        dtf = dt.to(wd)                                        # (B,H,L)
        decay = (-(A * dtf)).contiguous()                       # (B,H,L) >= 0
        trap = maps["m3_trap"].contiguous()                     # (B,H,L)
        lam = sigmoid_exact(trap.to(wd))
        v = x_cf.reshape(Bsz, H, P, L)                          # free view

        eng = self.backend
        if eng == "auto":
            eng = "cuda" if x_cf.is_cuda else "sequential"
        self._route("m3", "_m3_reference(fp64 oracle)" if eng != "cuda" else
                    "upstream_siso" if self.m3_kernel == "upstream"
                    else "ssd_mimo_scan")
        if eng != "cuda":
            if self.m3_angle_proj is not None:
                angles = (maps["m3_ang"].transpose(1, 2).unsqueeze(-2)
                          .expand(-1, -1, H, -1).to(wd))
                angles = F.pad(angles, (0, n3 // 2 - angles.shape[-1]))
            else:
                angles = torch.zeros(Bsz, L, H, n3 // 2, device=x_cf.device,
                                     dtype=wd)
            y = _m3_reference(Bt.unsqueeze(2), Ct.unsqueeze(2),
                              v.permute(0, 3, 1, 2), -decay, dtf, trap,
                              self.m3_B_bias, self.m3_C_bias, angles)
            return y.reshape(Bsz, L, self.d_inner).transpose(
                1, 2).to(x_cf.dtype)

        if self.m3_kernel == "upstream":
            from ..kernels.m3_adapter import siso
            angles = (maps["m3_ang"].transpose(1, 2).unsqueeze(2)
                      .expand(Bsz, L, H, -1).float()
                      if self.m3_angle_proj is not None else
                      dt.new_zeros(Bsz, L, H, n3 // 2, dtype=torch.float32))
            angles = F.pad(angles, (0, n3 // 2 - angles.shape[-1]))
            y = siso(Q=Ct.unsqueeze(2), K=Bt.unsqueeze(2),
                     V=v.permute(0, 3, 1, 2).contiguous(),
                     ADT=-decay, DT=dtf, Trap=trap.float(),
                     Q_bias=self.m3_C_bias, K_bias=self.m3_B_bias,
                     Angles=angles.contiguous(), chunk_size=64)
            return y.reshape(Bsz, L, self.d_inner).transpose(1, 2).to(x_cf.dtype)

        from ..kernels.ssd_mimo.autograd import ssd_mimo_scan
        theta = None
        if self.m3_angle_proj is not None:
            # SSD's fused phase kernel, as `SSDMimoSSM`
            ang = maps["m3_ang"].transpose(1, 2).to(wd)          # (B,L,ka)
            theta = _phase_law(ang.unsqueeze(2), dtf.contiguous(), None, H,
                              n3 // 2, -1.)
        # B/C stay head-shared; the bias and phase are folded in the scan tiles
        bk, ck = Bt[:, :, None, :, None], Ct[:, :, None, :, None]
        beta, gamma = self.m3_B_bias.unsqueeze(-1), self.m3_C_bias.unsqueeze(-1)
        # the two-tap trapezoid runs in the scan (raw dt + trap_lam)
        y, _ = ssd_mimo_scan(
            v.unsqueeze(3), decay, bk, ck, self.const_a_log,
            chunk_size=64, delta_w=dtf.contiguous(), trap_lam=lam.contiguous(),
            bc_per_head=True, beta=beta, gamma=gamma, theta=theta,
            theta_increment=False, bias_pre_rot=theta is not None)
        return y.reshape(Bsz, self.d_inner, L).to(x_cf.dtype)

    # ── m2 flat rows ─────────────────
    def _flat_decay(self, law, dt):
        """(log_a, decay_for_kernel, write_for_kernel) in (B,H,L), fp32.
        m2: decay = Delta_t on the static rate, the write is Delta_t too
        (write_for_kernel None = the kernel's own decay-side Delta)."""
        # fp64 stays fp64 (the CPU oracle route); everything else fp32
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        A_log = getattr(self, f"{law}_A_log").to(wd)
        dtf = dt.to(wd)
        decay, write = dtf, None
        log_a = -(decay * A_log.exp()[None, :, None])
        return log_a, decay, write

    def _flat_half_cf(self, law, x_cf, dt, Bt, Ct):
        """The m2 rows, tensors as (B, ., L): one `ssd_mimo_scan` launch with the
        static per-head rate as `a_log`, or the fp64 reference off CUDA."""
        Bsz, _, L = x_cf.shape
        H, P = self.H, self.P
        bn, cn = getattr(self, f"{law}_b_norm"), getattr(self, f"{law}_c_norm")
        bb, cb = getattr(self, f"{law}_B_bias"), getattr(self, f"{law}_C_bias")
        Bt = rms_norm_cast(Bt.transpose(1, 2), bn.weight, bn.eps)   # (B,L,n)
        Ct = rms_norm_cast(Ct.transpose(1, 2), cn.weight, cn.eps)
        log_a, decay, write = self._flat_decay(law, dt)
        v = x_cf.reshape(Bsz, H, P, L)                          # free view
        eng = self.backend
        if eng == "auto":
            eng = "cuda" if x_cf.is_cuda else "sequential"
        self._route(law, "ssd_mimo_scan" if eng == "cuda"
                    else "_flat_reference(fp64 oracle)")
        if eng != "cuda":
            y = _flat_reference(Bt, Ct, v.permute(0, 3, 1, 2), log_a,
                                decay if write is None else write, bb, cb)
            return y.reshape(Bsz, L, self.d_inner).transpose(1, 2).to(x_cf.dtype)
        from ..kernels.ssd_mimo.autograd import ssd_mimo_scan
        y, _ = ssd_mimo_scan(
            v.unsqueeze(3), decay.float().contiguous(),
            Bt.to(x_cf.dtype).contiguous(), Ct.to(x_cf.dtype).contiguous(),
            getattr(self, f"{law}_A_log").float(), chunk_size=64,
            delta_w=None if write is None else write.float().contiguous(),
            beta=bb.unsqueeze(-1), gamma=cb.unsqueeze(-1))       # (B,H,P,L)
        return y.reshape(Bsz, self.d_inner, L).to(x_cf.dtype)

    def _flat_step(self, law, x_t, dt, Bt, Ct, state, new_state):
        """M2 diagonal recurrence; no sequence allocation or oracle call."""
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        Bsz, H, P = x_t.shape[0], self.H, self.P
        k = getattr(self, f"{law}_b_norm")(Bt).to(wd)[:, None] + getattr(self, f"{law}_B_bias").to(wd)
        q = getattr(self, f"{law}_c_norm")(Ct).to(wd)[:, None] + getattr(self, f"{law}_C_bias").to(wd)
        rate = getattr(self, f"{law}_A_log").to(wd).exp()
        alpha = torch.exp(-rate * dt.to(wd))
        start = self.d_state_hm + self.d_state_m3
        end = start + getattr(self, f"d_state_{law}")
        old = state["hidden_state"][..., start:end].to(wd)
        hidden = alpha[..., None, None] * old + dt.to(wd)[..., None, None] * x_t.reshape(Bsz, H, P).to(wd)[..., None] * k[:, :, None]
        new_state["_hidden_update"] = hidden.to(state["hidden_state"].dtype)
        y = (hidden * q[:, :, None]).sum(-1)
        return y.reshape(Bsz, self.d_inner).to(x_t.dtype), new_state

    def _flat_offsets(self):
        """Row offsets of the flat tail: (m3, m2) starts."""
        o3 = self.d_state_hm
        o2 = o3 + self.d_state_m3
        return o3, o2

    def _forward_cf(self, x_cf, h_init=None, sel_x=None, want_state=False,
                   tok_raw=None, z_cf=None):
        """Channel-major forward: (B, d_inner, L) in and out (as `scan`, transposed).

        ``tok_raw`` is a pre-projected token slab (see `_scan_cf`).  ``z_cf``
        (B, d_inner, L) offers the SwiGLU gate to the scan kernel: afterwards
        ``self._h3_gate_in_kernel`` True means ``y`` is already ``y * silu(z)`` and
        the caller must not gate again.  Only offer a channel-major `z`
        (`HybridMambaSSM._h3_gate_offered`)."""
        self._route_begin(x_cf.is_cuda)
        if x_cf.shape[-1] == 0 or (self.fracs.n_flat and (
                self.m3_kernel == "native" or want_state or h_init is not None
                or self.h_init_param is not None)):
            self._h3_gate_in_kernel = False
            state = (State.from_hidden(self, h_init, batch_size=x_cf.shape[0])
                     if h_init is not None else None)
            y, state_final = self.scan(x_cf.transpose(1, 2), state=state,
                                         return_state=True, sel_x=sel_x, tok_raw=tok_raw)
            return y.transpose(1, 2), state_final.hidden_state
        y, h_final = self._scan_cf(x_cf, h_init, sel_x, want_state, tok_raw,
                                  z_cf=z_cf)
        if self.skip and not self._k11_skip_in_kernel:
            xd = x_cf * self._skip_channels().to(
                x_cf.dtype).unsqueeze(-1)
            # a module-side skip precedes the gate, so the gate was not in-kernel either
            y = y + xd
        return y, h_final

    def _run(self, x, h_init, sel_x, want_state, tok_raw=None):
        """Adapt B,L,D input/output to the internal channel-major scan.

        Apply D once if the scan did not fuse it. The output may be a view;
        callers must not assume contiguous token-major storage."""
        y, h_final = self._scan_cf(x.transpose(1, 2).contiguous(), h_init,
                                  sel_x, want_state, tok_raw)
        y = y.transpose(1, 2)
        if self.skip and not self._k11_skip_in_kernel:
            y = y + x * self._skip_channels().to(x.dtype)
        self.h_final = h_final
        return y, h_final

    def scan(self, x: torch.Tensor, *, sel_x: torch.Tensor | None = None, tok_raw: torch.Tensor | None = None,
             state: State | dict | None = None,
             return_state: bool = False) -> torch.Tensor | tuple[torch.Tensor, State]:
        """Return output only, or (output, complete State) when requested."""
        self._route_begin(x.is_cuda)
        if state is None and not return_state and self.fracs.n_flat == 0 and x.shape[1] > 0:
            return self._run(x, None, sel_x, want_state=False, tok_raw=tok_raw)[0]
        if x.shape[1] > 0 and self.backend != "sequential" and self.m3_kernel != "upstream":
            initial = ScanState(self).resolve_state(state, batch_size=x.shape[0], device=x.device,
                                         dtype=x.dtype)
            y, hidden, final = self._scan_cf(x.transpose(1, 2).contiguous(),
                initial.hidden_state if isinstance(initial, State) else initial["hidden_state"],
                sel_x, True, tok_raw, complete_state=initial)
            y = y.transpose(1, 2)
            if self.skip and not self._k11_skip_in_kernel:
                y = y + x * self._skip_channels().to(x.dtype)
            if return_state:
                final = ScanState(self)._conform_carries(final, state)
            self._final_state, self.h_final = final, hidden
            return (y, final) if return_state else y
        return self._forward_stepped(x, state, sel_x, return_state, tok_raw)

    def _forward_stepped(self, x, state, sel_x, return_state,
                         tok_raw=None):
        Bsz, L, _ = x.shape
        if L:
            # O(L) python loop of `step`: backend="sequential", or m3_kernel="upstream"
            # with flat rows or a State; never a kernel, so always named
            self._route("all", "stepped(per-token recurrence)")
        state = ScanState(self).resolve_state(state, batch_size=Bsz, device=x.device,
                                   dtype=x.dtype)
        ref = state
        ys = []
        for t in range(L):
            y_t, state = self.scan_step(
                x[:, t], state,
                sel_x=sel_x[:, t] if sel_x is not None else None,
                tok_raw=tok_raw[:, t] if tok_raw is not None else None)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) if ys else x.new_empty(Bsz, 0, self.out_width)
        state = ScanState(self)._conform_carries(state, ref)
        self._final_state = state
        self.h_final = state["hidden_state"]
        if return_state:
            return y, state
        return y

# ── fused in_proj for the layer ──

class _HMInProjFn(torch.autograd.Function):
    """Project drive and gate with one autograd node.

    The drive is channel-major, the gate uses its consumer layout, and backward
    combines their input gradients without a separate activation-sized sum."""

    @staticmethod
    def forward(ctx, xa, w, d_inner, z_cf=False):
        with torch.amp.autocast("cuda", enabled=False):
            dt, B = xa.dtype, xa.shape[0]
            wx = w[:d_inner].to(dt)
            wz = w[d_inner:].to(dt)
            xat = xa.transpose(1, 2)
            x_cf = torch.bmm(wx.expand(B, *wx.shape), xat)
            z = (torch.bmm(wz.expand(B, *wz.shape), xat) if z_cf
                 else F.linear(xa, wz))
        ctx.save_for_backward(xa, wx, wz)
        ctx.d_inner = d_inner
        ctx.z_cf = z_cf
        ctx.w_meta = (w.shape, w.dtype)
        return x_cf, z

    @staticmethod
    def backward(ctx, dx_cf, dz):
        xa, wx, wz = ctx.saved_tensors
        B, L, dm = xa.shape
        di = ctx.d_inner
        dx_cf = dx_cf if dx_cf.is_contiguous() else dx_cf.contiguous()
        dz = dz if dz.is_contiguous() else dz.contiguous()
        x_needs, w_needs = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dxa = None
        if x_needs:
            # token-major x gradient, so the z half accumulates into it in place
            dxa = torch.bmm(dx_cf.transpose(1, 2), wx.expand(B, di, dm))
            if ctx.z_cf:
                dxa.baddbmm_(dz.transpose(1, 2), wz.expand(B, di, dm))
            else:
                dxa = dxa.reshape(-1, dm)
                dxa.addmm_(dz.reshape(-1, di), wz)
                dxa = dxa.view(B, L, dm)
        dw = None
        if w_needs:
            shape, wdtype = ctx.w_meta
            dw = torch.empty(shape, dtype=wdtype, device=xa.device)
            dw[:di].copy_(torch.bmm(dx_cf, xa).sum(0))
            dw[di:].copy_(torch.bmm(dz, xa).sum(0) if ctx.z_cf
                          else dz.reshape(-1, di).t() @ xa.reshape(-1, dm))
        return dxa, dw, None, None


class _HMInProjTokFn(torch.autograd.Function):
    """`_HMInProjFn` plus the core's token maps (``proj_source="layer_input"``).

    Drive, gate and token slab are three products of one input: the drive and slab
    are returned channel-major, the gate in its consumer layout, and the backward
    sums the three input gradients into one token-major buffer."""

    @staticmethod
    def forward(ctx, xa, w, w_tok, d_inner, z_cf=False):
        with torch.amp.autocast("cuda", enabled=False):
            dt, B = xa.dtype, xa.shape[0]
            wx = w[:d_inner].to(dt)
            wz = w[d_inner:].to(dt)
            wt = w_tok.to(dt)
            xat = xa.transpose(1, 2)
            x_cf = torch.bmm(wx.expand(B, *wx.shape), xat)
            z = (torch.bmm(wz.expand(B, *wz.shape), xat) if z_cf
                 else F.linear(xa, wz))
            tok_cf = torch.bmm(wt.expand(B, *wt.shape), xat)
        ctx.save_for_backward(xa, wx, wz, wt)
        ctx.d_inner = d_inner
        ctx.z_cf = z_cf
        ctx.w_meta = (w.shape, w.dtype, w_tok.dtype)
        return x_cf, z, tok_cf

    @staticmethod
    def backward(ctx, dx_cf, dz, dtok):
        xa, wx, wz, wt = ctx.saved_tensors
        B, L, dm = xa.shape
        di = ctx.d_inner
        zeros = None
        if dx_cf is None:
            dx_cf = zeros = xa.new_zeros(B, di, L)
        if dz is None:
            dz = xa.new_zeros(*((B, di, L) if ctx.z_cf else (B, L, di)))
        dx_cf = dx_cf if dx_cf.is_contiguous() else dx_cf.contiguous()
        dz = dz if dz.is_contiguous() else dz.contiguous()
        if dtok is not None and not dtok.is_contiguous():
            dtok = dtok.contiguous()
        x_needs, w_needs, t_needs = ctx.needs_input_grad[:3]
        dxa = None
        if x_needs:
            dxa = torch.bmm(dx_cf.transpose(1, 2), wx.expand(B, di, dm))
            if ctx.z_cf:
                dxa.baddbmm_(dz.transpose(1, 2), wz.expand(B, di, dm))
            else:
                dxa = dxa.reshape(-1, dm)
                dxa.addmm_(dz.reshape(-1, di), wz)
                dxa = dxa.view(B, L, dm)
            if dtok is not None:
                dxa.baddbmm_(dtok.transpose(1, 2), wt.expand(B, *wt.shape))
        dw = dwt = None
        shape, wdtype, tdtype = ctx.w_meta
        if w_needs:
            dw = torch.empty(shape, dtype=wdtype, device=xa.device)
            if zeros is None:
                dw[:di].copy_(torch.bmm(dx_cf, xa).sum(0))
            else:
                dw[:di].zero_()
            dw[di:].copy_(torch.bmm(dz, xa).sum(0) if ctx.z_cf
                          else dz.reshape(-1, di).t() @ xa.reshape(-1, dm))
        if t_needs:
            dwt = (torch.bmm(dtok, xa).sum(0).to(tdtype) if dtok is not None
                   else torch.zeros_like(wt, dtype=tdtype))
        return dxa, dw, dwt, None, None


def _hm_in_proj(m, xa, z_cf=False):
    """``(x_cf, z)``: `in_proj` applied as the two halves.  The fused node is
    skipped under ``fuse_in_proj``, on CPU and outside grad mode.  ``z_cf``
    returns z channel-major (B, d_inner, L), as `mh_scan`'s fused gate takes it."""
    if (hm_glue_on("inproj") and not m.fuse_in_proj and xa.is_cuda
            and torch.is_grad_enabled()
            and (xa.requires_grad or m.in_proj.weight.requires_grad)):
        return _HMInProjFn.apply(xa, m.in_proj.weight, m.d_inner, z_cf)
    W = m.in_proj.weight
    x_cf = matmul_channel_major(W[:m.d_inner], xa, grad_layout="token")
    wz = W[m.d_inner:]
    return x_cf, (matmul_channel_major(wz, xa, grad_layout="token") if z_cf
                  else F.linear(xa, wz))


_LAYER_OWN = frozenset({"in_proj", "conv1d", "norm", "out_proj"})


@ssm_status("production", since="2026-08",
            kernel="mh_scan (hybrid_mamba_mh) + "
                   "ssd_mimo_scan (d_ssm_in = d_ssm_out = 1) per-frac dispatch",
            reason="The one SSM that calls BOTH kernel families. "
                   "State rows split by fracs between "
                   "mh_scan (laws A/dyn/mu, rotation, gates, betas) and "
                   "ssd_mimo_scan (flat M3/M2 laws); "
                   "outputs summed, which is exact because y is a sum over "
                   "rows.")
class HybridMambaSSM(BaseSSM, _HybridMambaScan):
    """HybridMamba layer: a state-space mixer whose state rows follow different decay laws.

    The layer is ``in_proj -> causal drive conv + SiLU -> scan -> gate -> [norm] ->
    out_proj``, the same stack as `SSDMimoSSM`.  The scan splits the ``d_state``
    state rows into blocks (``row_blocks``); each block has its own decay law and
    runs on the kernel that serves it.  The per-coordinate laws ``"coord_static"``, ``"coord_dynamic"``
    and ``"coord_potential"`` run in one ``mh_scan`` launch.  The per-head laws
    ``"head_dynamic"`` (Mamba-3 style: selected rate, trapezoid, phase) and
    ``"head_static"`` (Mamba-2 style: learned static rate) each run in one
    ``ssd_mimo_scan`` launch.  Block outputs are summed and the D skip is applied
    once.  The scan is width-preserving (output width ``d_inner``), so the
    SwiGLU gate ``y * silu(z)`` is applied on the module side.

    Parameters
    ----------
    config : HybridMambaConfig or dict
        The architecture.  Every field, its meaning, valid values and default are documented on
        `HybridMambaConfig`; a dict goes through `HybridMambaConfig.from_dict` (strict).  The layer width is the config's
        ``d_model`` (``d_inner`` when the config names no ``expand``); `HybridMambaConfig.build(d_model)` and
        `CausalLM` supply it otherwise.
    execution : Execution or dict, optional
        How the layer runs; see "Execution" below.  None keeps the defaults.

    Execution
    ---------
    How the layer runs is an `Execution` (or its dict), never part of the architecture: ``backend``,
    ``checkpoint`` and ``kernel_options`` (the keys this layer reads are listed at the end); ``precision`` is
    refused, ``chunk_size`` too unless it is ``"auto"`` (the MH scan chooses its chunking itself).

    * At construction: ``HybridMambaSSM(config, execution={"backend": "torch"})``.
    * At runtime: ``layer.set_execution({"precision": "ieee"})`` (only the fields given change).
    * Temporarily: ``with layer.execution({"backend": "torch"}): y = layer(x)`` (restored on exit).

    Inside a `CausalLM` or `BlockStack`, address one layer through ``lm.ssm_layers[i]``.

    Notes
    -----
    Checkpoints carry int64 markers for the |A| activations (``a_act_code``,
    ``dyn_act_code``) and the selection source (``proj_source_code``).  A load
    refuses a checkpoint whose declared values differ from the module's (rebuild with
    the checkpoint's ``proj_source`` / ``a_act`` / ``dyn_act``).  The SSD-MIMO options ``omega_layout`` and
    ``theta_init`` are refused here unless left at their defaults.
    """

    config_family = "hybrid_mamba"

    # ---- Execution: backend, kernel_options, checkpoint (-> ``ckpt``); precision / chunk_size are
    # refused (`_execution_fields`).  `kernel_options` carries the kernel levers and the layer switches
    # (kernel_options.LAYER_KEYS["hybrid_mamba"]): m3_kernel, mem, compile_gate.
    # The kernel part replaces `self.kernel_options`; named switches change, others stay.
    _layer_option_attrs = {"m3_kernel": "m3_kernel", "mem": "mem", "compile_gate": "_compile_gate",
                           "norm_kernel": "norm_kernel", "conv_kernel": "conv_kernel"}

    _execution_fields = ("backend", "checkpoint", "kernel_options")
    _execution_attr = {"checkpoint": "ckpt"}
    # chunk_size="auto" is what the MH scan does and is accepted; precision has no single fixed value (the scan runs
    # "ieee" operands for an fp32 input, "io" for 16-bit) and is refused whatever it says
    _execution_fixed = {"chunk_size": "auto"}
    _execution_hints = {"precision": "the MH scan has no operand-mode knob",
                        "chunk_size": "the MH scan chooses its chunking itself (kernel_options carries the levers)"}

    def _check_execution(self, ex):
        self._refuse_unsupported_execution(ex)
        f, out = ex.fields(), {}
        if "backend" in f:
            out["backend"] = check_backend(f["backend"], "HybridMambaSSM", BACKENDS)
            if out["backend"] in ("auto", "cuda") and (self.P & (self.P - 1)):
                # the Gluon backward needs a power-of-two P (else a CompilationError on the first backward)
                raise ValueError(
                    f"HybridMambaSSM: execution backend {out['backend']!r} needs a power-of-two "
                    f"d_head=d_inner/n_heads (got d_head={self.P}, d_inner={self.d_inner}, "
                    f"n_heads={self.H}); pick n_heads so that d_head is a power of two "
                    f"(d_head=1, i.e. n_heads=d_inner, is Mamba-1's per-channel dt)")
        kernel, lay = split_kernel_options(self, "hybrid_mamba", f.get("kernel_options"))
        if kernel is not None:
            out["kernel_options"] = kernel
        if lay["m3_kernel"] == "upstream" and self._upstream_refusal:
            raise ValueError(self._upstream_refusal)
        lay["mem"] = dict(lay["mem"]) if lay["mem"] else None
        from ..kernels.m3_adapter import check_provider
        check_provider(lay["m3_kernel"], out.get("backend", self.backend))
        check_norm_kernel(lay["norm_kernel"])
        check_conv_kernel(lay["conv_kernel"])
        if "checkpoint" in f:
            out["ckpt"] = bool(f["checkpoint"])
        out.update(lay)
        return out

    # Public names for the paper-notation attributes (read-only; kernel code keeps .H .P .N ...).
    d_head, n_heads = read_alias("P"), read_alias("H")
    d_ssm_in = d_ssm_out = property(lambda self: 1, doc="Read-only: 1 (this family has a single subsystem, R = O = 1).")

    def __init__(self, config: "HybridMambaConfig | dict", execution=None):
        """See the class docstring: the architecture is the `HybridMambaConfig` (or its strict dict), how the
        layer runs is ``execution`` (`Execution` or dict)."""
        config = as_family_config(config, HybridMambaConfig)
        kw = config.layer_kwargs()
        kw["d_model"] = config.layer_width()
        self._build(execution=execution, **kw)

    def _build(self, d_model, d_state=64, expand=None, d_conv=4, *, d_inner=None, execution=None,
               **core_options):
        """The constructor body: the config's keywords, resolved and validated by the scan."""
        nn.Module.__init__(self)
        from ..execution import Execution, as_execution
        # the default backend is checked too (P must be a power of two for "auto")
        _ex = as_execution(execution)
        _ex = dataclasses.replace(_ex, backend=_ex.backend or "auto")

        valid = (signature_names(HybridMambaSSM._build, _HybridMambaScan.__init__) | {"out_norm", "write_rule"}) - {"use_trap"}
        reject_unknown_kwargs([k for k in core_options if k not in valid],
                              "HybridMambaSSM", valid=valid)
        # the config names the quadrature; the scan keeps its boolean
        rule = core_options.pop("write_rule", None)
        core_options["use_trap"] = (None if rule is None else
                                    resolve_write_rule(rule, "HybridMambaSSM") == "trapezoid")

        from ..configuration import resolve_mixer_width
        d_inner = resolve_mixer_width(d_model, expand, d_inner)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self._compile_gate = False
        self.norm_kernel = "triton"

        # PyTorch's default nn.Linear init, as Mamba-3's in_proj and SSDMimoSSM
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

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
            row_blocks=[{"frac": 1.0, "decay": "coord_static"}],
            d_model=d_model, d_conv=d_conv,
        )
        # default heads from a fixed d_head: the CUDA backward needs P = d_inner/n_heads
        # to be a power of two, which a fixed n_heads breaks at most widths
        # out_norm=False: Mamba-1's epilogue (y*silu(z) -> out_proj); True = gated RMSNorm
        out_norm = bool(kw.pop("out_norm", False))
        if kw.get("n_heads") is None and kw.get("d_head") is None:
            kw["d_head"] = 64
        _HybridMambaScan.__init__(self, **kw)

        # _epilogue consumes the norm weight directly to fuse gate and RMSNorm.
        self.norm = RMSNorm(self.out_width) if out_norm else nn.Identity()
        # a BlockStack applies the residual-depth rescale
        self.out_proj = nn.Linear(self.out_width, d_model, bias=False)
        # the layer's recipe: the scan's recorded options plus the shared head
        model = {**self._resolved_constructor_config["model"], "out_norm": out_norm}
        # the coord rows' rule (None when there are none: the flat laws' rules are fixed, see `write_rules`)
        _ut = model.pop("use_trap")
        model["write_rule"] = None if _ut is None else "trapezoid" if _ut else "euler"
        record_constructor_config(
            self, locals(), constructor=HybridMambaSSM._build, exclude=("execution",), overrides=model)
        self.set_execution(_ex)

    @property
    def fuse_in_proj(self) -> bool:
        """True iff the token maps can join `in_proj` (``proj_source="layer_input"``).
        Read per call."""
        return self.proj_source == "layer_input"

    @property
    def write_rules(self) -> dict:
        """The write rule that really runs per row group: ``"coord"`` (the ``coord_*`` rows, the config's ``write_rule``,
        default trapezoid), ``"head_dynamic"`` (always trapezoid) and ``"head_static"`` (always euler); only the groups present."""
        out = {}
        if self.d_state_hm:
            out["coord"] = "trapezoid" if self.use_trap else "euler"
        if self.d_state_m3:
            out["head_dynamic"] = "trapezoid"
        if self.d_state_m2:
            out["head_static"] = "euler"
        return out

    _rg: frozenset = frozenset()     # HM_HELD_REGEN items live for this forward

    @_portable.scoped_forward
    def forward(self, x: torch.Tensor, *, state=None,
               return_state: bool = False):
        """Run the layer over ``x`` (B, L, d_model): in_proj, the scan, gate and out_proj.  ``state`` continues
        from a previous call (one `State`, from `init_state` or ``return_state=True``); ``return_state=True``
        returns ``(y, State)``, else ``y`` (B, L, d_model)."""
        rg = hm_held_regen() if (state is None and not return_state) else frozenset()
        self._rg = rg
        try:
            with _regen.scope(bool(rg)):
                return self._forward_impl(x, state=state, return_state=return_state)
        finally:
            self._rg = frozenset()

    def _forward_impl(self, x: torch.Tensor, *, state=None,
                      return_state: bool = False):
        """Run the mixer with a channel-major drive and optional complete state.

        Token maps can join in_proj only with layer-input selection. Carry
        routes preserve mixer convolution history and use the core state API.
        Gate layout follows the selected scan or gated-normalization consumer."""
        if state is not None:
            state = self.resolve_state(state, batch_size=x.shape[0], device=x.device,
                                       dtype=x.dtype)
        W = self.in_proj.weight
        xa = autocast_cast(x)
        # grad_layout="token": two-consumer gradient accumulation as in `SSDMimoSSM.forward`;
        # `_hm_in_proj` folds it into one node where it can.
        tok_raw = None
        # The layer offers its SwiGLU gate to the scan store, which needs the z half of
        # `in_proj` channel-major, so the decision precedes the projection; the core's
        # answer (`_h3_gate_in_kernel`) is read after the scan.
        want_state = return_state or state is not None
        # layer-input selection: drive, gate and token slab from one node
        # (`_HMInProjTokFn`); the carry route keeps the token-major slab.
        tok_node = (self.fuse_in_proj and not want_state and hm_glue_on("inproj")
                    and xa.is_cuda and torch.is_grad_enabled()
                    and (xa.requires_grad or W.requires_grad))
        want_z_cf = self._h3_gate_offered(tok_node) and not want_state
        if tok_node:
            x_cf, z, tok_cf = _HMInProjTokFn.apply(
                xa, W, self.tok_proj_weight(), self.d_inner, want_z_cf)
            # the core reads `tok_raw.transpose(1, 2)`: the contiguous channel-major slab
            tok_raw = tok_cf.transpose(1, 2)
        elif not self.fuse_in_proj:
            x_cf, z = _hm_in_proj(self, xa, z_cf=want_z_cf)
        else:
            x_cf = matmul_channel_major(W[:self.d_inner], xa,
                                        grad_layout="token")      # (B,D,L)
        if self.fuse_in_proj and not tok_node:
            # layer_input: the token maps read the block input, so z and all token
            # maps ride one d_model -> (d_inner + tok_width) GEMM
            zt = F.linear(xa, torch.cat(
                [W[self.d_inner:], self.tok_proj_weight()], dim=0))
            z, tok_raw = zt.split([self.d_inner, self.tok_width], -1)

        conv_tail = None
        if self.conv1d is not None:
            kc = self.d_conv
            # step()'s (B, C, d_conv) convention (index -1 = latest token): the
            # non-oldest slots are the d_conv-1 raw inputs preceding this chunk
            prefix = (state["conv_state"][:, :, 1:].to(x_cf.dtype)
                     if state is not None else None)
            x_in = x_cf if prefix is None else torch.cat([prefix, x_cf], -1)
            # channel-major entry: conv + SiLU in one launch, no transpose
            x_cf = causal_depthwise_conv1d_cl(
                x_in, self.conv1d.weight, self.conv1d.bias,
                kc, activation="silu", package=self._conv_impl)
            if prefix is not None:
                # causal conv: dropping the prefix outputs leaves this chunk's continuation
                x_cf = x_cf[..., prefix.shape[-1]:]
            x_cf = x_cf.contiguous()
            if "x" in self._rg and prefix is None:
                # HM_HELD_REGEN=x: rebuild this conv output in the backward
                _acd = (torch.is_autocast_enabled("cuda"),
                        torch.get_autocast_dtype("cuda"))
                _cw, _cb, _pkg = self.conv1d.weight, self.conv1d.bias, self._conv_impl

                def _x_regen(x_in=x_in, _cw=_cw, _cb=_cb, kc=kc, _acd=_acd, _pkg=_pkg):
                    with torch.autocast("cuda", dtype=_acd[1], enabled=_acd[0]):
                        return causal_depthwise_conv1d_cl(
                            x_in, _cw, _cb, kc, activation="silu", package=_pkg).contiguous()
                _regen.register(x_cf, _x_regen)
            if want_state:
                tail = x_in[..., -kc:]
                if tail.shape[-1] < kc:          # chunk shorter than d_conv
                    tail = torch.cat(
                        [tail.new_zeros(*tail.shape[:-1],
                                        kc - tail.shape[-1]), tail], dim=-1)
                conv_tail = tail if torch.is_grad_enabled() else tail.detach().clone()
        # selection input: the block input (layer_input) or the post-conv drive (main_branch)
        sel = xa if self.proj_source == "layer_input" else None
        if not want_state:
            y_cf, _ = self._forward_cf(
                x_cf, sel_x=sel, want_state=False, tok_raw=tok_raw,
                z_cf=z if want_z_cf else None)
            if getattr(self, "_h3_gate_in_kernel", False):
                # `y_cf` is already `y * silu(z)`; only the layout transition remains
                from ..layers.tails import (gate_transpose)
                return self.out_proj(gate_transpose(y_cf))
            if want_z_cf:
                # the core declined the gate offer; `gate_transpose` reads z through its strides
                z = z.transpose(1, 2)
            return self.out_proj(self._epilogue_cf(y_cf, z))
        # carry route: the core's per-token recurrence, channel-last
        ssm_state = (State({k: v for k, v in state.items() if k != "conv_state"})
                    if state is not None else None)
        y, out_ssm_state = self.scan(
            x_cf.transpose(1, 2).contiguous(), sel_x=sel, tok_raw=tok_raw,
            state=ssm_state, return_state=True)
        y = self.out_proj(self._epilogue(y, z))
        if not return_state:
            return y
        new_state = State(out_ssm_state)
        if conv_tail is not None:
            new_state["conv_state"] = conv_tail
        new_state = self._conform_carries(new_state, state)
        self._final_state = new_state
        return y, new_state


    def _epilogue(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply the gate and optional output normalization, fusing RMSNorm when possible."""
        if isinstance(self.norm, RMSNorm):
            return gated_rms_norm(y, self.norm.weight, self.norm.eps, z=z, kernel=self.norm_kernel)
        return self.norm(swiglu_gate(y, z, self._compile_gate))

    def _h3_gate_offered(self, tok_node: bool = False) -> bool:
        """Whether to project a channel-major gate for the scan: only the norm-free,
        HM-only route benefits, and a fused layer-input projection only via
        `_HMInProjTokFn` (`tok_node`).  The core makes the final check."""
        return (hm_glue_on("zk")
                and (tok_node or not self.fuse_in_proj)
                and not self._compile_gate
                and isinstance(self.norm, nn.Identity)
                and hasattr(self, "_forward_cf")
                and getattr(self, "d_state_m3", 1) == 0
                and getattr(self, "d_state_hm", 0) > 0)

    def _epilogue_cf(self, y_cf: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Convert channel-major scan output for the mixer epilogue.

        With RMSNorm, gate inside the norm; without it, fuse the gate into
        the layout conversion. Torch fallback preserves the same formula."""
        from ..layers.tails import (gate_transpose)
        if isinstance(self.norm, RMSNorm):
            return gated_rms_norm(gate_transpose(y_cf), self.norm.weight,
                                  self.norm.eps, z=z, kernel=self.norm_kernel)
        if self._compile_gate or not isinstance(self.norm, nn.Identity):
            return self._epilogue(gate_transpose(y_cf), z)
        out = gate_transpose(y_cf, z)
        if "out" in self._rg:
            # HM_HELD_REGEN=out: rebuild out_proj's saved input from y_cf / z
            _acd = (torch.is_autocast_enabled("cuda"),
                    torch.get_autocast_dtype("cuda"))

            def _out_regen(y_cf=y_cf, z=z, _acd=_acd):
                with torch.autocast("cuda", dtype=_acd[1], enabled=_acd[0]):
                    return gate_transpose(y_cf, z)
            _regen.register(out, _out_regen)
        return out

    def _init_state(self, batch_size: int, device=None, dtype=torch.float32) -> State:
        dev = device or next(self.parameters()).device
        state = State(self._scan_init_state(batch_size, device=dev, dtype=dtype))
        if self.conv1d is not None:
            state["conv_state"] = torch.zeros(
                batch_size, self.d_inner, self.d_conv, device=dev, dtype=dtype)
        return state

    @_portable.scoped_forward
    def step(self, x_t: torch.Tensor, state=None) -> tuple[torch.Tensor, State]:
        """One token: ``x_t`` (B, d_model) and the carried ``state`` (None starts from `init_state`) give
        ``(y_t (B, d_model), State)``; equals `forward` on the sequence up to rounding."""
        state = self.resolve_state(state, batch_size=x_t.shape[0], device=x_t.device,
                                   dtype=x_t.dtype)
        ref = state                               # the returned carries keep the incoming dtypes
        xz = self.in_proj(x_t)
        x_branch, z = xz.split([self.d_inner, self.d_inner], dim=-1)
        if self.conv1d is not None:
            # the carry may be wider than the activations (fp32 under a bf16 layer): the conv output returns to
            # the activation dtype, as in the forward; the carry keeps its own
            x_branch, conv_state = causal_conv_step(x_branch, state["conv_state"], self.conv1d.weight, self.conv1d.bias,
                                                    package=self._conv_impl, out_dtype=xz.dtype)
            state = {**state, "conv_state": conv_state}
        sel = x_t if self.proj_source == "layer_input" else None
        y_t, ssm_state = self.scan_step(x_branch, {k: v for k, v in state.items() if k != "conv_state"}, sel_x=sel)
        state = self._conform_carries(State({**state, **ssm_state}), ref)
        self._final_state = state
        # same gate + norm composition as the forward
        return self.out_proj(self._epilogue(y_t.to(xz.dtype), z)), state


from dataclasses import dataclass
from dataclasses import field as _field
from ..config import _CoreConfigBase, SpecFields, _refuse
from ..validation import WRITE_RULE_DEFAULT, resolve_write_rule

@dataclass(kw_only=True)
class HybridMambaConfig(SpecFields, _CoreConfigBase):
    """Architecture of a `HybridMambaSSM` layer as one typed, serializable object.

    A keyword-only dataclass: ``replace(**kw)`` copies with changes, ``from_dict`` is the strict dict
    constructor (unknown keys and wrongly typed values are a ``TypeError``), ``build(d_model=None,
    execution=None)`` returns the layer, and ``layer.to_recipe()`` records the full recipe.  How the
    layer runs (backend, checkpointing, kernel options) is an `Execution`, not part of the config.  These
    fields are the layer's only construction options.  ``None`` in an optional field means "the family
    default".  Fields the family cannot honor (``d_ssm_in != 1``, ``d_ssm_out`` other than 1, ``n_groups``)
    are refused.

    Field names are explanatory; each entry below also gives the paper's symbol where there is one
    (Paper 1, Section 5; the glossary table in ``docs/CONFIG_GLOSSARY.md`` lists them all).  The decay laws
    ``coord_static`` / ``coord_dynamic`` / ``coord_potential`` are the paper's "HM static / dynamic /
    potential" (per-coordinate), ``head_dynamic`` / ``head_static`` its "HM flat dynamic / static" (per head).

    Parameters
    ----------
    d_model : int, optional
        Layer width: the input and output width of the layer.  None: `build(d_model)` or the enclosing
        `CausalLM` supplies it; a layer built from the config alone then uses ``d_inner`` (when ``expand``
        is None).
    d_state : int, default 64
        N in the paper: total number of state rows per subsystem; the ``row_blocks`` sizes must sum
        to it.  Any split works: a flat law's row total (``head_dynamic``, ``head_static``) may be any
        size, the scan kernels pad or tile it exactly.  (Only ``kernel_options={"m3_kernel": "upstream"}``
        keeps its own requirement: head_dynamic rows and ``d_head`` in {16, 32, 64, 128}.)
    expand : float, optional
        Inner width factor, ``d_inner = int(d_model * expand)``; None means 2.0 unless ``d_inner`` is given.
        Giving both requires them to agree.
    d_conv : int, default 4
        k_conv in the paper: width of the causal depthwise conv on the drive (the SSM input), the
        convolution window.  ``0`` disables the conv (and the B/C conv with it).
    d_inner : int, optional
        d = H P in the paper, the drive width (= ``n_heads * d_head``).  Alone it is the width; with
        ``expand`` it must equal ``int(d_model * expand)``; neither: ``int(d_model * 2.0)``.
    n_heads, d_head : int, optional
        H heads (eta) and P coordinates per head in the paper, ``n_heads * d_head == d_inner``; give
        either.  Both omitted: ``d_head = 64`` in the layer.
    d_ssm_in, d_ssm_out, n_groups
        R, O (input and output rank) and G (channel groups) of the shared config; this family has
        R = O = 1 and supports only the defaults (``1``, ``None``, ``None``).
    dt_min, dt_max : float, default 1e-3, 0.1
        Range of the initial step size Delta_t = softplus(. + dt_bias): ``softplus(dt_bias)`` starts
        log-uniform in ``[dt_min, dt_max]``.
    dt_init_floor : float, default 1e-4
        Lower clamp on the sampled initial step size.
    dt_rank : int or {"adaptive", "auto"} or None, default "adaptive"
        Bottleneck of the step-size projection ``dt_proj(dt_low(u))``.  An int
        forces that rank, ``"auto"`` uses Mamba-1's ``ceil(d_model / 16)``,
        ``"adaptive"`` uses it only when it reduces projection multiplies, and
        ``None`` uses a dense ``Linear(d_sel, n_heads)``.
    force_dt_dense : bool, default False
        Force the dense step-size projection (conflicts with an explicit rank).
    dt_init : {"random", "constant", None}, default "random"
        Init of the ``dt_proj`` weight: uniform in ``+-std``, constant ``std``
        (``std = fan_in**-0.5 * dt_scale``), or ``None`` for the framework default.
    dt_scale : float, default 1.0
        Scale of that init; needs ``dt_init`` to be set explicitly if not 1.
    proj_source : {"layer_input", "main_branch"}, optional
        The coefficient source s_t of the paper: input of the token maps (B, C, step size, rates).
        ``"layer_input"`` reads the block input x_t (Mamba-2/3 style, selection width ``d_model``,
        B/C conv available); ``"main_branch"`` reads the post-conv drive u_t
        (Mamba-1 style, width ``d_inner``).  ``None`` resolves to ``"layer_input"``.
    bc_conv : bool, optional
        Depthwise causal conv + SiLU on the raw B/C projections (Mamba-2/3
        ``xBC`` conv).  Needs ``proj_source="layer_input"`` and ``d_conv > 0``;
        ``None`` means off.
    write_rule : {"trapezoid", "euler"}, optional
        Quadrature of the HM rows' forcing integral; default ``"trapezoid"``: the paper's two-endpoint
        write (Mamba-3, weight lambda_t); ``"euler"`` is exponential Euler (lambda = 1).  ``"zoh"`` is
        refused.  The flat ``head_dynamic`` rows always use the trapezoid and ``head_static`` the
        exponential-Euler rule.
    row_blocks : list of RowBlock or dict, default [{"frac": 1.0, "decay": "coord_static"}]
        Partition of the state rows; see `RowBlock` and
        `HybridMambaSSM`.  ``decay`` names the law: ``coord_static`` ("HM static", a = -exp(A_log) per
        (head, channel, row)), ``coord_dynamic`` ("HM dynamic", a_t = -act(W_a s_t)), ``coord_potential``
        ("HM potential", log alpha = Delta a + c (mu_t - mu_{t-1})), ``head_dynamic`` ("HM flat dynamic",
        per head and token) and ``head_static`` ("HM flat static", per head); an omitted ``decay`` resolves to
        ``coord_static`` and the recorded config carries it on every block.  ``complex=True`` blocks
        rotate (K_rot pairs).  Granularity: the coord laws are one law per row block inside one ``mh_scan`` launch
        (values per head, channel and row); each flat law is one scan with one decay scalar per head for all its rows.
        ``dyn_res``/``dyn_act``, ``mu_res``/``mu_cap`` and ``a_floor``/``a_act`` belong to ``coord_dynamic``,
        ``coord_potential`` and ``head_dynamic``: a non-default value without a block of that law is refused.
    learn_theta_freq, use_theta_h, theta_per_head : bool, optional
        Rotation options for ``complex`` HM rows: learn the pair frequencies
        ``omega_k = pi tanh(phi_k)`` (``theta_freq`` is phi_k), add a learned read offset ``theta_h``
        (nu in the paper), and emit one angle feature q_{k,t} per head instead of a head-shared one.
        Complex flat rows support none of them (they rotate at fixed frequency pi).
    omega_layout, theta_init : str, optional
        SSD-MIMO rotation options; non-default values are refused.
    b_norm, c_norm : bool, optional
        RMSNorm on the base maps B-hat and C-hat (default on).  Flat rows require both on.
    bc_bias : {"head", "shared", "none"}, optional
        Granularity of the static additive B/C biases beta, gamma: per head (default), one shared
        vector, or none.
    b_gate, c_gate : bool, optional
        The write / read gates phi_B, phi_C of the paper: data-dependent B/C gates (default off).
        HM rows require ``b_gate == c_gate``.
    bc_gate_act : str, default "one_plus"
        The gate activation phi: ``"one_plus"`` is ``1 + g`` (the paper's default); also ``"sigmoid"``,
        ``"exp"``, ``"tanh"``, ``"softplus"``, ``"raw"``, ``"one_plus_tanh"``.
    bc_gate_bias : bool, default False
        Learned gate bias.  Redundant (and refused) with ``bc_gate_pos="pre"``
        and a scale-factorising activation (``"one_plus"``, ``"exp"``).
    bc_gate_pos : {"pre", "post"}, default "pre"
        Gate before the rotation (host-side fold) or inside the scan after it.
    gate_init : {"kaiming_uniform", "zero", "kaiming_small"}, default "kaiming_uniform"
        Init of the B/C gate projections: ``nn.Linear``'s default draw, zeros
        (as SSD), or that draw scaled by 0.1*sqrt(3).
    dyn_res, mu_res : {"n", "h"}, default "n"
        Resolution of the ``coord_dynamic`` rate / ``coord_potential`` driver (mu_t) projections: per state
        row (``"n"``) or per head (``"h"``).
    dyn_act : {"heavy_tail", "softplus", "exp", "raw"}, default "heavy_tail"
        act in the paper's HM dynamic law a_t = -act(W_a s_t): the activation turning the
        ``coord_dynamic`` projection into a decay rate (a `layout.DYN_*` int code is accepted too).
        ``"heavy_tail"`` is ``ht(v) = 1 + v`` for ``v >= 0`` and ``1 / (1 - v)`` below.
    a_act : {"heavy_tail", "softplus"}, default "heavy_tail"
        The same choice for the ``head_dynamic`` rows' ``|A|`` ("HM flat dynamic"; as `SSDMimoSSM.a_act`).
    a_floor : float, default 1e-4
        Lower clamp (the 1e-4 of the paper's flat dynamic law) on ``|A|`` of the ``head_dynamic`` rows.
    mu_cap : float, default 4.0
        mu_cap in the paper, the bound of the ``coord_potential`` law, ``> 1``: any span's decay product
        lies in ``(0, mu_cap)``.  The potential gain is ``G_mu sigmoid(.)`` with
        ``G_mu = ln(mu_cap) / 2``.  Warns above 20.
    skip : bool, default True
        The D term of the paper (y_t = C_t h_t + D_t u_t; "D" in Mamba): ``y += D * u``.  False builds no D.
    skip_sharing : {"head", "channel"}, default "head"
        Shape of the D skip: one gain per head or per channel.
    norm_eps : float, default 1e-5
        Epsilon of the B/C RMSNorms.
    learnable_h_init : bool, default False
        Learn the initial state ``h0`` (``H x P x d_state``).
    out_norm : bool, default False
        The paper's output RMS normalization: insert a gated RMSNorm before ``out_proj`` (Mamba-2/3 style).  The
        default is Mamba-1's ``y * silu(z) -> out_proj`` with no norm.
    """

    row_blocks: list | tuple | None = _field(default_factory=lambda: [{"frac": 1.0, "decay": "coord_static"}])
    bc_gate_act: str | None = None
    bc_gate_bias: bool | None = None
    bc_gate_pos: str | None = None
    proj_source: str | None = None
    dt_rank: int | str | None = "adaptive"
    force_dt_dense: bool = False
    dt_init: str | None = "random"
    dt_scale: float = 1.0
    dyn_res: str = "n"
    dyn_act: str | int = "heavy_tail"
    mu_res: str = "n"
    mu_cap: float = MU_CAP_DEFAULT
    a_floor: float = 1e-4
    a_act: Literal["heavy_tail", "softplus"] = "heavy_tail"
    norm_eps: float = 1e-5
    learnable_h_init: bool = False
    gate_init: str = "kaiming_uniform"

    def __post_init__(self):
        super().__post_init__()
        if self.width() is not None:
            self._geometry()
        else:
            check_mu_cap(self.mu_cap)
        self._use_trap()
        spec = _config_spec.resolve("hybrid_mamba", d_state=self.d_state,
                                    **self.spec_kwargs())
        _check_gate_init("HybridMambaConfig", self.gate_init, spec["maps"])
        if self.write_rule is not None and not any(b["decay"] in _config_spec.HM_PER_COORD for b in spec["row_blocks"]):
            raise ValueError(
                "HybridMambaConfig: write_rule is set, but every row block is a flat law (head_dynamic / head_static), whose write "
                "is fixed by the law (head_dynamic trapezoid, head_static euler), so it would be ignored; drop write_rule "
                "or add coord_* rows")
        _config_spec.check_law_levers("HybridMambaConfig", spec["row_blocks"], _LAW_LEVERS, dict(
            dyn_res=self.dyn_res, dyn_act=self.dyn_act, mu_res=self.mu_res, mu_cap=self.mu_cap,
            a_floor=self.a_floor, a_act=self.a_act))

    def _geometry(self):
        check_mu_cap(self.mu_cap)
        return resolve_geometry(self.width(), n_heads=self.n_heads, d_head=self.d_head, default_d_head=64)

    def _use_trap(self) -> bool:
        """Map `write_rule` to the layer's `use_trap` bool via `resolve_write_rule`
        (None -> trapezoid; "zoh" and unknown names raise)."""
        rule = resolve_write_rule(
            WRITE_RULE_DEFAULT if self.write_rule is None else self.write_rule,
            "HybridMambaConfig")
        return rule == "trapezoid"

    _UNSUPPORTED = {
        "d_ssm_in": lambda c: c.d_ssm_in != 1,
        "d_ssm_out": lambda c: c.d_ssm_out not in (None, 1),
        "n_groups": lambda c: c.n_groups is not None,
    }

    def layer_kwargs(self) -> dict:
        """The keywords the layer builds from: every field this family has, as given (the layer resolves them)."""
        kw = super().layer_kwargs()
        for name in ("d_ssm_in", "d_ssm_out", "n_groups"):       # this family has a single subsystem, no groups
            del kw[name]
        return kw

    @staticmethod
    def _layer_class():
        return HybridMambaSSM


register_family("hybrid_mamba", config=HybridMambaConfig, layer=HybridMambaSSM)

if HybridMambaSSM.__doc__:      # the allowed ``kernel_options`` keys, generated from the schema
    HybridMambaSSM.__doc__ += "\n" + "\n".join("    " + line for line in _kernel_options.describe("hybrid_mamba").splitlines()) + "\n"
