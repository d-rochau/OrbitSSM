"""Pure-torch SSD-MIMO: an optimized sequential step and an optimized
chunked (parallel) scan.

Three implementations, kept apart on purpose:

* **oracle** -- ``tests/test_ssd_endpoint.py::physical_oracle``: the
  defining recurrence in PHYSICAL coordinates, one explicit loop per batch
  row, head, sibling and token, explicit rotation matrices.  It shares no
  code with this file.
* **sequential** -- :func:`ssd_step`: one token, O(1), every head and
  sibling batched, in FOLDED coordinates (the public ``State``).
* **parallel** -- :func:`ssd_chunked`: the whole sequence as a handful of
  batched matmuls over chunks (the SSD dual), same folded coordinates,
  plain autograd.  ``backend="torch"`` (and the CUDA route's fallback)
  takes it for L > SCAN_MAX_L and :func:`ssd_scan_seq` (the token
  recurrence, per-token objects batched over L, adjoint-recurrence
  backward `_LinearScanFn`) for shorter calls.

Both optimized routes are REARRANGED mathematics and are tested against the
oracle, not against each other.

Notation.  Head h, sibling r (the
kernel head is the pair (h, r)), channel group g, input column q (R of
them), output column o (O of them), state row n (N of them, K = N/2 pairs).
Per token t and pair (h, r):

    Bbar = diag(1 + gB[:, r]) Bhat + beta[h, r]          (N x R)
    Cbar = (1 + gC) * Chat + gamma[h, r]                 (N x O)
    Btil = R(Theta)^T Bbar,  Ctil = R(Theta + theta_h)^T Cbar   (folded maps)
    alpha = exp(dA),  dA = -|a| Delta                    (scalar decay)
    vtil[:, g] = Btil u[h, g, :]                         (folded write)
    f_t = alpha f_{t-1} + Delta [lam vtil_t + (1 - lam) alpha vtil_{t-1}]
    y[h, g, :] = silu(z) * sum_r chi[h, r] (Ctil^T f_t[:, g] + D[h, g, r, :] u[h, g, r])

``f`` is the folded state, ``State["hidden_state"]`` ([B, H, G*R, N] with
the flat P axis = g*R + r), ``State["write_prev"]`` is vtil_{t-1} in the
same layout, ``State["rot_ang"]`` is Theta ([B, H, R, ka]) and
``State["bc_conv_state"]`` the last ``d_conv`` raw B|C inputs.

The chunked route uses the collapsed one-tap form of the same law (contract
§1.1 "Folded recurrence and collapsed scan"): with n_t = Delta_{t+1}(1 -
lam_{t+1}) (n_{L-1} = 0), S_t = f_t + n_t vtil_t obeys S_t = alpha_t
S_{t-1} + w_t vtil_t, w_t = lam_t Delta_t + n_t, and the readout of f_t is
the readout of S_t minus n_t Ctil_t^T vtil_t -- i.e. the diagonal (s = t)
coefficient of the intra-chunk score is lam_t Delta_t instead of w_t.  No
separate correction term is formed.

Precision: every law and scan operation runs in fp32 (fp64 when the input is
fp64) with autocast disabled; only the fused token projection GEMM follows
the ambient autocast, as on every other route.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ..layers.state import State

__all__ = ("ssd_step", "ssd_chunked", "ssd_scan_seq", "fold_pairs",
           "token_law", "DEFAULT_CHUNK")

#: Pins the chunk length of the parallel route when set (tests, sweeps);
#: None = `default_chunk(N)`.  The module's integer ``chunk_size`` wins.
DEFAULT_CHUNK = None


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    """fp64 stays fp64; every other IO dtype computes the law in fp32."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


class Law(NamedTuple):
    """Per-token law quantities, leading shape ``lead`` = (B,) or (B, L).

    Bbar  [*lead, H|1, R|1, R, N]  gated + biased write map per (h, r),
                                   axes (sibling r, input column q, n)
    Cbar  [*lead, H|1, 1|R, O, N]  gated + biased read map per (h, r)
    dt    [*lead, H, R]            Delta (write clock)
    dA    [*lead, H, R]            log alpha = -|a| Delta
    lam   [*lead, H] or None       two-endpoint weight lambda
    incr  [*lead, H, R, ka] or None physical phase increment dTheta
    chi   [*lead|-, H, R] or None  sibling weight (None = unit weights)
    """
    Bbar: torch.Tensor
    Cbar: torch.Tensor
    dt: torch.Tensor
    dA: torch.Tensor
    lam: torch.Tensor | None
    incr: torch.Tensor | None
    chi: torch.Tensor | None


def _no_autocast(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)


def token_slab(core, u: torch.Tensor, tok_raw: torch.Tensor | None):
    """The fused token-map projection slab [*, tok_width] (one GEMM)."""
    if tok_raw is not None:
        if tok_raw.shape[-1] != core.tok_width:
            raise ValueError(
                f"tok_raw last dim {tok_raw.shape[-1]} != tok_width="
                f"{core.tok_width} (the fused token-map projection's "
                f"width; use `tok_proj_weight()` to build it)")
        return tok_raw
    w, b = core.tok_proj_weight()
    return F.linear(u, w, b)


def split_slab(core, raw: torch.Tensor) -> dict:
    out = dict(zip(core._proj_keys, raw.split(core._proj_widths, dim=-1)))
    from ..layers.glue import dt_fp32_on
    if dt_fp32_on(core.kernel_options) and core.dt_proj.bias is not None:
        out["_dtb"] = core.dt_proj.bias     # slab dt columns are bias-free
    return out


def bc_conv(core, bc_raw: torch.Tensor, prev: torch.Tensor | None, f):
    """Causal depthwise B|C convolution + SiLU with its carried window.

    bc_raw [B, L, C] raw (pre-conv) B|C projections; ``prev`` [B, C, d]
    the last ``d`` raw inputs before this call (None = sequence start,
    zeros).  Returns the convolved [B, L, C] and the new window [B, C, d]
    (the last ``d`` raw inputs ending at the final token -- the
    ``CausalDepthwiseConv.step`` convention).
    """
    conv = core.bc_conv1d.conv
    d = core.bc_conv1d.d_conv
    xin = _c(bc_raw, f).transpose(1, 2)                        # [B, C, L]
    if prev is None:
        prev = xin.new_zeros(xin.shape[0], xin.shape[1], d)
    if xin.shape[-1] == 1:
        # one token: the new window IS the conv input (fresh storage)
        window = torch.cat([_c(prev[..., 1:], f), xin], dim=-1)  # [B, C, d]
        y = F.conv1d(window, _c(conv.weight, f), _c(conv.bias, f),
                     groups=conv.groups)
        return F.silu(y).transpose(1, 2), window
    full = torch.cat([_c(prev, f), xin], dim=-1)               # [B, C, d+L]
    y = F.conv1d(full[..., 1:], _c(conv.weight, f), _c(conv.bias, f),
                 groups=conv.groups)                           # [B, C, L]
    return F.silu(y).transpose(1, 2), full[..., -d:]


def _c(t: torch.Tensor, f: torch.dtype) -> torch.Tensor:
    """``t.to(f)`` without a dispatcher call when it is already ``f``."""
    return t if t.dtype == f else t.to(f)


def _rms_last(x: torch.Tensor, norm, f) -> torch.Tensor:
    """``nn.RMSNorm(N)`` over the last axis of x [..., N] (written out: the
    CPU ATen kernel costs ~10 small ops per call)."""
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) \
        * _c(norm.weight, f)


def token_law(core, parts: dict, b_src: torch.Tensor, c_src: torch.Tensor,
              f: torch.dtype, lead: tuple) -> Law:
    """Every per-token object of the law, batched over (h, r).

    ``b_src`` / ``c_src`` are the (convolved, if ``bc_conv``) raw B / C
    projections [*lead, N*R] / [*lead, N*O] (column n*R + q); ``parts``
    the other slab slices.  Mirrors ``SSDMimoSSM._token_maps`` + the step's
    map assembly; the a-law tails are the module's own
    (``_a_mag_tail_eager`` / ``_a_mag_softplus_eager``), so boundary
    adjoints follow its convention.

    MAP LAYOUT: the state axis N is INNERMOST ([.., q, n] / [.., o, n]), so
    the RMSNorm, the pair rotation and every adjoint reduction over the
    small map axes run over contiguous N-rows.
    """
    from .ssd_mimo import _a_mag_tail_eager, _a_mag_softplus_eager

    H, R, O, N = core.H, core.R, core.O, core.N
    Bm = _c(b_src, f).view(*lead, N, R).transpose(-1, -2)
    Cm = _c(c_src, f).view(*lead, N, O).transpose(-1, -2)
    if len(lead) > 1:
        # sequences: materialise N-innermost once, so every map op and its
        # adjoint streams; a single token keeps the (free) transposed view
        Bm, Cm = Bm.contiguous(), Cm.contiguous()
    if core.b_norm:                                            # Bhat [*, q, n]
        Bm = _rms_last(Bm, core.B_norm, f)
    if core.c_norm:
        Cm = _rms_last(Cm, core.C_norm, f)                     # Chat [*, o, n]
    if core.bc_bias:
        if not core.head_bias:
            # bias_gran="shared": the R^N bias joins inside the norm block,
            # i.e. BEFORE the gates
            Bm = Bm + _c(core.B_bias, f)
            Cm = Cm + _c(core.C_bias, f)
    # write map: diag(1 + gB[:, r]) Bhat (+ beta[h, r]); sibling r picks a
    # gate column that row-scales ALL R input columns
    if core.use_g_b:
        gB = 1.0 + _c(parts["gB"], f).view(*lead, N, R).transpose(-1, -2)
        Bbar = (gB[..., None, :, None, :]                      # [*,1,R,1,N]
                * Bm[..., None, None, :, :])                   # [*,1,R,q,N]
    else:
        Bbar = Bm[..., None, None, :, :]                       # [*,1,1,q,N]
    if core.use_g_c:
        Cm = Cm * (1.0 + _c(parts["gC"], f).view(*lead, N, O).transpose(-1, -2))
    Cbar = Cm[..., None, None, :, :]                           # [*,1,1,o,N]
    if core.head_bias:
        # gate first, then the per-kernel-head bias (not gated); the
        # parameters are [H, R, N, R|O] -> viewed [H, R, R|O, N]
        Bbar = Bbar + _c(core.B_bias, f).transpose(-1, -2)     # [*,H,R,q,N]
        Cbar = Cbar + _c(core.C_bias, f).transpose(-1, -2)     # [*,H,R,o,N]

    dt_pre = _c(parts["dt"], f)
    if parts.get("_dtb") is not None:    # dt_fp32 option: bias kept out of the slab
        dt_pre = dt_pre + _c(parts["_dtb"], f)
    dt = F.softplus(dt_pre).view(*lead, H, R)                  # Delta
    if core.a_dyn:
        va = _c(parts["a"], f)
        if torch.is_grad_enabled():
            # the module's own tails: their boundary adjoints are the law's
            tail = (_a_mag_tail_eager if core.a_act == "heavy_tail"
                    else _a_mag_softplus_eager)
            a_mag = tail(va, 1e-4)                             # |a|
        elif core.a_act == "heavy_tail":
            # the same VALUES in fewer ops (no adjoint is formed here):
            # 1 + v (v >= 0), 1 / (1 - v) (v < 0), floored
            a_mag = torch.where(va >= 0, va + 1.0,
                                torch.reciprocal(1.0 - va)).clamp_min(1e-4)
        else:
            a_mag = F.softplus(va).clamp_min(1e-4)
        rate = (a_mag.view(*lead, H, R) if core.a_dyn_per_sibling
                else a_mag.view(*lead, H, 1))
    else:
        rate = core.A_log.to(f).exp()          # [H, R]
    dA = -(dt * rate)                                          # log alpha

    lam = torch.sigmoid(_c(parts["trap"], f)) if core.use_trap else None

    incr = None
    if core.rot_mode == "m3":
        K, ka = N // 2, core.n_rope_angles
        if core.omega_per_head:
            # omega_layout="head": one angle and one frequency per head, on every
            # rotated pair of the head
            ang = torch.tanh(_c(parts["theta"], f)).view(
                *lead, H, 1, 1).expand(*lead, H, 1, ka)        # [*,H,1,ka]
        else:
            ang = torch.tanh(_c(parts["theta"], f)).view(
                *lead, H if core.theta_per_head else 1, 1, ka) # [*,H|1,1,ka]
        if core.learn_theta_freq and core.omega_layout != "pair":
            # [H,1,1] (head) or [H,1,ka] (head_pair)
            omega = (math.pi * torch.tanh(_c(core.theta_freq, f))
                     ).view(H, 1, -1)
        elif core.learn_theta_freq:
            omega = math.pi * torch.tanh(_c(core.theta_freq, f))
        else:
            omega = math.pi
        # dTheta = -omega tanh(Ang) Delta on the ROTATING pairs only
        # (the clock, its carry and theta_h have width ka; `_pad_pairs`
        # hands the fold exact zeros on the real pairs)
        incr = (dt[..., None] * ang) * (-omega)                # [*,H,R,ka]

    chi = None
    if core.r_mix_mode != "none":
        chi = _c(core.r_mix, f)                                # zeta [H, R]
        if core.r_mix_dyn:
            chi = chi + _c(parts["rmix"], f).view(*lead, H, R)
    return Law(Bbar, Cbar, dt, dA, lam, incr, chi)


def _pad_pairs(core, ang: torch.Tensor) -> torch.Tensor:
    """A per-ROTATED-pair angle [..., ka] on all N/2 pairs: exact zeros
    (the identity) on the real pairs k >= ka."""
    pad = core.N // 2 - ang.shape[-1]
    return F.pad(ang, (0, pad)) if pad else ang


def _wrap2pi(x: torch.Tensor) -> torch.Tensor:
    """The running phase reduced to [-pi, pi] exactly as the native route
    does (`ssd_mimo._wrap2pi_t`; identity derivative)."""
    from .ssd_mimo import _wrap2pi_t
    return _wrap2pi_t(x)


def fold_pairs(v: torch.Tensor, ang: torch.Tensor, half: bool) -> torch.Tensor:
    """R(ang)^T on the state pairs of ``v`` [..., X, N]; ``ang`` [..., K].

    Pair k is (2k, 2k+1) ("interleaved") or (k, k+K) ("half"):
    e' = cos e + sin o,  o' = cos o - sin e  (per pair, broadcast over X).
    """
    c = torch.cos(ang).unsqueeze(-2)                           # [..., 1, K]
    s = torch.sin(ang).unsqueeze(-2)
    N = v.shape[-1]
    if half:
        e, o = v[..., :N // 2], v[..., N // 2:]
        return torch.cat((c * e + s * o, c * o - s * e), dim=-1)
    vv = v.view(*v.shape[:-1], N // 2, 2)
    e, o = vv[..., 0], vv[..., 1]
    return torch.stack((c * e + s * o, c * o - s * e), dim=-1).view(
        *torch.broadcast_shapes(v.shape[:-1], c.shape[:-1]), N)


def _skip(core, u: torch.Tensor, chi, f):
    """sum_r chi[h, r] D[h, g, r, :] u[h, g, r]  -> [..., H, G, O] (one
    token: one batched [1, R] @ [R, O] matmul per (h, g))."""
    uw = u if chi is None else u * chi.unsqueeze(-2)
    return torch.matmul(uw.unsqueeze(-2),
                        _c(core._skip_gains(), f)).squeeze(-2)


# ════════════════════════════════════════════════════════════════════════
#  Optimized sequential: one token, O(1)
# ════════════════════════════════════════════════════════════════════════

def _step_core(core, x_t, raw, gate_z, state, f):
    """The step's law and recurrence for one token (no validation).

    x_t [B, H*G*R] drive, raw [B, tok_width] token slab, gate_z [B,
    H*G*O] or None, ``state`` a resolved State.  Returns (y [B, H*G*O] in
    ``f``, dict of the new carries in their State dtypes)."""
    Bsz = x_t.shape[0]
    H, G, R, N, O = core.H, core.G, core.R, core.N, core.O
    with _no_autocast(x_t.device.type):
        parts = split_slab(core, raw)
        b_src, c_src = parts["B"], parts["C"]
        new = {}
        if core.bc_conv1d is not None:
            bc, window = bc_conv(
                core, torch.cat([b_src, c_src], dim=-1)[:, None],
                state["bc_conv_state"], f)
            b_src, c_src = bc[:, 0].split([N * R, N * O], dim=-1)
            new["bc_conv_state"] = window
        law = token_law(core, parts, b_src, c_src, f, (Bsz,))

        Btil, Ctil = law.Bbar, law.Cbar          # [B,H|1,R|1,R|O,N]
        if law.incr is not None:
            theta = _wrap2pi(_wrap2pi(_c(state["rot_ang"], f))
                             + law.incr)                       # Theta_t mod 2 pi
            half = core.pair_layout == "half"
            Btil = fold_pairs(Btil, _pad_pairs(core, theta), half)
            ang_c = (theta + _c(core.theta_h, f)[:, None, :]
                     if core.use_theta_h else theta)
            Ctil = fold_pairs(Ctil, _pad_pairs(core, ang_c), half)
            new["rot_ang"] = theta

        u = _c(x_t, f).reshape(Bsz, H, G, R)
        # folded write vtil[b,h,g,r,:] = sum_q u[b,h,g,q] Btil[b,h,r,q,:]:
        # one matmul with (r, n) as the output columns, eager and compiled.
        Bk = Btil.expand(Bsz, H, R, R, N).transpose(2, 3).reshape(
            Bsz, H, R, R * N)                                  # [B,H,q,(r,n)]
        v = torch.matmul(u, Bk).view(Bsz, H, G, R, N)
        f_old = _c(state["hidden_state"], f).view(Bsz, H, G, R, N)
        a5 = law.dA.exp().view(Bsz, H, 1, R, 1)                # alpha
        dt5 = law.dt.view(Bsz, H, 1, R, 1)                     # Delta
        if law.lam is not None:
            # f_t = alpha (f_{t-1} + Dt (1-lam) v_{t-1}) + Dt lam v_t
            lam5 = law.lam.view(Bsz, H, 1, 1, 1)
            v_prev = _c(state["write_prev"], f).view(Bsz, H, G, R, N)
            f_new = torch.addcmul((dt5 * lam5) * v, a5,
                                  torch.addcmul(f_old, dt5 * (1.0 - lam5),
                                                v_prev))
            new["write_prev"] = v
        else:
            f_new = torch.addcmul(dt5 * v, a5, f_old)
        new["hidden_state"] = f_new
        # read: y[b,h,g,:] = sum_r chi_r Ctil_r^T f[b,h,g,r,:] -- one matmul
        # over the joint (r, n) axis
        Cr = Ctil if law.chi is None else Ctil * law.chi[..., None, None]
        Ck = Cr.expand(Bsz, H, R, O, N).transpose(-1, -2).reshape(
            Bsz, H, R * N, O)                                  # [B,H,(r,n),o]
        y = torch.matmul(f_new.view(Bsz, H, G, R * N), Ck)     # [B,H,G,O]
        if core.skip:
            y = y + _skip(core, u, law.chi, f)
        y = y.reshape(Bsz, H * G * O)
        if gate_z is not None:
            y = y * F.silu(_c(gate_z, f))
    P = G * R
    out = {}
    for k in state:                          # the State's own key order
        if k not in new:
            continue
        v_ = new[k]
        if k in ("hidden_state", "write_prev"):
            v_ = v_.reshape(Bsz, H, P, N)
        out[k] = _c(v_, state[k].dtype)
    return y, out


def ssd_step(core, x_t: torch.Tensor, state, gate_z=None, sel_x=None,
             tok_raw=None):
    """One token of the law in folded coordinates (``SSDMimoSSM.step``).

    x_t [B, H*G*R]; gate_z [B, H*G*O] or None; returns (y [B, H*G*O],
    State).  No python loop over heads, siblings or outputs; the state
    stays in the public [B, H, G, R, N] layout, so the write and the read
    are one matmul each (the read also sums the siblings).
    """
    if gate_z is not None and gate_z.shape != (*x_t.shape[:-1], core.d_out):
        raise ValueError(
            f"gate_z must have shape {(*x_t.shape[:-1], core.d_out)}")
    from ..layers.state import ScanState
    state = ScanState(core).resolve_state(
        state, batch_size=x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
    u_sel = core._select(x_t, sel_x)
    raw = token_slab(core, u_sel, tok_raw)
    y, new = _step_core(core, x_t, raw, gate_z, state, compute_dtype(x_t))
    new_state = ScanState(core)._conform_carries(State(new), state)
    core.h_final = new_state["hidden_state"]
    core._final_state = new_state
    return y.to(x_t.dtype), new_state


def _fused_step_weight(mixer):
    """[in_proj ; token projections] as ONE weight + bias, so a mixer step
    is one GEMM.  Memoised (keyed on every source parameter's data_ptr and
    version) only while the mixer is PINNED to inference
    (``set_inference(True)``: the caller's promise of stable weights);
    otherwise rebuilt per call (a cat of the parameter rows)."""
    from ..layers.glue import param_cache_key, dt_fp32_on
    core = mixer
    dtz = dt_fp32_on(core.kernel_options)      # dt bias rides split_slab's `_dtb`, not the slab
    mods = [getattr(core, a) for a in core._proj_attrs]
    params = [mixer.in_proj.weight] + [m.weight for m in mods] + [
        m.bias for m in mods if m.bias is not None]
    cache = getattr(mixer, "_step_w_cache", None)
    pinned = getattr(mixer, "_inference_pin", None) is True
    if pinned and cache is not None:
        key = param_cache_key(params, dtz)
        if cache[0] == key:
            return cache[1], cache[2]
    w = torch.cat([mixer.in_proj.weight] + [core._proj_w(a) for a in core._proj_attrs],
                  dim=0)
    b = None
    if core._proj_any_bias:
        lead = mixer.in_proj.weight.shape[0]
        b = torch.cat([core._proj_zero.new_zeros(lead)] + [
            m.bias if (m.bias is not None
                       and not (dtz and a == "dt_proj"))
            else core._proj_zero[:n]
            for a, m, n in zip(core._proj_attrs, mods, core._proj_widths)])
    if pinned:
        mixer._step_w_cache = (param_cache_key(params, dtz), w, b)
    return w, b


def mixer_step_fast(mixer, x_t: torch.Tensor, state):
    """``SSDMimoSSM.step`` on the fast inference path (no graph).

    One GEMM for [drive | gate | token slab], the drive convolution step,
    the core law (`_step_core`) and ``out_proj``; one State validation.
    With ``mixer.step_inplace`` the new carries are written INTO the given
    State's tensors (fixed addresses, e.g. for CUDA-graph capture) and the
    same State object is returned; otherwise the State is new."""
    core = mixer
    Bsz = x_t.shape[0]
    state = mixer.resolve_state(state, batch_size=Bsz, device=x_t.device,
                                dtype=x_t.dtype)
    di, dg = mixer.d_inner, mixer.d_gate
    f = compute_dtype(x_t)
    with torch.no_grad():
        if mixer.fuse_in_proj:
            w, b = _fused_step_weight(mixer)
            xb, z, raw = F.linear(x_t, w, b).split((di, dg, core.tok_width),
                                                   dim=-1)
        else:
            xb, z = mixer.in_proj(x_t).split((di, dg), dim=-1)
            raw = None
        new = {}
        if mixer.conv1d is not None:
            window = torch.cat([_c(state["conv_state"][:, :, 1:], xb.dtype),
                                xb.unsqueeze(-1)], dim=-1)
            conv = mixer.conv1d
            xb = F.silu(torch.linalg.vecdot(
                window, _c(conv.weight[:, 0], xb.dtype))
                + _c(conv.bias, xb.dtype))
            new["conv_state"] = window
        if raw is None:
            raw = token_slab(core, xb, None)   # main_branch: maps read the drive
        y, carries = _step_core(core, xb, raw, z, state, f)
        new.update(carries)
        y = mixer.out_proj(mixer.norm(_c(y, x_t.dtype)))
        if getattr(mixer, "step_inplace", False):
            for k, v in new.items():
                state[k].copy_(v)
            out = state
        else:
            out = State({k: _c(new[k], state[k].dtype) if k in new else state[k]
                         for k in state})
    core.h_final = out["hidden_state"]
    mixer._final_state = out
    return y, out


# ════════════════════════════════════════════════════════════════════════
#  Sequence routes: shared preparation
# ════════════════════════════════════════════════════════════════════════

def grad_mode() -> str:
    """``SSD_MIMO_TORCH_GRAD`` for the token scan: "fn" (default) =
    `_LinearScanFn`, the adjoint-recurrence backward; "autograd" = the
    same loop under plain autograd (the verification variant).  The
    chunked dual is plain autograd only."""
    import os
    v = os.environ.get("SSD_MIMO_TORCH_GRAD", "fn").strip().lower()
    if v not in ("fn", "autograd", "auto"):
        raise ValueError(
            f"SSD_MIMO_TORCH_GRAD must be fn|autograd, got {v!r}")
    return "autograd" if v == "autograd" else "fn"


def default_chunk(N: int) -> int:
    """Chunk length of the parallel route: 16 up to N = 64, 32 above."""
    return 16 if N <= 64 else 32


#: Up to this length ``backend="torch"`` takes the token scan; the chunked
#: dual takes longer sequences.
SCAN_MAX_L = 8


def torch_route(L: int) -> str:
    """Which sequence route ``backend="torch"`` takes: "chunked" (the
    parallel dual) or "scan" (the token recurrence, `ssd_scan_seq`).
    ``SSD_MIMO_TORCH_ROUTE`` pins it; default: scan for L <= SCAN_MAX_L."""
    import os
    v = os.environ.get("SSD_MIMO_TORCH_ROUTE", "auto").strip().lower()
    if v not in ("auto", "chunked", "scan"):
        raise ValueError(
            f"SSD_MIMO_TORCH_ROUTE must be auto|chunked|scan, got {v!r}")
    if v == "auto":
        return "scan" if L <= SCAN_MAX_L else "chunked"
    return v


class SeqPrep(NamedTuple):
    """Per-token folded objects of a whole sequence, [B, L, ...]."""
    Btil: torch.Tensor          # [B,L,H,R,R,N]  folded write map (r, q, n)
    Cq: torch.Tensor            # [B,L,H,R,O,N]  chi * folded read map
    u: torch.Tensor             # [B,L,H,G,R]    drive
    law: Law
    theta: torch.Tensor | None  # [B,L,H,R,K]    Theta_t
    conv_state: torch.Tensor | None
    f: torch.dtype


def _prepare(core, x, u_sel, tok_raw, carry) -> SeqPrep:
    """Token law, B/C convolution and rotation fold for a whole sequence
    (plain autograd: every op here is per-token parallel)."""
    B, L, _ = x.shape
    H, G, R, N, O = core.H, core.G, core.R, core.N, core.O
    f = compute_dtype(x)
    raw = token_slab(core, u_sel, tok_raw)
    with _no_autocast(x.device.type):
        parts = split_slab(core, raw)
        b_src, c_src = parts["B"], parts["C"]
        conv_state = None
        if core.bc_conv1d is not None:
            bc, conv_state = bc_conv(
                core, torch.cat([b_src, c_src], dim=-1),
                carry.get("bc_conv_state"), f)
            b_src, c_src = bc.split([N * R, N * O], dim=-1)
        law = token_law(core, parts, b_src, c_src, f, (B, L))
        theta = None
        Btil, Ctil = law.Bbar, law.Cbar          # [B,L,H|1,R|1,R|O,N]
        if law.incr is not None:
            # Theta_t = Theta_{-1} + sum_{s<=t} dTheta_s
            # reduced mod 2 pi like the native route: only cos/sin of Theta
            # is read, and the reduction keeps the static read offset
            # theta_h (added AFTER it) from being rounded away against a
            # large accumulated or supplied phase
            theta = _wrap2pi(torch.cumsum(law.incr, dim=1))    # [B,L,H,R,K]
            if carry.get("rot_ang") is not None:
                theta = _wrap2pi(
                    theta + _wrap2pi(_c(carry["rot_ang"], f))[:, None])
            half = core.pair_layout == "half"
            Btil = fold_pairs(Btil, _pad_pairs(core, theta), half)
            ang_c = (theta + _c(core.theta_h, f)[:, None, :]
                     if core.use_theta_h else theta)
            Ctil = fold_pairs(Ctil, _pad_pairs(core, ang_c), half)
        Btil = Btil.expand(B, L, H, R, R, N)                   # [.., r, q, n]
        Cq = Ctil if law.chi is None else Ctil * law.chi[..., None, None]
        Cq = Cq.expand(B, L, H, R, O, N)                       # [.., r, o, n]
        u = _c(x, f).reshape(B, L, H, G, R)
    return SeqPrep(Btil, Cq, u, law, theta, conv_state, f)


def _initial_folded(prep: SeqPrep, h0, wp, lag0, B, H, G, R, N):
    """S_{-1} = f_{-1} (+ n_{-1} vtil_{-1} on the collapsed form), in the
    [B, H, R, N, G] scan layout; None for a zero start."""
    f = prep.f
    S0 = None
    if h0 is not None:
        S0 = _c(h0, f).view(B, H, G, R, N).permute(0, 1, 3, 4, 2)
    if wp is not None and lag0 is not None:
        shift = (_c(wp, f).view(B, H, G, R, N).permute(0, 1, 3, 4, 2)
                 * lag0[..., None, None])
        S0 = shift if S0 is None else S0 + shift
    return S0


def _finish(core, prep: SeqPrep, x, y_scan, gate_z, S_fin, want_state,
            carry, v_last, h0=None):
    """Skip, gate, output cast and the complete State (shared epilogue).

    y_scan [B, L, H, G, O]; S_fin [B, H, G, R, N] (public layout)."""
    B, L, _ = x.shape
    H, G, R, N, O = core.H, core.G, core.R, core.N, core.O
    f = prep.f
    with _no_autocast(x.device.type):
        y = y_scan
        if core.skip:
            # sum_r chi_r D[h,g,r,:] u[h,g,r], laid out [.., r, g, o] so that
            # every adjoint reduction (over g,o for chi, o for u, b,l for D)
            # runs over contiguous trailing or leading axes
            sk = (prep.u.transpose(-1, -2).contiguous()[..., None]
                  * _c(core._skip_gains(), f).permute(0, 2, 1, 3))
            if prep.law.chi is not None:
                sk = sk * prep.law.chi[..., None, None]         # [B,L,H,R,G,O]
            y = y + sk.sum(-3)
        y = y.reshape(B, L, H * G * O)
        if gate_z is not None:
            y = y * F.silu(_c(gate_z, f))
        y = y.to(x.dtype)
    P = G * R
    # the hidden carry in the declared State dtype: the supplied initial
    # state's (as `ssd_step` returns it), else the IO dtype (as `ssd_step`
    # initialises it) -- never a silent downcast of an fp32 carry
    hidden = S_fin.reshape(B, H, P, N).to(x.dtype if h0 is None else h0.dtype)
    state = None
    if want_state:
        state = State(hidden_state=hidden)
        if prep.theta is not None:
            rot = carry.get("rot_ang")
            state["rot_ang"] = prep.theta[:, -1].to(
                rot.dtype if rot is not None else f).clone()
        if prep.law.lam is not None:
            wp = carry.get("write_prev")
            state["write_prev"] = v_last.reshape(B, H, P, N).to(
                wp.dtype if wp is not None else x.dtype)
        if prep.conv_state is not None:
            cst = carry.get("bc_conv_state")
            state["bc_conv_state"] = prep.conv_state.to(
                cst.dtype if cst is not None else x.dtype).contiguous()
    return y, state, hidden


# ════════════════════════════════════════════════════════════════════════
#  Optimized parallel: chunked SSD dual
# ════════════════════════════════════════════════════════════════════════

def _segsum(x: torch.Tensor) -> torch.Tensor:
    """out[..., i, j] = sum_{k=j+1}^{i} x[..., k] for j <= i, -inf above.

    Built by a masked cumsum (Mamba-2's ``segsum``), so each entry sums only
    its own segment -- no difference of two long prefix sums."""
    n = x.shape[-1]
    xe = x[..., :, None].expand(*x.shape, n)                   # [.., k, j]
    lower = torch.ones(n, n, dtype=torch.bool, device=x.device).tril(-1)
    ss = torch.cumsum(xe.masked_fill(~lower, 0.0), dim=-2)
    keep = torch.ones(n, n, dtype=torch.bool, device=x.device).tril(0)
    return ss.masked_fill(~keep, float("-inf"))


def _chunk_fwd(Bq, Cq, U, dAz, wz, wdz, S0):
    """The chunked dual on chunk-major operands (differentiable torch ops).

    Bq [B,nc,H,R,Ri*T,N] (row q*T + s), Cq [B,nc,H,R,O*T,N] (row o*T + t),
    U [B,nc,H,Ri*T,G] (row q*T + s), dAz / wz / wdz [B,nc,H,R,T] (log
    decay, write weight w_s, diagonal weight), S0 [B,H,R,N,G] or None.
    Returns Y [B,nc,H,O*T,G] and the entering states Sin [B,nc+1,H,R,N,G]
    (Sin[:, nc] = final state).  The small map axes (o, q) are OUTSIDE the
    token axes, so every broadcast over them and every reduction of them
    runs over contiguous T-rows.

    Intra chunk:  y_t += sum_r sum_{s<=t} W_r[t,s] (Ctil_t^T Btil_s) u_s,
                  W[t,s] = exp(cs_t - cs_s) w_s (s < t), W[t,t] = wd_t.
    Chunk state:  Send = sum_s exp(cs_T - cs_s) w_s Btil_s u_s.
    Passing:      Sin[c] = sum_{j<=c} exp(A_j + .. + A_{c-1}) Z_j,
                  Z = (S0, Send_0, .., Send_{nc-1}).
    Read:         y_t += sum_r exp(cs_t) Ctil_t^T Sin[c].
    """
    B, nc, H, R, TRi, N = Bq.shape
    T = dAz.shape[-1]
    Ri, O, G = TRi // T, Cq.shape[-2] // T, U.shape[-1]
    cs = torch.cumsum(dAz, dim=-1)
    E = torch.exp(_segsum(dAz))                                # [B,nc,H,R,T,T]
    # The own-token (s = t) weight is ASSIGNED, never formed as w + (wd - w):
    # with a large next-token write n_t = w_t - wd_t that sum rounds the
    # own-token term away.
    eye = torch.eye(T, dtype=torch.bool, device=E.device)
    W = torch.where(eye, wdz[..., :, None], E * wz[..., None, :])
    S = torch.matmul(Cq, Bq.transpose(-1, -2))                 # [..,O*T,Ri*T]
    M = S.view(B, nc, H, R, O, T, Ri, T) * W.view(B, nc, H, R, 1, T, 1, T)
    M = M.sum(3) if R > 1 else M.squeeze(3)                    # [..,O,T,Ri,T]
    Y = torch.matmul(M.reshape(B, nc, H, O * T, Ri * T), U)    # [..,O*T,G]
    dend = E[..., -1, :] * wz                                  # [B,nc,H,R,T]
    Bw = (Bq.view(B, nc, H, R, Ri, T, N)
          * dend[:, :, :, :, None, :, None]).view(B, nc, H, R, TRi, N)
    Send = torch.matmul(Bw.transpose(-1, -2), U[:, :, :, None])  # [..,R,N,G]
    Z0 = S0 if S0 is not None else Send.new_zeros(B, H, R, N, G)
    Zm = torch.cat([Z0[:, None], Send], dim=1)                 # [B,nc+1,..]
    xz = F.pad(cs[..., -1].permute(0, 2, 3, 1), (1, 0))        # [B,H,R,nc+1]
    Dm = torch.exp(_segsum(xz))
    Sin = torch.einsum("bhrij,bjhrng->bihrng", Dm, Zm)
    P = torch.matmul(Cq, Sin[:, :nc])                          # [..,R,O*T,G]
    ys = (P.view(B, nc, H, R, O, T, G)
          * torch.exp(cs)[:, :, :, :, None, :, None])
    ys = ys.sum(3) if R > 1 else ys.squeeze(3)
    return Y + ys.reshape(B, nc, H, O * T, G), Sin


def _chunk_scan(Bt, Cq, u, dA, w, wd, S0, T):
    """[B, L, ...] operands -> chunk-major, scan, back.

    Bt [B,L,H,R,Ri,N], Cq [B,L,H,R,O,N], u [B,L,H,G,Ri], dA / w / wd
    [B,L,H,R], S0 [B,H,R,N,G] or None.  Returns y [B,L,H,G,O] and the
    final collapsed state [B,H,R,N,G]."""
    B, L, H, R, Ri, N = Bt.shape
    O, G = Cq.shape[-2], u.shape[3]
    nc = -(-L // T)
    pad = nc * T - L

    def chunked(t):                          # [B, L, ...] -> [B, nc, T, ...]
        t = _pad_len(t, pad)
        return t.reshape(B, nc, T, *t.shape[2:])

    Bq = (chunked(Bt).permute(0, 1, 3, 4, 5, 2, 6)             # [..,R,Ri,T,N]
          .reshape(B, nc, H, R, Ri * T, N))
    Cc = (chunked(Cq).permute(0, 1, 3, 4, 5, 2, 6)             # [..,R,O,T,N]
          .reshape(B, nc, H, R, O * T, N))
    Uc = (chunked(u).permute(0, 1, 3, 5, 2, 4)                 # [..,Ri,T,G]
          .reshape(B, nc, H, Ri * T, G))
    Bq, Cc, Uc = Bq.contiguous(), Cc.contiguous(), Uc.contiguous()
    dAz, wz, wdz = (chunked(t).permute(0, 1, 3, 4, 2).contiguous()
                    for t in (dA, w, wd))
    Y, Sin = _chunk_fwd(Bq, Cc, Uc, dAz, wz, wdz, S0)
    S_fin = Sin[:, -1]
    y = (Y.view(B, nc, H, O, T, G).permute(0, 1, 4, 2, 5, 3)
         .reshape(B, nc * T, H, G, O)[:, :L])
    return y, S_fin


def _pad_len(t: torch.Tensor, pad: int) -> torch.Tensor:
    """Zero-pad the token axis (dim 1) at the end."""
    if pad == 0:
        return t
    widths = [0, 0] * (t.dim() - 2) + [0, pad]
    return F.pad(t, widths)


def ssd_chunked(core, x: torch.Tensor, u_sel: torch.Tensor, *,
                h0: torch.Tensor | None = None, gate_z=None, tok_raw=None,
                carry: dict | None = None, want_state: bool = False,
                chunk_size: int | None = None):
    """The whole sequence through the chunked SSD dual.

    x [B, L, H*G*R] drive, u_sel the token-map source, h0 [B, H, P, N]
    folded initial state (None = zeros), ``carry`` the continuation carries
    beyond ``hidden_state`` (rot_ang / write_prev / bc_conv_state).
    Returns (y [B, L, H*G*O], State or None, final hidden [B, H, P, N]);
    the hidden state keeps its graph (``SSDMimoSSM.h_final``).
    """
    B, L, _ = x.shape
    H, G, R, N, O = core.H, core.G, core.R, core.N, core.O
    carry = carry or {}
    prep = _prepare(core, x, u_sel, tok_raw, carry)
    law = prep.law
    with _no_autocast(x.device.type):
        dt, dA = law.dt, law.dA                                # [B,L,H,R]
        # collapsed write: w_t = lam_t Dt_t + n_t, n_t = Dt_{t+1}(1 - lam_{t+1});
        # the diagonal (s = t) coefficient is lam_t Dt_t
        if law.lam is not None:
            lam = law.lam[..., None]
            lag = dt * (1.0 - lam)                             # Dt_t(1-lam_t)
            wd = lam * dt
            w = wd + F.pad(lag[:, 1:], (0, 0, 0, 0, 0, 1))     # n_{L-1} = 0
            lag0 = lag[:, 0]
        else:
            w = wd = dt
            lag0 = None
        wp = carry.get("write_prev")
        if wp is not None and lag0 is None:
            raise ValueError(
                "state carry 'write_prev' with write_rule='euler': the "
                "one-tap write has no two-endpoint lag to fold")
        S0 = _initial_folded(prep, h0, wp, lag0, B, H, G, R, N)
        T = max(1, min(int(chunk_size or DEFAULT_CHUNK or default_chunk(N)), L))
        y_scan, S_fin = _chunk_scan(prep.Btil, prep.Cq, prep.u, dA, w, wd,
                                    S0, T)
        v_last = None
        if want_state and law.lam is not None:
            v_last = torch.einsum("bhrqn,bhgq->bhgrn", prep.Btil[:, -1],
                                  prep.u[:, -1])
    return _finish(core, prep, x, y_scan, gate_z,
                   S_fin.permute(0, 1, 4, 2, 3), want_state, carry, v_last,
                   h0=h0)


# ════════════════════════════════════════════════════════════════════════
#  Sequential scan over a whole sequence (training through the recurrence)
# ════════════════════════════════════════════════════════════════════════

class _LinearScanFn(torch.autograd.Function):
    """f_t = a_t * f_{t-1} + b_t (t = 0..L-1), a scalar per (b, t, h, r).

    Forward and backward are ONE fused op per token each (``addcmul``); the
    backward is the adjoint recurrence F_t = g_t + a_{t+1} F_{t+1} run in
    reverse, with da_t = <F_t, f_{t-1}>, db_t = F_t, df_{-1} = a_0 F_0.
    Saves the states f_t (one state per token)."""

    @staticmethod
    def forward(ctx, a, b, f0):
        # a [B,L,H,1,R,1], b [B,L,H,G,R,N], f0 [B,H,G,R,N]
        fs = torch.empty_like(b)
        f = f0
        for t in range(b.shape[1]):
            f = torch.addcmul(b[:, t], a[:, t], f)
            fs[:, t] = f
        ctx.save_for_backward(a, fs, f0)
        return fs

    @staticmethod
    def backward(ctx, gfs):
        a, fs, f0 = ctx.saved_tensors
        L = fs.shape[1]
        Fa = torch.empty_like(fs)
        acc = gfs[:, L - 1]
        Fa[:, L - 1] = acc
        for t in range(L - 2, -1, -1):
            acc = torch.addcmul(gfs[:, t], a[:, t + 1], acc)
            Fa[:, t] = acc
        fprev = torch.cat([f0[:, None], fs[:, :-1]], dim=1)
        ga = (Fa * fprev).sum((3, 5), keepdim=True)
        return ga, Fa, a[:, 0] * Fa[:, 0]


def _linear_scan_autograd(a, b, f0):
    """The same recurrence under plain autograd (verification variant)."""
    fs, f = [], f0
    for t in range(b.shape[1]):
        f = torch.addcmul(b[:, t], a[:, t], f)
        fs.append(f)
    return torch.stack(fs, dim=1)


def ssd_scan_seq(core, x: torch.Tensor, u_sel: torch.Tensor, *,
                 h0: torch.Tensor | None = None, gate_z=None, tok_raw=None,
                 carry: dict | None = None, want_state: bool = False):
    """The whole sequence through the token-by-token recurrence.

    Everything per-token parallel (maps, writes, reads) is batched over L;
    only the state recurrence f_t = alpha_t f_{t-1} + b_t runs in a loop,
    inside ``_LinearScanFn`` (adjoint-recurrence backward).  The two-endpoint
    forcing is b_t = Dt_t lam_t vtil_t + alpha_t Dt_t (1 - lam_t) vtil_{t-1}.
    Same returns as :func:`ssd_chunked`.
    """
    B, L, _ = x.shape
    H, G, R, N, O = core.H, core.G, core.R, core.N, core.O
    carry = carry or {}
    prep = _prepare(core, x, u_sel, tok_raw, carry)
    law = prep.law
    f = prep.f
    with _no_autocast(x.device.type):
        # vtil_t[b,h,g,r,:] = sum_q u_t[b,h,g,q] Btil_t[b,h,r,q,:], all t
        Bk = prep.Btil.transpose(3, 4).reshape(B, L, H, R, R * N)
        v = torch.matmul(prep.u, Bk).view(B, L, H, G, R, N)
        a = torch.exp(law.dA).view(B, L, H, 1, R, 1)           # alpha_t
        dt6 = law.dt.view(B, L, H, 1, R, 1)
        if law.lam is not None:
            lam6 = law.lam.view(B, L, H, 1, 1, 1)
            wp = carry.get("write_prev")
            vp0 = (_c(wp, f).view(B, 1, H, G, R, N) if wp is not None
                   else v.new_zeros(B, 1, H, G, R, N))
            v_prev = torch.cat([vp0, v[:, :-1]], dim=1)        # vtil_{t-1}
            b = (dt6 * lam6) * v + (a * dt6 * (1.0 - lam6)) * v_prev
        else:
            b = dt6 * v
        f0 = (_c(h0, f).view(B, H, G, R, N) if h0 is not None
              else v.new_zeros(B, H, G, R, N))
        if grad_mode() == "fn":
            fs = _LinearScanFn.apply(a, b, f0)
        else:
            fs = _linear_scan_autograd(a, b, f0)
        # read: y_t[b,h,g,:] = sum_(r,n) f_t[b,h,g,r,n] Cq_t[b,h,r,n,:]
        y_scan = torch.matmul(fs.view(B, L, H, G, R * N),
                              prep.Cq.transpose(-1, -2).reshape(B, L, H, R * N, O))
    # clones: the carries must own compact storage, not views into [B, L, ..]
    return _finish(core, prep, x, y_scan, gate_z, fs[:, -1].clone(),
                   want_state, carry, v[:, -1].clone(), h0=h0)
