"""Parameter maps between this repository's mixers and official reference modules (`mamba_ssm`).

Feeds the exactness tests of the mixers against official Mamba-1/2/3
(m2exact, m3exact, HM(M1) at fp32 roundoff). The Mamba-2 map
follows the rule "with norms off we fit M2's mixer exactly".
"""

from __future__ import annotations

import torch


def mamba2_state_dict_to_ssd_mimo_mixer(
    official_sd: dict, *, d_inner: int, n_heads: int, d_state: int,
) -> dict:
    """Map an official ``mamba_ssm.modules.mamba2.Mamba2`` state dict onto the
    equivalent ``SSDMimoSSM`` state dict.

    The two mixers compute the same function only at one config: SISO
    (``d_ssm_in = d_ssm_out = 1``), ``proj_source="layer_input"``,
    ``b_norm=c_norm=False``, ``bc_bias="none"``, static decay (``head_static``),
    no rotation, no trapezoid, SwiGLU gate on, ``skip=True``, ``bc_conv=True``
    and official ``ngroups=1`` (SSD-MIMO shares one B/C across all heads).
    This is the `presets.mamba2_kwargs` config.

    Parameters
    ----------
    official_sd : dict
        State dict of the official module.
    d_inner, n_heads, d_state : int
        Inner width, head count H and state size N of the module.

    Returns
    -------
    dict
        New ``{our_param_name: tensor}`` dict of cloned tensors (the input is
        not mutated), including the ``pair_layout_code`` buffer. Load with
        ``mixer.load_state_dict(mapped, strict=False)``; at SISO every mixer
        parameter has an official counterpart, ``strict=False`` only admits
        ``r_mix`` (H, R), which exists at R > 1 and has none.

    Notes
    -----
    * Official ``in_proj.weight`` rows are ``[z, x, B (N), C (N), dt (H)]``;
      ours carries only ``[x, z]`` (opposite order), and B, C, dt live in the
      core's ``B_proj`` / ``C_proj`` / ``dt_proj``.
    * Official ``conv1d`` rows ``[x, B, C]`` split by channel (the conv is
      depthwise): the first ``d_inner`` go to ``conv1d``, the remaining ``2N``
      to ``bc_conv1d``.
    * ``dt_bias`` -> ``dt_proj.bias``; ``A_log`` (H,) -> ``A_log`` (H, 1), same
      rate ``exp(A_log)`` and sign convention; ``D`` (H,) -> ``D``, applied
      inside the gate at the post-conv-SiLU x on both sides.
    * ``norm.weight`` -> ``norm.weight``: a plain RMSNorm over the full width.
      SSD-MIMO applies the SwiGLU gate in the scan epilogue, so ``y`` is
      already ``scan * silu(z)`` when the norm runs, matching official's gated
      RMSNorm with ``norm_before_gate=False``.
    """
    N, H = d_state, n_heads
    w = official_sd["in_proj.weight"]
    z_rows, x_rows = w[:d_inner], w[d_inner:2 * d_inner]
    b_rows = w[2 * d_inner:2 * d_inner + N]
    c_rows = w[2 * d_inner + N:2 * d_inner + 2 * N]
    dt_rows = w[2 * d_inner + 2 * N:2 * d_inner + 2 * N + H]

    conv_w = official_sd["conv1d.weight"]           # (d_inner+2N, 1, d_conv)
    conv_b = official_sd["conv1d.bias"]              # (d_inner+2N,)

    mapped = {
        # mixer's own in_proj: OUR order is [x, z], official's is [z, x]
        "in_proj.weight": torch.cat([x_rows, z_rows], dim=0).clone(),
        "conv1d.weight": conv_w[:d_inner].clone(),
        "conv1d.bias": conv_b[:d_inner].clone(),
        "B_proj.weight": b_rows.clone(),
        "C_proj.weight": c_rows.clone(),
        "dt_proj.weight": dt_rows.clone(),
        "dt_proj.bias": official_sd["dt_bias"].clone(),
        "A_log": official_sd["A_log"].clone().unsqueeze(1),   # (H,) -> (H,1)
        "D": official_sd["D"].clone(),
        "bc_conv1d.conv.weight": conv_w[d_inner:d_inner + 2 * N].clone(),
        "bc_conv1d.conv.bias": conv_b[d_inner:d_inner + 2 * N].clone(),
        "norm.weight": official_sd["norm.weight"].clone(),
        "out_proj.weight": official_sd["out_proj.weight"].clone(),
        # B/C rows are copied verbatim, so the map is in the interleaved basis
        # (code 0); the core's load hook refuses a state dict that does not
        # declare its layout, so a `pair_layout="half"` mixer refuses it.
        # Created on the source device so the returned map is single-device.
        "pair_layout_code": torch.zeros((), dtype=torch.int64,
                                            device=w.device),
    }
    return mapped


def copy_official_mamba2_into_ssd_mimo_mixer(official, mixer) -> list[str]:
    """Copy every mappable parameter of an official ``Mamba2`` module into an
    M2-exact ``SSDMimoSSM``, in place (config and map:
    :func:`mamba2_state_dict_to_ssd_mimo_mixer`).

    Requires ``mixer.skip_sharing == "head"``. Returns the sorted names of
    the mixer's parameters that were not copied: empty at SISO (a non-empty
    list there means the config gained a counterpart-free parameter),
    ``r_mix`` at R > 1.
    """
    if mixer.skip_sharing != "head":
        raise ValueError("Official default parameter parity requires skip_sharing='head'; "
                         "use the named Mamba preset to preserve parameter tying.")
    d_inner = mixer.d_inner
    n_heads = mixer.H
    d_state = mixer.N
    mapped = mamba2_state_dict_to_ssd_mimo_mixer(
        official.state_dict(), d_inner=d_inner, n_heads=n_heads,
        d_state=d_state)
    params = dict(mixer.named_parameters())
    mixer.load_state_dict(mapped, strict=False)
    return sorted(n for n in params if n not in mapped)


# ── Mamba-1 ───────────────────────────────────────────────────────────
def copy_official_mamba1_into_hm_mixer(official, hm) -> list[str]:
    """Copy every parameter of an official Mamba-1
    (``mamba_ssm.modules.mamba_simple.Mamba``) into an M1-exact
    ``HybridMambaSSM`` (`presets.mamba1_kwargs`, with
    ``dt_rank == official.dt_rank``), in place.

    ``official.x_proj`` is one Linear(d_inner -> dt_rank + 2N) producing
    [dt, B, C]; HM splits it into ``dt_low`` and ``bc_proj``, which receive
    the matching row slices. All other pairs are same-shape tensors (in_proj,
    conv1d, dt_proj, dt bias, A_log, D, out_proj).

    Returns the sorted names of trainable parameters not copied (empty for the
    M1-exact config; the benchmark pair builder refuses a non-empty list).
    """
    dt_rank = official.dt_rank
    if hm.dt_low is None or hm.dt_low.weight.shape[0] != dt_rank:
        raise ValueError("M1 parameter parity needs dt_rank == official.dt_rank")
    pairs = {
        "in_proj.weight": official.in_proj.weight,
        "conv1d.weight": official.conv1d.weight,
        "conv1d.bias": official.conv1d.bias,
        "dt_low.weight": official.x_proj.weight[:dt_rank],
        "bc_proj.weight": official.x_proj.weight[dt_rank:],
        "dt_proj.weight": official.dt_proj.weight,
        "dt_bias": official.dt_proj.bias,
        "A_log": official.A_log,
        "D": official.D,
        "out_proj.weight": official.out_proj.weight,
    }
    if getattr(official.in_proj, "bias", None) is not None or getattr(official.out_proj, "bias", None) is not None:
        raise ValueError("M1 parameter parity: official in_proj/out_proj biases have no counterpart")
    params = dict(hm.named_parameters())
    with torch.no_grad():
        for name, src in pairs.items():
            dst = params[name]
            dst.copy_(src.detach().reshape(dst.shape))
    return sorted(n for n, p in params.items() if p.requires_grad and n not in pairs)


# ── Mamba-3 ───────────────────────────────────────────────────────────
def mamba3_state_dict_to_ssd_mimo_mixer(
    official_sd: dict, *, d_inner: int, n_heads: int, d_state: int,
) -> dict:
    """Map an official ``mamba_ssm.modules.mamba3.Mamba3`` state dict (SISO,
    ``is_mimo=False``, ``is_outproj_norm=False``) onto the equivalent
    ``SSDMimoSSM`` state dict.

    Target config is `presets.mamba3_kwargs`: SISO, ``proj_source="layer_input"``,
    ``d_conv=0``, a leading complex ``head_dynamic`` row half with
    ``theta_per_head=False`` (official's ``Angles`` projection is head-shared),
    ``bc_bias="head"``, ``b_norm=c_norm=True``, ``skip=True``,
    ``out_norm=False`` and official ``ngroups=1``.

    Parameters
    ----------
    official_sd : dict
        State dict of the official module.
    d_inner, n_heads, d_state : int
        Inner width, head count H and state size N of the module.

    Returns
    -------
    dict
        New ``{our_param_name: tensor}`` dict of cloned tensors, including the
        interleaved ``pair_layout_code`` buffer (loading into another layout
        needs an explicit basis conversion). Load with
        ``mixer.load_state_dict(mapped, strict=False)``. The named preset has
        no extra A / trapezoid projection biases; custom targets that enable
        them get those zeroed by `copy_official_mamba3_into_ssd_mimo_mixer`.

    Notes
    -----
    * Official ``in_proj.weight`` rows are ``[z, x, B (N), C (N), dd_dt (H),
      dd_A (H), trap (H), angle (ka)]``; ``ka``, the rotated-pair count
      (set by official ``rope_fraction``), is inferred from the row count.
      Ours carries ``[x, z]``; the rest map onto the core's ``B_proj`` /
      ``C_proj`` / ``dt_proj`` / ``a_dyn_proj`` / ``trap_proj`` /
      ``theta_proj``, all reading the same block input.
    * ``dt_bias`` -> ``dt_proj.bias``; ``D``, ``out_proj.weight`` and the
      B/C RMSNorm weights copy directly. Official's explicit ``eps=1e-5`` vs
      our dtype-dependent default is not bridged (see the exactness test's
      tolerance note).
    * ``B_bias`` / ``C_bias`` (H, 1, N) -> (H, 1, N, 1): an axis transpose, not
      a reshape; both are added inside the rotation (Mamba-3's fold order).
    * No entries for ``norm.weight`` (no output norm), ``conv1d`` /
      ``bc_conv1d`` (no conv), ``A_log`` (``head_dynamic`` decay is fully
      data-dependent) or ``r_mix`` (absent at SISO).
    """
    N, H = d_state, n_heads
    w = official_sd["in_proj.weight"]
    d_in_proj = w.shape[0]
    ka = d_in_proj - 2 * d_inner - 2 * N - 3 * H   # rotated-pair count
    if ka < 1:
        raise ValueError(
            f"mamba3_state_dict_to_ssd_mimo_mixer: inferred ka={ka} "
            f"rotated pairs from in_proj.weight's row count "
            f"({d_in_proj}) at d_inner={d_inner}, d_state={N}, "
            f"n_heads={H} -- check those match the official module")
    z_rows, x_rows = w[:d_inner], w[d_inner:2 * d_inner]
    b_rows = w[2 * d_inner:2 * d_inner + N]
    c_rows = w[2 * d_inner + N:2 * d_inner + 2 * N]
    dt_rows = w[2 * d_inner + 2 * N:2 * d_inner + 2 * N + H]
    a_rows = w[2 * d_inner + 2 * N + H:2 * d_inner + 2 * N + 2 * H]
    trap_rows = w[2 * d_inner + 2 * N + 2 * H:2 * d_inner + 2 * N + 3 * H]
    angle_rows = w[2 * d_inner + 2 * N + 3 * H:]
    assert angle_rows.shape[0] == ka

    mapped = {
        # mixer's own in_proj: OUR order is [x, z], official's is [z, x]
        "in_proj.weight": torch.cat([x_rows, z_rows], dim=0).clone(),
        "B_proj.weight": b_rows.clone(),
        "C_proj.weight": c_rows.clone(),
        "dt_proj.weight": dt_rows.clone(),
        "dt_proj.bias": official_sd["dt_bias"].clone(),
        "a_dyn_proj.weight": a_rows.clone(),
        "trap_proj.weight": trap_rows.clone(),
        "theta_proj.weight": angle_rows.clone(),
        "D": official_sd["D"].clone(),
        # (H, 1, N) -> (H, R=1, N, 1): transpose, then the sibling axis
        "B_bias": official_sd["B_bias"].clone().transpose(1, 2).unsqueeze(1),
        "C_bias": official_sd["C_bias"].clone().transpose(1, 2).unsqueeze(1),
        "B_norm.weight": official_sd["B_norm.weight"].clone(),
        "C_norm.weight": official_sd["C_norm.weight"].clone(),
        "out_proj.weight": official_sd["out_proj.weight"].clone(),
        # interleaved basis declared, on the source device (see the M2 map)
        "pair_layout_code": torch.zeros((), dtype=torch.int64,
                                            device=w.device),
    }
    return mapped


def mamba3_reference_a_act(official) -> str:
    """The |A| activation of an official Mamba-3 module (or its provider module):
    "heavy_tail" or "softplus".

    Probes for ``heavy_tail_activation`` in ``mamba_ssm/modules/mamba3.py``
    (added upstream; mamba_ssm 2.3.2.post1 wheels predate it and use
    softplus). The wheel version does not change across that commit, so the
    symbol is the probe.
    """
    import inspect
    provider = (official if inspect.ismodule(official)
                else inspect.getmodule(type(official)))
    return ("heavy_tail" if callable(getattr(provider, "heavy_tail_activation", None))
            else "softplus")


def copy_official_mamba3_into_ssd_mimo_mixer(official, mixer) -> list[str]:
    """Copy every mappable parameter of an official ``Mamba3`` module (SISO,
    default config) into an M3-exact ``SSDMimoSSM``, in place (config and map:
    :func:`mamba3_state_dict_to_ssd_mimo_mixer`).

    Requires the target's ``a_act`` to match the reference
    (`mamba3_reference_a_act`) and ``skip_sharing == "head"``. Returns the
    sorted unmapped parameter names (none for the named preset). Extra
    ``trap_proj.bias`` / ``a_dyn_proj.bias`` of custom targets are zeroed for
    copied-weight functional parity, which does not imply training-trajectory
    equivalence.
    """
    act = mamba3_reference_a_act(official)
    if getattr(mixer, "a_act", "heavy_tail") != act:
        raise ValueError(
            f"Mamba-3 parameter parity: the reference uses the {act!r} "
            f"|A| activation but the target has a_act={mixer.a_act!r}; build the "
            f"target with a_act={act!r} (mamba3_reference_a_act) to compare.")
    if mixer.skip_sharing != "head":
        raise ValueError("Official default parameter parity requires skip_sharing='head'; "
                         "use the named Mamba preset to preserve parameter tying.")
    d_inner = mixer.d_inner
    n_heads = mixer.H
    d_state = mixer.N
    mapped = mamba3_state_dict_to_ssd_mimo_mixer(
        official.state_dict(), d_inner=d_inner, n_heads=n_heads,
        d_state=d_state)
    params = dict(mixer.named_parameters())
    mixer.load_state_dict(mapped, strict=False)
    with torch.no_grad():
        for name in ("trap_proj.bias", "a_dyn_proj.bias"):
            if name in params:
                params[name].zero_()
    return sorted(n for n in params if n not in mapped)
