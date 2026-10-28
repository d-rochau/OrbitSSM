"""Config presets that reproduce the official Mamba-1/2/3 layers at a given size.

Each ``mambaN_config(d_model, ...)`` returns the family config of one of this
package's layers (build it with ``Layer(config)`` or ``config.build()``);
`build_official_like` constructs the layer.

* Mamba-1 -> `HybridMambaSSM` (per-channel dt),
* Mamba-2 -> `SSDMimoSSM` at ``d_ssm_in = d_ssm_out = 1`` (SISO), the M2-exact flags,
* Mamba-3 -> `SSDMimoSSM` with a leading complex ``head_dynamic`` row block,
  module-exact (single shared ``in_proj``, no conv, no output norm).

Exactness is validated against copied official parameters (the m2exact,
m3exact and HM(M1) parity tests) with the copy functions of `tests/_official_maps.py`; each preset's docstring
lists the config fields it sets.

Precision: the presets do not set the kernel ``precision``; the class default
(``"io"``, tf32 at fp32 IO) applies.  Exactness tests against an fp32 official
model pass ``execution={"precision": "ieee"}`` explicitly (a numerics-validation
mode), so a preset-built training model never runs the slow no-tensor-core path
by accident.
"""

from __future__ import annotations

import math

from .ssm import HybridMambaConfig, SSDMimoConfig

_FAMILIES = ("mamba1", "mamba2", "mamba3")


def _check_family(name: str) -> None:
    if name not in _FAMILIES:
        raise ValueError(f"unknown preset {name!r}, must be one of {_FAMILIES}")


def required_sizes(name: str) -> tuple[str, ...]:
    """The size argument(s) a preset cannot infer: always ``("d_model",)``.

    Every other keyword defaults to the official layer's own default.
    """
    _check_family(name)
    return ("d_model",)


def mamba1_config(d_model: int, *, expand: int = 2, d_state: int = 16,
                  d_conv: int = 4, dt_rank: int | str = "auto",
                  **overrides) -> HybridMambaConfig:
    """The `HybridMambaConfig` of ``HybridMambaSSM(mamba1_config(...))`` reproducing official
    Mamba-1 (``mamba_ssm.modules.mamba_simple.Mamba``) at width ``d_model``.

    Parameters
    ----------
    d_model : int
        Block width.
    expand : int, default 2
        Inner width factor, ``d_inner = expand * d_model``.
    d_state : int, default 16
        State size per channel.
    d_conv : int, default 4
        Depthwise conv width.
    dt_rank : int or "auto", default "auto"
        Rank of the dt bottleneck; "auto" is Mamba-1's ``ceil(d_model / 16)``.
    **overrides
        Config fields replacing the preset's (how to run the layer is ``execution=`` of the layer, e.g.
        ``execution={"backend": "cuda"}``).

    Notes
    -----
    Sets one real-decay row block (``decay="coord_static"``, no rotation), ``d_head=1``
    (per-channel dt, so ``n_heads = d_inner``), ``proj_source="main_branch"``
    (token maps read the post-conv drive), ``out_norm=False`` (epilogue is
    ``y * silu(z) -> out_proj``), no B/C norm or bias, and ``write_rule="euler"``
    (write_rule="euler"; Mamba-1 has no endpoint forcing).  Validated against
    official Mamba-1 at fp32 roundoff (copied parameters).
    """
    if dt_rank == "auto":
        dt_rank = math.ceil(d_model / 16)
    core = dict(
        row_blocks=[{"frac": 1.0, "decay": "coord_static", "complex": False}],
        d_head=1,               # P = 1: per-channel dt (n_heads = d_inner)
        dt_rank=int(dt_rank),
        dt_init="random", dt_scale=1.0,
        proj_source="main_branch",   # token maps read the post-conv drive
        out_norm=False,          # epilogue: y * silu(z) -> out_proj
        b_norm=False, c_norm=False, bc_bias="none",
        write_rule="euler",
    )
    core.update(overrides)
    return HybridMambaConfig(d_model=d_model, d_state=d_state, expand=float(expand), d_conv=d_conv, **core)


def mamba2_config(d_model: int, *, expand: int = 2, d_state: int = 128,
                  d_head: int = 64, n_groups: int = 1, d_conv: int = 4,
                  **overrides) -> SSDMimoConfig:
    """The `SSDMimoConfig` of ``SSDMimoSSM(mamba2_config(...))`` reproducing official
    Mamba-2 (``mamba_ssm.modules.mamba2.Mamba2``) at width ``d_model``.

    Parameters
    ----------
    d_model : int
        Block width.
    expand : int, default 2
        Inner width factor, ``d_inner = expand * d_model``.
    d_state : int, default 128
        State size N (official Mamba-2's constructor default).
    d_head : int, default 64
        Channels per head P; ``d_inner`` must be a multiple of it, and
        ``n_heads = d_inner // d_head``.
    n_groups : int, default 1
        Must be 1: SSD-MIMO shares one B/C across all heads, so official
        grouped-B/C layouts (official ``ngroups > 1``) are refused. The returned
        ``n_groups`` kwarg is ``d_head`` (SSD-MIMO's "channels per head"
        convention, see `compat.py`).
    d_conv : int, default 4
        Depthwise conv width.
    **overrides
        Config fields replacing the preset's.

    Notes
    -----
    Sets ``d_ssm_in = d_ssm_out = 1`` (SISO), ``proj_source="layer_input"``
    with ``bc_conv=True`` (Mamba-2's joint xBC conv), no B/C norms or biases,
    no gates or ``r_mix_dyn``, one real ``head_static`` row block (static
    per-head decay, no rotation), ``skip=True`` and ``write_rule="euler"``.
    Validated against
    official Mamba-2 (copied parameters; copy function `compat.copy_official_mamba2_into_ssd_mimo_mixer`).
    """
    if n_groups != 1:
        raise ValueError(
            f"mamba2_config: n_groups must be 1 (SSD-MIMO shares one B/C "
            f"across every head; there is no counterpart to official "
            f"Mamba-2's grouped B/C layout with n_groups={n_groups}), got "
            f"{n_groups}")
    d_inner = int(expand) * d_model
    if d_inner % d_head:
        raise ValueError(
            f"mamba2_config: d_inner={d_inner} (expand*d_model) is not a "
            f"multiple of d_head={d_head}")
    n_heads = d_inner // d_head
    core = dict(
        d_ssm_in=1, d_ssm_out=1, n_groups=d_head, n_heads=n_heads,
        proj_source="layer_input", bc_conv=True,
        # one static per-head rate, no rotation
        row_blocks=[{"frac": 1.0, "decay": "head_static", "complex": False}],
        skip=True, skip_sharing="head",
        write_rule="euler", out_norm=True,
        # no B/C norms, biases or gates (norms off fits M2 exactly)
        b_norm=False, c_norm=False, bc_bias="none",
        b_gate=False, c_gate=False, r_mix_dyn=False,
        dt_min=1e-3, dt_max=1e-1,
    )
    core.update(overrides)
    return SSDMimoConfig(d_model=d_model, d_state=d_state, expand=float(expand), d_conv=d_conv, **core)


def mamba3_config(d_model: int, *, expand: int = 2, d_state: int = 128,
                  d_head: int = 64, d_conv: int = 0,
                  **overrides) -> SSDMimoConfig:
    """The `SSDMimoConfig` of ``SSDMimoSSM(mamba3_config(...))`` reproducing official
    Mamba-3 (``mamba_ssm.modules.mamba3.Mamba3``, SISO, ``is_outproj_norm=False``)
    at width ``d_model``, module-level.

    Parameters
    ----------
    d_model : int
        Block width.
    expand : int, default 2
        Inner width factor, ``d_inner = expand * d_model``.
    d_state : int, default 128
        State size N (official Mamba-3's constructor default; with
        ``rope_fraction=0.5`` it rotates ``N // 4`` pairs).
    d_head : int, default 64
        Channels per head P; ``d_inner`` must be a multiple of it.
        The returned ``n_groups`` kwarg is ``d_head`` (see `mamba2_config`).
    d_conv : int, default 0
        Mamba-3 has no causal conv anywhere (neither on x nor on B/C), so
        ``bc_conv`` resolves to False.
    **overrides
        Config fields replacing the preset's.

    Notes
    -----
    Structure matched to the official module:

    * every one of z/x/B/C/dt/A/trap/angle is a slice of one ``in_proj`` of the
      block's raw input: ``proj_source="layer_input"``, ``d_conv=0``;
    * no output norm (``out_norm=False``); the ``silu(z)`` gate is fused into
      the scan epilogue and the result goes straight to ``out_proj``;
    * a per-head ``D`` skip (``skip=True``);
    * head-shared angles on the first half of the state pairs only:
      ``theta_per_head=False`` and ``row_blocks`` = a complex ``head_dynamic``
      half plus a real half (``head_dynamic`` is Mamba-3's data-dependent
      |A_t|), ``write_rule="trapezoid"``, fixed pi frequency, no read offset;
    * B/C RMSNorms with per-head Q/K biases folded before the rotation
      (``bc_bias="head"``, required by the rotation);
    * ``a_proj_bias=False`` and ``trap_proj_bias=False``: official's packed
      A/trap rows are bias-free; Delta keeps its bias (official ``dt_bias``).

    ``a_act`` stays at the class default ``"heavy_tail"``; for parity against
    an older reference pass ``a_act=compat.mamba3_reference_a_act(official)``
    (``"softplus"`` for mamba_ssm 2.3.2.post1).  Validated against
    official Mamba-3 (copied parameters; copy function `compat.copy_official_mamba3_into_ssd_mimo_mixer`).
    """
    d_inner = int(expand) * d_model
    if d_inner % d_head:
        raise ValueError(
            f"mamba3_config: d_inner={d_inner} (expand*d_model) is not a "
            f"multiple of d_head={d_head}")
    n_heads = d_inner // d_head
    core = dict(
        d_ssm_in=1, d_ssm_out=1, n_groups=d_head, n_heads=n_heads,
        proj_source="layer_input", write_rule="trapezoid",
        # per-head selected rate; rotated fraction 0.5 = complex half + real half
        row_blocks=[{"frac": 0.5, "decay": "head_dynamic", "complex": True},
              {"frac": 0.5, "decay": "head_dynamic", "complex": False}],
        # head-shared angles, fixed pi frequency, no read offset
        theta_per_head=False, use_theta_h=False, learn_theta_freq=False,
        # B/C RMSNorms + per-head Q/K biases folded before the rotation
        b_norm=True, c_norm=True, bc_bias="head",
        skip=True, skip_sharing="head",
        a_proj_bias=False, trap_proj_bias=False,
        b_gate=False, c_gate=False, r_mix_dyn=False,
        dt_min=1e-3, dt_max=1e-1, out_norm=False,
    )
    core.update(overrides)
    return SSDMimoConfig(d_model=d_model, d_state=d_state, expand=float(expand), d_conv=d_conv, **core)


_PRESET_FNS = {"mamba1": mamba1_config, "mamba2": mamba2_config,
              "mamba3": mamba3_config}


def build_official_like(name: str, d_model: int, *, execution=None, **config_fields):
    """Construct the layer for preset ``name`` ("mamba1", "mamba2" or "mamba3") at width ``d_model``,
    run as ``execution`` (an `Execution` or its dict).  ``config_fields`` forward to the matching
    ``mambaN_config`` function.  Mamba-1 builds a ``HybridMambaSSM``, Mamba-2/3 an ``SSDMimoSSM``."""
    _check_family(name)
    from .ssm import HybridMambaSSM, SSDMimoSSM
    cls = HybridMambaSSM if name == "mamba1" else SSDMimoSSM
    return cls(_PRESET_FNS[name](d_model, **config_fields), execution=execution)
