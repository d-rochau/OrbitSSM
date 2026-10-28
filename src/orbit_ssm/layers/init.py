"""Initialization: shared formulas, and the depth/LM steps with their tags.

Initialization is spread over four places that must run in this order:

  1. the layer constructors draw their own parameters (the formulas shared by
     the families are below: `init_dt_bias`, `FREQ_INIT`, `SMALL_SCALE`,
     `skip_param`, `freq_param`);
  2. `BlockStack` calls `rescale_residual_out` once (the depth rule);
  3. `CausalLM` calls `init_lm` (the LM step);
  4. `CausalLM` draws the untied head.

Which group is touched by which step (a blank cell = untouched):

  parameter group         constructor            stack rescale       LM step               tag
  ----------------------  ---------------------  ------------------  --------------------  ---------------------------
  residual-out weights    nn.Linear default      kaiming_uniform     --                    `_normalize_w_num_layers`
  (mixer out_proj,        (kaiming_uniform,      (a=sqrt 5) redrawn                        (`mark_residual_out`,
  SwiGLU down)            a=sqrt 5)              / sqrt(n_residuals)                       `tag_residual_mixer`)
  other Linear weights    nn.Linear default; the                     --
  (in_proj, token maps)   gate / mixing maps zero; theta_proj
                          zero or x SMALL_SCALE
  Linear biases           nn.Linear default                          zeroed                `_no_reinit` exempts
  (a / trap / m3 maps)                                                                     the SSD `dt_proj.bias`
  dt bias (SSD            `init_dt_bias`                             SSD: exempt (tag);    `_no_weight_decay`
  dt_proj.bias; HM                                                   HM: a plain
  dt_bias)                                                           Parameter, untouched
  D skip                  `skip_param` (ones)                        --                    `_no_weight_decay`
  theta_freq /            `freq_param` (FREQ_INIT)                   --                    `_no_weight_decay`
  phase_freq
  nn.Embedding            default N(0, 1)                            N(0, 0.02)
  untied lm_head          nn.Linear default                          (CausalLM)            trunc_normal(0.02)
  norms, conv, A_log,     each module's own      --                  --
  rotation / map biases   formula

Why the LM step and the tags stay: the blanket bias zeroing and the residual
rescale change values of Linears that a layer built STANDALONE keeps (every
`a_proj` / `trap_proj` bias is nn.Linear-uniform outside an LM), and the
rescale redraws (it consumes RNG), so neither can move into the constructors
without changing a standalone layer's initial parameters or the draw stream.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

#: raw theta_freq / phase_freq init: omega = pi * tanh(3.8) = 0.99900 pi
FREQ_INIT = 3.8

#: kaiming_uniform(a=sqrt(5)) is U(+-1/sqrt(fan_in)), Var(w) = 1/(3 fan_in), so on a
#: unit-RMS input the pre-activation has std 1/sqrt(3) whatever fan_in is; scaling
#: by 0.1 sqrt(3) gives std 0.1 (distinct random directions from step 0, small
#: perturbation).  SSD's theta_init="small" and HM's gate_init="kaiming_small".
SMALL_SCALE = 0.1 * 3 ** 0.5


def sample_dt(n: int, dt_min: float, dt_max: float, dt_init_floor: float) -> torch.Tensor:
    """n step sizes, log-uniform on [dt_min, dt_max], floored at `dt_init_floor`."""
    lo, hi = math.log(dt_min), math.log(dt_max)
    return torch.exp(torch.rand(n) * (hi - lo) + lo).clamp(min=dt_init_floor)


def inv_softplus(dt: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus, stable for small `dt`."""
    return dt + torch.log(-torch.expm1(-dt))


def init_dt_bias(n: int, dt_min: float, dt_max: float, dt_init_floor: float) -> torch.Tensor:
    """softplus(bias) starts log-uniform in [dt_min, dt_max] (Mamba's dt init)."""
    return inv_softplus(sample_dt(n, dt_min, dt_max, dt_init_floor))


def no_decay(p: nn.Parameter) -> nn.Parameter:
    """Tag `p` as exempt from weight decay."""
    p._no_weight_decay = True
    return p


def skip_param(shape) -> nn.Parameter:
    """The D skip gain: ones, no weight decay."""
    return no_decay(nn.Parameter(torch.ones(shape)))


def freq_param(shape, init: float = FREQ_INIT) -> nn.Parameter:
    """A learned raw phase frequency (omega = pi tanh(freq)): FREQ_INIT, no weight decay."""
    return no_decay(nn.Parameter(torch.full(shape, float(init))))


# ========================================================================== #
#  Residual-out tags, stack rescale, LM-scope init                           #
# ========================================================================== #
#  Policy is expressed as parameter-level tags set at the construction site:
#
#    ``bias._no_reinit = True``
#        Preserve this bias as it was initialized. Without the tag, the LM init
#        zeros every Linear bias (residual-stack convention).
#
#    ``weight._normalize_w_num_layers = True``  (set by ``mark_residual_out``)
#        This weight is the output projection of a residual branch. Every
#        ``BlockStack`` that contains it applies the GPT-2 / Megatron
#        residual-depth rescale at construction (``rescale_residual_out``):
#        ``kaiming_uniform_(a=sqrt(5))`` followed by division by
#        ``sqrt(n_residuals)``, the number of residual branches in the stack.
#        Matching by tag rather than by parameter name keeps internal
#        sub-projections that happen to be called ``out_proj`` from being
#        rescaled.
#
#  Who tags: ``ResidualBlock`` tags the two residual outputs it can vouch for
#  -- its own SwiGLU ``down`` and the ``out_proj`` of a built-in mixer family
#  (``tag_residual_mixer``, strict). Any other mixer tags its own residual
#  output projection (``tag_official_mamba_mixer`` for a vendored mamba_ssm
#  mixer, ``mark_residual_out`` otherwise); the block does not guess at the
#  internals of a mixer this package did not build.


def mark_residual_out(weight: nn.Parameter) -> None:
    """Tag a weight as a residual-branch output projection.

    Every ``BlockStack`` containing the weight rescales it for depth at
    construction (``rescale_residual_out``).
    """
    weight._normalize_w_num_layers = True


def tag_official_mamba_mixer(mixer: nn.Module) -> None:
    """Tag the residual output projection of a vendored mamba_ssm mixer.

    Mamba1/2/3 mixers expose the residual output as ``mixer.out_proj``. We do
    not modify vendored source; we only attach the marker so the enclosing
    ``BlockStack`` knows which weight to rescale.
    """
    out_proj = getattr(mixer, "out_proj", None)
    if isinstance(out_proj, nn.Linear):
        mark_residual_out(out_proj.weight)


def tag_residual_mixer(mixer: nn.Module) -> None:
    """Tag the output projection of any mixer feeding a residual branch.

    ``ResidualBlock`` calls this for every built-in mixer family, so the
    enclosing ``BlockStack`` gives that projection the Mamba/GPT-style
    ``1/sqrt(n_residuals)`` depth scale. The function is intentionally strict:
    a production mixer without a linear ``out_proj`` would otherwise silently
    miss the stabilization policy.
    """
    out_proj = getattr(mixer, "out_proj", None)
    if not isinstance(out_proj, nn.Linear):
        raise TypeError(
            f"{type(mixer).__name__} must expose nn.Linear out_proj so its "
            "residual branch can receive depth-aware initialization"
        )
    mark_residual_out(out_proj.weight)


def rescale_residual_out(module: nn.Module, *, n_residuals: int) -> int:
    """Apply the residual-depth rescale to every tagged weight in ``module``.

    GPT-2 / Megatron / mamba_ssm rule: a weight tagged by ``mark_residual_out``
    is redrawn from PyTorch's default ``nn.Linear`` init,
    ``kaiming_uniform_(a=sqrt(5))`` = U(-1/sqrt(fan_in), 1/sqrt(fan_in)), and
    divided by ``sqrt(n_residuals)``, where ``n_residuals`` counts the residual
    branches adding into the stream. The weight is redrawn rather than divided
    in place, so the result does not depend on the init the owning mixer gave
    it, and a second application does not compound. ``BlockStack`` calls
    this at construction; call it directly only for a model that is not built
    on the stack. Returns the number of weights rescaled.
    """
    if isinstance(n_residuals, bool) or not isinstance(n_residuals, int) or n_residuals < 1:
        raise ValueError(f"n_residuals must be a positive int, got {n_residuals!r}")
    count = 0
    for p in module.parameters():
        if getattr(p, "_normalize_w_num_layers", False):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p.div_(math.sqrt(n_residuals))
            count += 1
    return count


def init_lm(model: nn.Module, *, embedding_std: float = 0.02) -> None:
    """Apply the LM-scope init (the part of the policy the stack does not own).

    - All ``nn.Linear`` biases are zeroed unless ``_no_reinit`` is set on them.
    - All ``nn.Embedding`` weights are redrawn from ``N(0, embedding_std)``.

    The residual-depth rescale is NOT applied here: ``BlockStack`` applies
    it once, at construction (``rescale_residual_out``), so an LM built on the
    stack is rescaled exactly once.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=embedding_std)
