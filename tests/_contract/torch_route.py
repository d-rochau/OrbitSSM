"""The torch-route half of the contract suite (CPU).

The CUDA cells (tests/test_contract.py) exercise the kernels.  The torch / sequential routes, the shared laws (decay,
trapezoid, D skip, biases, gates), the epilogues and the State handling (carries, dtypes, continuation) are what a
CPU-only change can break, and no kernel cell sees them.  These checks run a family's LAYER on `backend="torch"` over a
small covering set of its torch-visible options, at fp32 / fp64 / bf16, against fp64 references:

  oracle        forward and EVERY gradient (input, parameters) against the family's INDEPENDENT fp64 oracle
                (`adapter.torch_oracle`), in absolute arithmetic (`_tolerance.terms_abs`)
  continuation  two calls joined by the returned State == one call, from an arbitrary complete initial State: the output,
                every carry of the final State, and the gradients through the carries (input, initial State, parameters),
                against the fp64 whole call
  step          the single-token step loop == the parallel forward, from the default `init_state()` and from the same
                arbitrary State: output, final State, gradients
  state dtypes  the default carries are promote(parameter dtype, fp32); a returned State keeps the dtype of every carry
                that was supplied; a bf16 layer steps from its default State

Tolerances follow the arithmetic.  fp32 / fp64: `|a - r| <= c u sum|terms|` element-wise with
sum|terms| from the fp64 reference run in absolute arithmetic and c = sqrt(arms * depth) + amplification (depth: the
module's own sums, `module_depth`, plus the recurrence's, `TorchLayer.scan_depth`; a gradient additionally sums B L
adjoints; amplification: the decay / phase magnitude the reference reports).  bf16: `tol16` over >= 5 draws with the
pooled signed-bias check (`assert_sixteen_bit_close`, falling back to the terms form for a unit whose reference is ~0).

An adapter plugs in with three hooks (tests/_contract/__init__.py):

  torch_cells()                   [Cell]: a deterministic pair-covering set of the family's torch-visible options
  torch_layer(cfg, dtype, seed)   a `TorchLayer`: the layer on backend="torch" in `dtype`, parameters off their
                                  initialisation (`perturb`), plus its independent oracle
  torch_dtypes(cfg)               optional, the dtype names the cell runs at (default fp32, fp64, bf16)
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import re
from typing import Callable, Optional

import torch
import torch.utils._pytree as pytree

from _tolerance import (DrawPool, NearZeroReference, TermsResult, _signed_bias, assert_close_terms, assert_rel_close,
                        draw_seeds, error_tail_fails, escalation_seeds, is_sixteen_bit, terms_abs, tol16, unit_roundoff)

#: the dtype of the MODULE (parameters, inputs, carries) per cell dtype name.  "amp" is the training regime: fp32 module and
#: inputs, the call under `torch.autocast(cpu, bfloat16)` -- bf16 activations, fp32 accumulation, fp32 carries
DTYPES = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.bfloat16, "amp": torch.float32}
#: bf16-rounded operands along one path of an autocast call (the terms of the 16-bit rule's depth): the six roundings of a
#: 16-bit LAYER's summand (IO_ROUNDINGS: in_proj output, conv + SiLU, the token-map projections, their norm / gate products,
#: the operand of the scan's contraction, the stored output) and the two operands of the scan's contraction, each rounded once
#: by the autocast matmul (the decay-weighted write and the readout).  The sums themselves accumulate in fp32 and the carries
#: stay fp32, so neither the sequence length nor the width of a sum enters this count (they enter the fp32 part of the
#: terms-form fallback, `Rule.f_y` / `f_g`); a gradient passes the same operations once more with bf16 cotangents
AMP_ROUNDINGS = 8


def amp_twins(cells, limit=None):
    """The mixed-precision cells: one twin of every bf16 LAYER cell (at most `limit` of them) (same options, same draws), dtype "amp".  The mode is the
    bf16 cell's arithmetic with the module in fp32 and the call under autocast, so the twin set inherits its option coverage
    and the fp32 / fp64 / bf16 cells keep the composition they were pinned with."""
    from .core import Cell
    out = []
    for c in cells:
        if limit is not None and len(out) >= limit:
            break
        if c.kind == "cover" and c.c.get("dtype") == "bf16":
            out.append(Cell(c.family, c.kind, c.name.replace("dtype=bf16", "dtype=amp", 1),
                            tuple(sorted({**c.c, "dtype": "amp"}.items()))))
    return out


def is_amp(name):
    return name == "amp"


def rule_dtype(name):
    """The format whose unit roundoff scales the bound: bf16 for the mixed-precision mode, else the module's dtype."""
    return torch.bfloat16 if is_amp(name) else DTYPES[name]


def autocast(name):
    """The context the cell's calls run in: CPU bf16 autocast for "amp", nothing otherwise."""
    return torch.autocast("cpu", dtype=torch.bfloat16) if is_amp(name) else contextlib.nullcontext()
#: roundings in the 16-bit format along one path of a 16-bit LAYER's summand (the terms form's c for a unit whose reference
#: is ~0 against its summands): in_proj output, conv + SiLU, the token-map projections (B, C, dt, gates), their norm / gate
#: products, the operand of the scan's contraction, and the stored output -- six.  The fp32 part of the arithmetic is
#: expressed in units of that format's u (mh_oracle.oracle_c does the same with the kernel's three).  PREMISE: the carries and the
#: per-token transition (clock, decay factor) run in the carries' working dtype (fp32), so the sequence length does not enter.
#: A step that forms the transition in the 16-bit format puts one rounding per token on the whole carry (sqrt(L) of them), which
#: this count does not cover -- that was a defect of the MH step, not of the count (mh_torch.REGRESSION_CELLS).
IO_ROUNDINGS = 6


@dataclasses.dataclass
class TorchLayer:
    """One cell's layer plus what the generic checks need to know about it."""
    layer: torch.nn.Module
    d_in: int                                   # width of the layer input
    #: oracle(layer64, x, stats) -> y: the family's INDEPENDENT physical law of the layer's fp64 parameters, written
    #: with plain differentiable torch ops (no production helper); `stats` (dict) receives "amp", the decay / phase
    #: amplification of the call.  None: the cell has no layer-level independent oracle (only G2-G4 run).
    oracle: Optional[Callable] = None
    scan_depth: int = 0                         # depth of the recurrence's longest sum feeding one output element
    B: int = 2
    L: int = 37
    #: False when the family's absolute-arithmetic sums are not a usable scale (a family with dense rotation products: sum|terms|
    #: grows like |Q|^t, a vacuous bound).  The references then run as plain fp64 and the scoring is NORMWISE against the
    #: reference (`score`), with the gradients of one call scaled by their largest norm.
    use_terms: bool = True
    #: carry_fix(layer, key, value, seed) -> a feasible value for the arbitrary initial State's carry `key` (or None: keep
    #: the random one); a carry returned here is a constant of the call, not a gradient leaf.  E.g. a co-moving frame
    #: must be a rotation (an arbitrary FEASIBLE State).
    carry_fix: Optional[Callable] = None
    #: carries that are phases modulo 2 pi (SSD `rot_ang`, stored wrapped to (-pi, pi]): the error of such a carry is the
    #: DISTANCE on the circle, so a value on either side of the branch cut is compared as the same angle
    circular: tuple = ()


def carry_dtype(dtype):
    """The default carry dtype of a layer whose parameters are `dtype`: never below fp32."""
    return torch.promote_types(dtype, torch.float32)


def perturb(layer, seed, scale=0.3):
    """Move every parameter off its initialisation, deterministically: a zero-initialised bias, gate or gain would
    otherwise make a defect in the code that reads it unobservable.  Multiplicative for non-zero entries, additive
    (at the tensor's own magnitude, floor 0.1) for the zero-initialised ones."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            if not p.is_floating_point():
                continue
            n = torch.randn(p.shape, generator=g, dtype=torch.float64)
            mag = p.detach().double().abs().mean().clamp_min(0.1)
            p.copy_((p.double() * (1 + 0.25 * scale * n) + (p.double() == 0) * scale * mag * n).to(p.dtype))
    return layer


def module_depth(layer):
    """Sum of every reduction a value passes through in the module's own (non-recurrence) arithmetic: a Linear's
    in_features + 2, a depthwise Conv's taps + 2, a norm's width; parallel branches are summed too (an upper bound)."""
    d = 0
    for mod in layer.modules():
        if isinstance(mod, torch.nn.Linear):
            d += mod.in_features + 2
        elif isinstance(mod, torch.nn.Conv1d):
            d += mod.kernel_size[0] * mod.in_channels // mod.groups + 2
        elif "Norm" in type(mod).__name__:
            ns = getattr(mod, "normalized_shape", None)
            d += int(ns[-1]) if ns else 0
    return d


class Pool(DrawPool):
    """`DrawPool` (pooled signed bias per tensor) plus the draw statistics of 16-bit GRADIENTS scored normwise: the bound is
    the typical scale (A7), so the median over the draws must lie inside it and no draw beyond twice it."""

    def __init__(self):
        super().__init__()
        self.errs = {}
        self.capture = None            # a dict: an ESCALATION draw collects (got, ref, bound) per tensor and checks nothing
        self.cap_errs = {}             # ... and (error, bound) of every normwise-scored gradient

    def err(self, name, err, bound):
        self.errs.setdefault(name, []).append((err, bound))

    def assert_unbiased(self, rerun=None, seeds=(), errs_of=None):
        """The pooled bias (`DrawPool`, with escalation) and the typical-scale rule of the gradient errors.  The five-draw screen
        (median <= bound, max <= 2 bound, each draw against its own bound) that flags a tensor escalates it to the full set of
        draws (`errs_of(seed) -> {tensor: (error, bound)}`), and the verdict is taken on the median (<= bound) and the 90th
        percentile (<= 2 bound), `_tolerance.error_tail_fails`.  Without a redraw the screen decides alone."""
        super().assert_unbiased(rerun, seeds)
        bad = []
        flagged = {n for n, d in self.errs.items() if error_tail_fails([e / b for e, b in d], screen=True)[0]}
        if flagged and errs_of is not None and seeds:
            for seed in seeds:
                for name, (e, b) in errs_of(seed).items():
                    if name in flagged:
                        self.err(name, e, b)
        for name in sorted(flagged):
            draws = self.errs[name]
            fails, med, tail = error_tail_fails([e / b for e, b in draws], screen=len(draws) < 10)
            if fails:
                bad.append(f"{name}: {len(draws)} draws, median error / bound {med:.2f} (<= 1), "
                           f"{'max' if len(draws) < 10 else '90th percentile'} {tail:.2f} (<= 2); "
                           f"errors {[f'{e:.3g}' for e, _ in draws][:8]}, bounds {sorted({round(b, 4) for _, b in draws})[:3]}")
        assert not bad, "16-bit gradient error outside the typical-scale rule:\n  " + "\n  ".join(bad)


class Rule:
    """The derived bound of one (layer, dtype, reference) triple: depth of an output and of a gradient, amplification."""

    def __init__(self, tl, dtype, amp, extra_depth=0, mixed=False):
        self.dtype = torch.bfloat16 if mixed else dtype
        self.mixed = mixed
        self.amp = amp
        self.depth_y = module_depth(tl.layer) + tl.scan_depth + extra_depth
        # terms-scaled: a parameter gradient sums B L adjoints, each as deep as the output; normwise (no terms scale): the adjoint
        # scan (one more chain) plus the contraction onto the B L tokens
        self.depth_g = (tl.B * tl.L + 1) * self.depth_y if tl.use_terms else 2 * self.depth_y + 2 * tl.B * tl.L
        # fp32 layer against the fp64 reference: one rounding arm; an fp64 layer against it: two
        self.arms = 2 if dtype == torch.float64 else 1
        if mixed:
            # mixed precision: the fp32 sums (depth above) stay the fp32 part of the terms-form fallback; the 16-bit rule's
            # depth counts the bf16-rounded operands (AMP_ROUNDINGS), twice for a gradient (bf16 cotangents)
            self.f_y, self.f_g = self.depth_y, self.depth_g
            # The decay / phase amplification of the call (`amp`: the accumulated log-decay and rotation angle the oracle reports)
            # multiplies the rounding of the bf16 operands that feed them (a projection's bf16 output error u |theta_t|
            # accumulates over the cumulative angle) and is added to the count, as in the normwise fp32 rule `2 (depth + amp) u`.
            # autocast also rounds every Linear / Conv weight to bf16 per call (the fp32 reference keeps the fp32 weight, unlike a
            # bf16 module whose reference shares its rounded weights): one rounding per such module, parallel branches summed
            n_weights = sum(isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)) for m in tl.layer.modules())
            self.depth_y = AMP_ROUNDINGS + n_weights + amp
            if not tl.use_terms:
                # No sum|terms| to scale a bf16-rounded operand's error by (no terms scale): a NORMWISE error over a sum of n terms
                # with rounded operands is u sqrt(n) of the terms' norm, which a cancelling sum turns into u sqrt(n) of its
                # value, so the summands of the fp32-accumulated sums (the fp32 depth) join the count, as in a bf16 cell
                self.depth_y += self.f_y
            # a parameter gradient reduces over the B L tokens, each term with bf16-rounded operands (tol16: "the token count
            # for a whole-sequence parameter reduction"); the normwise form adds the adjoint chain, as in a bf16 cell
            self.depth_g = ((tl.B * tl.L + 1) * self.depth_y if tl.use_terms else
                            2 * self.depth_y + 2 * tl.B * tl.L)

    def c(self, depth, u16=False):
        c32 = math.sqrt(self.arms * depth) + self.amp
        if not u16:
            return c32
        return IO_ROUNDINGS + c32 * unit_roundoff("fp32") / unit_roundoff(self.dtype)


#: a tensor whose reference norm is below this fraction of its group's scale is scored against the scale instead (the
#: normwise form of the guarded relative error: a gradient that cancels to ~0 must not turn rounding into a red)
NORMWISE_FLOOR = 1e-2


def score(name, got, ref, terms, rule, kind="y", pool=None, scale=None):
    """`got` (the layer's, any dtype) against `ref` (fp64) with sum|terms| `terms`, under the rule's bound.
    kind "y": outputs and carries; "g": gradients.  A reference of None (no path) requires no gradient or an exact zero.
    `terms` None: the NORMWISE form  ||got - ref|| / max(||ref||, NORMWISE_FLOOR scale) <= u (sqrt(arms depth) + amp)
    (16-bit: tol16(depth)), the signed bias pooled over draws (`scale`: the group's largest reference norm)."""
    depth = rule.depth_y if kind == "y" else rule.depth_g
    fdepth = (rule.f_y if kind == "y" else rule.f_g) if rule.mixed else depth      # the fp32 part of the terms form
    key = re.sub(r" seed \d+", "", name)           # the pooled statistics of a tensor span the draws
    if ref is None:
        assert got is None or got.count_nonzero() == 0, f"{name}: gradient where the oracle has none"
        return
    assert got is not None, f"{name}: no gradient"
    assert torch.isfinite(got).all(), f"{name}: non-finite"
    if terms is None:
        denom = max(float(ref.norm()), NORMWISE_FLOOR * (scale if scale is not None else 0.0), 1e-300)
        err = float((got.double() - ref).norm()) / denom
        # A7 normwise form of two routes rounding independently: 2 depth u, the decay / phase
        # amplification adding to the depth (a chunked route's exp(cumsum log a) differences amplify the cumsum's rounding).
        # 16-bit: tol16 for values; for a GRADIENT the derived 0.5 eps sqrt(depth) WITHOUT the upper band clamp (A7's band
        # top, 6.4 eps = 5 %, is a policy for forward values; the bf16 backward of a 37-token chain measures 3-8 % and a
        # clamped bound would call that a defect -- the fp32 cells hold the same gradients at 2 depth u32)
        if is_sixteen_bit(rule.dtype):
            bound = (tol16(rule.dtype, depth) if kind == "y" else
                     max(0.5 * torch.finfo(rule.dtype).eps * math.sqrt(rule.arms * depth), tol16(rule.dtype, 1)))
        else:
            bound = 2 * (depth + rule.amp) * unit_roundoff(rule.dtype)
        if pool is not None and pool.capture is not None:          # an escalation draw: the bias and error data only
            if is_sixteen_bit(rule.dtype) and got.numel() >= 256:
                pool.capture[key] = (got, ref, bound)
            if kind == "g" and is_sixteen_bit(rule.dtype):
                pool.cap_errs[key] = (err, bound)
            return
        if kind == "g" and is_sixteen_bit(rule.dtype) and pool is not None:
            pool.err(key, err, bound)          # judged over the draws (Pool.assert_unbiased)
        else:
            assert err <= bound, f"{name}: normwise relative error {err:.3e} > bound {bound:.3e} (depth {depth}, amp {rule.amp:.3g})"
        if pool is not None and is_sixteen_bit(rule.dtype) and got.numel() >= 256:
            pool.add(key, bound, *_signed_bias(got, ref))
        return
    if is_sixteen_bit(rule.dtype):
        # A7 16-bit policy: normwise tol16, guarded (a unit whose reference is ~0 against its own summands is scored in the
        # terms form instead).  The signed bias is pooled over the DRAWS only: one tensor's rounding errors are correlated
        # (a state's elements share the decay factors), so its per-draw bias scatters by more than bound / 4
        bound = tol16(rule.dtype, depth)
        if pool is not None and pool.capture is not None:          # an escalation draw: the bias data only
            if got.numel() >= 256:
                pool.capture[key] = (got, ref, bound)
            return
        # The relative form `err <= u sqrt(depth) ||ref||` follows from the terms form `err <= c u ||terms||` only when
        # ||ref|| >= (c / sqrt(depth)) ||terms|| (no cancellation between the summands); a unit below that (a carry that
        # cancels its summands: an accumulated angle) is scored in the terms form.  The guard of `assert_rel_close` is
        # ||ref|| < k u ||terms||, so k = c / (sqrt(depth) u).  The mixed-precision mode, whose c (rounded-operand count plus
        # the fp32 sums' share) is stated by `Rule.c`, uses it; the bf16 cells keep the default guard k = 1.
        k = max(1.0, rule.c(fdepth, True) / (math.sqrt(max(depth, 1)) * unit_roundoff(rule.dtype))) if rule.mixed else 1.0
        try:
            assert_rel_close(got, ref, bound, terms=terms, u=rule.dtype, k=k, name=key)
        except NearZeroReference:
            assert_close_terms(got, ref, terms, rule.dtype, rule.c(fdepth, True), bias=False, name=f"{key} [terms: ref ~ 0]")
        if pool is not None and got.numel() >= 256:
            pool.add(key, bound, *_signed_bias(got, ref))
    else:
        assert_close_terms(got, ref, terms, rule.dtype, rule.c(fdepth), name=name)


def reference(tl, l64, fn, *args, grad_outputs):
    """The fp64 reference run of `fn(*args)` with the module `l64`'s parameters tracked: `terms_abs` (absolute arithmetic)
    or, for a layer with `use_terms=False`, plain fp64 autograd with every terms field None."""
    if tl.use_terms:
        return terms_abs(fn, *args, grad_outputs=grad_outputs, params=l64)
    names = [n for n, p in l64.named_parameters(remove_duplicate=False) if p.requires_grad]
    params = dict(l64.named_parameters(remove_duplicate=False))
    leaves = [a for a in args if torch.is_tensor(a) and a.requires_grad]
    with torch.enable_grad():
        out = fn(*args)
        outs = [o for o in pytree.tree_leaves(out) if torch.is_tensor(o)]
        gos = [g for g in pytree.tree_leaves(grad_outputs) if g is not None]
        gs = torch.autograd.grad(outs, leaves + [params[n] for n in names], gos, allow_unused=True)
    none = pytree.tree_map(lambda _: None, out)
    return TermsResult(pytree.tree_map(lambda o: o.detach(), out), none, list(gs[:len(leaves)]), [None] * len(leaves),
                       dict(zip(names, gs[len(leaves):])), {n: None for n in names})


def _on_circle(tl, key, got, ref):
    """`got` moved by a multiple of 2 pi onto the branch of `ref` for a phase carry (`TorchLayer.circular`), else unchanged."""
    if key not in tl.circular:
        return got
    d = got.double() - ref
    return (ref + d - 2 * math.pi * torch.round(d / (2 * math.pi))).to(got.dtype)


def _randn(like, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(like.shape, generator=g, dtype=torch.float64) * scale).to(like.dtype)


def make_input(tl, dtype, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(tl.B, tl.L, tl.d_in, generator=g, dtype=torch.float64).to(dtype)


def _fp64_copy(layer):
    ref = copy.deepcopy(layer).double()
    for p in ref.parameters():
        p.requires_grad_(p.requires_grad)
    return ref


def arbitrary_state(tl, seed):
    """A complete, NONZERO State in the layer's default carry dtypes (an arbitrary feasible
    initial State), every floating carry a leaf unless the family fixes it to a constant (`TorchLayer.carry_fix`)."""
    layer = tl.layer
    st = layer.init_state(tl.B)
    out = {}
    for i, (k, v) in enumerate(st.items()):
        if v.is_floating_point():
            fixed = tl.carry_fix(layer, k, v, seed + 17 * i) if tl.carry_fix else None
            v = fixed if fixed is not None else _randn(v, seed + 17 * i, 0.5).requires_grad_()
        out[k] = v
    return type(st)(out)


def _step_reference(layer64, keys, length):
    """fn(x, *carries) -> (y, final carries): the fp64 SINGLE-TOKEN recurrence of the layer from the given State (the
    default State when no carries are passed), for `terms_abs`.  The sequential `step` is the one formulation of a
    layer that runs in absolute arithmetic everywhere (the chunked routes write into preallocated buffers); the
    parallel routes are scored against it, the step route itself against the independent oracle."""
    from orbit_ssm.layers.state import State

    def fn(x, *vals):
        st = State(dict(zip(keys, vals))) if vals else layer64.init_state(x.shape[0])
        ys = []
        for t in range(length):
            yt, st = layer64.step(x[:, t], st)
            ys.append(yt)
        return torch.stack(ys, 1), tuple(st[k] for k in keys)
    return fn


def _amp(tl, layer64, x64):
    if tl.oracle is None:
        return 0.0
    stats = {}
    with torch.no_grad():
        tl.oracle(layer64, x64, stats)
    return float(stats.get("amp", 0.0))


def _names(layer):
    return [n for n, p in layer.named_parameters(remove_duplicate=False) if p.requires_grad]


def _score_grads(prefix, got_grads, leaf_names, res, rule, pool, extra=None):
    """Score the gradients `got_grads` (order: `leaf_names`) against `res` (a `reference` result: res.grads for the tracked
    args in order, res.param_grads by parameter name)."""
    ref = dict(zip(extra or (), res.grads or ()))
    ref_t = dict(zip(extra or (), res.grad_terms or ()))
    ref.update(res.param_grads or {})
    ref_t.update(res.param_grad_terms or {})
    scale = max([float(v.norm()) for v in ref.values() if v is not None] or [0.0])
    for name, g in zip(leaf_names, got_grads):
        score(f"{prefix}grad {name}", g, ref.get(name), ref_t.get(name), rule, "g", pool, scale=scale)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _pooled(pool, draw, dtype, base):
    """Run ``draw(seed, pool)`` over the cell's draws, then judge the pooled bias with ESCALATION: a tensor whose mean
    exceeds bound/4 is redrawn on the continuation of the seed sequence (30 draws in total, HYBRID_SSM_BIAS_DRAWS) and
    judged on the full set (tests/_tolerance.py `DrawPool`).  The extra draws contribute their signed biases and, for a
    normwise-scored gradient, their error / bound (`Pool.assert_unbiased`, `error_tail_fails`).
    Returns False when a draw reports that the cell does not apply."""
    for seed in draw_seeds(dtype, base):
        if draw(seed, pool) is False:
            return False

    cache = {}

    def redraw(seed):
        if seed not in cache:
            cap = Pool()
            cap.capture = {}
            draw(seed, cap)
            cache[seed] = cap
        return cache[seed]
    pool.assert_unbiased(lambda seed: redraw(seed).capture, escalation_seeds(dtype, base),
                         errs_of=lambda seed: redraw(seed).cap_errs)
    return True


def check_oracle(ad, cell, dtype_name, base=0):
    """Forward + every gradient of the layer against the family's independent fp64 oracle."""
    dtype = DTYPES[dtype_name]
    pool = Pool()

    def draw(seed, pool):
        tl = ad.torch_layer(cell.c, dtype, seed)
        if tl.oracle is None:
            return False
        layer = tl.layer
        l64 = _fp64_copy(layer)                 # before any forward: a used module holds non-leaf caches
        x = make_input(tl, dtype, seed).requires_grad_()
        with autocast(dtype_name):
            y = layer(x)
        w = _randn(y, seed + 5)
        names = _names(layer)
        params = dict(layer.named_parameters(remove_duplicate=False))
        grads = torch.autograd.grad((y.double() * w.double()).sum(), [x] + [params[n] for n in names], allow_unused=True)
        stats = {}
        res = reference(tl, l64, lambda xx: tl.oracle(l64, xx, stats), x.detach().double().requires_grad_(),
                        grad_outputs=w.double())
        rule = Rule(tl, dtype, float(stats.get("amp", 0.0)), mixed=is_amp(dtype_name))
        tag = f"{cell.id} [{dtype_name} seed {seed}] "
        assert is_amp(dtype_name) or y.dtype == dtype, f"{tag}output dtype {y.dtype}"
        score(tag + "y", y, res.out, res.terms, rule, "y", pool)
        _score_grads(tag, grads, ["x"] + names, res, rule, pool, extra=["x"])
    return _pooled(pool, draw, rule_dtype(dtype_name), 7001 + base)


def _cut(tl):
    return tl.L // 2 + 1                        # never on a chunk boundary of the test geometries


def _assert_dtypes(ad, cell, tag, supplied, returned, route="forward"):
    """Every carry returned has the dtype of the carry supplied.  An adapter declares a DIAGNOSED deviation with its
    reason (`torch_state_dtype_exceptions(cfg, route) -> {carry: reason}`, route "forward" or "step"); a declared deviation must still occur (a stale
    declaration is a red: delete it when the deviation is fixed), an undeclared one is a red."""
    bad = {k for k in supplied if returned[k].dtype != supplied[k].dtype}
    declared = set(getattr(ad, "torch_state_dtype_exceptions", lambda c, r: {})(cell.c, route))
    assert not (bad - declared), (f"{tag}carries returned in another dtype than supplied: "
                                  f"{ {k: (str(supplied[k].dtype), str(returned[k].dtype)) for k in sorted(bad - declared)} }")
    assert not (declared - bad), f"{tag}declared dtype deviation(s) {sorted(declared - bad)} no longer occur: remove the declaration"


#: a carry the call has updated from a non-bf16 value is bf16-representable with probability 2^-16 per element (the low 16
#: mantissa bits of an fp32 number are zero); the check admits 1/10 of the nonzero elements
REPRESENTABLE_MAX = 0.1


def _bf16_representable_fraction(t):
    """Fraction of the nonzero elements of the fp32 tensor `t` whose low 16 mantissa bits are zero (exactly bf16 values)."""
    bits = t.detach().float().contiguous().view(torch.int32)
    nz = t.detach() != 0
    if int(nz.sum()) == 0:
        return 0.0
    return float((((bits & 0xFFFF) == 0) & nz).sum()) / float(nz.sum())


def _assert_carries_not_rounded(tag, l64, tl, st0, st_end, x64, n):
    """Mixed precision keeps the carries in fp32 BETWEEN tokens: a recurrent carry (its fp64 update over the call's tokens depends on its
    own previous value) that a call has advanced from the arbitrary fp32 State must hold values that are not bf16 numbers.
    A carry narrowed to bf16 anywhere on the path (per token, in the scan, or at return) is bf16-representable in every
    element and is still widened back to fp32 for the dtype check.  Carries that are copies of the activations (conv window,
    mu, write_prev) do not depend on their previous value after the call's tokens and are exempt by the dependence test; so are
    carries whose fp64 reference is itself representable (a constant frame, zeros)."""
    from orbit_ssm.layers.state import State
    keys = list(st0)
    vals = {k: st0[k].detach().double() for k in keys}

    def run(v):
        st = State(dict(v))
        for t in range(n):
            _, st = l64.step(x64[:, t], st)
        return st
    with torch.no_grad():
        base = run(vals)
        for k in keys:
            if not st0[k].is_floating_point():
                continue
            pert = dict(vals)
            pert[k] = vals[k] * 1.01 + 0.01
            moved = run(pert)[k]
            if float((moved - base[k]).abs().max()) == 0.0:
                continue                                     # does not depend on its previous value: an activation copy
            ref = base[k]
            if _bf16_representable_fraction(ref.float()) > REPRESENTABLE_MAX:
                continue                                     # the fp64 reference itself is made of bf16 numbers
            frac = _bf16_representable_fraction(st_end[k].float())
            assert frac <= REPRESENTABLE_MAX, (f"{tag}carry {k!r}: {frac:.2f} of its nonzero elements are exact bf16 numbers "
                                               f"after the call (expected ~2^-16 for an fp32 carry): the carry was rounded to bf16")


def check_continuation(ad, cell, dtype_name):
    """Two calls joined by the returned State == one call, from an arbitrary complete State."""
    dtype = DTYPES[dtype_name]
    pool = Pool()

    def draw(seed, pool):
        tl = ad.torch_layer(cell.c, dtype, seed)
        layer = tl.layer
        l64 = _fp64_copy(layer)
        x = make_input(tl, dtype, seed).requires_grad_()
        st0 = arbitrary_state(tl, seed)
        keys = list(st0)
        cut = _cut(tl)
        with autocast(dtype_name):
            y1, s1 = layer(x[:, :cut], state=st0, return_state=True)
            y2, s2 = layer(x[:, cut:], state=s1, return_state=True)
        yj = torch.cat([y1, y2], 1)
        tag = f"{cell.id} [{dtype_name} seed {seed}] continuation "
        # the State contract: same keys, a supplied carry keeps its dtype through the call
        assert set(s2) == set(keys), f"{tag}State keys {sorted(s2)} != {sorted(keys)}"
        _assert_dtypes(ad, cell, tag, st0, s1)
        _assert_dtypes(ad, cell, tag, st0, s2)
        w = _randn(yj, seed + 5)
        wc = {k: _randn(s2[k], seed + 31 + i) for i, k in enumerate(keys)}
        loss = (yj.double() * w.double()).sum() + sum((s2[k].double() * wc[k].double()).sum() for k in keys)
        names = _names(layer)
        params = dict(layer.named_parameters(remove_duplicate=False))
        lk = [k for k in keys if st0[k].requires_grad]
        leaves = [x] + [st0[k] for k in lk] + [params[n] for n in names]
        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        x64 = x.detach().double().requires_grad_()
        vals64 = [st0[k].detach().double().requires_grad_(st0[k].requires_grad) for k in keys]
        res = reference(tl, l64, _step_reference(l64, keys, tl.L), x64, *vals64,
                        grad_outputs=(w.double(), tuple(wc[k].double() for k in keys)))
        rule = Rule(tl, dtype, _amp(tl, l64, x64.detach()), extra_depth=tl.L, mixed=is_amp(dtype_name))
        if is_amp(dtype_name):
            _assert_carries_not_rounded(tag, l64, tl, st0, s1, x64.detach(), cut)
        score(tag + "y", yj, res.out[0], res.terms[0], rule, "y", pool)
        for k, v, r, t in zip(keys, [s2[k] for k in keys], res.out[1], res.terms[1]):
            score(tag + f"final {k}", _on_circle(tl, k, v, r), r, t, rule, "y", pool)
        _score_grads(tag, grads, ["x"] + [f"state.{k}" for k in lk] + names, res, rule, pool,
                     extra=["x"] + [f"state.{k}" for k in lk])
    _pooled(pool, draw, rule_dtype(dtype_name), 7101)


def check_step(ad, cell, dtype_name):
    """The single-token step loop against the fp64 references.

    default State: output and gradients (input, parameters) against the INDEPENDENT oracle (the loop from `init_state()`
    is the layer on a zero State); the final State against the fp64 loop.  arbitrary State: output, final State and
    every gradient, the gradients into the State included, against the fp64 loop (state handling and dtype contract)."""
    dtype = DTYPES[dtype_name]
    pool = Pool()

    def draw(seed, pool):
        tl = ad.torch_layer(cell.c, dtype, seed)
        layer = tl.layer
        l64 = _fp64_copy(layer)
        names = _names(layer)
        params = dict(layer.named_parameters(remove_duplicate=False))
        for arm in ("default", "given"):
            x = make_input(tl, dtype, seed).requires_grad_()
            st0 = layer.init_state(tl.B) if arm == "default" else arbitrary_state(tl, seed)
            keys = list(st0)
            st, ys = st0, []
            with autocast(dtype_name):
                for t in range(tl.L):
                    yt, st = layer.step(x[:, t], st)
                    ys.append(yt)
            ys = torch.stack(ys, 1)
            tag = f"{cell.id} [{dtype_name} seed {seed}] step({arm}) "
            assert is_amp(dtype_name) or ys.dtype == dtype, f"{tag}output dtype {ys.dtype}"
            _assert_dtypes(ad, cell, tag, st0, st, route="step")
            w = _randn(ys, seed + 5)
            wc = {k: _randn(st[k], seed + 31 + i) for i, k in enumerate(keys)}
            x64 = x.detach().double().requires_grad_()
            vals64 = [st0[k].detach().double().requires_grad_(st0[k].requires_grad) for k in keys] if arm == "given" else []
            leaf_state = [k for k in keys if st0[k].requires_grad] if arm == "given" else []
            res = reference(tl, l64, _step_reference(l64, keys, tl.L), x64, *vals64,
                            grad_outputs=(w.double(), tuple(wc[k].double() for k in keys)))
            rule = Rule(tl, dtype, _amp(tl, l64, x64.detach()), extra_depth=tl.L, mixed=is_amp(dtype_name))
            if is_amp(dtype_name) and arm == "given":
                _assert_carries_not_rounded(tag, l64, tl, st0, st, x64.detach(), tl.L)
            score(tag + "y", ys, res.out[0], res.terms[0], rule, "y", pool)
            for k, r, t in zip(keys, res.out[1], res.terms[1]):
                score(tag + f"final {k}", _on_circle(tl, k, st[k], r), r, t, rule, "y", pool)
            gy = None
            if arm == "default" and tl.oracle is not None:
                gy = torch.autograd.grad((ys.double() * w.double()).sum(), [x] + [params[n] for n in names],
                                         retain_graph=True, allow_unused=True)
            loss = (ys.double() * w.double()).sum() + sum((st[k].double() * wc[k].double()).sum() for k in keys)
            grads = torch.autograd.grad(loss, [x] + [st0[k] for k in leaf_state] + [params[n] for n in names],
                                        allow_unused=True)
            _score_grads(tag, grads, ["x"] + [f"state.{k}" for k in leaf_state] + names, res, rule, pool,
                         extra=["x"] + [f"state.{k}" for k in leaf_state])
            if gy is not None:
                # the loop from the default State against the independent oracle: y and the gradients of y
                ro = reference(tl, l64, lambda xx: tl.oracle(l64, xx, {}), x64, grad_outputs=w.double())
                score(tag + "y vs oracle", ys, ro.out, ro.terms, rule, "y", pool)
                _score_grads(tag + "oracle ", gy, ["x"] + names, ro, rule, pool, extra=["x"])
    _pooled(pool, draw, rule_dtype(dtype_name), 7201)


def check_default_carries(ad, cell, dtype_name):
    """`init_state()` allocates every carry in promote(parameter dtype, fp32) (one rule for every family)."""
    dtype = DTYPES[dtype_name]
    tl = ad.torch_layer(cell.c, dtype, 7301)
    want = carry_dtype(dtype)
    with autocast(dtype_name):
        st = tl.layer.init_state(tl.B)
    bad = {k: v.dtype for k, v in st.items() if v.is_floating_point() and v.dtype != want}
    assert not bad, f"{cell.id} [{dtype_name}]: default carries {bad}, expected {want}"
