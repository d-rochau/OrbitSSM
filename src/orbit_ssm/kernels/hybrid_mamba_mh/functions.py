"""autograd entry point for the HybridMamba-MH kernel (structure B).

The tensor arguments are passed as one ordered tuple rather than as named
kwargs, because `torch.autograd.Function.backward` must return exactly one
gradient per positional input; TENSOR_ARGS below is the single place that order
is defined, and `backward` walks the same list.

Both halves are transposed kernels. Pairing the backward with the forward's
checkpoints is `mh_scan(ckpt=True)`, the default. `ckpt_l` deletes the
backward's pass 1, at the price of the forward writing `hck`: the saving is
worth it wherever the buffer is moderate, and loses when `hck` is huge (the
forward's store then dominates). The `hck` size is per LAYER and must survive
to the backward, so that size, not the time ratio, is what decides this.
Compare tuned against tuned: `config=` disables the autotuner, so a
checkpointed arm launched through `config=` runs the offline heuristic.

Coarser checkpoints are still not a way out: pass 1 walks every element to build
any state at all, so its cost is set by L and not by how often it stores.

This matters for the comparison against structure A, which ALREADY checkpoints
-- `m1_fwd.cu:1169` writes h_carry, `m1_bwd.cu:1264` takes it as a parameter --
so the default path here has B doing strictly more work than the baseline it is
benchmarked against, while using less memory.

Three transforms happen on the host before the kernel sees anything, all for one
reason: the (H,B) grid recomputes anything (b,n,t)-shaped once per head, 64
times over at the reference shape. ONE of them lands on B/C and so rides in on
tensors the kernel already reads: the GATES, when their static bias is
head-shared. That is 2.6-3.0x on the gated forward -- one identical sigmoid per
head. `_prefold_gate_ok` guards it and falls back to the in-kernel fold, which
stays the only correct route when the tensor really is per-head: a G=H gate
bias, where the folded tensor would be (B,H,N,L), the streaming shape contract
forbids.  (The rotation has no host pre-fold: under the one rotation
law the angle is head-DEPENDENT and never leaves the scan kernel.)

The other two are decay drivers and cannot ride on B/C. The A_DYN activation
multiplies the decay, and `Delta_eff[b,h,t] * rate[b,n,t]` materialised is that
same forbidden shape, so it stays its own tensor and only the transcendental is
hoisted. The MU telescoping hoists whole, because the increment is the same
shape as `mu`. Neither needs a guard: both are functions of their own tensor
alone, so both are head-redundant
unconditionally.

All four are plain torch, so autograd supplies the chain rule back to
B_t/C_t/theta/g_B/g_C/a_dyn/mu/mu_init and no adjoint is written by hand.

AMP: no cast at the boundary. The kernel receives whatever dtype the model
produced (bf16 in practice) and upcasts per tile inside, so an autocast wrapper
that pre-cast the inputs would only add a full-size copy of B/C/x. `custom_fwd`
without `cast_inputs` gives that; the bodies additionally disable autocast
explicitly, since `cast_inputs=None` does NOT disable it inside the body.
"""
from __future__ import annotations

import functools
import os
import weakref

import torch

from .bwd_gluon import auto_block_l, mh_backward
from .. import portable as _portable
from .common_gluon import selector_window
from .fwd_gluon import mh_forward
from .layout import DYN_RAW, DYN_SOFTPLUS, GATE_RAW, GATE_SIGMOID, assert_pair_tied
from .layout import GATE_EXP, GATE_ONE_PLUS, GATE_SOFTPLUS, GATE_TANH

#: gate activations `common_gluon._gate` / `_gate_dv` implement
_KERNEL_GATE_ACTS = frozenset((GATE_RAW, GATE_SIGMOID, GATE_EXP, GATE_TANH, GATE_SOFTPLUS, GATE_ONE_PLUS))
from ...compile_compat import dynamo_opaque
from ...layers.proj import autocast_io
from ... import kernel_options as _ko
from .layout import dyn_rate as layout_dyn_rate
from .layout import gate_act as layout_gate_act

# There is no head-free `R(S)` host pre-fold: the SSD law's phase is
# head-dependent by construction (the increment is `Delta_t[h] * theta_t[k]`,
# the read offset is `theta_h[h,k]`).  The half_fold schedule below implements
# the full head-dependent law in Triton, with one subsequent real scan over
# all coordinates.


class _RefoldedMaps:
    """The folded (H,B,N,L) fp32 maps, rebuilt once on first unpack."""

    def __init__(self, fold_args, fold_kw):
        self.args, self.kw = fold_args, fold_kw
        self.versions = [None if t is None else t._version
                         for t in (*fold_args, *fold_kw.values()) if torch.is_tensor(t)]
        self.maps = None
        self.pending = {}

    def get(self, i):
        now = [t._version for t in (*self.args, *self.kw.values()) if torch.is_tensor(t)]
        if now != self.versions:
            raise RuntimeError("a folded-rotation input was modified in place "
                               "between forward and backward")
        value = None if self.maps is None else self.maps[i]()
        if value is None:
            from .half_fold import half_fold
            with torch.no_grad():
                b, c, _ = half_fold(*(t.detach() for t in self.args),
                                    **{k: v.detach() if torch.is_tensor(v) else v
                                       for k, v in self.kw.items()})
            self.maps = (weakref.ref(b), weakref.ref(c))
            # Keep the other map until its first unpack, then let the scan's
            # local saved-tensor tuple own both. A strong cache here retains
            # two H*B*N*L fp32 maps during the fold adjoint and after backward.
            self.pending = {0: b, 1: c}
            value = self.pending[i]
        self.pending.pop(i, None)
        return value


def _refold_saved_maps(B_fold, C_fold, fold_args, fold_kw):
    """Saved-tensor hooks for the folded route's scan: its two (H,B,N,L)
    fp32 maps are NOT kept for backward (2.4-2.5x the in-scan training
    memory at the largest cell); backward re-runs the
    deterministic fold kernel instead, so the rebuilt maps are bitwise the
    forward's.  Every other saved tensor is stored as usual."""
    if not torch.is_grad_enabled():
        import contextlib
        return contextlib.nullcontext()
    holder = _RefoldedMaps(fold_args, fold_kw)
    ids = {id(B_fold): 0, id(C_fold): 1}

    def pack(t):
        i = ids.get(id(t))
        return (holder, i) if i is not None else t

    def unpack(p):
        return p[0].get(p[1]) if isinstance(p, tuple) else p

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


#: The rotation route (`hybrid_mamba` kernel option `rotation`, env
#: HYBRID_MH_ROTATION): "fold" -- the head-dependent phase
#: folded into per-head maps ahead of ONE real scan, with mixed launches split
#: into the rotating window (folded) and the real rows (plain real scan) -- is
#: the default route.  "in_scan" (rotation inside the scan) is an explicit
#: alternative and the fallback for envelopes the fold does not serve;
#: every fallback is counted in `ROTATION_FALLBACKS` with its reason.
ROTATION_FALLBACKS: dict = {}
#: The route that served the LAST top-level `mh_scan` call (host-only
#: telemetry): "real" (no rotation), "fold",
#: "in_scan", "in_scan(<fallback reason>)", or "split+<rotating route>" when
#: the mixed-launch split served it.  Tests compare it with the route a
#: configuration should take, so a silent fallback cannot pass.
LAST_ROUTE: str | None = None
#: tests switch the mixed-launch split off to compare against the whole-N fold
_ROT_SPLIT = True
_IN_SPLIT = __import__("contextvars").ContextVar("mh_in_rot_split", default=False)


def rotation_route() -> str:
    from .autotune import _opt
    v = (_opt("ROTATION") or "fold").lower()
    if v not in ("fold", "in_scan"):
        raise ValueError("hybrid_mamba rotation must be 'fold' or 'in_scan', got %r" % v)
    return v


#: Narrowest head width P the fold serves.  Its per-head (H,B,N,L) maps grow
#: with H = D / P: at P = 1 the fold costs far more time and memory than
#: in-scan, while from P = 16 fold+split wins.  Env MH_FOLD_MIN_P overrides.
FOLD_MIN_P = 16


def _fold_refusal(theta, n_off, n_rows, P=None, mem=None):
    """Why the fold cannot serve this rotating launch, or None."""
    if theta.dim() != 3:
        return "per-head theta (B,H,K,L)"
    if n_off or n_rows is not None:
        return "row-window launch (n_off/n_rows)"
    if P is not None and P < int(os.environ.get("MH_FOLD_MIN_P", FOLD_MIN_P)):
        return "narrow heads (P < fold minimum)"
    if mem and mem.get("AGX") == 0 and mem.get("BCS") == 0:
        # the contract determinism pin: the fold backward reduces with
        # atomics, so a bit-reproducible backward must stay in-scan
        return "determinism pin (mem AGX=0, BCS=0)"
    return None


#: Below this head width rotating launches are supported but slow (warn, do
#: not refuse): in-scan+split costs about +50 % over real, until the
#: packed-head schedules serve rotation.
SLOW_ROTATION_P = 16
_WARNED_NARROW = False


def _warn_narrow_rotation(P):
    global _WARNED_NARROW
    if P < SLOW_ROTATION_P and not _WARNED_NARROW:
        _WARNED_NARROW = True
        import warnings
        warnings.warn(
            "hybrid_mamba: rotating (complex) rows with head width P=%d < %d "
            "run the unpacked in-scan schedule, roughly 1.5x the cost of the "
            "same launch without rotation; prefer P >= %d for complex rows."
            % (P, SLOW_ROTATION_P, SLOW_ROTATION_P), RuntimeWarning, stacklevel=3)


def _note_route(r):
    global LAST_ROUTE
    LAST_ROUTE = r


def _rot_window(theta, row_rot, n_off, n_rows, write_prev, mu_init, return_state):
    """(start, stop, N) of the rotating rows when they form ONE leading or
    trailing window of a mixed launch, else None (either rotation route).

    Mixed configs (e.g. C15: 25 % rotating rows) otherwise run every row
    through the rotation: the fold expands the real rows' shared B/C into
    per-head maps (2.5x the in-scan training peak at C15), in-scan computes
    the rotation and discards it on real rows.  The trapezoid weight
    (B,H,L) is row-free, so endpoint launches split too; a carried per-row
    write_prev, mu carries and state-returning calls (decode / chunked
    streaming: the whole-N route reports a terminal phase for EVERY pair,
    which the rotating window alone cannot) keep the whole-N fold."""
    if (not _ROT_SPLIT or _IN_SPLIT.get() or theta is None or row_rot is None
            or n_off or n_rows is not None or write_prev is not None
            or mu_init is not None or return_state):
        return None
    # host read of the selector, memoised per (tensor, version): a fresh
    # `.cpu()` here would be a CPU->CUDA copy that fails CUDA-graph capture;
    # the warm-up call fills the memo, the captured call only hits it
    rot = [bool(v) for v in selector_window(row_rot, 0, row_rot.numel())]
    n = len(rot)
    k = sum(rot)
    if k in (0, n) or k % 2:
        return None
    lead = all(rot[:k])
    if not lead and not all(rot[n - k:]):
        return None
    return (0, k, n) if lead else (n - k, n, n)


def _split_rot_scan(window, kw):
    """Rotating rows on their rotation route (sliced), the rest on the plain
    real scan through its row window; y is a sum over rows, so the two
    outputs add.  D skip is applied once (rotating call).  The z gate is
    linear in y but a row window refuses it (`mh_scan(z=...)` is the
    whole-N call), so it is applied once to the sum, in fp32 as the kernel
    epilogue does: Y * silu(z)."""
    r0, r1, n = window
    k0, k1 = r0 // 2, r1 // 2
    a_n, mu_n = kw["a_dyn_res"] == "n", kw["mu_res"] == "n"

    def rows(name, t):
        if t is None:
            return None
        if name in ("B_t", "C_t", "g_B", "g_C") or (name == "a_dyn" and a_n) or (name == "mu" and mu_n):
            return t[:, r0:r1]
        if name in ("A_log", "mu_gain", "h_init"):
            return t[..., r0:r1]
        if name in ("beta_a", "beta_b", "beta_c", "beta_gB", "beta_gC"):
            return t[:, r0:r1]
        if name in ("row_law", "row_rot"):
            return t[r0:r1]
        if name == "theta":
            return t[:, k0:k1] if t.dim() == 3 else t[:, :, k0:k1]
        if name in ("theta_h", "theta_init"):
            return t[..., k0:k1]
        return t
    skip = ("kernel_options", "validate", "return_state", "n_off", "n_rows", "z")
    rot_kw = {name: rows(name, v) for name, v in kw.items()
              if name not in ("x", "delta") + skip}
    real_kw = {name: v for name, v in kw.items() if name not in ("x", "delta") + skip}
    real_kw.update(theta=None, row_rot=None, theta_h=None, theta_init=None,
                   theta_delta=False, d_skip=None,
                   n_off=r1 if r0 == 0 else 0, n_rows=n - (r1 - r0))
    common = dict(validate=kw["validate"], kernel_options=kw["kernel_options"])
    token = _IN_SPLIT.set(True)
    try:
        y_rot = mh_scan(kw["x"], kw["delta"], **rot_kw, **common)
        rot_route = LAST_ROUTE
        y = y_rot + mh_scan(kw["x"], kw["delta"], **real_kw, **common)
    finally:
        _IN_SPLIT.reset(token)
    _note_route("split+" + str(rot_route))
    z = kw["z"]
    if z is None:
        return y
    zf = z.float()
    return (y.float() * (zf * torch.sigmoid(zf))).to(y.dtype)


#: Name of the environment switch for the ABI validation pass. Anything but
#: "0"/""/"false"/"no" turns it on process-wide.
VALIDATE_ENV = "HYBRID_SSM_MH_VALIDATE"


def _validate_default() -> bool:
    """Is the pair-tie check on for callers that did not say?

    Read per call, not at import: a test that wants the check off for one
    case sets the variable and gets it, and the suite-wide default in
    `tests/conftest.py` does not have to win an import race.

    OFF in production. The check compares whole tensors and asks `.any()`,
    which is a device->host sync on every `mh_scan` -- affordable in a test,
    not in a training step.
    """
    return os.environ.get(VALIDATE_ENV, "0").lower() \
        not in ("", "0", "false", "no")


# order is the ABI of this Function: forward's positional tail and backward's
# return tuple are both generated from it
TENSOR_ARGS = (
    "x", "delta", "A_log", "B_t", "C_t",
    "delta_w", "a_dyn", "mu", "mu_gain",
    "g_B", "g_C", "theta",
    "beta_a", "beta_b", "beta_c", "beta_gB", "beta_gC",
    "delta_bias", "h_init", "theta_init", "mu_init", "d_skip",
    "row_law", "row_rot",
    # H3: appended, not inserted -- this tuple IS the ABI and the two
    # non-differentiable selectors have to keep the positions the
    # `NON_DIFF` set names.
    "z", "endpoint_own",
    # SSD rotation law: static READ-side angle offset (H,K).
    # Appended for the same ABI reason.
    "theta_h",
)
NON_DIFF = frozenset({"row_law", "row_rot"})

TWO_PI = 6.283185307179586


def _shared(beta):
    """A static bias is head-shared iff it declares G=1."""
    return beta is None or beta.shape[0] == 1


def _prefold_gate_ok(theta, beta_b, beta_c,
                     g_B=None, g_C=None, beta_gB=None, beta_gC=None):
    """Can the B/C GATE be hoisted out of the head loop?

    One feature is left on this route: the gates, which are head-redundant
    when their static bias is head-shared.  A gated non-rotating problem
    pays 2.5-3.1x on the forward in-kernel, so the hoist is
    worth its two elementwise passes.

    Every condition is a `None` or a SHAPE test, never a value test: a
    value test would force a host sync on every training step.

    ROTATION disqualifies it, and that is a CORRECTNESS condition, not an
    optimisation: the order is bias -> rotate -> gate, so a gate
    folded onto B/C here would land BEFORE the rotation the kernel still
    applies.  Under the one rotation law (the SSD law) the angle is
    head-dependent and always stays in the kernel, so a rotating launch
    never takes this route.
    """
    if theta is not None:
        return False
    if not (_shared(beta_b) and _shared(beta_c)):
        return False
    return (g_B is not None or g_C is not None) \
        and _shared(beta_gB) and _shared(beta_gC)


# Which rotating launches take the pair-map routes: "all" (default: gated ones
# the per-row pair maps, ungated ones the raw maps, backward PRAW), "gated"
# or "off" (both on the in-kernel rotation).  A module attribute rather than
# an environment switch: it exists for tests.
PMAP_ROUTE = "all"


def _recomputed(fn, *args):
    """`fn(*args)` whose autograd graph keeps only `args`.

    The B/C folds (`_prefold_gate`, `_pair_maps`) are a few elementwise ops
    on (B, N, L) maps, but their autograd graph holds the fp32 intermediates
    -- the widened map and act(g) -- for the backward, several times the
    size of the input-dtype maps themselves.  Recomputing them in the
    backward is cheap and the same values come back (deterministic elementwise ops on the same inputs)."""
    if torch.is_grad_enabled() and any(
            torch.is_tensor(a) and a.requires_grad for a in args):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(fn, *args, use_reentrant=False,
                          preserve_rng_state=False)
    return fn(*args)


def _pmap_ok(theta, x, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC):
    """Which pair-map route a ROTATING launch takes: "gated", "raw" or None.

    It needs head-shared static biases (the maps are head-free) and P > 1
    (P == 1 is the backward's packed schedule, which keeps the in-kernel
    route).  "gated": act(g + beta_g) and the B/C bias folded into (B, 2N, L)
    per-row maps; "raw" (ungated, `PMAP_ROUTE == "all"`): the biased map
    itself, (B, N, L), each pair loaded once and its two rows' adjoints
    summed.  Shape tests only.
    """
    if PMAP_ROUTE == "off" or theta is None or x.shape[2] <= 1 \
            or not all(_shared(t) for t in (beta_b, beta_c, beta_gB, beta_gC)):
        return None
    if g_B is not None or g_C is not None:
        return "gated"
    return "raw" if PMAP_ROUTE == "all" else None


def _pair_maps_raw(T, beta):
    """The ungated route's maps: T with a head-shared bias folded in."""
    if beta is None:
        return T
    return (T.float() + beta[0].reshape(1, T.shape[1], 1)).to(T.dtype)


def _pair_maps(T, beta, g, beta_g, gate_act):
    """(B, N, L) map -> (B, 2N, L) pair maps: rows 2n, 2n+1 of the result are
    act(g_n + beta_g) * (T_2k + beta, T_2k+1 + beta) for n in {2k, 2k+1}."""
    Bs, N, L = T.shape
    v = T.float()
    if beta is not None:
        v = v + beta[0].reshape(1, N, 1)
    pair = v.reshape(Bs, N // 2, 1, 2, L)
    if g is None:
        out = pair.expand(Bs, N // 2, 2, 2, L)
    else:
        gv = g.float()
        if beta_g is not None:
            gv = gv + beta_g[0].reshape(1, N, 1)
        out = layout_gate_act(gv, gate_act).reshape(Bs, N // 2, 2, 1, L) * pair
    return out.reshape(Bs, 2 * N, L).to(T.dtype)


def _preact_gate_ok(theta, g_B, g_C, beta_gB, beta_gC, gate_act):
    """Can the gate ACTIVATION (not the product) leave a ROTATING kernel?

    Yes whenever there is a gate, it is not already raw, and its static bias
    is head-shared -- then act(g + beta) is the same for every head.  Shape
    and `None` tests only, like `_prefold_gate_ok`.
    """
    return theta is not None and (g_B is not None or g_C is not None) \
        and gate_act != GATE_RAW and _shared(beta_gB) and _shared(beta_gC)


def _preact_gate(g, beta_g, gate_act):
    if g is None:
        return None
    gv = g.float()
    if beta_g is not None:
        gv = gv + beta_g[0].reshape(1, g.shape[1], 1)
    return layout_gate_act(gv, gate_act)


def _prefold_gate(B_t, C_t, beta_b, beta_c,
                  g_B=None, g_C=None, beta_gB=None, beta_gC=None,
                  gate_act=GATE_SIGMOID):
    """The fold on a NON-rotating launch, done once on the host.

    bias -> gate, which is the kernel's own order with the (absent) rotation
    removed, and the kernel does nothing else to B/C before using them -- so
    this is exact, not an approximation.  Under `_prefold_gate_ok` every
    stage is a function of (b, n, t) alone, where the (H, B) grid recomputes
    all of it once per head: H = 64 times over at the reference shape.

    Plain autograd-visible torch ops, so the chain rule back to
    B_t/C_t/g_B/g_C and to the folded betas comes for free and no adjoint is
    written by hand.  The result is cast back to B_t's dtype on purpose: the
    kernel re-reads B/C once per head, so keeping the fold in fp32 would
    double that traffic and spend the prize on bandwidth.
    """
    N = B_t.shape[1]

    def fold(T, beta, g, bg):
        T = T.float()
        if beta is not None:
            T = T + beta[0].reshape(1, N, 1)      # G=1: the bias is head-shared
        if g is not None:
            gv = g.float()
            if bg is not None:
                gv = gv + bg[0].reshape(1, N, 1)
            T = T * layout_gate_act(gv, gate_act)
        return T

    return (fold(B_t, beta_b, g_B, beta_gB).to(B_t.dtype),
            fold(C_t, beta_c, g_C, beta_gC).to(C_t.dtype))


def _amp_fwd(fn):
    @functools.wraps(fn)
    def _inner(ctx, *args, **kwargs):
        with torch.amp.autocast("cuda", enabled=False):
            return fn(ctx, *args, **kwargs)
    return torch.amp.custom_fwd(_inner, device_type="cuda")


def _amp_bwd(fn):
    @functools.wraps(fn)
    def _inner(ctx, *args, **kwargs):
        with torch.amp.autocast("cuda", enabled=False):
            return fn(ctx, *args, **kwargs)
    return torch.amp.custom_bwd(_inner, device_type="cuda")


def _final_phase_vjp(seed, inputs, opts):
    """Adjoint of wrapped terminal phase, away from wrap discontinuities.

    Phase is accumulated for every pair in the launched row window, including
    nonrotating selectors. Unlaunched pairs only retain their incoming phase.
    Workspace scales with existing clock/feature tensors, not their product.
    """
    theta, delta = inputs["theta"], inputs["delta"]
    dtype = torch.float64 if theta.dtype == torch.float64 else torch.float32
    seed = seed.to(dtype)
    grads = {}
    if inputs.get("theta_init") is not None:
        grads["theta_init"] = seed
    lo = opts.get("n_off", 0) // 2
    rows = opts.get("n_rows")
    ph = theta.dim() == 4                # theta_per_head: (B, H, K, L)
    hi = theta.shape[-2] if rows is None else lo + rows // 2
    active = seed.clone()
    active[..., :lo] = 0
    active[..., hi:] = 0
    if opts.get("theta_delta", False):
        raw = delta.to(dtype)
        if inputs.get("delta_bias") is not None:
            raw = raw + inputs["delta_bias"].to(dtype)[None, :, None]
        softplus = opts.get("delta_softplus", True)
        clock = torch.nn.functional.softplus(raw) if softplus else raw
        if ph:
            grads["theta"] = active[..., None] * clock[:, :, None, :]
            clock_grad = torch.einsum("bhk,bhkl->bhl", active, theta.to(dtype))
        else:
            grads["theta"] = torch.bmm(active.transpose(1, 2), clock)
            clock_grad = torch.bmm(active, theta.to(dtype))
        if softplus:
            clock_grad = clock_grad * torch.where(raw > 20, 1., raw.sigmoid())
        grads["delta"] = clock_grad
        if inputs.get("delta_bias") is not None:
            grads["delta_bias"] = clock_grad.sum((0, 2))
    else:
        grads["theta"] = (active if ph else active.sum(1)).unsqueeze(-1).expand_as(theta)
    return grads


class _NoDoubleBackward(torch.autograd.Function):
    """Identity on a first-order gradient; differentiating it raises."""
    @staticmethod
    def forward(ctx, g, *deps):
        return g.view_as(g)

    @staticmethod
    def backward(ctx, *_):
        raise RuntimeError(
            "mh_scan: double backward is not supported -- the MH kernels' "
            "backward is not differentiable (a second derivative would "
            "silently miss the scan's part)")


class MHScan(torch.autograd.Function):

    @staticmethod
    @_amp_fwd
    def forward(ctx, opts, *tensors):
        # `mh_scan` drops h_final and theta_final unless return_state, so their
        # adjoints are structurally absent -- but materialised they arrive as
        # ZEROS, and a non-None gh_final turns on the whole HAS_GHL path: a
        # permuted fp32 seeding copy plus a branch inside a kernel already at
        # 255 registers.
        ctx.set_materialize_grads(False)
        # popped, not forwarded: `opts` is the kwargs both kernels take, and the
        # backward learns about the handover from `hck` itself. `mem` is a
        # backward-only dict for the same reason -- the forward owns none of
        # the buffers it names.
        ckpt = opts.pop("ckpt", True)
        # TRAIN-ONLY checkpoints: a forward no backward can follow -- grad mode off, or no
        # input that requires grad -- writes no `hck` at all and keeps the
        # plain recompute path.  Two conditions, because neither is enough:
        # `mh_scan` folds the CALLER's grad mode into `ckpt` (inside this
        # forward `torch.is_grad_enabled()` is always False, and under
        # no_grad `needs_input_grad` still reports requires_grad), and
        # `needs_input_grad` catches grad mode on with nothing to train.
        if not any(ctx.needs_input_grad[1:]):
            ckpt = False
        mem = opts.pop("mem", None)
        kw = {k: v for k, v in zip(TENSOR_ARGS, tensors) if v is not None}
        args = tuple(kw.pop(k) for k in TENSOR_ARGS[:5])
        # the checkpoint stride IS the backward's chunk length -- the
        # backward indexes the
        # slots with its own BLOCK_L, and pins it to this when `hck` arrives.
        bl = auto_block_l(args[0].shape[2], args[0].shape[3]) if ckpt else 0
        # CKBF is the one `mem` axis the FORWARD owns a piece of: the handed-
        # over checkpoint's dtype is decided at allocation, here. None lets
        # mh_forward apply its default (input dtype when 16-bit).
        Y, h_final, hck, theta_final = mh_forward(
            *args, **kw, **opts, ckpt_l=bl,
            ckpt_dt=torch.float32 if (mem or {}).get("CKBF") == 0 else None)
        # Saved tensors are released by autograd after backward; a tensor
        # attribute on ctx would survive as long as the output's grad_fn.
        ctx.save_for_backward(*tensors, hck)
        # the backward runs outside mh_scan's scope: carry its options
        ctx._kernel_options = _ko.current("hybrid_mamba")
        ctx.opts = opts
        ctx.mem = mem
        return Y, h_final, (torch.empty(0, device=Y.device) if theta_final is None
                           else theta_final)

    @staticmethod
    def backward(ctx, gy, gh_final, gtheta_final):
        # The kernels' backward is not itself differentiable.  Under
        # create_graph autograd would record nothing for it, and a second
        # derivative would silently keep only the host folds' part.
        # `once_differentiable` alone does not catch that: it refuses only
        # when the incoming GRADIENT requires grad, not when the saved
        # inputs do.  So the gradients are tied to every differentiable
        # input by a node whose backward raises.
        with torch.no_grad():
            out = MHScan._backward(ctx, gy, gh_final, gtheta_final)
        if torch.is_grad_enabled():
            deps = [t for t in (*ctx.saved_tensors, gy, gh_final, gtheta_final)
                    if torch.is_tensor(t) and t.requires_grad]
            if deps:
                out = tuple(None if o is None else _NoDoubleBackward.apply(o, *deps)
                            for o in out)
        return out

    @staticmethod
    @_amp_bwd
    @_ko.scoped_ctx("hybrid_mamba")
    def _backward(ctx, gy, gh_final, gtheta_final):
        *tensors, hck = ctx.saved_tensors
        kw = {k: v for k, v in zip(TENSOR_ARGS, tensors) if v is not None}
        args = tuple(kw.pop(k) for k in TENSOR_ARGS[:5])
        # gy is None iff Y is unused and the graph reaches here through h_final
        # alone; Y is x-shaped, which is the only thing the kernel needs.
        if gy is None:
            gy = torch.zeros_like(tensors[0])
        g = mh_backward(*args, gy.contiguous(),
                        gh_final=None if gh_final is None else gh_final.contiguous(),
                        hck=hck, mem=ctx.mem, **kw, **ctx.opts)
        # The raw launcher also returns reusable scratch for tuning callers.
        # Autograd only needs the ABI gradients: release unreferenced scratch
        # before allocating casts and terminal-phase adjoints.
        g = {name: g[name] for name in TENSOR_ARGS if name in g}
        if gtheta_final is not None and gtheta_final.numel():
            # The terminal phase is a sum of increments, independent of h.
            # Contract directly to B,K,L and B,H,L: never form B,H,K,L.
            phase_inputs = dict(zip(TENSOR_ARGS, tensors))
            phase_grads = _final_phase_vjp(gtheta_final, phase_inputs, ctx.opts)
            for name, value in phase_grads.items():
                g[name] = g[name] + value if name in g else value
        out = []
        for i, name in enumerate(TENSOR_ARGS):
            t = tensors[i]
            if t is None or name in NON_DIFF or not ctx.needs_input_grad[i + 1]:
                out.append(None)
            else:
                out.append(g[name].reshape(t.shape).to(t.dtype))
        return (None, *out)


# E2 (`src/orbit_ssm/compile_compat.py`): the mh family's public entry, and
# therefore its `torch.compile` boundary.  Its kernels are `@gluon.jit`
# (`gl`, not `tl`), which Inductor's kernel-reload harness cannot
# re-import -- so Dynamo must never capture them.  Pass-through in eager.
@dynamo_opaque
@_ko.scoped_kwarg("hybrid_mamba")
# Paper letters -> public layer names: H = n_heads, P = d_head, N = d_state.
# (HybridMambaSSM exposes n_heads and d_head as read-only properties.)
def mh_scan(
    x, delta, A_log, B_t, C_t, *,
    delta_w=None, a_dyn=None, mu=None, mu_gain=None,
    g_B=None, g_C=None, theta=None,
    beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None,
    delta_bias=None, h_init=None, theta_init=None, mu_init=None,
    d_skip=None, row_law=None, row_rot=None, z=None,
    delta_softplus=True, a_dyn_res="n", mu_res="n",
    dyn_act=DYN_SOFTPLUS, dyn_eps=0.0, gate_act=GATE_SIGMOID,
    n_off=0, n_rows=None, return_state=False, ckpt=True, mem=None,
    validate=None, trap_lam=None, write_prev=None,
    theta_delta=False, theta_h=None, kernel_options=None,
):
    """Differentiable MH scan. Returns Y, or (Y, h_final, theta_final).

    `kernel_options` (the `hybrid_mamba` family, `orbit_ssm.kernel_options`:
    autotune tier, fwd/bwd config pins, gate_bias_partial) is active for
    this call and for its backward; None or {} leaves the environment in
    charge.

    `trap_lam` optionally selects transported endpoint forcing; it contains
    activated weights in B,H,L. The existing effective write clock (delta_w
    when supplied) scales both taps. `write_prev` is the previous UNSCALED
    effective write in state coordinates, B,H,P,N; absent means zero. A
    continuation supplies it together with h_init (and existing phase/mu
    carries); h_final alone does not contain this independent history. Neither
    keyword changes a module constructor or adds a checkpoint parameter.

    `validate` runs `layout.assert_pair_tied` on the inputs before anything
    else: on a rotating pair every factor of the modulus must agree between
    the two coordinates, because the rotation is folded onto B/C rather than
    applied to the state, and that fold is a change of frame only for a
    scalar decay. It is OFF by default -- it is whole-tensor comparisons ending
    in `.any()`, i.e. a device->host sync per call. `validate=True` forces it
    on for one call, `validate=False` off; `None` takes the process-wide
    default from the `HYBRID_SSM_MH_VALIDATE` environment variable, which
    `tests/conftest.py` sets to 1 so the whole suite runs checked.

    `ckpt=True` has the forward write its chunk states and the backward read
    them instead of rebuilding them (see the module header), as structure A
    does. It reads as if it buys speed with a live (B, H, L/BLOCK_L + 1, N, P) fp32 buffer per layer, and it does
    not: `mh_backward` allocates that same buffer itself when `hck is None`, so
    the flag moves an existing allocation's LIFETIME out of the backward's
    workspace and into the saved set. This is the default. Chunk interiors are still reconstructed during backward;
    this does not checkpoint/re-execute the whole mixer. Saved boundaries
    remain live across stacked layers' forwards. `ckpt=False` explicitly
    trades that lifetime for the additional boundary-reconstruction pass.

    `mem` dials the backward's scratch, which IS a real peak. Three axes are
    on by default: AGX folds the NS `x_s` shards onto one slab with atomics
    (little time for a large peak reduction); BCS folds the H*NP BC slot
    buffers the same way, made cheap by the vectorised red
    (`red.global.add.v4.f32`), where the scalar fold is slow;
    CKBF stores the chunk checkpoints in the input dtype when
    that is 16-bit, halving the largest fixed buffer with no visible accuracy
    cost.

    BCBF (bf16 slot buffers) only bites where the fold is refused -- rows
    with BC biases, whose gradients are length-L reductions read back out of
    the slot buffer, and pre-sm_90 devices with no vector red.

    Atomics reorder sums, so the default backward is reproducible to fp32
    rounding, not to the bit. A seeded campaign that needs bit-identical
    gradients pins `mem=dict(AGX=0, BCS=0)` and runs the determinism
    test in that mode.
    """
    call = dict(locals())          # first statement: exactly the parameters
    _portable.refuse_gluon(
        "hybrid_mamba", "mh_scan (the MH rows' forward and backward are Gluon-only: `fwd_gluon._mh_fwd_kernel` / "
        "`bwd_gluon._mh_bwd_kernel`; gap G-MH)",
        "Use Execution(backend='torch') (`mh_torch_scan`, PyTorch ops) on this backend")
    if g_B is None and g_C is None:
        gate_act = GATE_RAW        # no gate: the activation is unused, keep it out of the compile key
    elif gate_act not in _KERNEL_GATE_ACTS:
        # e.g. one_plus_tanh (layout.GATE_ONE_PLUS_TANH) is a host-side fold of the
        # pre-rotation map only; the kernels' `_gate` would fall through to softplus
        raise ValueError(f"mh_scan: gate_act={gate_act} has no kernel implementation for "
                         f"in-scan gates g_B/g_C (kernel codes {sorted(_KERNEL_GATE_ACTS)}); "
                         f"fold it into B/C on the host (HybridMambaSSM gate_pos='pre')")
    window = _rot_window(theta, row_rot, n_off, n_rows, write_prev, mu_init,
                         return_state)
    if window:
        return _split_rot_scan(window, call)
    del call, window
    if write_prev is not None and trap_lam is None:
        raise ValueError("write_prev requires trap_lam")
    # AUTOCAST IO: under an active autocast region the scan's
    # IO dtype IS the autocast dtype, so the ACTIVATION inputs -- the
    # drive `x`, the maps `B_t` / `C_t` and the output gate `z` -- are
    # cast to it once, here (an upstream promotion to fp32 would otherwise
    # double the scan's x / Y / dX traffic and its checkpoint buffers).
    # Never cast (fp32 BY DESIGN): delta, delta_w, A_log, a_dyn, mu,
    # mu_gain, theta, theta_h, the beta_* / d_skip parameters, delta_bias,
    # trap_lam and the carries h_init / theta_init / mu_init / write_prev;
    # the gate pre-activations g_B / g_C keep their dtype (the kernel
    # evaluates act(g + beta) in fp32).  Identity outside autocast / fp64.
    x, B_t, C_t, z = (autocast_io(x), autocast_io(B_t), autocast_io(C_t),
                      autocast_io(z))
    # SSD rotation law: `theta` is the
    # head-SHARED angle feature (B,K,L) and the kernel scales it by this
    # head's effective delta (theta_delta), so the phase is per (b,h,k,t)
    # exactly as in ssd_mimo; `theta_h` (H,K) offsets the READ map only.
    # Neither is head-free; an external fold must retain the head axis.
    if theta_delta and theta is None:
        raise ValueError("theta_delta requires theta")
    if theta is not None:
        K_, L_ = B_t.shape[1] // 2, x.shape[-1]
        if tuple(theta.shape) not in ((x.shape[0], K_, L_), (x.shape[0], x.shape[1], K_, L_)):
            raise ValueError(
                "theta must be (B, K, L) (head-shared angle feature) or "
                "(B, H, K, L) (theta_per_head), K = N // 2")
    if theta_h is not None:
        if theta is None:
            raise ValueError("theta_h requires theta (a rotating launch)")
        if tuple(theta_h.shape) != (x.shape[1], B_t.shape[1] // 2):
            raise ValueError("theta_h must be (H, K) with K = N // 2")
    if theta_init is not None and theta is None:
        # the kernel turns rotation on for an angle, never for a carry alone;
        # `theta_init` is the SSD law's chunk-boundary carry and only means
        # anything beside a `theta`.
        raise ValueError(
            "theta_init is the rotation law's chunk carry and needs theta: "
            "a lone theta_init does not turn the kernel's rotation on, so it "
            "is refused")
    theta_delta = bool(theta_delta) and theta is not None
    endpoint_own = None
    if trap_lam is not None:
        # This opt-in changes only the forcing quadrature. Decay and all
        # effective-map folds keep their existing meaning and parameters.
        if trap_lam.shape != delta.shape:
            raise ValueError("trap_lam must have delta's B,H,L shape (activated lambda)")
        raw_delta = delta.float()
        if delta_bias is not None:
            raw_delta = raw_delta + delta_bias.float()[None, :, None]
        if delta_softplus:
            delta = torch.nn.functional.softplus(raw_delta)
            clock = delta if delta_w is None else torch.nn.functional.softplus(delta_w.float())
        else:
            delta = raw_delta
            clock = delta if delta_w is None else delta_w.float()
        delta_bias, delta_softplus = None, False
        from .endpoint import mh_endpoint_weights
        weight = trap_lam.float()
        if clock.ndim == 4:
            weight = weight[:, :, None, :].expand_as(clock)
        # (w, own): the kernels read the diagonal as own = lam*dt directly,
        # never as w - nxt (endpoint.py)
        delta_w, endpoint_own = mh_endpoint_weights(clock, weight)
        if write_prev is not None:
            if write_prev.shape != (x.shape[0], x.shape[1], x.shape[2], B_t.shape[1]):
                raise ValueError("write_prev must have state shape B,H,P,N")
            first = clock[..., 0] * (1 - weight[..., 0])
            if clock.ndim == 3:
                first = first[..., None]
            lag = first[..., None] * write_prev
            if n_off or n_rows is not None:
                end = B_t.shape[1] if n_rows is None else n_off + n_rows
                lag = torch.nn.functional.pad(lag[..., n_off:end], (n_off, B_t.shape[1] - end))
            h_init = lag if h_init is None else h_init + lag
    if z is not None:
        # Three refusals, and
        # each of them is the SAME ownership argument the D skip makes,
        # one step stronger because a gate is a PRODUCT with the whole
        # readout rather than a term added to it:
        #
        #  * an n-SHARD (`n_off`/`n_rows`) computes one SUMMAND of Y, and
        #    `gate(sum) != sum(gate)`.  Refused, never divided;
        #  * `z` is CHANNEL-major, (B, H, P, L) -- the layout `x` and `Y`
        #    already have.  A token-major `z` would be a stride-d_inner
        #    gather per element at P == 1, which is the whole reason the
        #    gate was worth moving; the caller transposes, or does not
        #    fuse;
        #  * it is an OUTPUT epilogue, so it applies to Y and to nothing
        #    the state carries: `h_final` and `theta_final` are ungated by
        #    construction and a chunked caller resumes from them
        #    unchanged.
        if n_off or n_rows is not None:
            raise NotImplementedError(
                "mh_scan(z=...) is the WHOLE-N call: an n_off/n_rows "
                "shard computes one summand of the readout and the "
                "output gate multiplies the whole of it")
        if z.shape != x.shape:
            raise ValueError(
                f"mh_scan(z=...) must be CHANNEL-major, the same "
                f"(B, H, P, L) shape as x -- got {tuple(z.shape)} "
                f"against {tuple(x.shape)}")
        if z.dtype != x.dtype:
            raise ValueError(
                f"mh_scan(z=...) must match x's dtype ({x.dtype}); "
                f"got {z.dtype}")
        if not z.is_contiguous():
            # the backward kernel loads z (and writes gz) at CONTIGUOUS
            # (B,H,P,L) offsets; a strided z (e.g. a slice of a joint
            # projection) would give NaN / silently wrong gradients while the
            # forward was exact.  The module always passes a contiguous z.
            z = z.contiguous()
    if d_skip is not None and (n_off or n_rows is not None):
        # an n-SHARD's Y is one SUMMAND of the readout, so a skip added
        # per shard would be counted once per shard.  Refused rather than
        # divided by the shard count, which would only be right when the
        # caller happens to launch every shard.
        raise NotImplementedError(
            "mh_scan(d_skip=...) is the WHOLE-N call: an n_off/n_rows "
            "shard computes one summand of the readout and the skip must "
            "be added exactly once")
    if _validate_default() if validate is None else validate:
        # BEFORE the two host hoists below: they are exact and preserve the
        # tie (a difference of tied columns is tied, an activation of tied
        # columns is tied), but the message should name the tensor the caller
        # passed, not a derived one. `a_dyn`/`mu` go in only at n-resolution
        # -- at h-resolution they carry no N axis to untie.
        assert_pair_tied(
            A_log, row_rot, beta_a=beta_a, mu_gain=mu_gain,
            row_law=row_law,
            a_dyn=a_dyn if a_dyn_res == "n" else None,
            mu=mu if mu_res == "n" else None,
            mu_init=mu_init if mu_res == "n" else None)
    if a_dyn is not None and dyn_act != DYN_RAW:
        # the rate is act(a_dyn)+eps and depends on a_dyn alone, so the
        # (H,B) grid evaluates one identical transcendental per head -- and at
        # n-resolution it does so inside the innermost n loop. Applying it once
        # here removes most of that law's forward cost and more of the
        # backward's, which then does not carry the activation's derivative
        # either. Exact,
        # not an approximation: the kernel's DYN_RAW branch is the identity.
        a_dyn = layout_dyn_rate(a_dyn.float(), dyn_act, dyn_eps)
        dyn_act, dyn_eps = DYN_RAW, 0.0
    mu_prediff = mu is not None
    if mu_prediff:
        # the MU driver is the telescoped increment mu_t - mu_{t-1},
        # so the (H,B) grid rebuilds one identical difference per head -- and at
        # n-resolution it does so in the innermost n loop, where the shift is a
        # gather plus a misaligned column-0 load on top of the mu load itself.
        # Unlike A_DYN's rate this hoists WHOLE rather than half: the increment
        # is the same (b,n,t) shape as mu, so nothing the contract forbids is
        # materialised and the kernel is left one plain load. fp32 because the
        # difference of two nearby values is where the precision goes.
        mu = mu.float()
        prev = (torch.zeros_like(mu[..., :1]) if mu_init is None
                else mu_init.reshape(*mu.shape[:-1], 1).float())
        mu = mu - torch.cat([prev, mu[..., :-1]], dim=-1)
        mu_init = None
    opts = dict(delta_softplus=delta_softplus, a_dyn_res=a_dyn_res,
                mu_res=mu_res, dyn_act=dyn_act, dyn_eps=dyn_eps,
                gate_act=gate_act, mu_prediff=mu_prediff,
                n_off=n_off, n_rows=n_rows,
                # train-only checkpoints: the caller's grad mode is read HERE,
                # because inside the Function's forward it is always off and
                # `needs_input_grad` still reports requires_grad under no_grad
                ckpt=bool(ckpt) and torch.is_grad_enabled(), mem=mem,
                theta_delta=theta_delta)
    route = rotation_route() if theta is not None else "in_scan"
    _note_route("real" if theta is None else route)
    if route == "fold":
        why = _fold_refusal(theta, n_off, n_rows, P=x.shape[2], mem=mem)
        if why:
            ROTATION_FALLBACKS[why] = ROTATION_FALLBACKS.get(why, 0) + 1
            route = "in_scan"
            _note_route(f"in_scan({why})")
            _warn_narrow_rotation(x.shape[2])
    if route == "fold":
        from .half_fold import half_fold
        fold_args = (B_t, C_t, delta, theta)
        fold_kw = dict(theta_h=theta_h, theta_init=theta_init,
                       beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C,
                       beta_gB=beta_gB, beta_gC=beta_gC, delta_bias=delta_bias,
                       row_rot=row_rot, theta_delta=theta_delta,
                       delta_softplus=delta_softplus, gate_act=gate_act)
        B_t, C_t, final_phase = half_fold(*fold_args, **fold_kw)
        # One real scan over ALL rows, in the public (interleaved) basis.
        opts.update(bc_head=True, theta_delta=False)
        with _refold_saved_maps(B_t, C_t, fold_args, fold_kw):
            y, last, _ = MHScan.apply(
                opts, x, delta, A_log, B_t, C_t,
                delta_w, a_dyn, mu, mu_gain,
                None, None, None, beta_a, None, None, None, None,
                delta_bias, h_init, None, mu_init, d_skip, row_law, None,
                z, endpoint_own, None)
        if not return_state:
            return y
        return y, last, final_phase
    # The scan's two bias-pair loads share a group stride. Normalize only
    # unequal groupings; autograd sums repeated groups back to each original
    # (G,N) parameter, so every bias retains its independently declared G.
    def common_groups(left, right):
        if left is None or right is None or left.shape[0] == right.shape[0]:
            return left, right
        import math
        groups = math.lcm(left.shape[0], right.shape[0])
        if not left.shape[0] or not right.shape[0] or x.shape[1] % groups:
            raise ValueError("B/C bias group counts must divide H")
        # Reduce the repeated-group adjoints in fp32 before rounding back
        # to a low-precision leaf; rounding each partial first loses small
        # grouped gradients when different heads nearly cancel.
        return (left.float().repeat_interleave(groups // left.shape[0], dim=0),
                right.float().repeat_interleave(groups // right.shape[0], dim=0))
    beta_b, beta_c = common_groups(beta_b, beta_c)
    beta_gB, beta_gC = common_groups(beta_gB, beta_gC)
    if _prefold_gate_ok(theta, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC):
        # The one host hoist.  It is the GATE's, and it is
        # reachable only on a NON-rotating launch: the order is
        # bias -> rotate -> gate, so a gate folded onto B/C here would land
        # on the wrong side of any rotation the kernel still has.  Under the
        # in-scan rotation law the angle is head-dependent, so this route
        # keeps its gate in the kernel too.
        B_t, C_t = _recomputed(_prefold_gate, B_t, C_t, beta_b, beta_c,
                               g_B, g_C, beta_gB, beta_gC, gate_act)
        # the kernel now sees a plain problem: no gate, and the betas it
        # would have folded in before each stage are already inside B/C.
        # Nulling all of these is what stops it applying any of them twice.
        beta_b = beta_c = g_B = g_C = beta_gB = beta_gC = None
    elif _pmap_ok(theta, x, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC) == "raw":
        # UNGATED rotating launch on the backward's pv path (PRAW): the maps
        # are B/C themselves, a head-shared bias folded in on the host
        if beta_b is not None or beta_c is not None:
            B_t = _recomputed(_pair_maps_raw, B_t, beta_b)
            C_t = _recomputed(_pair_maps_raw, C_t, beta_c)
        beta_b = beta_c = None
        opts["pmap"] = "raw"
    elif _pmap_ok(theta, x, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC):
        # Gated ROTATING launch: fold bias and gate into per-row PAIR MAPS,
        # act(g_n) * (B_2k, B_2k+1), so the kernels apply diag(g) R(Theta)
        # as one 2x2 per row (kernel constexpr PMAP).  Order bias -> rotate
        # -> gate is exact: g_n (c B_2k - s B_2k+1) = c (g_n B_2k)
        # - s (g_n B_2k+1).  Rounded to B_t's dtype, the same policy as the
        # non-rotating `_prefold_gate`.
        B_t = _recomputed(_pair_maps, B_t, beta_b, g_B, beta_gB, gate_act)
        C_t = _recomputed(_pair_maps, C_t, beta_c, g_C, beta_gC, gate_act)
        beta_b = beta_c = g_B = g_C = beta_gB = beta_gC = None
        opts["pmap"] = True
    elif _preact_gate_ok(theta, g_B, g_C, beta_gB, beta_gC, gate_act):
        # Under rotation the gate PRODUCT must stay after R(Theta) in the
        # kernel, but its ACTIVATION is a function of (b, n, t)
        # alone: evaluated in-kernel it is recomputed once per head and per
        # p-block, a software expf + rcp per element on both passes.  Taking
        # it here leaves the kernel a raw multiply (GATE_RAW), its backward
        # emits dL/dsigma summed over heads, and torch's autograd applies
        # the activation's derivative once, after that sum.  fp32, so the
        # factor the kernel multiplies by is not rounded to the input dtype.
        g_B, g_C = _preact_gate(g_B, beta_gB, gate_act), \
            _preact_gate(g_C, beta_gC, gate_act)
        beta_gB = beta_gC = None
        gate_act = GATE_RAW
        opts["gate_act"] = GATE_RAW
    Y, h_final, theta_final = MHScan.apply(
        opts, x, delta, A_log, B_t, C_t,
        delta_w, a_dyn, mu, mu_gain,
        g_B, g_C, theta,
        beta_a, beta_b, beta_c, beta_gB, beta_gC,
        delta_bias, h_init, theta_init, mu_init, d_skip, row_law, row_rot,
        z, endpoint_own, theta_h)
    if not return_state:
        return Y
    return Y, h_final, (None if theta_final.numel() == 0 else theta_final)
