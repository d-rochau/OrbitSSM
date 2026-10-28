"""MH oracle library (kept out of the test files so the contract suite imports no test module):
the fp64 oracle and scorers below, plus the half-fold test inputs.

Half-block fold: independent equations, boundary adjoints, one real scan.

Also the home of the MH A7 helpers (A7, tests/_tolerance.py)
the other MH files import: `oracle_terms` runs the fp64 oracle `mh_reference`
in absolute arithmetic, `amplification` and `oracle_c` derive the rounding
depth, `assert_oracle_close` scores every tensor per head (A13) with the
terms-scaled bound |a - r| <= c u sum|terms|.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from _tolerance import (DrawPool, NearZeroReference, assert_close_terms, assert_sixteen_bit_close,
                        assert_dtype_close, draw_seeds, is_sixteen_bit,
                        terms_abs, unit_roundoff)


# ---------------------------------------------------------------------------
# A7 helpers: an MH arm against the fp64 oracle `mh_reference`
# ---------------------------------------------------------------------------
ARG_NAMES = ("x", "delta", "A_log", "B_t", "C_t")
# mh_reference's own keywords (not `_prepare`'s)
_ORACLE_ONLY = ("h_init", "n_off", "n_rows", "d_skip", "z", "trap_lam",
                "write_prev")
# tensors whose leading axis is the head (or head group): scored per index of
# axis 0; everything else is batch-first and scored per index of axis 1
# (head, or row / pair for the head-shared B, C, gates and theta)
_HEAD_FIRST = ("A_log", "mu_gain", "theta_h", "delta_bias", "d_skip",
               "beta_a", "beta_b", "beta_c", "beta_gB", "beta_gC")
# factors of one scan term: Delta, x, B, rotation, gate, C, decay, lambda
_FACTORS = 8
# 16-bit IO: the roundings in IO precision on one path -- the checkpointed
# state (CKBF), the per-slot Delta partial (DEBF), the stored output/gradient
_IO_ROUNDINGS = 3


def leaf_names(kw):
    """Names of `args + [requires-grad kw tensors]`, the leaf order every MH
    test uses."""
    return list(ARG_NAMES) + [k for k, v in kw.items()
                              if torch.is_tensor(v) and v.requires_grad]


def head_axis(name):
    return 0 if name.split(".")[-1] in _HEAD_FIRST else 1


def mh_dims(args):
    b, h, p, length = args[0].shape
    return b, h, p, args[2].shape[-1], length


def terminal_phase(delta, kw):
    """The terminal phase `mh_scan(..., return_state=True)[2]` returns, NOT
    wrapped: theta_init + the launched window's increments
    (`functions._final_phase_vjp` is its adjoint)."""
    theta = kw["theta"]
    h = delta.shape[1]
    k = theta.shape[-2]
    raw = delta if kw.get("delta_bias") is None else \
        delta + kw["delta_bias"][None, :, None]
    clock = F.softplus(raw) if kw.get("delta_softplus", True) else raw
    th = theta[:, None] if theta.dim() == 3 else theta
    inc = th * clock[:, :, None] if kw.get("theta_delta") else \
        th.expand(-1, h, -1, -1)
    phase = inc.sum(-1) * window_mask(kw, k, delta.device)
    if kw.get("theta_init") is not None:
        phase = kw["theta_init"] + phase
    return phase


def window_mask(kw, k, device):
    lo = kw.get("n_off", 0) // 2
    rows = kw.get("n_rows")
    hi = k if rows is None else lo + rows // 2
    mask = torch.zeros(k, dtype=torch.float64, device=device)
    mask[lo:hi] = 1
    return mask


def wrap_phase(phase, kw):
    mask = window_mask(kw, phase.shape[-1], phase.device).bool()
    return torch.where(mask, phase.remainder(2 * torch.pi), phase)


def oracle_terms(args, kw, weights, *, phase=False):
    """`mh_reference` (fp64) in absolute arithmetic (`terms_abs`).

    `.out` = (Y, h_final[, unwrapped terminal phase]) -- the plain fp64
    oracle's values; `.terms` their sum|terms|; `.grads` / `.grad_terms` the
    oracle's gradients for `weights` (one per output, None to skip) of every
    requires-grad leaf, in the order `args + kw` (`leaf_names`)."""
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import mh_reference
    kw = dict(kw)
    if kw.get("h_init") is None:
        # the oracle otherwise allocates h from torch.zeros and writes the
        # terms state into it in place; an explicit zero start is the same
        # value and gives the in-place write a terms buffer
        b, h, p, n, _ = mh_dims(args)
        kw["h_init"] = torch.zeros(b, h, p, n, dtype=torch.float64,
                                   device=args[0].device)

    def fn(*a, **k):
        y, hf = mh_reference(*a, **k)
        return (y, hf, terminal_phase(a[1], k)) if phase else (y, hf)
    return terms_abs(fn, *args, grad_outputs=list(weights), **kw)


def amplification(args, kw):
    """The A7 amplification of the scan's nonlinear functions of rounded
    arguments, from the oracle's own front end (fp64):

    * decay: a_t = exp(log a_t) per token, each argument carrying O(1)
      roundings of |log a_t|; a chain of them carries u * sum_t |log a_t|
      -> + max over (b, h, p, n) of sum_t |log a_t|;
    * angle: the phase is a running sum of L increments, so its absolute
      error is u sqrt(L) |Theta| (typical) and every rotated factor carries
      it -> + sqrt(L) * max over (b, h, k) of |theta_init| + sum_t |inc_t|.
    """
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import _prepare
    det = [a.detach() for a in args]
    pk = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kw.items()
          if k not in _ORACLE_ONLY}
    with torch.no_grad():
        st = _prepare(*det, **pk)
        log_a = (st["c1"][None, ..., None] * st["s1"][:, :, None]
                 + st["c2"][None, ..., None] * st["s2"][:, :, None])
        amp = log_a.abs().sum(-1).amax().item()
        if kw.get("theta") is not None:
            ak = {k: (v.detach().double().abs() if torch.is_tensor(v)
                      and v.is_floating_point() else v)
                  for k, v in kw.items()}
            raw = det[1].double()
            if kw.get("delta_bias") is not None:
                raw = raw + kw["delta_bias"].detach().double()[None, :, None]
            clock = F.softplus(raw) if kw.get("delta_softplus", True) else raw
            ak["delta_bias"] = None
            ak["delta_softplus"] = False
            mag = terminal_phase(clock.abs(), ak)
            amp += math.sqrt(det[0].shape[-1]) * mag.amax().item()
    return amp


def oracle_depth(numel, dims):
    """The longest sum feeding an MH value with `numel` elements: L tokens,
    N rows, `_FACTORS` factors, and for a gradient the reduction over the
    output elements its leaf broadcasts to (B H P N L / numel)."""
    b, h, p, n, length = dims
    return length + n + _FACTORS + (b * h * p * n * length) // max(numel, 1)


def oracle_c(numel, dims, amp, u):
    """A7 rounding depth of an MH value with `numel` elements.

    fp32 part: a scan value is a sum over <= L tokens and N rows of terms of
    `_FACTORS` factors, and a gradient additionally reduces over the output
    elements its leaf broadcasts to (B H P N L / numel): typical
    sqrt(depth), plus the decay/angle amplification `amp`.  16-bit IO adds
    `_IO_ROUNDINGS` roundings in the IO format (the fp32 part is then
    expressed in units of that u).  Where both arms round, the caller passes
    T_arm + T_ref as the terms (tests/_tolerance.py) and the same c."""
    c32 = math.sqrt(oracle_depth(numel, dims)) + amp
    uu = unit_roundoff(u)
    if uu <= unit_roundoff("fp32"):
        return c32
    return _IO_ROUNDINGS + c32 * unit_roundoff("fp32") / uu


def assert_oracle_close(pairs, dims, amp, u, *, pool=None, label=""):
    """Every (name, actual, reference, sum|terms|), per head (`head_axis`).

    fp32 / fp64 u: the terms-scaled bound with c = `oracle_c`.  16-bit u
    (class (b), the kernel's IO rounding does not cancel): A7's 16-bit
    policy, tol16 with depth = `oracle_depth`, guarded per head, falling
    back to the terms-scaled form for a head whose reference is ~0
    (`assert_sixteen_bit_close`).  A reference of None (a leaf the oracle
    does not use) requires no gradient or an exact zero one.  `pool` pools
    the signed bias per name over a cell's draws (keep `label` fixed)."""
    for name, a, r, t in pairs:
        tag = f"{label}{name}"
        if r is None:
            assert a is None or a.count_nonzero() == 0, tag
            continue
        assert a is not None, f"{tag}: no gradient"
        c = oracle_c(r.numel(), dims, amp, u)
        if isinstance(u, torch.dtype) and is_sixteen_bit(u):
            assert_sixteen_bit_close(a, r, t, u, depth=oracle_depth(r.numel(), dims),
                                     c_terms=c, per=head_axis(name), pool=pool,
                                     name=tag)
        else:
            assert_close_terms(a, r, t, u, c, per=head_axis(name), name=tag,
                               pool=pool)


def oracle_pairs(outputs, grads, res, names, out_names=("y", "h_final")):
    """(name, actual, oracle value, oracle terms) for the outputs and every
    leaf gradient."""
    pairs = list(zip(out_names, outputs, res.out, res.terms))
    pairs += list(zip(names, grads, res.grads, res.grad_terms))
    return pairs


def oracle_phase_pair(actual, res, kw):
    """('phase', actual, wrapped oracle phase, sum|terms|) for
    `oracle_terms(..., phase=True)`: the unwrapped sum's terms plus the wrap's
    own summand |2 pi k| (the wrap keeps the sum's absolute error)."""
    wrapped = wrap_phase(res.out[2], kw)
    return ("phase", actual, wrapped, res.terms[2] + (wrapped - res.out[2]).abs())


def continuation_pairs(whole, split, grads_whole, grads_split, args, kw,
                       weights, cut):
    """(name, whole-call value, split-call value, sum|terms|) for a call over
    L tokens vs two calls joined at `cut` by the carried (h, phase[, mu]).

    Both arms are the same kernel formulation and both round, so the terms
    are T_whole + T_split (tests/_tolerance.py) with T_whole = T_split = the
    fp64 oracle's sum|terms| (a carried state is a partial sum of the same
    summands); the phase's include every wrap's summand |2 pi k| (whole: one
    wrap; split: the carried phase's and the final one's).  The two arms'
    errors are summed in the terms; c is one arm's (`oracle_c`)."""
    res = oracle_terms(args, kw, weights, phase=True)
    first = {k: (v[..., :cut] if k == "theta" else v) for k, v in kw.items()}
    carried = terminal_phase(args[1][..., :cut].detach().double(),
                             {k: (v.detach().double() if torch.is_tensor(v)
                                  and v.is_floating_point() else v)
                              for k, v in first.items()})
    rest = res.out[2] - carried
    joined = wrap_phase(carried, kw) + rest
    wraps = sum((wrap_phase(v, kw) - v).abs() for v in (res.out[2], carried, joined))
    terms = (2 * res.terms[0], 2 * res.terms[1], 2 * res.terms[2] + wraps)
    pairs = list(zip(("y", "h_final", "phase"), whole, split, terms))
    return pairs + list(zip(leaf_names(kw), grads_whole, grads_split,
                            [None if t is None else 2 * t for t in res.grad_terms]))


# ---------------------------------------------------------------------------
# A7 helpers: a module (HM core / mixer / LM) against its fp64 torch route
# ---------------------------------------------------------------------------
def module_reference(module, backend="sequential"):
    """An fp64 copy of `module` whose every HM core runs the torch route
    `backend` -- the reference arm's formulation, for `terms_abs`."""
    import copy
    ref = copy.deepcopy(module).double()
    for core in ref.modules():
        if type(core).__name__ == "HybridMambaSSM":
            core.set_execution(dict(backend=backend))
    return ref


def module_terms(ref, fn, *args, grad_outputs=None):
    """`terms_abs(fn, *args, params=ref)` with TIED parameters handled.

    terms_abs replaces a module's parameters by name, so a tied alias (the
    LM's lm_head.weight is embedding.weight) would stay a constant and its
    use would add nothing to the gradient terms.  Each alias gets its own
    copy of the value in `ref` (which must be a throwaway copy,
    `module_reference`), and the gradients / gradient terms of all aliases of
    a parameter are summed onto its first name -- a tied parameter's
    gradient is the sum of its uses'."""
    first, alias = {}, {}
    for name, p in list(ref.named_parameters(remove_duplicate=False)):
        if id(p) not in first:
            first[id(p)] = name
            continue
        alias[name] = first[id(p)]
        owner, _, leaf = name.rpartition(".")
        setattr(ref.get_submodule(owner) if owner else ref, leaf,
                torch.nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad))
    res = terms_abs(fn, *args, grad_outputs=grad_outputs, params=ref)
    grads, terms = dict(res.param_grads or {}), dict(res.param_grad_terms or {})
    for name, canon in alias.items():
        for d in (grads, terms):
            extra = d.pop(name, None)
            if extra is not None:
                d[canon] = extra if d.get(canon) is None else d[canon] + extra
    return res._replace(param_grads=grads, param_grad_terms=terms)


class ModuleDepth:
    """Rounding depth and amplification of a module's arithmetic, read off
    one fp64 forward (`run`, no grad) of its torch route.

    `depth`: the sum of every reduction on the longest path, each module
    counted once -- a Linear's in_features + 2, a Conv1d's taps x in/groups
    + 2, a norm's width, a scan's L + N + `_FACTORS` (summing parallel branches
    too, so an upper bound).  `amp`: every scan's decay/angle amplification
    (`amplification`, from the arguments the scan received)."""

    def __init__(self, ref, run):
        from orbit_ssm.kernels.hybrid_mamba_mh import torch_scan
        from orbit_ssm.kernels.hybrid_mamba_mh.reference import _prepare
        import inspect
        keep = set(inspect.signature(_prepare).parameters) | set(_ORACLE_ONLY)
        self.depth, self.amp, self.tokens = 0, 0.0, 0
        seen, hooks = set(), []

        def count(mod, inp, out):
            if id(mod) in seen:
                return
            seen.add(id(mod))
            # a dot of n products of 2 factors: n + 2 (A7's n + m)
            if isinstance(mod, torch.nn.Linear):
                self.depth += mod.in_features + 2
            elif isinstance(mod, torch.nn.Conv1d):
                self.depth += mod.kernel_size[0] * mod.in_channels // mod.groups + 2
            elif "Norm" in type(mod).__name__ and inp and torch.is_tensor(inp[0]):
                self.depth += inp[0].shape[-1]
        for mod in ref.modules():
            hooks.append(mod.register_forward_hook(count))
        real = torch_scan.mh_torch_scan
        inside = []

        def spy(*a, **k):
            if inside:                   # the scan's own tiling recursion
                return real(*a, **k)
            args = list(a) + [k.pop(n) for n in ARG_NAMES[len(a):]]
            b, _h, _p, length = args[0].shape
            self.depth += length + args[2].shape[-1] + _FACTORS
            self.tokens = max(self.tokens, b * length)
            self.amp += amplification(args, {
                key: v for key, v in k.items() if key in keep})
            inside.append(1)
            try:
                return real(*args, **k)
            finally:
                inside.pop()
        torch_scan.mh_torch_scan = spy
        try:
            with torch.no_grad():
                run()
        finally:
            torch_scan.mh_torch_scan = real
            for hk in hooks:
                hk.remove()

    def c(self, *, grad=False, u="fp32", io_roundings=0):
        """A7 c of one arm: sqrt(depth) + amp; a gradient adds the backward
        chain (depth again) and the reduction over the batch's tokens; a
        16-bit u adds `io_roundings` roundings in that format (the fp32 part
        is then expressed in units of that u).  Where both arms round, the
        caller passes T_arm + T_ref as the terms."""
        depth = self.depth * (2 if grad else 1) + (self.tokens if grad else 0)
        c32 = math.sqrt(depth) + self.amp
        uu = unit_roundoff(u)
        if uu <= unit_roundoff("fp32"):
            return c32
        return io_roundings + c32 * unit_roundoff("fp32") / uu




def inputs(length=65, p=1, dtype=torch.float32, mixed=True, n=8, seed=591):
    torch.manual_seed(seed)
    b, h = 2, 4
    def leaf(*shape, tied=False, scale=.15):
        v = torch.randn(*shape, device="cuda", dtype=torch.float32) * scale
        if tied:
            v = v.repeat_interleave(2, -1)
        return v.to(dtype).requires_grad_()
    args = [leaf(b,h,p,length), leaf(b,h,length),
            leaf(h,p,n//2,tied=True), leaf(b,n,length), leaf(b,n,length)]
    kw = dict(theta=leaf(b,n//2,length,scale=.03), theta_delta=True,
              theta_h=leaf(h,n//2), theta_init=leaf(b,h,n//2),
              delta_bias=leaf(h), beta_b=leaf(2,n), beta_c=leaf(h,n),
              g_B=leaf(b,n,length), g_C=leaf(b,n,length),
              beta_gB=leaf(h,n), beta_gC=leaf(1,n),
              h_init=leaf(b,h,p,n), delta_softplus=True)
    if mixed:
        kw['row_law']=torch.tensor([0,0,1,1,2,2,0,0],device="cuda",dtype=torch.int32)
        kw['row_rot']=torch.tensor([1,1,0,0,1,1,0,0],device="cuda",dtype=torch.bool)
        kw['a_dyn']=leaf(b,n//2,length).repeat_interleave(2,1).detach().requires_grad_()
        kw['mu']=leaf(b,n//2,length,scale=.03).repeat_interleave(2,1).detach().requires_grad_()
        kw['mu_gain']=leaf(h,p,n//2,tied=True,scale=.1)
        kw.update(a_dyn_res='n',mu_res='n')
    return args,kw


def fold_witness(B,C,delta,theta,wrap=True,**kw):
    B,C,delta,theta=[v.double() for v in (B,C,delta,theta)]
    bs,n,l=B.shape;h=delta.shape[1];k=n//2
    raw=delta+kw['delta_bias'].double()[None,:,None]
    dt=torch.nn.functional.softplus(raw) if kw['delta_softplus'] else raw
    phase=(theta[:,None]*dt[:,:,None] if kw['theta_delta'] else theta[:,None].expand(-1,h,-1,-1)).cumsum(-1)
    phase=phase+kw['theta_init'].double()[...,None]
    mask=kw.get('row_rot')
    if mask is None: mask=torch.ones(n,device=B.device,dtype=torch.bool)
    def one(x,bias,gate,gbias,offset):
        def expand(v): return v.double().repeat_interleave(h//v.shape[0],0)[None,:,:,None]
        x=x[:,None]+expand(bias)
        angle=phase+offset
        e,o=x[:,:,::2],x[:,:,1::2]
        re=e*angle.cos()-o*angle.sin();im=e*angle.sin()+o*angle.cos()
        y=torch.stack([re,im],dim=3).flatten(2,3)
        y=torch.where(mask[None,None,:,None],y,x)
        g=gate.double()[:,None]+expand(gbias)
        act=kw.get('gate_act',1)
        fac=[lambda z:z,torch.sigmoid,torch.exp,torch.tanh,torch.nn.functional.softplus][act](g)
        y=y*fac
        return y.permute(1,0,2,3)
    return one(B,kw['beta_b'],kw['g_B'],kw['beta_gB'],0),one(C,kw['beta_c'],kw['g_C'],kw['beta_gC'],kw['theta_h'].double()[None,:,:,None]),(phase[...,-1].remainder(2*torch.pi) if wrap else phase[...,-1])
















