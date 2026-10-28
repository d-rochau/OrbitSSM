
from __future__ import annotations
import functools
import os
import weakref
import torch
from .bwd_gluon import auto_block_l, mh_backward
from .common_gluon import selector_window
from .fwd_gluon import mh_forward
from .layout import DYN_RAW, DYN_SOFTPLUS, GATE_RAW, GATE_SIGMOID, assert_pair_tied
from .layout import GATE_EXP, GATE_ONE_PLUS, GATE_SOFTPLUS, GATE_TANH
_KERNEL_GATE_ACTS = frozenset((GATE_RAW, GATE_SIGMOID, GATE_EXP, GATE_TANH, GATE_SOFTPLUS, GATE_ONE_PLUS))
from ...compile_compat import dynamo_opaque
from ...layers.proj import autocast_io
from ... import kernel_options as _ko
from .layout import dyn_rate as layout_dyn_rate
from .layout import gate_act as layout_gate_act

class _RefoldedMaps:

    def __init__(self, fold_args, fold_kw):
        self.args, self.kw = (fold_args, fold_kw)
        self.versions = [None if t is None else t._version for t in (*fold_args, *fold_kw.values()) if torch.is_tensor(t)]
        self.maps = None
        self.pending = {}

    def get(self, i):
        now = [t._version for t in (*self.args, *self.kw.values()) if torch.is_tensor(t)]
        if now != self.versions:
            raise RuntimeError('a folded-rotation input was modified in place between forward and backward')
        value = None if self.maps is None else self.maps[i]()
        if value is None:
            from .half_fold import half_fold
            with torch.no_grad():
                b, c, _ = half_fold(*(t.detach() for t in self.args), **{k: v.detach() if torch.is_tensor(v) else v for k, v in self.kw.items()})
            self.maps = (weakref.ref(b), weakref.ref(c))
            self.pending = {0: b, 1: c}
            value = self.pending[i]
        self.pending.pop(i, None)
        return value

def _refold_saved_maps(B_fold, C_fold, fold_args, fold_kw):
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
ROTATION_FALLBACKS: dict = {}
_ROT_SPLIT = True
_IN_SPLIT = __import__('contextvars').ContextVar('mh_in_rot_split', default=False)

def rotation_route() -> str:
    from .autotune import _opt
    v = (_opt('ROTATION') or 'fold').lower()
    if v not in ('fold', 'in_scan'):
        raise ValueError("hybrid_mamba rotation must be 'fold' or 'in_scan', got %r" % v)
    return v
FOLD_MIN_P = 16

def _fold_refusal(theta, n_off, n_rows, P=None, mem=None):
    if theta.dim() != 3:
        return 'per-head theta (B,H,K,L)'
    if n_off or n_rows is not None:
        return 'row-window launch (n_off/n_rows)'
    if P is not None and P < int(os.environ.get('MH_FOLD_MIN_P', FOLD_MIN_P)):
        return 'narrow heads (P < fold minimum)'
    if mem and mem.get('AGX') == 0 and (mem.get('BCS') == 0):
        return 'determinism pin (mem AGX=0, BCS=0)'
    return None
SLOW_ROTATION_P = 16
_WARNED_NARROW = False

def _warn_narrow_rotation(P):
    global _WARNED_NARROW
    if P < SLOW_ROTATION_P and (not _WARNED_NARROW):
        _WARNED_NARROW = True
        import warnings
        warnings.warn('hybrid_mamba: rotating (complex) rows with head width P=%d < %d run the unpacked in-scan schedule, roughly 1.5x the cost of the same launch without rotation; prefer P >= %d for complex rows.' % (P, SLOW_ROTATION_P, SLOW_ROTATION_P), RuntimeWarning, stacklevel=3)

def _rot_window(theta, row_rot, n_off, n_rows, write_prev, mu_init, return_state):
    if not _ROT_SPLIT or _IN_SPLIT.get() or theta is None or (row_rot is None) or n_off or (n_rows is not None) or (write_prev is not None) or (mu_init is not None) or return_state:
        return None
    rot = [bool(v) for v in selector_window(row_rot, 0, row_rot.numel())]
    n = len(rot)
    k = sum(rot)
    if k in (0, n) or k % 2:
        return None
    lead = all(rot[:k])
    if not lead and (not all(rot[n - k:])):
        return None
    return (0, k, n) if lead else (n - k, n, n)

def _split_rot_scan(window, kw):
    r0, r1, n = window
    k0, k1 = (r0 // 2, r1 // 2)
    a_n, mu_n = (kw['a_dyn_res'] == 'n', kw['mu_res'] == 'n')

    def rows(name, t):
        if t is None:
            return None
        if name in ('B_t', 'C_t', 'g_B', 'g_C') or (name == 'a_dyn' and a_n) or (name == 'mu' and mu_n):
            return t[:, r0:r1]
        if name in ('A_log', 'mu_gain', 'h_init'):
            return t[..., r0:r1]
        if name in ('beta_a', 'beta_b', 'beta_c', 'beta_gB', 'beta_gC'):
            return t[:, r0:r1]
        if name in ('row_law', 'row_rot'):
            return t[r0:r1]
        if name == 'theta':
            return t[:, k0:k1] if t.dim() == 3 else t[:, :, k0:k1]
        if name in ('theta_h', 'theta_init'):
            return t[..., k0:k1]
        return t
    skip = ('kernel_options', 'validate', 'return_state', 'n_off', 'n_rows', 'z')
    rot_kw = {name: rows(name, v) for name, v in kw.items() if name not in ('x', 'delta') + skip}
    real_kw = {name: v for name, v in kw.items() if name not in ('x', 'delta') + skip}
    real_kw.update(theta=None, row_rot=None, theta_h=None, theta_init=None, theta_delta=False, d_skip=None, n_off=r1 if r0 == 0 else 0, n_rows=n - (r1 - r0))
    common = dict(validate=kw['validate'], kernel_options=kw['kernel_options'])
    token = _IN_SPLIT.set(True)
    try:
        y = mh_scan(kw['x'], kw['delta'], **rot_kw, **common) + mh_scan(kw['x'], kw['delta'], **real_kw, **common)
    finally:
        _IN_SPLIT.reset(token)
    z = kw['z']
    if z is None:
        return y
    zf = z.float()
    return (y.float() * (zf * torch.sigmoid(zf))).to(y.dtype)
VALIDATE_ENV = 'HYBRID_SSM_MH_VALIDATE'

def _validate_default() -> bool:
    return os.environ.get(VALIDATE_ENV, '0').lower() not in ('', '0', 'false', 'no')
TENSOR_ARGS = ('x', 'delta', 'A_log', 'B_t', 'C_t', 'delta_w', 'a_dyn', 'mu', 'mu_gain', 'g_B', 'g_C', 'theta', 'beta_a', 'beta_b', 'beta_c', 'beta_gB', 'beta_gC', 'delta_bias', 'h_init', 'theta_init', 'mu_init', 'd_skip', 'row_law', 'row_rot', 'z', 'endpoint_own', 'theta_h')
NON_DIFF = frozenset({'row_law', 'row_rot'})
TWO_PI = 6.283185307179586

def _shared(beta):
    return beta is None or beta.shape[0] == 1

def _prefold_gate_ok(theta, beta_b, beta_c, g_B=None, g_C=None, beta_gB=None, beta_gC=None):
    if theta is not None:
        return False
    if not (_shared(beta_b) and _shared(beta_c)):
        return False
    return (g_B is not None or g_C is not None) and _shared(beta_gB) and _shared(beta_gC)
PMAP_ROUTE = 'all'

def _recomputed(fn, *args):
    if torch.is_grad_enabled() and any((torch.is_tensor(a) and a.requires_grad for a in args)):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
    return fn(*args)

def _pmap_ok(theta, x, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC):
    if PMAP_ROUTE == 'off' or theta is None or x.shape[2] <= 1 or (not all((_shared(t) for t in (beta_b, beta_c, beta_gB, beta_gC)))):
        return None
    if g_B is not None or g_C is not None:
        return 'gated'
    return 'raw' if PMAP_ROUTE == 'all' else None

def _pair_maps_raw(T, beta):
    if beta is None:
        return T
    return (T.float() + beta[0].reshape(1, T.shape[1], 1)).to(T.dtype)

def _pair_maps(T, beta, g, beta_g, gate_act):
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
    return theta is not None and (g_B is not None or g_C is not None) and (gate_act != GATE_RAW) and _shared(beta_gB) and _shared(beta_gC)

def _preact_gate(g, beta_g, gate_act):
    if g is None:
        return None
    gv = g.float()
    if beta_g is not None:
        gv = gv + beta_g[0].reshape(1, g.shape[1], 1)
    return layout_gate_act(gv, gate_act)

def _prefold_gate(B_t, C_t, beta_b, beta_c, g_B=None, g_C=None, beta_gB=None, beta_gC=None, gate_act=GATE_SIGMOID):
    N = B_t.shape[1]

    def fold(T, beta, g, bg):
        T = T.float()
        if beta is not None:
            T = T + beta[0].reshape(1, N, 1)
        if g is not None:
            gv = g.float()
            if bg is not None:
                gv = gv + bg[0].reshape(1, N, 1)
            T = T * layout_gate_act(gv, gate_act)
        return T
    return (fold(B_t, beta_b, g_B, beta_gB).to(B_t.dtype), fold(C_t, beta_c, g_C, beta_gC).to(C_t.dtype))

def _amp_fwd(fn):

    @functools.wraps(fn)
    def _inner(ctx, *args, **kwargs):
        with torch.amp.autocast('cuda', enabled=False):
            return fn(ctx, *args, **kwargs)
    return torch.amp.custom_fwd(_inner, device_type='cuda')

def _amp_bwd(fn):

    @functools.wraps(fn)
    def _inner(ctx, *args, **kwargs):
        with torch.amp.autocast('cuda', enabled=False):
            return fn(ctx, *args, **kwargs)
    return torch.amp.custom_bwd(_inner, device_type='cuda')

def _final_phase_vjp(seed, inputs, opts):
    theta, delta = (inputs['theta'], inputs['delta'])
    dtype = torch.float64 if theta.dtype == torch.float64 else torch.float32
    seed = seed.to(dtype)
    grads = {}
    if inputs.get('theta_init') is not None:
        grads['theta_init'] = seed
    lo = opts.get('n_off', 0) // 2
    rows = opts.get('n_rows')
    ph = theta.dim() == 4
    hi = theta.shape[-2] if rows is None else lo + rows // 2
    active = seed.clone()
    active[..., :lo] = 0
    active[..., hi:] = 0
    if opts.get('theta_delta', False):
        raw = delta.to(dtype)
        if inputs.get('delta_bias') is not None:
            raw = raw + inputs['delta_bias'].to(dtype)[None, :, None]
        softplus = opts.get('delta_softplus', True)
        clock = torch.nn.functional.softplus(raw) if softplus else raw
        if ph:
            grads['theta'] = active[..., None] * clock[:, :, None, :]
            clock_grad = torch.einsum('bhk,bhkl->bhl', active, theta.to(dtype))
        else:
            grads['theta'] = torch.bmm(active.transpose(1, 2), clock)
            clock_grad = torch.bmm(active, theta.to(dtype))
        if softplus:
            clock_grad = clock_grad * torch.where(raw > 20, 1.0, raw.sigmoid())
        grads['delta'] = clock_grad
        if inputs.get('delta_bias') is not None:
            grads['delta_bias'] = clock_grad.sum((0, 2))
    else:
        grads['theta'] = (active if ph else active.sum(1)).unsqueeze(-1).expand_as(theta)
    return grads

class _NoDoubleBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, g, *deps):
        return g.view_as(g)

    @staticmethod
    def backward(ctx, *_):
        raise RuntimeError("mh_scan: double backward is not supported -- the MH kernels' backward is not differentiable (a second derivative would silently miss the scan's part)")

class MHScan(torch.autograd.Function):

    @staticmethod
    @_amp_fwd
    def forward(ctx, opts, *tensors):
        ctx.set_materialize_grads(False)
        ckpt = opts.pop('ckpt', True)
        if not any(ctx.needs_input_grad[1:]):
            ckpt = False
        mem = opts.pop('mem', None)
        kw = {k: v for k, v in zip(TENSOR_ARGS, tensors) if v is not None}
        args = tuple((kw.pop(k) for k in TENSOR_ARGS[:5]))
        bl = auto_block_l(args[0].shape[2], args[0].shape[3]) if ckpt else 0
        Y, h_final, hck, theta_final = mh_forward(*args, **kw, **opts, ckpt_l=bl, ckpt_dt=torch.float32 if (mem or {}).get('CKBF') == 0 else None)
        ctx.save_for_backward(*tensors, hck)
        ctx._kernel_options = _ko.current('hybrid_mamba')
        ctx.opts = opts
        ctx.mem = mem
        return (Y, h_final, torch.empty(0, device=Y.device) if theta_final is None else theta_final)

    @staticmethod
    def backward(ctx, gy, gh_final, gtheta_final):
        with torch.no_grad():
            out = MHScan._backward(ctx, gy, gh_final, gtheta_final)
        if torch.is_grad_enabled():
            deps = [t for t in (*ctx.saved_tensors, gy, gh_final, gtheta_final) if torch.is_tensor(t) and t.requires_grad]
            if deps:
                out = tuple((None if o is None else _NoDoubleBackward.apply(o, *deps) for o in out))
        return out

    @staticmethod
    @_amp_bwd
    @_ko.scoped_ctx('hybrid_mamba')
    def _backward(ctx, gy, gh_final, gtheta_final):
        *tensors, hck = ctx.saved_tensors
        kw = {k: v for k, v in zip(TENSOR_ARGS, tensors) if v is not None}
        args = tuple((kw.pop(k) for k in TENSOR_ARGS[:5]))
        if gy is None:
            gy = torch.zeros_like(tensors[0])
        g = mh_backward(*args, gy.contiguous(), gh_final=None if gh_final is None else gh_final.contiguous(), hck=hck, mem=ctx.mem, **kw, **ctx.opts)
        g = {name: g[name] for name in TENSOR_ARGS if name in g}
        if gtheta_final is not None and gtheta_final.numel():
            phase_inputs = dict(zip(TENSOR_ARGS, tensors))
            phase_grads = _final_phase_vjp(gtheta_final, phase_inputs, ctx.opts)
            for name, value in phase_grads.items():
                g[name] = g[name] + value if name in g else value
        out = []
        for i, name in enumerate(TENSOR_ARGS):
            t = tensors[i]
            if t is None or name in NON_DIFF or (not ctx.needs_input_grad[i + 1]):
                out.append(None)
            else:
                out.append(g[name].reshape(t.shape).to(t.dtype))
        return (None, *out)

@dynamo_opaque
@_ko.scoped_kwarg('hybrid_mamba')
def mh_scan(x, delta, A_log, B_t, C_t, *, delta_w=None, a_dyn=None, mu=None, mu_gain=None, g_B=None, g_C=None, theta=None, beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None, delta_bias=None, h_init=None, theta_init=None, mu_init=None, d_skip=None, row_law=None, row_rot=None, z=None, delta_softplus=True, a_dyn_res='n', mu_res='n', dyn_act=DYN_SOFTPLUS, dyn_eps=0.0, gate_act=GATE_SIGMOID, n_off=0, n_rows=None, return_state=False, ckpt=True, mem=None, validate=None, trap_lam=None, write_prev=None, theta_delta=False, theta_h=None, kernel_options=None):
    call = dict(locals())
    if g_B is None and g_C is None:
        gate_act = GATE_RAW
    elif gate_act not in _KERNEL_GATE_ACTS:
        raise ValueError(f"mh_scan: gate_act={gate_act} has no kernel implementation for in-scan gates g_B/g_C (kernel codes {sorted(_KERNEL_GATE_ACTS)}); fold it into B/C on the host (HybridMambaSSM gate_pos='pre')")
    window = _rot_window(theta, row_rot, n_off, n_rows, write_prev, mu_init, return_state)
    if window:
        return _split_rot_scan(window, call)
    del call, window
    if write_prev is not None and trap_lam is None:
        raise ValueError('write_prev requires trap_lam')
    x, B_t, C_t, z = (autocast_io(x), autocast_io(B_t), autocast_io(C_t), autocast_io(z))
    if theta_delta and theta is None:
        raise ValueError('theta_delta requires theta')
    if theta is not None:
        K_, L_ = (B_t.shape[1] // 2, x.shape[-1])
        if tuple(theta.shape) not in ((x.shape[0], K_, L_), (x.shape[0], x.shape[1], K_, L_)):
            raise ValueError('theta must be (B, K, L) (head-shared angle feature) or (B, H, K, L) (theta_per_head), K = N // 2')
    if theta_h is not None:
        if theta is None:
            raise ValueError('theta_h requires theta (a rotating launch)')
        if tuple(theta_h.shape) != (x.shape[1], B_t.shape[1] // 2):
            raise ValueError('theta_h must be (H, K) with K = N // 2')
    if theta_init is not None and theta is None:
        raise ValueError("theta_init is the rotation law's chunk carry and needs theta: a lone theta_init never turned the kernel's rotation on, so it was silently ignored (retired 2026-09-15)")
    theta_delta = bool(theta_delta) and theta is not None
    endpoint_own = None
    if trap_lam is not None:
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
        delta_bias, delta_softplus = (None, False)
        from .endpoint import mh_endpoint_weights
        weight = trap_lam.float()
        if clock.ndim == 4:
            weight = weight[:, :, None, :].expand_as(clock)
        delta_w, endpoint_own = mh_endpoint_weights(clock, weight)
        if write_prev is not None:
            if write_prev.shape != (x.shape[0], x.shape[1], x.shape[2], B_t.shape[1]):
                raise ValueError('write_prev must have state shape B,H,P,N')
            first = clock[..., 0] * (1 - weight[..., 0])
            if clock.ndim == 3:
                first = first[..., None]
            lag = first[..., None] * write_prev
            if n_off or n_rows is not None:
                end = B_t.shape[1] if n_rows is None else n_off + n_rows
                lag = torch.nn.functional.pad(lag[..., n_off:end], (n_off, B_t.shape[1] - end))
            h_init = lag if h_init is None else h_init + lag
    if z is not None:
        if n_off or n_rows is not None:
            raise NotImplementedError('mh_scan(z=...) is the WHOLE-N call: an n_off/n_rows shard computes one summand of the readout and the output gate multiplies the whole of it')
        if z.shape != x.shape:
            raise ValueError(f'mh_scan(z=...) must be CHANNEL-major, the same (B, H, P, L) shape as x -- got {tuple(z.shape)} against {tuple(x.shape)}')
        if z.dtype != x.dtype:
            raise ValueError(f"mh_scan(z=...) must match x's dtype ({x.dtype}); got {z.dtype}")
        if not z.is_contiguous():
            z = z.contiguous()
    if d_skip is not None and (n_off or n_rows is not None):
        raise NotImplementedError('mh_scan(d_skip=...) is the WHOLE-N call: an n_off/n_rows shard computes one summand of the readout and the skip must be added exactly once')
    if _validate_default() if validate is None else validate:
        assert_pair_tied(A_log, row_rot, beta_a=beta_a, mu_gain=mu_gain, row_law=row_law, a_dyn=a_dyn if a_dyn_res == 'n' else None, mu=mu if mu_res == 'n' else None, mu_init=mu_init if mu_res == 'n' else None)
    if a_dyn is not None and dyn_act != DYN_RAW:
        a_dyn = layout_dyn_rate(a_dyn.float(), dyn_act, dyn_eps)
        dyn_act, dyn_eps = (DYN_RAW, 0.0)
    mu_prediff = mu is not None
    if mu_prediff:
        mu = mu.float()
        prev = torch.zeros_like(mu[..., :1]) if mu_init is None else mu_init.reshape(*mu.shape[:-1], 1).float()
        mu = mu - torch.cat([prev, mu[..., :-1]], dim=-1)
        mu_init = None
    opts = dict(delta_softplus=delta_softplus, a_dyn_res=a_dyn_res, mu_res=mu_res, dyn_act=dyn_act, dyn_eps=dyn_eps, gate_act=gate_act, mu_prediff=mu_prediff, n_off=n_off, n_rows=n_rows, ckpt=bool(ckpt) and torch.is_grad_enabled(), mem=mem, theta_delta=theta_delta)
    route = rotation_route() if theta is not None else 'in_scan'
    if route == 'fold':
        why = _fold_refusal(theta, n_off, n_rows, P=x.shape[2], mem=mem)
        if why:
            ROTATION_FALLBACKS[why] = ROTATION_FALLBACKS.get(why, 0) + 1
            route = 'in_scan'
            _warn_narrow_rotation(x.shape[2])
    if route == 'fold':
        from .half_fold import half_fold
        fold_args = (B_t, C_t, delta, theta)
        fold_kw = dict(theta_h=theta_h, theta_init=theta_init, beta_b=beta_b, beta_c=beta_c, g_B=g_B, g_C=g_C, beta_gB=beta_gB, beta_gC=beta_gC, delta_bias=delta_bias, row_rot=row_rot, theta_delta=theta_delta, delta_softplus=delta_softplus, gate_act=gate_act)
        B_t, C_t, final_phase = half_fold(*fold_args, **fold_kw)
        opts.update(bc_head=True, theta_delta=False)
        with _refold_saved_maps(B_t, C_t, fold_args, fold_kw):
            y, last, _ = MHScan.apply(opts, x, delta, A_log, B_t, C_t, delta_w, a_dyn, mu, mu_gain, None, None, None, beta_a, None, None, None, None, delta_bias, h_init, None, mu_init, d_skip, row_law, None, z, endpoint_own, None)
        if not return_state:
            return y
        return (y, last, final_phase)

    def common_groups(left, right):
        if left is None or right is None or left.shape[0] == right.shape[0]:
            return (left, right)
        import math
        groups = math.lcm(left.shape[0], right.shape[0])
        if not left.shape[0] or not right.shape[0] or x.shape[1] % groups:
            raise ValueError('B/C bias group counts must divide H')
        return (left.float().repeat_interleave(groups // left.shape[0], dim=0), right.float().repeat_interleave(groups // right.shape[0], dim=0))
    beta_b, beta_c = common_groups(beta_b, beta_c)
    beta_gB, beta_gC = common_groups(beta_gB, beta_gC)
    if _prefold_gate_ok(theta, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC):
        B_t, C_t = _recomputed(_prefold_gate, B_t, C_t, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC, gate_act)
        beta_b = beta_c = g_B = g_C = beta_gB = beta_gC = None
    elif _pmap_ok(theta, x, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC) == 'raw':
        if beta_b is not None or beta_c is not None:
            B_t = _recomputed(_pair_maps_raw, B_t, beta_b)
            C_t = _recomputed(_pair_maps_raw, C_t, beta_c)
        beta_b = beta_c = None
        opts['pmap'] = 'raw'
    elif _pmap_ok(theta, x, beta_b, beta_c, g_B, g_C, beta_gB, beta_gC):
        B_t = _recomputed(_pair_maps, B_t, beta_b, g_B, beta_gB, gate_act)
        C_t = _recomputed(_pair_maps, C_t, beta_c, g_C, beta_gC, gate_act)
        beta_b = beta_c = g_B = g_C = beta_gB = beta_gC = None
        opts['pmap'] = True
    elif _preact_gate_ok(theta, g_B, g_C, beta_gB, beta_gC, gate_act):
        g_B, g_C = (_preact_gate(g_B, beta_gB, gate_act), _preact_gate(g_C, beta_gC, gate_act))
        beta_gB = beta_gC = None
        gate_act = GATE_RAW
        opts['gate_act'] = GATE_RAW
    Y, h_final, theta_final = MHScan.apply(opts, x, delta, A_log, B_t, C_t, delta_w, a_dyn, mu, mu_gain, g_B, g_C, theta, beta_a, beta_b, beta_c, beta_gB, beta_gC, delta_bias, h_init, theta_init, mu_init, d_skip, row_law, row_rot, z, endpoint_own, theta_h)
    if not return_state:
        return Y
    return (Y, h_final, None if theta_final.numel() == 0 else theta_final)
