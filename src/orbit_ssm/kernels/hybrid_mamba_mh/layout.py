
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
LAW_A_FIX, LAW_A_DYN, LAW_MU = (0, 1, 2)
LAW_NAMES = {'A': LAW_A_FIX, 'a_fix': LAW_A_FIX, 'fix': LAW_A_FIX, 'dyn': LAW_A_DYN, 'a_dyn': LAW_A_DYN, 'mu': LAW_MU}
ROUTED_TO_SSD = {'const': 'm3', 'm3': 'm3', 'm2': 'm2'}

def refuse_per_head_scalar_decay(name) -> None:
    law = ROUTED_TO_SSD.get(name if isinstance(name, str) else '')
    if law is None:
        return
    raise ValueError(f"decay {name!r} is a per-head SCALAR decay (constant along the state row n and the channel p): that is SSD structure and runs through ssd_mimo_scan at R = O = 1, not mh_scan. Use rows=[{{'decay': ..., ...}}] with a flat law ('head_dynamic' / 'head_static') on HybridMambaSSM, which dispatches those rows to the SSD kernel (implementation.md Sec.1.2, routing rule)")
DYN_RAW, DYN_SOFTPLUS, DYN_EXP = (0, 1, 2)

def dyn_rate(a_dyn, act, eps):
    if act == DYN_RAW:
        r = a_dyn
    elif act == DYN_EXP:
        r = torch.exp(a_dyn)
    elif act == DYN_SOFTPLUS:
        r = torch.nn.functional.softplus(a_dyn)
    else:
        raise ValueError(f'unknown dyn act {act}')
    return r + eps
GATE_RAW, GATE_SIGMOID, GATE_EXP, GATE_TANH, GATE_SOFTPLUS = (0, 1, 2, 3, 4)
GATE_ONE_PLUS = 5
GATE_ONE_PLUS_TANH = 6
GATE_NAMES = {'raw': GATE_RAW, 'identity': GATE_RAW, 'sigmoid': GATE_SIGMOID, 'exp': GATE_EXP, 'tanh': GATE_TANH, 'softplus': GATE_SOFTPLUS, 'one_plus': GATE_ONE_PLUS, 'one_plus_tanh': GATE_ONE_PLUS_TANH}

class _SigmoidExact(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return torch.sigmoid(v)

    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors
        e = torch.exp(-v.abs())
        return g * (e / ((1 + e) * (1 + e)))

class _TanhExact(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return torch.tanh(v)

    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors
        e = torch.exp(-2 * v.abs())
        return g * (4 * e / ((1 + e) * (1 + e)))

def sigmoid_exact(v):
    return _SigmoidExact.apply(v)

def gate_act(v, act):
    if act == GATE_RAW:
        return v
    if act == GATE_SIGMOID:
        return _SigmoidExact.apply(v)
    if act == GATE_EXP:
        return torch.exp(v)
    if act == GATE_TANH:
        return _TanhExact.apply(v)
    if act == GATE_SOFTPLUS:
        return torch.nn.functional.softplus(v)
    if act == GATE_ONE_PLUS:
        return 1 + v
    if act == GATE_ONE_PLUS_TANH:
        return 1 + _TanhExact.apply(v)
    raise ValueError(f'unknown gate act {act}')

@dataclass(frozen=True)
class Shapes:
    B: int
    H: int
    P: int
    N: int
    L: int

    @property
    def D(self) -> int:
        return self.H * self.P

    @property
    def K(self) -> int:
        return self.N // 2

def d_index(h: int, p: int, P: int) -> int:
    return h * P + p

def group_of(h: int, H: int, G: int) -> int:
    return h * G // H

def group_index(H: int, G: int, device=None) -> torch.Tensor:
    if H % G != 0:
        raise ValueError(f'group count G={G} must divide H={H} (contract Sec.1)')
    return torch.arange(H, device=device) * G // H

@dataclass(frozen=True)
class Frac:
    rows: int
    law: int
    rot: bool

def norm_fracs(fracs, N: int) -> list[Frac]:
    if fracs is None:
        return [Frac(rows=N, law=LAW_A_FIX, rot=False)]
    out: list[Frac] = []
    for i, spec in enumerate(fracs):
        if 'rows' in spec:
            rows = int(spec['rows'])
        elif 'frac' in spec:
            rows = int(round(float(spec['frac']) * N))
        else:
            raise ValueError(f"fracs[{i}]: needs 'rows' or 'frac'")
        decay = spec.get('decay', 'A')
        if decay in ('external', 'ext', 4):
            raise ValueError('external MH laws are retired; use an in-kernel decay law')
        refuse_per_head_scalar_decay(decay)
        law = LAW_NAMES[decay] if isinstance(decay, str) else int(decay)
        if law not in (LAW_A_FIX, LAW_A_DYN, LAW_MU):
            raise ValueError(f'unsupported MH law {law}')
        out.append(Frac(rows=rows, law=law, rot=bool(spec.get('rot', False))))
    total = sum((f.rows for f in out))
    if total != N:
        raise ValueError(f'fracs row counts sum to {total}, need exactly N={N}')
    off = 0
    for i, f in enumerate(out):
        if f.rot:
            if off % 2 or f.rows % 2:
                raise ValueError(f'fracs[{i}]: rotating block needs even offset and even rows (got off={off}, rows={f.rows}); contract Sec.3.4/Sec.4')
        off += f.rows
    return out

def fracs_to_rows(fracs: Sequence[Frac], N: int, device=None):
    row_law = torch.empty(N, dtype=torch.int32, device=device)
    row_rot = torch.zeros(N, dtype=torch.bool, device=device)
    off = 0
    for f in fracs:
        row_law[off:off + f.rows] = f.law
        row_rot[off:off + f.rows] = f.rot
        off += f.rows
    return (row_law, row_rot)

def law_mask(fracs: Sequence[Frac]) -> int:
    m = 0
    for f in fracs:
        m |= 1 << f.law
    return m

def _check(cond, msg):
    if not cond:
        raise ValueError(msg)

def validate_forward_inputs(x, delta, A_log, B_t, C_t, *, delta_w=None, a_dyn=None, mu=None, mu_gain=None, g_B=None, g_C=None, theta=None, nu=None, beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None, delta_bias=None, h_init=None, theta_init=None, mu_init=None, n_off=0, n_rows=None) -> Shapes:
    _check(x.dim() == 4, f'x must be (B,H,P,L), got {tuple(x.shape)}')
    Bs, H, P, L = x.shape
    _check(A_log.dim() == 3, f'A_log must be (H,P,N), got {tuple(A_log.shape)}')
    _check(A_log.shape[:2] == (H, P), f"A_log head/channel dims {tuple(A_log.shape[:2])} != x's ({H},{P})")
    N = A_log.shape[2]
    s = Shapes(B=Bs, H=H, P=P, N=N, L=L)
    _check(delta.shape == (Bs, H, L), f'delta must be (B,H,L)={(Bs, H, L)} -- p-varying delta would make a_bar p-dependent through a streaming tensor (contract Sec.2 (I), Sec.6.2); got {tuple(delta.shape)}')
    for name, t in (('B_t', B_t), ('C_t', C_t)):
        _check(t.shape == (Bs, N, L), f'{name} must be (B,N,L)={(Bs, N, L)}, got {tuple(t.shape)} -- H-resolution goes in the static bias, not the tensor (contract Sec.2 (III))')
    if delta_w is not None:
        _check(delta_w.shape in ((Bs, H, L), (Bs, H, P, L)), f'delta_w must be (B,H,L) or (B,H,P,L); got {tuple(delta_w.shape)}')
    for name, t in (('g_B', g_B), ('g_C', g_C)):
        if t is not None:
            _check(t.shape == (Bs, N, L), f'{name} must be (B,N,L); H-variation enters as beta_{name} (contract Sec.7). got {tuple(t.shape)}')
    for name, t in (('a_dyn', a_dyn), ('mu', mu)):
        if t is not None:
            _check(t.shape in ((Bs, N, L), (Bs, H, L)), f'{name} must be (B,N,L) XOR (B,H,L), never (B,H,N,L) (contract Sec.3.3); got {tuple(t.shape)}')
    for name, t in (('mu_gain', mu_gain),):
        if t is not None:
            _check(t.shape == (H, P, N), f'{name} must be (H,P,N)={(H, P, N)}')
    if theta is not None:
        _check(N % 2 == 0, 'rotation needs even N')
        _check(theta.shape in ((Bs, N // 2, L), (Bs, H, N // 2, L)), f'theta must be (B,K,L)={(Bs, N // 2, L)} or (B,H,K,L) (theta_per_head); got {tuple(theta.shape)}')
    if nu is not None:
        _check(nu.shape == (H, N // 2), f'nu must be (H,K)={(H, N // 2)} -- p-resolved nu destroys the fold and restores the trig cliff (contract Sec.2, Sec.15)')
    for name, t in (('beta_a', beta_a), ('beta_b', beta_b), ('beta_c', beta_c), ('beta_gB', beta_gB), ('beta_gC', beta_gC)):
        if t is not None:
            _check(t.dim() == 2 and t.shape[1] == N, f'{name} must be (G,N) with G|H; got {tuple(t.shape)}')
            _check(H % t.shape[0] == 0, f'{name} group count {t.shape[0]} must divide H={H}')
    if delta_bias is not None:
        _check(delta_bias.shape == (H,), f'delta_bias must be (H,) -- an (H,P) bias on the DECAY step would make a_bar p-dependent (contract Sec.2 (I)); use delta_w for per-channel write strength (Sec.6.2). got {tuple(delta_bias.shape)}')
    if h_init is not None:
        _check(h_init.shape == (Bs, H, P, N), f'h_init must be (B,H,P,N); got {tuple(h_init.shape)}')
    if theta_init is not None:
        _check(theta_init.shape == (Bs, H, N // 2), f'theta_init must be (B,H,K); got {tuple(theta_init.shape)}')
    if mu_init is not None:
        _check(mu_init.shape in ((Bs, N), (Bs, H)), f"mu_init must match mu's resolution; got {tuple(mu_init.shape)}")
    nr = N - n_off if n_rows is None else n_rows
    _check(0 <= n_off and nr > 0 and (n_off + nr <= N), f'row slice [{n_off},{n_off + nr}) out of range for N={N}')
    return s
MODULUS_FACTORS = (('A_log', -1), ('beta_a', -1), ('mu_gain', -1), ('a_dyn', 1), ('mu', 1), ('mu_init', 1), ('row_law', -1))

def _pair_untied(t, axis: int, pair_rot: torch.Tensor) -> bool:
    if t is None:
        return False
    v = t.movedim(axis, -1)
    if v.shape[-1] != pair_rot.numel() * 2:
        return False
    d = v[..., 0::2] != v[..., 1::2]
    return bool(d[..., pair_rot.to(d.device)].any())

def assert_pair_tied(A_log: torch.Tensor, row_rot: torch.Tensor, *, beta_a=None, mu_gain=None, a_dyn=None, mu=None, mu_init=None, row_law=None) -> None:
    N = A_log.shape[-1]
    if N % 2 or row_rot is None:
        return
    pair_rot = row_rot[0::2] & row_rot[1::2]
    if not pair_rot.any():
        return
    given = dict(A_log=A_log, beta_a=beta_a, mu_gain=mu_gain, a_dyn=a_dyn, mu=mu, mu_init=mu_init, row_law=row_law)
    bad = [name for name, axis in MODULUS_FACTORS if _pair_untied(given[name], axis, pair_rot)]
    if bad:
        raise ValueError(f"untied modulus on a rotating pair: {', '.join(bad)} differ between coordinates 2k and 2k+1 (contract Sec.3.4). Every factor of the pair's decay must satisfy t[..., 2k] == t[..., 2k+1] on rotating rows, because the rotation is folded onto B/C and diag(a_even, a_odd) commutes with R(Theta) only when the two are equal. Store as (H,P,K) and expand, or gather the odd coordinate from its even partner.")
