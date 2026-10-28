"""Index algebra and ABI validation for the HybridMamba-MH kernel (structure B).

    D = H * P,   d = h*P + p     (head-major, PINNED by contract)

The axis invariants this module enforces:

    (I)   no streaming tensor may carry a P axis   (except delta_w)
    (II)  static tensors may carry P iff they enter the state elementwise
          (A_log, mu_gain); write/read statics stay rank-1 in (p,n)
    (III) a streaming tensor may carry H only if it carries no N/K axis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..input_checks import check_cuda_input

# --- decay laws -----------------------------------------
LAW_A_FIX, LAW_A_DYN, LAW_MU = 0, 1, 2

LAW_NAMES = {
    "A": LAW_A_FIX, "a_fix": LAW_A_FIX, "fix": LAW_A_FIX,
    "dyn": LAW_A_DYN, "a_dyn": LAW_A_DYN,
    "mu": LAW_MU,
}

# The kernel's own names above are internal (and part of the calibration code hash); the decay-law names
# of the config surface (`config_spec.HM_PER_COORD`, the paper's "HM static / dynamic / potential") are
# looked up here, so a fracs spec may carry either.  Lower case on purpose: not an ALL_CAPS constant.
_coord_law_names = {"coord_static": LAW_A_FIX, "coord_dynamic": LAW_A_DYN, "coord_potential": LAW_MU}


def law_of(decay) -> int:
    """The kernel law code of a fracs ``decay`` (a config-surface name, a kernel name, or the int code)."""
    if not isinstance(decay, str):
        return int(decay)
    if decay in _coord_law_names:
        return _coord_law_names[decay]
    return LAW_NAMES[decay]

#: ROUTING RULE.  A row block whose decay is a
#: SCALAR PER HEAD -- constant along the state row `n` and the channel `p`
#: -- is SSD structure, not this family's, and must run through
#: `ssd_mimo_scan` at R = O = 1.  `mh_scan`'s axis is the per-CHANNEL decay
#: `(H, P, N)`; a per-head scalar on it is `H*P*N` copies of one number and
#: a second implementation of a law the SSD kernel already owns.  These
#: names are refused by name, here and at `mh_scan`'s entry point.
ROUTED_TO_SSD = {
    "const": "m3", "m3": "m3", "m2": "m2",
}


def refuse_per_head_scalar_decay(name) -> None:
    """Raise the routing-rule refusal for a per-head-scalar decay name."""
    law = ROUTED_TO_SSD.get(name if isinstance(name, str) else "")
    if law is None:
        return
    raise ValueError(
        f"decay {name!r} is a per-head SCALAR decay (constant along the "
        f"state row n and the channel p): that is SSD structure and runs "
        f"through ssd_mimo_scan at R = O = 1, not mh_scan. Use "
        f"rows=[{{'decay': ..., ...}}] with a flat law ('head_dynamic' / "
        f"'head_static') on HybridMambaSSM, which "
        f"dispatches those rows to the SSD kernel.")

# --- A_dyn rate activation -------------
DYN_RAW, DYN_SOFTPLUS, DYN_EXP = 0, 1, 2


def dyn_rate(a_dyn, act, eps):
    """The `rate = act(a_dyn) + eps`, in torch.

    Bound by construction, never clamped: a clamp that binds has zero gradient
    there. Lives here rather than in reference.py because functions.py
    pre-applies it on the host and must not import the fp64 oracle to do it.

    Note what is NOT an argument: `beta_a` rides on the STATIC side of the
    separable law (reference.py:142), so the rate depends on `a_dyn` alone and
    is head-redundant unconditionally -- unlike the gate, which needs its bias
    to be head-shared before it can be hoisted.
    """
    if act == DYN_RAW:
        r = a_dyn
    elif act == DYN_EXP:
        r = torch.exp(a_dyn)
    elif act == DYN_SOFTPLUS:
        r = torch.nn.functional.softplus(a_dyn)
    else:
        raise ValueError(f"unknown dyn act {act}")
    return r + eps


# --- gate activation --------------------------------------
GATE_RAW, GATE_SIGMOID, GATE_EXP, GATE_TANH, GATE_SOFTPLUS = 0, 1, 2, 3, 4
#: SSD's gate form, 1 + g
GATE_ONE_PLUS = 5
#: 1 + tanh(g): identity at g = 0, bounded in (0, 2), can switch a row off.
#: HOST-SIDE ONLY (gate_pos="pre" fold); no kernel implements it, so it is never a GATE_ACT constexpr
GATE_ONE_PLUS_TANH = 6

GATE_NAMES = {
    "raw": GATE_RAW, "identity": GATE_RAW,
    "sigmoid": GATE_SIGMOID,
    "exp": GATE_EXP,
    "tanh": GATE_TANH,
    "softplus": GATE_SOFTPLUS,
    "one_plus": GATE_ONE_PLUS,
    "one_plus_tanh": GATE_ONE_PLUS_TANH,
}


class _SigmoidExact(torch.autograd.Function):
    """sigmoid with its derivative from the pre-activation, e / (1 + e)^2 at
    e = exp(-|v|).  torch's own backward is y (1 - y) from the rounded output,
    which cancels where the gate saturates (relative error u e^v: 1e-3 at v = 10
    in fp32, total past v = 17) -- a trained gate/lambda sits there."""
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
    """tanh with 4 e / (1 + e)^2, e = exp(-2|v|), instead of 1 - y^2."""
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
    """`torch.sigmoid` whose backward stays relatively exact at saturation."""
    return _SigmoidExact.apply(v)


def gate_act(v, act):
    """The gate activation, in torch. Lives here rather than in
    reference.py because functions.py folds the gate on the host for a
    head-shared problem and must not import the fp64 oracle to do it.
    sigmoid/tanh differentiate from v (`_SigmoidExact`), like the kernels'
    `_gate_dv`."""
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
        # derivative 1 - tanh^2 taken from v (`_TanhExact`), not from the rounded output
        return 1 + _TanhExact.apply(v)
    raise ValueError(f"unknown gate act {act}")

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
    """The pinned flattening. Getting this backwards trains but is wrong."""
    return h * P + p


def group_of(h: int, H: int, G: int) -> int:
    """Contiguous head grouping g(h) = h*G/H. G=1 shared, G=H per-head."""
    return (h * G) // H


def group_index(H: int, G: int, device=None) -> torch.Tensor:
    if H % G != 0:
        raise ValueError(f"group count G={G} must divide H={H}")
    return (torch.arange(H, device=device) * G) // H


# ---------------------------------------------------------------------------
# fracs -- row composition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frac:
    rows: int
    law: int
    rot: bool


def norm_fracs(fracs, N: int) -> list[Frac]:
    """Normalise a fracs spec into explicit row counts.

    fracs=[{"rows": 32, "decay": "coord_static"},
           {"frac": 0.25, "decay": "coord_potential", "rot": True}]

    Rules: row counts sum to exactly N; rotating blocks have
    even rows AND even offset (pair alignment).
    """
    if fracs is None:
        return [Frac(rows=N, law=LAW_A_FIX, rot=False)]

    out: list[Frac] = []
    for i, spec in enumerate(fracs):
        if "rows" in spec:
            rows = int(spec["rows"])
        elif "frac" in spec:
            rows = int(round(float(spec["frac"]) * N))
        else:
            raise ValueError(f"fracs[{i}]: needs 'rows' or 'frac'")
        decay = spec.get("decay", "coord_static")
        if decay in ("external", "ext", 4):
            raise ValueError("external MH decay laws are not served; use an in-kernel decay law")
        refuse_per_head_scalar_decay(decay)
        law = law_of(decay)
        if law not in (LAW_A_FIX, LAW_A_DYN, LAW_MU):
            raise ValueError(f"unsupported MH law {law}")
        out.append(Frac(rows=rows, law=law, rot=bool(spec.get("rot", False))))

    total = sum(f.rows for f in out)
    if total != N:
        raise ValueError(f"fracs row counts sum to {total}, need exactly N={N}")

    off = 0
    for i, f in enumerate(out):
        if f.rot:
            if off % 2 or f.rows % 2:
                raise ValueError(
                    f"fracs[{i}]: rotating block needs even offset and even rows "
                    f"(got off={off}, rows={f.rows})"
                )
        off += f.rows
    return out


def fracs_to_rows(fracs: Sequence[Frac], N: int, device=None):
    """Expand fracs into per-row (law, rot) byte vectors.

    These are what the fused route consumes: the separable log-decay form makes a per-row law byte cost only two `where`s on
    p-free tiles, which is why the fused route is the preferred one.
    """
    row_law = torch.empty(N, dtype=torch.int32, device=device)
    row_rot = torch.zeros(N, dtype=torch.bool, device=device)
    off = 0
    for f in fracs:
        row_law[off:off + f.rows] = f.law
        row_rot[off:off + f.rows] = f.rot
        off += f.rows
    return row_law, row_rot


def law_mask(fracs: Sequence[Frac]) -> int:
    """Bitmask of laws present -- drives the compile-time gating of the decay laws."""
    m = 0
    for f in fracs:
        m |= 1 << f.law
    return m


# ---------------------------------------------------------------------------
# ABI validation
# ---------------------------------------------------------------------------

def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def validate_forward_inputs(
    x, delta, A_log, B_t, C_t, *,
    delta_w=None, a_dyn=None, mu=None, mu_gain=None,
    g_B=None, g_C=None, theta=None, nu=None,
    beta_a=None, beta_b=None, beta_c=None, beta_gB=None, beta_gC=None,
    delta_bias=None, h_init=None, theta_init=None, mu_init=None,
    n_off=0, n_rows=None,
) -> Shapes:
    """Validate the inputs of the CUDA (Gluon) forward launch (`fwd_gluon`, its one
    caller) and return their `Shapes`.  CUDA-only: every tensor must sit
    on x's CUDA device (named ValueError pointing at backend='torch'), then the
    contract's axis invariants are checked."""
    _check(x.dim() == 4, f"x must be (B,H,P,L), got {tuple(x.shape)}")
    # the CUDA (Gluon) launch: every tensor input on x's CUDA device
    check_cuda_input("mh_scan (CUDA route)", dict(
        x=x, delta=delta, A_log=A_log, B_t=B_t, C_t=C_t, delta_w=delta_w,
        a_dyn=a_dyn, mu=mu, mu_gain=mu_gain, g_B=g_B, g_C=g_C, theta=theta,
        nu=nu, beta_a=beta_a, beta_b=beta_b, beta_c=beta_c, beta_gB=beta_gB,
        beta_gC=beta_gC, delta_bias=delta_bias, h_init=h_init,
        theta_init=theta_init, mu_init=mu_init))
    Bs, H, P, L = x.shape
    _check(A_log.dim() == 3, f"A_log must be (H,P,N), got {tuple(A_log.shape)}")
    _check(A_log.shape[:2] == (H, P),
           f"A_log head/channel dims {tuple(A_log.shape[:2])} != x's ({H},{P})")
    N = A_log.shape[2]
    s = Shapes(B=Bs, H=H, P=P, N=N, L=L)

    _check(delta.shape == (Bs, H, L),
           f"delta must be (B,H,L)={(Bs, H, L)} -- p-varying delta would make "
           f"a_bar p-dependent through a streaming tensor; "
           f"got {tuple(delta.shape)}")
    for name, t in (("B_t", B_t), ("C_t", C_t)):
        _check(t.shape == (Bs, N, L),
               f"{name} must be (B,N,L)={(Bs, N, L)}, got {tuple(t.shape)} -- "
               f"H-resolution goes in the static bias, not the tensor")

    if delta_w is not None:
        _check(delta_w.shape in ((Bs, H, L), (Bs, H, P, L)),
               f"delta_w must be (B,H,L) or (B,H,P,L); got {tuple(delta_w.shape)}")
    for name, t in (("g_B", g_B), ("g_C", g_C)):
        if t is not None:
            _check(t.shape == (Bs, N, L),
                   f"{name} must be (B,N,L); H-variation enters as beta_{name}, "
                   f"got {tuple(t.shape)}")
    for name, t in (("a_dyn", a_dyn), ("mu", mu)):
        if t is not None:
            _check(t.shape in ((Bs, N, L), (Bs, H, L)),
                   f"{name} must be (B,N,L) XOR (B,H,L), never (B,H,N,L); "
                   f"got {tuple(t.shape)}")
    for name, t in (("mu_gain", mu_gain),):
        if t is not None:
            _check(t.shape == (H, P, N), f"{name} must be (H,P,N)={(H, P, N)}")
    if theta is not None:
        _check(N % 2 == 0, "rotation needs even N")
        _check(theta.shape in ((Bs, N // 2, L), (Bs, H, N // 2, L)),
               f"theta must be (B,K,L)={(Bs, N // 2, L)} or (B,H,K,L) "
               f"(theta_per_head); got {tuple(theta.shape)}")
    if nu is not None:
        _check(nu.shape == (H, N // 2),
               f"nu must be (H,K)={(H, N // 2)} -- p-resolved nu destroys the "
               f"fold and restores the trig cliff")
    for name, t in (("beta_a", beta_a), ("beta_b", beta_b), ("beta_c", beta_c),
                    ("beta_gB", beta_gB), ("beta_gC", beta_gC)):
        if t is not None:
            _check(t.dim() == 2 and t.shape[1] == N,
                   f"{name} must be (G,N) with G|H; got {tuple(t.shape)}")
            _check(H % t.shape[0] == 0,
                   f"{name} group count {t.shape[0]} must divide H={H}")
    if delta_bias is not None:
        _check(delta_bias.shape == (H,),
               f"delta_bias must be (H,) -- an (H,P) bias on the DECAY step "
               f"would make a_bar p-dependent; use "
               f"delta_w for per-channel write strength. "
               f"got {tuple(delta_bias.shape)}")
    if h_init is not None:
        _check(h_init.shape == (Bs, H, P, N),
               f"h_init must be (B,H,P,N); got {tuple(h_init.shape)}")
    if theta_init is not None:
        _check(theta_init.shape == (Bs, H, N // 2),
               f"theta_init must be (B,H,K); got {tuple(theta_init.shape)}")
    if mu_init is not None:
        _check(mu_init.shape in ((Bs, N), (Bs, H)),
               f"mu_init must match mu's resolution; got {tuple(mu_init.shape)}")
    nr = (N - n_off) if n_rows is None else n_rows
    _check(0 <= n_off and nr > 0 and n_off + nr <= N,
           f"row slice [{n_off},{n_off + nr}) out of range for N={N}")
    return s


#: Every per-coordinate factor of the pair modulus, as (name, N-axis) pairs.
#: `a_dyn`/`mu` appear only at n-resolution -- at h-resolution they carry no
#: N axis at all and cannot untie a pair.
MODULUS_FACTORS = (
    ("A_log", -1),          # c1 = -exp(A_log + beta_a)
    ("beta_a", -1),         # the decay bias, added to BOTH c1 forms
    ("mu_gain", -1),        # c2
    ("a_dyn", 1),           # s1, (B,N,L)
    ("mu", 1),              # s2, (B,N,L)
    ("mu_init", 1),         # s2's t=-1 term, (B,N)
    ("row_law", -1),        # the modulus FORM itself
)


def _pair_untied(t, axis: int, pair_rot: torch.Tensor) -> bool:
    """Does `t` differ between the two coordinates of any rotating pair?"""
    if t is None:
        return False
    v = t.movedim(axis, -1)
    if v.shape[-1] != pair_rot.numel() * 2:
        return False
    d = v[..., 0::2] != v[..., 1::2]
    return bool(d[..., pair_rot.to(d.device)].any())


def assert_pair_tied(A_log: torch.Tensor, row_rot: torch.Tensor, *,
                     beta_a=None, mu_gain=None,
                     a_dyn=None, mu=None, mu_init=None, row_law=None) -> None:
    """Rotating rows require constant modulus within a pair.

    The pair (2k, 2k+1) is ONE complex mode. The kernel never rotates the
    state: it folds R(Theta) onto B/C and runs a real diagonal scan.
    That fold is a change of FRAME only while the decay acts on the pair as a
    scalar, because diag(a_even, a_odd) commutes with R(Theta) iff
    a_even == a_odd. Untied, the kernel still computes something well defined
    -- and both fp64 oracles compute the SAME well-defined thing, which is
    exactly why no oracle test can see it -- but that something is no longer
    the complex system the pair is supposed to be.

    So this checks every per-coordinate factor of

        a_bar[p,n] = exp(c1[h,p,n] * s1[b,h,n,t] + c2[h,p,n] * s2[b,h,n,t])

    and not just `A_log`: the decay bias `beta_a` rides on c1 for every law
    (`reference.py`'s `a_off`), `mu_gain` is
    c2, and at n-resolution `a_dyn`/`mu`/`mu_init` are per-row streams inside
    s1/s2. `row_law` is checked too: two rows of one pair under different laws
    have different modulus FORMS whatever their coefficients (`norm_fracs`
    already refuses a rotating block at an odd offset or with an odd row
    count, so a fracs-built `row_law` cannot fail this).

    Exact equality, not a tolerance: the tie is a construction (store as
    (H,P,K) and expand, or gather the odd column from its even partner as
    `HybridMambaSSM` does for `A_log`), never a coincidence.

    Off the executed path by default -- it costs a device->host sync per
    call. `mh_scan(validate=True)` and `HYBRID_SSM_MH_VALIDATE=1` turn it on;
    the test suite runs with it on (`tests/conftest.py`).
    """
    N = A_log.shape[-1]
    if N % 2 or row_rot is None:
        return
    pair_rot = row_rot[0::2] & row_rot[1::2]
    if not pair_rot.any():
        return
    given = dict(A_log=A_log, beta_a=beta_a, mu_gain=mu_gain,
                 a_dyn=a_dyn, mu=mu, mu_init=mu_init, row_law=row_law)
    bad = [name for name, axis in MODULUS_FACTORS
           if _pair_untied(given[name], axis, pair_rot)]
    if bad:
        raise ValueError(
            f"untied modulus on a rotating pair: {', '.join(bad)} differ "
            f"between coordinates 2k and 2k+1. Every "
            f"factor of the pair's decay must satisfy t[..., 2k] == "
            f"t[..., 2k+1] on rotating rows, because the rotation is folded "
            f"onto B/C and diag(a_even, a_odd) commutes with R(Theta) only "
            f"when the two are equal. Store as (H,P,K) and expand, or gather "
            f"the odd coordinate from its even partner."
        )
