"""FAMILY-GENERIC state passing and BPTT, driven by what each family ADVERTISES.

The shared state contract makes two separate claims and
this file keeps them separate, because a family can hold one and not the
other:

  * **native carry acceptance** -- "Native
    kernels must accept arbitrary feasible complete initial States ... A
    forced step fallback or zero-history-only support does not meet this
    requirement."  Required of EVERY family.  Tier 1 below.
  * **full BPTT** -- "Full split versus unsplit gradient equivalence
    is required only for routes advertising full BPTT."  Tier 2 below,
    and it runs only where the advertisement says `bptt="full"`.

The point of the file is that the advertisement is **test-enforced**: a
capability asserted in a contract with no cell pinning it is exactly what
this suite exists to prevent.  So

  * every State-bearing SSM class in `orbit_ssm.ssm` must appear either
    in `ADVERTISED` or in `OUT_OF_SCOPE` with a reason --
    `test_every_state_bearing_family_is_declared` discovers them by
    introspection, so **adding a family cannot silently skip the suite**;
  * a family that does NOT advertise full BPTT SKIPS tier 2 with its
    declared reason printed, rather than passing quietly;
  * `test_the_advertisement_matches_the_route_actually_taken` fails a
    family that advertises native continuation and then demotes to the
    O(L) python step loop -- an advertisement may not be aspirational.

Numbers are fp64 on CPU, so tolerances are round-off, not model
tolerance.  `backend="sequential"` (SSD) / the step loop (all families)
is the token-by-token recurrence the State API is DEFINED by, and is the
reference every equality below is taken against.

MACHINE-READABLE ADVERTISEMENT.  The lookup prefers a module-supplied
`state_capability()` and falls back to the `ADVERTISED` table here.  No
family implements the method today; migrating the declaration into the
modules is the better home for it but touches files this suite does not
own (`src/orbit_ssm/ssm/*.py`), so it is left as a TODO rather than
half-built.  The discovery guard is what makes the fallback safe.
"""
import dataclasses
import importlib
import inspect
import pkgutil

import pytest
import torch

import orbit_ssm.ssm as _ssm_pkg
from orbit_ssm.layers.state import ScanState, State
from orbit_ssm.ssm.hybrid_mamba import HybridMambaSSM, HybridMambaConfig
from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM, SSDMimoConfig

from _tolerance import assert_close_terms
from test_state_lifecycle import argument_amplification, module_terms

from _design_warnings import SILENCE_SSD_DESIGN_WARNINGS

pytestmark = SILENCE_SSD_DESIGN_WARNINGS  # builds the warned configurations on purpose


# ── the advertisement ─────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Advertised:
    """What a family CLAIMS, and the contract line that claims it."""
    build: object                 # (backend=...) -> fp64 module
    carries: tuple                # the State keys this build must expose
    native_continuation: bool     # a carried call stays OFF the step loop
    bptt: str                     # "full" | "partial" | "none"
    no_adjoint: tuple             # carries accepted without a boundary adjoint
    reason: str                   # why, when a claim is not "full"/True
    contract: str                 # where the claim is written down
    step_loop: tuple              # method names that MEAN the O(L) loop
    needs_sel_x: bool = False     # forward/step want the block input too
    d_inner: int = 8


# Every builder SEEDS: the cells below construct the same family twice and
# compare, so two builds must be the same module, not two samples.
def _ssd(backend='torch', seed=0, **kw):
    """SSD-MIMO with all four carries live: hidden_state, rot_ang,
    write_prev (write_rule="trapezoid"), bc_conv_state (bc_conv=True)."""
    torch.manual_seed(seed)
    # the full layer; the cells drive its bare recurrence (`scan`, see `call`)
    return SSDMimoSSM(SSDMimoConfig(d_model=8, d_state=8, d_inner=8, d_ssm_in=2, d_ssm_out=1, n_groups=2, n_heads=2,
                      proj_source='layer_input',
                      row_blocks=[{'frac': 1.0, 'decay': 'head_dynamic', 'complex': True, 'gate': True}],
                      bc_conv=True, d_conv=3, write_rule='trapezoid',
                      **kw), execution={'backend': backend}).double()


_FLAT = ('head_dynamic', 'head_static')


def _mh(rows, backend='torch', seed=0, proj_source='main_branch', **kw):
    torch.manual_seed(seed)
    if any(f['decay'] in _FLAT for f in rows):
        # flat rows are served only with the per-head bias and, when
        # complex, the fixed-frequency rotation without read offset (one
        # global rotation setting)
        kw.setdefault('bc_bias', 'head')
        if any(f.get('complex') for f in rows):
            kw.setdefault('learn_theta_freq', False)
            kw.setdefault('use_theta_h', False)
    else:
        kw.setdefault('bc_bias', 'none')    # the pre-spec HM default
    d_model = kw.pop('d_model', 8)
    return HybridMambaSSM(HybridMambaConfig(d_model=d_model, d_state=sum(f['rows'] for f in rows), d_inner=8,
                          n_heads=2, row_blocks=rows, proj_source=proj_source,
                          learnable_h_init=True,
                          **kw), execution={'backend': backend}).double()


def _mh_a(backend='torch', **kw):
    """Per-coordinate decay rows only -- one mh_scan law, no flat rows."""
    return _mh([{'rows': 8, 'decay': 'coord_static'}], backend, **kw)


def _mh_mixed(backend='torch', **kw):
    """The mixed geometry the MH contract's State section describes:
    per-coordinate A rows plus FLAT head_dynamic/head_static rows, which is what
    brings `theta`, `write_prev`, `c3_*` and `bc_conv_state` into the
    State."""
    return _mh([{'rows': 4, 'decay': 'coord_static', 'complex': True},
                {'rows': 16, 'decay': 'head_dynamic', 'complex': True},
                {'rows': 16, 'decay': 'head_static'}], backend, **kw)


def _mh_conv(backend='torch', **kw):
    """Flat head_dynamic rows read from the LAYER INPUT, which is the only source
    that has a B/C convolution -- this is the build that puts
    `bc_conv_state` into an MH State at all."""
    return _mh([{'rows': 4, 'decay': 'coord_static', 'complex': True},
                {'rows': 16, 'decay': 'head_dynamic', 'complex': True}], backend,
               proj_source='layer_input', d_model=8, bc_conv=True,
               d_conv=3, **kw)


_SHARED = "the shared state contract"

ADVERTISED = {
    'ssd': Advertised(
        build=_ssd,
        carries=('hidden_state', 'rot_ang', 'write_prev', 'bc_conv_state'),
        native_continuation=True,
        bptt='full',
        no_adjoint=(),
        reason='',
        contract='SSD-MIMO state contract',
        step_loop=('_forward_sequential',),
        needs_sel_x=True,
    ),
    'mh_a': Advertised(
        build=_mh_a,
        carries=('hidden_state', 'write_prev'),
        # 575c775 routed PURE per-coordinate rows through `_scan_cf` with
        # differentiable carries; 17da861 extended it to MIXED rows, so
        # `n_flat` (flat rows) is no longer a condition.  The step loop is now
        # reached only for an empty chunk, backend="sequential", or
        # m3_kernel="upstream".
        native_continuation=True,
        bptt='full',
        no_adjoint=(),
        reason='',
        contract='HybridMamba state contract',
        step_loop=('_forward_stepped',),
    ),
    'mh_mixed': Advertised(
        build=_mh_mixed,
        # the MH contract's flat-row State section, all of it:
        carries=('hidden_state', 'theta', 'write_prev', 'c3_kp', 'c3_vp',
                 'c3_phase'),
        # MIXED rows take `_scan_cf` too.
        native_continuation=True,
        bptt='full',
        no_adjoint=(),
        reason='',
        contract='HybridMamba state contract',
        step_loop=('_forward_stepped',),
    ),
    'mh_conv': Advertised(
        build=_mh_conv,
        carries=('hidden_state', 'theta', 'write_prev', 'c3_kp', 'c3_vp',
                 'c3_phase', 'bc_conv_state'),
        # 17da861; `bc_conv_state` only exists under
        # proj_source="layer_input", so without this build the B/C
        # convolution carry would be advertised in
        # the HybridMamba state contract and never tested.
        native_continuation=True,
        bptt='full',
        no_adjoint=(),
        reason='',
        contract='HybridMamba state contract',
        step_loop=('_forward_stepped',),
        needs_sel_x=True,
    ),
}

#: State-bearing classes deliberately NOT exercised here, with the reason.
#: Every family is exercised through its layer class (`SSDMimoSSM`,
#: `HybridMambaSSM`; entries above) through the layer's
#: bare recurrence `scan` and its `ScanState` lifecycle.
OUT_OF_SCOPE = {
    # the private scan bases are what the layers' `scan` / `scan_step` are
    # inherited from; the layer carries the State and is covered above
    '_SSDMimoScan': 'scan base of SSDMimoSSM',
    '_HybridMambaScan': 'scan base of HybridMambaSSM',
}

try:       # families another suite of the same source tree pins (absent from a tree that has none)
    from _declared_elsewhere import DECLARED_ELSEWHERE
except ImportError:
    DECLARED_ELSEWHERE = {}

FAMILIES = sorted(ADVERTISED)
FULL_BPTT = [f for f in FAMILIES if ADVERTISED[f].bptt == 'full']


def capability(name):
    """Prefer a module-supplied declaration; fall back to the table.

    TODO(state-capability): the right home for this is a
    `state_capability()` method on each core, so the advertisement ships
    with the implementation instead of beside it.  That edit lands in
    `src/orbit_ssm/ssm/*.py`, which this suite does not own.
    """
    ad = ADVERTISED[name]
    mod = ad.build()
    declared = getattr(mod, 'state_capability', None)
    if declared is None:
        return ad
    d = declared()
    return dataclasses.replace(
        ad, bptt=d['bptt'], no_adjoint=tuple(d.get('no_adjoint', ())),
        native_continuation=d.get('native_continuation',
                                  ad.native_continuation))


# ── generic helpers ───────────────────────────────────────────────────
def lifecycle(mod):
    """The object whose `init_state` / `reset_state` define the State this
    suite pins: the layer's scan State (`ScanState`, no drive conv carry)."""
    return ScanState(mod)


def call(ad, mod, x, state=None, return_state=True):
    """One forward, supplying `sel_x` for families that read the block
    input separately from the main branch (SSD at
    proj_source="layer_input").  Both families run their bare recurrence
    (`scan`)."""
    kw = dict(sel_x=x) if ad.needs_sel_x else {}
    return mod.scan(x, state=state, return_state=return_state, **kw)


def run(ad, mod, x, splits, state=None):
    """Chain `mod` over `splits`, threading the returned State."""
    ys = []
    for a, b in splits:
        y, state = call(ad, mod, x[:, a:b], state)
        ys.append(y)
    return torch.cat(ys, dim=1), state


def seeded_state(mod, B, seed=11, scale=.05):
    """A complete State with every floating carry a NONZERO leaf -- the
    "arbitrary feasible complete initial State" of §3, not a zero seed."""
    g = torch.Generator().manual_seed(seed)
    st = lifecycle(mod).init_state(B, dtype=torch.float64)
    leaves = {}
    for k, v in list(st.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            t = torch.randn(v.shape, generator=g, dtype=v.dtype) * scale
            st[k] = t.requires_grad_()
            leaves[k] = st[k]
    return st, leaves


def state_loss(y, st):
    """Reaches the OUTPUT and every returned carry, with a different
    weight per carry so a swapped pair cannot cancel."""
    return y.square().sum() + sum(
        (i + 2) * v.square().sum() for i, v in enumerate(st.values())
        if torch.is_tensor(v) and v.is_floating_point())


# ── A7: the reference arm and its sum|terms| ────────
class Reference:
    """The WHOLE-sequence (or given-split) fp64 reference arm of a cell,
    with sum|terms| of its output, of every returned carry and -- for a
    loss -- of every gradient (``module_terms`` = ``terms_abs`` through
    the plain ops).

    Both arms are fp64 and differ only in where the sequence is cut (or in
    batch composition), so u = fp64 and, worst case per arm, gamma_{n+m}
    with two arms rounding independently:

        c = 2 * (n + m + amp)
        n = in/sel-proj d_inner + conv 3 + scan L * N * R + out d_inner
            (+ the B * L token reduction of a parameter gradient)
        m = 8 factors (proj weight, conv, B, decay, C, gate, D, out)
        amp = 1 + max |x| over the reference's exp/sin/cos/... arguments
              (``argument_amplification``: the rounding of a decay or angle
              argument is amplified by |x|; terms_abs counts f(x) as one
              term).
    """

    def __init__(self, ad, x, splits, state=None, loss=False, mod=None):
        mod = ad.build() if mod is None else mod
        keys = sorted(k for k, v in (state or {}).items()
                      if torch.is_tensor(v) and v.is_floating_point())
        if loss:
            x = x.detach().clone().requires_grad_()

        def ref(x, *vals):
            st = None
            if state is not None:
                st = State(state)
                for k, v in zip(keys, vals):
                    st[k] = v
            y, out = run(ad, mod, x, splits, state=st)
            if loss:
                return state_loss(y, out)
            return y, {k: v for k, v in out.items() if torch.is_tensor(v)}

        leaves = [state[k] for k in keys]
        self.keys, self.mod = keys, mod
        B, L = x.shape[:2]
        self.res = module_terms(ref, x, *leaves, module=mod)
        rank = getattr(mod, 'R', 1) or 1
        width = x.shape[-1]
        n = 2 * width + 3 + L * mod.hidden_state_shape()[-1] * rank
        if loss:
            n += B * L
        self.c = 2 * (n + 8 + argument_amplification(ref, x, *leaves))

    def check_grad(self, a, b, t, name):
        assert_close_terms(a, b, t, 'fp64', self.c, name=name)

    def check_y(self, y, ref_y, name='y', rows=None):
        terms = self.res.terms[0]
        if rows is not None:
            terms = terms[rows]
        assert_close_terms(y, ref_y, terms, 'fp64', self.c, name=name)

    def check_state(self, got, ref):
        tensors = {k for k, v in ref.items() if torch.is_tensor(v)}
        assert set(got) == set(ref), (sorted(got), sorted(ref))
        assert tensors == set(self.res.terms[1]), sorted(tensors)
        for k in ref:
            a, b = got[k], ref[k]
            if not (torch.is_tensor(a) and a.is_floating_point()):
                continue
            assert_close_terms(a, b, self.res.terms[1][k], 'fp64', self.c,
                               name=f"carry {k!r}")

    def grad_terms(self):
        """sum|terms| of [every incoming carry (sorted), input, params]."""
        pt = self.res.param_grad_terms or {}
        return (list(self.res.grad_terms[1:]) + [self.res.grad_terms[0]]
                + [pt.get(n) for n, _ in self.mod.named_parameters()])




class StepLoopProbe:
    """Counts entries into a family's O(L) python step loop."""

    def __init__(self, mod, names, monkeypatch):
        self.n = 0
        for name in names:
            orig = getattr(type(mod), name)

            def wrap(inner_self, *a, _o=orig, **kw):
                self.n += 1
                return _o(inner_self, *a, **kw)
            monkeypatch.setattr(type(mod), name, wrap)


def inputs(ad, B, L, seed=3):
    torch.manual_seed(seed)
    return torch.randn(B, L, ad.d_inner, dtype=torch.float64)


# ── the discovery guard: a new family cannot silently skip ────────────
def _discover():
    found = {}
    for m in pkgutil.iter_modules(_ssm_pkg.__path__):
        mod = importlib.import_module(f"orbit_ssm.ssm.{m.name}")
        for n, c in vars(mod).items():
            if (inspect.isclass(c) and issubclass(c, torch.nn.Module)
                    and c.__module__ == mod.__name__
                    and not n.startswith('_')     # private implementation subclasses
                    and hasattr(c, 'init_state') and hasattr(c, 'step')):
                found[n] = f"orbit_ssm.ssm.{m.name}"
    return found


def test_every_state_bearing_family_is_declared():
    """Every class in `orbit_ssm.ssm` that carries a State is either
    exercised by this suite or excluded BY NAME with a reason.  Without
    this cell, adding a family would silently add an unpinned contract
    claim -- the failure mode being guarded against."""
    covered = {type(ADVERTISED[f].build()).__name__ for f in FAMILIES}
    undeclared = {n: w for n, w in _discover().items()
                  if n not in covered and n not in OUT_OF_SCOPE
                  and n not in DECLARED_ELSEWHERE}
    assert not undeclared, (
        f"State-bearing SSM classes neither exercised nor excluded: "
        f"{undeclared}.  Add them to ADVERTISED (with the contract line "
        f"that advertises their capability) or to OUT_OF_SCOPE with a "
        f"reason.  See {_SHARED}.")


def test_an_advertisement_names_a_reason_whenever_it_is_not_full():
    """A partial claim must say WHY, and point at a contract line."""
    for name in FAMILIES:
        ad = capability(name)
        complete = ad.bptt == 'full' and ad.native_continuation
        assert complete or ad.reason.strip(), (
            f"{name}: advertises bptt={ad.bptt!r} / native_continuation="
            f"{ad.native_continuation} but gives no reason")
        assert ad.contract, name
        assert ad.bptt in ('full', 'partial', 'none'), (name, ad.bptt)


# ── TIER 1: required of EVERY family ──────────────────────────────────
@pytest.mark.parametrize('name', FAMILIES)
def test_the_declared_carry_set_is_really_in_the_state(name):
    """Pins that the carries this suite thinks it is exercising EXIST.

    Without it every cell below could be testing `hidden_state` alone
    and still be green -- which is how "each carry individually" turns
    into "the carries in combination, one of which may be absent"."""
    ad = capability(name)
    mod = ad.build()
    st = lifecycle(mod).init_state(2, dtype=torch.float64)
    missing = [k for k in ad.carries if k not in st]
    assert not missing, (f"{name}: advertised carries {missing} are not in "
                         f"init_state (present: {sorted(st)})")
    live = [k for k, v in st.items()
            if torch.is_tensor(v) and v.is_floating_point()]
    assert live, f"{name}: no floating carry at all"


@pytest.mark.parametrize('name', FAMILIES)
def test_an_arbitrary_feasible_complete_state_is_accepted(name):
    """§3: "Native kernels must accept arbitrary feasible complete
    initial States ... including custom nonzero previous-write/B-drive
    history."  A NONZERO seed in every carry, not a zero State."""
    ad = capability(name)
    mod = ad.build()
    x = inputs(ad, 2, 6)
    st, leaves = seeded_state(mod, 2)
    assert leaves, name
    before = {k: v.detach().clone() for k, v in st.items()
              if torch.is_tensor(v)}
    y, out = call(ad, mod, x, st)
    assert torch.isfinite(y).all()
    # the incoming State is an INPUT, not scratch space
    for k, v in before.items():
        torch.testing.assert_close(st[k], v, rtol=0, atol=0,
                                   msg=lambda m, k=k: f"{name} mutated {k}: {m}")
    assert out is not st


@pytest.mark.parametrize('name', FAMILIES)
@pytest.mark.parametrize('splits', [
    [(0, 1), (1, 9)],                     # single-token first chunk
    [(0, 4), (4, 6), (6, 9)],             # three segments
    [(0, 3), (3, 4), (4, 5), (5, 9)],     # FOUR segments, two of them short
], ids=['head1', 'three', 'four'])
def test_split_continuation_reproduces_the_whole_sequence(name, splits):
    """§3: "Whole-sequence, arbitrary splits and repeated steps agree in
    outputs and complete final State for the same initialization." """
    ad = capability(name)
    x = inputs(ad, 2, 9)
    whole, st_w = run(ad, ad.build(), x, [(0, 9)])
    split, st_s = run(ad, ad.build(), x, splits)
    ref = Reference(ad, x, [(0, 9)])
    ref.check_y(split, whole)
    ref.check_state(st_s, st_w)


@pytest.mark.parametrize('name', FAMILIES)
def test_continuation_from_a_nonzero_incoming_state_matches_the_steps(name):
    """The same split, but STARTING from an arbitrary complete State --
    §3's "nonzero initial hidden/auxiliary carries" coverage item."""
    ad = capability(name)
    mod = ad.build()
    x = inputs(ad, 2, 8)
    st0, _ = seeded_state(mod, 2)
    whole, st_w = run(ad, ad.build(), x, [(0, 8)], state=st0)
    split, st_s = run(ad, ad.build(), x, [(0, 2), (2, 3), (3, 8)], state=st0)
    stepped, st_q = run(ad, ad.build(), x, [(i, i + 1) for i in range(8)],
                        state=st0)
    ref = Reference(ad, x, [(0, 8)], state=st0)
    ref.check_y(split, whole, name='split')
    ref.check_y(stepped, whole, name='stepped')
    ref.check_state(st_s, st_w)
    ref.check_state(st_q, st_w)


@pytest.mark.parametrize('name', FAMILIES)
def test_selected_batch_rows_continue_independently(name):
    """§3 coverage item: "batch changes via explicit state selection"."""
    ad = capability(name)
    mod = ad.build()
    x = inputs(ad, 3, 8, seed=5)
    _, mid = run(ad, mod, x, [(0, 4)])
    idx = torch.tensor([2, 0, 2])
    picked = mid.index_select(idx)
    y_sel, _ = run(ad, mod, x.index_select(0, idx), [(4, 8)], state=picked)
    y_all, _ = run(ad, mod, x, [(4, 8)], state=mid)
    # the reference is the unselected continuation; `mid` enters both arms
    # as the same values (a selection is a pure data movement)
    ref = Reference(ad, x[:, 4:], [(0, 4)], state=mid, mod=mod)
    ref.check_y(y_sel, y_all.index_select(0, idx), rows=idx)


@pytest.mark.parametrize('name', FAMILIES)
def test_reset_state_mask_restarts_only_the_masked_rows(name):
    ad = capability(name)
    mod = ad.build()
    x = inputs(ad, 2, 8, seed=5)
    _, mid = run(ad, mod, x, [(0, 4)])
    _, keep = run(ad, mod, x, [(0, 4)])
    lifecycle(mod).reset_state(mid, mask=torch.tensor([True, False]))
    y, _ = run(ad, mod, x, [(4, 8)], state=mid)
    fresh, _ = run(ad, mod, x, [(4, 8)])
    kept, _ = run(ad, mod, x, [(4, 8)], state=keep)
    ref_fresh = Reference(ad, x[:, 4:], [(0, 4)], mod=mod)
    ref_kept = Reference(ad, x[:, 4:], [(0, 4)], state=keep, mod=mod)
    ref_fresh.check_y(y[0], fresh[0], name='reset row', rows=0)
    ref_kept.check_y(y[1], kept[1], name='kept row', rows=1)


@pytest.mark.parametrize('name', FAMILIES)
def test_the_advertisement_matches_the_route_actually_taken(name,
                                                            monkeypatch):
    """§3: "A forced step fallback ... does not meet this requirement."

    A family advertising native continuation must NOT enter its O(L)
    python step loop on a carried, state-returning call.  A family that
    does not advertise it is pinned the other way, so the day it gains a
    native route this cell FAILS and the advertisement gets updated --
    an advertisement may not drift in either direction."""
    ad = capability(name)
    mod = ad.build()
    x = inputs(ad, 2, 6)
    st, _ = seeded_state(mod, 2)
    probe = StepLoopProbe(mod, ad.step_loop, monkeypatch)
    y, out = call(ad, mod, x, st)
    assert torch.isfinite(y).all()
    if ad.native_continuation:
        assert probe.n == 0, (
            f"{name} advertises NATIVE continuation but entered "
            f"{ad.step_loop} {probe.n}x")
    else:
        assert probe.n > 0, (
            f"{name} does NOT advertise native continuation, yet never "
            f"entered {ad.step_loop}.  If it has gained a native route, "
            f"update ADVERTISED[{name!r}].native_continuation -- the "
            f"declared reason is now stale:\n  {ad.reason}")


# ── TIER 2: only for routes advertising FULL BPTT ─────────────────────
def _require_full_bptt(ad, name):
    if ad.bptt != 'full':
        pytest.skip(f"{name} advertises bptt={ad.bptt!r} ({ad.contract}): "
                    f"{ad.reason}")


@pytest.mark.parametrize('name', FAMILIES)
def test_gradients_flow_into_an_incoming_state(name):
    """§3: "gradients into initial hidden and starting rotation/phase
    remain required".  Every floating carry of the INCOMING State is a
    leaf; each must receive a finite, non-zero gradient, or be named in
    the advertisement's `no_adjoint`."""
    ad = capability(name)
    _require_full_bptt(ad, name)
    mod = ad.build()
    x = inputs(ad, 2, 6)
    st, leaves = seeded_state(mod, 2)
    y, out = call(ad, mod, x, st)
    keys = sorted(leaves)
    grads = torch.autograd.grad(y.square().sum(),
                                [leaves[k] for k in keys],
                                allow_unused=True)
    for k, g in zip(keys, grads):
        if k in ad.no_adjoint:
            continue
        assert g is not None, f"{name}: d{k} is None (not in no_adjoint)"
        assert torch.isfinite(g).all(), f"{name}: d{k} not finite"
        assert g.abs().max() > 0, (
            f"{name}: d{k} is exactly zero -- a zero adjoint reads as "
            f"converged; declare it in no_adjoint or wire it")


@pytest.mark.parametrize('name', FAMILIES)
def test_a_loss_on_a_returned_state_reaches_the_incoming_one(name):
    """The other direction: the loss touches ONLY the returned carries
    (not the output), and must still reach the incoming State's leaves.
    This is what makes a chunked training loop's carry a real edge."""
    ad = capability(name)
    _require_full_bptt(ad, name)
    mod = ad.build()
    x = inputs(ad, 2, 6)
    st, leaves = seeded_state(mod, 2)
    _, out = call(ad, mod, x, st)
    carry_only = sum((i + 2) * v.square().sum()
                     for i, v in enumerate(out.values())
                     if torch.is_tensor(v) and v.is_floating_point())
    keys = sorted(leaves)
    grads = torch.autograd.grad(carry_only, [leaves[k] for k in keys],
                                allow_unused=True)
    live = [k for k, g in zip(keys, grads)
            if g is not None and g.abs().max() > 0]
    assert live, (f"{name}: a loss on the RETURNED State reached no "
                  f"incoming carry at all -- the returned carries are "
                  f"detached from the incoming ones")


@pytest.mark.parametrize('name', FAMILIES)
@pytest.mark.parametrize('splits', [
    [(0, 4), (4, 8)],
    [(0, 3), (3, 6), (6, 8)],
    [(0, 1), (1, 3), (3, 4), (4, 8)],      # FOUR segments
], ids=['two', 'three', 'four'])
def test_split_gradients_equal_unsplit_gradients(name, splits):
    """THE BPTT cell.  §3: "Full split versus unsplit gradient
    equivalence is required only for routes advertising full BPTT."

    One loss over the concatenated output and the final State; the
    gradient w.r.t. the input, every parameter and every incoming carry
    must be the same whether the sequence ran whole or in segments."""
    ad = capability(name)
    _require_full_bptt(ad, name)
    x = inputs(ad, 2, 8)

    def arm(spl):
        mod = ad.build()
        st, leaves = seeded_state(mod, 2)
        xin = x.clone().requires_grad_()
        y, out = run(ad, mod, xin, spl, state=st)
        keys = sorted(leaves)
        wrt = [leaves[k] for k in keys] + [xin] + list(mod.parameters())
        g = torch.autograd.grad(state_loss(y, out), wrt, allow_unused=True)
        return y, keys, g

    y_ref, keys, g_ref = arm([(0, 8)])
    y_got, keys2, g_got = arm(splits)
    assert keys == keys2
    mod = ad.build()
    st, _ = seeded_state(mod, 2)
    fwd = Reference(ad, x, [(0, 8)], state=st, mod=mod)
    fwd.check_y(y_got, y_ref)
    ref = Reference(ad, x, [(0, 8)], state=st, loss=True, mod=mod)
    names = keys + ['input'] + [f'param {n}' for n, _ in mod.named_parameters()]
    assert ref.keys == keys
    for k, a, b, t in zip(names, g_got, g_ref, ref.grad_terms()):
        assert (a is None) == (b is None), f"{name}: {k} None-ness differs"
        if a is not None:
            ref.check_grad(a, b, t, name=f"{name} grad {k}")
    for k, a in zip(keys, g_got):
        if k not in ad.no_adjoint:
            assert a is not None and a.abs().max() > 0, f"{name}: d{k} zero"


@pytest.mark.parametrize('name', FAMILIES)
def test_each_carry_individually_carries_its_own_gradient(name):
    """Perturbing ONE carry must move the loss, for each carry alone.

    The combination cells above can be green while one carry is threaded
    as a constant: its gradient contribution is then hidden inside the
    others.  This isolates them -- the loss is differentiated w.r.t. a
    single leaf while the rest are plain tensors."""
    ad = capability(name)
    _require_full_bptt(ad, name)
    mod = ad.build()
    x = inputs(ad, 2, 6)
    base, _ = seeded_state(mod, 2)
    floating = [k for k, v in base.items()
                if torch.is_tensor(v) and v.is_floating_point()]
    dead = []
    for k in floating:
        st = base.__class__({kk: (vv.detach().clone() if torch.is_tensor(vv)
                                  else vv) for kk, vv in base.items()})
        st[k] = st[k].requires_grad_()
        y, out = run(ad, mod, x, [(0, 3), (3, 6)], state=st)
        g, = torch.autograd.grad(state_loss(y, out), [st[k]],
                                 allow_unused=True)
        if g is None or g.abs().max() == 0:
            dead.append(k)
    unexpected = [k for k in dead if k not in ad.no_adjoint]
    assert not unexpected, (
        f"{name}: carries {unexpected} are threaded through a SPLIT but "
        f"carry no gradient of their own; either they are being passed as "
        f"constants or they must be declared in no_adjoint")


@pytest.mark.parametrize('name', FAMILIES)
def test_truncated_bptt_cuts_history_only_where_the_caller_detaches(name):
    """§3: "Explicit `state.detach()` remains the caller's operation to
    cut all history."  Truncated BPTT must (a) still run, (b) give the
    SAME parameter gradient for the LAST segment as a fresh call from
    the detached State, and (c) differ from full BPTT -- otherwise the
    carry was never an edge in the graph to begin with."""
    ad = capability(name)
    _require_full_bptt(ad, name)
    x = inputs(ad, 2, 8)

    def full():
        mod = ad.build()
        st, _ = seeded_state(mod, 2)
        y, out = run(ad, mod, x, [(0, 4), (4, 8)], state=st)
        return torch.autograd.grad(state_loss(y, out),
                                   list(mod.parameters()), allow_unused=True)

    def truncated():
        mod = ad.build()
        st, _ = seeded_state(mod, 2)
        y0, mid = call(ad, mod, x[:, :4], st)
        cut = mid.detach()
        for v in cut.values():
            if torch.is_tensor(v) and v.is_floating_point():
                assert v.grad_fn is None and not v.requires_grad
        y1, out = call(ad, mod, x[:, 4:], cut)
        return torch.autograd.grad(
            state_loss(torch.cat((y0, y1), 1), out),
            list(mod.parameters()), allow_unused=True)

    g_full, g_trunc = full(), truncated()
    assert len(g_full) == len(g_trunc)
    live = [(a, b) for a, b in zip(g_full, g_trunc) if a is not None]
    assert live, name
    for a, b in live:
        assert torch.isfinite(b).all()
    # the cut must actually remove something: at least one parameter
    # gradient differs, or the carry was not a graph edge
    assert any((a - b).abs().max() > 1e-12 for a, b in live), (
        f"{name}: detaching the carry changed NO parameter gradient, so "
        f"the carry is not a live edge across the call boundary")
