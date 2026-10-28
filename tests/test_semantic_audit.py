"""Hand-calculated anchors for the equation oracle, independent of SSD code.

Production readiness is checked separately by tools/audit_ssm_semantics.py;
passing these tests does not mean production implements these equations.
"""
import math

import torch

from _tolerance import assert_close_terms, terms_abs
from tools.audit_ssm_semantics import endpoint_states


def test_endpoint_impulse_includes_transported_previous_write():
    a = torch.full((2, 1, 1), .5, dtype=torch.float64)
    v = torch.tensor([[1.], [0.]], dtype=torch.float64)
    h = endpoint_states(a, v, torch.ones(2), torch.tensor([1., .5]),
                        torch.zeros(1, dtype=torch.float64))
    # h0=1; h1=.5*1 + .5*.5*1 = .75 (ordinary Euler gives .5).
    torch.testing.assert_close(h[:, 0], torch.tensor([1., .75], dtype=torch.float64))


def test_current_clock_scales_both_endpoints():
    a = torch.full((2, 1, 1), .5, dtype=torch.float64)
    v = torch.tensor([[2.], [3.]], dtype=torch.float64)
    h = endpoint_states(a, v, torch.tensor([.25, 2.]), torch.full((2,), .5),
                        torch.tensor([4.], dtype=torch.float64))
    # h0=2+.25*.5*2=2.25; h1=1.125+2*(1.5+.5*.5*2)=5.125.
    torch.testing.assert_close(h[:, 0], torch.tensor([2.25, 5.125], dtype=torch.float64))


def test_previous_endpoint_uses_full_rotation_and_decay():
    a = torch.tensor([[[1., 0.], [0., 1.]], [[0., -.5], [.5, 0.]]],
                     dtype=torch.float64)
    v = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
    h = endpoint_states(a, v, torch.ones(2), torch.tensor([1., .5]),
                        torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(h[1], torch.tensor([0., .75], dtype=torch.float64))


def test_endpoint_gradients_by_hand():
    v = torch.tensor([[1.], [0.]], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor([1., .5], dtype=torch.float64, requires_grad=True)
    h0 = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
    h = endpoint_states(torch.full((2, 1, 1), .5, dtype=torch.float64), v,
                        torch.ones(2), lam, h0)
    dv, dl, dh = torch.autograd.grad(h[-1].sum(), (v, lam, h0))
    torch.testing.assert_close(dv[:, 0], torch.tensor([.75, .5], dtype=torch.float64))
    torch.testing.assert_close(dl, torch.tensor([.5, -.5], dtype=torch.float64))
    torch.testing.assert_close(dh, torch.tensor([.25], dtype=torch.float64))


def test_collapsed_endpoint_substitution_and_future_cancellation():
    # The correct auxiliary state is z[t]=h[t]+c[t]v[t], not
    # h[t]-lambda[t]*dt[t]*v[t]. Exercise noncommuting transitions.
    a = torch.tensor([[[.5, 0.], [0., .5]], [[0., -.4], [.4, 0.]],
                      [[.3, .1], [0., .2]]], dtype=torch.float64)
    v = torch.tensor([[1., 2.], [3., -1.], [2., 4.]], dtype=torch.float64,
                     requires_grad=True)
    dt = torch.tensor([.2, .5, .7], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor([.3, .6, .8], dtype=torch.float64, requires_grad=True)
    h0 = torch.tensor([.2, -.1], dtype=torch.float64)
    direct = endpoint_states(a, v, dt, lam, h0)

    def collapse(v, dt, lam):
        c = torch.cat((dt[1:] * (1 - lam[1:]), dt.new_zeros(1)))
        z = h0
        result = []
        for t in range(3):
            z = a[t] @ z + (lam[t] * dt[t] + c[t]) * v[t]
            result.append(z - c[t] * v[t])
        return torch.stack(result)

    collapsed = collapse(v, dt, lam)
    # A7: u=fp64 (two fp64 formulations of one recurrence, each rounds on its
    # own summands, so the bound carries BOTH arms' sum|terms|); c=sqrt(2)*
    # (n+m)=sqrt(2)*12 worst case: n = T*N = 6 summands per component, m <= 6
    # factors (3 transitions, lam, dt, v), two arms rounding independently.
    c_depth = math.sqrt(2) * (3 * 2 + 6)
    ref = terms_abs(lambda *t: endpoint_states(a, *t, h0).sum(), v, dt, lam)
    col = terms_abs(lambda *t: collapse(*t).sum(), v, dt, lam)
    ref_y = terms_abs(lambda *t: endpoint_states(a, *t, h0), v, dt, lam)
    col_y = terms_abs(lambda *t: collapse(*t), v, dt, lam)
    assert_close_terms(collapsed, direct, ref_y.terms + col_y.terms, "fp64",
                       c_depth, name="collapsed h")
    grads = zip(torch.autograd.grad(collapsed.sum(), (v, dt, lam),
                                    retain_graph=True),
                torch.autograd.grad(direct.sum(), (v, dt, lam)),
                ref.grad_terms, col.grad_terms, ("dv", "ddt", "dlam"))
    for got, expected, t_ref, t_col, name in grads:
        assert_close_terms(got, expected, t_ref + t_col, "fp64", c_depth,
                           name=name)
    # The future cancellation: d collapsed[0] / d(v, dt, lam)[1:] is the
    # difference of +c[0]v[0] and -c[0]v[0] -- exactly 0 in exact arithmetic,
    # compared against the size of the two terms that cancel.
    first = terms_abs(lambda *t: collapse(*t)[0].sum(), v, dt, lam)
    dv, dd, dl = torch.autograd.grad(collapsed[0].sum(), (v, dt, lam))
    for derivative, t_col, name in zip((dv, dd, dl), first.grad_terms,
                                       ("dv", "ddt", "dlam")):
        # A7: u=fp64, c=sqrt(2)*12 as above (the same products, one arm).
        assert_close_terms(derivative[1:], torch.zeros_like(derivative[1:]),
                           t_col[1:], "fp64", c_depth,
                           name=f"future {name}")

def test_rotation_does_not_commute_with_untied_coordinate_gate():
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    gate = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    # A gate may be part of the input map, but cannot be moved through a
    # rotation unless its two coordinates are tied (or the rotation degenerates).
    torch.testing.assert_close(rotation @ gate - gate @ rotation,
                               torch.tensor([[0., -1.], [-1., 0.]], dtype=torch.float64))
