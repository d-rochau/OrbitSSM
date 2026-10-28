"""P = 1 with rotation -- the pure Mamba-1 complex mode -- against the fp64 oracle.

`P = 1` and a rotation are the two features that pull `mh_scan` onto its two
most special-cased paths at once: the packed small-P schedule (`HPACK`) and the pair walk.  Both ROUTES the angle can take are exercised here:

    kernel  the bare per-token increment inside the scan
    ssd     the module's own law: `theta_delta` + the read offset `theta_h`

Both are IN-KERNEL: the head-free `R(S)` host/Triton pre-folds ("host" and
"fold" in this file earlier) were retired with the affine ramp
`nu`, because the one rotation law's angle is head-dependent by
construction.  Across both endpoint modes (the trapezoid forcing on and off) and both
checkpoint modes, at lengths that straddle the P = 1 chunk length (64) --
64, 65, 128, 130 -- because the chunk boundary is where a P = 1 backward
reads the forward's checkpoints.
"""
import pytest
import torch

from _contract.mh_oracle import (amplification, assert_oracle_close, leaf_names,
                               mh_dims, oracle_pairs, oracle_terms)

ROUTES = ["kernel", "ssd"]


def _case(length, route, dev="cuda"):
    """Inputs, the `mh_scan`/`mh_reference` kwargs, and the leaf list."""
    torch.manual_seed(4096 + length)
    b, h, p, n = 2, 4, 1, 8

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, device=dev) * scale).requires_grad_()

    x = r(b, h, p, length)
    dt = r(b, h, length).sigmoid().detach().requires_grad_()
    al = r(h, p, n // 2).repeat_interleave(2, -1).detach().requires_grad_()
    bm, cm = r(b, n, length), r(b, n, length)
    theta = r(b, n // 2, length, scale=.05)
    kw = dict(theta=theta, beta_b=r(1, n), beta_c=r(1, n),
              h_init=r(b, h, p, n), delta_softplus=False)
    if route == "ssd":
        kw.update(theta_delta=True, theta_h=r(h, n // 2))
    return [x, dt, al, bm, cm], kw


def _leaves(args, kw):
    return args + [v for v in kw.values()
                   if isinstance(v, torch.Tensor) and v.requires_grad]


def _compare(args, kw, actual, weight):
    """Y and every gradient against the fp64 oracle.  A7 class (b): the fp32
    kernel vs `mh_reference` on the same inputs, u=fp32, c = oracle_c (scan
    depth L + N + factors + a leaf's fanout, typical sqrt, + the decay/angle
    amplification), terms-scaled per head (`test_mh_half_fold`)."""
    got = torch.autograd.grad(actual, _leaves(args, kw), weight)
    res = oracle_terms(args, kw, (weight, None))
    assert_oracle_close(oracle_pairs((actual,), got, res, leaf_names(kw)),
                        mh_dims(args), amplification(args, kw), "fp32")


@pytest.mark.parametrize("length", [64, 65, 128, 130])
@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("route", ROUTES)
def test_mh_p1_rotation_matches_the_fp64_oracle(
        require_cuda, length, endpoint, checkpoint, route):
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    args, kw = _case(length, route)
    if endpoint:
        kw["trap_lam"] = (torch.randn(2, 4, length, device="cuda") * .2) \
            .sigmoid().detach().requires_grad_()
    actual = mh_scan(*args, **kw, ckpt=checkpoint)
    weight = torch.randn_like(actual)
    _compare(args, kw, actual, weight)


@pytest.mark.parametrize("length", [65, 128])
@pytest.mark.parametrize("route", ["kernel", "ssd"])
def test_mh_p1_checkpoint_stride_survives_a_pinned_block_l(
        require_cuda, length, route):
    """A `mem=`/`config=` BLOCK_L may not move the checkpoint stride.

    Pass 2 reads `hck` slot `j` for chunk `j` with no stride conversion, so a
    backward chunked at anything but the FORWARD's `auto_block_l(P, L)` reads
    the wrong incoming state at every chunk boundary -- right shapes, silently
    wrong gradients.  This is the defect: `autotune.v3_grid`'s
    `MAXNREG` corners halved BLOCK_L behind `tune_block_l=False`, and the
    `P = 1` + `ckpt=True` backward was wrong by 2-80 % on dB/dC/ddelta/dA
    wherever the timing happened to pick the capped corner.  Pinning the two
    axes of that corner by hand is the deterministic form of the same cell.
    """
    from orbit_ssm.kernels.hybrid_mamba_mh.bwd_gluon import auto_block_l
    from orbit_ssm.kernels.hybrid_mamba_mh.functions import mh_scan
    stride = auto_block_l(1, length)
    args, kw = _case(length, route)
    actual = mh_scan(*args, **kw, ckpt=True,
                     mem=dict(BLOCK_L=max(8, stride // 2), MAXNREG=128))
    _compare(args, kw, actual, torch.randn_like(actual))
