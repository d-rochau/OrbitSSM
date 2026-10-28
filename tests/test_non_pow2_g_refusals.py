"""The Gluon Kbc body and the SISO kernels refuse a non-power-of-two head-group tile BY NAME instead of failing to compile.

EVIDENCE (contract adapter, Blackwell, `reg_cp_fwd_gt_g48` / `reg_gate_epilogue_g48`): the real_gluon_kbc lever (calibrated
ON) sent a G = 48 backward to `kbc_hb_gluon`, whose `gl.allocate_shared_memory(..., [T, GT])` needs power-of-two dimensions ("got 64,
48"); a forced SISO forward at G = 48 died in `tl.arange(0, G_SIZE)`.  PINS the two envelope clauses (CPU: the clauses sit before any
device query)."""
import torch

from orbit_ssm.kernels.ssd_mimo import gluon_kbc_hb as gk
from orbit_ssm.kernels.ssd_mimo import siso_route as sr


def _kbc(GT):
    return gk.refusal(device="cuda", mma_io=True, rot_in_kernel=False, rot_per_head=False, has_theta_h=False, has_rot_bias=False,
                      has_beta=False, has_gamma=False, pre_rot=False, pair_half=False, sumds=False, dd_in_kbc=False,
                      ka_nodalpha=False, kbc_split=False, rot_abl=False, ts_abl=0, T=64, N=128, GT=GT, nsplit=1)


def test_gluon_kbc_refuses_a_non_power_of_two_g_tile_by_name():
    why = _kbc(48)
    assert why is not None and "power of two" in why
    for gt in (16, 32, 64):
        assert "power of two" not in _safe(gt)


def _safe(gt):
    try:
        return _kbc(gt) or ""        # None = served (a GPU host reaches the device query and serves the call)
    except Exception as e:              # the device query after the clause (no CUDA here): the clause did not refuse
        return "reached the device query: %s" % type(e).__name__


def test_siso_refuses_a_non_power_of_two_g_by_name():
    x = torch.zeros(1, dtype=torch.bfloat16)
    why = sr.siso_extra_refusal(x_raw=x, precision="io", T=64, N=64, G_SIZE=48, pair_half=False, theta=None)
    assert why is not None and "power of two" in why
    assert sr.siso_extra_refusal(x_raw=x, precision="io", T=64, N=64, G_SIZE=64, pair_half=False, theta=None) is None


def test_gluon_kbc_tile_is_always_a_power_of_two():
    """`gluon_tile` ("full": the widest tile <= 64 dividing G, multiple of 16) must return a power of two: G = 48 is 16 x 3, not one tile of
    48 (the Gluon shared-memory tile raised "shape must have power-of-2 ... got 64, 48" for the Blackwell reg_cp_fwd_gt_g48 cell)."""
    assert gk_tile(16, 3) == (16, 3)           # G = 48
    assert gk_tile(48, 1) == (16, 3)
    assert gk_tile(32, 2) == (64, 1)           # G = 64
    assert gk_tile(32, 1) == (32, 1)
    assert gk_tile(16, 5) == (16, 5)           # G = 80
    assert gk_tile(24, 2, "bwd") == (24, 2)    # "bwd" keeps the backward's tile (refused earlier by the power-of-two clause)
    assert gk_tile(16, 3, "48") == (16, 3)     # an explicit non-power-of-two request is ignored
    for g in (16, 32, 48, 64, 80, 96, 112, 128):
        gt, ngt = gk_tile(16, g // 16)
        assert gt & (gt - 1) == 0 and gt * ngt == g and gt % 16 == 0, (g, gt, ngt)


gk_tile = gk.gluon_tile
