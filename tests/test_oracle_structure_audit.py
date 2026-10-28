"""Discriminating physical recurrences and continuation, independent of folds."""

import pytest
import torch
from _tolerance import assert_close_terms, terms_abs
from orbit_ssm import SSDMimoConfig


def test_ssd_uncapped_clock_matches_physical_oracle_and_step():
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    from test_ssd_endpoint import physical_oracle
    m=SSDMimoSSM(SSDMimoConfig(d_model=8,d_state=4,d_inner=8,d_conv=0,d_ssm_in=1,d_ssm_out=1,n_heads=2,n_groups=4,proj_source='main_branch',
                 write_rule='trapezoid',
                 row_blocks=[{'frac':1.0,'decay':'head_dynamic','complex':True}]), execution={'backend':'torch'}).double()
    with torch.no_grad():
        m.dt_proj.bias.fill_(1.)  # Makes capped and uncapped laws distinct.
    x=torch.randn(1,5,8,dtype=torch.double)
    h0=m.init_state(1,dtype=torch.double)['hidden_state']
    expected,_=physical_oracle(m,x,x,h0)
    t_ref=terms_abs(lambda v, h: physical_oracle(m,v,v,h)[0], x, h0).terms
    got=m.scan(x)
    if isinstance(got,tuple): got=got[0]
    t_got=terms_abs(lambda v: (lambda o: o[0] if isinstance(o,tuple) else o)(m.scan(v)), x).terms
    # A7: u=fp64 (module routes vs the fp64 physical oracle), bound
    # gamma(route) + gamma(oracle).  Worst-case depth per arm, L = 5, N = 4
    # (2 complex pairs), d_inner 8: token maps 8-dot + 1 (9), softplus clock
    # and exp decay argument (<= 2: 3), scan n = 2(L+1) (trapezoid taps),
    # m = 3L + 3 (2-term pair rotation + decay per token; u, B, dt), + L for
    # the cumulative-angle / cumsum-decay arguments, readout 4-dot + 1 (5),
    # D skip + gate (3).
    L = x.shape[1]
    c = 9 + 3 + 2 * (L + 1) + 3 * L + 3 + L + 5 + 3
    assert_close_terms(got, expected, t_got + t_ref, "fp64", c, name="forward")
    state=m.init_state(1,dtype=torch.double); ys=[]
    for token in x.unbind(1):
        y,state=m.scan_step(token,state);ys.append(y)

    def steps(v):
        st=m.init_state(1,dtype=torch.double); out=[]
        for token in v.unbind(1):
            y_,st=m.scan_step(token,st);out.append(y_)
        return torch.stack(out,1)

    assert_close_terms(torch.stack(ys,1), expected, terms_abs(steps, x).terms + t_ref,
                       "fp64", c, name="step")


def test_exact_presets_use_uncapped_clock_and_removed_option_is_rejected():
    """The clock is an uncapped softplus with no option."""
    from orbit_ssm.presets import mamba2_config, mamba3_config
    from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig, SSDMimoSSM
    for preset in (mamba2_config, mamba3_config):
        cfg = preset(64, d_state=32, d_head=32)
        assert not hasattr(cfg, 'dt_softclamp')
        SSDMimoSSM(cfg, execution={'backend': 'torch'})
    cfg = SSDMimoConfig(d_model=64, d_inner=128, d_state=32)
    assert 'dt_softclamp' not in cfg.layer_kwargs()
    for value in (False, True):
        with pytest.raises(TypeError, match='dt_softclamp'):
            SSDMimoConfig(dt_softclamp=value)
        with pytest.raises(TypeError, match='dt_softclamp'):
            SSDMimoSSM(SSDMimoConfig(d_model=128, dt_softclamp=value))
