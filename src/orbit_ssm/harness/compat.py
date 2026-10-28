
from __future__ import annotations
import torch

def mamba2_state_dict_to_ssd_mimo_mixer(official_sd: dict, *, d_inner: int, n_heads: int, d_state: int) -> dict:
    N, H = (d_state, n_heads)
    w = official_sd['in_proj.weight']
    z_rows, x_rows = (w[:d_inner], w[d_inner:2 * d_inner])
    b_rows = w[2 * d_inner:2 * d_inner + N]
    c_rows = w[2 * d_inner + N:2 * d_inner + 2 * N]
    dt_rows = w[2 * d_inner + 2 * N:2 * d_inner + 2 * N + H]
    conv_w = official_sd['conv1d.weight']
    conv_b = official_sd['conv1d.bias']
    mapped = {'in_proj.weight': torch.cat([x_rows, z_rows], dim=0).clone(), 'conv1d.weight': conv_w[:d_inner].clone(), 'conv1d.bias': conv_b[:d_inner].clone(), 'ssm.B_proj.weight': b_rows.clone(), 'ssm.C_proj.weight': c_rows.clone(), 'ssm.dt_proj.weight': dt_rows.clone(), 'ssm.dt_proj.bias': official_sd['dt_bias'].clone(), 'ssm.A_log': official_sd['A_log'].clone().unsqueeze(1), 'ssm.D': official_sd['D'].clone(), 'ssm.bc_conv1d.conv.weight': conv_w[d_inner:d_inner + 2 * N].clone(), 'ssm.bc_conv1d.conv.bias': conv_b[d_inner:d_inner + 2 * N].clone(), 'norm.weight': official_sd['norm.weight'].clone(), 'out_proj.weight': official_sd['out_proj.weight'].clone(), 'ssm.pair_layout_code': torch.zeros((), dtype=torch.int64, device=w.device)}
    return mapped

def copy_official_mamba2_into_ssd_mimo_mixer(official, mixer) -> list[str]:
    if mixer.ssm.d_skip_sharing != 'head':
        raise ValueError("Official default parameter parity requires d_skip_sharing='head'; use the named Mamba preset to preserve parameter tying.")
    d_inner = mixer.d_inner
    n_heads = mixer.ssm.H
    d_state = mixer.ssm.N
    mapped = mamba2_state_dict_to_ssd_mimo_mixer(official.state_dict(), d_inner=d_inner, n_heads=n_heads, d_state=d_state)
    params = dict(mixer.named_parameters())
    mixer.load_state_dict(mapped, strict=False)
    return sorted((n for n in params if n not in mapped))

def copy_official_mamba1_into_hm_mixer(official, hm) -> list[str]:
    dt_rank = official.dt_rank
    if hm.ssm.dt_low is None or hm.ssm.dt_low.weight.shape[0] != dt_rank:
        raise ValueError('M1 parameter parity needs dt_rank == official.dt_rank')
    pairs = {'in_proj.weight': official.in_proj.weight, 'conv1d.weight': official.conv1d.weight, 'conv1d.bias': official.conv1d.bias, 'ssm.dt_low.weight': official.x_proj.weight[:dt_rank], 'ssm.bc_proj.weight': official.x_proj.weight[dt_rank:], 'ssm.dt_proj.weight': official.dt_proj.weight, 'ssm.dt_bias': official.dt_proj.bias, 'ssm.A_log': official.A_log, 'ssm.D': official.D, 'out_proj.weight': official.out_proj.weight}
    if getattr(official.in_proj, 'bias', None) is not None or getattr(official.out_proj, 'bias', None) is not None:
        raise ValueError('M1 parameter parity: official in_proj/out_proj biases have no counterpart')
    params = dict(hm.named_parameters())
    with torch.no_grad():
        for name, src in pairs.items():
            dst = params[name]
            dst.copy_(src.detach().reshape(dst.shape))
    return sorted((n for n, p in params.items() if p.requires_grad and n not in pairs))

def mamba3_state_dict_to_ssd_mimo_mixer(official_sd: dict, *, d_inner: int, n_heads: int, d_state: int) -> dict:
    N, H = (d_state, n_heads)
    w = official_sd['in_proj.weight']
    d_in_proj = w.shape[0]
    ka = d_in_proj - 2 * d_inner - 2 * N - 3 * H
    if ka < 1:
        raise ValueError(f"mamba3_state_dict_to_ssd_mimo_mixer: inferred ka={ka} rotated pairs from in_proj.weight's row count ({d_in_proj}) at d_inner={d_inner}, d_state={N}, n_heads={H} -- check those match the official module")
    z_rows, x_rows = (w[:d_inner], w[d_inner:2 * d_inner])
    b_rows = w[2 * d_inner:2 * d_inner + N]
    c_rows = w[2 * d_inner + N:2 * d_inner + 2 * N]
    dt_rows = w[2 * d_inner + 2 * N:2 * d_inner + 2 * N + H]
    a_rows = w[2 * d_inner + 2 * N + H:2 * d_inner + 2 * N + 2 * H]
    trap_rows = w[2 * d_inner + 2 * N + 2 * H:2 * d_inner + 2 * N + 3 * H]
    angle_rows = w[2 * d_inner + 2 * N + 3 * H:]
    assert angle_rows.shape[0] == ka
    mapped = {'in_proj.weight': torch.cat([x_rows, z_rows], dim=0).clone(), 'ssm.B_proj.weight': b_rows.clone(), 'ssm.C_proj.weight': c_rows.clone(), 'ssm.dt_proj.weight': dt_rows.clone(), 'ssm.dt_proj.bias': official_sd['dt_bias'].clone(), 'ssm.a_dyn_proj.weight': a_rows.clone(), 'ssm.trap_proj.weight': trap_rows.clone(), 'ssm.theta_proj.weight': angle_rows.clone(), 'ssm.D': official_sd['D'].clone(), 'ssm.B_bias': official_sd['B_bias'].clone().transpose(1, 2).unsqueeze(1), 'ssm.C_bias': official_sd['C_bias'].clone().transpose(1, 2).unsqueeze(1), 'ssm.B_norm.weight': official_sd['B_norm.weight'].clone(), 'ssm.C_norm.weight': official_sd['C_norm.weight'].clone(), 'out_proj.weight': official_sd['out_proj.weight'].clone(), 'ssm.pair_layout_code': torch.zeros((), dtype=torch.int64, device=w.device)}
    return mapped

def mamba3_reference_a_act(official) -> str:
    import inspect
    provider = official if inspect.ismodule(official) else inspect.getmodule(type(official))
    return 'heavy_tail' if callable(getattr(provider, 'heavy_tail_activation', None)) else 'softplus'

def copy_official_mamba3_into_ssd_mimo_mixer(official, mixer) -> list[str]:
    act = mamba3_reference_a_act(official)
    if getattr(mixer.ssm, 'a_act', 'heavy_tail') != act:
        raise ValueError(f'Mamba-3 parameter parity: the reference uses the {act!r} |A| activation but the target has a_act={mixer.ssm.a_act!r}; build the target with a_act={act!r} (mamba3_reference_a_act) to compare.')
    if mixer.ssm.d_skip_sharing != 'head':
        raise ValueError("Official default parameter parity requires d_skip_sharing='head'; use the named Mamba preset to preserve parameter tying.")
    d_inner = mixer.d_inner
    n_heads = mixer.ssm.H
    d_state = mixer.ssm.N
    mapped = mamba3_state_dict_to_ssd_mimo_mixer(official.state_dict(), d_inner=d_inner, n_heads=n_heads, d_state=d_state)
    params = dict(mixer.named_parameters())
    mixer.load_state_dict(mapped, strict=False)
    with torch.no_grad():
        for name in ('ssm.trap_proj.bias', 'ssm.a_dyn_proj.bias'):
            if name in params:
                params[name].zero_()
    return sorted((n for n in params if n not in mapped))
