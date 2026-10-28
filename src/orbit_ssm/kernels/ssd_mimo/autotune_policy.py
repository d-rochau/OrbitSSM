
def autotune_tier(value: str) -> str:
    tier = value.strip().lower()
    tier = {'off': 'pinned', 'auto': 'light'}.get(tier, tier)
    if tier not in ('pinned', 'light', 'full'):
        raise ValueError(f"SSD_MIMO_AUTOTUNE must be 'pinned', 'light' or 'full' (or the legacy 'off'/'auto'), got {value!r}")
    return tier

def describe_autotune() -> dict:
    import os
    from . import fused_bwd_kernel as fused
    from . import siso_fwd_kernel as forward
    from . import siso_bwd_kernel as backward
    from .fused_fwd_kernel import _env

    def configs(items):
        return [dict(warps=c.num_warps, stages=c.num_stages, maxnreg=c.maxnreg, kwargs=dict(c.kwargs)) for c in items]
    requested = autotune_tier(_env('AUTOTUNE', 'light'))
    grids = {'fused_backward': configs(fused._BWD_CFGS), 'fused_head_backward': configs(fused._HB_CFGS), 'fused_head_reduce_backward': configs(fused._KB2_HB_CFGS), 'siso_forward': configs(forward._siso_fwd_kernel.configs), 'siso_backward': configs(backward._siso_bwd_scan_kernel.configs), 'siso_rotation_adjoint': configs(backward._siso_rot_adjoint_kernel.configs)}
    return {'requested_global_tier': requested, 'imported_global_tier': fused._AUTOTUNE_TIER, 'request_matches_import': requested == fused._AUTOTUNE_TIER, 'grid_scope': 'process import; per-call options do not rebuild grids', 'grids': grids, 'current_overrides': {name: _env(name, '') or None for name in ('SISO_AUTOTUNE', 'SISO_ADJ_AUTOTUNE', 'SISO_BWD_ONECFG', 'SISO_ADJ_WARPS', 'SISO_ADJ_STAGES')}, 'fixed_gluon_backward': {'warp_groups': int(os.environ.get('SSD_MIMO_GLUON90_SISO_WN', '2')), 'pipeline_slots': int(os.environ.get('SSD_MIMO_GLUON90_SISO_STAGES', '2')), 'autotuned': False}}

def siso_autotune_tier(value: str, *, adjoint: bool=False) -> str:
    tier = value.strip().lower()
    aliases = {'0': 'pinned', '1': 'light' if adjoint else 'full'}
    if not adjoint:
        aliases['fast'] = 'light'
    tier = aliases.get(tier, tier)
    if tier not in ('pinned', 'light', 'full'):
        name = 'SSD_MIMO_SISO_ADJ_AUTOTUNE' if adjoint else 'SSD_MIMO_SISO_AUTOTUNE'
        legacy = '0/1' if adjoint else '0/1/fast'
        raise ValueError(f"{name} must be 'pinned', 'light' or 'full' (or the legacy {legacy}), got {value!r}")
    return tier
