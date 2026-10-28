
from __future__ import annotations
import argparse
from typing import Any, Callable
import torch
from . import baselines as bl
from . import shapes as sh
from .harness import KERNEL_PRECISION, MODULE_PARAM_DTYPE, Cell, amp_ctx, cuda_buckets, group_key, main_for, make_pass
from .measure import precision_ctx
GRIDS = ('smoke', 'ssm_mqmkvar', 'ssm_lm_spine', 'ssm_shared_set', 'ssm_shared_small', 'ssm_shared_medium', 'ssm_shared_large', 'ssm_shared_lm')
UNION_GRIDS = frozenset({'ssm_shared_set'})
FAMILIES = ('ssd_mimo', 'hybrid_mamba')
_BARE_CORE_FAMILIES = frozenset({'ssd_mimo', 'hybrid_mamba'})

def _hm_pin_no_bias(kw: dict) -> None:
    if 'bias_gran' not in kw and 'bias' not in (kw.get('maps') or {}):
        kw['bias_gran'] = 'none'

def _config(family: str, d_inner: int, d_state: int, **over):
    from orbit_ssm.config import HybridMambaConfig, SSDMimoConfig
    common = dict(d_inner=d_inner, d_state=d_state, backend='cuda')
    if family == 'ssd_mimo':
        R = over.pop('R', 1)
        O = over.pop('O', R)
        ngroups = over.pop('ngroups', max(16 // R, 1))
        return SSDMimoConfig(**common, R=R, O=O, ngroups=ngroups, proj_source='main_branch', **over)
    if family == 'hybrid_mamba':
        _hm_pin_no_bias(over)
        return HybridMambaConfig(**common, n_heads=over.pop('n_heads', max(d_inner // 16, 1)), b_norm=over.pop('b_norm', False), c_norm=over.pop('c_norm', False), proj_source='main_branch', **over)
    raise ValueError(f'unknown family {family!r}')

def build_mixer(family: str, d_inner: int, d_state: int, B: int, L: int, *, dtype=torch.float32, device='cuda', seed=0, **over):
    from orbit_ssm.config import build_ssm
    torch.manual_seed(seed)
    mod = build_ssm(_config(family, d_inner, d_state, **over))
    mod = mod.to(device=device, dtype=dtype)
    x = torch.randn(B, L, d_inner, device=device, dtype=dtype, requires_grad=True)

    def call():
        return mod(x)
    return (call, [x, *mod.parameters()], mod)

def build_full_mixer(family: str, d_model: int, d_state: int, B: int, L: int, *, dtype=torch.float32, device='cuda', seed=0, expand: float=2.0, d_conv: int=4, mixer_kwargs: dict | None=None):
    from orbit_ssm.model import HybridMambaMixer, SSDMimoMixer
    torch.manual_seed(seed)
    kw = dict(mixer_kwargs or {})
    if family == 'ssd_mimo':
        mode = kw.pop('mode', None)
        modular_checkpoint = kw.pop('modular_checkpoint', False)
        mod = SSDMimoMixer(d_model, d_state=d_state, expand=expand, d_conv=d_conv, mode=mode, modular_checkpoint=modular_checkpoint, ssm_kwargs=kw)
    elif family == 'hybrid_mamba':
        kw.setdefault('b_norm', False)
        kw.setdefault('c_norm', False)
        _hm_pin_no_bias(kw)
        mod = HybridMambaMixer(d_model, d_state=d_state, expand=expand, d_conv=d_conv, ssm_kwargs=kw)
    else:
        raise ValueError(f'build_full_mixer: {family!r} already owns its full projection stack -- use build_mixer/_ssm_cell instead')
    mod = mod.to(device=device, dtype=dtype)
    x = torch.randn(B, L, d_model, device=device, dtype=dtype, requires_grad=True)

    def call():
        return mod(x)
    return (call, [x, *mod.parameters()], mod)

def _cpu_param_count(build_module: Callable[[], Any]) -> int:
    mod = build_module()
    return sum((p.numel() for p in mod.parameters()))

def _ssm_cell(grid, family, arm, d_inner, d_state, B, L, precision, pass_, *, work, tags=None, **over):
    amp = amp_ctx(precision)

    def build():
        call, leaves, mod = build_mixer(family, d_inner, d_state, B, L, dtype=MODULE_PARAM_DTYPE, **over)
        _PARAMS[f'{family}/{arm}/{d_inner}'] = sum((p.numel() for p in mod.parameters()))
        return make_pass(call, leaves, pass_, amp=amp)
    return Cell(family='ssm', arm=arm, entry_point=f'orbit_ssm.config.build_ssm({family})', shape=dict(B=B, L=L, d_inner=d_inner, N=d_state, **{k: v for k, v in over.items() if isinstance(v, int)}), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), autocast=precision == 'bf16', tags=dict(tags or {}, mixer=family, mixer_level='core' if family in _BARE_CORE_FAMILIES else 'full'), build=build)

def _full_mixer_cell(grid, family, arm, d_model, d_state, B, L, precision, pass_, *, work, expand: float=2.0, tags=None, graphed=False, **mixer_kwargs):
    amp = amp_ctx(precision)

    def build():
        call, leaves, mod = build_full_mixer(family, d_model, d_state, B, L, dtype=MODULE_PARAM_DTYPE, expand=expand, mixer_kwargs=dict(mixer_kwargs))
        _PARAMS[f'{family}/{arm}/{d_model}'] = sum((p.numel() for p in mod.parameters()))
        return _graphed_closure(mod, leaves, pass_, precision) if graphed else make_pass(call, leaves, pass_, amp=amp)
    d_inner = int(d_model * expand)
    wrapper_cls = 'SSDMimoMixer' if family == 'ssd_mimo' else 'HybridMambaMixer'
    return Cell(family='ssm', arm=arm, entry_point=f'orbit_ssm.model.{wrapper_cls}', shape=dict(B=B, L=L, d_model=d_model, d_inner=d_inner, N=d_state, **{k: v for k, v in mixer_kwargs.items() if isinstance(v, int)}), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), autocast=precision == 'bf16', tags=dict(tags or {}, mixer=family, mixer_level='full'), build=build)
_GRAPHED_LIKE_FOR_LIKE = ('SM-R1:m2exact', 'SM-R1:m3exact', 'HMM:m1exact', 'official_mamba1', 'official_mamba2', 'official_mamba3')
GRAPH_ARM_SUFFIX = '@graph'
COMPILE_ARM_SUFFIX = '@compile'

def _compiled_closure(mod, leaves, pass_, precision, amp, arm):
    import time
    import torch._dynamo
    torch._dynamo.reset()
    x = leaves[0]
    try:
        cmod = torch.compile(mod)
        t0 = time.perf_counter()
        fn = make_pass(lambda: cmod(x), leaves, pass_, amp=amp)
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
    except Exception as exc:
        raise RuntimeError(f'{arm}: torch.compile failed ({type(exc).__name__}): {str(exc).splitlines()[0][:300]}') from exc
    fn.compile_seconds = secs
    fn.compile_mode = 'default'
    return fn

def _with_module(mod, leaves, amp, pass_, fn):
    x = leaves[0]

    def eager():
        if pass_ == 'fwd':
            with torch.no_grad(), amp():
                return mod(x)
        with amp():
            y = mod(x)
        y.float().square().mean().backward()
        for q in leaves:
            if isinstance(q, torch.Tensor):
                q.grad = None
        return y
    fn.module = mod
    fn.eager = eager
    return fn

def _graphed_closure(mod, leaves, pass_, precision):
    from orbit_ssm.harness import graphs as gph
    x = leaves[0]
    gs = gph.graphed_step(mod, (x,), fwd_bwd=pass_ == 'fwd_bwd', amp=lambda: gph.autocast(precision), warmup=11)

    def run():
        gs.graph.replay()
        return gs.static_output
    run.graphed_step = gs
    return run

def _official_cell(grid, variant, d_model, d_state, B, L, precision, pass_, *, work, headdim=64, ngroups=1, expand=2, tags=None, arm=None, graphed=False, compiled=False):
    amp = amp_ctx(precision)
    arm = arm or f'official_{variant}'
    if graphed:
        arm += GRAPH_ARM_SUFFIX
    if compiled:
        arm += COMPILE_ARM_SUFFIX

    def build():
        call, leaves, mod = bl.build_mamba_layer(variant, d_model, d_state=d_state, L=L, B=B, headdim=headdim, ngroups=ngroups, expand=expand, dtype=MODULE_PARAM_DTYPE)
        _PARAMS[f'official/{variant}/{d_model}'] = sum((p.numel() for p in mod.parameters()))
        return _with_module(mod, leaves, amp, pass_, _graphed_closure(mod, leaves, pass_, precision) if graphed else _compiled_closure(mod, leaves, pass_, precision, amp, arm) if compiled else make_pass(call, leaves, pass_, amp=amp))
    return Cell(family='ssm', arm=arm, entry_point=f'mamba_ssm.{variant}', shape=dict(B=B, L=L, d_inner=d_model * expand, N=d_state, d_model=d_model), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), autocast=precision == 'bf16', tags=dict(tags or {}, mixer=variant, **{'capture': 'cuda_graph'} if graphed else {}, **{'capture': 'torch_compile'} if compiled else {}), build=build)
_PARAMS: dict[str, int] = {}
_EXACT_PRESET = {'m1exact': 'mamba1', 'm2exact': 'mamba2', 'm3exact': 'mamba3'}
_EXACT_FAMILY = {'m1exact': 'hybrid_mamba', 'm2exact': 'ssd_mimo', 'm3exact': 'ssd_mimo'}

def build_preset_mixer(config_name: str, d_model: int, d_state: int, B: int, L: int, *, dtype=torch.float32, device='cuda', seed=0, headdim: int=64, **overrides):
    from orbit_ssm.harness import presets
    from orbit_ssm.model import HybridMambaMixer, SSDMimoMixer
    if config_name not in _EXACT_PRESET:
        raise ValueError(f'unknown exact config {config_name!r}, must be one of {sorted(_EXACT_PRESET)}')
    torch.manual_seed(seed)
    over = dict(overrides)
    mixer_mode = over.pop('mode', None)
    over.setdefault('backend', 'cuda')
    if config_name == 'm1exact':
        kw = presets.mamba1_kwargs(d_model, d_state=d_state, **over)
        mod = HybridMambaMixer(**kw)
    else:
        preset_fn = presets.mamba2_kwargs if config_name == 'm2exact' else presets.mamba3_kwargs
        kw = preset_fn(d_model, d_state=d_state, headdim=headdim, **over)
        mod = SSDMimoMixer(**kw, mode=mixer_mode)
    mod = mod.to(device=device, dtype=dtype)
    x = bl.module_input(B, L, d_model, device=device, dtype=dtype, seed=seed)

    def call():
        return mod(x)
    return (call, [x, *mod.parameters()], mod)
_OFFICIAL_OF = {'m1exact': 'mamba1', 'm2exact': 'mamba2', 'm3exact': 'mamba3'}
PAIR_GATE = {'bf16': (0.02, 0.05), 'tf32': (0.01, 0.03), 'ieee': (0.0001, 0.001)}
_PAIR_GATE_CACHE: dict = {}

def build_exact_pair(config_name: str, d_model: int, d_state: int, B: int, L: int, *, dtype=torch.float32, device='cuda', seed=0, headdim: int=64, **overrides):
    from orbit_ssm.harness import compat
    variant = _OFFICIAL_OF[config_name]
    _, _, official = bl.build_mamba_layer(variant, d_model, d_state=d_state, L=1, B=1, headdim=headdim, ngroups=1, expand=2, dtype=dtype, device=device, seed=seed)
    over = dict(overrides)
    if config_name == 'm3exact':
        over.setdefault('a_act', compat.mamba3_reference_a_act(official))
    _, _, ours = build_preset_mixer(config_name, d_model, d_state, 1, 1, dtype=dtype, device=device, seed=seed, headdim=headdim, **over)
    if config_name == 'm1exact':
        unmapped = compat.copy_official_mamba1_into_hm_mixer(official, ours)
    elif config_name == 'm2exact':
        unmapped = compat.copy_official_mamba2_into_ssd_mimo_mixer(official, ours)
    else:
        unmapped = compat.copy_official_mamba3_into_ssd_mimo_mixer(official, ours)
    if unmapped:
        raise RuntimeError(f'{config_name}: trainable parameters without an official counterpart: {unmapped}')
    return (official, ours)

def _rel(a, b) -> float:
    a, b = (a.detach().float(), b.detach().float())
    return float((a - b).norm() / b.norm().clamp_min(1e-30))

def exact_pair_gate(config_name, d_model, d_state, B, L, column, *, seed=0, headdim=64, **overrides) -> dict:
    precision = column
    key = (config_name, d_model, d_state, B, L, precision, headdim, tuple(sorted(overrides.items())))
    if key in _PAIR_GATE_CACHE:
        res = _PAIR_GATE_CACHE[key]
        if isinstance(res, str):
            raise AssertionError(res)
        return res
    official, ours = build_exact_pair(config_name, d_model, d_state, B, L, dtype=MODULE_PARAM_DTYPE, seed=seed, headdim=headdim, **overrides)
    amp = amp_ctx(precision)
    x = bl.module_input(B, L, d_model, device='cuda', dtype=MODULE_PARAM_DTYPE, seed=seed)
    out = {}
    dy = None
    with precision_ctx(precision):
        for name, mod in (('official', official), ('ours', ours)):
            xi = x.detach().clone().requires_grad_(True)
            with amp():
                y = mod(xi)
            if dy is None:
                g = torch.Generator(device='cuda').manual_seed(seed + 7)
                dy = torch.randn(y.shape, generator=g, device='cuda', dtype=torch.float32).to(y.dtype)
            y.backward(dy)
            out[name] = dict(y=y.detach().float(), dx=xi.grad.detach().float(), with_grad=sorted((n for n, q in mod.named_parameters() if q.grad is not None)), n_params=sum((1 for q in mod.parameters() if q.requires_grad)))
            del y, xi
    tol_y, tol_dx = PAIR_GATE[precision]
    res = dict(pair=f'{config_name}~official_{_OFFICIAL_OF[config_name]}', rel_y=_rel(out['ours']['y'], out['official']['y']), rel_dx=_rel(out['ours']['dx'], out['official']['dx']), tol_y=tol_y, tol_dx=tol_dx, grads_ours=f"{len(out['ours']['with_grad'])}/{out['ours']['n_params']}", grads_official=f"{len(out['official']['with_grad'])}/{out['official']['n_params']}", a_act=ours.ssm.a_act if config_name == 'm3exact' else None, weights='copied')
    del official, ours, out
    torch.cuda.empty_cache()
    bad = []
    if not res['rel_y'] < tol_y:
        bad.append(f"y rel {res['rel_y']:.3e} >= {tol_y}")
    if not res['rel_dx'] < tol_dx:
        bad.append(f"dx rel {res['rel_dx']:.3e} >= {tol_dx}")
    for side in ('ours', 'official'):
        have, tot = res[f'grads_{side}'].split('/')
        if have != tot:
            bad.append(f'{side}: only {have}/{tot} parameters received a gradient')
    if bad:
        msg = f"parity gate {res['pair']} B{B} L{L} d{d_model} N{d_state} {precision}: " + '; '.join(bad)
        _PAIR_GATE_CACHE[key] = msg
        raise AssertionError(msg)
    _PAIR_GATE_CACHE[key] = res
    return res

def _exact_mixer_cell(grid, arm, config_name, d_model, d_state, B, L, precision, pass_, *, work, headdim=64, tags=None, config_tag=None, graphed=False, compiled=False, **overrides):
    amp = amp_ctx(precision)
    family = _EXACT_FAMILY[config_name]
    if graphed:
        arm += GRAPH_ARM_SUFFIX
    if compiled:
        arm += COMPILE_ARM_SUFFIX
    cell_runtime: dict = {}

    def build():
        over = dict(overrides)
        if family == 'ssd_mimo':
            over.setdefault('precision', KERNEL_PRECISION[precision])
        gate = exact_pair_gate(config_name, d_model, d_state, B, L, precision, headdim=headdim, **over)
        official, mod = build_exact_pair(config_name, d_model, d_state, B, L, dtype=MODULE_PARAM_DTYPE, headdim=headdim, **over)
        del official
        x = bl.module_input(B, L, d_model, device='cuda', dtype=MODULE_PARAM_DTYPE, seed=0)
        call, leaves = (lambda: mod(x), [x, *mod.parameters()])
        cell_runtime['parity_gate'] = gate
        _PARAMS[f'{family}/{arm}/{d_model}'] = sum((p.numel() for p in mod.parameters()))
        return _with_module(mod, leaves, amp, pass_, _graphed_closure(mod, leaves, pass_, precision) if graphed else _compiled_closure(mod, leaves, pass_, precision, amp, arm) if compiled else make_pass(call, leaves, pass_, amp=amp))
    wrapper_cls = 'HybridMambaMixer' if family == 'hybrid_mamba' else 'SSDMimoMixer'
    preset_name = _EXACT_PRESET[config_name]
    return Cell(family='ssm', arm=arm, entry_point=f'orbit_ssm.harness.presets.{preset_name}_kwargs -> orbit_ssm.model.{wrapper_cls}', shape=dict(B=B, L=L, d_model=d_model, d_inner=d_model * 2, N=d_state, headdim=headdim), precision=precision, pass_=pass_, grid=grid, group=group_key(grid, work, precision, pass_), autocast=precision == 'bf16', tags=dict(tags or {}, mixer=family, mixer_level='full', config=config_tag or config_name, **{'capture': 'cuda_graph'} if graphed else {}, **{'capture': 'torch_compile'} if compiled else {}), build=build, runtime=cell_runtime)

def _exact_arm_param_audit(d_model: int, d_state: int, headdim: int, B: int, L: int, *, d_state_m1: int=16) -> dict[str, dict]:
    B = L = 1
    from orbit_ssm.harness import presets
    from orbit_ssm.model import HybridMambaMixer, SSDMimoMixer
    ds = {'mamba1': d_state_m1, 'mamba2': d_state, 'mamba3': d_state}
    official_n: dict[str, int] = {}
    for variant in ('mamba1', 'mamba2', 'mamba3'):
        official_n[variant] = _cpu_param_count(lambda variant=variant: bl.build_mamba_layer(variant, d_model, d_state=ds[variant], L=L, B=B, headdim=headdim, ngroups=1, expand=2, dtype=MODULE_PARAM_DTYPE, device='cpu')[2])
    builders = {'m1exact': lambda: HybridMambaMixer(**presets.mamba1_kwargs(d_model, d_state=d_state_m1)), 'm2exact': lambda: SSDMimoMixer(**presets.mamba2_kwargs(d_model, d_state=d_state, headdim=headdim)), 'm3exact': lambda: SSDMimoMixer(**presets.mamba3_kwargs(d_model, d_state=d_state, headdim=headdim))}
    tags: dict[str, dict] = {}
    for name, build_fn in builders.items():
        variant = _EXACT_PRESET[name]
        n = _cpu_param_count(build_fn)
        target = official_n[variant]
        ratio = n / target
        ok = abs(ratio - 1.0) <= 0.005
        try:
            assert ok, f"{name}: {n} params vs official_{variant} target {target} (ratio {ratio:.4f}) is outside ruling I6's +-0.5% exact-config param-match gate"
        except AssertionError as exc:
            print(f'[param-match] WARNING: {exc}', flush=True)
        tags[name] = dict(config=name, param_match_target_arm=f'official_{variant}', param_match_target_n=target, param_match_n=n, param_match_ratio=round(ratio, 4), param_match_ok=ok, d_state=ds[variant])
    return tags
_HM_COMPLEX_A = ({'frac': 1.0, 'decay': 'A', 'complex': True, 'gate': True},)
_HM_GATES_POST = dict(use_g_b=True, use_g_c=True, gate_act='sigmoid', gate_pos='post')
_HM_FEATURE_OVERRIDES: dict[str, dict] = {'rot': dict(rows=_HM_COMPLEX_A), 'gates': dict(_HM_GATES_POST), 'all': dict(rows=_HM_COMPLEX_A, **_HM_GATES_POST, bias_gran='head_bc', gate_bias=True), 'all1': dict(rows=_HM_COMPLEX_A, use_g_b=True, use_g_c=True, gate_act='sigmoid', gate_pos='pre', bias_gran='shared', gate_bias=True)}
_SSD_COMPLEX_STATIC = ({'frac': 1.0, 'decay': 'head_static', 'complex': True, 'gate': True},)
_SSD_FEATURE_OVERRIDES: dict[str, dict] = {'rot': dict(rows=_SSD_COMPLEX_STATIC, bias_gran='head_bc', b_norm=True, c_norm=True), 'gates': dict(use_g_b=True, use_g_c=True), 'all': dict(rows=_SSD_COMPLEX_STATIC, use_g_b=True, use_g_c=True, bias_gran='head_bc', b_norm=True, c_norm=True)}
_FEATURE_NAMES = ('rot', 'gates', 'all')
_HM_FEATURE_NAMES = _FEATURE_NAMES + ('all1',)

def _feature_tags(base_arm: str, base_n: int, feature: str, build_fn: Callable[[], Any], **extra) -> dict:
    n = _cpu_param_count(build_fn)
    return dict(extra, feature=feature, base_arm=base_arm, base_param_n=base_n, param_n=n, param_ratio=round(n / max(base_n, 1), 4))

def _hm_exact_feature_cells(g: str, d_model: int, d_state: int, B: int, L: int, prec: str, p: str, w: str, base_n: int) -> list[Cell]:
    from orbit_ssm.harness import presets
    from orbit_ssm.model import HybridMambaMixer
    cells: list[Cell] = []
    for feat in _HM_FEATURE_NAMES:
        over = _HM_FEATURE_OVERRIDES[feat]
        arm = f'HMM:m1exact+{feat}'
        tags = _feature_tags('HMM:m1exact', base_n, feat, lambda over=over: HybridMambaMixer(**presets.mamba1_kwargs(d_model, d_state=d_state, **over)))
        cells.append(_exact_mixer_cell(g, arm, 'm1exact', d_model, d_state, B, L, prec, p, work=w, tags=tags, **over))
    return cells

def _hm_design_feature_cells(g: str, arm_base: str, d_model: int, n_heads: int, d_state: int, B: int, L: int, prec: str, p: str, w: str, base_n: int, feature_names: tuple[str, ...]=_FEATURE_NAMES) -> list[Cell]:
    cells: list[Cell] = []
    for feat in feature_names:
        over = _HM_FEATURE_OVERRIDES[feat]
        arm = f'{arm_base}+{feat}'
        mixer_kwargs = dict(n_heads=n_heads, proj_source='main_branch', **over)
        tags = _feature_tags(arm_base, base_n, feat, lambda mixer_kwargs=mixer_kwargs: build_full_mixer('hybrid_mamba', d_model, d_state, 1, 1, dtype=MODULE_PARAM_DTYPE, mixer_kwargs=dict(mixer_kwargs), device='cpu')[2], config='default')
        cells.append(_full_mixer_cell(g, 'hybrid_mamba', arm, d_model, d_state, B, L, prec, p, work=w, tags=tags, **mixer_kwargs))
    return cells

def _ssd_exact_feature_cells(g: str, d_model: int, d_state: int, B: int, L: int, prec: str, p: str, w: str, headdim: int, base_n: int) -> list[Cell]:
    from orbit_ssm.harness import presets
    from orbit_ssm.model import SSDMimoMixer
    cells: list[Cell] = []
    for feat in _FEATURE_NAMES:
        over = _SSD_FEATURE_OVERRIDES[feat]
        arm = f'SM-R1:m2exact+{feat}'
        tags = _feature_tags('SM-R1:m2exact', base_n, feat, lambda over=over: SSDMimoMixer(**presets.mamba2_kwargs(d_model, d_state=d_state, headdim=headdim, **over)))
        cells.append(_exact_mixer_cell(g, arm, 'm2exact', d_model, d_state, B, L, prec, p, work=w, headdim=headdim, tags=tags, **over))
    return cells

def _graphed_like_for_like_cells(g, d_model, B, L, prec, p, w, exact_tags):
    if prec != 'bf16':
        return []
    cells = [_exact_mixer_cell(g, 'SM-R1:m2exact', 'm2exact', d_model, sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64, tags=exact_tags['m2exact'], graphed=True), _exact_mixer_cell(g, 'SM-R1:m3exact', 'm3exact', d_model, sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64, tags=exact_tags['m3exact'], graphed=True), _exact_mixer_cell(g, 'HMM:m1exact', 'm1exact', d_model, sh.EXACT_D_STATE_M1, B, L, prec, p, work=w, tags=exact_tags['m1exact'], graphed=True)]
    for variant, ds in (('mamba1', sh.EXACT_D_STATE_M1), ('mamba2', sh.EXACT_D_STATE_M23), ('mamba3', sh.EXACT_D_STATE_M23)):
        cells.append(_official_cell(g, variant, d_model, ds, B, L, prec, p, work=w, headdim=64, tags=dict(config='official'), graphed=True))
    return cells

def grid_smoke(graphed: bool=False) -> list[Cell]:
    g = 'smoke'
    s = sh.SMOKE_SSM
    d, n, B, L = (s['d_model'], s['d_state'], s['B'], s['L'])
    w = f'B{B}L{L}d{d}N{n}'
    cells: list[Cell] = []
    for prec in ('tf32', 'bf16'):
        for p in ('fwd', 'fwd_bwd'):
            for fam in FAMILIES:
                over: dict[str, Any] = {}
                if fam == 'ssd_mimo':
                    over = dict(R=2, O=2, ngroups=8)
                elif fam == 'hybrid_mamba':
                    over = dict(n_heads=8)
                cells.append(_ssm_cell(g, fam, fam, d, n, B, L, prec, p, work=w, **over))
            for variant, hd in (('mamba1', 64), ('mamba2', 32), ('mamba3', 32)):
                cells.append(_official_cell(g, variant, d, n, B, L, prec, p, work=w, headdim=hd))
    return cells

def grid_ssm_mqmkvar(graphed: bool=False) -> list[Cell]:
    g = 'ssm_mqmkvar'
    B, L = (sh.MQMKVAR_SHAPE['B'], sh.MQMKVAR_SHAPE['L'])
    n_mhr = sh.MQMKVAR_SHAPE['N_mhr']
    d_model = sh.MQMKVAR_SHAPE['d_model']
    w = f'B{B}L{L}d{d_model}'
    other = {a['id']: a for a in sh.MQMKVAR_OTHER_ARMS}
    hm = other['HM']
    exact_tags = _exact_arm_param_audit(d_model, sh.EXACT_D_STATE_M23, 64, B, L, d_state_m1=sh.EXACT_D_STATE_M1)
    exact_tags_n64 = _exact_arm_param_audit(d_model, sh.EXACT_D_STATE_M23_N64, 64, B, L, d_state_m1=sh.EXACT_D_STATE_M1)
    hmm_n_heads = hm['d_inner'] // 64
    hmm_base_n = _cpu_param_count(lambda: build_full_mixer('hybrid_mamba', hm['d_model'], n_mhr, 1, 1, dtype=MODULE_PARAM_DTYPE, device='cpu', mixer_kwargs=dict(n_heads=hmm_n_heads, proj_source='main_branch'))[2])

    def wtags(arm: dict, **extra) -> dict:
        return dict(extra, d_model=arm['d_model'], width_convention=arm['width_convention'])
    cells: list[Cell] = []
    for prec in ('ieee', 'tf32', 'bf16'):
        for p in ('fwd', 'fwd_bwd'):
            for arm in sh.MQMKVAR_S_ARMS:
                cells.append(_ssm_cell(g, 'ssd_mimo', f"core:{arm['id']}", arm['d_inner'], n_mhr, B, L, prec, p, work=w, tags=dict(spec_arm=arm['id'], d_model=arm['width'], width_convention='spec', config='default'), R=arm['R'], O=arm['O'], ngroups=arm['ngroups'], rows=arm['rows'], r_mix_dyn=arm['R'] > 1))
                cells.append(_full_mixer_cell(g, 'ssd_mimo', f"SM{arm['id'][1:]}", arm['width'], n_mhr, B, L, prec, p, work=w, tags=dict(spec_arm=arm['id'], d_model=arm['width'], width_convention='spec', config='default'), R=arm['R'], O=arm['O'], ngroups=arm['ngroups'], rows=arm['rows'], r_mix_dyn=arm['R'] > 1, proj_source='main_branch'))
            cells.append(_ssm_cell(g, hm['family'], 'core:HM', hm['d_inner'], n_mhr, B, L, prec, p, work=w, tags=wtags(hm, config='default'), n_heads=hm['d_inner'] // hm['head_dim']))
            cells.append(_full_mixer_cell(g, 'hybrid_mamba', 'HMM', hm['d_model'], n_mhr, B, L, prec, p, work=w, tags=wtags(hm, head_dim=64, config='default'), n_heads=hm['d_inner'] // 64, proj_source='main_branch'))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m2exact', 'm2exact', d_model, sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64, tags=exact_tags['m2exact']))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m2exact@N64', 'm2exact', d_model, sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64, config_tag='m2exact_n64', tags=dict(exact_tags_n64['m2exact'], config='m2exact_n64')))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m3exact', 'm3exact', d_model, sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64, tags=exact_tags['m3exact']))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m3exact@N64', 'm3exact', d_model, sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64, config_tag='m3exact_n64', tags=dict(exact_tags_n64['m3exact'], config='m3exact_n64')))
            cells.append(_exact_mixer_cell(g, 'HMM:m1exact', 'm1exact', d_model, sh.EXACT_D_STATE_M1, B, L, prec, p, work=w, tags=exact_tags['m1exact']))
            cells.extend(_hm_exact_feature_cells(g, d_model, sh.EXACT_D_STATE_M1, B, L, prec, p, w, exact_tags['m1exact']['param_match_n']))
            cells.extend(_hm_design_feature_cells(g, 'HMM', hm['d_model'], hmm_n_heads, n_mhr, B, L, prec, p, w, hmm_base_n))
            cells.extend(_ssd_exact_feature_cells(g, d_model, sh.EXACT_D_STATE_M23, B, L, prec, p, w, 64, exact_tags['m2exact']['param_match_n']))
            for variant, hd, ds in (('mamba1', 64, sh.EXACT_D_STATE_M1), ('mamba2', 64, sh.EXACT_D_STATE_M23), ('mamba3', 64, sh.EXACT_D_STATE_M23)):
                cells.append(_official_cell(g, variant, d_model, ds, B, L, prec, p, work=w, headdim=hd, tags=dict(config='official')))
            for variant, hd in (('mamba2', 64), ('mamba3', 64)):
                cells.append(_official_cell(g, variant, d_model, sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=hd, arm=f'official_{variant}@N64', tags=dict(config='official_n64')))
            if graphed:
                cells.extend(_graphed_like_for_like_cells(g, d_model, B, L, prec, p, w, exact_tags))
    return cells

def _lm_spine_param_audit(D: int, N: int, B: int, L: int) -> dict[str, dict]:
    B = L = 1
    d_model_off = D // 2
    m2_target = _cpu_param_count(lambda: bl.build_mamba_layer('mamba2', d_model_off, d_state=N, L=L, B=B, headdim=64, ngroups=1, expand=2, dtype=MODULE_PARAM_DTYPE, device='cpu')[2])
    specs = {'SM-R1': dict(family='ssd_mimo', d_model=d_model_off, d_state=N, mixer_kwargs=dict(R=1, O=1, ngroups=64, proj_source='main_branch')), 'SM-R2O2': dict(family='ssd_mimo', d_model=d_model_off, d_state=128, mixer_kwargs=dict(R=2, O=2, ngroups=64, proj_source='main_branch', r_mix_dyn=True)), 'SM-R4O4': dict(family='ssd_mimo', d_model=d_model_off, d_state=128, mixer_kwargs=dict(R=4, O=4, ngroups=128, proj_source='main_branch', r_mix_dyn=True)), 'HMM-LM': dict(family='hybrid_mamba', d_model=d_model_off, d_state=N, mixer_kwargs=dict(n_heads=D // 64, proj_source='main_branch'))}
    tags: dict[str, dict] = {}
    for name, spec in specs.items():
        n = _cpu_param_count(lambda spec=spec: build_full_mixer(spec['family'], spec['d_model'], spec['d_state'], B, L, dtype=MODULE_PARAM_DTYPE, mixer_kwargs=spec['mixer_kwargs'], device='cpu')[2])
        ratio = n / m2_target
        ok = abs(ratio - 1.0) <= 0.05
        try:
            assert ok, f"{name}: {n} params vs official_mamba2 target {m2_target} (ratio {ratio:.3f}) is outside the spine cell's +-5% full-mixer param-match gate"
        except AssertionError as exc:
            print(f'[param-match] WARNING: {exc}', flush=True)
        tags[name] = dict(param_match_target_arm='official_mamba2', param_match_target_n=m2_target, param_match_n=n, param_match_ratio=round(ratio, 4), param_match_ok=ok, config='default')
    return tags

def grid_ssm_lm_spine(graphed: bool=False) -> list[Cell]:
    g = 'ssm_lm_spine'
    B, D, L, N = (sh.LM_SPINE['B'], sh.LM_SPINE['D'], sh.LM_SPINE['L'], sh.LM_SPINE['N'])
    w = f'B{B}L{L}d{D}'
    d_model_off = D // 2
    param_tags = _lm_spine_param_audit(D, N, B, L)
    exact_tags = _exact_arm_param_audit(d_model_off, sh.EXACT_D_STATE_M23, 64, B, L, d_state_m1=sh.EXACT_D_STATE_M1)
    exact_tags_n64 = _exact_arm_param_audit(d_model_off, sh.EXACT_D_STATE_M23_N64, 64, B, L, d_state_m1=sh.EXACT_D_STATE_M1)
    cells: list[Cell] = []
    for prec in ('ieee', 'tf32', 'bf16'):
        for p in ('fwd', 'fwd_bwd'):
            cells.append(_ssm_cell(g, 'ssd_mimo', 'core:S-R1', D, N, B, L, prec, p, work=w, R=1, O=1, ngroups=64, tags=dict(config='default')))
            cells.append(_ssm_cell(g, 'ssd_mimo', 'core:S-R2O2', D, 128, B, L, prec, p, work=w, R=2, O=2, ngroups=64, r_mix_dyn=True, tags=dict(config='default')))
            cells.append(_ssm_cell(g, 'ssd_mimo', 'core:S-R4O4', D, 128, B, L, prec, p, work=w, R=4, O=4, ngroups=128, r_mix_dyn=True, tags=dict(gate='r4o4_18ms', config='default')))
            cells.append(_ssm_cell(g, 'hybrid_mamba', 'core:HM-LM', D, N, B, L, prec, p, work=w, n_heads=D // 16, tags=dict(config='default')))
            cells.append(_full_mixer_cell(g, 'ssd_mimo', 'SM-R1', d_model_off, N, B, L, prec, p, work=w, tags=param_tags['SM-R1'], R=1, O=1, ngroups=64, proj_source='main_branch'))
            if prec == 'bf16':
                pc3 = (('SM-R1:fused', 'fused', False), ('SM-R1:modular', 'modular', False), ('SM-R1:modular+checkpoint', 'modular', True))
                for arm, mode, checkpoint in pc3:
                    tags = dict(config='pc3_m3', mixer_mode=mode, modular_checkpoint=checkpoint, comparison='pc3_modular_mixer')
                    over = dict(R=1, O=1, ngroups=64, proj_source='layer_input', mode=mode, modular_checkpoint=checkpoint)
                    cells.append(_full_mixer_cell(g, 'ssd_mimo', arm, d_model_off, N, B, L, prec, p, work=w, tags=tags, **over))
                    if graphed:
                        cells.append(_full_mixer_cell(g, 'ssd_mimo', arm + GRAPH_ARM_SUFFIX, d_model_off, N, B, L, prec, p, work=w, tags=tags, graphed=True, **over))
            cells.append(_full_mixer_cell(g, 'ssd_mimo', 'SM-R2O2', d_model_off, 128, B, L, prec, p, work=w, tags=param_tags['SM-R2O2'], R=2, O=2, ngroups=64, r_mix_dyn=True, proj_source='main_branch'))
            cells.append(_full_mixer_cell(g, 'ssd_mimo', 'SM-R4O4', d_model_off, 128, B, L, prec, p, work=w, tags=dict(param_tags['SM-R4O4'], gate='r4o4_18ms'), R=4, O=4, ngroups=128, r_mix_dyn=True, proj_source='main_branch'))
            cells.append(_full_mixer_cell(g, 'hybrid_mamba', 'HMM-LM', d_model_off, N, B, L, prec, p, work=w, tags=param_tags['HMM-LM'], n_heads=D // 64, proj_source='main_branch'))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m2exact', 'm2exact', d_model_off, sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64, tags=exact_tags['m2exact']))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m2exact@N64', 'm2exact', d_model_off, sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64, config_tag='m2exact_n64', tags=dict(exact_tags_n64['m2exact'], config='m2exact_n64')))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m3exact', 'm3exact', d_model_off, sh.EXACT_D_STATE_M23, B, L, prec, p, work=w, headdim=64, tags=exact_tags['m3exact']))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m3exact@N64', 'm3exact', d_model_off, sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=64, config_tag='m3exact_n64', tags=dict(exact_tags_n64['m3exact'], config='m3exact_n64')))
            cells.append(_exact_mixer_cell(g, 'HMM:m1exact', 'm1exact', d_model_off, sh.EXACT_D_STATE_M1, B, L, prec, p, work=w, tags=exact_tags['m1exact']))
            cells.extend(_hm_exact_feature_cells(g, d_model_off, sh.EXACT_D_STATE_M1, B, L, prec, p, w, exact_tags['m1exact']['param_match_n']))
            cells.extend(_hm_design_feature_cells(g, 'HMM-LM', d_model_off, D // 64, N, B, L, prec, p, w, param_tags['HMM-LM']['param_match_n'], feature_names=_HM_FEATURE_NAMES))
            cells.extend(_ssd_exact_feature_cells(g, d_model_off, sh.EXACT_D_STATE_M23, B, L, prec, p, w, 64, exact_tags['m2exact']['param_match_n']))
            for variant, hd, ds in (('mamba1', 64, sh.EXACT_D_STATE_M1), ('mamba2', 64, sh.EXACT_D_STATE_M23), ('mamba3', 64, sh.EXACT_D_STATE_M23)):
                cells.append(_official_cell(g, variant, D // 2, ds, B, L, prec, p, work=w, headdim=hd, tags=dict(gate_ref=True, config='official')))
            for variant, hd in (('mamba2', 64), ('mamba3', 64)):
                cells.append(_official_cell(g, variant, D // 2, sh.EXACT_D_STATE_M23_N64, B, L, prec, p, work=w, headdim=hd, arm=f'official_{variant}@N64', tags=dict(gate_ref=True, config='official_n64')))
            if graphed:
                cells.extend(_graphed_like_for_like_cells(g, d_model_off, B, L, prec, p, w, exact_tags))
    return cells

def _graphed_like_for_like_cells_shared(g, d_model, B, L, w, tag):
    cells = []
    for p in ('fwd', 'fwd_bwd'):
        for arm, cfg in (('SM-R1:m2exact', 'm2exact'), ('SM-R1:m3exact', 'm3exact')):
            cells.append(_exact_mixer_cell(g, arm, cfg, d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config=cfg, shape=tag), graphed=True))
        for variant in ('mamba2', 'mamba3'):
            cells.append(_official_cell(g, variant, d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config='official', shape=tag), graphed=True))
    return cells

def _compiled_like_for_like_cells_shared(g, d_model, B, L, w, tag):
    cells = []
    for p in ('fwd', 'fwd_bwd'):
        for arm, cfg in (('SM-R1:m2exact', 'm2exact'), ('SM-R1:m3exact', 'm3exact')):
            cells.append(_exact_mixer_cell(g, arm, cfg, d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config=cfg, shape=tag), compiled=True))
        for variant in ('mamba2', 'mamba3'):
            cells.append(_official_cell(g, variant, d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config='official', shape=tag), compiled=True))
    return cells

def grid_ssm_shared_set(graphed: bool=False) -> list[Cell]:
    from orbit_ssm.kernels import shape_sets as ss
    g = 'ssm_shared_set'
    cells: list[Cell] = []
    for tag, (B, d_inner, L) in ss.SET.items():
        d_model = d_inner // 2
        w = f'{tag}:B{B}L{L}d{d_model}N{sh.EXACT_D_STATE_M23}'
        for p in ('fwd', 'fwd_bwd'):
            cells.append(_exact_mixer_cell(g, 'SM-R1:m2exact', 'm2exact', d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config='m2exact', shape=tag)))
            cells.append(_exact_mixer_cell(g, 'SM-R1:m3exact', 'm3exact', d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config='m3exact', shape=tag)))
            for variant in ('mamba2', 'mamba3'):
                cells.append(_official_cell(g, variant, d_model, sh.EXACT_D_STATE_M23, B, L, 'bf16', p, work=w, headdim=64, tags=dict(config='official', shape=tag)))
        cells += _graphed_like_for_like_cells_shared(g, d_model, B, L, w, tag)
        cells += _compiled_like_for_like_cells_shared(g, d_model, B, L, w, tag)
    return cells

def _shared_part(prefix: str) -> Callable[..., list[Cell]]:

    def fn(graphed: bool=False) -> list[Cell]:
        return [c for c in grid_ssm_shared_set(graphed) if c.tags['shape'].startswith(prefix)]
    return fn
_GRID_FNS: dict[str, Callable[[], list[Cell]]] = {'ssm_shared_set': grid_ssm_shared_set, 'ssm_shared_small': _shared_part('small_'), 'ssm_shared_medium': _shared_part('medium_'), 'ssm_shared_large': _shared_part('large_'), 'ssm_shared_lm': _shared_part('lm_'), 'smoke': grid_smoke, 'ssm_mqmkvar': grid_ssm_mqmkvar, 'ssm_lm_spine': grid_ssm_lm_spine}

def build_grid(name: str, args: argparse.Namespace) -> list[Cell]:
    return _GRID_FNS[name](graphed=bool(getattr(args, 'graphed', False)))
_LM_SPINE_TF32_ARMS = frozenset(('core:S-R1', 'core:S-R2O2', 'core:S-R4O4', 'core:HM-LM', 'SM-R1', 'SM-R2O2', 'SM-R4O4', 'HMM-LM', 'official_mamba1', 'official_mamba2', 'official_mamba3'))
_GRAPHED_PAPER_ARMS = frozenset((a + GRAPH_ARM_SUFFIX for a in _GRAPHED_LIKE_FOR_LIKE))
_LM_SPINE_BF16_ONLY_ARMS = frozenset(('SM-R1:m2exact', 'SM-R1:m3exact', 'HMM:m1exact', 'HMM:m1exact+rot', 'HMM:m1exact+gates', 'HMM:m1exact+all', 'HMM:m1exact+all1', 'HMM-LM+rot', 'HMM-LM+gates', 'HMM-LM+all', 'HMM-LM+all1', 'SM-R1:m2exact+rot', 'SM-R1:m2exact+gates', 'SM-R1:m2exact+all', 'SM-R1:fused', 'SM-R1:modular', 'SM-R1:modular+checkpoint', 'SM-R1:fused@graph', 'SM-R1:modular@graph', 'SM-R1:modular+checkpoint@graph')) | _GRAPHED_PAPER_ARMS

def _paper_ssm_lm_spine(c: Cell) -> bool:
    if c.arm in _LM_SPINE_TF32_ARMS:
        return c.precision in ('tf32', 'bf16')
    if c.arm in _LM_SPINE_BF16_ONLY_ARMS:
        return c.precision == 'bf16'
    return False
_MQMKVAR_TF32_ARMS = frozenset(('core:S1', 'core:S2', 'core:S3', 'core:S4', 'core:S5', 'SM1', 'SM2', 'SM3', 'SM4', 'SM5', 'core:HM', 'HMM', 'official_mamba1', 'official_mamba2', 'official_mamba3'))
_MQMKVAR_BF16_ONLY_ARMS = frozenset(('SM-R1:m2exact', 'SM-R1:m3exact', 'HMM:m1exact', 'HMM:m1exact+rot', 'HMM:m1exact+gates', 'HMM:m1exact+all', 'HMM:m1exact+all1', 'HMM+rot', 'HMM+gates', 'HMM+all', 'SM-R1:m2exact+rot', 'SM-R1:m2exact+gates', 'SM-R1:m2exact+all')) | _GRAPHED_PAPER_ARMS

def _paper_ssm_mqmkvar(c: Cell) -> bool:
    if c.arm in _MQMKVAR_TF32_ARMS:
        return c.precision in ('tf32', 'bf16')
    if c.arm in _MQMKVAR_BF16_ONLY_ARMS:
        return c.precision == 'bf16'
    return False
PAPER_SETS: dict[str, Callable[[Cell], bool]] = {'ssm_mqmkvar': _paper_ssm_mqmkvar, 'ssm_lm_spine': _paper_ssm_lm_spine, 'ssm_shared_set': lambda c: True, 'ssm_shared_small': lambda c: True, 'ssm_shared_medium': lambda c: True, 'ssm_shared_large': lambda c: True, 'ssm_shared_lm': lambda c: True}

def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--profile', action='store_true', help='also record the self-CUDA fwd/bwd/sp/glue bucket split (adds a profiled pass per cell)')
    parser.add_argument('--graphed', action='store_true', help='G34: also emit an `<arm>@graph` row for each like-for-like arm (ours AND the officials), whose timed closure is one CUDA-graph replay of a whole captured fwd(+bwd) step (orbit_ssm.harness.graphs).  Same group, same budget and canary rules as the eager rows; bf16 only. Both mixers are host-bound at these cells (G32/G33), so this is the regime in which a wall ratio prices the kernels')

def _extra_meta(args: argparse.Namespace) -> dict:
    return dict(profile=bool(getattr(args, 'profile', False)), graphed=bool(getattr(args, 'graphed', False)), params_by_arm=_PARAMS)

def _post_cell(args: argparse.Namespace):
    want_profile = bool(getattr(args, 'profile', False))

    def hook(cell, closure):
        out: dict = {}
        if cell.grid == 'ssm_shared_set' and cell.arm.startswith('SM-R1:') and ('@' not in cell.arm):
            from .route_record import served
            try:
                out.update(served(closure))
            except Exception as exc:
                out['route_error'] = f'{type(exc).__name__}: {exc}'
        if getattr(closure, 'compile_seconds', None) is not None:
            out['compile_seconds'] = closure.compile_seconds
            out['compile_mode'] = getattr(closure, 'compile_mode', None)
        gs = getattr(closure, 'graphed_step', None)
        if gs is not None:
            out['graph_memory'] = dict(gs.memory)
            out['graph_notes'] = list(gs.notes)
            out['graph_warnings'] = [w for w in gs.capture_warnings if 'deprecated' not in w]
        if want_profile:
            try:
                out['cuda_buckets'] = cuda_buckets(closure)
            except Exception as exc:
                out['cuda_buckets'] = None
                out['cuda_buckets_error'] = f'{type(exc).__name__}: {exc}'
        return out
    return hook
if __name__ == '__main__':
    main_for(build_grid, GRIDS, 'benchmarks.bench_ssms', add_args=_add_args, extra_meta=_extra_meta, post_cell=_post_cell, paper_filter=PAPER_SETS.get)
