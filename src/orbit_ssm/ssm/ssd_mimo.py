
from __future__ import annotations
import math
import os as _os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv import CausalDepthwiseConv
from ..layers.glue import glue_on, param_cache_key, dt_fp32_on
from ..layers.proj import linear_keep_layout, matmul_channel_major, resolve_proj_source, cached_cat, split_in_proj
from ..layers.norms import rms_norm_cast
from ..layers.state import BaseSSM, State, ssm_status
from .. import kernel_options as _kernel_options
from ..config import BACKENDS, WRITE_RULE_DEFAULT, check_backend, reject_unknown_kwargs, resolve_write_rule, _valid_chunk_size
from .. import config_spec as _config_spec
from ..configuration import Configurable, positive_int, resolve_ssd_geometry, record_constructor_config, MixerConfigurable, mixer_core_options, record_mixer_config

def _ref_f(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float64 if t.dtype == torch.float64 else torch.float32)
_Y_LAYOUTS = ('token', 'contig', 'channel')

def _ua_gate_epilogue_off() -> bool:
    from ..kernels.ssd_mimo import autograd as _ua
    return not _ua._gate_epilogue_reads_strided()

def _y_layout() -> str:
    mode = _os.environ.get('SSD_MIMO_Y_LAYOUT', 'token')
    if mode not in _Y_LAYOUTS:
        raise ValueError(f'SSD_MIMO_Y_LAYOUT must be one of {_Y_LAYOUTS}, got {mode!r}')
    return mode

def _y_is_token_major(y: torch.Tensor) -> bool:
    stride, shape = (y.stride(), y.shape)
    ch = 1
    for size, st in zip(reversed(shape[1:-1]), reversed(stride[1:-1])):
        if size == 1:
            continue
        if st != ch:
            return False
        ch *= size
    return stride[-1] == ch

def _torch_impl() -> str:
    v = _os.environ.get('SSD_MIMO_TORCH_IMPL', 'fast').strip().lower()
    if v not in ('fast', 'legacy'):
        raise ValueError(f'SSD_MIMO_TORCH_IMPL must be fast|legacy, got {v!r}')
    return v
_R_MIX_MODES = ('auto', 'learned', 'none')
THETA_SMALL_SCALE = 0.1 * 3 ** 0.5
_THETA_FREQ_INIT = 3.8
_COMBINED_CHUNK = 64
_CHUNK_RANK_TABLE = {1: 64, 2: 32, 4: 32, 8: 16}
_CHUNK_CANDIDATES = (64, 32, 16)

def _valid_chunk_size(v) -> bool:
    from ..config import _valid_chunk_size as _v
    return _v(v)
_CHUNK_WIDE_W = 64
_CHUNK_WIDE_T = 32
_CHUNK_RANK_TABLE_FWD = {1: 64, 2: 64, 4: 16, 8: 16}
_CHUNK_WIDE_W_FWD = 64
_CHUNK_WIDE_T_FWD = 32

def _call_size(kw: dict) -> dict:
    x = kw.get('x_k')
    if isinstance(x, torch.Tensor) and x.dim() == 5:
        B, Hk, _, _, L = x.shape
        dt = torch.get_autocast_dtype('cuda') if x.is_cuda and torch.is_autocast_enabled('cuda') else x.dtype
        return dict(L=int(L), work=int(B) * int(Hk) * int(L), dtype=dt)
    return {}

def _needs_bwd(kw: dict) -> bool:
    return bool(torch.is_grad_enabled() and any((isinstance(t, torch.Tensor) and t.requires_grad for t in kw.values())))

def _gluon_siso_default(kw: dict, with_route: bool=True) -> bool:
    x = kw.get('x_k')
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    io_bf16 = torch.get_autocast_dtype('cuda') == torch.bfloat16 if torch.is_autocast_enabled('cuda') else x.dtype == torch.bfloat16
    from ..kernels.ssd_mimo.siso_route import siso_enabled
    if with_route:
        return bool(siso_enabled(x.device, rotating=True, io_bf16=io_bf16, T=64, bh=x.shape[0] * x.shape[1], L=x.shape[-1], needs_bwd=_needs_bwd(kw), thh=kw.get('thh_k') is not None))
    return bool(siso_enabled(x.device, rotating=True, io_bf16=io_bf16, T=64))

def _calibrated_rot_route(kw: dict, fp32_only: bool=False):
    x = kw.get('x_k')
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return None
    from ..kernels.ssd_mimo import rot_route as rr
    if rr.explicit_siso() or rr.explicit_rot_bwd():
        return None
    io = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
    if io == torch.float32:
        return rr.pick(x.shape[0] * x.shape[1], x.shape[-1], x.device, _needs_bwd(kw), kw.get('thh_k') is not None, dtype='fp32')
    if io != torch.bfloat16 or fp32_only:
        return None
    return rr.pick(x.shape[0] * x.shape[1], x.shape[-1], x.device, _needs_bwd(kw), kw.get('thh_k') is not None)

def _scan_chunk(R: int=1, G: int | None=None, O: int | None=None, needs_bwd: bool=True, real: bool | None=None, L: int | None=None, work: int | None=None, dtype=None) -> int:
    import os as _os
    env = _os.environ.get('SSD_MIMO_CHUNK')
    if env:
        return int(env)
    if real and R == 1 and (O is None or O == 1):
        from ..kernels.ssd_mimo.calibration_chunkpar import real_chunk
        t = real_chunk('train_fwd' if needs_bwd else 'fwd_only', G, L, work, dtype)
        if t is not None:
            return t
    elif real is not None and R > 1 and (O is None or O == R):
        from ..kernels.ssd_mimo.calibration_chunkpar import rank_chunk
        t = rank_chunk(bool(real), R, R, 'train_fwd' if needs_bwd else 'fwd_only', G, dtype)
        if t is not None:
            return t
    table = _CHUNK_RANK_TABLE if needs_bwd else _CHUNK_RANK_TABLE_FWD
    wide_w = _CHUNK_WIDE_W if needs_bwd else _CHUNK_WIDE_W_FWD
    wide_t = _CHUNK_WIDE_T if needs_bwd else _CHUNK_WIDE_T_FWD
    T = table.get(R, _COMBINED_CHUNK)
    if G is not None:
        W = int(G) * (R * (R if O is None else O) if needs_bwd else 1)
        if T > wide_t and W >= wide_w:
            T = wide_t
    return T

def _chunk_pair_on() -> bool:
    import os as _os
    return _os.environ.get('SSD_MIMO_CHUNK_PAIR', '0').lower() in ('1', 'true', 'yes', 'on')

def _chunk_pair_search_on() -> bool:
    import os as _os
    return _os.environ.get('SSD_MIMO_CHUNK_PAIR_SEARCH', '1').lower() not in ('0', 'false', 'no', 'off')
_CHUNK_BWD_TABLE: dict = {}

def _scan_chunk_bwd(R: int=1, G: int | None=None, O: int | None=None, T_fwd: int=64, real: bool | None=None, L: int | None=None, work: int | None=None, dtype=None) -> int | None:
    import os as _os
    env = _os.environ.get('SSD_MIMO_CHUNK_BWD', '').strip().lower()
    if env and env not in ('auto', '0', ''):
        return int(env)
    tb = _CHUNK_BWD_TABLE.get((R, R if O is None else O, G))
    if real and R == 1 and (O is None or O == 1) and (not _os.environ.get('SSD_MIMO_CHUNK')):
        from ..kernels.ssd_mimo.calibration_chunkpar import real_chunk
        tc = real_chunk('train_bwd', G, L, work, dtype)
        if tc is not None:
            tb = tc
    elif real is not None and R > 1 and (O is None or O == R) and (not _os.environ.get('SSD_MIMO_CHUNK')):
        from ..kernels.ssd_mimo.calibration_chunkpar import rank_chunk
        tc = rank_chunk(bool(real), R, R, 'train_bwd', G, dtype)
        if tc is not None:
            tb = tc
    if tb is None or tb == T_fwd:
        return None
    return tb

def _dt_tail_eager(dt_lin: torch.Tensor) -> torch.Tensor:
    return F.softplus(dt_lin)

def _clamp_min_pass(x: torch.Tensor, lo: float) -> torch.Tensor:
    return torch.where(x >= lo, x, x.clamp_min(lo).detach())

def _clamp_max_pass(x: torch.Tensor, hi: float) -> torch.Tensor:
    return torch.where(x <= hi, x, x.clamp_max(hi).detach())

def _a_mag_tail_eager(a_lin: torch.Tensor, floor: float) -> torch.Tensor:
    neg = _clamp_max_pass(a_lin, 0.0)
    pos = _clamp_min_pass(a_lin, 0.0)
    return _clamp_min_pass(pos + torch.reciprocal(1 - neg), floor)

def _a_mag_softplus_eager(a_lin: torch.Tensor, floor: float) -> torch.Tensor:
    return _clamp_min_pass(F.softplus(a_lin), floor)
A_ACTS = ('heavy_tail', 'softplus')
_dt_tail_compiled = None
_a_mag_tail_compiled: dict = {}
_DT_TAIL_ORIG = _dt_tail_eager
_A_MAG_TAIL_ORIG = _a_mag_tail_eager
_A_MAG_SOFTPLUS_ORIG = _a_mag_softplus_eager

def _bc_narrow_conv_ready(joint: torch.Tensor) -> bool:
    from ..layers.conv import channel_last_conv_view_ok
    return channel_last_conv_view_ok(joint) and joint.stride(1) > joint.shape[-1]

def _tail_law_is_stock() -> bool:
    return _dt_tail_eager is _DT_TAIL_ORIG and _a_mag_tail_eager is _A_MAG_TAIL_ORIG and (_a_mag_softplus_eager is _A_MAG_SOFTPLUS_ORIG)

def _kernel_phase_pre_omega(raw: torch.Tensor, dt_raw_k: torch.Tensor, *, H: int, R: int, K: int, ka: int, per_head: bool, sign: float) -> torch.Tensor:
    from ..kernels.ssd_mimo.m3_epilogue import m3_phase_increments
    B, L = (raw.shape[0], raw.shape[1])
    raw_f = _ref_f(raw)
    if per_head:
        raw_h = raw_f.view(B, L, H, ka)
    else:
        raw_h = raw_f.view(B, L, 1, ka).expand(B, L, H, ka).contiguous()
    if R == 1:
        return m3_phase_increments(raw_h, dt_raw_k, K, sign)
    parts = []
    for r in range(R):
        dt_r = dt_raw_k[:, r * H:(r + 1) * H, :].contiguous()
        parts.append(m3_phase_increments(raw_h, dt_r, K, sign))
    return torch.stack(parts, dim=2).reshape(B, L, R * H, K)

class _OmegaFoldFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, raw, dt_raw_k, omega, H, R, K, ka, per_head, sign):
        with torch.no_grad():
            pre = _kernel_phase_pre_omega(raw, dt_raw_k, H=H, R=R, K=K, ka=ka, per_head=per_head, sign=sign)
            out = pre * omega if omega is not None else pre
        ctx.save_for_backward(raw, dt_raw_k, omega if omega is not None else raw.new_empty(0))
        ctx.H, ctx.R, ctx.K, ctx.ka = (H, R, K, ka)
        ctx.per_head, ctx.sign = (per_head, sign)
        ctx.has_omega = omega is not None
        ctx.need_raw = raw.requires_grad
        ctx.need_dt = dt_raw_k.requires_grad
        ctx.need_omega = ctx.has_omega and omega.requires_grad
        return out

    @staticmethod
    def backward(ctx, dout):
        raw, dt_raw_k, omega_s = ctx.saved_tensors
        omega = omega_s if ctx.has_omega else None
        need_raw, need_dt, need_omega = (ctx.need_raw, ctx.need_dt, ctx.need_omega)
        if not (need_raw or need_dt or need_omega):
            return (None, None, None, None, None, None, None, None, None)
        dout = dout.contiguous()
        with torch.enable_grad():
            raw_l = raw.detach().requires_grad_(need_raw or need_dt)
            dt_l = dt_raw_k.detach().requires_grad_(need_dt)
            pre = _kernel_phase_pre_omega(raw_l, dt_l, H=ctx.H, R=ctx.R, K=ctx.K, ka=ctx.ka, per_head=ctx.per_head, sign=ctx.sign)
        domega = (dout * pre).sum(dim=(0, 1, 2)) if need_omega else None
        dpre = dout * omega if omega is not None else dout
        draw = ddt = None
        if need_raw or need_dt:
            leaves = ([raw_l] if need_raw else []) + ([dt_l] if need_dt else [])
            grads = torch.autograd.grad(pre, leaves, grad_outputs=dpre, allow_unused=True)
            it = iter(grads)
            draw = next(it) if need_raw else None
            ddt = next(it) if need_dt else None
        return (draw, ddt, domega, None, None, None, None, None, None)

def _phase_chain_torch(raw: torch.Tensor, dt_raw_k: torch.Tensor, omega, *, H: int, R: int, K: int, ka: int, per_head: bool, sign: float, omega_per_head: bool=False) -> torch.Tensor:
    B, L = (raw.shape[0], raw.shape[1])
    Hk = R * H
    raw_f = _ref_f(raw)
    if omega_per_head:
        ang = raw_f.view(B, L, H, 1).expand(B, L, H, ka)
        ang = F.pad(torch.tanh(ang) * math.pi, (0, K - ka))
        if omega is not None:
            ang = ang * omega.view(H, 1)
        if R > 1:
            ang = ang.unsqueeze(2).expand(B, L, R, H, K).reshape(B, L, Hk, K)
        return sign * (ang * dt_raw_k.permute(0, 2, 1)[..., None])
    ang = raw_f.view(B, L, H, ka) if per_head else raw_f.view(B, L, ka)
    if ka < K:
        ang = F.pad(ang, (0, K - ka))
    ang = torch.tanh(ang) * math.pi
    if not per_head:
        ang = ang.unsqueeze(2).expand(B, L, H, K)
    if R > 1:
        ang = ang.unsqueeze(2).expand(B, L, R, H, K).reshape(B, L, Hk, K)
    incr = sign * (ang * dt_raw_k.permute(0, 2, 1)[..., None])
    if omega is not None:
        incr = incr * (omega if omega.dim() == 1 else omega.repeat(R, 1))
    return incr

def _wrap2pi_t(x: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return x - two_pi * torch.round(x.detach() / two_pi)

def _theta_increments(raw: torch.Tensor, dt_raw_k: torch.Tensor, omega, *, H: int, R: int, K: int, ka: int, per_head: bool, use_kernel: bool, sign: float, omega_per_head: bool=False) -> torch.Tensor:
    if omega_per_head:
        return _phase_chain_torch(raw, dt_raw_k, omega, H=H, R=R, K=K, ka=ka, per_head=per_head, sign=sign, omega_per_head=True)
    if use_kernel and R > 1:
        return _OmegaFoldFn.apply(raw, dt_raw_k, omega, H, R, K, ka, per_head, sign)
    if use_kernel:
        incr = _kernel_phase_pre_omega(raw, dt_raw_k, H=H, R=R, K=K, ka=ka, per_head=per_head, sign=sign)
        if omega is not None:
            incr = incr * omega
        return incr
    return _phase_chain_torch(raw, dt_raw_k, omega, H=H, R=R, K=K, ka=ka, per_head=per_head, sign=sign)

def _dt_tail(dt_lin: torch.Tensor, use_compile: bool) -> torch.Tensor:
    if use_compile:
        global _dt_tail_compiled
        if _dt_tail_compiled is None:
            _dt_tail_compiled = torch.compile(_dt_tail_eager)
        return _dt_tail_compiled(dt_lin)
    return _dt_tail_eager(dt_lin)

def _a_mag_tail(a_lin: torch.Tensor, floor: float, use_compile: bool, act: str='heavy_tail') -> torch.Tensor:
    fn = _a_mag_tail_eager if act == 'heavy_tail' else _a_mag_softplus_eager
    if use_compile:
        if act not in _a_mag_tail_compiled:
            _a_mag_tail_compiled[act] = torch.compile(fn)
        return _a_mag_tail_compiled[act](a_lin, floor)
    return fn(a_lin, floor)

def _bc_bias_init(shape, mode: str, R: int, N: int) -> torch.Tensor:
    base = torch.ones(shape)
    if mode == 'ones' or len(shape) < 4 or R < 2:
        return base
    r_off = torch.linspace(0.0, math.log(2.0), R)
    k = torch.arange(R).float()[:, None]
    n = torch.arange(N).float()[None, :]
    dct = torch.cos(math.pi * k * (n + 0.5) / N)
    return base + (r_off[:, None] * dct)[None, :, :, None]

@ssm_status('production')
class SSDMimoSSM(Configurable, BaseSSM):
    config_family = 'ssd_mimo'

    def __init__(self, d_inner: int, d_state: int=64, *, R: int=2, O: int | None=None, ngroups: int | None=None, n_heads: int | None=None, heads: int | None=None, H: int | None=None, head_subsystems: int | None=None, P: int | None=None, rows: list | None=None, rotation: dict | None=None, maps: dict | None=None, r_mix: str='auto', r_mix_dyn: bool | None=None, backend: str='auto', precision: str='io', m3_kernel: str='native', dt_min: float=0.001, dt_max: float=0.1, dt_init_floor: float=0.0001, a_proj_bias: bool=True, trap_proj_bias: bool=True, write_rule: str=WRITE_RULE_DEFAULT, d_skip: bool=True, d_skip_sharing: str='input_output', pair_layout: str='interleaved', proj_source: str='layer_input', d_model: int | None=None, bc_conv: bool | None=None, d_conv: int=4, compile_token_maps: bool=False, a_dyn_per_sibling: bool | None=None, a_act: Literal['heavy_tail', 'softplus']='heavy_tail', chunk_size: int | str='auto', kernel_options: dict | None=None, allow_chunk_autotune: bool=True, theta_per_head: bool | None=None, learn_theta_freq: bool | None=None, use_theta_h: bool | None=None, b_norm: bool | None=None, c_norm: bool | None=None, bc_norm: bool | None=None, bias_gran: str | None=None, use_g_b: bool | None=None, use_g_c: bool | None=None, bc_bias_init: str | None=None, **unexpected):
        super().__init__()
        _config_spec.reject_removed(unexpected, 'SSDMimoSSM')
        reject_unknown_kwargs(unexpected, 'SSDMimoSSM')
        self.kernel_options = _kernel_options.validate('ssd_mimo', kernel_options)
        d_inner = positive_int(d_inner, 'd_inner')
        d_state = positive_int(d_state, 'd_state')
        if not 0 < dt_min <= dt_max or not math.isfinite(dt_max):
            raise ValueError('require 0 < dt_min <= dt_max < infinity')
        if not math.isfinite(dt_init_floor) or dt_init_floor <= 0:
            raise ValueError('dt_init_floor must be finite and positive')
        self.proj_source, d_sel, self.bc_conv = resolve_proj_source(proj_source, d_inner, d_model, bc_conv, who='SSDMimoSSM', d_conv=d_conv)
        self.d_sel = d_sel
        check_backend(backend, 'SSDMimoSSM', BACKENDS)
        from ..kernels.m3_adapter import check_provider
        check_provider(m3_kernel, backend)
        self.m3_kernel = m3_kernel
        self.backend = backend
        if precision not in ('ieee', 'tf32', 'io', 'bf16_ops'):
            raise ValueError(f'precision must be ieee|tf32|io (or the opt-in bf16_ops), got {precision!r}')
        write_rule = resolve_write_rule(write_rule, 'SSDMimoSSM')
        self.precision = precision
        self._compile_token_maps = bool(compile_token_maps)
        if not _valid_chunk_size(chunk_size):
            raise ValueError(f"chunk_size must be 'auto', 'measure' or a power-of-two >= 16, got {chunk_size!r}")
        self.chunk_size = chunk_size
        self.allow_chunk_autotune = bool(allow_chunk_autotune)
        R = positive_int(R, 'R')
        if R not in (1, 2, 4, 8):
            raise ValueError(f'R must be one of 1,2,4,8, got {R}')
        N = d_state
        n_heads, P, G = resolve_ssd_geometry(d_inner, R=R, ngroups=ngroups, n_heads=n_heads, heads=heads, H=H, head_subsystems=head_subsystems, P=P)
        self.G_kernel = -(-G // 16) * 16
        O = R if O is None else positive_int(O, 'O')
        if O not in (1, 2, 4, 8):
            raise ValueError(f'O must be one of 1,2,4,8 (kernel o-tiling), got {O}')
        self.O = O
        if m3_kernel == 'upstream' and (R != O or N not in (16, 32, 64, 128) or G * R not in (32, 64, 128)):
            raise ValueError('upstream M3 requires R=O, N in {16,32,64,128}, G*R in {32,64,128}')
        self.d_out = n_heads * G * O
        self.out_width = self.d_out
        self.H, self.P, self.G, self.R, self.N = (n_heads, P, G, R, N)
        self.d_inner = d_inner
        spec = _config_spec.resolve('ssd_mimo', d_state=N, R=R, O=O, rows=rows, rotation=rotation, maps=maps, **_config_spec.flat_kwargs(locals()))
        self.spec = spec
        n_complex = sum((b['rows'] for b in spec['rows'] if b['complex']))
        a_dyn = spec['rows'][0]['decay'] == _config_spec.HEAD_DYNAMIC
        rot = spec['rotation'] or {}
        rot_mode = 'm3' if n_complex else None
        use_theta_h = bool(rot.get('use_theta_h', False))
        learn_theta_freq = bool(rot.get('learn_theta_freq', False))
        theta_per_head = bool(rot.get('theta_per_head', False))
        omega_layout = rot.get('omega_layout', 'pair')
        theta_init = rot.get('theta_init', 'kaiming')
        mp = spec['maps']
        use_g_b, use_g_c = (mp['gate_b'], mp['gate_c'])
        bias_gran = mp['bias']
        if m3_kernel == 'upstream' and (not n_complex or use_theta_h or learn_theta_freq):
            raise ValueError('upstream M3 requires a complex row block with use_theta_h=False and learn_theta_freq=False')
        if m3_kernel == 'upstream' and omega_layout != 'pair':
            raise ValueError(f"upstream M3 runs Mamba-3's per-pair angles; rotation.omega_layout={omega_layout!r} is the native route only")
        if (R == 1) != (self.O == 1):
            warnings.warn(f"SSDMimoSSM: R={R}, O={self.O} -- exactly one of them is 1, so each SUBSYSTEM's local temporal block C_t Phi(t,s) B_s has rank at most min(N, R, O) = 1, the same ceiling as SISO, while the larger rank is still paid for in state, parameters and work (thm:crosstalk, paper/theory_compat.tex). The LAYER is not bounded by this: it aggregates R subsystems through the zeta sibling mix, so the configuration still buys state and parameters and may buy layer-level rank.  Use R == O to lift the per-subsystem ceiling itself.", RuntimeWarning, stacklevel=2)
        self.d_skip = bool(d_skip)
        if d_skip_sharing not in ('input_output', 'channel', 'head'):
            raise ValueError('d_skip_sharing must be input_output, channel, or head')
        self.d_skip_sharing = d_skip_sharing
        if self.d_skip:
            shape = {'input_output': (n_heads, G, R, self.O), 'channel': (n_heads, G, R), 'head': (n_heads,)}[d_skip_sharing]
            self.D = nn.Parameter(torch.ones(shape))
            self.D._no_weight_decay = True
        if r_mix not in _R_MIX_MODES:
            raise ValueError(f'SSDMimoSSM: r_mix must be one of {_R_MIX_MODES}, got {r_mix!r}')
        self.r_mix_mode = ('none' if R == 1 else 'learned') if r_mix == 'auto' else r_mix
        if r_mix_dyn is None:
            r_mix_dyn = R > 1 and self.r_mix_mode != 'none'
        if self.r_mix_mode == 'none':
            if r_mix_dyn:
                raise ValueError(f"SSDMimoSSM(r_mix_dyn=True) needs an active sibling mix, but r_mix resolved to 'none' (R={R}); pass r_mix='learned' to keep it")
            self.r_mix = None
            self.r_mix_dyn = False
        else:
            self.r_mix = nn.Parameter(torch.full((n_heads, R), 1.0 / R))
            self.r_mix_dyn = bool(r_mix_dyn)
            if r_mix_dyn:
                self.r_mix_proj = nn.Linear(d_sel, n_heads * R, bias=False)
                nn.init.zeros_(self.r_mix_proj.weight)
        self._register_load_state_dict_pre_hook(self._validate_pair_layout)
        self.B_proj = nn.Linear(d_sel, N * R, bias=False)
        self.C_proj = nn.Linear(d_sel, N * O, bias=False)
        self.bc_conv1d = CausalDepthwiseConv(N * R + N * O, d_conv) if self.bc_conv else None
        self.use_g_b = bool(use_g_b)
        self.use_g_c = bool(use_g_c)
        if R > 1 and (not self.use_g_b):
            warnings.warn(f'SSDMimoSSM(R={R}, use_g_b=False): sibling-specific write gating is disabled; biases or different clocks may still distinguish siblings.', RuntimeWarning, stacklevel=2)
        if O > 1 and (not self.use_g_c):
            warnings.warn(f'SSDMimoSSM(O={O} > 1, use_g_c=False): output-specific read gating is disabled; the read projection still has distinct output columns.', RuntimeWarning, stacklevel=2)
        gate_cx = any((b['gate'] for b in spec['rows'] if b['complex']))
        gate_re = any((b['gate'] for b in spec['rows'] if not b['complex']))
        self.gate_blocks = (gate_cx, gate_re)
        self.gate_pair_tied = n_complex > 0 and gate_cx
        self.gate_masked = n_complex > 0 and (not gate_cx) or (N > n_complex and (not gate_re))
        n_gate = (n_complex // 2 if gate_cx else 0) + (N - n_complex if gate_re else 0)
        self.n_gate_rows = n_gate
        if self.use_g_b:
            self.gB_proj = nn.Linear(d_sel, n_gate * R, bias=False)
            nn.init.zeros_(self.gB_proj.weight)
        if self.use_g_c:
            self.gC_proj = nn.Linear(d_sel, n_gate * O, bias=False)
            nn.init.zeros_(self.gC_proj.weight)
        dt_out = n_heads * R
        self.dt_proj = nn.Linear(d_sel, dt_out, bias=True)
        with torch.no_grad():
            dt = torch.exp(torch.rand(dt_out) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.a_dyn = bool(a_dyn)
        self.a_dyn_per_sibling = bool(a_dyn_per_sibling)
        if a_act not in A_ACTS:
            raise ValueError(f'a_act must be one of {A_ACTS}, got {a_act!r}')
        self.a_act = a_act
        if self.a_dyn:
            self.A_log = None
            rate_width = n_heads * R if self.a_dyn_per_sibling else n_heads
            self.a_dyn_proj = nn.Linear(d_sel, rate_width, bias=a_proj_bias)
            if self.a_dyn_proj.bias is not None:
                nn.init.zeros_(self.a_dyn_proj.bias)
            if self.a_dyn_proj.bias is not None:
                self.a_dyn_proj.bias._no_weight_decay = True
        else:
            r_off = torch.linspace(0.0, math.log(2.0), R)
            hbase = torch.exp(torch.linspace(math.log(0.05), math.log(2.0), n_heads))
            A = hbase[:, None] * torch.exp(r_off)[None, :]
            self.A_log = nn.Parameter(A.log())
            self.A_log._no_weight_decay = True
        self.bias_gran = bias_gran
        self.bc_bias = bias_gran != 'none'
        self.head_bias = bias_gran == 'head_bc'
        self.b_norm = bool(mp['b_norm'])
        self.c_norm = bool(mp['c_norm'])
        self.bc_bias_init = mp['bias_init']
        if self.b_norm:
            self.B_norm = nn.RMSNorm(N, eps=1e-05)
        if self.c_norm:
            self.C_norm = nn.RMSNorm(N, eps=1e-05)
        if self.bc_bias:
            bshape = {'shared': (N,), 'head_bc': (n_heads, R, N, R)}[bias_gran]
            cshape = {'shared': (N,), 'head_bc': (n_heads, R, N, O)}[bias_gran]
            self.B_bias = nn.Parameter(_bc_bias_init(bshape, self.bc_bias_init, R, N))
            self.C_bias = nn.Parameter(_bc_bias_init(cshape, self.bc_bias_init, R, N))
        self.use_theta_h = bool(use_theta_h) and n_complex > 0
        if self.use_theta_h:
            self.theta_h = nn.Parameter(torch.zeros(n_heads, n_complex // 2))
            self.theta_h._no_weight_decay = True
        self.rot_mode = rot_mode
        if pair_layout not in ('interleaved', 'half'):
            raise ValueError(f"pair_layout must be 'interleaved' or 'half', got {pair_layout!r}")
        self.pair_layout = pair_layout
        if pair_layout == 'half':
            if self.rot_mode != 'm3':
                raise NotImplementedError("pair_layout='half' is the rotation law's layout: it needs a complex row block in `rows`")
            if N % 32:
                raise NotImplementedError(f"pair_layout='half' needs N a multiple of 32, got {N}")
        self.register_buffer('pair_layout_code', torch.tensor(1 if pair_layout == 'half' else 0, dtype=torch.int64), persistent=True)
        self.register_buffer('gate_tie_idx', self._gate_tie_index(N, n_complex, pair_layout, self.gate_blocks) if self.gate_pair_tied or self.gate_masked else None, persistent=False)
        self._register_load_state_dict_pre_hook(self._refuse_per_coordinate_complex_gates)
        self.theta_per_head = theta_per_head
        self.omega_layout = omega_layout if n_complex > 0 else 'pair'
        self.omega_per_head = self.omega_layout == 'head'
        self.theta_init = theta_init
        if self.rot_mode == 'm3':
            self.n_rope_angles = n_complex // 2
            self.n_theta_feat = 1 if self.omega_per_head else self.n_rope_angles
            theta_width = (n_heads if self.theta_per_head else 1) * self.n_theta_feat
            self.theta_proj = nn.Linear(d_sel, theta_width, bias=False)
            if theta_init == 'zero':
                nn.init.zeros_(self.theta_proj.weight)
            elif theta_init == 'small':
                with torch.no_grad():
                    self.theta_proj.weight.mul_(THETA_SMALL_SCALE)
            self.learn_theta_freq = bool(learn_theta_freq)
            if self.learn_theta_freq:
                shape = {'pair': (self.n_rope_angles,), 'head_pair': (n_heads, self.n_rope_angles), 'head': (n_heads,)}[self.omega_layout]
                self.theta_freq = nn.Parameter(torch.full(shape, _THETA_FREQ_INIT))
                self.theta_freq._no_weight_decay = True
        else:
            self.learn_theta_freq = False
        self.bias_pre_rot = self.rot_mode == 'm3'
        self.write_rule = write_rule
        if self.use_trap:
            self.trap_proj = nn.Linear(d_sel, n_heads, bias=trap_proj_bias)
            if self.trap_proj.bias is not None:
                nn.init.zeros_(self.trap_proj.bias)
        self._fuse_proj = True
        specs = [('B', 'B_proj', N * R), ('C', 'C_proj', N * self.O)]
        if self.use_g_b:
            specs.append(('gB', 'gB_proj', N * R))
        if self.use_g_c:
            specs.append(('gC', 'gC_proj', N * self.O))
        specs.append(('dt', 'dt_proj', dt_out))
        if self.a_dyn:
            specs.append(('a', 'a_dyn_proj', self.a_dyn_proj.out_features))
        if self.use_trap:
            specs.append(('trap', 'trap_proj', n_heads))
        if self.rot_mode == 'm3':
            specs.append(('theta', 'theta_proj', self.theta_proj.out_features))
        if self.r_mix_dyn:
            specs.append(('rmix', 'r_mix_proj', n_heads * R))
        self._proj_keys = tuple((s[0] for s in specs))
        self._proj_attrs = tuple((s[1] for s in specs))
        self._proj_widths = tuple((s[2] for s in specs))
        self._tail_owns_all = set((s[0] for s in specs)) <= {'B', 'C', 'gB', 'gC', 'dt', 'a'}
        self.tok_width = sum(self._proj_widths)
        self._bc_adjacent = self._proj_keys[:2] == ('B', 'C')
        self._bc_width = sum(self._proj_widths[:2])
        self._proj_any_bias = any((getattr(self, a).bias is not None for a in self._proj_attrs))
        self.register_buffer('_proj_zero', torch.zeros(self.tok_width), persistent=False)
        self._tok_proj_w_cache: dict = {}
        record_constructor_config(self, locals(), constructor=SSDMimoSSM.__init__, overrides=dict(kernel_options=self.kernel_options, ngroups=G, r_mix=self.r_mix_mode, write_rule=self.write_rule, d_skip=self.d_skip, r_mix_dyn=self.r_mix_dyn, a_dyn_per_sibling=self.a_dyn_per_sibling, a_act=self.a_act, bc_conv=self.bc_conv, proj_source=self.proj_source, **spec), exclude=_config_spec.FLAT_KWARGS, execution=('kernel_options', 'backend', 'precision', 'm3_kernel', 'compile_token_maps', 'chunk_size', 'allow_chunk_autotune'))

    @property
    def use_trap(self) -> bool:
        return self.write_rule == 'trapezoid'

    def _refuse_per_coordinate_complex_gates(self, state_dict, prefix, *_args):
        if not self.gate_pair_tied:
            return
        for name in ('gB_proj.weight', 'gC_proj.weight', 'beta_gB', 'beta_gC'):
            mod = getattr(self, name.split('.')[0], None)
            t = state_dict.get(prefix + name)
            if t is None or mod is None:
                continue
            own = mod.weight if name.endswith('.weight') else mod
            axis = 0 if name.endswith('.weight') else -1
            if t.shape[axis] != own.shape[axis] and t.shape[axis] == self.N * (self.R if name.startswith('gB') else self.O):
                raise RuntimeError(f'{prefix}{name}: this checkpoint gates the complex rows PER COORDINATE ({t.shape[axis]} rows); gates on complex row pairs are pair-tied since 2026-09-26 (one factor per pair, {own.shape[axis]} rows).  There is no migration -- retrain.')

    @classmethod
    def _gate_tie_index(cls, N: int, n_complex: int, pair_layout: str, gated=(True, True)):
        gate_cx, gate_re = gated
        with torch.device('cpu'):
            K = n_complex // 2 if gate_cx else 0
            n_gate = K + (N - n_complex if gate_re else 0)
            n = torch.arange(N)
            cx = n // 2 if gate_cx else torch.full_like(n, n_gate)
            re = K + n - n_complex if gate_re else torch.full_like(n, n_gate)
            inter = torch.where(n < n_complex, cx, re)
            if pair_layout == 'half':
                inter = inter[cls._pair_perm(N, to_half=True)]
        return inter.long()

    def _proj_w(self, attr: str) -> torch.Tensor:
        w = getattr(self, attr).weight
        idx = getattr(self, 'gate_tie_idx', None)
        if idx is None or attr not in ('gB_proj', 'gC_proj'):
            return w
        blk = self.R if attr == 'gB_proj' else self.O
        w = w.view(self.n_gate_rows, blk, w.shape[-1])
        if self.gate_masked:
            w = torch.cat([w, w.new_zeros(1, blk, w.shape[-1])], 0)
        return w.index_select(0, idx).reshape(self.N * blk, w.shape[-1])

    def _proj_w_grad(self, attr: str, g: torch.Tensor) -> torch.Tensor:
        idx = getattr(self, 'gate_tie_idx', None)
        if idx is None or attr not in ('gB_proj', 'gC_proj'):
            return g
        blk = self.R if attr == 'gB_proj' else self.O
        d = g.shape[-1]
        n_out = self.n_gate_rows + int(self.gate_masked)
        out = g.new_zeros(n_out, blk, d)
        out.index_add_(0, idx, g.reshape(self.N, blk, d))
        return out[:self.n_gate_rows].reshape(-1, d)

    @staticmethod
    def _pair_perm(N: int, to_half: bool, device=None):
        fwd = torch.cat([torch.arange(0, N, 2, device=device), torch.arange(1, N, 2, device=device)])
        if to_half:
            return fwd
        inv = torch.empty_like(fwd)
        inv[fwd] = torch.arange(N, device=device)
        return inv

    def _pair_layout_targets(self):
        N, R, O = (self.N, self.R, self.O)
        out = [('B_proj.weight', 0, R), ('C_proj.weight', 0, O)]
        tied = getattr(self, 'gate_tie_idx', None) is not None
        if getattr(self, 'use_g_b', False) and (not tied):
            out.append(('gB_proj.weight', 0, R))
        if getattr(self, 'use_g_c', False) and (not tied):
            out.append(('gC_proj.weight', 0, O))
        if getattr(self, 'b_norm', False):
            out.append(('B_norm.weight', 0, 1))
        if getattr(self, 'c_norm', False):
            out.append(('C_norm.weight', 0, 1))
        if getattr(self, 'bc_bias', False):
            if getattr(self, 'head_bias', False):
                out.append(('B_bias', 2, 1))
                out.append(('C_bias', 2, 1))
            else:
                out.append(('B_bias', 0, 1))
                out.append(('C_bias', 0, 1))
        return out

    def _validate_pair_layout(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'pair_layout_code'
        code = state_dict.get(key)
        if code is None:
            raise ValueError(f'{key} is absent: this state dict does not declare its pair-layout code, so the state-coordinate basis of its B/C rows is unknown and loading it into pair_layout={self.pair_layout!r} could silently reinterpret the state axis. Set {key!r} to the layout it was written in (0 = interleaved, 1 = half) and pass it through convert_pair_layout before loading.')
        if int(code) != int(self.pair_layout_code):
            raise ValueError('pair_layout differs: explicitly convert_pair_layout before loading')

    def convert_pair_layout(self, state_dict):
        code = int(state_dict['pair_layout_code'])
        if code not in (0, 1):
            raise ValueError('invalid pair_layout_code')
        state_dict = dict(state_dict)
        dst_half = self.pair_layout == 'half'
        if code == int(dst_half):
            return state_dict
        state_dict['pair_layout_code'] = self.pair_layout_code.clone()
        N = self.N
        idx = self._pair_perm(N, to_half=dst_half)
        for name, axis, blk in self._pair_layout_targets():
            t = state_dict[name]
            if t.shape[axis] != N * blk:
                raise ValueError(f'{name}: expected state axis of width {N * blk}')
            if blk == 1:
                state_dict[name] = t.index_select(axis, idx.to(t.device)).contiguous()
            else:
                v = t.unflatten(axis, (N, blk)).index_select(axis, idx.to(t.device))
                state_dict[name] = v.flatten(axis, axis + 1).contiguous()
        if self.bc_conv1d is not None:
            b_idx = (idx[:, None] * self.R + torch.arange(self.R)).flatten()
            c_idx = (idx[:, None] * self.O + torch.arange(self.O)).flatten() + N * self.R
            conv_idx = torch.cat((b_idx, c_idx))
            for name in ('bc_conv1d.conv.weight', 'bc_conv1d.conv.bias'):
                t = state_dict[name]
                state_dict[name] = t.index_select(0, conv_idx.to(t.device)).contiguous()
        return state_dict

    def tok_proj_weight(self):
        mods = [getattr(self, a) for a in self._proj_attrs]
        ws = [self._proj_w(a) for a in self._proj_attrs]
        if not glue_on('wcat'):
            w = torch.cat(ws, dim=0)
            b = None
            if self._proj_any_bias:
                dtz = dt_fp32_on()
                b = torch.cat([m.bias if m.bias is not None and (not (dtz and a == 'dt_proj')) else self._proj_zero[:n] for a, m, n in zip(self._proj_attrs, mods, self._proj_widths)])
            return (w, b)
        cache = self._tok_proj_w_cache
        w_key = param_cache_key([m.weight for m in mods])
        w = cached_cat(cache.setdefault('w', {}), w_key, ws, dim=0)
        b = None
        if self._proj_any_bias:
            dtz = dt_fp32_on()
            b_tensors = [m.bias if m.bias is not None and (not (dtz and a == 'dt_proj')) else self._proj_zero[:n] for a, m, n in zip(self._proj_attrs, mods, self._proj_widths)]
            b_key = param_cache_key([m.bias for m in mods if m.bias is not None], dtz)
            b = cached_cat(cache.setdefault('b', {}), b_key, b_tensors, dim=0)
        return (w, b)

    def tok_proj_params(self):
        out = []
        for a in self._proj_attrs:
            mod = getattr(self, a)
            out.append(mod.weight)
            if mod.bias is not None:
                out.append(mod.bias)
        return out

    def split_tok_raw(self, raw: torch.Tensor) -> dict:
        if raw.shape[-1] != self.tok_width:
            raise ValueError(f"tok_raw last dim {raw.shape[-1]} != tok_width={self.tok_width} (the fused token-map projection's width; use `tok_proj_weight()` to build it)")
        out = dict(zip(self._proj_keys, raw.split(self._proj_widths, dim=-1)))
        if dt_fp32_on() and self.dt_proj.bias is not None:
            out['_dtb'] = self.dt_proj.bias
        if self._bc_adjacent:
            out['_bc'] = raw.narrow(-1, 0, self._bc_width)
        return out

    def _proj_split(self, u: torch.Tensor, channel_major: bool=False) -> dict:
        if not self._fuse_proj:
            dtz = dt_fp32_on() and self.dt_proj.bias is not None
            out = {k: F.linear(u, self._proj_w(a), None if dtz and a == 'dt_proj' else getattr(self, a).bias) for k, a in zip(self._proj_keys, self._proj_attrs)}
            if dtz:
                out['_dtb'] = self.dt_proj.bias
            return out
        return self.split_tok_raw(self.proj_slab(u, channel_major))

    def proj_slab(self, u: torch.Tensor, channel_major: bool=False) -> torch.Tensor:
        w, b = self.tok_proj_weight()
        if channel_major:
            outc = matmul_channel_major(w, u)
            if b is not None:
                outc = outc + b.to(outc.dtype)[:, None]
            out = outc.transpose(1, 2)
        else:
            out = linear_keep_layout(u, w, b)
        return out

    def _bc_source(self, parts, prefix=None):
        Bt, Ct = (parts['B'], parts['C'])
        if self.bc_conv1d is None:
            if prefix is not None:
                raise ValueError("state carry 'bc_conv_state' on a core built with bc_conv=False: there is no B/C convolution to prefix")
            return (Bt, Ct)
        if prefix is not None:
            if getattr(self, '_bc_capture', None) is not None:
                raise NotImplementedError("state carry 'bc_conv_state' inside the combined SSD-MIMO node: the node captures the convolution's own input and `combined_refusal` excludes carries -- ask it first")
            k1 = self.bc_conv1d.d_conv - 1
            cur = parts.get('_bc')
            if cur is None:
                cur = torch.cat([Bt, Ct], dim=-1)
            joint = torch.cat([prefix.to(cur.dtype), cur], dim=1)
            from ..layers.conv import causal_depthwise_conv1d
            bc = causal_depthwise_conv1d(joint.contiguous(), self.bc_conv1d.conv.weight, self.bc_conv1d.conv.bias, self.bc_conv1d.d_conv, activation='silu', allow_channel_last=True)[:, k1:]
            return bc.split([self.N * self.R, self.N * self.O], dim=-1)
        joint = parts.get('_bc')
        if joint is None:
            joint = torch.cat([Bt, Ct], dim=-1)
        cap = getattr(self, '_bc_capture', None)
        conv = self.bc_conv1d.conv
        view_ok = glue_on('bc') and _bc_narrow_conv_ready(joint)
        if cap is None:
            if view_ok:
                from ..layers.conv import causal_depthwise_conv1d_narrow_cl
                bc = causal_depthwise_conv1d_narrow_cl(joint, conv.weight, conv.bias, activation='silu')
            else:
                from ..layers.conv import causal_depthwise_conv1d
                bc = causal_depthwise_conv1d(joint.contiguous(), conv.weight, conv.bias, self.bc_conv1d.d_conv, activation='silu', allow_channel_last=True)
        else:
            from ..layers.conv import causal_conv1d_cl_fwd
            xcl = joint.transpose(1, 2)
            cap.append(xcl)
            out = torch.empty(joint.shape, device=joint.device, dtype=joint.dtype).transpose(1, 2) if view_ok else None
            bc = causal_conv1d_cl_fwd(xcl, conv.weight, conv.bias, 'silu', out=out).transpose(1, 2)
        return bc.split([self.N * self.R, self.N * self.O], dim=-1)

    def _tail_will_fuse(self, u: torch.Tensor) -> bool:
        from .. import tails as _tails
        return bool(self._fuse_proj and _tails.tails_enabled() and _tail_law_is_stock() and u.is_cuda and (u.dtype != torch.float64))

    def _fused_token_tails(self, parts, L: int, bc_prefix=None):
        from .. import tails as _tails
        if not (self._fuse_proj and _tails.tails_enabled() and _tail_law_is_stock()):
            return None
        N, R, O, H = (self.N, self.R, self.O, self.H)
        b_src, c_src = self._bc_source(parts, prefix=bc_prefix)
        gb_src = parts.get('gB')
        gc_src = parts.get('gC')
        dt_src = parts['dt']
        dtb = parts.get('_dtb')
        a_src = parts.get('a')
        if not _tails.can_fuse(b_src, c_src, gb_src, gc_src, dt_src, a_src):
            return None
        shared_bias = bool(self.bc_bias) and (not self.head_bias)
        fold_gb = bool(self.use_g_b) and R == 1
        fold_gc = bool(self.use_g_c)
        need_b = bool(self.b_norm or shared_bias or fold_gb)
        need_c = bool(self.c_norm or shared_bias or fold_gc)

        def _eps(norm):
            e = norm.eps
            return float(torch.finfo(torch.float32).eps if e is None else e)
        cfg = _tails.TailCfg(N=N, R=R, O=O, H=H, L=L, bc_norm=bool(self.b_norm or self.c_norm), b_norm=bool(self.b_norm), c_norm=bool(self.c_norm), shared_bias=shared_bias, fold_gb=fold_gb, fold_gc=fold_gc, a_dyn=bool(self.a_dyn), a_per_sibling=bool(self.a_dyn_per_sibling), has_dt_bias=dtb is not None, has_a_bias=False, eps_b=_eps(self.B_norm) if self.b_norm else 0.0, eps_c=_eps(self.C_norm) if self.c_norm else 0.0, a_floor=0.0001, a_heavy_tail=self.a_act == 'heavy_tail')

        def _odt(base, bias, gate):
            dt = base.dtype
            if bias is not None:
                dt = torch.promote_types(dt, bias.dtype)
            if gate is not None:
                dt = torch.promote_types(dt, gate.dtype)
            return dt
        Bt, Ct, dt_k, dec_k = _tails.token_tails(cfg, b_src=b_src if need_b else None, c_src=c_src if need_c else None, gb_src=gb_src if fold_gb else None, gc_src=gc_src if fold_gc else None, dt_src=dt_src, a_src=a_src if self.a_dyn else None, w_b=self.B_norm.weight if self.b_norm else None, w_c=self.C_norm.weight if self.c_norm else None, bias_b=self.B_bias if shared_bias else None, bias_c=self.C_bias if shared_bias else None, bias_dt=dtb, bias_a=None, need_b=need_b, need_c=need_c, out_dtype_b=_odt(b_src, self.B_bias if shared_bias else None, gb_src if fold_gb else None), out_dtype_c=_odt(c_src, self.C_bias if shared_bias else None, gc_src if fold_gc else None))
        B = dt_src.shape[0]
        if not need_b:
            Bt = b_src.view(B, L, N, R)
        if not need_c:
            Ct = c_src.view(B, L, N, O)
        gB = None if fold_gb or not self.use_g_b else 1.0 + gb_src.view(B, L, N, R)
        return (Bt, Ct, gB, dt_k, dec_k)

    def _token_maps(self, x: torch.Tensor, bc_conv_state=None, parts=None, wide: bool=True, bc_prefix=None):
        B, L, _ = x.shape
        H, P, G, R, N = (self.H, self.P, self.G, self.R, self.N)
        if parts is None:
            parts = self._proj_split(x)
        Bt = parts['B']
        Ct = parts['C']
        new_conv_state = None
        if self.bc_conv1d is not None:
            if bc_conv_state is None:
                Bt, Ct = self._bc_source(parts, prefix=bc_prefix)
            else:
                bc = torch.cat([Bt, Ct], dim=-1)
                bc, new_conv_state = self.bc_conv1d.step(bc.squeeze(1), bc_conv_state)
                bc = bc.unsqueeze(1)
                Bt, Ct = bc.split([N * R, N * self.O], dim=-1)
        Bt = Bt.view(B, L, N, R)
        Ct = Ct.view(B, L, N, self.O)
        if self.b_norm:
            Bt = rms_norm_cast(Bt.transpose(-1, -2), self.B_norm.weight, self.B_norm.eps).transpose(-1, -2)
        if self.c_norm:
            Ct = rms_norm_cast(Ct.transpose(-1, -2), self.C_norm.weight, self.C_norm.eps).transpose(-1, -2)
        if self.bc_bias:
            if self.head_bias:
                pass
            else:
                Bt = Bt + self.B_bias[:, None]
                Ct = Ct + self.C_bias[:, None]
        gB = 1.0 + parts['gB'].view(B, L, N, R) if self.use_g_b else None
        gC = 1.0 + parts['gC'].view(B, L, N, self.O) if self.use_g_c else None
        if parts.get('_dtb') is not None:
            _w = torch.promote_types(parts['dt'].dtype, torch.float32)
            dt_raw = _dt_tail(parts['dt'].to(_w) + parts['_dtb'].to(_w), self._compile_token_maps)
        else:
            dt_raw = _dt_tail(parts['dt'], self._compile_token_maps)
        if self.a_dyn:
            a_mag = _a_mag_tail(parts['a'], 0.0001, self._compile_token_maps, self.a_act)
            if self.a_dyn_per_sibling:
                dt_dec_raw = dt_raw.view(B, L, H, R) * a_mag.view(B, L, H, R)
            else:
                dt_dec_raw = dt_raw.view(B, L, H, R) * a_mag.unsqueeze(-1)
            dt_dec = dt_dec_raw.view(B, L, H, 1, R).expand(B, L, H, G, R).reshape(B, L, H, P) if wide else dt_dec_raw
        else:
            dt_dec = None
        dt = dt_raw.view(B, L, H, 1, R).expand(B, L, H, G, R).reshape(B, L, H, P) if wide else dt_raw.view(B, L, H, R)
        if bc_conv_state is not None:
            return (Bt, Ct, gB, gC, dt, dt_dec, new_conv_state)
        return (Bt, Ct, gB, gC, dt, dt_dec)

    def _select(self, x: torch.Tensor, sel_x: torch.Tensor | None) -> torch.Tensor:
        if self.proj_source == 'layer_input':
            if sel_x is None:
                raise ValueError("SSDMimoSSM(proj_source='layer_input') needs sel_x (the block input, width d_model) in forward/step; the mixer passes it — or build the core with proj_source='main_branch'")
            u = sel_x
        else:
            u = x
        if u.shape[-1] != self.d_sel:
            raise ValueError(f'selection input width {u.shape[-1]} != d_sel={self.d_sel} (proj_source={self.proj_source!r})')
        return u

    def forward(self, x: torch.Tensor, *, gate_z: torch.Tensor | None=None, sel_x: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None, state: State | dict | None=None, return_state: bool=False, out_norm=None) -> torch.Tensor | tuple[torch.Tensor, State]:
        output_gate = gate_z if self.R > 1 or self.O > 1 else None
        if gate_z is not None and gate_z.shape != (*x.shape[:-1], self.d_out):
            raise ValueError(f'gate_z must have shape {(*x.shape[:-1], self.d_out)}')
        if output_gate is not None:
            if out_norm is not None:
                raise NotImplementedError('MIMO output norm must follow the output gate in the mixer')
            gate_z = None
        h_init = None
        if state is not None:
            state = self.resolve_state(state, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
            h_init = state.hidden_state if isinstance(state, State) else state['hidden_state']
        self._carry_dtype = h_init.dtype if h_init is not None else None
        extra_carry = {k: v for k, v in state.items() if k != 'hidden_state'} if state is not None else {}
        backend = self.backend
        if backend == 'auto':
            backend = 'cuda' if x.is_cuda else 'torch'
        if self.m3_kernel == 'upstream' and (not x.is_cuda):
            raise ValueError("m3_kernel='upstream' batched forward requires CUDA")
        u = self._select(x, sel_x)
        if out_norm is not None and (backend != 'cuda' or extra_carry or return_state):
            raise NotImplementedError(f'out_norm= is fused into the CUDA gate epilogue; this call resolved to backend={backend!r}, extra_carry={bool(extra_carry)}.  Ask `out_norm_refusal` first and apply the norm module-side when it names a clause.')
        self._keep_state_graph = bool(return_state and torch.is_grad_enabled())
        chunk_why = self._carry_refusal_chunked(extra_carry)
        cuda_why = self._carry_refusal_cuda(extra_carry) if chunk_why is None else None
        if backend == 'cuda' and cuda_why is not None:
            backend = 'torch'
        norm_pending = out_norm is not None
        if x.shape[1] == 0 or (extra_carry and chunk_why is not None):
            self._route_note = chunk_why
            y = self._forward_sequential(x, u, h_init, gate_z, init_extra=extra_carry, tok_raw=tok_raw)
            full_state = self._seq_final_state
        elif backend == 'cuda':
            self._route_note = None
            full_state = None
            fell_back = False
            try:
                y = self._forward_cuda(x, u, h_init, gate_z, tok_raw, out_norm=out_norm, carry=extra_carry)
            except NotImplementedError as why:
                self._route_note = f'cuda route envelope: {why}'
                y, full_state = self._forward_torch(x, u, h_init, gate_z, tok_raw, carry=extra_carry, return_state=return_state)
                fell_back = True
            if not fell_back:
                norm_pending = False
            if not fell_back and return_state:
                full_state = self._collect_fast_state(x, u, tok_raw, carry=extra_carry)
        elif backend == 'torch':
            self._route_note = cuda_why
            y, full_state = self._forward_torch(x, u, h_init, gate_z, tok_raw, carry=extra_carry, return_state=return_state)
        elif backend == 'sequential':
            self._route_note = None
            y = self._forward_sequential(x, u, h_init, gate_z, init_extra=extra_carry, tok_raw=tok_raw)
            full_state = self._seq_final_state
        else:
            raise ValueError(f'unknown backend {backend!r}')
        if norm_pending:
            w_n, eps_n = out_norm
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + eps_n) * w_n.float()).to(y.dtype)
        if output_gate is not None:
            y = y * F.silu(output_gate)
        if not return_state:
            return y
        self._final_state = full_state
        return (y, full_state)
    FAST_CARRY_KEYS = ('rot_ang', 'write_prev', 'bc_conv_state')

    def _carry_refusal_chunked(self, carry) -> str | None:
        if not carry:
            return None
        unknown = sorted(set(carry) - set(self.FAST_CARRY_KEYS))
        if unknown:
            return f'state carries {unknown} are not threaded by the chunked/kernel input build (threaded: {list(self.FAST_CARRY_KEYS)})'
        if 'rot_ang' in carry and self.rot_mode != 'm3':
            return f"state carry 'rot_ang' with rot_mode={self.rot_mode!r}: the scan has no phase to offset"
        if 'write_prev' in carry and (not self.use_trap):
            return "state carry 'write_prev' with write_rule='euler': the scan's one-tap write has no trapezoid lag to fold"
        if 'bc_conv_state' in carry and self.bc_conv1d is None:
            return "state carry 'bc_conv_state' with bc_conv=False: there is no B/C convolution to prefix"
        return None

    def _carry_refusal_cuda(self, carry) -> str | None:
        if not carry:
            return None
        if self.m3_kernel == 'upstream':
            return f"m3_kernel='upstream': the upstream M3 adapter takes no initial state at all, so state carries {sorted(carry)} cannot ride it"
        return None

    def last_route(self) -> str | None:
        return getattr(self, '_route_note', None)

    def state_grad_refusal(self, key: str) -> str | None:
        have = set(self._state_schema().keys())
        if key not in have:
            return f'state carry {key!r} is not carried by this config (carried: {sorted(have)}), so it has no gradient; asking for one is a configuration error, not a zero'
        return None

    def _collect_fast_state(self, x: torch.Tensor, u: torch.Tensor, tok_raw: torch.Tensor | None=None, carry: dict | None=None) -> State:
        carry = carry or {}
        st = State(hidden_state=self.h_final)
        if self.rot_mode == 'm3':
            st['rot_ang'] = self._rot_ang_final
        if self.use_trap:
            key = 'write_prev'
            st[key] = self._trap_w_final(x, u, tok_raw, carry=carry)
        if self.bc_conv1d is not None:
            kc = self.bc_conv1d.d_conv
            if tok_raw is None:
                tail = u[:, -kc:]
                parts_t = self._proj_split(tail)
            else:
                parts_t = self.split_tok_raw(tok_raw[:, -kc:])
            bc_tail = torch.cat([parts_t['B'], parts_t['C']], dim=-1)
            if bc_tail.shape[1] < kc:
                bc_tail = torch.cat([self._bc_window_pad(carry.get('bc_conv_state'), kc, kc - bc_tail.shape[1], bc_tail), bc_tail], dim=1)
            st['bc_conv_state'] = self._state_leaf(bc_tail.transpose(1, 2))
        return st

    @staticmethod
    def _bc_window_pad(prev, kc: int, need: int, like: torch.Tensor):
        if prev is None:
            return like.new_zeros(like.shape[0], need, like.shape[-1])
        return prev[:, :, kc - need:].transpose(1, 2).to(like.dtype)

    def _state_leaf(self, t: torch.Tensor) -> torch.Tensor:
        return t if getattr(self, '_keep_state_graph', False) else t.detach()

    def _trap_w_final(self, x: torch.Tensor, u: torch.Tensor, tok_raw: torch.Tensor | None=None, carry: dict | None=None):
        carry = carry or {}
        L = u.shape[1]
        prev = State(hidden_state=torch.zeros_like(self.h_final, dtype=x.dtype))
        if self.rot_mode == 'm3':
            prev['rot_ang'] = self._rot_ang_prev
        key = 'write_prev'
        prev[key] = torch.zeros_like(self.h_final, dtype=x.dtype)
        if self.bc_conv1d is not None:
            kc = self.bc_conv1d.d_conv
            if tok_raw is None:
                parts_t = self._proj_split(u[:, -kc - 1:-1]) if L > 1 else None
            else:
                parts_t = self.split_tok_raw(tok_raw[:, -kc - 1:-1]) if L > 1 else None
            if parts_t is None:
                bc_tail = None
            else:
                bc_tail = torch.cat([parts_t['B'], parts_t['C']], dim=-1)
            width = self.N * self.R + self.N * self.O
            if bc_tail is None or bc_tail.shape[1] < kc:
                have = 0 if bc_tail is None else bc_tail.shape[1]
                like = bc_tail if bc_tail is not None else u.new_zeros(u.shape[0], 0, width)
                pad = self._bc_window_pad(carry.get('bc_conv_state'), kc, kc - have, like)
                bc_tail = pad if bc_tail is None else torch.cat([pad, bc_tail], dim=1)
            prev['bc_conv_state'] = self._state_leaf(bc_tail.transpose(1, 2))
        saved_h, saved_state = (self.h_final, getattr(self, '_final_state', None))
        keep = getattr(self, '_keep_state_graph', False)
        with torch.enable_grad() if keep else torch.no_grad():
            _, st_last = self._step_legacy(x[:, -1], prev, sel_x=u[:, -1] if self.proj_source == 'layer_input' else None, tok_raw=tok_raw[:, -1] if tok_raw is not None else None)
        self.h_final, self._final_state = (saved_h, saved_state)
        return self._state_leaf(st_last[key])

    def _forward_torch(self, x: torch.Tensor, u: torch.Tensor, h_init: torch.Tensor | None=None, gate_z: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None, carry: dict | None=None, return_state: bool=False) -> tuple[torch.Tensor, State | None]:
        if _torch_impl() == 'legacy':
            y = self._forward_torch_legacy(x, u, h_init, gate_z, tok_raw, carry=carry)
            st = self._collect_fast_state(x, u, tok_raw, carry=carry) if return_state else None
            return (y, st)
        from .ssd_mimo_torch import ssd_chunked, ssd_scan_seq, torch_route
        if torch_route(x.shape[1]) == 'scan':
            y, st, hidden = ssd_scan_seq(self, x, u, h0=h_init, gate_z=gate_z, tok_raw=tok_raw, carry=carry, want_state=return_state)
        else:
            chunk = self.chunk_size if isinstance(self.chunk_size, int) and (not isinstance(self.chunk_size, bool)) else None
            y, st, hidden = ssd_chunked(self, x, u, h0=h_init, gate_z=gate_z, tok_raw=tok_raw, carry=carry, want_state=return_state, chunk_size=chunk)
        self.h_final = hidden
        return (y, st)

    def _forward_torch_legacy(self, x: torch.Tensor, u: torch.Tensor, h_init: torch.Tensor | None=None, gate_z: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None, carry: dict | None=None) -> torch.Tensor:
        from ..kernels.ssd_mimo.reference import unified_ssd_chunked
        kw = self._prepare_kernel_inputs(x, u, h_init, gate_z, tok_raw, carry=carry)
        y, s_final = unified_ssd_chunked(kw['x_k'], kw['delta_k'], kw['B_k'], kw['C_k'], kw['a_k'], beta=kw['beta_k'], gamma=kw['gamma_k'], theta=kw['theta_k'], theta_h=kw['thh_k'], rot_bias=kw['rb_k'], bias_pre_rot=kw['bias_pre_rot'], delta_w=kw['delta_wk'], x0=kw['x0'], gate_z=kw['gz_k'], d_skip=kw['dsk_k'], bc_per_head=True, pair_layout=self.pair_layout)
        y = self._apply_m3_correction(y, kw)
        return self._finish_forward(x, u, y, s_final, kw.get('rmix_k'), gate_z=gate_z)

    def _forward_sequential(self, x: torch.Tensor, u: torch.Tensor, h_init: torch.Tensor | None=None, gate_z: torch.Tensor | None=None, init_extra: dict | None=None, tok_raw: torch.Tensor | None=None) -> torch.Tensor:
        B, L, _ = x.shape
        state = self.init_state(B, device=x.device, dtype=x.dtype, h_init=h_init, overrides=init_extra)
        ys = []
        for t in range(L):
            y_t, state = self.step(x[:, t], state, gate_z=gate_z[:, t] if gate_z is not None else None, sel_x=u[:, t] if self.proj_source == 'layer_input' else None, tok_raw=tok_raw[:, t] if tok_raw is not None else None)
            ys.append(y_t)
        self._seq_final_state = state
        self.h_final = state['hidden_state']
        return torch.stack(ys, dim=1) if ys else x.new_empty(B, 0, self.d_out)

    def _prepare_kernel_inputs(self, x: torch.Tensor, u: torch.Tensor, h_init: torch.Tensor | None=None, gate_z: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None, in_kernel_trap: bool=False, in_kernel_phase: bool=False, m3_host_fold: bool=False, rot_host_fold: bool | None=None, carry: dict | None=None) -> dict:
        B, L, D = x.shape
        H, P, G, R, N = (self.H, self.P, self.G, self.R, self.N)
        carry = carry or {}
        host_fold = bool(m3_host_fold) if rot_host_fold is None else bool(rot_host_fold) or bool(m3_host_fold)
        bc_prefix = None
        if carry.get('bc_conv_state') is not None and self.bc_conv1d is not None:
            kc = self.bc_conv1d.d_conv
            if kc > 1:
                bc_prefix = carry['bc_conv_state'][:, :, 1:].transpose(1, 2)
        parts = self._proj_split(u, channel_major=self._tail_owns_all and u.dim() == 3 and (u.stride(-1) != 1) and self._tail_will_fuse(u)) if tok_raw is None else self.split_tok_raw(tok_raw)
        Hk = R * H
        xi = x.view(B, L, H, G, R).permute(0, 2, 3, 4, 1)

        def _dk(v):
            return v.permute(0, 3, 2, 1).reshape(B, Hk, L).to(torch.float64 if v.dtype == torch.float64 else torch.float32).contiguous()
        fused = self._fused_token_tails(parts, L, bc_prefix=bc_prefix)
        if fused is not None:
            Bt, Ct, gB, dt_k, dec_k = fused
            gC = None
            delta_k = dec_k if self.a_dyn else dt_k
            delta_wk = dt_k if self.a_dyn else None
            dt = dt_dec = None
        else:
            Bt, Ct, gB, gC, dt, dt_dec = self._token_maps(u, parts=parts, wide=False, bc_prefix=bc_prefix)
            dt_k = _dk(dt)
            delta_k = dt_k if dt_dec is None else _dk(dt_dec)
            delta_wk = None if dt_dec is None else dt_k
        endpoint = None
        trap_nxt = None
        trap_lam_k = None
        lam_k = None
        if self.use_trap:
            dt_w = dt_k
            _tr = parts['trap']
            lam = torch.sigmoid(_tr.float() if _tr.dtype in (torch.bfloat16, torch.float16) else _tr)
            lam_k = lam.permute(0, 2, 1).unsqueeze(1).expand(B, R, H, L).reshape(B, Hk, L).to(dt_w.dtype)
            if in_kernel_trap:
                trap_lam_k = lam_k
                delta_wk = dt_w
            elif x.is_cuda and x.dtype != torch.float64:
                from ..kernels.ssd_mimo.endpoint import endpoint_weights
                delta_wk, trap_nxt = endpoint_weights(dt_w, lam_k)
            else:
                trap_nxt = F.pad((dt_w * (1 - lam_k))[..., 1:], (0, 1))
                delta_wk = lam_k * dt_w + trap_nxt
        x_k = xi.unsqueeze(1).expand(B, R, H, G, R, L).reshape(B, Hk, G, R, L)
        if gB is None:
            B_k = Bt.unsqueeze(2)
        else:
            B_k = gB.permute(0, 1, 3, 2).unsqueeze(-1) * Bt.unsqueeze(2)
        if self.O == 1:
            gated_c = Ct if gC is None else Ct * gC
            C_k = gated_c.permute(0, 1, 3, 2)
        else:
            gated_c = Ct if gC is None else gC * Ct
            C_k = gated_c[:, :, None]
        if self.A_log is None:
            a_k = _ref_f(torch.zeros(R * H, device=x.device, dtype=x.dtype))
        else:
            a_k = _ref_f(self.A_log.permute(1, 0).reshape(R * H))

        def _to_kernel_state(v):
            return _ref_f(v.view(B, H, G, R, N).permute(0, 3, 1, 2, 4).reshape(B, R * H, G, N))
        x0 = None
        if h_init is not None:
            x0 = _to_kernel_state(h_init)
        w_prev = carry.get('write_prev')
        if w_prev is not None:
            if lam_k is None:
                raise ValueError("state carry 'write_prev' with write_rule='euler': the scan's one-tap write has no trapezoid lag to fold")
            c0 = dt_k[:, :, 0] * (1.0 - lam_k[:, :, 0])
            lag = _to_kernel_state(w_prev) * c0[:, :, None, None]
            x0 = lag if x0 is None else x0 + lag
        beta_k = None
        gamma_k = None
        if self.head_bias:
            beta_k = self.B_bias.permute(1, 0, 2, 3).reshape(R * H, N, R).to(a_k.dtype).contiguous()
            gamma_k = self.C_bias.permute(1, 0, 2, 3).reshape(R * H, N, self.O).to(a_k.dtype).contiguous()
        gz_k = None
        if gate_z is not None:
            if R == 1:
                from ..kernels.ssd_mimo.autograd import _gate_epilogue_reads_strided
                gz_k = gate_z if _gate_epilogue_reads_strided() else gate_z.contiguous()
            else:
                gz_k = gate_z.view(B, L, H, G, R).permute(0, 1, 4, 2, 3).reshape(B, L, R * H * G).contiguous()
        dsk_k = None
        if self.d_skip and self._dskip_route(x, gate_z) == 'in_scan':
            dsk_k = self._skip_gains().to(a_k.dtype).reshape(H, G, 1)
        thh_k = None
        rb_k = None

        def _expand_heads(v):
            if v.shape[2] == Hk:
                return v
            if v.shape[2] == 1:
                return v.expand(B, L, Hk, *v.shape[3:])
            return v.unsqueeze(3).expand(B, L, R, H, *v.shape[3:]).reshape(B, L, Hk, *v.shape[3:])

        def _rot_fold(v, th):
            c, s_ = (torch.cos(th), torch.sin(th))
            if v.dim() == 5:
                c, s_ = (c[..., None], s_[..., None])
            if self.pair_layout == 'half':
                Kh = N // 2
                e = v[:, :, :, :Kh]
                o = v[:, :, :, Kh:]
                return torch.cat([c * e + s_ * o, -s_ * e + c * o], dim=3)
            e = v[:, :, :, 0::2]
            o = v[:, :, :, 1::2]
            out = torch.stack([c * e + s_ * o, -s_ * e + c * o], dim=4)
            return out.reshape(v.shape[:3] + (N,) + v.shape[4:])
        if self.use_theta_h:
            nu_k = self._theta_h_pairs().unsqueeze(0).expand(R, H, N // 2).reshape(Hk, N // 2)
        if self.use_theta_h and (not host_fold):
            thh_k = nu_k.contiguous()
        elif self.use_theta_h:
            C5 = _expand_heads(C_k if C_k.dim() == 5 else C_k.unsqueeze(-1))
            C_k = _rot_fold(_ref_f(C5), nu_k[None, None]).to(x.dtype)
            if self.O == 1:
                C_k = C_k[..., 0]
            if gamma_k is not None:
                g5 = gamma_k.unsqueeze(0).unsqueeze(0)
                gamma_k = _rot_fold(_ref_f(g5), nu_k[None, None])[0, 0].to(gamma_k.dtype).contiguous()
        theta_k = None
        theta_lo_k = None
        theta_incr = False
        m3_pre = None
        if self.rot_mode == 'm3':
            ka = self.n_rope_angles
            dt_raw_k = dt_k
            use_phase_kernel = x.is_cuda and parts['theta'].dtype != torch.float64 and (_os.environ.get('SSD_MIMO_M3_PHASE_KERNEL', '1') == '1')
            omega = None
            if self.learn_theta_freq and self.omega_layout == 'head':
                omega = torch.tanh(_ref_f(self.theta_freq))
            elif self.learn_theta_freq and self.omega_layout == 'head_pair':
                omega = F.pad(torch.tanh(_ref_f(self.theta_freq)), (0, N // 2 - ka))
            elif self.learn_theta_freq:
                omega = F.pad(torch.tanh(_ref_f(self.theta_freq)), (0, N // 2 - ka))
            raw_theta = parts['theta']
            phase_fused = use_phase_kernel and _os.environ.get('SSD_MIMO_PHASE_FUSED', '1') == '1'
            if phase_fused:
                from ..kernels.ssd_mimo.m3_epilogue import phase_law
                incr = None
                phase_lo = torch.bfloat16 if x_k.dtype in (torch.bfloat16, torch.float16) and self.precision != 'ieee' and (_os.environ.get('SSD_MIMO_PHASE_LO', '1') == '1') and (_os.environ.get('SSD_MIMO_THETA_BWD_FP32', '1') == '0') else None
                if self.omega_per_head and _os.environ.get('SSD_MIMO_OMEGA_HEAD_KERNEL', '1') == '0':
                    ang_b = raw_theta.reshape(B, L, H, 1).expand(B, L, H, ka)
                    dt_b = dt_raw_k if omega is None else dt_raw_k * omega.repeat(R).to(dt_raw_k.dtype)[None, :, None]
                    theta_k = phase_law(ang_b, dt_b, None, H, N // 2, sign=-1.0, lo_dtype=phase_lo)
                else:
                    theta_k = phase_law(raw_theta.reshape(B, L, H if self.theta_per_head else 1, self.n_theta_feat), dt_raw_k, omega, H, N // 2, sign=-1.0, lo_dtype=phase_lo, ka=ka if self.omega_per_head else 0, omega_layout=self.omega_layout)
                if phase_lo is not None:
                    theta_k, theta_lo_k = theta_k
            else:
                incr = _theta_increments(raw_theta, dt_raw_k, omega, H=H, R=R, K=N // 2, ka=ka, per_head=self.theta_per_head, use_kernel=use_phase_kernel and self.omega_layout == 'pair', sign=-1.0, omega_per_head=self.omega_per_head)
            theta_incr = in_kernel_phase and (not host_fold) and (not phase_fused) and (_os.environ.get('SSD_MIMO_PHASE_INKERNEL', '1') == '1')
            if not phase_fused:
                theta_k = incr if theta_incr else _wrap2pi_t(incr.cumsum(dim=1))
            rot_carry = carry.get('rot_ang')
            if rot_carry is not None:
                off = F.pad(_ref_f(rot_carry).permute(0, 2, 1, 3).reshape(B, Hk, ka), (0, N // 2 - ka))
                if theta_incr:
                    theta_k = incr.cumsum(dim=1)
                    theta_incr = False
                theta_k = _wrap2pi_t(theta_k + off[:, None].to(theta_k.dtype))
                if theta_lo_k is not None:
                    theta_lo_k = theta_k.to(theta_lo_k.dtype)

            def _ang_carry(v):
                v = v[..., :ka]
                if R == 1:
                    return self._state_leaf(v.unsqueeze(2)).clone()
                return self._state_leaf(v.reshape(B, R, H, ka).permute(0, 2, 1, 3)).clone()

            def _ang_prev_carry():
                if rot_carry is None:
                    return torch.zeros_like(self._rot_ang_final)
                return self._state_leaf(_ref_f(rot_carry)).clone()
            if theta_incr:
                tot = incr.sum(dim=1)
                self._rot_ang_final = _ang_carry(_wrap2pi_t(tot))
                self._rot_ang_prev = _ang_prev_carry() if L == 1 else _ang_carry(_wrap2pi_t(tot - incr[:, -1]))
            else:
                self._rot_ang_final = _ang_carry(theta_k[:, -1])
                self._rot_ang_prev = _ang_prev_carry() if L == 1 else _ang_carry(theta_k[:, -2])
        bias_pre_rot = self.bias_pre_rot
        if self.rot_mode == 'm3' and host_fold:
            B_pre = _ref_f(_expand_heads(B_k))
            C_pre = _ref_f(_expand_heads(C_k))
            if beta_k is not None:
                B_pre = B_pre + beta_k.to(B_pre.dtype)[None, None]
            if gamma_k is not None:
                gam = gamma_k if C_pre.dim() == 5 else gamma_k[..., 0]
                C_pre = C_pre + gam.to(C_pre.dtype)[None, None]
            m3_pre = (B_pre, C_pre)
            B_k = _rot_fold(B_pre, theta_k).to(x.dtype)
            C_k = _rot_fold(C_pre, theta_k).to(x.dtype)
            beta_k = gamma_k = theta_k = None
            bias_pre_rot = False
        m3_corr = None
        if trap_nxt is not None and self.rot_mode == 'm3' and x.is_cuda and (x.dtype != torch.float64):
            from ..kernels.ssd_mimo.endpoint import endpoint_local_map
            if m3_pre is None:
                eb, ec, ebias, cbias = (B_k, C_k, beta_k, gamma_k)
            else:
                eb, ec = m3_pre
                ebias = cbias = None
            matrix = endpoint_local_map(eb if eb.dim() == 5 else eb.unsqueeze(-1), ec if ec.dim() == 5 else ec.unsqueeze(-1), None, ebias, cbias, None, thh_k, Hk)
            endpoint = (matrix, trap_nxt)
        elif trap_nxt is not None and self.rot_mode == 'm3' and (R == 1) and (self.O == 1) and (m3_pre is None) and (C_k.shape[2] == 1):
            Bv = B_k[:, :, 0, :, 0] if B_k.dim() == 5 else B_k[:, :, 0]
            Cv = C_k[:, :, 0, :, 0] if C_k.dim() == 5 else C_k[:, :, 0]
            Bv, Cv = (_ref_f(Bv), _ref_f(Cv))
            bb = beta_k[:, :, 0]
            cb = gamma_k[:, :, 0]
            if thh_k is not None:
                Cr = _rot_fold(Cv[:, :, None, :, None].expand(B, L, Hk, N, 1), thh_k[None, None])[..., 0]
                cr = _rot_fold(cb[None, None, :, :, None], thh_k[None, None])[0, 0, :, :, 0]
                dot = torch.einsum('bln,blhn->blh', Bv, Cr) + Bv @ cr.t() + torch.einsum('hn,blhn->blh', bb, Cr) + (bb * cr).sum(-1)
            else:
                dot = (Bv * Cv).sum(-1)[:, :, None] + Bv @ cb.t() + Cv @ bb.t() + (bb * cb).sum(-1)
            m3_corr = trap_nxt[:, :, None, :] * dot.permute(0, 2, 1)[:, :, None, :] * _ref_f(x_k[:, :, :, 0, :])
        elif trap_nxt is not None and self.rot_mode == 'm3':
            Oo = self.O
            if m3_pre is None:
                Bg = _ref_f(_expand_heads(B_k if B_k.dim() == 5 else B_k.unsqueeze(-1)))
                Cg = _ref_f(_expand_heads(C_k if C_k.dim() == 5 else C_k.unsqueeze(-1)))
                if beta_k is not None:
                    Bg = Bg + beta_k.to(Bg.dtype)[None, None]
                if gamma_k is not None:
                    Cg = Cg + gamma_k.to(Cg.dtype)[None, None]
                if thh_k is not None:
                    Cg = _rot_fold(Cg, thh_k[None, None])
                M = torch.einsum('blknr,blkno->blkro', Bg, Cg)
            else:
                B_pre, C_pre = m3_pre
                C_pre5 = C_pre if C_pre.dim() == 5 else C_pre.unsqueeze(-1)
                M = torch.einsum('blknr,blkno->blkro', B_pre, C_pre5)
            corr = torch.einsum('bkgrl,blkro->bkogl', _ref_f(x_k), M)
            corr = corr * trap_nxt[:, :, None, None, :]
            m3_corr = corr[:, :, 0] if Oo == 1 else corr
        if trap_nxt is not None and self.rot_mode != 'm3':
            phase = theta_k.cumsum(1) if theta_incr else theta_k
            if x.is_cuda and x.dtype != torch.float64:
                from ..kernels.ssd_mimo.endpoint import endpoint_local_map
                local_map = endpoint_local_map(B_k, C_k if C_k.dim() == 5 else C_k.unsqueeze(-1), phase, beta_k, gamma_k, rb_k, thh_k, Hk)
            else:
                be = _ref_f(_expand_heads(B_k))
                ce = _ref_f(_expand_heads(C_k))
                if ce.dim() == 4:
                    ce = ce.unsqueeze(-1)
                if phase is not None:
                    be, ce = (_rot_fold(be, phase), _rot_fold(ce, phase))
                if rb_k is not None:
                    be = _rot_fold(be, rb_k[None, None])
                if beta_k is not None:
                    be = be + beta_k[None, None]
                if gamma_k is not None:
                    ce = ce + gamma_k[None, None]
                if thh_k is not None:
                    ce = _rot_fold(ce, thh_k[None, None])
                local_map = torch.einsum('blknr,blkno->blkro', be, ce)
            if x.is_cuda and x.dtype != torch.float64:
                endpoint = (local_map, trap_nxt)
            else:
                corr = torch.einsum('bkgrl,blkro->bkogl', _ref_f(x_k), local_map)
                corr = corr * trap_nxt[:, :, None, None, :]
                m3_corr = corr[:, :, 0] if self.O == 1 else corr
        return dict(x_k=x_k, delta_k=delta_k, delta_wk=delta_wk, B_k=B_k, C_k=C_k, a_k=a_k, x0=x0, beta_k=beta_k, gamma_k=gamma_k, gz_k=gz_k, dsk_k=dsk_k, trap_lam_k=trap_lam_k, thh_k=thh_k, rb_k=rb_k, theta_k=theta_k, theta_lo_k=theta_lo_k, theta_incr=theta_incr, bias_pre_rot=bias_pre_rot, rmix_k=parts.get('rmix'), m3_corr=m3_corr, endpoint=endpoint)

    def _forward_upstream_m3(self, x, u, gate_z, tok_raw):
        from ..kernels.m3_adapter import siso, mimo
        B, L, _ = x.shape
        H, R, O, G, N = (self.H, self.R, self.O, self.G, self.N)
        Hk = H * R
        parts = self._proj_split(u) if tok_raw is None else self.split_tok_raw(tok_raw)
        fused = self._fused_token_tails(parts, L)
        if fused is not None:
            bt, ct, gb, dt_k, dec_k = fused
            gc = None
            adt = -dec_k if self.a_dyn else -dt_k * self.A_log.T.reshape(1, Hk, 1).float().exp()
        else:
            clock_parts = {**parts, 'dt': parts['dt'].float()}
            if 'a' in parts:
                clock_parts['a'] = parts['a'].float()
            bt, ct, gb, gc, dt, dec = self._token_maps(u, parts=clock_parts, wide=False)
            dk = lambda t: t.permute(0, 3, 2, 1).reshape(B, Hk, L).float().contiguous()
            dt_k = dk(dt)
            adt = -dk(dec) if dec is not None else -dt_k * self.A_log.T.reshape(1, Hk, 1).float().exp()
        trap = parts['trap'].transpose(1, 2)[:, None].expand(B, R, H, L).reshape(B, Hk, L).float().contiguous()
        ka = self.n_rope_angles
        angles = parts['theta'].reshape(B, L, H if self.theta_per_head else 1, ka)
        angles = angles.expand(B, L, H, ka)[:, :, None].expand(B, L, R, H, ka).reshape(B, L, Hk, ka).float()
        angles = F.pad(angles, (0, N // 2 - ka)).contiguous()
        with torch.no_grad():
            inc = -angles.tanh() * math.pi * dt_k.transpose(1, 2)[..., None]
            total = inc.sum(1)
            carry = lambda a: a[..., :ka].reshape(B, R, H, ka).permute(0, 2, 1, 3).clone()
            self._rot_ang_final = carry(_wrap2pi_t(total))
            self._rot_ang_prev = carry(_wrap2pi_t(total - inc[:, -1]))
        bk = bt[:, :, None] if gb is None else gb.permute(0, 1, 3, 2)[..., None] * bt[:, :, None]
        ck = (ct if gc is None else ct * gc)[:, :, None]
        bk = bk.expand(B, L, R, N, R)
        ck = ck.expand(B, L, R, N, O)
        beta = self.B_bias.permute(1, 0, 2, 3).reshape(Hk, N, R) if self.head_bias else x.new_zeros(Hk, N, R)
        gamma = self.C_bias.permute(1, 0, 2, 3).reshape(Hk, N, O) if self.head_bias else x.new_zeros(Hk, N, O)
        if R == 1:
            v = x.reshape(B, L, H, G).contiguous()
            z = None if gate_z is None else gate_z.reshape(B, L, H, G)
            result = siso(Q=ck[..., 0], K=bk[..., 0], V=v, ADT=adt, DT=dt_k, Trap=trap, Q_bias=gamma[..., 0].contiguous(), K_bias=beta[..., 0].contiguous(), Angles=angles, D=self.D if self.d_skip and self.d_skip_sharing == 'head' else None, Z=z if not self.d_skip or self.d_skip_sharing == 'head' else None, chunk_size=64, return_final_states=True)
            y, _, final, _, _ = result
            if self.d_skip and self.d_skip_sharing != 'head':
                yt = y.float() + v.float() * self._skip_gains().reshape(H, G).float()
                if z is not None:
                    yt = yt * F.silu(z.float())
                y = yt.to(y.dtype)
            y = y.permute(0, 2, 3, 1)
        else:
            if x.dtype != torch.bfloat16:
                raise ValueError("upstream MIMO currently requires BF16 value inputs (use the mixer's BF16 autocast)")
            index = torch.arange(G * R, device=x.device) % R
            psi = (torch.arange(R, device=x.device)[:, None] == index[None]).float()
            psi = psi[None].expand(H, R, G * R).contiguous()
            order = torch.cat((torch.arange(0, N, 2, device=x.device), torch.arange(1, N, 2, device=x.device)))
            bk, ck = (bk.index_select(3, order), ck.index_select(3, order))
            beta, gamma = (beta[:H].index_select(1, order), gamma[:H].index_select(1, order))

            def maps_to_batch(t):
                return t.permute(0, 2, 1, 4, 3).reshape(B * R, L, R, 1, N).contiguous().to(x.dtype)

            def tokens_to_batch(t):
                return t.reshape(B, L, R, H, -1).permute(0, 2, 1, 3, 4).reshape(B * R, L, H, -1).contiguous()
            result = mimo(Q=maps_to_batch(ck), K=maps_to_batch(bk), V=x.reshape(B, L, H, G * R)[:, None].expand(B, R, L, H, G * R).reshape(B * R, L, H, G * R).contiguous(), ADT=adt.reshape(B * R, H, L), DT=dt_k.reshape(B * R, H, L), Trap=trap.reshape(B * R, H, L).to(x.dtype), Q_bias=gamma.transpose(1, 2).contiguous().float(), K_bias=beta.transpose(1, 2).contiguous().float(), MIMO_V=psi, MIMO_Z=None, MIMO_Out=None, Angles=tokens_to_batch(angles), D=None, Z=None, chunk_size=64 // R, rotary_dim_divisor=2, dtype=x.dtype, return_state=True)
            y, _, final, _, _ = result
            y = y.reshape(B, R, L, O, H, G, R).float().sum(-1).to(x.dtype)
            y = y.permute(0, 1, 4, 3, 5, 2).reshape(B, Hk, O, G, L).contiguous()
            final = final.reshape(B, Hk, G, R, N).sum(3)
            final = final.reshape(B, Hk, G, 2, N // 2).transpose(-1, -2).reshape(B, Hk, G, N)
            if gate_z is not None:
                z = gate_z.reshape(B, L, H, G, R).permute(0, 4, 2, 3, 1).reshape(B, Hk, G, L)
                y = (y.float() * F.silu(z.float())[:, :, None]).to(x.dtype)
        return self._finish_forward(x, u, y, final, parts.get('rmix'), gate_z=gate_z)

    def _resolve_chunk_size(self, kw: dict) -> tuple[int, str]:
        if getattr(self, 'precision', None) == 'bf16_ops' and (not torch.is_autocast_enabled('cuda')):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return self._resolve_chunk_size_impl(kw)
        return self._resolve_chunk_size_impl(kw)

    def _resolve_chunk_size_impl(self, kw: dict) -> tuple[int, str]:
        import os as _os
        env = _os.environ.get('SSD_MIMO_CHUNK')
        if env:
            return (int(env), 'env')
        cs = self.chunk_size
        if isinstance(cs, int):
            return (cs, 'explicit')
        nb = _needs_bwd(kw)
        if cs == 'auto':
            if self.R == 1 and self.O == 1 and (self.rot_mode is not None) and _gluon_siso_default(kw, with_route=False):
                r = _calibrated_rot_route(kw)
                if r is None or r[0] == 'siso':
                    return (64, 'auto-gluon90-siso')
                return (r[1], 'auto-route-' + r[0])
            if self.R == 1 and self.O == 1 and (self.rot_mode is not None) and (getattr(self, 'precision', 'tf32') != 'ieee'):
                r = _calibrated_rot_route(kw, fp32_only=True)
                if r is not None:
                    return (r[1], 'auto-route32-' + r[0])
            return (_scan_chunk(self.R, self.G_kernel, self.O, nb, real=self.rot_mode is None, **_call_size(kw)), 'auto')
        if cs == 'measure':
            if not self.allow_chunk_autotune:
                return (_scan_chunk(self.R, self.G_kernel, self.O, nb, real=self.rot_mode is None, **_call_size(kw)), 'auto-fallback')
            return (self._measure_chunk_size(kw), 'measure')
        raise ValueError(f"chunk_size must be an int, 'auto' or 'measure'; got {cs!r}")

    def _resolve_chunk_size_bwd(self, kw: dict, chunk_T: int, chunk_mode: str):
        ko = self.kernel_options.get('chunk_size_bwd', 'auto')
        if ko != 'auto':
            return int(ko)
        import os as _os
        env = _os.environ.get('SSD_MIMO_CHUNK_BWD', '').strip().lower()
        if env and env not in ('', 'auto', '0'):
            return int(env)
        if not _needs_bwd(kw):
            return None
        if chunk_mode == 'measure':
            tb = getattr(self, '_chunk_bwd_measured', None)
            return None if tb is None or tb == chunk_T else tb
        return _scan_chunk_bwd(self.R, self.G_kernel, self.O, chunk_T, real=self.rot_mode is None, **_call_size(kw))

    def _measure_chunk_size(self, kw: dict) -> int:
        from ..kernels.ssd_mimo import autograd as _ua
        from ..kernels.ssd_mimo.smem import chunk_admissible, chunk_cache_key, chunk_cache_lookup, chunk_cache_store
        x = kw['x_k']
        needs_bwd = _needs_bwd(kw)
        key = chunk_cache_key(self.R, self.O, self.N, x.dtype, self.precision, 'fwd_bwd' if needs_bwd else 'fwd', tuple(x.shape), device=x.device)
        cached = chunk_cache_lookup(key)
        if cached is not None:
            return cached
        leaf_names = ('x_k', 'delta_k', 'B_k', 'C_k', 'a_k', 'x0', 'delta_wk', 'theta_k', 'thh_k', 'gz_k', 'dsk_k', 'trap_lam_k', 'beta_k', 'gamma_k', 'rb_k')

        def _detached(name):
            t = kw.get(name)
            if not isinstance(t, torch.Tensor):
                return t
            return t.detach().requires_grad_(bool(needs_bwd and t.requires_grad))
        det = {name: _detached(name) for name in leaf_names}
        call_kwargs = dict(x0=det['x0'], precision=self.precision, bc_per_head=True, delta_w=det['delta_wk'], theta=det['theta_k'], theta_lo=kw.get('theta_lo_k'), theta_h=det['thh_k'], theta_increment=kw['theta_incr'], gate_z=det['gz_k'], d_skip=det['dsk_k'], trap_lam=det['trap_lam_k'], beta=det['beta_k'], gamma=det['gamma_k'], rot_bias=det['rb_k'], bias_pre_rot=kw['bias_pre_rot'], pair_layout=self.pair_layout, kernel_options=self.kernel_options)
        leaves = [det[n] for n in ('x_k', 'delta_k', 'B_k', 'C_k', 'a_k')]

        def trial(T, T_bwd=None):
            extra = {} if T_bwd in (None, T) else dict(chunk_size_bwd=T_bwd)

            def _run():
                if not needs_bwd:
                    with torch.no_grad():
                        _ua.ssd_mimo_scan(*leaves, chunk_size=T, chunk_mode='measure', **call_kwargs)
                    return
                for t in det.values():
                    if isinstance(t, torch.Tensor):
                        t.grad = None
                y, s = _ua.ssd_mimo_scan(*leaves, chunk_size=T, chunk_mode='measure', **extra, **call_kwargs)
                (y.float().sum() + s.float().sum()).backward()
            return _run

        def time_it(run):
            try:
                run()
                torch.cuda.synchronize()
            except Exception:
                torch.cuda.empty_cache()
                return None
            samples = []
            for _ in range(3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    run()
                except Exception:
                    return None
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            if not samples:
                return None
            samples.sort()
            return samples[len(samples) // 2]
        timings = {}
        for T in chunk_admissible():
            t = time_it(trial(T))
            if t is not None:
                timings[T] = t
        if not timings:
            self._chunk_bwd_measured = None
            return _scan_chunk(self.R, self.G_kernel, self.O, needs_bwd)
        best = min(timings, key=timings.get)
        best_bwd, pair_timings = (None, {})
        if needs_bwd and _chunk_pair_search_on():
            for Tb in chunk_admissible():
                if Tb == best:
                    continue
                t = time_it(trial(best, Tb))
                if t is not None:
                    pair_timings[Tb] = t
            if pair_timings:
                cand = min(pair_timings, key=pair_timings.get)
                if pair_timings[cand] < timings[best]:
                    best_bwd = cand
        self._chunk_bwd_measured = best_bwd
        chunk_cache_store(key, best, dict(diag=timings, pair=pair_timings, T_bwd=best_bwd))
        return best

    def _forward_cuda(self, x: torch.Tensor, u: torch.Tensor, h_init: torch.Tensor | None=None, gate_z: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None, out_norm=None, carry: dict | None=None) -> torch.Tensor:
        if self.m3_kernel == 'upstream':
            if h_init is not None or out_norm is not None or carry:
                raise NotImplementedError('upstream M3 batched route does not support h_init or fused out_norm; use the state/step API for continuation')
            return self._forward_upstream_m3(x, u, gate_z, tok_raw)
        from ..kernels.ssd_mimo import autograd as _ua
        from ..kernels.ssd_mimo.common import _ASCALAR_R_MAX
        if self.R > _ASCALAR_R_MAX:
            raise NotImplementedError(f"backend='cuda' supports R <= {_ASCALAR_R_MAX} (UNIFIED_SSD_SCALAR_RMAX).")
        rot_kernel_ok = self.rot_mode == 'm3' and self.R in (1, 2, 4, 8) and (self.O in (1, 2, 4, 8)) and (self.N >= 16) and (self.N % 16 == 0) and (self.G_kernel % 16 == 0) and (_os.environ.get('SSD_MIMO_ROT_INKERNEL', '1') == '1')
        rot_host_fold = self.rot_mode == 'm3' and (not rot_kernel_ok)
        trap_geom_ok = self.R in (1, 2, 4, 8) and self.O in (1, 2, 4, 8) and (self.N >= 16) and (self.N % 16 == 0) and (self.G_kernel % 16 == 0)
        trap_kernel_ok = rot_kernel_ok if self.rot_mode == 'm3' else self.rot_mode is None and trap_geom_ok
        in_kernel_trap = self.use_trap and trap_kernel_ok and (self.R == 1 and self.O == 1 or _os.environ.get('SSD_MIMO_TRAP_INKERNEL_RO', '1') == '1') and (_os.environ.get('SSD_MIMO_TRAP_INKERNEL', '1') == '1')
        in_kernel_phase = rot_kernel_ok
        kw = self._prepare_kernel_inputs(x, u, h_init, gate_z, tok_raw, in_kernel_trap=in_kernel_trap, in_kernel_phase=in_kernel_phase, rot_host_fold=rot_host_fold, carry=carry)
        G, Gp = (self.G, self.G_kernel)
        Hk = self.R * self.H
        Bsz, L = (x.shape[0], x.shape[1])
        if Gp != G:
            kw['x_k'] = F.pad(kw['x_k'], (0, 0, 0, 0, 0, Gp - G))
            if kw['x0'] is not None:
                kw['x0'] = F.pad(kw['x0'], (0, 0, 0, Gp - G))
            if kw['gz_k'] is not None:
                kw['gz_k'] = F.pad(kw['gz_k'].view(Bsz, L, Hk, G), (0, Gp - G)).reshape(Bsz, L, Hk * Gp)
            if kw['dsk_k'] is not None:
                kw['dsk_k'] = F.pad(kw['dsk_k'], (0, 0, 0, Gp - G))
        layout = _y_layout()
        want_token = layout == 'token' and self.O == 1 and (Gp == G) and (kw.get('m3_corr') is None) and (kw.get('endpoint') is None)
        if out_norm is not None and (not want_token):
            raise NotImplementedError('the fused output norm needs the token-major store and a readout with nothing after it (O == 1, no G pad, no host m3 correction); run with SSD_MIMO_OUT_NORM_FUSED=0')
        chunk_T, chunk_mode = self._resolve_chunk_size(kw)
        chunk_T_bwd = None
        if chunk_mode == 'auto' and _chunk_pair_on() and _needs_bwd(kw):
            chunk_T = _scan_chunk(self.R, self.G_kernel, self.O, False)
            chunk_T_bwd = _scan_chunk(self.R, self.G_kernel, self.O, True)
        else:
            chunk_T_bwd = self._resolve_chunk_size_bwd(kw, chunk_T, chunk_mode)
        if chunk_T_bwd == chunk_T:
            chunk_T_bwd = None
        from ..kernels.ssd_mimo import levers as _levers
        _nka = getattr(self, 'n_rope_angles', 0)
        _decl = _nka if _nka and 4 * _nka == self.N and (kw['theta_k'] is not None) and (self.pair_layout == 'interleaved') else None
        with _levers.declare_active_pairs(_decl):
            y, s_final = _ua.ssd_mimo_scan(kw['x_k'], kw['delta_k'], kw['B_k'], kw['C_k'], kw['a_k'], chunk_size=chunk_T, chunk_size_bwd=chunk_T_bwd, chunk_mode=chunk_mode, kernel_options=self.kernel_options, x0=kw['x0'], y_token_major=want_token, precision=self.precision, bc_per_head=True, delta_w=kw['delta_wk'], theta=kw['theta_k'], theta_h=kw['thh_k'], theta_lo=kw.get('theta_lo_k'), theta_increment=kw['theta_incr'], gate_z=kw['gz_k'], d_skip=kw['dsk_k'], trap_lam=kw['trap_lam_k'], beta=kw['beta_k'], gamma=kw['gamma_k'], rot_bias=kw['rb_k'], bias_pre_rot=kw['bias_pre_rot'], pair_layout=self.pair_layout, out_norm_w=None if out_norm is None else out_norm[0], out_norm_eps=1e-05 if out_norm is None else out_norm[1])
        if Gp != G:
            if self.O == 1:
                y = y.reshape(Bsz, Hk, Gp, L)[:, :, :G]
            else:
                y = y.reshape(Bsz, Hk, self.O, Gp, L)[:, :, :, :G]
            s_final = s_final.reshape(Bsz, Hk, Gp, -1)[:, :, :G]
        endpoint_kw = kw if self._can_fuse_endpoint(y, kw) else None
        if endpoint_kw is None:
            y = self._apply_m3_correction(y, kw)
        return self._finish_forward(x, u, y, s_final, kw.get('rmix_k'), gate_z=gate_z, y_contig=layout == 'contig', endpoint_kw=endpoint_kw)

    def _can_fuse_endpoint(self, y, kw):
        return kw.get('endpoint') is not None and self.R == self.O == 2 and y.is_cuda and (y.dtype in (torch.float32, torch.bfloat16, torch.float16)) and (_os.environ.get('SSD_ENDPOINT_MIX_FUSED', '1') == '1') and (_os.environ.get('SSD_MIMO_RMIX_EPILOGUE', '1') == '1')

    def _endpoint_operands(self, y, kw):
        matrix, coefficient = kw['endpoint']
        channels = y.shape[-2]
        xk = kw['x_k'][:, :, :channels]
        gate = kw.get('gz_k')
        if gate is not None:
            gate = gate.view(gate.shape[0], gate.shape[1], xk.shape[1], -1)
            gate = gate[..., :channels].reshape(gate.shape[0], gate.shape[1], -1).contiguous()
        return (xk, matrix, coefficient, gate)

    def _apply_m3_correction(self, y, kw):
        if kw.get('endpoint') is not None:
            from ..kernels.ssd_mimo.endpoint import endpoint_correct
            return endpoint_correct(y, *self._endpoint_operands(y, kw))
        corr = kw.get('m3_corr')
        if corr is None:
            return y
        corr = corr.reshape(y.shape)
        gz = kw.get('gz_k')
        if gz is not None:
            Bsz, L = (gz.shape[0], gz.shape[1])
            g = F.silu(_ref_f(gz)).view(Bsz, L, y.shape[1], -1)
            g = g[..., :y.shape[-2]].permute(0, 2, 3, 1)
            corr = corr * (g[:, :, None] if y.dim() == 5 else g)
        return y - corr.to(y.dtype)

    def _skip_gains(self):
        if self.d_skip_sharing == 'head':
            return self.D[:, None, None, None].expand(self.H, self.G, self.R, self.O)
        if self.d_skip_sharing == 'channel':
            return self.D[..., None].expand(self.H, self.G, self.R, self.O)
        return self.D

    def _skip_gains_kernel(self, dtype):
        return self._skip_gains().to(dtype).contiguous()

    def _dskip_route(self, ref, gate_z=None):
        mode = _os.environ.get('SSD_MIMO_DSKIP_EPILOGUE', 'auto').lower()
        if mode not in ('auto', 'on', 'off', 'host', '0', '1'):
            raise ValueError('SSD_MIMO_DSKIP_EPILOGUE must be auto/on/off/host/0/1')
        if not self.d_skip:
            return 'off'
        siso = self.R == self.O == 1
        if siso and self.m3_kernel == 'upstream':
            return 'in_scan'
        if mode == 'host':
            return 'host'
        fusable = gate_z is not None
        if siso and mode not in ('on', '1'):
            return 'in_scan' if fusable else 'host'
        if mode not in ('off', '0') and ref.is_cuda and (self.backend in ('cuda', 'auto')) and (ref.dtype in (torch.float32, torch.float16, torch.bfloat16)):
            return 'epilogue'
        return 'in_scan' if siso and fusable else 'host'

    def _dskip_in_epilogue(self, ref, gate_z=None):
        return self._dskip_route(ref, gate_z) == 'epilogue'

    def _input_skip(self, drive, gate_z, rmix):
        lead = drive.shape[:-1]
        values = _ref_f(drive).reshape(*lead, self.H, self.G, self.R)
        if gate_z is not None:
            values = values * F.silu(_ref_f(gate_z)).reshape_as(values)
        if self.r_mix_mode != 'none':
            weight = _ref_f(self.r_mix)
            if rmix is not None and self.r_mix_dyn:
                weight = weight + _ref_f(rmix).reshape(*lead, self.H, self.R)
            values = values * weight.unsqueeze(-2)
        return torch.einsum('...hgr,hgro->...hgo', values, _ref_f(self._skip_gains())).reshape(*lead, self.d_out)

    def _h_final_dtype(self, x: torch.Tensor) -> torch.dtype:
        dt = getattr(self, '_carry_dtype', None)
        return x.dtype if dt is None else dt

    def _finish_forward(self, x, u, y, s_final, rmix=None, *, gate_z=None, y_contig=False, endpoint_kw=None):
        if rmix is None and self.r_mix_dyn:
            rmix = self.r_mix_proj(u)
        route = self._dskip_route(x, gate_z)
        if route == 'epilogue':
            B, L = x.shape[:2]
            H, G, R, O = (self.H, self.G, self.R, self.O)
            c = None if self.r_mix_mode == 'none' else self.r_mix
            rm = rmix.reshape(B, L, H, R).to(y.dtype) if self.r_mix_dyn and c is not None else None
            d = self._skip_gains_kernel(y.dtype)
            if endpoint_kw is not None:
                from ..kernels.ssd_mimo.endpoint_mix import endpoint_mix
                out = endpoint_mix(y, *self._endpoint_operands(y, endpoint_kw), c, rm, H, R, O, G, x.dtype, d=d)
            else:
                from ..kernels.ssd_mimo.rmix_epilogue import rmix_reduce
                drive = x.reshape(B, L, H, G, R).contiguous()
                gate = None if gate_z is None else gate_z.reshape(B, L, H, G, R).contiguous()
                out = rmix_reduce(y, c, rm, H, R, O, G, x.dtype, d=d, drive=drive, gate=gate)
            self.h_final = s_final.reshape(B, R, H, G, self.N).permute(0, 2, 3, 1, 4).reshape(B, H, self.P, self.N).to(self._h_final_dtype(x)).clone()
            return out
        out = self._reduce_siblings(x, u, y, s_final, rmix, y_contig=y_contig, endpoint_kw=endpoint_kw)
        if route == 'host':
            out = out + self._input_skip(x, gate_z, rmix).to(out.dtype)
        return out

    def _reduce_siblings(self, x, u, y, s_final, rmix=None, *, y_contig: bool=False, endpoint_kw=None) -> torch.Tensor:
        if rmix is None and self.r_mix_dyn:
            rmix = self.r_mix_proj(u)
        B, L, _ = x.shape
        H, P, G, R, N = (self.H, self.P, self.G, self.R, self.N)
        if endpoint_kw is not None:
            from ..kernels.ssd_mimo.endpoint_mix import endpoint_mix
            c_st = None if self.r_mix_mode == 'none' else self.r_mix
            rm = rmix.view(B, L, H, R).to(y.dtype) if self.r_mix_dyn and self.r_mix_mode != 'none' else None
            out = endpoint_mix(y, *self._endpoint_operands(y, endpoint_kw), c_st, rm, H, R, self.O, G, x.dtype)
            self.h_final = s_final.reshape(B, R, H, G, N).permute(0, 2, 3, 1, 4).reshape(B, H, P, N).to(self._h_final_dtype(x)).clone()
            return out
        if self.O == 1 and _y_is_token_major(y):
            yt = y.permute(0, 3, 1, 2)
            yv = yt.reshape(B, L, R, H, G) if R > 1 else yt
            if self.r_mix_mode == 'none':
                y_out = yv.sum(dim=2) if R > 1 else yv
            elif R == 1:
                c = self.r_mix.to(yv.dtype).reshape(H)
                if self.r_mix_dyn:
                    c = c[None, None, :] + rmix.view(B, L, H).to(yv.dtype)
                    y_out = yv * c[..., None]
                else:
                    y_out = yv * c[None, None, :, None]
            else:
                ck = self.r_mix.to(yv.dtype).permute(1, 0)[None, None, :, :, None]
                if self.r_mix_dyn:
                    ck = ck + rmix.view(B, L, H, R).permute(0, 1, 3, 2).to(yv.dtype)[..., None]
                y_out = (yv * ck).sum(dim=2)
            y_out = y_out.reshape(B, L, H * G).to(x.dtype)
            self.h_final = s_final.reshape(B, R, H, G, N).permute(0, 2, 3, 1, 4).reshape(B, H, P, N).to(self._h_final_dtype(x)).clone() if R > 1 else s_final.reshape(B, H, G, N).to(self._h_final_dtype(x)).clone()
            return y_out
        if R == 1 and self.O == 1:
            yk = y.reshape(B, H, G, L)
            if self.r_mix_mode != 'none':
                c = self.r_mix.to(yk.dtype).reshape(H)
                if self.r_mix_dyn:
                    cd = c[None, :, None] + rmix.view(B, L, H).to(yk.dtype).permute(0, 2, 1)
                    yk = yk * cd[:, :, None, :]
                else:
                    yk = yk * c[None, :, None, None]
            y_out = yk.permute(0, 3, 1, 2).reshape(B, L, H * G).to(x.dtype)
            if y_contig:
                y_out = y_out.contiguous()
            self.h_final = s_final.reshape(B, H, G, N).to(self._h_final_dtype(x)).clone()
            return y_out
        if _os.environ.get('SSD_MIMO_RMIX_EPILOGUE', '1') == '1' and self.backend != 'torch' and y.is_cuda and (y.dtype in (torch.float32, torch.bfloat16, torch.float16)):
            from ..kernels.ssd_mimo.rmix_epilogue import rmix_reduce
            c_st = None if self.r_mix_mode == 'none' else self.r_mix
            rm = rmix.view(B, L, H, R).to(y.dtype) if self.r_mix_dyn and self.r_mix_mode != 'none' else None
            y_out = rmix_reduce(y, c_st, rm, H, R, self.O, G, x.dtype)
            self.h_final = s_final.reshape(B, R, H, G, N).permute(0, 2, 3, 1, 4).reshape(B, H, P, N).to(self._h_final_dtype(x)).clone()
            return y_out
        O = self.O
        off = self.r_mix_mode == 'none'
        c = None if off else self.r_mix.to(y.dtype).permute(1, 0)
        if self.O == 1:
            yk = y.reshape(B, R, H, G, L)
            if off:
                y_out = yk.sum(dim=1)
            else:
                ck = c[None, :, :, None, None]
                if self.r_mix_dyn:
                    ck = ck + rmix.view(B, L, H, R).permute(0, 3, 2, 1).to(y.dtype)[:, :, :, None, :]
                y_out = (yk * ck).sum(dim=1)
            y_out = y_out.permute(0, 3, 1, 2)
        else:
            yk = y.reshape(B, R, H, O, G, L)
            if off:
                y_out = yk.sum(dim=1)
            else:
                ck = c[None, :, :, None, None, None]
                if self.r_mix_dyn:
                    ck = ck + rmix.view(B, L, H, R).permute(0, 3, 2, 1).to(y.dtype)[:, :, :, None, None, :]
                y_out = (yk * ck).sum(dim=1)
            y_out = y_out.permute(0, 4, 1, 3, 2)
        y_out = y_out.reshape(B, L, H * G * self.O).to(x.dtype)
        if y_contig:
            y_out = y_out.contiguous()
        self.h_final = s_final.reshape(B, R, H, G, N).permute(0, 2, 3, 1, 4).reshape(B, H, P, N).to(self._h_final_dtype(x)).clone()
        return y_out

    def combined_refusal(self, x, *, gate_z=None, h_init=None, extra_carry=None, return_state=False):
        from .. import tails as _tails
        from ..layers.conv import _causal_conv1d_fn
        if self.R != 1 or self.O != 1:
            return f'combined is wired at R = O = 1 (R={self.R}, O={self.O})'
        backend = self.backend
        if backend == 'auto':
            backend = 'cuda' if x.is_cuda else 'torch'
        if backend != 'cuda':
            return f'backend={backend!r} (the combined node is the CUDA route)'
        if not x.is_cuda or x.dtype == torch.float64:
            return f'CUDA, non-fp64 only (dtype={x.dtype}, cuda={x.is_cuda})'
        if gate_z is None:
            return 'no gate: the fused epilogue owns the combined output store'
        if self.d_skip and self._dskip_route(x, gate_z) != 'in_scan':
            return 'D skip outside the scan requires its ordinary autograd path'
        if h_init is not None or extra_carry or return_state:
            return 'a state carry (h_init / continuation / return_state)'
        if self.G_kernel != self.G:
            return f'G={self.G} is zero-padded to {self.G_kernel}'
        if self.r_mix_mode != 'none':
            return f'r_mix={self.r_mix_mode!r} (the sibling mix)'
        if self.rot_mode is not None:
            return f'rot_mode={self.rot_mode!r}: its angle map is host-side differentiable work the combined adjoint does not cover'
        if self.use_trap:
            return 'the trapezoid'
        if _y_layout() != 'token':
            return f'SSD_MIMO_Y_LAYOUT={_y_layout()!r} (token-major only)'
        if not (self._fuse_proj and _tails.tails_enabled() and _tail_law_is_stock()):
            return 'the eager token-map tail (SSD_MIMO_TAILS=0 or a dial law)'
        if self.bc_conv1d is not None and _causal_conv1d_fn is None:
            return 'the causal_conv1d package is not importable'
        if self.N & self.N - 1 or not 16 <= self.N <= 128:
            return f'N={self.N} takes the tiled / padded scan repair'
        if x.dim() == 3 and x.shape[1] % _COMBINED_CHUNK:
            return f'L={x.shape[1]} is not a multiple of the scan chunk ({_COMBINED_CHUNK}), so the scan pads it'
        return None

    def out_norm_refusal(self, x, *, gate_z=None, state=None, return_state=False):
        if self.m3_kernel == 'upstream':
            return 'upstream M3: output norm remains in the enclosing mixer'
        from .. import tails as _tails
        if _os.environ.get('SSD_MIMO_OUT_NORM_FUSED', '1') != '1':
            return 'SSD_MIMO_OUT_NORM_FUSED=0'
        backend = self.backend
        if backend == 'auto':
            backend = 'cuda' if x.is_cuda else 'torch'
        if backend != 'cuda' or not x.is_cuda or x.dtype == torch.float64:
            return f'backend={backend!r} / dtype={x.dtype}: the fused output norm is the CUDA epilogue kernel'
        if gate_z is None:
            return 'no gate: the fused norm rides the gate epilogue kernel'
        if self.d_skip and self._dskip_route(x, gate_z) != 'in_scan':
            return 'D skip outside the scan requires its ordinary autograd path'
        if state is not None or return_state:
            return 'a state carry (the step / continuation route)'
        if self.R != 1 or self.O != 1:
            return f'R={self.R}, O={self.O}: the fused norm is wired at R = O = 1 (at O > 1 the gate broadcasts over o, so dZ needs a segmented reduction inside the token row)'
        if self.r_mix_mode != 'none':
            return f'r_mix={self.r_mix_mode!r}: the sibling mix runs AFTER the scan, so the norm would see the wrong row'
        if self.G_kernel != self.G:
            return f'G={self.G} is zero-padded to {self.G_kernel}'
        if _y_layout() != 'token':
            return f'SSD_MIMO_Y_LAYOUT={_y_layout()!r} (token-major only)'
        if self.use_trap and self.rot_mode != 'm3':
            return "the HOST two-tap trapezoid correction runs after the scan's readout"
        if self.rot_mode == 'm3' and self.use_trap and (not (_os.environ.get('SSD_MIMO_TRAP_INKERNEL', '1') == '1' and (not self.use_theta_h))):
            return 'the host trapezoid correction runs after the readout'
        if _ua_gate_epilogue_off():
            return 'SSD_MIMO_GATE_INKERNEL=1 / SSD_MIMO_GATE_FUSED=0: the fused norm rides the DEDICATED epilogue kernel'
        if self.N & self.N - 1 or not 16 <= self.N <= 128:
            return f'N={self.N} takes the tiled / padded scan repair, and a per-N-tile norm is not the row norm'
        return None

    def _combined_core_fwd(self, x, u, gate_z, tok_raw, out_norm=None):
        from .. import tails as _tails
        from ..kernels.ssd_mimo import autograd as _ua
        own_slab = tok_raw is None
        with torch.no_grad():
            cm = False
            slab = tok_raw
            if own_slab:
                cm = self._tail_owns_all and u.dim() == 3 and (u.stride(-1) != 1) and self._tail_will_fuse(u)
                slab = self.proj_slab(u, channel_major=cm)
            self._bc_capture = bcap = []
            try:
                with _tails.capture_tails() as tslot, _ua.capture_scan() as sslot:
                    y_out = self._forward_cuda(x, u, None, gate_z, slab, out_norm=out_norm)
            finally:
                self._bc_capture = None
        if len(sslot) != 1 or len(tslot) != 1:
            raise RuntimeError(f'combined SSD-MIMO node: expected exactly one captured scan and one captured tail, got {len(sslot)} / {len(tslot)} -- the call took a repaired (N-tiled / chunk-halved) route that `combined_refusal` should have excluded')
        tape = dict(scan=sslot[0], tail=tslot[0], slab=slab, own_slab=own_slab, slab_cm=cm, u=u, bc_in=bcap[0] if bcap else None)
        return (y_out, tape)

    def _combined_core_bwd(self, tape, dy_out, dz_out=None, dslab_out=None):
        scan, tail = (tape['scan'], tape['tail'])
        slab, u = (tape['slab'], tape['u'])
        H, G, N, R, O = (self.H, self.G, self.N, self.R, self.O)
        Bsz, L = (slab.shape[0], slab.shape[1])
        T = scan.meta['T']
        nc = L // T
        if scan.meta['L_padded'] != L or nc * T != L:
            raise AssertionError(f"combined SSD-MIMO node: L={L} is not nc*T={nc}*{T} (the scan padded it to {scan.meta['L_padded']}); `combined_refusal` should have excluded this call")
        grads = {}
        if not dy_out.is_contiguous():
            dy_out = dy_out.contiguous()
        dY = dy_out.view(Bsz, nc, T, H, 1, G).permute(0, 3, 1, 2, 4, 5)
        scan.ctx.dz_out = dz_out
        g = scan.backward(dY, None)
        del dY, dy_out
        scan.ctx.saved_tensors = ()
        tape['scan'] = None
        dx_branch = g['x_in'].reshape(Bsz, H, G, L).permute(0, 3, 1, 2).reshape(Bsz, L, H * G)
        d_bt = g['B_tok'].reshape(Bsz, L, N, R)
        d_ct = g['C_tok'].reshape(Bsz, L, N, O)
        if self.a_dyn:
            d_dec_k, d_dt_k = (g['delta'], g['delta_w'])
        else:
            d_dec_k, d_dt_k = (None, g['delta'])
        if self.A_log is not None and g['a_log'] is not None:
            d_a = g['a_log'].view(R, H).permute(1, 0).to(self.A_log.dtype)
            grads['A_log'] = d_a
        if self.head_bias:
            if g['beta'] is not None:
                grads['B_bias'] = g['beta'].view(R, H, N, R).sum(0).to(self.B_bias.dtype)
            if g['gamma'] is not None:
                grads['C_bias'] = g['gamma'].view(R, H, N, O).sum(0).to(self.C_bias.dtype)
        if self.d_skip and g['d_skip'] is not None:
            grads['D'] = (g['d_skip'].sum(dim=(1, 2)) if self.d_skip_sharing == 'head' else g['d_skip']).reshape_as(self.D).to(self.D.dtype)
        if self.use_theta_h and g.get('theta_h') is not None:
            grads['theta_h'] = g['theta_h'].view(R, H, N // 2).sum(0)[:, :self.theta_h.shape[-1]].to(self.theta_h.dtype)
        if g.get('out_norm_w') is not None:
            grads['@out_norm_w'] = g['out_norm_w']
        shared_bias = bool(self.bc_bias) and (not self.head_bias)
        need_b = bool(self.b_norm or shared_bias or (self.use_g_b and R == 1))
        need_c = bool(self.c_norm or shared_bias or self.use_g_c)
        td = tail.backward(d_bt.reshape(Bsz, L, N, R) if need_b else None, d_ct.reshape(Bsz, L, N, O) if need_c else None, d_dt_k, d_dec_k)
        d_b_src = td['b_src'] if need_b else d_bt.reshape(Bsz, L, N * R)
        d_c_src = td['c_src'] if need_c else d_ct.reshape(Bsz, L, N * O)
        g['B_tok'] = g['C_tok'] = g['delta'] = g['delta_w'] = None
        d_dt_k = d_dec_k = None
        tail.ctx.saved_tensors = ()
        tape['tail'] = None
        if self.b_norm and td['w_b'] is not None:
            grads['B_norm.weight'] = td['w_b']
        if self.c_norm and td['w_c'] is not None:
            grads['C_norm.weight'] = td['w_c']
        if shared_bias:
            if td['bias_b'] is not None:
                grads['B_bias'] = td['bias_b']
            if td['bias_c'] is not None:
                grads['C_bias'] = td['bias_c']
        d_parts = {'dt': td['dt_src']}
        if self.a_dyn:
            d_parts['a'] = td['a_src']
        if self.use_g_b:
            d_parts['gB'] = td['gb_src']
        if self.use_g_c:
            d_parts['gC'] = td['gc_src']
        if self.bc_conv1d is None:
            d_parts['B'], d_parts['C'] = (d_b_src, d_c_src)
        else:
            from ..layers.conv import causal_conv1d_cl_bwd
            d_bc = torch.cat([d_b_src.reshape(Bsz, L, N * R), d_c_src.reshape(Bsz, L, N * O)], dim=-1)
            conv = self.bc_conv1d.conv
            bc_in = tape['bc_in']
            dx_cl = None if bc_in.stride(2) == bc_in.shape[1] else torch.empty(bc_in.shape[0], bc_in.shape[2], bc_in.shape[1], device=bc_in.device, dtype=bc_in.dtype).transpose(1, 2)
            d_in, d_w, d_b = causal_conv1d_cl_bwd(bc_in, conv.weight, conv.bias, d_bc.transpose(1, 2), 'silu', dx=dx_cl)
            d_in = d_in.transpose(1, 2)
            grads['bc_conv1d.conv.weight'] = d_w
            grads['bc_conv1d.conv.bias'] = d_b
            d_parts['B'], d_parts['C'] = d_in.split([N * R, N * O], dim=-1)
            if self._bc_adjacent:
                d_parts['_bc'] = d_in
            del d_bc
            tape['bc_in'] = None
        d_dtb = td.get('bias_dt')
        del d_b_src, d_c_src, d_bt, d_ct, td
        for key, attr, width in zip(self._proj_keys, self._proj_attrs, self._proj_widths):
            mod = getattr(self, attr)
            gp = d_parts.get(key)
            if mod.bias is not None and gp is not None:
                if attr == 'dt_proj' and d_dtb is not None:
                    grads[f'{attr}.bias'] = d_dtb.to(mod.bias.dtype)
                    continue
                grads[f'{attr}.bias'] = gp.reshape(-1, width).sum(0).to(mod.bias.dtype)
        d_slab = torch.zeros(Bsz, L, self.tok_width, device=slab.device, dtype=slab.dtype) if dslab_out is None else dslab_out
        off = 0
        joint = d_parts.pop('_bc', None)
        if joint is not None:
            d_slab[..., :self._bc_width].copy_(joint.reshape(Bsz, L, self._bc_width))
            off = self._bc_width
        for key, width in zip(self._proj_keys, self._proj_widths):
            if joint is not None and key in ('B', 'C'):
                continue
            gp = d_parts.get(key)
            if gp is not None:
                d_slab[..., off:off + width].copy_(gp.reshape(Bsz, L, width))
            elif dslab_out is not None:
                d_slab[..., off:off + width].zero_()
            off += width
        du = None
        if tape['own_slab']:
            du, dw_tok = self._proj_slab_bwd(u, d_slab, tape['slab_cm'])
            grads.update(self.scatter_tok_grads(dw_tok))
            d_slab = None
        return (dx_branch, du, d_slab, grads)

    def scatter_tok_grads(self, dw_tok) -> dict:
        grads: dict = {}
        off = 0
        for attr, width in zip(self._proj_attrs, self._proj_widths):
            mod = getattr(self, attr)
            grads[f'{attr}.weight'] = self._proj_w_grad(attr, dw_tok[off:off + width]).to(mod.weight.dtype)
            off += width
        return grads

    def _proj_slab_bwd(self, u, d_out, channel_major):
        w, _ = self.tok_proj_weight()
        Bsz = u.shape[0]
        if channel_major:
            d_outc = d_out.transpose(1, 2)
            dw = torch.bmm(d_outc, u).sum(0)
            du = torch.bmm(w.to(d_outc.dtype).t().expand(Bsz, w.shape[1], w.shape[0]), d_outc).transpose(1, 2)
        else:
            f = d_out.reshape(-1, d_out.shape[-1])
            dw = f.t() @ u.reshape(-1, u.shape[-1])
            du = (f @ w.to(f.dtype)).view(*u.shape)
        return (du, dw)

    def step(self, x_t: torch.Tensor, state: State | dict | None=None, gate_z: torch.Tensor | None=None, sel_x: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None) -> tuple[torch.Tensor, State]:
        if _torch_impl() == 'legacy':
            return self._step_legacy(x_t, state, gate_z=gate_z, sel_x=sel_x, tok_raw=tok_raw)
        from .ssd_mimo_torch import ssd_step
        return ssd_step(self, x_t, state, gate_z=gate_z, sel_x=sel_x, tok_raw=tok_raw)

    def _step_legacy(self, x_t: torch.Tensor, state: State | dict | None=None, gate_z: torch.Tensor | None=None, sel_x: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None) -> tuple[torch.Tensor, State]:
        output_gate = gate_z if self.R > 1 or self.O > 1 else None
        if gate_z is not None and gate_z.shape != (*x_t.shape[:-1], self.d_out):
            raise ValueError(f'gate_z must have shape {(*x_t.shape[:-1], self.d_out)}')
        if output_gate is not None:
            gate_z = None
        Bsz = x_t.shape[0]
        H, P, G, R, N, O = (self.H, self.P, self.G, self.R, self.N, self.O)
        state = self.resolve_state(state, batch_size=Bsz, device=x_t.device, dtype=x_t.dtype)
        cache = state
        h_old = cache['hidden_state']
        u_t = self._select(x_t, sel_x)
        u_seq = u_t.unsqueeze(1)
        parts = self._proj_split(u_seq) if tok_raw is None else self.split_tok_raw(tok_raw.unsqueeze(1))
        if self.bc_conv1d is not None:
            Bt, Ct, gB, gC, dt, dt_dec, bc_state = self._token_maps(u_seq, bc_conv_state=cache['bc_conv_state'], parts=parts)
            cache = {**cache, 'bc_conv_state': bc_state}
        else:
            Bt, Ct, gB, gC, dt, dt_dec = self._token_maps(u_seq, parts=parts)
        Bt = Bt.squeeze(1)
        Ct = Ct.squeeze(1)
        if gB is not None:
            gB = gB.squeeze(1)
        if gC is not None:
            gC = gC.squeeze(1)
        dt = dt.squeeze(1)
        if dt_dec is not None:
            dt_dec = dt_dec.squeeze(1)
        lam_trap = None
        if self.use_trap:
            if 'dt_prev' in cache:
                raise ValueError('legacy SSD dt_prev cache used the incorrect clock-blend law; restart from the sequence prefix under endpoint semantics')
            lam_trap = torch.sigmoid(parts['trap'].squeeze(1))[:, :, None]
            dt_w = torch.ones_like(dt)
        else:
            dt_w = dt
        if self.A_log is None:
            rates = torch.ones(H, P, device=dt.device, dtype=dt.dtype)
        else:
            rates = self.A_log.exp()
            rates = rates[:, None, :].expand(H, G, R).reshape(H, P)
        d_dec = dt if dt_dec is None else dt_dec
        a_t = torch.exp(-d_dec.unsqueeze(-1) * rates.unsqueeze(-1))
        theta_t = None
        theta_hr = None
        if self.rot_mode == 'm3':
            if self.rot_mode == 'm3':
                dw_hr = _ref_f(dt.view(Bsz, H, G, R)[:, :, 0, :])
                ka = self.n_rope_angles
                raw = _ref_f(parts['theta'].squeeze(1))
                if self.omega_per_head:
                    raw = raw.view(Bsz, H, 1, 1).expand(Bsz, H, 1, ka)
                elif self.theta_per_head:
                    raw = raw.view(Bsz, H, 1, ka)
                else:
                    raw = raw.view(Bsz, 1, 1, ka).expand(Bsz, H, 1, ka)
                ang = torch.tanh(raw)
                if self.learn_theta_freq and self.omega_layout != 'pair':
                    omega = (math.pi * torch.tanh(_ref_f(self.theta_freq))).view(H, 1, -1)
                elif self.learn_theta_freq:
                    omega = math.pi * torch.tanh(_ref_f(self.theta_freq))
                else:
                    omega = math.pi
                incr = -dw_hr[..., None] * ang * omega
            rot_ang = _wrap2pi_t(cache['rot_ang'] + incr)
            cache = {**cache, 'rot_ang': rot_ang}
            theta_hr = F.pad(rot_ang, (0, N // 2 - ka)) if ka < N // 2 else rot_ang
        xi = x_t.view(Bsz, H, G, R)
        if self.bias_pre_rot and R == 1:
            Bt_eff = (Bt if gB is None else gB * Bt)[:, None]
            Bt_eff = Bt_eff + self.B_bias[:, 0].to(Bt.dtype)[None]
            w = torch.einsum('bhnr,bhgr->bhgn', Bt_eff, xi)
            if lam_trap is not None:
                drive = w[:, :, :, None, :].expand(Bsz, H, G, R, N)
            else:
                drive = dt_w.view(Bsz, H, G, R)[..., None] * w[:, :, :, None, :]
        elif self.bias_pre_rot:
            Bb = self.B_bias.to(Bt.dtype)
            if gB is None:
                Beff = Bt[:, None, None] + Bb[None]
                w = torch.einsum('bhrnq,bhgq->bhgrn', Beff, xi)
            else:
                gB_r = gB.permute(0, 2, 1)
                Beff = gB_r[:, None, :, :, None] * Bt[:, None, None] + Bb[None]
                w = torch.einsum('bhrnq,bhgq->bhgrn', Beff, xi)
            if lam_trap is not None:
                drive = w
            else:
                drive = dt_w.view(Bsz, H, G, R)[..., None] * w
        else:
            w = torch.einsum('bnr,bhgr->bhgn', Bt, xi)
            drive = dt_w.view(Bsz, H, G, R)[..., None] * w[:, :, :, None, :]
            if gB is not None:
                drive = drive * gB.permute(0, 2, 1).view(Bsz, 1, 1, R, N)

        def rotate_pairs(v, cs, sn, axis=-1):
            axis %= v.ndim
            if self.pair_layout == 'half':
                e, o = v.chunk(2, dim=axis)
                return torch.cat((cs * e + sn * o, -sn * e + cs * o), dim=axis)
            even = [slice(None)] * v.ndim
            odd = even.copy()
            even[axis], odd[axis] = (slice(0, None, 2), slice(1, None, 2))
            e, o = (v[tuple(even)], v[tuple(odd)])
            return torch.stack((cs * e + sn * o, -sn * e + cs * o), dim=axis + 1).flatten(axis, axis + 1)
        if theta_hr is not None:
            ang_b = theta_hr
            cs_b = torch.cos(ang_b)[:, :, None, :, :]
            sn_b = torch.sin(ang_b)[:, :, None, :, :]
            drive = rotate_pairs(drive, cs_b, sn_b)
        drive = drive.reshape(Bsz, H, G, R, N)
        drive = drive.reshape(Bsz, H, P, N)
        if self.head_bias and (not self.bias_pre_rot):
            bd = torch.einsum('hrnq,bhgq->bhgrn', self.B_bias.to(xi.dtype), xi)
            bd = dt_w.view(Bsz, H, G, R)[..., None] * bd
            drive = drive + bd.reshape(Bsz, H, P, N)
        if lam_trap is not None:
            lam_p = lam_trap.reshape(Bsz, H, 1, 1)
            dtp = dt.reshape(Bsz, H, P, 1)
            key = 'write_prev'
            w_prev = cache.get(key)
            drive_now = lam_p * dtp * drive
            if w_prev is not None:
                drive_now = drive_now + a_t * (1.0 - lam_p) * dtp * w_prev
            cache = {**cache, key: drive}
            drive = drive_now
        h_new = a_t * h_old + drive
        gated = Ct if gC is None else gC * Ct
        h_view = h_new.view(Bsz, H, G, R, N)
        if self.bias_pre_rot and R == 1 and (O == 1):
            K = N // 2
            ce = gated[:, None] + self.C_bias[:, 0].to(gated.dtype)[None]
            ce = ce[..., 0]
            ang_c = theta_hr[:, :, 0, :]
            if self.use_theta_h:
                ang_c = ang_c + self._theta_h_pairs()[None]
            cs = torch.cos(ang_c)
            sn = torch.sin(ang_c)
            read = rotate_pairs(ce, cs, sn)
            yr = (h_view[:, :, :, 0, :] * read[:, :, None, :]).sum(-1)[:, :, :, None, None]
        elif self.bias_pre_rot:
            K = N // 2
            ce = gated[:, None, None] + self.C_bias.to(gated.dtype)[None]
            ang_c = theta_hr
            if self.use_theta_h:
                ang_c = ang_c + self._theta_h_pairs()[None, :, None, :]
            cs = torch.cos(ang_c)[..., None]
            sn = torch.sin(ang_c)[..., None]
            read = rotate_pairs(ce, cs, sn, axis=-2)
            yr = torch.einsum('bhgrn,bhrno->bhgro', h_view, read)
        elif self.use_theta_h or theta_hr is not None:
            K = N // 2
            ang_c = None
            if self.use_theta_h:
                ang_c = self._theta_h_pairs()[None, :, None, :]
            if theta_hr is not None:
                ang_c = theta_hr if ang_c is None else ang_c + theta_hr
            cs = torch.cos(ang_c)[..., None]
            sn = torch.sin(ang_c)[..., None]
            read = rotate_pairs(gated[:, None, None], cs, sn, axis=-2)
            read = read.expand(Bsz, H, R, N, O)
            yr = torch.einsum('bhgrn,bhrno->bhgro', h_view, read)
        else:
            read = gated.unsqueeze(1).expand(Bsz, R, N, O)
            yr = torch.einsum('bhgrn,brno->bhgro', h_view, read)
        if self.head_bias and (not self.bias_pre_rot):
            c_bias = self.C_bias.to(h_new.dtype)
            y_gam = torch.einsum('hrno,bhgrn->bhgro', c_bias, h_view)
            yr = yr + y_gam
        if self.d_skip:
            yr = yr + x_t.reshape(Bsz, H, G, R).to(yr.dtype)[..., None] * self._skip_gains().to(yr.dtype)[None]
        if gate_z is not None:
            gz = F.silu(gate_z).view(Bsz, H, G, R)[..., None]
            yr = yr * gz
        if self.r_mix_mode == 'none':
            y = yr.sum(dim=3)
        else:
            c = self.r_mix.to(yr.dtype)[None, :, None, :]
            if self.r_mix_dyn:
                c = c + parts['rmix'].view(Bsz, H, 1, R).to(yr.dtype)
            y = (yr * c[..., None]).sum(dim=3)
        self.h_final = h_new
        new_state = State(hidden_state=h_new)
        if self.rot_mode == 'm3':
            new_state['rot_ang'] = cache['rot_ang']
        if self.use_trap:
            key = 'write_prev'
            new_state[key] = cache[key]
        if self.bc_conv1d is not None:
            new_state['bc_conv_state'] = cache['bc_conv_state']
        self._final_state = new_state
        y = y.reshape(Bsz, H * G * O)
        if output_gate is not None:
            y = y * F.silu(output_gate)
        return (y, new_state)

    def theta_omega(self) -> torch.Tensor:
        if getattr(self, 'learn_theta_freq', False):
            return math.pi * torch.tanh(self.theta_freq)
        if getattr(self, 'omega_layout', 'pair') == 'head':
            return torch.full((self.H,), math.pi, device=next(self.parameters()).device)
        return torch.full((getattr(self, 'n_rope_angles', self.N // 2),), math.pi, device=next(self.parameters()).device)

    def _theta_h_pairs(self) -> torch.Tensor | None:
        if not self.use_theta_h:
            return None
        th = _ref_f(self.theta_h)
        pad = self.N // 2 - th.shape[-1]
        return F.pad(th, (0, pad)) if pad else th

    def _default_state_dtype(self) -> torch.dtype:
        return torch.promote_types(next(self.parameters()).dtype, torch.float32)

    def _init_state(self, batch_size: int, device: torch.device | None=None, dtype: torch.dtype | None=None) -> State:
        dev = device or next(self.parameters()).device
        if dtype is None:
            dtype = self._default_state_dtype()
        state = State(hidden_state=torch.zeros(batch_size, self.H, self.P, self.N, device=dev, dtype=dtype))
        if self.rot_mode == 'm3':
            ang_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            state['rot_ang'] = torch.zeros(batch_size, self.H, self.R, self.n_rope_angles, device=dev, dtype=ang_dtype)
        if self.use_trap:
            state['write_prev'] = torch.zeros_like(state['hidden_state'])
        if self.bc_conv1d is not None:
            state['bc_conv_state'] = self.bc_conv1d.init_state(batch_size, device=dev, dtype=dtype)
        return state

    def hidden_state_shape(self) -> tuple[int, ...]:
        return (self.H, self.P, self.N)
from dataclasses import dataclass
from typing import Literal
from ..config import SSMCoreConfig, SpecFields, refuse_retired_kwargs, _refuse

@refuse_retired_kwargs
@dataclass
class SSDMimoConfig(SpecFields, SSMCoreConfig):
    R: int = 2
    pair_layout: str = 'interleaved'
    a_dyn_per_sibling: bool = False
    a_act: Literal['heavy_tail', 'softplus'] = 'heavy_tail'
    allow_chunk_autotune: bool = True
    r_mix: str = 'auto'
    r_mix_dyn: bool | None = None
    bc_bias_init: str | None = None
    d_skip: bool = True
    d_skip_sharing: str = 'input_output'
    a_proj_bias: bool = True
    trap_proj_bias: bool = True
    dt_init_floor: float = 0.0001
    m3_kernel: str = 'native'
    compile_token_maps: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._geometry()
        _config_spec.resolve('ssd_mimo', d_state=self.d_state, R=self.R, O=self.O, **self.spec_kwargs())

    def _geometry(self):
        return resolve_ssd_geometry(self.d_inner, R=self.R, ngroups=self.ngroups, n_heads=self.n_heads, heads=self.heads, H=self.H, head_subsystems=self.head_subsystems, P=self.P)

    def to_ssm_kwargs(self) -> dict:
        heads, _, groups = self._geometry()
        return dict(d_inner=self.d_inner, d_state=self.d_state, d_model=self.d_model, d_conv=self.d_conv, R=self.R, O=self.O, ngroups=groups, heads=heads, compile_token_maps=self.compile_token_maps, **self.spec_kwargs(), pair_layout=self.pair_layout, a_dyn_per_sibling=self.a_dyn_per_sibling, a_act=self.a_act, allow_chunk_autotune=self.allow_chunk_autotune, r_mix=self.r_mix, r_mix_dyn=self.r_mix_dyn, backend=self.backend, precision=self.precision, m3_kernel=self.m3_kernel, dt_min=self.dt_min, dt_max=self.dt_max, dt_init_floor=self.dt_init_floor, a_proj_bias=self.a_proj_bias, trap_proj_bias=self.trap_proj_bias, write_rule=WRITE_RULE_DEFAULT if self.write_rule is None else self.write_rule, d_skip=self.d_skip, d_skip_sharing=self.d_skip_sharing, proj_source=self.proj_source, bc_conv=self.bc_conv, chunk_size=self.chunk_size, kernel_options=self.kernel_options)

    def build(self):
        return SSDMimoSSM(**self.to_ssm_kwargs())
from ..compile_compat import dynamo_opaque
from typing import Any
from ..layers.state import BaseMixer
from ..layers.norms import RMSNorm
from ..config import reject_retired_keys
from ..layers.proj import autocast_cast, matmul_channel_major
from ..layers.conv import causal_depthwise_conv1d, causal_depthwise_conv1d_cl
from ..layers.glue import glue_on, param_cache_key, dt_fp32_on

def _fused_in_proj(m, w, dt):
    fold = glue_on('bias')
    cached = glue_on('wcat')
    key = None
    if cached:
        key = param_cache_key([w] + m.ssm.tok_proj_params(), dt, fold, dt_fp32_on())
        hit = getattr(m, '_in_proj_cache', None)
        if hit is not None and hit[0] == key:
            return hit[1]
    wt, bt = m.ssm.tok_proj_weight()
    wcat = torch.cat([w[m.d_inner:], wt], dim=0).to(dt)
    bcat = None
    if bt is not None and fold:
        bcat, bt = (F.pad(bt.to(dt), (m.d_inner, 0)), None)
    val = (wcat, bcat, bt)
    if cached:
        m._in_proj_cache = (key, val)
    return val

class _SSDMimoCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mixer, x, *params):
        from ..layers.conv import causal_conv1d_cl_fwd
        m = mixer
        Bsz, L, _ = x.shape
        dt = torch.get_autocast_dtype('cuda') if x.is_cuda and torch.is_autocast_enabled('cuda') else x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            xa = x if x.dtype == dt else x.to(dt)
            w = m.in_proj.weight
            wx = w[:m.d_inner].to(dt)
            xcl = torch.bmm(wx.expand(Bsz, *wx.shape), xa.transpose(1, 2))
            tok_raw = None
            wcat = bt = None
            if m.fuse_in_proj:
                wcat, bcat, bt = _fused_in_proj(m, w, dt)
                zt = F.linear(xa, wcat, bcat)
                z, tok_raw = zt.split([m.d_inner, m.ssm.tok_width], -1)
                if bt is not None:
                    tok_raw = tok_raw + bt.to(tok_raw.dtype)
            else:
                wcat = w[m.d_inner:].to(dt)
                z = F.linear(xa, wcat)
            xcl_pre = xcl
            if m.conv1d is not None:
                xcl = causal_conv1d_cl_fwd(xcl, m.conv1d.weight, m.conv1d.bias, 'silu')
            x_branch = xcl.transpose(1, 2)
            sel = xa if m.ssm.proj_source == 'layer_input' else None
            u = m.ssm._select(x_branch, sel)
            fused_norm = getattr(m, '_fused_out_norm', None)
            y_core, tape = m.ssm._combined_core_fwd(x_branch, u, z, tok_raw, out_norm=fused_norm)
            rstd = nw = None
            yn = y_core
            if fused_norm is None and (not isinstance(m.norm, nn.Identity)):
                nw = m.norm.weight.to(y_core.dtype)
                yn, rstd = torch.ops.aten._fused_rms_norm(y_core, [y_core.shape[-1]], nw, m.norm.eps)
            wo = m.out_proj.weight.to(dt)
            out = F.linear(yn, wo)
        ctx.mixer = m
        ctx.tape = tape
        ctx.amp_dtype = dt
        ctx.x_dtype = x.dtype
        ctx.saved = (xa, xcl_pre, y_core, yn, rstd, nw, wx, wcat, wo)
        ctx.needs = tuple((p.requires_grad for p in params))
        ctx.x_needs = x.requires_grad
        return out

    @staticmethod
    def backward(ctx, dout):
        from ..layers.conv import causal_conv1d_cl_bwd
        m = ctx.mixer
        if ctx.saved is None:
            raise RuntimeError("the combined SSD-MIMO mixer node releases its saved tensors as it consumes them -- that is what keeps its peak at the multi-node path's -- so it cannot serve a SECOND backward through the same graph.  Build the mixer with combined=False (or set `.combined = False`) if you need retain_graph=True.")
        xa, xcl_pre, y_core, yn, rstd, nw, wx, wcat, wo = ctx.saved
        dt, tape = (ctx.amp_dtype, ctx.tape)
        Bsz, L, dm = xa.shape
        di, tw = (m.d_inner, m.ssm.tok_width)
        grads: dict = {}
        dout = dout if dout.is_contiguous() else dout.contiguous()
        f_out = dout.reshape(-1, dout.shape[-1])
        dyn = (f_out @ wo).view(Bsz, L, wo.shape[1])
        grads['out_proj.weight'] = (f_out.t() @ yn.reshape(-1, wo.shape[1])).to(m.out_proj.weight.dtype)
        if rstd is None:
            dy_core = dyn
        else:
            dy_core, dwn = torch.ops.aten._fused_rms_norm_backward(dyn, y_core, [y_core.shape[-1]], rstd, nw, [True, True])
            grads['norm.weight'] = dwn.to(m.norm.weight.dtype)
        del dout, f_out, dyn, yn
        ctx.saved = None
        if m.fuse_in_proj:
            dslab = torch.empty(Bsz, L, di + tw, device=xa.device, dtype=dt)
            dz_out, dtok_out = (dslab[..., :di], dslab[..., di:])
        else:
            dslab = dz_out = torch.empty(Bsz, L, di, device=xa.device, dtype=dt)
            dtok_out = None
        dx_branch, du, _dtok, cgrads = m.ssm._combined_core_bwd(tape, dy_core, dz_out=dz_out, dslab_out=dtok_out)
        del dy_core, y_core, dz_out, dtok_out
        ctx.tape = None
        for k, v in cgrads.items():
            if k == '@out_norm_w':
                grads['norm.weight'] = v.to(m.norm.weight.dtype)
            else:
                grads['ssm.' + k] = v
        if du is not None:
            dx_branch = dx_branch + du.reshape(dx_branch.shape)
        dxcl = dx_branch.transpose(1, 2)
        if m.conv1d is not None:
            dxcl, dcw, dcb = causal_conv1d_cl_bwd(xcl_pre, m.conv1d.weight, m.conv1d.bias, dxcl, 'silu')
            grads['conv1d.weight'] = dcw.to(m.conv1d.weight.dtype)
            grads['conv1d.bias'] = dcb.to(m.conv1d.bias.dtype)
        del dx_branch, xcl_pre
        dw_in = torch.empty_like(m.in_proj.weight)
        if glue_on('dwin'):
            dw_in[:di].copy_(torch.bmm(dxcl, xa).sum(0))
        else:
            dw_in[:di] = torch.bmm(dxcl, xa).sum(0).to(dw_in.dtype)
        dxa = torch.bmm(dxcl.transpose(1, 2), wx.expand(Bsz, *wx.shape))
        f_slab = dslab.reshape(-1, dslab.shape[-1])
        dxa = dxa.reshape(-1, dm)
        if glue_on('addmm'):
            dxa.addmm_(f_slab, wcat)
        else:
            dxa = torch.addmm(dxa, f_slab, wcat)
        dxa = dxa.view(Bsz, L, dm)
        dw_cat = f_slab.t() @ xa.reshape(-1, dm)
        dw_src = dw_cat[:di] if m.fuse_in_proj else dw_cat
        if glue_on('dwin'):
            dw_in[di:].copy_(dw_src)
        else:
            dw_in[di:] = dw_src.to(dw_in.dtype)
        grads['in_proj.weight'] = dw_in
        if m.fuse_in_proj:
            for k, v in m.ssm.scatter_tok_grads(dw_cat[di:]).items():
                grads['ssm.' + k] = v
        dx = dxa.to(ctx.x_dtype) if ctx.x_needs else None
        out = [None, dx]
        for (name, p), need in zip(m._combined_named_params(), ctx.needs):
            g = grads.get(name) if need else None
            if g is not None and tuple(g.shape) != tuple(p.shape):
                g = g.reshape(p.shape)
            out.append(g)
        return tuple(out)

@dynamo_opaque
def _combined_apply(mixer, x, params):
    return _SSDMimoCombinedFn.apply(mixer, x, *params)

class SSDMimoMixer(MixerConfigurable, BaseMixer):
    config_family = 'ssd_mimo'
    config_execution_fields = ('mode', 'modular_checkpoint')

    def __init__(self, d_model: int, d_state: int=64, expand: float | None=2.0, d_conv: int=4, bc_norm: None=None, combined: bool=False, mode: str | None=None, modular_checkpoint: bool=False, ssm_kwargs: dict | None=None, *, d_inner: int | None=None):
        super().__init__()
        if isinstance(ssm_kwargs, dict):
            reject_retired_keys(ssm_kwargs, 'SSDMimoMixer.ssm_kwargs')
        if bc_norm is not None:
            raise _config_spec.ConfigSpecError(f'SSDMimoMixer: {_config_spec.MIXER_BC_NORM_REMOVED}')
        from ..layers.widths import resolve_mixer_width
        d_inner = resolve_mixer_width(d_model, expand, d_inner)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        if d_conv > 0:
            self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, bias=True)
        else:
            self.conv1d = None
        kw = mixer_core_options(ssm_kwargs, config_type=SSDMimoConfig, d_inner=d_inner, d_state=d_state, d_model=d_model, d_conv=d_conv)
        out_norm = bool(kw.pop('out_norm', False))
        self.ssm = SSDMimoSSM(**kw)
        self.d_gate = self.ssm.d_out
        if self.d_gate != d_inner:
            self.in_proj = nn.Linear(d_model, d_inner + self.d_gate, bias=False)
        self.norm = RMSNorm(self.ssm.d_out, fused=True) if out_norm else nn.Identity()
        self.out_proj = nn.Linear(self.ssm.d_out, d_model, bias=False)
        self.fuse_in_proj = self.ssm.proj_source == 'layer_input'
        self._prologue_wcat_cache: dict = {}
        if mode is None:
            mode = 'fused' if combined else 'modular'
        if mode not in ('fused', 'modular'):
            raise ValueError(f"SSDMimoMixer mode must be 'fused' or 'modular', got {mode!r}")
        if modular_checkpoint and mode != 'modular':
            raise ValueError("modular_checkpoint is only meaningful with mode='modular'")
        import os as _os
        self.mode = mode
        self.modular_checkpoint = bool(modular_checkpoint)
        self.combined = mode == 'fused' and _os.environ.get('SSD_MIMO_COMBINED', '1') == '1'
        self._combined_refusal = 'off'
        self._out_norm_refusal = 'not run'
        self._fused_out_norm = None
        self._registered_m2 = _os.environ.get('SSD_MIMO_REGISTERED_M2', '0') == '1'
        if self._registered_m2:
            from ..layers import conv as _registered_conv
            from ..kernels.ssd_mimo import registered as _registered_scan
        self._registered_m2_refusal = 'not run'
        record_mixer_config(self, expand=expand, out_norm=out_norm, execution=dict(mode=self.mode, modular_checkpoint=self.modular_checkpoint))

    def _combined_named_params(self):
        out = [('in_proj.weight', self.in_proj.weight)]
        if self.conv1d is not None:
            out.append(('conv1d.weight', self.conv1d.weight))
            out.append(('conv1d.bias', self.conv1d.bias))
        if not isinstance(self.norm, nn.Identity):
            out.append(('norm.weight', self.norm.weight))
        out.append(('out_proj.weight', self.out_proj.weight))
        out.extend((('ssm.' + n, p) for n, p in self.ssm.named_parameters()))
        return out

    def _resolve_out_norm(self, x_branch, *, gate_z, state, return_state):
        if isinstance(self.norm, nn.Identity):
            self._out_norm_refusal = 'no output norm'
            self._fused_out_norm = None
            return None
        why = self.ssm.out_norm_refusal(x_branch, gate_z=gate_z, state=state, return_state=return_state)
        self._out_norm_refusal = why
        self._fused_out_norm = None if why is not None else (self.norm.weight, self.norm.eps)
        return self._fused_out_norm

    def forward(self, x: torch.Tensor, *, state=None, return_state: bool=False):
        if state is not None:
            state = self.resolve_state(state, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if getattr(self, '_registered_m2', False):
            why = self._registered_m2_why(x, state, return_state)
            self._registered_m2_refusal = why
            if why is None:
                return self._forward_registered_m2(x)
        if self.combined:
            why = self.ssm.combined_refusal(x, gate_z=x, extra_carry=state is not None, return_state=return_state)
            self._combined_refusal = why
            if why is None:
                self._resolve_out_norm(x, gate_z=x, state=None, return_state=False)
                return _combined_apply(self, x, [p for _, p in self._combined_named_params()])
        else:
            self._combined_refusal = 'off'
        return self._forward_multinode(x, state=state, return_state=return_state)

    def _registered_m2_why(self, x, state, return_state):
        import os
        from ..layers.conv import _causal_conv1d_fn
        from ..kernels.ssd_mimo.autograd import _FUSED_SSD
        from ..kernels.ssd_mimo.fused_fwd_kernel import _env
        s = self.ssm
        if self.mode != 'modular' or self.modular_checkpoint:
            return 'requires non-checkpointed modular mode'
        if state is not None or return_state:
            return 'stateful call'
        if not x.is_cuda or s.backend not in ('cuda', 'auto'):
            return 'requires CUDA backend and input'
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        if dtype not in (torch.float32, torch.bfloat16):
            return 'requires FP32 or BF16 IO'
        if any((p.dtype != torch.float32 for p in self.parameters())):
            return 'requires FP32 parameters'
        if x.ndim != 3 or x.shape[1] < 64 or x.shape[1] % 64:
            return 'requires full T64 chunks'
        if (s.N, s.G, s.R, s.O) != (128, 64, 1, 1):
            return 'requires N128/G64/R1/O1'
        if s.a_dyn or s.rot_mode is not None or s.use_trap or s.use_g_b or s.use_g_c or s.use_theta_h:
            return 'requires static unrotated ungated maps'
        if s.bc_bias or s.b_norm or s.c_norm or (not s.d_skip) or (s.proj_source != 'layer_input') or (not self.fuse_in_proj):
            return 'requires M2 projection, skip and map configuration'
        if self.d_conv != 4 or s.bc_conv1d is None or s.bc_conv1d.d_conv != 4 or (_causal_conv1d_fn is None):
            return 'requires both width-four extension convolutions'
        if isinstance(self.norm, nn.Identity) or self.norm.eps != 1e-05:
            return 'requires output RMSNorm eps=1e-5'
        if s.precision not in ('tf32', 'io') or not _tail_law_is_stock() or (not _FUSED_SSD):
            return 'requires stock tail law and tf32/io precision setting'
        if not glue_on('bias'):
            return 'requires projection bias epilogue'
        if os.environ.get('UNIFIED_SSD_SALL_FP32', '0') != '0':
            return 'unsupported setting: UNIFIED_SSD_SALL_FP32'
        for name, default, required in (('SISO_FAST', '0', '1'), ('MMA_IO', '0', '1'), ('GATE_INKERNEL', '0', '0'), ('GATE_FUSED', '1', '1'), ('GATE_OUT_IO', '1', '1')):
            if _env(name, default) != required:
                return 'unsupported setting: ' + name
        return None

    def _forward_registered_m2(self, x):
        from ..layers.conv import conv_fwd
        from ..kernels.ssd_mimo.registered import m2_fwd
        s = self.ssm
        batch, length, _ = x.shape
        xa = autocast_cast(x)
        xcl = matmul_channel_major(self.in_proj.weight[:self.d_inner], xa, grad_layout='token')
        wt, bt = s.tok_proj_weight()
        wz = self.in_proj.weight[self.d_inner:]
        if glue_on('wcat'):
            wcat_key = param_cache_key([self.in_proj.weight] + s.tok_proj_params())
            wcat = cached_cat(self._prologue_wcat_cache, wcat_key, [wz, wt], dim=0)
        else:
            wcat = torch.cat((wz, wt))
        zt = F.linear(xa, wcat, F.pad(bt.to(xa.dtype), (self.d_inner, 0)))
        z, raw = zt.split((self.d_inner, s.tok_width), -1)
        parts = s.split_tok_raw(raw)
        xcl = conv_fwd(xcl, self.conv1d.weight.squeeze(1), self.conv1d.bias, True)
        bcconv = s.bc_conv1d.conv
        bc = conv_fwd(parts['_bc'].transpose(1, 2), bcconv.weight.squeeze(1), bcconv.bias, True).transpose(1, 2)
        b, c = bc.split(s.N, -1)
        dt = _dt_tail(parts['dt'].float() + (parts['_dtb'].float() if parts.get('_dtb') is not None else 0.0), False).transpose(1, 2).contiguous()
        dc = dt.reshape(batch, s.H, length // 64, 64)
        a = s.A_log.reshape(s.H).float()
        skip = s._skip_gains().float().reshape(s.H, 64, 1).contiguous()
        y, _, _, _, _ = m2_fwd(c.reshape(batch, length // 64, 64, 128), b.reshape(batch, length // 64, 64, 128), xcl.reshape(batch, s.H, 64, 1, length), dt, dc, a, z, skip, self.norm.weight)
        self._combined_refusal = 'off'
        self._out_norm_refusal = None
        return self.out_proj(y.permute(0, 2, 3, 1, 4, 5).reshape(batch, length, self.d_inner))

    def _forward_multinode(self, x: torch.Tensor, *, state=None, return_state: bool=False):
        want_state = return_state or state is not None
        if self.modular_checkpoint and torch.is_grad_enabled() and (not want_state):
            from torch.utils.checkpoint import checkpoint
            y = checkpoint(self._checkpointed_modular_body, x, use_reentrant=False)
            return self.out_proj(y)
        from ..layers import regen as _regen
        _sp = getattr(self.ssm, 'spec', None) or {}
        _real = self.ssm.R == 1 and self.ssm.O == 1 and (not _sp.get('rotation')) and (not any((b.get('complex') for b in _sp.get('rows', ()))))
        self._regen_real = _real
        _rg = not want_state and _regen.mode(_real) != ''
        with _regen.scope(_rg):
            return self._forward_multinode_body(x, state, want_state, return_state, _rg)

    def _forward_multinode_body(self, x, state, want_state, return_state, _rg):
        from ..layers import regen as _regen
        self._x_regen_fn = None
        xa, x_branch, z, tok_raw, conv_tail = self._forward_prologue(x, state=state, want_state=want_state)
        if _rg and _regen.wants('x', self._regen_real) and (self._x_regen_fn is not None):
            _regen.register(x_branch, self._x_regen_fn)
        self._x_regen_fn = None
        sel = xa if self.ssm.proj_source == 'layer_input' else None
        ssm_state = State({k: v for k, v in state.items() if k != 'conv_state'}) if state is not None else None
        fused_norm = self._resolve_out_norm(x_branch, gate_z=z, state=ssm_state, return_state=want_state)
        core_result = self.ssm(x_branch, gate_z=z, sel_x=sel, tok_raw=tok_raw, state=ssm_state, return_state=want_state, out_norm=fused_norm)
        y, out_ssm_state = core_result if want_state else (core_result, None)
        y = self.out_proj(y if fused_norm is not None else self.norm(y))
        if not return_state:
            return y
        new_state = State(out_ssm_state)
        if conv_tail is not None:
            new_state['conv_state'] = conv_tail
        self._final_state = new_state
        return (y, new_state)

    def _forward_prologue(self, x: torch.Tensor, *, state=None, want_state: bool=False):
        w = self.in_proj.weight
        xa = autocast_cast(x)
        tok_raw = None
        if self.fuse_in_proj:
            wt, bt = self.ssm.tok_proj_weight()
            b_ep = bt if bt is not None and glue_on('bias') else None
            xcl, z, tok_raw = split_in_proj(xa, w, wt, b_ep, self.d_inner)
            if bt is not None and b_ep is None:
                tok_raw = tok_raw + bt.to(tok_raw.dtype)
        else:
            xcl, z, _ = split_in_proj(xa, w, None, None, self.d_inner)
        conv_tail = None
        if self.conv1d is not None:
            kc = self.d_conv
            prefix = state['conv_state'][:, :, 1:].to(xcl.dtype) if state is not None else None
            x_in = xcl if prefix is None else torch.cat([prefix, xcl], dim=-1)
            xcl = causal_depthwise_conv1d_cl(x_in, self.conv1d.weight, self.conv1d.bias, kc, activation='silu')
            if prefix is None and getattr(self, '_x_regen_fn', 0) is None:
                _acd = (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
                _cw, _cb = (self.conv1d.weight, self.conv1d.bias)

                def _x_regen(x_in=x_in, _cw=_cw, _cb=_cb, kc=kc, _acd=_acd):
                    with torch.autocast('cuda', dtype=_acd[1], enabled=_acd[0]):
                        return causal_depthwise_conv1d_cl(x_in, _cw, _cb, kc, activation='silu')
                self._x_regen_fn = _x_regen
            if prefix is not None:
                xcl = xcl[..., prefix.shape[-1]:].contiguous()
            if want_state:
                tail = x_in[..., -kc:]
                if tail.shape[-1] < kc:
                    tail = torch.cat([tail.new_zeros(*tail.shape[:-1], kc - tail.shape[-1]), tail], dim=-1)
                conv_tail = tail if torch.is_grad_enabled() else tail.detach().clone()
        x_branch = xcl.transpose(1, 2)
        return (xa, x_branch, z, tok_raw, conv_tail)

    def _checkpointed_modular_body(self, x: torch.Tensor):
        xa, x_branch, z, tok_raw, _ = self._forward_prologue(x)
        sel = xa if self.ssm.proj_source == 'layer_input' else None
        fused_norm = self._resolve_out_norm(x_branch, gate_z=z, state=None, return_state=False)
        y = self.ssm(x_branch, gate_z=z, sel_x=sel, tok_raw=tok_raw, state=None, return_state=False, out_norm=fused_norm)
        return y if fused_norm is not None else self.norm(y)

    def _default_state_dtype(self) -> torch.dtype:
        return torch.promote_types(next(self.parameters()).dtype, torch.float32)

    def _init_state(self, batch_size: int, device: torch.device | None=None, dtype: torch.dtype | None=None) -> State:
        dev = device or next(self.parameters()).device
        if dtype is None:
            dtype = self._default_state_dtype()
        state = self.ssm.init_state(batch_size, device=dev, dtype=dtype)
        if self.conv1d is not None:
            state['conv_state'] = torch.zeros(batch_size, self.d_inner, self.d_conv, device=dev, dtype=dtype)
        return state
    step_inplace: bool = False

    def step(self, x_t: torch.Tensor, state=None) -> tuple[torch.Tensor, State]:
        if self._fast_inference() and _torch_impl() == 'fast':
            from .ssd_mimo_torch import mixer_step_fast
            return mixer_step_fast(self, x_t, state)
        state = self.resolve_state(state, batch_size=x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        xz = self.in_proj(x_t)
        x_branch, z = xz.split([self.d_inner, self.d_gate], dim=-1)
        if self.conv1d is not None:
            conv_state = torch.cat([state['conv_state'][:, :, 1:], x_branch.unsqueeze(-1)], dim=-1)
            conv_w = self.conv1d.weight.squeeze(1).to(x_branch.dtype)
            conv_b = self.conv1d.bias.to(x_branch.dtype)
            x_branch = F.silu((conv_state * conv_w).sum(-1) + conv_b)
            state = {**state, 'conv_state': conv_state}
        sel = x_t if self.ssm.proj_source == 'layer_input' else None
        y_t, ssm_state = self.ssm.step(x_branch, {k: v for k, v in state.items() if k != 'conv_state'}, gate_z=z, sel_x=sel)
        state = State({**state, **ssm_state})
        self._final_state = state
        return (self.out_proj(self.norm(y_t)), state)
