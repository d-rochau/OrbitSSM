
from __future__ import annotations
import math
import os as _os
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..kernels.hybrid_mamba_mh.layout import DYN_EXP, DYN_RAW, DYN_SOFTPLUS, GATE_NAMES, LAW_A_DYN, LAW_A_FIX, LAW_MU, LAW_NAMES, Frac, dyn_rate, fracs_to_rows, gate_act, sigmoid_exact, group_index, norm_fracs
from ..layers.conv import CausalDepthwiseConv, causal_depthwise_conv1d_cl
from ..layers.proj import autocast_cast, matmul_channel_major, matmul_channel_major_tee, resolve_proj_source
from ..layers.glue import hm_glue_on
from ..layers import regen as _regen

def hm_held_regen() -> frozenset:
    v = _os.environ.get('HM_HELD_REGEN')
    if v is None:
        v = 'all'
    v = v.strip().lower()
    if v in ('', 'off', '0', 'false', 'no'):
        return frozenset()
    items = {t.strip() for t in v.split(',') if t.strip()}
    if 'all' in items:
        return frozenset(('x', 'out', 'delta', 'dtlin', 'amag'))
    bad = items - {'x', 'out', 'delta', 'dtlin', 'amag'}
    if bad:
        raise ValueError(f'HM_HELD_REGEN: unknown item(s) {sorted(bad)}')
    return frozenset(items)
from ..layers.norms import RMSNorm, gated_rms_norm, rms_norm_cast, swiglu_gate
from ..layers.state import BaseMixer, BaseSSM, State, ssm_status
from .ssd_mimo import A_ACTS, _THETA_FREQ_INIT, _a_mag_tail
from ..kernels.hybrid_mamba_mh.torch_scan import phase_bmm
from ..config import BACKENDS, check_backend, reject_retired_keys, reject_unknown_kwargs
from .. import config_spec as _config_spec
from ..configuration import Configurable, positive_int, resolve_heads, resolve_dt_rank, record_constructor_config, MixerConfigurable, mixer_core_options, record_mixer_config
GATE_INITS = ('zero', 'kaiming_uniform', 'kaiming_small')
GATE_SMALL_STD = 0.1
GATE_SMALL_SCALE = GATE_SMALL_STD * 3 ** 0.5

def _init_gate_weight(lin, gate_init):
    if gate_init == 'zero':
        nn.init.zeros_(lin.weight)
    elif gate_init == 'kaiming_small':
        with torch.no_grad():
            lin.weight.mul_(GATE_SMALL_SCALE)

def _flat_phase_tile(batch, heads, pairs):
    return max(64, (16 << 20) // (batch * heads * pairs) // 64 * 64)

def _expand_group(beta: torch.Tensor | None, H: int) -> torch.Tensor | None:
    if beta is None:
        return None
    G = beta.shape[0]
    return beta[group_index(H, G, device=beta.device)]
DYN_ACTS = ('heavy_tail', 'softplus', 'exp', 'raw')
_DYN_NAMES = {'raw': DYN_RAW, 'identity': DYN_RAW, 'softplus': DYN_SOFTPLUS, 'exp': DYN_EXP}
_DYN_CODE_NAMES = {DYN_RAW: 'raw', DYN_SOFTPLUS: 'softplus', DYN_EXP: 'exp'}
_HM_LAWS = (LAW_A_FIX, LAW_A_DYN, LAW_MU)
_SCALE_GATES = ('one_plus', 'exp')
FLAT_LAWS = ('m3', 'm2')
_BLOCK_KEYS = set(_config_spec.BLOCK_KEYS)
_MAX_BLOCKS = 8
THETA_H_CONVENTION = 1
ACT_MARKER_CODES = {'raw': 0, 'softplus': 1, 'exp': 2, 'heavy_tail': 3}
_ACT_MARKER_NAMES = {v: k for k, v in ACT_MARKER_CODES.items()}
_ACT_MARKERS = {'a_act': ('a_act_code', 'm3_a_proj.weight', A_ACTS), 'dyn_act': ('dyn_act_code', 'a_dyn_proj.weight', DYN_ACTS)}
PROJ_SOURCE_MARKER_CODES = {'main_branch': 0, 'layer_input': 1}
_PROJ_SOURCE_MARKER_NAMES = {v: k for k, v in PROJ_SOURCE_MARKER_CODES.items()}
_SEL_MAP_ATTRS = ('bc_proj', 'a_dyn_proj', 'mu_proj', 'theta_proj', 'trap_proj', 'gB_proj', 'gC_proj', 'm3_a_proj', 'm3_trap_proj', 'm3_angle_proj')
_FLAT_LAW_OF = {_config_spec.HEAD_DYNAMIC: 'm3', _config_spec.HEAD_STATIC: 'm2'}

def _phase_law(*args, **kwargs):
    from ..kernels.ssd_mimo.m3_epilogue import phase_law
    return phase_law(*args, **kwargs)
MU_CAP_DEFAULT = 4.0
MU_CAP_WARN = 20.0

def check_mu_cap(mu_cap) -> float:
    try:
        cap = float(mu_cap)
    except (TypeError, ValueError):
        raise ValueError(f'mu_cap must be a real number > 1, got {mu_cap!r}') from None
    if not (cap > 1.0 and math.isfinite(cap)):
        raise ValueError(f"mu_cap must be finite and > 1 (the mu potential's span factor lies in (1/mu_cap, mu_cap)), got {mu_cap!r}")
    if cap > MU_CAP_WARN:
        warnings.warn(f"mu_cap={cap:g} > {MU_CAP_WARN:g}: the mu law allows large transient growth -- a span's decay product (and so the state) can amplify by up to {cap:g}x", stacklevel=3)
    return cap

def mu_gain_bound(mu_cap: float) -> float:
    return 0.5 * math.log(mu_cap)

def describe_rows(rows: Sequence[dict]) -> str:
    parts = []
    for b in rows:
        size = f"{b['rows']}" if 'rows' in b else f"{b['frac']:g}f"
        parts.append(f"{size}x{b['decay']}{('+cx' if b.get('complex') else '')}")
    return ' | '.join(parts)

@dataclass(frozen=True)
class CompiledFracs:
    user: tuple[dict, ...]
    hm: tuple[Frac, ...]
    n_hm: int
    n3: int
    n3_complex: int
    n2: int = 0
    hm_gate: tuple[bool, ...] = ()
    flat_gated: bool = False

    @property
    def n_flat(self) -> int:
        return self.n3 + self.n2

def compile_rows(rows: Sequence[dict], d_state: int) -> CompiledFracs:
    spec = _config_spec.resolve('hybrid_mamba', d_state=d_state, rows=rows)
    fracs = [dict(b, decay=_FLAT_LAW_OF.get(b['decay'], b['decay'])) for b in spec['rows']]
    public = [b['decay'] for b in spec['rows']]
    user: list[dict] = []
    for i, spec in enumerate(fracs):
        extra = set(spec) - _BLOCK_KEYS
        if extra:
            hint = ''
            if extra & {'rot', 'rotate'}:
                hint = " (the old schema's 'rotate'/'rot' key is now 'complex')"
            raise ValueError(f'rows[{i}]: unknown keys {sorted(extra)}{hint}')
        if 'rows' in spec:
            rows = int(spec['rows'])
        elif 'frac' in spec:
            rows = int(round(float(spec['frac']) * d_state))
        else:
            raise ValueError(f"rows[{i}]: needs 'rows' or 'frac'")
        if rows <= 0:
            raise ValueError(f'rows[{i}]: resolved to {rows} rows')
        decay = spec.get('decay', 'A')
        cx = bool(spec.get('complex', False))
        if isinstance(decay, str) and decay in FLAT_LAWS:
            law = ('flat', decay)
            if cx and decay != 'm3':
                raise ValueError(f"rows[{i}]: complex=True is served on flat rows by the 'head_dynamic' law only (got {public[i]!r})")
        else:
            law = LAW_NAMES[decay] if isinstance(decay, str) else int(decay)
            if law not in _HM_LAWS:
                raise ValueError(f"rows[{i}]: decay {decay!r} is not usable here; choose 'A'/'dyn'/'mu' (mh_scan, per-coordinate) or 'head_dynamic'/'head_static' (ssd_mimo, flat rows)")
        user.append({'rows': rows, 'decay': decay, 'complex': cx, 'gate': bool(spec['gate']), '_law': law})
    total = sum((b['rows'] for b in user))
    if total != d_state:
        raise ValueError(f'rows sum to {total}, need exactly d_state={d_state}')
    hm_specs = [b for b in user if not isinstance(b['_law'], tuple)]
    flat_gated = any((b['gate'] for b in user if isinstance(b['_law'], tuple)))
    flat = {law: [b for b in user if b['_law'] == ('flat', law)] for law in FLAT_LAWS}
    n3_complex = sum((b['rows'] for b in flat['m3'] if b['complex']))
    n3 = sum((b['rows'] for b in flat['m3']))
    n2 = sum((b['rows'] for b in flat['m2']))
    n_hm = d_state - n3 - n2
    if n3_complex % 2:
        raise ValueError(f'complex head_dynamic rows must come in pairs, got {n3_complex}')
    for law, n in (('head_dynamic', n3), ('head_static', n2)):
        if n > 0 and (n & n - 1 or not 16 <= n <= 128):
            raise ValueError(f"{law} rows total {n} must be a power of two in [16, 128] (it is that launch's ssd_mimo state width N)")
    try:
        hm = tuple(norm_fracs([{'rows': b['rows'], 'decay': b['_law'], 'rot': b['complex']} for b in hm_specs], n_hm)) if n_hm > 0 else ()
    except ValueError as e:
        raise ValueError(f'rows invalid after packing (flat rows move to the tail, so HM offsets are counted with flat blocks removed): {e}') from e
    for b, name in zip(user, public):
        del b['_law']
        b['decay'] = name
    return CompiledFracs(user=tuple(user), hm=hm, n_hm=n_hm, n3=n3, n3_complex=n3_complex, n2=n2, hm_gate=tuple((b['gate'] for b in hm_specs)), flat_gated=flat_gated)

def _m3_reference(Bt, Ct, v, ADT, DT, Trap, K_bias, Q_bias, angles, *, S_init=None, kp_init=None, vp_init=None, phase_init=None, return_state=False):
    Bt, Ct, v = (Bt.double(), Ct.double(), v.double())
    ADT, DT = (ADT.double(), DT.double())
    lam = torch.sigmoid(Trap.double())
    alpha = torch.exp(ADT)
    dt_blh = DT.transpose(1, 2)
    phase0 = phase_init.double().unsqueeze(1) if phase_init is not None else 0.0
    theta = phase0 + torch.cumsum(torch.tanh(angles.double()) * math.pi * dt_blh.unsqueeze(-1), dim=1)
    k = Bt + K_bias.double().unsqueeze(0).unsqueeze(0)
    q = Ct + Q_bias.double().unsqueeze(0).unsqueeze(0)
    Ka = angles.shape[-1]
    cos, sin = (torch.cos(theta), torch.sin(theta))

    def rot(t):
        e, o = (t[..., 0::2], t[..., 1::2])
        er = torch.cat([e[..., :Ka] * cos - o[..., :Ka] * sin, e[..., Ka:]], dim=-1)
        og = torch.cat([e[..., :Ka] * sin + o[..., :Ka] * cos, o[..., Ka:]], dim=-1)
        return torch.stack([er, og], dim=-1).flatten(-2)
    k, q = (rot(k), rot(q))
    Bsz, L, H, P = v.shape
    n3 = k.shape[-1]
    S = v.new_zeros(Bsz, H, P, n3) if S_init is None else S_init.double().clone()
    kp = k.new_zeros(Bsz, H, n3) if kp_init is None else kp_init.double().clone()
    vp = v.new_zeros(Bsz, H, P) if vp_init is None else vp_init.double().clone()
    ys = []
    for t in range(L):
        a = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)
        d = DT[:, :, t].unsqueeze(-1).unsqueeze(-1)
        lm = lam[:, :, t].unsqueeze(-1).unsqueeze(-1)
        kt, vt = (k[:, t], v[:, t])
        S = a * S + a * d * (1 - lm) * vp.unsqueeze(-1) * kp.unsqueeze(-2) + lm * d * vt.unsqueeze(-1) * kt.unsqueeze(-2)
        kp, vp = (kt, vt)
        ys.append(torch.einsum('bhpn,bhn->bhp', S, q[:, t]))
    Y = torch.stack(ys, dim=1)
    if not return_state:
        return Y
    state = dict(S=S, kp=kp, vp=vp, phase=theta[:, -1])
    return (Y, state)

def _flat_reference(Bt, Ct, v, log_a, write, B_bias, C_bias, *, S_init=None, return_state=False):
    Bt, Ct, v = (Bt.double(), Ct.double(), v.double())
    a = torch.exp(log_a.double())
    w = write.double()
    k = Bt.unsqueeze(2) + B_bias.double()
    q = Ct.unsqueeze(2) + C_bias.double()
    Bsz, L, H, P = v.shape
    n = k.shape[-1]
    S = v.new_zeros(Bsz, H, P, n) if S_init is None else S_init.double().clone()
    ys = []
    for t in range(L):
        S = a[:, :, t, None, None] * S + w[:, :, t, None, None] * v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
        ys.append(torch.einsum('bhpn,bhn->bhp', S, q[:, t]))
    Y = torch.stack(ys, dim=1)
    return (Y, S) if return_state else Y

@ssm_status('production', since='2026-08', kernel='mh_scan (structure B, hybrid_mamba_mh) + ssd_mimo_scan (R = O = 1) per-frac dispatch', reason='THE migration target, and the one SSM that calls BOTH kernel families. State rows split by fracs between mh_scan (laws A/dyn/mu, rotation, gates, betas) and ssd_mimo_scan (flat M3/M2 laws); outputs summed, which is exact because y is a sum over rows.')
class HybridMambaSSM(Configurable, BaseSSM):
    supports_fused_scan_gate = False
    external_value_dim = 0
    bc_from_residual = False
    config_family = 'hybrid_mamba'

    def __init__(self, d_inner: int, d_state: int, *, n_heads: int | None=None, heads: int | None=None, H: int | None=None, head_subsystems: int | None=None, P: int | None=None, rows: Sequence[dict] | None=None, rotation: dict | None=None, maps: dict | None=None, proj_source: str | None=None, d_model: int | None=None, bc_conv: bool | None=None, d_conv: int=4, dt_rank: int | str | None='adaptive', force_dt_dense: bool=False, dt_init: str | None='random', dt_scale: float=1.0, dt_min: float=0.001, dt_max: float=0.1, dt_init_floor: float=0.0001, dyn_res: str='n', dyn_act: str='heavy_tail', mu_res: str='n', mu_cap: float=MU_CAP_DEFAULT, a_floor: float=0.0001, a_act: Literal['heavy_tail', 'softplus']='heavy_tail', norm_eps: float=1e-05, d_skip_mode: str='head', use_trap: bool=True, backend: str='auto', ckpt: bool=True, m3_kernel: str='native', mem: dict | None=None, kernel_options: dict | None=None, learnable_h_init: bool=False, gate_init: str='kaiming_uniform', theta_per_head: bool | None=None, learn_theta_freq: bool | None=None, use_theta_h: bool | None=None, b_norm: bool | None=None, c_norm: bool | None=None, bc_norm: bool | None=None, bias_gran: str | None=None, use_g_b: bool | None=None, use_g_c: bool | None=None, gate_act: str | None=None, gate_bias: bool | None=None, gate_pos: str | None=None, **unexpected: Any) -> None:
        super().__init__()
        _config_spec.reject_removed(unexpected, 'HybridMambaSSM')
        reject_unknown_kwargs(unexpected, 'HybridMambaSSM')
        from ..kernel_options import validate
        self.kernel_options = validate('hybrid_mamba', kernel_options)
        d_inner = positive_int(d_inner, 'd_inner')
        d_state = positive_int(d_state, 'd_state')
        n_heads, _P = resolve_heads(d_inner, n_heads=n_heads, heads=heads, H=H, head_subsystems=head_subsystems, P=P)
        check_backend(backend, 'HybridMambaSSM', BACKENDS)
        if backend in ('auto', 'cuda') and _P & _P - 1:
            raise ValueError(f"HybridMambaSSM: head width P=d_inner/n_heads={_P} must be a power of two for the CUDA backward (d_inner={d_inner}, n_heads={n_heads}); pick n_heads so that P is a power of two (P=1, i.e. n_heads=d_inner, is Mamba-1's per-channel dt)")
        if dyn_res not in ('n', 'h') or mu_res not in ('n', 'h'):
            raise ValueError(f"dyn_res/mu_res must be 'n' or 'h', got {dyn_res!r}/{mu_res!r}")
        self.mu_cap = check_mu_cap(mu_cap)
        self.d_inner = d_inner
        self.d_state = d_state
        self.H = n_heads
        self.P = d_inner // n_heads
        self.out_width = d_inner
        from ..kernels.m3_adapter import check_provider
        check_provider(m3_kernel, backend)
        self.m3_kernel = m3_kernel
        self.backend = backend
        self.ckpt = bool(ckpt)
        self.mem = dict(mem) if mem else None
        self.a_floor = float(a_floor)
        if gate_init not in GATE_INITS:
            raise ValueError(f'gate_init must be one of {GATE_INITS}, got {gate_init!r}')
        self.gate_init = gate_init
        self.a_act = a_act
        self.dyn_act = dyn_act
        self.dyn_res = dyn_res
        self.mu_res = mu_res
        spec = _config_spec.resolve('hybrid_mamba', d_state=d_state, rows=rows, rotation=rotation, maps=maps, **_config_spec.flat_kwargs(locals()))
        self.spec = spec
        rot = spec['rotation'] or {}
        mp = spec['maps']
        learn_theta_freq = bool(rot.get('learn_theta_freq', False))
        use_theta_h = bool(rot.get('use_theta_h', False))
        theta_per_head = bool(rot.get('theta_per_head', False))
        for key, default in (('omega_layout', 'pair'), ('theta_init', 'kaiming')):
            if rot.get(key, default) != default:
                raise NotImplementedError(f'HybridMambaSSM: rotation.{key}={rot[key]!r} is not implemented (SSD-MIMO only)')
        gates_b, gates_c = (bool(mp['gate_b']), bool(mp['gate_c']))
        gate_act, gate_bias, gate_pos = (mp['gate_act'], mp['gate_bias'], mp['gate_pos'])
        b_norm, c_norm = (bool(mp['b_norm']), bool(mp['c_norm']))
        self.bias_gran = mp['bias']
        self._gate_act = GATE_NAMES[gate_act]
        self.gate_pos = gate_pos
        plan = compile_rows(spec['rows'], d_state)
        self.fracs = plan
        self.d_state_hm = plan.n_hm
        self.d_state_m3 = plan.n3
        if m3_kernel == 'upstream' and (plan.n3 not in (16, 32, 64, 128) or self.P not in (16, 32, 64, 128)):
            raise ValueError('upstream M3 requires N_m3 and P in {16,32,64,128}, and M3 rows')
        self.d_state_m2 = plan.n2
        n_hm = plan.n_hm
        laws = {f.law for f in plan.hm}
        has_rot = any((f.rot for f in plan.hm))
        if has_rot and n_hm % 2:
            raise ValueError(f'HM rotation needs even N_hm, got {n_hm}')
        self._host_buffers = {}
        _law, _rot = ([], [])
        for f in plan.hm:
            _law += [int(f.law)] * f.rows
            _rot += [bool(f.rot)] * f.rows
        assert len(_law) == n_hm
        self._register_host_buffer('row_law', _law, torch.int32)
        self._register_host_buffer('row_rot', _rot, torch.bool)
        if plan.n_flat:
            gaps = [name for name, bad in (('b_norm=False', not b_norm), ('c_norm=False', not c_norm), (f'bias={self.bias_gran!r}', self.bias_gran != 'head_bc'), ('gate_b on a gated flat block', gates_b and plan.flat_gated), ('gate_c on a gated flat block', gates_c and plan.flat_gated), ('learn_theta_freq', plan.n3_complex and learn_theta_freq), ('use_theta_h', plan.n3_complex and use_theta_h), ('theta_per_head', plan.n3_complex and theta_per_head)) if bad]
            if gaps:
                raise NotImplementedError(f"HybridMambaSSM: {', '.join(gaps)} on the flat head_dynamic/head_static rows is not yet implemented (they serve b_norm=c_norm=True, bias='head_bc', no gates -- set rows[i]['gate']=False on them --, and complex flat rows rotate at the fixed frequency pi with no read offset).  The rotation is ONE global setting, so it is never dropped silently for the flat rows: pass rotation=dict(learn_theta_freq=False, use_theta_h=False) explicitly (this also applies to complex HM rows in the same core), or make the flat rows real")
        if n_hm and gate_bias and (gate_pos == 'pre') and (gate_act in _SCALE_GATES):
            raise ValueError(f"HybridMambaSSM: gate_bias=True with gate_pos='pre' and gate_act={gate_act!r} is a redundant parametrisation (the bias factors into the B/C norm weight); drop gate_bias, or use gate_pos='post' or a non-factorising gate_act")
        if n_hm and (gates_b or gates_c) and (gate_act == 'one_plus_tanh') and (gate_pos != 'pre'):
            raise NotImplementedError("HybridMambaSSM: gate_act='one_plus_tanh' with gate_pos='post' is not implemented (host-side pre-rotation gate only); use gate_pos='pre'")
        if n_hm and gates_b != gates_c:
            raise NotImplementedError('HybridMambaSSM: separate B/C gates (gate_b != gate_c) are not yet implemented on the HM rows; set use_g_b == use_g_c')
        bc_bias = self.bias_gran != 'none' and n_hm > 0
        hm_gated = n_hm > 0 and any(plan.hm_gate)
        gates_b, gates_c = (gates_b and hm_gated, gates_c and hm_gated)
        if (gates_b or gates_c) and gate_pos == 'post' and (not all(plan.hm_gate)) and (gate_act not in _SCALE_GATES):
            raise NotImplementedError(f"HybridMambaSSM: a gate=False row block with gate_pos='post' needs gate_act in {_SCALE_GATES} (act(0) == 1), got {gate_act!r}")
        b_norm_hm, c_norm_hm = (b_norm and n_hm > 0, c_norm and n_hm > 0)
        self._proj_source_explicit = proj_source is not None
        self._sel_request = dict(d_model=d_model, bc_conv=bc_conv, d_conv=d_conv, dt_rank=dt_rank, force_dt_dense=force_dt_dense)
        self.proj_source, d_sel, self.bc_conv = resolve_proj_source('layer_input' if proj_source is None else proj_source, d_inner, d_model, bc_conv, who='HybridMambaSSM', d_conv=d_conv)
        self.d_sel = d_sel
        dt_rank = resolve_dt_rank(dt_rank, d_sel=d_sel, heads=self.H, d_model=d_model, force_dt_dense=force_dt_dense)
        if dt_init not in (None, 'random', 'constant'):
            raise ValueError("dt_init must be None, 'random', or 'constant'")
        if not math.isfinite(dt_scale) or dt_scale <= 0:
            raise ValueError('dt_scale must be finite and positive')
        if dt_init is None and dt_scale != 1.0:
            raise ValueError("dt_scale requires explicit dt_init='random' or 'constant'")
        if not 0 < dt_min <= dt_max or not math.isfinite(dt_max):
            raise ValueError('require 0 < dt_min <= dt_max < infinity')
        if not math.isfinite(dt_init_floor) or dt_init_floor <= 0:
            raise ValueError('dt_init_floor must be finite and positive')
        self.dt_init, self.dt_scale = (dt_init, float(dt_scale))
        self.dt_rank = None if dt_rank is None else int(dt_rank)
        if self.dt_rank is not None:
            if self.dt_rank <= 0:
                raise ValueError(f'dt_rank must be positive, got {dt_rank}')
            self.dt_low = nn.Linear(d_sel, self.dt_rank, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.H, bias=False)
        else:
            self.register_module('dt_low', None)
            self.dt_proj = nn.Linear(d_sel, self.H, bias=False)
        if dt_init is not None:
            std = self.dt_proj.in_features ** (-0.5) * dt_scale
            if dt_init == 'constant':
                nn.init.constant_(self.dt_proj.weight, std)
            else:
                nn.init.uniform_(self.dt_proj.weight, -std, std)
        _lo, _hi = (math.log(dt_min), math.log(dt_max))
        _dt = torch.exp(torch.rand(self.H) * (_hi - _lo) + _lo)
        _dt = _dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        self.dt_bias._no_weight_decay = True
        self.bc_proj = nn.Linear(d_sel, 2 * d_state, bias=False)
        self.register_module('bc_conv1d', CausalDepthwiseConv(2 * d_state, d_conv) if self.bc_conv else None)
        G = self.H if self.bias_gran == 'head_bc' else 1
        G_gate = 1 if gate_pos == 'pre' else self.H
        self.b_norm = nn.RMSNorm(n_hm, eps=norm_eps) if b_norm_hm else None
        self.c_norm = nn.RMSNorm(n_hm, eps=norm_eps) if c_norm_hm else None
        self.beta_b = nn.Parameter(torch.zeros(G, n_hm)) if bc_bias else None
        self.beta_c = nn.Parameter(torch.zeros(G, n_hm)) if bc_bias else None
        _rep = [n - 1 if _rot[n] and n % 2 else n for n in range(n_hm)]

        def _store(law_set):
            rows = [n for n in range(n_hm) if _law[n] in law_set and _rep[n] == n]
            col = {n: i for i, n in enumerate(rows)}
            idx = [col.get(_rep[n], len(rows)) for n in range(n_hm)]
            if idx == list(range(n_hm)):
                return (rows, None)
            return (rows, idx)
        self.has_rot = has_rot
        a_rows, a_idx = _store((LAW_A_FIX, LAW_MU))
        self._register_host_buffer('a_log_idx', a_idx, torch.long)
        ladder, off = ([], 0)
        for f in plan.hm:
            if f.law in (LAW_A_FIX, LAW_MU):
                ladder += list(range(1, (f.rows // 2 if f.rot else f.rows) + 1))
            off += f.rows
        assert len(ladder) == len(a_rows)
        if a_rows:
            A = torch.tensor(ladder, dtype=torch.float32).repeat(self.H, self.P, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
        else:
            self.A_log = None
        if LAW_A_DYN in laws:
            dyn_rows, dyn_idx = _store((LAW_A_DYN,))
            self.a_dyn_proj = nn.Linear(d_sel, len(dyn_rows) if dyn_res == 'n' else self.H, bias=False)
        else:
            dyn_idx, self.a_dyn_proj = (None, None)
        self._register_host_buffer('dyn_idx', dyn_idx if dyn_res == 'n' else None, torch.long)
        if LAW_MU in laws:
            mu_rows, mu_idx = _store((LAW_MU,))
            self.n_mu = len(mu_rows)
            self.mu_proj = nn.Linear(d_sel, self.n_mu if mu_res == 'n' else self.H, bias=False)
            self.mu_gain_logit = nn.Parameter(torch.zeros(self.H, self.P, self.n_mu))
            self.mu_gain_logit._no_weight_decay = True
            self._register_load_state_dict_pre_hook(self._refuse_unbounded_mu)
        else:
            mu_idx, self.n_mu = (None, 0)
            self.mu_proj = None
            self.mu_gain_logit = None
        self._register_host_buffer('mu_idx', mu_idx, torch.long)
        self.learn_theta_freq = bool(learn_theta_freq) and has_rot
        self.use_theta_h = bool(use_theta_h) and has_rot
        self.theta_per_head = bool(theta_per_head) and has_rot
        if has_rot:
            rot_pairs = [k for k, r in enumerate(_rot[0::2]) if r]
            self.n_rot_pairs = len(rot_pairs)
            self._register_host_buffer('rot_pair_idx', rot_pairs, torch.long)
            n_rot = self.n_rot_pairs
            self.theta_proj = nn.Linear(d_sel, (self.H if self.theta_per_head else 1) * n_rot, bias=False)
            if self.learn_theta_freq:
                self.theta_freq = nn.Parameter(torch.full((n_rot,), _THETA_FREQ_INIT))
                self.theta_freq._no_weight_decay = True
            if self.use_theta_h:
                self.theta_h = nn.Parameter(torch.zeros(self.H, n_rot))
                self.theta_h._no_weight_decay = True
                self._register_host_buffer('theta_h_convention', lambda: THETA_H_CONVENTION, torch.int64, persistent=True)
                self._register_load_state_dict_pre_hook(self._migrate_theta_h_convention)
        else:
            self.n_rot_pairs = 0
            self.rot_pair_idx = None
            self.theta_proj = None
        self.use_trap = bool(use_trap) and n_hm > 0
        if self.use_trap:
            self.trap_proj = nn.Linear(d_sel, self.H, bias=True)
            nn.init.zeros_(self.trap_proj.bias)
        else:
            self.trap_proj = None
        self.gate_pair_tied = bool(has_rot)
        _gated = []
        for f, on in zip(plan.hm, plan.hm_gate):
            _gated += [on] * f.rows
        gate_rows = [n for n in range(n_hm) if _gated[n] and _rep[n] == n]
        _col = {n: i for i, n in enumerate(gate_rows)}
        n_gate = len(gate_rows)
        _gidx = [_col[_rep[n]] if _gated[n] else n_gate for n in range(n_hm)]
        self.n_gate_rows = n_gate
        self.gate_masked = not all(_gated)
        self._register_host_buffer('gate_tie_idx', None if _gidx == list(range(n_hm)) else _gidx, torch.long)
        self._register_host_buffer('gate_row_on', _gated if self.gate_masked else None, torch.bool)
        self._register_load_state_dict_pre_hook(self._refuse_per_coordinate_complex_gates)
        self.gB_proj = nn.Linear(d_sel, n_gate, bias=False) if gates_b else None
        self.gC_proj = nn.Linear(d_sel, n_gate, bias=False) if gates_c else None
        for _g in (self.gB_proj, self.gC_proj):
            if _g is not None:
                _init_gate_weight(_g, gate_init)
        self.beta_gB = nn.Parameter(torch.zeros(G_gate, n_gate)) if gates_b and gate_bias else None
        self.beta_gC = nn.Parameter(torch.zeros(G_gate, n_gate)) if gates_c and gate_bias else None
        n3, n3c = (plan.n3, plan.n3_complex)
        if n3 > 0:
            self.m3_a_proj = nn.Linear(d_sel, self.H, bias=True)
            self.m3_trap_proj = nn.Linear(d_sel, self.H, bias=True)
            nn.init.zeros_(self.m3_a_proj.bias)
            nn.init.zeros_(self.m3_trap_proj.bias)
            self.m3_a_proj.bias._no_weight_decay = True
            self.m3_angle_proj = nn.Linear(d_sel, n3c // 2, bias=False) if n3c > 0 else None
            self.m3_B_bias = nn.Parameter(torch.ones(self.H, n3))
            self.m3_C_bias = nn.Parameter(torch.ones(self.H, n3))
            self._register_host_buffer('const_a_log', [0.0] * self.H, torch.float32)
            self.m3_b_norm = nn.RMSNorm(n3, eps=1e-05)
            self.m3_c_norm = nn.RMSNorm(n3, eps=1e-05)
        else:
            self.m3_a_proj = None
            self.m3_trap_proj = None
            self.m3_angle_proj = None
            self.m3_B_bias = None
            self.m3_C_bias = None
            self.m3_b_norm = None
            self.m3_c_norm = None
        for law, n in (('m2', plan.n2),):
            if n > 0:
                a_log = torch.linspace(math.log(0.05), math.log(2.0), self.H)
                p_a = nn.Parameter(a_log)
                p_a._no_weight_decay = True
                setattr(self, f'{law}_A_log', p_a)
                setattr(self, f'{law}_B_bias', nn.Parameter(torch.ones(self.H, n)))
                setattr(self, f'{law}_C_bias', nn.Parameter(torch.ones(self.H, n)))
                setattr(self, f'{law}_b_norm', nn.RMSNorm(n, eps=1e-05))
                setattr(self, f'{law}_c_norm', nn.RMSNorm(n, eps=1e-05))
            else:
                for nm in ('A_log', 'B_bias', 'C_bias', 'b_norm', 'c_norm'):
                    setattr(self, f'{law}_{nm}', None)
        markers = False
        for field, (buf, weight, _legal) in _ACT_MARKERS.items():
            if getattr(self, weight.split('.')[0]) is not None:
                self._register_host_buffer(buf, lambda f=field: ACT_MARKER_CODES[getattr(self, f)], torch.int64, persistent=True)
                markers = True
        if markers:
            self._register_load_state_dict_pre_hook(self._adopt_act_markers)
        self._register_host_buffer('proj_source_code', lambda: PROJ_SOURCE_MARKER_CODES[self.proj_source], torch.int64, persistent=True)
        self._register_load_state_dict_pre_hook(self._adopt_proj_source_marker)
        if d_skip_mode not in ('head', 'channel'):
            raise ValueError("d_skip_mode must be 'head' or 'channel'")
        self.d_skip_mode = d_skip_mode
        self.D = nn.Parameter(torch.ones((self.H,) if d_skip_mode == 'head' else (self.H, self.P)))
        self.D._no_weight_decay = True
        self._k11_skip_in_kernel = False
        self._h3_gate_in_kernel = False
        self.h_init_param = nn.Parameter(torch.zeros(self.H, self.P, self.d_state)) if learnable_h_init else None
        record_constructor_config(self, locals(), constructor=HybridMambaSSM.__init__, overrides=dict(kernel_options=self.kernel_options, bc_conv=self.bc_conv, proj_source=self.proj_source, use_trap=self.use_trap, **spec), exclude=_config_spec.FLAT_KWARGS, execution=('kernel_options', 'backend', 'ckpt', 'm3_kernel', 'mem'))
        if self.mu_proj is None:
            self._resolved_constructor_config['model'].pop('mu_cap')

    def theta_omega(self) -> torch.Tensor:
        if self.learn_theta_freq:
            return math.pi * torch.tanh(self.theta_freq)
        return torch.full((self.n_rot_pairs,), math.pi, device=self.theta_proj.weight.device)

    def _pairs_to_kernel(self, t, dim=-1):
        if t is None:
            return None
        K = self.d_state_hm // 2
        if self.n_rot_pairs == K:
            return t
        dim = dim % t.dim()
        shape = list(t.shape)
        shape[dim] = K
        return t.new_zeros(shape).index_copy(dim, self.rot_pair_idx, t)

    def _theta_feature(self, q):
        if self.theta_per_head:
            Bq, K = (q.shape[0], self.n_rot_pairs)
            f = self._theta_feature_shared(q.reshape(Bq * self.H, K, *q.shape[2:]))
            return f.reshape(Bq, self.H, K, *q.shape[2:])
        return self._theta_feature_shared(q)

    def _theta_feature_shared(self, q):
        om = self.theta_omega()
        if q.dim() == 2:
            return om.to(q.dtype) * torch.tanh(q)
        if not (q.is_cuda and q.dtype != torch.float64):
            return om.to(q.dtype)[None, :, None] * torch.tanh(q)
        from ..kernels.ssd_mimo.m3_epilogue import m3_phase_increments
        Bsz, K, L = q.shape
        Kp = 1 << max(0, K - 1).bit_length()
        ang = m3_phase_increments(q.float().transpose(1, 2).unsqueeze(2), q.new_ones(Bsz, 1, L, dtype=torch.float32), Kp, 1.0)
        ang = ang[..., :K].squeeze(2).transpose(1, 2)
        w = (om * (1.0 / math.pi)).to(ang.dtype)
        return (ang * w[None, :, None]).to(q.dtype)

    def _theta_h_kernel(self):
        return self._pairs_to_kernel(-self.theta_h) if self.use_theta_h else None

    def _migrate_theta_h_convention(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'theta_h_convention'
        code = state_dict.get(key)
        if code is None:
            th = prefix + 'theta_h'
            if th in state_dict:
                state_dict[th] = -state_dict[th]
                warnings.warn("HybridMambaSSM: state dict without 'theta_h_convention' (written before 2026-09-23): theta_h is loaded NEGATED into the unified read-offset convention; the model function is unchanged", UserWarning, stacklevel=3)
            state_dict[key] = self.theta_h_convention.detach().clone()
            return
        if int(code) != THETA_H_CONVENTION:
            raise ValueError(f'{key}={int(code)} is not a known read-offset convention (this version reads {THETA_H_CONVENTION}; an absent key means the pre-2026-09-23 MH convention and is migrated)')

    @property
    def a_act(self) -> str:
        return self._a_act

    @a_act.setter
    def a_act(self, name) -> None:
        if name not in A_ACTS:
            raise ValueError(f'a_act must be one of {A_ACTS}, got {name!r}')
        self._a_act = name
        self._sync_act_record('a_act', name)

    @staticmethod
    def _canonical_dyn_act(value) -> str | None:
        if isinstance(value, str):
            name = 'raw' if value == 'identity' else value
            return name if name in DYN_ACTS else None
        if isinstance(value, bool):
            return None
        try:
            return _DYN_CODE_NAMES.get(int(value))
        except (TypeError, ValueError):
            return None

    @property
    def dyn_act(self) -> str:
        return self._dyn_act_sel

    @dyn_act.setter
    def dyn_act(self, value) -> None:
        name = self._canonical_dyn_act(value)
        if name is None:
            raise ValueError(f'dyn_act must be one of {DYN_ACTS} (or a layout.DYN_* code), got {value!r}')
        self._dyn_act_sel = name
        self._dyn_act = DYN_RAW if name == 'heavy_tail' else _DYN_NAMES[name]
        self._sync_act_record('dyn_act', name)

    def _sync_act_record(self, field: str, name: str) -> None:
        buf = self._buffers.get(_ACT_MARKERS[field][0])
        if buf is not None:
            with torch.no_grad():
                buf.fill_(ACT_MARKER_CODES[name])
        rec = self.__dict__.get('_resolved_constructor_config')
        if rec is not None:
            old = rec['model'].get(field)
            same = old == name if field == 'a_act' else self._canonical_dyn_act(old) == name
            if not same:
                rec['model'][field] = name

    def _adopt_act_markers(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for field, (buf, weight, legal) in _ACT_MARKERS.items():
            if buf not in self._buffers:
                continue
            key = prefix + buf
            code = state_dict.get(key)
            current = getattr(self, field)
            if code is None:
                if prefix + weight not in state_dict:
                    continue
                name = 'softplus' if current == 'heavy_tail' else current
                if name != current:
                    warnings.warn(f"HybridMambaSSM: state dict without '{buf}' (written before 2026-09-23, when HM rows had no heavy_tail): {field} 'heavy_tail' -> 'softplus', the activation its weights were trained with", UserWarning, stacklevel=3)
                    setattr(self, field, name)
                state_dict[key] = self._buffers[buf].detach().clone()
                continue
            name = _ACT_MARKER_NAMES.get(int(code))
            if name not in legal:
                raise ValueError(f'{key}={int(code)} is not a known {field} marker (this version reads {[ACT_MARKER_CODES[n] for n in legal]} = {list(legal)}; an absent key means a pre-2026-09-23 softplus checkpoint)')
            if name != current:
                warnings.warn(f'HybridMambaSSM: the state dict declares {field}={name!r} (the activation its weights were trained with); the module was built with {current!r} and adopts {name!r}', UserWarning, stacklevel=3)
                setattr(self, field, name)

    def _adopt_proj_source_marker(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'proj_source_code'
        code = state_dict.get(key)
        if code is not None:
            name = _PROJ_SOURCE_MARKER_NAMES.get(int(code))
            if name is None:
                raise ValueError(f'{key}={int(code)} is not a known proj_source marker (this version reads {PROJ_SOURCE_MARKER_CODES}; an absent key means a checkpoint written before 2026-09-23)')
            if name != self.proj_source:
                self._adopt_or_refuse_proj_source(name, 'the state dict declares it', legacy=False)
            return
        if prefix + 'bc_proj.weight' not in state_dict:
            return
        name, why = self._infer_legacy_proj_source(state_dict, prefix)
        if name is None:
            return
        if name != self.proj_source:
            self._adopt_or_refuse_proj_source(name, why, legacy=True)
        state_dict[key] = self.proj_source_code.detach().clone()

    def _infer_legacy_proj_source(self, state_dict, prefix):
        if prefix + 'bc_conv1d.conv.weight' in state_dict:
            return ('layer_input', 'it carries a B/C conv, a layer_input feature')
        width = state_dict[prefix + 'bc_proj.weight'].shape[-1]
        d_model = self._sel_request['d_model']
        if d_model is not None and d_model != self.d_inner:
            if width == self.d_inner:
                return ('main_branch', f'token-map input width {width} = d_inner')
            if width == d_model:
                return ('layer_input', f'token-map input width {width} = d_model')
            return (None, None)
        if self._proj_source_explicit:
            return (self.proj_source, 'explicit, the widths cannot tell')
        return ('main_branch', 'd_inner == d_model, so the widths cannot tell: the pre-2026-09-23 default')

    def _adopt_or_refuse_proj_source(self, name: str, why: str, *, legacy: bool) -> None:
        current = self.proj_source
        if self._proj_source_explicit:
            raise ValueError(f"HybridMambaSSM: the state dict's token maps were trained on proj_source={name!r} ({why}), but this module was built with proj_source={current!r} explicitly; rebuild it with proj_source={name!r}, or leave proj_source unset to adopt the checkpoint's source")
        try:
            self._switch_proj_source(name)
        except ValueError as err:
            raise ValueError(f"HybridMambaSSM: cannot adopt the state dict's proj_source={name!r} ({why}): {err}") from err
        if legacy:
            msg = f"HybridMambaSSM: state dict without 'proj_source_code' (written before 2026-09-23, when the default was 'main_branch'): proj_source {current!r} -> {name!r}, the source its token maps were trained on ({why})"
        else:
            msg = f'HybridMambaSSM: the state dict declares proj_source={name!r} (the source its token maps were trained on); the module (proj_source left to the default) read {current!r} and adopts {name!r}'
        warnings.warn(msg, UserWarning, stacklevel=4)

    def _switch_proj_source(self, name: str) -> None:
        req = self._sel_request
        source, d_sel, bc_conv = resolve_proj_source(name, self.d_inner, req['d_model'], req['bc_conv'], who='HybridMambaSSM', d_conv=req['d_conv'])
        rank = resolve_dt_rank(req['dt_rank'], d_sel=d_sel, heads=self.H, d_model=req['d_model'], force_dt_dense=req['force_dt_dense'])
        rank = None if rank is None else int(rank)
        ref = self.bc_proj.weight

        def dt_init(lin):
            if self.dt_init is not None:
                std = lin.in_features ** (-0.5) * self.dt_scale
                if self.dt_init == 'constant':
                    nn.init.constant_(lin.weight, std)
                else:
                    nn.init.uniform_(lin.weight, -std, std)

        def zero_bias(lin):
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        def fresh(old, in_f, out_f, init=None):
            bias = old is not None and old.bias is not None
            if old is not None and (old.in_features, old.out_features) == (in_f, out_f):
                return old
            with torch.random.fork_rng(devices=[]):
                lin = nn.Linear(in_f, out_f, bias=bias)
                if init is not None:
                    init(lin)
            src = old if old is not None else self.dt_proj
            lin = lin.to(device=src.weight.device, dtype=src.weight.dtype)
            for new_p, old_p in ((lin.weight, src.weight), (lin.bias, src.bias)):
                if new_p is not None and old_p is not None:
                    new_p.requires_grad_(old_p.requires_grad)
                    new_p.__dict__.update(old_p.__dict__)
            return lin
        if rank is None:
            self.dt_low = None
            self.dt_proj = fresh(self.dt_proj, d_sel, self.H, dt_init)
        else:
            self.dt_low = fresh(self.dt_low, d_sel, rank)
            self.dt_proj = fresh(self.dt_proj, rank, self.H, dt_init)
        g_init = (lambda lin: _init_gate_weight(lin, self.gate_init)) if self.gate_init != 'kaiming_uniform' else None
        inits = {'trap_proj': zero_bias, 'm3_a_proj': zero_bias, 'm3_trap_proj': zero_bias, 'gB_proj': g_init, 'gC_proj': g_init}
        for attr in _SEL_MAP_ATTRS:
            old = getattr(self, attr)
            if old is not None:
                setattr(self, attr, fresh(old, d_sel, old.out_features, inits.get(attr)))
        if not bc_conv:
            self.bc_conv1d = None
        elif self.bc_conv1d is None:
            with torch.random.fork_rng(devices=[]):
                conv = CausalDepthwiseConv(2 * self.d_state, req['d_conv'])
            self.bc_conv1d = conv.to(device=ref.device, dtype=ref.dtype).requires_grad_(ref.requires_grad)
        self.proj_source, self.d_sel, self.bc_conv = (source, d_sel, bc_conv)
        self.dt_rank = rank
        with torch.no_grad():
            self.proj_source_code.fill_(PROJ_SOURCE_MARKER_CODES[source])
        rec = self.__dict__.get('_resolved_constructor_config')
        if rec is not None:
            rec['model'].update(proj_source=source, bc_conv=bc_conv, dt_rank=rank)

    def _register_host_buffer(self, name, values, dtype, persistent=False):
        if values is None:
            self.register_buffer(name, None, persistent=persistent)
            return
        self._host_buffers[name] = values if callable(values) else list(values)
        self.register_buffer(name, self._host_value(name, dtype), persistent=persistent)

    def _host_value(self, name, dtype, device=None):
        v = self._host_buffers[name]
        return torch.tensor(v() if callable(v) else v, dtype=dtype, device=device)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        with torch.no_grad():
            for name in getattr(self, '_host_buffers', {}):
                buf = getattr(self, name, None)
                if buf is not None and buf.device.type != 'meta':
                    buf.copy_(self._host_value(name, buf.dtype, 'cpu'))
        return out

    @staticmethod
    def _rows_to_kernel(t, idx, dim=-1):
        if t is None or idx is None:
            return t
        dim = dim % t.dim()
        pad = [0, 0] * (t.dim() - 1 - dim) + [0, 1]
        return F.pad(t, pad).index_select(dim, idx)

    def _A_log_kernel(self):
        if self.A_log is None:
            return self.dt_bias.new_zeros(self.H, self.P, self.d_state_hm)
        return self._rows_to_kernel(self.A_log, self.a_log_idx, -1)

    def _mu_gain_full(self):
        return mu_gain_bound(self.mu_cap) * torch.sigmoid(self.mu_gain_logit)

    def _bound_mu_map(self, maps):
        if 'mu' in maps:
            maps['mu'] = torch.tanh(maps['mu'])
        return maps

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
            if t.shape[axis] != own.shape[axis] and t.shape[axis] == self.d_state_hm:
                raise RuntimeError(f'{prefix}{name}: this checkpoint gates the complex rows PER COORDINATE ({t.shape[axis]} rows); gates on complex row pairs are pair-tied since 2026-09-26 (one factor per pair, {own.shape[axis]} rows).  There is no migration -- retrain.')

    def _gate_maps(self, maps):
        idx = self.gate_tie_idx
        if idx is not None:
            for k in ('g_B', 'g_C'):
                g = maps.get(k)
                if g is not None:
                    if self.gate_masked:
                        g = torch.cat([g, g.new_zeros(g[:, :1].shape)], 1)
                    maps[k] = g.index_select(1, idx)
        return maps

    def _gate_bias(self, beta):
        if beta is None or self.gate_tie_idx is None:
            return beta
        if self.gate_masked:
            beta = torch.cat([beta, beta.new_zeros(beta[..., :1].shape)], -1)
        return beta.index_select(-1, self.gate_tie_idx)

    @staticmethod
    def _refuse_unbounded_mu(state_dict, prefix, *_args):
        if prefix + 'mu_gain' in state_dict:
            raise RuntimeError(f'{prefix}mu_gain: this checkpoint carries the retired UNBOUNDED mu law (log a += mu_gain (mu_t - mu_{{t-1}}), mu = mu_proj sel).  The mu law is bounded since 2026-09-25 (mu = tanh(mu_proj sel), gain G sigmoid(mu_gain_logit)); there is no migration -- retrain.')

    def _mu_gain_kernel(self):
        if self.mu_proj is None:
            return None
        return self._rows_to_kernel(self._mu_gain_full(), self.mu_idx, -1)

    def _dyn_rate(self, a_dyn: torch.Tensor) -> torch.Tensor:
        if self.dyn_act == 'heavy_tail':
            return _a_mag_tail(a_dyn, 0.0, False, 'heavy_tail')
        return dyn_rate(a_dyn, self._dyn_act, 0.0)

    def _dyn_scan_input(self, a_dyn, eng: str, x_dtype: torch.dtype):
        if a_dyn is None or self.dyn_act != 'heavy_tail':
            return (a_dyn, self._dyn_act)
        wd = torch.float64 if x_dtype == torch.float64 and eng != 'cuda' else torch.float32
        if 'amag' in hm_held_regen() and torch.is_grad_enabled() and a_dyn.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return (checkpoint(lambda a: _a_mag_tail(a.to(wd), 0.0, False, 'heavy_tail'), a_dyn, use_reentrant=False), DYN_RAW)
        return (_a_mag_tail(a_dyn.to(wd), 0.0, False, 'heavy_tail'), DYN_RAW)
    _KERNEL_ROUTES = ('mh_scan', 'ssd_mimo_scan', 'upstream_siso')

    def _route_begin(self, on_cuda: bool | None=None) -> None:
        self._routes = ()
        self._route_note = None
        self._refusal_note = None
        self._wants_kernel = self.backend == 'cuda' or (self.backend == 'auto' and bool(on_cuda))

    def _route(self, block: str, name: str) -> None:
        self._routes = getattr(self, '_routes', ()) + ((block, name),)
        if name not in self._KERNEL_ROUTES:
            note = f'{block}: {name}'
            prev = getattr(self, '_route_note', None)
            self._route_note = note if prev is None else f'{prev}; {note}'
            if getattr(self, '_wants_kernel', False):
                prev = getattr(self, '_refusal_note', None)
                self._refusal_note = note if prev is None else f'{prev}; {note}'

    def last_route(self) -> str | None:
        return getattr(self, '_route_note', None)

    def last_refusal(self) -> str | None:
        return getattr(self, '_refusal_note', None)

    def last_routes(self) -> tuple:
        return tuple(getattr(self, '_routes', ()))

    def hidden_state_shape(self):
        return (self.H, self.P, self.d_state)

    def _base_initial_state(self):
        if self.h_init_param is None:
            return None
        return self.h_init_param

    def _init_state(self, batch_size: int, device=None, dtype: torch.dtype=torch.float32) -> State:
        dev = device or next(self.parameters()).device
        state = State(hidden_state=self.initial_state(batch_size).to(device=dev, dtype=dtype))
        phase_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        n_hm, n3 = (self.d_state_hm, self.d_state_m3)
        if n_hm > 0:
            if self.has_rot:
                state['theta'] = torch.zeros(batch_size, self.H, self.n_rot_pairs, device=dev, dtype=phase_dtype)
            if self.mu_proj is not None:
                mu_width = self.mu_proj.out_features
                state['mu'] = torch.zeros(batch_size, mu_width, device=dev, dtype=dtype)
            if self.trap_proj is not None:
                state['write_prev'] = torch.zeros(batch_size, self.H, self.P, n_hm, device=dev, dtype=dtype)
        if self.bc_conv1d is not None:
            state['bc_conv_state'] = self.bc_conv1d.init_state(batch_size, device=dev, dtype=dtype)
        if n3 > 0:
            state['c3_kp'] = torch.zeros(batch_size, self.H, n3, device=dev, dtype=dtype)
            state['c3_vp'] = torch.zeros(batch_size, self.H, self.P, device=dev, dtype=dtype)
            if self.m3_angle_proj is not None:
                state['c3_phase'] = torch.zeros(batch_size, self.H, self.m3_angle_proj.out_features, device=dev, dtype=phase_dtype)
        return state

    def step(self, x_t: torch.Tensor, state=None, sel_x: torch.Tensor | None=None, tok_raw: torch.Tensor | None=None):
        Bsz = x_t.shape[0]
        state = self.resolve_state(state, batch_size=Bsz, device=x_t.device, dtype=x_t.dtype)
        if self.proj_source == 'layer_input':
            if sel_x is None:
                raise ValueError("HybridMambaSSM(proj_source='layer_input') needs sel_x (the block input, width d_model) in step(); the mixer passes it -- or build the core with proj_source='main_branch'")
            u_t = sel_x
        else:
            u_t = x_t
        if u_t.shape[-1] != self.d_sel:
            raise RuntimeError(f'selection input is width {u_t.shape[-1]}, expected {self.d_sel} (proj_source={self.proj_source!r})')
        n_hm, n3 = (self.d_state_hm, self.d_state_m3)
        maps = self._gate_maps(self._bound_mu_map({k: v.squeeze(-1) for k, v in (self._sel_slice(tok_raw.unsqueeze(-1)) if tok_raw is not None else self._sel_project(u_t.unsqueeze(-1))).items()}))
        dt_lin = maps['dt'] if self.dt_low is None else self.dt_proj(maps['dt'])
        dt = F.softplus(dt_lin + self.dt_bias)
        bc = maps['bc']
        new_state = State(state)
        next_hidden = []
        if self.bc_conv1d is not None:
            bc, bc_conv_state = self.bc_conv1d.step(bc, state['bc_conv_state'])
            new_state['bc_conv_state'] = bc_conv_state
        Bt_all, Ct_all = bc.split([self.d_state, self.d_state], dim=-1)
        y = torch.zeros(Bsz, self.d_inner, device=x_t.device, dtype=x_t.dtype)
        if n_hm > 0:
            y_hm, new_state = self._hm_step(maps, x_t, dt, Bt_all[..., :n_hm], Ct_all[..., :n_hm], state, new_state)
            next_hidden.append(new_state.pop('_hidden_update'))
            y = y + y_hm.to(y.dtype)
        o3, o2 = self._flat_offsets()
        if n3 > 0:
            y_c3, new_state = self._m3_step(maps, x_t, dt, Bt_all[..., o3:o2], Ct_all[..., o3:o2], state, new_state)
            next_hidden.append(new_state.pop('_hidden_update'))
            y = y + y_c3.to(y.dtype)
        if self.d_state_m2 > 0:
            y_f, new_state = self._flat_step('m2', x_t, dt, Bt_all[..., o2:], Ct_all[..., o2:], state, new_state)
            next_hidden.append(new_state.pop('_hidden_update'))
            y = y + y_f.to(y.dtype)
        y = y + x_t * self._skip_channels().to(x_t.dtype)
        new_state['hidden_state'] = next_hidden[0] if len(next_hidden) == 1 else torch.cat(next_hidden, dim=-1)
        self._final_state = new_state
        self.h_final = new_state['hidden_state']
        return (y, new_state)

    def _hm_step(self, maps, x_t, dt, Bt, Ct, state, new_state):
        H, P, n_hm = (self.H, self.P, self.d_state_hm)
        Bsz = x_t.shape[0]
        if self.b_norm is not None:
            Bt = self.b_norm(Bt)
        if self.c_norm is not None:
            Ct = self.c_norm(Ct)
        A_log = self._A_log_kernel()
        c1 = -A_log.exp()
        row_law = self.row_law
        is_dyn = row_law == LAW_A_DYN
        is_mu = row_law == LAW_MU
        s1 = dt[:, :, None].expand(Bsz, H, n_hm).clone()
        if self.a_dyn_proj is not None:
            a_dyn_t = maps['a_dyn']
            rate = self._dyn_rate(a_dyn_t)
            if self.dyn_res == 'n':
                rate_b = self._rows_to_kernel(rate, self.dyn_idx, 1)[:, None, :].expand(Bsz, H, n_hm)
            else:
                rate_b = rate[:, :, None].expand(Bsz, H, n_hm)
            s1 = torch.where(is_dyn.view(1, 1, n_hm), dt[:, :, None] * rate_b, s1)
        c2 = torch.zeros_like(c1)
        s2 = torch.zeros(Bsz, H, n_hm, device=x_t.device, dtype=c1.dtype)
        mu_t = None
        if self.mu_proj is not None:
            mu_gain = self._mu_gain_kernel()
            c2 = torch.where(is_mu.view(1, 1, n_hm), mu_gain, torch.zeros_like(mu_gain))
            mu_t = maps['mu']
            mu_prev = state.get('mu')
            if mu_prev is None:
                mu_prev = torch.zeros_like(mu_t)
            if self.mu_res == 'n':
                mu_b = self._rows_to_kernel(mu_t, self.mu_idx, 1)[:, None, :].expand(Bsz, H, n_hm)
                prev_b = self._rows_to_kernel(mu_prev, self.mu_idx, 1)[:, None, :].expand(Bsz, H, n_hm)
            else:
                mu_b = mu_t[:, :, None].expand(Bsz, H, n_hm)
                prev_b = mu_prev[:, :, None].expand(Bsz, H, n_hm)
            s2 = torch.where(is_mu.view(1, 1, n_hm), mu_b - prev_b, s2)
        log_a = c1[None] * s1[:, :, None, :] + c2[None] * s2[:, :, None, :]
        a_bar = torch.exp(log_a)
        pre = self.gate_pos == 'pre'
        if pre and self.gB_proj is not None:
            Bt = self._pre_gate(Bt[..., None], maps['g_B'][..., None], self._gate_bias(self.beta_gB))[..., 0]
            Ct = self._pre_gate(Ct[..., None], maps['g_C'][..., None], self._gate_bias(self.beta_gC))[..., 0]
        Bh = Bt[:, None, :].expand(Bsz, H, n_hm).clone()
        Ch = Ct[:, None, :].expand(Bsz, H, n_hm).clone()
        bb = _expand_group(self.beta_b, H)
        cc = _expand_group(self.beta_c, H)
        if bb is not None:
            Bh = Bh + bb[None]
        if cc is not None:
            Ch = Ch + cc[None]
        theta_t = None
        if self.has_rot:
            feat = self._theta_feature(maps['theta'].to(dt.dtype))
            theta_t = state['theta'] + dt[:, :, None] * (feat if self.theta_per_head else feat[:, None, :])
            new_state['theta'] = theta_t
            ang_b = self._pairs_to_kernel(theta_t)
            ang_c = ang_b if not self.use_theta_h else ang_b + self._theta_h_kernel()[None].to(theta_t.dtype)

            def _rot(v, ang):
                cos_t, sin_t = (torch.cos(ang), torch.sin(ang))
                vp = v.view(Bsz, H, n_hm // 2, 2)
                e, o = (vp[..., 0], vp[..., 1])
                rotated = torch.stack([e * cos_t - o * sin_t, e * sin_t + o * cos_t], dim=-1).reshape(Bsz, H, n_hm)
                rot_mask = self.row_rot.view(1, 1, n_hm)
                return torch.where(rot_mask, rotated, v)
            Bh, Ch = (_rot(Bh, ang_b), _rot(Ch, ang_c))
        if self.gB_proj is not None and (not pre):
            gv = maps['g_B'][:, None, :].expand(Bsz, H, n_hm).clone()
            bgB = _expand_group(self._gate_bias(self.beta_gB), H)
            if bgB is not None:
                gv = gv + bgB[None]
            Bh = Bh * gate_act(gv, self._gate_act)
        if self.gC_proj is not None and (not pre):
            gv = maps['g_C'][:, None, :].expand(Bsz, H, n_hm).clone()
            bgC = _expand_group(self._gate_bias(self.beta_gC), H)
            if bgC is not None:
                gv = gv + bgC[None]
            Ch = Ch * gate_act(gv, self._gate_act)
        h_old = state['hidden_state'][..., :self.d_state_hm]
        x_hp = x_t.view(Bsz, H, P)
        W_t = x_hp.unsqueeze(-1) * Bh[:, :, None, :]
        if self.trap_proj is not None:
            lam = sigmoid_exact(maps['trap'].to(W_t.dtype))[:, :, None, None]
            W_prev = state.get('write_prev')
            if W_prev is None:
                W_prev = torch.zeros_like(W_t)
            drive = dt[:, :, None, None] * (lam * W_t + (1.0 - lam) * a_bar * W_prev)
            new_state['write_prev'] = W_t
        else:
            drive = dt[:, :, None, None] * W_t
        h_new = a_bar * h_old + drive
        y_t = (Ch[:, :, None, :] * h_new).sum(-1)
        new_state['_hidden_update'] = h_new
        if mu_t is not None:
            new_state['mu'] = mu_t
        return (y_t.reshape(Bsz, self.d_inner), new_state)

    def _m3_rate(self, a_lin: torch.Tensor) -> torch.Tensor:
        return _a_mag_tail(a_lin, self.a_floor, False, self.a_act)

    def _m3_step(self, maps, x_t, dt, Bt, Ct, state, new_state):
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        Bsz, H, P, n3 = (x_t.shape[0], self.H, self.P, self.d_state_m3)
        dt = dt.to(wd)
        rate = self._m3_rate(maps['m3_a'].to(wd))
        alpha = torch.exp(-rate * dt)[..., None, None]
        lam = sigmoid_exact(maps['m3_trap'].to(wd))[..., None, None]
        d = dt[..., None, None]
        k = self.m3_b_norm(Bt).to(wd)[:, None] + self.m3_B_bias.to(wd)
        q = self.m3_c_norm(Ct).to(wd)[:, None] + self.m3_C_bias.to(wd)
        if self.m3_angle_proj is not None:
            increment = torch.tanh(maps['m3_ang'].to(wd))[:, None] * math.pi * dt[..., None]
            phase = state['c3_phase'] + increment
            full = F.pad(phase, (0, n3 // 2 - phase.shape[-1]))
            cos, sin = (full.cos(), full.sin())

            def rotate(v):
                e, o = (v[..., 0::2], v[..., 1::2])
                return torch.stack((e * cos - o * sin, e * sin + o * cos), -1).flatten(-2)
            k, q = (rotate(k), rotate(q))
            new_state['c3_phase'] = phase
        v = x_t.reshape(Bsz, H, P).to(wd)
        start = self.d_state_hm
        old = state['hidden_state'][..., start:start + n3].to(wd)
        previous = state['c3_vp'].to(wd)[..., None] * state['c3_kp'].to(wd)[:, :, None]
        hidden = alpha * old + alpha * d * (1 - lam) * previous + lam * d * v[..., None] * k[:, :, None]
        new_state['_hidden_update'] = hidden.to(state['hidden_state'].dtype)
        new_state['c3_kp'] = k.to(state['c3_kp'].dtype)
        new_state['c3_vp'] = v.to(state['c3_vp'].dtype)
        y = (hidden * q[:, :, None]).sum(-1)
        return (y.reshape(Bsz, self.d_inner).to(x_t.dtype), new_state)
    _SEL_ORDER = ('dt', 'bc', 'a_dyn', 'mu', 'theta', 'trap', 'g_B', 'g_C', 'm3_a', 'm3_trap', 'm3_ang')

    def _sel_modules(self):
        m = {'dt': self.dt_low if self.dt_low is not None else self.dt_proj, 'bc': self.bc_proj, 'a_dyn': self.a_dyn_proj, 'mu': self.mu_proj, 'theta': self.theta_proj, 'trap': self.trap_proj, 'g_B': self.gB_proj, 'g_C': self.gC_proj, 'm3_a': self.m3_a_proj, 'm3_trap': self.m3_trap_proj, 'm3_ang': self.m3_angle_proj}
        return [(k, m[k]) for k in self._SEL_ORDER if m[k] is not None]

    def _sel_project(self, u_cf):
        return self._sel_slice(matmul_channel_major(self.tok_proj_weight(), u_cf.transpose(1, 2)))

    def tok_proj_weight(self) -> torch.Tensor:
        mods = self._sel_modules()
        if len(mods) == 1:
            return mods[0][1].weight
        return torch.cat([m.weight for _, m in mods], 0)

    @property
    def tok_width(self) -> int:
        return sum((m.weight.shape[0] for _, m in self._sel_modules()))

    def _sel_slice(self, proj):
        out, off = ({}, 0)
        for key, m in self._sel_modules():
            rows = m.weight.shape[0]
            sl = proj[:, off:off + rows]
            if m.bias is not None:
                sl = sl + m.bias.to(sl.dtype).view(1, rows, *[1] * (proj.dim() - 2))
            out[key] = sl
            off += rows
        return out

    def _sel_input_cf(self, x_cf, sel_x):
        if self.proj_source == 'layer_input':
            if sel_x is None:
                raise ValueError("HybridMambaSSM(proj_source='layer_input') needs sel_x (the block input, width d_model); the mixer passes it -- or build the core with proj_source='main_branch'")
            u_cf = sel_x.transpose(1, 2)
        else:
            u_cf = x_cf
        if u_cf.shape[1] != self.d_sel:
            raise RuntimeError(f'selection input is width {u_cf.shape[1]}, expected {self.d_sel} (proj_source={self.proj_source!r})')
        return u_cf

    def _scan_cf(self, x_cf, h_init, sel_x, want_state, tok_raw=None, z_cf=None, complete_state=None):
        if tok_raw is not None:
            maps = self._sel_slice(tok_raw.transpose(1, 2))
        elif self.proj_source == 'main_branch':
            x_cf, proj = matmul_channel_major_tee(self._sel_input_cf(x_cf, sel_x), self.tok_proj_weight())
            maps = self._sel_slice(proj)
        else:
            maps = self._sel_project(self._sel_input_cf(x_cf, sel_x))
        maps = self._gate_maps(self._bound_mu_map(maps))
        n_hm, n3 = (self.d_state_hm, self.d_state_m3)
        n_flat = self.fracs.n_flat
        if self.dt_low is None:
            dt_lin = maps['dt']
        else:
            dt_lin = matmul_channel_major(self.dt_proj.weight, maps['dt'].transpose(1, 2))
            if 'dtlin' in hm_held_regen() and torch.is_grad_enabled() and dt_lin.requires_grad and self._delta_prep_in_kernel(x_cf, n_flat, n_hm):
                _acd = (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
                _w, _lo = (self.dt_proj.weight, maps['dt'].transpose(1, 2))

                def _dtlin_regen(_w=_w, _lo=_lo, _acd=_acd):
                    with torch.autocast('cuda', dtype=_acd[1], enabled=_acd[0]):
                        return matmul_channel_major(_w, _lo)
                _regen.register(dt_lin, _dtlin_regen)
        bc = maps['bc']
        bc_raw = bc
        if self.bc_conv1d is not None:
            prefix = complete_state['bc_conv_state'][..., 1:].to(bc.dtype) if complete_state is not None else None
            if prefix is not None:
                conv = self.bc_conv1d.conv
                bc = F.silu(F.conv1d(torch.cat((prefix, bc), dim=-1), conv.weight.to(bc.dtype), conv.bias.to(bc.dtype), groups=bc.shape[1]))
            else:
                bc = causal_depthwise_conv1d_cl(bc, self.bc_conv1d.conv.weight, self.bc_conv1d.conv.bias, self.bc_conv1d.d_conv, activation='silu')
        Bt_all, Ct_all = (bc[:, :self.d_state], bc[:, self.d_state:])
        dt = bc_normed = None
        dt_in_kernel = self._delta_prep_in_kernel(x_cf, n_flat, n_hm)
        if dt_in_kernel:
            dt = dt_lin
        elif n_flat == 0 and n_hm > 0:
            got = self._fused_hm_tails(dt_lin, Bt_all[:, :n_hm], Ct_all[:, :n_hm])
            if got is not None:
                dt, Bhm, Chm = got
                Bt_all, Ct_all, bc_normed = (Bhm, Chm, True)
        if dt is None:
            dt = F.softplus(dt_lin + self.dt_bias.unsqueeze(-1))
        elif 'delta' in hm_held_regen() and (not dt_in_kernel) and (self.b_norm is None) and dt.is_cuda and (dt is not dt_lin):
            from ..layers.tails import dt_softplus_cf as _dtsp
            _acd = (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
            _bias = self.dt_bias

            def _dt_regen(dt_lin=dt_lin, _bias=_bias, _acd=_acd):
                with torch.autocast('cuda', dtype=_acd[1], enabled=_acd[0]):
                    return _dtsp(dt_lin, _bias)
            _regen.register(dt, _dt_regen)
        h_final = None
        in_kernel = n_hm > 0 and self._d_skip_in_kernel(x_cf)
        self._k11_skip_in_kernel = in_kernel
        gate_in_kernel = z_cf is not None and self._z_gate_in_kernel(x_cf, n_flat, n_hm, in_kernel, z_cf)
        self._h3_gate_in_kernel = gate_in_kernel
        if n_hm > 0:
            y, h_final = self._hm_half_cf(maps, x_cf, dt, Bt_all[:, :n_hm], Ct_all[:, :n_hm], h_init, want_state=want_state, d_skip=self.D if in_kernel else None, bc_normed=bool(bc_normed), dt_raw=dt_in_kernel, z=z_cf if gate_in_kernel else None, complete_state=complete_state)
        else:
            y = torch.zeros_like(x_cf)
        o3, o2 = self._flat_offsets()
        flat_finals = []
        if complete_state is not None:
            for law, start, end in (('m3', o3, o2), ('m2', o2, self.d_state)):
                if end > start:
                    yf, hf = self._flat_state_scan(law, maps, x_cf, dt, Bt_all[:, start:end], Ct_all[:, start:end], complete_state, start, end)
                    y = y + yf.to(y.dtype)
                    flat_finals.append(hf)
            pieces = ([h_final] if n_hm else []) + flat_finals
            h_final = torch.cat(pieces, -1)
        if n3 > 0 and complete_state is None:
            y = y + self._m3_half_cf(maps, x_cf, dt, Bt_all[:, o3:o2], Ct_all[:, o3:o2]).to(y.dtype)
        if self.d_state_m2 > 0 and complete_state is None:
            y = y + self._flat_half_cf('m2', x_cf, dt, Bt_all[:, o2:], Ct_all[:, o2:]).to(y.dtype)
        if complete_state is not None:
            before_last = State(complete_state)
            if 'theta' in before_last:
                wd = before_last['theta'].dtype
                clock = F.softplus(dt_lin.to(wd) + self.dt_bias.to(wd)[None, :, None])
                feature = self._theta_feature(maps['theta'].to(wd))
                before_last['theta'] = before_last['theta'] + (torch.einsum('bhl,bhkl->bhk', clock[..., :-1], feature[..., :-1]) if self.theta_per_head else phase_bmm(clock[..., :-1], feature[..., :-1].transpose(1, 2)))
            if 'mu' in before_last and x_cf.shape[-1] > 1:
                before_last['mu'] = maps['mu'][..., -2]
            if 'c3_phase' in before_last:
                wd = before_last['c3_phase'].dtype
                feature = torch.tanh(maps['m3_ang'].to(wd)) * math.pi
                terminal = phase_bmm(dt.to(wd)[..., :-1], feature[..., :-1].transpose(1, 2))
                before_last['c3_phase'] = before_last['c3_phase'] + terminal
            if self.bc_conv1d is not None:
                cache = complete_state['bc_conv_state']
                history = torch.cat((cache, bc_raw.to(cache.dtype)), -1)
                kc = self.bc_conv1d.d_conv
                before_last['bc_conv_state'] = history[..., -(kc + 1):-1].contiguous()
            _, final = self.step(x_cf[..., -1], before_last, sel_x=None if sel_x is None else sel_x[:, -1], tok_raw=None if tok_raw is None else tok_raw[:, -1])
            final['hidden_state'] = h_final.to(complete_state['hidden_state'].dtype)
            return (y, h_final, final)
        return (y, h_final)

    def _z_gate_in_kernel(self, x_cf, n_flat: int, n_hm: int, skip_in_kernel: bool, z_cf) -> bool:
        if not (n_hm > 0 and n_flat == 0 and skip_in_kernel):
            return False
        if not hm_glue_on('zk'):
            return False
        if z_cf.shape != x_cf.shape or z_cf.dtype != x_cf.dtype or (not z_cf.is_contiguous()):
            return False
        eng = self.backend
        if eng == 'auto':
            eng = 'cuda' if x_cf.is_cuda else 'sequential'
        return eng == 'cuda'

    def _delta_prep_in_kernel(self, x_cf, n_flat: int, n_hm: int) -> bool:
        if not (n_hm > 0 and n_flat == 0):
            return False
        if self.b_norm is not None or self.c_norm is not None:
            return False
        if not hm_glue_on('dtk'):
            return False
        eng = self.backend
        if eng == 'auto':
            eng = 'cuda' if x_cf.is_cuda else 'sequential'
        return eng == 'cuda'

    def _skip_channels(self):
        return self.D.repeat_interleave(self.P) if self.d_skip_mode == 'head' else self.D.flatten()

    def _d_skip_in_kernel(self, x_cf) -> bool:
        eng = self.backend
        if eng == 'auto':
            eng = 'cuda' if x_cf.is_cuda else 'sequential'
        return eng == 'cuda' and self.d_skip_mode == 'head' and (_os.environ.get('HYBRID_SSM_MH_DSKIP_INKERNEL', '1') == '1')

    def _fused_hm_tails(self, dt_lin, Bt, Ct):
        from .. import tails as _t
        if not _t.tails_enabled():
            return None
        both = (self.b_norm is not None) == (self.c_norm is not None)
        if not both:
            return None
        if self.b_norm is None:
            return (_t.dt_softplus_cf(dt_lin, self.dt_bias), Bt, Ct)
        b_src, c_src = (Bt.transpose(1, 2), Ct.transpose(1, 2))
        d_src = dt_lin.transpose(1, 2)
        if not _t.can_fuse(b_src, c_src, d_src):
            return None
        N = Bt.shape[1]
        norm = self.b_norm is not None
        cfg = _t.TailCfg(N=N, R=1, O=1, H=self.H, L=Bt.shape[-1], bc_norm=norm, shared_bias=False, fold_gb=False, fold_gc=False, a_dyn=False, has_dt_bias=True, has_a_bias=False, eps_b=float(self.b_norm.eps) if norm else 0.0, eps_c=float(self.c_norm.eps) if norm else 0.0, a_floor=0.0, dt_dtype=torch.float32, bc_channel_major=True)
        Bo, Co, dt, _ = _t.token_tails(cfg, b_src=b_src if norm else None, c_src=c_src if norm else None, gb_src=None, gc_src=None, dt_src=d_src, a_src=None, w_b=self.b_norm.weight if norm else None, w_c=self.c_norm.weight if norm else None, bias_b=None, bias_c=None, bias_dt=self.dt_bias, bias_a=None, need_b=norm, need_c=norm, out_dtype_b=b_src.dtype, out_dtype_c=c_src.dtype)
        if not norm:
            return (dt, Bt, Ct)
        return (dt, Bo.squeeze(-1).transpose(1, 2), Co.squeeze(-1).transpose(1, 2))

    def _pre_gate(self, M, g, beta):
        if g is None:
            return M
        wd = torch.promote_types(torch.promote_types(M.dtype, g.dtype), torch.float32)
        gv = g.to(wd)
        if beta is not None:
            gv = gv + beta.to(wd).reshape(1, -1, 1)
        fac = gate_act(gv, self._gate_act)
        on = getattr(self, 'gate_row_on', None)
        if on is not None:
            fac = torch.where(on.reshape(1, -1, 1), fac, torch.ones_like(fac))
        return (fac * M.to(wd)).to(M.dtype)

    def _hm_half_cf(self, maps, x_cf, delta, Bt, Ct, h_init, want_state, d_skip=None, bc_normed=False, dt_raw=False, z=None, complete_state=None):
        Bsz, _, L = x_cf.shape
        if self.b_norm is not None and (not bc_normed):
            Bt = rms_norm_cast(Bt.transpose(1, 2), self.b_norm.weight, self.b_norm.eps).transpose(1, 2)
        if self.c_norm is not None and (not bc_normed):
            Ct = rms_norm_cast(Ct.transpose(1, 2), self.c_norm.weight, self.c_norm.eps).transpose(1, 2)
        x_t = x_cf.reshape(Bsz, self.H, self.P, L)
        A_log = self._A_log_kernel()
        h0 = None
        if h_init is not None or self.h_init_param is not None:
            h0 = self.initial_state(Bsz, h_init)[..., :self.d_state_hm].contiguous()
        pre = self.gate_pos == 'pre' and self.gB_proj is not None

        def _m(key, idx=None):
            t = maps.get(key)
            if t is None:
                return None
            return self._rows_to_kernel(t, idx, 1).contiguous()
        eng = self.backend
        if eng == 'auto':
            eng = 'cuda' if x_cf.is_cuda else 'torch'
        a_dyn, dyn_act = self._dyn_scan_input(_m('a_dyn', self.dyn_idx), eng, x_cf.dtype)
        kw = dict(delta_softplus=bool(dt_raw), delta_bias=self.dt_bias if dt_raw else None, row_law=self.row_law, row_rot=self.row_rot, a_dyn=a_dyn, a_dyn_res=self.dyn_res, dyn_act=dyn_act, mu=_m('mu', self.mu_idx if self.mu_res == 'n' else None), mu_gain=self._mu_gain_kernel(), mu_res=self.mu_res, theta=None if maps.get('theta') is None else self._pairs_to_kernel(self._theta_feature(_m('theta').to(torch.float64 if x_cf.dtype == torch.float64 else torch.float32)), dim=-2), theta_delta=maps.get('theta') is not None, theta_h=self._theta_h_kernel(), g_B=None if pre else _m('g_B'), g_C=None if pre else _m('g_C'), gate_act=self._gate_act, beta_a=None, beta_b=self.beta_b, beta_c=self.beta_c, beta_gB=None if pre else self._gate_bias(self.beta_gB), beta_gC=None if pre else self._gate_bias(self.beta_gC), h_init=h0, theta_init=None if complete_state is None else self._pairs_to_kernel(complete_state.get('theta')), mu_init=None if complete_state is None else self._rows_to_kernel(complete_state.get('mu'), self.mu_idx, 1) if self.mu_res == 'n' else complete_state.get('mu'), write_prev=None if complete_state is None else complete_state.get('write_prev'), d_skip=d_skip, z=None if z is None else z.reshape(Bsz, self.H, self.P, L), trap_lam=sigmoid_exact(_m('trap').to(torch.float64 if x_cf.dtype == torch.float64 else torch.float32)) if maps.get('trap') is not None else None)
        if pre:
            Bt = self._pre_gate(Bt, _m('g_B'), self._gate_bias(self.beta_gB))
            Ct = self._pre_gate(Ct, _m('g_C'), self._gate_bias(self.beta_gC))
        args = (x_t, delta.contiguous(), A_log, Bt.contiguous(), Ct.contiguous())
        self._route('hm', 'mh_scan' if eng == 'cuda' else 'mh_torch_scan(sequential)' if eng == 'sequential' else 'mh_torch_scan')
        if eng == 'cuda':
            from ..kernels.hybrid_mamba_mh.functions import mh_scan
            out = mh_scan(*args, ckpt=self.ckpt, mem=self.mem, return_state=want_state, kernel_options=self.kernel_options or None, **kw)
            Y, h_final = (out[0], out[1]) if want_state else (out, None)
        else:
            from ..kernels.hybrid_mamba_mh.torch_scan import mh_torch_scan
            Y, h_final = mh_torch_scan(*args, sequential=eng == 'sequential', **kw)
            if not want_state:
                h_final = None
        y = Y.reshape(Bsz, self.d_inner, L)
        return (y.to(x_cf.dtype), None if h_final is None else h_final.to(x_cf.dtype))

    def _flat_state_scan(self, law, maps, x_cf, dt, Bt, Ct, state, start, end):
        if law != 'm3' or self.m3_angle_proj is None:
            return self._flat_state_scan_chunk(law, maps, x_cf, dt, Bt, Ct, state, start, end)
        tile = _flat_phase_tile(x_cf.shape[0], self.H, (end - start) // 2)
        if x_cf.shape[-1] <= tile:
            return self._flat_state_scan_chunk(law, maps, x_cf, dt, Bt, Ct, state, start, end)
        from torch.utils.checkpoint import checkpoint
        wd = torch.float64 if x_cf.dtype == torch.float64 else torch.float32
        h = state['hidden_state'][..., start:end]
        phase, kp, vp = (state['c3_phase'], state['c3_kp'], state['c3_vp'])

        def run(h, phase, kp, vp, x, clock, b, c, rate, trap, angle):
            local = State(hidden_state=F.pad(h, (start, self.d_state - end)), c3_phase=phase, c3_kp=kp, c3_vp=vp)
            local_maps = {'m3_a': rate, 'm3_trap': trap, 'm3_ang': angle}
            y, hf = self._flat_state_scan_chunk(law, local_maps, x, clock, b, c, local, start, end)
            feature = torch.tanh(angle.to(wd)) * math.pi
            terminal = phase_bmm(clock.to(wd), feature.transpose(1, 2))
            ph = phase.to(wd) + terminal
            key = self.m3_b_norm(b[..., -1]).to(wd)[:, None] + self.m3_B_bias.to(wd)
            e, o = (key[..., 0::2], key[..., 1::2])
            phf = F.pad(ph, (0, (end - start) // 2 - ph.shape[-1]))
            key = torch.stack((e * phf.cos() - o * phf.sin(), e * phf.sin() + o * phf.cos()), -1).flatten(-2)
            return (y, hf, ph, key, x[..., -1].reshape(x.shape[0], self.H, self.P).to(wd))
        outputs = []
        for t in range(0, x_cf.shape[-1], tile):
            sl = slice(t, t + tile)
            args = (h, phase, kp, vp, x_cf[..., sl], dt[..., sl], Bt[..., sl], Ct[..., sl], maps['m3_a'][..., sl], maps['m3_trap'][..., sl], maps['m3_ang'][..., sl])
            if self.ckpt and torch.is_grad_enabled() and any((a.requires_grad for a in args)):
                y, h, phase, kp, vp = checkpoint(run, *args, use_reentrant=False)
            else:
                y, h, phase, kp, vp = run(*args)
            outputs.append(y)
        return (torch.cat(outputs, -1), h)

    def _flat_state_scan_chunk(self, law, maps, x_cf, dt, Bt, Ct, state, start, end):
        wd = torch.float64 if x_cf.dtype == torch.float64 else torch.float32
        batch, _, length = x_cf.shape
        n = end - start
        dt = dt.to(wd)
        v = x_cf.reshape(batch, self.H, self.P, length).to(wd)
        k = getattr(self, f'{law}_b_norm')(Bt.transpose(1, 2)).to(wd).transpose(1, 2)[:, None]
        q = getattr(self, f'{law}_c_norm')(Ct.transpose(1, 2)).to(wd).transpose(1, 2)[:, None]
        beta = getattr(self, f'{law}_B_bias').to(wd).unsqueeze(-1)
        gamma = getattr(self, f'{law}_C_bias').to(wd).unsqueeze(-1)
        native = self.backend == 'cuda' or (self.backend == 'auto' and x_cf.is_cuda)
        phase = theta = None
        lam = previous = None
        if law == 'm3':
            log_a = -self._m3_rate(maps['m3_a'].to(wd)) * dt
            lam = sigmoid_exact(maps['m3_trap'].to(wd))
            previous = state['c3_vp'].to(wd)[..., None] * state['c3_kp'].to(wd)[:, :, None]
            if self.m3_angle_proj is not None:
                if native:
                    angle = maps['m3_ang'].transpose(1, 2).to(wd).unsqueeze(2)
                    theta = _phase_law(angle, dt, None, self.H, n // 2, -1.0)
                    c3 = state['c3_phase'].to(wd)
                    theta = theta - F.pad(c3, (0, n // 2 - c3.shape[-1]))[:, None]
                else:
                    inc = torch.tanh(maps['m3_ang'].to(wd))[:, None] * math.pi * dt[:, :, None]
                    inc = F.pad(inc, (0, 0, 0, n // 2 - inc.shape[-2]))
                    c3 = state['c3_phase'].to(wd)
                    phase = F.pad(c3, (0, n // 2 - c3.shape[-1]))[..., None] + inc.cumsum(-1)
        else:
            log_a, _, _ = self._flat_decay(law, dt)
        hidden = state['hidden_state'][..., start:end].to(wd)
        self._route(law, 'ssd_mimo_scan' if native else 'flat_torch_scan')
        if native:
            from ..kernels.ssd_mimo.autograd import ssd_mimo_scan
            bk = k.permute(0, 3, 1, 2).unsqueeze(-1).to(x_cf.dtype).contiguous()
            cq = q.permute(0, 3, 1, 2).unsqueeze(-1).to(x_cf.dtype).contiguous()
            if lam is not None:
                hidden = hidden + (dt[..., 0] * (1 - lam[..., 0]))[..., None, None] * previous
            y, final = ssd_mimo_scan(v.to(x_cf.dtype).unsqueeze(3).contiguous(), (-log_a).contiguous(), bk, cq, dt.new_zeros(self.H), delta_w=dt.contiguous(), trap_lam=None if lam is None else lam.contiguous(), beta=beta, gamma=gamma, theta=theta, bias_pre_rot=True, x0=hidden.contiguous(), bc_per_head=True, chunk_size=64, precision='ieee' if x_cf.dtype == torch.float32 else 'io')
            y = y.reshape(batch, self.H, self.P, length)
        else:
            from ..kernels.hybrid_mamba_mh.flat_scan import flat_torch_scan
            y, final = flat_torch_scan(v, k, q, log_a, dt, hidden, lam=lam, previous=previous, beta=beta, gamma=gamma, phase=phase, ckpt=self.ckpt)
        return (y.reshape(batch, self.d_inner, length).to(x_cf.dtype), final)

    def _m3_half_cf(self, maps, x_cf, dt, Bt, Ct):
        if self.m3_kernel == 'upstream' and (not x_cf.is_cuda):
            raise ValueError("m3_kernel='upstream' batched forward requires CUDA")
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        Bsz, _, L = x_cf.shape
        H, P, n3 = (self.H, self.P, self.d_state_m3)
        if self.m3_angle_proj is not None and self.m3_kernel == 'native':
            tile = _flat_phase_tile(Bsz, H, n3 // 2)
            if L > tile:
                initial = self.init_state(Bsz, device=x_cf.device, dtype=x_cf.dtype)
                y, _ = self._flat_state_scan('m3', maps, x_cf, dt, Bt, Ct, initial, self.d_state_hm, self.d_state_hm + n3)
                return y
        Bt = rms_norm_cast(Bt.transpose(1, 2), self.m3_b_norm.weight, self.m3_b_norm.eps)
        Ct = rms_norm_cast(Ct.transpose(1, 2), self.m3_c_norm.weight, self.m3_c_norm.eps)
        A = -self._m3_rate(maps['m3_a'].to(wd))
        dtf = dt.to(wd)
        decay = (-(A * dtf)).contiguous()
        trap = maps['m3_trap'].contiguous()
        lam = sigmoid_exact(trap.to(wd))
        v = x_cf.reshape(Bsz, H, P, L)
        eng = self.backend
        if eng == 'auto':
            eng = 'cuda' if x_cf.is_cuda else 'sequential'
        self._route('m3', '_m3_reference(fp64 oracle)' if eng != 'cuda' else 'upstream_siso' if self.m3_kernel == 'upstream' else 'ssd_mimo_scan')
        if eng != 'cuda':
            if self.m3_angle_proj is not None:
                angles = maps['m3_ang'].transpose(1, 2).unsqueeze(-2).expand(-1, -1, H, -1).to(wd)
                angles = F.pad(angles, (0, n3 // 2 - angles.shape[-1]))
            else:
                angles = torch.zeros(Bsz, L, H, n3 // 2, device=x_cf.device, dtype=wd)
            y = _m3_reference(Bt.unsqueeze(2), Ct.unsqueeze(2), v.permute(0, 3, 1, 2), -decay, dtf, trap, self.m3_B_bias, self.m3_C_bias, angles)
            return y.reshape(Bsz, L, self.d_inner).transpose(1, 2).to(x_cf.dtype)
        if self.m3_kernel == 'upstream':
            from ..kernels.m3_adapter import siso
            angles = maps['m3_ang'].transpose(1, 2).unsqueeze(2).expand(Bsz, L, H, -1).float() if self.m3_angle_proj is not None else dt.new_zeros(Bsz, L, H, n3 // 2, dtype=torch.float32)
            angles = F.pad(angles, (0, n3 // 2 - angles.shape[-1]))
            y = siso(Q=Ct.unsqueeze(2), K=Bt.unsqueeze(2), V=v.permute(0, 3, 1, 2).contiguous(), ADT=-decay, DT=dtf, Trap=trap.float(), Q_bias=self.m3_C_bias, K_bias=self.m3_B_bias, Angles=angles.contiguous(), chunk_size=64)
            return y.reshape(Bsz, L, self.d_inner).transpose(1, 2).to(x_cf.dtype)
        from ..kernels.ssd_mimo.autograd import ssd_mimo_scan
        theta = None
        if self.m3_angle_proj is not None:
            ang = maps['m3_ang'].transpose(1, 2).to(wd)
            theta = _phase_law(ang.unsqueeze(2), dtf.contiguous(), None, H, n3 // 2, -1.0)
        bk, ck = (Bt[:, :, None, :, None], Ct[:, :, None, :, None])
        beta, gamma = (self.m3_B_bias.unsqueeze(-1), self.m3_C_bias.unsqueeze(-1))
        y, _ = ssd_mimo_scan(v.unsqueeze(3), decay, bk, ck, self.const_a_log, chunk_size=64, delta_w=dtf.contiguous(), trap_lam=lam.contiguous(), bc_per_head=True, beta=beta, gamma=gamma, theta=theta, theta_increment=False, bias_pre_rot=theta is not None)
        return y.reshape(Bsz, self.d_inner, L).to(x_cf.dtype)

    def _flat_decay(self, law, dt):
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        A_log = getattr(self, f'{law}_A_log').to(wd)
        dtf = dt.to(wd)
        decay, write = (dtf, None)
        log_a = -(decay * A_log.exp()[None, :, None])
        return (log_a, decay, write)

    def _flat_half_cf(self, law, x_cf, dt, Bt, Ct):
        Bsz, _, L = x_cf.shape
        H, P = (self.H, self.P)
        bn, cn = (getattr(self, f'{law}_b_norm'), getattr(self, f'{law}_c_norm'))
        bb, cb = (getattr(self, f'{law}_B_bias'), getattr(self, f'{law}_C_bias'))
        Bt = rms_norm_cast(Bt.transpose(1, 2), bn.weight, bn.eps)
        Ct = rms_norm_cast(Ct.transpose(1, 2), cn.weight, cn.eps)
        log_a, decay, write = self._flat_decay(law, dt)
        v = x_cf.reshape(Bsz, H, P, L)
        eng = self.backend
        if eng == 'auto':
            eng = 'cuda' if x_cf.is_cuda else 'sequential'
        self._route(law, 'ssd_mimo_scan' if eng == 'cuda' else '_flat_reference(fp64 oracle)')
        if eng != 'cuda':
            y = _flat_reference(Bt, Ct, v.permute(0, 3, 1, 2), log_a, decay if write is None else write, bb, cb)
            return y.reshape(Bsz, L, self.d_inner).transpose(1, 2).to(x_cf.dtype)
        from ..kernels.ssd_mimo.autograd import ssd_mimo_scan
        y, _ = ssd_mimo_scan(v.unsqueeze(3), decay.float().contiguous(), Bt.to(x_cf.dtype).contiguous(), Ct.to(x_cf.dtype).contiguous(), getattr(self, f'{law}_A_log').float(), chunk_size=64, delta_w=None if write is None else write.float().contiguous(), beta=bb.unsqueeze(-1), gamma=cb.unsqueeze(-1))
        return y.reshape(Bsz, self.d_inner, L).to(x_cf.dtype)

    def _flat_step(self, law, x_t, dt, Bt, Ct, state, new_state):
        wd = torch.float64 if dt.dtype == torch.float64 else torch.float32
        Bsz, H, P = (x_t.shape[0], self.H, self.P)
        k = getattr(self, f'{law}_b_norm')(Bt).to(wd)[:, None] + getattr(self, f'{law}_B_bias').to(wd)
        q = getattr(self, f'{law}_c_norm')(Ct).to(wd)[:, None] + getattr(self, f'{law}_C_bias').to(wd)
        rate = getattr(self, f'{law}_A_log').to(wd).exp()
        alpha = torch.exp(-rate * dt.to(wd))
        start = self.d_state_hm + self.d_state_m3
        end = start + getattr(self, f'd_state_{law}')
        old = state['hidden_state'][..., start:end].to(wd)
        hidden = alpha[..., None, None] * old + dt.to(wd)[..., None, None] * x_t.reshape(Bsz, H, P).to(wd)[..., None] * k[:, :, None]
        new_state['_hidden_update'] = hidden.to(state['hidden_state'].dtype)
        y = (hidden * q[:, :, None]).sum(-1)
        return (y.reshape(Bsz, self.d_inner).to(x_t.dtype), new_state)

    def _flat_offsets(self):
        o3 = self.d_state_hm
        o2 = o3 + self.d_state_m3
        return (o3, o2)

    def _forward_cf(self, x_cf, h_init=None, sel_x=None, want_state=False, tok_raw=None, z_cf=None):
        self._route_begin(x_cf.is_cuda)
        if x_cf.shape[-1] == 0 or (self.fracs.n_flat and (self.m3_kernel == 'native' or want_state or h_init is not None or (self.h_init_param is not None))):
            self._h3_gate_in_kernel = False
            state = State.from_hidden(self, h_init, batch_size=x_cf.shape[0]) if h_init is not None else None
            y, state_final = self.forward(x_cf.transpose(1, 2), state=state, return_state=True, sel_x=sel_x, tok_raw=tok_raw)
            return (y.transpose(1, 2), state_final.hidden_state)
        y, h_final = self._scan_cf(x_cf, h_init, sel_x, want_state, tok_raw, z_cf=z_cf)
        if not self._k11_skip_in_kernel:
            xd = x_cf * self._skip_channels().to(x_cf.dtype).unsqueeze(-1)
            y = y + xd
        return (y, h_final)

    def _run(self, x, h_init, sel_x, want_state, tok_raw=None):
        y, h_final = self._scan_cf(x.transpose(1, 2).contiguous(), h_init, sel_x, want_state, tok_raw)
        y = y.transpose(1, 2)
        if not self._k11_skip_in_kernel:
            y = y + x * self._skip_channels().to(x.dtype)
        self.h_final = h_final
        return (y, h_final)

    def forward(self, x, *, sel_x=None, state=None, return_state=False, tok_raw=None):
        self._route_begin(x.is_cuda)
        if state is None and (not return_state) and (self.fracs.n_flat == 0) and (x.shape[1] > 0):
            return self._run(x, None, sel_x, want_state=False, tok_raw=tok_raw)[0]
        if x.shape[1] > 0 and self.backend != 'sequential' and (self.m3_kernel != 'upstream'):
            initial = self.resolve_state(state, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
            y, hidden, final = self._scan_cf(x.transpose(1, 2).contiguous(), initial.hidden_state if isinstance(initial, State) else initial['hidden_state'], sel_x, True, tok_raw, complete_state=initial)
            y = y.transpose(1, 2)
            if not self._k11_skip_in_kernel:
                y = y + x * self._skip_channels().to(x.dtype)
            self._final_state, self.h_final = (final, hidden)
            return (y, final) if return_state else y
        return self._forward_stepped(x, state, sel_x, return_state, tok_raw)

    def _forward_stepped(self, x, state, sel_x, return_state, tok_raw=None):
        Bsz, L, _ = x.shape
        if L:
            self._route('all', 'stepped(per-token recurrence)')
        state = self.resolve_state(state, batch_size=Bsz, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            y_t, state = self.step(x[:, t], state, sel_x=sel_x[:, t] if sel_x is not None else None, tok_raw=tok_raw[:, t] if tok_raw is not None else None)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) if ys else x.new_empty(Bsz, 0, self.out_width)
        self._final_state = state
        self.h_final = state['hidden_state']
        if return_state:
            return (y, state)
        return y

class _HMInProjFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xa, w, d_inner, z_cf=False):
        with torch.amp.autocast('cuda', enabled=False):
            dt, B = (xa.dtype, xa.shape[0])
            wx = w[:d_inner].to(dt)
            wz = w[d_inner:].to(dt)
            xat = xa.transpose(1, 2)
            x_cf = torch.bmm(wx.expand(B, *wx.shape), xat)
            z = torch.bmm(wz.expand(B, *wz.shape), xat) if z_cf else F.linear(xa, wz)
        ctx.save_for_backward(xa, wx, wz)
        ctx.d_inner = d_inner
        ctx.z_cf = z_cf
        ctx.w_meta = (w.shape, w.dtype)
        return (x_cf, z)

    @staticmethod
    def backward(ctx, dx_cf, dz):
        xa, wx, wz = ctx.saved_tensors
        B, L, dm = xa.shape
        di = ctx.d_inner
        dx_cf = dx_cf if dx_cf.is_contiguous() else dx_cf.contiguous()
        dz = dz if dz.is_contiguous() else dz.contiguous()
        x_needs, w_needs = (ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        dxa = None
        if x_needs:
            dxa = torch.bmm(dx_cf.transpose(1, 2), wx.expand(B, di, dm))
            if ctx.z_cf:
                dxa.baddbmm_(dz.transpose(1, 2), wz.expand(B, di, dm))
            else:
                dxa = dxa.reshape(-1, dm)
                dxa.addmm_(dz.reshape(-1, di), wz)
                dxa = dxa.view(B, L, dm)
        dw = None
        if w_needs:
            shape, wdtype = ctx.w_meta
            dw = torch.empty(shape, dtype=wdtype, device=xa.device)
            dw[:di].copy_(torch.bmm(dx_cf, xa).sum(0))
            dw[di:].copy_(torch.bmm(dz, xa).sum(0) if ctx.z_cf else dz.reshape(-1, di).t() @ xa.reshape(-1, dm))
        return (dxa, dw, None, None)

class _HMInProjTokFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xa, w, w_tok, d_inner, z_cf=False):
        with torch.amp.autocast('cuda', enabled=False):
            dt, B = (xa.dtype, xa.shape[0])
            wx = w[:d_inner].to(dt)
            wz = w[d_inner:].to(dt)
            wt = w_tok.to(dt)
            xat = xa.transpose(1, 2)
            x_cf = torch.bmm(wx.expand(B, *wx.shape), xat)
            z = torch.bmm(wz.expand(B, *wz.shape), xat) if z_cf else F.linear(xa, wz)
            tok_cf = torch.bmm(wt.expand(B, *wt.shape), xat)
        ctx.save_for_backward(xa, wx, wz, wt)
        ctx.d_inner = d_inner
        ctx.z_cf = z_cf
        ctx.w_meta = (w.shape, w.dtype, w_tok.dtype)
        return (x_cf, z, tok_cf)

    @staticmethod
    def backward(ctx, dx_cf, dz, dtok):
        xa, wx, wz, wt = ctx.saved_tensors
        B, L, dm = xa.shape
        di = ctx.d_inner
        zeros = None
        if dx_cf is None:
            dx_cf = zeros = xa.new_zeros(B, di, L)
        if dz is None:
            dz = xa.new_zeros(*((B, di, L) if ctx.z_cf else (B, L, di)))
        dx_cf = dx_cf if dx_cf.is_contiguous() else dx_cf.contiguous()
        dz = dz if dz.is_contiguous() else dz.contiguous()
        if dtok is not None and (not dtok.is_contiguous()):
            dtok = dtok.contiguous()
        x_needs, w_needs, t_needs = ctx.needs_input_grad[:3]
        dxa = None
        if x_needs:
            dxa = torch.bmm(dx_cf.transpose(1, 2), wx.expand(B, di, dm))
            if ctx.z_cf:
                dxa.baddbmm_(dz.transpose(1, 2), wz.expand(B, di, dm))
            else:
                dxa = dxa.reshape(-1, dm)
                dxa.addmm_(dz.reshape(-1, di), wz)
                dxa = dxa.view(B, L, dm)
            if dtok is not None:
                dxa.baddbmm_(dtok.transpose(1, 2), wt.expand(B, *wt.shape))
        dw = dwt = None
        shape, wdtype, tdtype = ctx.w_meta
        if w_needs:
            dw = torch.empty(shape, dtype=wdtype, device=xa.device)
            if zeros is None:
                dw[:di].copy_(torch.bmm(dx_cf, xa).sum(0))
            else:
                dw[:di].zero_()
            dw[di:].copy_(torch.bmm(dz, xa).sum(0) if ctx.z_cf else dz.reshape(-1, di).t() @ xa.reshape(-1, dm))
        if t_needs:
            dwt = torch.bmm(dtok, xa).sum(0).to(tdtype) if dtok is not None else torch.zeros_like(wt, dtype=tdtype)
        return (dxa, dw, dwt, None, None)

def _hm_in_proj(m, xa, z_cf=False):
    if hm_glue_on('inproj') and (not m.fuse_in_proj) and xa.is_cuda and torch.is_grad_enabled() and (xa.requires_grad or m.in_proj.weight.requires_grad):
        return _HMInProjFn.apply(xa, m.in_proj.weight, m.d_inner, z_cf)
    W = m.in_proj.weight
    x_cf = matmul_channel_major(W[:m.d_inner], xa, grad_layout='token')
    wz = W[m.d_inner:]
    return (x_cf, matmul_channel_major(wz, xa, grad_layout='token') if z_cf else F.linear(xa, wz))

class HybridMambaMixer(MixerConfigurable, BaseMixer):
    config_family = 'hybrid_mamba'
    config_execution_fields = ('compile_gate',)

    def __init__(self, d_model: int, d_state: int=64, expand: float | None=2.0, d_conv: int=4, ssm_kwargs: dict | None=None, compile_gate: bool=False, *, d_inner: int | None=None):
        super().__init__()
        if isinstance(ssm_kwargs, dict):
            reject_retired_keys(ssm_kwargs, 'HybridMambaMixer.ssm_kwargs')
        from ..layers.widths import resolve_mixer_width
        d_inner = resolve_mixer_width(d_model, expand, d_inner)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self._compile_gate = bool(compile_gate)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        if d_conv > 0:
            self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, bias=True)
        else:
            self.conv1d = None
        kw = mixer_core_options(ssm_kwargs, config_type=HybridMambaConfig, d_inner=d_inner, d_state=d_state, rows=[{'frac': 1.0, 'decay': 'A'}], d_model=d_model, d_conv=d_conv)
        head_dim = kw.pop('head_dim', None)
        out_norm = bool(kw.pop('out_norm', False))
        if head_dim is not None:
            from ..configuration import resolve_alias
            kw['head_subsystems'] = resolve_alias('head_subsystems', head_dim=head_dim, head_subsystems=kw.get('head_subsystems'), P=kw.get('P'))
        if not any((kw.get(k) is not None for k in ('n_heads', 'heads', 'H', 'head_subsystems', 'P'))):
            kw['head_subsystems'] = 64
        self.ssm = HybridMambaSSM(**kw)
        self.norm = RMSNorm(self.ssm.out_width) if out_norm else nn.Identity()
        self.out_proj = nn.Linear(self.ssm.out_width, d_model, bias=False)
        record_mixer_config(self, expand=expand, out_norm=out_norm, execution=dict(compile_gate=self._compile_gate))

    @property
    def fuse_in_proj(self) -> bool:
        return self.ssm.proj_source == 'layer_input'

    def resolved_config(self) -> dict:
        document = super().resolved_config()
        core = self.ssm.resolved_config()
        document['model']['core'] = core['model']
        document['execution']['core_execution'] = core['execution']
        return document
    _rg: frozenset = frozenset()

    def forward(self, x: torch.Tensor, *, state=None, return_state: bool=False):
        rg = hm_held_regen() if state is None and (not return_state) else frozenset()
        self._rg = rg
        try:
            with _regen.scope(bool(rg)):
                return self._forward_impl(x, state=state, return_state=return_state)
        finally:
            self._rg = frozenset()

    def _forward_impl(self, x: torch.Tensor, *, state=None, return_state: bool=False):
        if state is not None:
            state = self.resolve_state(state, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        W = self.in_proj.weight
        xa = autocast_cast(x)
        tok_raw = None
        want_state = return_state or state is not None
        tok_node = self.fuse_in_proj and (not want_state) and hm_glue_on('inproj') and xa.is_cuda and torch.is_grad_enabled() and (xa.requires_grad or W.requires_grad)
        want_z_cf = self._h3_gate_offered(tok_node) and (not want_state)
        if tok_node:
            x_cf, z, tok_cf = _HMInProjTokFn.apply(xa, W, self.ssm.tok_proj_weight(), self.d_inner, want_z_cf)
            tok_raw = tok_cf.transpose(1, 2)
        elif not self.fuse_in_proj:
            x_cf, z = _hm_in_proj(self, xa, z_cf=want_z_cf)
        else:
            x_cf = matmul_channel_major(W[:self.d_inner], xa, grad_layout='token')
        if self.fuse_in_proj and (not tok_node):
            zt = F.linear(xa, torch.cat([W[self.d_inner:], self.ssm.tok_proj_weight()], dim=0))
            z, tok_raw = zt.split([self.d_inner, self.ssm.tok_width], -1)
        conv_tail = None
        if self.conv1d is not None:
            kc = self.d_conv
            prefix = state['conv_state'][:, :, 1:].to(x_cf.dtype) if state is not None else None
            x_in = x_cf if prefix is None else torch.cat([prefix, x_cf], -1)
            x_cf = causal_depthwise_conv1d_cl(x_in, self.conv1d.weight, self.conv1d.bias, kc, activation='silu')
            if prefix is not None:
                x_cf = x_cf[..., prefix.shape[-1]:]
            x_cf = x_cf.contiguous()
            if 'x' in self._rg and prefix is None:
                _acd = (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
                _cw, _cb = (self.conv1d.weight, self.conv1d.bias)

                def _x_regen(x_in=x_in, _cw=_cw, _cb=_cb, kc=kc, _acd=_acd):
                    with torch.autocast('cuda', dtype=_acd[1], enabled=_acd[0]):
                        return causal_depthwise_conv1d_cl(x_in, _cw, _cb, kc, activation='silu').contiguous()
                _regen.register(x_cf, _x_regen)
            if want_state:
                tail = x_in[..., -kc:]
                if tail.shape[-1] < kc:
                    tail = torch.cat([tail.new_zeros(*tail.shape[:-1], kc - tail.shape[-1]), tail], dim=-1)
                conv_tail = tail if torch.is_grad_enabled() else tail.detach().clone()
        sel = xa if self.ssm.proj_source == 'layer_input' else None
        if not want_state:
            y_cf, _ = self.ssm._forward_cf(x_cf, sel_x=sel, want_state=False, tok_raw=tok_raw, z_cf=z if want_z_cf else None)
            if getattr(self.ssm, '_h3_gate_in_kernel', False):
                from ..layers.tails import gate_transpose
                return self.out_proj(gate_transpose(y_cf))
            if want_z_cf:
                z = z.transpose(1, 2)
            return self.out_proj(self._epilogue_cf(y_cf, z))
        ssm_state = State({k: v for k, v in state.items() if k != 'conv_state'}) if state is not None else None
        y, out_ssm_state = self.ssm(x_cf.transpose(1, 2).contiguous(), sel_x=sel, tok_raw=tok_raw, state=ssm_state, return_state=True)
        y = self.out_proj(self._epilogue(y, z))
        if not return_state:
            return y
        new_state = State(out_ssm_state)
        if conv_tail is not None:
            new_state['conv_state'] = conv_tail
        self._final_state = new_state
        return (y, new_state)

    def _epilogue(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if isinstance(self.norm, RMSNorm):
            return gated_rms_norm(y, self.norm.weight, self.norm.eps, z=z)
        return self.norm(swiglu_gate(y, z, self._compile_gate))

    def _h3_gate_offered(self, tok_node: bool=False) -> bool:
        return hm_glue_on('zk') and (tok_node or not self.fuse_in_proj) and (not self._compile_gate) and isinstance(self.norm, nn.Identity) and hasattr(self.ssm, '_forward_cf') and (getattr(self.ssm, 'd_state_m3', 1) == 0) and (getattr(self.ssm, 'd_state_hm', 0) > 0)

    def _epilogue_cf(self, y_cf: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from ..layers.tails import gate_transpose
        if isinstance(self.norm, RMSNorm):
            return gated_rms_norm(gate_transpose(y_cf), self.norm.weight, self.norm.eps, z=z)
        if self._compile_gate or not isinstance(self.norm, nn.Identity):
            return self._epilogue(gate_transpose(y_cf), z)
        out = gate_transpose(y_cf, z)
        if 'out' in self._rg:
            _acd = (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))

            def _out_regen(y_cf=y_cf, z=z, _acd=_acd):
                with torch.autocast('cuda', dtype=_acd[1], enabled=_acd[0]):
                    return gate_transpose(y_cf, z)
            _regen.register(out, _out_regen)
        return out

    def _init_state(self, batch_size: int, device=None, dtype=torch.float32) -> State:
        dev = device or next(self.parameters()).device
        state = self.ssm.init_state(batch_size, device=dev, dtype=dtype)
        if self.conv1d is not None:
            state['conv_state'] = torch.zeros(batch_size, self.d_inner, self.d_conv, device=dev, dtype=dtype)
        return state

    def step(self, x_t: torch.Tensor, state=None) -> tuple[torch.Tensor, State]:
        state = self.resolve_state(state, batch_size=x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        xz = self.in_proj(x_t)
        x_branch, z = xz.split([self.d_inner, self.d_inner], dim=-1)
        if self.conv1d is not None:
            conv_state = torch.cat([state['conv_state'][:, :, 1:], x_branch.unsqueeze(-1)], dim=-1)
            conv_w = self.conv1d.weight.squeeze(1).to(x_branch.dtype)
            conv_b = self.conv1d.bias.to(x_branch.dtype)
            x_branch = F.silu((conv_state * conv_w).sum(-1) + conv_b)
            state = {**state, 'conv_state': conv_state}
        sel = x_t if self.ssm.proj_source == 'layer_input' else None
        y_t, ssm_state = self.ssm.step(x_branch, {k: v for k, v in state.items() if k != 'conv_state'}, sel_x=sel)
        state = State({**state, **ssm_state})
        self._final_state = state
        return (self.out_proj(self._epilogue(y_t, z)), state)
from dataclasses import dataclass
from dataclasses import field as _field
from ..config import SSMCoreConfig, SpecFields, refuse_retired_kwargs, _refuse, resolve_write_rule, WRITE_RULE_DEFAULT

@refuse_retired_kwargs
@dataclass
class HybridMambaConfig(SpecFields, SSMCoreConfig):
    rows: list | None = _field(default_factory=lambda: [{'frac': 1.0, 'decay': 'A'}])
    gate_act: str | None = None
    gate_bias: bool | None = None
    gate_pos: str | None = None
    proj_source: str | None = None
    dt_rank: int | str | None = 'adaptive'
    force_dt_dense: bool = False
    dt_init: str | None = 'random'
    dt_scale: float = 1.0
    dt_init_floor: float = 0.0001
    dyn_res: str = 'n'
    dyn_act: str = 'heavy_tail'
    mu_res: str = 'n'
    mu_cap: float = MU_CAP_DEFAULT
    a_floor: float = 0.0001
    a_act: Literal['heavy_tail', 'softplus'] = 'heavy_tail'
    d_skip_mode: str = 'head'
    norm_eps: float = 1e-05
    mem: dict | None = None
    learnable_h_init: bool = False
    gate_init: str = 'kaiming_uniform'
    ckpt: bool = True
    m3_kernel: str = 'native'

    def __post_init__(self):
        super().__post_init__()
        self._geometry()
        self._use_trap()
        _config_spec.resolve('hybrid_mamba', d_state=self.d_state, **self.spec_kwargs())

    def _geometry(self):
        check_mu_cap(self.mu_cap)
        return resolve_heads(self.d_inner, n_heads=self.n_heads, heads=self.heads, H=self.H, head_subsystems=self.head_subsystems, P=self.P)

    def _use_trap(self) -> bool:
        rule = resolve_write_rule(WRITE_RULE_DEFAULT if self.write_rule is None else self.write_rule, 'HybridMambaConfig')
        return rule == 'trapezoid'
    _UNSUPPORTED = {'R': lambda c: c.R != 1, 'O': lambda c: c.O not in (None, 1), 'ngroups': lambda c: c.ngroups is not None, 'precision': lambda c: c.precision != 'io', 'chunk_size': lambda c: c.chunk_size != 'auto'}

    def to_ssm_kwargs(self) -> dict:
        _refuse(self, 'HybridMambaSSM', self._UNSUPPORTED)
        kw = dict(d_inner=self.d_inner, d_state=self.d_state, d_model=self.d_model, d_conv=self.d_conv, proj_source=self.proj_source, bc_conv=self.bc_conv, dt_rank=self.dt_rank, force_dt_dense=self.force_dt_dense, dt_init=self.dt_init, dt_scale=self.dt_scale, dt_init_floor=self.dt_init_floor, dt_min=self.dt_min, dt_max=self.dt_max, dyn_res=self.dyn_res, dyn_act=self.dyn_act, mu_res=self.mu_res, mu_cap=self.mu_cap, a_floor=self.a_floor, a_act=self.a_act, **self.spec_kwargs(), d_skip_mode=self.d_skip_mode, backend=self.backend, ckpt=self.ckpt, norm_eps=self.norm_eps, mem=self.mem, kernel_options=self.kernel_options, learnable_h_init=self.learnable_h_init, gate_init=self.gate_init, m3_kernel=self.m3_kernel, use_trap=self._use_trap())
        kw['heads'], _ = self._geometry()
        return kw

    def build(self):
        return HybridMambaSSM(**self.to_ssm_kwargs())
