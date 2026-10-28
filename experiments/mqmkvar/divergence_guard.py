
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any
import torch
import torch.nn.functional as F
try:
    from lightning.pytorch.callbacks import Callback
except Exception:
    Callback = object

@dataclass(frozen=True)
class GuardConfig:
    check_every: int = 250
    delta_max: float = 500.0
    delta_sustain: int = 3
    grad_ratio: float = 1000.0
    grad_window: int = 20
    grad_sustain: int = 5
    grad_min_history: int = 10
    abort_nonfinite: bool = True
    save_state_on_abort: bool = True
    detail_stats: bool = False

    def __post_init__(self):
        for name in ('check_every', 'delta_sustain', 'grad_window', 'grad_sustain', 'grad_min_history'):
            if int(getattr(self, name)) < 1:
                raise ValueError(f'GuardConfig.{name} must be >= 1')
        for name in ('delta_max', 'grad_ratio'):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f'GuardConfig.{name} must be finite and > 0')

    @classmethod
    def from_dict(cls, d: dict | None) -> 'GuardConfig':
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown GuardConfig fields: {sorted(unknown)}')
        return cls(**d)

class TrainingDiverged(RuntimeError):

    def __init__(self, reason: str, step: int, metrics: dict):
        super().__init__(f'training diverged at step {step}: {reason}')
        self.reason = reason
        self.step = int(step)
        self.metrics = metrics

    def record(self) -> dict:
        return dict(status='diverged', reason=self.reason, step=self.step, last_metrics=self.metrics)

def _dt_lin(parts):
    dtb = parts.get('_dtb')
    if dtb is None:
        return parts['dt']
    return parts['dt'].float() + dtb.float()

def _finite_or_none(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else repr(v)

class GuardState:

    def __init__(self, config: GuardConfig):
        self.config = config
        self.grad_history: list[float] = []
        self.delta_run = 0
        self.grad_run = 0

    def step_check(self, loss: float | None, grad_norm: float | None) -> str | None:
        if not self.config.abort_nonfinite:
            return None
        if grad_norm is not None and (not math.isfinite(grad_norm)):
            return f'nonfinite_grad_norm ({grad_norm!r})'
        if loss is not None and (not math.isfinite(loss)):
            return f'nonfinite_loss ({loss!r})'
        return None

    def periodic_check(self, max_delta: float | None, grad_norm: float | None) -> str | None:
        c = self.config
        reason = None
        if max_delta is not None:
            if not math.isfinite(max_delta):
                if c.abort_nonfinite:
                    return f'nonfinite_delta ({max_delta!r})'
            elif max_delta > c.delta_max:
                self.delta_run += 1
                if self.delta_run >= c.delta_sustain:
                    reason = f'delta_runaway: max Delta {max_delta:.4g} > {c.delta_max:g} for {self.delta_run} consecutive checks'
            else:
                self.delta_run = 0
        if grad_norm is not None and math.isfinite(grad_norm):
            hist = self.grad_history[-c.grad_window:]
            if len(hist) >= c.grad_min_history:
                med = statistics.median(hist)
                if med > 0 and grad_norm > c.grad_ratio * med:
                    self.grad_run += 1
                    if self.grad_run >= c.grad_sustain and reason is None:
                        reason = f'grad_norm_explosion: {grad_norm:.4g} > {c.grad_ratio:g} x running median {med:.4g} for {self.grad_run} consecutive checks'
                else:
                    self.grad_run = 0
            self.grad_history.append(float(grad_norm))
            if len(self.grad_history) > 4 * c.grad_window:
                del self.grad_history[:-c.grad_window]
        return reason

    def state_dict(self) -> dict:
        return dict(grad_history=list(self.grad_history), delta_run=self.delta_run, grad_run=self.grad_run)

    def load_state_dict(self, sd: dict) -> None:
        self.grad_history = [float(x) for x in sd.get('grad_history', [])]
        self.delta_run = int(sd.get('delta_run', 0))
        self.grad_run = int(sd.get('grad_run', 0))

def ssd_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, torch.nn.Module]]:
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    out = []
    for name, mod in model.named_modules():
        ssm = getattr(mod, 'ssm', None)
        if isinstance(ssm, SSDMimoSSM) and mod is not ssm:
            out.append((name, mod, ssm))
    return out

@torch.no_grad()
def ssd_token_law_stats(ssm, u: torch.Tensor) -> dict:
    from orbit_ssm.ssm import ssd_mimo as sm
    if ssm.proj_source != 'layer_input':
        raise NotImplementedError('guard stats assume proj_source="layer_input" (u = mixer input)')
    with torch.autocast(device_type=u.device.type, enabled=False):
        w, b = ssm.tok_proj_weight()
        slab = F.linear(u.float(), w.float(), None if b is None else b.float())
        parts = ssm.split_tok_raw(slab)
        dt = F.softplus(_dt_lin(parts))
        out = dict(delta_max=float(dt.max()), delta_mean=float(dt.mean()), delta_p99=float(torch.quantile(dt.flatten()[::max(1, dt.numel() // 1000000)].float(), 0.99)))
        if getattr(ssm, 'a_dyn', False) and 'a' in parts:
            tail = sm._a_mag_tail_eager if ssm.a_act == 'heavy_tail' else sm._a_mag_softplus_eager
            a = tail(parts['a'], 0.0001)
            out.update(a_min=float(a.min()), a_max=float(a.max()), a_mean=float(a.mean()))
            B_, L_ = dt.shape[:2]
            H, R = (ssm.H, ssm.R)
            aa = a.view(B_, L_, H, -1)
            out['delta_a_max'] = float((dt.view(B_, L_, H, R) * aa).max())
    return out

def _q(x: torch.Tensor, q: float) -> float:
    x = x.flatten()
    return float(torch.quantile(x[::max(1, x.numel() // 1000000)].float(), q))

@torch.no_grad()
def ssd_detail_stats(ssm, u: torch.Tensor) -> dict:
    from orbit_ssm.ssm import ssd_mimo as sm
    from orbit_ssm.layers.norms import rms_norm_cast
    H, R, N = (ssm.H, ssm.R, ssm.N)
    out: dict = {}
    with torch.autocast(device_type=u.device.type, enabled=False):
        w, b = ssm.tok_proj_weight()
        slab = F.linear(u.float(), w.float(), None if b is None else b.float())
        parts = ssm.split_tok_raw(slab)
        Bsz, L = slab.shape[:2]
        dt = F.softplus(_dt_lin(parts)).view(Bsz, L, H, R)
        out['delta_r'] = {f'r{r}': dict(max=float(dt[..., r].max()), p99=_q(dt[..., r], 0.99), median=_q(dt[..., r], 0.5)) for r in range(R)}
        gap = dt.max(-1).values - dt.min(-1).values
        out['sibling_gap'] = dict(max=float(gap.max()), per_head_mean=[round(float(v), 5) for v in gap.mean((0, 1))], per_head_max=[round(float(v), 4) for v in gap.amax((0, 1))])
        out['delta_mean_hr'] = [[round(float(v), 5) for v in row] for row in dt.mean((0, 1))]
        out['delta_max_hr'] = [[round(float(v), 4) for v in row] for row in dt.amax((0, 1))]
        if getattr(ssm, 'a_dyn', False) and 'a' in parts:
            tail = sm._a_mag_tail_eager if ssm.a_act == 'heavy_tail' else sm._a_mag_softplus_eager
            a = tail(parts['a'], 0.0001)
            a = a.view(Bsz, L, H, R) if ssm.a_dyn_per_sibling else a.view(Bsz, L, H, 1)
            key = 'a_hr' if ssm.a_dyn_per_sibling else 'a_h'
            out[key + '_mean'] = [[round(float(v), 5) for v in row] for row in a.mean((0, 1))]
            out[key + '_min'] = [[round(float(v), 6) for v in row] for row in a.amin((0, 1))]
            expo = dt * a
            out['alpha_lt_1e-3_r'] = {f'r{r}': float((expo[..., r] > 6.907755).float().mean()) for r in range(R)}
            out['delta_a_max_r'] = {f'r{r}': float(expo[..., r].max()) for r in range(R)}
            out['sat_gt10'] = float((expo > 10.0).float().mean())
            out['sat_gt10_r'] = {f'r{r}': float((expo[..., r] > 10.0).float().mean()) for r in range(R)}
        if 'B' in parts:
            Bt = parts['B'].reshape(Bsz, L, N, R)
            if ssm.b_norm:
                Bn = rms_norm_cast(Bt.transpose(-1, -2), ssm.B_norm.weight.float(), ssm.B_norm.eps).transpose(-1, -2)
            else:
                Bn = Bt.float()
            flatB = Bn.reshape(Bsz * L, N * R)
            b2 = flatB.square().sum(-1, keepdim=True)
            if getattr(ssm, 'head_bias', False):
                beta = ssm.B_bias.float().reshape(H * R, N * R)
                cross = flatB @ beta.t()
                bnorm = (b2 + 2 * cross + beta.square().sum(-1)).clamp_min(0).sqrt().view(Bsz, L, H, R)
                out['beta_norm_hr'] = [[round(float(v), 4) for v in row] for row in beta.norm(dim=-1).view(H, R)]
            else:
                bnorm = b2.sqrt().view(Bsz, L, 1, 1).expand(Bsz, L, H, R)
            wg = dt * bnorm
            out['write_gain_r'] = {f'r{r}': dict(max=float(wg[..., r].max()), p99=_q(wg[..., r], 0.99), median=_q(wg[..., r], 0.5)) for r in range(R)}
            if 'gB' in parts:
                g = 1.0 + parts['gB']
                out['gB_absmax'] = float(g.abs().max())
    return out

def _grad_norm(params) -> float | None:
    grads = [p.grad for p in params if p is not None and p.grad is not None]
    if not grads:
        return None
    return float(torch.linalg.vector_norm(torch.stack([g.detach().float().norm() for g in grads])))

class DivergenceGuard(Callback):

    def __init__(self, config: GuardConfig, log_path: str | Path | None=None, echo: bool=True):
        super().__init__()
        self.config = config
        self.state = GuardState(config)
        self.log_path = None if log_path is None else Path(log_path)
        self.echo = echo
        self.last_record: dict = {}
        self.summary = dict(checks=0, delta_max_seen=0.0, a_min_seen=None, grad_max_seen=0.0, routes=set())
        self._armed = False
        self._inputs: dict[str, torch.Tensor] = {}
        self._outputs: dict[str, dict] = {}
        self._handles = []
        self._layers = []
        self._grad_total = None
        self._grad_groups: dict = {}
        self._last_loss = None

    @property
    def state_key(self) -> str:
        return 'DivergenceGuard'

    def state_dict(self) -> dict:
        return dict(state=self.state.state_dict(), config=asdict(self.config))

    def load_state_dict(self, state_dict: dict) -> None:
        self.state.load_state_dict(state_dict.get('state', {}))

    def setup(self, trainer, pl_module, stage=None):
        if self._handles:
            return
        self._layers = ssd_layers(pl_module.model)
        for name, mixer, _ssm in self._layers:

            def pre(mod, args, kwargs, _name=name):
                if self._armed and args:
                    self._inputs[_name] = args[0].detach()
            self._handles.append(mixer.register_forward_pre_hook(pre, with_kwargs=True))
            if self.config.detail_stats:

                def post(mod, args, output, _name=name):
                    if self._armed:
                        y = output[0] if isinstance(output, (tuple, list)) else output
                        with torch.no_grad():
                            self._outputs[_name] = dict(rms=float(y.detach().float().square().mean().sqrt()), absmax=float(y.detach().abs().max()))
                self._handles.append(mixer.register_forward_hook(post))

    def teardown(self, trainer, pl_module, stage=None):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _is_check(self, step: int) -> bool:
        return step % self.config.check_every == 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._armed = self._is_check(trainer.global_step)
        self._inputs = {}
        self._outputs = {}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        step = trainer.global_step
        params = [p for p in pl_module.model.parameters() if p.grad is not None]
        norms = torch._foreach_norm([p.grad for p in params]) if params else []
        total = float(torch.linalg.vector_norm(torch.stack(norms).float())) if norms else None
        self._grad_total = total
        if self._is_check(step):
            groups = {}
            for name, _mixer, ssm in self._layers:
                groups[name] = dict(dt_proj_weight=_grad_norm([ssm.dt_proj.weight]), dt_bias=_grad_norm([ssm.dt_proj.bias]), a_dyn_proj=_grad_norm(list(ssm.a_dyn_proj.parameters())) if getattr(ssm, 'a_dyn_proj', None) is not None else None)
                if self.config.detail_stats:
                    groups[name].update(B_proj=_grad_norm([ssm.B_proj.weight]), out_proj=_grad_norm(list(_mixer.out_proj.parameters())))
            self._grad_groups = groups
        reason = self.state.step_check(None, total)
        if reason:
            self._abort(trainer, pl_module, reason, step, dict(grad_norm=_finite_or_none(total)))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step - 1
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        loss = float(loss.detach()) if torch.is_tensor(loss) else None if loss is None else float(loss)
        self._last_loss = loss
        reason = self.state.step_check(loss, None)
        if reason:
            self._abort(trainer, pl_module, reason, step, dict(loss=_finite_or_none(loss), grad_norm=_finite_or_none(self._grad_total)))
        if not self._armed:
            return
        self._armed = False
        cm = getattr(trainer, 'callback_metrics', None) or {}
        acc = cm.get('train/accuracy_step', cm.get('train/accuracy'))
        rec = dict(event='guard', step=step, time=time.time(), loss=_finite_or_none(loss), train_acc=_finite_or_none(acc.detach().float().item()) if torch.is_tensor(acc) else None, grad_norm=_finite_or_none(self._grad_total), grad_groups=self._grad_groups, layers={})
        max_delta = None
        for name, mixer, ssm in self._layers:
            lay = {}
            u = self._inputs.get(name)
            if u is not None:
                lay.update(ssd_token_law_stats(ssm, u))
                max_delta = lay['delta_max'] if max_delta is None else max(max_delta, lay['delta_max'])
                if self.config.detail_stats:
                    lay['detail'] = ssd_detail_stats(ssm, u)
            if name in self._outputs:
                lay['mixer_out'] = self._outputs[name]
            lay['route'] = ssm.last_route()
            lay['combined_refusal'] = getattr(mixer, '_combined_refusal', None)
            rec['layers'][name] = lay
            self.summary['routes'].add(json.dumps([lay['route'], lay['combined_refusal']]))
            if 'a_min' in lay:
                a0 = self.summary['a_min_seen']
                self.summary['a_min_seen'] = lay['a_min'] if a0 is None else min(a0, lay['a_min'])
        self._inputs = {}
        rec['max_delta'] = max_delta
        self.summary['checks'] += 1
        if max_delta is not None and math.isfinite(max_delta):
            self.summary['delta_max_seen'] = max(self.summary['delta_max_seen'], max_delta)
        if self._grad_total is not None and math.isfinite(self._grad_total):
            self.summary['grad_max_seen'] = max(self.summary['grad_max_seen'], self._grad_total)
        reason = self.state.periodic_check(max_delta, self._grad_total)
        rec['delta_run'] = self.state.delta_run
        rec['grad_run'] = self.state.grad_run
        self.last_record = rec
        self._emit(rec)
        if reason:
            self._abort(trainer, pl_module, reason, step, rec)

    def _emit(self, rec: dict) -> None:
        line = json.dumps(rec, allow_nan=False, default=str)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a') as fh:
                fh.write(line + '\n')
        if self.echo:
            print('GUARD ' + line, flush=True)

    def summary_dict(self) -> dict:
        s = dict(self.summary)
        s['routes'] = sorted(s['routes'])
        s['config'] = asdict(self.config)
        return s

    def _abort(self, trainer, pl_module, reason: str, step: int, metrics: dict):
        metrics = dict(metrics)
        if self.last_record and metrics is not self.last_record:
            metrics.setdefault('last_check', self.last_record)
        rec = dict(event='abort', step=step, reason=reason, time=time.time(), metrics=metrics)
        if self.config.save_state_on_abort and self.log_path is not None:
            path = self.log_path.with_name('diverged_model_state.pt')
            try:
                torch.save({k: v.detach().cpu() for k, v in pl_module.model.state_dict().items()}, path)
                rec['state_path'] = str(path)
            except Exception as exc:
                rec['state_save_error'] = repr(exc)
        self._emit(rec)
        raise TrainingDiverged(reason, step, metrics | {'summary': self.summary_dict()})
