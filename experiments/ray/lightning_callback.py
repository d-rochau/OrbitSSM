from __future__ import annotations
from typing import Any
from lightning.pytorch.callbacks import Callback
from ray import tune

def _scalarize_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str | bool]:
    payload: dict[str, float | int | str | bool] = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except Exception:
                continue
        if isinstance(value, (float, int, str, bool)):
            payload[key] = value
    return payload

class RayTuneReportCallback(Callback):

    def __init__(self, report_train_metrics_every_n_steps: int | None=None):
        super().__init__()
        self.report_train_metrics_every_n_steps = report_train_metrics_every_n_steps

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = _scalarize_metrics(trainer.callback_metrics)
        metrics.setdefault('epoch', int(trainer.current_epoch))
        metrics.setdefault('global_step', int(trainer.global_step))
        tune.report(metrics)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.report_train_metrics_every_n_steps:
            return
        if trainer.global_step == 0 or trainer.global_step % self.report_train_metrics_every_n_steps != 0:
            return
        metrics = _scalarize_metrics(trainer.callback_metrics)
        metrics.setdefault('epoch', int(trainer.current_epoch))
        metrics.setdefault('global_step', int(trainer.global_step))
        tune.report(metrics)
