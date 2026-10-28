
from __future__ import annotations
from collections import defaultdict
from typing import Any
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from experiments.models.zoo import count_trainable_parameters
from torchmetrics import Metric
from zoology.train import compute_metrics

class SumRatio(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('num', default=torch.zeros((), dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('den', default=torch.zeros((), dtype=torch.float64), dist_reduce_fx='sum')

    def update(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += torch.as_tensor(num, dtype=torch.float64, device=self.num.device)
        self.den += torch.as_tensor(den, dtype=torch.float64, device=self.den.device)

    def compute(self) -> torch.Tensor:
        return (self.num / self.den).to(torch.float32)

class MQMKVARSupervisedTaskModule(LightningModule):

    def __init__(self, model, experiment_spec, data_metadata: dict[str, Any]):
        super().__init__()
        self.model = model
        self.experiment_spec = experiment_spec
        self.data_metadata = data_metadata
        self.loss_name = experiment_spec.task.loss
        self.ignore_index = int(experiment_spec.task.kwargs.get('ignore_index', -100))
        self.slice_keys = tuple(experiment_spec.task.kwargs.get('slice_keys', ()))
        self.answer_width = experiment_spec.task.kwargs.get('answer_width')
        if self.answer_width is not None:
            self.answer_width = int(self.answer_width)
            if self.answer_width < 1:
                raise ValueError('task answer_width must be >= 1')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.validation_rows: list[dict[str, Any]] = []
        self.test_rows: list[dict[str, Any]] = []
        names = ['loss', 'accuracy', 'example_accuracy']
        if self.answer_width is not None:
            names += ['answer_exact_accuracy', 'example_exact_accuracy']
        self.epoch_metrics = nn.ModuleDict({f'{stage}_{name}': SumRatio() for stage in ('train', 'val', 'test') for name in names})
        self.num_parameters = count_trainable_parameters(self.model)
        self.save_hyperparameters({'experiment_name': experiment_spec.name, 'model_name': experiment_spec.model.name, 'task_name': experiment_spec.task.name, 'loss': self.loss_name, 'primary_metric': experiment_spec.task.primary_metric, 'train_examples': data_metadata.get('train_examples'), 'val_examples': data_metadata.get('val_examples'), 'num_parameters': self.num_parameters})

    def forward(self, inputs, return_embeddings: bool=False):
        return self.model(inputs, return_embeddings=return_embeddings)

    def _shared_step(self, batch):
        inputs, targets, slices = batch
        if self.loss_name != 'ce':
            raise NotImplementedError(f'MQMKVAR Lightning path currently supports only ce loss, got {self.loss_name}')
        logits = self.model(inputs, return_embeddings=False)
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targets.flatten())
        preds = logits.argmax(dim=-1)
        mask = targets != self.ignore_index
        hits = (preds == targets) & mask
        n_ex = mask.sum(dim=-1)
        c_ex = hits.sum(dim=-1)
        tokens = n_ex.sum()
        has = n_ex > 0
        f32 = torch.float32
        stats = {'accuracy': (c_ex.sum().to(f32), tokens.to(f32)), 'loss': (torch.where(tokens > 0, loss.detach().to(f32) * tokens, 0.0), tokens.to(f32)), 'example_accuracy': ((c_ex[has].to(f32) / n_ex[has].to(f32)).sum(), has.sum().to(f32))}
        if self.answer_width is not None:
            exact_answers = []
            for example_preds, example_targets, example_mask in zip(preds, targets, mask):
                active_preds = example_preds[example_mask]
                active_targets = example_targets[example_mask]
                if active_targets.numel() % self.answer_width != 0:
                    raise ValueError(f'active target count {active_targets.numel()} is not divisible by answer_width={self.answer_width}')
                correct = (active_preds == active_targets).reshape(-1, self.answer_width)
                exact_answers.append(correct.all(dim=-1).to(f32))
            per_ex = [(e.sum(), e.numel()) for e in exact_answers if e.numel() > 0]
            stats['answer_exact_accuracy'] = (sum((e.sum() for e in exact_answers), torch.zeros((), device=logits.device)), torch.tensor(float(sum((e.numel() for e in exact_answers))), device=logits.device))
            stats['example_exact_accuracy'] = (sum((s / k for s, k in per_ex), torch.zeros((), device=logits.device)), torch.tensor(float(len(per_ex)), device=logits.device))
        return (loss, preds, targets, slices, stats)

    def _log_stats(self, stage: str, stats: dict, *, on_step: bool) -> None:
        for name, (num, den) in stats.items():
            metric = self.epoch_metrics[f'{stage}_{name}']
            if on_step:
                metric(num, den)
            else:
                metric.update(num, den)
            self.log(f'{stage}/{name}', metric, on_step=on_step, on_epoch=True, prog_bar=name in ('loss', 'accuracy'), metric_attribute='epoch_metrics.' + f'{stage}_{name}')

    def training_step(self, batch, batch_idx):
        loss, _preds, _targets, _slices, stats = self._shared_step(batch)
        self._log_stats('train', stats, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, slices, stats = self._shared_step(batch)
        self._log_stats('val', stats, on_step=False)
        self.validation_rows.extend(compute_metrics(preds.detach().cpu(), targets.detach().cpu(), slices, ignore_index=self.ignore_index))
        return {'val/loss': loss}

    def _log_slices(self, stage: str, rows: list) -> None:
        grouped = defaultdict(list)
        for row in rows:
            for key in self.slice_keys:
                if key in row:
                    grouped[key, row[key]].append(row['accuracy'])
        for (key, value), accuracies in grouped.items():
            metric_name = f'{stage}/{key}/accuracy-{value}'
            metric_value = sum(accuracies) / len(accuracies)
            self.log(metric_name, metric_value, prog_bar=False, on_step=False, on_epoch=True)
        rows.clear()

    def on_validation_epoch_end(self) -> None:
        self._log_slices('val', self.validation_rows)

    def test_step(self, batch, batch_idx):
        loss, preds, targets, slices, stats = self._shared_step(batch)
        self._log_stats('test', stats, on_step=False)
        self.test_rows.extend(compute_metrics(preds.detach().cpu(), targets.detach().cpu(), slices, ignore_index=self.ignore_index))
        return {'test/loss': loss}

    def on_test_epoch_end(self) -> None:
        self._log_slices('test', self.test_rows)

    def configure_optimizers(self):
        optimizer_spec = self.experiment_spec.optimizer
        optimizer_name = optimizer_spec.name.lower()
        optimizer_kwargs = dict(optimizer_spec.kwargs)
        optimizer_kwargs.setdefault('lr', float(self.experiment_spec.train.learning_rate))
        optimizer_kwargs.setdefault('weight_decay', float(self.experiment_spec.train.weight_decay))
        from experiments.lm.train.optim import make_param_groups
        wd = float(optimizer_kwargs.pop('weight_decay'))
        groups = make_param_groups(self, wd)
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(groups, **optimizer_kwargs)
        else:
            raise ValueError(f'Unsupported optimizer in MQMKVAR Lightning path: {optimizer_spec.name}')
        scheduler_spec = self.experiment_spec.scheduler
        if not scheduler_spec.name:
            return optimizer
        warmup_epochs = int(self.experiment_spec.train.lr_warmup_epochs or 0)
        total_epochs = int(self.experiment_spec.train.epochs or 1)
        scheduler_name = scheduler_spec.name.lower()
        scheduler_kwargs = dict(scheduler_spec.kwargs)
        if scheduler_name == 'cosineannealinglr':
            is_step = scheduler_spec.interval == 'step'
            if is_step:
                total_iters = int(self.experiment_spec.train.max_steps or 1)
            else:
                total_iters = total_epochs
            cosine_kwargs = dict(scheduler_kwargs)
            cosine_kwargs.setdefault('T_max', max(1, total_iters - warmup_epochs))
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cosine_kwargs)
            if warmup_epochs > 0 and (not is_step):
                warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-08, end_factor=1.0, total_iters=warmup_epochs)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
            else:
                scheduler = cosine_sched
        else:
            raise ValueError(f'Unsupported scheduler in MQMKVAR Lightning path: {scheduler_spec.name}')
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': scheduler_spec.interval, 'monitor': scheduler_spec.monitor}}
