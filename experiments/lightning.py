
from __future__ import annotations
import importlib
import os
import random
from typing import Any, Callable
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

def _load_symbol(import_path: str) -> Any:
    module_name, separator, attribute = import_path.partition(':')
    if not separator or not module_name or (not attribute):
        raise ValueError(f"Invalid import path '{import_path}'. Use 'package.module:attribute' format.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_name}' has no attribute '{attribute}'") from exc

def _call_builder(builder: Callable[..., Any], *args, **kwargs) -> Any:
    try:
        return builder(**kwargs)
    except TypeError as keyword_exc:
        try:
            return builder(*args)
        except TypeError:
            raise keyword_exc

def _default_count_trainable_parameters(model) -> int:
    return sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

def run_lightning_trial(spec, *, extra_callbacks=(), report_to_ray=True, resume_from_checkpoint=None) -> dict[str, Any]:
    if spec.integration is None:
        raise ValueError('Lightning backend requires ExperimentSpec.integration')
    if spec.train.epochs is None and spec.train.max_steps is None:
        raise ValueError('Lightning path requires TrainSpec.epochs or TrainSpec.max_steps')
    missing_fields = [field_name for field_name in ('model_builder', 'data_builder', 'task_builder') if not getattr(spec.integration, field_name)]
    if missing_fields:
        missing = ', '.join(missing_fields)
        raise ValueError(f'Lightning integration is missing required hooks: {missing}')
    data_builder = _load_symbol(spec.integration.data_builder)
    model_builder = _load_symbol(spec.integration.model_builder)
    task_builder = _load_symbol(spec.integration.task_builder)
    parameter_counter = _load_symbol(spec.integration.parameter_counter) if spec.integration.parameter_counter else _default_count_trainable_parameters
    datamodule = _call_builder(data_builder, spec.data, spec.train, spec.seeds, data_spec=spec.data, train_spec=spec.train, seed_spec=spec.seeds)
    datamodule.setup('fit')
    model_seed = int(spec.seeds.model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    model = _call_builder(model_builder, spec.model, spec.data, model_spec=spec.model, data_spec=spec.data)
    data_metadata = getattr(datamodule, 'metadata', {})
    module = _call_builder(task_builder, model, spec, data_metadata, model=model, experiment_spec=spec, data_metadata=data_metadata)
    if report_to_ray:
        from experiments.ray.lightning_callback import RayTuneReportCallback
    callbacks = ([RayTuneReportCallback(report_train_metrics_every_n_steps=spec.logger.report_train_metrics_every_n_steps)] if report_to_ray else []) + list(extra_callbacks)
    logger = False
    if spec.logger.wandb_mode != 'disabled':
        logger = WandbLogger(project=spec.logger.wandb_project, entity=spec.logger.wandb_entity, group=spec.logger.wandb_group, tags=list(spec.tags) + list(spec.logger.tags), mode=spec.logger.wandb_mode, name=spec.name)
    train_kwargs = dict(spec.train.kwargs)
    max_epochs = int(spec.train.epochs) if spec.train.epochs is not None else -1
    max_steps = int(spec.train.max_steps) if spec.train.max_steps is not None else -1
    use_gpu = spec.resources.gpus > 0
    if use_gpu and (not torch.cuda.is_available()):
        raise RuntimeError(f"spec requests {spec.resources.gpus} GPU(s) but torch.cuda.is_available()=False. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}. Check cluster GPU allocation.")
    if use_gpu:
        print(f"[lightning] GPU trial: cuda_avail={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')!r}, gpus_requested={spec.resources.gpus}", flush=True)
    use_ddp = spec.resources.ddp and spec.resources.gpus > 1
    trainer = Trainer(accelerator='gpu' if use_gpu else 'cpu', devices=spec.resources.gpus if use_gpu else 1, strategy='ddp' if use_ddp else 'auto', max_epochs=max_epochs, max_steps=max_steps, precision=train_kwargs.get('precision', '32-true'), log_every_n_steps=int(spec.logger.log_every_n_steps), val_check_interval=train_kwargs.get('val_check_interval', None), enable_checkpointing=bool(train_kwargs.get('enable_checkpointing', False)), default_root_dir=train_kwargs.get('default_root_dir', os.path.join('runs', 'lightning')), gradient_clip_val=train_kwargs.get('gradient_clip_val', None), gradient_clip_algorithm=train_kwargs.get('gradient_clip_algorithm', 'norm'), enable_progress_bar=bool(train_kwargs.get('enable_progress_bar', True)), callbacks=callbacks, logger=logger)
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

    def _scalarize(metric_dict):
        out = {}
        for key, value in metric_dict.items():
            if hasattr(value, 'item'):
                try:
                    value = value.item()
                except Exception:
                    continue
            if isinstance(value, (float, int, str, bool)):
                out[key] = value
        return out
    callback_metrics = _scalarize(trainer.callback_metrics)
    if getattr(datamodule, 'final_test_loader', None) is not None:
        trainer.test(module, datamodule=datamodule, verbose=False)
        callback_metrics.update(_scalarize(trainer.callback_metrics))
    try:
        param_result = _call_builder(parameter_counter, model, model=model, model_spec=spec.model, data_spec=spec.data)
    except TypeError:
        param_result = _default_count_trainable_parameters(model)
    if isinstance(param_result, dict):
        num_parameters = int(param_result.get('total', _default_count_trainable_parameters(model)))
        param_breakdown = {k: int(v) for k, v in param_result.items() if k != 'total'}
    else:
        num_parameters = int(param_result)
        param_breakdown = {}
    primary_metric = spec.task.primary_metric
    result = {'status': 'ok', 'primary_metric': primary_metric, 'primary_value': callback_metrics.get(primary_metric), 'best_metrics': callback_metrics, 'final_metrics': callback_metrics, 'epochs_ran': int(trainer.current_epoch + 1), 'global_step': int(trainer.global_step), 'num_parameters': num_parameters, 'dataset_metadata': data_metadata}
    for key, value in param_breakdown.items():
        result[f'num_{key}_parameters'] = value
    return result
