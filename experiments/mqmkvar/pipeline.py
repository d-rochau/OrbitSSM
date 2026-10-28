
from __future__ import annotations
from dataclasses import replace
from typing import Any
from experiments.lightning import run_lightning_trial
from experiments.mqmkvar.data import MQMKVARDataModule
from experiments.mqmkvar.presets import mqmkvar_lightning_integration
from experiments.mqmkvar.task import MQMKVARSupervisedTaskModule

def build_mqmkvar_datamodule(*, data_spec, train_spec, seed_spec):
    return MQMKVARDataModule(data_spec, train_spec, seed_spec)

def build_mqmkvar_task_module(*, model, experiment_spec, data_metadata: dict[str, Any]):
    return MQMKVARSupervisedTaskModule(model=model, experiment_spec=experiment_spec, data_metadata=data_metadata)

def run_mqmkvar_lightning_trial(spec) -> dict[str, Any]:
    if spec.integration is None:
        spec = replace(spec, integration=mqmkvar_lightning_integration())
    return run_lightning_trial(spec)
