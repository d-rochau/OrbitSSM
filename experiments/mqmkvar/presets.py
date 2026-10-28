
from __future__ import annotations
from experiments.specs import IntegrationSpec

def mqmkvar_lightning_integration() -> IntegrationSpec:
    return IntegrationSpec(model_builder='experiments.models.zoo:build_mqmkvar_model', data_builder='experiments.mqmkvar.pipeline:build_mqmkvar_datamodule', task_builder='experiments.mqmkvar.pipeline:build_mqmkvar_task_module', parameter_counter='experiments.models.zoo:count_parameter_breakdown')
