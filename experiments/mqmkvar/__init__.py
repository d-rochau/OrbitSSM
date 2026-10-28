
__all__ = ['MQMKVARDataModule', 'MQMKVARSupervisedTaskModule', 'build_mqmkvar_datamodule', 'build_mqmkvar_task_module', 'build_mqmkvar_model', 'count_trainable_parameters', 'mqmkvar_lightning_integration', 'run_mqmkvar_lightning_trial']

def __getattr__(name):
    if name == 'MQMKVARDataModule':
        from .data import MQMKVARDataModule
        return MQMKVARDataModule
    if name in {'build_mqmkvar_model', 'count_trainable_parameters'}:
        from .models import build_mqmkvar_model, count_trainable_parameters
        return {'build_mqmkvar_model': build_mqmkvar_model, 'count_trainable_parameters': count_trainable_parameters}[name]
    if name == 'mqmkvar_lightning_integration':
        from .presets import mqmkvar_lightning_integration
        return mqmkvar_lightning_integration
    if name == 'run_mqmkvar_lightning_trial':
        from .pipeline import run_mqmkvar_lightning_trial
        return run_mqmkvar_lightning_trial
    if name in {'build_mqmkvar_datamodule', 'build_mqmkvar_task_module'}:
        from .pipeline import build_mqmkvar_datamodule, build_mqmkvar_task_module
        return {'build_mqmkvar_datamodule': build_mqmkvar_datamodule, 'build_mqmkvar_task_module': build_mqmkvar_task_module}[name]
    if name == 'MQMKVARSupervisedTaskModule':
        from .task import MQMKVARSupervisedTaskModule
        return MQMKVARSupervisedTaskModule
    raise AttributeError(name)
