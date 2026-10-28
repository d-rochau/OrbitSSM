
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any

@dataclass(frozen=True)
class SeedSpec:
    model_seed: int
    data_seed: int

@dataclass(frozen=True)
class ResourceSpec:
    gpus: int = 1
    cpus: int = 4
    nodes: int = 1
    ddp: bool = False
    ray_init_mode: str = 'local'
    ray_address: str | None = None
    ray_node_ip_address: str | None = None
    expected_cluster_gpus: int | None = None
    expected_cluster_nodes: int | None = None
    cluster_ready_timeout_s: float | None = None

@dataclass(frozen=True)
class ModelSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DataSpec:
    task: str
    mode: str
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class IntegrationSpec:
    model_builder: str
    data_builder: str
    task_builder: str
    parameter_counter: str | None = None

@dataclass(frozen=True)
class TaskSpec:
    name: str
    loss: str = 'ce'
    metrics: tuple[str, ...] = ('loss',)
    primary_metric: str = 'val/loss'
    primary_mode: str = 'min'
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class OptimizerSpec:
    name: str = 'AdamW'
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SchedulerSpec:
    name: str | None = None
    interval: str = 'epoch'
    monitor: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class LoggerSpec:
    wandb_mode: str = 'disabled'
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    log_every_n_steps: int = 50
    report_train_metrics_every_n_steps: int | None = None
    tags: tuple[str, ...] = ()

@dataclass(frozen=True)
class TrainSpec:
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    epochs: int | None = None
    max_steps: int | None = None
    lr_warmup_epochs: int = 0
    lr_warmup_steps: int = 0
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs is None and self.max_steps is None:
            raise ValueError('TrainSpec needs either epochs or max_steps')
        if self.epochs is not None and self.max_steps is not None:
            raise ValueError('TrainSpec should use epochs or max_steps, not both')

@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model: ModelSpec
    data: DataSpec
    train: TrainSpec
    seeds: SeedSpec
    integration: IntegrationSpec | None = None
    task: TaskSpec = field(default_factory=lambda: TaskSpec(name='supervised'))
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = field(default_factory=SchedulerSpec)
    logger: LoggerSpec = field(default_factory=LoggerSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def expand_grid(base_name: str, axes: dict[str, list[Any]], builder) -> list[ExperimentSpec]:
    if not axes:
        return [builder(name=base_name)]
    keys = list(axes.keys())
    experiments: list[ExperimentSpec] = []
    for values in product(*(axes[key] for key in keys)):
        point = dict(zip(keys, values))
        suffix = '_'.join((f'{key}{value}' for key, value in point.items()))
        experiments.append(builder(name=f'{base_name}_{suffix}', **point))
    return experiments
