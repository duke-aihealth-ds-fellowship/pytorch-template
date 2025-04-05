from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel
from tomllib import load


class PathConfig(BaseModel):
    dataset: Path
    hyperparameters: Path
    checkpoint: Path
    logs: Path
    metrics: Path
    attribution: Path
    importance_plot: Path
    metrics_plot: Path

    def init_paths(self) -> None:
        for field_value in self.__dict__.values():
            if isinstance(field_value, Path):
                if field_value.suffix:
                    field_value.parent.mkdir(parents=True, exist_ok=True)
                else:
                    field_value.mkdir(parents=True, exist_ok=True)


class SimulationConfig(BaseModel):
    intercept: float
    d_features: int
    scale: float
    n_samples: int
    m_timepoints: int
    variance: float
    noise: float
    gamma_shape: float
    gamma_rate: float


class DatasetConfig(BaseModel):
    train_size: float
    stratify: str


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ModelConfig(BaseModel):
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    compile: bool
    output_dim: int = -1  # set at run time


class LossConfig(BaseModel):
    ignore_index: int
    reduction: str


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float


class SchedulerConfig(BaseModel):
    gamma: float


class TrainerConfig(BaseModel):
    max_epochs: int
    gradient_clip: float
    device: str


class TunerConfig(BaseModel):
    n_trials: int
    direction: str


class HParamsConfig(BaseModel):
    path: Path
    max_epochs: dict
    hidden_dim: dict
    num_heads: dict
    num_layers: dict
    dropout: dict
    lr: dict
    weight_decay: dict


class EvaluatorConfig(BaseModel):
    n_bootstraps: int


class PlotConfig(BaseModel):
    style: Literal["white", "dark", "whitegrid", "darkgrid"]
    font_scale: float
    palette: str


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Config(BaseModel):
    seed: int
    dev_run: bool
    task: str
    simulate: bool
    tune: bool
    use_best: bool
    train: bool
    evaluate: bool
    importance: bool
    plot: bool
    compile: bool
    proportions: list[float]
    path: PathConfig
    simulation: SimulationConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    tuner: TunerConfig
    hparams: HParamsConfig
    evaluator: EvaluatorConfig
    plots: PlotConfig

    def set_dev_run(self) -> None:
        self.trainer.max_epochs = 1
        self.tuner.n_trials = 2
        self.hparams.max_epochs["low"] = 1
        self.hparams.max_epochs["high"] = 1
        self.evaluator.n_bootstraps = 3

    def model_post_init(self, __context: Any) -> None:
        if self.trainer.device == "auto":
            self.trainer.device = get_device()
        if self.dev_run:
            self.set_dev_run()


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
