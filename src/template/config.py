from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel


class TokenizerConfig(BaseModel):
    path: str
    vocab_size: int
    max_length: int


class DatasetConfig(BaseModel):
    name: str
    path: str
    train_size: float
    stratify: str


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ModelConfig(BaseModel):
    compile: bool
    embedding_dim: int
    hidden_dim: int
    vocab_size: int = -1  # set at run time
    padding_idx: int = -1  # set at run time
    output_dim: int = -1  # set at run time


class OptimizerConfig(BaseModel):
    lr: float
    momentum: float
    weight_decay: float


class TrainerConfig(BaseModel):
    max_epochs: int
    gradient_clip: float
    device: str = "cpu"  # set at initialization


class TunerConfig(BaseModel):
    n_trials: int
    direction: str
    checkpoint: Path
    hparams_path: Path


class HParamsConfig(BaseModel):
    max_epochs: dict
    hidden_dim: dict
    lr: dict
    weight_decay: dict


class EvaluatorConfig(BaseModel):
    n_bootstraps: int
    aggregate: bool


class ImportanceConfig(BaseModel):
    num_samples: int
    plot_path: Path


class PlotConfig(BaseModel):
    path: Path
    style: str
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
    regenerate: bool
    tune: bool
    train: bool
    evaluate: bool
    feature_importance: bool
    data_dir: Path
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    tuner: TunerConfig
    hparams: HParamsConfig
    evaluator: EvaluatorConfig
    importance: ImportanceConfig
    plots: PlotConfig

    def model_post_init(self, __context: Any) -> None:
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.dataset.path = str(self.data_dir) + "/" + self.dataset.name
        self.tuner.hparams_path = self.data_dir / self.tuner.hparams_path
        self.tuner.checkpoint = self.data_dir / self.tuner.checkpoint
        self.importance.plot_path = self.data_dir / self.importance.plot_path
        self.tokenizer.path = str(self.data_dir) + "/" + self.tokenizer.path
        self.plots.path = self.data_dir / self.plots.path
        self.model.vocab_size = self.tokenizer.vocab_size
        self.trainer.device = get_device()
