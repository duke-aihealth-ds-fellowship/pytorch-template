from pathlib import Path
from typing import Any, Literal

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
    path: Path


class AttributionConfig(BaseModel):
    num_samples: int
    path: Path


class PlotConfig(BaseModel):
    path: str
    style: Literal["white", "dark", "whitegrid", "darkgrid"]
    font_scale: float
    palette: str
    importance: str


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
    use_best: bool
    train: bool
    evaluate: bool
    importance: bool
    plot: bool
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
    attribution: AttributionConfig
    plots: PlotConfig

    # TODO automate path construction
    def model_post_init(self, __context: Any) -> None:
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.tokenizer.path = str(self.data_dir) + "/" + self.tokenizer.path
        self.dataset.path = str(self.data_dir) + "/" + self.dataset.name
        self.tuner.hparams_path = self.data_dir / self.tuner.hparams_path
        self.tuner.checkpoint = self.data_dir / self.tuner.checkpoint
        self.evaluator.path = self.data_dir / self.evaluator.path
        self.plots.importance = str(
            self.data_dir / self.plots.path / self.plots.importance
        )
        self.attribution.path = self.data_dir / self.attribution.path
        self.model.vocab_size = self.tokenizer.vocab_size
        self.trainer.device = get_device()
