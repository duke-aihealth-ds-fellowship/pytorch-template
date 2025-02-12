from pathlib import Path

import torch
from pydantic import BaseModel


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ModelConfig(BaseModel):
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    n_layers: int
    padding_idx: int
    output_dim: int = -1  # set at run time


class OptimizerConfig(BaseModel):
    lr: float
    momentum: float
    weight_decay: float


class TrainerConfig(BaseModel):
    max_epochs: int
    gradient_clip: float
    eval_every_n_epochs: int
    device: str | None = None  # set at initialization


class TunerConfig(BaseModel):
    n_trials: int
    prune: bool
    checkpoint: Path
    hyperparameters: Path
    hidden_dim: dict
    n_layers: dict
    lr: dict
    weight_decay: dict


class EvaluatorConfig(BaseModel):
    n_bootstraps: int
    aggregate: bool


class Config(BaseModel):
    random_state: int
    verbose: bool
    train: bool
    tune: bool
    evaluate: bool
    importance: bool
    train_size: float
    data_dir: Path
    combine_train_val: bool
    model: ModelConfig
    optimizer: OptimizerConfig
    dataloader: DataLoaderConfig
    trainer: TrainerConfig
    tuner: TunerConfig
    evaluator: EvaluatorConfig


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
