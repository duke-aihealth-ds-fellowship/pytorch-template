from pathlib import Path
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    embedding_dim: int
    hidden_dim: int
    n_layers: int
    output_dim: int
    padding_idx: int
    vocab_size: int | None = Field(default=None)


class OptimizerConfig(BaseModel):
    lr: float
    momentum: float
    weight_decay: float


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool


class TrainerConfig(BaseModel):
    max_epochs: int
    gradient_clip: float
    device: str


class TunerConfig(BaseModel):
    n_trials: int


class EvaluatorConfig(BaseModel):
    n_bootstraps: int


class Config(BaseModel):
    random_seed: int
    verbose: bool
    train: bool
    tune: bool
    evaluate: bool
    train_size: float
    checkpoint_path: Path
    model: ModelConfig
    optimizer: OptimizerConfig
    dataloader: DataLoaderConfig
    trainer: TrainerConfig
    tuner: TunerConfig
    evaluator: EvaluatorConfig
