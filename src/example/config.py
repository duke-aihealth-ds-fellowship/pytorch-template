from pydantic import BaseModel


class Model(BaseModel):
    embedding_dim: int
    hidden_dim: int
    n_layers: int
    output_dim: int


class Optimizer(BaseModel):
    lr: float
    momentum: float
    weight_decay: float


class Training(BaseModel):
    max_epochs: int
    batch_size: int
    gradient_clip: float


class Config(BaseModel):
    random_seed: int
    tune: bool
    evaluate: bool
    n_trials: int
    train_size: float
    n_bootstraps: int
    padding_idx: int
    num_workers: int
    device: str
    model: Model
    optimizer: Optimizer
    training: Training
