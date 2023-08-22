from pydantic import BaseModel


class Model(BaseModel):
    n_layers: int
    hidden_dim: int


class Optimizer(BaseModel):
    learning_rate: float
    momentum: float


class Config(BaseModel):
    random_seed: int
    model: Model
    optimizer: Optimizer
