from functools import partial
import json

from optuna import create_study, Trial
from optuna.samplers import TPESampler
import torch.nn as nn

from example.config import Config
from example.dataset import Splits
from example.evaluate import evaluate_model
from example.train import train_model


def sample_hyperparameters(trial: Trial, config: Config):
    config.model.hidden_dim = trial.suggest_categorical(
        name="hidden_dim", choices=[16, 32, 64]
    )
    config.model.n_layers = trial.suggest_int(name="n_layers", low=1, high=6)
    config.optimizer.lr = trial.suggest_float(name="lr", low=1e-5, high=1e-2)
    config.optimizer.weight_decay = trial.suggest_float(
        name="weight_decay", low=1e-10, high=1e-2
    )
    config.optimizer.momentum = trial.suggest_float(name="momentum", low=0.9, high=0.99)
    return config


def objective(trial: Trial, dataloaders: Splits, config: Config) -> float:
    config = sample_hyperparameters(trial, config)
    model = train_model(dataloaders=dataloaders, config=config)
    metrics = {"val_loss": nn.CrossEntropyLoss()}
    val_loss = evaluate_model(model=model, dataloader=dataloaders.val, metrics=metrics)
    return val_loss["val_loss"].item()


def tune_model(dataloaders: Splits, config: Config):
    sampler = TPESampler(seed=config.random_seed)
    study = create_study(sampler=sampler, direction="minimize", study_name="ABCD")
    objective_function = partial(objective, dataloaders=dataloaders, config=config)
    study.optimize(func=objective_function, n_trials=config.tuner.n_trials)
    if config.verbose:
        print("Best model hyperparameters:\n")
        print(json.dumps(study.best_params, indent=4))
        print(f"Best model checkpoint saved in: {config.checkpoint_path}")
