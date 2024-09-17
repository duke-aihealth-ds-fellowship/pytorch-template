from functools import partial
import json

from optuna import create_study, Trial
from optuna.samplers import TPESampler
import torch

from template.checkpoint import get_best_checkpoint_path
from template.config import Config
from template.dataset import DataLoaders
from template.train import train_model


def sample_hyperparameters(trial: Trial, config: Config):
    config.model.hidden_dim = trial.suggest_categorical(**config.hparams.hidden_dim)
    config.model.n_layers = trial.suggest_int(**config.hparams.n_layers)
    config.optimizer.lr = trial.suggest_float(**config.hparams.lr)
    config.optimizer.weight_decay = trial.suggest_float(**config.hparams.weight_decay)
    config.optimizer.momentum = trial.suggest_float(**config.hparams.momentum)
    return config


def objective(trial: Trial, dataloaders: DataLoaders, config: Config) -> float:
    config = sample_hyperparameters(trial, config)
    train_model(dataloaders=dataloaders, config=config)
    best_checkpoint_path = get_best_checkpoint_path(config.checkpoint)
    checkpoint = torch.load(best_checkpoint_path, weights_only=True)
    return checkpoint["val_loss"].item()


def tune_model(dataloaders: DataLoaders, config: Config):
    sampler = TPESampler(seed=config.random_state)
    study = create_study(
        sampler=sampler, direction=config.checkpoint.mode, study_name="tune"
    )
    objective_function = partial(objective, dataloaders=dataloaders, config=config)
    study.optimize(func=objective_function, n_trials=config.tuner.n_trials)
    if config.verbose:
        print("Best model hyperparameters:")
        print(json.dumps(study.best_params, indent=4))
        print(f"Best model checkpoint saved in: {config.checkpoint.path}")
