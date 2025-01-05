import json

import torch
import torch.nn as nn
from optuna import Trial, create_study
from optuna.samplers import TPESampler

from template.config import Config
from template.dataset import DataLoaders
from template.train import Trainer


def sample_hyperparameters(trial: Trial, cfg: Config):
    cfg.model.hidden_dim = trial.suggest_categorical(**cfg.hparams.hidden_dim)
    cfg.model.n_layers = trial.suggest_int(**cfg.hparams.n_layers)
    cfg.optimizer.lr = trial.suggest_float(**cfg.hparams.lr)
    cfg.optimizer.weight_decay = trial.suggest_float(**cfg.hparams.weight_decay)
    cfg.optimizer.momentum = trial.suggest_float(**cfg.hparams.momentum)
    return cfg


class Objective:
    def __init__(self, dataloaders: DataLoaders, cfg: Config):
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.best_validation_loss = float("inf")

    def __call__(self, trial: Trial) -> float:
        cfg = sample_hyperparameters(trial, self.cfg.hparams)
        trainer = Trainer(cfg=cfg, dataloaders=self.dataloaders)
        validation_loss = trainer.train()
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.save_checkpoint(trainer.model, cfg)
        return validation_loss

    def save_checkpoint(self, model: nn.Module):
        hyperparameters = self.cfg.model.model_dump()
        with open(self.cfg.tuner.best_hyperparameters, "w") as f:
            json.dump(hyperparameters, f)
        torch.save(model.state_dict(), self.cfg.tuner.best_checkpoint)


def tune_hyperparameters(dataloaders: DataLoaders, cfg: Config):
    sampler = TPESampler(seed=cfg.random_state)
    study = create_study(
        sampler=sampler, direction=cfg.tuner.direction, study_name="tune"
    )
    objective = Objective(dataloaders=dataloaders, cfg=cfg)
    study.optimize(func=objective, n_trials=cfg.tuner.n_trials)
    if cfg.verbose:
        print("Best model hyperparameters:")
        print(json.dumps(study.best_params, indent=4))
        print(f"Best model checkpoint saved in: {cfg.tuner.best_checkpoint}")


def load_best_checkpoint(cfg: Config, model_class: type[nn.Module]) -> nn.Module:
    model_weights = torch.load(cfg.tuner.best_checkpoint, weights_only=True)
    hyperparameters = json.loads(cfg.tuner.best_hyperparameters)
    model = model_class(**hyperparameters)
    return model.load_state_dict(model_weights)
