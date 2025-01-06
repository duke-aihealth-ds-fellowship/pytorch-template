import json

import torch
import torch.nn as nn
from optuna import Trial, create_study
from optuna.samplers import TPESampler

from template.config import Config
from template.dataset import DataLoaders
from template.train import make_trainer


class Objective:
    def __init__(self, dataloaders: DataLoaders, cfg: Config):
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.best_validation_loss = float("inf")

    def sample_hyperparameters(self, trial: Trial):
        cfg = self.cfg.model_copy(deep=True)
        cfg.model.hidden_dim = trial.suggest_categorical(**cfg.tuner.hidden_dim)
        cfg.model.n_layers = trial.suggest_int(**cfg.tuner.n_layers)
        cfg.optimizer.lr = trial.suggest_float(**cfg.tuner.lr)
        cfg.optimizer.weight_decay = trial.suggest_float(**cfg.tuner.weight_decay)
        cfg.optimizer.momentum = trial.suggest_float(**cfg.tuner.momentum)
        return cfg

    def save_checkpoint(self, cfg: Config, model: nn.Module):
        hyperparameters = cfg.model.model_dump()
        with open(cfg.tuner.hyperparameters, "w") as f:
            json.dump(hyperparameters, f)
        torch.save(model.state_dict(), cfg.tuner.checkpoint)

    def __call__(self, trial: Trial) -> float:
        cfg = self.sample_hyperparameters(trial=trial)
        trainer = make_trainer(cfg=cfg, dataloaders=self.dataloaders)
        validation_loss = trainer.train()
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.save_checkpoint(cfg=cfg, model=trainer.model)
        return validation_loss


def tune_hyperparameters(dataloaders: DataLoaders, cfg: Config):
    sampler = TPESampler(seed=cfg.random_state)
    study = create_study(sampler=sampler, direction="minimize", study_name="tune")
    objective = Objective(dataloaders=dataloaders, cfg=cfg)
    study.optimize(func=objective, n_trials=cfg.tuner.n_trials)
    if cfg.verbose:
        print("Best model hyperparameters:")
        print(json.dumps(study.best_params, indent=4))
        print(f"Best model checkpoint saved to: {cfg.tuner.checkpoint}")
        print(f"Best model hyperparameters saved to: {cfg.tuner.hyperparameters}")


def load_best_checkpoint(cfg: Config, model_class: type[nn.Module]) -> nn.Module:
    model_weights = torch.load(cfg.tuner.checkpoint, weights_only=True)
    with open(cfg.tuner.hyperparameters, "r") as f:
        hyperparameters = json.load(f)
    model: nn.Module = model_class(**hyperparameters)
    model.load_state_dict(model_weights)
    model.to(cfg.trainer.device)
    return model
