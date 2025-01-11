import json

import optuna
import torch
import torch.nn as nn
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler

from template.config import Config
from template.dataset import DataLoaders
from template.train import make_trainer


class Objective:
    def __init__(self, dataloaders: DataLoaders, cfg: Config):
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.best_validation_loss = float("inf")

    def sample_hyperparameters(self, trial: optuna.Trial):
        cfg = self.cfg.model_copy(deep=True)
        cfg.model.hidden_dim = 2 ** trial.suggest_int(**cfg.tuner.hidden_dim)
        cfg.model.n_layers = trial.suggest_int(**cfg.tuner.n_layers)
        cfg.optimizer.lr = trial.suggest_float(**cfg.tuner.lr)
        cfg.optimizer.weight_decay = trial.suggest_float(**cfg.tuner.weight_decay)
        return cfg

    def __call__(self, trial: Trial) -> float:
        cfg = self.sample_hyperparameters(trial=trial)
        trainer = make_trainer(cfg=cfg, dataloaders=self.dataloaders)
        validation_loss = trainer.train()
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            hyperparameters = cfg.model.model_dump()
            with open(cfg.tuner.hyperparameters, "w") as f:
                json.dump(hyperparameters, f)
            torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
        return validation_loss


def make_study(cfg: Config):
    sampler = QMCSampler(seed=cfg.random_state)
    if cfg.tuner.prune:
        pruner = HyperbandPruner(
            min_resource=cfg.trainer.max_epochs // 4,
            max_resource=cfg.trainer.max_epochs,
        )
    else:
        pruner = None
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )
    return optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="tune",
    )


def tune_hyperparameters(dataloaders: DataLoaders, cfg: Config):
    objective = Objective(cfg=cfg, dataloaders=dataloaders)
    half_trials = cfg.tuner.n_trials // 2
    study = make_study(cfg=cfg)
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.random_state)
    study.optimize(func=objective, n_trials=half_trials)
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
