import json

import optuna
import torch
import torch.nn as nn
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler

from template.config import Config, TunerConfig
from template.dataset import DataLoaders
from template.model import set_hyperparameters
from template.train import make_trainer


class Objective:
    def __init__(self, loaders: DataLoaders, cfg: Config):
        self.loaders = loaders
        self.cfg = cfg
        self.best_validation_loss = float("inf")

    def sample_hyperparameters(self, trial: optuna.Trial):
        cfg: TunerConfig = self.cfg.tuner.model_copy(deep=True)
        return {
            "hidden_dim": trial.suggest_int(**cfg.hidden_dim),
            "n_layers": trial.suggest_int(**cfg.n_layers),
            "lr": trial.suggest_float(**cfg.lr),
            "weight_decay": trial.suggest_float(**cfg.weight_decay),
        }

    def __call__(self, trial: Trial) -> float:
        hyperparams = self.sample_hyperparameters(trial=trial)
        cfg = set_hyperparameters(cfg=self.cfg, **hyperparams)
        trainer = make_trainer(cfg=cfg, dataloaders=self.dataloaders)
        trainer.train(self.loaders.train, self.loaders.val)
        if trainer.eval_loss < self.best_validation_loss:
            self.best_validation_loss = trainer.validation_loss
            with open(self.cfg.tuner.hyperparameters, "w") as f:
                json.dump(hyperparams, f)
            torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
        return trainer.eval_loss


def make_study(cfg: Config, sampler: optuna.samplers.BaseSampler) -> optuna.study.Study:
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
    sampler = QMCSampler(seed=cfg.random_state)
    study = make_study(cfg=cfg, sampler=sampler)
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.random_state)
    study.optimize(func=objective, n_trials=half_trials)
    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=4))


def load_best_checkpoint(cfg: Config, model_class: type[nn.Module]) -> nn.Module:
    model_weights = torch.load(cfg.tuner.checkpoint, weights_only=True)
    with open(cfg.tuner.hyperparameters, "r") as file:
        hyperparams = json.load(file)
    cfg = set_hyperparameters(cfg=cfg, **hyperparams)
    model: nn.Module = model_class(**cfg.model.model_dump(exclude={"compile"}))
    model.load(model_weights)
    model.to(cfg.trainer.device)
    return model
