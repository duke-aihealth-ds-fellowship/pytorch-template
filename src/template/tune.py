import json

import optuna
import torch
from optuna import Trial
from optuna.samplers import QMCSampler, TPESampler

from template.config import Config
from template.dataset import DataLoaders
from template.model import set_hyperparameters
from template.train import make_trainer


class Objective:
    def __init__(self, loaders: DataLoaders, cfg: Config):
        self.loaders = loaders
        self.cfg = cfg
        self.best_eval_loss = float("inf")

    def sample_hyperparameters(self, trial: optuna.Trial):
        return {
            "max_epochs": trial.suggest_int(**self.cfg.hparams.max_epochs),
            "hidden_dim": trial.suggest_int(**self.cfg.hparams.hidden_dim),
            "lr": trial.suggest_float(**self.cfg.hparams.lr),
            "weight_decay": trial.suggest_float(**self.cfg.hparams.weight_decay),
        }

    def __call__(self, trial: Trial) -> float:
        hyperparams = self.sample_hyperparameters(trial=trial)
        cfg = set_hyperparameters(cfg=self.cfg, **hyperparams)
        trainer = make_trainer(cfg=cfg)
        trainer.train(self.loaders.train, self.loaders.val)
        if trainer.eval_loss < self.best_eval_loss:
            self.best_eval_loss = trainer.eval_loss
            with open(self.cfg.hparams.path, "w") as f:
                json.dump(hyperparams, f)
            torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
        return trainer.eval_loss


def make_study(cfg: Config, sampler: optuna.samplers.BaseSampler) -> optuna.study.Study:
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )
    return optuna.create_study(
        storage=storage,
        sampler=sampler,
        direction=cfg.tuner.direction,
        study_name="tune",
    )


def tune_hyperparameters(loaders: DataLoaders, cfg: Config):
    objective = Objective(cfg=cfg, loaders=loaders)
    half_trials = cfg.tuner.n_trials // 2
    sampler = QMCSampler(seed=cfg.main.seed)
    study = make_study(cfg=cfg, sampler=sampler)
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.main.seed)
    study.optimize(func=objective, n_trials=half_trials)
    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=4))


def load_best_checkpoint(cfg: Config, model_class):
    with open(cfg.hparams.path, "r") as file:
        hyperparams = json.load(file)
    cfg = set_hyperparameters(cfg=cfg, **hyperparams)
    with torch.device("meta"):
        model = model_class(**cfg.model.model_dump())
    model_weights = torch.load(cfg.tuner.checkpoint, weights_only=True, mmap=True)
    model.load_state_dict(model_weights, assign=True)
    model.to(cfg.trainer.device)
    return model
