import torch

from template.config import get_config
from template.dataset import get_splits, make_dataloaders
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.plots import plot
from template.train import train_model
from template.tune import tune_hyperparameters


def main():
    cfg = get_config()

    torch.manual_seed(cfg.main.seed)
    torch.set_float32_matmul_precision("high")

    splits = get_splits(cfg=cfg)
    loaders = make_dataloaders(splits=splits, cfg=cfg)

    if cfg.main.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.main.train:
        train_model(cfg=cfg, loaders=loaders)
    if cfg.main.evaluate:
        evaluate_model(cfg=cfg, loader=loaders.test)
    if cfg.main.importance:
        feature_importance(cfg=cfg, loader=loaders.test)
    if cfg.main.plot:
        plot(cfg=cfg)


if __name__ == "__main__":
    main()
