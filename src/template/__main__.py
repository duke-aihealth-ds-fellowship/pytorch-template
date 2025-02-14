import torch
from tomllib import load

from template.config import Config
from template.dataset import get_splits, make_dataloaders
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.plots import plot_attributions
from template.train import train_model
from template.tune import tune_hyperparameters


def main():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    cfg = Config(**cfg_data)

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    splits = get_splits(cfg=cfg)
    loaders = make_dataloaders(splits=splits, cfg=cfg)

    if cfg.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.train:
        train_model(cfg=cfg, loaders=loaders)
    if cfg.evaluate:
        evaluate_model(cfg=cfg, loader=loaders.test)
    if cfg.importance:
        feature_importance(cfg=cfg, loaders=loaders)
    if cfg.plot:
        plot_attributions(cfg=cfg)


if __name__ == "__main__":
    main()
