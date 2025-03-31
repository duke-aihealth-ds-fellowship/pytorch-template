import torch

from template.config import get_config
from template.dataset import get_data, make_dataloaders
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.plots import plot
from template.train import train_model
from template.tune import tune_hyperparameters


def main():
    cfg = get_config()

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    data = get_data(cfg=cfg)
    loaders = make_dataloaders(data=data, cfg=cfg)

    if cfg.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.train:
        train_model(cfg=cfg, loaders=loaders)
    if cfg.evaluate:
        evaluate_model(cfg=cfg, loader=loaders.test)
    if cfg.importance:
        feature_importance(cfg=cfg, loader=loaders.test)
    if cfg.plot:
        plot(cfg=cfg)


if __name__ == "__main__":
    main()
