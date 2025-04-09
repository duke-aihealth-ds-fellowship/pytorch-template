import torch

from template.config import get_config
from template.dataset import make_dataloaders
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.plots import plot
from template.simulation import simulate
from template.tune import train_model, tune_hyperparameters
from template.uncertainty import quantify_uncertainty


def main():
    cfg = get_config()

    cfg.path.init_paths()

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    if cfg.simulate or cfg.path:
        simulate(cfg=cfg)

    loaders = make_dataloaders(cfg=cfg)

    if cfg.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.train:
        train_model(cfg=cfg, loaders=loaders)
    if cfg.evaluate:
        evaluate_model(cfg=cfg, loader=loaders.test)
    if cfg.uq:
        quantify_uncertainty(cfg=cfg, loaders=loaders)
    if cfg.importance:
        feature_importance(cfg=cfg, loader=loaders.test)
    if cfg.plot:
        plot(cfg=cfg)


if __name__ == "__main__":
    main()
