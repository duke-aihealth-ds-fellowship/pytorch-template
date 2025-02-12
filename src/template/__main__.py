import torch
from datasets import concatenate_datasets, load_dataset
from tomllib import load

from template.config import Config, get_device
from template.dataset import make_dataloaders
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.train import train_model
from template.tune import tune_hyperparameters


def main():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    cfg = Config(**cfg_data)
    cfg.trainer.device = get_device()
    cfg.data_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(cfg.random_state)
    torch.set_float32_matmul_precision("high")

    dataset = load_dataset(cfg.dataset.name)
    dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    cfg.model.output_dim = len(dataset.unique("label"))
    loaders = make_dataloaders(dataset=dataset, cfg=cfg)

    if cfg.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.train:
        train_model(cfg=cfg, loaders=loaders, use_best=True)
    if cfg.evaluate:
        results = evaluate_model(cfg=cfg, loader=loaders.test)
        print(results)
    if cfg.feature_importance:
        feature_importance(cfg=cfg, loaders=loaders)


if __name__ == "__main__":
    main()
