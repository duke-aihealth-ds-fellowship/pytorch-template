import torch
from tomllib import load

from template.config import Config
from template.dataset import make_dataloaders
from template.evaluate import evaluate_model
from template.examples import make_fake_sequence_dataset
from template.train import train_model
from template.tune import tune_hyperparameters


def main():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    cfg = Config(**cfg_data)
    torch.manual_seed(cfg.random_state)
    torch.set_float32_matmul_precision("high")

    cfg.data_dir.mkdir(exist_ok=True, parents=True)
    df = make_fake_sequence_dataset()
    loaders = make_dataloaders(data=df, cfg=cfg)

    cfg.model.output_dim = df["label"].n_unique()
    if cfg.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.train:
        train_model(cfg=cfg, loaders=loaders, use_best=True)
    if cfg.evaluate:
        results = evaluate_model(cfg=cfg, loader=loaders.test)
        print(results)
    if cfg.importance:
        pass  # TODO


if __name__ == "__main__":
    main()
