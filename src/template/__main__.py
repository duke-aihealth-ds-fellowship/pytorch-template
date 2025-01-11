import torch
from tomllib import load

from template.config import Config
from template.dataset import make_dataloaders
from template.evaluate import evaluate_model
from template.examples import make_fake_sequence_dataset
from template.train import make_trainer
from template.tune import tune_hyperparameters


def main():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    cfg = Config(**cfg_data)
    cfg.data_dir.mkdir(exist_ok=True, parents=True)
    df = make_fake_sequence_dataset()
    dataloaders = make_dataloaders(data=df, cfg=cfg)
    cfg.model.output_dim = df["label"].n_unique()
    if cfg.tune:
        tune_hyperparameters(dataloaders, cfg=cfg)
    if cfg.train:
        trainer = make_trainer(cfg=cfg, dataloaders=dataloaders)
        trainer.train()
        torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
    if cfg.evaluate:
        evaluate_model(cfg=cfg, dataloader=dataloaders.test)


if __name__ == "__main__":
    main()
