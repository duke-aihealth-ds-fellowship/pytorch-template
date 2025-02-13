import torch
from tomllib import load
from transformers import PreTrainedTokenizerFast

from template.config import Config
from template.dataset import get_splits, make_dataloaders
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.train import train_model
from template.tune import tune_hyperparameters


def main():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    cfg = Config(**cfg_data)

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    splits = get_splits(cfg=cfg)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer.path)
    cfg.model.output_dim = len(splits["train"].unique("label"))
    cfg.model.padding_idx = tokenizer.pad_token_id  # type: ignore
    loaders = make_dataloaders(splits=splits, tokenizer=tokenizer, cfg=cfg)

    if cfg.tune:
        tune_hyperparameters(loaders, cfg=cfg)
    if cfg.train:
        train_model(cfg=cfg, loaders=loaders, use_best=True)
    if cfg.evaluate:
        results = evaluate_model(cfg=cfg, loader=loaders.test)
        print(results)
    if cfg.feature_importance:
        shap_values = feature_importance(cfg=cfg, loaders=loaders)
        print(shap_values)


if __name__ == "__main__":
    main()
