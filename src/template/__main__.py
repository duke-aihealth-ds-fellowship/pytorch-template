import torch
from tomllib import load

from template.config import Config, get_device
from template.dataset import make_dataloaders, make_splits
from template.evaluate import evaluate_model
from template.importance import feature_importance
from template.tokenize import get_tokenizer, tokenize_dataset
from template.train import train_model
from template.tune import tune_hyperparameters

# https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling
# from transformers import DataCollatorForLanguageModeling

# tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


def main():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    cfg = Config(**cfg_data)
    cfg.trainer.device = get_device()
    cfg.data_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    splits = make_splits(cfg=cfg)
    tokenizer = get_tokenizer(cfg=cfg, dataset=splits["train"])
    splits = tokenize_dataset(tokenizer=tokenizer, splits=splits, cfg=cfg)
    cfg.model.output_dim = len(splits["train"].unique("label"))
    cfg.model.vocab_size = tokenizer.vocab_size
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
        feature_importance(cfg=cfg, loaders=loaders)


if __name__ == "__main__":
    main()
