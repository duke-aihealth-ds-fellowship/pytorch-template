from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

from template.config import Config
from template.tokenizer import make_tokenizer, tokenize_dataset


def make_splits(cfg: Config) -> DatasetDict:
    dataset = load_dataset(cfg.dataset.name)
    dataset_dict = cast(DatasetDict, dataset)
    dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["test"]])
    if cfg.dev_run:
        dataset = dataset.select(range(cfg.dataloader.batch_size * 10))
    split = partial(
        Dataset.train_test_split, seed=cfg.seed, stratify_by_column=cfg.dataset.stratify
    )
    train_val_test = split(dataset, train_size=cfg.dataset.train_size)
    val_test = split(train_val_test["test"], test_size=0.5)
    splits = DatasetDict(
        {
            "train": train_val_test["train"],
            "val": val_test["train"],
            "test": val_test["test"],
        }
    )
    splits["train_val"] = concatenate_datasets([splits["train"], splits["val"]])
    return splits


def get_splits(cfg: Config) -> DatasetDict:
    if not Path(cfg.dataset.path).exists() or cfg.regenerate:
        splits = make_splits(cfg=cfg)
        tokenizer = make_tokenizer(cfg=cfg, dataset=splits["train"])
        splits = tokenize_dataset(tokenizer=tokenizer, splits=splits, cfg=cfg)
    else:
        splits = DatasetDict.load_from_disk(cfg.dataset.path)
    return splits


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_val: DataLoader


def make_dataloaders(
    splits: DatasetDict, tokenizer: PreTrainedTokenizerFast, cfg: Config
) -> DataLoaders:
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = partial(DataLoader, collate_fn=collate_fn, **cfg.dataloader.model_dump())
    return DataLoaders(
        train=loader(dataset=splits["train"], shuffle=True, drop_last=True),
        val=loader(dataset=splits["val"]),
        test=loader(dataset=splits["test"]),
        train_val=loader(dataset=splits["train_val"], shuffle=True, drop_last=True),
    )
