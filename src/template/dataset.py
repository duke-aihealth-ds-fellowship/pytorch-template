from dataclasses import dataclass
from functools import partial
from typing import cast

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

from template.config import Config
from template.tokenizer import make_tokenizer, tokenize_dataset


def subset_splits(dataset: Dataset, batch_size: int, seed: int):
    dataset = dataset.shuffle(seed=seed)
    train = dataset.select(range(batch_size))
    val = dataset.select(range(batch_size, 2 * batch_size))
    test = dataset.select(range(2 * batch_size, 3 * batch_size))
    return DatasetDict({"train": train, "val": val, "test": test})


def make_splits(cfg: Config) -> DatasetDict:
    ds = load_dataset(cfg.dataset.path)
    ds = cast(DatasetDict, ds)
    num_classes = len(ds["train"].unique("label"))
    ds.cast_column("label", ClassLabel(num_classes=num_classes))
    splits = [split for split in ds.values()]
    ds = concatenate_datasets(splits)
    if cfg.main.dev_run:
        splits = subset_splits(ds, cfg.dataloader.batch_size, cfg.main.seed)
    else:
        split = partial(
            Dataset.train_test_split,
            seed=cfg.main.seed,
            stratify_by_column=cfg.dataset.stratify,
        )
        train_val_test = split(ds, train_size=cfg.dataset.train_size)
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
    dataset_path = cfg.main.data / cfg.dataset.path
    if not dataset_path.exists() or cfg.main.regenerate:
        splits = make_splits(cfg=cfg)
        tokenizer = make_tokenizer(cfg=cfg, dataset=splits["train"])
        splits = tokenize_dataset(tokenizer=tokenizer, splits=splits, cfg=cfg)
        splits.save_to_disk(str(dataset_path))
    else:
        splits = DatasetDict.load_from_disk(dataset_path)
    return splits


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_val: DataLoader


def make_dataloaders(splits: DatasetDict, cfg: Config) -> DataLoaders:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer.path)
    cfg.model.output_dim = len(splits["train"].unique("label"))
    cfg.model.padding_idx = tokenizer.pad_token_id  # type: ignore
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = partial(DataLoader, collate_fn=collate_fn, **cfg.dataloader.model_dump())
    return DataLoaders(
        train=loader(dataset=splits["train"], shuffle=True, drop_last=True),
        val=loader(dataset=splits["val"]),
        test=loader(dataset=splits["test"]),
        train_val=loader(dataset=splits["train_val"], shuffle=True, drop_last=True),
    )
