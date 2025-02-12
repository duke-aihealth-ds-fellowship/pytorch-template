from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import polars as pl
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from template.config import Config


class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        df: pl.DataFrame = self.df[idx]
        inputs = torch.tensor(df["text"].item(), dtype=torch.int)
        labels = torch.tensor(df["label"].item(), dtype=torch.int)
        return inputs, labels


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_val: DataLoader


def collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels)
    return inputs, labels


def train_val_test_split(data: Iterable, cfg: Config):
    split = partial(train_test_split, random_state=cfg.random_state)
    train, temp = split(data, train_size=cfg.train_size)
    val, test = split(temp, train_size=0.5, shuffle=False)
    return train, val, test


def init_datasets(cfg: Config):
    dataset = load_dataset(cfg.dataset.name)
    dataset = pl.DataFrame(dataset["train"] + dataset["test"])
    train, val, test = train_val_test_split(dataset, cfg=cfg)
    train = SequenceDataset(train)
    val = SequenceDataset(val)
    test = SequenceDataset(test)
    train_val = ConcatDataset([train, val])
    return train, val, test, train_val


def make_dataloaders(data: Iterable, cfg: Config) -> DataLoaders:
    train, val, test, train_val = init_datasets(data, cfg=cfg)
    loader = partial(DataLoader, collate_fn=collate_fn, **cfg.dataloader.model_dump())
    return DataLoaders(
        train=loader(dataset=train, shuffle=True),
        val=loader(dataset=val),
        test=loader(dataset=test),
        train_val=loader(dataset=train_val, shuffle=True),
    )
