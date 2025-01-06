from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from template.config import DataLoaderConfig


@dataclass
class Datasets:
    train: Dataset
    validation: Dataset
    test: Dataset


class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        df: pl.DataFrame = self.df[idx]
        inputs = torch.tensor(df["input"].item(), dtype=torch.int)
        labels = torch.tensor(df["label"].item(), dtype=torch.int)
        return inputs, labels


# TODO SequenceDataset is specific to the example dataset, make generic
def make_splits(data: Iterable, train_size: float, random_state: int) -> Datasets:
    train, validation_test = train_test_split(
        data, train_size=train_size, random_state=random_state, shuffle=True
    )
    validation, test = train_test_split(
        validation_test, train_size=0.5, random_state=random_state, shuffle=False
    )
    return Datasets(
        train=SequenceDataset(train),
        validation=SequenceDataset(validation),
        test=SequenceDataset(test),
    )


def collate_batch(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels)
    return inputs, labels


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def make_dataloaders(splits: Datasets, cfg: DataLoaderConfig):
    dataloader = partial(DataLoader, collate_fn=collate_batch, **cfg.model_dump())
    return DataLoaders(
        train=dataloader(dataset=splits.train, shuffle=True),
        validation=dataloader(dataset=splits.validation, shuffle=False),
        test=dataloader(dataset=splits.test, shuffle=False),
    )
