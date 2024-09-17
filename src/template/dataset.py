from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from template.config import DataLoaderConfig


@dataclass
class Datasets:
    train: Dataset
    val: Dataset
    test: Dataset


class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        df: pl.DataFrame = self.df[idx]
        inputs = torch.tensor(df["input"].item(), dtype=torch.long)
        labels = torch.tensor(df["label"].item(), dtype=torch.float)
        return inputs, labels


# TODO SequenceDataset is specific to the example dataset, make generic
def make_splits(data: Iterable, train_size: float, random_state: int) -> Datasets:
    train, val_test = train_test_split(
        data, train_size=train_size, random_state=random_state, shuffle=True
    )
    val, test = train_test_split(
        val_test, train_size=0.5, random_state=random_state, shuffle=False
    )
    return Datasets(
        train=SequenceDataset(train),
        val=SequenceDataset(val),
        test=SequenceDataset(test),
    )


def collate_batch(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels).unsqueeze(1)
    return inputs, labels


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_dataloaders(splits: Datasets, dataloader_config: DataLoaderConfig):
    dataloader = partial(
        DataLoader, collate_fn=collate_batch, **dataloader_config.model_dump()
    )
    return DataLoaders(
        train=dataloader(dataset=splits.train, shuffle=True),
        val=dataloader(dataset=splits.val, shuffle=False),
        test=dataloader(dataset=splits.test, shuffle=False),
    )
