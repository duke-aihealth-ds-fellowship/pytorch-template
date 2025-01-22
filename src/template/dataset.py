from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import polars as pl
import torch
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
        inputs = torch.tensor(df["input"].item(), dtype=torch.int)
        labels = torch.tensor(df["label"].item(), dtype=torch.int)
        return inputs, labels


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_validation: DataLoader


def collate_batch(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels)
    return inputs, labels


def make_dataloaders(data: Iterable, cfg: Config) -> DataLoaders:
    train, validation_test = train_test_split(
        data, train_size=cfg.train_size, random_state=cfg.random_state, shuffle=True
    )
    validation, test = train_test_split(
        validation_test, train_size=0.5, random_state=cfg.random_state, shuffle=False
    )
    dataloader = partial(
        DataLoader, collate_fn=collate_batch, **cfg.dataloader.model_dump()
    )
    train_dataset = SequenceDataset(train)
    validation_dataset = SequenceDataset(validation)
    test_dataset = SequenceDataset(test)
    train_validation_dataset = ConcatDataset([train_dataset, validation_dataset])
    return DataLoaders(
        train=dataloader(dataset=train_dataset, shuffle=True),
        validation=dataloader(dataset=validation_dataset, shuffle=False),
        test=dataloader(dataset=test_dataset, shuffle=False),
        train_validation=dataloader(dataset=train_validation_dataset, shuffle=True),
    )
