from dataclasses import dataclass
from functools import partial
from torch.utils.data import Dataset, DataLoader
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import numpy as np

from example.config import DataLoaderConfig


def make_fake_dataset():
    n_samples = 200
    max_sequence_length = 10
    n_classes = 2
    vocab_size = 50
    sequence_ids = np.arange(n_samples)
    sequence_lengths = np.random.randint(1, max_sequence_length + 1, size=n_samples)
    sequences = [
        np.random.randint(0, vocab_size, size=length) for length in sequence_lengths
    ]
    labels = np.random.randint(0, n_classes, size=n_samples).tolist()
    return pl.DataFrame({"id": sequence_ids, "input": sequences, "label": labels})


def make_splits(df: pl.DataFrame, train_size: float, random_state: int):
    train, val_test = train_test_split(
        df, train_size=train_size, random_state=random_state, shuffle=True
    )
    val, test = train_test_split(
        val_test, train_size=0.5, random_state=random_state, shuffle=False
    )
    return train, val, test


class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df: pl.DataFrame = self.df[idx]
        inputs = torch.tensor(df["input"].item(), dtype=torch.long)
        labels = torch.tensor(df["label"].item(), dtype=torch.float32)
        return inputs, labels


def collate_batch(batch):
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels).unsqueeze(1)
    return inputs, labels


@dataclass
class Splits:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_dataloaders(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    dataloader_config: DataLoaderConfig,
):
    train_dataset = SequenceDataset(train)
    val_dataset = SequenceDataset(val)
    test_dataset = SequenceDataset(test)
    dataloader = partial(
        DataLoader, collate_fn=collate_batch, **dataloader_config.model_dump()
    )
    train_loader = dataloader(dataset=train_dataset, shuffle=True)
    val_loader = dataloader(dataset=val_dataset, shuffle=False)
    test_loader = dataloader(dataset=test_dataset, shuffle=False)
    return Splits(train=train_loader, val=val_loader, test=test_loader)
