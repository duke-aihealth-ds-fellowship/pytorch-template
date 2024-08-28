from functools import partial
from torch.utils.data import Dataset, DataLoader
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import numpy as np


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
        self.dfs: list[pl.DataFrame] = df.partition_by("id")

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, idx):
        df: pl.DataFrame = self.dfs[idx]
        X = torch.tensor(df["input"][0]).long()
        y = torch.tensor(df["label"].to_numpy()).float()
        return X, y


def collate_batch(batch, device):
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels)
    inputs = inputs.to(device)
    labels = labels.to(device)
    return inputs, labels


def make_dataloaders(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    batch_size: int,
    num_workers: int,
    device: str,
):
    train_dataset = SequenceDataset(train)
    val_dataset = SequenceDataset(val)
    test_dataset = SequenceDataset(test)
    collate_fn = partial(collate_batch, device=device)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
