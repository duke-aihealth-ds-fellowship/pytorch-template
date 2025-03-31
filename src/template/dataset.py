from dataclasses import dataclass
from functools import partial

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from template.config import Config
from template.simulation import simulate


def train_val_test_split(data: TensorDict, proportions: list[float]) -> TensorDict:
    n = len(data)
    train_size = int(n * proportions[0])
    val_size = int(n * proportions[1])
    test_size = n - train_size - val_size
    train, val, test = data.split([train_size, val_size, test_size], dim=0)
    train_val = torch.cat([train, val], dim=0)
    return TensorDict(
        {"train": train, "val": val, "test": test, "train_val": train_val}
    )


def save_data(data: TensorDict, cfg: Config):
    data.save(cfg.path.data)
    train, val, test = train_val_test_split(data=data, proportions=cfg.proportions)
    train.save(cfg.path.train)
    val.save(cfg.path.val)
    test.save(cfg.path.test)
    return data


def get_data(cfg: Config):
    if not cfg.dataset.path.exists() or cfg.main.regenerate:
        data = simulate(cfg=cfg)
        save_data(data=data, path=cfg.dataset.path)
    else:
        data = TensorDict.load(cfg.dataset.path)
    return data


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_val: DataLoader


# TODO add padding collate_fn
def collate_fn(batch):
    pass


def make_dataloaders(data: TensorDict, cfg: Config) -> DataLoaders:
    cfg.model.output_dim = len(data["train"].unique("label"))
    loader = partial(DataLoader, collate_fn=collate_fn, **cfg.dataloader.model_dump())
    train_loader = partial(loader, shuffle=True, drop_last=True)
    return DataLoaders(
        train=train_loader(dataset=data["train"]),
        val=loader(dataset=data["val"]),
        test=loader(dataset=data["test"]),
        train_val=train_loader(dataset=data["train_val"]),
    )
