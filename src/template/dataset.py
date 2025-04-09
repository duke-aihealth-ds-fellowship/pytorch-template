from dataclasses import dataclass
from functools import partial

from tensordict import LazyStackedTensorDict, TensorDict
from torch.utils.data import DataLoader, Dataset

from template.config import Config


def train_val_test_split(data: TensorDict, proportions: list[float]) -> TensorDict:
    n = len(data)
    train_size = int(n * proportions[0])
    val_size = int(n * proportions[1])
    test_size = n - train_size - val_size
    train, val, test = data.split([train_size, val_size, test_size], dim=0)
    train_val = LazyStackedTensorDict.lazy_stack([train, val])
    result = TensorDict({})
    result["train"] = train
    result["val"] = val
    result["test"] = test
    result["train_val"] = train_val
    return result


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_val: DataLoader


def get_target(cfg: Config) -> str:
    if cfg.task == "cls":
        target = "label"
    elif cfg.task == "reg":
        target = "target"
    elif cfg.task == "tte":
        target = "indicator"
    else:
        raise ValueError(f"Unknown task: {cfg.task}. Choose from 'cls', 'reg', 'tte'.")
    return target


class TensorDictDataset(Dataset):
    def __init__(self, tensordict: TensorDict):
        self.tensordict = tensordict

    def __len__(self):
        return len(self.tensordict)

    def __getitem__(self, idx):
        return self.tensordict[idx]


def get_output_dim(data: TensorDict, cfg: Config) -> int:
    if cfg.task == "bc":
        return 1
    elif cfg.task == "cls":
        target = get_target(cfg=cfg)
        return len(data["train"][target].unique())
    elif cfg.task == "tte":
        raise NotImplementedError("Time-to-event task is not implemented yet")
    elif cfg.task == "reg":
        raise NotImplementedError("Regression task is not implemented yet")
    else:
        raise ValueError(
            f"Unknown task: {cfg.task}. Choose from 'bc', 'cls', 'tte', 'reg'."
        )


def make_dataloaders(cfg: Config) -> DataLoaders:
    data = TensorDict.load(cfg.path.dataset)
    cfg.model.output_dim = get_output_dim(data=data, cfg=cfg)
    loader_kwargs = cfg.dataloader.model_dump()
    loader = partial(DataLoader, **loader_kwargs)
    train_loader = partial(loader, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    data = TensorDict.load(cfg.path.dataset)
    return DataLoaders(
        train=train_loader(dataset=data["train"]),
        val=loader(dataset=data["val"]),
        test=loader(dataset=data["test"]),
        train_val=train_loader(dataset=data["train_val"]),
    )
