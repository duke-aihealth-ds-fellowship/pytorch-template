from collections.abc import Callable
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.wrappers import BootStrapper


def get_predictions(model: nn.Module, dataloader: DataLoader):
    model.eval()
    outputs_list = []
    labels_list = []
    for inputs, labels in dataloader:
        with torch.no_grad():
            outputs = model(inputs)
        outputs_list.append(outputs)
        labels_list.append(labels)
    return torch.cat(outputs_list), torch.cat(labels_list)


def bootstrap_metric(
    metric, outputs: torch.Tensor, labels: torch.Tensor, n_bootstraps: int
):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=False, std=False, raw=True
    )
    bootstrap.to(outputs.device)
    bootstrap.update(outputs, labels)
    return bootstrap.compute()["raw"]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    metrics: dict[str, Callable],
    n_bootstraps: int | None = None,
):
    outputs, labels = get_predictions(model=model, dataloader=dataloader)
    results = {}
    for name, metric in metrics.items():
        metric.to(outputs.device)
        if n_bootstraps:
            result = bootstrap_metric(metric, outputs, labels, n_bootstraps)
        else:
            result = metric(outputs, labels)
        results[name] = result.cpu().numpy()
    return results


# TODO generate metrics for subsets of the data
def evaluate_subsets(df: pl.DataFrame, groups: list[str]):
    pass
