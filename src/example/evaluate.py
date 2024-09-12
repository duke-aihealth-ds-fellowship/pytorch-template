from collections.abc import Callable
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.wrappers import BootStrapper


def get_predictions(
    model: nn.Module, dataloader: DataLoader, device: torch.device | str
):
    model.to(device)
    model.eval()
    output_batches = []
    label_batches = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        output_batches.append(outputs)
        label_batches.append(labels)
    return torch.cat(output_batches), torch.cat(label_batches)


def bootstrap_metric(
    metric, outputs: torch.Tensor, labels: torch.Tensor, n_bootstraps: int
):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=True, std=True, raw=True
    )
    bootstrap.to(outputs.device)
    bootstrap.update(outputs, labels)
    return bootstrap.compute()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    metrics: dict[str, Callable],
    device: torch.device | str,
    n_bootstraps: int | None = None,
) -> dict:
    outputs, labels = get_predictions(model=model, dataloader=dataloader, device=device)
    results = {}
    for name, metric in metrics.items():
        if issubclass(type(metric), Metric):
            labels = labels.long()
        if n_bootstraps:
            results[name] = bootstrap_metric(metric, outputs, labels, n_bootstraps)
        else:
            metric.to(outputs.device)
            results[name] = metric(outputs, labels)
    return results


# TODO generate metrics for subsets of the data
def evaluate_subsets(df: pl.DataFrame, groups: list[str]):
    pass
