import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
)
from torchmetrics.wrappers import BootStrapper

from template.config import Config


@torch.no_grad()
def evaluate_model(
    cfg: Config, model: nn.Module, loader: DataLoader, aggregate=False
) -> dict:
    model.eval()
    metrics = MetricCollection(
        [MulticlassAUROC, MulticlassAveragePrecision, MulticlassAccuracy]
    )
    if cfg.evaluator.n_bootstraps:
        metrics = BootStrapper(
            metrics,
            num_bootstraps=cfg.evaluator.n_bootstraps,
            mean=aggregate,
            std=aggregate,
            raw=not aggregate,
        )
    metrics = [metric(num_classes=cfg.model.output_dim) for metric in metrics]
    for inputs, labels in loader:
        inputs = inputs.to(cfg.trainer.device)
        labels = labels.to(cfg.trainer.device)
        outputs = model(inputs)

        metrics.update(outputs, labels)
    results = metrics.compute()
    return results


# TODO generate metrics for subsets of the data
def evaluate_subsets(df: pl.DataFrame, groups: list[str]):
    pass
