import polars as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
)
from torchmetrics.wrappers import BootStrapper

from template.config import Config
from template.model import EmbeddingModel
from template.tune import load_best_checkpoint


@torch.no_grad()
def evaluate_model(cfg: Config, loader: DataLoader) -> dict:
    model = load_best_checkpoint(cfg=cfg, model_class=EmbeddingModel)
    model.eval()
    metrics = [MulticlassAUROC, MulticlassAveragePrecision, MulticlassAccuracy]
    metrics = [metric(num_classes=cfg.model.output_dim) for metric in metrics]
    metrics = MetricCollection(metrics)
    if cfg.evaluator.n_bootstraps:
        metrics = BootStrapper(
            metrics,
            num_bootstraps=cfg.evaluator.n_bootstraps,
            mean=cfg.evaluator.aggregate,
            std=cfg.evaluator.aggregate,
            raw=not cfg.evaluator.aggregate,
        )
    metrics = metrics.to(cfg.trainer.device)
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
