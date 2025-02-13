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
from tqdm import tqdm

from template.config import Config
from template.model import EmbeddingModel
from template.tune import load_best_checkpoint


def make_metrics(cfg: Config):
    metrics = {
        "AUROC": MulticlassAUROC,
        "AP": MulticlassAveragePrecision,
        "Accuracy": MulticlassAccuracy,
    }
    for name, metric in metrics.items():
        metric = metric(num_classes=cfg.model.output_dim)
        if cfg.evaluator.n_bootstraps > 0:
            metric = BootStrapper(
                metric,
                num_bootstraps=cfg.evaluator.n_bootstraps,
                mean=cfg.evaluator.aggregate,
                std=cfg.evaluator.aggregate,
                raw=not cfg.evaluator.aggregate,
            )
        metrics[name] = metric
    return MetricCollection(metrics).to(cfg.trainer.device)


@torch.no_grad()
def evaluate_model(cfg: Config, loader: DataLoader) -> dict:
    model = load_best_checkpoint(cfg=cfg, model_class=EmbeddingModel)
    model.eval()
    metrics = make_metrics(cfg=cfg)
    for batch in tqdm(loader, desc="Evaluating"):
        inputs = batch["input_ids"].to(cfg.trainer.device)
        labels = batch["labels"].to(cfg.trainer.device)
        outputs = model(inputs)
        metrics.update(outputs, labels)
    results = metrics.compute()
    return results


# TODO generate metrics for subsets of the data
def evaluate_subsets(df: pl.DataFrame, groups: list[str]):
    pass
