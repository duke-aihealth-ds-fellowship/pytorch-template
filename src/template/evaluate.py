import polars as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)
from torchmetrics.wrappers import BootStrapper

from template.config import Config
from template.model import EmbeddingModel
from template.tune import load_best_checkpoint


def get_predictions(cfg: Config, dataloader: DataLoader):
    model = load_best_checkpoint(cfg=cfg, model_class=EmbeddingModel)
    model.to(cfg.trainer.device)
    model.eval()
    output_batches = []
    label_batches = []
    for inputs, labels in dataloader:
        inputs = inputs.to(cfg.trainer.device)
        labels = labels.to(cfg.trainer.device)
        with torch.no_grad():
            outputs = model(inputs)
        output_batches.append(outputs)
        label_batches.append(labels)
    return torch.cat(output_batches).cpu(), torch.cat(label_batches).cpu()


def bootstrap_metric(
    metric, outputs: torch.Tensor, labels: torch.Tensor, n_bootstraps: int
):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=True, std=True, raw=True
    )
    bootstrap.update(outputs, labels)
    return bootstrap.compute()


def evaluate_model(
    cfg: Config,
    dataloader: DataLoader,
    n_bootstraps: int = 0,
) -> dict:
    outputs, labels = get_predictions(cfg=cfg, dataloader=dataloader)
    results = {}
    metrics = [BinaryAUROC(), BinaryAveragePrecision(), BinaryAccuracy()]
    for metric in metrics:
        name = metric.__class__.__name__
        if n_bootstraps:
            results[name] = bootstrap_metric(
                metric,
                outputs,
                labels,
                n_bootstraps=cfg.evaluator.n_bootstraps,
            )
        else:
            results[name] = metric(outputs, labels)
    return results


# TODO generate metrics for subsets of the data
def evaluate_subsets(df: pl.DataFrame, groups: list[str]):
    pass
