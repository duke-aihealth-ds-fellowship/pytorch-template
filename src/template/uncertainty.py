from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from template.config import Config
from template.dataset import DataLoaders
from template.model import Transformer
from template.tune import load_checkpoint, train_model


class ModelEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor) -> Tensor:
        predictions = [model(x) for model in self.models]
        return torch.stack(predictions, dim=0)


def ensemble_predictions(loaders: DataLoaders, cfg: Config):
    models = []
    for _ in repeat(None, cfg.uncertainty.ensemble_size):
        model = train_model(loaders=loaders, cfg=cfg, save=False)
        models.append(model)
    ensemble_model = ModelEnsemble(models=models)
    ensemble_model.to(cfg.trainer.device)
    ensemble_model.eval()
    predictions = []
    labels = []
    for batch in loaders.test:
        batch = batch.to(cfg.trainer.device)
        with torch.no_grad():
            outputs = ensemble_model(batch["features"]).permute(1, 0, 2, 3)
        predictions.append(outputs)
        batch_labels = batch["label"].unsqueeze(1).expand_as(outputs)
        labels.append(batch_labels)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels


@torch.no_grad()
def monte_carlo_dropout(
    model: Transformer, loader: DataLoader, cfg: Config
) -> tuple[Tensor, Tensor]:
    model.train()
    batch_predictions = []
    batch_labels = []
    for batch in loader:
        batch = batch.to(cfg.trainer.device)
        mc_predictions = []
        for _ in repeat(None, cfg.uncertainty.mc_dropout_samples):
            outputs = model(batch["features"])
            mc_predictions.append(outputs)
        mc_predictions = torch.stack(mc_predictions, dim=1)
        batch_predictions.append(mc_predictions)
        expanded_labels = batch["label"].unsqueeze(1).expand_as(mc_predictions)
        batch_labels.append(expanded_labels)
    predictions = torch.cat(batch_predictions, dim=0)
    labels = torch.cat(batch_labels, dim=0)
    return predictions, labels


def plot_uncertainty_over_time(
    predictions: Tensor, labels: Tensor, method: str
) -> None:
    batch_size, num_samples, seq_len, output_dim = predictions.shape
    time_indices = np.repeat(np.arange(seq_len), output_dim)
    time_indices = np.tile(time_indices, batch_size * num_samples)
    probability = torch.sigmoid(predictions)
    df = pl.DataFrame(
        {
            "Label": labels.flatten().numpy(),
            "Probabilty": probability.flatten().numpy(),
            "Time index": time_indices,
        }
    )
    sns.lineplot(data=df, x="Time index", y="Probabilty", hue="Label")
    plt.savefig(f"data/plots/{method}_uncertainty_over_time.png")


def quantify_uncertainty(cfg: Config, loaders: DataLoaders) -> None:
    if cfg.uncertainty.method == "ensemble":
        predictions, labels = ensemble_predictions(loaders=loaders, cfg=cfg)
    elif cfg.uncertainty.method == "mc_dropout":
        model = load_checkpoint(cfg=cfg, model_class=Transformer)
        predictions, labels = monte_carlo_dropout(
            model=model, loader=loaders.test, cfg=cfg
        )
    else:
        raise ValueError(
            f"Unknown UQ method: {cfg.uncertainty.method}. Choose from 'ensemble' or 'mc_dropout'."
        )
    plot_uncertainty_over_time(
        predictions=predictions, labels=labels, method=cfg.uncertainty.method
    )
