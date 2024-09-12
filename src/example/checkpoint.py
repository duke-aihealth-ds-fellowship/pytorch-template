from pathlib import Path
from re import search
from numpy import argmin, argmax
import torch
import torch.nn as nn

from example.config import CheckpointConfig, ModelConfig, OptimizerConfig


def checkpoint_model(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"epoch_{epoch}_train_loss_{train_loss:.3f}_val_loss_{val_loss:.3f}.pt"
    filepath = path / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_class": type(model),
            "model_config": model_config.model_dump(),
            "optimizer_config": optimizer_config.model_dump(),
        },
        filepath,
    )


def cleanup_checkpoints(checkpoint_config: CheckpointConfig) -> None:
    checkpoint_paths = list(checkpoint_config.path.glob("*"))
    best_checkpoint = get_best_checkpoint_path(checkpoint_config)
    for checkpoint in checkpoint_paths:
        if checkpoint == best_checkpoint:
            checkpoint.unlink()


def get_best_checkpoint_path(checkpoint_config: CheckpointConfig) -> Path:
    checkpoint_paths = list(checkpoint_config.path.glob("*.pt"))
    metrics = [
        float(match.group(0))
        for filepath in checkpoint_paths
        if (match := search(r"(\d+\.\d+)\.pt", filepath.stem))
    ]
    match checkpoint_config.mode:
        case "minimize":
            index = argmin(metrics)
        case "maximize":
            index = argmax(metrics)
        case _:
            raise ValueError("mode must be 'minimize' or 'maximize'")
    return checkpoint_paths[index]


def load_best_checkpoint(checkpoint_config: CheckpointConfig) -> nn.Module:
    checkpoint_path = get_best_checkpoint_path(checkpoint_config)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model_class = checkpoint["model_class"]
    model = model_class(**checkpoint["model_config"])
    return model.load_state_dict(checkpoint["model_state_dict"])
