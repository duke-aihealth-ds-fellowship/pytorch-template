from pathlib import Path
from re import search
from numpy import argmin, argmax
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from template.config import CheckpointConfig, ModelConfig, OptimizerConfig


def checkpoint_model(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"epoch_{epoch}_train_loss_{train_loss:.4f}_val_loss_{val_loss:.4f}.pt"
    filepath = path / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": model_config.model_dump(),
            "optimizer_config": optimizer_config.model_dump(),
        },
        filepath,
    )


def remove_worse_checkpoints(
    best_checkpoint_filepath: Path, checkpoint_path: Path
) -> None:
    for checkpoint in checkpoint_path.glob("*.pt"):
        if not checkpoint.samefile(best_checkpoint_filepath):
            checkpoint.unlink()


# TODO may want the regex pattern to be configurable
def get_best_checkpoint_path(checkpoint_config: CheckpointConfig) -> Path:
    checkpoint_paths = list(checkpoint_config.path.glob("*.pt"))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_config.path}.")
    metrics = []
    for filepath in checkpoint_paths:
        match = search(r"val_loss_(\d+\.\d+)", filepath.stem)
        if match:
            metrics.append(float(match.group(1)))
    match checkpoint_config.mode:
        case "minimize":
            index = argmin(metrics)
        case "maximize":
            index = argmax(metrics)
        case _:
            raise ValueError("mode must be 'minimize' or 'maximize'")
    return checkpoint_paths[index]


# TODO load optimizer as well so training can be resumed from a checkpoint
def load_best_checkpoint(filepath: Path, model_class: type[nn.Module]) -> nn.Module:
    checkpoint = torch.load(filepath, weights_only=True)
    model = model_class(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
