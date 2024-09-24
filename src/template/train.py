import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from tqdm import tqdm

from template.checkpoint import (
    checkpoint_model,
    get_best_checkpoint_path,
    remove_worse_checkpoints,
)
from template.config import Config
from template.dataset import DataLoaders
from template.evaluate import evaluate_model
from template.model import EmbeddingModel


def make_components(config: Config):
    model = EmbeddingModel(**config.model.model_dump())
    optimizer = SGD(model.parameters(), **config.optimizer.model_dump())
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion


def train_model(dataloaders: DataLoaders, config: Config) -> None:
    model, optimizer, criterion = make_components(config=config)
    model_device = torch.device(config.trainer.device)
    model.to(model_device)
    progress_bar = tqdm(range(config.trainer.max_epochs), desc="Epoch")
    min_val_loss = float("inf")
    early_stopping = 0
    for epoch in progress_bar:
        if early_stopping > config.trainer.early_stopping_patience:
            break
        epoch_train_loss = 0
        for inputs, labels in dataloaders.train:
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.trainer.gradient_clip)
            optimizer.step()
            epoch_train_loss += train_loss.item() * inputs.size(0)
        epoch_train_loss = epoch_train_loss / len(dataloaders.train.dataset)
        if epoch % config.trainer.eval_every_n_epochs == 0:
            val_loss = evaluate_model(
                model=model,
                dataloader=dataloaders.val,
                metrics={"val_loss": criterion},
                device=model_device,
            )["val_loss"]
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint_model(
                path=config.checkpoint.path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=epoch_train_loss,
                val_loss=min_val_loss,
                model_config=config.model,
                optimizer_config=config.optimizer,
            )
            progress_bar.set_postfix_str(
                f"train loss: {epoch_train_loss.item():.4f}; val loss: {val_loss:.4f}"
            )
            early_stopping = 0
        early_stopping += 1
        best_checkpoint_path = get_best_checkpoint_path(config.checkpoint)
        remove_worse_checkpoints(
            best_checkpoint_path, checkpoint_path=config.checkpoint.path
        )
