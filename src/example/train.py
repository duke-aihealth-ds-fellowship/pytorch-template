import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

from example.checkpoint import checkpoint_model
from example.config import Config, ModelConfig, OptimizerConfig
from example.dataset import Splits
from example.model import EmbeddingModel
from example.evaluate import evaluate_model


def make_components(model_config: ModelConfig, optimizer_config: OptimizerConfig):
    model = EmbeddingModel(**model_config.model_dump())
    optimizer = SGD(model.parameters(), **optimizer_config.model_dump())
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion


# TODO add early stopping, gradient clipping, and DataParallel
def train_model(dataloaders: Splits, config: Config):
    model, optimizer, criterion = make_components(
        **config.model.model_dump(), **config.optimizer.model_dump()
    )
    model_device = torch.device(config.trainer.device)
    model.to(device=model_device)
    progress_bar = tqdm(range(config.trainer.max_epochs), desc="Epoch")
    min_val_loss = float("inf")
    for epoch in progress_bar:
        for inputs, labels in dataloaders.train:
            inputs, labels = inputs.to(model_device), labels.to(model_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
        val_loss = evaluate_model(
            model=model, dataloader=dataloaders.val, metrics={"val_loss": criterion}
        )
        if val_loss["val_loss"] < min_val_loss:
            min_val_loss = val_loss["val_loss"]
            checkpoint_model(
                path=config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss.item(),
                val_loss=min_val_loss,
                model_config=config.model,
                optimizer_config=config.optimizer,
            )
        progress_bar.set_postfix_str(
            f"train loss: {train_loss.item():.4f}; val loss: {val_loss['val_loss']:.4f}"
        )
    return model
