import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from example.model import EmbeddingModel
from example.evaluate import evaluate_model


def make_components(
    vocab_size: int,
    embedding_dim: int,
    padding_idx: int,
    output_dim: int,
    hidden_dim: int,
    n_layers: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: str,
):
    model = EmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_dim=output_dim,
    )
    model.to(device=torch.device(device))
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion


def checkpoint_model(path, model, optimizer, loss, epoch):
    filepath = path + f"epoch_{epoch}_loss_{loss:.3f}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )


# barebones training loop
# TODO early stopping
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    max_epochs: int,
    checkpoint_path: str | None = None,
):
    progress_bar = tqdm(range(max_epochs), desc="Epoch")
    min_val_loss = float("inf")
    for epoch in progress_bar:
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
        val_loss = evaluate_model(
            model=model, dataloader=val_loader, metrics={"val_loss": criterion}
        )
        if val_loss["val_loss"] < min_val_loss:
            min_val_loss = val_loss["val_loss"]
            if checkpoint_path:
                checkpoint_model(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    loss=min_val_loss,
                    epoch=epoch,
                )
        progress_bar.set_postfix_str(
            f"train loss: {train_loss.item():.3f}; val loss: {val_loss['val_loss']:.3f}"
        )
    return model
