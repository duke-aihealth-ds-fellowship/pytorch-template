import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from example.model import EmbeddingModel


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


# barebones training loop
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    max_epochs: int,
):
    progress_bar = tqdm(range(max_epochs), desc="Epochs")
    for _ in progress_bar:
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        progress_bar.set_postfix_str(f"Training loss: {loss.item():.3f}")
    return model
