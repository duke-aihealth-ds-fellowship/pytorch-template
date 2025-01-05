import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from tqdm import tqdm

from template.config import Config
from template.dataset import DataLoaders
from template.model import EmbeddingModel


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        dataloaders: DataLoaders,
        cfg: Config,
        device: str | torch.device,
    ):
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.device = device

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.trainer.gradient_clip
        )
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        loss = 0
        for inputs, labels in self.dataloaders.validation:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss += self.criterion(outputs, labels)
        validation_loss = loss / len(self.dataloaders.validation)
        return validation_loss

    def train(self) -> None:
        progress_bar = tqdm(range(self.cfg.trainer.max_epochs), desc="Epoch")
        for epoch in progress_bar:
            self.model.train()
            train_loss = 0
            for inputs, labels in self.dataloaders.train:
                loss = self.train_step(inputs, labels)
                train_loss += loss
            train_loss = train_loss / len(self.dataloaders.train)
            if epoch % self.cfg.trainer.eval_every_n_epochs == 0:
                validation_loss = self.validate()
                progress_bar.set_postfix_str(
                    f"Train loss: {train_loss.item():.4f}, "
                    "Validation loss: {validation_loss:.4f}"
                )
        return validation_loss


def make_trainer(cfg: Config, dataloaders: DataLoaders) -> Trainer:
    model = EmbeddingModel(**cfg.model.model_dump())
    optimizer = SGD(model.parameters(), **cfg.optimizer.model_dump())
    criterion = nn.BCEWithLogitsLoss()
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders=dataloaders,
        cfg=cfg,
        device=cfg.trainer.device,
    )
