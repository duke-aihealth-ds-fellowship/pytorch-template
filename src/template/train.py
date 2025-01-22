import json

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from tqdm import tqdm

from template.config import Config, TrainerConfig
from template.dataset import DataLoaders
from template.model import EmbeddingModel, set_hyperparameters


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        dataloaders: DataLoaders,
        cfg: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.progress_bar = tqdm(range(cfg.max_epochs), desc="Epoch")
        self.cfg = cfg
        self.train_loss = float("inf")
        self.validation_loss = float("inf")

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.gradient_clip)
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def validate(self) -> None:
        self.model.eval()
        loss = 0
        for inputs, labels in self.dataloaders.validation:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            outputs = self.model(inputs)
            loss += self.criterion(outputs, labels)
        loss /= len(self.dataloaders.validation)
        self.validation_loss = loss

    def train(self, validate: bool = True) -> None:
        self.model.to(self.cfg.device)
        for epoch in self.progress_bar:
            self.model.train()
            loss = 0
            for inputs, labels in self.dataloaders.train:
                loss += self.train_step(inputs, labels)
            loss /= len(self.dataloaders.train)
            self.train_loss = loss
            if epoch % self.cfg.eval_every_n_epochs == 0:
                postfix = f"Train loss: {loss.item():.4f}"
                if validate:
                    self.validate()
                    postfix += f", Val loss: {self.validation_loss:.4f}"
                self.progress_bar.set_postfix_str(postfix)


def make_trainer(cfg: Config, dataloaders: DataLoaders) -> Trainer:
    model = EmbeddingModel(**cfg.model.model_dump())
    optimizer = SGD(model.parameters(), **cfg.optimizer.model_dump())
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders=dataloaders,
        cfg=cfg.trainer,
    )


def train_model(
    dataloaders: DataLoaders,
    cfg: Config,
    use_best: bool = False,
    validate: bool = True,
    combine_train_val: bool = False,
):
    if combine_train_val:
        dataloaders.train = dataloaders.train_validation
    if use_best:
        hyperparameters = json.load(open(cfg.tuner.hyperparameters, "r"))
        cfg = set_hyperparameters(cfg=cfg, **hyperparameters)
    trainer = make_trainer(cfg=cfg, dataloaders=dataloaders)
    trainer.train(validate=validate)
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
