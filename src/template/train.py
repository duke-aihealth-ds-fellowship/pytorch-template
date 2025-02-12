import json

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
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
        cfg: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.train_loss = float("inf")
        self.validation_loss = float("inf")

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        inputs = inputs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.gradient_clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        loss = 0
        for inputs, labels in loader:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            outputs = self.model(inputs)
            loss += self.criterion(outputs, labels)
        loss /= len(loader)
        return loss

    def train(self, train_loader: DataLoader, eval_loader: DataLoader) -> float:
        self.model.to(self.cfg.device)
        num_steps = self.cfg.max_epochs * len(train_loader)
        progress_bar = tqdm(range(num_steps), desc="Steps")
        for epoch in progress_bar:
            self.model.train()
            loss = 0
            for inputs, labels in train_loader:
                loss += self.train_step(inputs, labels)
            loss /= len(train_loader)
            self.train_loss = loss
            if epoch % self.cfg.eval_every_n_epochs == 0:
                postfix = f"Train loss: {loss.item():.4f}"
                self.eval_loss = self.evaluate(loader=eval_loader)
                postfix += f", Val loss: {self.validation_loss:.4f}"
                progress_bar.set_postfix_str(postfix)
        return loss

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> list[torch.Tensor]:
        self.model.eval()
        predictions = []
        for inputs, labels in loader:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            outputs = self.model(inputs)
            predictions.append(outputs)
        return predictions


def make_trainer(cfg: Config) -> Trainer:
    model = EmbeddingModel(**cfg.model.model_dump())
    optimizer = SGD(model.parameters(), **cfg.optimizer.model_dump())
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model, optimizer=optimizer, criterion=criterion, cfg=cfg.trainer
    )


def train_model(loaders: DataLoaders, cfg: Config, use_best: bool = False):
    if use_best:
        with open(cfg.tuner.hyperparameters, "r") as file:
            hyperparameters = json.load(file)
        cfg = set_hyperparameters(cfg=cfg, **hyperparameters)
        trainer = make_trainer(cfg=cfg)
        trainer.train(loaders.train_validation, loaders.test)
    else:
        trainer = make_trainer(cfg=cfg)
        trainer.train(loaders.train, loaders.validation)
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
