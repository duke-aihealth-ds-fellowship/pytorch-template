import json
from typing import Callable

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
        model: nn.Module | Callable,
        optimizer: Optimizer,
        criterion: nn.Module,
        cfg: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.train_loss = float("inf")
        self.eval_loss = float("inf")

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        inputs = batch["input_ids"].to(self.cfg.device)
        labels = batch["labels"].to(self.cfg.device)
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
        eval_loss = 0
        for batch in loader:
            inputs = batch["input_ids"].to(self.cfg.device)
            labels = batch["labels"].to(self.cfg.device)
            outputs = self.model(inputs)
            eval_loss += self.criterion(outputs, labels).item()
        eval_loss /= len(loader)
        return eval_loss

    def update_progress(self, progress_bar: tqdm):
        postfix = f"Train loss: {self.train_loss:.4f}, Eval loss: {self.eval_loss:.4f}"
        progress_bar.set_postfix_str(postfix)
        progress_bar.update()

    def train(self, train_loader: DataLoader, eval_loader: DataLoader) -> float:
        self.model.to(self.cfg.device)
        num_batches = len(train_loader)
        num_steps = self.cfg.max_epochs * num_batches
        progress_bar = tqdm(total=num_steps, desc="Steps")
        self.eval_loss = self.evaluate(loader=eval_loader)
        self.model.train()
        for _ in range(self.cfg.max_epochs):
            train_loss = 0
            self.model.train()
            for step, batch in enumerate(train_loader, start=1):
                train_loss += self.train_step(batch)
                self.train_loss = train_loss / step
                self.update_progress(progress_bar)
            self.eval_loss = self.evaluate(loader=eval_loader)
            self.update_progress(progress_bar)
        return train_loss

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> list[torch.Tensor]:
        self.model.eval()
        predictions = []
        for batch in loader:
            inputs = batch["input_ids"].to(self.cfg.device)
            outputs = self.model(inputs)
            predictions.append(outputs)
        return predictions


def make_trainer(cfg: Config) -> Trainer:
    model = EmbeddingModel(**cfg.model.model_dump(exclude={"compile"}))
    if cfg.model.compile:
        model = torch.compile(model)
    optimizer = SGD(model.parameters(), **cfg.optimizer.model_dump(), fused=True)
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model, optimizer=optimizer, criterion=criterion, cfg=cfg.trainer
    )


def train_model(loaders: DataLoaders, cfg: Config, use_best: bool = False):
    if use_best:
        with open(cfg.tuner.hparams_path, "r") as file:
            hyperparameters = json.load(file)
        cfg = set_hyperparameters(cfg=cfg, **hyperparameters)
        trainer = make_trainer(cfg=cfg)
        trainer.train(loaders.train_val, loaders.test)
    else:
        trainer = make_trainer(cfg=cfg)
        trainer.train(loaders.train, loaders.val)
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
