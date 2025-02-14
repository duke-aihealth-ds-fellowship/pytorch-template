import json
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from template.config import Config
from template.dataset import DataLoaders
from template.model import EmbeddingModel, set_hyperparameters


class Trainer:
    def __init__(
        self,
        model: nn.Module | Callable,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: LRScheduler,
        max_epochs: int,
        gradient_clip: float,
        device: str,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.device = device
        self.train_loss = float("inf")
        self.eval_loss = float("inf")
        self.progress_bar: tqdm
        self.epoch: int

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        inputs = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        train_loss = 0
        for step, batch in enumerate(loader, start=1):
            train_loss += self.train_step(batch)
            self.train_loss = train_loss / step
            self.update_progress()
        self.scheduler.step()

    def evaluate_step(self, batch: dict[str, torch.Tensor]) -> float:
        inputs = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.model(inputs)
        return self.criterion(outputs, labels).item()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        eval_loss = 0
        for batch in loader:
            eval_loss += self.evaluate_step(batch)
        eval_loss /= len(loader)
        return eval_loss

    def update_progress(self):
        lr = self.scheduler.get_last_lr()[0]
        postfix = (
            f"Epoch {self.epoch}, Train loss: {self.train_loss:.4f}, "
            f"Eval loss: {self.eval_loss:.4f}, lr: {lr:.2e}"
        )
        self.progress_bar.set_postfix_str(postfix)
        self.progress_bar.update()

    def train(self, train_loader: DataLoader, eval_loader: DataLoader) -> float:
        num_steps = self.max_epochs * len(train_loader)
        self.progress_bar = tqdm(total=num_steps, desc="Training steps")
        self.eval_loss = self.evaluate(loader=eval_loader)
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.train_epoch(train_loader)
            self.eval_loss = self.evaluate(loader=eval_loader)
            self.update_progress()
        return self.train_loss

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> list[torch.Tensor]:
        self.model.eval()
        predictions = []
        for batch in loader:
            inputs = batch["input_ids"].to(self.device)
            outputs = self.model(inputs)
            predictions.append(outputs)
        return predictions


def make_trainer(cfg: Config) -> Trainer:
    model = EmbeddingModel(**cfg.model.model_dump())
    if cfg.compile:
        model = torch.compile(model)
    model.to(cfg.trainer.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), **cfg.optimizer.model_dump())
    scheduler = CosineAnnealingWarmRestarts(optimizer, **cfg.scheduler.model_dump())
    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        **cfg.trainer.model_dump(),
    )


def train_model(loaders: DataLoaders, cfg: Config):
    if cfg.use_best:
        with open(cfg.tuner.hparams_path, "r") as file:
            hyperparameters = json.load(file)
        cfg = set_hyperparameters(cfg=cfg, **hyperparameters)
        trainer = make_trainer(cfg=cfg)
        trainer.train(loaders.train_val, loaders.test)
    else:
        trainer = make_trainer(cfg=cfg)
        trainer.train(loaders.train, loaders.val)
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
