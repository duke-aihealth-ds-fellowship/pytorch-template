import json
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from template.config import Config
from template.dataset import DataLoaders
from template.model import EmbeddingModel, set_hyperparameters


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        model: nn.Module | Callable,
        criterion: nn.Module,
        optimizer: Optimizer,
        warmup_scheduler: LRScheduler,
        scheduler: LRScheduler,
        metric: Metric,
        max_epochs: int,
        gradient_clip: float,
        device: str,
    ):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.scheduler = scheduler
        self.metric = metric
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.device = device
        self.train_loss = float("inf")
        self.eval_loss = float("inf")
        self.eval_metric = 0.0
        num_steps = max_epochs * len(train_loader)
        self.progress_bar = tqdm(total=num_steps, desc="Train steps")

    def update_progress(self):
        lr = self.scheduler.get_last_lr()[0]
        postfix = (
            f"lr: {lr:.4e}, Train loss: {self.train_loss:.4f}"
            f", Eval loss: {self.eval_loss:.4f}"
            f", Eval metric: {self.eval_metric:.4f}"
        )
        self.progress_bar.set_postfix_str(postfix)
        self.progress_bar.update()

    def to_device(
        self, batch: dict[str, Tensor], keys: list[str]
    ) -> tuple[Tensor, ...]:
        return tuple(batch[key].to(self.device) for key in keys)

    def train_step(self, batch: dict[str, Tensor]) -> float:
        inputs, labels = self.to_device(batch, keys=["input_ids", "labels"])
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
        self.warmup_scheduler.step()
        return loss.item()

    def train_epoch(self):
        self.model.train()
        loss = 0.0
        for step, batch in enumerate(self.train_loader, start=1):
            loss += self.train_step(batch)
            self.train_loss = loss / step
            self.update_progress()
        self.scheduler.step()

    def train(self) -> float:
        for _ in range(1, self.max_epochs + 1):
            self.train_epoch()
            self.evaluate()
        self.progress_bar.close()
        return self.train_loss

    @torch.no_grad()
    def evaluate_step(self, batch: dict[str, Tensor]) -> float:
        inputs, labels = self.to_device(batch, keys=["input_ids", "labels"])
        outputs = self.model(inputs)
        self.metric.update(outputs, labels)
        return self.criterion(outputs, labels).item()

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        loss = 0.0
        for batch in self.eval_loader:
            loss += self.evaluate_step(batch)
        self.eval_metric = self.metric.compute()
        self.metric.reset()
        self.eval_loss = loss / len(self.eval_loader)
        return self.eval_loss

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> list[Tensor]:
        self.model.eval()
        predictions = []
        for batch in loader:
            inputs = batch["input_ids"].to(self.device)
            outputs = self.model(inputs)
            predictions.append(outputs)
        return predictions


def make_trainer(
    train_loader: DataLoader, eval_loader: DataLoader, cfg: Config
) -> Trainer:
    model = EmbeddingModel(**cfg.model.model_dump())
    if cfg.main.compile:
        model = torch.compile(model)
    model.to(cfg.trainer.device)
    criterion = nn.CrossEntropyLoss(**cfg.loss.model_dump())
    optimizer = SGD(model.parameters(), **cfg.optimizer.model_dump())
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=len(train_loader)
    )
    scheduler = ExponentialLR(optimizer, **cfg.scheduler.model_dump())
    metric = MulticlassAccuracy(
        num_classes=cfg.model.output_dim, ignore_index=cfg.loss.ignore_index
    )
    metric.to(cfg.trainer.device)
    return Trainer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        warmup_scheduler=warmup_scheduler,
        scheduler=scheduler,
        metric=metric,
        **cfg.trainer.model_dump(),
    )


def train_model(loaders: DataLoaders, cfg: Config):
    if cfg.main.use_best:
        with open(cfg.hparams.path, "r") as file:
            hyperparameters = json.load(file)
        cfg = set_hyperparameters(cfg=cfg, **hyperparameters)
        trainer = make_trainer(loaders.train_val, loaders.test, cfg=cfg)
        trainer.train()
    else:
        trainer = make_trainer(loaders.train, loaders.val, cfg=cfg)
        trainer.train()
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
