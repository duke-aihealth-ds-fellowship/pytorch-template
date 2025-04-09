from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
from torchmetrics.regression import MeanAbsoluteError
from tqdm import tqdm

from template.config import Config, SchedulerConfig

# from template.loss import DiscreteFailureTimeNLL
from template.model import Transformer


class Scheduler:
    def __init__(self, scheduler: LRScheduler, warmup_scheduler: LRScheduler) -> None:
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler

    def step(self):
        self.scheduler.step()

    def warmup_step(self):
        self.warmup_scheduler.step()


def to_device(
    batch: dict[str, Tensor], keys: list[str], device: str
) -> tuple[Tensor, ...]:
    return tuple(batch[key].to(device) for key in keys)


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        model: nn.Module | Callable,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Scheduler,
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
        self.td_keys = ["features", "labels"]

    def update_progress(self):
        postfix = (
            f"lr: {self.optimizer.param_groups[0]['lr']:.2e}, "
            f"Train loss: {self.train_loss:.3f}, "
            f"Eval loss: {self.eval_loss:.3f}, "
            f", Eval metric: {self.eval_metric:.3f}"
        )
        self.progress_bar.set_postfix_str(postfix)
        self.progress_bar.update()

    def train_step(self, batch: TensorDict) -> float:
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch["features"])
        loss = self.criterion(outputs, batch["label"])
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
        self.scheduler.warmup_step()
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
        for _ in range(self.max_epochs):
            self.train_epoch()
            self.evaluate()
        self.progress_bar.close()
        return self.train_loss

    @torch.no_grad()
    def evaluate_step(self, batch: TensorDict) -> float:
        batch = batch.to(self.device)
        outputs = self.model(batch["features"])
        self.metric.update(outputs, batch["label"])
        return self.criterion(outputs, batch["label"]).item()

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


def make_model(cfg: Config) -> nn.Module:
    if isinstance(cfg.simulation.parameters, list):
        cfg.model.input_dim = len(cfg.simulation.parameters)
    else:
        cfg.model.input_dim = cfg.simulation.parameters
    model = Transformer(**cfg.model.model_dump())
    if cfg.compile:
        model = torch.compile(model)
    model.to(cfg.trainer.device)
    return model  # type: ignore


def make_loss_fn(cfg: Config) -> nn.Module:
    if cfg.task == "bc":
        return nn.BCEWithLogitsLoss()
    elif cfg.task == "cls":
        return nn.CrossEntropyLoss(ignore_index=cfg.loss.ignore_index)
    elif cfg.task == "reg":
        return nn.MSELoss()
    # elif cfg.task == "tte":
    #     return DiscreteFailureTimeNLL(ignore_index=cfg.loss.ignore_index)
    else:
        raise ValueError(
            f"Unknown task: {cfg.task}. Choose from 'bc', 'cls', 'reg', 'tte', 'mxc'."
        )


def make_metric(cfg: Config) -> Metric:
    if cfg.task == "bc":
        return BinaryAUROC()
    elif cfg.task == "cls":
        return MulticlassAUROC(num_classes=cfg.model.output_dim)
    elif cfg.task == "reg":
        return MeanAbsoluteError()
    # elif cfg.task == "tte":
    #     return TimeVaryingAUC(ignore_index=cfg.loss.ignore_index)
    else:
        raise ValueError(
            f"Unknown task: {cfg.task}. Choose from 'bc', 'cls', 'reg', 'tte', 'mxc'."
        )


def make_scheduler(optimizer: Optimizer, cfg: SchedulerConfig) -> Scheduler:
    if cfg.warmup_steps == 0:
        start_factor = 1.0
    else:
        start_factor = cfg.start_factor
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )
    lr_scheduler = ExponentialLR(optimizer, gamma=cfg.gamma)
    scheduler = Scheduler(scheduler=lr_scheduler, warmup_scheduler=warmup_scheduler)
    return scheduler


def make_trainer(
    train_loader: DataLoader, eval_loader: DataLoader, cfg: Config
) -> Trainer:
    model = make_model(cfg=cfg)
    criterion = make_loss_fn(cfg=cfg)
    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    scheduler = make_scheduler(optimizer=optimizer, cfg=cfg.scheduler)
    metric = make_metric(cfg=cfg)
    metric.to(cfg.trainer.device)
    return Trainer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
        **cfg.trainer.model_dump(),
    )
