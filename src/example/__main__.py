from toml import load
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import polars as pl

from example.evaluate import evaluate_model
from example.tune import tune_model
from example.config import Config
from example.dataset import make_dataloaders, make_fake_dataset, make_splits
from example.train import make_components, train_model


def main():
    config_data = load("config.toml")
    config = Config(**config_data)

    df = make_fake_dataset()
    print(df)

    train, val, test = make_splits(
        df, train_size=config.train_size, random_state=config.random_seed
    )
    vocab_size = train["input"].explode().n_unique()
    train_loader, val_loader, test_loader = make_dataloaders(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        device=config.device,
    )
    if config.tune:
        model = tune_model(
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=config.n_trials,
            vocab_size=vocab_size,
            embedding_dim=config.model.embedding_dim,
            output_dim=config.model.output_dim,
            padding_idx=config.padding_idx,
            max_epochs=config.training.max_epochs,
            device=config.device,
            random_seed=config.random_seed,
        )
    else:
        model, optimizer, criterion = make_components(
            vocab_size=vocab_size,
            embedding_dim=config.model.embedding_dim,
            padding_idx=config.padding_idx,
            device=config.device,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            n_layers=config.model.n_layers,
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=config.training.max_epochs,
        )
    if config.evaluate:
        # you can add or remove metrics here
        test_metrics = {"AUROC": BinaryAUROC(), "AP": BinaryAveragePrecision()}
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            metrics=test_metrics,
            n_bootstraps=config.n_bootstraps,
        )
        # warning, this code is only for demonstration purposes and
        # will likely break for multiclass or multilabel metrics
        metrics = pl.DataFrame(metrics).melt()
        metric_summary = metrics.groupby("variable").agg(
            pl.col("value").mean().alias("mean"), pl.col("value").std().alias("std")
        )
        print(metric_summary)


if __name__ == "__main__":
    main()
