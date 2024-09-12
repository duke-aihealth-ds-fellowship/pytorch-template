from toml import load
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from example.evaluate import evaluate_model
from example.tune import tune_model
from example.config import Config
from example.dataset import make_dataloaders, make_fake_dataset, make_splits
from example.train import train_model
from example.checkpoint import load_best_checkpoint
import json


def main():
    config_data = load("config.toml")
    config = Config(**config_data)

    df = make_fake_dataset()
    print(df)

    train, val, test = make_splits(
        df, train_size=config.train_size, random_state=config.random_seed
    )
    dataloaders = make_dataloaders(
        train=train, val=val, test=test, dataloader_config=config.dataloader
    )
    vocab_size = train["input"].explode().n_unique()
    config.model.vocab_size = vocab_size
    if config.tune:
        study = tune_model(dataloaders, config=config)
        print("Best model hyperparameters:\n", json.dumps(study.best_params, indent=4))
        print(f"Best model checkpoint saved in: {config.checkpoint_path}")
    if config.train:
        model = train_model(dataloaders=dataloaders, config=config)
    else:
        model = load_best_checkpoint(config.checkpoint_path)
    if config.evaluate:
        test_metrics = {"AUROC": BinaryAUROC(), "AP": BinaryAveragePrecision()}
        metrics = evaluate_model(
            model=model,
            dataloader=dataloaders.test,
            metrics=test_metrics,
            n_bootstraps=config.evaluator.n_bootstraps,
        )
        print(metrics)


if __name__ == "__main__":
    main()
