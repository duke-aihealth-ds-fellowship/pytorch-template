from tomllib import load
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from example.config import Config
from example.dataset import make_dataloaders, make_splits
from example.evaluate import evaluate_model
from example.tune import tune_model
from example.train import train_model
from example.checkpoint import load_best_checkpoint
from example.examples import make_fake_sequence_dataset


def main():
    with open("config.toml", "rb") as f:
        config_data = load(f)
    config = Config(**config_data)
    df = make_fake_sequence_dataset()
    print(df)
    # Make train, validation, and test splits
    splits = make_splits(
        df, train_size=config.train_size, random_state=config.random_state
    )
    dataloaders = make_dataloaders(splits=splits, dataloader_config=config.dataloader)
    # Tune model hyperparameters
    if config.tune:
        tune_model(dataloaders, config=config)
    if config.train:  # Train model with default hyperparameters
        model = train_model(dataloaders=dataloaders, config=config)
    else:  # Load the best model from a previous training or tuning run
        model = load_best_checkpoint(config.checkpoint_path)
    if config.evaluate:  # Generate model performance metrics from the test set
        test_metrics = {"AUROC": BinaryAUROC(), "AP": BinaryAveragePrecision()}
        metrics = evaluate_model(
            model=model,
            dataloader=dataloaders.test,
            metrics=test_metrics,
            device=config.trainer.device,
            n_bootstraps=config.evaluator.n_bootstraps,
        )
        auroc = metrics["AUROC"]
        ap = metrics["AP"]
        print(f"AUROC: {auroc['mean']} ± {auroc['std']}")
        print(f"AP: {ap['mean']} ± {ap['std']}")


if __name__ == "__main__":
    main()
