from tomllib import load
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from example.evaluate import evaluate_model
from example.tune import tune_model
from example.config import Config
from example.dataset import make_dataloaders, make_fake_dataset, make_splits
from example.train import train_model
from example.checkpoint import load_best_checkpoint


def main():
    with open("config.toml", "rb") as f:
        config_data = load(f)
    config = Config(**config_data)

    df = make_fake_dataset()
    print(df)

    train, val, test = make_splits(
        df, train_size=config.train_size, random_state=config.random_seed
    )
    dataloaders = make_dataloaders(
        train=train, val=val, test=test, dataloader_config=config.dataloader
    )
    # TODO This is specific to the example dataset
    config.model.vocab_size = train["input"].explode().n_unique()
    if config.tune:
        tune_model(dataloaders, config=config)
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
