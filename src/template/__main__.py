from tomllib import load
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from template.config import Config
from template.dataset import make_dataloaders, make_splits
from template.evaluate import evaluate_model
from template.model import EmbeddingModel
from template.tune import tune_model
from template.train import train_model
from template.checkpoint import get_best_checkpoint_path, load_best_checkpoint
from template.examples import make_fake_sequence_dataset


def main():
    with open("config.toml", "rb") as f:
        config_data = load(f)
    config = Config(**config_data)
    df = make_fake_sequence_dataset()
    splits = make_splits(
        df, train_size=config.train_size, random_state=config.random_state
    )
    dataloaders = make_dataloaders(splits=splits, dataloader_config=config.dataloader)
    if config.tune:
        tune_model(dataloaders, config=config)
    if config.train:
        model = train_model(dataloaders=dataloaders, config=config)
    best_checkpoint_path = get_best_checkpoint_path(config.checkpoint)
    model = load_best_checkpoint(best_checkpoint_path, model_class=EmbeddingModel)
    if config.evaluate:
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
        print(f"AUROC: {auroc['mean']:.2f} ± {auroc['std']:.2f}")
        print(f"AP: {ap['mean']:.2f} ± {ap['std']:.2f}")
        # The raw bootstrap values (which are often easier to plot with
        # packages like seaborn) can be accessed via auroc["raw"] or ap["raw"]


if __name__ == "__main__":
    main()
