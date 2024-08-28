# TODO add hyperparameter tuning code here
from functools import partial
from optuna import create_study, Trial
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from example.evaluate import evaluate_predictions, get_predictions
from example.train import make_components, train_model


def objective(
    trial: Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    embedding_dim: int,
    padding_idx: int,
    output_dim: int,
    max_epochs: int,
    device: str,
) -> float:
    hyperparameters = {
        "hidden_dim": trial.suggest_categorical(
            name="hidden_dim", choices=[16, 32, 64]
        ),
        "n_layers": trial.suggest_int(name="n_layers", low=1, high=6),
        "lr": trial.suggest_float(name="lr", low=1e-5, high=1e-2),
        "weight_decay": trial.suggest_float(name="weight_decay", low=1e-10, high=1e-2),
        "momentum": trial.suggest_float(name="momentum", low=0.9, high=0.99),
    }
    model, optimizer, criterion = make_components(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        output_dim=output_dim,
        device=device,
        **hyperparameters,
    )
    model = train_model(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=max_epochs,
    )
    outputs, labels = get_predictions(model=model, dataloader=val_loader)
    metrics = evaluate_predictions(
        outputs=outputs, labels=labels, metrics={"val_loss": criterion}
    )
    return metrics["val_loss"].item()


def tune_model(
    train_loader,
    val_loader,
    n_trials: int,
    vocab_size: int,
    embedding_dim: int,
    padding_idx: int,
    output_dim: int,
    max_epochs: int,
    device: str,
    random_seed: int,
):
    sampler = TPESampler(seed=random_seed)
    study = create_study(
        sampler=sampler,
        direction="minimize",
        study_name="ABCD",
    )
    objective_function = partial(
        objective,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        output_dim=output_dim,
        max_epochs=max_epochs,
        device=device,
    )
    study.optimize(func=objective_function, n_trials=n_trials)
    model, optimizer, criterion = make_components(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        output_dim=output_dim,
        device=device,
        **study.best_params,
    )
    model = train_model(
        model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=max_epochs,
    )
    return model
