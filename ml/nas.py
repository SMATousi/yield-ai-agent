import math

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import YieldDataset
from ml.model import YieldMLP, RMYieldMLP
from ml.trainer import run_epoch,run_epoch_rm

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    train_ds: YieldDataset,
    val_ds: YieldDataset,
    input_dim: int,
    device: torch.device,
    n_epochs: int,
    patience: int,
) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = YieldMLP(input_dim, n_layers, hidden_size, activation, dropout_rate, use_batch_norm).to(device)
    opt_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(n_epochs):
        run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_rmse = run_epoch(model, val_loader, None, criterion, device, train=False)

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_rmse

def _objectiveRM(
    trial: optuna.Trial,
    train_ds: YieldDataset,
    val_ds: YieldDataset,
    input_dim: int,
    n_rm_levels: int,

    device: torch.device,
    n_epochs: int,
    patience: int,
) -> float:
    rm_embedding_dim = trial.suggest_int("rm_embedding_dim", 2, 16)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = RMYieldMLP(input_dim,n_rm_levels,rm_embedding_dim, n_layers, hidden_size, activation, dropout_rate, use_batch_norm).to(device)
    opt_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(n_epochs):
        run_epoch_rm(model, train_loader, optimizer, criterion, device, train=True)
        val_rmse = run_epoch_rm(model, val_loader, None, criterion, device, train=False)

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_rmse


def run_nas(
    train_ds: YieldDataset,
    val_ds: YieldDataset,
    input_dim: int,
    device: torch.device,
    n_trials: int = 75,
    n_epochs: int = 100,
    patience: int = 10,
) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # don't prune in the first 5 trials or first 10 epochs of a trial
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: _objective(trial, train_ds, val_ds, input_dim, device, n_epochs, patience),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def run_nas_rm(
    train_ds: YieldDataset,
    val_ds: YieldDataset,
    input_dim: int,
    n_rm_levels: int,
    device: torch.device,
    n_trials: int = 75,
    n_epochs: int = 100,
    patience: int = 10,
) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # don't prune in the first 5 trials or first 10 epochs of a trial
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: _objectiveRM(trial, train_ds, val_ds, input_dim, n_rm_levels, device, n_epochs, patience),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study