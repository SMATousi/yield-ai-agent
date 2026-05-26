import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.model import YieldMLP, RMYieldMLP


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
) -> float:
    """Run one epoch; return RMSE."""
    model.train(train)
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for X, rm_id_dummy, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
    return math.sqrt(total_loss / n)

def run_epoch_rm(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
) -> float:
    """Run one epoch; return RMSE."""
    model.train(train)
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for X, rm_id, y in loader:
            X, y = X.to(device), y.to(device)
            rm_id = rm_id.to(device).long()
            pred = model(X, rm_id)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
    return math.sqrt(total_loss / n)


def train_model(
    model: YieldMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    n_epochs: int,
    patience: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, list[dict]]:
    """Train with early stopping; return (best_val_rmse, history)."""
    opt_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_rmse = run_epoch(model, val_loader, None, criterion, device, train=False)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4d}  train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_rmse, history

def train_model_rm(
    model: RMYieldMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    n_epochs: int,
    patience: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, list[dict]]:
    """Train with early stopping; return (best_val_rmse, history)."""
    opt_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch_rm(model, train_loader, optimizer, criterion, device, train=True)
        val_rmse = run_epoch_rm(model, val_loader, None, criterion, device, train=False)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4d}  train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_rmse, history