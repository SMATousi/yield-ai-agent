"""Step 5: Neural Architecture Search with Optuna.

Runs TPE-based NAS over the MLP search space and saves the best model + study.
Replace make_synthetic_data() with real data once steps 1-3 are complete.

Usage:
    python run_nasRM.py --n-trials 10          # quick smoke test
    python run_nasRM.py --n-trials 75          # full prototype run
    python run_nasRM.py --n-trials 150         # full production run
"""
import argparse
import json
import os
import pickle

import torch

from ml.dataset import N_FEATURES, YieldDataset, env_split, source_env_split, rm_safe_source_env_split, preprocessing_data,rm_safe_env_split
from ml.model import RMYieldMLP
from ml.nas import run_nas_rm
from ml.trainer import run_epoch_rm, train_model_rm

import torch.nn as nn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def evaluate_rm_with_predictions(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    n = 0

    y_obs = []
    y_pred = []

    with torch.no_grad():
        for X, rm_id, y in loader:
            X = X.to(device)
            rm_id = rm_id.to(device)
            y = y.to(device)

            pred = model(X, rm_id)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * len(y)
            n += len(y)

            y_obs.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_obs = np.concatenate(y_obs)
    y_pred = np.concatenate(y_pred)

    rmse = np.sqrt(total_loss / n)
    r2 = r2_score(y_obs, y_pred)

    return rmse, r2, y_obs, y_pred

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=10, help="NAS trials (10=smoke test, 75-150=full)")
    p.add_argument("--n-epochs", type=int, default=100, help="Max epochs per trial")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience per trial")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"NAS: {args.n_trials} trials, up to {args.n_epochs} epochs each, patience={args.patience}")

    print("Loading data...")
    df = pd.read_csv("data.csv") #make_synthetic_data(seed=args.seed)

    train_df, val_df, test_df = rm_safe_env_split(df, seed=args.seed)

    #train_df, val_df, test_df = source_env_split(df, seed=args.seed)

    #train_df, val_df, test_df = rm_safe_source_env_split(df,seed=args.seed) 

    train_df, val_df, test_df = preprocessing_data((train_df, val_df, test_df))

    print(
        f"  train: {len(train_df):,} obs  "
        f"val: {len(val_df):,} obs  "
        f"test: {len(test_df):,} obs  (held out)"
    )

    train_ds = YieldDataset(train_df)
    val_ds = YieldDataset(val_df)
    test_ds = YieldDataset(test_df)

    print("\nRunning NAS...")
    study = run_nas_rm(
        train_ds, val_ds,
        input_dim=N_FEATURES,
        n_rm_levels=df["rm"].nunique(),
        device=device,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        patience=args.patience,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}  val RMSE={best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    print("\nRetraining best architecture on train+val...")
    
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    trainval_ds = YieldDataset(trainval_df)
    from torch.utils.data import DataLoader
    trainval_loader = DataLoader(trainval_ds, batch_size=best.params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    final_model = RMYieldMLP(
        input_dim=N_FEATURES,
        n_rm_levels=df["rm"].nunique(), 
        rm_embedding_dim=best.params["rm_embedding_dim"],
        n_layers=best.params["n_layers"],
        hidden_size=best.params["hidden_size"],
        activation=best.params["activation"],
        dropout_rate=best.params["dropout_rate"],
        use_batch_norm=best.params["use_batch_norm"],
    ).to(device)

    # retrain until convergence on train+val (use val as a dummy — same split)
    val_loader = DataLoader(val_ds, batch_size=256)
    train_model_rm(
        final_model, trainval_loader, val_loader,
        lr=best.params["learning_rate"],
        weight_decay=best.params["weight_decay"],
        optimizer_name=best.params["optimizer"],
        n_epochs=args.n_epochs,
        patience=args.patience,
        device=device,
        verbose=True,
    )

    criterion = nn.MSELoss()

    test_rmse, test_r2, y_obs, y_pred = evaluate_rm_with_predictions(
        final_model,
        test_loader,
        criterion,
        device,
    )

    print(f"\nFinal test RMSE (held-out environments): {test_rmse:.4f}")
    print(f"Final test R²   (held-out environments): {test_r2:.4f}")

    min_value = min(y_obs.min(), y_pred.min())
    max_value = max(y_obs.max(), y_pred.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(y_obs, y_pred, alpha=0.5)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Observed yield")
    plt.ylabel("Predicted yield")
    plt.title(f"Observed vs Predicted | RMSE = {test_rmse:.2f}, R² = {test_r2:.2f}")
    plt.tight_layout()
    plt.show()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(final_model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "optuna_study.pkl"), "wb") as f:
        pickle.dump(study, f)
    eval_report = {
        "test_rmse": test_rmse,
        "best_trial": best.number,
        "best_val_rmse": best.value,
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
    }
    with open(os.path.join(args.out_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Artifacts saved to {args.out_dir}/")
    print(f"  model.pt, optuna_study.pkl, eval_report.json")


if __name__ == "__main__":
    main()
