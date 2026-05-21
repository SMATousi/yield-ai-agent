"""Step 4: Baseline MLP training.

Trains a fixed-architecture MLP on synthetic data and reports val RMSE.
Replace make_synthetic_data() with real data once steps 1-3 are complete.

Usage:
    python train.py
    python train.py --n-layers 3 --hidden-size 256 --n-epochs 200
"""
import argparse
import json
import os

import torch

from ml.dataset import N_FEATURES, make_loaders, make_synthetic_data, env_split, preprocessing_data
from ml.model import YieldMLP
from ml.trainer import run_epoch, train_model

import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--activation", default="relu", choices=["relu", "gelu", "silu"])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    df = pd.read_csv("data.csv") #make_synthetic_data(seed=args.seed)

    #train_df, val_df, test_df = env_split(df, seed=args.seed)

    train_df, val_df, test_df = preprocessing_data(df,"feat_1" ,seed=args.seed)

    print(
        f"  train: {len(train_df):,} obs  "
        f"val: {len(val_df):,} obs  "
        f"test: {len(test_df):,} obs"
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    model = YieldMLP(
        input_dim=N_FEATURES,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        activation=args.activation,
        dropout_rate=args.dropout,
        use_batch_norm=args.batch_norm,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    print("Training...")
    best_val_rmse, history = train_model(
        model, train_loader, val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        n_epochs=args.n_epochs,
        patience=args.patience,
        device=device,
    )

    import math
    import torch.nn as nn
    criterion = nn.MSELoss()
    test_rmse = run_epoch(model, test_loader, None, criterion, device, train=False)
    print(f"\nBest val RMSE : {best_val_rmse:.4f}")
    print(f"Test RMSE     : {test_rmse:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nArtifacts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
