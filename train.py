"""Baseline training script — fixed architecture, no NAS.

Supports both YieldMLP and YieldFTTransformer via --model-type.
For FT-Transformer fine-tuning from a pre-trained checkpoint use --pretrained.

Usage:
    python train.py                                          # MLP with defaults
    python train.py --model-type ft_transformer              # FT-Transformer
    python train.py --model-type ft_transformer \\
        --pretrained artifacts/model_vt.pt                  # fine-tune pre-trained backbone
    python train.py --data vt.csv --out-dir artifacts/vt    # specify data file and output dir
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from ml.dataset import get_feature_cols, make_loaders, rm_safe_env_split, preprocessing_data,env_split
from ml.model import YieldFTTransformer, YieldMLP
from ml.trainer import run_epoch, train_model

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--model-type", default="mlp", choices=["mlp", "ft_transformer"])
    p.add_argument("--pretrained", default=None,
                   help="Path to pre-trained FT-Transformer checkpoint for fine-tuning")

    # MLP hyperparameters
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--activation", default="relu", choices=["relu", "gelu", "silu", "tanh", "sig"])
    p.add_argument("--batch-norm", action="store_true")

    # FT-Transformer hyperparameters
    p.add_argument("--d-token", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ffn-factor", type=float, default=2.0)

    # Shared training hyperparameters
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def build_model(args, input_dim: int, n_num: int, cat_cardinalities: list[int]) -> nn.Module:
    if args.model_type == "mlp":
        return YieldMLP(
            input_dim=input_dim,
            n_layers=args.n_layers,
            hidden_size=args.hidden_size,
            activation=args.activation,
            dropout_rate=args.dropout,
            use_batch_norm=args.batch_norm,
        )
    return YieldFTTransformer(
        n_num_features=n_num,
        cat_cardinalities=cat_cardinalities,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ffn_factor=args.d_ffn_factor,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)

    train_df, val_df, test_df = rm_safe_env_split(df, seed=args.seed)
    train_df, val_df, test_df, y_scaler = preprocessing_data((train_df, val_df, test_df))
    print(
        f"  train: {len(train_df):,} obs  "
        f"val: {len(val_df):,} obs  "
        f"test: {len(test_df):,} obs"
    )

    feature_cols = get_feature_cols(train_df)
    input_dim = len(feature_cols)
    n_num = input_dim - 1 if "rm_id" in feature_cols else input_dim
    cat_cardinalities = [int(train_df["rm_id"].max()) + 1] if "rm_id" in feature_cols else []

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, batch_size=args.batch_size, feature_cols=feature_cols
    )

    model = build_model(args, input_dim, n_num, cat_cardinalities).to(device)

    if args.pretrained:
        print(f"Loading backbone from {args.pretrained} ...")
        state = torch.load(args.pretrained, map_location=device)
        # Load only matching keys so the head can be re-initialized
        current = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        current.update(compatible)
        model.load_state_dict(current)
        print(f"  Loaded {len(compatible)}/{len(state)} parameter tensors from checkpoint")

        # Stage 1: head-only (backbone frozen)
        model.freeze_backbone()
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"Fine-tuning stage 1: head only ({n_frozen} tensors frozen)")
        train_model(
            model, train_loader, val_loader,
            lr=args.lr * 10,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            n_epochs=30,
            patience=args.patience,
            device=device,
            y_scaler=y_scaler,
        )

        # Stage 2: full network (backbone unfrozen)
        model.unfreeze_backbone()
        print("Fine-tuning stage 2: full network")
        lr_stage2 = min(args.lr, 1e-4)

    else:
        lr_stage2 = args.lr

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ({args.model_type}): {n_params:,} parameters")
    print("Training...")
    best_val_rmse, history = train_model(
        model, train_loader, val_loader,
        lr=lr_stage2,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        n_epochs=args.n_epochs,
        patience=args.patience,
        device=device,
        y_scaler=y_scaler,
    )

    criterion = nn.MSELoss()
    test_rmse = run_epoch(model, test_loader, None, criterion, device, y_scaler, train=False)
    print(f"\nBest val RMSE : {best_val_rmse:.4f}")
    print(f"Test RMSE     : {test_rmse:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"model_{args.model_type}.pt"
    torch.save(model.state_dict(), os.path.join(args.out_dir, fname))
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nArtifacts saved to {args.out_dir}/  ({fname})")


if __name__ == "__main__":
    main()
