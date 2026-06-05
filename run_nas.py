"""Neural Architecture Search with Optuna.

Runs TPE-based NAS for YieldMLP or YieldFTTransformer, then retrains the best
architecture on train+val environments and evaluates on held-out test environments.

Usage:
    python run_nas.py --n-trials 10                              # quick smoke test (MLP)
    python run_nas.py --n-trials 75                              # full MLP run
    python run_nas.py --model-type ft_transformer --n-trials 50  # FT-Transformer NAS
    python run_nas.py --data vt.csv --out-dir artifacts/vt       # specify data / output
"""
import argparse
import json
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import YieldDataset, get_feature_cols, env_split, preprocessing_data
from ml.model import YieldFTTransformer, YieldMLP
from ml.nas import run_nas, run_nas_ft
from ml.trainer import run_epoch, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--model-type", default="mlp", choices=["mlp", "ft_transformer"])
    p.add_argument("--n-trials", type=int, default=10,
                   help="NAS trials (10=smoke test, 75=full MLP, 50=full FT-Transformer)")
    p.add_argument("--n-epochs", type=int, default=100, help="Max epochs per trial")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience per trial")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def _rebuild_mlp(params: dict, input_dim: int) -> YieldMLP:
    return YieldMLP(
        input_dim=input_dim,
        n_layers=params["n_layers"],
        hidden_size=params["hidden_size"],
        activation=params["activation"],
        dropout_rate=params["dropout_rate"],
        use_batch_norm=params["use_batch_norm"],
    )


def _rebuild_ft(params: dict, n_num: int, cat_cardinalities: list[int]) -> YieldFTTransformer:
    return YieldFTTransformer(
        n_num_features=n_num,
        cat_cardinalities=cat_cardinalities,
        d_token=params["d_token"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        d_ffn_factor=params["d_ffn_factor"],
        dropout=params["dropout"],
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type : {args.model_type}")
    print(f"NAS        : {args.n_trials} trials, up to {args.n_epochs} epochs, patience={args.patience}")

    print(f"\nLoading data from {args.data} ...")
    df = pd.read_csv(args.data)
    train_df, val_df, test_df = env_split(df, seed=args.seed)
    train_df, val_df, test_df, y_scaler = preprocessing_data((train_df, val_df, test_df))
    print(
        f"  train: {len(train_df):,} obs  "
        f"val: {len(val_df):,} obs  "
        f"test: {len(test_df):,} obs  (held out)"
    )

    feature_cols = get_feature_cols(train_df)
    input_dim = len(feature_cols)
    n_num = input_dim - 1 if "rm_id" in feature_cols else input_dim
    cat_cardinalities = [int(train_df["rm_id"].max()) + 1] if "rm_id" in feature_cols else []
    print(f"  features: {input_dim} total  ({n_num} continuous, {len(cat_cardinalities)} categorical)")

    train_ds = YieldDataset(train_df, feature_cols)
    val_ds = YieldDataset(val_df, feature_cols)
    test_ds = YieldDataset(test_df, feature_cols)

    print("\nRunning NAS...")
    if args.model_type == "mlp":
        study = run_nas(
            train_ds, val_ds,
            input_dim=input_dim,
            device=device,
            n_trials=args.n_trials,
            n_epochs=args.n_epochs,
            patience=args.patience,
        )
    else:
        study = run_nas_ft(
            train_ds, val_ds,
            n_num_features=n_num,
            cat_cardinalities=cat_cardinalities,
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

    print("\nRetraining best architecture on train+val ...")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    trainval_ds = YieldDataset(trainval_df, feature_cols)
    trainval_loader = DataLoader(trainval_ds, batch_size=best.params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    if args.model_type == "mlp":
        final_model = _rebuild_mlp(best.params, input_dim).to(device)
    else:
        final_model = _rebuild_ft(best.params, n_num, cat_cardinalities).to(device)

    train_model(
        final_model, trainval_loader, val_loader,
        lr=best.params["learning_rate"],
        weight_decay=best.params["weight_decay"],
        optimizer_name=best.params["optimizer"],
        n_epochs=args.n_epochs,
        patience=args.patience,
        device=device,
        y_scaler=y_scaler,
        verbose=True,
    )

    criterion = nn.MSELoss()
    test_rmse = run_epoch(final_model, test_loader, None, criterion, device, y_scaler, train=False)
    print(f"\nFinal test RMSE (held-out environments): {test_rmse:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"model_{args.model_type}.pt"
    torch.save(final_model.state_dict(), os.path.join(args.out_dir, fname))
    with open(os.path.join(args.out_dir, "optuna_study.pkl"), "wb") as f:
        pickle.dump(study, f)
    eval_report = {
        "model_type": args.model_type,
        "test_rmse": test_rmse,
        "best_trial": best.number,
        "best_val_rmse": best.value,
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
        "input_dim": input_dim,
        "n_num_features": n_num,
        "cat_cardinalities": cat_cardinalities,
    }
    with open(os.path.join(args.out_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Artifacts saved to {args.out_dir}/")
    print(f"  {fname}, optuna_study.pkl, eval_report.json")


if __name__ == "__main__":
    main()
