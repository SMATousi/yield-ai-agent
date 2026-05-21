import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

N_FEATURES = 50
FEATURE_COLS = [f"feat_{i}" for i in range(N_FEATURES)]


def make_synthetic_data(n_obs: int = 8500, n_envs: int = 608, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset mimicking the real 50-predictor schema (placeholder for steps 1-3)."""
    rng = np.random.default_rng(seed)
    env_ids = rng.integers(0, n_envs, size=n_obs)
    X = rng.standard_normal((n_obs, N_FEATURES)).astype(np.float32)
    coef = rng.standard_normal(N_FEATURES).astype(np.float32)
    env_effect = rng.standard_normal(n_envs).astype(np.float32)
    # linear signal + environment effect + noise
    yield_ = X @ coef + env_effect[env_ids] + rng.standard_normal(n_obs).astype(np.float32) * 0.5
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["env_id"] = env_ids
    df["yield"] = yield_
    return df


def env_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by environment ID so no environment appears in more than one split."""
    envs = df["env_id"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(envs)
    n = len(envs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_envs = set(envs[:n_test])
    val_envs = set(envs[n_test : n_test + n_val])
    train_envs = set(envs[n_test + n_val :])
    return (
        df[df["env_id"].isin(train_envs)].reset_index(drop=True),
        df[df["env_id"].isin(val_envs)].reset_index(drop=True),
        df[df["env_id"].isin(test_envs)].reset_index(drop=True),
    )

def source_env_split(
        df: pd.DataFrame,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

     """Split by environment ID, stratified by dataset_source."""

     df_env = (
         df[["env_id","dataset_source"]]
         .drop_duplicates()
         .reset_index(drop = True)
         )
     
     train_vals_envs, test_envs = train_test_split(
         df_env,
         test_size = test_frac,
         random_state= seed,
         stratify = df_env["dataset_source"]
     )

     val_size_adj = val_frac / (1 - test_frac)

     train_envs, val_envs = train_test_split(
         train_vals_envs,
         test_size = val_size_adj,
         random_state = seed,
         stratify = train_vals_envs["dataset_source"]
     )

     train_envs = set(train_envs["env_id"])
     val_envs = set(val_envs["env_id"])
     test_envs = set(test_envs["env_id"])

     return (
         df[df["env_id"].isin(train_envs)].reset_index(drop=True),
         df[df["env_id"].isin(val_envs)].reset_index(drop=True),
         df[df["env_id"].isin(test_envs)].reset_index(drop=True),
         )



class YieldDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["yield"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(YieldDataset(train_df), batch_size=batch_size, shuffle=True),
        DataLoader(YieldDataset(val_df), batch_size=256),
        DataLoader(YieldDataset(test_df), batch_size=256),
    )
