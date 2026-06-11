import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


N_FEATURES = 57

FEATURE_COLS = [f"feat_{i}" for i in range(N_FEATURES)]

# Columns that are never model inputs
_NON_FEATURE_COLS = {"yield", "env_id", "region", "dataset_source"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return predictor columns with rm_id (categorical) always last.

    This ordering is required by YieldFTTransformer, which expects
    len(cat_cardinalities) categorical features at the tail of the tensor.
    """
    cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    if "rm_id" in cols:
        cols = [c for c in cols if c != "rm_id"] + ["rm_id"]
    return cols


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

def are_rm_levels_in_train(dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rm_column_name: str = "rm",) -> bool:
    train_df, val_df, test_df = dfs

    train_rms = set(train_df[rm_column_name].unique())
    val_rms = set(val_df[rm_column_name].unique())
    test_rms = set(test_df[rm_column_name].unique())

    return val_rms.issubset(train_rms) and test_rms.issubset(train_rms)

def rm_safe_env_split(
        df: pd.DataFrame,
        rm_column_name: str = "rm",
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_df, val_df, test_df = env_split(df, val_frac,test_frac,seed)

    all_rms = are_rm_levels_in_train((train_df, val_df, test_df), rm_column_name)

    iteration = 0

    while not all_rms and iteration < 50:
          iteration += 1
          seed += 1

          train_df, val_df, test_df = env_split(df, val_frac,test_frac,seed)
          all_rms = are_rm_levels_in_train((train_df, val_df, test_df), rm_column_name)

    if not all_rms:
        raise ValueError("Could not find a split where all RM levels in val/test are present in train.")

    print(f"Seed used for splitting: {seed} after {iteration} retries.")
    return train_df, val_df, test_df

def impute_NA(dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rs_column_name: str, pop_column_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = dfs

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    
    imputer = SimpleImputer(strategy="most_frequent")

    train_df[[rs_column_name,pop_column_name]] = imputer.fit_transform(train_df[[rs_column_name,pop_column_name]])
    val_df[[rs_column_name,pop_column_name]] = imputer.transform(val_df[[rs_column_name,pop_column_name]])
    test_df[[rs_column_name,pop_column_name]] = imputer.transform(test_df[[rs_column_name,pop_column_name]])

    return train_df, val_df, test_df

def standardize_data(dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rm_column_name: str = "rm") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    train_df, val_df, test_df = dfs

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    feature_cols = [
        col for col in train_df.columns
        if col not in ["yield","env_id","region","dataset_source",rm_column_name,"feat_56","feat_57"] 
    ]

    scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    train_df[["yield"]] = y_scaler.fit_transform(train_df[["yield"]])
    val_df[["yield"]] = y_scaler.transform(val_df[["yield"]])
    test_df[["yield"]] = y_scaler.transform(test_df[["yield"]])

    return train_df, val_df, test_df, y_scaler

def encode_rm(dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rm_column_name: str = "rm") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = dfs

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    rm_levels = sorted(train_df[rm_column_name].unique())
    rm_ids = {mg: i for i, mg in enumerate(rm_levels)}

    train_df["rm_id"] = train_df[rm_column_name].map(rm_ids)
    val_df["rm_id"] = val_df[rm_column_name].map(rm_ids)
    test_df["rm_id"] = test_df[rm_column_name].map(rm_ids)

    train_df = train_df.drop(columns=[rm_column_name])
    val_df = val_df.drop(columns=[rm_column_name])
    test_df = test_df.drop(columns=[rm_column_name])

    return train_df, val_df, test_df

def preprocessing_data(dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                       rm_column_name: str = "rm",
                       rs_column_name: str = "feat_1", 
                       pop_column_name: str = "feat_0"
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_df, val_df, test_df = dfs

    train_df, val_df, test_df = impute_NA((train_df, val_df, test_df),rs_column_name,pop_column_name)

    train_df, val_df, test_df, y_scaler = standardize_data((train_df, val_df, test_df), rm_column_name)

    train_df, val_df, test_df = encode_rm((train_df, val_df, test_df),rm_column_name)

    return train_df, val_df, test_df, y_scaler

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
    feature_cols: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    return (
        DataLoader(YieldDataset(train_df, feature_cols), batch_size=batch_size, shuffle=True),
        DataLoader(YieldDataset(val_df, feature_cols), batch_size=256),
        DataLoader(YieldDataset(test_df, feature_cols), batch_size=256),
    )
