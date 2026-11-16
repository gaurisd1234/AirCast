"""
Data loading, preprocessing, chronological splitting, and window generation for AirCast.
"""

import os
import urllib.request
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


CONTINUOUS_COLUMNS = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
CATEGORICAL_COLUMNS = ["cbwd"]


def download_and_load_raw_data(raw_url: str, save_path: str) -> pd.DataFrame:
    """
    Download Beijing PM2.5 raw dataset if not present on disk, then load into a DataFrame.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Downloading raw dataset from {raw_url} to {save_path}...")
        urllib.request.urlretrieve(raw_url, save_path)
        print("Download complete.")
    else:
        print(f"Using cached raw dataset at {save_path}.")
    
    df = pd.read_csv(save_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw pollution dataframe:
    1. Drop initial unmeasured 24 hours of 2010-01-01 where target is absent.
    2. Construct unified datetime index.
    3. Causal forward-fill missing target values (measurements carry forward).
    4. Engineer cyclical temporal features (hour, month, day-of-week).
    5. One-hot encode wind direction ('cbwd').
    """
    # Drop initial unmeasured 24 hours where sensor was not yet active
    df_clean = df.iloc[24:].copy().reset_index(drop=True)

    # Build datetime column for chronological ordering and temporal features
    df_clean["datetime"] = pd.to_datetime(df_clean[["year", "month", "day", "hour"]])
    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)

    # Causal forward fill for missing PM2.5 concentrations (zero leakage)
    df_clean["pm2.5"] = df_clean["pm2.5"].ffill()

    # Cyclical temporal encoding for hour of day (period = 24)
    # Why: 23:00 and 00:00 are 1 hour apart, but numerically 23 and 0 appear distant.
    # Sine/cosine projections place them adjacent on the unit circle.
    hour = df_clean["datetime"].dt.hour
    df_clean["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_clean["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Cyclical temporal encoding for month of year (period = 12)
    month = df_clean["datetime"].dt.month - 1  # 0-indexed month
    df_clean["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_clean["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Cyclical temporal encoding for day of week (period = 7)
    dayofweek = df_clean["datetime"].dt.dayofweek
    df_clean["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df_clean["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    # One-hot encode categorical wind direction (cbwd: NE, NW, SE, cv)
    cbwd_dummies = pd.get_dummies(df_clean["cbwd"], prefix="cbwd", dtype=float)
    df_clean = pd.concat([df_clean, cbwd_dummies], axis=1)

    # Drop raw date/time components and non-feature columns
    columns_to_drop = ["No", "year", "month", "day", "hour", "cbwd"]
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])

    # Ensure target pm2.5 is the very first column for convenient index tracking
    cols = ["pm2.5"] + [col for col in df_clean.columns if col not in ["pm2.5", "datetime"]] + ["datetime"]
    df_clean = df_clean[cols]

    return df_clean


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data strictly in chronological order without shuffling.
    Train: [0, train_end)
    Val:   [train_end, val_end)
    Test:  [val_end, total_len)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def fit_scalers_and_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    continuous_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], StandardScaler]:
    """
    Fit StandardScaler STRICTLY on the training set continuous features.
    Transform train, validation, and test sets using the fitted parameters.
    Returns transformed dataframes along with target mean and std for unscaling.
    """
    # Verify no NaNs remain before scaling
    train_target_median = float(train_df["pm2.5"].median())
    for d in (train_df, val_df, test_df):
        d["pm2.5"] = d["pm2.5"].fillna(train_target_median)

    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])

    # Extract target pm2.5 scaler parameters (pm2.5 is index 0 in continuous_cols)
    pm25_idx = continuous_cols.index("pm2.5")
    target_mean = float(scaler.mean_[pm25_idx])
    target_std = float(scaler.scale_[pm25_idx])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[continuous_cols] = scaler.transform(train_df[continuous_cols])
    val_scaled[continuous_cols] = scaler.transform(val_df[continuous_cols])
    test_scaled[continuous_cols] = scaler.transform(test_df[continuous_cols])

    scaler_info = {
        "target_mean": target_mean,
        "target_std": target_std,
        "target_median": train_target_median
    }

    return train_scaled, val_scaled, test_scaled, scaler_info, scaler


def create_sliding_windows(
    data: np.ndarray,
    target_idx: int = 0,
    lookback: int = 72,
    horizon: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding windows for multi-horizon forecasting.
    Input X shape:  (N, lookback, num_features)
    Target Y shape: (N, horizon)
    
    Zero leakage check:
    For any sample i, input covers time [i : i + lookback - 1],
    and target covers future time [i + lookback : i + lookback + horizon - 1].
    There is 0 temporal overlap between historical input and future target.
    """
    n_samples = len(data) - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Data length ({len(data)}) is too short for lookback ({lookback}) + horizon ({horizon})"
        )

    X = np.empty((n_samples, lookback, data.shape[1]), dtype=np.float32)
    Y = np.empty((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        # Lookback input window
        X[i] = data[i : i + lookback]
        # Multi-horizon future target values
        Y[i] = data[i + lookback : i + lookback + horizon, target_idx]

    return X, Y


class AirQualityDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding window inputs X and targets Y.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, float], List[str]]:
    """
    Full leak-free data pipeline:
    1. Load raw data
    2. Preprocess & engineer features
    3. Split chronologically into train, val, test
    4. Fit scalers on train only, transform all splits
    5. Generate sliding windows (lookback=72, horizon=24)
    6. Construct PyTorch DataLoaders (shuffle=True on train only, False on val/test)
    """
    raw_url = config["data"]["raw_url"]
    raw_path = config["data"]["raw_path"]
    lookback = config["data"]["lookback_hours"]
    horizon = config["data"]["max_horizon"]
    batch_size = config["training"]["batch_size"]

    df_raw = download_and_load_raw_data(raw_url, raw_path)
    df_clean = preprocess_data(df_raw)

    train_df, val_df, test_df = split_chronological(
        df_clean,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"]
    )

    feature_cols = [c for c in train_df.columns if c != "datetime"]
    target_idx = feature_cols.index("pm2.5")

    train_scaled, val_scaled, test_scaled, scaler_info, _ = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )

    train_arr = train_scaled[feature_cols].values.astype(np.float32)
    val_arr = val_scaled[feature_cols].values.astype(np.float32)
    test_arr = test_scaled[feature_cols].values.astype(np.float32)

    X_train, Y_train = create_sliding_windows(train_arr, target_idx, lookback, horizon)
    X_val, Y_val = create_sliding_windows(val_arr, target_idx, lookback, horizon)
    X_test, Y_test = create_sliding_windows(test_arr, target_idx, lookback, horizon)

    train_dataset = AirQualityDataset(X_train, Y_train)
    val_dataset = AirQualityDataset(X_val, Y_val)
    test_dataset = AirQualityDataset(X_test, Y_test)

    # Note: Shuffling window samples during training breaks correlation between adjacent batches,
    # which stabilizes SGD optimization. Val and test loaders are NEVER shuffled to maintain order.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_info, feature_cols
