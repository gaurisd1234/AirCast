"""
Tests for data ingestion, preprocessing, leak-free splitting, and sliding window generation.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from src.data import (
    preprocess_data,
    split_chronological,
    fit_scalers_and_transform,
    create_sliding_windows,
    AirQualityDataset,
    CONTINUOUS_COLUMNS
)


@pytest.fixture
def sample_raw_df():
    """Create a synthetic dataframe mimicking Beijing PM2.5 schema."""
    dates = pd.date_range("2010-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        "No": range(1, 201),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "pm2.5": [np.nan if i < 24 else (50.0 + (i % 20) * 5.0) for i in range(200)],
        "DEWP": np.linspace(-20, 10, 200),
        "TEMP": np.linspace(-10, 25, 200),
        "PRES": np.linspace(1010, 1030, 200),
        "cbwd": ["NW", "SE", "NE", "cv"] * 50,
        "Iws": np.random.uniform(0.5, 25.0, 200),
        "Is": [0 if i % 10 != 0 else 1 for i in range(200)],
        "Ir": [0 if i % 15 != 0 else 2 for i in range(200)]
    })
    return df


def test_preprocess_drops_initial_na_and_creates_cyclical(sample_raw_df):
    clean = preprocess_data(sample_raw_df)
    # The first 24 unmeasured rows should be dropped
    assert len(clean) == 200 - 24
    assert clean["pm2.5"].isna().sum() == 0
    # Check cyclical feature identity: sin^2 + cos^2 == 1
    hour_circ = clean["hour_sin"]**2 + clean["hour_cos"]**2
    np.testing.assert_allclose(hour_circ, 1.0, atol=1e-5)
    month_circ = clean["month_sin"]**2 + clean["month_cos"]**2
    np.testing.assert_allclose(month_circ, 1.0, atol=1e-5)


def test_chronological_split_no_leakage(sample_raw_df):
    clean = preprocess_data(sample_raw_df)
    train_df, val_df, test_df = split_chronological(clean, 0.70, 0.15, 0.15)
    
    assert len(train_df) + len(val_df) + len(test_df) == len(clean)
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()


def test_scaler_fit_strictly_on_train(sample_raw_df):
    clean = preprocess_data(sample_raw_df)
    train_df, val_df, test_df = split_chronological(clean, 0.70, 0.15, 0.15)
    train_scaled, val_scaled, test_scaled, scaler_info, scaler = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )

    # Train continuous features should have approximately 0 mean and 1 variance
    for col in CONTINUOUS_COLUMNS:
        np.testing.assert_allclose(train_scaled[col].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(train_scaled[col].std(ddof=0), 1.0, atol=1e-5)
    
    # Target mean and std must match the train target column
    assert abs(scaler_info["target_mean"] - train_df["pm2.5"].mean()) < 1e-4
    assert abs(scaler_info["target_std"] - train_df["pm2.5"].std(ddof=0)) < 1e-4


def test_sliding_window_shapes_and_temporal_alignment():
    # Synthetic array with 50 timesteps, 5 features
    data = np.arange(50 * 5, dtype=np.float32).reshape(50, 5)
    lookback = 10
    horizon = 4
    X, Y = create_sliding_windows(data, target_idx=0, lookback=lookback, horizon=horizon)

    expected_samples = 50 - lookback - horizon + 1  # 50 - 10 - 4 + 1 = 37
    assert X.shape == (expected_samples, lookback, 5)
    assert Y.shape == (expected_samples, horizon)

    # First sample: input covers timesteps 0..9, target covers 10..13 for feature 0
    np.testing.assert_allclose(X[0, :, 0], data[0:10, 0])
    np.testing.assert_allclose(Y[0], data[10:14, 0])

    # Ensure zero overlap: the last input step < the first target step
    assert X[0, -1, 0] < Y[0, 0]


def test_dataset_pytorch_tensor_output():
    X = np.random.randn(20, 15, 8).astype(np.float32)
    Y = np.random.randn(20, 4).astype(np.float32)
    dataset = AirQualityDataset(X, Y)

    assert len(dataset) == 20
    x_item, y_item = dataset[0]
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)
    assert x_item.shape == (15, 8)
    assert y_item.shape == (4,)
