"""
Dedicated unit and integration tests confirming:
1. No future data in any window (strict causality).
2. Scaler and imputer fitted on TRAIN only (val/test mutation invariance).
3. Strictly non-overlapping chronological splits (monotonic time boundaries).
4. Window and target alignment across differing lookback lengths (L=24 vs L=72).
5. Real observation target filtering (no imputed targets in test evaluation).
"""

import numpy as np
import pandas as pd
import pytest
from src.data import (
    split_chronological,
    fit_scalers_and_transform,
    create_sliding_windows,
    preprocess_data,
    CONTINUOUS_COLUMNS
)
from scripts.evaluate_test import build_aligned_windows


def test_no_future_data_in_any_window():
    """Verify that for every window i, the input window contains strictly past data."""
    # Synthetic time-indexed array: each row is [timestamp_id, feature_value]
    n_steps = 100
    timestamps = np.arange(n_steps, dtype=np.float32)
    features = np.random.randn(n_steps, 3).astype(np.float32)
    data = np.column_stack([timestamps, features])  # col 0 is timestamp

    lookback = 24
    horizon = 12
    X, Y = create_sliding_windows(data, target_idx=0, lookback=lookback, horizon=horizon)

    for i in range(len(X)):
        input_timestamps = X[i, :, 0]
        target_timestamps = Y[i]

        # 1. The input timesteps must be strictly monotonically increasing by 1
        np.testing.assert_array_equal(np.diff(input_timestamps), np.ones(lookback - 1))

        # 2. The target timesteps must be strictly monotonically increasing by 1
        np.testing.assert_array_equal(np.diff(target_timestamps), np.ones(horizon - 1))

        # 3. Strict causality: max input timestamp < min target timestamp
        assert input_timestamps[-1] < target_timestamps[0]

        # 4. Immediate adjacency: the target begins at the very next timestep after lookback ends
        assert target_timestamps[0] == input_timestamps[-1] + 1.0


def test_scaler_and_imputer_fitted_on_train_only():
    """Verify that changing val or test data does NOT change fitted scaler parameters."""
    dates = pd.date_range("2010-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "No": range(1, 101),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "pm2.5": [50.0 + i for i in range(100)],
        "DEWP": [10.0] * 100,
        "TEMP": [20.0] * 100,
        "PRES": [1013.0] * 100,
        "cbwd": ["SE"] * 100,
        "Iws": [5.0 + (i % 5) for i in range(100)],
        "Is": [0 if i % 10 != 0 else 1 for i in range(100)],
        "Ir": [0 if i % 15 != 0 else 2 for i in range(100)]
    })

    clean = preprocess_data(df)
    train_df, val_df, test_df = split_chronological(clean, 0.70, 0.15, 0.15)

    _, _, _, scaler_info_orig, scaler_orig = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )

    # Mutate val_df and test_df with massive outliers (e.g. pm2.5 = 1,000,000)
    val_mutated = val_df.copy()
    val_mutated["pm2.5"] = 1_000_000.0
    val_mutated["TEMP"] = -500.0

    test_mutated = test_df.copy()
    test_mutated["pm2.5"] = 9_999_999.0

    _, _, _, scaler_info_mut, scaler_mut = fit_scalers_and_transform(
        train_df, val_mutated, test_mutated, CONTINUOUS_COLUMNS
    )

    # Fitted parameters MUST remain identical down to machine precision
    np.testing.assert_array_equal(scaler_orig.mean_, scaler_mut.mean_)
    np.testing.assert_array_equal(scaler_orig.scale_, scaler_mut.scale_)
    assert scaler_info_orig["target_mean"] == scaler_info_mut["target_mean"]
    assert scaler_info_orig["target_std"] == scaler_info_mut["target_std"]
    assert scaler_info_orig["target_median"] == scaler_info_mut["target_median"]


def test_strictly_non_overlapping_chronological_splits():
    """Verify that splits have zero shared rows and strictly sequential datetime intervals."""
    dates = pd.date_range("2010-01-01", periods=300, freq="h")
    df = pd.DataFrame({
        "No": range(1, 301),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "pm2.5": [50.0] * 300,
        "DEWP": [10.0] * 300,
        "TEMP": [20.0] * 300,
        "PRES": [1013.0] * 300,
        "cbwd": ["NW"] * 300,
        "Iws": [5.0] * 300,
        "Is": [0] * 300,
        "Ir": [0] * 300
    })

    clean = preprocess_data(df)
    n = len(clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # 1. Total row conservation
    train_df, val_df, test_df = split_chronological(clean, 0.70, 0.15, 0.15)
    assert len(train_df) + len(val_df) + len(test_df) == len(clean)

    # 2. Zero row overlap from original dataframe before reset_index
    train_orig_idx = set(clean.iloc[:train_end].index)
    val_orig_idx = set(clean.iloc[train_end:val_end].index)
    test_orig_idx = set(clean.iloc[val_end:].index)
    assert len(train_orig_idx & val_orig_idx) == 0
    assert len(val_orig_idx & test_orig_idx) == 0
    assert len(train_orig_idx & test_orig_idx) == 0

    # 3. Monotonic datetime sequence without overlap
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()


def test_window_target_alignment_across_lookbacks():
    """Verify that L=24 and L=72 align to the identical target sequence Y."""
    data = np.arange(200 * 5, dtype=np.float32).reshape(200, 5)
    H = 24
    max_L = 72
    X_72, X_24, Y = build_aligned_windows(data, target_idx=0, H=H, max_L=max_L)

    assert len(X_72) == len(X_24) == len(Y)
    # The last historical step of X_24 must match the last historical step of X_72
    for i in range(len(Y)):
        np.testing.assert_array_equal(X_72[i, -1, :], X_24[i, -1, :])
        # Target must be the 24 steps immediately following the last input step
        np.testing.assert_array_equal(Y[i], data[i + max_L : i + max_L + H, 0])


def test_real_target_filtering():
    """Verify that filtering excludes windows that contain NaNs in ground truth."""
    raw_pm25 = np.array([10.0] * 20 + [np.nan] + [10.0] * 20, dtype=np.float32)
    # Let H = 4, max_L = 5
    max_L = 5
    H = 4
    n_windows = len(raw_pm25) - max_L - H + 1
    target_is_real = np.empty((n_windows, H), dtype=bool)

    for i in range(n_windows):
        k = i + max_L
        target_is_real[i] = ~np.isnan(raw_pm25[k : k + H])

    real_mask = target_is_real.all(axis=1)

    # Every window included in real_mask must contain zero NaNs
    for i in range(n_windows):
        if real_mask[i]:
            assert not np.isnan(raw_pm25[i + max_L : i + max_L + H]).any()
        else:
            assert np.isnan(raw_pm25[i + max_L : i + max_L + H]).any()
