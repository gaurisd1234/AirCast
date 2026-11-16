"""
Unit tests for new baselines (Seasonal Naive 24h/168h, HistGBDT), multi-seed metrics, and block bootstrap.
"""

import numpy as np
import pytest
from src.baselines import SeasonalNaiveForecaster, HistGradientBoostingForecaster


def test_seasonal_naive_24h_predictions():
    # 5 samples, lookback=24 timesteps, 2 features. Target at index 0.
    X = np.zeros((5, 24, 2), dtype=np.float32)
    # Assign known values: at time t-23 (index 0) value is 10.0; at time t (index 23) value is 99.0
    for i in range(5):
        X[i, :, 0] = np.arange(1, 25) + i * 10.0

    sn24 = SeasonalNaiveForecaster(period=24, target_idx=0)
    sn24.fit(X)
    preds = sn24.predict(X, horizon=24)

    assert preds.shape == (5, 24)
    # Forecast h=1 corresponds to y_{t-23} (first element of lookback)
    np.testing.assert_allclose(preds[:, 0], X[:, 0, 0])
    # Forecast h=24 corresponds to y_t (last element of lookback)
    np.testing.assert_allclose(preds[:, 23], X[:, 23, 0])


def test_seasonal_naive_168h_predictions():
    # Lookback=168 timesteps (7 days * 24 hours)
    X = np.zeros((3, 168, 2), dtype=np.float32)
    for i in range(3):
        X[i, :, 0] = np.arange(168) + i * 100.0

    sn168 = SeasonalNaiveForecaster(period=168, target_idx=0)
    sn168.fit(X)
    preds = sn168.predict(X, horizon=24)

    assert preds.shape == (3, 24)
    # Forecast h=1 corresponds to y_{t-167} (index 0 of the 168-step lookback)
    np.testing.assert_allclose(preds[:, 0], X[:, 0, 0])
    # Forecast h=24 corresponds to y_{t-144} (index 23 of the 168-step lookback)
    np.testing.assert_allclose(preds[:, 23], X[:, 23, 0])


def test_seasonal_naive_insufficient_lookback_raises_error():
    X = np.zeros((2, 50, 2), dtype=np.float32)
    sn168 = SeasonalNaiveForecaster(period=168, target_idx=0)
    with pytest.raises(ValueError, match="is shorter than seasonality period"):
        sn168.predict(X, horizon=24)


def test_hist_gradient_boosting_forecaster_fit_predict():
    np.random.seed(42)
    X = np.random.randn(50, 10, 4).astype(np.float32)
    y = np.random.randn(50, 6).astype(np.float32)

    hgbt = HistGradientBoostingForecaster(max_iter=5, random_state=42)
    hgbt.fit(X, y)
    preds = hgbt.predict(X)

    assert preds.shape == (50, 6)
    assert np.isfinite(preds).all()


def test_block_bootstrap_resampling_logic():
    from scripts.run_block_bootstrap import compute_block_bootstrap_ci

    np.random.seed(42)
    # Model A is strictly better than Model B by ~5 units on average
    errors_A = np.random.normal(40, 5, 240)
    errors_B = np.random.normal(45, 5, 240)
    # Diff = A - B -> should be strictly negative (~ -5)
    diff = errors_A - errors_B

    ci_low, ci_high, mean_diff = compute_block_bootstrap_ci(diff, block_size=24, n_resamples=200, seed=42)
    assert ci_low < ci_high
    assert ci_low < -3.0
    assert ci_high < 0.0  # Statistically significant difference
