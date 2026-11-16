"""
Tests for classical baseline models and evaluation metrics.
"""

import numpy as np
import pytest
from src.metrics import (
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_smape,
    evaluate_forecasts
)
from src.baselines import (
    PersistenceForecaster,
    MovingAverageForecaster,
    RidgeForecaster
)


def test_metric_computations_exact_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    expected_mae = (2.0 + 2.0 + 3.0) / 3.0
    expected_rmse = np.sqrt((4.0 + 4.0 + 9.0) / 3.0)

    assert abs(compute_mae(y_true, y_pred) - expected_mae) < 1e-6
    assert abs(compute_rmse(y_true, y_pred) - expected_rmse) < 1e-6

    # Perfect predictions give R2 = 1.0
    assert abs(compute_r2(y_true, y_true) - 1.0) < 1e-6
    # Constant mean predictor gives R2 = 0.0
    mean_pred = np.full_like(y_true, np.mean(y_true))
    assert abs(compute_r2(y_true, mean_pred) - 0.0) < 1e-6


def test_evaluate_forecasts_shapes_and_keys():
    y_true = np.random.uniform(10, 100, (50, 24))
    y_pred = y_true + np.random.normal(0, 5, (50, 24))

    results = evaluate_forecasts(y_true, y_pred, horizons=[1, 6, 12, 24])
    assert "overall" in results
    assert "per_horizon" in results
    for h in [1, 6, 12, 24]:
        assert f"t+{h}" in results["per_horizon"]
        assert "MAE" in results["per_horizon"][f"t+{h}"]
        assert "RMSE" in results["per_horizon"][f"t+{h}"]


def test_persistence_forecaster():
    # 5 samples, lookback=10, 3 features. Target is at index 0.
    X = np.zeros((5, 10, 3))
    # Give the last step (t) of feature 0 distinct values
    for i in range(5):
        X[i, -1, 0] = (i + 1) * 10.0

    forecaster = PersistenceForecaster(target_idx=0)
    forecaster.fit(X)
    pred = forecaster.predict(X, horizon=4)

    assert pred.shape == (5, 4)
    for i in range(5):
        expected_val = (i + 1) * 10.0
        np.testing.assert_allclose(pred[i], [expected_val] * 4)


def test_moving_average_forecaster():
    # Lookback=4 timesteps with values [10, 20, 30, 40] -> mean = 25.0
    X = np.zeros((1, 4, 2))
    X[0, :, 0] = [10.0, 20.0, 30.0, 40.0]

    forecaster = MovingAverageForecaster(target_idx=0)
    forecaster.fit(X)
    pred = forecaster.predict(X, horizon=3)

    assert pred.shape == (1, 3)
    np.testing.assert_allclose(pred[0], [25.0, 25.0, 25.0])


def test_ridge_forecaster():
    np.random.seed(42)
    X = np.random.randn(30, 10, 4)
    y = np.random.randn(30, 5)

    forecaster = RidgeForecaster(alpha=1.0)
    forecaster.fit(X, y)
    pred = forecaster.predict(X)

    assert pred.shape == (30, 5)
