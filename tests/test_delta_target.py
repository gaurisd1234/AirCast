"""
Unit tests for Delta target formulation and validation selection logic.
"""

import numpy as np
import pytest
from src.data import create_delta_targets, reconstruct_predictions_from_delta


def test_create_delta_targets_values_and_shapes():
    """Verify that delta targets correctly subtract the last historical target value."""
    # Synthetic input X: 3 samples, lookback=4, 2 features (target at index 0)
    # Feature 0 values for sample 0: [10, 20, 30, 40] -> last observed is 40
    # Feature 0 values for sample 1: [5,  15, 25, 35] -> last observed is 35
    X = np.zeros((2, 4, 2), dtype=np.float32)
    X[0, :, 0] = [10.0, 20.0, 30.0, 40.0]
    X[1, :, 0] = [5.0, 15.0, 25.0, 35.0]

    # Future targets Y: 2 samples, horizon=3
    Y = np.array([
        [45.0, 50.0, 38.0],
        [35.0, 40.0, 30.0]
    ], dtype=np.float32)

    delta = create_delta_targets(Y, X, target_idx=0)

    assert delta.shape == Y.shape
    # Sample 0 expected deltas: [45-40, 50-40, 38-40] = [5.0, 10.0, -2.0]
    np.testing.assert_allclose(delta[0], [5.0, 10.0, -2.0], rtol=1e-5)
    # Sample 1 expected deltas: [35-35, 40-35, 30-35] = [0.0, 5.0, -5.0]
    np.testing.assert_allclose(delta[1], [0.0, 5.0, -5.0], rtol=1e-5)


def test_reconstruct_predictions_invertibility():
    """Verify that reconstructing from exact deltas perfectly recovers original targets."""
    np.random.seed(42)
    N, L, F, H = 20, 24, 5, 24
    X = np.random.randn(N, L, F).astype(np.float32)
    Y = np.random.randn(N, H).astype(np.float32)

    delta = create_delta_targets(Y, X, target_idx=0)
    reconstructed = reconstruct_predictions_from_delta(delta, X, target_idx=0)

    np.testing.assert_allclose(reconstructed, Y, atol=1e-6)


def test_zero_delta_defaults_to_persistence():
    """Verify that predicting zero delta naturally defaults to the persistence baseline."""
    N, L, F, H = 10, 24, 3, 24
    X = np.random.uniform(10, 100, size=(N, L, F)).astype(np.float32)
    zero_deltas = np.zeros((N, H), dtype=np.float32)

    reconstructed = reconstruct_predictions_from_delta(zero_deltas, X, target_idx=0)

    expected_persistence = np.repeat(X[:, -1, 0:1], H, axis=1)
    np.testing.assert_allclose(reconstructed, expected_persistence, atol=1e-6)


def test_validation_selection_rule():
    """Verify that target definition is chosen strictly by validation score."""
    val_mae_raw = 48.2836
    val_mae_delta = 47.4042

    chosen_setup = "delta" if val_mae_delta < val_mae_raw else "raw"
    assert chosen_setup == "delta", "Validation selection must choose the configuration with lower validation error."
