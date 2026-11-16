"""
Tests for final test evaluation window alignment, unscaling, and metric table generation.
"""

import numpy as np
import pytest
from scripts.evaluate_test import build_aligned_windows, unscale


def test_build_aligned_windows():
    # 150 timesteps, 4 features. Target is index 0.
    data = np.arange(150 * 4, dtype=np.float32).reshape(150, 4)
    H = 24
    max_L = 72
    X_72, X_24, Y = build_aligned_windows(data, target_idx=0, H=H, max_L=max_L)

    expected_samples = 150 - max_L - H + 1  # 150 - 72 - 24 + 1 = 55
    assert X_72.shape == (expected_samples, 72, 4)
    assert X_24.shape == (expected_samples, 24, 4)
    assert Y.shape == (expected_samples, H)

    # For the first sample:
    # Target starts at index max_L = 72
    np.testing.assert_array_equal(Y[0], data[72 : 72 + H, 0])
    # X_72 covers 0:72
    np.testing.assert_array_equal(X_72[0, :, 0], data[0:72, 0])
    # X_24 covers 48:72
    np.testing.assert_array_equal(X_24[0, :, 0], data[48:72, 0])
    # The last step of X_24 matches the last step of X_72
    np.testing.assert_array_equal(X_72[0, -1, :], X_24[0, -1, :])


def test_unscale_function():
    scaled = np.array([0.0, 1.0, -1.0])
    mean = 100.0
    std = 15.0
    unscaled = unscale(scaled, mean, std)
    np.testing.assert_allclose(unscaled, [100.0, 115.0, 85.0])
