"""
Classical baseline forecasters for multi-horizon air quality prediction.
Includes Persistence (Naive), Moving Average, and Multi-Output Ridge Regression.
"""

from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge


class PersistenceForecaster:
    """
    Persistence / Naive Baseline.
    Predicts that future PM2.5 levels will remain exactly equal to the most
    recently observed PM2.5 value at time t: y_{t+h} = y_t for all h.
    """
    def __init__(self, target_idx: int = 0):
        self.target_idx = target_idx

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PersistenceForecaster":
        # Persistence has no trainable parameters; fit is a no-op
        return self

    def predict(self, X: np.ndarray, horizon: int = 24) -> np.ndarray:
        """
        X shape: (N, L, F)
        Returns: (N, horizon)
        """
        # Extract the target value at the most recent historical timestep (index -1)
        last_observed = X[:, -1, self.target_idx]  # Shape: (N,)
        # Broadcast across all future horizon steps
        return np.repeat(last_observed[:, np.newaxis], horizon, axis=1)


class MovingAverageForecaster:
    """
    Historical Moving Average Baseline.
    Predicts future PM2.5 as the arithmetic mean of observed PM2.5 concentrations
    over the historical lookback window: y_{t+h} = (1/L) * sum(y_{t-k}) for all h.
    """
    def __init__(self, target_idx: int = 0):
        self.target_idx = target_idx

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MovingAverageForecaster":
        # Moving average has no trainable parameters; fit is a no-op
        return self

    def predict(self, X: np.ndarray, horizon: int = 24) -> np.ndarray:
        """
        X shape: (N, L, F)
        Returns: (N, horizon)
        """
        # Average the target variable across all lookback timesteps (axis=1)
        target_series = X[:, :, self.target_idx]  # Shape: (N, L)
        window_mean = np.mean(target_series, axis=1)  # Shape: (N,)
        # Broadcast mean across all future horizon steps
        return np.repeat(window_mean[:, np.newaxis], horizon, axis=1)


class RidgeForecaster:
    """
    Multi-Output Ridge Regression Baseline.
    Flattens the 3D historical window (N, L, F) into a 2D matrix (N, L*F)
    and solves the regularized least-squares problem:
    min_W ||Y - X_flat W||_2^2 + alpha ||W||_2^2
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeForecaster":
        """
        X shape: (N, L, F)
        y shape: (N, H)
        """
        n_samples = X.shape[0]
        # Flatten temporal and feature dimensions into a single feature vector per sample
        X_flat = X.reshape(n_samples, -1)
        self.model.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (N, L, F)
        Returns: (N, H)
        """
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        return self.model.predict(X_flat)
