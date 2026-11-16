"""
Classical baseline forecasters for multi-horizon air quality prediction.
Includes Persistence (Naive), Moving Average, and Multi-Output Ridge Regression.
"""

from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


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


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Forecaster.
    Predicts future PM2.5 as the observed value from exactly one seasonal cycle prior:
    y_{t+h} = y_{t+h-m}, where m is the seasonal period (e.g. m=24 for diurnal, m=168 for weekly).
    
    For horizon h in 1..H:
    - If m=24:  forecast h=1 uses y_{t-23}, forecast h=24 uses y_t.
    - If m=168: forecast h=1 uses y_{t-167}, forecast h=24 uses y_{t-144}.
    """
    def __init__(self, period: int = 24, target_idx: int = 0):
        self.period = period
        self.target_idx = target_idx

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SeasonalNaiveForecaster":
        # Seasonal Naive has no trainable parameters; fit is a no-op
        return self

    def predict(self, X: np.ndarray, horizon: int = 24) -> np.ndarray:
        """
        X shape: (N, L, F) where L >= self.period.
        Returns: (N, horizon)
        """
        L = X.shape[1]
        if L < self.period:
            raise ValueError(
                f"Lookback length L={L} is shorter than seasonality period m={self.period}. "
                f"Cannot extract seasonal lag without sufficient history."
            )
        # Extract the target column slice corresponding to [t - period + 1 : t - period + horizon]
        # In 0-indexed lookback where index L-1 is time t:
        # time (t - period + 1) is at index (L - 1) - period + 1 = L - period
        start_idx = L - self.period
        end_idx = start_idx + horizon
        return X[:, start_idx:end_idx, self.target_idx]


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


class HistGradientBoostingForecaster:
    """
    Multi-Output Histogram-Based Gradient Boosted Decision Tree Forecaster.
    Wraps HistGradientBoostingRegressor inside MultiOutputRegressor to fit
    24 direct tree ensembles mapping flattened lag, meteorological, and cyclical
    features to all future horizons H=24.
    """
    def __init__(
        self,
        max_iter: int = 50,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        random_state: int = 42
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        base_estimator = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.model = MultiOutputRegressor(base_estimator, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingForecaster":
        """
        X shape: (N, L, F)
        y shape: (N, H)
        """
        n_samples = X.shape[0]
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
