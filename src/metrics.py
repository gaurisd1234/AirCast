"""
Evaluation metrics for multi-horizon air quality forecasting.
Includes MAE, RMSE, R-squared (R2), and symmetric MAPE (sMAPE).
"""

from typing import Dict, List, Union
import numpy as np


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in original physical units (ug/m3)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in original physical units (ug/m3)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R^2).
    Measures proportion of variance explained by predictions relative to mean.
    Values can be negative if predictions are worse than historical mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) bounded between 0% and 200%.
    Why sMAPE over standard MAPE: Avoids division by zero or explosive percentages
    when true air pollution levels approach 0 ug/m3.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / denominator))


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] = [1, 6, 12, 24]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute comprehensive metrics overall and broken down for each focal horizon.
    
    y_true shape: (N, H)
    y_pred shape: (N, H)
    horizons: list of 1-indexed future steps (e.g., [1, 6, 12, 24])
    """
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    results = {
        "overall": {
            "MAE": compute_mae(y_true, y_pred),
            "RMSE": compute_rmse(y_true, y_pred),
            "R2": compute_r2(y_true, y_pred),
            "sMAPE": compute_smape(y_true, y_pred),
        },
        "per_horizon": {}
    }

    max_h = y_true.shape[1]
    for h in horizons:
        if h <= max_h:
            # 1-indexed horizon h maps to index h - 1 in the target array
            idx = h - 1
            yt_h = y_true[:, idx]
            yp_h = y_pred[:, idx]
            results["per_horizon"][f"t+{h}"] = {
                "MAE": compute_mae(yt_h, yp_h),
                "RMSE": compute_rmse(yt_h, yp_h),
                "R2": compute_r2(yt_h, yp_h),
                "sMAPE": compute_smape(yt_h, yp_h),
            }

    return results
