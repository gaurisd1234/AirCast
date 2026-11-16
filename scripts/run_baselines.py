"""
Run classical baselines (Persistence, Moving Average, Ridge Regression) on Train and evaluate on Validation.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.utils import load_config, set_seed, save_json
from src.data import (
    preprocess_data,
    split_chronological,
    fit_scalers_and_transform,
    create_sliding_windows,
    CONTINUOUS_COLUMNS
)
from src.baselines import PersistenceForecaster, MovingAverageForecaster, RidgeForecaster
from src.metrics import evaluate_forecasts


def unscale_targets(y_scaled: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Transform standardized predictions/targets back to physical ug/m3 units."""
    return y_scaled * target_std + target_mean


def main():
    config = load_config("configs/default_config.yaml")
    set_seed(config["seed"])

    print("Loading data for baseline evaluation...")
    df_raw = pd.read_csv(config["data"]["raw_path"])
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

    lookback = config["data"]["lookback_hours"]
    horizon = config["data"]["max_horizon"]
    eval_horizons = config["data"]["evaluation_horizons"]

    X_train, Y_train_scaled = create_sliding_windows(
        train_scaled[feature_cols].values.astype(np.float32), target_idx, lookback, horizon
    )
    X_val, Y_val_scaled = create_sliding_windows(
        val_scaled[feature_cols].values.astype(np.float32), target_idx, lookback, horizon
    )

    # Convert true validation targets to original physical scale (ug/m3)
    y_val_true = unscale_targets(Y_val_scaled, scaler_info["target_mean"], scaler_info["target_std"])

    results = {}

    # 1. Persistence Baseline
    print("\n--- Evaluating Persistence Baseline ---")
    persistence = PersistenceForecaster(target_idx=target_idx)
    persistence.fit(X_train)
    # Persistence predicts scaled values directly from scaled input
    y_val_pred_scaled = persistence.predict(X_val, horizon=horizon)
    y_val_pred_pers = unscale_targets(y_val_pred_scaled, scaler_info["target_mean"], scaler_info["target_std"])
    results["Persistence"] = evaluate_forecasts(y_val_true, y_val_pred_pers, eval_horizons)

    # 2. Moving Average Baseline
    print("\n--- Evaluating Moving Average Baseline ---")
    ma = MovingAverageForecaster(target_idx=target_idx)
    ma.fit(X_train)
    y_val_pred_scaled = ma.predict(X_val, horizon=horizon)
    y_val_pred_ma = unscale_targets(y_val_pred_scaled, scaler_info["target_mean"], scaler_info["target_std"])
    results["MovingAverage"] = evaluate_forecasts(y_val_true, y_val_pred_ma, eval_horizons)

    # 3. Ridge Regression with alpha validation tuning
    print("\n--- Tuning & Evaluating Multi-Output Ridge Regression ---")
    alpha_candidates = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha = None
    best_ridge_val_mae = float("inf")
    best_ridge_pred = None

    for alpha in alpha_candidates:
        ridge = RidgeForecaster(alpha=alpha)
        ridge.fit(X_train, Y_train_scaled)
        pred_scaled = ridge.predict(X_val)
        pred_unscaled = unscale_targets(pred_scaled, scaler_info["target_mean"], scaler_info["target_std"])
        mae = float(np.mean(np.abs(y_val_true - pred_unscaled)))
        print(f"  Ridge alpha={alpha:7.1f} -> Validation MAE: {mae:.4f} ug/m3")
        if mae < best_ridge_val_mae:
            best_ridge_val_mae = mae
            best_alpha = alpha
            best_ridge_pred = pred_unscaled

    print(f"Selected best Ridge alpha: {best_alpha} with Val MAE: {best_ridge_val_mae:.4f} ug/m3")
    results["Ridge"] = evaluate_forecasts(y_val_true, best_ridge_pred, eval_horizons)
    results["Ridge"]["best_alpha"] = best_alpha

    # Display clean comparison table
    print("\n" + "=" * 70)
    print(f"{'Model':<16} | {'Overall MAE':<12} | {'Overall RMSE':<12} | {'Overall R2':<10} | {'Overall sMAPE':<12}")
    print("-" * 70)
    for model_name, res in results.items():
        ov = res["overall"]
        print(f"{model_name:<16} | {ov['MAE']:<12.4f} | {ov['RMSE']:<12.4f} | {ov['R2']:<10.4f} | {ov['sMAPE']:<11.2f}%")
    print("=" * 70)

    print("\nPer-Horizon MAE (ug/m3) on Validation:")
    print(f"{'Model':<16} | " + " | ".join([f"{f't+{h}':<10}" for h in eval_horizons]))
    print("-" * 65)
    for model_name, res in results.items():
        horiz_str = " | ".join([f"{res['per_horizon'][f't+{h}']['MAE']:<10.4f}" for h in eval_horizons])
        print(f"{model_name:<16} | {horiz_str}")

    save_path = os.path.join(config["results_dir"], "baselines_val_results.json")
    save_json(results, save_path)
    print(f"\nBaseline results saved to {save_path}")


if __name__ == "__main__":
    main()
