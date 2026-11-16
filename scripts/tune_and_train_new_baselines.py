"""
Tuning and evaluation for Seasonal Naive (24h, 168h) and HistGradientBoostingRegressor.
Hyperparameters chosen strictly on the VALIDATION set, then evaluated on the held-out test set.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.utils import load_config, set_seed, save_json
from src.data import (
    download_and_load_raw_data,
    preprocess_data,
    split_chronological,
    fit_scalers_and_transform,
    CONTINUOUS_COLUMNS
)
from src.baselines import SeasonalNaiveForecaster, HistGradientBoostingForecaster
from src.metrics import evaluate_forecasts, compute_mae


def unscale(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


def build_aligned_windows_with_168(data_arr: np.ndarray, full_clean_arr: np.ndarray, split_start_idx: int, target_idx: int, H: int = 24, max_L: int = 72):
    """
    Construct aligned sliding windows for L=24, L=72, and L=168 (using continuous history).
    """
    n_samples = len(data_arr) - max_L - H + 1
    X_24 = np.empty((n_samples, 24, data_arr.shape[1]), dtype=np.float32)
    X_72 = np.empty((n_samples, 72, data_arr.shape[1]), dtype=np.float32)
    X_168 = np.empty((n_samples, 168, data_arr.shape[1]), dtype=np.float32)
    Y = np.empty((n_samples, H), dtype=np.float32)

    for i in range(n_samples):
        k = i + max_L
        X_24[i] = data_arr[k - 24 : k]
        X_72[i] = data_arr[k - 72 : k]
        Y[i] = data_arr[k : k + H, target_idx]

        # Global continuous index for 168h lookback (available when global_k >= 168)
        global_k = split_start_idx + k
        if global_k >= 168:
            X_168[i] = full_clean_arr[global_k - 168 : global_k]

    return X_24, X_72, X_168, Y


def main():
    config = load_config("configs/default_config.yaml")
    set_seed(config["seed"])
    results_dir = config["results_dir"]
    horizon = config["data"]["max_horizon"]
    eval_horizons = config["data"]["evaluation_horizons"]

    print("Loading data for new baselines...")
    raw_df = pd.read_csv(config["data"]["raw_path"])
    clean_df = preprocess_data(raw_df)

    n = len(clean_df)
    train_end = int(n * config["data"]["train_ratio"])
    val_end = int(n * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))

    train_df = clean_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = clean_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = clean_df.iloc[val_end:].copy().reset_index(drop=True)

    feature_cols = [c for c in train_df.columns if c != "datetime"]
    target_idx = feature_cols.index("pm2.5")

    train_scaled, val_scaled, test_scaled, scaler_info, _ = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )
    target_mean = scaler_info["target_mean"]
    target_std = scaler_info["target_std"]

    # Full standardized array for continuous historical lookbacks (168h)
    full_scaled_df = pd.concat([train_scaled, val_scaled, test_scaled], ignore_index=True)
    full_arr = full_scaled_df[feature_cols].values.astype(np.float32)

    train_arr = train_scaled[feature_cols].values.astype(np.float32)
    val_arr = val_scaled[feature_cols].values.astype(np.float32)
    test_arr = test_scaled[feature_cols].values.astype(np.float32)

    X_train_24, _, X_train_168, Y_train = build_aligned_windows_with_168(
        train_arr, full_arr, split_start_idx=0, target_idx=target_idx, H=horizon, max_L=72
    )
    X_val_24, _, X_val_168, Y_val_scaled = build_aligned_windows_with_168(
        val_arr, full_arr, split_start_idx=train_end, target_idx=target_idx, H=horizon, max_L=72
    )
    X_test_24, _, X_test_168, Y_test_scaled = build_aligned_windows_with_168(
        test_arr, full_arr, split_start_idx=val_end, target_idx=target_idx, H=horizon, max_L=72
    )

    Y_val_true = unscale(Y_val_scaled, target_mean, target_std)

    # Real observation mask for test set
    raw_pm25 = raw_df.iloc[24:].reset_index(drop=True).iloc[val_end:]["pm2.5"].values
    n_windows = len(X_test_24)
    target_is_real = np.empty((n_windows, horizon), dtype=bool)
    for i in range(n_windows):
        k = i + 72
        target_is_real[i] = ~np.isnan(raw_pm25[k : k + horizon])
    real_mask = target_is_real.all(axis=1)

    Y_test_real = unscale(Y_test_scaled[real_mask], target_mean, target_std)

    # -------------------------------------------------------------
    # 1. Validation Tuning for HistGradientBoostingRegressor
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("HYPERPARAMETER TUNING FOR HistGradientBoostingRegressor (VALIDATION SET ONLY)")
    print("=" * 75)

    gbdt_candidates = [
        {"max_iter": 30, "learning_rate": 0.05, "min_samples_leaf": 30},
        {"max_iter": 50, "learning_rate": 0.05, "min_samples_leaf": 20},
        {"max_iter": 50, "learning_rate": 0.10, "min_samples_leaf": 20},
        {"max_iter": 80, "learning_rate": 0.05, "min_samples_leaf": 20}
    ]

    best_val_mae = float("inf")
    best_gbdt_cfg = None
    best_gbdt_model = None

    for i, cfg in enumerate(gbdt_candidates, 1):
        print(f"Testing Candidate {i}: {cfg}...")
        forecaster = HistGradientBoostingForecaster(
            max_iter=cfg["max_iter"],
            learning_rate=cfg["learning_rate"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=config["seed"]
        )
        forecaster.fit(X_train_24, Y_train)
        preds_scaled = forecaster.predict(X_val_24)
        preds = unscale(preds_scaled, target_mean, target_std)
        val_mae = compute_mae(Y_val_true, preds)
        print(f"  -> Validation MAE: {val_mae:.4f} ug/m3")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_gbdt_cfg = cfg
            best_gbdt_model = forecaster

    print(f"\nWinning HistGBDT Config on Validation: {best_gbdt_cfg} (Val MAE: {best_val_mae:.4f} ug/m3)")

    # -------------------------------------------------------------
    # 2. Evaluate on Test Set (N=5,839 Real-Observation Windows)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATING NEW BASELINES ON HELD-OUT TEST SET (REAL OBSERVATIONS ONLY)")
    print("=" * 75)

    new_results = {}
    preds_dict = {}

    # Seasonal Naive 24h
    sn24 = SeasonalNaiveForecaster(period=24, target_idx=target_idx)
    pred_sn24_scaled = sn24.predict(X_test_24, horizon=horizon)
    pred_sn24 = unscale(pred_sn24_scaled, target_mean, target_std)[real_mask]
    new_results["SeasonalNaive_24h"] = evaluate_forecasts(Y_test_real, pred_sn24, eval_horizons)
    preds_dict["SeasonalNaive_24h"] = pred_sn24

    # Seasonal Naive 168h
    sn168 = SeasonalNaiveForecaster(period=168, target_idx=target_idx)
    pred_sn168_scaled = sn168.predict(X_test_168, horizon=horizon)
    pred_sn168 = unscale(pred_sn168_scaled, target_mean, target_std)[real_mask]
    new_results["SeasonalNaive_168h"] = evaluate_forecasts(Y_test_real, pred_sn168, eval_horizons)
    preds_dict["SeasonalNaive_168h"] = pred_sn168

    # Best HistGradientBoostingRegressor
    pred_gbdt_scaled = best_gbdt_model.predict(X_test_24)
    pred_gbdt = unscale(pred_gbdt_scaled, target_mean, target_std)[real_mask]
    new_results["HistGradientBoosting"] = evaluate_forecasts(Y_test_real, pred_gbdt, eval_horizons)
    new_results["HistGradientBoosting"]["chosen_hyperparameters"] = best_gbdt_cfg
    new_results["HistGradientBoosting"]["validation_mae"] = best_val_mae
    preds_dict["HistGradientBoosting"] = pred_gbdt

    # Save test predictions for bootstrap analysis
    for name, arr in preds_dict.items():
        np.save(os.path.join(results_dir, f"test_preds_{name}.npy"), arr)

    # Print summary table
    print(f"{'Model':<25} | {'Overall MAE':<12} | {'Overall RMSE':<12} | {'Overall R2':<10} | {'Overall sMAPE':<12}")
    print("-" * 75)
    for name, res in new_results.items():
        ov = res["overall"]
        print(f"{name:<25} | {ov['MAE']:<12.4f} | {ov['RMSE']:<12.4f} | {ov['R2']:<10.4f} | {ov['sMAPE']:<11.2f}%")
    print("=" * 75)

    print("\nPER-HORIZON MAE (ug/m3) ON TEST:")
    print(f"{'Model':<25} | " + " | ".join([f"{f't+{h}':<10}" for h in eval_horizons]))
    print("-" * 70)
    for name, res in new_results.items():
        horiz_str = " | ".join([f"{res['per_horizon'][f't+{h}']['MAE']:<10.4f}" for h in eval_horizons])
        print(f"{name:<25} | {horiz_str}")

    save_path = os.path.join(results_dir, "new_baselines_results.json")
    save_json(new_results, save_path)
    print(f"\nNew baselines results saved to {save_path}")


if __name__ == "__main__":
    main()
