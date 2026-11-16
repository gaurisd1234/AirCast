"""
Final Unbiased Test Set Evaluation for AirCast.
Evaluates classical baselines, Vanilla RNN, and LSTM on held-out test data.
All models are evaluated on identical target timestamps for fair comparison.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from src.utils import load_config, set_seed, save_json, get_device
from src.data import (
    download_and_load_raw_data,
    preprocess_data,
    split_chronological,
    fit_scalers_and_transform,
    CONTINUOUS_COLUMNS
)
from src.baselines import PersistenceForecaster, MovingAverageForecaster, RidgeForecaster
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.models.lstm import LSTMAirQuality
from src.trainer import predict_model
from src.metrics import evaluate_forecasts
from torch.utils.data import DataLoader, TensorDataset


def unscale(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


def build_aligned_windows(data_arr: np.ndarray, target_idx: int, H: int = 24, max_L: int = 72):
    """
    Construct aligned sliding windows for different lookback lengths (24h and 72h)
    so that both lookback windows predict the EXACT same target sequence.
    """
    n_samples = len(data_arr) - max_L - H + 1
    X_72 = np.empty((n_samples, 72, data_arr.shape[1]), dtype=np.float32)
    X_24 = np.empty((n_samples, 24, data_arr.shape[1]), dtype=np.float32)
    Y = np.empty((n_samples, H), dtype=np.float32)

    for i in range(n_samples):
        # Forecast origin point is k = i + max_L
        k = i + max_L
        X_72[i] = data_arr[k - 72 : k]
        X_24[i] = data_arr[k - 24 : k]
        Y[i] = data_arr[k : k + H, target_idx]

    return X_72, X_24, Y


def main():
    config = load_config("configs/default_config.yaml")
    set_seed(config["seed"])
    device = get_device()
    print(f"Running final test evaluation on device: {device}")

    # Load and preprocess data
    raw_df = download_and_load_raw_data(config["data"]["raw_url"], config["data"]["raw_path"])
    clean_df = preprocess_data(raw_df)

    train_df, val_df, test_df = split_chronological(
        clean_df,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"]
    )

    feature_cols = [c for c in train_df.columns if c != "datetime"]
    target_idx = feature_cols.index("pm2.5")
    num_features = len(feature_cols)
    horizon = config["data"]["max_horizon"]
    eval_horizons = config["data"]["evaluation_horizons"]

    train_scaled, val_scaled, test_scaled, scaler_info, _ = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )

    target_mean = scaler_info["target_mean"]
    target_std = scaler_info["target_std"]

    # Construct aligned windows for Train and Test
    train_arr = train_scaled[feature_cols].values.astype(np.float32)
    test_arr = test_scaled[feature_cols].values.astype(np.float32)

    X_train_72, X_train_24, Y_train = build_aligned_windows(train_arr, target_idx, horizon, max_L=72)
    X_test_72, X_test_24, Y_test_scaled = build_aligned_windows(test_arr, target_idx, horizon, max_L=72)

    Y_test_true = unscale(Y_test_scaled, target_mean, target_std)
    print(f"Test evaluation samples: {len(Y_test_true):,} aligned sequences of 24h targets.")

    results = {}
    predictions_dict = {}

    # 1. Persistence Baseline (last observed value at t=0)
    print("\n[1/7] Evaluating Persistence Baseline on Test...")
    persistence = PersistenceForecaster(target_idx=target_idx)
    pred_pers_scaled = persistence.predict(X_test_72, horizon=horizon)
    pred_pers = unscale(pred_pers_scaled, target_mean, target_std)
    results["Persistence"] = evaluate_forecasts(Y_test_true, pred_pers, eval_horizons)
    predictions_dict["Persistence"] = pred_pers

    # 2. Moving Average Baseline (over 72h)
    print("[2/7] Evaluating Moving Average (72h) Baseline on Test...")
    ma_72 = MovingAverageForecaster(target_idx=target_idx)
    pred_ma_scaled = ma_72.predict(X_test_72, horizon=horizon)
    pred_ma = unscale(pred_ma_scaled, target_mean, target_std)
    results["MovingAverage_72h"] = evaluate_forecasts(Y_test_true, pred_ma, eval_horizons)
    predictions_dict["MovingAverage_72h"] = pred_ma

    # 3. Moving Average Baseline (over 24h)
    print("[3/7] Evaluating Moving Average (24h) Baseline on Test...")
    ma_24 = MovingAverageForecaster(target_idx=target_idx)
    pred_ma24_scaled = ma_24.predict(X_test_24, horizon=horizon)
    pred_ma24 = unscale(pred_ma24_scaled, target_mean, target_std)
    results["MovingAverage_24h"] = evaluate_forecasts(Y_test_true, pred_ma24, eval_horizons)
    predictions_dict["MovingAverage_24h"] = pred_ma24

    # 4. Multi-Output Ridge Regression (trained on train with best validation alpha=10.0)
    print("[4/7] Evaluating Ridge Regression (alpha=10.0) on Test...")
    ridge = RidgeForecaster(alpha=10.0)
    ridge.fit(X_train_72, Y_train)
    pred_ridge_scaled = ridge.predict(X_test_72)
    pred_ridge = unscale(pred_ridge_scaled, target_mean, target_std)
    results["Ridge_alpha10"] = evaluate_forecasts(Y_test_true, pred_ridge, eval_horizons)
    predictions_dict["Ridge_alpha10"] = pred_ridge

    # Prepare PyTorch Test DataLoaders
    test_loader_72 = DataLoader(
        TensorDataset(torch.tensor(X_test_72), torch.tensor(Y_test_scaled)),
        batch_size=128,
        shuffle=False
    )
    test_loader_24 = DataLoader(
        TensorDataset(torch.tensor(X_test_24), torch.tensor(Y_test_scaled)),
        batch_size=128,
        shuffle=False
    )

    # 5. Vanilla RNN (L=72 Baseline)
    print("[5/7] Evaluating Vanilla RNN (L=72) on Test...")
    rnn_72 = VanillaRNNAirQuality(
        num_features=num_features,
        hidden_dim=config["models"]["vanilla_rnn"]["hidden_dim"],
        num_layers=config["models"]["vanilla_rnn"]["num_layers"],
        horizon=horizon,
        dropout=config["models"]["vanilla_rnn"]["dropout"]
    ).to(device)
    rnn_72_ckpt = os.path.join(config["results_dir"], "checkpoints", "vanilla_rnn_best.pt")
    rnn_72.load_state_dict(torch.load(rnn_72_ckpt, weights_only=True, map_location=device))
    pred_rnn72_scaled = predict_model(rnn_72, test_loader_72, device)
    pred_rnn72 = unscale(pred_rnn72_scaled, target_mean, target_std)
    results["Vanilla_RNN_L72"] = evaluate_forecasts(Y_test_true, pred_rnn72, eval_horizons)
    predictions_dict["Vanilla_RNN_L72"] = pred_rnn72

    # 6. Vanilla RNN Tuned (L=24)
    print("[6/7] Evaluating Best Tuned Vanilla RNN (L=24) on Test...")
    rnn_24 = VanillaRNNAirQuality(
        num_features=num_features,
        hidden_dim=64,
        num_layers=2,
        horizon=horizon,
        dropout=0.2
    ).to(device)
    rnn_24_ckpt = os.path.join(config["results_dir"], "checkpoints", "tune_Vanilla_RNN_ConfigC_Lookback24h.pt")
    rnn_24.load_state_dict(torch.load(rnn_24_ckpt, weights_only=True, map_location=device))
    pred_rnn24_scaled = predict_model(rnn_24, test_loader_24, device)
    pred_rnn24 = unscale(pred_rnn24_scaled, target_mean, target_std)
    results["Vanilla_RNN_L24_Tuned"] = evaluate_forecasts(Y_test_true, pred_rnn24, eval_horizons)
    predictions_dict["Vanilla_RNN_L24_Tuned"] = pred_rnn24

    # 7. LSTM Tuned (L=24)
    print("[7/7] Evaluating Best Tuned LSTM (L=24) on Test...")
    lstm_24 = LSTMAirQuality(
        num_features=num_features,
        hidden_dim=64,
        num_layers=2,
        horizon=horizon,
        dropout=0.2
    ).to(device)
    lstm_24_ckpt = os.path.join(config["results_dir"], "checkpoints", "tune_LSTM_ConfigC_Lookback24h.pt")
    lstm_24.load_state_dict(torch.load(lstm_24_ckpt, weights_only=True, map_location=device))
    pred_lstm24_scaled = predict_model(lstm_24, test_loader_24, device)
    pred_lstm24 = unscale(pred_lstm24_scaled, target_mean, target_std)
    results["LSTM_L24_Tuned"] = evaluate_forecasts(Y_test_true, pred_lstm24, eval_horizons)
    predictions_dict["LSTM_L24_Tuned"] = pred_lstm24

    # Save true targets and predictions array for visualization script
    np.save(os.path.join(config["results_dir"], "test_y_true.npy"), Y_test_true)
    for model_name, preds in predictions_dict.items():
        np.save(os.path.join(config["results_dir"], f"test_preds_{model_name}.npy"), preds)

    # Build and print full comparison tables
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION SUMMARY (HELD-OUT TEST: APRIL - DECEMBER 2014)")
    print("=" * 80)
    print(f"{'Model':<25} | {'Overall MAE':<12} | {'Overall RMSE':<12} | {'Overall R2':<10} | {'Overall sMAPE':<12}")
    print("-" * 80)
    for name, res in results.items():
        ov = res["overall"]
        print(f"{name:<25} | {ov['MAE']:<12.4f} | {ov['RMSE']:<12.4f} | {ov['R2']:<10.4f} | {ov['sMAPE']:<11.2f}%")
    print("=" * 80)

    print("\nPER-HORIZON MAE (ug/m3) BREAKDOWN ON TEST:")
    print(f"{'Model':<25} | " + " | ".join([f"{f't+{h}':<10}" for h in eval_horizons]))
    print("-" * 75)
    for name, res in results.items():
        horiz_str = " | ".join([f"{res['per_horizon'][f't+{h}']['MAE']:<10.4f}" for h in eval_horizons])
        print(f"{name:<25} | {horiz_str}")
    print("=" * 75)

    print("\nPER-HORIZON RMSE (ug/m3) BREAKDOWN ON TEST:")
    print(f"{'Model':<25} | " + " | ".join([f"{f't+{h}':<10}" for h in eval_horizons]))
    print("-" * 75)
    for name, res in results.items():
        horiz_str = " | ".join([f"{res['per_horizon'][f't+{h}']['RMSE']:<10.4f}" for h in eval_horizons])
        print(f"{name:<25} | {horiz_str}")
    print("=" * 75)

    # Save to JSON
    out_json = os.path.join(config["results_dir"], "final_test_metrics.json")
    save_json(results, out_json)
    print(f"\nFinal test metrics saved to {out_json}")

    # Build CSV table
    csv_rows = []
    for name, res in results.items():
        row = {
            "Model": name,
            "Overall_MAE": res["overall"]["MAE"],
            "Overall_RMSE": res["overall"]["RMSE"],
            "Overall_R2": res["overall"]["R2"],
            "Overall_sMAPE": res["overall"]["sMAPE"],
        }
        for h in eval_horizons:
            row[f"MAE_t+{h}"] = res["per_horizon"][f"t+{h}"]["MAE"]
            row[f"RMSE_t+{h}"] = res["per_horizon"][f"t+{h}"]["RMSE"]
            row[f"R2_t+{h}"] = res["per_horizon"][f"t+{h}"]["R2"]
            row[f"sMAPE_t+{h}"] = res["per_horizon"][f"t+{h}"]["sMAPE"]
        csv_rows.append(row)

    df_results = pd.DataFrame(csv_rows)
    out_csv = os.path.join(config["results_dir"], "final_test_metrics.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Final test CSV saved to {out_csv}")


if __name__ == "__main__":
    main()
