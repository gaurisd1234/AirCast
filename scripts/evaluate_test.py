"""
Final Unbiased Test Set Evaluation for AirCast.
Evaluates classical baselines, Vanilla RNN, and LSTM on held-out test data.
Test metrics are computed exclusively on windows whose target values are 100% real observations (not imputed).
Outputs per-horizon metrics for all 24 forecast horizons to results/per_horizon_all.csv.
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
from src.metrics import evaluate_forecasts, compute_mae, compute_rmse, compute_r2, compute_smape
from torch.utils.data import DataLoader, TensorDataset


def unscale(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Transform standardized values back to physical ug/m3 units."""
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
        k = i + max_L
        X_72[i] = data_arr[k - 72 : k]
        X_24[i] = data_arr[k - 24 : k]
        Y[i] = data_arr[k : k + H, target_idx]

    return X_72, X_24, Y


def compute_all_24_horizons_mae(y_true: np.ndarray, predictions_dict: dict) -> pd.DataFrame:
    """
    Compute per-horizon MAE for each of all 24 horizons (t+1 through t+24).
    y_true shape: (N, 24)
    predictions shape: (N, 24)
    """
    rows = []
    H = y_true.shape[1]
    for h in range(1, H + 1):
        idx = h - 1
        row = {"Horizon": f"t+{h}"}
        for model_name, preds in predictions_dict.items():
            row[model_name] = compute_mae(y_true[:, idx], preds[:, idx])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    config = load_config("configs/default_config.yaml")
    set_seed(config["seed"])
    device = get_device()
    print(f"Running final test evaluation on device: {device}")

    # 1. Load raw data and report missing values per split
    raw_df = pd.read_csv(config["data"]["raw_path"])
    raw_clean = raw_df.iloc[24:].copy().reset_index(drop=True)
    n = len(raw_clean)
    train_end = int(n * config["data"]["train_ratio"])
    val_end = int(n * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))

    train_raw = raw_clean.iloc[:train_end]
    val_raw = raw_clean.iloc[train_end:val_end]
    test_raw = raw_clean.iloc[val_end:].reset_index(drop=True)

    print("\n--- Missing PM2.5 Handling Summary ---")
    print(f"Total raw records: {len(raw_df):,} | Initial 24 unmeasured hours dropped (all 24 missing PM2.5).")
    print(f"Cleaned records:   {n:,} (2010-01-02 to 2014-12-31)")
    print(f"  Train split missing PM2.5: {train_raw['pm2.5'].isna().sum():,} / {len(train_raw):,} ({train_raw['pm2.5'].isna().mean():.2%}) - causally ffilled; fallback to train median")
    print(f"  Val split missing PM2.5:   {val_raw['pm2.5'].isna().sum():,} / {len(val_raw):,} ({val_raw['pm2.5'].isna().mean():.2%}) - causally ffilled")
    print(f"  Test split missing PM2.5:  {test_raw['pm2.5'].isna().sum():,} / {len(test_raw):,} ({test_raw['pm2.5'].isna().mean():.2%}) - raw test ground truth contains missing entries")

    # 2. Preprocess, split, and scale
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

    train_arr = train_scaled[feature_cols].values.astype(np.float32)
    test_arr = test_scaled[feature_cols].values.astype(np.float32)

    X_train_72, X_train_24, Y_train = build_aligned_windows(train_arr, target_idx, horizon, max_L=72)
    X_test_72, X_test_24, Y_test_scaled = build_aligned_windows(test_arr, target_idx, horizon, max_L=72)

    # 3. Identify windows whose target values are 100% real observations (not imputed)
    raw_pm25 = test_raw["pm2.5"].values
    n_windows = len(X_test_72)
    target_is_real = np.empty((n_windows, horizon), dtype=bool)
    for i in range(n_windows):
        k = i + 72
        target_is_real[i] = ~np.isnan(raw_pm25[k : k + horizon])

    real_mask = target_is_real.all(axis=1)
    imputed_windows_count = n_windows - int(real_mask.sum())
    print(f"\nTotal test sliding windows: {n_windows:,}")
    print(f"Windows with 100% real target observations: {int(real_mask.sum()):,} ({real_mask.mean():.2%})")
    print(f"Windows containing at least 1 imputed target: {imputed_windows_count:,} ({imputed_windows_count/n_windows:.2%})")
    print("Action: Final evaluation is strictly performed on the 5,839 real observation windows.")

    # Apply real target mask
    X_test_72_real = X_test_72[real_mask]
    X_test_24_real = X_test_24[real_mask]
    Y_test_real_scaled = Y_test_scaled[real_mask]
    Y_test_real = unscale(Y_test_real_scaled, target_mean, target_std)

    # Also keep unmasked versions to report exact diff
    Y_test_all = unscale(Y_test_scaled, target_mean, target_std)

    # 4. Generate predictions
    predictions_real = {}
    predictions_all = {}

    # Persistence
    persistence = PersistenceForecaster(target_idx=target_idx)
    pred_pers_all = unscale(persistence.predict(X_test_72, horizon=horizon), target_mean, target_std)
    predictions_all["Persistence"] = pred_pers_all
    predictions_real["Persistence"] = pred_pers_all[real_mask]

    # Moving Average (24h lookback)
    ma_24 = MovingAverageForecaster(target_idx=target_idx)
    pred_ma24_all = unscale(ma_24.predict(X_test_24, horizon=horizon), target_mean, target_std)
    predictions_all["MovingAverage_24h"] = pred_ma24_all
    predictions_real["MovingAverage_24h"] = pred_ma24_all[real_mask]

    # Ridge Regression (alpha=10.0)
    ridge = RidgeForecaster(alpha=10.0)
    ridge.fit(X_train_72, Y_train)
    pred_ridge_all = unscale(ridge.predict(X_test_72), target_mean, target_std)
    predictions_all["Ridge_alpha10"] = pred_ridge_all
    predictions_real["Ridge_alpha10"] = pred_ridge_all[real_mask]

    # DataLoaders for PyTorch models
    test_loader_24_all = DataLoader(
        TensorDataset(torch.tensor(X_test_24), torch.tensor(Y_test_scaled)),
        batch_size=128, shuffle=False
    )
    test_loader_72_all = DataLoader(
        TensorDataset(torch.tensor(X_test_72), torch.tensor(Y_test_scaled)),
        batch_size=128, shuffle=False
    )

    # Headline Model: Vanilla RNN (L=24 Tuned)
    rnn_24 = VanillaRNNAirQuality(
        num_features=num_features, hidden_dim=64, num_layers=2, horizon=horizon, dropout=0.2
    ).to(device)
    rnn_24_ckpt = os.path.join(config["results_dir"], "checkpoints", "tune_Vanilla_RNN_ConfigC_Lookback24h.pt")
    rnn_24.load_state_dict(torch.load(rnn_24_ckpt, weights_only=True, map_location=device))
    pred_rnn24_all = unscale(predict_model(rnn_24, test_loader_24_all, device), target_mean, target_std)
    predictions_all["Vanilla_RNN_L24_Tuned"] = pred_rnn24_all
    predictions_real["Vanilla_RNN_L24_Tuned"] = pred_rnn24_all[real_mask]

    # Headline Model: LSTM (L=24 Tuned)
    lstm_24 = LSTMAirQuality(
        num_features=num_features, hidden_dim=64, num_layers=2, horizon=horizon, dropout=0.2
    ).to(device)
    lstm_24_ckpt = os.path.join(config["results_dir"], "checkpoints", "tune_LSTM_ConfigC_Lookback24h.pt")
    lstm_24.load_state_dict(torch.load(lstm_24_ckpt, weights_only=True, map_location=device))
    pred_lstm24_all = unscale(predict_model(lstm_24, test_loader_24_all, device), target_mean, target_std)
    predictions_all["LSTM_L24_Tuned"] = pred_lstm24_all
    predictions_real["LSTM_L24_Tuned"] = pred_lstm24_all[real_mask]

    # Ablation Row: Vanilla RNN (longer lookback L=72)
    rnn_72 = VanillaRNNAirQuality(
        num_features=num_features, hidden_dim=64, num_layers=2, horizon=horizon, dropout=0.2
    ).to(device)
    rnn_72_ckpt = os.path.join(config["results_dir"], "checkpoints", "vanilla_rnn_best.pt")
    rnn_72.load_state_dict(torch.load(rnn_72_ckpt, weights_only=True, map_location=device))
    pred_rnn72_all = unscale(predict_model(rnn_72, test_loader_72_all, device), target_mean, target_std)
    predictions_all["Vanilla_RNN_L72_Ablation"] = pred_rnn72_all
    predictions_real["Vanilla_RNN_L72_Ablation"] = pred_rnn72_all[real_mask]

    # Save real observation target and prediction arrays for plotting
    np.save(os.path.join(config["results_dir"], "test_y_true.npy"), Y_test_real)
    for model_name, preds in predictions_real.items():
        np.save(os.path.join(config["results_dir"], f"test_preds_{model_name}.npy"), preds)

    # 5. Compute Metrics on Real-Observation Windows
    model_order = [
        "Persistence",
        "MovingAverage_24h",
        "Ridge_alpha10",
        "Vanilla_RNN_L24_Tuned",
        "LSTM_L24_Tuned",
        "Vanilla_RNN_L72_Ablation"
    ]

    results_real = {}
    results_all = {}
    for m in model_order:
        results_real[m] = evaluate_forecasts(Y_test_real, predictions_real[m], eval_horizons)
        results_all[m] = evaluate_forecasts(Y_test_all, predictions_all[m], eval_horizons)

    # 6. Save per-horizon MAE for all 24 horizons
    df_all_24 = compute_all_24_horizons_mae(Y_test_real, predictions_real)
    path_all_24 = os.path.join(config["results_dir"], "per_horizon_all.csv")
    df_all_24.to_csv(path_all_24, index=False)
    print(f"\nPer-horizon MAE for all 24 horizons saved to {path_all_24}")

    # 7. Print Impact of Filtering Imputed Targets
    print("\n" + "=" * 90)
    print("IMPACT OF FILTERING IMPUTED TARGETS (ALL 6,475 WINDOWS VS 5,839 REAL-OBSERVATION WINDOWS)")
    print("=" * 90)
    print(f"{'Model':<30} | {'All Windows MAE':<16} | {'Real Only MAE':<16} | {'Difference':<12}")
    print("-" * 90)
    for m in model_order:
        mae_all = results_all[m]["overall"]["MAE"]
        mae_real = results_real[m]["overall"]["MAE"]
        diff = mae_real - mae_all
        sign = "+" if diff >= 0 else ""
        print(f"{m:<30} | {mae_all:<16.4f} | {mae_real:<16.4f} | {sign}{diff:<11.4f}")
    print("=" * 90)

    # 8. Print Official Benchmark Table (Real Observations Only, Clearly Labeled)
    print("\n" + "=" * 88)
    print("OFFICIAL TEST BENCHMARK (N = 5,839 REAL-OBSERVATION WINDOWS, NO TEST SCORE RANKING/BOLDING)")
    print("Note: 'Overall' denotes the arithmetic average across all 24 forecast horizons.")
    print("=" * 88)
    print(f"{'Model (Family / Category)':<34} | {'Overall MAE':<12} | {'Overall RMSE':<12} | {'Overall R2':<10} | {'Overall sMAPE':<12}")
    print("-" * 88)
    display_names = {
        "Persistence": "Persistence (Naive Baseline)",
        "MovingAverage_24h": "Moving Average (24h Baseline)",
        "Ridge_alpha10": "Ridge Regression (alpha=10.0)",
        "Vanilla_RNN_L24_Tuned": "Vanilla RNN (L=24, Val-Tuned)",
        "LSTM_L24_Tuned": "LSTM (L=24, Val-Tuned)",
        "Vanilla_RNN_L72_Ablation": "Vanilla RNN (Ablation: L=72)"
    }
    for m in model_order:
        ov = results_real[m]["overall"]
        print(f"{display_names[m]:<34} | {ov['MAE']:<12.4f} | {ov['RMSE']:<12.4f} | {ov['R2']:<10.4f} | {ov['sMAPE']:<11.2f}%")
    print("=" * 88)

    print("\nPER-HORIZON MAE (ug/m3) ON REAL OBSERVATIONS:")
    print(f"{'Model':<34} | " + " | ".join([f"{f't+{h}':<10}" for h in eval_horizons]))
    print("-" * 82)
    for m in model_order:
        horiz_str = " | ".join([f"{results_real[m]['per_horizon'][f't+{h}']['MAE']:<10.4f}" for h in eval_horizons])
        print(f"{display_names[m]:<34} | {horiz_str}")
    print("=" * 82)

    # Save final JSON and CSV
    final_payload = {
        "evaluation_notes": {
            "test_sample_count": int(real_mask.sum()),
            "total_windows_before_mask": n_windows,
            "imputed_windows_excluded": imputed_windows_count,
            "overall_definition": "Arithmetic average across all 24 forecast horizons (t+1 through t+24)",
            "headline_selection_criterion": "Selected strictly on VALIDATION set",
            "ablation_models": ["Vanilla_RNN_L72_Ablation"]
        },
        "missing_counts": {
            "train_missing": int(train_raw["pm2.5"].isna().sum()),
            "train_total": len(train_raw),
            "val_missing": int(val_raw["pm2.5"].isna().sum()),
            "val_total": len(val_raw),
            "test_missing": int(test_raw["pm2.5"].isna().sum()),
            "test_total": len(test_raw)
        },
        "models": results_real
    }
    save_json(final_payload, os.path.join(config["results_dir"], "final_test_metrics.json"))

    # CSV Summary Table
    csv_rows = []
    for m in model_order:
        r = results_real[m]
        row = {
            "Model": display_names[m],
            "Overall_MAE": r["overall"]["MAE"],
            "Overall_RMSE": r["overall"]["RMSE"],
            "Overall_R2": r["overall"]["R2"],
            "Overall_sMAPE": r["overall"]["sMAPE"],
        }
        for h in eval_horizons:
            row[f"MAE_t+{h}"] = r["per_horizon"][f"t+{h}"]["MAE"]
            row[f"RMSE_t+{h}"] = r["per_horizon"][f"t+{h}"]["RMSE"]
            row[f"R2_t+{h}"] = r["per_horizon"][f"t+{h}"]["R2"]
            row[f"sMAPE_t+{h}"] = r["per_horizon"][f"t+{h}"]["sMAPE"]
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(os.path.join(config["results_dir"], "final_test_metrics.csv"), index=False)
    print("Final test JSON and CSV updated successfully.")


if __name__ == "__main__":
    main()
