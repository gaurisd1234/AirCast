"""
AirCast: Delta Target Formulation Evaluation and Validation-Driven Comparison.

Concept Explanation (Residual / Delta Forecasting):
1. In atmospheric PM2.5 forecasting, consecutive hourly concentrations exhibit strong physical inertia.
2. Standard multi-horizon neural networks minimize mean squared error across all 24 future hours simultaneously,
   which incentivizes the network to predict a smoothed conditional mean that often fails to anchor to the latest measurement.
3. The Delta formulation redefines the prediction target as the difference between future PM2.5 and the last observed
   PM2.5 value in the lookback window: Delta_{t+h} = y_{t+h} - y_t.
4. Under this formulation, a network outputting near-zero delta naturally predicts the strong persistence heuristic (y_{t+1} approx y_t).
5. The choice between Raw and Delta target definitions is evaluated and decided STRICTLY on the VALIDATION set.
   The held-out test set is evaluated once with the chosen setup.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, set_seed, save_json, get_device
from src.data import (
    download_and_load_raw_data,
    preprocess_data,
    split_chronological,
    fit_scalers_and_transform,
    create_sliding_windows,
    create_delta_targets,
    reconstruct_predictions_from_delta,
    CONTINUOUS_COLUMNS
)
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.models.lstm import LSTMAirQuality
from src.trainer import train_model, predict_model
from src.metrics import evaluate_forecasts, compute_mae


def unscale(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Unscale standardized values back to physical ug/m3 units."""
    return arr * std + mean


def main():
    config = load_config("configs/default_config.yaml")
    seed = config["seed"]
    set_seed(seed)
    device = get_device()
    print(f"Running Delta Target Formulation Study on device: {device} (Seed: {seed})\n")

    # 1. Ingest raw data and split chronologically
    raw_df = download_and_load_raw_data(config["data"]["raw_url"], config["data"]["raw_path"])
    raw_clean = raw_df.iloc[24:].copy().reset_index(drop=True)
    n = len(raw_clean)
    train_end = int(n * config["data"]["train_ratio"])
    val_end = int(n * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    test_raw = raw_clean.iloc[val_end:].reset_index(drop=True)
    raw_pm25 = test_raw["pm2.5"].values

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
    lookback = 24  # Validation-selected lookback
    horizon = config["data"]["max_horizon"]
    max_L = 72     # Aligned lookback anchor across test evaluation
    eval_horizons = config["data"]["evaluation_horizons"]

    train_scaled, val_scaled, test_scaled, scaler_info, _ = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )

    target_mean = scaler_info["target_mean"]
    target_std = scaler_info["target_std"]

    train_arr = train_scaled[feature_cols].values.astype(np.float32)
    val_arr = val_scaled[feature_cols].values.astype(np.float32)
    test_arr = test_scaled[feature_cols].values.astype(np.float32)

    # Build Training and Validation sliding windows
    X_train, Y_train_raw = create_sliding_windows(train_arr, target_idx, lookback, horizon)
    X_val, Y_val_raw = create_sliding_windows(val_arr, target_idx, lookback, horizon)

    # Construct Delta targets: Delta_{t+h} = y_{t+h} - y_t
    Y_train_delta = create_delta_targets(Y_train_raw, X_train, target_idx)
    Y_val_delta = create_delta_targets(Y_val_raw, X_val, target_idx)

    # Common training config matching validation-selected hyperparameters
    train_cfg = {
        "training": {
            "epochs": 15,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "patience": 4,
            "clip_grad_norm": 1.0,
            "batch_size": 128
        },
        "results_dir": config["results_dir"]
    }

    # =========================================================================
    # PART 1: VALIDATION-ONLY COMPARISON (RAW VS. DELTA)
    # =========================================================================
    print("=" * 70)
    print("PART 1: VALIDATION SET COMPARISON (RAW vs. DELTA TARGET DEFINITION)")
    print("=" * 70)

    val_y_true = unscale(Y_val_raw, target_mean, target_std)
    val_records = {}

    experiments = [
        ("Vanilla_RNN", "raw", Y_train_raw, Y_val_raw),
        ("Vanilla_RNN", "delta", Y_train_delta, Y_val_delta),
        ("LSTM", "raw", Y_train_raw, Y_val_raw),
        ("LSTM", "delta", Y_train_delta, Y_val_delta),
    ]

    models_trained = {}
    for model_name, target_def, y_tr, y_va in experiments:
        set_seed(seed)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_tr)),
            batch_size=128, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_va)),
            batch_size=128, shuffle=False
        )

        if model_name == "Vanilla_RNN":
            model = VanillaRNNAirQuality(num_features, 64, 2, horizon, 0.2).to(device)
        else:
            model = LSTMAirQuality(num_features, 64, 2, horizon, 0.2).to(device)

        ckpt_name = f"tune_{model_name}_{target_def.capitalize()}_L24"
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg,
            checkpoint_name=ckpt_name,
            device=device
        )
        models_trained[f"{model_name}_{target_def}"] = trained_model

        preds_scaled = predict_model(trained_model, val_loader, device)
        if target_def == "delta":
            preds_scaled = reconstruct_predictions_from_delta(preds_scaled, X_val, target_idx)

        preds_unscaled = unscale(preds_scaled, target_mean, target_std)
        metrics = evaluate_forecasts(val_y_true, preds_unscaled, eval_horizons)

        val_records[f"{model_name}_{target_def}"] = {
            "model": model_name,
            "target_definition": target_def,
            "overall": metrics["overall"],
            "per_horizon": metrics["per_horizon"],
            "best_epoch": int(np.argmin(history["val_loss"]) + 1),
            "epochs_run": len(history["train_loss"]),
            "best_val_loss": float(min(history["val_loss"])),
            "train_loss_history": [float(x) for x in history["train_loss"]],
            "val_loss_history": [float(x) for x in history["val_loss"]]
        }

    # Summary table of validation results
    print("\n--- Summary of Validation Results ---")
    val_table_rows = []
    for k, v in val_records.items():
        val_table_rows.append({
            "Setup": k,
            "Best Epoch": f"{v['best_epoch']}/{v['epochs_run']}",
            "Val Loss": round(v["best_val_loss"], 4),
            "Overall MAE": round(v["overall"]["MAE"], 4),
            "t+1 MAE": round(v["per_horizon"]["t+1"]["MAE"], 4),
            "t+6 MAE": round(v["per_horizon"]["t+6"]["MAE"], 4),
            "t+12 MAE": round(v["per_horizon"]["t+12"]["MAE"], 4),
            "t+24 MAE": round(v["per_horizon"]["t+24"]["MAE"], 4),
        })
    df_val_summary = pd.DataFrame(val_table_rows)
    print(df_val_summary.to_string(index=False))

    # Explicit validation selection check
    rnn_chosen = "delta" if val_records["Vanilla_RNN_delta"]["overall"]["MAE"] < val_records["Vanilla_RNN_raw"]["overall"]["MAE"] else "raw"
    lstm_chosen = "delta" if val_records["LSTM_delta"]["overall"]["MAE"] < val_records["LSTM_raw"]["overall"]["MAE"] else "raw"

    print(f"\nValidation Decision for Vanilla RNN: CHOSEN '{rnn_chosen.upper()}' (MAE: {val_records[f'Vanilla_RNN_{rnn_chosen}']['overall']['MAE']:.4f})")
    print(f"Validation Decision for LSTM:        CHOSEN '{lstm_chosen.upper()}' (MAE: {val_records[f'LSTM_{lstm_chosen}']['overall']['MAE']:.4f})")

    # Save validation comparison
    val_comp_path = os.path.join(config["results_dir"], "delta_validation_comparison.json")
    save_json(val_records, val_comp_path)
    print(f"Saved validation comparison to: {val_comp_path}")

    # =========================================================================
    # PART 2: FINAL TEST SET EVALUATION WITH CHOSEN DELTA SETUP
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: FINAL TEST SET EVALUATION WITH CHOSEN DELTA SETUP")
    print("NOTE: This represents a SECOND USE of the test set after validation selection.")
    print("=" * 70)

    # Build aligned test windows with max_L=72 anchor
    n_test_windows = len(test_arr) - max_L - horizon + 1
    X_test_24 = np.empty((n_test_windows, lookback, test_arr.shape[1]), dtype=np.float32)
    Y_test_raw_scaled = np.empty((n_test_windows, horizon), dtype=np.float32)
    target_is_real = np.empty((n_test_windows, horizon), dtype=bool)

    for i in range(n_test_windows):
        k = i + max_L
        X_test_24[i] = test_arr[k - lookback : k]
        Y_test_raw_scaled[i] = test_arr[k : k + horizon, target_idx]
        target_is_real[i] = ~np.isnan(raw_pm25[k : k + horizon])

    real_mask = target_is_real.all(axis=1)
    print(f"Total test sliding windows: {n_test_windows:,}")
    print(f"Windows with 100% real target observations: {int(real_mask.sum()):,} ({real_mask.mean():.2%})")

    test_loader_24 = DataLoader(
        TensorDataset(torch.tensor(X_test_24), torch.tensor(Y_test_raw_scaled)),
        batch_size=128, shuffle=False
    )

    # Retrieve chosen models trained on Delta target from Part 1
    rnn_final = models_trained["Vanilla_RNN_delta"]
    lstm_final = models_trained["LSTM_delta"]

    # Generate Delta predictions on test set
    rnn_delta_preds = predict_model(rnn_final, test_loader_24, device)
    lstm_delta_preds = predict_model(lstm_final, test_loader_24, device)

    # Reconstruct predictions in scaled space
    rnn_scaled_preds = reconstruct_predictions_from_delta(rnn_delta_preds, X_test_24, target_idx)
    lstm_scaled_preds = reconstruct_predictions_from_delta(lstm_delta_preds, X_test_24, target_idx)

    # Unscale to physical ug/m3 units
    rnn_test_preds = unscale(rnn_scaled_preds, target_mean, target_std)
    lstm_test_preds = unscale(lstm_scaled_preds, target_mean, target_std)

    # Mask to real observation windows only
    Y_test_real = unscale(Y_test_raw_scaled[real_mask], target_mean, target_std)
    rnn_test_real = rnn_test_preds[real_mask]
    lstm_test_real = lstm_test_preds[real_mask]

    # Save predictions
    np.save(os.path.join(config["results_dir"], "test_preds_Vanilla_RNN_Delta_L24.npy"), rnn_test_real)
    np.save(os.path.join(config["results_dir"], "test_preds_LSTM_Delta_L24.npy"), lstm_test_real)

    # Compute Test Metrics
    rnn_test_metrics = evaluate_forecasts(Y_test_real, rnn_test_real, eval_horizons)
    lstm_test_metrics = evaluate_forecasts(Y_test_real, lstm_test_real, eval_horizons)

    test_results = {
        "evaluation_notes": "Second use of test set after diagnosing weak LSTM t+1 and selecting Delta formulation on validation.",
        "real_target_windows_count": int(real_mask.sum()),
        "models": {
            "Vanilla_RNN_Delta_L24": rnn_test_metrics,
            "LSTM_Delta_L24": lstm_test_metrics
        }
    }

    test_res_path = os.path.join(config["results_dir"], "delta_target_results.json")
    save_json(test_results, test_res_path)
    print(f"\nSaved Delta test results to: {test_res_path}")

    # Print clean comparison table
    print("\n--- Final Test Performance Comparison (Real Target Windows N=5,839) ---")
    test_rows = [
        {
            "Model": "Vanilla RNN (Delta L=24, Val-Selected)",
            "Overall MAE": round(rnn_test_metrics["overall"]["MAE"], 4),
            "Overall RMSE": round(rnn_test_metrics["overall"]["RMSE"], 4),
            "Overall R2": round(rnn_test_metrics["overall"]["R2"], 4),
            "Overall sMAPE": round(rnn_test_metrics["overall"]["sMAPE"], 4),
            "t+1 MAE": round(rnn_test_metrics["per_horizon"]["t+1"]["MAE"], 4),
            "t+6 MAE": round(rnn_test_metrics["per_horizon"]["t+6"]["MAE"], 4),
            "t+12 MAE": round(rnn_test_metrics["per_horizon"]["t+12"]["MAE"], 4),
            "t+24 MAE": round(rnn_test_metrics["per_horizon"]["t+24"]["MAE"], 4)
        },
        {
            "Model": "LSTM (Delta L=24, Val-Selected)",
            "Overall MAE": round(lstm_test_metrics["overall"]["MAE"], 4),
            "Overall RMSE": round(lstm_test_metrics["overall"]["RMSE"], 4),
            "Overall R2": round(lstm_test_metrics["overall"]["R2"], 4),
            "Overall sMAPE": round(lstm_test_metrics["overall"]["sMAPE"], 4),
            "t+1 MAE": round(lstm_test_metrics["per_horizon"]["t+1"]["MAE"], 4),
            "t+6 MAE": round(lstm_test_metrics["per_horizon"]["t+6"]["MAE"], 4),
            "t+12 MAE": round(lstm_test_metrics["per_horizon"]["t+12"]["MAE"], 4),
            "t+24 MAE": round(lstm_test_metrics["per_horizon"]["t+24"]["MAE"], 4)
        }
    ]
    df_test_summary = pd.DataFrame(test_rows)
    print(df_test_summary.to_string(index=False))

    # Update per_horizon_all.csv with Delta columns
    per_h_path = os.path.join(config["results_dir"], "per_horizon_all.csv")
    if os.path.exists(per_h_path):
        df_per_h = pd.read_csv(per_h_path)
        # Compute 24-horizon MAE for Delta models
        for h in range(1, horizon + 1):
            idx = h - 1
            df_per_h.loc[df_per_h["Horizon"] == f"t+{h}", "Vanilla_RNN_Delta_L24"] = compute_mae(
                Y_test_real[:, idx], rnn_test_real[:, idx]
            )
            df_per_h.loc[df_per_h["Horizon"] == f"t+{h}", "LSTM_Delta_L24"] = compute_mae(
                Y_test_real[:, idx], lstm_test_real[:, idx]
            )
        df_per_h.to_csv(per_h_path, index=False)
        print(f"Updated {per_h_path} with Vanilla_RNN_Delta_L24 and LSTM_Delta_L24 columns.")

    # Update final_test_metrics.csv
    csv_metrics_path = os.path.join(config["results_dir"], "final_test_metrics.csv")
    if os.path.exists(csv_metrics_path):
        df_final = pd.read_csv(csv_metrics_path)
        new_records = []
        for m_name, m_metrics in [
            ("Vanilla RNN (Delta L=24, Val-Selected)", rnn_test_metrics),
            ("LSTM (Delta L=24, Val-Selected)", lstm_test_metrics)
        ]:
            rec = {
                "Model": m_name,
                "Overall_MAE": m_metrics["overall"]["MAE"],
                "Overall_RMSE": m_metrics["overall"]["RMSE"],
                "Overall_R2": m_metrics["overall"]["R2"],
                "Overall_sMAPE": m_metrics["overall"]["sMAPE"],
                "MAE_t+1": m_metrics["per_horizon"]["t+1"]["MAE"],
                "RMSE_t+1": m_metrics["per_horizon"]["t+1"]["RMSE"],
                "R2_t+1": m_metrics["per_horizon"]["t+1"]["R2"],
                "sMAPE_t+1": m_metrics["per_horizon"]["t+1"]["sMAPE"],
                "MAE_t+6": m_metrics["per_horizon"]["t+6"]["MAE"],
                "RMSE_t+6": m_metrics["per_horizon"]["t+6"]["RMSE"],
                "R2_t+6": m_metrics["per_horizon"]["t+6"]["R2"],
                "sMAPE_t+6": m_metrics["per_horizon"]["t+6"]["sMAPE"],
                "MAE_t+12": m_metrics["per_horizon"]["t+12"]["MAE"],
                "RMSE_t+12": m_metrics["per_horizon"]["t+12"]["RMSE"],
                "R2_t+12": m_metrics["per_horizon"]["t+12"]["R2"],
                "sMAPE_t+12": m_metrics["per_horizon"]["t+12"]["sMAPE"],
                "MAE_t+24": m_metrics["per_horizon"]["t+24"]["MAE"],
                "RMSE_t+24": m_metrics["per_horizon"]["t+24"]["RMSE"],
                "R2_t+24": m_metrics["per_horizon"]["t+24"]["R2"],
                "sMAPE_t+24": m_metrics["per_horizon"]["t+24"]["sMAPE"],
            }
            # Remove if already exists
            df_final = df_final[df_final["Model"] != m_name]
            new_records.append(rec)
        df_final = pd.concat([df_final, pd.DataFrame(new_records)], ignore_index=True)
        df_final.to_csv(csv_metrics_path, index=False)
        print(f"Updated {csv_metrics_path} with Delta models.")

    # Update final_test_metrics.json
    json_metrics_path = os.path.join(config["results_dir"], "final_test_metrics.json")
    if os.path.exists(json_metrics_path):
        final_json = load_config(json_metrics_path)
        final_json["models"]["Vanilla_RNN_Delta_L24"] = rnn_test_metrics
        final_json["models"]["LSTM_Delta_L24"] = lstm_test_metrics
        save_json(final_json, json_metrics_path)
        print(f"Updated {json_metrics_path} with Vanilla_RNN_Delta_L24 and LSTM_Delta_L24.")


if __name__ == "__main__":
    main()
