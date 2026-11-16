"""
Multi-seed retraining and evaluation of frozen validation-selected Vanilla RNN and LSTM models.
Runs 5 fixed seeds (42, 43, 44, 45, 46) and reports empirical Mean and Standard Deviation.
"""

import copy
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from src.utils import load_config, set_seed, save_json, get_device
from src.data import get_dataloaders, preprocess_data, fit_scalers_and_transform, CONTINUOUS_COLUMNS
from torch.utils.data import DataLoader, TensorDataset
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.models.lstm import LSTMAirQuality
from src.trainer import train_model, predict_model
from src.metrics import compute_mae, compute_rmse, compute_smape, compute_r2


def unscale(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


def run_single_seed_experiment(
    model_type: str,
    seed: int,
    base_config: dict,
    real_mask: np.ndarray,
    target_mean: float,
    target_std: float,
    eval_horizons: list,
    device: torch.device
) -> dict:
    set_seed(seed)
    cfg = copy.deepcopy(base_config)
    cfg["seed"] = seed
    cfg["data"]["lookback_hours"] = 24
    cfg["training"]["batch_size"] = 128
    cfg["training"]["epochs"] = 15
    cfg["training"]["patience"] = 4
    cfg["training"]["learning_rate"] = 0.001

    train_loader, val_loader, test_loader, _, feature_cols = get_dataloaders(cfg)
    num_features = len(feature_cols)
    horizon = cfg["data"]["max_horizon"]

    if model_type == "vanilla_rnn":
        model = VanillaRNNAirQuality(
            num_features=num_features,
            hidden_dim=64,
            num_layers=2,
            horizon=horizon,
            dropout=0.2
        ).to(device)
    elif model_type == "lstm":
        model = LSTMAirQuality(
            num_features=num_features,
            hidden_dim=64,
            num_layers=2,
            horizon=horizon,
            dropout=0.2
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt_name = f"multiseed_{model_type}_seed{seed}"
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        checkpoint_name=ckpt_name,
        device=device
    )

    # Predict on test set
    preds_scaled = predict_model(model, test_loader, device)

    # Gather test ground truth from test_loader
    test_y_scaled = []
    for _, y_b in test_loader:
        test_y_scaled.append(y_b.numpy())
    test_y_scaled = np.concatenate(test_y_scaled, axis=0)

    # Filter to real-observation windows
    preds_real = unscale(preds_scaled[real_mask], target_mean, target_std)
    y_real = unscale(test_y_scaled[real_mask], target_mean, target_std)

    # Compute overall metrics (average over 24 horizons)
    metrics = {
        "seed": seed,
        "overall": {
            "MAE": compute_mae(y_real, preds_real),
            "RMSE": compute_rmse(y_real, preds_real),
            "sMAPE": compute_smape(y_real, preds_real),
            "R2": compute_r2(y_real, preds_real)
        },
        "per_horizon": {}
    }

    for h in eval_horizons:
        idx = h - 1
        metrics["per_horizon"][f"t+{h}"] = {
            "MAE": compute_mae(y_real[:, idx], preds_real[:, idx]),
            "RMSE": compute_rmse(y_real[:, idx], preds_real[:, idx]),
            "sMAPE": compute_smape(y_real[:, idx], preds_real[:, idx]),
            "R2": compute_r2(y_real[:, idx], preds_real[:, idx])
        }

    return metrics, preds_real


def aggregate_seed_metrics(seed_metrics_list: list, eval_horizons: list) -> dict:
    """Compute empirical mean and sample standard deviation (ddof=1) across seeds."""
    agg = {"overall": {}, "per_horizon": {}}

    for metric in ["MAE", "RMSE", "sMAPE", "R2"]:
        vals = [m["overall"][metric] for m in seed_metrics_list]
        agg["overall"][metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1))
        }

    for h in eval_horizons:
        h_key = f"t+{h}"
        agg["per_horizon"][h_key] = {}
        for metric in ["MAE", "RMSE", "sMAPE", "R2"]:
            vals = [m["per_horizon"][h_key][metric] for m in seed_metrics_list]
            agg["per_horizon"][h_key][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1))
            }

    return agg


def main():
    base_config = load_config("configs/default_config.yaml")
    device = get_device()
    results_dir = base_config["results_dir"]
    eval_horizons = base_config["data"]["evaluation_horizons"]
    seeds = [42, 43, 44, 45, 46]

    print(f"Running multi-seed evaluation across 5 seeds {seeds} on device: {device}\n")

    # Load data and align windows exactly with evaluate_test.py
    raw_df = pd.read_csv(base_config["data"]["raw_path"])
    clean_df = preprocess_data(raw_df)
    n = len(clean_df)
    train_end = int(n * base_config["data"]["train_ratio"])
    val_end = int(n * (base_config["data"]["train_ratio"] + base_config["data"]["val_ratio"]))

    train_df = clean_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = clean_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = clean_df.iloc[val_end:].copy().reset_index(drop=True)

    feature_cols = [c for c in train_df.columns if c != "datetime"]
    target_idx = feature_cols.index("pm2.5")
    num_features = len(feature_cols)
    horizon = 24

    train_scaled, val_scaled, test_scaled, scaler_info, _ = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )
    target_mean = scaler_info["target_mean"]
    target_std = scaler_info["target_std"]

    train_arr = train_scaled[feature_cols].values.astype(np.float32)
    val_arr = val_scaled[feature_cols].values.astype(np.float32)
    test_arr = test_scaled[feature_cols].values.astype(np.float32)

    from scripts.evaluate_test import build_aligned_windows
    X_train_72, X_train_24, Y_train = build_aligned_windows(train_arr, target_idx, horizon, max_L=72)
    X_val_72, X_val_24, Y_val_scaled = build_aligned_windows(val_arr, target_idx, horizon, max_L=72)
    X_test_72, X_test_24, Y_test_scaled = build_aligned_windows(test_arr, target_idx, horizon, max_L=72)

    # Establish real-observation mask on aligned test windows (N=5,839)
    test_raw_pm25 = raw_df.iloc[24:].reset_index(drop=True).iloc[val_end:]["pm2.5"].values
    n_windows = len(X_test_24)
    target_is_real = np.empty((n_windows, horizon), dtype=bool)
    for i in range(n_windows):
        k = i + 72
        target_is_real[i] = ~np.isnan(test_raw_pm25[k : k + horizon])
    real_mask = target_is_real.all(axis=1)

    X_test_24_real = torch.tensor(X_test_24[real_mask])
    Y_test_real_scaled = torch.tensor(Y_test_scaled[real_mask])
    test_loader_real = DataLoader(TensorDataset(X_test_24_real, Y_test_real_scaled), batch_size=128, shuffle=False)
    Y_test_real = unscale(Y_test_scaled[real_mask], target_mean, target_std)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_24), torch.tensor(Y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_24), torch.tensor(Y_val_scaled)), batch_size=128, shuffle=False)

    all_models_multiseed = {}

    for model_type in ["vanilla_rnn", "lstm"]:
        display_name = "Vanilla RNN (L=24)" if model_type == "vanilla_rnn" else "LSTM (L=24)"
        print("\n" + "=" * 75)
        print(f"EVALUATING 5 SEEDS FOR {display_name}")
        print("=" * 75)

        seed_results = []
        seed_predictions = []

        for seed in seeds:
            ckpt_path = os.path.join(results_dir, "checkpoints", f"multiseed_{model_type}_seed{seed}.pt")
            if model_type == "vanilla_rnn":
                model = VanillaRNNAirQuality(num_features=num_features, hidden_dim=64, num_layers=2, horizon=horizon, dropout=0.2).to(device)
            else:
                model = LSTMAirQuality(num_features=num_features, hidden_dim=64, num_layers=2, horizon=horizon, dropout=0.2).to(device)

            model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
            preds_scaled = predict_model(model, test_loader_real, device)
            preds_real = unscale(preds_scaled, target_mean, target_std)

            metrics = {
                "seed": seed,
                "overall": {
                    "MAE": compute_mae(Y_test_real, preds_real),
                    "RMSE": compute_rmse(Y_test_real, preds_real),
                    "sMAPE": compute_smape(Y_test_real, preds_real),
                    "R2": compute_r2(Y_test_real, preds_real)
                },
                "per_horizon": {}
            }
            for h in eval_horizons:
                idx = h - 1
                metrics["per_horizon"][f"t+{h}"] = {
                    "MAE": compute_mae(Y_test_real[:, idx], preds_real[:, idx]),
                    "RMSE": compute_rmse(Y_test_real[:, idx], preds_real[:, idx]),
                    "sMAPE": compute_smape(Y_test_real[:, idx], preds_real[:, idx]),
                    "R2": compute_r2(Y_test_real[:, idx], preds_real[:, idx])
                }

            seed_results.append(metrics)
            seed_predictions.append(preds_real)
            ov = metrics["overall"]
            print(f"Seed {seed} Test Results: MAE={ov['MAE']:.4f} | RMSE={ov['RMSE']:.4f} | sMAPE={ov['sMAPE']:.2f}%")

        agg = aggregate_seed_metrics(seed_results, eval_horizons)
        all_models_multiseed[model_type] = {
            "per_seed": seed_results,
            "aggregated": agg
        }

        # Save ensemble mean prediction across the 5 seeds on the exact 5,839 real windows
        ens_preds = np.mean(seed_predictions, axis=0)
        np.save(os.path.join(results_dir, f"test_preds_{model_type}_5seed_mean.npy"), ens_preds)

    # Display multi-seed comparison table
    print("\n" + "=" * 95)
    print("MULTI-SEED TEST PERFORMANCE (5 SEEDS: 42, 43, 44, 45, 46, REAL OBSERVATIONS ONLY)")
    print("=" * 95)
    print(f"{'Model':<20} | {'Overall MAE (mean ± std)':<26} | {'Overall RMSE (mean ± std)':<26} | {'Overall sMAPE (mean ± std)':<22}")
    print("-" * 95)
    for model_type, name in [("vanilla_rnn", "Vanilla RNN (L=24)"), ("lstm", "LSTM (L=24)")]:
        ov = all_models_multiseed[model_type]["aggregated"]["overall"]
        mae_str = f"{ov['MAE']['mean']:.4f} ± {ov['MAE']['std']:.4f}"
        rmse_str = f"{ov['RMSE']['mean']:.4f} ± {ov['RMSE']['std']:.4f}"
        smape_str = f"{ov['sMAPE']['mean']:.2f}% ± {ov['sMAPE']['std']:.2f}%"
        print(f"{name:<20} | {mae_str:<26} | {rmse_str:<26} | {smape_str:<22}")
    print("=" * 95)

    print("\nPER-HORIZON MAE (ug/m3) (mean ± std) ACROSS 5 SEEDS:")
    header = f"{'Model':<20} | " + " | ".join([f"{f't+{h}':<18}" for h in eval_horizons])
    print(header)
    print("-" * len(header))
    for model_type, name in [("vanilla_rnn", "Vanilla RNN (L=24)"), ("lstm", "LSTM (L=24)")]:
        h_strs = []
        for h in eval_horizons:
            h_data = all_models_multiseed[model_type]["aggregated"]["per_horizon"][f"t+{h}"]["MAE"]
            h_strs.append(f"{h_data['mean']:.2f} ± {h_data['std']:.2f}")
        print(f"{name:<20} | " + " | ".join([f"{s:<18}" for s in h_strs]))
    print("=" * len(header))

    out_path = os.path.join(results_dir, "multiseed_results.json")
    save_json(all_models_multiseed, out_path)
    print(f"\nMulti-seed results saved to {out_path}")


if __name__ == "__main__":
    main()
