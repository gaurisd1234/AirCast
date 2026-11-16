"""
Train and validate Vanilla (Elman) RNN on AirCast dataset.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.utils import load_config, set_seed, save_json, get_device
from src.data import get_dataloaders
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.trainer import train_model, predict_model
from src.metrics import evaluate_forecasts


def main():
    config = load_config("configs/default_config.yaml")
    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    print("Building leak-free dataloaders...")
    train_loader, val_loader, test_loader, scaler_info, feature_cols = get_dataloaders(config)
    num_features = len(feature_cols)
    horizon = config["data"]["max_horizon"]
    eval_horizons = config["data"]["evaluation_horizons"]

    print(f"Num features: {num_features}, Max Horizon: {horizon}")

    rnn_config = config["models"]["vanilla_rnn"]
    model = VanillaRNNAirQuality(
        num_features=num_features,
        hidden_dim=rnn_config["hidden_dim"],
        num_layers=rnn_config["num_layers"],
        horizon=horizon,
        dropout=rnn_config["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Vanilla RNN initialized with {total_params:,} trainable parameters.")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_name="vanilla_rnn_best",
        device=device
    )

    # Evaluate on VALIDATION set (Test set remains untouched)
    print("\nGenerating predictions on Validation set...")
    val_preds_scaled = predict_model(model, val_loader, device)

    # Gather true validation targets from DataLoader
    val_y_scaled = []
    for _, y_b in val_loader:
        val_y_scaled.append(y_b.numpy())
    val_y_scaled = np.concatenate(val_y_scaled, axis=0)

    # Unscale predictions and ground truth to physical units (ug/m3)
    target_mean = scaler_info["target_mean"]
    target_std = scaler_info["target_std"]
    val_preds = val_preds_scaled * target_std + target_mean
    val_y_true = val_y_scaled * target_std + target_mean

    val_metrics = evaluate_forecasts(val_y_true, val_preds, eval_horizons)
    val_metrics["history"] = history
    val_metrics["num_parameters"] = total_params

    print("\n" + "=" * 70)
    print(f"{'Model':<16} | {'Overall MAE':<12} | {'Overall RMSE':<12} | {'Overall R2':<10} | {'Overall sMAPE':<12}")
    print("-" * 70)
    ov = val_metrics["overall"]
    print(f"{'Vanilla RNN':<16} | {ov['MAE']:<12.4f} | {ov['RMSE']:<12.4f} | {ov['R2']:<10.4f} | {ov['sMAPE']:<11.2f}%")
    print("=" * 70)

    print("\nPer-Horizon MAE (ug/m3) on Validation:")
    print(f"{'Model':<16} | " + " | ".join([f"{f't+{h}':<10}" for h in eval_horizons]))
    print("-" * 65)
    horiz_str = " | ".join([f"{val_metrics['per_horizon'][f't+{h}']['MAE']:<10.4f}" for h in eval_horizons])
    print(f"{'Vanilla RNN':<16} | {horiz_str}")

    out_path = os.path.join(config["results_dir"], "vanilla_rnn_val_results.json")
    save_json(val_metrics, out_path)
    print(f"\nVanilla RNN validation results saved to {out_path}")


if __name__ == "__main__":
    main()
