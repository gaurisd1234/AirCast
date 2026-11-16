"""
Hyperparameter tuning and lookback ablation study evaluated STRICTLY on the Validation set.
Tests learning rate, hidden capacity, regularization, and lookback window (L=24 vs L=72).
"""

import copy
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.utils import load_config, set_seed, save_json, get_device
from src.data import get_dataloaders
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.models.lstm import LSTMAirQuality
from src.trainer import train_model, predict_model
from src.metrics import evaluate_forecasts


def run_single_config(
    model_type: str,
    custom_cfg: dict,
    checkpoint_name: str,
    device: torch.device
) -> dict:
    """Train a specific configuration and evaluate solely on Validation."""
    set_seed(custom_cfg["seed"])

    train_loader, val_loader, _, scaler_info, feature_cols = get_dataloaders(custom_cfg)
    num_features = len(feature_cols)
    horizon = custom_cfg["data"]["max_horizon"]
    eval_horizons = custom_cfg["data"]["evaluation_horizons"]

    m_cfg = custom_cfg["models"][model_type]
    if model_type == "vanilla_rnn":
        model = VanillaRNNAirQuality(
            num_features=num_features,
            hidden_dim=m_cfg["hidden_dim"],
            num_layers=m_cfg["num_layers"],
            horizon=horizon,
            dropout=m_cfg["dropout"]
        ).to(device)
    elif model_type == "lstm":
        model = LSTMAirQuality(
            num_features=num_features,
            hidden_dim=m_cfg["hidden_dim"],
            num_layers=m_cfg["num_layers"],
            horizon=horizon,
            dropout=m_cfg["dropout"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=custom_cfg,
        checkpoint_name=checkpoint_name,
        device=device
    )

    # Evaluate on VALIDATION set
    val_preds_scaled = predict_model(model, val_loader, device)
    val_y_scaled = []
    for _, y_b in val_loader:
        val_y_scaled.append(y_b.numpy())
    val_y_scaled = np.concatenate(val_y_scaled, axis=0)

    target_mean = scaler_info["target_mean"]
    target_std = scaler_info["target_std"]
    val_preds = val_preds_scaled * target_std + target_mean
    val_y_true = val_y_scaled * target_std + target_mean

    metrics = evaluate_forecasts(val_y_true, val_preds, eval_horizons)
    metrics["num_parameters"] = total_params
    metrics["history"] = history
    metrics["best_val_loss"] = min(history["val_loss"])
    metrics["config"] = {
        "lookback_hours": custom_cfg["data"]["lookback_hours"],
        "hidden_dim": m_cfg["hidden_dim"],
        "num_layers": m_cfg["num_layers"],
        "dropout": m_cfg["dropout"],
        "learning_rate": custom_cfg["training"]["learning_rate"],
        "weight_decay": custom_cfg["training"]["weight_decay"],
        "batch_size": custom_cfg["training"]["batch_size"]
    }

    return metrics


def main():
    base_config = load_config("configs/default_config.yaml")
    device = get_device()
    print(f"Running validation hyperparameter & ablation sweep on device: {device}\n")

    experiments = [
        # --- Vanilla RNN Experiments ---
        {
            "name": "Vanilla_RNN_ConfigA_Baseline",
            "model_type": "vanilla_rnn",
            "overrides": {
                "training.batch_size": 128,
                "training.epochs": 15,
                "training.patience": 4,
                "training.learning_rate": 0.001,
                "models.vanilla_rnn.hidden_dim": 64,
                "models.vanilla_rnn.dropout": 0.2,
                "data.lookback_hours": 72
            }
        },
        {
            "name": "Vanilla_RNN_ConfigB_Regularized",
            "model_type": "vanilla_rnn",
            "overrides": {
                "training.batch_size": 128,
                "training.epochs": 15,
                "training.patience": 4,
                "training.learning_rate": 0.0005,
                "training.weight_decay": 0.0001,
                "models.vanilla_rnn.hidden_dim": 32,
                "models.vanilla_rnn.dropout": 0.3,
                "data.lookback_hours": 72
            }
        },
        {
            "name": "Vanilla_RNN_ConfigC_Lookback24h",
            "model_type": "vanilla_rnn",
            "overrides": {
                "training.batch_size": 128,
                "training.epochs": 15,
                "training.patience": 4,
                "training.learning_rate": 0.001,
                "models.vanilla_rnn.hidden_dim": 64,
                "models.vanilla_rnn.dropout": 0.2,
                "data.lookback_hours": 24
            }
        },
        # --- LSTM Experiments ---
        {
            "name": "LSTM_ConfigA_Baseline",
            "model_type": "lstm",
            "overrides": {
                "training.batch_size": 128,
                "training.epochs": 15,
                "training.patience": 4,
                "training.learning_rate": 0.001,
                "models.lstm.hidden_dim": 64,
                "models.lstm.dropout": 0.2,
                "data.lookback_hours": 72
            }
        },
        {
            "name": "LSTM_ConfigB_Regularized_LowerLR",
            "model_type": "lstm",
            "overrides": {
                "training.batch_size": 128,
                "training.epochs": 15,
                "training.patience": 4,
                "training.learning_rate": 0.0005,
                "training.weight_decay": 0.0001,
                "models.lstm.hidden_dim": 32,
                "models.lstm.dropout": 0.3,
                "data.lookback_hours": 72
            }
        },
        {
            "name": "LSTM_ConfigC_Lookback24h",
            "model_type": "lstm",
            "overrides": {
                "training.batch_size": 128,
                "training.epochs": 15,
                "training.patience": 4,
                "training.learning_rate": 0.001,
                "models.lstm.hidden_dim": 64,
                "models.lstm.dropout": 0.2,
                "data.lookback_hours": 24
            }
        }
    ]

    all_tuning_results = {}

    for exp in experiments:
        exp_name = exp["name"]
        model_type = exp["model_type"]
        print(f"\n{'='*70}\nRunning Experiment: {exp_name}\n{'='*70}")

        cfg = copy.deepcopy(base_config)
        for key, val in exp["overrides"].items():
            parts = key.split(".")
            curr = cfg
            for p in parts[:-1]:
                curr = curr[p]
            curr[parts[-1]] = val

        metrics = run_single_config(
            model_type=model_type,
            custom_cfg=cfg,
            checkpoint_name=f"tune_{exp_name}",
            device=device
        )
        all_tuning_results[exp_name] = metrics

        ov = metrics["overall"]
        print(f"\nResult for {exp_name}:")
        print(f"  Val MAE: {ov['MAE']:.4f} ug/m3 | Val RMSE: {ov['RMSE']:.4f} | Best Val Loss: {metrics['best_val_loss']:.6f}")

    # Summary Table of All Tuning Runs
    print("\n" + "=" * 80)
    print(f"{'Experiment Name':<34} | {'Val MAE':<10} | {'Val RMSE':<10} | {'Val R2':<8} | {'Best Val Loss':<12}")
    print("-" * 80)
    for name, res in all_tuning_results.items():
        ov = res["overall"]
        print(f"{name:<34} | {ov['MAE']:<10.4f} | {ov['RMSE']:<10.4f} | {ov['R2']:<8.4f} | {res['best_val_loss']:<12.6f}")
    print("=" * 80)

    # Identify best configuration for Vanilla RNN and LSTM
    best_rnn_exp = min(
        [k for k in all_tuning_results if k.startswith("Vanilla_RNN")],
        key=lambda k: all_tuning_results[k]["overall"]["MAE"]
    )
    best_lstm_exp = min(
        [k for k in all_tuning_results if k.startswith("LSTM")],
        key=lambda k: all_tuning_results[k]["overall"]["MAE"]
    )

    print(f"\nBest Vanilla RNN Config on Validation: {best_rnn_exp} (MAE: {all_tuning_results[best_rnn_exp]['overall']['MAE']:.4f})")
    print(f"Best LSTM Config on Validation:        {best_lstm_exp} (MAE: {all_tuning_results[best_lstm_exp]['overall']['MAE']:.4f})")

    tuning_summary = {
        "best_vanilla_rnn": {
            "name": best_rnn_exp,
            "metrics": all_tuning_results[best_rnn_exp]["overall"],
            "config": all_tuning_results[best_rnn_exp]["config"],
            "checkpoint": f"tune_{best_rnn_exp}.pt"
        },
        "best_lstm": {
            "name": best_lstm_exp,
            "metrics": all_tuning_results[best_lstm_exp]["overall"],
            "config": all_tuning_results[best_lstm_exp]["config"],
            "checkpoint": f"tune_{best_lstm_exp}.pt"
        },
        "all_experiments": all_tuning_results
    }

    out_file = os.path.join(base_config["results_dir"], "validation_tuning_results.json")
    save_json(tuning_summary, out_file)
    print(f"\nFull validation tuning results saved to {out_file}")


if __name__ == "__main__":
    main()
