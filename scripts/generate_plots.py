"""
Generate comprehensive publication-quality visualization plots for AirCast.
Saves all figures to figures/ using repository-relative paths.
Headline models match validation-selected configurations; L=72 is labeled as ablation.
All text labels use plain Unicode (µg/m³) without mathtext ($...$) to ensure full font compatibility.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import load_config


def main():
    config = load_config("configs/default_config.yaml")
    results_dir = config["results_dir"]
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    metrics_json = os.path.join(results_dir, "final_test_metrics.json")
    if not os.path.exists(metrics_json):
        print("Run evaluate_test.py first to produce test metrics.")
        return

    test_data = load_config(metrics_json)
    test_metrics = test_data["models"]
    y_true = np.load(os.path.join(results_dir, "test_y_true.npy"))

    # Headline models selected on validation, plus baselines and ablation row
    models_to_plot = [
        ("Persistence", "Persistence (Naive)", "#7f7f7f"),
        ("MovingAverage_24h", "Moving Average (24h)", "#bcbd22"),
        ("Ridge_alpha10", "Ridge Regression (alpha=10)", "#1f77b4"),
        ("HistGradientBoosting", "HistGradientBoosting (Val-Tuned)", "#17becf"),
        ("Vanilla_RNN_L24_Tuned", "Vanilla RNN (L=24, Val-Tuned)", "#ff7f0e"),
        ("LSTM_L24_Tuned", "LSTM (L=24, Val-Tuned)", "#2ca02c"),
        ("Vanilla_RNN_Delta_L24", "Vanilla RNN (Delta L=24, Val-Selected)", "#d62728"),
        ("LSTM_Delta_L24", "LSTM (Delta L=24, Val-Selected)", "#9467bd"),
        ("Vanilla_RNN_L72_Ablation", "Vanilla RNN (Ablation: L=72)", "#8c564b")
    ]

    predictions = {}
    for model_key, _, _ in models_to_plot:
        pred_file = os.path.join(results_dir, f"test_preds_{model_key}.npy")
        if os.path.exists(pred_file):
            predictions[model_key] = np.load(pred_file)

    # -------------------------------------------------------------
    # Plot 1: Per-Horizon MAE Comparison Bar Chart (Focal Horizons)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    horizons = ["t+1", "t+6", "t+12", "t+24"]
    x = np.arange(len(horizons))
    width = 0.09

    plotted_models = [m for m in models_to_plot if m[0] in test_metrics]
    n_models = len(plotted_models)

    for idx, (m_key, m_label, color) in enumerate(plotted_models):
        maes = [test_metrics[m_key]["per_horizon"][h]["MAE"] for h in horizons]
        offset = (idx - (n_models - 1) / 2.0) * width
        ax.bar(x + offset, maes, width, label=m_label, color=color, alpha=0.85)

    ax.set_ylabel("Mean Absolute Error (µg/m³)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Horizon Forecasting Error Comparison on Held-Out Test Set (Real Observations)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11)
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    for out_dir in [figures_dir, os.path.join(results_dir, "plots")]:
        plt.savefig(os.path.join(out_dir, "per_horizon_mae_comparison.png"), dpi=200)
    plt.close(fig)
    print("[Plot 1] Saved: figures/per_horizon_mae_comparison.png")

    # -------------------------------------------------------------
    # Plot 2: Sample Forecast Trajectories (24-Hour Multi-Step)
    # -------------------------------------------------------------
    sample_indices = [150, 600, 1200]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    horizon_steps = np.arange(1, 25)

    for i, s_idx in enumerate(sample_indices):
        ax = axes[i]
        ax.plot(horizon_steps, y_true[s_idx], "k-o", linewidth=2.5, label="Ground Truth PM2.5", zorder=10)
        for m_key, m_label, color in models_to_plot:
            if m_key in predictions:
                linestyle = ":" if "Ablation" in m_label else "--"
                ax.plot(horizon_steps, predictions[m_key][s_idx], linestyle=linestyle, linewidth=1.8, label=m_label, color=color)

        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
        ax.set_title(f"Episode {i+1} (Sample #{s_idx}): 24-Hour Multi-Horizon Forecast Trajectory", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, ncol=3, fontsize=8)
        if i == 2:
            ax.set_xlabel("Forecast Step Ahead (Hours Ahead: t+1 to t+24)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    for out_dir in [figures_dir, os.path.join(results_dir, "plots")]:
        plt.savefig(os.path.join(out_dir, "forecast_trajectories_sample.png"), dpi=200)
    plt.close(fig)
    print("[Plot 2] Saved: figures/forecast_trajectories_sample.png")

    # -------------------------------------------------------------
    # Plot 3: Error Residual Distributions (y_true - y_pred)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    comp_models = [
        ("Persistence", "Persistence (Naive)", "#7f7f7f"),
        ("Vanilla_RNN_L24_Tuned", "Vanilla RNN (L=24, Val-Tuned)", "#ff7f0e"),
        ("LSTM_L24_Tuned", "LSTM (L=24, Val-Tuned)", "#2ca02c")
    ]

    for i, (m_key, m_label, color) in enumerate(comp_models):
        ax = axes[i]
        if m_key in predictions:
            residuals = (y_true - predictions[m_key]).flatten()
            mean_err = np.mean(residuals)
            std_err = np.std(residuals)
            ax.hist(residuals, bins=60, range=(-150, 150), color=color, alpha=0.75, edgecolor="black", density=True)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
            ax.axvline(mean_err, color="blue", linestyle="-", linewidth=1.5, label=f"Mean: {mean_err:.2f}")
            ax.set_title(f"{m_label}\n(Std: {std_err:.2f} µg/m³)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Residual (True - Pred) [µg/m³]", fontsize=10)
            if i == 0:
                ax.set_ylabel("Probability Density", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Residual Error Distributions on Real Observations (Held-Out Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for out_dir in [figures_dir, os.path.join(results_dir, "plots")]:
        plt.savefig(os.path.join(out_dir, "residual_distributions.png"), dpi=200)
    plt.close(fig)
    print("[Plot 3] Saved: figures/residual_distributions.png")

    # -------------------------------------------------------------
    # Plot 4: Training Convergence Curves (from saved tuning history)
    # -------------------------------------------------------------
    tuning_file = os.path.join(results_dir, "validation_tuning_results.json")
    if os.path.exists(tuning_file):
        tuning_data = load_config(tuning_file)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        rnn_hist = tuning_data["all_experiments"]["Vanilla_RNN_ConfigC_Lookback24h"]["history"]
        lstm_hist = tuning_data["all_experiments"]["LSTM_ConfigC_Lookback24h"]["history"]

        ax = axes[0]
        epochs_rnn = range(1, len(rnn_hist["train_loss"]) + 1)
        ax.plot(epochs_rnn, rnn_hist["train_loss"], "b-o", label="Train MSE Loss")
        ax.plot(epochs_rnn, rnn_hist["val_loss"], "r--s", label="Val MSE Loss")
        ax.set_title("Vanilla RNN (L=24h, Val-Tuned) Convergence", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        ax = axes[1]
        epochs_lstm = range(1, len(lstm_hist["train_loss"]) + 1)
        ax.plot(epochs_lstm, lstm_hist["train_loss"], "b-o", label="Train MSE Loss")
        ax.plot(epochs_lstm, lstm_hist["val_loss"], "r--s", label="Val MSE Loss")
        ax.set_title("LSTM (L=24h, Val-Tuned) Convergence", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle("Training and Validation Loss Dynamics (Scaled MSE)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        for out_dir in [figures_dir, os.path.join(results_dir, "plots")]:
            plt.savefig(os.path.join(out_dir, "training_loss_curves.png"), dpi=200)
        plt.close(fig)
        print("[Plot 4] Saved: figures/training_loss_curves.png")

    # -------------------------------------------------------------
    # Plot 5: Per-Horizon Progression (All 24 Horizons Step-by-Step)
    # -------------------------------------------------------------
    per_h_csv = os.path.join(results_dir, "per_horizon_all.csv")
    if os.path.exists(per_h_csv):
        df_per_h = pd.read_csv(per_h_csv)
        fig, ax = plt.subplots(figsize=(12, 6))
        horizon_steps = np.arange(1, 25)

        line_configs = [
            ("Persistence", "Persistence (Naive)", "#7f7f7f", "-"),
            ("MovingAverage_24h", "Moving Average (24h)", "#bcbd22", ":"),
            ("Ridge_alpha10", "Ridge Regression (alpha=10)", "#1f77b4", "--"),
            ("HistGradientBoosting", "HistGradientBoosting (Val-Tuned)", "#17becf", "-."),
            ("Vanilla_RNN_L24_Tuned", "Vanilla RNN (L=24, Val-Tuned)", "#ff7f0e", "-"),
            ("LSTM_L24_Tuned", "LSTM (L=24, Val-Tuned)", "#2ca02c", "-"),
            ("Vanilla_RNN_Delta_L24", "Vanilla RNN (Delta L=24, Val-Selected)", "#d62728", "-"),
            ("LSTM_Delta_L24", "LSTM (Delta L=24, Val-Selected)", "#9467bd", "-"),
            ("Vanilla_RNN_L72_Ablation", "Vanilla RNN (Ablation: L=72)", "#8c564b", ":")
        ]

        for col, label, color, ls in line_configs:
            if col in df_per_h.columns:
                ax.plot(horizon_steps, df_per_h[col], marker="o", markersize=3, label=label, color=color, linestyle=ls)

        ax.set_xlabel("Forecast Horizon Step (Hours Ahead: t+1 to t+24)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Absolute Error (µg/m³)", fontsize=11, fontweight="bold")
        ax.set_title("Per-Horizon MAE Progression Across 24 Forecast Steps (April – December 2014)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")
        plt.tight_layout()
        for out_dir in [figures_dir, os.path.join(results_dir, "plots")]:
            plt.savefig(os.path.join(out_dir, "per_horizon_mae_progression.png"), dpi=200)
        plt.close(fig)
        print("[Plot 5] Saved: figures/per_horizon_mae_progression.png")

    print("\nAll visualization plots generated successfully in figures/ and results/plots/")


if __name__ == "__main__":
    main()
