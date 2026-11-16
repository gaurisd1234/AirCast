"""
Generate comprehensive publication-quality visualization plots for AirCast.
Saves all figures to results/plots/
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_config


def main():
    config = load_config("configs/default_config.yaml")
    results_dir = config["results_dir"]
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_json = os.path.join(results_dir, "final_test_metrics.json")
    if not os.path.exists(metrics_json):
        print("Run evaluate_test.py first to produce test metrics.")
        return

    test_metrics = load_config(metrics_json)
    y_true = np.load(os.path.join(results_dir, "test_y_true.npy"))

    models_to_plot = [
        ("Persistence", "#7f7f7f"),
        ("MovingAverage_24h", "#bcbd22"),
        ("Ridge_alpha10", "#1f77b4"),
        ("Vanilla_RNN_L24_Tuned", "#ff7f0e"),
        ("LSTM_L24_Tuned", "#2ca02c")
    ]

    predictions = {}
    for model_name, _ in models_to_plot:
        pred_file = os.path.join(results_dir, f"test_preds_{model_name}.npy")
        if os.path.exists(pred_file):
            predictions[model_name] = np.load(pred_file)

    # -------------------------------------------------------------
    # Plot 1: Per-Horizon MAE Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = ["t+1", "t+6", "t+12", "t+24"]
    x = np.arange(len(horizons))
    width = 0.15

    for idx, (m_name, color) in enumerate(models_to_plot):
        if m_name in test_metrics:
            maes = [test_metrics[m_name]["per_horizon"][h]["MAE"] for h in horizons]
            ax.bar(x + (idx - 2) * width, maes, width, label=m_name.replace("_", " "), color=color, alpha=0.85)

    ax.set_ylabel("Mean Absolute Error (μg/m³)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Horizon Forecasting Error Comparison on Held-Out Test Set", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "per_horizon_mae_comparison.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[Plot 1] Saved: {p1}")

    # -------------------------------------------------------------
    # Plot 2: Sample Forecast Trajectories (24-Hour Multi-Step)
    # -------------------------------------------------------------
    sample_indices = [150, 600, 1200]  # Representative episodes in test set
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    horizon_steps = np.arange(1, 25)

    for i, s_idx in enumerate(sample_indices):
        ax = axes[i]
        ax.plot(horizon_steps, y_true[s_idx], "k-o", linewidth=2.5, label="Ground Truth PM2.5", zorder=10)
        for m_name, color in models_to_plot:
            if m_name in predictions:
                ax.plot(horizon_steps, predictions[m_name][s_idx], "--", linewidth=1.8, label=m_name.replace("_", " "), color=color)

        ax.set_ylabel("PM2.5 (μg/m³)", fontsize=11)
        ax.set_title(f"Episode {i+1} (Test Sample Index #{s_idx}): 24-Hour Trajectory Forecast", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, ncol=3, fontsize=9)
        if i == 2:
            ax.set_xlabel("Future Forecast Step (Hours Ahead: t+1 to t+24)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    p2 = os.path.join(plots_dir, "forecast_trajectories_sample.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[Plot 2] Saved: {p2}")

    # -------------------------------------------------------------
    # Plot 3: Error Residual Distributions (y_true - y_pred)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    comp_models = [
        ("Persistence", "#7f7f7f"),
        ("Vanilla_RNN_L24_Tuned", "#ff7f0e"),
        ("LSTM_L24_Tuned", "#2ca02c")
    ]

    for i, (m_name, color) in enumerate(comp_models):
        ax = axes[i]
        if m_name in predictions:
            residuals = (y_true - predictions[m_name]).flatten()
            mean_err = np.mean(residuals)
            std_err = np.std(residuals)
            ax.hist(residuals, bins=60, range=(-150, 150), color=color, alpha=0.75, edgecolor="black", density=True)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
            ax.axvline(mean_err, color="blue", linestyle="-", linewidth=1.5, label=f"Mean: {mean_err:.2f}")
            ax.set_title(f"{m_name.replace('_', ' ')}\n(Std: {std_err:.2f} μg/m³)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Residual (True - Pred) [μg/m³]", fontsize=10)
            if i == 0:
                ax.set_ylabel("Probability Density", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Residual Error Distributions on Held-Out Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "residual_distributions.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"[Plot 3] Saved: {p3}")

    # -------------------------------------------------------------
    # Plot 4: Training Convergence Curves (from saved tuning history)
    # -------------------------------------------------------------
    tuning_file = os.path.join(results_dir, "validation_tuning_results.json")
    if os.path.exists(tuning_file):
        tuning_data = load_config(tuning_file)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        rnn_hist = tuning_data["all_experiments"]["Vanilla_RNN_ConfigC_Lookback24h"]["history"]
        lstm_hist = tuning_data["all_experiments"]["LSTM_ConfigC_Lookback24h"]["history"]

        # Vanilla RNN plot
        ax = axes[0]
        epochs_rnn = range(1, len(rnn_hist["train_loss"]) + 1)
        ax.plot(epochs_rnn, rnn_hist["train_loss"], "b-o", label="Train MSE Loss")
        ax.plot(epochs_rnn, rnn_hist["val_loss"], "r--s", label="Val MSE Loss")
        ax.set_title("Vanilla RNN (L=24h) Convergence", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        # LSTM plot
        ax = axes[1]
        epochs_lstm = range(1, len(lstm_hist["train_loss"]) + 1)
        ax.plot(epochs_lstm, lstm_hist["train_loss"], "b-o", label="Train MSE Loss")
        ax.plot(epochs_lstm, lstm_hist["val_loss"], "r--s", label="Val MSE Loss")
        ax.set_title("LSTM (L=24h) Convergence", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("Training and Validation Loss Dynamics (Scaled MSE)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p4 = os.path.join(plots_dir, "training_loss_curves.png")
        plt.savefig(p4, dpi=200)
        plt.close()
        print(f"[Plot 4] Saved: {p4}")

    print("\nAll visualization plots generated successfully in results/plots/")


if __name__ == "__main__":
    main()
