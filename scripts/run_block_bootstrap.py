"""
Moving Block Bootstrap (24-hour blocks, 1,000 resamples) for multi-horizon air quality forecasting.
Computes 95% Confidence Intervals for pairwise MAE differences between models.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple
import numpy as np
from src.utils import load_config, set_seed, save_json


def compute_block_bootstrap_ci(
    diff_per_sample: np.ndarray,
    block_size: int = 24,
    n_resamples: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute 95% confidence interval for mean difference using moving block bootstrap.
    
    Why block bootstrap: In time series, errors of adjacent windows exhibit temporal autocorrelation.
    Standard i.i.d. bootstrap underestimates variance and yields artificially narrow intervals.
    Resampling contiguous 24-hour blocks preserves daily temporal dependence structures.
    """
    rng = np.random.default_rng(seed)
    n = len(diff_per_sample)
    n_blocks = int(np.ceil(n / block_size))

    # Pre-compute valid starting indices for contiguous blocks of length block_size
    max_start = n - block_size
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        # Sample starting indices of blocks uniformly with replacement
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        # Construct resampled index array
        resampled_indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        bootstrap_means[b] = np.mean(diff_per_sample[resampled_indices])

    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    mean_diff = float(np.mean(diff_per_sample))

    return ci_lower, ci_upper, mean_diff


def main():
    config = load_config("configs/default_config.yaml")
    results_dir = config["results_dir"]
    set_seed(config["seed"])

    print("Running 24-Hour Moving Block Bootstrap (1,000 resamples)...")
    y_true = np.load(os.path.join(results_dir, "test_y_true.npy"))

    # Load test predictions
    preds_rnn = np.load(os.path.join(results_dir, "test_preds_Vanilla_RNN_L24_Tuned.npy"))
    preds_lstm = np.load(os.path.join(results_dir, "test_preds_LSTM_L24_Tuned.npy"))
    preds_ridge = np.load(os.path.join(results_dir, "test_preds_Ridge_alpha10.npy"))
    preds_persistence = np.load(os.path.join(results_dir, "test_preds_Persistence.npy"))

    gbdt_path = os.path.join(results_dir, "test_preds_HistGradientBoosting.npy")
    if not os.path.exists(gbdt_path):
        print(f"Error: {gbdt_path} not found. Run tune_and_train_new_baselines.py first.")
        return
    preds_gbdt = np.load(gbdt_path)

    # Compute sample-level MAE (mean absolute error averaged over all 24 horizons for each test window)
    mae_rnn = np.mean(np.abs(y_true - preds_rnn), axis=1)
    mae_lstm = np.mean(np.abs(y_true - preds_lstm), axis=1)
    mae_ridge = np.mean(np.abs(y_true - preds_ridge), axis=1)
    mae_gbdt = np.mean(np.abs(y_true - preds_gbdt), axis=1)
    mae_pers = np.mean(np.abs(y_true - preds_persistence), axis=1)

    comparisons_single_seed = [
        ("LSTM minus RNN (Seed 42)", mae_lstm - mae_rnn),
        ("RNN minus Ridge (Seed 42)", mae_rnn - mae_ridge),
        ("RNN minus Gradient Boosting (Seed 42)", mae_rnn - mae_gbdt),
        ("RNN minus Persistence (Seed 42)", mae_rnn - mae_pers)
    ]

    # Also compute for 5-seed ensemble mean if available
    ens_rnn_path = os.path.join(results_dir, "test_preds_vanilla_rnn_5seed_mean.npy")
    ens_lstm_path = os.path.join(results_dir, "test_preds_lstm_5seed_mean.npy")
    comparisons_5seed = []
    if os.path.exists(ens_rnn_path) and os.path.exists(ens_lstm_path):
        preds_rnn_ens = np.load(ens_rnn_path)
        preds_lstm_ens = np.load(ens_lstm_path)
        mae_rnn_ens = np.mean(np.abs(y_true - preds_rnn_ens), axis=1)
        mae_lstm_ens = np.mean(np.abs(y_true - preds_lstm_ens), axis=1)
        comparisons_5seed = [
            ("LSTM minus RNN (5-Seed Ensemble)", mae_lstm_ens - mae_rnn_ens),
            ("RNN minus Ridge (5-Seed Ensemble)", mae_rnn_ens - mae_ridge),
            ("RNN minus Gradient Boosting (5-Seed Ensemble)", mae_rnn_ens - mae_gbdt),
            ("RNN minus Persistence (5-Seed Ensemble)", mae_rnn_ens - mae_pers)
        ]

    bootstrap_results = {}

    print("\n" + "=" * 90)
    print("BLOCK BOOTSTRAP 95% CONFIDENCE INTERVALS (24-HOUR BLOCKS, 1,000 RESAMPLES)")
    print("=" * 90)
    print(f"{'Comparison':<45} | {'Mean Diff':<12} | {'95% CI Lower':<14} | {'95% CI Upper':<14} | {'Significant?':<12}")
    print("-" * 90)

    for group_name, comp_list in [("Single Seed (42)", comparisons_single_seed), ("5-Seed Ensemble", comparisons_5seed)]:
        if not comp_list:
            continue
        print(f"--- {group_name} ---")
        for name, diff in comp_list:
            ci_low, ci_high, mean_d = compute_block_bootstrap_ci(diff, block_size=24, n_resamples=1000, seed=config["seed"])
            is_sig = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
            sig_str = "YES (p < 0.05)" if is_sig else "NO (crosses 0)"

            bootstrap_results[name] = {
                "mean_diff": mean_d,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "statistically_significant": is_sig
            }
            sign = "+" if mean_d >= 0 else ""
            print(f"{name:<45} | {sign}{mean_d:<11.4f} | {ci_low:<14.4f} | {ci_high:<14.4f} | {sig_str}")

    print("=" * 90)

    out_file = os.path.join(results_dir, "bootstrap_results.json")
    save_json(bootstrap_results, out_file)
    print(f"\nBootstrap results saved to {out_file}")


if __name__ == "__main__":
    main()
