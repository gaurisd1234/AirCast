"""
Script to prepare and validate AirCast dataset, verifying leak-free splits and window shapes.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from src.utils import load_config, set_seed, save_json
from src.data import (
    download_and_load_raw_data,
    preprocess_data,
    split_chronological,
    fit_scalers_and_transform,
    create_sliding_windows,
    CONTINUOUS_COLUMNS
)


def main():
    config = load_config("configs/default_config.yaml")
    set_seed(config["seed"])

    raw_df = download_and_load_raw_data(config["data"]["raw_url"], config["data"]["raw_path"])
    clean_df = preprocess_data(raw_df)

    # Save processed dataframe for inspectability
    clean_df.to_csv(config["data"]["processed_path"], index=False)
    print(f"Cleaned dataset saved to {config['data']['processed_path']} with shape {clean_df.shape}")

    train_df, val_df, test_df = split_chronological(
        clean_df,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"]
    )

    print("Chronological Split Summary:")
    print(f"  Train: {train_df['datetime'].min()} to {train_df['datetime'].max()} | Rows: {len(train_df)}")
    print(f"  Val:   {val_df['datetime'].min()} to {val_df['datetime'].max()} | Rows: {len(val_df)}")
    print(f"  Test:  {test_df['datetime'].min()} to {test_df['datetime'].max()} | Rows: {len(test_df)}")

    # Check zero temporal overlap
    assert train_df["datetime"].max() < val_df["datetime"].min(), "Leakage detected: train overlaps with val!"
    assert val_df["datetime"].max() < test_df["datetime"].min(), "Leakage detected: val overlaps with test!"

    train_scaled, val_scaled, test_scaled, scaler_info, scaler = fit_scalers_and_transform(
        train_df, val_df, test_df, CONTINUOUS_COLUMNS
    )

    feature_cols = [c for c in train_df.columns if c != "datetime"]
    target_idx = feature_cols.index("pm2.5")

    lookback = config["data"]["lookback_hours"]
    horizon = config["data"]["max_horizon"]

    X_train, Y_train = create_sliding_windows(train_scaled[feature_cols].values, target_idx, lookback, horizon)
    X_val, Y_val = create_sliding_windows(val_scaled[feature_cols].values, target_idx, lookback, horizon)
    X_test, Y_test = create_sliding_windows(test_scaled[feature_cols].values, target_idx, lookback, horizon)

    print("\nWindow Tensor Shapes:")
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")
    print(f"  X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")

    print("\nTarget Scaler Statistics (fitted strictly on TRAIN):")
    print(f"  Target Mean (pm2.5): {scaler_info['target_mean']:.4f} ug/m3")
    print(f"  Target Std  (pm2.5): {scaler_info['target_std']:.4f} ug/m3")
    print(f"  Target Median(pm2.5): {scaler_info['target_median']:.4f} ug/m3")

    # Check for NaNs or Infinities in window arrays
    for name, arr in [("X_train", X_train), ("Y_train", Y_train),
                      ("X_val", X_val), ("Y_val", Y_val),
                      ("X_test", X_test), ("Y_test", Y_test)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"WARNING: {name} contains {nan_count} NaNs and {inf_count} Infs!")
        else:
            print(f"  [OK] {name} contains zero NaNs / Infs.")

    summary = {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "features": feature_cols,
        "num_features": len(feature_cols),
        "train_range": [str(train_df["datetime"].min()), str(train_df["datetime"].max())],
        "val_range": [str(val_df["datetime"].min()), str(val_df["datetime"].max())],
        "test_range": [str(test_df["datetime"].min()), str(test_df["datetime"].max())],
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "lookback_hours": lookback,
        "max_horizon": horizon,
        "scaler_info": scaler_info
    }

    os.makedirs(config["results_dir"], exist_ok=True)
    save_json(summary, os.path.join(config["results_dir"], "data_summary.json"))
    print(f"\nData summary written to {os.path.join(config['results_dir'], 'data_summary.json')}")


if __name__ == "__main__":
    main()
