"""
Unit tests for Streamlit application data loading and component contracts.
"""

import numpy as np
import pandas as pd
from app import load_benchmark_data


def test_load_benchmark_data():
    df_metrics, df_per_h, y_true, predictions = load_benchmark_data()

    assert df_metrics is not None
    assert isinstance(df_metrics, pd.DataFrame)
    assert "Overall_MAE" in df_metrics.columns
    assert len(df_metrics) >= 6

    assert df_per_h is not None
    assert isinstance(df_per_h, pd.DataFrame)
    assert len(df_per_h) == 24  # All 24 horizons

    assert y_true is not None
    assert isinstance(y_true, np.ndarray)
    assert y_true.shape == (5839, 24)

    assert "Persistence (Naive)" in predictions
    assert "Vanilla RNN (L=24, Val-Tuned)" in predictions
    assert "LSTM (L=24, Val-Tuned)" in predictions
    assert predictions["Persistence (Naive)"].shape == (5839, 24)
