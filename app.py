"""
AirCast Streamlit Dashboard.
Interactive model and horizon comparison on the held-out test period.
Run via: streamlit run app.py
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Streamlit page configuration
st.set_page_config(
    page_title="AirCast: Air Quality Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_benchmark_data():
    """Load precomputed test metrics, predictions, and ground truth."""
    metrics_csv = os.path.join("results", "final_test_metrics.csv")
    per_h_csv = os.path.join("results", "per_horizon_all.csv")
    y_true_path = os.path.join("results", "test_y_true.npy")

    df_metrics = pd.read_csv(metrics_csv) if os.path.exists(metrics_csv) else None
    df_per_h = pd.read_csv(per_h_csv) if os.path.exists(per_h_csv) else None
    y_true = np.load(y_true_path) if os.path.exists(y_true_path) else None

    # Load predictions dictionary
    predictions = {}
    model_keys = [
        ("Persistence", "Persistence (Naive)"),
        ("SeasonalNaive_24h", "Seasonal Naive (24h)"),
        ("MovingAverage_24h", "Moving Average (24h)"),
        ("Ridge_alpha10", "Ridge Regression (alpha=10)"),
        ("HistGradientBoosting", "HistGradientBoosting (Val-Tuned)"),
        ("Vanilla_RNN_L24_Tuned", "Vanilla RNN (L=24, Val-Tuned)"),
        ("LSTM_L24_Tuned", "LSTM (L=24, Val-Tuned)"),
        ("Vanilla_RNN_Delta_L24", "Vanilla RNN (Delta L=24, Val-Selected)"),
        ("LSTM_Delta_L24", "LSTM (Delta L=24, Val-Selected)"),
        ("Vanilla_RNN_L72_Ablation", "Vanilla RNN (Ablation: L=72)")
    ]

    for key, label in model_keys:
        p_path = os.path.join("results", f"test_preds_{key}.npy")
        if os.path.exists(p_path):
            predictions[label] = np.load(p_path)

    return df_metrics, df_per_h, y_true, predictions


def main():
    st.title("🌫️ AirCast: Multi-Horizon Air Quality Forecasting in PyTorch")
    st.markdown(
        "*Benchmarking Classical Baselines, HistGradientBoosting, Vanilla RNN, and LSTM "
        "on the UCI Beijing PM2.5 Dataset.*"
    )

    st.info(
        "**Evaluation Honesty Disclosure:** The test set was used for the initial evaluation "
        "and reused for multi-seed runs and the added baselines with frozen configurations; "
        "no hyperparameter was changed after viewing test results."
    )

    df_metrics, df_per_h, y_true, predictions = load_benchmark_data()

    if df_metrics is None or y_true is None:
        st.error("Results files not found in results/. Run `python scripts/evaluate_test.py` first.")
        return

    # Sidebar Controls
    st.sidebar.header("Navigation & Model Controls")
    available_models = list(predictions.keys())
    default_selected = [
        "Persistence (Naive)",
        "HistGradientBoosting (Val-Tuned)",
        "Vanilla RNN (L=24, Val-Tuned)",
        "LSTM (L=24, Val-Tuned)"
    ]
    selected_models = st.sidebar.multiselect(
        "Select Models to Display",
        options=available_models,
        default=[m for m in default_selected if m in available_models]
    )

    # Tabs for structured comparison
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Benchmark Summary",
        "📈 Per-Horizon Error Dynamics",
        "🔍 Forecast Trajectory Inspector",
        "⚖️ Limitations & Scientific Notes"
    ])

    # -------------------------------------------------------------
    # Tab 1: Benchmark Summary
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Official Test Set Metrics (N = 5,839 Real Observation Windows)")
        st.caption("Note: 'Overall' denotes the unweighted arithmetic average across all 24 forecast horizons.")

        # Display formatted table
        display_df = df_metrics.copy()
        st.dataframe(
            display_df.style.format({
                "Overall_MAE": "{:.4f}",
                "Overall_RMSE": "{:.4f}",
                "Overall_R2": "{:.4f}",
                "Overall_sMAPE": "{:.2f}%",
                "MAE_t+1": "{:.4f}",
                "MAE_t+6": "{:.4f}",
                "MAE_t+12": "{:.4f}",
                "MAE_t+24": "{:.4f}"
            }),
            use_container_width=True
        )

        st.markdown("### 🎲 Multi-Seed Retraining (Mean ± Standard Deviation)")
        st.markdown(
            "Validation-selected Vanilla RNN and LSTM architectures retrained across 5 fixed seeds "
            "(`42, 43, 44, 45, 46`) with no hyperparameter changes:"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vanilla RNN (L=24) Test MAE", "40.03 ± 0.70 μg/m³", "Best Mean")
        with col2:
            st.metric("LSTM (L=24) Test MAE", "40.67 ± 0.32 μg/m³", "Lower Std Dev")

        st.markdown("### 📦 24-Hour Block Bootstrap 95% Confidence Intervals (1,000 Resamples)")
        st.markdown(
            "- **RNN minus Ridge:** Mean Diff = **-1.63 μg/m³** [95% CI: -2.34, -0.96] -> **Significant ($p < 0.05$)**\n"
            "- **RNN minus Persistence:** Mean Diff = **-4.42 μg/m³** [95% CI: -6.31, -2.64] -> **Significant ($p < 0.05$)**\n"
            "- **RNN minus HistGBDT:** Mean Diff = **+0.39 μg/m³** [95% CI: -0.56, +1.33] -> **Indistinguishable from noise (crosses 0)**\n"
            "- **LSTM minus RNN:** Mean Diff = **+0.69 μg/m³** [95% CI: +0.03, +1.30] (Seed 42) / [-0.08, +1.36] (Ensemble) -> **No significant advantage for LSTM**"
        )

    # -------------------------------------------------------------
    # Tab 2: Per-Horizon Error Dynamics
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Error Compounding Across 24 Forecast Horizons")
        if df_per_h is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            horizon_steps = np.arange(1, 25)

            # Map display labels to per_horizon_all.csv column names
            name_mapping = {
                "Persistence (Naive)": "Persistence",
                "Moving Average (24h)": "MovingAverage_24h",
                "Ridge Regression (alpha=10)": "Ridge_alpha10",
                "HistGradientBoosting (Val-Tuned)": "HistGradientBoosting",
                "Vanilla RNN (L=24, Val-Tuned)": "Vanilla_RNN_L24_Tuned",
                "LSTM (L=24, Val-Tuned)": "LSTM_L24_Tuned",
                "Vanilla RNN (Ablation: L=72)": "Vanilla_RNN_L72_Ablation"
            }

            for label in selected_models:
                col_name = name_mapping.get(label)
                if col_name and col_name in df_per_h.columns:
                    ax.plot(horizon_steps, df_per_h[col_name], marker="o", markersize=3, label=label)

            ax.set_xlabel("Forecast Horizon Step (Hours Ahead: t+1 to t+24)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Mean Absolute Error (μg/m³)", fontsize=11, fontweight="bold")
            ax.set_title("Per-Horizon MAE Progression (April – December 2014)", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=True, fontsize=9)
            st.pyplot(fig)
            plt.close()

    # -------------------------------------------------------------
    # Tab 3: Forecast Trajectory Inspector
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Multi-Step Forecast Trajectory vs. Ground Truth")
        st.caption("Select any test sequence to inspect how models predict 24 hours ahead.")

        max_sample = len(y_true) - 1
        sample_idx = st.slider("Test Sample Index", min_value=0, max_value=max_sample, value=150, step=25)

        fig, ax = plt.subplots(figsize=(11, 5))
        horizon_steps = np.arange(1, 25)

        ax.plot(horizon_steps, y_true[sample_idx], "k-o", linewidth=2.5, label="Ground Truth PM2.5", zorder=10)
        for label in selected_models:
            if label in predictions:
                ax.plot(horizon_steps, predictions[label][sample_idx], "--", linewidth=1.8, label=label)

        ax.set_xlabel("Hours Ahead (t+1 to t+24)", fontsize=11, fontweight="bold")
        ax.set_ylabel("PM2.5 (μg/m³)", fontsize=11, fontweight="bold")
        ax.set_title(f"Test Window #{sample_idx}: 24-Hour Forecast Trajectory", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)
        st.pyplot(fig)
        plt.close()

    # -------------------------------------------------------------
    # Tab 4: Limitations & Scientific Notes
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Honest Scientific Limitations & Findings")
        st.markdown(r"""
        1. **Single Monitoring Station:** Data is collected strictly from the US Embassy station in Beijing. Spatial air quality varies across topography and traffic corridors, so spatial transferability cannot be assessed.
        2. **Single City / Single Climate Basin:** Beijing sits in a mountain-basin terrain prone to severe thermal inversions and seasonal sandstorms. Dynamics may not generalize directly to maritime or coastal cities.
        3. **Missing Data Handling:** 1.34% of test PM2.5 readings were missing and causally forward-filled. To preserve ground truth integrity, test evaluation was strictly filtered to the 5,839 windows containing 100% un-imputed ground truth observations.
        4. **Deep Models vs. Gradient Boosting:** HistGradientBoosting achieves 39.78 μg/m³ MAE, which is nominally superior to both Vanilla RNN (40.17 μg/m³) and LSTM (40.86 μg/m³). Deep sequential models did not beat gradient boosting on this tabular lag dataset.
        5. **Persistence at Short Horizons:** At horizon $t+1$, Persistence achieves 11.12 μg/m³ MAE due to atmospheric inertia, beating deep models (12.05 to 21.17 μg/m³). Learned models only outperform Persistence at horizons $t \ge 6$.
        """)


if __name__ == "__main__":
    main()
