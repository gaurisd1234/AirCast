"""
AirCast Streamlit Dashboard.
Interactive model and horizon comparison on the held-out test period.
Built with Streamlit native charts (st.line_chart) and pre-generated static figures (st.image)
to guarantee zero dependency on runtime font engines and backend display servers.
Run via: streamlit run app.py
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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

    try:
        df_metrics, df_per_h, y_true, predictions = load_benchmark_data()
    except Exception as e:
        st.error(f"Failed to load benchmark data: {e}")
        return

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Benchmark Summary",
        "📈 Per-Horizon Error Dynamics",
        "🔍 Forecast Trajectory Inspector",
        "🔬 Residuals & Convergence",
        "⚖️ Limitations & Scientific Notes"
    ])

    # -------------------------------------------------------------
    # Tab 1: Benchmark Summary
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Official Test Set Metrics (N = 5,839 Real Observation Windows)")
        st.caption("Note: 'Overall' denotes the unweighted arithmetic average across all 24 forecast horizons.")

        try:
            display_df = df_metrics.copy()
            # Clean column names for formatting
            format_dict = {}
            for col in display_df.columns:
                if "MAE" in col or "RMSE" in col or "R2" in col:
                    format_dict[col] = "{:.4f}"
                elif "sMAPE" in col:
                    format_dict[col] = "{:.2f}%"

            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Could not render metrics table with formatting: {e}. Falling back to unformatted table.")
            st.dataframe(df_metrics, use_container_width=True)

        st.markdown("### 🎲 Multi-Seed Retraining (Mean ± Standard Deviation)")
        st.markdown(
            "Validation-selected Vanilla RNN and LSTM architectures retrained across 5 fixed seeds "
            "(`42, 43, 44, 45, 46`) with no hyperparameter changes:"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vanilla RNN (L=24) Test MAE", "40.03 ± 0.70 µg/m³", "Best Mean")
        with col2:
            st.metric("LSTM (L=24) Test MAE", "40.67 ± 0.32 µg/m³", "Lower Std Dev")

        st.markdown("### 📦 24-Hour Block Bootstrap 95% Confidence Intervals (1,000 Resamples)")
        st.markdown(
            "- **RNN minus Ridge:** Mean Diff = **-1.63 µg/m³** [95% CI: -2.34, -0.96] -> **Significant (p < 0.05)**\n"
            "- **RNN minus Persistence:** Mean Diff = **-4.42 µg/m³** [95% CI: -6.31, -2.64] -> **Significant (p < 0.05)**\n"
            "- **RNN minus HistGBDT:** Mean Diff = **+0.39 µg/m³** [95% CI: -0.56, +1.33] -> **Indistinguishable from noise (crosses 0)**\n"
            "- **LSTM minus RNN:** Mean Diff = **+0.69 µg/m³** [95% CI: +0.03, +1.30] (Seed 42) / [-0.08, +1.36] (Ensemble) -> **No significant advantage for LSTM**"
        )

    # -------------------------------------------------------------
    # Tab 2: Per-Horizon Error Dynamics
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Error Compounding Across 24 Forecast Horizons")
        st.caption(
            "All figures are pre-generated to figures/*.png in scripts/run_all.py for robust display, "
            "complemented by interactive native Streamlit controls."
        )

        # 1. Pre-generated progression line chart
        fig_progression = os.path.join("figures", "per_horizon_mae_progression.png")
        try:
            if os.path.exists(fig_progression):
                st.image(
                    fig_progression,
                    caption="Figure: Per-Horizon MAE Progression across all 24 Forecast Horizons (April – December 2014)",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Could not display pre-generated progression figure: {e}")

        # 2. Interactive line chart using Streamlit native line_chart
        if df_per_h is not None and len(selected_models) > 0:
            st.markdown("#### 🎛️ Interactive Comparison (Filtered by Sidebar Selection)")
            name_mapping = {
                "Persistence (Naive)": "Persistence",
                "Moving Average (24h)": "MovingAverage_24h",
                "Ridge Regression (alpha=10)": "Ridge_alpha10",
                "HistGradientBoosting (Val-Tuned)": "HistGradientBoosting",
                "Vanilla RNN (L=24, Val-Tuned)": "Vanilla_RNN_L24_Tuned",
                "LSTM (L=24, Val-Tuned)": "LSTM_L24_Tuned",
                "Vanilla RNN (Delta L=24, Val-Selected)": "Vanilla_RNN_Delta_L24",
                "LSTM (Delta L=24, Val-Selected)": "LSTM_Delta_L24",
                "Vanilla RNN (Ablation: L=72)": "Vanilla_RNN_L72_Ablation"
            }
            try:
                interactive_df = pd.DataFrame(index=[f"t+{h}" for h in range(1, 25)])
                for label in selected_models:
                    col = name_mapping.get(label)
                    if col and col in df_per_h.columns:
                        interactive_df[label] = df_per_h[col].values

                if not interactive_df.empty:
                    st.line_chart(interactive_df)
            except Exception as e:
                st.warning(f"Could not render interactive line chart: {e}")

        # 3. Pre-generated focal bar chart
        fig_bar = os.path.join("figures", "per_horizon_mae_comparison.png")
        try:
            if os.path.exists(fig_bar):
                st.image(
                    fig_bar,
                    caption="Figure: Multi-Horizon Focal Comparison Bar Chart (t+1, t+6, t+12, t+24)",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Could not display focal comparison bar chart: {e}")

    # -------------------------------------------------------------
    # Tab 3: Forecast Trajectory Inspector (Native Streamlit Chart)
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Multi-Step Forecast Trajectory vs. Ground Truth")
        st.caption(
            "Rendered via Streamlit native charts (st.line_chart) for smooth interactivity, tooltips, and font-independence."
        )

        max_sample = len(y_true) - 1
        sample_idx = st.slider("Select Test Window Index", min_value=0, max_value=max_sample, value=150, step=25)

        try:
            chart_data = {"Ground Truth PM2.5": y_true[sample_idx]}
            for label in selected_models:
                if label in predictions:
                    chart_data[label] = predictions[label][sample_idx]

            trajectory_df = pd.DataFrame(chart_data, index=[f"t+{h}" for h in range(1, 25)])
            st.line_chart(trajectory_df)
            st.caption(f"Displaying 24-hour forecast trajectory for test window #{sample_idx}. Unit: µg/m³.")
        except Exception as e:
            st.warning(f"Could not render interactive trajectory chart: {e}")

        # Static multi-episode reference image
        fig_sample = os.path.join("figures", "forecast_trajectories_sample.png")
        try:
            if os.path.exists(fig_sample):
                st.markdown("#### 📷 Multi-Episode Benchmark Reference")
                st.image(
                    fig_sample,
                    caption="Pre-generated Benchmark Episodes (Samples #150, #600, #1200)",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Could not display static sample trajectory figure: {e}")

    # -------------------------------------------------------------
    # Tab 4: Residual Distributions & Training Dynamics
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Statistical Residual Distributions & Convergence Dynamics")

        fig_residuals = os.path.join("figures", "residual_distributions.png")
        try:
            if os.path.exists(fig_residuals):
                st.image(
                    fig_residuals,
                    caption="Figure: Residual Error Distributions (True - Predicted) on Held-Out Real Test Observations",
                    use_container_width=True
                )
            else:
                st.info("Residual distribution plot not found. Run `python scripts/generate_plots.py` to generate it.")
        except Exception as e:
            st.warning(f"Could not display residual distribution figure: {e}")

        fig_training = os.path.join("figures", "training_loss_curves.png")
        try:
            if os.path.exists(fig_training):
                st.image(
                    fig_training,
                    caption="Figure: Training and Validation Loss Dynamics (Scaled MSE with Early Stopping)",
                    use_container_width=True
                )
            else:
                st.info("Training loss curve plot not found. Run `python scripts/generate_plots.py` to generate it.")
        except Exception as e:
            st.warning(f"Could not display training loss curves figure: {e}")

    # -------------------------------------------------------------
    # Tab 5: Limitations & Scientific Notes
    # -------------------------------------------------------------
    with tab5:
        st.subheader("Honest Scientific Limitations & Findings")
        st.markdown(
            "1. **Single Monitoring Station:** Data is collected strictly from the US Embassy station in Beijing. "
            "Spatial air quality varies across topography and traffic corridors, so spatial transferability cannot be assessed.\n"
            "2. **Single City / Single Climate Basin:** Beijing sits in a mountain-basin terrain prone to severe thermal inversions "
            "and seasonal sandstorms. Dynamics may not generalize directly to maritime or coastal cities.\n"
            "3. **Missing Data Handling:** 1.34% of test PM2.5 readings were missing and causally forward-filled. To preserve ground truth "
            "integrity, test evaluation was strictly filtered to the 5,839 windows containing 100% un-imputed ground truth observations.\n"
            "4. **Deep Models vs. Gradient Boosting:** HistGradientBoosting achieves 39.78 µg/m³ MAE, which is nominally superior to both "
            "Vanilla RNN (40.17 µg/m³) and LSTM (40.86 µg/m³). Deep sequential models did not beat gradient boosting on this tabular lag dataset.\n"
            "5. **Persistence at Short Horizons:** At horizon t+1, Persistence achieves 11.23 µg/m³ MAE due to atmospheric inertia, beating raw-target deep models "
            "(12.05 to 21.17 µg/m³). Learned models only outperform Persistence at horizons t >= 6.\n"
            "6. **Delta Formulation Advantage:** Predicting the delta from the last observed measurement (target minus last observation) resolved the short-horizon "
            "underfitting problem, bringing t+1 MAE down to 10.97 µg/m³ for Vanilla RNN and 11.04 µg/m³ for LSTM, matching or beating Persistence."
        )


if __name__ == "__main__":
    main()
