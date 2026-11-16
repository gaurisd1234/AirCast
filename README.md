# AirCast: Multi-Horizon Air Quality Forecasting in PyTorch

AirCast is a strictly leak-free, multi-horizon air quality forecasting benchmark in PyTorch. It forecasts fine particulate matter ($\text{PM}_{2.5}$) concentrations up to 24 hours into the future using 5 years of hourly atmospheric observations (2010–2014, $N = 43,800$ hours) from the Beijing benchmark dataset.

The system rigorously benchmarks classical statistical heuristics (**Persistence**, **Seasonal Naive 24h & 168h**, **Moving Average**, **Multi-Output Ridge Regression**), tree ensembles (**HistGradientBoostingRegressor**), and recurrent deep neural networks (**Vanilla RNN** and **Long Short-Term Memory / LSTM**).

> **Evaluation Honesty Disclosure:**
> The test set was used for the initial evaluation and reused for multi-seed runs and the added baselines with frozen configurations; no hyperparameter was changed after viewing test results.

---

## 1. Official Test Set Benchmark

> **Protocol & Evaluation Notes:**
> - **Validation-Selected Headline Models:** The headline neural network configurations (**Vanilla RNN L=24 Tuned** and **LSTM L=24 Tuned**) and the GBDT model were tuned and selected strictly based on **Validation** performance. The $L=72$ Vanilla RNN is reported solely as a clearly labeled ablation row (**Vanilla RNN Ablation: longer lookback L=72**).
> - **Evaluation Population:** Evaluated exclusively on the **5,839 test sliding windows whose target values are 100% real ground-truth observations** (excluding 636 windows containing forward-filled target values).
> - **"Overall" Definition:** "Overall" denotes the unweighted arithmetic average across all 24 individual forecast horizons ($t+1$ through $t+24$). Per-horizon MAEs for all 24 horizons are preserved in [results/per_horizon_all.csv](results/per_horizon_all.csv).
> - **No Test-Score Bolding or Ranking:** In accordance with rigorous scientific reporting standards, models are presented in architectural order without bolding or ranking by test performance.

### Overall Benchmark Metrics (Held-Out Test: April – December 2014, $N = 5,839$ Real Observation Windows)

| Model (Family / Category) | Overall MAE ($\mu\text{g/m}^3$) | Overall RMSE ($\mu\text{g/m}^3$) | Overall $R^2$ | Overall sMAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| Persistence (Naive Baseline) | 44.5884 | 69.4731 | 0.2143 | 56.42% |
| Seasonal Naive (24h Baseline) | 59.7027 | 87.5903 | -0.2490 | 71.99% |
| Seasonal Naive (168h Baseline) | 79.7635 | 112.1086 | -1.0460 | 88.72% |
| Moving Average (24h Baseline) | 54.1790 | 77.8615 | 0.0131 | 65.47% |
| Ridge Regression ($\alpha=10.0$) | 41.8042 | 60.0270 | 0.4134 | 55.01% |
| HistGradientBoosting (Val-Tuned) | 39.7841 | 58.7397 | 0.4383 | 50.52% |
| Vanilla RNN (L=24, Val-Tuned Headline, Seed 42) | 40.1705 | 58.9300 | 0.4347 | 51.24% |
| LSTM (L=24, Val-Tuned Headline, Seed 42) | 40.8567 | 59.1379 | 0.4307 | 52.30% |
| Vanilla RNN (Ablation: longer lookback L=72) | 39.9202 | 57.8648 | 0.4549 | 51.83% |

---

### Per-Horizon MAE Breakdown ($\mu\text{g/m}^3$) for Focal Horizons

| Model (Family / Category) | $t+1$ hour | $t+6$ hours | $t+12$ hours | $t+24$ hours |
| :--- | :---: | :---: | :---: | :---: |
| Persistence (Naive Baseline) | 11.1192 | 35.5110 | 48.7571 | 59.0966 |
| Seasonal Naive (24h Baseline) | 60.0783 | 59.9673 | 59.7174 | 59.0966 |
| Seasonal Naive (168h Baseline) | 80.0257 | 79.9125 | 79.7976 | 79.5076 |
| Moving Average (24h Baseline) | 38.5089 | 47.4134 | 54.6775 | 65.4971 |
| Ridge Regression ($\alpha=10.0$) | 11.3804 | 33.6048 | 45.0959 | 55.3254 |
| HistGradientBoosting (Val-Tuned) | 10.7834 | 31.3449 | 42.4792 | 53.3961 |
| Vanilla RNN (L=24, Val-Tuned Headline) | 12.0478 | 32.3587 | 43.0546 | 52.9862 |
| LSTM (L=24, Val-Tuned Headline) | 21.1678 | 34.1671 | 42.4470 | 52.4101 |
| Vanilla RNN (Ablation: longer lookback L=72) | 13.9881 | 32.0653 | 42.6672 | 52.1592 |

*Full per-horizon errors across all 24 horizons are available in [results/per_horizon_all.csv](results/per_horizon_all.csv).*

---

## 2. Multi-Seed Retraining (Mean ± Standard Deviation)

To quantify initialization and mini-batch shuffle variance, the frozen validation-selected Vanilla RNN and LSTM architectures were retrained across **5 fixed seeds** (`42, 43, 44, 45, 46`) without changing any hyperparameters.

### Overall Performance Across 5 Seeds ($N = 5,839$ Real Observation Windows)

| Model | Overall MAE ($\text{mean} \pm \text{std}$) | Overall RMSE ($\text{mean} \pm \text{std}$) | Overall sMAPE ($\text{mean} \pm \text{std}$) |
| :--- | :---: | :---: | :---: |
| **Vanilla RNN (L=24)** | 40.0283 ± 0.6968 | 58.4393 ± 0.4773 | 53.26% ± 1.73% |
| **LSTM (L=24)** | 40.6652 ± 0.3218 | 59.3191 ± 0.6705 | 52.24% ± 0.48% |

### Per-Horizon MAE Breakdown Across 5 Seeds ($\mu\text{g/m}^3$)

| Model | $t+1$ hour | $t+6$ hours | $t+12$ hours | $t+24$ hours |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla RNN (L=24)** | 14.98 ± 2.27 | 32.01 ± 0.63 | 42.31 ± 0.85 | 52.84 ± 1.12 |
| **LSTM (L=24)** | 18.65 ± 1.74 | 32.89 ± 0.76 | 42.48 ± 0.32 | 53.11 ± 0.66 |

*Key finding:* LSTM exhibits lower variance across random seeds ($\sigma_{\text{MAE}} = 0.32$ vs $0.70\,\mu\text{g/m}^3$), indicating smoother gradient descent, but Vanilla RNN achieves a slightly lower mean error ($40.03$ vs $40.67\,\mu\text{g/m}^3$).

---

## 3. Block Bootstrap 95% Confidence Intervals (24-Hour Blocks, 1,000 Resamples)

Because time-series forecasting errors violate independence assumptions due to serial correlation, statistical significance was assessed via a **moving block bootstrap** (24-hour block length, 1,000 resamples).

$$\Delta \text{MAE} = \text{MAE}_A - \text{MAE}_B$$

| Comparison | Mean Difference ($\mu\text{g/m}^3$) | 95% CI Lower | 95% CI Upper | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM minus RNN (Seed 42)** | +0.6862 | +0.0280 | +1.2957 | **YES ($p < 0.05$)** |
| **RNN minus Ridge (Seed 42)** | -1.6337 | -2.3407 | -0.9630 | **YES ($p < 0.05$)** |
| **RNN minus Gradient Boosting (Seed 42)** | +0.3864 | -0.5551 | +1.3282 | **NO (crosses 0)** |
| **RNN minus Persistence (Seed 42)** | -4.4179 | -6.3119 | -2.6419 | **YES ($p < 0.05$)** |
| *Ensemble:* LSTM minus RNN (5-Seed Mean) | +0.6724 | -0.0828 | +1.3576 | **NO (crosses 0)** |
| *Ensemble:* RNN minus Ridge (5-Seed Mean) | -2.5629 | -3.3268 | -1.9142 | **YES ($p < 0.05$)** |
| *Ensemble:* RNN minus HistGBDT (5-Seed Mean) | -0.5429 | -1.5547 | +0.4017 | **NO (crosses 0)** |
| *Ensemble:* RNN minus Persistence (5-Seed Mean) | -5.3471 | -7.3095 | -3.5537 | **YES ($p < 0.05$)** |

---

## 4. Honest Scientific Summary

1. **Which differences are distinguishable from noise?**
   - **RNN vs. Ridge:** **Yes ($p < 0.05$).** Both single-seed and multi-seed CIs exclude zero ($[-2.34, -0.96]$ and $[-3.33, -1.91]$). Non-linear recurrent models yield a genuine, statistically significant improvement over regularized linear autoregression.
   - **RNN vs. Persistence:** **Yes ($p < 0.05$).** CIs exclude zero ($[-6.31, -2.64]$). Recurrent networks are decisively superior to naive persistence overall.
   - **RNN vs. Gradient Boosting:** **No.** The 95% CI for the difference crosses zero in both single-seed ($[-0.56, +1.33]$) and 5-seed ensemble tests ($[-1.55, +0.40]$). The performance difference between Vanilla RNN and HistGradientBoosting is indistinguishable from random sampling noise.
   - **LSTM vs. Vanilla RNN:** **No.** While seed 42 narrowly excluded zero ($[+0.03, +1.30]$), the 5-seed ensemble CI crosses zero ($[-0.08, +1.36]$). The LSTM's gated memory cells offer no statistically significant accuracy benefit over a Vanilla RNN on this 24-hour air quality benchmark.
2. **Do the deep recurrent models beat gradient boosting?**
   - **No.** `HistGradientBoostingRegressor` achieves an overall test MAE of **39.7841 $\mu\text{g/m}^3$**, which is nominally superior to single-seed Vanilla RNN ($40.17\,\mu\text{g/m}^3$) and single-seed LSTM ($40.86\,\mu\text{g/m}^3$), and statistically on par with the 5-seed RNN ensemble ($39.24\,\mu\text{g/m}^3$). Gradient boosting trains substantially faster and achieves competitive accuracy without recurrent backpropagation through time.
3. **At which horizons does persistence still win?**
   - **At $t+1$ hour**, Persistence achieves a test MAE of **11.1192 $\mu\text{g/m}^3$**, outperforming single-seed Vanilla RNN ($12.05\,\mu\text{g/m}^3$) and LSTM ($21.17\,\mu\text{g/m}^3$), as well as their 5-seed means ($14.98$ and $18.65\,\mu\text{g/m}^3$). Only `HistGradientBoosting` narrowly matches/beats Persistence at $t+1$ ($10.78\,\mu\text{g/m}^3$). Persistence wins at $t+1$ due to short-term atmospheric inertia, but degrades rapidly at $t \ge 6$ ($35.51$ at $t+6$, $48.76$ at $t+12$, $59.10\,\mu\text{g/m}^3$ at $t+24$).

---

## 5. Visualizations

### Per-Horizon MAE Comparison
![Per-Horizon MAE Comparison](figures/per_horizon_mae_comparison.png)

### Sample Multi-Step Forecast Trajectories
![Forecast Trajectories](figures/forecast_trajectories_sample.png)

### Error Residual Distributions
![Residual Distributions](figures/residual_distributions.png)

### Training & Validation Loss Dynamics
![Training Loss Curves](figures/training_loss_curves.png)

---

## 6. Test Suite & Verification

The test suite contains **36 automated unit and integration tests** verifying data leakage protections, scaler/imputer isolation, non-overlapping splits, window alignment, real-observation filtering, new baselines, and block bootstrap logic:
```powershell
python -m pytest tests/
```
Output:
```text
tests\test_baselines_metrics.py .....                                    [ 13%]
tests\test_data.py .....                                                 [ 27%]
tests\test_evaluation.py ..                                              [ 33%]
tests\test_leakage_and_alignment.py .....                                [ 47%]
tests\test_lstm.py ....                                                  [ 58%]
tests\test_new_comparisons.py .....                                      [ 72%]
tests\test_tuning.py ..                                                  [ 77%]
tests\test_utils.py ....                                                 [ 88%]
tests\test_vanilla_rnn.py ....                                           [100%]
============================= 36 passed in 13.44s =============================
```
