# Results

Summary tables behind every number in the report. The test period for all metrics is 1998–2008.

| File | Contents | Made by |
|---|---|---|
| `eligible_catchments.csv` | The 128 selected catchments, with attributes, flow completeness and regime label | `05_select_catchments.py` |
| `basins_train.txt`, `basins_test_128.txt` | LSTM training catchments (663) and test catchments (128) | `08_prepare_lstm_data.py` |
| `gr_parameters_optionA.csv`, `gr_parameters_optionB.csv` | Calibrated parameters (X1–X8) and train/test NSE and KGE for GR4J, GR6J, CemaNeige-GR4J and CemaNeige-GR6J, one row per catchment and model | `gr_run.R` (merged from the per-catchment files) |
| `gr_metrics_optionA.csv` | All metrics for GR4J and GR6J, training and test periods | `07_gr_metrics_optionA.py` |
| `metrics_test_all_models.csv` | Option A: all metrics for every model, the 8 LSTM seeds and the LSTM ensemble | `10_evaluate_optionA.py`, `12_snow_experiment.py` |
| `metrics_test_optionB.csv` | The same for Option B inputs | `14_evaluate_optionB.py` |
| `paired_tests.csv` | Paired Wilcoxon tests (Holm-corrected) between GR4J, GR6J and the LSTM, Option A | `11_compare_models_optionA.py` |
| `sensitivity_A_vs_B.csv` | Change in NSE from Option A to Option B for each model | `15_sensitivity_A_vs_B.py` |

**Metric columns** follow neuralhydrology's names: `NSE`, `KGE`, `Alpha-NSE`, `Beta-KGE`, `Beta-NSE`, `Pearson-r`, `FHV`, `FMS`, `FLV`, `Peak-Timing`, `Peak-MAPE`, `MSE`, `RMSE`.

**Not included** (too large for GitHub, and they can be regenerated with the scripts): daily simulated flows for each catchment, per-catchment parameter files, and the trained LSTM weights.
