# Scripts

These scripts reproduce the whole project, in the order they were run. They are taken from the Colab notebook (`notebooks/Code.ipynb`) and made runnable outside Colab:
- Paths come from `config.py`.
- Figures are saved to `outputs/figures/` instead of being displayed.
- Google Drive and `pip` lines are removed. Install the packages from `requirements.txt` instead.

## Setup

1. Install the Python packages in `requirements.txt` (PyTorch with CUDA first) and, in R, `install.packages("airGR")`.
2. Tell the scripts where the project folder is:
   - Windows (PowerShell): `$env:RRB_ROOT = "C:\path\to\rainfall_runoff_benchmark"`
   - Linux or macOS: `export RRB_ROOT=/path/to/rainfall_runoff_benchmark`

   If `RRB_ROOT` is not set, the Google Drive path used in Colab is assumed.
3. Run the scripts from this folder.

## Run order

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `01_download_camels_gb_v2.py` | Downloads CAMELS-GB v2 from the EIDC (needs an EIDC access token) | 1–2 h |
| 2 | `02_check_download.py` | Checks the download is complete | seconds |
| 3 | `03_explore_data.py` | File inventory, attribute tables, example series, decoding of the QC flags | 1 min |
| 4 | `04_benchmark_overview.py` | The 137 benchmark catchments vs all 671 | 1 min |
| 5 | `05_select_catchments.py` | Applies the selection rules, giving 128 catchments with regime labels | 2 min |
| 6 | `06_run_gr_models.py --option A` | Calibrates GR4J, GR6J and both CemaNeige versions (calls `gr_run.R`) | 15–40 min |
| 7 | `07_gr_metrics_optionA.py` | Metrics for the GR models | 2 min |
| 8 | `08_prepare_lstm_data.py` | Converts the data to neuralhydrology's format | 5–15 min |
| 9 | `train_lstm.py --root <folder> --option A` | Trains the 8-seed LSTM ensemble (GPU) | ~9 h |
| 10 | `09_check_lstm_seed.py` *(optional)* | Quick check of one trained seed | 2 min |
| 11 | `10_evaluate_optionA.py` | Evaluates the LSTM ensemble and builds `metrics_test_all_models.csv` | 10 min |
| 12 | `11_compare_models_optionA.py` | Paired tests, regimes, NSE distribution and maps | 1 min |
| 13 | `12_snow_experiment.py` | Effect of the CemaNeige snow module | 2 min |
| 14 | `13_case_study_hydrographs.py` | Case-study figures | 1 min |
| 15 | `06_run_gr_models.py --option B` and `train_lstm.py --root <folder> --option B` | Repeat with the Option B inputs | ~10 h |
| 16 | `14_evaluate_optionB.py` | Evaluates all models under Option B | 10 min |
| 17 | `15_sensitivity_A_vs_B.py` | Compares the two input sets | 1 min |
| 18 | `make_summary_figures.py` | The two summary figures used in the report | seconds |

## Other files

- `config.py`: shared paths and helper functions (attribute loading, figure saving).
- `gr_run.R`: GR calibration in airGR for any model and either input option. Usage: `Rscript gr_run.R <model> <A|B> [project_root]`.
- `legacy/`: the first Option A GR scripts (`gr_calibrate.R`, `gr_cemaneige.R`), kept for the record. `gr_run.R` uses the same protocol and replaces both.

**Note:** `06`, `07`, `09`–`15` and `train_lstm.py` need neuralhydrology, R/airGR or a GPU. Their logic is unchanged from the notebook cells that produced the reported results.
