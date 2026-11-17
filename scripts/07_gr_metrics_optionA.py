# =============================================================================
# 07 - Metrics for the GR models, Option A (train and test periods)
#
# Uses neuralhydrology's metric functions, so the LSTM is scored with
# identical code. Saves gr_parameters_optionA.csv and gr_metrics_optionA.csv.
#
# Run:   python 07_gr_metrics_optionA.py
# Needs: 06 (option A)
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 11 - Full evaluation metrics for GR4J and GR6J (train and test periods)
# Uses neuralhydrology's metric functions so the LSTM is later scored with
# identical code (NSE, KGE + components, FHV/FMS/FLV, peak timing)
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wilcoxon
from neuralhydrology.evaluation.metrics import calculate_all_metrics

ROOT = PROJECT_ROOT
GR_DIR = ROOT / "outputs/gr_models"
PERIODS = {"train": ("1988-01-01", "1997-12-31"), "test": ("1998-01-01", "2008-12-31")}

# --- Catchment list with regime labels (from the selection step) -------------
elig = pd.read_csv(ROOT / "outputs/catchment_selection/eligible_catchments.csv")
elig["gauge_id"] = elig["gauge_id"].astype(str)

# --- Step 1: merge calibrated parameters from all saved files ----------------
par = pd.concat([pd.read_csv(f) for f in sorted((GR_DIR / "params").glob("*.csv"))],
                ignore_index=True)
par["gauge_id"] = par["gauge_id"].astype(str)
par = par.drop(columns=["error"], errors="ignore")
print("Calibrated:", par.groupby("model")["gauge_id"].nunique().to_dict())
par.to_csv(GR_DIR / "gr_parameters_optionA.csv", index=False)

# --- Step 2: metrics for every model / catchment / period --------------------
def metrics_for(sim_file, start, end):
    """Read one simulation file and compute all metrics for one period."""
    df = pd.read_csv(sim_file, parse_dates=["date"]).set_index("date").loc[start:end]
    obs = xr.DataArray(df["q_obs"].values, coords={"date": df.index}, dims=["date"])
    sim = xr.DataArray(df["q_sim"].values, coords={"date": df.index}, dims=["date"])
    return calculate_all_metrics(obs, sim, resolution="1D")

rows = []
for f in sorted((GR_DIR / "sim").glob("*.csv")):
    model, gid = f.stem.split("_", 1)
    for period, (start, end) in PERIODS.items():
        try:
            m = metrics_for(f, start, end)
        except Exception as e:                        # keep going if one fails
            m = {"error": str(e)}
        rows.append({"model": model, "gauge_id": gid, "period": period, **m})

met = pd.DataFrame(rows)
met = met.merge(elig[["gauge_id", "gauge_name", "regime", "aridity", "baseflow_index"]],
                on="gauge_id", how="left")
met.to_csv(GR_DIR / "gr_metrics_optionA.csv", index=False)
if "error" in met and met["error"].notna().any():
    print("Metric errors:\n", met.loc[met["error"].notna(), ["model", "gauge_id", "period", "error"]])

# --- Step 3: median of each metric, per model and period ---------------------
metric_cols = [c for c in met.columns if c not in
               ["model", "gauge_id", "period", "gauge_name", "regime", "aridity",
                "baseflow_index", "error"]]
print("\nMedian across catchments:")
print(met.groupby(["period", "model"])[metric_cols].median().round(3).T.to_string())

# --- Step 4: GR4J vs GR6J, catchment by catchment (test period) --------------
test = met[met["period"] == "test"].pivot(index="gauge_id", columns="model",
                                          values=["NSE", "KGE", "FLV", "FHV"])
print("\nTest period, GR6J vs GR4J (paired, same catchments):")
for metric, better in [("NSE", "higher"), ("KGE", "higher"), ("FLV", "closer to 0"),
                       ("FHV", "closer to 0")]:
    a, b = test[(metric, "GR4J")], test[(metric, "GR6J")]
    ok = a.notna() & b.notna()
    if better == "higher":
        wins = (b[ok] > a[ok]).sum()
        p = wilcoxon(b[ok], a[ok]).pvalue
    else:                                             # bias metrics: smaller |bias| wins
        wins = (b[ok].abs() < a[ok].abs()).sum()
        p = wilcoxon(b[ok].abs(), a[ok].abs()).pvalue
    print(f"  {metric}: GR6J better in {wins}/{ok.sum()} catchments "
          f"(Wilcoxon signed-rank p = {p:.3g})")

# --- Step 5: median test NSE and FLV by aridity/BFI regime -------------------
print("\nTest-period medians by regime:")
print(met[met["period"] == "test"].pivot_table(index="regime", columns="model",
      values=["NSE", "FLV"], aggfunc="median").round(3).to_string())

# --- Step 6: weakest catchments (test NSE), to investigate later -------------
print("\nLowest 10 test NSE (GR4J):")
print(met[(met["period"] == "test") & (met["model"] == "GR4J")]
      .nsmallest(10, "NSE")[["gauge_id", "gauge_name", "regime", "NSE", "KGE", "FHV", "FLV"]]
      .round(3).to_string(index=False))
