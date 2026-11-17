# =============================================================================
# 12 - Snow experiment: effect of the CemaNeige snow module
#
# Scores CemaNeige-GR4J/-GR6J (Option A), adds them to the comparison
# table, and tests whether the snow module closes the gap to the LSTM.
#
# Run:   python 12_snow_experiment.py
# Needs: 06 (option A, CemaNeige models), 10
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 20 - Effect of the CemaNeige snow module, and whether it closes the gap
# to the LSTM. Same metric function (neuralhydrology) and test period as before
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
from neuralhydrology.evaluation.metrics import calculate_all_metrics

ROOT = PROJECT_ROOT
GR_DIR, CMP_DIR = ROOT / "outputs/gr_models", ROOT / "outputs/comparison"
TEST = ("1998-01-01", "2008-12-31")

# --- Step 1: test-period metrics for the two CemaNeige models ----------------
rows = []
for f in sorted((GR_DIR / "sim").glob("CemaNeige*.csv")):
    model, gid = f.stem.split("_", 1)
    df = pd.read_csv(f, parse_dates=["date"]).set_index("date").loc[TEST[0]:TEST[1]]
    obs = xr.DataArray(df["q_obs"].values, coords={"date": df.index}, dims=["date"])
    sim = xr.DataArray(df["q_sim"].values, coords={"date": df.index}, dims=["date"])
    rows.append({"model": model, "gauge_id": gid, **calculate_all_metrics(obs, sim, resolution="1D")})
cn = pd.DataFrame(rows)
print("CemaNeige catchments evaluated:", cn.groupby("model")["gauge_id"].nunique().to_dict())

# --- Step 2: add to the full comparison table --------------------------------
allm = pd.read_csv(CMP_DIR / "metrics_test_all_models.csv", dtype={"gauge_id": str})
allm = allm[~allm["model"].str.startswith("CemaNeige")]            # avoid duplicates on rerun
info = allm.drop_duplicates("gauge_id")[["gauge_id", "gauge_name", "regime", "aridity", "baseflow_index"]]
allm = pd.concat([allm, cn.merge(info, on="gauge_id", how="left")], ignore_index=True)
allm.to_csv(CMP_DIR / "metrics_test_all_models.csv", index=False)

MODELS = ["GR4J", "CemaNeigeGR4J", "GR6J", "CemaNeigeGR6J", "LSTM_ensemble"]
main = allm[allm["model"].isin(MODELS)]
cols = ["NSE", "KGE", "Pearson-r", "FHV", "FMS", "FLV", "Peak-Timing"]
print("\nTest-period medians (1998-2008):")
print(main.groupby("model")[cols].median().reindex(MODELS).round(3).T.to_string())

# --- Step 3: paired NSE tests (Holm-corrected) -------------------------------
nse = main.pivot(index="gauge_id", columns="model", values="NSE")
pairs = [("CemaNeigeGR4J", "GR4J"), ("CemaNeigeGR6J", "GR6J"),
         ("LSTM_ensemble", "CemaNeigeGR4J"), ("LSTM_ensemble", "CemaNeigeGR6J")]
tests = []
for a, b in pairs:
    gain = (nse[a] - nse[b]).dropna()
    tests.append({"comparison": f"{a} vs {b}", "a_better_in": f"{(gain > 0).sum()}/{len(gain)}",
                  "median_gain": gain.median(), "p_raw": wilcoxon(gain[gain != 0]).pvalue})
tests = pd.DataFrame(tests)
tests["p_holm"] = multipletests(tests["p_raw"], method="holm")[1]
print("\nPaired NSE comparisons (median_gain > 0 favours the first model):")
print(tests.round(4).to_string(index=False))

# --- Step 4: is the improvement linked to snow? ------------------------------
clim = pd.read_csv(ROOT / "data/camels_gb_v2/Catchment_Attributes/camels_gb_v2_climatic_attributes.csv",
                   dtype={"gauge_id": str}).set_index("gauge_id")
snow = clim["frac_snow"].reindex(nse.index)
best_gr        = nse[["GR4J", "GR6J"]].max(axis=1)
best_gr_snow   = nse[["CemaNeigeGR4J", "CemaNeigeGR6J"]].max(axis=1)
snow_gain      = best_gr_snow - best_gr                       # what the snow module adds
lstm_gap_before = nse["LSTM_ensemble"] - best_gr              # LSTM advantage, no snow module
lstm_gap_after  = nse["LSTM_ensemble"] - best_gr_snow         # LSTM advantage, with snow module

print("\nSpearman correlation with frac_snow (fraction of precipitation falling as snow):")
for name, s in [("NSE gain from snow module", snow_gain),
                ("LSTM advantage over GR (no snow)", lstm_gap_before),
                ("LSTM advantage over GR (with snow)", lstm_gap_after)]:
    rho, p = spearmanr(snow, s, nan_policy="omit")
    print(f"  {name:<36} rho = {rho:+.2f}  (p = {p:.3g})")

# --- Step 5: the snowiest catchments in detail -------------------------------
tab = pd.DataFrame({"frac_snow": snow, "GR_best": best_gr, "GR_snow_best": best_gr_snow,
                    "LSTM": nse["LSTM_ensemble"], "gap_before": lstm_gap_before,
                    "gap_after": lstm_gap_after})
tab = tab.join(info.set_index("gauge_id")["gauge_name"])
print("\n10 snowiest catchments (NSE, test period):")
print(tab.nlargest(10, "frac_snow").round(3).to_string())
print(f"\nMedian LSTM advantage: {lstm_gap_before.median():+.3f} without snow module -> "
      f"{lstm_gap_after.median():+.3f} with snow module")
