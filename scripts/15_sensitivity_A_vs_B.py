# =============================================================================
# 15 - Sensitivity of the results to the input data (Option A vs B)
#
# Medians, paired tests within B, change from A to B, agreement on the
# best model, regimes and the snow finding under Option B.
#
# Run:   python 15_sensitivity_A_vs_B.py
# Needs: 12, 14
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401
import pandas as pd

ROOT = PROJECT_ROOT
CMP_DIR = ROOT / "outputs/comparison"

# =============================================================================
# CELL 26 - Sensitivity to forcing data: Option A (CEH-GEAR + CHESS-PE) vs
# Option B (HadUK-Grid + Hydro-PE). Same catchments, periods and metrics.
# =============================================================================
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests

MAIN = ["GR4J", "GR6J", "CemaNeigeGR4J", "CemaNeigeGR6J", "LSTM_ensemble"]
met_a = pd.read_csv(CMP_DIR / "metrics_test_all_models.csv", dtype={"gauge_id": str})
met_a["forcing"] = "A"
met_b = pd.read_csv(CMP_DIR / "metrics_test_optionB.csv", dtype={"gauge_id": str})
both = pd.concat([met_a, met_b], ignore_index=True)
both = both[both["model"].isin(MAIN)]
info = met_a.drop_duplicates("gauge_id").set_index("gauge_id")[["gauge_name", "regime"]]

# --- 1. Medians side by side -------------------------------------------------
print("Median test metrics by forcing option:")
for metric in ["NSE", "KGE", "FHV", "FLV", "Peak-Timing"]:
    t = both.pivot_table(index="model", columns="forcing", values=metric, aggfunc="median").reindex(MAIN)
    t["B - A"] = t["B"] - t["A"]
    print(f"\n{metric}:\n{t.round(3).to_string()}")

# --- 2. Does the ranking hold within Option B? (paired NSE, Holm) ------------
nse_b = met_b[met_b["model"].isin(MAIN)].pivot(index="gauge_id", columns="model", values="NSE")
pairs = [("LSTM_ensemble", "GR6J"), ("LSTM_ensemble", "CemaNeigeGR6J"),
         ("CemaNeigeGR6J", "GR6J"), ("GR6J", "GR4J")]
tests = []
for a, b in pairs:
    g = (nse_b[a] - nse_b[b]).dropna()
    tests.append({"comparison (Option B)": f"{a} vs {b}", "a_better_in": f"{(g > 0).sum()}/{len(g)}",
                  "median_gain": g.median(), "p_raw": wilcoxon(g[g != 0]).pvalue})
tests = pd.DataFrame(tests)
tests["p_holm"] = multipletests(tests["p_raw"], method="holm")[1]
print("\nPaired NSE comparisons within Option B:")
print(tests.round(4).to_string(index=False))

# --- 3. How much does each model change from A to B? -------------------------
rows = []
for m in MAIN:
    w = both[both["model"] == m].pivot(index="gauge_id", columns="forcing", values="NSE").dropna()
    d = w["B"] - w["A"]
    rho, _ = spearmanr(w["A"], w["B"])
    rows.append({"model": m, "median_dNSE_(B-A)": d.median(), "B_better_in": f"{(d > 0).sum()}/{len(d)}",
                 "p_wilcoxon": wilcoxon(d[d != 0]).pvalue,
                 "spearman_rho_A_vs_B": rho})
print("\nChange in NSE from Option A to Option B (same catchments):")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# --- 4. Best model per catchment: same answer under A and B? -----------------
best = {}
for opt in ["A", "B"]:
    w = both[both["forcing"] == opt].pivot(index="gauge_id", columns="model", values="NSE")
    best[opt] = w[["GR4J", "GR6J", "LSTM_ensemble"]].idxmax(axis=1)       # as in the main maps
agree = (best["A"] == best["B"]).mean()
print(f"\nBest model (GR4J/GR6J/LSTM) by NSE - counts:\n"
      f"{pd.DataFrame({'A': best['A'].value_counts(), 'B': best['B'].value_counts()}).fillna(0).astype(int)}")
print(f"Same best model under A and B in {100 * agree:.0f}% of catchments")

# --- 5. Regimes and the snow finding under Option B --------------------------
print("\nMedian test NSE by regime, Option B:")
print(met_b[met_b["model"].isin(MAIN)].merge(info, left_on="gauge_id", right_index=True)
      .pivot_table(index="regime", columns="model", values="NSE", aggfunc="median")[MAIN].round(3).to_string())

clim = pd.read_csv(ROOT / "data/camels_gb_v2/Catchment_Attributes/camels_gb_v2_climatic_attributes.csv",
                   dtype={"gauge_id": str}).set_index("gauge_id")
snow = clim["frac_snow"].reindex(nse_b.index)
gap_no_snow = nse_b["LSTM_ensemble"] - nse_b[["GR4J", "GR6J"]].max(axis=1)
gap_snow = nse_b["LSTM_ensemble"] - nse_b[["CemaNeigeGR4J", "CemaNeigeGR6J"]].max(axis=1)
print("\nOption B - Spearman correlation of the LSTM advantage with frac_snow:")
for name, s in [("vs GR without snow module", gap_no_snow), ("vs GR with snow module", gap_snow)]:
    rho, p = spearmanr(snow, s)
    print(f"  {name:<28} rho = {rho:+.2f} (p = {p:.3g})")

pd.DataFrame(rows).to_csv(CMP_DIR / "sensitivity_A_vs_B.csv", index=False)
print(f"\nSaved: {CMP_DIR / 'sensitivity_A_vs_B.csv'}")
