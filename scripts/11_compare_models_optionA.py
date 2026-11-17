# =============================================================================
# 11 - Compare GR4J, GR6J and the LSTM ensemble (Option A)
#
# Paired Wilcoxon tests (Holm-corrected), results by regime, NSE
# distribution and maps of where each model wins.
#
# Run:   python 11_compare_models_optionA.py
# Needs: 10
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401
import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
CMP_DIR = ROOT / "outputs/comparison"

# =============================================================================
# CELL 18 - Compare GR4J, GR6J and the LSTM ensemble: paired tests, regimes,
# NSE distribution and maps of where each model wins
# =============================================================================
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DATA_DIR = ROOT / "data/camels_gb_v2"
MODELS = ["GR4J", "GR6J", "LSTM_ensemble"]
COLORS = {"GR4J": "#2a78d6", "GR6J": "#eb6834", "LSTM_ensemble": "#1baf7a"}  # CVD-checked
LABELS = {"GR4J": "GR4J", "GR6J": "GR6J", "LSTM_ensemble": "LSTM (8-seed ensemble)"}

df = pd.read_csv(CMP_DIR / "metrics_test_all_models.csv", dtype={"gauge_id": str})
df = df[df["model"].isin(MODELS)]

# --- Step 1: paired Wilcoxon tests with Holm correction ----------------------
# "higher" = higher is better; "abs" = smaller |bias| is better; "lower" = lower is better
CRITERIA = {"NSE": "higher", "KGE": "higher", "FHV": "abs", "FMS": "abs",
            "FLV": "abs", "Peak-Timing": "lower"}
PAIRS = [("LSTM_ensemble", "GR4J"), ("LSTM_ensemble", "GR6J"), ("GR6J", "GR4J")]

tests = []
for metric, rule in CRITERIA.items():
    wide = df.pivot(index="gauge_id", columns="model", values=metric)
    for a, b in PAIRS:
        x, y = wide[a], wide[b]
        ok = x.notna() & y.notna()
        if rule == "higher":
            gain = x[ok] - y[ok]                    # > 0 means a is better
        elif rule == "lower":
            gain = y[ok] - x[ok]
        else:
            gain = y[ok].abs() - x[ok].abs()        # > 0 means a has smaller |bias|
        tests.append({"metric": metric, "comparison": f"{a} vs {b}",
                      "a_better_in": f"{(gain > 0).sum()}/{ok.sum()}",
                      "median_gain": gain.median(),
                      "p_raw": wilcoxon(gain[gain != 0]).pvalue})
tests = pd.DataFrame(tests)
tests["p_holm"] = multipletests(tests["p_raw"], method="holm")[1]
tests["significant"] = np.where(tests["p_holm"] < 0.05, "yes", "no")
print("Paired comparisons (test period). median_gain > 0 favours the first model:")
print(tests.round({"median_gain": 3, "p_raw": 4, "p_holm": 4}).to_string(index=False))
tests.to_csv(CMP_DIR / "paired_tests.csv", index=False)

# --- Step 2: results by aridity/BFI regime -----------------------------------
print("\nMedian test NSE by regime:")
print(df.pivot_table(index="regime", columns="model", values="NSE", aggfunc="median").round(3).to_string())
print("\nMedian test FLV (low-flow bias, %) by regime:")
print(df.pivot_table(index="regime", columns="model", values="FLV", aggfunc="median").round(1).to_string())

nse = df.pivot(index="gauge_id", columns="model", values="NSE")
best = nse[MODELS].idxmax(axis=1).rename("best_model")
regime = df.drop_duplicates("gauge_id").set_index("gauge_id")["regime"]
print("\nBest model by NSE - number of catchments per regime:")
print(pd.crosstab(regime, best, margins=True).to_string())

# --- Step 3: cumulative distribution of test NSE (as in Kratzert/Lees) -------
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in MODELS:
    v = np.sort(nse[m].dropna().values)
    cdf = np.arange(1, len(v) + 1) / len(v)
    ax.step(v, cdf, where="post", lw=2, color=COLORS[m],
            label=f"{LABELS[m]} (median {np.median(v):.3f})")
ax.set_xlim(0.4, 1.0)
ax.set_xlabel("Test-period NSE (1998-2008)")
ax.set_ylabel("Fraction of catchments")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"NSE across {len(nse)} benchmark catchments")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(CMP_DIR / "nse_cdf.png", dpi=200)
plt.close("all")

# --- Step 4: maps - best model, and LSTM gain over the best GR model ---------
gdf = gpd.read_file(DATA_DIR / "Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp")
gdf["gauge_id"] = gdf["ID"].astype(int).astype(str)
gdf = gdf.merge(best, left_on="gauge_id", right_index=True, how="left")
gdf["d_nse"] = (nse["LSTM_ensemble"] - nse[["GR4J", "GR6J"]].max(axis=1)).reindex(gdf["gauge_id"]).values
sel = gdf[gdf["best_model"].notna()]

fig, axes = plt.subplots(1, 2, figsize=(13, 10))

# (a) which model has the highest NSE in each catchment
ax = axes[0]
gdf.plot(ax=ax, facecolor="#f0efec", edgecolor="#d6d5d0", linewidth=0.2)
for m in MODELS:
    part = sel[sel["best_model"] == m]
    if len(part):
        part.plot(ax=ax, facecolor=COLORS[m], edgecolor="white", linewidth=0.4)
counts = sel["best_model"].value_counts()
ax.legend(handles=[Patch(facecolor=COLORS[m], label=f"{LABELS[m]} ({counts.get(m, 0)})")
                   for m in MODELS], loc="upper left", frameon=False, title="Highest test NSE")
ax.set_title("(a) Best model per catchment")
ax.set_axis_off()

# (b) LSTM ensemble NSE minus the better of GR4J/GR6J (red = GR better, blue = LSTM better)
ax = axes[1]
div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
lim = np.nanpercentile(np.abs(sel["d_nse"]), 95)
gdf.plot(ax=ax, facecolor="#f0efec", edgecolor="#d6d5d0", linewidth=0.2)
sel.plot(ax=ax, column="d_nse", cmap=div, norm=TwoSlopeNorm(0, -lim, lim),
         edgecolor="white", linewidth=0.4, legend=True,
         legend_kwds={"label": "NSE(LSTM) - NSE(best GR)", "shrink": 0.5})
ax.set_title("(b) LSTM gain over the better GR model")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(CMP_DIR / "maps_best_model.png", dpi=200)
plt.close("all")

# --- Step 5: catchments where the LSTM gains or loses most -------------------
tab = pd.DataFrame({"NSE_GR4J": nse["GR4J"], "NSE_GR6J": nse["GR6J"],
                    "NSE_LSTM": nse["LSTM_ensemble"],
                    "gain": nse["LSTM_ensemble"] - nse[["GR4J", "GR6J"]].max(axis=1)})
tab = tab.join(df.drop_duplicates("gauge_id").set_index("gauge_id")[["gauge_name", "regime"]])
print("\nLargest LSTM gains:")
print(tab.nlargest(8, "gain").round(3).to_string())
print("\nWhere GR beats the LSTM most:")
print(tab.nsmallest(8, "gain").round(3).to_string())
print(f"\nFigures and tables saved in: {CMP_DIR}")
