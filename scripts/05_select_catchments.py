# =============================================================================
# 05 - Select the benchmark catchments
#
# Rules: UK Benchmark Network, runoff ratio <= 1, >= 90% flow data in both
# the training (1988-97) and test (1998-2008) periods -> 128 catchments.
# Adds aridity x baseflow-index regime labels (terciles).
# Note: an early version also drew a random sample of 50; the final study
# uses all 128 eligible catchments, so that step is not included here.
#
# Run:   python 05_select_catchments.py
# Needs: 01
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT, DATA_DIR, save_figure, load_attributes, gauge_from_name  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt

attrs = load_attributes(verbose=False)
DAILY_DIR = DATA_DIR / "Catchment_Timeseries/hydro-meteorological/daily"
daily_files = {gauge_from_name(p): p for p in DAILY_DIR.glob("*.csv")}

import numpy as np
import geopandas as gpd

# --- Settings ----------------------------------------------------------------
PERIODS = {                                  # Lees et al. (2021)
    "validation": ("1975-01-01", "1980-12-31"),
    "train":      ("1988-01-01", "1997-12-31"),
    "test":       ("1998-01-01", "2008-12-31"),
}
FORCING_A = ["precipitation_cehgear", "pet_chess", "temperature_chess"]
MIN_FLOW_COMPLETE = 90.0                     # % of days with flow, train and test (our choice)
OUT_DIR = DATA_DIR.parent.parent / "outputs" / "catchment_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: one attribute table, one row per catchment ----------------------
keep = {
    "topographic":    ["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area", "elev_mean"],
    "climatic":       ["gauge_id", "p_mean", "aridity", "frac_snow"],
    "hydrologic":     ["gauge_id", "q_mean", "runoff_ratio", "baseflow_index"],
    "humaninfluence": ["gauge_id", "benchmark_catch"],
    "hydrometry":     ["gauge_id", "station_type", "max_gauging_flow", "max_hourly_flow",
                       "quncert_meta"],
}
cat = None
for table, cols in keep.items():
    part = attrs[table][cols]
    cat = part if cat is None else cat.merge(part, on="gauge_id", how="left")
cat["gauge_id"] = cat["gauge_id"].astype(str)
cat["extrap_ratio"] = cat["max_hourly_flow"] / cat["max_gauging_flow"]

# --- Step 2: benchmark catchments with a plausible water balance -------------
pool = cat[cat["benchmark_catch"].eq("Y")].copy()
print(f"Benchmark catchments:                         {len(pool)}")
dropped_rr = pool[pool["runoff_ratio"] > 1]
pool = pool[pool["runoff_ratio"] <= 1]
print(f"After removing runoff ratio > 1:              {len(pool)}  "
      f"(removed {list(dropped_rr['gauge_id'])})")

# --- Step 3: flow completeness in each period, from the daily files ----------
def period_completeness(gid):
    """% of days with flow (and with all Option-A forcings) in each period."""
    df = pd.read_csv(daily_files[gid], parse_dates=["date"]).set_index("date")
    out = {"gauge_id": gid}
    for name, (start, end) in PERIODS.items():
        sub = df.loc[start:end]
        out[f"flow_{name}_%"] = 100 * sub["discharge_spec"].notna().mean() if len(sub) else 0.0
        out[f"forcing_{name}_%"] = 100 * sub[FORCING_A].notna().all(axis=1).mean() if len(sub) else 0.0
    return out

comp = pd.DataFrame([period_completeness(g) for g in pool["gauge_id"]]).round(1)
pool = pool.merge(comp, on="gauge_id")

ok = (pool["flow_train_%"] >= MIN_FLOW_COMPLETE) & (pool["flow_test_%"] >= MIN_FLOW_COMPLETE)
print(f"With >= {MIN_FLOW_COMPLETE}% flow in train AND test:    {ok.sum()}  "
      f"(removed {(~ok).sum()})")
print("Forcing complete in train and test for all:",
      bool((pool.loc[ok, ['forcing_train_%', 'forcing_test_%']] == 100).all().all()))
eligible = pool[ok].copy()

# --- Step 4: aridity x BFI classes (terciles of the eligible pool) -----------
eligible["aridity_class"] = pd.qcut(eligible["aridity"], 3, labels=["wet", "mid", "dry"])
eligible["bfi_class"] = pd.qcut(eligible["baseflow_index"], 3, labels=["flashy", "mid", "baseflow"])
eligible["regime"] = eligible["aridity_class"].astype(str) + "/" + eligible["bfi_class"].astype(str)


print("\nEligible catchments per regime (aridity/BFI):")
print(eligible["regime"].value_counts().sort_index().to_string())

# --- Step 5: save -------------------------------------------------------------
eligible.to_csv(OUT_DIR / "eligible_catchments.csv", index=False)
print(f"\n{len(eligible)} eligible catchments saved to {OUT_DIR / 'eligible_catchments.csv'}")

# --- Step 6: map of the eligible catchments, coloured by regime --------------
gdf = gpd.read_file(DATA_DIR / "Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp")
gdf["gauge_id"] = gdf["ID"].astype(int).astype(str)
fig, ax = plt.subplots(figsize=(7, 10))
gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="lightgrey", linewidth=0.2)
gdf.merge(eligible[["gauge_id", "regime"]], on="gauge_id").plot(
    ax=ax, column="regime", cmap="tab10", edgecolor="black", linewidth=0.4,
    legend=True, legend_kwds={"loc": "upper left", "fontsize": 7, "title": "aridity/BFI"})
ax.set_title(f"Eligible benchmark catchments ({len(eligible)}) by regime")
ax.set_axis_off()
save_figure("eligible_catchments_map")
