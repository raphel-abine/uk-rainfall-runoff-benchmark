# =============================================================================
# 04 - The 137 UK Benchmark Network catchments vs all 671
#
# Data quality, spread of climate and baseflow index, and a map.
#
# Run:   python 04_benchmark_overview.py
# Needs: 01
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT, DATA_DIR, save_figure, load_attributes, gauge_from_name  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt

attrs = load_attributes(verbose=False)

# =============================================================================
# CELL 6 - The 137 benchmark (near-natural) catchments vs all 671
# =============================================================================
import geopandas as gpd

# --- Join the attribute tables into one table, one row per catchment ---------
keep = {
    "topographic":    ["gauge_id", "gauge_name", "area", "elev_mean", "dpsbar"],
    "climatic":       ["gauge_id", "p_mean", "pet_mean", "aridity", "frac_snow"],
    "hydrologic":     ["gauge_id", "q_mean", "runoff_ratio", "baseflow_index", "Q95", "Q5"],
    "humaninfluence": ["gauge_id", "benchmark_catch", "surfacewater_abs",
                       "groundwater_abs", "num_reservoir"],
    "hydrometry":     ["gauge_id", "station_type", "daily_flow_perc_complete",
                       "hourly_flow_perc_complete", "max_gauging_flow", "max_hourly_flow",
                       "quncert_meta", "station_quality_hourlyflow_issues"],
}
cat = None
for table, cols in keep.items():
    part = attrs[table][cols]
    cat = part if cat is None else cat.merge(part, on="gauge_id", how="left")

cat["benchmark"] = cat["benchmark_catch"].eq("Y")
# how far the highest hourly flow goes beyond the highest manual gauging
cat["extrap_ratio"] = cat["max_hourly_flow"] / cat["max_gauging_flow"]
bench = cat[cat["benchmark"]]
print(f"Benchmark catchments: {len(bench)} of {len(cat)}")

# --- Compare benchmark catchments with all catchments ------------------------
cols = ["area", "elev_mean", "p_mean", "aridity", "baseflow_index",
        "runoff_ratio", "daily_flow_perc_complete", "hourly_flow_perc_complete", "extrap_ratio"]
summary = pd.concat({
    "all_671_median": cat[cols].median(),
    "bench_median": bench[cols].median(),
    "bench_min": bench[cols].min(),
    "bench_max": bench[cols].max(),
}, axis=1).round(2)
print("\nBenchmark vs all catchments:")
print(summary.to_string())

# --- Data-quality issues within the benchmark set ----------------------------
print("\nQuality checks (benchmark catchments):")
print("  runoff ratio > 1 (water-balance problem):", (bench["runoff_ratio"] > 1).sum())
print("  daily flow < 90% complete:              ", (bench["daily_flow_perc_complete"] < 90).sum())
print("  hourly flow < 90% complete:             ", (bench["hourly_flow_perc_complete"] < 90).sum())
print("  hourly flow issue flagged:              ", bench["station_quality_hourlyflow_issues"].notna().sum())
print("  peak flow > 2x highest gauging:         ", (bench["extrap_ratio"] > 2).sum())
print("\nDischarge uncertainty information (benchmark):")
print(bench["quncert_meta"].value_counts().to_string())
print("\nStation types (benchmark):")
print(bench["station_type"].value_counts().head(10).to_string())

# --- Distributions: is the benchmark set varied enough? ----------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, ["aridity", "baseflow_index", "area", "elev_mean"]):
    ax.hist(cat[col].dropna(), bins=30, alpha=0.5, label="all 671")
    ax.hist(bench[col].dropna(), bins=30, alpha=0.8, label="benchmark")
    ax.set_title(col)
    if col == "area":
        ax.set_xscale("log")
axes[0].legend()
plt.tight_layout()
save_figure("benchmark_distributions")

# --- Map: benchmark catchments coloured by baseflow index --------------------
gdf = gpd.read_file(DATA_DIR / "Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp")
gdf = gdf.merge(cat, left_on="ID", right_on="gauge_id", how="left")

fig, ax = plt.subplots(figsize=(7, 10))
gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="grey", linewidth=0.2)
gdf[gdf["benchmark"]].plot(ax=ax, column="baseflow_index", cmap="viridis",
                           edgecolor="black", linewidth=0.3, legend=True,
                           legend_kwds={"label": "Baseflow index", "shrink": 0.5})
ax.set_title(f"Benchmark catchments ({len(bench)}) coloured by baseflow index")
ax.set_axis_off()
save_figure("benchmark_map_bfi")
