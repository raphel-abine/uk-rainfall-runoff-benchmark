# =============================================================================
# 03 - Explore CAMELS-GB v2
#
# File inventory, attribute tables, daily and hourly series for one gauge,
# map of all catchments, and decoding of the hourly discharge QC flags
# (3-digit XYZ codes, Coxon et al. 2026, Supplement S1).
# Figures are saved to outputs/figures/.
#
# Run:   python 03_explore_data.py
# Needs: 01
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT, DATA_DIR, save_figure, load_attributes, gauge_from_name  # noqa: F401

# =============================================================================
# CELL 1 - Setup and inventory of the CAMELS-GB v2 files
# =============================================================================
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = (PROJECT_ROOT / "data/camels_gb_v2")

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# --- For each folder: number of files, total size, and example file names ---
for folder in sorted({p.parent for p in DATA_DIR.rglob("*") if p.is_file()}):
    files = sorted(f for f in folder.iterdir() if f.is_file())
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\n{folder.relative_to(DATA_DIR)}  ({len(files)} files, {size_mb:,.1f} MB)")
    for f in files[:5]:                                # show up to 5 example names
        print(f"    {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
    if len(files) > 5:
        print(f"    ... and {len(files)-5} more")

# =============================================================================
# Attribute tables (rows with stray commas are repaired - see config.read_csv_robust)
# =============================================================================
attrs = load_attributes(verbose=True)
print("\nTables loaded:", list(attrs.keys()))
print("\nhydrometry column types (should now be numeric where expected):")
print(attrs["hydrometry"].dtypes.value_counts().to_string())

# =============================================================================
# CELL 3b - Daily hydro-meteorological series for one example catchment
# =============================================================================
DAILY_DIR = DATA_DIR / "Catchment_Timeseries/hydro-meteorological/daily"
HOURLY_DIR = DATA_DIR / "Catchment_Timeseries/hydro-meteorological/hourly"

daily_files = {gauge_from_name(p): p for p in DAILY_DIR.glob("*.csv")}
hourly_files = {gauge_from_name(p): p for p in HOURLY_DIR.glob("*.csv")}
print(f"{len(daily_files)} daily files, {len(hourly_files)} hourly files")

gauge_id = "10002"                                     # change to look at another gauge
names = attrs["topographic"].set_index("gauge_id")["gauge_name"]
print(f"Gauge {gauge_id}: {names.get(int(gauge_id), '?')}")

daily = pd.read_csv(daily_files[gauge_id])
date_col = next(c for c in daily.columns if "date" in c.lower() or "time" in c.lower())
daily[date_col] = pd.to_datetime(daily[date_col])
daily = daily.set_index(date_col)
print("Columns:", list(daily.columns))

# --- Period each variable actually has data for -----------------------------
coverage = pd.DataFrame({
    "first_value": daily.apply(lambda s: s.first_valid_index()),
    "last_value": daily.apply(lambda s: s.last_valid_index()),
    "missing_%": (100 * daily.isna().mean()).round(2),
})
print("\nData coverage per variable:")
print(coverage.to_string())

# --- Plot every numeric variable ---------------------------------------------
num_cols = daily.select_dtypes("number").columns
fig, axes = plt.subplots(len(num_cols), 1, figsize=(14, 2.0 * len(num_cols)), sharex=True)
for ax, col in zip(axes, num_cols):
    ax.plot(daily.index, daily[col], lw=0.4)
    ax.set_ylabel(col, rotation=0, ha="right", fontsize=8)
axes[0].set_title(f"CAMELS-GB v2 daily series - gauge {gauge_id} ({names.get(int(gauge_id), '')})")
plt.tight_layout()
save_figure("explore_daily_series_10002")

# =============================================================================
# CELL 4 - Hourly series for the same catchment, zoomed on the largest event
# =============================================================================
hourly = pd.read_csv(hourly_files[gauge_id])
h_date_col = next(c for c in hourly.columns if "date" in c.lower() or "time" in c.lower())
hourly[h_date_col] = pd.to_datetime(hourly[h_date_col])
hourly = hourly.set_index(h_date_col)
print("File:", hourly_files[gauge_id].name)
print("Columns:", list(hourly.columns))

coverage_h = pd.DataFrame({
    "first_value": hourly.apply(lambda s: s.first_valid_index()),
    "last_value": hourly.apply(lambda s: s.last_valid_index()),
    "missing_%": (100 * hourly.isna().mean()).round(2),
})
print("\nData coverage per variable:")
print(coverage_h.to_string())

# --- Zoom in on the largest hourly flow event --------------------------------
h_num = hourly.select_dtypes("number").columns
flow_col = next((c for c in h_num if "discharge" in c.lower() or "flow" in c.lower()), h_num[0])
peak = hourly[flow_col].idxmax()
window = hourly.loc[peak - pd.Timedelta(days=5): peak + pd.Timedelta(days=5)]

fig, axes = plt.subplots(len(h_num), 1, figsize=(14, 2.0 * len(h_num)), sharex=True)
axes = [axes] if len(h_num) == 1 else axes
for ax, col in zip(axes, h_num):
    ax.plot(window.index, window[col], lw=0.8)
    ax.set_ylabel(col, rotation=0, ha="right", fontsize=8)
axes[0].set_title(f"Hourly data around the largest event ({peak:%d %b %Y}) - gauge {gauge_id}")
plt.tight_layout()
save_figure("explore_hourly_largest_event_10002")

# =============================================================================
# CELL 5 - Map of the catchment boundaries
# =============================================================================
import geopandas as gpd

BOUND_DIR = DATA_DIR / "Catchment_Boundaries"
shp = sorted(BOUND_DIR.glob("*.shp"))
print("Shapefiles found:", [s.name for s in shp])

gdf = gpd.read_file(shp[0])
print("\nCRS:", gdf.crs)
print("Columns:", list(gdf.columns))
print("Number of catchments:", len(gdf))

ax = gdf.plot(figsize=(7, 10), edgecolor="black", linewidth=0.2, facecolor="lightsteelblue")
ax.set_title("CAMELS-GB v2 catchments (671)")
ax.set_axis_off()
save_figure("explore_all_catchments_map")

# =============================================================================
# Decode the 3-digit hourly discharge flags (XYZ) - Coxon et al. (2026), Supplement S1
# =============================================================================
NRFA_X = {0: "agrees with NRFA", 1: ">5% off NRFA daily", 2: ">20% off POT peaks",
          3: ">20% off annual maxima", 4: "off daily + POT", 5: "off daily + AMAX"}
BASIC_Y = {0: "OK", 1: "negative", 2: "relative spike", 3: "absolute spike", 4: "drop",
           5: "fluctuation", 6: "truncated low", 7: "truncated high",
           8: "relative + absolute spike", 9: "drop + negative/truncated low"}
HIGH_Z = {0: "not a high flow", 1: "unrealistically high", 2: "exceptionally high",
          3: "big event, no antecedent rain", 4: "big event, no regional high flow",
          5: "no rain and no regional high flow", 6: "validated event",
          7: "exceptional + unsupported", 8: "exceptional + no rain/regional", 9: "exceptional, validated"}

# Turn e.g. 6 -> "006" and split into the three digits
codes = hourly["discharge_flag"].dropna().astype(int).astype(str).str.zfill(3)
decoded = pd.DataFrame({
    "nrfa_check":  codes.str[0].astype(int).map(NRFA_X),
    "basic_qc":    codes.str[1].astype(int).map(BASIC_Y),
    "high_flow_qc": codes.str[2].astype(int).map(HIGH_Z),
})

# Percentage of hours in each category, for each check
for col in decoded.columns:
    print(f"\n{col} (% of hours):")
    print((100 * decoded[col].value_counts(normalize=True)).round(3).to_string())
