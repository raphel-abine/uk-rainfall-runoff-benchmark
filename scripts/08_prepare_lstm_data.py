# =============================================================================
# 08 - Convert CAMELS-GB v2 to neuralhydrology's GenericDataset format
#
# time_series/<gauge_id>.nc (coordinate 'date', all A and B variables),
# attributes/static_attributes.csv (22 descriptors, no flow signatures),
# and the training (663) and test (128) catchment lists.
#
# Run:   python 08_prepare_lstm_data.py
# Needs: 05
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 12 - Convert CAMELS-GB v2 into neuralhydrology "GenericDataset" format
# time_series/<gauge_id>.nc (coordinate 'date') + attributes/static_attributes.csv
# Run once; output is saved to Drive and reused for every seed and for Option B
# =============================================================================

import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = PROJECT_ROOT
SRC = ROOT / "data/camels_gb_v2"
NH = ROOT / "data/nh_camels_gb_v2"                       # neuralhydrology data_dir
(NH / "time_series").mkdir(parents=True, exist_ok=True)
(NH / "attributes").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/lstm").mkdir(parents=True, exist_ok=True)

TRAIN = ("1988-01-01", "1997-12-31")
DYNAMIC_A = ["precipitation_cehgear", "pet_chess", "temperature_chess"]
TARGET = "discharge_spec"

# --- Step 1: daily time series -> one netCDF per catchment -------------------
daily_dir = SRC / "Catchment_Timeseries/hydro-meteorological/daily"
train_flow_days = {}
for i, f in enumerate(sorted(daily_dir.glob("*.csv")), 1):
    gid = re.search(r"timeseries_(\d+)_", f.name).group(1)
    df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    df.index.name = "date"                                 # required coordinate name
    out = NH / "time_series" / f"{gid}.nc"
    if not out.exists():
        df.astype("float32").to_xarray().to_netcdf(out)   # keeps all variables (A and B)
    train_flow_days[gid] = int(df.loc[TRAIN[0]:TRAIN[1], TARGET].notna().sum())
    if i % 100 == 0:
        print(f"  {i} catchments converted")
print(f"Time series files: {len(list((NH / 'time_series').glob('*.nc')))}")

# --- Step 2: static attributes (no discharge-derived signatures) -------------
def read_attr(name):
    """Read one attribute table; repair rows with stray commas (hydrometry issue)."""
    p = SRC / f"Catchment_Attributes/camels_gb_v2_{name}_attributes.csv"
    return pd.read_csv(p, encoding="latin-1", on_bad_lines="skip")

STATIC = {
    "topographic": ["area", "elev_mean", "dpsbar"],
    "climatic":    ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
                    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"],
    "soil":        ["sand_perc", "silt_perc", "clay_perc", "porosity_hypres",
                    "conductivity_hypres", "soil_depth_pelletier"],
    "landcover":   ["dwood_perc_2015", "ewood_perc_2015", "crop_perc_2015",
                    "urban_perc_2015"],                    # LCM2015, as in CAMELS-GB v1
}
attrs = None
for table, cols in STATIC.items():
    part = read_attr(table)[["gauge_id"] + cols]
    attrs = part if attrs is None else attrs.merge(part, on="gauge_id", how="outer")
attrs["gauge_id"] = attrs["gauge_id"].astype(str)
attrs = attrs.set_index("gauge_id")
static_list = [c for cols in STATIC.values() for c in cols]
print(f"\nStatic attributes ({len(static_list)}): {static_list}")

# --- Step 3: training catchments = complete attributes + >= 1 yr flow in train
has_attrs = attrs.notna().all(axis=1)
has_flow = pd.Series(train_flow_days).reindex(attrs.index).fillna(0) >= 365
train_basins = sorted(attrs.index[has_attrs & has_flow], key=int)
print(f"\nCatchments with missing attributes:  {list(attrs.index[~has_attrs])}")
print(f"Catchments with < 1 yr training flow: {list(attrs.index[has_attrs & ~has_flow])}")
print(f"Training catchments: {len(train_basins)}")

attrs.loc[train_basins].to_csv(NH / "attributes" / "static_attributes.csv")

# --- Step 4: basin lists -----------------------------------------------------
elig = pd.read_csv(ROOT / "outputs/catchment_selection/eligible_catchments.csv")
test_basins = [g for g in elig["gauge_id"].astype(str) if g in train_basins]
missing = sorted(set(elig["gauge_id"].astype(str)) - set(test_basins))
print(f"Test catchments (of our 128): {len(test_basins)}  | not usable: {missing}")

(ROOT / "outputs/lstm/basins_train.txt").write_text("\n".join(train_basins) + "\n")
(ROOT / "outputs/lstm/basins_test_128.txt").write_text("\n".join(test_basins) + "\n")
print("Basin lists saved.")
