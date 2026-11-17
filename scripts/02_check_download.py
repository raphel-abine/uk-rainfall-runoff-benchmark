# =============================================================================
# 02 - Check the CAMELS-GB v2 download is complete
#
# Counts files and size per folder and flags empty (failed) files.
# Can be run while 01 is still downloading.
#
# Run:   python 02_check_download.py
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# Check progress of the CAMELS-GB v2 download in Google Drive
# Run this in a SEPARATE Colab notebook while the download is still running.
# =============================================================================

from pathlib import Path

# --- Connect to the same Google Drive ----------------------------------------
OUT_DIR = (PROJECT_ROOT / "data/camels_gb_v2")

# --- Number of files expected in each folder (from the listing) -------------
EXPECTED = {
    "Catchment_Attributes": 9,
    "Catchment_Boundaries": 7,
    "Catchment_Timeseries/groundwater/daily": 23,
    "Catchment_Timeseries/groundwater/monthly": 55,
    "Catchment_Timeseries/hydro-meteorological/daily": 671,
    "Catchment_Timeseries/hydro-meteorological/hourly": 671,
}

# --- Count files and size in each folder -------------------------------------
print(f"{'Folder':<50} {'Done':>6} / {'Total':<6} {'Size (GB)':>10}  Status")
print("-" * 90)
total_done, total_expected, total_bytes = 0, 0, 0
for folder, expected in EXPECTED.items():
    path = OUT_DIR / folder
    files = [f for f in path.glob("*") if f.is_file()] if path.exists() else []
    n = len(files)
    size = sum(f.stat().st_size for f in files)
    status = "complete" if n >= expected else ("in progress" if n > 0 else "not started")
    print(f"{folder:<50} {n:>6} / {expected:<6} {size/1e9:>10.2f}  {status}")
    total_done += n
    total_expected += expected
    total_bytes += size

# --- Overall summary ---------------------------------------------------------
print("-" * 90)
pct = 100 * total_done / total_expected
print(f"{'TOTAL':<50} {total_done:>6} / {total_expected:<6} {total_bytes/1e9:>10.2f}  ({pct:.1f}% of files)")

# --- Flag any empty files (would point to a failed download) -----------------
empty = [f for f in OUT_DIR.rglob("*") if f.is_file() and f.stat().st_size == 0]
if empty:
    print(f"\nWarning: {len(empty)} empty files found, e.g. {empty[0].name}")
else:
    print("\nNo empty files found.")
