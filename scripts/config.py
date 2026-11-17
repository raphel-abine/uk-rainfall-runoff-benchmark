# =============================================================================
# config.py - Shared paths and helpers for all project scripts
#
# Set the project folder once with the environment variable RRB_ROOT, e.g.
#   Windows (PowerShell):  $env:RRB_ROOT = "C:\Users\<you>\Desktop\rainfall_runoff_benchmark"
#   Linux / macOS:         export RRB_ROOT=/path/to/rainfall_runoff_benchmark
# If it is not set, the Google Drive folder used in Colab is assumed.
#
# Expected layout under the project folder:
#   data/camels_gb_v2/        raw CAMELS-GB v2 (script 01)
#   data/nh_camels_gb_v2/     neuralhydrology input data (script 08)
#   outputs/                  everything the scripts produce
# =============================================================================
import csv
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # scripts save figures instead of showing them
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

PROJECT_ROOT = Path(os.environ.get("RRB_ROOT", "/content/drive/MyDrive/rainfall_runoff_benchmark"))
DATA_DIR = PROJECT_ROOT / "data" / "camels_gb_v2"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)


def save_figure(name):
    """Save the current matplotlib figure to outputs/figures/<name>.png and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Figure saved: {path}")


def gauge_from_name(path):
    """Gauge ID sits between 'timeseries_' and the next '_' in the file name."""
    return re.search(r"timeseries_(\d+)_", Path(path).name).group(1)


def to_numeric_where_possible(df):
    """Convert each column to numbers if every non-empty value is numeric."""
    df = df.replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # keep the conversion only if nothing that had a value was lost
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def read_csv_robust(path, verbose=True):
    """
    Read a CSV. If some rows have more fields than the header (a text value
    with an unquoted comma - happens in the hydrometry table), merge the
    surplus fields back into the last column so no catchment is dropped.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            break                                    # go to the repair below

    with open(path, encoding="latin-1", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    n = len(header)
    bad = [(i + 2, r) for i, r in enumerate(body) if len(r) != n]
    if verbose:
        print(f"  !! {len(bad)} rows repaired (had more than {n} fields): "
              f"file lines {[b[0] for b in bad]}")
    fixed = [r[:n - 1] + [",".join(r[n - 1:])] if len(r) > n else r + [""] * (n - len(r))
             for r in body]
    return to_numeric_where_possible(pd.DataFrame(fixed, columns=header))


def load_attributes(verbose=True):
    """All CAMELS-GB v2 attribute tables, keyed by short name (e.g. 'topographic')."""
    attrs = {}
    for f in sorted((DATA_DIR / "Catchment_Attributes").glob("*.csv")):
        df = read_csv_robust(f, verbose=verbose)
        key = f.stem.replace("camels_gb_v2_", "").replace("_attributes", "")
        attrs[key] = df
        if verbose:
            print(f"### {key:<16} {df.shape[0]:>4} rows x {df.shape[1]:>3} columns")
    return attrs
