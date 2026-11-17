# =============================================================================
# 13 - Case-study hydrographs (Findhorn, Itchen, Thet, Gelyn)
#
# One water year (linear and log scale) and flow duration curves for
# observed flow, GR6J, CemaNeige-GR6J and the LSTM ensemble.
#
# Run:   python 13_case_study_hydrographs.py
# Needs: 12
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 21 - Case-study hydrographs: observed vs GR6J, CemaNeige-GR6J, LSTM ensemble
# Per catchment: (a) one water year, linear  (b) same year, log scale
#                (c) flow duration curves for the whole test period 1998-2008
# Water year: snowy catchments -> largest spring (Mar-May) flow; others -> largest peak
# Each catchment is wrapped in try/except so any error is printed in full
# =============================================================================
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = PROJECT_ROOT
CMP_DIR = ROOT / "outputs/comparison"
OUT = CMP_DIR / "case_studies"
OUT.mkdir(parents=True, exist_ok=True)
TEST = ("1998-01-01", "2008-12-31")
FULL_WATER_YEARS = range(1999, 2009)        # 1998/99 ... 2007/08 (complete Oct-Sep years)
SNOWY = 0.05                                # frac_snow threshold for choosing a melt year

# --- Case studies (edit freely) ----------------------------------------------
CASES = {
    "7001":  "Snow-influenced (Cairngorms)",
    "42016": "Chalk, groundwater-dominated",
    "33019": "Largest GR advantage (dry, permeable)",
    "67010": "Small, flashy upland (Wales)",
}
# Optional manual override, e.g. {"7001": 2001} (2001 = Oct 2000 - Sep 2001)
WATER_YEAR = {}

# --- Models in legend order; colours fixed and CVD-checked -------------------
MODELS = {
    "GR6J":          {"label": "GR6J",           "color": "#eb6834", "ls": "-",
                      "file": lambda g: ROOT / f"outputs/gr_models/sim/GR6J_{g}.csv"},
    "CemaNeigeGR6J": {"label": "CemaNeige-GR6J", "color": "#eda100", "ls": (0, (4, 2)),
                      "file": lambda g: ROOT / f"outputs/gr_models/sim/CemaNeigeGR6J_{g}.csv"},
    "LSTM_ensemble": {"label": "LSTM ensemble",  "color": "#1baf7a", "ls": "-",
                      "file": lambda g: ROOT / f"outputs/lstm/sim/ensemble/{g}.csv"},
}
DRAW_ORDER = ["GR6J", "LSTM_ensemble", "CemaNeigeGR6J"]   # dashed line drawn last, on top
OBS_COLOR = "#0b0b0b"
NOTE_COLOR = "#52514e"

metrics = pd.read_csv(CMP_DIR / "metrics_test_all_models.csv", dtype={"gauge_id": str})
attrs = (pd.read_csv(ROOT / "outputs/catchment_selection/eligible_catchments.csv", dtype={"gauge_id": str})
           .set_index("gauge_id"))
clim = (pd.read_csv(ROOT / "data/camels_gb_v2/Catchment_Attributes/camels_gb_v2_climatic_attributes.csv",
                    dtype={"gauge_id": str}).set_index("gauge_id"))

def load_series(gid):
    """Observed + simulated daily flow (mm/day) for the test period, aligned by date."""
    out = None
    for key, m in MODELS.items():
        d = pd.read_csv(m["file"](gid), parse_dates=["date"]).set_index("date").loc[TEST[0]:TEST[1]]
        if out is None:
            out = d[["q_obs"]].copy()
        out[key] = d["q_sim"]
    return out

def water_year(index):
    """Water year label: Oct-Dec belong to the following year (Oct 2000 -> 2001)."""
    return (index + pd.offsets.DateOffset(months=3)).year

def choose_year(df, snow_frac):
    """Snowy: year with the largest mean Mar-May observed flow. Otherwise: largest peak."""
    wy = pd.Series(water_year(df.index), index=df.index)
    keep = wy.isin(FULL_WATER_YEARS)
    obs, wy = df.loc[keep, "q_obs"], wy[keep]
    if snow_frac >= SNOWY:
        spring = obs[obs.index.month.isin([3, 4, 5])]
        return int(spring.groupby(wy[spring.index]).mean().idxmax()), "largest spring (Mar-May) flow"
    return int(wy[obs.idxmax()]), "largest observed peak"

def fdc(q):
    """Flow duration curve: exceedance probability (%) and flows sorted high to low."""
    q = np.sort(q.dropna().to_numpy())[::-1]
    return 100 * np.arange(1, len(q) + 1) / (len(q) + 1), q

def style(ax):
    """Recessive grid and axes."""
    ax.grid(alpha=0.25, lw=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

def plot_lines(ax, x_obs, y_obs, series, label_fn=None):
    """Draw observed + models; return handles in legend order (Observed, then MODELS order)."""
    handles = {"obs": ax.plot(x_obs, y_obs, color=OBS_COLOR, lw=2, label="Observed")[0]}
    for k in DRAW_ORDER:
        x, y = series[k]
        v = MODELS[k]
        handles[k] = ax.plot(x, y, color=v["color"], ls=v["ls"], lw=1.6,
                             label=label_fn(k) if label_fn else v["label"])[0]
    return [handles["obs"]] + [handles[k] for k in MODELS]

summary = []
for gid, story in CASES.items():
    try:
        df = load_series(gid)
        a = attrs.loc[gid]
        snow_frac = float(clim.loc[gid, "frac_snow"])

        # --- choose the water year to display --------------------------------
        if gid in WATER_YEAR:
            year, why = WATER_YEAR[gid], "chosen manually"
        else:
            year, why = choose_year(df, snow_frac)
        sel = df[water_year(df.index) == year]

        m = metrics[metrics["gauge_id"] == gid].set_index("model")
        same_as_gr6j = abs(m.loc["CemaNeigeGR6J", "NSE"] - m.loc["GR6J", "NSE"]) < 0.005

        fig = plt.figure(figsize=(13, 11.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.42)

        # (a) linear hydrograph - floods and timing
        ax1 = fig.add_subplot(gs[0])
        h = plot_lines(ax1, sel.index, sel["q_obs"],
                       {k: (sel.index, sel[k]) for k in MODELS},
                       lambda k: f"{MODELS[k]['label']} (NSE {m.loc[k, 'NSE']:.2f}, KGE {m.loc[k, 'KGE']:.2f})")
        ax1.legend(handles=h, frameon=False, ncol=2, fontsize=9, loc="upper right")
        title = f"(a) Water year {year-1}/{str(year)[-2:]} ({why}) - linear scale"
        if same_as_gr6j:
            title += (f"\nCemaNeige-GR6J ≈ GR6J here (snow fraction {snow_frac:.2f}); "
                      "the dashed line overlies GR6J")
        ax1.set_title(title, loc="left", fontsize=11)
        ax1.set_ylabel("Flow (mm/day)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        style(ax1)

        # (b) same year, log scale - recessions and low flows
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        plot_lines(ax2, sel.index, sel["q_obs"], {k: (sel.index, sel[k]) for k in MODELS})
        ax2.set_yscale("log")
        ax2.set_ylabel("Flow (mm/day, log)")
        ax2.set_title("(b) Same year - log scale (recessions and low flows)", loc="left", fontsize=11)
        style(ax2)

        # (c) flow duration curves, whole test period (days with observations only)
        ax3 = fig.add_subplot(gs[2])
        has_obs = df["q_obs"].notna()
        p_obs, q_obs = fdc(df["q_obs"])
        h = plot_lines(ax3, p_obs, q_obs,
                       {k: fdc(df.loc[has_obs, k]) for k in MODELS},
                       lambda k: f"{MODELS[k]['label']} (FHV {m.loc[k, 'FHV']:+.0f}%, FLV {m.loc[k, 'FLV']:+.0f}%)")
        ax3.legend(handles=h, frameon=False, fontsize=9, loc="upper right")
        ax3.set_yscale("log")
        ax3.set_xlabel("Exceedance probability (%)")
        ax3.set_ylabel("Flow (mm/day, log)")
        ax3.set_title("(c) Flow duration curves, test period 1998-2008", loc="left", fontsize=11)
        style(ax3)

        fig.suptitle(f"{a['gauge_name']} ({gid}) - {story}\n"
                     f"Area {a['area']:.0f} km² | BFI {a['baseflow_index']:.2f} | aridity {a['aridity']:.2f} | "
                     f"snow fraction {snow_frac:.2f} | regime {a['regime']}",
                     fontsize=12, x=0.02, ha="left")
        fig.savefig(OUT / f"case_{gid}.png", dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"OK  {gid} {a['gauge_name']}: water year {year-1}/{str(year)[-2:]} ({why}) -> saved")

        summary.append({"gauge_id": gid, "catchment": a["gauge_name"], "story": story,
                        "water_year": f"{year-1}/{str(year)[-2:]}", "why": why,
                        **{f"NSE_{k}": round(m.loc[k, "NSE"], 3) for k in MODELS}})
    except Exception:
        print(f"\nERROR in catchment {gid}:")
        traceback.print_exc()
        plt.close("all")

print()
print(pd.DataFrame(summary).to_string(index=False))
print(f"\nFigures saved in: {OUT}")
