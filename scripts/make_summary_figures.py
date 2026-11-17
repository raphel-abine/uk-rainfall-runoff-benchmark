"""
make_summary_figures.py - Two summary figures for the benchmark report.

Values are the test-period (1998-2008) results printed by the evaluation
notebooks (Cells 20 and 26). Re-running those cells regenerates the inputs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d6d5d0"
COL = {"no_snow": "#eb6834", "snow": "#eda100", "lstm": "#1baf7a"}   # same as case studies

def tidy(ax):
    """Recessive grid, no top/right spines."""
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

# ---------------------------------------------------------------------------
# Figure 1 - Snow experiment: the 8 snowiest catchments (Option A forcing)
# ---------------------------------------------------------------------------
snow = [  # name, snow fraction, best GR (no snow), best GR + CemaNeige, LSTM ensemble
    ("Muick at Invermuick",         0.17, 0.708, 0.842, 0.850),
    ("Findhorn at Shenachie",       0.15, 0.544, 0.722, 0.810),
    ("Avon at Delnashaugh",         0.15, 0.546, 0.735, 0.813),
    ("Dee at Woodend",              0.13, 0.655, 0.824, 0.875),
    ("Dulnain at Balnaan Bridge",   0.11, 0.629, 0.761, 0.830),
    ("Prosen Water at Prosen Br.",  0.10, 0.755, 0.832, 0.853),
    ("Ericht at Craighall",         0.09, 0.759, 0.824, 0.871),
    ("Meig at Glenmeanie",          0.08, 0.760, 0.827, 0.880),
]
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(snow))[::-1]
for yi, (name, fs, g, gs, l) in zip(y, snow):
    ax.plot([g, l], [yi, yi], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(g,  yi, s=70, color=COL["no_snow"], zorder=3, edgecolor="white", lw=1.5)
    ax.scatter(gs, yi, s=70, color=COL["snow"],    zorder=3, edgecolor="white", lw=1.5)
    ax.scatter(l,  yi, s=70, color=COL["lstm"],    zorder=3, edgecolor="white", lw=1.5)
ax.set_yticks(y)
ax.set_yticklabels([f"{s[0]}  ({s[1]:.2f})" for s in snow], color=INK)
ax.set_xlim(0.5, 0.9)
ax.set_xlabel("Test-period NSE (1998-2008)", color=INK2)
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=COL[k], label=lab) for k, lab in
           [("no_snow", "Best GR model, no snow module"),
            ("snow", "Best GR model + CemaNeige snow module"),
            ("lstm", "LSTM ensemble")]]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3, fontsize=9)
ax.set_title("Snow module closes most of the LSTM's lead in the snowiest catchments\n"
             "(catchment, snow fraction of precipitation)", loc="left", fontsize=11, color=INK)
tidy(ax)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(OUT / "snow_experiment.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# Figure 2 - Sensitivity to forcing data: median NSE and KGE, Option A vs B
# ---------------------------------------------------------------------------
models = ["GR4J", "GR6J", "CemaNeige-GR4J", "CemaNeige-GR6J", "LSTM ensemble"]
nse = {"A": [0.863, 0.863, 0.868, 0.869, 0.885], "B": [0.851, 0.851, 0.864, 0.866, 0.878]}
kge = {"A": [0.870, 0.873, 0.878, 0.882, 0.872], "B": [0.862, 0.861, 0.880, 0.881, 0.861]}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
y = np.arange(len(models))[::-1]
for ax, (metric, vals) in zip(axes, [("NSE", nse), ("KGE", kge)]):
    for yi, a, b in zip(y, vals["A"], vals["B"]):
        ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(a, yi, s=70, color=INK, zorder=3)
        ax.scatter(b, yi, s=70, facecolor="white", edgecolor=INK, lw=1.8, zorder=3)
    ax.set_xlim(0.84, 0.895)
    ax.set_xlabel(f"Median test-period {metric}", color=INK2)
    ax.set_title(metric, loc="left", fontsize=11, color=INK)
    tidy(ax)
axes[0].set_yticks(y)
axes[0].set_yticklabels(models, color=INK)
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=INK,
                      label="Option A: CEH-GEAR rainfall + CHESS-PE"),
           plt.Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="white",
                      markeredgecolor=INK, label="Option B: HadUK-Grid rainfall + Hydro-PE")]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=9)
fig.suptitle("Changing the input data shifts all models slightly; the ranking on NSE holds",
             x=0.01, ha="left", fontsize=12, color=INK)
plt.tight_layout(rect=(0, 0.08, 1, 0.95))
plt.savefig(OUT / "forcing_sensitivity.png", dpi=200)
plt.close()
print("Saved:", sorted(p.name for p in OUT.glob("*.png")))
