# =============================================================================
# 14 - Evaluate all models with Option B inputs (HadUK-Grid + Hydro-PE)
#
# Saves outputs/comparison/metrics_test_optionB.csv.
#
# Run:   python 14_evaluate_optionB.py
# Needs: 06 --option B, train_lstm.py --option B (8 seeds), GPU
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 25 - Option B evaluation: GR4J, GR6J, CemaNeige-GR4J/-GR6J and the 8-seed
# LSTM ensemble on the test period (1998-2008), same metric code as Option A
# =============================================================================

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from neuralhydrology.nh_run import eval_run
from neuralhydrology.evaluation.metrics import calculate_all_metrics

ROOT = PROJECT_ROOT
GR_B = ROOT / "outputs/gr_models_optionB"
RUN_DIR = ROOT / "outputs/lstm/runs"
SIM_B = ROOT / "outputs/lstm/sim_optionB"
CMP_DIR = ROOT / "outputs/comparison"
TEST = ("1998-01-01", "2008-12-31")
GR_MODELS = ["GR4J", "GR6J", "CemaNeigeGR4J", "CemaNeigeGR6J"]
SEEDS, EPOCH = range(1, 9), 30

def nh_metrics(dates, obs, sim):
    """All neuralhydrology metrics for one observed/simulated series."""
    o = xr.DataArray(np.asarray(obs), coords={"date": pd.to_datetime(dates)}, dims=["date"])
    s = xr.DataArray(np.asarray(sim), coords={"date": pd.to_datetime(dates)}, dims=["date"])
    return calculate_all_metrics(o, s, resolution="1D")

# --- Step 1: GR models (Option B) - completeness check and test metrics ------
rows = []
for m in GR_MODELS:
    files = sorted((GR_B / "sim").glob(f"{m}_*.csv"))
    print(f"{m:<15} Option B simulations: {len(files)}/128")
    for f in files:
        gid = f.stem.split("_", 1)[1]
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date").loc[TEST[0]:TEST[1]]
        rows.append({"model": m, "gauge_id": gid, **nh_metrics(d.index, d["q_obs"], d["q_sim"])})

# --- Step 2: evaluate each Option B LSTM seed on the test period -------------
def finished_run(seed):
    runs = [d for d in sorted(RUN_DIR.glob(f"lstm_optionB_seed{seed}_*"))
            if (d / f"model_epoch{EPOCH:03d}.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"No finished Option B run for seed {seed}")
    return runs[-1]

for seed in SEEDS:
    out_dir = SIM_B / f"seed{seed}"
    if out_dir.exists() and len(list(out_dir.glob("*.csv"))) >= 128:
        print(f"Seed {seed}: simulations already saved")
        continue
    run = finished_run(seed)
    res_file = run / "test" / f"model_epoch{EPOCH:03d}" / "test_results.p"
    if not res_file.exists():
        eval_run(run_dir=run, period="test", epoch=EPOCH)
    with open(res_file, "rb") as fh:
        results = pickle.load(fh)
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, res in results.items():
        ds = res["1D"]["xr"].isel(time_step=-1)                  # predict_last_n = 1
        pd.DataFrame({"date": pd.to_datetime(ds["date"].values),
                      "q_obs": ds["discharge_spec_obs"].values,
                      "q_sim": ds["discharge_spec_sim"].values}
                     ).to_csv(out_dir / f"{gid}.csv", index=False)
    print(f"Seed {seed}: {len(results)} catchments evaluated")

# --- Step 3: ensemble = mean of the 8 seeds; metrics for seeds and ensemble --
(SIM_B / "ensemble").mkdir(parents=True, exist_ok=True)
for gid in sorted(p.stem for p in (SIM_B / "seed1").glob("*.csv")):
    seeds_df = [pd.read_csv(SIM_B / f"seed{s}" / f"{gid}.csv", parse_dates=["date"]).set_index("date")
                for s in SEEDS]
    ens = seeds_df[0][["q_obs"]].copy()
    ens["q_sim"] = pd.concat([d["q_sim"] for d in seeds_df], axis=1).mean(axis=1)
    ens.reset_index().to_csv(SIM_B / "ensemble" / f"{gid}.csv", index=False)
    rows.append({"model": "LSTM_ensemble", "gauge_id": gid,
                 **nh_metrics(ens.index, ens["q_obs"], ens["q_sim"])})
    for s, d in zip(SEEDS, seeds_df):
        rows.append({"model": f"LSTM_seed{s}", "gauge_id": gid,
                     **nh_metrics(d.index, d["q_obs"], d["q_sim"])})

# --- Step 4: save and summarise ---------------------------------------------
met_b = pd.DataFrame(rows)
met_b["forcing"] = "B"
met_b.to_csv(CMP_DIR / "metrics_test_optionB.csv", index=False)

MAIN = GR_MODELS + ["LSTM_ensemble"]
cols = ["NSE", "KGE", "Pearson-r", "FHV", "FMS", "FLV", "Peak-Timing"]
print("\nOption B - test-period medians (1998-2008):")
print(met_b[met_b["model"].isin(MAIN)].groupby("model")[cols].median().reindex(MAIN).round(3).T.to_string())
seed_nse = met_b[met_b["model"].str.startswith("LSTM_seed")].groupby("model")["NSE"].median()
print(f"\nSingle-seed median NSE (B): {seed_nse.min():.3f} - {seed_nse.max():.3f}")
