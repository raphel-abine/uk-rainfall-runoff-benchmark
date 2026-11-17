# =============================================================================
# 10 - Evaluate the Option A LSTM ensemble and combine with GR results
#
# Runs each of the 8 seeds on the test period, averages them into the
# ensemble, scores everything with the same metric code.
# Saves outputs/comparison/metrics_test_all_models.csv.
#
# Run:   python 10_evaluate_optionA.py
# Needs: 07, train_lstm.py --option A (8 seeds), GPU
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 17 - Evaluate all 8 LSTM seeds on the test period (1998-2008), build the
# ensemble (mean of seed predictions, Lees et al. 2021) and score everything
# with the same metric function used for GR4J/GR6J
# =============================================================================

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from neuralhydrology.nh_run import eval_run
from neuralhydrology.evaluation.metrics import calculate_all_metrics

ROOT = PROJECT_ROOT
RUN_DIR = ROOT / "outputs/lstm/runs"
SIM_DIR = ROOT / "outputs/lstm/sim"
CMP_DIR = ROOT / "outputs/comparison"
CMP_DIR.mkdir(parents=True, exist_ok=True)
SEEDS, EPOCH = range(1, 9), 30

def finished_run(seed):
    """Run folder for this seed that contains the final-epoch weights."""
    runs = [d for d in sorted(RUN_DIR.glob(f"lstm_optionA_seed{seed}_*"))
            if (d / f"model_epoch{EPOCH:03d}.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"No finished run for seed {seed}")
    return runs[-1]

def nh_metrics(df):
    """All neuralhydrology metrics for a DataFrame with date, q_obs, q_sim."""
    dates = pd.to_datetime(df["date"]).values
    obs = xr.DataArray(df["q_obs"].values, coords={"date": dates}, dims=["date"])
    sim = xr.DataArray(df["q_sim"].values, coords={"date": dates}, dims=["date"])
    return calculate_all_metrics(obs, sim, resolution="1D")

# --- Step 1: run each seed on the test period and save its simulations -------
for seed in SEEDS:
    out_dir = SIM_DIR / f"seed{seed}"
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
    print(f"Seed {seed}: {len(results)} catchments evaluated and saved")

# --- Step 2: ensemble = mean of the 8 seed simulations ----------------------
basins = sorted(p.stem for p in (SIM_DIR / "seed1").glob("*.csv"))
ens_dir = SIM_DIR / "ensemble"
ens_dir.mkdir(parents=True, exist_ok=True)
rows = []
for gid in basins:
    seeds_df = [pd.read_csv(SIM_DIR / f"seed{s}" / f"{gid}.csv", parse_dates=["date"]).set_index("date")
                for s in SEEDS]
    # sanity check: every seed was evaluated against the same observations
    assert all(np.allclose(d["q_obs"], seeds_df[0]["q_obs"], equal_nan=True) for d in seeds_df)
    ens = seeds_df[0][["q_obs"]].copy()
    ens["q_sim"] = pd.concat([d["q_sim"] for d in seeds_df], axis=1).mean(axis=1)
    ens.reset_index().to_csv(ens_dir / f"{gid}.csv", index=False)

    rows.append({"model": "LSTM_ensemble", "gauge_id": gid, **nh_metrics(ens.reset_index())})
    for s, d in zip(SEEDS, seeds_df):                            # individual seeds too
        rows.append({"model": f"LSTM_seed{s}", "gauge_id": gid, **nh_metrics(d.reset_index())})

lstm = pd.DataFrame(rows)

# --- Step 3: combine with GR4J / GR6J test-period metrics --------------------
gr = pd.read_csv(ROOT / "outputs/gr_models/gr_metrics_optionA.csv", dtype={"gauge_id": str})
gr = gr[gr["period"] == "test"].drop(columns=["period"])
elig = pd.read_csv(ROOT / "outputs/catchment_selection/eligible_catchments.csv", dtype={"gauge_id": str})
allm = pd.concat([gr, lstm], ignore_index=True)
allm = allm.drop(columns=["gauge_name", "regime", "aridity", "baseflow_index"], errors="ignore")
allm = allm.merge(elig[["gauge_id", "gauge_name", "regime", "aridity", "baseflow_index"]],
                  on="gauge_id", how="left")
allm.to_csv(CMP_DIR / "metrics_test_all_models.csv", index=False)

# --- Step 4: headline medians ------------------------------------------------
cols = ["NSE", "KGE", "Alpha-NSE", "Beta-KGE", "Pearson-r", "FHV", "FMS", "FLV", "Peak-Timing"]
main = allm[allm["model"].isin(["GR4J", "GR6J", "LSTM_ensemble"])]
print(f"\nTest period 1998-2008, {main['gauge_id'].nunique()} catchments - medians:")
print(main.groupby("model")[cols].median().round(3).T.to_string())

seed_nse = allm[allm["model"].str.startswith("LSTM_seed")].groupby("model")["NSE"].median()
print(f"\nSingle-seed median NSE: {seed_nse.min():.3f} - {seed_nse.max():.3f} "
      f"(ensemble: {main.loc[main['model'] == 'LSTM_ensemble', 'NSE'].median():.3f})")
print(f"\nSaved: {CMP_DIR / 'metrics_test_all_models.csv'}")
