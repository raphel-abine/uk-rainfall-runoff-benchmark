# =============================================================================
# 09 - Quick check of one trained LSTM seed (optional)
#
# Validation log, test-period metrics vs GR4J/GR6J, and saved simulations.
#
# Run:   python 09_check_lstm_seed.py
# Needs: train_lstm.py (seed 1, option A), 07
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# CELL 14 - Check one LSTM seed: validation log, test-period metrics, save sims
# Same metric function as GR4J/GR6J (neuralhydrology calculate_all_metrics)
# =============================================================================
import pickle
from pathlib import Path
import pandas as pd
from neuralhydrology.nh_run import eval_run
from neuralhydrology.evaluation.metrics import calculate_all_metrics

ROOT = PROJECT_ROOT
RUN_DIR = ROOT / "outputs/lstm/runs"
SIM_DIR = ROOT / "outputs/lstm/sim"
SEED, EPOCH = 1, 30

run = sorted(RUN_DIR.glob(f"lstm_optionA_seed{SEED}_*"))[-1]       # latest run folder
print("Run folder:", run.name)

# --- Step 1: validation scores written during training -----------------------
print("\nValidation (1975-80, 100 random catchments):")
for line in (run / "output.log").read_text().splitlines():
    if "validation" in line.lower() and "nse" in line.lower():
        print("  ", line.split(": ", 1)[-1])

# --- Step 2: run the trained model on the test period (128 catchments) -------
res_file = run / "test" / f"model_epoch{EPOCH:03d}" / "test_results.p"
if not res_file.exists():
    eval_run(run_dir=run, period="test", epoch=EPOCH)
with open(res_file, "rb") as fh:
    results = pickle.load(fh)

# --- Step 3: metrics per catchment + save simulated series -------------------
out_dir = SIM_DIR / f"seed{SEED}"
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
for gid, res in results.items():
    ds = res["1D"]["xr"].isel(time_step=-1)                      # predict_last_n = 1
    obs, sim = ds["discharge_spec_obs"], ds["discharge_spec_sim"]
    pd.DataFrame({"date": pd.to_datetime(ds["date"].values),
                  "q_obs": obs.values, "q_sim": sim.values}).to_csv(out_dir / f"{gid}.csv", index=False)
    rows.append({"gauge_id": str(gid), **calculate_all_metrics(obs, sim, resolution="1D")})

lstm = pd.DataFrame(rows)
lstm["model"] = f"LSTM_seed{SEED}"
print(f"\nTest catchments evaluated: {len(lstm)}")

# --- Step 4: compare with GR4J and GR6J (test period, same 128 catchments) ---
gr = pd.read_csv(ROOT / "outputs/gr_models/gr_metrics_optionA.csv", dtype={"gauge_id": str})
gr = gr[gr["period"] == "test"]
cols = ["NSE", "KGE", "Alpha-NSE", "Beta-KGE", "FHV", "FMS", "FLV", "Peak-Timing"]
both = pd.concat([gr[["model", "gauge_id"] + cols], lstm[["model", "gauge_id"] + cols]])
print("\nTest-period medians (1998-2008):")
print(both.groupby("model")[cols].median().round(3).T.to_string())

# catchment-by-catchment NSE: how often does this single seed beat each GR model?
piv = both.pivot(index="gauge_id", columns="model", values="NSE")
for m in ["GR4J", "GR6J"]:
    print(f"LSTM seed {SEED} NSE > {m} in {(piv[f'LSTM_seed{SEED}'] > piv[m]).sum()}/{len(piv)} catchments")

lstm.to_csv(ROOT / f"outputs/lstm/metrics_seed{SEED}_test.csv", index=False)
