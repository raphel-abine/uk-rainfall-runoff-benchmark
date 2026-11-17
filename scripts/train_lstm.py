# =============================================================================
# train_lstm.py - Train the LSTM ensemble with neuralhydrology
# Protocol follows Lees et al. (2021): cudalstm, hidden size 64, dropout 0.4,
# 365-day input sequence, NSE loss, 30 epochs, 8 random seeds (ensemble mean).
#
# Usage (from a Python environment with neuralhydrology and a CUDA GPU):
#     python train_lstm.py --root <project_root> --option A --seeds 1 2 3 4 5 6 7 8
# project_root must contain data/nh_camels_gb_v2/ (time_series/*.nc, attributes/)
# and outputs/lstm/basins_train.txt and basins_test_128.txt.
# Completed seeds are skipped, so the script can be stopped and restarted.
# =============================================================================
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from neuralhydrology.nh_run import start_run

EPOCHS = 30

# --- Dynamic inputs for each forcing option ----------------------------------
FORCING = {
    "A": ["precipitation_cehgear", "pet_chess", "temperature_chess"],
    "B": ["precipitation_haduk", "pet_hydrope", "temperature_haduk"],
}


def write_config(root, option, seed, workers, static_list):
    """YAML config for one ensemble member (paths written with forward slashes)."""
    nh = (root / "data" / "nh_camels_gb_v2").as_posix()
    run_dir = (root / "outputs" / "lstm" / "runs").as_posix()
    lstm = (root / "outputs" / "lstm").as_posix()
    yml = f"""
# --- experiment ---
experiment_name: lstm_option{option}_seed{seed}
run_dir: {run_dir}
seed: {seed}
device: cuda:0

# --- basins and periods (Lees et al., 2021) ---
train_basin_file: {lstm}/basins_train.txt
validation_basin_file: {lstm}/basins_train.txt
test_basin_file: {lstm}/basins_test_128.txt
train_start_date: '01/01/1988'
train_end_date: '31/12/1997'
validation_start_date: '01/01/1975'
validation_end_date: '31/12/1980'
test_start_date: '01/01/1998'
test_end_date: '31/12/2008'

# --- validation monitoring (does not change training) ---
validate_every: 5
validate_n_random_basins: 100
metrics: [NSE]
save_validation_results: False

# --- model (Lees et al., 2021) ---
model: cudalstm
head: regression
output_activation: linear
hidden_size: 64
initial_forget_bias: 3
output_dropout: 0.4

# --- training ---
optimizer: Adam
loss: NSE
learning_rate:
  0: 1e-3
  10: 5e-4
  25: 1e-4
batch_size: 256
epochs: {EPOCHS}
clip_gradient_norm: 1
predict_last_n: 1
seq_length: 365
num_workers: {workers}
log_interval: 5
log_tensorboard: False
log_n_figures: 0
save_weights_every: 5

# --- data (Option {option} forcing) ---
dataset: generic
data_dir: {nh}
dynamic_inputs: {FORCING[option]}
target_variables: [discharge_spec]
clip_targets_to_zero: [discharge_spec]
static_attributes: {static_list}
"""
    cfg_dir = root / "outputs" / "lstm" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"lstm_option{option}_seed{seed}.yml"
    path.write_text(yml)
    return path


def seed_done(root, option, seed):
    """A seed is complete if a run folder holds the final-epoch weights."""
    run_dir = root / "outputs" / "lstm" / "runs"
    return any((d / f"model_epoch{EPOCHS:03d}.pt").exists()
               for d in run_dir.glob(f"lstm_option{option}_seed{seed}_*"))


def main():
    ap = argparse.ArgumentParser(description="Train LSTM ensemble members (neuralhydrology)")
    ap.add_argument("--root", type=Path, required=True, help="project root folder")
    ap.add_argument("--option", choices=["A", "B"], required=True, help="forcing option")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--workers", type=int, default=2, help="data-loading workers (0 if errors)")
    args = ap.parse_args()

    root = args.root.resolve()
    attr_file = root / "data" / "nh_camels_gb_v2" / "attributes" / "static_attributes.csv"
    assert torch.cuda.is_available(), "PyTorch cannot see a GPU"
    assert attr_file.exists(), f"Converted data not found: {attr_file}"
    (root / "outputs" / "lstm" / "runs").mkdir(parents=True, exist_ok=True)
    static_list = pd.read_csv(attr_file, nrows=1).columns[1:].tolist()
    print(f"GPU: {torch.cuda.get_device_name(0)} | Option {args.option} | seeds {args.seeds}")

    for seed in args.seeds:
        if seed_done(root, args.option, seed):
            print(f"Seed {seed}: already trained - skipped")
            continue
        print(f"\n===== Option {args.option}, seed {seed} - started {time.strftime('%H:%M')} =====")
        t0 = time.time()
        start_run(config_file=write_config(root, args.option, seed, args.workers, static_list))
        print(f"Seed {seed} finished in {(time.time() - t0) / 60:.0f} min")

    print(f"\nOption {args.option} status:",
          {s: ("done" if seed_done(root, args.option, s) else "pending") for s in range(1, 9)})


if __name__ == "__main__":          # required on Windows for multi-worker data loading
    main()
