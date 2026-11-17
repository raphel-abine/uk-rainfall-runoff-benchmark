# =============================================================================
# 06 - Calibrate GR4J, GR6J, CemaNeige-GR4J and CemaNeige-GR6J (airGR)
#
# Runs gr_run.R for every model, one catchment at a time with live
# progress. Finished catchments are skipped, so it can be restarted.
# Install airGR in R first:  install.packages("airGR")
#
# Run:   python 06_run_gr_models.py --option A     (and again with --option B)
# Needs: 05, R with airGR (Rscript on the PATH, or set RSCRIPT)
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
import argparse
import os
import subprocess
from pathlib import Path

from config import PROJECT_ROOT

MODELS = ["GR4J", "GR6J", "CemaNeigeGR4J", "CemaNeigeGR6J"]

ap = argparse.ArgumentParser(description="Calibrate the GR models with airGR")
ap.add_argument("--option", choices=["A", "B"], required=True, help="forcing option")
ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
args = ap.parse_args()

rscript = os.environ.get("RSCRIPT", "Rscript")
script = Path(__file__).resolve().parent / "gr_run.R"
for model in args.models:
    print(f"\n===== {model}, forcing option {args.option} =====", flush=True)
    subprocess.run([rscript, str(script), model, args.option, PROJECT_ROOT.as_posix()], check=True)
