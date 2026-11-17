# =============================================================================
# 01 - Download CAMELS-GB v2 from the EIDC (UKCEH)
#
# Walks the dataset folders and downloads every file (11.4 GB).
# Needs an EIDC Personal Access Token: set EIDC_TOKEN, or paste it when asked.
# Files already downloaded are skipped, so it can be re-run after a break.
#
# Run:   python 01_download_camels_gb_v2.py
# Paths: set the environment variable RRB_ROOT to the project folder
#        (default: the Google Drive folder used in Colab) - see config.py
# =============================================================================
import os
from config import PROJECT_ROOT  # noqa: F401

# =============================================================================
# Download ALL of CAMELS-GB v2 (UKCEH / EIDC) in Google Colab -> Google Drive
# Walks through the dataset's folders and downloads every file inside them.
# DOI: 10.5285/9a46d428-958f-4ac1-86eb-94eee70c0955
# =============================================================================
# Attribution (must be used in all outputs):
# "Contains data supplied by UK Centre for Ecology & Hydrology, British
#  Geological Survey, Environment Agency, Natural Resources Wales and
#  Scottish Environmental Protection Agency."
# =============================================================================

import re
import shutil
import getpass
from pathlib import Path
from collections import Counter
from urllib.parse import urljoin, urlparse, unquote

import requests

# --- Settings ----------------------------------------------------------------
DRY_RUN = False           # True = only list folders/files; False = download
SKIP_FOLDERS = []         # empty = download every folder
DATASET_ID = "9a46d428-958f-4ac1-86eb-94eee70c0955"
BASE_URL = f"https://catalogue.ceh.ac.uk/datastore/eidchub/{DATASET_ID}/"

# --- Connect Google Drive so files survive after Colab closes ----------------
OUT_DIR = (PROJECT_ROOT / "data/camels_gb_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Token (asked once per session; reused if the cell is run again) ---------
token = os.environ.get("EIDC_TOKEN") or getpass.getpass("Paste your EIDC Personal Access Token: ")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {token}"})

# --- Walk the folder tree and collect file URLs ------------------------------
def list_page(url):
    """Return (subfolder_urls, file_urls) found on one datastore page."""
    r = session.get(url, timeout=60)
    r.raise_for_status()
    folders, files = set(), set()
    for href in re.findall(r'href="([^"#]+)"', r.text):
        if "?" in href:                       # skip sort/order links
            continue
        full = urljoin(url, href)
        # keep only links that stay inside this folder and go deeper
        if not full.startswith(url) or len(full) <= len(url):
            continue
        if full.endswith("/"):
            folders.add(full)
        elif not full.endswith((".css", ".js", ".ico", ".png", ".svg")):
            files.add(full)
    return folders, files

all_files, to_visit, seen = [], [BASE_URL], set()
while to_visit:
    url = to_visit.pop()
    if url in seen:
        continue
    seen.add(url)
    rel = unquote(url[len(BASE_URL):])
    if any(rel.startswith(s.strip("/") + "/") for s in SKIP_FOLDERS):
        print(f"Skipping folder: {rel}")
        continue
    folders, files = list_page(url)
    all_files.extend(sorted(files))
    to_visit.extend(sorted(folders))

# --- Summary per folder ------------------------------------------------------
rel_paths = [unquote(urlparse(f).path).split(DATASET_ID + "/", 1)[1] for f in all_files]
counts = Counter(p.rsplit("/", 1)[0] if "/" in p else "(top level)" for p in rel_paths)
print(f"\n{len(rel_paths)} files to get from {len(counts)} folders:")
for folder, n in sorted(counts.items()):
    print(f"  {folder:<50} {n:>6} files")

# --- Check free space on Google Drive before starting ------------------------
free_gb = shutil.disk_usage(OUT_DIR).free / 1e9
print(f"\nFree space at download location: {free_gb:,.1f} GB")

# --- Download (only when DRY_RUN = False) ------------------------------------
if not DRY_RUN:
    failed = []
    for i, (url, rel) in enumerate(zip(all_files, rel_paths), 1):
        dest = OUT_DIR / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue                                  # already downloaded
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with session.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
        except Exception as e:                        # keep going if one file fails
            failed.append((rel, str(e)))
            if dest.exists():
                dest.unlink()                         # remove partial file
        print(f"\r[{i}/{len(all_files)}] {rel[:70]:<70}", end="")

    # --- Report total size downloaded ----------------------------------------
    total_gb = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file()) / 1e9
    print(f"\nDone. {total_gb:,.2f} GB in: {OUT_DIR}")
    if failed:
        print(f"{len(failed)} files failed - run the cell again to retry them:")
        for rel, err in failed[:10]:
            print("  ", rel, "->", err)
