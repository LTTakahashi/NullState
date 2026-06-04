#!/bin/bash
# Download all 3 datasets using aria2c (parallel connections) then run the pipeline.
set -e
cd /workspace/NullState
source /workspace/nullstate_env/bin/activate

echo "=== NullState H100 Data Download + Pipeline ==="
echo "Started at: $(date)"
echo ""

# --- Download HNOCA from Zenodo (18 GB) with 16 parallel connections ---
if [ -f data/hnoca.h5ad ] && [ $(stat -c%s data/hnoca.h5ad 2>/dev/null || echo 0) -gt 18000000000 ]; then
    echo ">>> hnoca.h5ad already exists ($(du -h data/hnoca.h5ad | cut -f1)). Skipping."
else
    echo ">>> Downloading HNOCA from Zenodo with aria2c (16 connections)..."
    rm -f data/hnoca.h5ad
    aria2c -x 16 -s 16 -k 10M --file-allocation=none \
        -d data -o hnoca.h5ad \
        "https://zenodo.org/records/14161275/files/hnoca_cleanedmeta.h5ad"
fi
echo ""

# --- Download Cao Fetal Atlas via Census ---
if [ -f data/cao_fetal.h5ad ] && [ $(stat -c%s data/cao_fetal.h5ad 2>/dev/null || echo 0) -gt 4000000000 ]; then
    echo ">>> cao_fetal.h5ad already exists ($(du -h data/cao_fetal.h5ad | cut -f1)). Skipping."
else
    echo ">>> Downloading Cao Fetal Atlas via Census..."
    python scripts/download_census2.py
fi
echo ""

# --- Download HDBCA via Census ---
if [ -f data/hdbca.h5ad ] && [ $(stat -c%s data/hdbca.h5ad 2>/dev/null || echo 0) -gt 10000000000 ]; then
    echo ">>> hdbca.h5ad already exists ($(du -h data/hdbca.h5ad | cut -f1)). Skipping."
else
    echo ">>> Downloading HDBCA via Census..."
    python scripts/download_hdbca.py
fi
echo ""

echo ">>> All data files:"
ls -lh data/
echo ""

# --- Build reference if needed ---
if [ -f results/pilot/reference.h5ad ]; then
    echo ">>> Reference already exists. Validating..."
    python scripts/validate_reference.py
else
    echo ">>> Building reference..."
    python scripts/run_pilot.py --skip-to-step 1 --force --config-dir config/
    echo ">>> Validating reference..."
    python scripts/validate_reference.py
fi
echo ""

# --- Run full pipeline Step 2+ ---
echo ">>> Starting training pipeline (Steps 2-4) with checkpoints..."
echo ">>> GPU training starting at: $(date)"
python scripts/run_pilot.py --skip-to-step 2 --config-dir config/

echo ""
echo "=== Pipeline complete at: $(date) ==="
