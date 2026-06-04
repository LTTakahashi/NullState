#!/bin/bash
# =============================================================================
# NullState — Cloud Instance Setup Script
# Run this ON the RunPod instance after SSH-ing in.
# Usage: bash cloud_setup.sh
# =============================================================================

set -e

echo "=============================================="
echo "  NullState Cloud Environment Setup"
echo "=============================================="

# --- 1. System-level dependencies ---
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git rsync htop tmux > /dev/null 2>&1
echo "  ✓ System packages installed"

# --- 2. Python environment ---
echo "[2/6] Setting up Python environment..."

# RunPod templates usually have conda pre-installed
if command -v conda &> /dev/null; then
    echo "  Found conda, creating nullstate environment..."
    conda create -n nullstate python=3.11 -y -q
    # Write activation to bashrc so it persists
    echo "conda activate nullstate" >> ~/.bashrc
    eval "$(conda shell.bash hook)"
    conda activate nullstate
else
    echo "  No conda found, using system Python + venv..."
    python3 -m venv /workspace/nullstate_env
    source /workspace/nullstate_env/bin/activate
    echo "source /workspace/nullstate_env/bin/activate" >> ~/.bashrc
fi

# --- 3. Install dependencies ---
echo "[3/6] Installing Python packages..."

# Core ML / single-cell stack
pip install -q --upgrade pip
pip install -q \
    "scanpy>=1.10" \
    "scvi-tools>=1.1" \
    "anndata>=0.10" \
    "torch>=2.0" \
    numpy scipy pandas matplotlib seaborn

# Data retrieval
pip install -q cellxgene-census requests

# Gene sets & enrichment
pip install -q gseapy decoupler

# Graph clustering
pip install -q leidenalg igraph louvain

# Utilities
pip install -q pyyaml rich tqdm jsonschema

# Ontology
pip install -q obonet pronto

# Testing
pip install -q pytest pytest-cov

echo "  ✓ All Python packages installed"

# --- 4. Verify GPU ---
echo "[4/6] Verifying GPU access..."
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  ⚠ WARNING: No GPU detected!')
"

# --- 5. Verify scvi-tools ---
echo "[5/6] Verifying scvi-tools..."
python3 -c "
import scvi
print(f'  scvi-tools version: {scvi.__version__}')
import scanpy as sc
print(f'  scanpy version: {sc.__version__}')
import anndata as ad
print(f'  anndata version: {ad.__version__}')
print('  ✓ All imports successful')
"

# --- 6. Create workspace structure ---
echo "[6/6] Setting up workspace..."
mkdir -p /workspace/NullState/data
mkdir -p /workspace/NullState/results/pilot

echo ""
echo "=============================================="
echo "  ✓ Cloud setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Upload your data files to /workspace/NullState/data/"
echo "  2. Upload your code to /workspace/NullState/"
echo "  3. cd /workspace/NullState && python scripts/run_pilot.py --skip-to-step 2"
echo "=============================================="
