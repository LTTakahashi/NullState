#!/bin/bash
# =============================================================================
# NullState — Deploy Local Files to RunPod
# Run this FROM your WSL/local machine.
#
# Usage:
#   bash scripts/deploy_to_cloud.sh <RUNPOD_SSH_ADDRESS> [SSH_PORT]
#
# Example:
#   bash scripts/deploy_to_cloud.sh root@209.74.123.45 22222
#
# The SSH address and port are shown in your RunPod dashboard under
# "Connect" → "SSH over exposed TCP".
# =============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/deploy_to_cloud.sh <SSH_USER@HOST> [SSH_PORT]"
    echo ""
    echo "Find your SSH address in the RunPod dashboard:"
    echo "  Pod → Connect → SSH over exposed TCP"
    echo "  It looks like: ssh root@209.74.123.45 -p 22222"
    exit 1
fi

REMOTE="$1"
PORT="${2:-22}"
REMOTE_DIR="/workspace/NullState"

echo "=============================================="
echo "  NullState — Deploying to Cloud"
echo "  Remote: $REMOTE (port $PORT)"
echo "=============================================="

# --- 1. Upload code (lightweight) ---
echo "[1/4] Uploading project code..."
rsync -avz --progress \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='*.h5ad' \
    ./ "$REMOTE:$REMOTE_DIR/"

echo "  ✓ Code uploaded"

# --- 2. Run cloud setup ---
echo "[2/4] Running cloud environment setup..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "$REMOTE" \
    "cd $REMOTE_DIR && bash scripts/cloud_setup.sh"

echo "  ✓ Environment configured"

# --- 3. Upload data files ---
echo "[3/4] Uploading data files (this may take a while for large .h5ad files)..."

# Only upload files that exist
for f in data/hnoca.h5ad data/hdbca.h5ad data/cao_fetal.h5ad; do
    if [ -f "$f" ]; then
        echo "  Uploading $f..."
        rsync -avz --progress \
            -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
            "$f" "$REMOTE:$REMOTE_DIR/$f"
    else
        echo "  ⚠ $f not found locally, skipping"
    fi
done

# Upload built reference if it exists
if [ -f "results/pilot/reference.h5ad" ]; then
    echo "  Uploading reference.h5ad..."
    ssh -p "$PORT" -o StrictHostKeyChecking=no "$REMOTE" "mkdir -p $REMOTE_DIR/results/pilot/"
    rsync -avz --progress \
        -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        results/pilot/reference.h5ad "$REMOTE:$REMOTE_DIR/results/pilot/reference.h5ad"
fi

echo "  ✓ Data uploaded"

# --- 4. Final verification ---
echo "[4/4] Verifying deployment..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "$REMOTE" "
    cd $REMOTE_DIR
    echo 'Files on remote:'
    ls -lh data/ 2>/dev/null || echo '  (no data files yet)'
    ls -lh results/pilot/reference.h5ad 2>/dev/null || echo '  (no reference yet)'
    echo ''
    echo 'Python verification:'
    python3 -c 'import scvi; import torch; print(f\"scvi={scvi.__version__}, CUDA={torch.cuda.is_available()}\")'
"

echo ""
echo "=============================================="
echo "  ✓ Deployment complete!"
echo ""
echo "  To start the pilot, SSH into the pod and run:"
echo "    ssh -p $PORT $REMOTE"
echo "    cd $REMOTE_DIR"
echo "    python scripts/run_pilot.py --skip-to-step 2"
echo "=============================================="
