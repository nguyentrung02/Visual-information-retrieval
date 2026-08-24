#!/bin/bash
# ============================================================================
#  One-time setup script for running the GDZ retrieval pipeline on the
#  GWDG / NHR-Nord SCC (Grete GPU nodes).
#
#  Run this ON A LOGIN NODE (e.g.  ssh <your-user>@glogin-gpu.hpc.gwdg.de)
#  before submitting the Slurm job.
#
#  It creates a conda env with all dependencies and pre-downloads the
#  CLIP model so that the compute job can run offline (HF_HUB_OFFLINE=1).
#
#  NOTE: requirements.txt is used for local development.  On the SCC we
#  install packages individually because PyTorch must come from the CUDA
#  index, and query-agent-benchmarking needs a patched local clone.
# ============================================================================
set -euo pipefail

module purge
module load miniforge3 gcc cuda

# --- 1. Create conda environment -------------------------------------------
ENV_NAME="gdz-retrieval"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo ">>> Creating conda env: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.11
fi

source activate "$ENV_NAME"

# --- 2. Install PyTorch (CUDA build) ---------------------------------------
# SCC Grete nodes have A100 GPUs with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- 3. Install all Python dependencies from requirements.txt ---------------
# weaviate-client is required as a transitive dependency of query-agent-benchmarking
# (its __init__.py imports weaviate at module level).  Having the library installed
# does NOT mean Weaviate Cloud is used — run-image-retrieval.py is pure brute-force.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"

# query-agent-benchmarking with patches — install from the local clone
: "${WORK:=${HOME}/work}"   # fallback if $WORK is not set by the module stack
QAB_DIR="${QAB_DIR:-$WORK/workspaces/query-agent-benchmarking}"
if [ -d "$QAB_DIR" ]; then
    pip install -e "$QAB_DIR"
else
    echo "WARNING: query-agent-benchmarking not found at $QAB_DIR"
    echo "         Install manually: pip install -e /path/to/query-agent-benchmarking"
fi

# --- 4. Pre-download the CLIP model on the login node ----------------------
# This caches the model weights under $WORK (not $HOME) so the compute
# node can use them with HF_HUB_OFFLINE=1.
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
mkdir -p "$HF_HOME"

python -c "
from transformers import CLIPModel, CLIPProcessor
print('Downloading openai/clip-vit-base-patch32 ...')
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('Done.')
"

echo "=== Setup complete ==="
echo "Activate the env on compute nodes with:  source activate $ENV_NAME"
echo "Submit the job with:                     sbatch scripts/slurm-image-job.sh"
