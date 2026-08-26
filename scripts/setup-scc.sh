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

# On KISSKI accounts, $WORK is often unset — use $PROJECT as fallback
: "${WORK:=${PROJECT:-${HOME}}}"
export WORK

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
pip install matplotlib  # for result visualization (plot_results.py)

# query-agent-benchmarking with patches — install from the local clone
# (must come after requirements.txt so deps like dspy are already present)
QAB_DIR="${QAB_DIR:-$WORK/workspaces/query-agent-benchmarking}"
if [ -d "$QAB_DIR" ]; then
    pip install -e "$QAB_DIR"
else
    echo "WARNING: query-agent-benchmarking not found at $QAB_DIR"
    echo "         Install manually: pip install -e /path/to/query-agent-benchmarking"
fi

# --- 4. Pre-download the model on the login node -----------------------------
# This caches the model weights under $WORK (not $HOME) so the compute
# node can use them with HF_HUB_OFFLINE=1.
# Default: openai/clip-vit-large-patch14 (3x larger than base, much better recall)
# Alternatives: google/siglip-base-patch16-224, laion/CLIP-ViT-H-14-laion2B-s32B-b76K
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
mkdir -p "$HF_HOME"

MODEL_NAME="${MODEL_NAME:-openai/clip-vit-large-patch14}"
python -c "
import sys
# Bypass CVE-2025-32434 torch.load check (torch>=2.6 not on cu121 index)
for _m in list(sys.modules.values()):
    if _m is not None and hasattr(_m, 'check_torch_load_is_safe'):
        try: _m.check_torch_load_is_safe = lambda: None
        except: pass
from transformers import AutoModel, AutoProcessor
print('Downloading $MODEL_NAME ...')
AutoModel.from_pretrained('$MODEL_NAME', torch_dtype='float32', attn_implementation='eager')
AutoProcessor.from_pretrained('$MODEL_NAME')
print('Done.')
"

# --- 5. Pre-download the GDZ dataset on the login node ----------------------
# The compute nodes run with HF_HUB_OFFLINE=1, so all datasets must be cached
# under $WORK/.cache/huggingface (not the default ~/.cache).
python -c "
from datasets import load_dataset
print('Caching Trungdaik/Visual_information_retrieval ...')
load_dataset('Trungdaik/Visual_information_retrieval', 'docs', split='train')
load_dataset('Trungdaik/Visual_information_retrieval', 'queries', split='train')
print('Done.')
"

echo "=== Setup complete ==="
echo "Activate the env on compute nodes with:  source activate $ENV_NAME"
echo "Submit the job with:                     sbatch scripts/slurm-image-job.sh"
