#!/bin/bash
#SBATCH --job-name=gdz-image-retrieval
#SBATCH -p kisski               # KISSKI GPU partition on Grete (A100 nodes)
#SBATCH --account=kisski-nlpbg    # your SLURM project account (from sbalance)
#SBATCH -G A100:1                # 1× A100 GPU (kisski partition is shared)
#SBATCH -t 06:00:00              # walltime — always set a limit
#SBATCH --mail-type=all
#SBATCH -o %x-%j.out             # %x = job-name, %j = job id

# ---------------------------------------------------------------------------
# Environment setup — follows GWDG HPC best practices (see SKILL.md)
# ---------------------------------------------------------------------------
module purge
module load miniforge3 gcc cuda

# Activate the conda env created by setup-scc.sh on the login node
source activate gdz-retrieval 2>/dev/null || source activate base

# Keep HF caches on fast $PROJECT or $WORK storage, NOT $HOME (60 GiB quota)
: "${WORK:=${PROJECT:-${HOME}}}"
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$WORK/.cache/huggingface/transformers}"
export HF_HUB_OFFLINE=1        # model pre-downloaded on login node
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Locate directories on the compute node.
# This script lives at: $HOME/Visual-information-retrieval/scripts/slurm-image-job.sh
# query-agent-benchmarking is at: $PROJECT/workspaces/query-agent-benchmarking
# Results land in: .../query-agent-benchmarking/console/results/
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:=${PROJECT:-${HOME}}}"

# Try sibling dir first, then fall back to $PROJECT/workspaces/
QAB_DIR="${QAB_DIR:-../query-agent-benchmarking}"
if [ ! -d "$QAB_DIR" ]; then
    QAB_DIR="${PROJECT}/workspaces/query-agent-benchmarking"
fi
cd "$QAB_DIR" || { echo "ERROR: query-agent-benchmarking not found at $QAB_DIR" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Run the brute-force CLIP image retrieval
# Uses cosine similarity — no Weaviate required at runtime.
# ---------------------------------------------------------------------------
python -u "$SCRIPT_DIR/run-image-retrieval.py" \
    --model openai/clip-vit-large-patch14 \
    --max-docs 3021 \
    --max-queries 180 \
    --batch-size 64 \
    --tiles 3 \
    --output-dir console/results
