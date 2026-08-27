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
# Visual-information-retrieval is at: $HOME/Visual-information-retrieval
# query-agent-benchmarking is at:     $PROJECT/workspaces/query-agent-benchmarking
# Results land in:                    <qab>/console/results/
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve QAB dir: explicit override > $PROJECT/workspaces > sibling dir
if [ -n "${QAB_DIR:-}" ]; then
    :
elif [ -d "${PROJECT}/workspaces/query-agent-benchmarking" ]; then
    QAB_DIR="${PROJECT}/workspaces/query-agent-benchmarking"
elif [ -d "../query-agent-benchmarking" ]; then
    QAB_DIR="$(cd "../query-agent-benchmarking" && pwd)"
else
    QAB_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/query-agent-benchmarking"
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
