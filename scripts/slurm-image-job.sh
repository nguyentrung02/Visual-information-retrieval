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
# Results land in:                    $SCRIPT_DIR/results/
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fallback: if run-image-retrieval.py not found via BASH_SOURCE
# (happens when sbatch script is in /tmp/ which isn't on compute nodes)
if [ ! -f "$SCRIPT_DIR/run-image-retrieval.py" ]; then
    for _fallback in \
        "${HOME}/Visual-information-retrieval/scripts" \
        "${PROJECT}/workspaces/Visual-information-retrieval/scripts"; do
        if [ -f "$_fallback/run-image-retrieval.py" ]; then
            SCRIPT_DIR="$_fallback"
            break
        fi
    done
fi

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
# Ablation knobs (override via env vars when submitting)
# ---------------------------------------------------------------------------
TILES="${SLURM_TILES:-3}"
NO_CENTER="${SLURM_NO_CENTER:-1}"
PROMPT="${SLURM_PROMPT:-1}"
OUTPUT_NAME="${SLURM_OUTPUT_NAME:-gdz-image-v4-tiles-prompt-nocenter}"

# Build the --no-center and --prompt-template flags conditionally
CENTER_FLAG=""
if [ "$NO_CENTER" = "1" ] || [ "$NO_CENTER" = "true" ]; then
    CENTER_FLAG="--no-center"
fi

PROMPT_FLAG=""
if [ "$PROMPT" != "0" ] && [ "$PROMPT" != "false" ]; then
    PROMPT_FLAG='--prompt-template "a scanned page of a scientific paper about {query}"'
fi

# ---------------------------------------------------------------------------
# Run the brute-force CLIP image retrieval
# Uses cosine similarity — no Weaviate required at runtime.
# ---------------------------------------------------------------------------
python -u "$SCRIPT_DIR/run-image-retrieval.py" \
    --model openai/clip-vit-large-patch14 \
    --max-docs 3021 \
    --max-queries 180 \
    --batch-size 64 \
    --tiles "$TILES" \
    $CENTER_FLAG \
    $PROMPT_FLAG \
    --output-name "$OUTPUT_NAME" \
    --output-dir "$SCRIPT_DIR/results"
