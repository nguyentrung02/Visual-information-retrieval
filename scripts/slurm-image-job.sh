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
# Locate the repository on the compute node
# Under Slurm, BASH_SOURCE points to a spooled copy in /var/spool/slurmd,
# so SLURM_SUBMIT_DIR is the reliable path to the original repo.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/..}}"
cd "$REPO_DIR" || exit 1

# ---------------------------------------------------------------------------
# Run the brute-force CLIP image retrieval
# (already uses cosine similarity — no Weaviate required)
# ---------------------------------------------------------------------------
python -u "$REPO_DIR/scripts/run-image-retrieval.py" \
    --model openai/clip-vit-base-patch32 \
    --max-docs 3021 \
    --max-queries 180 \
    --batch-size 64 \
    --output-dir console/results
