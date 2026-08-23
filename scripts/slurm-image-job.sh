#!/bin/bash
#SBATCH --job-name=gdz-image-retrieval
#SBATCH -p scc-gpu
#SBATCH -G A100:1
#SBATCH -t 04:00:00
#SBATCH --mail-type=all
#SBATCH -o slurm-%j.out

module purge
module load miniforge3 gcc cuda

source activate base

# Hugging Face caches on fast storage, not $HOME
export HF_HOME="$WORK/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/.cache/huggingface/transformers"
export HF_HUB_OFFLINE=0

cd "$WORK/workspaces/query-agent-benchmarking" || cd "$SLURM_SUBMIT_DIR"
# If own-proj/ is moved, update the cd path above to point to your repo root.

python -u own-proj/run-image-retrieval.py \
    --model openai/clip-vit-base-patch32 \
    --max-docs 3021 \
    --max-queries 180 \
    --batch-size 64 \
    --output-dir console/results
