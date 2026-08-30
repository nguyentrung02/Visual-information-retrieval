# Visual Information Retrieval — Scripts

This directory contains all scripts for the GDZ visual information retrieval
benchmark.  Results are saved to `scripts/results/` (CLIP) and
`console/results/` (text retrieval via Weaviate).

## Files

| File | Purpose |
|------|---------|
| `run-text-retrieval.py` | Text retrieval via Weaviate Cloud (BM25, vector, hybrid) |
| `run-image-retrieval.py` | CLIP image-to-text retrieval (brute-force cosine similarity, local or GPU) |
| `slurm-image-job.sh` | Slurm batch script for SCC Grete GPU nodes |
| `setup-scc.sh` | One-time environment setup script for the SCC login node |
| `analyze_results.py` | Generate per-paper breakdown and failure-mode analysis |
| `compare_methods.py` | Compare metrics across result files |
| `requirements.txt` | Python dependencies |
| `.env.example` | Weaviate Cloud credentials template (text retrieval only) |

## Sub-directories

- `OCR_&_extract_images/` — PDF extraction and Tesseract OCR scripts
- `results/` — presentation-ready analysis output (`analysis.json`, `failure-modes.md`)

## Quick start (local CPU)

```bash
pip install -r requirements.txt
pip install -e /path/to/query-agent-benchmarking   # patched library

# Text retrieval (requires Weaviate Cloud credentials in .env)
python run-text-retrieval.py

# Image retrieval (CPU is slow; 4 min for 3 021 images with CLIP-B/32)
python run-image-retrieval.py
```

## Quick start (SCC GPU)

```bash
# On the SCC login node:
bash setup-scc.sh        # one-time env + model download
source activate gdz-retrieval
sbatch slurm-image-job.sh
squeue --me              # monitor
```

See the main [README.md](../README.md) for full documentation, results, and
references.
