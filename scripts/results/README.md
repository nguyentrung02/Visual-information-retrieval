# Experiment Results

The raw generated result files remain under `scripts/results/` so the
original console can display them.

The presentation-ready derived analysis is generated here:

```powershell
python scripts/analyze_results.py `
  --text scripts/results/gdz-text-local.json `
  --image scripts/results/gdz-image-v4-tiles-prompt-nocenter-trial-1.json `
  --output scripts/results/analysis.json
```

Files:

- `analysis.json`: aggregate metrics, per-paper breakdowns, and failure-mode query IDs.
- `failure-modes.md`: concise interpretation for the presentation.

The CLIP image file is a free baseline, not the final visual-document method.
