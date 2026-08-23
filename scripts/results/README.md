# Experiment Results

The raw generated result files remain under the repository's existing
`console/results/` directory so the original console can display them.

The presentation-ready derived analysis is generated here:

```powershell
python own-proj/analyze_results.py `
  --text console/results/gdz-text-local.json `
  --image console/results/gdz-image-full-local-v2.json `
  --output own-proj/results/analysis.json
```

Files:

- `analysis.json`: aggregate metrics, per-paper breakdowns, and failure-mode query IDs.
- `failure-modes.md`: concise interpretation for the presentation.

The CLIP image file is a free baseline, not the final visual-document method.
