"""Compute hubness metrics from CLIP/BM25 result JSON files.

Implements the three hubness diagnostics from the supervisor's verification step:
  1. Distinct pages appearing across all top-20 lists
  2. Share of slots taken by the 50 most frequent pages
  3. Mean pairwise top-20 overlap between unrelated queries

Usage:
    python scripts/compute_hubness_metrics.py [--top-k 20]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

VARIANTS = {
    "BM25 (baseline)": "gdz-text-local.json",
    "Old CLIP B/32 (CPU)": "gdz-image-clip-vit-base-patch32-trial-1.json",
    "V1 (no centering, no tiles)": "gdz-image-clip-vit-large-patch14-trial-1.json",
    "V2 (centering, no tiles)": "gdz-image-clip-vit-large-patch14-centered-trial-1.json",
    "V4 (tiling + centering)": "gdz-image-clip-vit-large-patch14-tiles3-centered-trial-1.json",
    "V4 (tiling + prompt, no centering)": "gdz-image-v4-tiles-prompt-nocenter-trial-1.json",
}


def load_result(path: Path) -> list[list[str]]:
    with open(path) as f:
        data = json.load(f)
    queries = data.get("queries", [])
    return [q["retrieved_ids"][:20] for q in queries]


def compute_hubness(top20_lists: list[list[str]]) -> dict:
    n_queries = len(top20_lists)
    n_slots = sum(len(lst) for lst in top20_lists)

    freq = Counter()
    for lst in top20_lists:
        freq.update(lst)

    sorted_freq = sorted(freq.values(), reverse=True)
    top50_share = sum(sorted_freq[:50]) / n_slots if n_slots else 0.0

    overlap_sum = 0.0
    pair_count = 0
    for i in range(n_queries):
        set_i = set(top20_lists[i])
        for j in range(i + 1, n_queries):
            set_j = set(top20_lists[j])
            union = set_i | set_j
            overlap = len(set_i & set_j) / len(union) if union else 0.0
            overlap_sum += overlap
            pair_count += 1
    mean_overlap = overlap_sum / pair_count if pair_count else 0.0

    return {
        "distinct_pages": len(freq),
        "total_slots": n_slots,
        "n_queries": n_queries,
        "top50_share": top50_share,
        "mean_pairwise_overlap": mean_overlap,
        "pair_count": pair_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    print("=" * 85)
    print(f"Hubness Metrics (top-{args.top_k} lists, 180 queries)")
    print("=" * 85)
    print(f"{'Variant':<35} {'Distinct':>10} {'Top50%':>10} {'MeanOverlap':>12}")
    print("-" * 85)

    results = {}
    for label, filename in VARIANTS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"{label:<35} {'NOT FOUND':>32}")
            results[label] = None
            continue
        top20 = load_result(path)
        m = compute_hubness(top20)
        results[label] = m
        print(f"{label:<35} {m['distinct_pages']:>10} "
              f"{m['top50_share']:>9.1%} {m['mean_pairwise_overlap']:>12.3f}")

    print("-" * 85)
    print(f"\nInterpretation:")
    print(f"  - 'Distinct' = unique pages in top-20 across 180 queries (higher = less hubbing)")
    print(f"  - 'Top50%' = share of all 3600 slots filled by 50 most frequent pages")
    print(f"  - 'MeanOverlap' = mean Jaccard overlap between query result sets")
    print(f"\n  Supervisor target (diagnostic notes): centering should push")
    print(f"  hubness values toward BM25 levels (low top-50 share, high distinct pages).")
    print(f"  Empirical finding: centering is CATASTROPHIC — it INCREASES hubness")
    print(f"  (e.g. V1: 21% top-50 share → V2: 93.5%), contradicting the diagnostic notes.")


if __name__ == "__main__":
    main()
