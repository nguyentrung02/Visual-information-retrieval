#!/usr/bin/env python
"""Generate visualizations from GDZ retrieval benchmark results.

Reads the JSON files produced by run_search_eval and creates diagnostic plots:

  1. recall_bar.png      — Recall@1/5/20 (and nDCG@10) bar chart
  2. rank_dist.png       — Histogram of ground-truth rank positions
  3. query_latency.png   — Boxplot of per-query latency
  4. success_heatmap.png — Per-query hit/miss at k=1,5,20,50

Usage:
    python scripts/plot_results.py \
        --results console/results/gdz-image-clip-vit-base-patch32-trial-1.json \
        [--metrics console/results/gdz-image-clip-vit-base-patch32-trial-1-metrics.json] \
        [--outdir plots/]

If --metrics is omitted, the script re-derives Recall@K from the trial JSON.
Multiple --results files can be given; labels are taken from the filename.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_trial(path):
    with open(path) as f:
        data = json.load(f)
    queries = data["queries"]
    return queries, data.get("metadata", {})


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def compute_recall_at_k(queries, k):
    scores = []
    for q in queries:
        gold = set(q["ground_truth_ids"])
        retrieved = q["retrieved_ids"][:k]
        hits = len(gold & set(retrieved))
        denom = max(len(gold), len(retrieved[:k])) if retrieved[:k] else 1
        scores.append(hits / denom if denom else 0.0)
    return scores


def compute_rank(queries):
    """Return list of 0-based ranks of the first ground-truth hit per query."""
    ranks = []
    for q in queries:
        gold = set(q["ground_truth_ids"])
        for idx, rid in enumerate(q["retrieved_ids"]):
            if rid in gold:
                ranks.append(idx)
                break
        else:
            ranks.append(len(q["retrieved_ids"]))
    return ranks


def plot_recall_bar(queries_list, labels, outdir):
    ks = [1, 5, 20]
    n = len(queries_list)
    x = np.arange(len(ks))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (qs, label) in enumerate(zip(queries_list, labels)):
        means = []
        stds = []
        for k in ks:
            scores = compute_recall_at_k(qs, k)
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall")
    ax.set_title("Recall@K — Image Retrieval Comparison")
    ax.legend()
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(outdir / "recall_bar.png", dpi=150)


def plot_rank_histogram(queries_list, labels, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for qs, label in zip(queries_list, labels):
        ranks = compute_rank(qs)
        # Use log scale bins for rank distribution
        max_rank = max(ranks) if ranks else 1
        bins = np.logspace(0, np.log10(max_rank + 1), 30)
        ax.hist(ranks, bins=bins, alpha=0.6, label=label, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Ground-truth rank (log scale)")
    ax.set_ylabel("Number of queries")
    ax.set_title("Rank Distribution — Where the correct image falls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "rank_dist.png", dpi=150)
    plt.close(fig)
    print(f"  → rank_dist.png")


def plot_query_latency(queries_list, labels, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))

    datasets = []
    for qs, label in zip(queries_list, labels):
        latencies = [q["time_taken"] for q in qs]
        datasets.append(latencies)

    bp = ax.boxplot(datasets, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set2.colors[: len(labels)]):
        patch.set_facecolor(color)
    ax.set_ylabel("Query latency (s)")
    ax.set_title("Per-Query Latency (single GPU, batch_size=64)")
    fig.tight_layout()
    fig.savefig(outdir / "query_latency.png", dpi=150)
    plt.close(fig)
    print(f"  → query_latency.png")


def plot_success_heatmap(queries_list, labels, outdir):
    """Heatmap: for each query (row), whether the ground truth is in the
    top-k retrieved (colored cell) at k = 1, 5, 20, 50."""
    ks = [1, 5, 20, 50]
    max_rows = max(len(qs) for qs in queries_list)
    n = len(queries_list)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, max(6, max_rows / 8)), squeeze=False)
    for col, (qs, label) in enumerate(zip(queries_list, labels)):
        matrix = np.zeros((len(qs), len(ks)))
        for i, q in enumerate(qs):
            gold = set(q["ground_truth_ids"])
            for j, k in enumerate(ks):
                retrieved = set(q["retrieved_ids"][:k])
                matrix[i, j] = 1.0 if gold & retrieved else 0.0
        ax = axes[0, col]
        sns_ok = matrix.shape[0] > 1
        ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"@{k}" for k in ks])
        ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel("Query index")
            ax.set_yticks([])
    fig.suptitle("Per-Query Success at k=1/5/20/50", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "success_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"  → success_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True,
                        help="Path(s) to *-trial-1.json files")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Optional metrics JSON files (for nDCG etc.)")
    parser.add_argument("--outdir", default="plots", type=Path)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    queries_list = []
    labels = []

    for path_str in args.results:
        path = Path(path_str)
        qs, meta = load_trial(path)
        label = meta.get("agent_name", path.stem.replace("-trial-1", ""))
        queries_list.append(qs)
        labels.append(label)
        print(f"Loaded {len(qs)} queries from {path.name} ({label})")

    print("\nGenerating plots:")
    plot_recall_bar(queries_list, labels, outdir)
    plot_rank_histogram(queries_list, labels, outdir)
    plot_query_latency(queries_list, labels, outdir)
    plot_success_heatmap(queries_list, labels, outdir)
    print("\nAll done!")


if __name__ == "__main__":
    main()
