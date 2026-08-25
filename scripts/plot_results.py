#!/usr/bin/env python
"""Generate visualizations from GDZ retrieval benchmark results.

Reads either trial JSON files (*-trial-1.json) or metrics JSON files
(*-trial-1-metrics.json) and creates diagnostic plots:

  1. recall_bar.png      — Recall@1/5/20 bar chart comparing all methods
  2. rank_dist.png       — Histogram of ground-truth rank positions
  3. query_latency.png   — Boxplot of per-query latency
  4. success_heatmap.png — Per-query hit/miss at k=1,5,20,50

Usage:
    python scripts/plot_results.py \
        --results console/results/gdz-image-clip-scc-results.json \
        --results console/results/gdz-bm25-search-1-20260704-231542-results-trial-1.json \
        --results console/results/gdz-hybrid-search-1-20260704-231324-results-trial-1.json \
        --results console/results/gdz-vector-search-1-20260704-231740-results-trial-1.json \
        [--metrics console/results/gdz-*-trial-1-metrics.json ...] \
        [--outdir plots/]

If --metrics files are provided, pre-computed Recall@K and query_times
are used (preferred, handles Recall@100 correctly even when trial JSON
only stores top-20 retrieved_ids).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ResultLoader:
    """Load trial JSON and optional metrics JSON, provide unified access."""

    def __init__(self, trial_path, metrics_path=None):
        with open(trial_path) as f:
            self.trial = json.load(f)
        self.queries = self.trial.get("queries", [])
        self.meta = self.trial.get("metadata", {})
        if metrics_path:
            with open(metrics_path) as f:
                self.metrics = json.load(f)
        else:
            self.metrics = None
        self.label = self._infer_label(trial_path)

    def _infer_label(self, path):
        name = Path(path).stem
        if "image" in name or "clip" in name:
            return "CLIP"
        if "bm25" in name:
            return "BM25"
        if "hybrid" in name:
            return "Hybrid"
        if "vector" in name:
            return "Vector"
        return name

    def get_recall_scores(self, k):
        """Return per-query recall score at k."""
        if self.metrics:
            key = f"recall_at_{k}_scores"
            if key in self.metrics:
                return self.metrics[key]
        return [self._recall_at_k(q, k) for q in self.queries]

    def get_avg_recall(self, k):
        scores = self.get_recall_scores(k)
        return np.mean(scores) if scores else 0.0

    def get_query_times(self):
        if self.metrics and "query_times" in self.metrics:
            return self.metrics["query_times"]
        return [q.get("time_taken", 0.0) for q in self.queries]

    @staticmethod
    def _recall_at_k(query, k):
        gold = set(query["ground_truth_ids"])
        retrieved = set(query["retrieved_ids"][:k])
        hits = len(gold & retrieved)
        return hits / len(gold) if len(gold) > 0 else 0.0

    def get_ranks(self):
        """Return 0-based rank of first ground-truth hit per query.
        -1 if not found in available retrieved_ids."""
        ranks = []
        for q in self.queries:
            gold = set(q["ground_truth_ids"])
            found = -1
            for idx, rid in enumerate(q["retrieved_ids"]):
                if rid in gold:
                    found = idx
                    break
            ranks.append(found)
        return ranks


def plot_recall_bar(loaders, outdir):
    ks = [1, 5, 20]
    n = len(loaders)
    x = np.arange(len(ks))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, ldr in enumerate(loaders):
        means = []
        stds = []
        for k in ks:
            scores = ldr.get_recall_scores(k)
            means.append(np.mean(scores) if scores else 0.0)
            stds.append(np.std(scores) if len(scores) > 1 else 0.0)
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=ldr.label, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall")
    ax.set_title("Recall@K — Image Retrieval Comparison")
    ax.legend()
    ax.set_ylim(0, 0.35)
    fig.tight_layout()
    fig.savefig(outdir / "recall_bar.png", dpi=150)
    plt.close(fig)
    print("  -> recall_bar.png")


def plot_rank_histogram(loaders, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ldr in loaders:
        ranks = ldr.get_ranks()
        found_ranks = [r for r in ranks if r >= 0]
        if not found_ranks:
            ax.hist([], bins=20, alpha=0.6, label=f"{ldr.label} (0 found)")
            continue
        max_rank = max(found_ranks)
        bins = np.linspace(0, max(max_rank, 20), 30)
        ax.hist(found_ranks, bins=bins, alpha=0.6, label=f"{ldr.label} ({len(found_ranks)} found)")

    ax.set_xlabel("Ground-truth rank (0-based)")
    ax.set_ylabel("Number of queries")
    ax.set_title("Rank Distribution — where the correct image falls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "rank_dist.png", dpi=150)
    plt.close(fig)
    print("  -> rank_dist.png")


def plot_query_latency(loaders, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    datasets = []
    labels = []
    for ldr in loaders:
        times = ldr.get_query_times()
        if times:
            datasets.append(times)
            labels.append(ldr.label)

    if not datasets:
        print("  -> query_latency.png (skipped: no timing data)")
        return

    bp = ax.boxplot(datasets, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set2.colors[: len(labels)]):
        patch.set_facecolor(color)
    ax.set_ylabel("Query latency (s)")
    ax.set_title("Per-Query Latency")
    fig.tight_layout()
    fig.savefig(outdir / "query_latency.png", dpi=150)
    plt.close(fig)
    print("  -> query_latency.png")


def plot_success_heatmap(loaders, outdir):
    ks = [1, 5, 20, 50]
    n = len(loaders)
    max_rows = max(len(ldr.queries) for ldr in loaders)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, max(6, max_rows / 8)), squeeze=False)
    for col, ldr in enumerate(loaders):
        qs = ldr.queries
        matrix = np.zeros((len(qs), len(ks)))
        for i, q in enumerate(qs):
            gold = set(q["ground_truth_ids"])
            for j, k in enumerate(ks):
                retrieved = set(q["retrieved_ids"][:k])
                matrix[i, j] = 1.0 if gold & retrieved else 0.0
        ax = axes[0, col]
        ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"@{k}" for k in ks])
        ax.set_title(ldr.label, fontsize=10)
        if col == 0:
            ax.set_ylabel("Query index")
            ax.set_yticks([])
    fig.suptitle("Per-Query Success at k=1/5/20/50", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "success_heatmap.png", dpi=120)
    plt.close(fig)
    print("  -> success_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True,
                        help="Path to a *-trial-1.json trial result file (repeat for multiple)")
    parser.add_argument("--metrics", action="append", default=None,
                        help="Optional metrics JSON file (auto-discovered if omitted)")
    parser.add_argument("--outdir", default="plots", type=Path)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    loaders = []
    metrics_map = {}
    if args.metrics:
        for mpath in args.metrics:
            stem = Path(mpath).stem
            trial_stem = stem.replace("-metrics", "")
            metrics_map[trial_stem] = mpath

    for path_str in args.results:
        path = Path(path_str)
        trial_stem = path.stem
        metrics_path = metrics_map.get(trial_stem)
        if not metrics_path:
            auto = path.parent / f"{trial_stem}-metrics.json"
            if auto.exists():
                metrics_path = str(auto)
        ldr = ResultLoader(str(path), metrics_path)
        loaders.append(ldr)
        print(f"Loaded {len(ldr.queries)} queries from {path.name} ({ldr.label})")

    print("\nRecall summary:")
    for ldr in loaders:
        r1 = ldr.get_avg_recall(1)
        r5 = ldr.get_avg_recall(5)
        r20 = ldr.get_avg_recall(20)
        print(f"  {ldr.label:<12} R@1={r1*100:.2f}%  R@5={r5*100:.2f}%  R@20={r20*100:.2f}%")

    print("\nGenerating plots:")
    plot_recall_bar(loaders, outdir)
    plot_rank_histogram(loaders, outdir)
    plot_query_latency(loaders, outdir)
    plot_success_heatmap(loaders, outdir)
    print("\nAll done!")


if __name__ == "__main__":
    main()
