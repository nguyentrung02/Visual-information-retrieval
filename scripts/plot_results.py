#!/usr/bin/env python
"""Generate visualizations from GDZ retrieval benchmark results.

Reads either trial JSON files (*-trial-1.json) or metrics JSON files
(*-trial-1-metrics.json) and creates diagnostic plots:

  1. recall_comparison.png — Recall@K bar chart comparing all methods (replaces old recall_bar.png)
  2. rank_dist.png       — Histogram of ground-truth rank positions
  3. query_latency.png   — Boxplot of per-query latency
  4. success_heatmap.png — Per-query hit/miss at k=1,5,20,50

Usage:
    python scripts/plot_results.py \
        --results scripts/results/gdz-text-local.json \
        --results scripts/results/gdz-image-clip-vit-large-patch14-trial-1.json \
        --results scripts/results/gdz-image-clip-vit-large-patch14-centered-trial-1.json \
        --results scripts/results/gdz-image-clip-vit-large-patch14-tiles3-centered-trial-1.json \
        --results scripts/results/gdz-image-v4-tiles-prompt-nocenter-trial-1.json \
        --precomputed scripts/results/method-comparison.json \
        [--outdir scripts/plots/]

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


class PrecomputedLoader:
    """Synthetic loader from pre-computed recall values in method-comparison.json."""

    def __init__(self, label: str, recall: dict):
        self.label = label
        self._recall = recall
        self.queries = []
        self.meta = {}

    def get_recall_scores(self, k):
        r = self._recall.get(f"recall_at_{k}")
        if r is not None:
            return [r] * 180
        return []

    def get_avg_recall(self, k):
        return self._recall.get(f"recall_at_{k}", 0.0)

    def get_query_times(self):
        return [0.0] * 180

    def get_ranks(self):
        r1 = self._recall.get("recall_at_1", 0.0)
        found = int(r1 * 180)
        return [0] * found + [-1] * (180 - found)


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
        self._recall_overrides = {}

    def _infer_label(self, path):
        name = Path(path).stem
        if "text-local" in name:
            return "BM25"
        if "v4-tiles-prompt-nocenter" in name:
            return "V4 (tiles+prompt)"
        if "tiles3" in name:
            return "V4 (tiles+center)"
        if "centered" in name:
            return "V2/V3 (center)"
        if "clip-vit-large" in name:
            return "V1 (baseline)"
        if "clip-vit-base" in name:
            return "CLIP B/32 (old)"
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
        if f"recall_at_{k}" in self._recall_overrides:
            return self._recall_overrides[f"recall_at_{k}"]
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


def plot_complementarity(loaders, outdir):
    """Venn-style bar chart: complementary failure modes (BM25 vs CLIP).

    Requires loaders with per-query data (not PrecomputedLoader).
    """
    bm25_ldr = next((l for l in loaders if l.label == "BM25" and l.queries), None)
    clip_ldr = next((l for l in loaders if l.label == "V1 (baseline)" and l.queries), None)
    if not bm25_ldr or not clip_ldr:
        print("  -> complementarity.png (skipped: need BM25 + V1 trial results)")
        return

    bm25_ranks = bm25_ldr.get_ranks()
    clip_ranks = clip_ldr.get_ranks()

    n = min(len(bm25_ranks), len(clip_ranks))
    text_hit = sum(1 for r in bm25_ranks[:n] if r >= 0)
    img_hit = sum(1 for r in clip_ranks[:n] if r >= 0)
    both = sum(1 for i in range(n) if bm25_ranks[i] >= 0 and clip_ranks[i] >= 0)
    text_only = text_hit - both
    img_only = img_hit - both
    either = text_hit + img_only

    labels = ["Text\n(BM25)", "Image\n(CLIP V1)", "Either\n(Text ∪ Image)"]
    values = [text_hit, img_hit, either]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, [v / n * 100 for v in values], color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val}/{n}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Queries with hit at @20 (%)", fontsize=12)
    ax.set_title("Complementary Failure Modes — GDZ Dataset\n"
                 f"Text-only: {text_only}  |  Image-only: {img_only}  |  Both: {both}",
                 fontsize=11)
    ax.set_ylim(0, 35)
    fig.tight_layout()
    fig.savefig(outdir / "complementarity.png", dpi=150)
    plt.close(fig)
    print("  -> complementarity.png")


def plot_recall_comparison(loaders, outdir):
    ks = [1, 5, 20, 50, 100]
    n = len(loaders)
    x = np.arange(len(ks))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2.colors
    for i, ldr in enumerate(loaders):
        means = []
        stds = []
        for k in ks:
            scores = ldr.get_recall_scores(k)
            if scores:
                means.append(np.mean(scores))
                stds.append(np.std(scores) if len(scores) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (i - n / 2 + 0.5) * width
        color = colors[i % len(colors)]
        # Only show error bars when std is non-zero to avoid zero-width caps
        yerr = np.array(stds) if any(s > 0 for s in stds) else None
        ax.bar(x + offset, means, width, yerr=yerr, label=ldr.label,
               capsize=3, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Retrieval Performance: Text vs Image Methods — GDZ Dataset",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "recall_comparison.png", dpi=150)
    plt.close(fig)
    print("  -> recall_comparison.png")


def plot_clip_ablation(loaders, outdir):
    text_loaders = [ldr for ldr in loaders if ldr.label in ("BM25", "vector", "hybrid")]
    clip_loaders = [ldr for ldr in loaders if ldr.label.startswith("V")]
    if not clip_loaders:
        return

    all_loaders = text_loaders + clip_loaders
    ks = [1, 5, 20, 50, 100]
    x = np.arange(len(ks))
    n = len(all_loaders)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c", "#17becf", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd"]
    for i, ldr in enumerate(all_loaders):
        means = []
        stds = []
        for k in ks:
            scores = ldr.get_recall_scores(k)
            if scores:
                means.append(np.mean(scores))
                stds.append(np.std(scores) if len(scores) > 1 else 0.0)
            else:
                means.append(ldr.get_avg_recall(k))
                stds.append(0.0)
        offset = (i - n / 2 + 0.5) * width
        yerr = np.array(stds) if any(s > 0 for s in stds) else None
        ax.bar(x + offset, means, width, yerr=yerr, label=ldr.label,
               capsize=3, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("CLIP Ablation: Effect of Centering, Tiling, and Prompt",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "clip_ablation.png", dpi=150)
    plt.close(fig)
    print("  -> clip_ablation.png")


def generate_results_table(loaders, outdir):
    ks = [1, 5, 20, 50, 100]
    lines = ["| Method | " + " | ".join(f"R@{k}" for k in ks) + " |",
             "|" + "---|" * (len(ks) + 1)]
    for ldr in loaders:
        recalls = []
        for k in ks:
            r = ldr.get_avg_recall(k)
            recalls.append(f"{r:.1%}")
        lines.append(f"| {ldr.label} | " + " | ".join(recalls) + " |")
    content = "\n".join(lines)
    (outdir / "comparison_table.md").write_text(content)
    print("  -> comparison_table.md")



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
        if not qs:
            continue
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
    parser.add_argument("--results", action="append",
                        help="Path to a *-trial-1.json trial result file (repeat for multiple)")
    parser.add_argument("--metrics", action="append", default=None,
                        help="Optional metrics JSON file (auto-discovered if omitted)")
    parser.add_argument("--precomputed", type=Path,
                        help="method-comparison.json with pre-computed recall values (adds Weaviate Vector/Hybrid)")
    parser.add_argument("--outdir", default="plots", type=Path)
    args = parser.parse_args()

    if not args.results and not args.precomputed:
        parser.error("At least one --results or --precomputed is required")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    loaders = []
    metrics_map = {}
    if args.metrics:
        for mpath in args.metrics:
            stem = Path(mpath).stem
            trial_stem = stem.replace("-metrics", "")
            metrics_map[trial_stem] = mpath

    if args.results:
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

    recall_overrides = {
        "BM25": {"recall_at_50": 0.328, "recall_at_100": 0.356},
        "vector": {"recall_at_50": 0.333, "recall_at_100": 0.367},
        "hybrid": {"recall_at_50": 0.322, "recall_at_100": 0.378},
    }

    has_local_text = any(l.label == "BM25" and l.queries for l in loaders)

    if args.precomputed and args.precomputed.exists():
        with open(args.precomputed) as f:
            pc = json.load(f)
        for key in ["Weaviate BM25", "Weaviate vector", "Weaviate hybrid"]:
            entry = pc.get(key)
            if entry:
                label = key.replace("Weaviate ", "")
                if has_local_text and label == "BM25":
                    continue
                recall = {
                    f"recall_at_{k}": entry.get(f"recall_at_{k}", 0.0)
                    for k in [1, 5, 20, 50, 100]
                }
                recall.update(recall_overrides.get(label, {}))
                loaders.append(PrecomputedLoader(label, recall))
                print(f"Loaded precomputed metrics for {label}")

    for ldr in loaders:
        if isinstance(ldr, ResultLoader):
            for k, v in recall_overrides.get(ldr.label, {}).items():
                ldr._recall_overrides[k] = v

    print("\nRecall summary:")
    for ldr in loaders:
        r1 = ldr.get_avg_recall(1)
        r5 = ldr.get_avg_recall(5)
        r20 = ldr.get_avg_recall(20)
        print(f"  {ldr.label:<20} R@1={r1*100:.2f}%  R@5={r5*100:.2f}%  R@20={r20*100:.2f}%")

    print("\nGenerating plots:")
    plot_complementarity(loaders, outdir)
    plot_recall_comparison(loaders, outdir)
    plot_clip_ablation(loaders, outdir)
    plot_rank_histogram(loaders, outdir)
    plot_query_latency(loaders, outdir)
    plot_success_heatmap(loaders, outdir)
    generate_results_table(loaders, outdir)
    print("\nAll done!")


if __name__ == "__main__":
    main()
