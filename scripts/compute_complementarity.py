"""Compute complementary failure analysis: identify queries where BM25 and CLIP
exclusively succeed, validating the IRPAPERS "complementary failure modes" finding.

Reads BM25 (gdz-text-local.json) and CLIP V1 (gdz-image-clip-vit-large-patch14-trial-1.json)
result files from scripts/results/.

Matches queries by question text (BM25 uses "1_8" as query_id while CLIP uses "q0").

Usage:
    python scripts/compute_complementarity.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    queries = data.get("queries", [])
    return {q["question"]: q for q in queries}


def check_hit(query: dict, k: int) -> bool:
    gold = set(query["ground_truth_ids"])
    retrieved = set(query.get("retrieved_ids", [])[:k])
    return bool(gold & retrieved)


def main():
    bm25_path = RESULTS_DIR / "gdz-text-local.json"
    clip_path = RESULTS_DIR / "gdz-image-clip-vit-large-patch14-trial-1.json"

    bm25 = load_results(bm25_path)
    clip = load_results(clip_path)

    common_questions = sorted(set(bm25.keys()) & set(clip.keys()))
    n = len(common_questions)

    results = {
        "text_only": [],
        "image_only": [],
        "both": [],
        "neither": [],
    }

    k = 20
    for q in common_questions:
        bm25_hit = check_hit(bm25[q], k)
        clip_hit = check_hit(clip[q], k)

        if bm25_hit and clip_hit:
            results["both"].append(bm25[q]["query_id"])
        elif bm25_hit and not clip_hit:
            results["text_only"].append(bm25[q]["query_id"])
        elif not bm25_hit and clip_hit:
            results["image_only"].append(clip[q]["query_id"])
        else:
            results["neither"].append(bm25[q]["query_id"])

    total_hit_bm25 = len(results["both"]) + len(results["text_only"])
    total_hit_clip = len(results["both"]) + len(results["image_only"])

    print("=" * 70)
    print(f"Complementary Failure Analysis (Recall@{k}, {n} queries)")
    print("=" * 70)
    print(f"\nBM25 total hits@{k} (exclusive + shared): {total_hit_bm25} ")
    print(f"CLIP V1 total hits@{k} (exclusive + shared): {total_hit_clip}")
    print()
    print(f"  Both succeed (shared):     {len(results['both']):>3d}  ({len(results['both'])/n*100:.1f}%)")
    print(f"  Text only succeeds:        {len(results['text_only']):>3d}  ({len(results['text_only'])/n*100:.1f}%)")
    print(f"  Image only succeeds:       {len(results['image_only']):>3d}  ({len(results['image_only'])/n*100:.1f}%)")
    print(f"  Neither succeeds:          {len(results['neither']):>3d}  ({len(results['neither'])/n*100:.1f}%)")
    print()
    print("IRPAPERS reported: 22 text-exclusive, 18 image-exclusive (out of 180)")
    print()

    with open(RESULTS_DIR / "complementarity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / 'complementarity.json'}")


if __name__ == "__main__":
    main()
