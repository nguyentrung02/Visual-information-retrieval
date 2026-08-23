"""Compare all available GDZ retrieval result files."""

import argparse
import json
from pathlib import Path


METHODS = {
    "Weaviate BM25": "gdz-bm25-search-1-20260704-231542-results-trial-1.json",
    "Weaviate hybrid": "gdz-hybrid-search-1-20260704-231324-results-trial-1.json",
    "Weaviate vector": "gdz-vector-search-1-20260704-231740-results-trial-1.json",
    "Standalone BM25": "gdz-text-local.json",
    "Standalone CLIP image": "gdz-image-full-local-v2.json",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def hit(query: dict, k: int) -> bool:
    gold = {str(value) for value in query["ground_truth_ids"]}
    return any(str(value) in gold for value in query["retrieved_ids"][:k])


def score(result: dict, k: int) -> float:
    queries = result["queries"]
    return sum(hit(query, k) for query in queries) / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "console" / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "method-comparison.json",
    )
    args = parser.parse_args()

    comparison = {}
    for method, filename in METHODS.items():
        path = args.results_dir / filename
        if not path.exists():
            print(f"Skipping missing result: {path}")
            continue
        result = load(path)
        comparison[method] = {
            "source": str(path),
            "dataset": result["metadata"].get("dataset"),
            "documents": result["metadata"].get("num_documents"),
            "queries": len(result["queries"]),
            "recall_at_1": round(score(result, 1), 3),
            "recall_at_5": round(score(result, 5), 3),
            "recall_at_20": round(score(result, 20), 3),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
