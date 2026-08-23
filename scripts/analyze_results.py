"""Analyze standalone retrieval results for presentation-ready failure modes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def hit(query: dict, k: int) -> bool:
    gold = {str(value) for value in query["ground_truth_ids"]}
    return any(value in gold for value in query["retrieved_ids"][:k])


def metrics(result: dict) -> dict:
    queries = result["queries"]
    return {
        f"recall_at_{k}": round(sum(hit(query, k) for query in queries) / len(queries), 3)
        for k in (1, 5, 20)
    } | {"queries": len(queries), "documents": result["metadata"]["num_documents"]}


def paper_breakdown(result: dict) -> dict:
    groups = defaultdict(list)
    for query in result["queries"]:
        paper = str(query["ground_truth_ids"][0]).split("_", 1)[0]
        groups[paper].append(query)
    return {
        paper: {
            "queries": len(queries),
            "recall_at_1": round(sum(hit(query, 1) for query in queries) / len(queries), 3),
            "recall_at_5": round(sum(hit(query, 5) for query in queries) / len(queries), 3),
            "recall_at_20": round(sum(hit(query, 20) for query in queries) / len(queries), 3),
        }
        for paper, queries in sorted(groups.items())
    }


def failure_modes(result: dict) -> dict:
    modes = defaultdict(list)
    for query in result["queries"]:
        h1, h5, h20 = (hit(query, k) for k in (1, 5, 20))
        if h1:
            mode = "hit_at_1"
        elif h5:
            mode = "found_at_5_not_1"
        elif h20:
            mode = "found_at_20_not_5"
        else:
            mode = "missed_at_20"
        modes[mode].append(query["query_id"])
    return {mode: {"count": len(ids), "query_ids": ids} for mode, ids in sorted(modes.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=Path("console/results/gdz-text-local.json"))
    parser.add_argument("--image", type=Path, default=Path("console/results/gdz-image-full-local-v2.json"))
    parser.add_argument("--output", type=Path, default=Path("own-proj/results/analysis.json"))
    args = parser.parse_args()

    text = load(args.text)
    image = load(args.image)
    report = {
        "text": {
            "metadata": text["metadata"],
            "metrics": metrics(text),
            "paper_breakdown": paper_breakdown(text),
            "failure_modes": failure_modes(text),
        },
        "image_clip_baseline": {
            "metadata": image["metadata"],
            "metrics": metrics(image),
            "paper_breakdown": paper_breakdown(image),
            "failure_modes": failure_modes(image),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value["metrics"] for key, value in report.items()}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
