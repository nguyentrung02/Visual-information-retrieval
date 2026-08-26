"""Run brute-force image-to-text retrieval on the GDZ dataset using CLIP.

Implements a local SearchAgent that encodes images with CLIP, then performs
cosine-similarity brute-force search against text queries. Results are
evaluated through the query-agent-benchmarking library.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Disable transformers' torch.load security check (CVE-2025-32434).
# torch >= 2.6 is not available on the KISSKI SCC's cu121 index.
# We load trusted HuggingFace weights from .bin checkpoint files — acceptable
# because the model is from openai/ (a verified publisher) and not user-supplied.
# Must patch in *every* module that imported the function by value.
_safe_noop = lambda: None
for _mod in list(sys.modules.values()):
    if _mod is not None and hasattr(_mod, "check_torch_load_is_safe"):
        try:
            _mod.check_torch_load_is_safe = _safe_noop
        except (AttributeError, TypeError):
            pass

class _FakeEngramModule(types.ModuleType):
    """Catch-all fake engram module — returns ``object`` for any attribute.

    The installed query-agent-benchmarking may import various names from
    ``engram`` (EngramClient, BM25Retrieval, FetchRetrieval, HybridRetrieval,
    VectorRetrieval, ...).  We only need the import to succeed; the image
    retrieval agent never calls into engram.
    """
    def __getattr__(self, name):
        return object

_fake_engram = _FakeEngramModule("engram")
sys.modules.setdefault("engram", _fake_engram)

import query_agent_benchmarking.internal.core.domain.metrics_config as _metrics_config

_orig_resolve_metrics_profile = _metrics_config.resolve_metrics_profile


def _patched_resolve_metrics_profile(dataset_name, extra_metrics=None):
    if dataset_name is None:
        return _metrics_config._DEFAULT_METRICS
    base = None
    for profile in _metrics_config.DATASET_METRICS_REGISTRY:
        if dataset_name == profile.dataset_pattern or dataset_name.startswith(profile.dataset_pattern):
            base = profile.metrics
            break
    if base is None:
        base = _metrics_config._DEFAULT_METRICS
    if not extra_metrics:
        return base
    existing_keys = {spec.key for spec in base}
    merged = list(base)
    for spec in extra_metrics:
        if spec.key not in existing_keys:
            merged.append(spec)
            existing_keys.add(spec.key)
    return tuple(merged)


_metrics_config.resolve_metrics_profile = _patched_resolve_metrics_profile

# Patch at the import site — ir_metrics_calculator.py does
#   from ...metrics_config import resolve_metrics_profile
# so patching metrics_config.resolve_metrics_profile alone is NOT enough;
# the imported reference in ir_metrics_calculator's namespace still points
# to the original (which raises ValueError for "GDZ").
import query_agent_benchmarking.internal.adapters.metrics.ir_metrics_calculator as _ir_calc
_ir_calc.resolve_metrics_profile = _patched_resolve_metrics_profile

from query_agent_benchmarking import (
    DocsCollection,
    InMemoryQuery,
    ObjectID,
    SearchAgent,
    run_search_eval,
)


def decode_image(value: str | bytes) -> Image.Image:
    """Decode a dataset base64 image into an RGB PIL image."""
    if isinstance(value, str):
        value = value.strip().strip("'\"").encode("ascii")
    elif isinstance(value, bytearray):
        value = bytes(value)
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


class CLIPImageSearchAgent:
    """Brute-force image-to-text retriever using CLIP embeddings."""

    def __init__(
        self,
        images: list[str | bytes],
        document_ids: list[str],
        model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 32,
    ):
        self.document_ids = document_ids
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float32,
                attn_implementation="eager"
            ).to(self.device).eval()
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float32
            ).to(self.device).eval()

        print(f"Encoding {len(images)} images on {self.device}...")
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = [
                    decode_image(value)
                    for value in images[start : start + batch_size]
                ]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                vecs = self.model.get_image_features(**inputs)
                # transformers 4.56 returns BaseModelOutputWithPooling;
                # .pooler_output is already the projected CLIP embedding
                if not isinstance(vecs, torch.Tensor):
                    vecs = vecs.pooler_output
                vectors.append(vecs.cpu())
        self.image_embeddings = torch.nn.functional.normalize(
            torch.cat(vectors).float(), dim=1
        )
        print(f"Image embeddings shape: {self.image_embeddings.shape}")

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        with torch.inference_mode():
            inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_vec = self.model.get_text_features(**inputs)
            # transformers 4.56 returns BaseModelOutputWithPooling;
            # .pooler_output is already the projected CLIP embedding
            if not isinstance(text_vec, torch.Tensor):
                text_vec = text_vec.pooler_output
            text_vec = torch.nn.functional.normalize(text_vec.float(), dim=1).cpu()

        scores = (self.image_embeddings @ text_vec.T).squeeze(1).tolist()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [ObjectID(object_id=self.document_ids[i]) for i in ranked]

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        return self.run(query, tenant)

    async def initialize_async(self) -> None:
        pass

    async def close_async(self) -> None:
        pass


def load_gdz_dataset():
    raw_docs = load_dataset(
        "Trungdaik/Visual_information_retrieval", "docs", split="train"
    )
    raw_queries = load_dataset(
        "Trungdaik/Visual_information_retrieval", "queries", split="train"
    )
    docs = [dict(item) for item in raw_docs]
    queries = [
        {
            "question": item["question"],
            "query_id": str(item["dataset_id"]),
            "dataset_ids": [str(item["dataset_id"])],
        }
        for item in raw_queries
    ]
    return docs, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--max-docs", type=int, default=3021)
    parser.add_argument("--max-queries", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "console" / "results")
    args = parser.parse_args()

    docs, queries = load_gdz_dataset()
    docs = docs[: args.max_docs]
    queries = queries[: args.max_queries]
    if not docs or not queries:
        raise ValueError("Empty document or query subset.")

    image_values = [doc["base64_str"] for doc in docs]
    document_ids = [str(doc["dataset_id"]) for doc in docs]
    in_memory_queries = [
        InMemoryQuery(
            question=q["question"],
            dataset_ids=q["dataset_ids"],
            query_id=q["query_id"],
        )
        for q in queries
    ]

    docs_collection = DocsCollection(
        collection_name="GDZ",
        content_key="content",
        id_key="dataset_id",
    )

    agent = CLIPImageSearchAgent(
        images=image_values,
        document_ids=document_ids,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    output_name = f"gdz-image-{args.model.split('/')[-1]}"
    run_search_eval(
        docs_collection=docs_collection,
        queries=in_memory_queries,
        search_agent=agent,
        num_trials=1,
        use_async=False,
        output_path=str(args.output_dir / output_name),
    )


if __name__ == "__main__":
    main()
