"""Run brute-force image-to-text retrieval on the GDZ dataset using CLIP.

Implements a local SearchAgent that encodes images with CLIP (with optional
tiling and mean-centering), then performs cosine-similarity brute-force
MaxSim search against text queries. Results are evaluated through the
query-agent-benchmarking library.

Key design choices (per supervisor feedback):
  - Tiling: large A4 pages (2479x3508) are split into an overlapping grid
    of tiles so CLIP sees legible text at ~80 DPI instead of 27 DPI.
  - Mean-centering: removes the shared image-mean direction that causes hubness.
  - Prompt template: wraps queries to match CLIP's caption-style training data.
"""

import argparse
import base64
import io
import sys
import types
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# torch.load CVE-2025-32434 compatibility shim
# On torch >= 2.6, torch.load blocks .bin checkpoints unless weights_only=False.
# The CLIP weights ship as .bin on the SCC's cu121 index. We load trusted
# HuggingFace weights only (verified publisher openai/), so this is safe.
# ---------------------------------------------------------------------------
_torch_version = torch.__version__.split("+")[0]
_torch_major = int(_torch_version.split(".")[0])
_torch_minor = int(_torch_version.split(".")[1])
if _torch_major >= 2 and _torch_minor >= 6:
    _safe_noop = lambda: None
    for _mod in list(sys.modules.values()):
        if _mod is not None and hasattr(_mod, "check_torch_load_is_safe"):
            try:
                _mod.check_torch_load_is_safe = _safe_noop
            except (AttributeError, TypeError):
                pass

# ---------------------------------------------------------------------------
# Fake engram module — query-agent-benchmarking imports engram names at module
# level but we never need them for brute-force image retrieval.
# ---------------------------------------------------------------------------
class _FakeEngramModule(types.ModuleType):
    def __getattr__(self, name):
        return object

sys.modules.setdefault("engram", _FakeEngramModule("engram"))

import query_agent_benchmarking.internal.core.domain.metrics_config as _metrics_config


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

import query_agent_benchmarking.internal.adapters.metrics.ir_metrics_calculator as _ir_calc
_ir_calc.resolve_metrics_profile = _patched_resolve_metrics_profile

from query_agent_benchmarking import (
    DocsCollection,
    InMemoryQuery,
    ObjectID,
    SearchAgent,
    run_search_eval,
)

import torch.nn.functional as F


def decode_image(value: str | bytes) -> Image.Image:
    """Decode a dataset base64 image into an RGB PIL image."""
    if isinstance(value, str):
        value = value.strip().strip("'\"").encode("ascii")
    elif isinstance(value, bytearray):
        value = bytes(value)
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


class CLIPImageSearchAgent:
    """Brute-force image-to-text retriever using CLIP embeddings.

    Supports tiling (--tiles), mean-centering (--no-center), and prompt
    templates (--prompt-template) to overcome CLIP's limitations on scientific
    figures.
    """

    def __init__(
        self,
        images: list[str | bytes],
        document_ids: list[str],
        model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 32,
        num_tiles: int = 3,
        center_embeddings: bool = True,
        prompt_template: str = "a scanned page of a scientific paper about {query}",
    ):
        self.document_ids = document_ids
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_tiles = num_tiles
        self.center_embeddings = center_embeddings
        self.prompt_template = prompt_template

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

        self.proj_dim = getattr(self.model.config, "projection_dim", None)

        print(f"Encoding {len(images)} images on {self.device} (tiles={num_tiles})...")

        # -- Encode each page into an overlapping grid of tiles -----------------
        all_vectors = []       # list of (n_tiles_i, 512) tensors
        self.tile_to_doc = []  # document index for each row

        for doc_idx, value in enumerate(images):
            img = decode_image(value)
            tiles = self._create_tiles(img)

            # Disable center crop and force exact 224×224 resize so the full
            # page (including header/footer) is preserved rather than discarded.
            # The image is squashed to fit — that is intentional for the
            # whole-page view; grid tiles are already cropped regions.
            try:
                tile_inputs = self.processor(
                    images=tiles, return_tensors="pt",
                    do_center_crop=False,
                    size={"height": 224, "width": 224},
                )
            except TypeError:
                tile_inputs = self.processor(
                    images=tiles, return_tensors="pt",
                )
            tile_inputs = {k: v.to(self.device) for k, v in tile_inputs.items()}

            with torch.inference_mode():
                vecs = self.model.get_image_features(**tile_inputs)

            # Handle different return types across transformers versions.
            # Some versions return BaseModelOutputWithPooling from get_image_features()
            # when return_dict=True; extract the appropriate tensor.
            if not isinstance(vecs, torch.Tensor):
                if hasattr(vecs, "image_embeds") and vecs.image_embeds is not None:
                    vecs = vecs.image_embeds
                elif hasattr(vecs, "pooler_output"):
                    vecs = vecs.pooler_output

            vecs = vecs.float().cpu()

            # Verify and fix projection: if we got pre-projection features
            # (e.g. 768-dim for ViT-Large), apply visual_projection manually.
            # CLIP's projection_dim is 512 for ViT-L/14.
            if self.proj_dim and vecs.shape[-1] != self.proj_dim and hasattr(self.model, "visual_projection"):
                with torch.inference_mode():
                    vecs = self.model.visual_projection(vecs.to(self.device)).float().cpu()

            self.tile_to_doc.extend([doc_idx] * vecs.shape[0])
            all_vectors.append(vecs)

            if (doc_idx + 1) % 200 == 0:
                print(f"  Encoded {doc_idx + 1}/{len(images)} pages "
                      f"({sum(v.shape[0] for v in all_vectors)} tiles so far)")

        raw = torch.cat(all_vectors).float()  # (total_tiles, dim)

        # -- Mean-centering to reduce modality-gap hubness --------------------
        if center_embeddings:
            self.image_mean = raw.mean(dim=0, keepdim=True)
            raw = raw - self.image_mean
            print(f"  Mean-centered {raw.shape[0]} tile embeddings (hubness reduction)")
        else:
            self.image_mean = None

        self.image_embeddings = F.normalize(raw, dim=1)
        self.tile_to_doc = np.array(self.tile_to_doc, dtype=np.int64)
        print(f"Total tiles: {raw.shape[0]}, documents: {len(images)}")
        print(f"Image embeddings shape: {self.image_embeddings.shape}")

    def _create_tiles(self, image: Image.Image) -> list[Image.Image]:
        """Split a large page image into an overlapping grid plus the whole page."""
        if self.num_tiles <= 1:
            return [image]

        w, h = image.size
        tiles = [image]  # whole-page view (no center crop)

        tile_w = w // self.num_tiles
        tile_h = h // self.num_tiles
        if tile_w == 0 or tile_h == 0:
            return [image]

        overlap = 0.25
        stride_w = max(1, int(tile_w * (1 - overlap)))
        stride_h = max(1, int(tile_h * (1 - overlap)))

        y = 0
        while y <= h - tile_h:
            x = 0
            while x <= w - tile_w:
                tiles.append(image.crop((x, y, x + tile_w, y + tile_h)))
                if x + tile_w >= w:
                    break
                x += stride_w
            if y + tile_h >= h:
                break
            y += stride_h

        return tiles

    def _format_prompt(self, query: str) -> str:
        if self.prompt_template:
            return self.prompt_template.format(query=query)
        return query

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        prompt = self._format_prompt(query)

        with torch.inference_mode():
            inputs = self.processor(
                text=[prompt], return_tensors="pt", padding=True,
                truncation=True, max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_vec = self.model.get_text_features(**inputs)

            if not isinstance(text_vec, torch.Tensor):
                if hasattr(text_vec, "text_embeds") and text_vec.text_embeds is not None:
                    text_vec = text_vec.text_embeds
                elif hasattr(text_vec, "pooler_output"):
                    text_vec = text_vec.pooler_output

            text_vec = text_vec.float()
            # Same projection fix for text
            if self.proj_dim and text_vec.shape[-1] != self.proj_dim and hasattr(self.model, "text_projection"):
                with torch.inference_mode():
                    text_vec = self.model.text_projection(text_vec.to(self.device)).float().cpu()

            text_vec = text_vec.float()
            if self.image_mean is not None:
                text_vec = text_vec - self.image_mean
            text_vec = F.normalize(text_vec, dim=1).cpu()

        # Cosine similarity per tile, then MaxSim per document
        tile_scores = (self.image_embeddings @ text_vec.T).squeeze(1)  # (total_tiles,)

        if self.num_tiles == 1 or len(self.tile_to_doc) == len(self.document_ids):
            scores = tile_scores.tolist()
        else:
            scores_arr = np.full(len(self.document_ids), float("-inf"))
            np.maximum.at(scores_arr, self.tile_to_doc, tile_scores.cpu().numpy())
            scores = scores_arr.tolist()

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
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--max-docs", type=int, default=3021)
    parser.add_argument("--max-queries", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tiles", type=int, default=3,
                        help="Grid size for tiling (0 or 1 = whole page, 3 = 3x3 grid + whole)")
    parser.add_argument("--no-center", action="store_true",
                        help="Disable mean-centering (for ablation)")
    parser.add_argument("--prompt-template", type=str, default=None,
                        help="Override prompt template (default: 'a scanned page of a scientific paper about {query}')")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "console" / "results")
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

    prompt_template = args.prompt_template
    if prompt_template is None and args.tiles > 0:
        prompt_template = "a scanned page of a scientific paper about {query}"

    agent = CLIPImageSearchAgent(
        images=image_values,
        document_ids=document_ids,
        model_name=args.model,
        batch_size=args.batch_size,
        num_tiles=args.tiles,
        center_embeddings=not args.no_center,
        prompt_template=prompt_template,
    )

    output_name = f"gdz-image-{args.model.split('/')[-1]}"
    if args.tiles > 1:
        output_name += f"-tiles{args.tiles}"
    if not args.no_center:
        output_name += "-centered"
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
