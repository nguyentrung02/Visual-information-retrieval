# Visual Information Retrieval Benchmarking

Retrieval benchmarks for the [Trungdaik/Visual_information_retrieval](https://huggingface.co/datasets/Trungdaik/Visual_information_retrieval) dataset using [query-agent-benchmarking](https://github.com/weaviate/query-agent-benchmarking).

This project evaluates three text-retrieval strategies (BM25, dense vector, hybrid) against Weaviate Cloud, and a brute-force CLIP image-to-text retriever locally or on the GWDG SCC GPU cluster.

## Motivation

The GDZ dataset pairs scientific documents with text queries. We want to compare:
- **Lexical search** (BM25) — exact token overlap.
- **Dense search** (Sentence-BERT / Weaviate `text2vec-weaviate`) — semantic similarity.
- **Hybrid search** (RRF fusion) — combining lexical and dense signals.
- **Image search** (CLIP) — querying document images via text embeddings.

## Method

### Dataset

- **Source:** `Trungdaik/Visual_information_retrieval`
- **Documents:** 3,021 scientific papers with `dataset_id`, `transcription` (text), and `base64_str` (first page image).
- **Queries:** 180 text questions, each with one gold `dataset_id`.

### Text Retrieval (Weaviate Cloud)

1. Load docs and queries from HuggingFace.
2. Create a single Weaviate collection (`GDZ_Default`) with the `text2vec-weaviate` vectorizer.
3. Upload all documents (batched to avoid gRPC payload limits).
4. Run three search agents via `run_search_eval()`:
   - `bm25-search`
   - `vector-search`
   - `hybrid-search`
5. Queries are passed in-memory as `InMemoryQuery` objects (no second collection needed).

### Image Retrieval (CLIP, brute-force)

1. Encode all document images with `openai/clip-vit-base-patch32`.
2. For each query, encode the question text and compute cosine similarity against all image embeddings.
3. Rank by similarity and evaluate with `run_search_eval()`.

### Metrics

- Recall@1, Recall@5, Recall@20, Recall@50, Recall@100
- nDCG@10
- Average query time

## Repository Structure

```
own-proj/
├── run-text-retrieval.py       # Text retrieval via Weaviate Cloud
├── run-image-retrieval.py      # CLIP image-to-text retrieval (local / SCC)
├── slurm-image-job.sh          # Slurm job script for SCC Grete GPU
├── requirements.txt            # Python dependencies
├── .env.example                # Weaviate credentials template
└── README.md                   # This file
```

## Prerequisites

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/own-proj
pip install -r requirements.txt
```

### 2. Configure Weaviate Cloud (text retrieval only)

Copy `.env.example` to `.env` and fill in your Weaviate Cloud credentials:

```bash
cp .env.example .env
# Edit .env with your WEAVIATE_URL and WEAVIATE_API_KEY
```

The free tier of Weaviate Cloud has a daily limit on `text2vec-weaviate` requests (~10,000). For the full 3,021 docs × 180 queries, vector and hybrid search may hit this limit. BM25 does not count against this quota.

### 3. (Optional) SCC GPU for image retrieval

If you want to run CLIP encoding on the GWDG SCC:

```bash
ssh <your-user>@glogin-gpu.hpc.gwdg.de
cd $WORK/workspaces/<your-project>
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
sbatch own-proj/slurm-image-job.sh
```

See `SKILL.md` for full SCC documentation.

## How to Repduce

### Setup

```bash
cd own-proj
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your WEAVIATE_URL and WEAVIATE_API_KEY
```

### Text Retrieval (local / Weaviate)

```bash
# Quick smoke test (10 docs, 5 queries)
python run-text-retrieval.py --max-docs 10 --max-queries 5

# Full dataset (3,021 docs, 180 queries)
python run-text-retrieval.py
```

Results are saved to `console/results/gdz-{bm25,vector,hybrid}-search-*.json`.

### Image Retrieval (CLIP, local CPU)

```bash
python run-image-retrieval.py --max-docs 10 --max-queries 5
```

### Image Retrieval (CLIP, SCC GPU)

```bash
sbatch slurm-image-job.sh
```

### Compare Methods

```bash
python compare_methods.py
# Or with custom results directory:
python compare_methods.py --results-dir /path/to/results
```

## Results

*(Fill this section after running the experiments.)*

| Method | Recall@1 | Recall@5 | Recall@20 | nDCG@10 | Avg Query Time |
|---|---|---|---|---|---|
| BM25 | — | — | — | — | — |
| Vector | — | — | — | — | — |
| Hybrid | — | — | — | — | — |
| CLIP (image) | — | — | — | — | — |

## Known Issues & Patches

The published `query-agent-benchmarking` package requires two small patches to work with this code. If you install from PyPI, you need to apply these manually or install the patched version from the main repo:

1. **`engram_dspy_agent.py`** — change `from engram import RetrievalConfig` to `from engram import RetrievalConfigModel as RetrievalConfig`.
2. **`metrics_config.py`** — add a fallback to default metrics for unknown dataset names (e.g., `GDZ`).

Alternatively, install the library directly from the cloned repo:
```bash
pip install -e path/to/query-agent-benchmarking
```

## License

MIT
