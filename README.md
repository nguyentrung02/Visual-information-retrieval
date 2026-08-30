# Visual Document Retrieval on Historical Scientific Pages

**Group repository:** https://github.com/nguyentrung02/Visual-information-retrieval  
**Dataset (Hugging Face):** https://huggingface.co/datasets/Trungdaik/Visual_information_retrieval  
**Evaluation framework:** https://github.com/weaviate/query-agent-benchmarking  
**Tutor responsible:** Terry Ruas, Constantin Dalinghaus  
**Group leader:** Trung Nguyen (nguyentrung02)  
**Group members:** Trung Nguyen

---

## Setup instructions

### Prerequisites

- Python 3.10+
- pip package manager
- (Optional) Weaviate Cloud account for text retrieval via Weaviate
- (Optional) Access to an NVIDIA GPU for CLIP image retrieval (e.g., GWDG SCC Grete)

### 1. Clone and install

```bash
git clone https://github.com/nguyentrung02/Visual-information-retrieval.git
cd Visual-information-retrieval/scripts
pip install -r requirements.txt
```

### 2. Weaviate Cloud credentials (text retrieval only)

Text retrieval (BM25, vector, hybrid) uses Weaviate Cloud.  Copy the env
template and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your WEAVIATE_URL and WEAVIATE_API_KEY
```

The free tier of Weaviate Cloud has a daily limit (~10 000 `text2vec-weaviate`
requests).  BM25 does not count against this quota.

### 3. Install query-agent-benchmarking with patches

The published PyPI release of `query-agent-benchmarking` requires two small
patches.  Install from source instead:

```bash
git clone https://github.com/weaviate/query-agent-benchmarking.git
cd query-agent-benchmarking
pip install -e .
```

Or apply the patches manually (see `scripts/README.md` → "Known Issues").

### 4. Run experiments

```bash
# Text retrieval (via Weaviate Cloud) — smoke test then full run
python run-text-retrieval.py --max-docs 10 --max-queries 5
python run-text-retrieval.py

# Image retrieval (CLIP, brute-force cosine similarity) — smoke test then full
python run-image-retrieval.py --max-docs 10 --max-queries 5
python run-image-retrieval.py

# Analyse failures and compare methods
python analyze_results.py
python compare_methods.py
```

### 5. Run on the GWDG SCC (GPU)

GPU work must be done on the Grete login node. Your SLURM project account
determines which partitions and filesystems are available.

### Experiments (SCC GPU)

```bash
# 1. SSH to the Grete GPU login node
ssh -i <key-file> <username>@glogin-gpu.hpc.gwdg.de

# 2. Clone repos to $HOME
cd $HOME
git clone https://github.com/nguyentrung02/Visual-information-retrieval.git
git clone https://github.com/weaviate/query-agent-benchmarking.git

# 3. One-time environment setup (conda env + CLIP model download)
cd Visual-information-retrieval
bash scripts/setup-scc.sh

# 4. Submit the image retrieval job
source activate gdz-retrieval
sbatch scripts/slurm-image-job.sh

# 5. Monitor
squeue --me
```

For ablation experiments, use env-var overrides (documented in `SKILL.md`):

```bash
# V1 (baseline: no tiling, no centering, no prompt)
SLURM_TILES=0 SLURM_NO_CENTER=0 SLURM_PROMPT=0 SLURM_OUTPUT_NAME="gdz-image-v1" \
    sbatch scripts/slurm-image-job.sh

# V2 (centering only)
SLURM_TILES=0 SLURM_PROMPT=0 SLURM_OUTPUT_NAME="gdz-image-v2" \
    sbatch scripts/slurm-image-job.sh

# V4b (default: tiling + prompt, no centering)
sbatch scripts/slurm-image-job.sh
```

Outputs are written to `scripts/results/gdz-image-<name>.json`.

---

## Methodology

### Research question

Can image-based retrieval match text-based retrieval on historical scientific
documents **without** OCR?  We compare four retrieval strategies on the GDZ
corpus and analyse where each fails.

### Dataset

- **Source:** 8 volumes from the Göttinger Digitalisierungszentrum (GDZ) —
  geophysics and mathematics, published 2000s.
- **Size:** 3 021 page-level records, each with:
  - `transcription` — Tesseract OCR text (eng + deu)
  - `base64_str` — the first page image (300 DPI PNG)
- **Queries:** 180 needle-in-a-haystack questions, one gold page per query.
- Published on Hugging Face: `Trungdaik/Visual_information_retrieval`

| Paper | Title (abbreviated) | Pages | Queries |
|-------|--------------------|-------|---------|
| 1 | Journal of Geophysics | 205 | 32 |
| 2 | Journal für die reine und angewandte Mathematik | 228 | 18 |
| 3 | Journal of Fourier Analysis (Vol 5) | 641 | 20 |
| 4 | Journal of Fourier Analysis (Vol 6) | 679 | 39 |
| 5 | Journal of Fourier Analysis (Vol 7) | 649 | 38 |
| 6 | Metrika | 285 | 11 |
| 7 | Aequationes Mathematicae | 333 | 22 |

### Retrieval methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **BM25** (lexical) | Term-frequency scoring over OCR transcriptions | Weaviate Cloud BM25 |
| **Vector** (dense) | Semantic search via `text2vec-weaviate` embeddings | Weaviate Cloud vector search |
| **Hybrid** | RRF fusion of BM25 + vector | Weaviate Cloud hybrid search |
| **CLIP** (image) | Image-to-text retrieval via cosine similarity of CLIP embeddings | Local brute-force (`openai/clip-vit-large-patch14`) |

### Evaluation

All experiments use the [`query-agent-benchmarking`](https://github.com/weaviate/query-agent-benchmarking)
framework with 1 trial and the following metrics:

- **Recall@K** (K = 1, 5, 20, 50, 100) — whether the gold page appears in the top-K.
- **nDCG@10** — ranking quality at depth 10.
- **Average query time** — end-to-end latency per query.

Each query has exactly one gold document, so Recall@K is binary per query.

### Why results are low overall

Papers 3, 4, and 5 contain queries that are **abstract paraphrase-style**
questions (e.g., *"what algebraic and topological property must the operator
sequence $S_n$ satisfy"*), often with LaTeX math or symbolic notation. These
share very few rare tokens with the target page text. In contrast, Papers 1, 2,
6, and 7 questions tend to quote or closely paraphrase article titles and
section headings, which BM25 matches directly via exact terms.

Additionally, Papers 3, 4, and 5 are volumes of the **same journal** (*Journal
of Fourier Analysis*), so they share terminology, author names, and notation.
BM25 frequently retrieves the correct paper but struggles to discriminate the
exact page within or across volumes. Length correlates with failure (longer
papers → more similar pages → harder discrimination) but is not the mechanism —
it is the **query construction + cross-volume vocabulary overlap** that drives
0% Recall@1 for Papers 3–5.

---

## Experiments

### Experiment 1: Text retrieval baselines (BM25, vector, hybrid)

**What:** Evaluate three text-search strategies on OCR transcriptions via
Weaviate Cloud.

**How:** Upload 3 021 documents to a single Weaviate collection, then run
`bm25-search`, `vector-search`, and `hybrid-search` agents via the
benchmarking library.  Queries are passed in-memory as `InMemoryQuery` objects.

**Expected:** BM25 should outperform dense retrieval because the queries
contain specific terminology (altitudes, dates, instrument names, numerical
values) that benefits from exact token matching.

### Experiment 2: Image retrieval (CLIP, brute-force)

**What:** Encode all 3 021 page images with CLIP, compute cosine similarity
against text queries, and evaluate.

**How:** The `run-image-retrieval.py` script encodes image embeddings once
(GPU), then for each query encodes the question text and computes dot-product
cosine similarity against all image embeddings (brute-force, no index).

**Expected:** CLIP underperforms text methods because it is trained on
natural images, not scientific document pages.  A document-specific visual
model (e.g., ColPali) is expected to improve results.

### Experiment 3: Failure-mode analysis

**What:** Categorise every query into one of four failure modes.

**How:** For each retrieval method, classify queries as `hit@1`,
`found@5_not_1`, `found@20_not_5`, or `missed@20`.  Breakdown per paper
identifies which volumes are hardest.

---

## Results

### Text retrieval — Recall@K and nDCG

| Method | Recall@1 | Recall@5 | Recall@20 | Recall@50 | Recall@100 | nDCG@10 | Avg query (ms) |
|--------|---------:|---------:|----------:|----------:|-----------:|--------:|---------------:|
| BM25 | 17.8% | 26.1% | 28.3% | 32.8% | 35.6% | 22.7% | 51.5 |
| Vector | 8.9% | 15.6% | 25.0% | 33.3% | 36.7% | 13.4% | 40.2 |
| Hybrid | 13.9% | 20.0% | 28.9% | 32.2% | 37.8% | 19.3% | 78.6 |

### CLIP image retrieval (brute-force, clip-vit-large-patch14 on SCC GPU)

| Variant | Config | Recall@1 | Recall@5 | Recall@20 | Recall@100 | Top-50 share | Distinct pages |
|---------|--------|---------:|---------:|----------:|-----------:|-------------:|---------------:|
| V1 | Baseline (no centering, no tiling) | 2.78% | 3.89% | 10.00% | 18.89% | 21.4% | 1,139 |
| V2 | + Mean-centering (hubness reduction) | 0.00% | 0.56% | 2.22% | 7.22% | 93.5% | 105 |
| V3 | + Centering + prompt template | 0.00% | 0.00% | 2.78% | 7.78% | 93.5% | 105 |
| V4a | + Tiling (3×3 grid + whole page), centering | 0.56% | 4.44% | 8.33% | 18.33% | 82.1% | 196 |
| V4b | **+ Tiling + prompt template, NO centering** | **7.22%** | **16.11%** | **23.33%** | **31.67%** | **13.8%** | **1,424** |
| BM25 | Sparse retrieval (baseline) | 18.33% | 26.67% | 28.33% | 28.33% | 16.5% | 1,255 |

**Results are in** `scripts/results/gdz-image-clip-vit-large-patch14-trial-1.json`
(V1), `gdz-image-clip-vit-large-patch14-centered-trial-1.json` (V2/V3),
`gdz-image-clip-vit-large-patch14-tiles3-centered-trial-1.json` (V4a), and
`gdz-image-v4-tiles-prompt-nocenter-trial-1.json` (V4b).

**Key finding:** V4b (tiling + prompt, no centering) is the best CLIP
configuration — R@1=7.22%, a 2.6x improvement over V1. Mean-centering is
**catastrophic** in every ablation (V2: R@1=0.00%, 93.5% top-50 share;
V4a: R@1=0.56%, 82.1% top-50 share). Tiling provides the biggest recall gain
(~80 DPI per tile vs 27 DPI), and the prompt template further improves
alignment with CLIP's caption-training distribution. V4b's hubness actually
*beats* BM25 (1424 distinct pages vs 1255, 13.8% vs 16.5% top-50 share).

### Complementary failure modes (BM25 vs CLIP V1)

Comparing BM25 (text) and CLIP V1 (image) at Recall@20 over 180 queries:

| Category | Count | % |
|---|---|---|
| Both succeed | 13 | 7.2% |
| Text only succeeds | 38 | 21.1% |
| Image only succeeds | 5 | 2.8% |
| Neither succeeds | 124 | 68.9% |

Text retrieval succeeds on 51 queries at @20, CLIP on 18. They share 13 successes
and have complementary failure modes: 38 queries that BM25 finds but CLIP misses,
plus 5 that CLIP finds but BM25 misses. This validates the IRPAPERS finding that
text- and image-based retrieval exhibit **complementary failure modes**, though the
asymmetry is more pronounced here (38 text-exclusive vs 5 image-exclusive) because
CLIP-ViT-L/14 is a general-domain model, not a document-tuned model like ColPali.

### Per-paper Recall@1 (BM25)

| Paper | Pages | Queries | Recall@1 | Recall@5 | Recall@20 |
|-------|-------|---------|---------:|---------:|----------:|
| 1 | 205 | 32 | 34.4% | 40.6% | 40.6% |
| 2 | 228 | 18 | 61.1% | 94.4% | 100.0% |
| 3 | 641 | 20 | 0.0% | 5.0% | 10.0% |
| 4 | 679 | 39 | 0.0% | 2.6% | 2.6% |
| 5 | 649 | 38 | 0.0% | 2.6% | 2.6% |
| 6 | 285 | 11 | 36.4% | 36.4% | 54.5% |
| 7 | 333 | 22 | 27.3% | 45.5% | 45.5% |

Use `python full_analysis.py` (in the `query-agent-benchmarking` repo) to
generate the equivalent table for Vector and Hybrid.

### Why Recall@5 ≈ Recall@20 for BM25

The gold document's rank follows a **bimodal distribution**: it is either
found very early (top 5) or ranked far below position 20.  Of 180 queries:

| Rank bucket | Queries found | Cumulative |
|-------------|--------------:|------------:|
| Position 1 | 32 | 17.8% |
| Position 2–5 | 15 | 26.1% |
| Position 6–20 | 4 | 28.3% |
| Position 21–50 | 8 | 32.8% |
| Position 51–100 | 5 | 35.6% |
| Not in top 100 | 116 | 100% |

Only 4 additional queries are resolved by expanding from top-5 to top-20,
hence the small jump.  This is a data characteristic (long documents with
shared vocabulary), not a bug — the full 3 021-document ranking confirms the
pattern.

### Failure-mode summary (BM25, Recall@1)

| Category | Count |
|----------|------:|
| All three methods succeed | 23 / 180 |
| All three methods fail | 146 / 180 |
| BM25 only succeeds (lexical) | 9 |
| Vector only succeeds (semantic) | 2 |

### Discussion

1. **BM25 is the strongest text baseline.**  Scientific queries with
   specific terminology, dates, and numerical values favour exact token
   matching over semantic embeddings.

2. **Dense search catches a different error set.**  Vector-only successes
   tend to be author/title queries (e.g., "Who are the authors of the 1997
   paper titled 'Tangent star cones'?").

3. **Hybrid never dominates either alone method** because RRF with k=60
   dilutes strong lexical signals.  A weighted hybrid would likely improve.

4. **CLIP benefits greatly from tiling + prompt, and centering is harmful.**
   The model is trained on natural images, not scientific document pages. A
   structural issue is CLIP's default preprocessor resizing A4 pages
   (2479×3508px) to 224×224 with center-crop — destroying text legibility
   (≈27 DPI) and deleting the title/page-number band. Tiling (3×3 overlapping
   grid + whole page, MaxSim scoring) recovers ≈80 DPI per tile and lifts
   R@1 from 2.78% to 7.22%. The prompt template ("a scanned page of a
   scientific paper about {query}") further improves alignment with CLIP's
   caption-training distribution. Mean-centering — proposed to reduce the
   modality gap — proves **catastrophic** in every configuration: it collapses
   both image and text embeddings toward the mean, destroying recall (R@1 drops
   to 0–0.56%) and creating severe hubbing (top-50 share 82–93.5%, distinct
   pages 105–196). The recommended configuration is **tiling + prompt, no
   centering**. A document-specific model (e.g., ColPali, patch-level
  embeddings at native resolution) is the planned improvement and requires
  GPU execution on the SCC.

5. **Query construction + cross-volume vocabulary overlap predict failure.**
   Papers 3–5 (same journal, overlapping terminology) have ~0% Recall@1 not
   because long papers are inherently harder — but because their queries are
   abstract paraphrases with LaTeX that share rare tokens across volumes.
   Length correlates with the outcome but is not the mechanism.

---

## Visualizations

The per-paper recall chart shows the **query-construction + cross-volume overlap**
effect rather than a pure length effect:

```
Recall@1 by Paper (BM25)
Paper 2 (228p):  ████████████████ 61%
Paper 1 (205p):  █████████ 34%
Paper 6 (285p):  █████████ 36%
Paper 7 (333p):  ██████ 27%
Paper 3 (641p):  ───────── 0%
Paper 4 (679p):  ───────── 0%
Paper 5 (649p):  ───────── 0%
```

Full per-query results with ranked lists are in
`scripts/results/gdz-{bm25,vector,hybrid}-trial-1.json` (Weaviate) and
`scripts/results/gdz-image-clip-vit-large-patch14-*.json` (CLIP V1–V4) and
`scripts/results/gdz-image-v4-tiles-prompt-nocenter-trial-1.json` (V4b).
Pre-computed recall for Vector/Hybrid is in `scripts/results/method-comparison.json`.

To regenerate the comparison plots (Recall@K bar chart, rank distribution,
clip ablation, complementarity) with all methods side-by-side, see
`scripts/plots/README.md` for details. Quick command:

```bash
python scripts/plot_results.py \
    --results scripts/results/gdz-text-local.json \
    --results scripts/results/gdz-image-clip-vit-large-patch14-trial-1.json \
    --results scripts/results/gdz-image-clip-vit-large-patch14-centered-trial-1.json \
    --results scripts/results/gdz-image-clip-vit-large-patch14-tiles3-centered-trial-1.json \
    --results scripts/results/gdz-image-v4-tiles-prompt-nocenter-trial-1.json \
    --precomputed scripts/results/method-comparison.json \
    --outdir scripts/plots
```

---

## References

1. Shorten, C., et al. (2026). *IRPAPERS: A Benchmark for Information
   Retrieval on Scientific Papers.* [GitHub](https://github.com/weaviate/IRPAPERS)
2. Weaviate. *query-agent-benchmarking* — evaluation framework for retrieval
   agents. [GitHub](https://github.com/weaviate/query-agent-benchmarking)
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks.*
4. Radford, A., et al. (2021). *Learning Transferable Visual Models From
   Natural Language Supervision (CLIP).*
5. Nauroy, T., et al. (2024). *ColPali: Efficient Document Retrieval with
   Vision-Language Models for Visual Document Retrieval.*
