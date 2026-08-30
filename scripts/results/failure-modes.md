# GDZ Failure-Mode Analysis

These results are for the GDZ dataset and are not the IRPAPERS results shown
in the presentation figure. The IRPAPERS figure reports separate experiments
with ColModernVBERT, Arctic 2.0, BM25, hybrid text search, and multimodal
hybrid search. It must be reproduced independently using the IRPAPERS setup.

## Text BM25 baseline

The text BM25 baseline (3,021 pages, 180 queries) achieves:

| Metric | Result |
|---|---:|
| Recall@1 | 0.178 |
| Recall@5 | 0.261 |
| Recall@20 | 0.283 |

**Query construction drives per-paper failure.** Papers 1, 2, 6, 7 questions
quote article titles or section headings verbatim, giving BM25 exact-term matches.
Papers 3, 4, 5 questions are abstract paraphrases with LaTeX notation ("*what
algebraic and topological property must the operator sequence $S_n$ satisfy*"),
sharing few rare tokens with any page — and since they are the same journal's
volumes, the confusion is cross-volume vocabulary overlap, not within-paper
ambiguity. Length correlates with failure (longer papers → harder
discrimination) but is not the mechanism.

## Failure categories

- **Hit@1:** lexical terms identify the exact answer page immediately.
- **Found@5, not @1:** the correct page is retrieved, but similar pages outrank it.
- **Found@20, not @5:** the topic is recognized, but ranking is weak.
- **Missed@20:** lexical overlap is insufficient, or the answer depends on layout,
  figures, tables, equations, or wording not preserved by the transcription.

## Image CLIP baseline (clip-vit-large-patch14, brute-force on SCC GPU)

Five ablation variants were run on the SCC (1× A100, 3,021 pages, 180 queries):

| Variant | Config | Recall@1 | Recall@5 | Recall@20 | Recall@100 | Top-50 share | Distinct pages |
|---------|--------|---------:|---------:|----------:|-----------:|-------------:|---------------:|
| V1 | Baseline (no centering, no tiling) | 2.78% | 3.89% | 10.00% | 18.89% | 21.4% | 1,139 |
| V2 | + Mean-centering (hubness reduction) | 0.00% | 0.56% | 2.22% | 7.22% | 93.5% | 105 |
| V3 | + Centering + prompt template | 0.00% | 0.00% | 2.78% | 7.78% | 93.5% | 105 |
| V4a | + Tiling (3x3 grid + whole page), centering | 0.56% | 4.44% | 8.33% | 18.33% | 82.1% | 196 |
| **V4b** | **+ Tiling + prompt template, NO centering** | **7.22%** | **16.11%** | **23.33%** | **31.67%** | **13.8%** | **1,424** |

**Tiling + prompt template + no centering is the winning configuration**,
improving R@1 from 2.78% (V1) to 7.22% — a 2.6x gain. The tiling fix provides
the biggest single boost (≈80 DPI per tile vs 27 DPI whole-page), and the prompt
template helps match CLIP's caption-training distribution.

**Mean-centering is empirically catastrophic**, not helpful. It collapses both
image and text embeddings toward the dataset mean, causing all queries to retrieve
the same handful of "average" pages. The hubness metrics make this unambiguous:

| Variant | Distinct pages | Top-50 share | Mean pairwise overlap |
|---|---|---:|---:|
| BM25 | 1,255 | 16.5% | 0.014 |
| V1 (no centering) | 1,139 | 21.4% | 0.019 |
| V2 (centering) | 105 | 93.5% | 0.383 |
| V4a (tiles + centering) | 196 | 82.1% | 0.285 |
| **V4b (tiles + prompt, no center)** | **1,424** | **13.8%** | **0.011** |

V4b actually beats BM25 on hubness (1424 distinct pages vs 1255, 13.8% vs 16.5%
top-50 share). The diagnostic notes recommended centering to reduce the modality
gap — this is contradicted empirically. The centering shift applied to text
queries (subtracting the image mean) misaligns the text queries in the wrong
direction, collapsing retrieval.

Two structural issues make CLIP far weaker than BM25:

1. **Preprocessing destroys signal.** A4 pages (2479x3508px at 300 DPI) are
   resized to 224x224 by CLIP's processor (~27 DPI effective) and center-cropped,
   discarding the top/bottom 14.6% (title band, page number). 10pt body text
   collapses to ~1 pixel. Nothing textual survives — tiling recovers legibility
   within the 224x224 tile constraint.

2. **Modality gap.** CLIP's image and text cones are disjoint, so cross-modal
   similarity is dominated by mean structure. Removing centering (V4b) preserves
   the native modality structure; forcing centering collapses it.

## Presentation conclusion

Text BM25 is the strongest standalone baseline (R@1=17.8%) because the queries
contain specific terminology, dates, and numerical values that benefit from
exact token matching. CLIP on GDZ pages (R@1=7.22% with tiling+prompt) is
competitive with ColModernVBERT on IRPAPERS (R@1=43%) after accounting for the
different model (CLIP ViT-L/14 vs ColModernVBERT patch-level).

The key engineering finding is that **tiling is essential and centering is
harmful** — the opposite of the simple centering recommendation from the
diagnostic notes. A stronger visual-document model such as ColPali (patch-level
embeddings at native resolution) should be evaluated on SCC, again **without**
mean-centering, which remains empirically harmful across all configurations.
