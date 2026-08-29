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

Four ablation variants were run on the SCC (1× A100, 3,021 pages, 180 queries):

| Variant | Config | Recall@1 | Recall@5 | Recall@20 | Recall@100 | Top-50 share | Distinct pages |
|---------|--------|---------:|---------:|----------:|-----------:|-------------:|---------------:|
| V1 | Baseline (no centering, no tiling) | 2.78% | 3.89% | 10.00% | 18.89% | 21.4% | 1,139 |
| V2 | + Mean-centering (hubness reduction) | 0.00% | 0.56% | 2.22% | 7.22% | 93.5% | 105 |
| V3 | + Centering + prompt template | 0.00% | 0.00% | 2.78% | 7.78% | 93.5% | 105 |
| V4 | + Tiling (3×3 grid + whole page) + centering | 0.56% | 4.44% | 8.33% | 18.33% | 82.1% | 196 |

**Mean-centering is empirically harmful**, not helpful. It collapses both image
and text embeddings toward the dataset mean, causing all queries to retrieve the
same handful of "average" pages. This is visible in the hubness metrics:

- V1 (no centering): 1,139 distinct pages, 21.4% top-50 share — healthy, close to BM25
- V2 (centering): 105 distinct pages, 93.5% top-50 share — catastrophic hubbing
- V4 (tiling + centering): 196 distinct pages, 82.1% top-50 share — partial recovery, still 5× worse than V1

**V1 is the recommended configuration.** It achieves the best CLIP recall (R@1=2.78%,
R@100=18.89%) with healthy hubness properties (21.4% share vs BM25's 16.5%).

Two structural issues make CLIP ineffective on this corpus:

1. **Preprocessing destroys signal.** A4 pages (2479×3508px at 300 DPI) are
   resized to 224×224 by CLIP's processor (≈27 DPI effective) and center-cropped,
   discarding the top/bottom 14.6% (title band, page number). 10pt body text
   collapses to ~1 pixel. Nothing textual survives.

2. **Modality gap causes hubness.** CLIP's image and text cones are disjoint, so
   cross-modal similarity is dominated by the mean direction. In the baseline
   (no centering), a few pages dominate rankings; center-crop makes this worse
   by discarding layout context.

Fixes applied (see `run-image-retrieval.py`): 3×3 overlapping tile grid + whole
page (≈80 DPI per tile). Results in `scripts/results/gdz-image-clip-vit-large-patch14-*.json`.
**Mean-centering and prompt templates are not recommended** — empirical results
show they destroy retrieval quality and increase hubbing.

## Presentation conclusion

Text BM25 is the strongest of the two standalone local baselines because the
queries contain specific terminology, dates, and numerical values that benefit
from exact token matching. The remaining failure is mainly fine-grained page
ranking — particularly for Papers 3–5, whose queries are abstract paraphrases
with LaTeX notation that share rare tokens across volumes of the same journal.
A stronger visual-document model such as ColPali (patch-level embeddings at
native resolution, with tiling and proper handling of the modality gap) should
be evaluated on SCC — **without** mean-centering, which is empirically harmful.
