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

## Image CLIP baseline

The baseline CLIP ViT-B/32 run (before tiling/centering) achieves approximately:

| Metric | Result |
|---|---:|
| Recall@1 | 0.006 |
| Recall@5 | 0.017 |
| Recall@20 | 0.044 |

Two structural issues make CLIP ineffective on this corpus:

1. **Preprocessing destroys signal.** A4 pages (2479×3508px at 300 DPI) are
   resized to 224×224 by CLIP's processor (≈27 DPI effective) and center-cropped,
   discarding the top/bottom 14.6% (title band, page number). 10pt body text
   collapses to ~1 pixel. Nothing textual survives.

2. **Modality gap → hubness.** CLIP's image and text cones are disjoint, so
   cross-modal similarity is dominated by the mean image direction. Page 4_111
   alone appears in 62 of 180 top-20 lists. The ranking is largely a fixed
   popularity order with the query barely perturbing it.

Fixes applied (see `run-image-retrieval.py`): 3×3 overlapping tile grid + whole
page (≈80 DPI per tile), mean-centering with query-time centering, prompt
template (`"a scanned page of a scientific paper about {query}"`). Results in
`console/results/gdz-image-clip-vit-large-patch14-tiles3-centered-*.json`.

## Presentation conclusion

Text BM25 is the strongest of the two standalone local baselines because the
queries contain specific terminology, dates, and numerical values that benefit
from exact token matching. The remaining failure is mainly fine-grained page
ranking — particularly for Papers 3–5, whose queries are abstract paraphrases
with LaTeX notation that share rare tokens across volumes of the same journal.
A stronger visual-document model such as ColPali (patch-level embeddings at
native resolution, with tiling and proper handling of the modality gap) should
be evaluated on SCC.
