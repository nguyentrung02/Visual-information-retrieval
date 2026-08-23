# GDZ Failure-Mode Analysis

These results are for the GDZ dataset and are not the IRPAPERS results shown
in the presentation figure. The IRPAPERS figure reports separate experiments
with ColModernVBERT, Arctic 2.0, BM25, hybrid text search, and multimodal
hybrid search. It must be reproduced independently using the IRPAPERS setup.

## Text BM25 baseline

The standalone run evaluates 3,021 pages and 180 queries. It achieves:

| Metric | Result |
|---|---:|
| Recall@1 | 0.178 |
| Recall@5 | 0.261 |
| Recall@20 | 0.283 |

The main pattern is paper-local retrieval. When BM25 finds the correct page,
nearby pages from the same paper often fill the remaining ranks. This is good
for identifying the relevant document, but weak for distinguishing the exact
page containing the answer.

The per-paper breakdown should be discussed in the presentation. Papers 1 and
2 are substantially easier for lexical retrieval, while papers 3, 4, and 5
contain many queries where the relevant page is not in the top 20. Papers 6 and
7 are intermediate.

## Failure categories

- **Hit@1:** lexical terms identify the exact answer page immediately.
- **Found@5, not @1:** the correct page is retrieved, but similar pages outrank it.
- **Found@20, not @5:** the topic is recognized, but ranking is weak.
- **Missed@20:** lexical overlap is insufficient, or the answer depends on layout,
  figures, tables, equations, or wording not preserved by the transcription.

## Image CLIP baseline

The full free CLIP run achieves approximately:

| Metric | Result |
|---|---:|
| Recall@1 | 0.006 |
| Recall@5 | 0.017 |
| Recall@20 | 0.044 |

CLIP often retrieves pages from the correct general visual/document family but
fails to distinguish the exact scientific page. This is expected from a
natural-image CLIP model and is evidence that it should remain a baseline, not
the final visual method.

## Presentation conclusion

Text BM25 is the strongest of the two standalone local baselines because the
queries are detailed scientific questions and the pages have usable
transcriptions. The remaining failure is mainly fine-grained page ranking, not
corpus identification. A stronger visual-document model such as ColPali should
be evaluated on SCC because it represents text patches and page layout rather
than treating the entire page as a generic image.
