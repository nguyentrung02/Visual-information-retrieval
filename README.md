# Visual Document Retrieval

A replication study of the [IRPAPERS](https://github.com/weaviate/IRPAPERS) benchmark on scientific documents from the [Göttinger Digitalisierungszentrum](https://www.sub.uni-goettingen.de/en/digitalisation/gdz/).

## Goal

Compare **image‑based retrieval** (ColModernVBERT) with **text‑based retrieval** (OCR + hybrid search) on non‑OCR’d historical and modern scientific volumes. Determine whether visual retrieval can match text retrieval without ever running OCR – and where each approach fails.

## Dataset

- **Source:** multiple volumes from the GDZ (geophysics, mathematics), English and some German pages.
- **Current size:** *7* volumes, *3020* pages.
- **Images:** high‑resolution PNG, extracted per book.
- **Transcriptions:** Tesseract OCR (English + German).
- **Questions:** needle‑in‑a‑haystack format.

## Repository structure

- `scripts/` – extraction, OCR.
- `data/` – ID mappings, PDFs, question CSV (add later).

## Status

- [x] Corpus extraction & OCR
- [ ] Question creation (in progress)
- [ ] Retrieval pipeline (planned)
- [ ] Evaluation & failure analysis (planned)

## Acknowledgements

- IRPAPERS benchmark: Shorten et al. (2026)
- Göttinger Digitalisierungszentrum