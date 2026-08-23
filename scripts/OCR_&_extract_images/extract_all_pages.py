"""
Extract all PDFs to per-book folders of PNG page images.

Usage:
    python extract_pages.py
"""

import fitz  # PyMuPDF
import os
from pathlib import Path

PDF_DIR = Path("../data/PDFs")
OUTPUT_ROOT = Path("../data/page_images")
DPI = 300

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    for pdf_path in pdf_files:
        book_name = pdf_path.stem  # filename without .pdf
        book_output_dir = OUTPUT_ROOT / book_name
        book_output_dir.mkdir(exist_ok=True)

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=DPI)
            img_name = f"{page_num + 1}.png"
            img_path = book_output_dir / img_name
            pix.save(str(img_path))
            print(f"[{book_name}] Saved page {page_num+1}/{total_pages} -> {img_path}")

        doc.close()
        print(f"Finished {book_name}")

    print("All PDFs processed.")

if __name__ == "__main__":
    main()