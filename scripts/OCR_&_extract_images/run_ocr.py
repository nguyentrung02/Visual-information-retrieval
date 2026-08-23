"""
Run Tesseract OCR on all page images (per-book folders).

Usage:
    python run_ocr.py
"""

import pytesseract
from PIL import Image
import os
from pathlib import Path

# ===== CONFIGURATION =====
IMAGE_ROOT = Path("../data/page_images")
TEXT_ROOT = Path("../data/OCR_text")
LANG = "eng+deu"           # use "eng+deu" for mixed German / English
# =========================

# Uncomment and adjust if Tesseract is not in PATH (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def main():
    TEXT_ROOT.mkdir(exist_ok=True)

    for book_name in sorted(os.listdir(IMAGE_ROOT)):
        book_path = IMAGE_ROOT / book_name
        if not book_path.is_dir():
            continue

        output_book_path = TEXT_ROOT / book_name
        output_book_path.mkdir(exist_ok=True)

        for img_filename in sorted(os.listdir(book_path)):
            if not img_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = book_path / img_filename
            img = Image.open(img_path)

            text = pytesseract.image_to_string(img, lang=LANG)

            base = os.path.splitext(img_filename)[0]
            txt_path = output_book_path / f"{base}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"OCR done: {book_name}/{img_filename} -> {book_name}/{base}.txt")

    print("All OCR finished.")

if __name__ == "__main__":
    main()