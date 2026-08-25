"""Local PDF text extraction for the Text RAG MVP."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from furiosa_rag.models import PageText


class PdfTextExtractor:
    def extract(self, pdf_path: str | Path) -> list[PageText]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        reader = PdfReader(path)
        pages = [
            PageText(page_number=index, text=(page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, start=1)
        ]
        if not any(page.text for page in pages):
            raise ValueError("PDF contains no extractable text; OCR is not implemented yet")
        return pages
