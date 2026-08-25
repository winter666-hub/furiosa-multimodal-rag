"""Memory-first rendering of individual PDF pages for Vision requests."""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf


class PdfPageRenderer:
    def __init__(self, *, dpi: float = 144.0) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be greater than zero")
        self.dpi = dpi

    def render_png(self, pdf_path: str | Path, page_number: int) -> bytes:
        """Render a one-based PDF page number to PNG bytes without writing to disk."""
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")
        if page_number <= 0:
            raise ValueError("page_number is one-based and must be greater than zero")

        with pymupdf.open(path) as document:
            if page_number > document.page_count:
                raise ValueError(
                    f"page_number {page_number} exceeds PDF page count {document.page_count}"
                )
            page = document.load_page(page_number - 1)
            scale = self.dpi / 72.0
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            return pixmap.tobytes("png")

    def render_data_url(self, pdf_path: str | Path, page_number: int) -> str:
        encoded = base64.b64encode(self.render_png(pdf_path, page_number)).decode("ascii")
        return f"data:image/png;base64,{encoded}"
