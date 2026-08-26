"""Memory-first rendering of individual PDF pages for Vision requests."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True, slots=True)
class HighlightRectangle:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class PageTextHighlights:
    page_width: float
    page_height: float
    rectangles: tuple[HighlightRectangle, ...]


def _search_anchors(excerpt: str) -> tuple[str, ...]:
    normalized = " ".join(excerpt.split())
    if not normalized:
        return ()
    sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    words = normalized.split()
    candidates = (sentence[:150], " ".join(words[:18]), " ".join(words[:10]))
    return tuple(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))


def find_text_highlights(
    pdf_path: str | Path, page_number: int, excerpt: str
) -> PageTextHighlights:
    """Find a small set of PDF-space rectangles for a retrieved text excerpt."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if page_number <= 0:
        raise ValueError("page_number is one-based and must be greater than zero")

    with pymupdf.open(path) as document:
        if page_number > document.page_count:
            raise ValueError("page_number exceeds PDF page count")
        page = document.load_page(page_number - 1)
        rectangles: list[pymupdf.Rect] = []
        for anchor in _search_anchors(excerpt):
            rectangles = page.search_for(anchor)
            if rectangles:
                break
        return PageTextHighlights(
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            rectangles=tuple(
                HighlightRectangle(
                    x=float(rect.x0),
                    y=float(rect.y0),
                    width=float(rect.width),
                    height=float(rect.height),
                )
                for rect in rectangles[:20]
            ),
        )


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
