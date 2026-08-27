"""Memory-first rendering of individual PDF pages for Vision requests."""

from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PDF_POINTS_PER_INCH = 72.0
PDF_PAGE_RENDER_SCALE = 3.0
PDF_PAGE_RENDER_DPI = PDF_POINTS_PER_INCH * PDF_PAGE_RENDER_SCALE
DEFAULT_MAX_RENDER_PIXELS = 20_000_000
MIN_HIGHLIGHT_MATCH_SCORE = 0.2


class RenderPixelLimitError(ValueError):
    """Raised before allocating a pixmap that exceeds the configured pixel budget."""


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
    words = normalized.split()
    candidates: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        sentence_words = sentence.split()
        if len(sentence_words) >= 6 or len(words) < 6:
            candidates.append(" ".join(sentence_words[:24]))
    for size in (24, 18, 12, 8):
        if len(words) < size:
            continue
        starts = range(0, len(words) - size + 1, max(1, size // 2))
        candidates.extend(" ".join(words[start : start + size]) for start in starts)
        final_start = len(words) - size
        candidates.append(" ".join(words[final_start:]))
    if not candidates:
        candidates.append(normalized)
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _word_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _block_for_rectangle(
    rectangle: pymupdf.Rect, blocks: list[tuple[float, float, float, float, str, int, int]]
) -> tuple[int, pymupdf.Rect, str] | None:
    center = (rectangle.x0 + rectangle.x1) / 2, (rectangle.y0 + rectangle.y1) / 2
    for index, block in enumerate(blocks):
        block_rect = pymupdf.Rect(block[:4])
        if block_rect.contains(center):
            return index, block_rect, block[4]
    return None


def _match_score(
    excerpt_tokens: set[str], anchor: str, block_text: str, block_rect: pymupdf.Rect, page_height: float
) -> float:
    block_tokens = _word_tokens(block_text)
    overlap = len(excerpt_tokens & block_tokens)
    excerpt_coverage = overlap / max(1, len(excerpt_tokens))
    anchor_specificity = min(len(_word_tokens(anchor)), 24) / 24
    edge_penalty = 0.15 if block_rect.y0 < page_height * 0.08 or block_rect.y1 > page_height * 0.92 else 0
    return excerpt_coverage + anchor_specificity * 0.2 - edge_penalty


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
        excerpt_tokens = _word_tokens(excerpt)
        blocks = [block for block in page.get_text("blocks") if block[4].strip()]
        matches_by_block: dict[int, tuple[float, list[pymupdf.Rect]]] = {}
        for anchor in _search_anchors(excerpt):
            for rectangle in page.search_for(anchor):
                located_block = _block_for_rectangle(rectangle, blocks)
                if located_block is None:
                    continue
                block_index, block_rect, block_text = located_block
                score = _match_score(
                    excerpt_tokens, anchor, block_text, block_rect, float(page.rect.height)
                )
                previous = matches_by_block.get(block_index)
                if previous is None or score > previous[0]:
                    matches_by_block[block_index] = (score, [rectangle])
                elif math.isclose(score, previous[0]):
                    previous[1].append(rectangle)
        best_match = (
            max(matches_by_block.values(), key=lambda candidate: candidate[0])
            if matches_by_block
            else None
        )
        rectangles = best_match[1] if best_match and best_match[0] >= MIN_HIGHLIGHT_MATCH_SCORE else []
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
    def __init__(
        self, *, dpi: float = PDF_PAGE_RENDER_DPI, max_pixels: int = DEFAULT_MAX_RENDER_PIXELS
    ) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be greater than zero")
        if max_pixels <= 0:
            raise ValueError("max_pixels must be greater than zero")
        self.dpi = dpi
        self.max_pixels = max_pixels

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
            scale = self.dpi / PDF_POINTS_PER_INCH
            render_width = math.ceil(page.rect.width * scale)
            render_height = math.ceil(page.rect.height * scale)
            if render_width * render_height > self.max_pixels:
                raise RenderPixelLimitError("PDF page is too large to preview.")
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            return pixmap.tobytes("png")

    def render_data_url(self, pdf_path: str | Path, page_number: int) -> str:
        encoded = base64.b64encode(self.render_png(pdf_path, page_number)).decode("ascii")
        return f"data:image/png;base64,{encoded}"
