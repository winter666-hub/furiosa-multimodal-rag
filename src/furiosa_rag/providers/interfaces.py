"""Interfaces for optional providers; this module performs no external I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DocumentInput:
    content: bytes
    filename: str
    media_type: str


@dataclass(frozen=True, slots=True)
class ParsedPage:
    page_number: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    pages: tuple[ParsedPage, ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float | None = None


class DocumentParser(Protocol):
    """Convert a binary document into provider-neutral pages."""

    def parse(self, document: DocumentInput) -> ParsedDocument: ...


class OcrProvider(Protocol):
    """Extract text from one image without prescribing an OCR vendor."""

    def extract_text(self, image: bytes, *, media_type: str) -> str: ...


class SearchProvider(Protocol):
    """Return normalized search results from a future search backend."""

    def search(self, query: str, *, limit: int = 5) -> Sequence[SearchResult]: ...


class TranslationProvider(Protocol):
    """Translate text independently of a specific translation service."""

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
    ) -> str: ...

