"""Provider-neutral extension points for optional external capabilities."""

from .interfaces import (
    DocumentInput,
    DocumentParser,
    OcrProvider,
    ParsedDocument,
    ParsedPage,
    SearchProvider,
    SearchResult,
    TranslationProvider,
)

__all__ = [
    "DocumentInput",
    "DocumentParser",
    "OcrProvider",
    "ParsedDocument",
    "ParsedPage",
    "SearchProvider",
    "SearchResult",
    "TranslationProvider",
]

