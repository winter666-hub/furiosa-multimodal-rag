"""Page-preserving configurable word chunking."""

from __future__ import annotations

from furiosa_rag.models import Chunk, PageText


class PageChunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for page in pages:
            words = page.text.split()
            page_chunk_index = 1
            for start in range(0, len(words), step):
                text = " ".join(words[start : start + self.chunk_size]).strip()
                if not text:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"page-{page.page_number}-chunk-{page_chunk_index}",
                        page_number=page.page_number,
                        text=text,
                    )
                )
                page_chunk_index += 1
                if start + self.chunk_size >= len(words):
                    break
        return chunks
