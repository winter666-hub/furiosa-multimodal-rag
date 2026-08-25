"""Persistent document chunk and embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from furiosa_rag.models import Chunk


CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CachedDocument:
    chunks: list[Chunk]
    embeddings: list[list[float]]


class DocumentEmbeddingCache:
    def __init__(self, cache_dir: str | Path = "data/cache/embeddings") -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def file_hash(pdf_path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(pdf_path).open("rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def cache_key(
        self,
        pdf_path: str | Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
    ) -> str:
        identity = {
            "version": CACHE_VERSION,
            "pdf_sha256": self.file_hash(pdf_path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.npz"

    def load(self, cache_key: str) -> CachedDocument | None:
        cache_path = self.path_for(cache_key)
        if not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                version = int(data["cache_version"][0])
                if version != CACHE_VERSION:
                    return None
                chunk_ids = data["chunk_ids"].tolist()
                page_numbers = data["page_numbers"].tolist()
                texts = data["texts"].tolist()
                embedding_array = np.asarray(data["embeddings"], dtype=np.float32)
        except (OSError, ValueError, KeyError, TypeError):
            return None

        if not (len(chunk_ids) == len(page_numbers) == len(texts) == len(embedding_array)):
            return None
        chunks = [
            Chunk(chunk_id=str(chunk_id), page_number=int(page_number), text=str(text))
            for chunk_id, page_number, text in zip(chunk_ids, page_numbers, texts, strict=True)
        ]
        return CachedDocument(chunks=chunks, embeddings=embedding_array.tolist())

    def save(
        self,
        cache_key: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same non-zero length")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.path_for(cache_key)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_key}.", suffix=".npz", dir=self.cache_dir
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary_path,
                cache_version=np.asarray([CACHE_VERSION], dtype=np.int64),
                chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks]),
                page_numbers=np.asarray([chunk.page_number for chunk in chunks], dtype=np.int64),
                texts=np.asarray([chunk.text for chunk in chunks]),
                embeddings=np.asarray(embeddings, dtype=np.float32),
                metadata=np.asarray([json.dumps(metadata or {}, sort_keys=True)]),
            )
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return cache_path

