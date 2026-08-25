"""In-memory NumPy cosine similarity retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from furiosa_rag.models import Chunk, RetrievedChunk


class CosineRetriever:
    def search(
        self,
        query_embedding: Sequence[float],
        chunks: Sequence[Chunk],
        chunk_embeddings: Sequence[Sequence[float]],
        *,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not chunks or len(chunks) != len(chunk_embeddings):
            raise ValueError("chunks and chunk_embeddings must have the same non-zero length")

        matrix = np.asarray(chunk_embeddings, dtype=np.float32)
        query = np.asarray(query_embedding, dtype=np.float32)
        if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
            raise ValueError("embedding dimensions do not match")

        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
        denominators = matrix_norms * query_norm
        scores = np.divide(
            matrix @ query,
            denominators,
            out=np.zeros_like(matrix_norms),
            where=denominators != 0,
        )
        indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
        return [
            RetrievedChunk(chunk=chunks[int(index)], retrieval_score=float(scores[index]))
            for index in indices
        ]
