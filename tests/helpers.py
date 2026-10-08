"""Small builders shared by the tests: toy chunks, hits and a fake embedder.

The tests never load the real embedding model or call Ollama, so they run in
seconds and offline. (Only the tokenizer file, already in the local cache
after the first build, is needed by the chunking tests.)
"""

from __future__ import annotations

import numpy as np

from src.retrieve import Hit
from src.schema import Chunk


def make_chunk(chunk_id: str, text: str = "text", **fields) -> Chunk:
    defaults = dict(
        doc_id="is875",
        doc_title="IS 875 (Part 3):1987",
        section=chunk_id,
        text=text,
        pdf_pages=[1],
        printed_pages=["1"],
    )
    defaults.update(fields)
    return Chunk(chunk_id=chunk_id, **defaults)


def make_hit(chunk: Chunk, reason: str = "") -> Hit:
    return Hit(chunk=chunk, score=0.0, attached_reason=reason)


class FakeEmbedder:
    """Stands in for SentenceTransformer: every text maps to the same unit vector."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, text, normalize_embeddings: bool = True, **_):
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        if isinstance(text, list):  # Like SentenceTransformer: a list gives one row per text.
            return np.tile(vector, (len(text), 1))
        return vector


def unit_embeddings(n: int, dim: int = 4) -> np.ndarray:
    """n identical unit vectors, so dense ranking ties and BM25 decides the order."""
    matrix = np.zeros((n, dim), dtype=np.float32)
    matrix[:, 0] = 1.0
    return matrix
