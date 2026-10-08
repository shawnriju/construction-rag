"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion.

Why hybrid: the corpus is full of exact identifiers ("Clause 10CC",
"Table 28", "Section 34(2)(a)(iii)") that BM25 matches reliably, while
paraphrased questions ("what happens if the contractor is late?") need
semantic matching. RRF combines the two rankings without having to calibrate
their very different score scales.

After fusion, amendment expansion makes sure superseding IS 875 amendments
travel together with the provisions they change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    BGE_QUERY_PREFIX,
    CANDIDATES_PER_RETRIEVER,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    INDEX_META_PATH,
    LOW_CONFIDENCE_COSINE,
    MAX_ATTACHED,
    RRF_K,
    TOP_K,
)
from src.index import chunks_fingerprint
from src.schema import Chunk, load_chunks

# --- Tokenisation for BM25 -----------------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or shall that the "
    "this to was were which with any such than then there these those under what when "
    "where who will would can does do i me my we our you your how".split()
)
# Keeps clause ids intact: "10cc", "6.2.2.8", "34", "c-2.1".
_TOKEN = re.compile(r"[a-z]-\d+(?:\.\d+)*|\d+(?:\.\d+)*[a-z]{0,2}|[a-z]+")
_SPACED_CLAUSE_ID = re.compile(r"\b(\d+)\s+([a-z]{1,2})\b(?=\s|$|[),.;:])")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = _SPACED_CLAUSE_ID.sub(r"\1\2", text)  # "clause 10 cc" -> "clause 10cc"
    return [t for t in _TOKEN.findall(text) if t not in _STOPWORDS]


# --- Results -------------------------------------------------------------------


@dataclass
class Hit:
    """One retrieved chunk plus how it got there (for --debug and the UI)."""

    chunk: Chunk
    score: float                       # RRF score (or raw score in single-retriever modes).
    ranks: dict[str, int] = field(default_factory=dict)  # Retriever name -> 1-based rank.
    attached_reason: str = ""          # Non-empty if added by amendment expansion.


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    best_cosine: float

    @property
    def low_confidence(self) -> bool:
        return self.best_cosine < LOW_CONFIDENCE_COSINE


# --- Retriever -----------------------------------------------------------------


class Retriever:
    """Holds the chunks, the BM25 index and the embedding matrix."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, model: SentenceTransformer):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])
        self._amendments_by_target = self._index_amendments(chunks)

    @classmethod
    def load(cls, chunks_path: Path = CHUNKS_PATH) -> "Retriever":
        """Load prebuilt artifacts; refuses to run on a stale index."""
        chunks = load_chunks(chunks_path)
        meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        if meta["chunks_fingerprint"] != chunks_fingerprint(chunks):
            raise RuntimeError("Embeddings are out of date with chunks.jsonl. Run: python -m src.index")
        model = SentenceTransformer(meta["embed_model"], device="cpu")
        return cls(chunks, np.load(EMBEDDINGS_PATH), model)

    # -- single retrievers --

    def bm25_ranking(self, query: str, k: int = CANDIDATES_PER_RETRIEVER) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_ranking(self, query: str, k: int = CANDIDATES_PER_RETRIEVER) -> list[tuple[int, float]]:
        vector = self.model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.embeddings @ vector
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    # -- fusion --

    @staticmethod
    def rrf(rankings: dict[str, list[tuple[int, float]]]) -> list[tuple[int, float, dict[str, int]]]:
        """Reciprocal Rank Fusion: score = sum over retrievers of 1 / (k + rank)."""
        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for name, ranking in rankings.items():
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
                ranks.setdefault(idx, {})[name] = rank
        order = sorted(fused, key=fused.get, reverse=True)
        return [(idx, fused[idx], ranks[idx]) for idx in order]

    def search(self, query: str, top_k: int = TOP_K, mode: str = "hybrid", expand: bool = True) -> SearchResult:
        """Retrieve the top chunks for a question.

        Args:
            mode: "hybrid" (default), "bm25" or "dense" - the latter two exist for the ablation study.
            expand: Attach amendments to the provisions they modify (and vice versa).
        """
        dense = self.dense_ranking(query)
        best_cosine = dense[0][1] if dense else 0.0

        if mode == "bm25":
            ranked = [(i, s, {"bm25": r}) for r, (i, s) in enumerate(self.bm25_ranking(query), start=1)]
        elif mode == "dense":
            ranked = [(i, s, {"dense": r}) for r, (i, s) in enumerate(dense, start=1)]
        elif mode == "hybrid":
            ranked = self.rrf({"bm25": self.bm25_ranking(query), "dense": dense})
        else:
            raise ValueError(f"Unknown mode: {mode}")

        hits = [Hit(self.chunks[i], score, ranks) for i, score, ranks in ranked[:top_k]]
        if expand:
            hits = self._expand_amendments(hits)
        return SearchResult(query, hits, best_cosine)

    # -- amendment expansion --

    @staticmethod
    def _index_amendments(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        """Provision id ("Table 28", "Cl. 6.2.2.8") -> amendment chunks that modify it."""
        index: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            for target in chunk.amends:
                index.setdefault(target, []).append(chunk)
        return index

    def _related(self, chunk: Chunk) -> list[tuple[Chunk, str]]:
        """Chunks that must accompany `chunk`, each with a human-readable reason."""
        if chunk.is_amendment:  # Show the original text the amendment changes.
            targets = set(chunk.amends)
            return [
                (c, f"original text of {', '.join(sorted(targets & set(c.covers)))}, amended by {chunk.section}")
                for c in self.chunks
                if not c.is_amendment and targets & set(c.covers)
            ]
        return [  # Show amendments that change provisions inside this chunk.
            (amendment, f"{amendment.section} amends {target}")
            for target in chunk.covers
            for amendment in self._amendments_by_target.get(target, [])
        ]

    def _expand_amendments(self, hits: list[Hit]) -> list[Hit]:
        """Insert related amendment/original chunks directly after the hit that needs them."""
        seen = {h.chunk.chunk_id for h in hits}
        expanded: list[Hit] = []
        attached = 0
        for hit in hits:
            expanded.append(hit)
            for chunk, reason in self._related(hit.chunk):
                if chunk.chunk_id in seen or attached >= MAX_ATTACHED:
                    continue
                seen.add(chunk.chunk_id)
                expanded.append(Hit(chunk, 0.0, attached_reason=reason))
                attached += 1
        return expanded
