"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion.

Why hybrid: the corpus is full of exact identifiers ("Clause 10CC",
"Table 28", "Section 34(2)(a)(iii)") that BM25 matches reliably, while
paraphrased questions ("what happens if the contractor is late?") need
semantic matching. RRF combines the two rankings without having to calibrate
their very different score scales.

After fusion, rare identifiers named in the question are pinned: "Article 142"
occurs in a single chunk, but a paraphrased question ("What relief did the court
grant under Article 142?") can push that chunk out of the top k because dense
search misses it. A chunk holding a rare identifier the user typed is included.

Then two expansions add chunks the LLM needs alongside the hits:
  * sibling expansion ("small-to-big"): a curated table split into parts to
    fit the embedder is searched part by part, but handed to the LLM whole.
    Dense embeddings can't tell table rows apart by number, so the part with
    the asked-for row is often not the part that was found.
  * amendment expansion: superseding IS 875 amendments travel together with
    the provisions they change.
"""

from __future__ import annotations

import json
import re
from collections import Counter
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
    MAX_PINNED,
    MAX_SIBLINGS_ATTACHED,
    RARE_IDENTIFIER_MAX_CHUNKS,
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
# Short English words that follow numbers ("Section 34 of", "Article 142 in", "Tables 2 to 5")
# and must NOT be glued on as if they were a clause suffix ("34of" would never match "34").
_NOT_A_SUFFIX = frozenset("of in to is as at on or by be an if it no so up we do he me my us am".split())


def _join_spaced_clause_id(match: re.Match) -> str:
    number, letters = match.groups()
    return match.group(0) if letters in _NOT_A_SUFFIX else number + letters


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = _SPACED_CLAUSE_ID.sub(_join_spaced_clause_id, text)  # "clause 10 cc" -> "clause 10cc"
    return [t for t in _TOKEN.findall(text) if t not in _STOPWORDS]


# --- Results -------------------------------------------------------------------


@dataclass
class Hit:
    """One retrieved chunk plus how it got there (for --debug and the UI)."""

    chunk: Chunk
    score: float                       # RRF score (or raw score in single-retriever modes).
    ranks: dict[str, int] = field(default_factory=dict)  # Retriever name -> 1-based rank.
    attached_reason: str = ""          # Non-empty if added by sibling or amendment expansion.


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
        corpus = [tokenize(c.index_text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self._chunk_tokens = [set(tokens) for tokens in corpus]
        self._chunk_frequency = Counter(token for tokens in self._chunk_tokens for token in tokens)
        self._amendments_by_target = self._index_amendments(chunks)
        self._parts_by_parent = self._index_parts(chunks)

    @classmethod
    def load(cls, chunks_path: Path = CHUNKS_PATH) -> "Retriever":
        """Load prebuilt artifacts; refuses to run on a stale index."""
        chunks = load_chunks(chunks_path)
        meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        if meta["chunks_fingerprint"] != chunks_fingerprint(chunks):
            raise RuntimeError("Embeddings are out of date with chunks.jsonl. Run: python -m src.index")
        # No fallback to another model: the stored vectors only match the model that made them.
        try:
            model = SentenceTransformer(meta["embed_model"], device="cpu")
        except OSError as error:
            raise RuntimeError(
                f"Could not load the embedding model '{meta['embed_model']}' that built the index. "
                "The first run downloads it from the Hugging Face Hub (~130 MB): check the internet "
                f"connection and try again. ({error})"
            ) from error
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

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        mode: str = "hybrid",
        expand: bool = True,
        siblings: bool = True,
        pin: bool = True,
    ) -> SearchResult:
        """Retrieve the top chunks for a question.

        Args:
            mode: "hybrid" (default), "bm25" or "dense" - the latter two exist for the ablation study.
            expand: Attach amendments to the provisions they modify (and vice versa).
            siblings: Attach the other parts of a split curated table.
            pin: Include chunks holding a rare identifier named in the question.
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
        if pin:
            hits = self._pin_rare_identifiers(query, hits)
        if siblings:
            hits = self._attach_siblings(hits)
        if expand:
            hits = self._expand_amendments(hits)
        return SearchResult(query, hits, best_cosine)

    # -- identifier pinning --

    def _pin_rare_identifiers(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Append chunks that contain a rare number-like identifier from the question (best BM25 first)."""
        rare = [
            token for token in dict.fromkeys(tokenize(query))
            if any(ch.isdigit() for ch in token) and 0 < self._chunk_frequency[token] <= RARE_IDENTIFIER_MAX_CHUNKS
        ]
        if not rare:
            return hits
        seen = {h.chunk.chunk_id for h in hits}
        scores = self.bm25.get_scores(tokenize(query))
        pinned: list[Hit] = []
        for token in rare:
            holders = sorted((i for i, tokens in enumerate(self._chunk_tokens) if token in tokens), key=lambda i: -scores[i])
            for i in holders:
                if len(pinned) >= MAX_PINNED:
                    break
                if self.chunks[i].chunk_id not in seen:
                    seen.add(self.chunks[i].chunk_id)
                    pinned.append(Hit(self.chunks[i], float(scores[i]), attached_reason=f"contains '{token}' from the question"))
        return hits + pinned

    # -- sibling expansion --

    @staticmethod
    def _index_parts(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        """Split table name ("Table 2") -> its parts, in document order."""
        index: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.parent:
                index.setdefault(chunk.parent, []).append(chunk)
        return index

    def _attach_siblings(self, hits: list[Hit]) -> list[Hit]:
        """Insert the missing parts of a split table directly after its first retrieved part."""
        seen = {h.chunk.chunk_id for h in hits}
        expanded: list[Hit] = []
        attached = 0
        for hit in hits:
            expanded.append(hit)
            for part in self._parts_by_parent.get(hit.chunk.parent, []):
                if part.chunk_id in seen or attached >= MAX_SIBLINGS_ATTACHED:
                    continue
                seen.add(part.chunk_id)
                reason = f"rest of {part.parent}: {hit.chunk.section} was retrieved"
                expanded.append(Hit(part, 0.0, attached_reason=reason))
                attached += 1
        return expanded

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
