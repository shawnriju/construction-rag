"""Before/after check for the token-based chunking fix.

Compares the current build with an older commit (default: 74f8585, the
word-based chunking). The old chunks and embeddings are read straight from git,
so nothing needs to be checked out. Deterministic, no LLM needed.

Usage:
    python -m eval.compare_chunking [--old-ref 74f8585]

Three checks:
  1. Truncation - chunks whose embedded text (breadcrumb + body) is longer
     than the embedder's 512-token window, so their tail is silently dropped.
  2. Tail findability - for every chunk the old build truncated, a passage
     from the lost tail is used as a dense query, and we report the rank of the
     chunk that holds it. The old embedding never saw that text.
  3. Table lookups - questions about specific Table 2 rows and Appendix A
     cities: does the needed row reach the LLM's context (after expansion)?
     Run for old / new without sibling expansion / new with it.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
from dataclasses import dataclass

import numpy as np

from src.config import EMBED_MAX_TOKENS
from src.ingest.build import embedded_tokens
from src.ingest.text import _tokenizer
from src.retrieve import Retriever
from src.schema import Chunk

DEFAULT_OLD_REF = "74f8585"
_TAIL_QUERY_WORDS = 25  # Long enough to be distinctive, short like a real passage query.
_LOCATE_WORDS = 8       # Words used to find which chunk holds the tail passage.


@dataclass
class TableProbe:
    question: str
    table: str     # Section prefix of the curated table that must supply the answer.
    patterns: tuple[str, ...]  # Regexes for the rows/entries the answer needs (all of them).


# Answers come from the hand-checked curated tables (data/curated/is875.yaml).
# The two interpolation questions need rows that sit in different Table 2 parts.
PROBES = [
    TableProbe("What is the k2 factor at 10 m height for terrain category 1, class A?", "Table 2", (r"\|\s*10\s*\|",)),
    TableProbe("k2 multiplier at 30 m height in terrain category 3 for a class B building", "Table 2", (r"\|\s*30\s*\|",)),
    TableProbe("What is the k2 factor at 100 m height for terrain category 2?", "Table 2", (r"\|\s*100\s*\|",)),
    TableProbe("k2 value for a 250 m tall structure in terrain category 4, class C", "Table 2", (r"\|\s*250\s*\|",)),
    TableProbe("What is the k2 factor at 400 m height?", "Table 2", (r"\|\s*400\s*\|",)),
    TableProbe("What is the k2 factor at 450 m height for terrain category 2, class A?", "Table 2", (r"\|\s*450\s*\|",)),
    TableProbe("Design wind speed multiplier k2 at 500 m height in terrain category 1", "Table 2", (r"\|\s*500\s*\|",)),
    TableProbe("k2 at 40 m height for terrain category 2, class A, by interpolation", "Table 2",
               (r"\|\s*30\s*\|", r"\|\s*50\s*\|")),
    TableProbe("Interpolate the k2 factor for a 225 m tall tower in terrain category 3", "Table 2",
               (r"\|\s*200\s*\|", r"\|\s*250\s*\|")),
    TableProbe("What is the basic wind speed for Agra?", "Appendix A", (r"Agra 47",)),
    TableProbe("Basic wind speed in Delhi", "Appendix A", (r"Delhi 47",)),
    TableProbe("What is the basic wind speed in Mumbai?", "Appendix A", (r"Bombay 44",)),
    TableProbe("What is the basic wind speed in Chennai?", "Appendix A", (r"Madras 50",)),
    TableProbe("Basic wind speed for Thiruvananthapuram", "Appendix A", (r"Trivandrum 39",)),
    TableProbe("What is the basic wind speed at Vadodara?", "Appendix A", (r"Vadodara 44",)),
    TableProbe("Basic wind speed for Visakhapatnam", "Appendix A", (r"Visakhapatnam 50",)),
]


def _git_bytes(ref: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load_old(ref: str, model) -> Retriever:
    lines = _git_bytes(ref, "artifacts/chunks.jsonl").decode("utf-8").splitlines()
    chunks = [Chunk(**json.loads(line)) for line in lines if line.strip()]
    embeddings = np.load(io.BytesIO(_git_bytes(ref, "artifacts/embeddings.npy")))
    return Retriever(chunks, embeddings, model)


def _squash(text: str) -> str:
    return " ".join(text.split())


def check_truncation(old: Retriever, new: Retriever) -> list[Chunk]:
    print(f"\n1. Chunks over the {EMBED_MAX_TOKENS}-token window (their tail is invisible to dense search)")
    over_old = [c for c in old.chunks if embedded_tokens(c) > EMBED_MAX_TOKENS]
    over_new = [c for c in new.chunks if embedded_tokens(c) > EMBED_MAX_TOKENS]
    print(f"   old: {len(over_old)} of {len(old.chunks)}    new: {len(over_new)} of {len(new.chunks)}")
    return over_old


def _lost_tail(chunk: Chunk) -> str:
    """The part of the chunk's embedded text past the window, i.e. what the old embedding never saw."""
    content_limit = EMBED_MAX_TOKENS - 2  # [CLS] and [SEP]
    offsets = _tokenizer().encode(chunk.index_text, add_special_tokens=False).offsets
    return chunk.index_text[offsets[content_limit - 1][1]:]


def _dense_rank(retriever: Retriever, query: str, locate: str) -> str:
    holders = {i for i, c in enumerate(retriever.chunks) if locate in _squash(c.text)}
    if not holders:
        return "n/a"
    ranking = retriever.dense_ranking(query, k=len(retriever.chunks))
    rank = next(r for r, (i, _) in enumerate(ranking, start=1) if i in holders)
    return str(rank)


def check_tails(old: Retriever, new: Retriever, truncated: list[Chunk]) -> None:
    print("\n2. Dense rank of the chunk holding text from a lost tail (1 = best; lower is better)")
    print(f"   {'old chunk':<46}{'old rank':>9}{'new rank':>9}")
    for chunk in truncated:
        words = _lost_tail(chunk).split()[1:]  # Drop the first word: the cut may fall mid-word.
        query = " ".join(words[:_TAIL_QUERY_WORDS])
        locate = " ".join(words[:_LOCATE_WORDS])
        print(f"   {chunk.chunk_id:<46}{_dense_rank(old, query, locate):>9}{_dense_rank(new, query, locate):>9}")


def _probe(retriever: Retriever, probe: TableProbe, siblings: bool) -> tuple[str, int]:
    """'retrieved' if search found every needed row, 'attached' if expansion supplied one, else 'MISSING'."""
    hits = retriever.search(probe.question, siblings=siblings).hits
    table_hits = [
        h for h in hits if h.chunk.source == "curated" and h.chunk.section.startswith(probe.table)
    ]
    status = "retrieved"
    for pattern in probe.patterns:
        holder = next((h for h in table_hits if re.search(pattern, h.chunk.text)), None)
        if holder is None:
            return "MISSING", len(hits)
        if holder.attached_reason:
            status = "attached"
    return status, len(hits)


def check_tables(old: Retriever, new: Retriever) -> None:
    print("\n3. Does the row that answers the question reach the LLM's context?")
    configs = [("old", old, False), ("new, no siblings", new, False), ("new + siblings", new, True)]
    print(f"   {'question':<62}" + "".join(f"{name:>18}" for name, _, _ in configs))
    found = {name: 0 for name, _, _ in configs}
    sent = {name: 0 for name, _, _ in configs}
    for probe in PROBES:
        row = f"   {probe.question[:60]:<62}"
        for name, retriever, siblings in configs:
            status, n_hits = _probe(retriever, probe, siblings)
            found[name] += status != "MISSING"
            sent[name] += n_hits
            row += f"{status:>18}"
        print(row)
    print(f"   {'answer row in context':<62}" + "".join(f"{f'{found[n]}/{len(PROBES)}':>18}" for n in found))
    print(f"   {'avg chunks sent to the LLM':<62}" + "".join(f"{sent[n] / len(PROBES):>18.1f}" for n in sent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-ref", default=DEFAULT_OLD_REF, help="git commit of the old build")
    args = parser.parse_args()

    new = Retriever.load()
    old = load_old(args.old_ref, new.model)
    print(f"old = commit {args.old_ref} ({len(old.chunks)} chunks), new = working tree ({len(new.chunks)} chunks)")
    truncated = check_truncation(old, new)
    check_tails(old, new, truncated)
    check_tables(old, new)


if __name__ == "__main__":
    main()
