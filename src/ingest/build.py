"""Ingestion entry point: PDFs + curated content -> artifacts/chunks.jsonl.

Usage:
    python -m src.ingest.build
"""

from __future__ import annotations

from collections import Counter

from src.config import CHUNKS_PATH, CURATED_DIR, PDF_DIR
from src.ingest import cpwd, curated, is875, ssangyong
from src.ingest.text import word_count
from src.schema import Chunk, save_chunks


def build_chunks() -> list[Chunk]:
    chunks = [
        *cpwd.parse(PDF_DIR),
        *ssangyong.parse(PDF_DIR),
        *is875.parse(PDF_DIR),
        *curated.load(CURATED_DIR),
    ]
    _validate(chunks)
    return chunks


def _validate(chunks: list[Chunk]) -> None:
    """Fail loudly on problems that would silently corrupt citations."""
    duplicates = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate chunk ids: {duplicates[:10]}")
    empty = [c.chunk_id for c in chunks if not c.text.strip()]
    if empty:
        raise ValueError(f"Empty chunks: {empty[:10]}")
    no_pages = [c.chunk_id for c in chunks if not c.pdf_pages]
    if no_pages:
        raise ValueError(f"Chunks without page provenance: {no_pages[:10]}")


def _print_summary(chunks: list[Chunk]) -> None:
    print(f"{'document':<12}{'chunks':>8}{'words(avg)':>12}{'amendments':>12}{'curated':>9}")
    for doc_id in dict.fromkeys(c.doc_id for c in chunks):
        docs = [c for c in chunks if c.doc_id == doc_id]
        avg = sum(word_count(c.text) for c in docs) / len(docs)
        amendments = sum(c.is_amendment for c in docs)
        curated_count = sum(c.source == "curated" for c in docs)
        print(f"{doc_id:<12}{len(docs):>8}{avg:>12.0f}{amendments:>12}{curated_count:>9}")
    print(f"{'total':<12}{len(chunks):>8}")


def main() -> None:
    chunks = build_chunks()
    save_chunks(chunks, CHUNKS_PATH)
    _print_summary(chunks)
    print(f"\nWrote {CHUNKS_PATH.relative_to(CHUNKS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
