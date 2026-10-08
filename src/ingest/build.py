"""Ingestion entry point: PDFs + curated content -> artifacts/chunks.jsonl.

Usage:
    python -m src.ingest.build
"""

from __future__ import annotations

from collections import Counter

from src.config import CHUNKS_PATH, CURATED_DIR, EMBED_MAX_TOKENS, PDF_DIR
from src.ingest import cpwd, curated, is875, ssangyong
from src.ingest.text import token_count, word_count
from src.schema import Chunk, load_chunks, save_chunks

_SPECIAL_TOKENS = 2  # [CLS] and [SEP], which also count against the window.


def build_chunks() -> list[Chunk]:
    chunks = [
        *cpwd.parse(PDF_DIR),
        *ssangyong.parse(PDF_DIR),
        *is875.parse(PDF_DIR),
        *curated.load(CURATED_DIR),
    ]
    _validate(chunks)
    return chunks


def embedded_tokens(chunk: Chunk) -> int:
    """Tokens the embedder sees for this chunk (breadcrumb + text + special tokens)."""
    return token_count(chunk.index_text) + _SPECIAL_TOKENS


def _validate(chunks: list[Chunk]) -> None:
    """Fail loudly on problems that would silently corrupt citations or retrieval."""
    duplicates = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate chunk ids: {duplicates[:10]}")
    empty = [c.chunk_id for c in chunks if not c.text.strip()]
    if empty:
        raise ValueError(f"Empty chunks: {empty[:10]}")
    no_pages = [c.chunk_id for c in chunks if not c.pdf_pages]
    if no_pages:
        raise ValueError(f"Chunks without page provenance: {no_pages[:10]}")
    # The embedder truncates past its window without any error, so the tail of
    # an oversized chunk would be invisible to dense retrieval.
    oversized = [f"{c.chunk_id} ({embedded_tokens(c)})" for c in chunks if embedded_tokens(c) > EMBED_MAX_TOKENS]
    if oversized:
        raise ValueError(f"Chunks over the {EMBED_MAX_TOKENS}-token embedding window: {oversized[:10]}")


def _print_summary(chunks: list[Chunk]) -> None:
    print(f"{'document':<12}{'chunks':>8}{'words(avg)':>12}{'tokens(max)':>13}{'amendments':>12}{'curated':>9}")
    for doc_id in dict.fromkeys(c.doc_id for c in chunks):
        docs = [c for c in chunks if c.doc_id == doc_id]
        avg = sum(word_count(c.text) for c in docs) / len(docs)
        longest = max(embedded_tokens(c) for c in docs)
        amendments = sum(c.is_amendment for c in docs)
        curated_count = sum(c.source == "curated" for c in docs)
        print(f"{doc_id:<12}{len(docs):>8}{avg:>12.0f}{longest:>13}{amendments:>12}{curated_count:>9}")
    print(f"{'total':<12}{len(chunks):>8}")


def _print_id_changes(old_ids: list[str], chunks: list[Chunk]) -> None:
    """Show which chunk ids appeared or disappeared since the last build.

    Gold questions and fine-tuning pairs reference chunk ids, so a rebuild that
    renames chunks must be visible, not silent.
    """
    new_ids = [c.chunk_id for c in chunks]
    removed = [cid for cid in old_ids if cid not in set(new_ids)]
    added = [cid for cid in new_ids if cid not in set(old_ids)]
    print(f"\nChunk ids vs previous build: {len(old_ids)} -> {len(new_ids)}, "
          f"{len(removed)} removed, {len(added)} added, {len(new_ids) - len(added)} unchanged ids")
    for cid in removed:
        print(f"  - {cid}")
    for cid in added:
        print(f"  + {cid}")


def main() -> None:
    old_ids = [c.chunk_id for c in load_chunks(CHUNKS_PATH)] if CHUNKS_PATH.exists() else []
    chunks = build_chunks()
    save_chunks(chunks, CHUNKS_PATH)
    _print_summary(chunks)
    _print_id_changes(old_ids, chunks)
    print(f"\nWrote {CHUNKS_PATH.relative_to(CHUNKS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
