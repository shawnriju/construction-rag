"""Builds the dense index: one normalized embedding per chunk.

BM25 is not persisted - it is rebuilt from chunks.jsonl at load time in a
few milliseconds, which avoids pickling and keeps artifacts small.

Usage:
    python -m src.index                      # embed with config.EMBED_MODEL
    python -m src.index --model <hf-id|path> # e.g. the fine-tuned model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBED_MODEL, EMBEDDINGS_PATH, INDEX_META_PATH
from src.schema import Chunk, load_chunks


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    """Hash of chunk ids + texts, so a stale index is detected at load time."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(chunk.index_text.encode())
    return digest.hexdigest()[:16]


def embed_passages(model: SentenceTransformer, chunks: list[Chunk]) -> np.ndarray:
    """Embed chunk texts (BGE passages need no instruction prefix)."""
    vectors = model.encode(
        [c.index_text for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build(model_name: str) -> None:
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Embedding {len(chunks)} chunks with {model_name} ...")
    start = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = embed_passages(model, chunks)
    np.save(EMBEDDINGS_PATH, embeddings)
    meta = {
        "embed_model": model_name,
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
        "chunks_fingerprint": chunks_fingerprint(chunks),
    }
    INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {embeddings.shape} embeddings in {time.perf_counter() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=EMBED_MODEL, help="Embedding model (HF id or local path).")
    build(parser.parse_args().model)


if __name__ == "__main__":
    main()
