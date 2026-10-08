"""Checks on the committed artifacts, so a stale or broken index is caught before commit."""

import json

import numpy as np
import pytest

from src.config import CHUNKS_PATH, EMBED_MAX_TOKENS, EMBEDDINGS_PATH, INDEX_META_PATH
from src.index import chunks_fingerprint
from src.ingest.build import embedded_tokens
from src.schema import load_chunks


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(CHUNKS_PATH)


def test_index_matches_the_chunks(chunks):
    meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    assert meta["chunks_fingerprint"] == chunks_fingerprint(chunks), "Run: python -m src.index"
    assert np.load(EMBEDDINGS_PATH).shape[0] == len(chunks)


def test_every_chunk_fits_the_embedding_window(chunks):
    assert max(embedded_tokens(c) for c in chunks) <= EMBED_MAX_TOKENS


def test_chunk_ids_are_unique_and_every_chunk_has_pages(chunks):
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.pdf_pages and c.text.strip() for c in chunks)


def test_every_amendment_targets_a_provision_present_in_the_corpus(chunks):
    # Otherwise retrieval can never attach that amendment to the text it changes.
    covered = {p for c in chunks if not c.is_amendment for p in c.covers}
    missing = {t for c in chunks if c.is_amendment for t in c.amends if t not in covered}
    assert not missing, f"Amendment targets that no chunk covers: {missing}"


def test_ocr_lost_provisions_are_covered_by_exactly_one_chunk(chunks):
    from src.ingest.is875 import ABSORBED_PROVISIONS

    for absorbed in ABSORBED_PROVISIONS:
        hosts = [c.chunk_id for c in chunks if absorbed.provision in c.covers]
        assert len(hosts) == 1, f"{absorbed.provision} covered by {hosts}"


def test_modern_city_names_sit_next_to_the_old_ones(chunks):
    # Added at build time from data/curated/place_aliases.yaml. Inline names made
    # "wind speed in Chennai" reliable on qwen2.5:3b (10/10 vs 8/10 with a separate note).
    appendix = " ".join(c.text for c in chunks if c.parent == "Appendix A")
    for inline in ("Madras [now Chennai] 50", "Bombay [now Mumbai] 44", "Calcutta [now Kolkata] 50"):
        assert inline in appendix


def test_split_tables_have_all_parts(chunks):
    parts: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.parent:
            parts.setdefault(chunk.parent, []).append(chunk.section)
    assert set(parts) == {"Table 2", "Appendix A"}
    for name, sections in parts.items():
        assert sections == [f"{name} (part {i}/{len(sections)})" for i in range(1, len(sections) + 1)]
