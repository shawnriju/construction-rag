"""Loads hand-curated content (data/curated/is875.yaml) as chunks.

Two kinds of entries:
  * tables   - clean transcriptions of tables the OCR destroyed.
  * amendments - one chunk per amendment item, flagged `is_amendment=True`
    with the provisions it `amends`, so retrieval can attach it to the
    original text it supersedes (see src/retrieve.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.ingest.is875 import DOC_ID, DOC_TITLE
from src.schema import Chunk

CURATED_FILE = "is875.yaml"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _table_chunk(entry: dict) -> Chunk:
    return Chunk(
        chunk_id=f"{DOC_ID}:curated:{_slug(entry['section'])}",
        doc_id=DOC_ID,
        doc_title=DOC_TITLE,
        part=entry.get("part", ""),
        section=entry["section"],
        title=entry["title"],
        text=entry["text"].strip(),
        pdf_pages=entry["pdf_pages"],
        printed_pages=entry["printed_pages"],
        covers=entry.get("covers", [entry["section"]]),
        source="curated",
    )


def _amendment_chunks(amendment: dict) -> list[Chunk]:
    number, date = amendment["number"], amendment["date"]
    chunks = []
    for i, item in enumerate(amendment["items"], start=1):
        targets = ", ".join(item["amends"]) or "all clauses"
        chunks.append(
            Chunk(
                chunk_id=f"{DOC_ID}:amd{number}:item-{i}",
                doc_id=DOC_ID,
                doc_title=DOC_TITLE,
                part="Amendments",
                section=f"Amendment No. {number}, item {i}",
                title=f"Amendment No. {number} ({date}) - amends {targets}",
                text=f"Amendment No. {number} ({date}) to IS 875 (Part 3):1987. {item['text']}",
                pdf_pages=amendment["pdf_pages"],
                printed_pages=[],
                is_amendment=True,
                amends=item["amends"],
                source="curated",
            )
        )
    return chunks


def load(curated_dir: Path) -> list[Chunk]:
    with (curated_dir / CURATED_FILE).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    chunks = [_table_chunk(entry) for entry in data["tables"]]
    for amendment in data["amendments"]:
        chunks.extend(_amendment_chunks(amendment))
    return chunks
