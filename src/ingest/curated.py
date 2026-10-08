"""Loads hand-curated content (data/curated/is875.yaml) as chunks.

Two kinds of entries:
  * tables   - clean transcriptions of tables the OCR destroyed.
  * amendments - one chunk per amendment item, flagged `is_amendment=True`
    with the provisions it `amends`, so retrieval can attach it to the
    original text it supersedes (see src/retrieve.py).

The transcriptions are kept exactly as printed. Anything the curator adds
(today's names for renamed cities, from data/curated/place_aliases.yaml) is
applied here at build time and clearly marked, never written into the transcription.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.ingest.is875 import DOC_ID, DOC_TITLE
from src.schema import Chunk

CURATED_FILE = "is875.yaml"
PLACE_ALIASES_FILE = "place_aliases.yaml"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --- Curator enrichment: today's place names --------------------------------------


@dataclass(frozen=True)
class PlaceAliases:
    """Old place name -> today's name, added next to the old name in selected entries."""

    apply_to: frozenset[str] = frozenset()   # Curated sections to annotate, e.g. {"Appendix A"}.
    legend: str = ""                          # Line explaining the brackets, added once per annotated text.
    aliases: dict[str, str] = field(default_factory=dict)

    def applies_to(self, section: str) -> bool:
        return section in self.apply_to

    def annotate(self, text: str) -> str:
        """'Madras 50' -> 'Madras [now Chennai] 50' (whole words only), plus the legend if anything changed."""
        annotated = text
        for old, new in self.aliases.items():
            annotated = re.sub(rf"\b{re.escape(old)}\b(?! \[now )", f"{old} [now {new}]", annotated)
        if annotated == text:
            return text
        return f"{annotated}\n{self.legend}" if self.legend else annotated


def load_place_aliases(curated_dir: Path) -> PlaceAliases:
    path = curated_dir / PLACE_ALIASES_FILE
    if not path.exists():
        return PlaceAliases()
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PlaceAliases(
        apply_to=frozenset(data.get("apply_to", [])),
        legend=data.get("legend", ""),
        aliases=dict(data.get("aliases", {})),
    )


# --- Tables and amendments ------------------------------------------------------------


def _table_chunk(entry: dict, section: str, title: str, text: str, parent: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"{DOC_ID}:curated:{_slug(section)}",
        doc_id=DOC_ID,
        doc_title=DOC_TITLE,
        part=entry.get("part", ""),
        section=section,
        title=title,
        text=text.strip(),
        pdf_pages=entry["pdf_pages"],
        printed_pages=entry["printed_pages"],
        covers=entry.get("covers", [entry["section"]]),
        source="curated",
        parent=parent,
    )


def _table_chunks(entry: dict, places: PlaceAliases) -> list[Chunk]:
    """One chunk per table, or one per part for a table too long for the embedder.

    A split table repeats its `header` (caption + column headings) and `footer`
    (notes) in every part, so each part can be read and cited on its own.
    All parts keep `covers` = the whole table, so amendments still attach, and
    `parent` = the table's name, so retrieval can attach the missing parts.
    """
    def body(text: str) -> str:
        text = text.strip()
        return places.annotate(text) if places.applies_to(entry["section"]) else text

    if "parts" not in entry:
        return [_table_chunk(entry, entry["section"], entry["title"], body(entry["text"]))]
    header, footer, parts = entry["header"].strip(), entry.get("footer", "").strip(), entry["parts"]
    chunks = []
    for i, part in enumerate(parts, start=1):
        text = "\n\n".join(t for t in (header, body(part["text"]), footer) if t)
        section = f"{entry['section']} (part {i}/{len(parts)})"
        title = f"{entry['title']} ({part['label']})"
        chunks.append(_table_chunk(entry, section, title, text, parent=entry["section"]))
    return chunks


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
    places = load_place_aliases(curated_dir)
    sections = {entry["section"] for entry in data["tables"]}
    unknown = places.apply_to - sections
    if unknown:  # A typo in apply_to would silently disable the enrichment.
        raise ValueError(f"{PLACE_ALIASES_FILE}: apply_to names unknown sections {sorted(unknown)}")
    chunks = [chunk for entry in data["tables"] for chunk in _table_chunks(entry, places)]
    for amendment in data["amendments"]:
        chunks.extend(_amendment_chunks(amendment))
    return chunks
