"""The Chunk data model shared by ingestion, indexing, retrieval and the UI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    """One retrievable unit of text plus everything needed to cite it.

    Attributes:
        chunk_id: Stable, human-readable id, e.g. "cpwd:clause-10CC:1".
        doc_id: Short document key: "cpwd", "ssangyong" or "is875".
        doc_title: Display name used in citations.
        section: Citation label, e.g. "Clause 10CC", "¶ 23-24", "Cl. 5.3.2".
        title: Optional heading of the section (e.g. a margin heading).
        text: The cleaned chunk body (without the breadcrumb).
        pdf_pages: 1-based PDF page numbers the text came from.
        printed_pages: Page numbers as printed on the document itself.
        part: Larger division of the document, e.g. "Clauses of Contract".
        covers: Provision ids contained in this chunk (e.g. ["Cl. 6.2.2.8",
            "Cl. 6.2.2.9"]). Used to find amendments that modify the chunk.
        is_amendment: True if this chunk is an amendment to another provision.
        amends: Provision ids this chunk modifies (only for amendments).
        source: "pdf" for extracted text, "curated" for hand-transcribed content.
        parent: For one part of a curated table that was split to fit the
            embedder (e.g. "Table 2"): the whole table's name. Retrieval
            attaches the other parts, so the LLM always sees the full table.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    pdf_pages: list[int]
    printed_pages: list[str]
    title: str = ""
    part: str = ""
    covers: list[str] = field(default_factory=list)
    is_amendment: bool = False
    amends: list[str] = field(default_factory=list)
    source: str = "pdf"
    parent: str = ""

    @property
    def breadcrumb(self) -> str:
        """Where the chunk sits in its document, e.g. 'CPWD GCC 2020 > Clauses of Contract > Clause 2'."""
        parts = [self.doc_title, self.part, self.section]
        crumb = " > ".join(p for p in parts if p)
        return f"{crumb} - {self.title}" if self.title else crumb

    @property
    def index_text(self) -> str:
        """Text that gets embedded and BM25-indexed: breadcrumb gives context the body may lack."""
        return f"{self.breadcrumb}\n{self.text}"

    @property
    def page_label(self) -> str:
        """Human-readable page reference, e.g. 'p. 28 (PDF 30)' or 'pp. 28-29 (PDF 30-31)'."""
        pdf = _span([str(p) for p in self.pdf_pages])
        if not self.printed_pages:  # e.g. amendment slips, which carry no page number.
            return f"PDF p. {pdf}"
        printed = _span(self.printed_pages)
        prefix = "pp." if len(set(self.printed_pages)) > 1 else "p."
        return f"{prefix} {printed} (PDF {pdf})" if printed != pdf else f"{prefix} {printed}"

    @property
    def citation(self) -> str:
        """Full citation string, e.g. 'CPWD GCC 2020, Clause 10CC, p. 28 (PDF 30)'."""
        return f"{self.doc_title}, {self.section}, {self.page_label}"


def _span(pages: list[str]) -> str:
    """Collapse a page list to 'first' or 'first-last'."""
    unique = list(dict.fromkeys(pages))  # De-duplicate but keep order.
    if not unique:
        return "?"
    return unique[0] if len(unique) == 1 else f"{unique[0]}-{unique[-1]}"


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
