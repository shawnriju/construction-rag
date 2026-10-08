"""Parser for Ssangyong Engineering v. NHAI, Supreme Court of India, 2019.

Layout facts this parser relies on (verified against the PDF):
  * Every line of the judgment is its own text block. Body text starts at
    x ~ 99; quotations from other judgments/reports are indented (x >= 140).
  * The Court's own paragraphs start with a block whose first line is "N."
    at the body margin. Quoted passages also contain numbers ("34. ...") but
    they are indented, and we additionally require the numbers to be sequential.
  * Line spacing is ~32 pt; a gap of ~44 pt separates paragraphs.
  * Section headings ("Most Basic Notions of Justice") are 1-2 short lines at
    the body margin, without a full stop, directly before a paragraph number.
  * The page number printed at the bottom equals the PDF page number.

Output: chunks of whole paragraphs (e.g. "¶ 23-24"); a paragraph longer than
the size limit is split into parts ("¶ 33 (part 2/4)").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest.pdf import TextBlock, read_blocks
from src.ingest.text import Paragraph, Unit, group_units, join_lines, word_count
from src.schema import Chunk

DOC_ID = "ssangyong"
DOC_TITLE = "Ssangyong v NHAI (SC 2019)"
PDF_NAME = "Ssangyong-v-NHAI_SupremeCourt-arbitration_EN.pdf"

BODY_MAX_X0 = 110          # Blocks left of this are at the body margin.
SIDEBAR_MAX_X1 = 99        # Digital-signature stamp in the left margin.
FOOTER_MIN_Y = 760         # Page number at the bottom.
PARAGRAPH_GAP = 38         # y0 jump bigger than normal line spacing (~32).
HEADING_MAX_WORDS = 15

_PARA_START = re.compile(r"^(\d{1,3})\.$")
_SIGNATURE_LINE = re.compile(r"^[.…\s]*J\.?$")


@dataclass
class _Group:
    """Consecutive lines with the same indentation and no paragraph gap."""

    lines: list[str]
    x0: float
    pdf_page: int
    para_number: int | None = None  # Set if the group opens a numbered paragraph.

    @property
    def text(self) -> str:
        return join_lines(self.lines)


@dataclass
class _LegalParagraph:
    number: int | None  # None for the case header before paragraph 1.
    heading: str
    pieces: list[Paragraph] = field(default_factory=list)


# --- Step 1: blocks -> line groups --------------------------------------------


def _body_blocks(pages: dict[int, list[TextBlock]]) -> list[TextBlock]:
    blocks = []
    for page_no in sorted(pages):
        for block in pages[page_no]:
            if block.y0 > FOOTER_MIN_Y or block.x1 <= SIDEBAR_MAX_X1:
                continue
            blocks.append(block)
    return blocks


def _groups(blocks: list[TextBlock]) -> list[_Group]:
    groups: list[_Group] = []
    prev: TextBlock | None = None
    for block in blocks:
        lines = [ln for ln in block.lines if not _SIGNATURE_LINE.match(ln)]
        if not lines:
            continue
        starts_paragraph = bool(_PARA_START.match(lines[0])) and block.x0 < BODY_MAX_X0
        new_group = (
            prev is None
            or starts_paragraph
            or block.pdf_page != prev.pdf_page  # One group per page keeps page citations exact.
            or abs(block.x0 - prev.x0) > 10  # Indentation changed (quote starts/ends).
            or block.y0 - prev.y0 > PARAGRAPH_GAP
        )
        if new_group:
            groups.append(_Group([], block.x0, block.pdf_page))
        groups[-1].lines.extend(lines)
        prev = block
    return groups


def _mark_paragraph_starts(groups: list[_Group]) -> None:
    """Tag groups that open the Court's numbered paragraphs (numbers must be sequential)."""
    expected = 1
    for group in groups:
        match = _PARA_START.match(group.lines[0])
        if match and group.x0 < BODY_MAX_X0 and int(match.group(1)) == expected:
            group.para_number = expected
            group.lines = group.lines[1:]  # Drop the bare "N." line.
            expected += 1


def _is_heading(group: _Group, next_group: _Group | None) -> bool:
    text = group.text
    return (
        next_group is not None
        and next_group.para_number is not None
        and group.para_number is None
        and group.x0 < BODY_MAX_X0
        and word_count(text) <= HEADING_MAX_WORDS
        # Headings have no sentence punctuation; '"' rules out the tail of a quotation.
        # (Not ')': real headings end with "Section 34(2)(a)(iii)".)
        and not text.endswith((".", ",", ";", ":", '"'))
    )


# --- Step 2: line groups -> numbered paragraphs --------------------------------


def _legal_paragraphs(groups: list[_Group]) -> list[_LegalParagraph]:
    paragraphs = [_LegalParagraph(number=None, heading="Case details")]
    heading = ""
    for i, group in enumerate(groups):
        next_group = groups[i + 1] if i + 1 < len(groups) else None
        if paragraphs[-1].number is not None and _is_heading(group, next_group):
            heading = group.text
            continue
        if group.para_number is not None:
            paragraphs.append(_LegalParagraph(group.para_number, heading))
        if group.lines:
            page = str(group.pdf_page)
            paragraphs[-1].pieces.append(Paragraph(group.text, group.pdf_page, page))
    return paragraphs


# --- Step 3: paragraphs -> chunks ----------------------------------------------


def _to_unit(paragraph: _LegalParagraph) -> Unit:
    if paragraph.number is None:  # Parties, court and appeal number.
        return Unit("Case details", "case-details", "", paragraph.pieces, mergeable=False)
    return Unit(str(paragraph.number), paragraph.heading, paragraph.heading, paragraph.pieces)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower().replace("¶", "p")).strip("-")


def parse(pdf_dir: Path) -> list[Chunk]:
    """Parse the Ssangyong judgment into paragraph-based, citation-ready chunks."""
    groups = _groups(_body_blocks(read_blocks(pdf_dir / PDF_NAME)))
    _mark_paragraph_starts(groups)
    units = [_to_unit(p) for p in _legal_paragraphs(groups)]

    chunks = []
    for window in group_units(units):
        is_header = window.keys == ["Case details"]
        label = "Case details" if is_header else window.label("¶ ")
        pages = sorted({p.pdf_page for p in window.paragraphs})
        chunks.append(
            Chunk(
                chunk_id=f"{DOC_ID}:{_slug(label)}",
                doc_id=DOC_ID,
                doc_title=DOC_TITLE,
                part="Judgment (R.F. Nariman, J.)",
                section=label,
                title=window.title,
                text="\n".join(p.text for p in window.paragraphs),
                pdf_pages=pages,
                printed_pages=[str(p) for p in pages],
                covers=[] if is_header else [f"¶ {k}" for k in window.keys],
            )
        )
    return chunks
