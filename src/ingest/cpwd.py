"""Parser for the CPWD General Conditions of Contract 2020 (Construction Works).

Layout facts this parser relies on (verified against the PDF):
  * Every page carries a running header (the part name, e.g. "CLAUSES OF
    CONTRACT") at y < 80 and a footer "<printed page> / 165 Years of
    Engineering Excellence" at the bottom.
  * Rows of "1234567890..." are decoration and must be removed.
  * The General Rules, Conditions and Clauses parts are two-column: body text
    starts at x ~ 165 and short "margin headings" (clause titles such as
    "Performance Guarantee") sit at x ~ 54 with a right edge below x ~ 160.
  * Clause boundaries are blocks whose first line is exactly "Clause <id>",
    e.g. "Clause 10CC", "CLAUSE 19A", "Clause 1 A".

Output: one chunk per clause (long clauses split into several windows), and
size-bounded windows of numbered items for every other part.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest.pdf import TextBlock, read_blocks
from src.ingest.text import (
    Paragraph,
    is_ruler_line,
    join_lines,
    looks_like_legacy_hindi,
    normalize,
    pack,
)
from src.schema import Chunk

DOC_ID = "cpwd"
DOC_TITLE = "CPWD GCC 2020"
PDF_NAME = "CPWD-GCC-2020_construction-contract_EN.pdf"

# Pages 1-3 are the office memo, cover and a bilingual index; 80-95 are
# bilingual register proformas (mostly Hindi in a legacy font); 112 is the back cover.
INCLUDED_PAGES = [*range(4, 80), *range(96, 112)]

HEADER_MAX_Y = 80
FOOTER_MARKER = "Engineering Excellence"
MARGIN_MAX_X0 = 100
MARGIN_MAX_X1 = 160

CLAUSES_PART = "Clauses of Contract"
TWO_COLUMN_PARTS = {"General Rules & Directions", "Conditions of Contract", CLAUSES_PART}

# Running-header text -> display name used in citations.
PART_NAMES = {
    "GENERAL GUIDELINES": "General Guidelines",
    "TENDER & CONTRACT": "Tender Form (CPWD-7/8)",
    "GENERAL RULES & DIRECTIONS": "General Rules & Directions",
    "CONDITIONS OF CONTRACT": "Conditions of Contract",
    "CLAUSES OF CONTRACT": CLAUSES_PART,
    "INTEGRITY PACT": "Integrity Pact",
    "INTEGRITY AGREEMENT": "Integrity Agreement",
    "C.P.W.D. SAFETY CODE": "CPWD Safety Code",
    "MODEL RULES": "Model Rules (Health & Sanitation)",
    "CONTRACTOR'S LABOUR REGULATIONS": "Contractor's Labour Regulations",
    "APPENDIX - XV": "Appendix XV (Secured Advance Indenture)",
    "APPENDIX - XVI": "Appendix XVI (Extension of Time Form)",
    "APPENDIX - XVII": "Appendix XVII (Arbitration Notice)",
    "BANK GUARANTEE BOND": "Bank Guarantee Bond Forms",
    "PROFORMA OF SCHEDULES": "Proforma of Schedules",
    "ANNEXURE": "Annexure (Road Roller Quantities)",
}

_CLAUSE_LINE = re.compile(r"clause\s+(\d+)\s*([a-z]{1,2})?", re.IGNORECASE)
_NUMBERED_ITEM = re.compile(r"^(\d{1,2}[A-Z]?)\.\s")


@dataclass
class _Item:
    """One element of the cleaned page stream: a heading or a body paragraph."""

    kind: str  # "heading" | "body"
    text: str
    part: str
    pdf_page: int
    printed_page: str
    clause_id: str | None = None  # Set when the paragraph opens a new clause.


@dataclass
class _Section:
    """A run of items that becomes one or more chunks."""

    part: str
    label: str
    title: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)


# --- Step 1: page blocks -> clean item stream ---------------------------------


def _clause_id(first_line: str) -> str | None:
    """'Clause 10 CC' -> '10CC'; returns None if the line is not a clause heading."""
    match = _CLAUSE_LINE.fullmatch(first_line.strip())
    if not match:
        return None
    number, letters = match.groups()
    return f"{number}{(letters or '').upper()}"


def _part_name(header: str) -> str:
    key = normalize(header).upper()
    return PART_NAMES.get(key, key.title())


def _printed_page(blocks: list[TextBlock], pdf_page: int) -> str:
    for block in blocks:
        if FOOTER_MARKER in block.text and block.lines[0].isdigit():
            return block.lines[0]
    return str(pdf_page - 2)  # Holds for every page in this edition.


def _is_short_caps_title(text: str) -> bool:
    """True for in-page titles like 'GOVERNMENT OF INDIA' or 'SCHEDULE 'F''.

    Requires a real word (4+ letters) and no '=' so formulas such as
    'N = 0.85 M' are not mistaken for titles.
    """
    has_word = any(sum(c.isalpha() for c in w) >= 4 for w in text.split())
    return has_word and "=" not in text and text.upper() == text and len(text.split()) <= 8


def _page_items(pdf_page: int, blocks: list[TextBlock], current_part: str) -> tuple[list[_Item], str]:
    """Clean one page into items. Returns the items and the part this page belongs to."""
    part = current_part
    for block in blocks:
        if block.y1 < HEADER_MAX_Y and not is_ruler_line(block.text.replace("\n", " ")):
            part = _part_name(block.text)
            break
    printed = _printed_page(blocks, pdf_page)

    items: list[_Item] = []
    for block in blocks:
        if block.y1 < HEADER_MAX_Y or FOOTER_MARKER in block.text:
            continue
        lines = [ln for ln in block.lines if not is_ruler_line(ln) and not looks_like_legacy_hindi(ln)]
        if not lines:
            continue

        is_margin = part in TWO_COLUMN_PARTS and block.x0 < MARGIN_MAX_X0 and block.x1 < MARGIN_MAX_X1
        clause_id = _clause_id(lines[0])
        if is_margin:
            items.append(_Item("heading", join_lines(lines), part, pdf_page, printed))
        elif clause_id:
            body = join_lines(lines[1:])
            items.append(_Item("body", body, part, pdf_page, printed, clause_id=clause_id))
        elif _is_short_caps_title(join_lines(lines)):
            items.append(_Item("heading", join_lines(lines), part, pdf_page, printed))
        else:
            items.append(_Item("body", join_lines(lines), part, pdf_page, printed))
    return items, part


def _item_stream(pdf_path: Path) -> list[_Item]:
    pages = read_blocks(pdf_path, INCLUDED_PAGES)
    stream: list[_Item] = []
    part = ""
    for pdf_page in INCLUDED_PAGES:
        items, part = _page_items(pdf_page, pages[pdf_page], part)
        stream.extend(items)
    return stream


# --- Step 2: item stream -> sections ------------------------------------------


def _is_part_title(heading: str, part: str) -> bool:
    """A heading that merely repeats the part name, e.g. 'C.P.W.D. SAFETY CODE'."""
    simplify = lambda s: re.sub(r"[^a-z]", "", s.lower())  # noqa: E731
    return simplify(heading) in simplify(part) or simplify(_part_name(heading)) == simplify(part)


def _clause_sections(items: list[_Item]) -> list[_Section]:
    """One section per 'Clause N'. The first margin heading inside a clause is its title."""
    sections: list[_Section] = []
    current: _Section | None = None
    for item in items:
        if item.clause_id:
            current = _Section(CLAUSES_PART, f"Clause {item.clause_id}")
            sections.append(current)
        if current is None:  # Part title before Clause 1 - nothing to keep.
            continue
        if item.kind == "heading":
            if not current.title:
                current.title = item.text
            else:  # Secondary heading (e.g. "Mobilization advance" inside 10B).
                current.paragraphs.append(Paragraph(f"[{item.text}]", item.pdf_page, item.printed_page))
        elif item.text:
            current.paragraphs.append(Paragraph(item.text, item.pdf_page, item.printed_page))
    return sections


def _generic_sections(items: list[_Item]) -> list[_Section]:
    """One section per part; headings are kept inline as [bracketed] context."""
    sections: list[_Section] = []
    for item in items:
        if not sections or sections[-1].part != item.part:
            sections.append(_Section(item.part, label=""))
        if item.kind == "heading" and _is_part_title(item.text, item.part):
            continue
        text = f"[{item.text}]" if item.kind == "heading" else item.text
        if item.clause_id:  # e.g. "Clause 5" rows inside the Schedule F proforma.
            text = f"Clause {item.clause_id}: {text}"
        if text:
            sections[-1].paragraphs.append(Paragraph(text, item.pdf_page, item.printed_page))
    return sections


# --- Step 3: sections -> chunks -----------------------------------------------


def _leading_int(label: str) -> int:
    digits = re.match(r"\d+", label)
    return int(digits.group()) if digits else -1


def _window_items(window: list[Paragraph], carried: str | None) -> list[str]:
    """Top-level item numbers in a window, e.g. ['6', '7'].

    Only increasing numbers count, so a nested list ('1.' inside item 11) does
    not produce nonsense like 'items 12-1'. If the window starts mid-item, the
    item carried over from the previous window is included first.
    """
    numbers: list[str] = []
    starts_with_item = False
    for i, p in enumerate(window):
        # Numbered headings ("[4. DRINKING WATER]") and "Article N" are always top-level.
        is_authoritative = p.text.startswith(("[", "Article"))
        text = p.text.lstrip("[")
        match = (
            _NUMBERED_ITEM.match(text)
            or re.match(r"^Clause (\w+):", text)
            or re.match(r"^Article (\d+)\b", text)
        )
        if not match:
            continue
        starts_with_item = starts_with_item or i == 0
        number = match.group(1)
        last = numbers[-1] if numbers else carried
        if is_authoritative and last is not None and _leading_int(number) < _leading_int(last):
            numbers = []  # Earlier numbers were a nested list; the real sequence restarts here.
            carried = None
        if is_authoritative or last is None or _leading_int(number) >= _leading_int(last):
            numbers.append(number)
    if carried and not starts_with_item:
        numbers.insert(0, f"{carried} (cont.)")
    return numbers


def _item_label(part: str, numbers: list[str]) -> str:
    """Citation label for a non-clause window, e.g. 'CPWD Safety Code, items 6-7'."""
    if not numbers:
        return part
    first, last = numbers[0].replace(" (cont.)", ""), numbers[-1]
    if len(numbers) == 1 or first == last:
        return f"{part}, item {numbers[0] if len(numbers) == 1 else first}"
    return f"{part}, items {first}-{last}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _to_chunks(sections: list[_Section], clause_mode: bool) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        carried_item: str | None = None
        for i, window in enumerate(pack(section.paragraphs), start=1):
            if clause_mode:
                label = section.label
                chunk_id = f"{DOC_ID}:{_slug(label)}:{i}"
            else:
                numbers = _window_items(window, carried_item)
                if numbers:
                    carried_item = numbers[-1].replace(" (cont.)", "")
                label = _item_label(section.part, numbers)
                chunk_id = f"{DOC_ID}:{_slug(section.part)}:p{window[0].pdf_page}-{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=DOC_ID,
                    doc_title=DOC_TITLE,
                    part=section.part,
                    section=label,
                    title=section.title,
                    text="\n".join(p.text for p in window),
                    pdf_pages=sorted({p.pdf_page for p in window}),
                    printed_pages=list(dict.fromkeys(p.printed_page for p in window)),
                )
            )
    return chunks


def parse(pdf_dir: Path) -> list[Chunk]:
    """Parse the CPWD GCC PDF into citation-ready chunks."""
    items = _item_stream(pdf_dir / PDF_NAME)
    clause_items = [it for it in items if it.part == CLAUSES_PART]
    other_items = [it for it in items if it.part != CLAUSES_PART]
    return _to_chunks(_clause_sections(clause_items), clause_mode=True) + _to_chunks(
        _generic_sections(other_items), clause_mode=False
    )
