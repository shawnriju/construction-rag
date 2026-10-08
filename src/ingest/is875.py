"""Parser for IS 875 (Part 3):1987 - Wind Loads (a scanned PDF with an OCR layer).

Layout facts this parser relies on (verified against the PDF):
  * Two-column pages. Blocks are read left column top-to-bottom, then right.
  * Running header "IS : 875 ( Part 3 ) - 1987" at the top, page number at
    the bottom. Printed page = PDF page - 4 for the pages we read.
  * Clauses start with their number ("5.3.2.1 Terrain - ..."). The OCR
    sometimes merges lines across columns, so a clause number can also appear
    in the middle of a block. Because OCR also produces stray numbers, a
    candidate is only accepted if it is a plausible *next* clause after the
    previous one (child, next sibling, or next ancestor, allowing one skipped
    number). This sequence check is what makes mid-block detection safe.
  * Tables ("TABLE 5 ...") and appendices ("APPENDIX B") are their own units.
    Table OCR text is poor; the most important tables are replaced by
    hand-curated versions of Tables 1, 2, 28 and Appendix A (see data/curated/is875.yaml and src/ingest/curated.py).

Known limitation: on a few pages (e.g. PDF 41-42) the OCR interleaves the two
columns line by line, so some sentences there are out of order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.ingest.text import Paragraph, Unit, Window, group_units, normalize
from src.schema import Chunk

DOC_ID = "is875"
DOC_TITLE = "IS 875 (Part 3):1987"
PDF_NAME = "IS875-3-1987_wind-loads_EN.pdf"

# 7-8 foreword, 9-62 body and appendices. Skipped: cover/contents (1-6),
# the wind map image (13), a blank page (14), the BIS back page (63) and the
# amendment pages (64-67), which are curated by hand instead.
INCLUDED_PAGES = [7, 8, 9, 10, 11, 12, *range(15, 63)]
PRINTED_PAGE_OFFSET = 4

HEADER_MAX_Y = 50
FOOTER_MARGIN = 70
COLUMN_TOLERANCE = 15

# Sections whose OCR text is replaced by curated content.
CURATED_SECTIONS = {"Table 1", "Table 2", "Table 28", "Appendix A"}

# Clause number (optionally appendix-prefixed, e.g. "C-2.1") followed by a capital letter.
_CLAUSE = re.compile(r"(?:^|(?<=\s))((?:[A-D]-)?[\dlI]{1,2}(?:\.\d{1,2}){0,4})(\.?)\s+(?=[A-Z(])")
# Top-level sections are always printed as "1. SCOPE", "5. WIND SPEED AND PRESSURE".
_CAPS_WORD = re.compile(r"[A-Z]{3,}\b")
_TABLE = re.compile(r"^TABLE\s+(\d{1,2})\b\s*(.*)")
_APPENDIX = re.compile(r"^APPENDIX\s+([A-D])\b\s*(.*)")
_TITLE_DASH = re.compile(r"\s[-–]\s")

# A clause id as a sortable tuple; appendix "C-2.1" -> (103, 2, 1).
ClauseKey = tuple[int, ...]


@dataclass
class _Segment:
    """A piece of text that starts a new unit (clause/table/appendix) or continues one."""

    text: str
    pdf_page: int
    starts: str | None = None  # Unit key, e.g. "Cl. 5.3.2", "Table 5", "Appendix B".


# --- Clause-number plausibility -------------------------------------------------


def _parse_key(raw: str) -> ClauseKey | None:
    """'5.3.2' -> (5, 3, 2); 'C-2.1' -> (103, 2, 1); OCR 'l'/'I' are read as '1'."""
    raw = raw.replace("l", "1").replace("I", "1")
    match = re.fullmatch(r"(?:([A-D])-)?(\d+(?:\.\d+)*)", raw)
    if not match:
        return None
    letter, numbers = match.groups()
    key = tuple(int(n) for n in numbers.split("."))
    return ((ord(letter) - ord("A") + 101,) + key) if letter else key


def _key_label(key: ClauseKey) -> str:
    if key[0] > 100:
        letter = chr(key[0] - 101 + ord("A"))
        return f"Cl. {letter}-" + ".".join(str(n) for n in key[1:])
    return "Cl. " + ".".join(str(n) for n in key)


def _is_plausible_next(prev: ClauseKey, new: ClauseKey, max_skip: int = 2) -> bool:
    """True if `new` can follow `prev` in a numbered hierarchy.

    Allowed: first child (5.3 -> 5.3.1), next sibling at any level
    (5.3.2.2 -> 5.3.2.3 / 5.3.3 / 5.4 / 6), with up to `max_skip - 1` numbers
    skipped in case the OCR missed a heading.
    """
    if new == prev + (1,):
        return True
    for depth in range(len(prev)):
        if len(new) == depth + 1 and new[:depth] == prev[:depth]:
            if 0 < new[depth] - prev[depth] <= max_skip:
                return True
    return False


# --- Step 1: pages -> segments --------------------------------------------------


def _ordered_blocks(page: fitz.Page) -> list[tuple[float, str]]:
    """Body blocks in reading order, without header/footer.

    Pages are read column by column (left, then right). But when a full-width
    table starts mid-page under two-column text, reading the whole page
    left-then-right would pull the top-right text *below* the table. So the
    page is first cut into horizontal bands at every full-width TABLE/APPENDIX
    heading, and each band is read left-then-right.
    """
    mid = page.rect.width / 2
    height = page.rect.height
    blocks = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        clean = normalize(" ".join(text.split()))
        if not clean:
            continue
        if y1 < HEADER_MAX_Y and "875" in clean:
            continue
        if y0 > height - FOOTER_MARGIN and clean.isdigit():
            continue
        blocks.append((x0, y0, x1, clean))

    band_starts = sorted(
        y0 for x0, y0, x1, text in blocks
        if (_TABLE.match(text) or _APPENDIX.match(text))
        and x0 < mid - COLUMN_TOLERANCE and x1 > mid + COLUMN_TOLERANCE  # Spans both columns.
    )
    edges = [float("-inf"), *band_starts, float("inf")]

    ordered: list[tuple[float, str]] = []
    for top, bottom in zip(edges, edges[1:]):
        band = [b for b in blocks if top <= b[1] < bottom]
        left = sorted((b for b in band if b[0] < mid - COLUMN_TOLERANCE), key=lambda b: b[1])
        right = sorted((b for b in band if b[0] >= mid - COLUMN_TOLERANCE), key=lambda b: b[1])
        ordered.extend((b[1], b[3]) for b in left + right)
    return ordered


class _SegmentBuilder:
    """Walks blocks in order and cuts them into segments at accepted unit starts."""

    def __init__(self) -> None:
        self.prev_clause: ClauseKey = (0,)  # Foreword clauses are 0.1, 0.2, ...
        self.prev_table = 0
        self.segments: list[_Segment] = []

    def add_block(self, text: str, pdf_page: int) -> None:
        table, appendix = _TABLE.match(text), _APPENDIX.match(text)
        if table and 0 < int(table.group(1)) - self.prev_table <= 3:
            self.prev_table = int(table.group(1))
            self.segments.append(_Segment(text, pdf_page, starts=f"Table {table.group(1)}"))
            return
        if appendix:
            letter = appendix.group(1)
            self.prev_clause = (ord(letter) - ord("A") + 101,)  # Next expected: "<letter>-1".
            self.segments.append(_Segment(text, pdf_page, starts=f"Appendix {letter}"))
            return
        self._split_on_clauses(text, pdf_page)

    def _split_on_clauses(self, text: str, pdf_page: int) -> None:
        cursor = 0
        for match in _CLAUSE.finditer(text):
            key = _parse_key(match.group(1))
            if key is None or not _is_plausible_next(self.prev_clause, key):
                continue
            is_top_level = len(key) == 1
            if is_top_level and not (match.group(2) and _CAPS_WORD.match(text, match.end())):
                continue  # e.g. "Part 2 Imposed loads" in the foreword is not clause 2.
            if match.start() > cursor:
                self.segments.append(_Segment(text[cursor:match.start()].strip(), pdf_page))
            self.prev_clause = key
            self.segments.append(_Segment("", pdf_page, starts=_key_label(key)))
            cursor = match.end()
        rest = text[cursor:].strip()
        if rest:
            self.segments.append(_Segment(rest, pdf_page))


def _segments(pdf_path: Path) -> list[_Segment]:
    builder = _SegmentBuilder()
    with fitz.open(pdf_path) as doc:
        for pdf_page in INCLUDED_PAGES:
            for _, text in _ordered_blocks(doc[pdf_page - 1]):
                builder.add_block(text, pdf_page)
    return builder.segments


# --- Step 2: segments -> units --------------------------------------------------


def _title(key: str, first_text: str) -> str:
    """Heading of a unit, e.g. 'Terrain' from '5.3.2.1 Terrain - Selection of ...'."""
    if key.startswith(("Table", "Appendix")):
        head = _TABLE.match(first_text) or _APPENDIX.match(first_text)
        return head.group(2)[:120] if head else ""
    dash = _TITLE_DASH.search(first_text)
    if dash and len(first_text[: dash.start()].split()) <= 10:
        return first_text[: dash.start()].strip()
    words = first_text.split()
    caps = [w for w in words[:12] if w.upper() == w and any(c.isalpha() for c in w)]
    if len(caps) >= 2 and words[: len(caps)] == caps:  # "WIND PRESSURES AND FORCES ..."
        return " ".join(caps)
    return first_text if len(words) <= 8 else ""


def _group_key(key: str) -> str:
    """Units merge only within the same top-level clause or appendix."""
    if key.startswith("Table"):
        return key
    if key.startswith("Appendix"):
        return key[-1]
    body = key.removeprefix("Cl. ")
    return body.split("-")[0] if "-" in body else body.split(".")[0]


def _units(segments: list[_Segment]) -> list[Unit]:
    units: list[Unit] = []
    for segment in segments:
        if segment.starts:
            key = segment.starts
            units.append(Unit(key, _group_key(key), "", mergeable=not key.startswith("Table")))
            continue
        if not units:  # Foreword text before 0.1.
            units.append(Unit("Foreword", "0", "Foreword"))
        unit = units[-1]
        if not unit.paragraphs:
            unit.title = _title(unit.key, segment.text)
        printed = str(segment.pdf_page - PRINTED_PAGE_OFFSET)
        unit.paragraphs.append(Paragraph(segment.text, segment.pdf_page, printed))
    return [u for u in units if u.key not in CURATED_SECTIONS]


# --- Step 3: units -> chunks ----------------------------------------------------


def _window_label(window: Window) -> str:
    """'Cl. 5.3.2', 'Cl. 5.3.2-5.3.2.2', 'Appendix B, Cl. B-1-B-2', 'Table 5 (part 1/2)'."""
    first, last = window.keys[0], window.keys[-1]
    if first == last:
        label = first
    elif first.startswith("Cl.") and last.startswith("Cl."):
        label = f"{first}-{last.removeprefix('Cl. ')}"
    else:
        label = f"{first}, {last}" if len(window.keys) == 2 else f"{first} to {last}"
    return f"{label} (part {window.part_no}/{window.part_count})" if window.part_count > 1 else label


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _section_titles(units: list[Unit]) -> dict[str, str]:
    """Top-level clause number -> breadcrumb, e.g. {'5': '5. Wind Speed and Pressure'}."""
    titles = {"0": "Foreword"}
    for unit in units:
        number = unit.key.removeprefix("Cl. ")
        if unit.key.startswith("Cl.") and number.isdigit() and unit.title:
            titles[number] = f"{number}. {unit.title.title()}"
    return titles


def _part(window: Window, titles: dict[str, str]) -> str:
    key = window.keys[-1]
    if key.startswith(("Table", "Appendix")) or key == "Foreword":
        return "Tables" if key.startswith("Table") else ("Appendices" if key.startswith("Appendix") else "Foreword")
    group = _group_key(key)
    return titles.get(group, "Appendices" if not group.isdigit() else "")


def parse(pdf_dir: Path) -> list[Chunk]:
    """Parse the IS 875 OCR text into clause-based chunks (curated content excluded)."""
    units = _units(_segments(pdf_dir / PDF_NAME))
    titles = _section_titles(units)
    chunks = []
    for window in group_units(units):
        label = _window_label(window)
        chunks.append(
            Chunk(
                chunk_id=f"{DOC_ID}:{_slug(label)}",
                doc_id=DOC_ID,
                doc_title=DOC_TITLE,
                part=_part(window, titles),
                section=label,
                title=window.title,
                text="\n".join(p.text for p in window.paragraphs),
                pdf_pages=sorted({p.pdf_page for p in window.paragraphs}),
                printed_pages=list(dict.fromkeys(p.printed_page for p in window.paragraphs)),
                covers=list(window.keys),
            )
        )
    return chunks
