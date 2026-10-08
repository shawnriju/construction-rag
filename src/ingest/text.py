"""Text-cleaning and chunk-packing helpers shared by all document parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from src.config import EMBED_TOKENIZER, MAX_CHUNK_TOKENS, MIN_TAIL_TOKENS

# --- Cleaning ----------------------------------------------------------------

_UNICODE_FIXES = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "ﬁ": "fi", "ﬂ": "fl",
}
_DOT_LEADER = re.compile(r"\.{4,}|…{2,}")
_MULTI_SPACE = re.compile(r"[ \t]+")
_RULER_LINE = re.compile(r"^[\d\s]{20,}$")  # CPWD's "1234567890..." decoration rows.


def normalize(text: str) -> str:
    """Fix typographic characters, drop dot leaders and collapse whitespace."""
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    text = _DOT_LEADER.sub(" ... ", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def is_ruler_line(line: str) -> bool:
    return bool(_RULER_LINE.match(line))


def join_lines(lines: list[str]) -> str:
    """Join wrapped PDF lines into one string, repairing words split by hyphens.

    "Engineer-" + "in-Charge" keeps the hyphen (compound word);
    "construc-" + "tion" drops it (a word broken at the line end).
    """
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            next_word = line.split(" ", 1)[0]
            out = out + line if "-" in next_word else out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return normalize(out)


# Legacy Hindi fonts (e.g. Kruti Dev) extract as Latin gibberish such as
# "ds- yks- fu- fo-" or "mRd`'Vrk". These tokens have odd punctuation inside the
# word or an upper-case letter in the middle of a lower-case word.
_ODD_TOKEN = re.compile(r"[a-z][A-Z]|[;`~@¼½©®ç]|^[a-z]{1,3}-$")


def looks_like_legacy_hindi(line: str) -> bool:
    tokens = [t for t in line.split() if any(c.isalpha() for c in t)]
    if len(tokens) < 2:
        return False
    odd = sum(1 for t in tokens if _ODD_TOKEN.search(t))
    return odd / len(tokens) >= 0.3


def word_count(text: str) -> int:
    return len(text.split())


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    """The embedder's tokenizer (tokenizer.json only, no model weights).

    The local cache is tried first: a cached copy needs no network call, and
    on a restricted network the online check can hang instead of failing.
    """
    try:
        path = hf_hub_download(EMBED_TOKENIZER, "tokenizer.json", local_files_only=True)
    except Exception:  # Not cached yet: fresh machine.
        path = hf_hub_download(EMBED_TOKENIZER, "tokenizer.json")
    return Tokenizer.from_file(path)


@lru_cache(maxsize=None)
def token_count(text: str) -> int:
    """Embedder tokens in `text`, without [CLS]/[SEP].

    Counts add up across whitespace joins (the tokenizer splits on whitespace
    first), so the packers can sum the counts of individual pieces.
    """
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


# --- Packing paragraphs into chunks ------------------------------------------


@dataclass
class Paragraph:
    """A cleaned paragraph together with the page it came from."""

    text: str
    pdf_page: int
    printed_page: str


_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"'])")


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split at word boundaries into pieces of at most `max_tokens` (last resort)."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in text.split():
        tokens = token_count(word)
        if current and current_tokens + tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_long(paragraph: Paragraph, max_tokens: int) -> list[Paragraph]:
    """Split an over-long paragraph at sentence boundaries (hard-split as a last resort)."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph.text):
        candidate = f"{current} {sentence}".strip()
        if current and token_count(candidate) > max_tokens:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    final: list[str] = []
    for piece in pieces:  # A single giant "sentence" (e.g. a formula dump) still gets split.
        final.extend(_hard_split(piece, max_tokens))
    return [Paragraph(p, paragraph.pdf_page, paragraph.printed_page) for p in final]


def pack(paragraphs: list[Paragraph], max_tokens: int = MAX_CHUNK_TOKENS) -> list[list[Paragraph]]:
    """Greedily group consecutive paragraphs into windows of at most `max_tokens`.

    Paragraph boundaries are respected; only a single paragraph that is longer
    than `max_tokens` on its own is split (at sentence boundaries).
    """
    windows: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    current_tokens = 0
    for paragraph in paragraphs:
        too_long = token_count(paragraph.text) > max_tokens
        for piece in _split_long(paragraph, max_tokens) if too_long else [paragraph]:
            tokens = token_count(piece.text)
            if current and current_tokens + tokens > max_tokens:
                windows.append(current)
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += tokens
    if current:
        windows.append(current)

    # A tiny trailing window ("...this judgment.") is useless on its own; fold it
    # into the previous one. The overflow is at most MIN_TAIL_TOKENS, which the
    # breadcrumb reserve and the 512-token check in build.py leave room for.
    if len(windows) > 1 and sum(token_count(p.text) for p in windows[-1]) < MIN_TAIL_TOKENS:
        windows[-2].extend(windows.pop())
    return windows


# --- Grouping labelled units (paragraphs, clauses) into chunks ------------------


@dataclass
class Unit:
    """A citable unit of a document, e.g. judgment paragraph 23 or clause 5.3.2.

    Attributes:
        key: Identifier used in the citation ("23", "5.3.2", "Table 28").
        group: Units are only merged with neighbours of the same group
            (e.g. the same section heading or top-level clause).
        title: Heading shown with the chunk.
        paragraphs: The unit's text, with page provenance.
        mergeable: False keeps the unit on its own (e.g. a table).
    """

    key: str
    group: str
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    mergeable: bool = True

    @property
    def tokens(self) -> int:
        return sum(token_count(p.text) for p in self.paragraphs)


@dataclass
class Window:
    """One future chunk: either several whole units, or one part of a long unit."""

    keys: list[str]
    title: str
    paragraphs: list[Paragraph]
    part_no: int = 1
    part_count: int = 1

    def label(self, prefix: str) -> str:
        """Citation label, e.g. '¶ 23-24', 'Cl. 5.3.2', '¶ 33 (part 2/4)'."""
        first, last = self.keys[0], self.keys[-1]
        name = f"{prefix}{first}" if first == last else f"{prefix}{first}-{last}"
        return f"{name} (part {self.part_no}/{self.part_count})" if self.part_count > 1 else name


def group_units(units: list[Unit], max_tokens: int = MAX_CHUNK_TOKENS) -> list[Window]:
    """Merge consecutive small units (same group) into windows; split oversized units.

    Units are never cut in the middle unless a single unit alone exceeds
    `max_tokens`, in which case it becomes several "part i/n" windows.
    """
    windows: list[Window] = []
    batch: list[Unit] = []

    def flush() -> None:
        if batch:
            paragraphs = [p for unit in batch for p in unit.paragraphs]
            windows.append(Window([u.key for u in batch], batch[0].title, paragraphs))
            batch.clear()

    for unit in units:
        if not unit.paragraphs:
            continue
        if unit.tokens > max_tokens:
            # A heading-only stub just before a long unit ("7. DYNAMIC EFFECTS")
            # is folded into the long unit's first part instead of standing alone.
            stub: list[Unit] = []
            if batch and batch[0].group == unit.group and sum(u.tokens for u in batch) < MIN_TAIL_TOKENS:
                stub = batch[:]
                batch.clear()
            flush()
            parts = pack(unit.paragraphs, max_tokens)
            for i, part in enumerate(parts, start=1):
                keys = [u.key for u in stub] + [unit.key] if i == 1 else [unit.key]
                lead = [p for u in stub for p in u.paragraphs] if i == 1 else []
                title = stub[0].title if (i == 1 and stub and stub[0].title) else unit.title
                windows.append(Window(keys, title, lead + part, part_no=i, part_count=len(parts)))
            continue
        can_join = (
            batch
            and unit.mergeable
            and batch[-1].mergeable
            and batch[0].group == unit.group
            and sum(u.tokens for u in batch) + unit.tokens <= max_tokens
        )
        if not can_join:
            flush()
        batch.append(unit)
    flush()
    return windows
