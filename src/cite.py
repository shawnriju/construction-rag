"""Citation validation for generated answers.

The model is told to put [S#] markers on every factual sentence. Small models
do not always comply, so we check mechanically instead of trusting it:
  * invalid    - markers pointing to a source number that was never provided
  * uncited    - substantive sentences without any marker
  * ungrounded - numbers in a cited sentence that appear in none of the
                 sources it cites (e.g. "pz = 0.6 ..." cited to a foreword
                 that contains no 0.6). Wrong numbers are the most dangerous
                 citation error in engineering and contract answers.

The number check is a deterministic heuristic, not proof of support: it can
miss a wrong claim with the right numbers, and it flags a number the model
spelled differently ("12 months" vs "twelve months"). Its warnings mean
"check this", not "this is wrong".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

NOT_FOUND = "Not found in the provided documents."

_BRACKET = re.compile(r"\[([^\]]+)\]")
_SOURCE_NUMBER = re.compile(r"S\s*(\d+)", re.IGNORECASE)
_MARKER = re.compile(r"\[\s*S\s*\d+(?:\s*,\s*S?\s*\d+)*\s*\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_WORDS_FOR_CITATION = 6  # Short connective lines ("In summary:") need no citation.
# Abbreviations common in these documents whose full stop does not end a sentence:
# "Cl. 5.3", "p. 15", "pp. 28-29", "No. 2", "s. 34", "Fig. 13", "i.e.", "e.g.".
_ABBREVIATION_END = re.compile(
    r"(?:^|[\s(\[])(?:cl|cls|no|nos|p|pp|para|paras|sec|s|fig|figs|art|amd|vol|ch|vs|viz|rs|i\.e|e\.g)\.$",
    re.IGNORECASE,
)
# Markers at the start of a fragment, e.g. the "[S2]" in "...is 1.8. [S2] Next sentence" (cited after the full stop).
_LEADING_MARKERS = re.compile(r"^((?:\[[^\]]*\][\s.,;:]*)+)(.*)$", re.DOTALL)

# Numbers: OCR writes decimals as 0'6 or 0·6; thousands may be 1,000.
_OCR_DECIMAL = re.compile(r"(?<=\d)['’·](?=\d)")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Number words, so "twelve months" in a source supports "12 months" in an answer (and vice versa).
_NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
     "ninety": "90", "hundred": "100"}
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)           # Valid source numbers used (1-based).
    invalid: list[int] = field(default_factory=list)         # Source numbers that don't exist.
    uncited_sentences: list[str] = field(default_factory=list)
    # (number, sentence) pairs: the number appears in none of the sources the sentence cites.
    ungrounded_numbers: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.invalid and not self.uncited_sentences and not self.ungrounded_numbers


def source_numbers(text: str) -> list[int]:
    """All source numbers referenced in `text`, e.g. '[S1][S3, S4]' -> [1, 3, 4]."""
    numbers = []
    for group in _BRACKET.findall(text):
        numbers.extend(int(n) for n in _SOURCE_NUMBER.findall(group))
    return numbers


def numbers_in(text: str) -> set[str]:
    """Numbers in `text`, normalised so equal values compare equal: '1.00' == '1', "0'6" == '0.6',
    '1,000' == '1000', 'twelve' == '12'.

    Signs are ignored (a source saying "Substitute '-0.8' for '0.8'" contains both anyway).
    """
    text = _THOUSANDS_COMMA.sub("", _OCR_DECIMAL.sub(".", text))
    text = _NUMBER_WORD.sub(lambda m: f" {_NUMBER_WORDS[m.group(1).lower()]} ", text)
    found = set()
    for number in _NUMBER.findall(text):
        whole, _, fraction = number.partition(".")
        fraction = fraction.rstrip("0")
        found.add(f"{int(whole)}.{fraction}" if fraction else str(int(whole)))
    return found


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, without breaking at abbreviations.

    Markers that open a fragment are moved to the sentence before it when that
    sentence has none, so "...is 1.8. [S2]" counts as one cited sentence.
    """
    sentences: list[str] = []
    for fragment in _SENTENCE_SPLIT.split(text):
        fragment = fragment.strip()
        leading = _LEADING_MARKERS.match(fragment)
        if sentences and leading and not source_numbers(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {leading.group(1).strip()}"
            fragment = leading.group(2).strip()
        if not fragment:
            continue
        if sentences and _ABBREVIATION_END.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {fragment}"
        else:
            sentences.append(fragment)
    return sentences


def check_citations(
    answer: str,
    num_sources: int,
    source_texts: Sequence[str] | None = None,
    question: str = "",
) -> CitationReport:
    """Check the answer's [S#] markers.

    Args:
        source_texts: The text of each numbered source as the model saw it
            (source_texts[0] is [S1]). If given, numbers are checked too.
        question: Numbers the user typed may be repeated without a source.
    """
    if answer.strip().startswith(NOT_FOUND):
        return CitationReport()

    used = source_numbers(answer)
    report = CitationReport(
        cited=sorted({n for n in used if 1 <= n <= num_sources}),
        invalid=sorted({n for n in used if not 1 <= n <= num_sources}),
    )
    source_numbers_by_index = (
        [numbers_in(_MARKER.sub(" ", text)) for text in source_texts] if source_texts is not None else None
    )
    allowed = numbers_in(question)

    for sentence in split_sentences(answer):
        stripped = sentence.strip(" -*•\t")
        cited_here = [n for n in source_numbers(stripped) if 1 <= n <= num_sources]
        if not source_numbers(stripped):
            if len(stripped.split()) >= _MIN_WORDS_FOR_CITATION:
                report.uncited_sentences.append(stripped)
            continue
        if source_numbers_by_index is None or not cited_here:
            continue
        supported = set().union(*(source_numbers_by_index[n - 1] for n in cited_here))
        for number in sorted(numbers_in(_MARKER.sub(" ", stripped)) - supported - allowed):
            report.ungrounded_numbers.append((number, stripped))
    return report
