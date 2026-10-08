"""Citation validation for generated answers.

The model is told to end every factual sentence with [S#] markers. Small
models do not always comply, so we check mechanically instead of trusting it:
  * invalid  - markers pointing to a source number that was never provided
  * uncited  - substantive sentences without any marker
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NOT_FOUND = "Not found in the provided documents."

_BRACKET = re.compile(r"\[([^\]]+)\]")
_SOURCE_NUMBER = re.compile(r"S\s*(\d+)", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_WORDS_FOR_CITATION = 6  # Short connective lines ("In summary:") need no citation.


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)           # Valid source numbers used (1-based).
    invalid: list[int] = field(default_factory=list)         # Source numbers that don't exist.
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.invalid and not self.uncited_sentences


def source_numbers(text: str) -> list[int]:
    """All source numbers referenced in `text`, e.g. '[S1][S3, S4]' -> [1, 3, 4]."""
    numbers = []
    for group in _BRACKET.findall(text):
        numbers.extend(int(n) for n in _SOURCE_NUMBER.findall(group))
    return numbers


def check_citations(answer: str, num_sources: int) -> CitationReport:
    if answer.strip().startswith(NOT_FOUND):
        return CitationReport()

    used = source_numbers(answer)
    report = CitationReport(
        cited=sorted({n for n in used if 1 <= n <= num_sources}),
        invalid=sorted({n for n in used if not 1 <= n <= num_sources}),
    )
    for sentence in _SENTENCE_SPLIT.split(answer):
        stripped = sentence.strip(" -*•\t")
        if len(stripped.split()) >= _MIN_WORDS_FOR_CITATION and not source_numbers(stripped):
            report.uncited_sentences.append(stripped)
    return report
