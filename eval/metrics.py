"""Pure scoring functions for the evaluation (no models, no I/O), so they are easy to test.

Conventions (docs/PLAN.md section 4):
  - A question is "found" at rank r if r is the rank of the first retrieved chunk that is
    one of its evidence chunks. Hit@k and MRR use that rank.
  - "Fully supported" means every evidence chunk reaches the LLM (e.g. Table 28 *and* its amendment).
  - With ~30 questions, results are shown as counts ("22/26"), and two systems are compared
    per question (wins/losses) rather than by a percentage gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.cite import NOT_FOUND, numbers_in

_CITATION_MARKER = re.compile(r"\[\s*S\s*\d+[^\]]*\]", re.IGNORECASE)  # [S1], [S1, S3], [S2][S4]
_MINUS_SIGNS = "-−–"  # hyphen-minus, minus sign, en dash (models use all three)
_NUMERIC_VALUE = re.compile(rf"[{_MINUS_SIGNS}]?\s*\d[\d.,]*")
_LEADING_NOISE = re.compile(r"^[\s\"'*>_`]+")  # Quotes, bold or blockquote marks before the phrase.


@dataclass(frozen=True)
class Rate:
    """A count out of a total, printed as '22/26 (85%)'."""

    hits: int
    total: int

    @classmethod
    def of(cls, flags: Iterable[bool]) -> "Rate":
        flags = list(flags)
        return cls(sum(flags), len(flags))

    def __str__(self) -> str:
        if not self.total:
            return "-"
        return f"{self.hits}/{self.total} ({100 * self.hits / self.total:.0f}%)"


def first_gold_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> int | None:
    """1-based rank of the first evidence chunk in a ranked list, or None if none was retrieved."""
    gold = set(gold_ids)
    return next((rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in gold), None)


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None, cutoff: int) -> float:
    """1/rank if the first evidence chunk is within the cutoff, else 0 (i.e. MRR@cutoff)."""
    return 1.0 / rank if hit_at(rank, cutoff) else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def found_in(context_ids: Iterable[str], gold_ids: Iterable[str]) -> bool:
    """At least one evidence chunk is among the chunks sent to the LLM."""
    return not set(gold_ids).isdisjoint(context_ids)


def fully_supported(context_ids: Iterable[str], gold_ids: Iterable[str]) -> bool:
    """Every evidence chunk is among the chunks sent to the LLM."""
    return set(gold_ids) <= set(context_ids)


def is_abstention(answer: str) -> bool:
    """The answer declines with the exact not-found phrase (the same convention as `src.cite`).

    Only the opening counts: a partial answer that says "not found" about one detail is not an abstention.
    """
    opening = _LEADING_NOISE.sub("", answer).lower()
    return opening.startswith(NOT_FOUND.rstrip(".").lower())


def missing_values(answer: str, required: Iterable[str]) -> list[str]:
    """The required key values that the answer does not state.

    A numeric value is compared as a number (so '12' matches 'twelve' and '1.80' matches '1.8');
    a negative one must also carry its minus sign (-0.8 is not 0.8 for a pressure coefficient).
    Anything else must appear as text, ignoring case. Citation markers are removed first,
    so the '1' in '[S1]' never counts as a stated value.
    """
    text = _CITATION_MARKER.sub(" ", answer)
    stated = numbers_in(text)
    missing = []
    for value in required:
        value = value.strip()
        if _NUMERIC_VALUE.fullmatch(value):
            present = numbers_in(value) <= stated
            if present and value[0] in _MINUS_SIGNS:
                magnitude = re.escape(value[1:].strip())
                # Trailing zeros are allowed ("-0.80"), further digits are not ("-0.85").
                present = re.search(rf"[{_MINUS_SIGNS}]\s*{magnitude}0*(?!\d)", text) is not None
        else:
            present = value.lower() in text.lower()
        if not present:
            missing.append(value)
    return missing


@dataclass(frozen=True)
class WinsLosses:
    """Per-question comparison of system B against system A."""

    wins: tuple[str, ...]    # Question ids where B is better.
    losses: tuple[str, ...]  # Question ids where B is worse.
    ties: int

    def __str__(self) -> str:
        wins = f" ({', '.join(self.wins)})" if self.wins else ""
        losses = f" ({', '.join(self.losses)})" if self.losses else ""
        return f"fixed {len(self.wins)}{wins}, broke {len(self.losses)}{losses}, same {self.ties}"


def wins_losses(
    ranks_a: dict[str, int | None], ranks_b: dict[str, int | None], k: int, mrr_cutoff: int
) -> WinsLosses:
    """Compare two systems per question: Hit@k decides, the reciprocal rank breaks ties.

    This is the pre-committed rule for the fine-tune (docs/PLAN.md section 3): B ships only if it
    has more wins than losses on hybrid Hit@5, ties broken by MRR.
    """
    wins, losses, ties = [], [], 0
    for qid in ranks_a:
        a = (hit_at(ranks_a[qid], k), reciprocal_rank(ranks_a[qid], mrr_cutoff))
        b = (hit_at(ranks_b[qid], k), reciprocal_rank(ranks_b[qid], mrr_cutoff))
        if b > a:
            wins.append(qid)
        elif b < a:
            losses.append(qid)
        else:
            ties += 1
    return WinsLosses(tuple(wins), tuple(losses), ties)


@dataclass(frozen=True)
class ThresholdChoice:
    """How well a low-confidence cutoff separates answerable from unanswerable questions.

    A question is flagged when its best dense cosine is below the threshold.
    """

    threshold: float
    unanswerable_flagged: Rate  # Want high: these should be flagged.
    answerable_flagged: Rate    # Want low: these are false alarms.

    @property
    def correct(self) -> int:
        return self.unanswerable_flagged.hits + (self.answerable_flagged.total - self.answerable_flagged.hits)


def score_threshold(threshold: float, answerable: Sequence[float], unanswerable: Sequence[float]) -> ThresholdChoice:
    return ThresholdChoice(
        threshold,
        Rate.of(score < threshold for score in unanswerable),
        Rate.of(score < threshold for score in answerable),
    )


def best_threshold(answerable: Sequence[float], unanswerable: Sequence[float]) -> ThresholdChoice:
    """The cutoff that classifies the most questions correctly.

    Candidates are the midpoints between neighbouring observed scores (rounded to 3 decimals),
    so the choice doesn't sit exactly on one question's score. Ties go to fewer false alarms
    on answerable questions, because a warning on a good answer erodes trust in the warning.
    """
    scores = sorted(set(answerable) | set(unanswerable))
    if not scores:
        raise ValueError("No scores to calibrate on")
    candidates = [round((lo + hi) / 2, 3) for lo, hi in zip(scores, scores[1:])] or [scores[0]]
    choices = [score_threshold(t, answerable, unanswerable) for t in candidates]
    return max(choices, key=lambda c: (c.correct, -c.answerable_flagged.hits))
