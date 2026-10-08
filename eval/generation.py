"""Answer-quality evaluation on the frozen gold set (run it with `python -m eval.run generation`).

Every question goes through `RAGPipeline.ask()`, exactly as the CLI asks it, so the scores
describe the real system. Needs the LLM backend (Ollama) running; about 15 minutes for 33
questions on CPU/GTX 1650. Answers are deterministic (temperature 0, fixed seed).

Writes `eval/results/generation_<label>.md` (report + every answer for a manual read) and
`generation_<label>.json` (per-question results).

Scored automatically (PLAN.md section 4):
  - correct:   answerable -> no abstention and every `must_include` value stated;
               unanswerable -> the answer abstains with the exact not-found phrase.
               Questions without `must_include` are left to the manual review.
  - failure source: a wrong answer whose evidence did reach the LLM is a generation
               failure; one whose evidence never arrived is a retrieval failure.
  - cites evidence: the answer cites at least one evidence passage.
  - warnings: the pipeline's own citation checks (invalid markers, uncited sentences,
               numbers not in the cited sources).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval.gold import GoldQuestion
from eval.metrics import Rate, found_in, fully_supported, is_abstention, mean, missing_values
from eval.report import group_by, markdown_table, yes_no
from src.config import EVAL_RESULTS_DIR, GOLD_PATH
from src.pipeline import Answer, RAGPipeline

GENERATED = "generated"  # Answer.mode when the LLM answered (otherwise "retrieval-only").


@dataclass
class GenerationResult:
    """Everything measured for one answer (serialised to JSON as-is)."""

    id: str
    doc: str
    type: str
    wording: str
    answerable: bool
    question: str
    expected: str
    answer: str
    mode: str                     # "generated", or "retrieval-only" if the LLM failed on this question.
    abstained: bool
    missing: list[str]            # `must_include` values the answer does not state.
    correct: bool | None          # None = no automatic check possible (manual review).
    found: bool                   # At least one evidence passage reached the LLM.
    full: bool                    # Every evidence passage reached the LLM.
    cites_evidence: bool          # The answer cites at least one evidence passage.
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def failure_source(self) -> str:
        """'' if not a known failure; else whether retrieval or generation is to blame."""
        if self.correct is not False or not self.answerable:
            return ""
        return "generation" if self.found else "retrieval"


def score_answer(question: GoldQuestion, answer: Answer) -> GenerationResult:
    """Score one pipeline answer against its gold record (no LLM call here, so it is testable)."""
    gold = question.gold_chunk_ids
    context = [hit.chunk.chunk_id for hit in answer.sources]
    cited = {answer.sources[n - 1].chunk.chunk_id for n in answer.citations.cited}
    generated = answer.mode == GENERATED
    abstained = generated and is_abstention(answer.text)
    missing = missing_values(answer.text, question.must_include) if generated else list(question.must_include)

    if not generated:
        correct: bool | None = False
    elif not question.answerable:
        correct = abstained
    elif abstained:
        correct = False
    elif question.must_include:
        correct = not missing
    else:
        correct = None

    return GenerationResult(
        id=question.id, doc=question.doc, type=question.type, wording=question.wording,
        answerable=question.answerable, question=question.question, expected=question.answer,
        answer=answer.text, mode=answer.mode, abstained=abstained, missing=missing, correct=correct,
        found=found_in(context, gold), full=fully_supported(context, gold),
        cites_evidence=not cited.isdisjoint(gold), warnings=list(answer.warnings),
        seconds=round(answer.seconds, 1),
    )


def evaluate_generation(
    pipeline: RAGPipeline, questions: list[GoldQuestion], progress: Callable[[str], None] = lambda _: None
) -> list[GenerationResult]:
    results = []
    for n, question in enumerate(questions, start=1):
        result = score_answer(question, pipeline.ask(question.question))
        progress(f"[{n}/{len(questions)}] {result.id}: {_verdict(result)} ({result.seconds:.0f} s)")
        results.append(result)
    return results


# --- Report -------------------------------------------------------------------


def _verdict(result: GenerationResult) -> str:
    if result.mode != GENERATED:
        return "LLM failed"
    if result.correct is None:
        return "manual check"
    return "correct" if result.correct else "wrong"


def _auto_rate(results: list[GenerationResult]) -> str:
    return str(Rate.of(r.correct for r in results if r.correct is not None))


def _breakdown(results: list[GenerationResult], field_name: str) -> list[str]:
    rows = [
        [key, str(len(group)), _auto_rate(group), str(Rate.of(r.cites_evidence for r in group)),
         str(Rate.of(not r.warnings for r in group))]
        for key, group in group_by(results, lambda r: getattr(r, field_name)).items()
    ]
    return markdown_table([field_name, "n", "correct (auto)", "cites evidence", "no warnings"], rows)


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line.strip() else ">" for line in text.strip().splitlines()]


def format_report(results: list[GenerationResult], label: str, model: str) -> str:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    scored = [r for r in answerable if r.correct is not None]
    wrong = [r for r in scored if not r.correct]
    llm_failures = [r for r in results if r.mode != GENERATED]

    lines = [
        f"# Answer-quality evaluation - {label}",
        "",
        f"Gold set: `{GOLD_PATH.name}`, {len(answerable)} answerable + {len(unanswerable)} unanswerable questions. "
        f"Generator: `{model}` (temperature 0, fixed seed). Generated by `python -m eval.run generation`.",
        "",
    ]
    if llm_failures:
        lines += [f"**{len(llm_failures)} question(s) got no generated answer (LLM error or timeout): "
                  f"{', '.join(r.id for r in llm_failures)}. They count as wrong.**", ""]
    lines += [
        "## 1. Summary",
        "",
        *markdown_table(
            ["Measure", "Result"],
            [
                ["Answerable: correct (all key values stated, automatic)", str(Rate.of(r.correct for r in scored))],
                ["Answerable: wrongly said \"not found\"", str(Rate.of(r.abstained for r in answerable))],
                ["Answerable: cites at least one evidence passage", str(Rate.of(r.cites_evidence for r in answerable))],
                ["Unanswerable: correctly said \"not found\"", str(Rate.of(r.abstained for r in unanswerable))],
                ["All: answers with no citation warning", str(Rate.of(not r.warnings for r in results))],
                ["Average seconds per question", f"{mean([r.seconds for r in results]):.1f}"],
            ],
        ),
        "",
        f"Not scored automatically (no key values; see section 4): {', '.join(r.id for r in answerable if r.correct is None) or 'none'}.",
        "",
        "## 2. Retrieval or generation? (wrong answerable answers)",
        "",
        "A wrong answer whose evidence reached the LLM is a **generation** failure (the model had what it needed); "
        "one whose evidence never arrived is a **retrieval** failure.",
        "",
        *markdown_table(
            ["Failure source", "count", "questions"],
            [[source, str(len(group)), ", ".join(r.id for r in group)]
             for source, group in group_by(wrong, lambda r: r.failure_source).items()] or [["none", "0", ""]],
        ),
        "",
        "## 3. Breakdowns",
        "",
        *_breakdown(results, "doc"),
        "",
        *_breakdown(results, "type"),
        "",
        *_breakdown(answerable, "wording"),
        "",
        "## 4. Per question",
        "",
        *markdown_table(
            ["id", "verdict", "missing values", "evidence reached LLM", "cites evidence", "warnings", "seconds"],
            [
                [r.id, _verdict(r), ", ".join(r.missing) or "-", yes_no(r.found) if r.answerable else "-",
                 yes_no(r.cites_evidence) if r.answerable else "-", str(len(r.warnings)), f"{r.seconds:.0f}"]
                for r in results
            ],
        ),
        "",
        "## 5. Answers for manual review",
        "",
        "Mark `[x]` when the answer is right and its citations support it; note anything wrong underneath.",
        "",
    ]
    for r in results:
        lines += [f"### {r.id} - {_verdict(r)}", "", f"- [ ] **Q:** {r.question}", f"- **Expected:** {r.expected}", ""]
        lines += _quote(r.answer) + [""]
        lines += [f"- Warning: {warning}" for warning in r.warnings]
        if r.warnings:
            lines.append("")
    return "\n".join(lines)


def write_results(
    results: list[GenerationResult], label: str, model: str, out_dir: Path = EVAL_RESULTS_DIR
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"generation_{label}.md"
    data = out_dir / f"generation_{label}.json"
    report.write_text(format_report(results, label, model), encoding="utf-8")
    data.write_text(json.dumps([asdict(r) for r in results], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return report, data
