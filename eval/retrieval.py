"""Retrieval evaluation on the frozen gold set (run it with `python -m eval.run retrieval`).

Writes `eval/results/retrieval_<label>.md` (the report) and `retrieval_<label>.json`
(per-question results, so two runs - e.g. base vs fine-tuned embedder - can be compared
question by question later). Deterministic, no LLM needed, about a minute on CPU.

What is measured (answerable questions only; unanswerable ones are used for the threshold):
  1. Ranking quality per retriever (BM25, dense, hybrid): Hit@1/5/10 and MRR@10, using the
     rank of the first evidence chunk in the plain ranked list (no pinning or expansion).
  2. What reaches the LLM with the real pipeline (top 6 + pinning + table parts + amendments),
     scored as "found" and "fully supported", and the same with each feature switched off.
  3. A low-confidence cutoff on the best dense cosine, calibrated on the unanswerable questions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.gold import GoldQuestion
from eval.metrics import (
    Rate,
    WinsLosses,
    best_threshold,
    first_gold_rank,
    found_in,
    fully_supported,
    hit_at,
    mean,
    reciprocal_rank,
    score_threshold,
    wins_losses,
)
from eval.report import group_by, markdown_table, yes_no
from src.config import CANDIDATES_PER_RETRIEVER, EVAL_RESULTS_DIR, GOLD_PATH, LOW_CONFIDENCE_COSINE, TOP_K
from src.retrieve import Retriever

RANKED_MODES = ("bm25", "dense", "hybrid")
HIT_CUTOFFS = (1, 5, 10)
MRR_CUTOFF = 10
HEADLINE_K = 5  # Hit@5 is the headline number and the fine-tune's success rule (docs/PLAN.md section 3).
# Long enough to hold every candidate either retriever returns, so a rank beyond 10 is still visible.
RANKED_DEPTH = 2 * CANDIDATES_PER_RETRIEVER

# The real pipeline, then one feature off at a time, then none of them (the "plain top 6").
CONTEXT_CONFIGS: dict[str, dict[str, bool]] = {
    "full pipeline": {},
    "without pinning": {"pin": False},
    "without table parts": {"siblings": False},
    "without amendments": {"expand": False},
    "plain top 6": {"pin": False, "siblings": False, "expand": False},
}
FULL_PIPELINE = "full pipeline"


@dataclass
class RetrievalResult:
    """Everything measured for one question (serialised to JSON as-is)."""

    id: str
    doc: str
    type: str
    wording: str
    answerable: bool
    ranks: dict[str, int | None]   # Retriever -> rank of the first evidence chunk (None = not in the list).
    found: dict[str, bool]         # Context config -> at least one evidence chunk sent to the LLM.
    full: dict[str, bool]          # Context config -> every evidence chunk sent to the LLM.
    chunks_sent: dict[str, int]    # Context config -> number of chunks sent to the LLM.
    best_cosine: float


def evaluate_question(retriever: Retriever, question: GoldQuestion) -> RetrievalResult:
    gold = question.gold_chunk_ids
    ranks, found, full, sent = {}, {}, {}, {}
    best_cosine = 0.0
    for mode in RANKED_MODES:
        result = retriever.search(question.question, top_k=RANKED_DEPTH, mode=mode, expand=False, siblings=False, pin=False)
        ranks[mode] = first_gold_rank([h.chunk.chunk_id for h in result.hits], gold)
        best_cosine = result.best_cosine  # Same in every mode: always the best dense hit.
    for name, switches in CONTEXT_CONFIGS.items():
        context = [h.chunk.chunk_id for h in retriever.search(question.question, top_k=TOP_K, **switches).hits]
        found[name] = found_in(context, gold)
        full[name] = fully_supported(context, gold)
        sent[name] = len(context)
    return RetrievalResult(
        question.id, question.doc, question.type, question.wording, question.answerable,
        ranks, found, full, sent, round(best_cosine, 4),
    )


def evaluate_retrieval(
    retriever: Retriever, questions: list[GoldQuestion], progress: Callable[[str], None] = lambda _: None
) -> list[RetrievalResult]:
    results = []
    for n, question in enumerate(questions, start=1):
        progress(f"[{n}/{len(questions)}] {question.id}")
        results.append(evaluate_question(retriever, question))
    return results


# --- Report -------------------------------------------------------------------


def _rank(rank: int | None) -> str:
    return str(rank) if rank is not None else f">{RANKED_DEPTH}"


def _ranked_row(name: str, results: list[RetrievalResult], mode: str) -> list[str]:
    ranks = [r.ranks[mode] for r in results]
    hits = [str(Rate.of(hit_at(rank, k) for rank in ranks)) for k in HIT_CUTOFFS]
    return [name, *hits, f"{mean([reciprocal_rank(rank, MRR_CUTOFF) for rank in ranks]):.3f}"]


def _breakdown(results: list[RetrievalResult], field_name: str) -> list[str]:
    rows = [
        [
            key, str(len(group)),
            *(str(Rate.of(hit_at(r.ranks[mode], HEADLINE_K) for r in group)) for mode in RANKED_MODES),
            str(Rate.of(r.found[FULL_PIPELINE] for r in group)),
            str(Rate.of(r.full[FULL_PIPELINE] for r in group)),
        ]
        for key, group in group_by(results, lambda r: getattr(r, field_name)).items()
    ]
    header = [field_name, "n", *(f"{m} Hit@{HEADLINE_K}" for m in RANKED_MODES), "found (pipeline)", "fully supported"]
    return markdown_table(header, rows)


def format_report(results: list[RetrievalResult], label: str) -> str:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    by_mode = {mode: {r.id: r.ranks[mode] for r in answerable} for mode in RANKED_MODES}

    lines = [
        f"# Retrieval evaluation - {label}",
        "",
        f"Gold set: `{GOLD_PATH.name}`, {len(answerable)} answerable + {len(unanswerable)} unanswerable questions. "
        "Generated by `python -m eval.run retrieval`; deterministic.",
        "",
        "## 1. Ranking quality (answerable questions)",
        "",
        "Rank of the first evidence passage in each retriever's ranked list (no pinning or expansion). "
        f"Hit@k = that passage is in the top k; MRR@{MRR_CUTOFF} = mean of 1/rank (0 if below {MRR_CUTOFF}).",
        "",
        *markdown_table(["Retriever", *(f"Hit@{k}" for k in HIT_CUTOFFS), f"MRR@{MRR_CUTOFF}"],
                [_ranked_row(mode, answerable, mode) for mode in RANKED_MODES]),
        "",
        f"Per question, Hit@{HEADLINE_K} then MRR (hybrid against each single retriever):",
        "",
        f"- hybrid vs bm25: {wins_losses(by_mode['bm25'], by_mode['hybrid'], HEADLINE_K, MRR_CUTOFF)}",
        f"- hybrid vs dense: {wins_losses(by_mode['dense'], by_mode['hybrid'], HEADLINE_K, MRR_CUTOFF)}",
        "",
        "## 2. What reaches the LLM",
        "",
        f"Chunks actually sent to the LLM (top {TOP_K} hybrid + additions). Found = at least one evidence passage; "
        "fully supported = all of them (e.g. the original *and* its amendment).",
        "",
        *markdown_table(
            ["Pipeline", "found", "fully supported", "avg chunks sent"],
            [
                [name, str(Rate.of(r.found[name] for r in answerable)), str(Rate.of(r.full[name] for r in answerable)),
                 f"{mean([r.chunks_sent[name] for r in answerable]):.1f}"]
                for name in CONTEXT_CONFIGS
            ],
        ),
        "",
        "## 3. Breakdowns",
        "",
        *_breakdown(answerable, "doc"),
        "",
        *_breakdown(answerable, "type"),
        "",
        *_breakdown(answerable, "wording"),
        "",
        *_threshold_section(answerable, unanswerable),
        "",
        "## 5. Per question",
        "",
        *markdown_table(
            ["id", "wording", *RANKED_MODES, "found", "fully supported", "best cosine"],
            [
                [r.id, r.wording, *(_rank(r.ranks[m]) for m in RANKED_MODES),
                 yes_no(r.found[FULL_PIPELINE]), yes_no(r.full[FULL_PIPELINE]), f"{r.best_cosine:.3f}"]
                for r in answerable
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _threshold_section(answerable: list[RetrievalResult], unanswerable: list[RetrievalResult]) -> list[str]:
    lines = ["## 4. Low-confidence cutoff (best dense cosine)", ""]
    if not unanswerable:
        return lines + ["No unanswerable questions, nothing to calibrate.", ""]
    ok = [r.best_cosine for r in answerable]
    nope = [r.best_cosine for r in unanswerable]
    current = score_threshold(LOW_CONFIDENCE_COSINE, ok, nope)
    best = best_threshold(ok, nope)
    rows = [
        [f"current ({LOW_CONFIDENCE_COSINE})", str(current.unanswerable_flagged), str(current.answerable_flagged)],
        [f"best on this set ({best.threshold})", str(best.unanswerable_flagged), str(best.answerable_flagged)],
    ]
    return lines + [
        f"Answerable: min {min(ok):.3f}, median {sorted(ok)[len(ok) // 2]:.3f}. "
        f"Unanswerable: min {min(nope):.3f}, max {max(nope):.3f}.",
        "",
        *markdown_table(["Cutoff", "unanswerable flagged (want all)", "answerable flagged (want none)"], rows),
        "",
        "Unanswerable questions: " + ", ".join(f"{r.id} {r.best_cosine:.3f}" for r in unanswerable),
    ]


def write_results(results: list[RetrievalResult], label: str, out_dir: Path = EVAL_RESULTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"retrieval_{label}.md"
    data = out_dir / f"retrieval_{label}.json"
    report.write_text(format_report(results, label), encoding="utf-8")
    data.write_text(json.dumps([asdict(r) for r in results], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return report, data



# --- Comparing two runs (base vs fine-tuned embedder) ---------------------------------


def load_results(label: str, out_dir: Path = EVAL_RESULTS_DIR) -> list[RetrievalResult]:
    """Per-question results written by an earlier `write_results(..., label)`."""
    rows = json.loads((out_dir / f"retrieval_{label}.json").read_text(encoding="utf-8"))
    return [RetrievalResult(**row) for row in rows]


@dataclass(frozen=True)
class Verdict:
    """The pre-committed success rule (docs/PLAN.md section 3), applied to two runs."""

    hybrid: WinsLosses
    broken_amendment_questions: tuple[str, ...]

    @property
    def ships(self) -> bool:
        return len(self.hybrid.wins) > len(self.hybrid.losses) and not self.broken_amendment_questions


def verdict(base: list[RetrievalResult], new: list[RetrievalResult]) -> Verdict:
    """New ships only if it wins more questions than it loses on hybrid Hit@5 (ties broken by MRR)
    and every amendment question that was fully supported stays fully supported."""
    base_answerable = {r.id: r for r in base if r.answerable}
    new_answerable = {r.id: r for r in new if r.answerable}
    if base_answerable.keys() != new_answerable.keys():
        raise ValueError("The two runs were made on different gold questions.")
    hybrid = wins_losses({q: r.ranks["hybrid"] for q, r in base_answerable.items()},
                         {q: r.ranks["hybrid"] for q, r in new_answerable.items()}, HEADLINE_K, MRR_CUTOFF)
    broken = tuple(
        q for q, r in base_answerable.items()
        if r.type == "amendment" and r.full[FULL_PIPELINE] and not new_answerable[q].full[FULL_PIPELINE]
    )
    return Verdict(hybrid, broken)


def format_comparison(base: list[RetrievalResult], new: list[RetrievalResult], base_label: str, new_label: str) -> str:
    base_answerable = [r for r in base if r.answerable]
    new_answerable = [r for r in new if r.answerable]
    decision = verdict(base, new)  # Also checks that both runs cover the same questions.
    new_by_id = {r.id: r for r in new_answerable}
    rows = []
    for mode in ("dense", "hybrid"):
        rows.append(_ranked_row(f"{mode} ({base_label})", base_answerable, mode))
        rows.append(_ranked_row(f"{mode} ({new_label})", new_answerable, mode))
    dense = wins_losses({r.id: r.ranks["dense"] for r in base_answerable},
                        {r.id: r.ranks["dense"] for r in new_answerable}, HEADLINE_K, MRR_CUTOFF)
    by_wording = {
        wording: str(Rate.of(hit_at(r.ranks["hybrid"], HEADLINE_K) for r in group))
        for wording, group in group_by(new_answerable, lambda r: r.wording).items()
    }
    base_by_wording = {
        wording: str(Rate.of(hit_at(r.ranks["hybrid"], HEADLINE_K) for r in group))
        for wording, group in group_by(base_answerable, lambda r: r.wording).items()
    }
    lines = [
        f"# Retrieval comparison - {base_label} vs {new_label}",
        "",
        "Generated by `python -m eval.run compare`. Same frozen gold set, same pipeline; only the embedder differs.",
        "",
        "## Verdict (rule fixed before training, docs/PLAN.md section 3)",
        "",
        f"- Hybrid Hit@{HEADLINE_K} per question ({new_label} vs {base_label}): {decision.hybrid}",
        f"- Amendment questions no longer fully supported: {', '.join(decision.broken_amendment_questions) or 'none'}",
        f"- **{new_label.upper() + ' SHIPS' if decision.ships else 'KEEP ' + base_label.upper()}**: "
        + ("more wins than losses and no amendment question broken."
           if decision.ships else "the rule needs more wins than losses and no broken amendment question."),
        "",
        "## Ranking quality",
        "",
        *markdown_table(["Retriever", *(f"Hit@{k}" for k in HIT_CUTOFFS), f"MRR@{MRR_CUTOFF}"], rows),
        "",
        f"- Dense only, per question: {dense}",
        f"- Hybrid Hit@{HEADLINE_K} by wording: "
        + ", ".join(f"{w} {base_by_wording[w]} -> {by_wording[w]}" for w in by_wording),
        "",
        "## What reaches the LLM (full pipeline)",
        "",
        *markdown_table(
            ["Run", "found", "fully supported"],
            [[label, str(Rate.of(r.found[FULL_PIPELINE] for r in group)), str(Rate.of(r.full[FULL_PIPELINE] for r in group))]
             for label, group in ((base_label, base_answerable), (new_label, new_answerable))],
        ),
        "",
        "## Per question (rank of the first evidence passage)",
        "",
        *markdown_table(
            ["id", "wording", f"dense {base_label}", f"dense {new_label}", f"hybrid {base_label}", f"hybrid {new_label}"],
            [[b.id, b.wording, _rank(b.ranks["dense"]), _rank(new_by_id[b.id].ranks["dense"]), _rank(b.ranks["hybrid"]),
              _rank(new_by_id[b.id].ranks["hybrid"])] for b in base_answerable],
        ),
        "",
    ]
    return "\n".join(lines)
