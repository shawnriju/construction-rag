"""Evaluation metrics and the retrieval evaluation (toy retriever: offline, no model)."""

import json

import pytest

from eval.gold import Evidence, GoldQuestion
from eval.metrics import (
    Rate,
    best_threshold,
    first_gold_rank,
    found_in,
    fully_supported,
    hit_at,
    reciprocal_rank,
    score_threshold,
    wins_losses,
)
from eval.retrieval import (
    CONTEXT_CONFIGS,
    FULL_PIPELINE,
    RetrievalResult,
    evaluate_retrieval,
    format_comparison,
    format_report,
    verdict,
    write_results,
)
from src.cite import NOT_FOUND
from src.retrieve import Retriever
from tests.helpers import FakeEmbedder, make_chunk, unit_embeddings

# --- metrics --------------------------------------------------------------------


def test_rate_prints_count_and_percentage():
    assert str(Rate.of([True, True, False])) == "2/3 (67%)"
    assert str(Rate.of([])) == "-"


def test_first_gold_rank_uses_the_best_ranked_evidence_chunk():
    assert first_gold_rank(["a", "b", "c"], {"c", "b"}) == 2
    assert first_gold_rank(["a"], {"z"}) is None


@pytest.mark.parametrize("rank, k, expected", [(1, 1, True), (5, 5, True), (6, 5, False), (None, 10, False)])
def test_hit_at(rank, k, expected):
    assert hit_at(rank, k) is expected


def test_reciprocal_rank_is_zero_below_the_cutoff():
    assert reciprocal_rank(4, 10) == 0.25
    assert reciprocal_rank(11, 10) == 0.0
    assert reciprocal_rank(None, 10) == 0.0


def test_found_needs_one_evidence_chunk_and_fully_supported_needs_all():
    assert found_in(["t28", "x"], ["t28", "amd"]) and not fully_supported(["t28", "x"], ["t28", "amd"])
    assert fully_supported(["amd", "t28"], ["t28", "amd"])


def test_wins_losses_compares_hit_first_then_rank():
    a = {"q1": 7, "q2": 1, "q3": 2, "q4": None}
    b = {"q1": 2, "q2": 9, "q3": 1, "q4": None}
    result = wins_losses(a, b, k=5, mrr_cutoff=10)
    # q1: miss -> hit (win); q2: hit -> miss (loss); q3: both hit, better rank (win); q4: tie.
    assert result.wins == ("q1", "q3") and result.losses == ("q2",) and result.ties == 1
    assert str(result) == "fixed 2 (q1, q3), broke 1 (q2), same 1"


def test_threshold_flags_scores_below_it():
    choice = score_threshold(0.7, answerable=[0.8, 0.65], unanswerable=[0.6, 0.75])
    assert (choice.unanswerable_flagged.hits, choice.answerable_flagged.hits) == (1, 1)


def test_best_threshold_separates_when_possible_and_prefers_fewer_false_alarms():
    assert best_threshold([0.8, 0.9], [0.5, 0.6]).threshold == 0.7
    # Overlap: 0.62 and 0.75 both classify 3 of 4 correctly; 0.75 would flag the 0.7 answer, so 0.62 wins.
    choice = best_threshold(answerable=[0.7, 0.8], unanswerable=[0.6, 0.72])
    assert choice.threshold == 0.65 and choice.answerable_flagged.hits == 0


# --- runner -----------------------------------------------------------------------

TABLE_28 = make_chunk("t28", "force coefficient single frame flat sided", section="Table 28", covers=["Table 28"])
AMD = make_chunk("amd", "substitute 1.8 for 1.0", is_amendment=True, amends=["Table 28"], printed_pages=[])
FILLER = [make_chunk(f"f{i}", f"unrelated clause number {i}") for i in range(8)]


def _question(qid, question, gold=(), qtype="amendment", wording="document"):
    evidence = tuple(Evidence(cid, cid, (1,), "x") for cid in gold)
    answer = "1.8" if gold else NOT_FOUND
    doc = "is875" if gold else "none"
    return GoldQuestion(qid, question, doc, qtype if gold else "unanswerable", wording, answer, evidence=evidence)


@pytest.fixture(scope="module")
def results():
    chunks = [TABLE_28, *FILLER, AMD]
    embeddings = unit_embeddings(len(chunks))
    # The amendment points away from every query, so dense ranks it last and it can only reach
    # the top 6 through amendment expansion (otherwise ties would decide).
    embeddings[-1] = [0.0, 1.0, 0.0, 0.0]
    retriever = Retriever(chunks, embeddings, FakeEmbedder())
    questions = [
        _question("q1", "force coefficient flat sided frame", gold=("t28", "amd")),
        _question("q2", "something about nothing", qtype="unanswerable", wording="natural"),
    ]
    return evaluate_retrieval(retriever, questions)


def test_amendment_counts_only_when_expansion_is_on(results):
    q1 = results[0]
    assert q1.ranks["bm25"] == 1
    assert q1.full[FULL_PIPELINE] and not q1.full["without amendments"]
    assert all(q1.found[name] for name in CONTEXT_CONFIGS)


def test_report_and_json_are_written(results, tmp_path):
    report, data = write_results(results, "toy", out_dir=tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "# Retrieval evaluation - toy" in text
    assert "| full pipeline | 1/1 (100%) | 1/1 (100%) |" in text
    assert [row["id"] for row in json.loads(data.read_text(encoding="utf-8"))] == ["q1", "q2"]


def test_report_without_unanswerable_questions_skips_calibration(results):
    assert "nothing to calibrate" in format_report(results[:1], "toy")


# --- comparing two runs ------------------------------------------------------------------


def _run(ranks: dict[str, int | None], amendment_full: bool = True) -> list[RetrievalResult]:
    results = []
    for qid, rank in ranks.items():
        qtype = "amendment" if qid == "amd" else "fact"
        full = amendment_full if qid == "amd" else True
        results.append(RetrievalResult(qid, "is875", qtype, "natural", True, {m: rank for m in ("bm25", "dense", "hybrid")},
                                       {FULL_PIPELINE: True}, {FULL_PIPELINE: full}, {FULL_PIPELINE: 6}, 0.8))
    return results


def test_new_embedder_ships_with_more_wins_and_no_broken_amendment():
    base = _run({"a": 9, "b": 1, "amd": 1})
    assert verdict(base, _run({"a": 2, "b": 1, "amd": 1})).ships
    assert not verdict(base, _run({"a": 2, "b": 8, "amd": 1})).ships                       # 1 win, 1 loss
    assert not verdict(base, _run({"a": 2, "b": 1, "amd": 1}, amendment_full=False)).ships  # amendment broken


def test_comparison_refuses_runs_on_different_questions():
    with pytest.raises(ValueError, match="different gold questions"):
        verdict(_run({"a": 1}), _run({"b": 1}))


def test_comparison_report_states_the_verdict():
    text = format_comparison(_run({"a": 9, "b": 1, "amd": 1}), _run({"a": 2, "b": 1, "amd": 1}), "base", "finetuned")
    assert "**FINETUNED SHIPS**" in text
    assert "| a | natural | 9 | 2 | 9 | 2 |" in text
