"""Answer-quality scoring: key values, abstention, failure source, and the runner (fake LLM, offline)."""

import json

import pytest

from eval.generation import evaluate_generation, score_answer, write_results
from eval.gold import Evidence, GoldQuestion
from eval.metrics import is_abstention, missing_values
from src.cite import NOT_FOUND, CitationReport
from src.pipeline import Answer, RAGPipeline
from src.retrieve import SearchResult
from tests.helpers import make_chunk, make_hit

# --- key values and abstention -------------------------------------------------------


@pytest.mark.parametrize(
    "answer, required, missing",
    [
        ("The coefficient is 1.8 [S2].", ["1.8"], []),
        ("The coefficient is 1.80 [S2].", ["1.8"], []),          # trailing zero
        ("Liable for twelve months [S1].", ["12"], []),          # number word
        ("See source [S1].", ["1"], ["1"]),                      # marker digits don't count
        ("Cpi is -0.8 [S1].", ["-0.8"], []),
        ("Cpi is −0.80 [S1].", ["-0.8"], []),               # Unicode minus, trailing zero
        ("Cpi is 0.8 [S1].", ["-0.8"], ["-0.8"]),                # sign missing
        ("Cpi is -0.85 [S1].", ["-0.8"], ["-0.8"]),              # different value
        ("A bypass in madhya pradesh [S1].", ["Madhya Pradesh"], []),
        ("Refer to IS 15498 : 2004 [S2].", ["15498", "2004"], []),
        ("Refer to the cyclone code [S2].", ["15498"], ["15498"]),
    ],
)
def test_missing_values(answer, required, missing):
    assert missing_values(answer, required) == missing


@pytest.mark.parametrize(
    "answer, expected",
    [
        (NOT_FOUND, True),
        ("not found in the provided documents", True),
        ('**"Not found in the provided documents."**', True),
        ("The rate is 1%. The cap is not found in the provided documents.", False),
        ("The rate is 1% [S1].", False),
    ],
)
def test_is_abstention(answer, expected):
    assert is_abstention(answer) is expected


# --- scoring one answer ----------------------------------------------------------------

T28 = make_chunk("t28", "phi 0.2 = 1.0")
AMD = make_chunk("amd", "Substitute 1.8 for 1.0")
OTHER = make_chunk("other", "unrelated")


def _question(qid="q1", must_include=("1.8",), gold=("t28", "amd")):
    evidence = tuple(Evidence(cid, cid, (1,), "x") for cid in gold)
    if not gold:
        return GoldQuestion(qid, "Delhi Metro penalty?", "none", "unanswerable", "natural", NOT_FOUND)
    return GoldQuestion(qid, "Force coefficient?", "is875", "amendment", "document", "1.8",
                        must_include=tuple(must_include), evidence=evidence)


def _answer(text, sources=(T28, AMD), cited=(2,), mode="generated", warnings=()):
    return Answer("Q", text, mode, [make_hit(c) for c in sources], CitationReport(cited=list(cited)), list(warnings))


def test_correct_answer_citing_the_amendment():
    result = score_answer(_question(), _answer("It is 1.8 [S2]."))
    assert result.correct and result.cites_evidence and result.full and result.failure_source == ""


def test_wrong_value_with_evidence_in_context_is_a_generation_failure():
    result = score_answer(_question(), _answer("It is 1.0 [S1].", cited=(1,)))
    assert result.correct is False and result.missing == ["1.8"]
    assert result.failure_source == "generation"


def test_wrong_value_without_evidence_in_context_is_a_retrieval_failure():
    result = score_answer(_question(), _answer("It is 1.0 [S1].", sources=(OTHER,), cited=(1,)))
    assert result.failure_source == "retrieval" and not result.cites_evidence


def test_abstaining_on_an_answerable_question_is_wrong():
    result = score_answer(_question(), _answer(NOT_FOUND, cited=()))
    assert result.abstained and result.correct is False


def test_question_without_key_values_needs_a_manual_check():
    assert score_answer(_question(must_include=()), _answer("It is 1.8 [S2].")).correct is None


def test_unanswerable_question_is_correct_only_when_the_answer_abstains():
    assert score_answer(_question(gold=()), _answer(NOT_FOUND, sources=(OTHER,), cited=())).correct is True
    assert score_answer(_question(gold=()), _answer("It is 5% [S1].", sources=(OTHER,), cited=(1,))).correct is False


def test_retrieval_only_fallback_counts_as_wrong():
    result = score_answer(_question(), _answer("Most relevant passages", mode="retrieval-only", cited=()))
    assert result.correct is False and result.missing == ["1.8"] and not result.abstained


# --- runner through the real pipeline ----------------------------------------------------


class ScriptedRetriever:
    def search(self, question, top_k, mode):
        return SearchResult(question, [make_hit(T28), make_hit(AMD)], best_cosine=0.9)


class ScriptedLLM:
    name = "scripted"

    def available(self):
        return True

    def complete(self, system, prompt):
        return NOT_FOUND if "Delhi Metro" in prompt else "The force coefficient is 1.8 [S2]."


def test_evaluate_generation_runs_the_pipeline_and_writes_the_report(tmp_path):
    pipeline = RAGPipeline(retriever=ScriptedRetriever(), llm=ScriptedLLM())
    messages = []
    results = evaluate_generation(pipeline, [_question(), _question("q2", gold=())], progress=messages.append)
    assert [r.correct for r in results] == [True, True]
    assert messages[0].startswith("[1/2] q1: correct")

    report, data = write_results(results, "toy", "scripted", out_dir=tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "Generator: `scripted`" in text
    assert '| Unanswerable: correctly said "not found" | 1/1 (100%) |' in text
    assert "> The force coefficient is 1.8 [S2]." in text
    assert [row["id"] for row in json.loads(data.read_text(encoding="utf-8"))] == ["q1", "q2"]
