"""The pipeline must never fail because of the LLM: it degrades to retrieval-only."""

from src.generate import LLMError
from src.pipeline import RAGPipeline
from src.retrieve import SearchResult
from tests.helpers import make_chunk, make_hit

HITS = [make_hit(make_chunk("c1", "Compensation is 1% per month.")), make_hit(make_chunk("c2"))]


class FakeRetriever:
    def __init__(self, best_cosine: float = 0.9):
        self.best_cosine = best_cosine

    def search(self, question, top_k, mode):
        return SearchResult(question, HITS, self.best_cosine)


class FakeLLM:
    name = "fake"

    def __init__(self, reachable: bool = True, answer: str | None = "The rate is 1% per month of delay [S1].",
                 error: Exception | None = None):
        self.reachable, self.answer, self.error = reachable, answer, error

    def available(self):
        return self.reachable

    def complete(self, system, prompt):
        if self.error:
            raise self.error
        return self.answer


def ask(llm, **kwargs):
    return RAGPipeline(retriever=FakeRetriever(**kwargs), llm=llm).ask("question")


def test_generated_answer_is_checked_for_citations():
    answer = ask(FakeLLM())
    assert answer.mode == "generated"
    assert answer.citations.cited == [1]
    assert answer.warnings == []


def test_unreachable_llm_falls_back_to_retrieval_only():
    answer = ask(FakeLLM(reachable=False))
    assert answer.mode == "retrieval-only"
    assert answer.sources == HITS
    assert "not reachable" in answer.warnings[0]


def test_llm_error_falls_back_to_retrieval_only():
    answer = ask(FakeLLM(error=LLMError("timed out")))
    assert answer.mode == "retrieval-only"
    assert "timed out" in answer.warnings[0]


def test_use_llm_false_is_retrieval_only_without_warning():
    answer = RAGPipeline(retriever=FakeRetriever(), llm=FakeLLM()).ask("q", use_llm=False)
    assert answer.mode == "retrieval-only"
    assert answer.warnings == []


def test_low_confidence_and_citation_problems_are_warned():
    answer = ask(FakeLLM(answer="The compensation is capped at ten percent of the value [S7]."), best_cosine=0.1)
    joined = " ".join(answer.warnings)
    assert "Low retrieval confidence" in joined
    assert "non-existent sources: S7" in joined
