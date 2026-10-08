from src.ask import _parse_args, format_answer
from src.cite import CitationReport
from src.pipeline import Answer
from tests.helpers import make_chunk, make_hit

TABLE = make_chunk("t28", "| 0.2 | 1.0 |", section="Table 28", covers=["Table 28"], source="curated")
AMENDMENT = make_chunk("amd", "Substitute '1.8' for '1.0'.", section="Amendment No. 2, item 10",
                       is_amendment=True, amends=["Table 28"], printed_pages=[], source="curated")
SOURCES = [make_hit(TABLE), make_hit(AMENDMENT, reason="Amendment No. 2, item 10 amends Table 28")]


def test_generated_answer_marks_cited_sources_and_amendments():
    answer = Answer("q", "It is 1.8 [S2].", "generated", SOURCES, CitationReport(cited=[2]))
    out = format_answer(answer)
    assert " *[S2]" in out and "  [S1]" in out
    assert "amended by [S2]" in out
    assert "[AMENDMENT, curated]" in out


def test_retrieval_only_shows_passage_snippets_and_warnings():
    answer = Answer("q", "Most relevant passages:", "retrieval-only", SOURCES, warnings=["LLM not reachable"])
    out = format_answer(answer)
    assert "> | 0.2 | 1.0 |" in out
    assert "! LLM not reachable" in out


def test_debug_shows_ranks():
    hit = make_hit(TABLE)
    hit.ranks = {"bm25": 1, "dense": 3}
    out = format_answer(Answer("q", "x", "retrieval-only", [hit]), debug=True)
    assert "bm25 #1, dense #3" in out


def test_cli_arguments():
    args = _parse_args(["What is k2?", "--mode", "bm25", "--no-llm"])
    assert (args.question, args.mode, args.no_llm, args.debug) == ("What is k2?", "bm25", True, False)
