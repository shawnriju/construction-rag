import pytest
import requests

from src.generate import LLMError, NoLLM, OllamaLLM, amendment_links, build_prompt, get_llm, tidy_answer
from tests.helpers import make_chunk, make_hit

TABLE = make_chunk("table-28", "| 0.2 | 1.0 |", section="Table 28", covers=["Table 28"])
AMENDMENT = make_chunk(
    "amd2-10", "Substitute '1.8' for '1.0'.", section="Amendment No. 2, item 10",
    is_amendment=True, amends=["Table 28"], printed_pages=[],
)
OTHER = make_chunk("cl-6-3", "Unrelated clause.", section="Cl. 6.3", covers=["Cl. 6.3"])


def test_amendment_links_map_originals_to_their_amendments():
    hits = [make_hit(TABLE), make_hit(OTHER), make_hit(AMENDMENT)]
    assert amendment_links(hits) == {1: [3]}


def test_prompt_puts_the_amendment_inside_the_original_block():
    prompt = build_prompt("What is Cf?", [make_hit(TABLE), make_hit(AMENDMENT)])
    original_block = prompt.split("\n\n[S2]")[0]
    assert "AMENDED by" in original_block
    assert "Substitute '1.8' for '1.0'." in original_block
    assert "This is an AMENDMENT to [S1]." in prompt


def test_prompt_repeats_the_question_after_the_sources():
    prompt = build_prompt("What is Cf?", [make_hit(OTHER)])
    assert prompt.startswith("Question: What is Cf?")
    assert prompt.rstrip().endswith("Answer (cite sources as [S#]):")
    assert prompt.count("Question: What is Cf?") == 2


def test_tidy_answer_strips_latex_delimiters():
    raw = "The formula is:\n\n\n\\[ p_z = 0.6 v_z^2 \\]\n\nwhere \\( p_z \\) is the pressure [S7]."
    assert tidy_answer(raw) == "The formula is:\n\n p_z = 0.6 v_z^2 \n\nwhere  p_z  is the pressure [S7]."


def test_prompt_without_amendments_has_no_amendment_notes():
    assert "AMENDED" not in build_prompt("q", [make_hit(OTHER)])


def test_ollama_requests_are_deterministic(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": " answer "}}

    def capture(url, json, **_):
        sent.update(json)
        return Response()

    monkeypatch.setattr(requests, "post", capture)
    assert OllamaLLM().complete("system", "prompt") == "answer"
    assert sent["options"]["temperature"] == 0.0
    assert "seed" in sent["options"]


def test_ollama_timeout_becomes_llm_error(monkeypatch):
    def timeout(*_, **__):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", timeout)
    with pytest.raises(LLMError, match="did not answer"):
        OllamaLLM().complete("system", "prompt")


def test_ollama_malformed_response_becomes_llm_error(monkeypatch):
    class BadResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"unexpected": True}

    monkeypatch.setattr(requests, "post", lambda *_, **__: BadResponse())
    with pytest.raises(LLMError):
        OllamaLLM().complete("system", "prompt")


def test_ollama_unreachable_is_reported_as_unavailable(monkeypatch):
    def refuse(*_, **__):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "get", refuse)
    assert OllamaLLM().available() is False


def test_backends():
    assert isinstance(get_llm("none"), NoLLM)
    with pytest.raises(LLMError):
        NoLLM().complete("s", "p")
    with pytest.raises(ValueError):
        get_llm("unknown")
