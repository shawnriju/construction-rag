"""The gold set: the committed file must match the committed chunks, and the validator must catch drift."""

import json

import pytest

from eval.gold import Evidence, GoldQuestion, load_gold, render_review_sheet, validate
from src.cite import NOT_FOUND
from src.config import CHUNKS_PATH, GOLD_PATH
from src.schema import load_chunks
from tests.helpers import make_chunk


def _question(**fields) -> GoldQuestion:
    defaults = dict(
        id="q1",
        question="What is the force coefficient?",
        doc="is875",
        type="amendment",
        wording="document",
        answer="1.8",
        evidence=(Evidence("t28", "Table 28", (5,), "phi 0.2 = 1.0"),),
    )
    defaults.update(fields)
    return GoldQuestion(**defaults)


CHUNKS = [make_chunk("t28", text="Force coefficients: phi 0.2  =  1.0 for flat-sided members.", pdf_pages=[5])]


def test_committed_gold_set_matches_the_committed_chunks():
    # Fails if a rebuild renamed a gold chunk, moved it to other pages, or changed its text.
    problems = validate(load_gold(GOLD_PATH), load_chunks(CHUNKS_PATH))
    assert not problems, "\n".join(problems)


def test_a_valid_question_has_no_problems():
    # The quote matches although the chunk text has extra spaces and different case.
    assert validate([_question(evidence=(Evidence("t28", "Table 28", (5,), "PHI 0.2 = 1.0"),))], CHUNKS) == []


@pytest.mark.parametrize(
    "evidence, expected",
    [
        (Evidence("gone", "Table 28", (5,), "phi 0.2"), "not in the corpus"),
        (Evidence("t28", "Table 28", (6,), "phi 0.2"), "is on PDF [5]"),
        (Evidence("t28", "Table 28", (5,), "phi 0.3 = 1.0"), "quote not found"),
    ],
)
def test_drifted_evidence_is_reported(evidence, expected):
    problems = validate([_question(evidence=(evidence,))], CHUNKS)
    assert len(problems) == 1 and expected in problems[0]


def test_record_level_mistakes_are_reported():
    questions = [
        _question(),
        _question(),  # duplicate id
        _question(id="q2", type="opinion"),
        _question(id="q3", evidence=()),  # answerable without evidence
        _question(id="q4", type="unanswerable", doc="none", answer="Probably 50."),
    ]
    problems = validate(questions, CHUNKS)
    assert any("duplicate id" in p for p in problems)
    assert any("q2: type" in p for p in problems)
    assert any("q3: answerable question without evidence" in p for p in problems)
    assert any(p.startswith("q4: unanswerable") for p in problems)


def test_unanswerable_question_with_the_exact_phrase_is_valid():
    question = _question(type="unanswerable", doc="none", answer=NOT_FOUND, evidence=())
    assert validate([question], CHUNKS) == []


def test_load_gold_names_the_bad_line(tmp_path):
    path = tmp_path / "gold.jsonl"
    good = {"id": "q1", "question": "Q?", "doc": "none", "type": "unanswerable", "wording": "natural", "answer": NOT_FOUND}
    path.write_text(json.dumps(good) + "\n\n" + json.dumps({"id": "q2"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        load_gold(path)


def test_review_sheet_gives_the_printed_page_of_the_quoted_pdf_page_only():
    chunk = make_chunk("c25", text="give its decision within 60 days", pdf_pages=[48, 49], printed_pages=["46", "47"])
    question = _question(evidence=(Evidence("c25", "Clause 25", (49,), "within 60 days"),))
    assert "Clause 25, PDF p. 49 (printed p. 47)" in render_review_sheet([question], [chunk])


def test_review_sheet_shows_where_to_look():
    sheet = render_review_sheet([_question(note="Amendment case.")], CHUNKS)
    assert "- [ ] **Q:** What is the force coefficient?" in sheet
    assert "Table 28, PDF p. 5" in sheet
    assert "> phi 0.2 = 1.0" in sheet
