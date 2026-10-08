"""Training-pair construction: parsing, filters, hard negatives, split (toy retriever, no LLM or model)."""

import json

from finetune.make_pairs import (
    Pair,
    copies_phrase,
    filter_candidates,
    generate,
    hard_negative,
    load_candidates,
    parse_questions,
    related,
    section_family,
    spread_sample,
    split,
)
from src.generate import LLMError
from src.retrieve import Retriever
from tests.helpers import FakeEmbedder, make_chunk, unit_embeddings


def test_parse_questions_strips_markers_and_keeps_real_questions():
    raw = '1. What is the cap on delay compensation?\n- "How long is the defect period?"\nNote: none\n2) Why?'
    assert parse_questions(raw, limit=5) == ["What is the cap on delay compensation?", "How long is the defect period?"]
    assert parse_questions(raw, limit=1) == ["What is the cap on delay compensation?"]


def test_copies_phrase_detects_a_long_run_of_passage_words():
    passage = "The contractor shall submit an irrevocable Performance Guarantee of 5% of the tendered amount."
    assert copies_phrase("Must the contractor submit an irrevocable Performance Guarantee of 5%?", passage)
    assert not copies_phrase("How big is the security a contractor has to give?", passage)


def test_section_family_ignores_part_labels():
    assert section_family(make_chunk("a", section="Table 2 (part 3/4)")) == "Table 2"
    assert section_family(make_chunk("b", section="Clause 10CC")) == "Clause 10CC"


T2_PART1 = make_chunk("t2-1", section="Table 2 (part 1/4)", parent="Table 2", covers=["Table 2"])
T2_PART3 = make_chunk("t2-3", section="Table 2 (part 3/4)", parent="Table 2", covers=["Table 2"])
T28 = make_chunk("t28", section="Table 28", covers=["Table 28"])
AMD_T28 = make_chunk("amd", section="Amendment No. 2, item 10", is_amendment=True, amends=["Table 28"])
CL_10CC_1 = make_chunk("10cc-1", section="Clause 10CC", doc_id="cpwd")
CL_10CC_2 = make_chunk("10cc-2", section="Clause 10CC", doc_id="cpwd")
CL_10CA = make_chunk("10ca", section="Clause 10CA", doc_id="cpwd")


def test_related_chunks_are_never_negatives_for_each_other():
    assert related(T2_PART1, T2_PART3)          # same table
    assert related(T28, AMD_T28)                # amendment link
    assert related(CL_10CC_1, CL_10CC_2)        # same clause, other window
    assert not related(CL_10CC_1, CL_10CA)      # neighbouring clause: a good near miss
    assert not related(T28, T2_PART1)


def test_hard_negative_skips_related_chunks():
    chunks = [CL_10CC_1, CL_10CC_2, CL_10CA]
    assert hard_negative(CL_10CC_1, [0, 1, 2], chunks) is CL_10CA
    assert hard_negative(CL_10CC_1, [0, 1], chunks) is None


def test_spread_sample_covers_the_whole_list():
    chunks = [make_chunk(f"c{i}") for i in range(10)]
    assert [c.chunk_id for c in spread_sample(chunks, 3)] == ["c0", "c3", "c6"]
    assert len(spread_sample(chunks, 50)) == 10


def test_split_is_deterministic_and_disjoint():
    pairs = [Pair(f"q{i}", "c", "n") for i in range(20)]
    train, val = split(pairs, fraction=0.1, seed=1)
    assert len(val) == 2 and len(train) == 18
    assert {p.query for p in train}.isdisjoint(p.query for p in val)
    assert split(pairs, fraction=0.1, seed=1) == (train, val)


class ScriptedLLM:
    name = "scripted"

    def __init__(self, fail_on: str = ""):
        self.fail_on = fail_on

    def complete(self, system, prompt):
        if self.fail_on and self.fail_on in prompt:
            raise LLMError("timeout")
        return "What is the first question here?\nWhat is the second question here?"


def test_generate_appends_per_chunk_and_resumes(tmp_path):
    path = tmp_path / "candidates.jsonl"
    chunks = [make_chunk("a", "alpha text"), make_chunk("b", "beta text")]
    generate(chunks, ScriptedLLM(fail_on="beta"), path=path)
    records = load_candidates(path)
    assert records[0]["questions"] == ["What is the first question here?", "What is the second question here?"]
    assert records[1]["error"] == "timeout"

    generate(chunks, ScriptedLLM(), path=path)  # Only the failed chunk is asked again.
    assert [r["chunk_id"] for r in load_candidates(path)] == ["a", "b", "b"]


def test_filter_candidates_applies_each_filter():
    passage = make_chunk("p", "delay compensation is one percent per month capped at ten percent", section="Clause 2")
    neighbour = make_chunk("n", "delay extension of time for hindrances", section="Clause 5")
    filler = [make_chunk(f"f{i}", f"unrelated words {i}", section=f"Clause {i + 10}") for i in range(4)]
    chunks = [passage, neighbour, *filler]
    retriever = Retriever(chunks, unit_embeddings(len(chunks)), FakeEmbedder())
    candidates = [{"chunk_id": "p", "questions": [
        "How much delay compensation can be charged?",                       # kept
        "How much delay compensation can be charged ?",                      # near-duplicate
        "What does the passage say about delay?",                            # mentions the passage
        "Is it one percent per month capped at ten percent?",                # copies a phrase
    ]}]
    # FakeEmbedder maps every text to the same vector, so any gold question looks identical (cosine 1):
    # with gold questions, the leak guard drops everything; without, it drops nothing.
    pairs, dropped = filter_candidates(candidates, retriever, gold_questions=[])
    assert [(p.query, p.chunk_id, p.negative_id) for p in pairs] == [("How much delay compensation can be charged?", "p", "n")]
    assert sorted(d.reason for d in dropped) == ["copies a phrase from its passage", "near-duplicate", "refers to 'the passage'"]

    pairs, dropped = filter_candidates(candidates, retriever, gold_questions=["Any gold question?"])
    assert pairs == [] and "too similar to a gold question" in {d.reason for d in dropped}


def test_candidates_file_is_json_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"chunk_id": "a", "questions": []}) + "\n\n", encoding="utf-8")
    assert load_candidates(path) == [{"chunk_id": "a", "questions": []}]
    assert load_candidates(tmp_path / "missing.jsonl") == []
