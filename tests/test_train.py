"""Training-data helpers (the training loop itself needs the GPU venv and is not unit-tested)."""

import json

import pytest

from finetune.make_pairs import Pair
from finetune.train import load_pairs, retrieval_eval_inputs, training_columns
from src.config import BGE_QUERY_PREFIX
from tests.helpers import make_chunk

CHUNKS = {c.chunk_id: c for c in (make_chunk("p", "delay compensation"), make_chunk("n", "extension of time"))}


def test_training_columns_match_how_search_embeds_text():
    columns = training_columns([Pair("How much for delay?", "p", "n")], CHUNKS)
    assert columns["anchor"] == [BGE_QUERY_PREFIX + "How much for delay?"]
    assert columns["positive"] == [CHUNKS["p"].index_text]   # breadcrumb + body, as in the index
    assert columns["negative"] == [CHUNKS["n"].index_text]


def test_validation_inputs_search_the_whole_corpus():
    queries, corpus, relevant = retrieval_eval_inputs([Pair("How much for delay?", "p", "n")], CHUNKS)
    assert set(corpus) == {"p", "n"} and relevant == {"q0": {"p"}}
    assert queries["q0"].startswith(BGE_QUERY_PREFIX)


def test_load_pairs_rejects_unknown_chunk_ids(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"query": "Q?", "chunk_id": "p", "negative_id": "gone"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gone"):
        load_pairs(path, CHUNKS)
