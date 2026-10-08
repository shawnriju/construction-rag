"""Amendment and sibling expansion: the parts of retrieval the answers depend on."""

from src.config import MAX_SIBLINGS_ATTACHED
from src.retrieve import Retriever, tokenize
from tests.helpers import FakeEmbedder, make_chunk, unit_embeddings

TABLE_28 = make_chunk("t28", "force coefficient single frame flat sided", section="Table 28", covers=["Table 28"])
AMD_ITEM_10 = make_chunk(
    "amd2-10", "substitute 1.8 for 1.0", section="Amendment No. 2, item 10",
    is_amendment=True, amends=["Table 28"], printed_pages=[],
)
# Each part has one distinctive word, so a query can target exactly one part.
TABLE_2_PARTS = [
    make_chunk(f"t2-{i}", f"k2 factor height rows {word}", section=f"Table 2 (part {i}/4)",
               covers=["Table 2"], parent="Table 2", source="curated")
    for i, word in enumerate(["low", "middle", "tall", "highest"], start=1)
]
FILLER = [make_chunk(f"f{i}", f"unrelated clause number {i}") for i in range(6)]


# The fake embedder gives every chunk the same vector, so ranking tests use BM25
# alone; expansion runs after ranking and is the same in every mode.
def retriever(chunks):
    return Retriever(chunks, unit_embeddings(len(chunks)), FakeEmbedder())


def ids(result):
    return [h.chunk.chunk_id for h in result.hits]


def test_tokenize_keeps_clause_ids_intact():
    assert tokenize("Clause 10 CC and Cl. 6.2.2.8, see C-2.1") == ["clause", "10cc", "cl", "6.2.2.8", "see", "c-2.1"]


def test_tokenize_does_not_glue_english_words_to_numbers():
    # Regression: "Section 34 of the Act" used to become "34of", so a query for "Section 34" missed it.
    assert tokenize("Section 34 of the Act, Article 142 in Ssangyong, Tables 2 to 5") == [
        "section", "34", "act", "article", "142", "ssangyong", "tables", "2", "5",
    ]
    assert tokenize("Clause 19 A and a 10 m height") == ["clause", "19a", "10m", "height"]


def test_retrieved_original_brings_its_amendment():
    result = retriever([TABLE_28, AMD_ITEM_10, *FILLER]).search("force coefficient flat sided", top_k=1, mode="bm25")
    assert ids(result) == ["t28", "amd2-10"]
    assert "amends Table 28" in result.hits[1].attached_reason


def test_retrieved_amendment_brings_the_original():
    result = retriever([TABLE_28, AMD_ITEM_10, *FILLER]).search("substitute 1.8", top_k=1, mode="bm25")
    assert ids(result) == ["amd2-10", "t28"]


def test_expansion_can_be_switched_off_for_ablation():
    result = retriever([TABLE_28, AMD_ITEM_10, *FILLER]).search("force coefficient flat sided", top_k=1, mode="bm25", expand=False)
    assert ids(result) == ["t28"]


def test_one_table_part_brings_the_others_in_document_order():
    result = retriever([*TABLE_2_PARTS, *FILLER]).search("tall", top_k=1, mode="bm25")
    assert ids(result) == ["t2-3", "t2-1", "t2-2", "t2-4"]
    assert all("rest of Table 2" in h.attached_reason for h in result.hits[1:])


def test_sibling_parts_are_not_duplicated_and_respect_the_cap():
    result = retriever([*TABLE_2_PARTS, *FILLER]).search("k2 factor height rows", top_k=4, mode="bm25")
    assert sorted(ids(result)) == sorted(c.chunk_id for c in TABLE_2_PARTS)
    attached = [h for h in result.hits if h.attached_reason]
    assert len(attached) <= MAX_SIBLINGS_ATTACHED


def test_siblings_can_be_switched_off():
    result = retriever([*TABLE_2_PARTS, *FILLER]).search("tall", top_k=1, mode="bm25", siblings=False)
    assert ids(result) == ["t2-3"]


# --- Identifier pinning ------------------------------------------------------------------
# Dense mode with the fake embedder ranks chunks in list order, so the chunk holding the
# identifier (last in the list) is never in the top 1 on its own merit.

RULING = make_chunk("ruling", "invoking our power under Article 142 we uphold the minority award")
COMMON = [make_chunk(f"t{i}", f"Table 28 row {i}") for i in range(4)]  # "28" in 4 chunks: too common to pin.


def test_rare_identifier_from_the_question_is_pinned():
    result = retriever([*FILLER, RULING]).search("What relief was granted under Article 142?", top_k=1, mode="dense")
    assert ids(result) == ["f0", "ruling"]
    assert "'142'" in result.hits[1].attached_reason


def test_common_identifier_is_not_pinned():
    result = retriever([*FILLER, *COMMON]).search("What does Table 28 say?", top_k=1, mode="dense")
    assert ids(result) == ["f0"]


def test_pinning_can_be_switched_off():
    result = retriever([*FILLER, RULING]).search("Article 142", top_k=1, mode="dense", pin=False)
    assert ids(result) == ["f0"]
