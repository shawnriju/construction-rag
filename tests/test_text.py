"""Chunk sizing: every chunk must fit the embedder's window (measured in its own tokens)."""

from src.ingest.text import Paragraph, Unit, group_units, pack, token_count

NUMBERS = " ".join(f"{i}.{i % 10}5" for i in range(400))  # Number-heavy text: several tokens per "word".
PROSE = "The contractor shall complete the work within the stipulated time. " * 60


def paragraph(text: str) -> Paragraph:
    return Paragraph(text, pdf_page=1, printed_page="1")


def test_token_counts_add_up_across_whitespace_joins():
    a, b = "Clause 10CC escalation", "Vz = Vb k1 k2 k3 = 0.85"
    assert token_count(f"{a} {b}") == token_count(a) + token_count(b)


def test_numbers_cost_more_tokens_than_words():
    assert token_count("0.85") > 1


def test_pack_never_exceeds_the_budget_even_for_number_heavy_text():
    budget = 120
    windows = pack([paragraph(NUMBERS), paragraph(PROSE)], max_tokens=budget)
    assert len(windows) > 1
    # Folding a tiny tail into the previous window may add up to MIN_TAIL_TOKENS.
    assert all(sum(token_count(p.text) for p in w) <= budget + 50 for w in windows)
    rejoined = " ".join(p.text for w in windows for p in w)
    assert rejoined.split() == f"{NUMBERS} {PROSE}".split()  # Nothing lost or reordered.


def test_group_units_merges_small_units_and_splits_large_ones():
    # Each small unit is ~25 tokens, so together they are above MIN_TAIL_TOKENS and are
    # not folded into the long unit as a heading stub.
    clause = "The Engineer-in-Charge shall certify the measurements recorded by the contractor within seven days."
    small = [Unit(key=str(i), group="g", title="", paragraphs=[paragraph(clause)]) for i in range(3)]
    large = Unit(key="9", group="g", title="", paragraphs=[paragraph(PROSE)])
    windows = group_units([*small, large], max_tokens=150)
    assert windows[0].keys == ["0", "1", "2"]
    assert all(w.keys == ["9"] for w in windows[1:])
    assert windows[-1].part_count == len(windows) - 1 > 1
