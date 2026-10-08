"""Curated data: transcriptions stay as printed; curator additions are applied separately."""

import pytest
import yaml

from src.config import CURATED_DIR
from src.ingest.curated import CURATED_FILE, PLACE_ALIASES_FILE, PlaceAliases, load

PLACES = PlaceAliases(apply_to=frozenset({"Appendix A"}), legend="[legend]", aliases={"Madras": "Chennai"})


def _data_lines(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]


def test_transcription_file_contains_no_curator_additions():
    data = "\n".join(_data_lines(CURATED_DIR / CURATED_FILE))
    assert "[now " not in data
    assert "curator" not in data.lower()


def test_annotate_adds_the_modern_name_and_the_legend():
    assert PLACES.annotate("Madras 50; Madurai 39.") == "Madras [now Chennai] 50; Madurai 39.\n[legend]"


def test_annotate_matches_whole_words_only_and_never_doubles():
    assert PLACES.annotate("Madrasi 50") == "Madrasi 50"
    assert PLACES.annotate("Madras [now Chennai] 50") == "Madras [now Chennai] 50"


def test_enrichment_applies_only_to_the_listed_entries():
    chunks = load(CURATED_DIR)
    appendix = " ".join(c.text for c in chunks if c.parent == "Appendix A")
    others = " ".join(c.text for c in chunks if c.parent != "Appendix A")
    assert "Madras [now Chennai] 50" in appendix
    assert "[now " not in others


def test_unknown_apply_to_section_is_an_error(tmp_path):
    (tmp_path / CURATED_FILE).write_text(yaml.safe_dump({"tables": [], "amendments": []}), encoding="utf-8")
    (tmp_path / PLACE_ALIASES_FILE).write_text(yaml.safe_dump({"apply_to": ["Apendix A"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown sections"):
        load(tmp_path)


def test_missing_alias_file_means_no_enrichment(tmp_path):
    entry = {"section": "Appendix A", "title": "t", "pdf_pages": [1], "printed_pages": ["1"], "text": "Madras 50"}
    (tmp_path / CURATED_FILE).write_text(yaml.safe_dump({"tables": [entry], "amendments": []}), encoding="utf-8")
    assert load(tmp_path)[0].text == "Madras 50"


# --- Amendment explanations ------------------------------------------------------------

# Curator wording that once sat inside amendment `text` fields (moved to `explanation`, 2026-10-08).
MOVED_CURATOR_PHRASES = ["is therefore", "then reads", "For open-ended cylinders", "caption correction",
                         "units of kinematic viscosity", "This replaces the original", "General notations showing"]


def test_amendment_text_fields_hold_only_printed_wording():
    data = yaml.safe_load((CURATED_DIR / CURATED_FILE).read_text(encoding="utf-8"))
    texts = " ".join(item["text"] for amendment in data["amendments"] for item in amendment["items"])
    for phrase in MOVED_CURATOR_PHRASES:
        assert phrase not in texts, phrase


def test_explanation_is_appended_and_marked_as_curator_text():
    from src.ingest.curated import amendment_text

    item = {"text": "Substitute '1.8' for '1.0'.", "explanation": "The value is therefore 1.8."}
    assert amendment_text(item) == "Substitute '1.8' for '1.0'. [Curator's explanation: The value is therefore 1.8.]"
    assert amendment_text({"text": "Delete 'closed'."}) == "Delete 'closed'."


def test_built_amendment_chunks_carry_the_marked_explanation():
    table_28_amendment = next(c for c in load(CURATED_DIR) if c.is_amendment and c.amends == ["Table 28"])
    assert "Substitute '1.8' for '1.0'. [Curator's explanation: " in table_28_amendment.text
