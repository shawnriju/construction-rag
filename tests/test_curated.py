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
