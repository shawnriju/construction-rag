from src.cite import NOT_FOUND, check_citations, numbers_in, source_numbers, split_sentences

FOREWORD = "[S1] IS 875, Cl. 0.3.2, p. 3\nThe foreword explains the revision of the code."
PRESSURE = "[S2] IS 875, Cl. 5.4, p. 13\nDesign Wind Pressure - pz = 0'6 Vz2 where pz is in N/m2."


def test_source_numbers_reads_all_marker_styles():
    assert source_numbers("a [S1][S3, S4] b [s2]") == [1, 3, 4, 2]


def test_abbreviations_do_not_end_a_sentence():
    text = "See Clause 2, p. 15 (PDF 17) and Cl. 5.3, i.e. the main rule, for the rate [S1]."
    assert split_sentences(text) == [text]


def test_marker_after_the_full_stop_belongs_to_the_previous_sentence():
    text = "The force coefficient for flat-sided members is 1.8. [S2] Linear interpolation is permitted here [S1]."
    assert split_sentences(text) == [
        "The force coefficient for flat-sided members is 1.8. [S2]",
        "Linear interpolation is permitted here [S1].",
    ]


def test_leading_marker_stays_when_the_previous_sentence_is_already_cited():
    sentences = split_sentences("The rate is 1% per month [S1]. [S2] gives the overall cap.")
    assert sentences[1].startswith("[S2]")


def test_uncited_long_sentences_are_reported():
    report = check_citations("The compensation is capped at ten percent of the tendered value.", 3)
    assert report.uncited_sentences == ["The compensation is capped at ten percent of the tendered value."]
    assert not report.ok


def test_short_connective_lines_need_no_citation():
    assert check_citations("In summary:\nThe rate is 1% per month of delay [S1].", 1).ok


def test_markers_to_missing_sources_are_invalid():
    report = check_citations("The value is 1.8 for flat-sided members [S2][S9].", 3)
    assert report.cited == [2]
    assert report.invalid == [9]


def test_not_found_answer_needs_no_citations():
    assert check_citations(NOT_FOUND, 6).ok


# --- Number grounding -------------------------------------------------------------


def test_numbers_are_normalised_so_equal_values_match():
    assert numbers_in("0'6, 1.00, 1,000 and 0.80") == {"0.6", "1", "1000", "0.8"}


def test_number_words_match_digits():
    # Real false alarm: the source says "twelve months", the answer "12 months".
    source = "[S1] CPWD GCC 2020, Clause 17\nwithin twelve months (six months for small works)"
    answer = "The contractor is liable for 12 months, or 6 months for small works [S1]."
    assert check_citations(answer, 1, [source]).ungrounded_numbers == []


def test_number_cited_to_a_source_without_it_is_flagged():
    # The real failure: the formula was cited to the foreword, which has no 0.6.
    answer = "The design wind pressure is pz = 0.6 times the velocity squared [S1]."
    report = check_citations(answer, 2, [FOREWORD, PRESSURE])
    assert [number for number, _ in report.ungrounded_numbers] == ["0.6"]
    assert not report.ok


def test_number_cited_to_the_right_source_passes_even_in_ocr_form():
    answer = "The design wind pressure is pz = 0.6 times the velocity squared [S2]."
    assert check_citations(answer, 2, [FOREWORD, PRESSURE]).ungrounded_numbers == []


def test_marker_digits_and_numbers_from_the_question_are_not_flagged():
    answer = "For a solidity ratio of 0.2 the code gives the pressure formula [S2]."
    report = check_citations(answer, 2, [FOREWORD, PRESSURE], question="What applies at solidity ratio 0.2?")
    assert report.ungrounded_numbers == []


def test_number_check_is_skipped_without_source_texts_or_markers():
    assert check_citations("The value is 9.9 for all members here [S1].", 1).ungrounded_numbers == []
    report = check_citations("The value is 9.9 for all flat-sided members.", 1, [FOREWORD])
    assert report.ungrounded_numbers == [] and report.uncited_sentences
