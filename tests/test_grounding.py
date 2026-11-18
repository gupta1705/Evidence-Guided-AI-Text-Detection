"""Explanation grounding: feature citations and invented numbers."""

import math

from src.features import compute_features
from src.grounding import (
    check_grounding,
    check_grounding_for_prompt,
    extract_numbers,
    matches_supplied,
    named_features,
    supplied_numbers_by_feature,
    supplied_numbers_from_prompt,
)
from src.prompts import build_data_block

REFERENCE = {
    "sentence_length_cv": {"human_median": 0.58, "human_q1": 0.44, "human_q3": 0.73, "ai_median": 0.39,
                           "ai_q1": 0.31, "ai_q3": 0.48, "scale": 0.3},
    "mattr": {"human_median": 0.81, "human_q1": 0.77, "human_q3": 0.85, "ai_median": 0.76, "ai_q1": 0.71,
              "ai_q3": 0.78, "scale": 0.08},
}
TEXT = ("In 1987 the committee met 45 times. It produced a long report. Members argued about every detail, and "
        "the chair kept careful notes.")


def _supplied():
    feats = compute_features(TEXT)
    return feats, supplied_numbers_from_prompt(build_data_block(TEXT, feats, REFERENCE))


def test_supplied_numbers_come_from_feature_section_only():
    feats, supplied = _supplied()
    assert 1987.0 not in supplied and 45.0 not in supplied  # numbers inside the assessed text
    assert float(feats["word_count"]) in supplied
    assert 0.58 in supplied and 0.48 in supplied
    assert 100.0 in supplied and 300.0 in supplied


def test_grounded_evidence():
    feats, supplied = _supplied()
    value = f"{feats['sentence_length_cv']:.4f}"
    evidence = [
        f"sentence_length_cv is {value}, closer to the AI reference median 0.3900 than the human median 0.5800.",
        f"The text has {feats['word_count']} words, below the 100-word minimum.",
    ]
    r = check_grounding(evidence, supplied)
    assert r.grounded_evidence_rate == 1.0 and not r.has_invented_number


def test_aliases_and_rounding_tolerance():
    feats, supplied = _supplied()
    rounded = round(feats["sentence_length_cv"], 2)
    r = check_grounding([f"Sentence length variation of {rounded} is closer to the AI median 0.39."], supplied)
    assert r.grounded_evidence_rate == 1.0
    assert named_features("The vocabulary diversity (moving-average type-token ratio) is high") == ["mattr"]
    assert "word_count" in named_features("Only 212 words were supplied")


def test_invented_numbers_detected():
    _, supplied = _supplied()
    r = check_grounding(
        ["mattr is 0.9100, far above the AI median.", "There is a 73% chance the text is AI-generated."],
        supplied,
        ["Results may vary by 12 points."],
    )
    assert r.grounded_evidence_rate == 0.0
    assert r.has_invented_number
    assert 0.91 in r.invented_numbers and 73.0 in r.invented_numbers and 12.0 in r.invented_numbers


def test_evidence_without_feature_name_is_not_grounded():
    _, supplied = _supplied()
    r = check_grounding(["The writing feels very polished.", "No human/AI reference ranges were supplied."], supplied)
    assert r.grounded_evidence_rate == 0.0 and not r.has_invented_number


def test_empty_evidence_rate_is_nan():
    r = check_grounding([], [1.0])
    assert math.isnan(r.grounded_evidence_rate) and not r.has_invented_number


def test_number_extraction_and_tolerance():
    assert extract_numbers("values 0.4123, 1,204 words, .41 and V4 then 100-word") == [0.4123, 1204.0, 0.41, 100.0]
    assert matches_supplied(0.41, [0.4123])  # abs diff 0.0023 <= 0.006
    assert matches_supplied(1210, [1204])  # rel diff 0.5% <= 1%
    assert not matches_supplied(0.43, [0.4123])


def test_numbers_glued_to_letters_are_ignored_whole():
    """No backtracking to the integer part of a decimal that is glued to letters ('5.8335bits' is not 5)."""
    text = "mattr is 0.8119, closer to 0.8088; 5.8335bits; 1,204 words; -0.25; 3.5x; V4; 20%; .5x"
    assert extract_numbers(text) == [0.8119, 0.8088, 1204.0, 0.25, 20.0]


def _prompt():
    feats = compute_features(TEXT)
    return feats, build_data_block(TEXT, feats, REFERENCE)


def test_supplied_numbers_by_feature_line():
    feats, prompt = _prompt()
    lines = supplied_numbers_by_feature(prompt)
    assert lines["word_count"] == [float(feats["word_count"])]
    assert lines["mattr"][1:] == [0.81, 0.77, 0.85, 0.76, 0.71, 0.78]


def test_strict_grounding_needs_the_named_features_own_numbers():
    feats, prompt = _prompt()
    value = f"{feats['mattr']:.4f}"
    evidence = [
        f"mattr is {value}, closer to the human reference median 0.8100.",  # strictly grounded
        "Vocabulary diversity is typical of AI text.",  # names a feature but quotes nothing
        "mattr is 0.5800, closer to the AI reference median.",  # 0.58 is sentence_length_cv's human median
        f"The text has {feats['word_count']} words, below the 100-word minimum.",  # count + constant
    ]
    r = check_grounding_for_prompt(evidence, [], prompt)
    assert [i["grounded"] for i in r.items] == [True, True, True, True]  # project definition
    assert [i["strictly_grounded"] for i in r.items] == [True, False, False, True]
    assert r.grounded_evidence_rate == 1.0 and r.strict_grounded_rate == 0.5
    assert not r.has_invented_number  # a misattributed number is still a supplied number
    # Without per-feature numbers the strict rate is undefined.
    assert math.isnan(check_grounding(evidence, supplied_numbers_from_prompt(prompt)).strict_grounded_rate)
