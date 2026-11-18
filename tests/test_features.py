"""Each feature on hand-computed examples."""

import math
import random

import pytest

from src.features import (
    FEATURE_NAMES,
    FEATURES,
    compute_features,
    mattr,
    repeated_trigram_rate,
    sentence_length_cv,
    shannon_entropy,
    valid_feature_count,
    word_entropy,
)


def test_feature_metadata_complete():
    assert FEATURE_NAMES == ["sentence_length_cv", "mattr", "repeated_trigram_rate", "word_entropy"]
    for name, info in FEATURES.items():
        assert info.name == name and info.label and info.description and info.unit
    assert "not GPT-2" in FEATURES["word_entropy"].description
    assert "not GPT-2 predictive" in word_entropy.__doc__


def test_sentence_length_cv_population_std():
    # lengths 3 and 6: mean 4.5, population std 1.5 -> 1/3
    assert sentence_length_cv([3, 6]) == pytest.approx(1 / 3)
    assert sentence_length_cv([4, 4, 4]) == 0.0
    assert math.isnan(sentence_length_cv([7]))
    assert math.isnan(sentence_length_cv([]))


def test_mattr_short_text_is_plain_ttr():
    value, short = mattr(["a", "b", "a", "b"])
    assert value == 0.5 and short is True
    value, short = mattr([])
    assert math.isnan(value) and short is True


def test_mattr_sliding_windows_hand_computed():
    words = ["a"] * 50 + ["b"]  # windows: 1 type, then 2 types -> (1 + 2) / 2 / 50
    value, short = mattr(words)
    assert value == pytest.approx(0.03) and short is False
    distinct = [f"w{i}" for i in range(50)] + ["w0"]
    assert mattr(distinct)[0] == pytest.approx(1.0)


def test_mattr_matches_brute_force():
    rng = random.Random(0)
    words = [rng.choice("abcdefghijklmnopqrstuvwxyz") * rng.randint(1, 2) for _ in range(173)]
    brute = sum(len(set(words[i:i + 50])) / 50 for i in range(len(words) - 49)) / (len(words) - 49)
    assert mattr(words)[0] == pytest.approx(brute)


def test_repeated_trigram_rate():
    # trigrams: (a,b,c) (b,c,a) (c,a,b) (a,b,c) -> 1 - 3/4
    assert repeated_trigram_rate(list("abcabc")) == pytest.approx(0.25)
    assert repeated_trigram_rate(["a", "b"]) == 0.0
    assert repeated_trigram_rate(["a", "b", "c"]) == 0.0


def test_word_entropy_windows():
    value, controlled = word_entropy(["a"] * 50 + ["b"] * 50)
    assert value == pytest.approx(1.0) and controlled is True
    words = ["a"] * 100 + ["p", "q", "r", "s"] * 25  # windows: 0 bits and 2 bits -> mean 1
    assert word_entropy(words)[0] == pytest.approx(1.0)
    words_with_remainder = words + ["z"] * 30  # remainder (< 100 words) is ignored
    assert word_entropy(words_with_remainder)[0] == pytest.approx(1.0)


def test_word_entropy_short_text_not_length_controlled():
    value, controlled = word_entropy(["a", "b", "c", "d"])
    assert value == pytest.approx(2.0) and controlled is False
    assert shannon_entropy(["x", "x"]) == 0.0
    assert math.isnan(shannon_entropy([]))


def test_compute_features_small_text():
    feats = compute_features("The cat sat. The dog ran far away!")
    assert feats["word_count"] == 8
    assert feats["sentence_count"] == 2
    assert feats["sentence_length_cv"] == pytest.approx(0.25)  # lengths 3 and 5
    assert feats["mattr"] == pytest.approx(7 / 8)  # plain TTR: 'the' repeats
    assert feats["repeated_trigram_rate"] == 0.0
    assert feats["short_window"] is True and feats["length_controlled"] is False
    assert feats["truncated"] is False


def test_compute_features_truncates_to_300_words_but_counts_full_text():
    text = " ".join(f"word{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}." for i in range(350))
    feats = compute_features(text)
    assert feats["word_count"] == 350
    assert feats["analysed_words"] == 300
    assert feats["sentence_count"] == 350
    assert feats["truncated"] is True
    assert feats["length_controlled"] is True


def test_compute_features_normalises_first():
    raw = "I do n't know . It 's fine , really ."
    assert compute_features(raw) == compute_features("I don't know. It's fine, really.")


def test_valid_feature_count():
    feats = compute_features("Only one sentence here without an end")
    assert math.isnan(feats["sentence_length_cv"])
    assert valid_feature_count(feats) == 3
