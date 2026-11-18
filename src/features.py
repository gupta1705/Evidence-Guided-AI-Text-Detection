"""The four statistical features named in the project form, computed in plain Python.

No neural model is involved. Every feature is computed on the normalised text truncated to its first
300 words (length control); ``word_count`` and ``sentence_count`` describe the full text.

Features
--------
1. ``sentence_length_cv``   - sentence-length variation: population std / mean of sentence lengths in words.
2. ``mattr``                - moving-average type-token ratio over windows of 50 words.
3. ``repeated_trigram_rate`` - share of repeated word sequences: ``1 - unique trigrams / total trigrams``.
4. ``word_entropy``         - empirical word-frequency (unigram) Shannon entropy in bits, averaged over
   consecutive non-overlapping 100-word windows.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from src.config import ENTROPY_WINDOW, MATTR_WINDOW, MAX_WORDS
from src.text_utils import normalize_text, split_sentences, tokenize, truncate_words


@dataclass(frozen=True)
class FeatureInfo:
    """Metadata for one feature, shared by the prompts, grounding checks and the app."""

    name: str
    label: str
    description: str
    unit: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


FEATURES: dict[str, FeatureInfo] = {
    "sentence_length_cv": FeatureInfo(
        name="sentence_length_cv",
        label="Sentence-length variation",
        description="Coefficient of variation (population std / mean) of sentence lengths in words.",
        unit="ratio (std/mean)",
        aliases=(
            "sentence length variation",
            "sentence-length variation",
            "variation in sentence length",
            "sentence length variability",
            "sentence length cv",
            "sentence-length cv",
            "coefficient of variation",
            "burstiness",
        ),
    ),
    "mattr": FeatureInfo(
        name="mattr",
        label="Vocabulary diversity (MATTR)",
        description="Moving-average type-token ratio over sliding 50-word windows "
        "(plain type-token ratio for texts under 50 words).",
        unit="ratio (0-1)",
        aliases=(
            "moving-average type-token ratio",
            "moving average type-token ratio",
            "moving average type token ratio",
            "type-token ratio",
            "type token ratio",
            "vocabulary diversity",
            "lexical diversity",
        ),
    ),
    "repeated_trigram_rate": FeatureInfo(
        name="repeated_trigram_rate",
        label="Repeated word sequences",
        description="Share of word trigrams that repeat an earlier trigram: 1 - unique/total trigrams.",
        unit="ratio (0-1)",
        aliases=(
            "repeated trigram rate",
            "repeated trigrams",
            "repeated word sequences",
            "repeated word sequence",
            "trigram repetition",
            "repetition rate",
        ),
    ),
    "word_entropy": FeatureInfo(
        name="word_entropy",
        label="Empirical word-frequency entropy",
        description="Shannon entropy (bits) of the word-frequency distribution, averaged over consecutive "
        "100-word windows. Empirical frequency entropy, not GPT-2 predictive entropy.",
        unit="bits",
        aliases=(
            "word entropy",
            "word-frequency entropy",
            "word frequency entropy",
            "empirical entropy",
            "unigram entropy",
            "frequency entropy",
            "entropy",
        ),
    ),
}

FEATURE_NAMES: list[str] = list(FEATURES)

# Descriptive counts that also appear in the prompt's feature block (not used by the baseline score).
COUNT_INFO: dict[str, FeatureInfo] = {
    "word_count": FeatureInfo(
        name="word_count",
        label="Word count",
        description="Number of tokenizer words in the full normalised text.",
        unit="words",
        aliases=("word count", "number of words", "length of the text", "text length"),
    ),
    "sentence_count": FeatureInfo(
        name="sentence_count",
        label="Sentence count",
        description="Number of sentences (with at least one word) in the full normalised text.",
        unit="sentences",
        aliases=("sentence count", "number of sentences"),
    ),
}


def sentence_length_cv(sentence_lengths: list[int]) -> float:
    """Population std / mean of sentence lengths; NaN with fewer than two sentences or zero mean."""
    if len(sentence_lengths) < 2:
        return math.nan
    mean = sum(sentence_lengths) / len(sentence_lengths)
    if mean == 0:
        return math.nan
    var = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)
    return math.sqrt(var) / mean


def mattr(words: list[str], window: int = MATTR_WINDOW) -> tuple[float, bool]:
    """Moving-average type-token ratio.

    Returns ``(value, short_window)``. With at least ``window`` words the value is the mean TTR over all
    ``len(words) - window + 1`` sliding windows; with fewer words it is the plain TTR of all words and
    ``short_window`` is True. An empty word list gives NaN.
    """
    n = len(words)
    if n == 0:
        return math.nan, True
    if n < window:
        return len(set(words)) / n, True
    counts = Counter(words[:window])
    total_types = len(counts)
    for i in range(window, n):
        out_word, in_word = words[i - window], words[i]
        counts[out_word] -= 1
        if counts[out_word] == 0:
            del counts[out_word]
        counts[in_word] += 1
        total_types += len(counts)
    n_windows = n - window + 1
    return total_types / (n_windows * window), False


def repeated_trigram_rate(words: list[str]) -> float:
    """``1 - unique_trigrams / total_trigrams``; 0.0 when there are fewer than three words."""
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1.0 - len(set(trigrams)) / len(trigrams)


def shannon_entropy(words: list[str]) -> float:
    """Shannon entropy in bits of the empirical word-frequency distribution of ``words``."""
    n = len(words)
    if n == 0:
        return math.nan
    return -sum((c / n) * math.log2(c / n) for c in Counter(words).values())


def word_entropy(words: list[str], window: int = ENTROPY_WINDOW) -> tuple[float, bool]:
    """Empirical word-frequency (unigram) entropy in bits, averaged over 100-word windows.

    This is the entropy of the observed word frequencies inside each window - a measure of how evenly the
    vocabulary is used. It is empirical frequency entropy, not GPT-2 predictive (per-token surprisal) entropy;
    no language model is used.

    Returns ``(value, length_controlled)``. The text is cut into consecutive non-overlapping windows of
    ``window`` words (a shorter remainder is ignored) and the window entropies are averaged, which makes
    texts of different lengths comparable. With fewer than ``window`` words the entropy of all words is
    returned and ``length_controlled`` is False.
    """
    n_windows = len(words) // window
    if n_windows == 0:
        return shannon_entropy(words), False
    values = [shannon_entropy(words[k * window:(k + 1) * window]) for k in range(n_windows)]
    return sum(values) / n_windows, True


def compute_features(text: str, max_words: int = MAX_WORDS) -> dict:
    """Compute the four features and descriptive counts for ``text``.

    The text is normalised, then the features are computed on its first ``max_words`` words. Values are
    returned at full precision; the prompt builder shows them with 4 decimals.

    Returns:
        A dict with keys ``word_count`` and ``sentence_count`` (full text), ``analysed_words`` (words used
        for the features), the four features (float, NaN when undefined), and the flags ``truncated``,
        ``short_window`` (MATTR fell back to plain TTR) and ``length_controlled`` (entropy used 100-word
        windows).
    """
    norm = normalize_text(text)
    full_words = tokenize(norm)
    full_sentences = [s for s in split_sentences(norm) if tokenize(s)]

    head = truncate_words(norm, max_words)
    words = tokenize(head)
    sentence_lengths = [len(tokenize(s)) for s in split_sentences(head)]
    sentence_lengths = [n for n in sentence_lengths if n > 0]

    mattr_value, short_window = mattr(words)
    entropy_value, length_controlled = word_entropy(words)
    return {
        "word_count": len(full_words),
        "sentence_count": len(full_sentences),
        "analysed_words": len(words),
        "sentence_length_cv": sentence_length_cv(sentence_lengths),
        "mattr": mattr_value,
        "repeated_trigram_rate": repeated_trigram_rate(words),
        "word_entropy": entropy_value,
        "truncated": len(words) < len(full_words),
        "short_window": short_window,
        "length_controlled": length_controlled,
    }



def valid_feature_count(features: dict) -> int:
    """Number of the four features that are defined (not NaN / missing)."""
    return sum(1 for name in FEATURE_NAMES if is_number(features.get(name)))


def is_number(value: object) -> bool:
    """True for a finite int/float (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
