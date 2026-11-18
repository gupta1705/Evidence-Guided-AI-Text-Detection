"""Typo perturbations, deterministic variant seeds, prompt injection placement and style changes."""

import random
import zlib

import pytest

from src.config import INJECTION_TEXT
from src.perturb import (
    STYLE_OPS,
    TYPO_OPS,
    inject_instruction,
    perturb,
    perturb_visible,
    perturb_word,
    style_change,
    typo_variant,
    variant_seed,
)
from src.text_utils import count_words, truncate_words

TEXT = "The quick brown fox jumps over the lazy dog while the farmer watches carefully from the barn."


@pytest.mark.parametrize("op", TYPO_OPS)
def test_perturb_word_keeps_first_letter(op):
    rng = random.Random(3)
    out = perturb_word("keyboard", op, rng)
    assert out[0] == "k" and out != "" and abs(len(out) - len("keyboard")) <= 1
    with pytest.raises(ValueError):
        perturb_word("keyboard", "bogus", rng)


def test_perturb_is_deterministic_and_edits_n_words():
    a = perturb(TEXT, "swap", n_chars=2, seed=11)
    assert a == perturb(TEXT, "swap", n_chars=2, seed=11)
    changed = [x for x, y in zip(TEXT.split(), a.split()) if x != y]
    assert 1 <= len(changed) <= 2  # a swap of two equal letters can be a no-op
    assert perturb("a an to", "drop", n_chars=2, seed=1) == "a an to"  # no eligible words


def test_variant_seed_is_stable_crc32():
    assert variant_seed("q1_h0", "swap", 2, 0) == zlib.crc32(b"q1_h0|swap|2|0")
    assert variant_seed("q1_h0", "swap", 2, 0) != variant_seed("q1_h0", "swap", 2, 1)


def test_typo_variant_cycles_operations():
    ops = [typo_variant(TEXT, "id", j)[1] for j in range(5)]
    assert ops == ["swap", "drop", "key", "add", "swap"]


def test_perturbation_stays_in_visible_window():
    head = " ".join(["alpha"] * 300)
    tail = "omega " * 50
    text = head + " " + tail.strip()
    out = perturb_visible(text, "add", n_chars=2, seed=5, max_words=300)
    assert out.endswith(tail.strip())
    assert truncate_words(out, 300) != head


def test_injection_short_text_appended():
    out = inject_instruction(TEXT)
    assert out == TEXT + " " + INJECTION_TEXT


def test_injection_long_text_inside_first_300_words():
    text = " ".join(f"w{chr(97 + i % 26)}" for i in range(400))
    out = inject_instruction(text)
    visible = truncate_words(out, 300)
    assert INJECTION_TEXT in visible
    assert count_words(out) == count_words(text) + count_words(INJECTION_TEXT)


@pytest.mark.parametrize("op", STYLE_OPS)
def test_style_changes(op):
    text = "It is clear that we do not know. Cannot stop, will not stop."
    out = style_change(text, op)
    expected = {
        "contract": "It's clear that we don't know. Can't stop, won't stop.",
        "expand": text,
        "lowercase": text.lower(),
        "no_commas": "It is clear that we do not know. Cannot stop will not stop.",
    }[op]
    assert out == expected
    assert style_change("It's here and we don't care.", "expand") == "It is here and we do not care."
