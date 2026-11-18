"""Robustness perturbations: character typos (Pruthi et al. 2019), style changes and prompt injection.

The typo code is the shared, already-tested implementation from the project specification. Variant seeds
come from a stable CRC32 hash (never Python's salted ``hash()``), so every run perturbs the same words.
Perturbations are applied inside the analysed window (first 300 words) so that the model actually sees them.
"""

from __future__ import annotations

import random
import re
import string
import zlib

from src.config import INJECTION_TEXT, MAX_WORDS
from src.text_utils import count_words, split_at_word_limit, truncate_words

TYPO_OPS: tuple[str, ...] = ("swap", "drop", "key", "add")
STYLE_OPS: tuple[str, ...] = ("contract", "expand", "lowercase", "no_commas")

KEYBOARD = {  # neighbouring keys on a QWERTY keyboard
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij",
    "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz", "d": "sefx", "f": "drgc",
    "g": "fthv", "h": "gyjb", "j": "hukn", "k": "jilm", "l": "kop", "z": "asx", "x": "zsdc",
    "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def perturb_word(word: str, op: str, rng: random.Random) -> str:
    """Apply one character-level edit to an alphabetic word of length >= 3."""
    i = rng.randrange(1, len(word) - 1)            # keep the first letter unchanged
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "key":
        c = word[i].lower()
        return word[:i] + rng.choice(KEYBOARD.get(c, c)) + word[i + 1:]
    if op == "add":
        return word[:i] + rng.choice(string.ascii_lowercase) + word[i:]
    raise ValueError(f"unknown operation: {op}")


def perturb(text: str, op: str, n_chars: int = 1, seed: int = 0) -> str:
    """Return `text` with `n_chars` randomly chosen words edited once each."""
    rng = random.Random(seed)
    tokens = re.split(r"(\W+)", text)               # keep spaces and punctuation
    eligible = [k for k, t in enumerate(tokens) if t.isalpha() and len(t) >= 3]
    for k in rng.sample(eligible, min(n_chars, len(eligible))):
        tokens[k] = perturb_word(tokens[k], op, rng)
    return "".join(tokens)


def variant_seed(text_id: str, op: str, level: int, j: int = 0) -> int:
    """Deterministic seed for one perturbed variant (CRC32 of ``"{text_id}|{op}|{level}|{j}"``)."""
    return zlib.crc32(f"{text_id}|{op}|{level}|{j}".encode("utf-8"))


def perturb_visible(text: str, op: str, n_chars: int, seed: int, max_words: int = MAX_WORDS) -> str:
    """Apply :func:`perturb` to the analysed prefix (first ``max_words`` words) and keep the remainder."""
    head, rest = split_at_word_limit(text, max_words)
    head = perturb(head, op, n_chars=n_chars, seed=seed)
    return f"{head} {rest}".strip()


def typo_variant(text: str, text_id: str, j: int, n_chars: int = 2, max_words: int = MAX_WORDS) -> tuple[str, str]:
    """The ``j``-th text of the typo set: ``n_chars`` edits with the operation cycled by ``j``.

    Returns:
        ``(perturbed_text, op)``.
    """
    op = TYPO_OPS[j % len(TYPO_OPS)]
    return perturb_visible(text, op, n_chars, variant_seed(text_id, op, n_chars, j), max_words), op


def inject_instruction(text: str, payload: str = INJECTION_TEXT, max_words: int = MAX_WORDS) -> str:
    """Append ``payload`` inside the text so that it stays within the analysed first ``max_words`` words.

    Short texts get the payload at the end. For longer texts the payload is inserted right after the first
    ``max_words - len(payload)`` words, so the model still sees it.
    """
    k = count_words(payload)
    if count_words(text) + k <= max_words:
        return f"{text.rstrip()} {payload}".strip()
    head = truncate_words(text, max(0, max_words - k))
    rest = text[len(head):].strip() if text.startswith(head) else ""
    return f"{head} {payload} {rest}".strip()


_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"), ("is not", "isn't"),
    ("are not", "aren't"), ("was not", "wasn't"), ("were not", "weren't"), ("cannot", "can't"),
    ("will not", "won't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
    ("could not", "couldn't"), ("have not", "haven't"), ("has not", "hasn't"), ("it is", "it's"),
    ("that is", "that's"), ("there is", "there's"), ("i am", "i'm"), ("you are", "you're"),
    ("they are", "they're"), ("we are", "we're"),
)


def _replace_phrases(text: str, pairs: tuple[tuple[str, str], ...]) -> str:
    """Case-insensitive whole-word phrase replacement that keeps a leading capital letter."""
    for src, dst in pairs:
        pattern = re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE)

        def repl(m: re.Match, dst: str = dst) -> str:
            out = dst
            if m.group()[0].isupper():
                out = out[0].upper() + out[1:]
            return out

        text = pattern.sub(repl, text)
    return text


def style_change(text: str, op: str) -> str:
    """Deterministic style rewrite.

    * ``contract``  - "do not" -> "don't", "it is" -> "it's", ...
    * ``expand``    - the reverse.
    * ``lowercase`` - lower-case everything.
    * ``no_commas`` - remove all commas.
    """
    if op == "contract":
        return _replace_phrases(text, _CONTRACTIONS)
    if op == "expand":
        return _replace_phrases(text, tuple((b, a) for a, b in _CONTRACTIONS))
    if op == "lowercase":
        return text.lower()
    if op == "no_commas":
        return re.sub(r"\s*,", "", text)
    raise ValueError(f"unknown style operation: {op}")


def style_variant(text: str, j: int) -> tuple[str, str]:
    """The ``j``-th text of the style set (operation cycled by ``j``). Returns ``(text, op)``."""
    op = STYLE_OPS[j % len(STYLE_OPS)]
    return style_change(text, op), op
