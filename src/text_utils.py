"""Text normalisation, the fixed word tokenizer, the sentence splitter and length control.

HC3 human answers are pre-tokenised ("word ." / "do n't") while ChatGPT answers are not, and ChatGPT
answers contain line breaks far more often - sometimes as real line breaks, sometimes as the literal
escape ``\\n`` and sometimes removed altogether ("situations.One"). A detector (or an LLM reading the text)
would learn those formatting artefacts instead of writing style, so every text (both classes; reference,
validation and test data; app input) passes through :func:`normalize_text` before anything else is
computed. Rules 1-8 are the shared project rules; rules 1b and 9-12 were added after measuring artefacts
that the shared rules leave behind (see ``NORMALISATION_RULES`` and ``data/summary.json``).
"""

from __future__ import annotations

import re

from src.config import MAX_WORDS

# Rule 1: control whitespace -> space.
_CONTROL_WS = re.compile(r"[\r\n\t]")
# Rule 1b: literal escape sequences "\n", "\r", "\t" (backslash + letter) used as line breaks. A run that is
# followed by a lower-case letter is kept, so LaTeX commands such as "\neg" or "\theta" survive.
_ESCAPED_BREAK = re.compile(r"(?:\\[nrt])+(?![a-z])")
# Rule 9: typographic apostrophe U+2019 -> "'" (so contractions and the tokenizer see one apostrophe).
_CURLY_APOSTROPHE = "’"
# Rule 2: whitespace before closing punctuation.
_SPACE_BEFORE_CLOSE = re.compile(r"\s+([.,!?;:%)\]}])")
# Rule 3: whitespace after opening brackets.
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
# Rule 4: detached contractions, case-insensitive.
_CONTRACTION = re.compile(r"\s+(n't|'(?:s|re|ve|ll|d|m))\b", re.IGNORECASE)
# Rule 5: spaced hyphen between word characters.
_SPACED_HYPHEN = re.compile(r"(?<=\w)\s+-\s+(?=\w)")
# Rule 6: whitespace just inside a pair of double quotes (pairs are matched left to right).
_QUOTED = re.compile(r'"([^"]*)"')
# Rule 7: "# 1" -> "#1".
_HASH_NUMBER = re.compile(r"#\s+(?=\d)")
# Rule 10: spaced slash between word characters ("km / h" -> "km/h").
_SPACED_SLASH = re.compile(r"(?<=\w)\s+/\s+(?=\w)")
# Rule 11: a single quote standing alone between spaces (pre-tokenised quoting such as "' prove '").
_DETACHED_QUOTE = re.compile(r"(?<!\S)'(?=[\s.,!?;:)\]]|$)")
_POSSESSIVE_BEFORE = re.compile(r"[a-z](?:s|in)\s+$", re.IGNORECASE)  # "judges '", "makin '"
_WORD_AFTER = re.compile(r"\s+[a-z]")
# Rule 12: sentence boundary that lost its space ("situations.One", "(CPI).The", "Form 1040.It").
_GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9)\]\"'][.!?])(?=[A-Z][a-z])")
# Rule 8: runs of whitespace.
_MULTI_WS = re.compile(r"\s+")

# Human-readable list of every rule, written into data/summary.json and the README.
NORMALISATION_RULES: tuple[str, ...] = (
    "1. \\r, \\n and \\t become a space.",
    "1b. Literal escape sequences \\n, \\r, \\t (backslash + letter) become a space unless a lower-case "
    "letter follows (LaTeX commands such as \\neg are kept).",
    "9. The typographic apostrophe U+2019 becomes \"'\".",
    "2. Whitespace before closing punctuation . , ! ? ; : % ) ] } is removed.",
    "3. Whitespace after opening brackets ( [ { is removed.",
    "4. Detached contractions (n't 's 're 've 'll 'd 'm) are re-attached, case-insensitive.",
    "5. Spaced hyphens between word characters are tightened (oscar - winning -> oscar-winning).",
    "6. Whitespace just inside double quotes is removed (pairs matched left to right).",
    "7. '# 1' becomes '#1'.",
    "10. Spaced slashes between word characters are tightened (km / h -> km/h).",
    "11. Single quotes standing alone between spaces are tightened: with an even number they are paired "
    "left to right (' prove ' -> 'prove'); with an odd number, one after a word ending in -s or -in that is "
    "followed by a word is attached to that word first (judges ' vision -> judges' vision).",
    "12. A sentence boundary without a space gets one: [a-z0-9)]\"'] + [.!?] + capitalised word "
    "(situations.One -> situations. One); abbreviations such as U.S.A are not affected.",
    "8. Runs of whitespace collapse to one space; the result is stripped.",
)

# The fixed tokenizer from the project specification.
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")
# Sentence boundary: after ., ! or ? when followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_CHUNK = re.compile(r"\S+")


def _tighten_single_quotes(text: str) -> str:
    """Rule 11: tighten single quotes that stand alone between spaces (see ``NORMALISATION_RULES``)."""
    quotes = [m.start() for m in _DETACHED_QUOTE.finditer(text)]
    if not quotes:
        return text
    attach_left: set[int] = set()
    if len(quotes) % 2:
        for i in quotes:
            if _POSSESSIVE_BEFORE.search(text[:i]) and _WORD_AFTER.match(text, i + 1):
                attach_left.add(i)
                break
    rest = [i for i in quotes if i not in attach_left]
    opening: set[int] = set()
    if len(rest) % 2 == 0:
        opening = set(rest[0::2])
        attach_left |= set(rest[1::2])
    # Edit from right to left so that earlier positions stay valid.
    for i in sorted(opening | attach_left, reverse=True):
        if i in opening:
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            text = text[:i + 1] + text[j:]
        else:
            j = i
            while j > 0 and text[j - 1].isspace():
                j -= 1
            text = text[:j] + text[i:]
    return text


def normalize_text(text: str) -> str:
    """Return ``text`` with tokenisation and layout artefacts removed.

    The rules are applied in this order (each one is unit-tested):

    1. ``\\r``, ``\\n`` and ``\\t`` become a space.
       1b. Literal escape sequences ``\\n``, ``\\r``, ``\\t`` (a backslash followed by the letter, as stored
       in some HC3 ChatGPT answers) become a space unless a lower-case letter follows (keeps ``\\neg``).
    9. The typographic apostrophe U+2019 becomes ``'``.
    2. Whitespace before closing punctuation ``. , ! ? ; : % ) ] }`` is removed ("word ." -> "word.").
    3. Whitespace after opening brackets ``( [ {`` is removed.
    4. Detached contractions are re-attached: " n't", " 's", " 're", " 've", " 'll", " 'd", " 'm"
       (case-insensitive).
    5. Spaced hyphens between word characters are tightened ("oscar - winning" -> "oscar-winning").
    6. Whitespace just inside double quotes is removed (``" Best Seller "`` -> ``"Best Seller"``).
       Quote marks are paired left to right, so ``a "b" c "d"`` is left unchanged.
    7. "# 1" becomes "#1".
    10. Spaced slashes between word characters are tightened ("km / h" -> "km/h").
    11. Single quotes standing alone between spaces are tightened (``' prove '`` -> ``'prove'``; with an
        odd number, a plural possessive or elision such as "judges ' vision" is attached first).
    12. A missing space after a sentence end is restored ("situations.One" -> "situations. One") when
        ``.``, ``!`` or ``?`` follows a lower-case letter, digit, ``)``, ``]`` or a quote and precedes a
        capitalised word. Abbreviations such as "U.S.A" are left alone.
    8. Runs of whitespace collapse to one space and the result is stripped.

    Rules 1-8 are the shared project rules; 1b and 9-12 remove further class-dependent HC3 artefacts
    (rates per class before and after are in ``data/summary.json``).

    Args:
        text: Any string. ``None`` is treated as an empty string.

    Returns:
        The normalised text.
    """
    if not text:
        return ""
    text = _CONTROL_WS.sub(" ", str(text))
    text = _ESCAPED_BREAK.sub(" ", text)
    text = text.replace(_CURLY_APOSTROPHE, "'")
    text = _SPACE_BEFORE_CLOSE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _CONTRACTION.sub(r"\1", text)
    text = _SPACED_HYPHEN.sub("-", text)
    text = _QUOTED.sub(lambda m: '"' + m.group(1).strip() + '"', text)
    text = _HASH_NUMBER.sub("#", text)
    text = _SPACED_SLASH.sub("/", text)
    text = _tighten_single_quotes(text)
    text = _GLUED_SENTENCE.sub(" ", text)
    return _MULTI_WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lower-case word tokens with the project's fixed tokenizer.

    The pattern is exactly ``[a-z]+(?:'[a-z]+)?|\\d+(?:\\.\\d+)?`` applied to the lower-cased text, so words
    keep one internal apostrophe ("don't") and numbers keep one decimal part ("3.5"). Callers pass
    normalised text, in which the typographic apostrophe has already become "'" (rule 9).
    """
    return WORD_PATTERN.findall(text.lower())


def count_words(text: str) -> int:
    """Number of tokenizer words in ``text``."""
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """Split normalised text after ``.``, ``!`` or ``?`` followed by whitespace; drop empty pieces."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Return the longest prefix of whitespace-delimited chunks that holds at most ``max_words`` words.

    Truncation happens on word boundaries and keeps punctuation attached to the kept words. Words are
    counted with :func:`tokenize`, so the result always has ``count_words(result) <= max_words``; it has
    exactly ``max_words`` words unless a chunk such as "well-known" (two tokenizer words) straddles the
    limit, in which case that chunk is left out.
    """
    if max_words <= 0:
        return ""
    total = 0
    end = 0
    for match in _WHITESPACE_CHUNK.finditer(text):
        n = count_words(match.group())
        if total + n > max_words:
            break
        total += n
        end = match.end()
        if total == max_words:
            break
    else:
        return text.strip()
    return text[:end].strip()


def split_at_word_limit(text: str, max_words: int = MAX_WORDS) -> tuple[str, str]:
    """Split ``text`` into the analysed prefix (see :func:`truncate_words`) and the unseen remainder."""
    head = truncate_words(text, max_words)
    start = text.find(head) if head else 0
    rest = text[start + len(head):].strip()
    return head, rest
