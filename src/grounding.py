"""Explanation-grounding checks for the LLM's evidence items.

An evidence item is **grounded** when

1. it names at least one supplied feature - its exact name (``sentence_length_cv``) or a documented alias
   (``FEATURES[...].aliases``, e.g. "sentence length variation"; "212 words" names ``word_count``), and
2. every number in it matches a supplied number within tolerance (absolute 0.006 or relative 1%).

Supplied numbers are the numbers in the prompt's COMPUTED FEATURES section (feature values, reference
medians and quartiles, word and sentence counts) plus the 100-word threshold and the 300-word length limit.
Numbers quoted from the assessed text itself are *not* supplied statistics.

``has_invented_number`` is True when any evidence or limitation item contains a number that matches no
supplied number.

This definition (the project's) accepts an item that quotes no number at all, or that quotes a supplied
number belonging to a *different* feature. A stricter companion check is therefore reported as well: an
item is **strictly grounded** when it names a supplied feature, quotes at least one number, and every
number it quotes appears on the COMPUTED FEATURES line of a feature it names (its value, medians or
quartiles) or is the 100/300-word constant. ``strict_grounded_rate`` is the share of such items; it is
what the V2 rule ("quote its supplied value and the closer median") asks for.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from src.config import MAX_WORDS, MIN_WORDS_DECISION
from src.features import COUNT_INFO, FEATURES
from src.prompts import features_section

ABS_TOL = 0.006
REL_TOL = 0.01
CONSTANT_NUMBERS: tuple[float, ...] = (float(MIN_WORDS_DECISION), float(MAX_WORDS))

# Numbers such as 212, 1,204, 0.4123 or .41. A number glued to letters ("V4", "x2", "3.5x", "5.83bits") is
# ignored as a whole: the "(?!\.\d)" stops the engine from backtracking to the integer part of a decimal.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z_\d]|\.\d)"
    r"|(?<![\w.])\.\d+(?![A-Za-z_\d])"
)
_FEATURE_LINE = re.compile(r"^- (\w+):")

_COUNT_PATTERNS: dict[str, re.Pattern] = {
    "word_count": re.compile(r"\b\d[\d,]*\s+words?\b"),
    "sentence_count": re.compile(r"\b\d+\s+sentences?\b"),
}


def _norm(text: str) -> str:
    """Lower-case and turn underscores/hyphens into spaces so names and aliases compare equally."""
    return " ".join(re.sub(r"[_\-‐-–]", " ", text.lower()).split())


def _name_patterns() -> dict[str, list[re.Pattern]]:
    patterns: dict[str, list[re.Pattern]] = {}
    for info in [*FEATURES.values(), *COUNT_INFO.values()]:
        phrases = {_norm(info.name), *(_norm(a) for a in info.aliases)}
        patterns[info.name] = [re.compile(r"\b" + re.escape(p) + r"\b") for p in sorted(phrases)]
    return patterns


NAME_PATTERNS: dict[str, list[re.Pattern]] = _name_patterns()


def extract_numbers(text: str) -> list[float]:
    """All numbers in ``text`` as floats (thousands separators removed)."""
    return [float(m.group().replace(",", "")) for m in NUMBER_RE.finditer(text or "")]


def supplied_numbers_from_prompt(prompt: str) -> list[float]:
    """Numbers the model was given: the COMPUTED FEATURES section plus the 100- and 300-word constants."""
    return sorted(set(extract_numbers(features_section(prompt))) | set(CONSTANT_NUMBERS))


def supplied_numbers_by_feature(prompt: str) -> dict[str, list[float]]:
    """Numbers on each ``- name: ...`` line of the COMPUTED FEATURES section, keyed by feature/count name."""
    out: dict[str, list[float]] = {}
    for line in features_section(prompt).splitlines():
        m = _FEATURE_LINE.match(line.strip())
        if m:
            out[m.group(1)] = extract_numbers(line[line.index(":") + 1:])
    return out


def matches_supplied(x: float, supplied: Iterable[float], abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    """True if ``x`` equals some supplied number within the absolute or relative tolerance."""
    return any(abs(x - s) <= abs_tol or abs(x - s) <= rel_tol * abs(s) for s in supplied)


def named_features(item: str) -> list[str]:
    """Supplied features (or descriptive counts) that ``item`` names explicitly or via an alias."""
    norm = _norm(item)
    found = [name for name, pats in NAME_PATTERNS.items() if any(p.search(norm) for p in pats)]
    for name, pat in _COUNT_PATTERNS.items():
        if name not in found and pat.search(norm):
            found.append(name)
    return found


@dataclass
class GroundingResult:
    """Grounding outcome for one response."""

    items: list[dict] = field(default_factory=list)
    grounded_evidence_rate: float = math.nan
    strict_grounded_rate: float = math.nan
    has_invented_number: bool = False
    invented_numbers: list[float] = field(default_factory=list)


def check_grounding(
    evidence: list[str],
    supplied: Iterable[float],
    limitations: Iterable[str] = (),
    by_feature: Mapping[str, Iterable[float]] | None = None,
) -> GroundingResult:
    """Check each evidence item and look for invented numbers in evidence and limitations.

    Args:
        evidence: Evidence strings from the parsed response.
        supplied: Numbers available to the model (see :func:`supplied_numbers_from_prompt`).
        limitations: Limitation strings (checked for invented numbers only).
        by_feature: Numbers per feature line (see :func:`supplied_numbers_by_feature`); needed for the
            strict check, which is NaN without it.

    Returns:
        ``grounded_evidence_rate`` and ``strict_grounded_rate`` are NaN when there is no evidence item.
    """
    supplied = list(supplied)
    lines = {k: list(v) for k, v in by_feature.items()} if by_feature is not None else None
    items = []
    invented_all: list[float] = []
    for text in evidence:
        names = named_features(text)
        numbers = extract_numbers(text)
        invented = [x for x in numbers if not matches_supplied(x, supplied)]
        invented_all += invented
        item = {
            "text": text,
            "features": names,
            "numbers": numbers,
            "invented": invented,
            "grounded": bool(names) and not invented,
            "strictly_grounded": None,
        }
        if lines is not None:
            own = [x for name in names for x in lines.get(name, [])] + list(CONSTANT_NUMBERS)
            item["strictly_grounded"] = bool(names) and bool(numbers) and all(matches_supplied(x, own) for x in numbers)
        items.append(item)
    for text in limitations:
        invented_all += [x for x in extract_numbers(text) if not matches_supplied(x, supplied)]
    rate = sum(i["grounded"] for i in items) / len(items) if items else math.nan
    strict = sum(bool(i["strictly_grounded"]) for i in items) / len(items) if items and lines is not None else math.nan
    return GroundingResult(
        items=items,
        grounded_evidence_rate=rate,
        strict_grounded_rate=strict,
        has_invented_number=bool(invented_all),
        invented_numbers=invented_all,
    )


def check_grounding_for_prompt(evidence: list[str], limitations: Iterable[str], prompt: str) -> GroundingResult:
    """:func:`check_grounding` with the supplied numbers (overall and per feature) read from ``prompt``."""
    return check_grounding(evidence, supplied_numbers_from_prompt(prompt), limitations,
                           supplied_numbers_by_feature(prompt))
