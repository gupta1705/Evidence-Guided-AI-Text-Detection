"""Robust extraction and validation of the LLM's JSON assessment.

The expected object is ``{"label": ..., "evidence": [...], "limitations": [...]}`` with the label one of
"Likely AI-generated", "Likely human-written" or "Uncertain".

* ``json_valid`` is the strict schema check: exactly those three keys, the label spelled exactly as one of
  the allowed values, and non-empty lists of non-empty strings for evidence and limitations. It is the
  same check for every prompt version.
* ``counts_ok`` additionally checks the V4 counts (2-4 evidence items, 1-3 limitations); it is reported
  separately (``counts_compliance`` in the evaluation) because only V4 asks for them.
* The *effective* label (``label``) is the response's label only when the response passes the strict
  schema (``json_valid``); every invalid response - malformed JSON, missing or extra keys, a wrongly spelled
  label, evidence that is not a list - counts as "Uncertain", so coverage and accuracy only use responses a
  pipeline could accept. ``parsed_label`` keeps the allowed label the response named (ignoring case and
  surrounding whitespace) even when the schema failed, so lenient metrics can be recomputed from the logs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.config import LABEL_UNCERTAIN, VALID_LABELS

REQUIRED_KEYS = frozenset({"label", "evidence", "limitations"})
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\n?|```")
_LABEL_LOOKUP = {label.lower(): label for label in VALID_LABELS}


@dataclass
class ParseResult:
    """Outcome of parsing one raw LLM response."""

    parsed: dict | None
    json_valid: bool
    label: str
    label_valid: bool
    parsed_label: str | None = None
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    counts_ok: bool = False
    errors: list[str] = field(default_factory=list)


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences such as ```json ... ```."""
    return _FENCE_RE.sub("", text or "").strip()


def iter_balanced_objects(text: str):
    """Yield every top-level balanced ``{...}`` substring, in order (string- and escape-aware)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return
        yield text[start:end + 1]
        start = text.find("{", end + 1)


def extract_json_object(text: str) -> dict | None:
    """Return the first balanced ``{...}`` in ``text`` that parses as a JSON object, else None."""
    cleaned = strip_code_fences(text)
    for candidate in iter_balanced_objects(cleaned):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def canonical_label(value: object) -> str | None:
    """Map a label to its canonical spelling (case-insensitive); None if it is not an allowed label."""
    if not isinstance(value, str):
        return None
    return _LABEL_LOOKUP.get(" ".join(value.split()).lower())


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


def schema_errors(obj: dict) -> list[str]:
    """List the strict-schema violations of a parsed object (empty list = valid)."""
    errors: list[str] = []
    keys = set(obj)
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected keys: {sorted(extra)}")
    if "label" in obj and obj["label"] not in VALID_LABELS:
        errors.append(f"invalid label: {obj['label']!r}")
    for key in ("evidence", "limitations"):
        if key not in obj:
            continue
        value = obj[key]
        if not isinstance(value, list) or not value:
            errors.append(f"{key} must be a non-empty list")
        elif not all(isinstance(v, str) and v.strip() for v in value):
            errors.append(f"{key} must contain only non-empty strings")
    return errors


def parse_response(raw: str | None) -> ParseResult:
    """Parse a raw LLM response into a :class:`ParseResult` (never raises)."""
    obj = extract_json_object(raw or "")
    if obj is None:
        return ParseResult(parsed=None, json_valid=False, label=LABEL_UNCERTAIN, label_valid=False,
                           errors=["no JSON object found"])
    errors = schema_errors(obj)
    named = canonical_label(obj.get("label"))
    evidence = _string_list(obj.get("evidence"))
    limitations = _string_list(obj.get("limitations"))
    counts_ok = not errors and 2 <= len(evidence) <= 4 and 1 <= len(limitations) <= 3
    return ParseResult(
        parsed=obj,
        json_valid=not errors,
        label=obj["label"] if not errors else LABEL_UNCERTAIN,
        label_valid=named is not None,
        parsed_label=named,
        evidence=evidence,
        limitations=limitations,
        counts_ok=counts_ok,
        errors=errors,
    )
