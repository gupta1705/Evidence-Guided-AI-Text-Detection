"""Robust JSON extraction and schema validation."""

import json

from src.parsing import canonical_label, extract_json_object, parse_response, strip_code_fences

VALID = {
    "label": "Likely AI-generated",
    "evidence": ["mattr is 0.7400, closer to the AI reference median 0.7567.", "word_count is 212."],
    "limitations": ["Not proof of authorship; manual review is recommended."],
}


def test_valid_json():
    r = parse_response(json.dumps(VALID))
    assert r.json_valid and r.label_valid and r.counts_ok
    assert r.label == "Likely AI-generated"
    assert r.evidence == VALID["evidence"] and r.limitations == VALID["limitations"]


def test_fenced_json_and_surrounding_prose():
    raw = "Here is my assessment:\n```json\n" + json.dumps(VALID, indent=2) + "\n```\nHope this helps."
    r = parse_response(raw)
    assert r.json_valid and r.label == "Likely AI-generated"
    assert strip_code_fences("```json\n{}\n```") == "{}"


def test_braces_inside_strings_and_earlier_non_json_braces():
    obj = {**VALID, "evidence": ["a {curly} remark", "word_count is 212."]}
    raw = "Template {not json} then " + json.dumps(obj)
    r = parse_response(raw)
    assert r.json_valid and r.evidence[0] == "a {curly} remark"


def test_malformed_json_becomes_uncertain():
    for raw in ['{"label": "Likely AI-generated", "evidence": ["unterminated', "no json here", "", None]:
        r = parse_response(raw)
        assert not r.json_valid and r.parsed is None
        assert r.label == "Uncertain" and not r.label_valid and r.parsed_label is None


def test_wrong_label_is_invalid_and_uncertain():
    r = parse_response(json.dumps({**VALID, "label": "AI"}))
    assert not r.json_valid and r.label == "Uncertain" and not r.label_valid
    assert any("invalid label" in e for e in r.errors)


def test_wrongly_cased_label_fails_schema_and_counts_as_uncertain():
    r = parse_response(json.dumps({**VALID, "label": "likely human-written"}))
    assert not r.json_valid and r.label == "Uncertain"
    assert r.label_valid and r.parsed_label == "Likely human-written"  # kept for lenient re-analysis


def test_extra_or_missing_keys_fail_strict_schema_and_count_as_uncertain():
    extra = parse_response(json.dumps({**VALID, "confidence": 0.9}))
    assert not extra.json_valid and extra.label == "Uncertain" and extra.parsed_label == "Likely AI-generated"
    missing = parse_response(json.dumps({"label": "Likely AI-generated", "evidence": ["x"]}))
    assert not missing.json_valid and missing.label == "Uncertain"
    assert any("missing" in e for e in missing.errors)
    wrong_type = parse_response(json.dumps({**VALID, "evidence": "a single string"}))
    assert not wrong_type.json_valid and wrong_type.evidence == ["a single string"]
    assert wrong_type.label == "Uncertain"
    many = parse_response('{"label": "likely ai-generated", "evidence": "x", "limitations": [], "confidence": 0.9}')
    assert len(many.errors) == 4 and many.label == "Uncertain"


def test_valid_response_keeps_its_label():
    for label in ("Likely AI-generated", "Likely human-written", "Uncertain"):
        r = parse_response(json.dumps({**VALID, "label": label}))
        assert r.json_valid and r.label == label == r.parsed_label


def test_counts_ok_flag():
    one_item = parse_response(json.dumps({**VALID, "evidence": ["only one"]}))
    assert one_item.json_valid and not one_item.counts_ok
    five = parse_response(json.dumps({**VALID, "evidence": ["e"] * 5}))
    assert five.json_valid and not five.counts_ok


def test_helpers():
    assert extract_json_object("[1, 2]") is None
    assert extract_json_object('{"a": 1} {"b": 2}') == {"a": 1}
    assert canonical_label("  Uncertain ") == "Uncertain"
    assert canonical_label(3) is None
