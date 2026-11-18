"""Dataset construction: cleaning, question-disjoint balanced splits and artefact rates."""

import json

import pandas as pd
import pytest

from src.config import DATA_DIR, MIN_WORDS_DATA, SPLITS
from src.data import (
    ARTEFACT_PATTERNS,
    COLUMNS,
    INFO_PATTERNS,
    artefact_rates,
    clean_answers,
    load_answers,
    load_split,
    prepare_text,
    remove_url_placeholders,
    resolve_hc3_path,
    sample_balanced,
    scaled_sizes,
    split_by_question,
    word_band,
)
from src.text_utils import NORMALISATION_RULES
from tests.conftest import SMALL_SIZES


def test_splits_balanced_disjoint_and_sized(pipeline):
    splits = {name: load_split(name, pipeline["data"]) for name in SPLITS}
    for name, df in splits.items():
        assert list(df.columns) == COLUMNS
        assert len(df) == SMALL_SIZES[name]
        assert (df["label"] == 0).sum() == (df["label"] == 1).sum()
        assert (df["word_count"] >= MIN_WORDS_DATA).all()
        assert set(df["provenance"]) == {"human", "ai-generated"}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not set(splits[a]["question_id"]) & set(splits[b]["question_id"])
    assert pipeline["summary"]["disjointness"]["disjoint"] is True


def test_artefact_rates_near_equal_after_normalisation(pipeline):
    rates = pipeline["summary"]["artefact_rates"]
    before = rates["before_normalisation_all_answers"]
    after = rates["after_normalisation_dataset"]
    # The synthetic corpus really has the artefacts ...
    assert before["human"]["space_period"] > 0.5
    assert before["ai"]["escaped_newline"] > 0.1 and before["ai"]["glued_sentence"] > 0.1
    assert before["human"]["url_placeholder"] > 0.1 and before["human"]["spaced_slash"] > 0.1
    # ... and none survives preparation in either class.
    for key in ARTEFACT_PATTERNS:
        assert after["human"][key] < 0.02, key
        assert after["ai"][key] < 0.02, key
    assert set(INFO_PATTERNS) <= set(after["human"])
    assert pipeline["summary"]["normalisation_rules"] == list(NORMALISATION_RULES)


def test_url_placeholders_removed():
    assert remove_url_placeholders("the [piston effect] (URL_0). Next") == "the piston effect. Next"
    assert prepare_text("see [ WWVB Service ] ( URL_0 ) now , URL_1 The end .") == "see WWVB Service now, The end."
    assert prepare_text("a ( URL_3 ) b") == "a b"
    assert prepare_text("Here : URL_0") == "Here:"
    assert artefact_rates(["has URL_2", "clean"])["url_placeholder"] == 0.5


def test_summary_contents(pipeline):
    summary = json.loads((pipeline["data"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 42
    for name in SPLITS:
        s = summary["splits"][name]
        assert s["by_class"]["human"] == s["by_class"]["ai"]
        assert sum(v["human"] + v["ai"] for v in s["word_count_bands"].values()) == s["n"]
        assert sum(v["human"] + v["ai"] for v in s["by_source"].values()) == s["n"]
    cleaning = summary["cleaning"]
    assert cleaning["removed_empty"] >= 1
    assert cleaning["removed_under_min_words"] >= 1
    assert cleaning["removed_duplicates"] >= 1


def test_clean_answers_rules():
    df = pd.DataFrame({
        "question_id": [0, 0, 1, 1, 2, 3],
        "source": ["s"] * 6,
        "label": [0, 1, 0, 1, 0, 1],
        "provenance": ["human", "ai-generated"] * 3,
        "answer_index": [0, 0, 0, 0, 0, 0],
        "text": ["", "a " * 25, "b " * 25, "b " * 25, "c " * 25, "short text"],
        "word_count": [0, 25, 25, 25, 25, 2],
    })
    cleaned, stats = clean_answers(df)
    assert stats["removed_empty"] == 1
    assert stats["removed_under_min_words"] == 1
    assert stats["removed_label_conflicts"] == 2  # "b ..." appears as human and as AI
    assert list(cleaned["text"]) == ["a " * 25, "c " * 25]


def test_split_by_question_keeps_questions_together():
    rows = []
    for q in range(20):
        for label in (0, 1):
            for j in range(2):
                rows.append({"question_id": q, "source": "s", "label": label,
                             "provenance": "x", "answer_index": j, "text": f"{q}-{label}-{j}", "word_count": 30})
    df = pd.DataFrame(rows)
    splits = split_by_question(df, {"reference": 10, "validation": 4, "test": 4}, seed=42)
    seen = {}
    for name, part in splits.items():
        assert (part["label"] == 0).sum() == (part["label"] == 1).sum()
        for q in part["question_id"]:
            assert seen.setdefault(q, name) == name
    with pytest.raises(ValueError):
        split_by_question(df, {"reference": 40, "validation": 4, "test": 4})
    with pytest.raises(ValueError):
        split_by_question(df, {"reference": 3, "validation": 4, "test": 4})


def test_split_is_deterministic(hc3_file):
    answers, _ = load_answers(hc3_file)
    cleaned, _ = clean_answers(answers)
    a = split_by_question(cleaned, SMALL_SIZES, seed=42)
    b = split_by_question(cleaned, SMALL_SIZES, seed=42)
    for name in SPLITS:
        assert a[name]["id"].tolist() == b[name]["id"].tolist()


def test_resolve_hc3_path(tmp_path, hc3_file):
    assert resolve_hc3_path(hc3_file) == hc3_file
    with pytest.raises(FileNotFoundError):
        resolve_hc3_path(tmp_path / "missing.jsonl")


def test_scaled_sizes_and_bands():
    assert scaled_sizes(None) == {"reference": 3000, "validation": 600, "test": 600}
    small = scaled_sizes(280)
    assert small == {"reference": 200, "validation": 40, "test": 40}
    assert all(v % 2 == 0 for v in scaled_sizes(37).values())
    assert [word_band(n) for n in (20, 99, 100, 199, 200, 5000)] == ["<100", "<100", "100-199", "100-199",
                                                                   ">=200", ">=200"]


def test_sample_balanced():
    df = pd.DataFrame({"label": [0] * 10 + [1] * 10, "x": range(20)})
    out = sample_balanced(df, 6, seed=1)
    assert len(out) == 6 and (out["label"] == 0).sum() == 3
    assert len(sample_balanced(df, None)) == 20


def test_real_dataset_summary_if_built():
    """When the real HC3 splits have been built, check the artefact contract on them too."""
    path = DATA_DIR / "summary.json"
    if not path.is_file():
        pytest.skip("real dataset not built (python -m src.data)")
    summary = json.loads(path.read_text(encoding="utf-8"))
    after = summary["artefact_rates"]["after_normalisation_dataset"]
    for key in ARTEFACT_PATTERNS:
        assert after["human"][key] < 0.02 and after["ai"][key] < 0.02
    assert summary["disjointness"]["disjoint"] is True
