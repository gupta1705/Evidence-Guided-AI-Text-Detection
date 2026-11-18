"""Baseline maths, threshold/margin fitting and the frozen decision rule."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.baseline import Baseline, fit_margin, fit_threshold, label_metrics, reference_stats
from src.config import LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN


def _ref_frame():
    return pd.DataFrame({"label": [0, 0, 0, 0, 1, 1, 1, 1], "mattr": [1, 2, 3, 4, 5, 6, 7, 8]}, dtype=float)


def test_reference_stats_hand_computed():
    stats = reference_stats(_ref_frame(), ["mattr"])["mattr"]
    assert stats["human_median"] == 2.5 and stats["ai_median"] == 6.5
    assert stats["human_q1"] == pytest.approx(1.75) and stats["human_q3"] == pytest.approx(3.25)
    assert stats["pooled_iqr"] == pytest.approx(3.5)  # quantiles of 1..8: 2.75 and 6.25
    assert stats["pooled_std"] == pytest.approx(np.std(np.arange(1, 9), ddof=1))
    assert stats["scale"] == pytest.approx(3.5)


def test_scale_floor():
    flat = pd.DataFrame({"label": [0, 0, 1, 1], "mattr": [0.5, 0.5, 0.5, 0.5]})
    assert reference_stats(flat, ["mattr"])["mattr"]["scale"] == 1e-6
    spiky = pd.DataFrame({"label": [0] * 4 + [1] * 4, "mattr": [0, 0, 0, 0, 0, 0, 0, 10.0]})
    s = reference_stats(spiky, ["mattr"])["mattr"]
    assert s["pooled_iqr"] == 0.0
    assert s["scale"] == pytest.approx(0.25 * s["pooled_std"])


def test_score_is_distance_difference():
    b = Baseline(reference=reference_stats(_ref_frame(), ["mattr"]), features=["mattr"])
    d_h, d_a, k = b.distances({"mattr": 6.5})
    assert (d_h, d_a, k) == (pytest.approx(4 / 3.5), 0.0, 1)
    assert b.score({"mattr": 6.5}) == pytest.approx(4 / 3.5)
    assert b.score({"mattr": 2.5}) == pytest.approx(-4 / 3.5)
    assert math.isnan(b.score({"mattr": math.nan}))


def test_score_averages_available_features():
    ref = {
        "mattr": {"human_median": 0.0, "ai_median": 1.0, "scale": 1.0},
        "word_entropy": {"human_median": 0.0, "ai_median": 2.0, "scale": 2.0},
    }
    b = Baseline(reference=ref, features=["mattr", "word_entropy"])
    # mattr: d_h 1, d_a 0; entropy: d_h 1, d_a 0 -> score 1
    assert b.score({"mattr": 1.0, "word_entropy": 2.0}) == pytest.approx(1.0)
    # a missing feature is skipped, not counted as zero
    assert b.score({"mattr": 1.0, "word_entropy": math.nan}) == pytest.approx(1.0)


def test_fit_threshold_midpoint_and_ties():
    t, bacc = fit_threshold(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
    assert t == 0.0 and bacc == 1.0
    # Candidates 0.5, 2.5 and 4.5 tie (balanced accuracy 2/3) -> the middle one is chosen.
    t, bacc = fit_threshold(np.arange(6, dtype=float), np.array([0, 1, 0, 1, 0, 1]))
    assert t == 2.5 and bacc == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        fit_threshold(np.array([1.0, 2.0]), np.array([1, 1]))


def test_fit_margin_smallest_delta_meeting_targets():
    scores = np.array([-3.0, -2.0, -0.1, 0.1, 2.0, 3.0])
    y = np.array([0, 0, 1, 0, 1, 1])  # the two texts closest to t=0 are wrong
    delta, info = fit_margin(scores, y, 0.0, target_accuracy=0.85, min_coverage=0.6)
    assert delta == pytest.approx(0.1)
    assert info["target_met"] and info["decided_accuracy"] == 1.0
    assert info["coverage"] == pytest.approx(4 / 6)


def test_fit_margin_fallback_when_target_unreachable():
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y = np.array([1, 0, 0, 1, 1, 0])
    delta, info = fit_margin(scores, y, 0.0, target_accuracy=0.99, min_coverage=0.6)
    assert not info["target_met"]
    assert info["coverage"] >= 0.6


def test_decision_rule():
    ref = {name: {"human_median": 0.0, "ai_median": 1.0, "scale": 1.0, "human_q1": 0, "human_q3": 0,
                  "ai_q1": 1, "ai_q3": 1}
           for name in ("sentence_length_cv", "mattr", "repeated_trigram_rate", "word_entropy")}
    b = Baseline(reference=ref, threshold=0.0, margin=0.2, min_words=100)
    ai = {"word_count": 150, "sentence_length_cv": 1.0, "mattr": 1.0, "repeated_trigram_rate": 1.0, "word_entropy": 1.0}
    human = {**ai, "sentence_length_cv": 0.0, "mattr": 0.0, "repeated_trigram_rate": 0.0, "word_entropy": 0.0}
    middle = {**ai, "sentence_length_cv": 0.55, "mattr": 0.55, "repeated_trigram_rate": 0.45, "word_entropy": 0.5}
    assert b.predict(ai)["label"] == LABEL_AI
    assert b.predict(human)["label"] == LABEL_HUMAN
    assert b.predict(middle)["label"] == LABEL_UNCERTAIN
    short = {**ai, "word_count": 99}
    assert b.predict(short)["label"] == LABEL_UNCERTAIN
    assert b.predict(short, apply_min_words=False)["label"] == LABEL_AI
    one_feature = {"word_count": 150, "mattr": 1.0, "sentence_length_cv": math.nan,
                   "repeated_trigram_rate": math.nan, "word_entropy": math.nan}
    assert b.predict(one_feature)["reason"] == "fewer than 2 valid features"


def test_explain_closer_to():
    b = Baseline(reference=reference_stats(_ref_frame(), ["mattr"]), features=["mattr"])
    rows = b.explain({"mattr": 6.0})
    assert rows[0]["closer_to"] == "AI" and rows[0]["human_median"] == 2.5 and rows[0]["ai_median"] == 6.5
    assert b.explain({"mattr": 1.0})[0]["closer_to"] == "human"
    assert b.explain({"mattr": 4.5})[0]["closer_to"] == "tie"
    assert b.explain({"mattr": math.nan})[0]["closer_to"] == "n/a"


def test_label_metrics_hand_computed():
    y = np.array([0, 0, 0, 1, 1, 1])
    labels = [LABEL_HUMAN, LABEL_AI, LABEL_UNCERTAIN, LABEL_AI, LABEL_AI, LABEL_HUMAN]
    m = label_metrics(y, labels)
    assert m["coverage"] == pytest.approx(5 / 6)
    assert m["decided_accuracy"] == pytest.approx(3 / 5)  # correct: h, ai, ai
    assert m["decided_f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert m["human_fpr"] == pytest.approx(1 / 3)
    assert m["ai_recall"] == pytest.approx(2 / 3)


def test_metrics_agree_with_scikit_learn():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 200)
    labels = rng.choice([LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN], 200)
    decided = labels != LABEL_UNCERTAIN
    pred = (labels[decided] == LABEL_AI).astype(int)
    m = label_metrics(y, labels)
    assert m["decided_accuracy"] == pytest.approx(accuracy_score(y[decided], pred))
    assert m["decided_f1"] == pytest.approx(f1_score(y[decided], pred))

    scores = rng.normal(size=200) + y
    t, bacc = fit_threshold(scores, y)
    assert bacc == pytest.approx(balanced_accuracy_score(y, (scores > t).astype(int)))


def test_frozen_baseline_file(pipeline):
    data = json.loads(pipeline["model"].read_text(encoding="utf-8"))
    for key in ("features", "reference_stats", "threshold", "margin", "min_words", "validation_metrics", "created"):
        assert key in data
    for stats in data["reference_stats"].values():
        assert stats["scale"] >= 1e-6
    loaded = Baseline.load(pipeline["model"])
    assert loaded.threshold == data["threshold"] and loaded.margin == data["margin"]
    assert loaded.to_dict()["reference_stats"] == data["reference_stats"]


def test_validation_report_with_and_without_rule(pipeline):
    table = pd.read_csv(pipeline["results"] / "baseline_validation.csv")
    assert set(table["rule"]) == {"with 100-word rule", "without word rule"}
    assert set(table["subset"]) == {"all", ">=100", "<100", "100-199", ">=200"}
    with_rule = table[(table["rule"] == "with 100-word rule") & (table["subset"] == "<100")].iloc[0]
    assert with_rule["coverage"] == 0.0
    rows = table[table["rule"] == "with 100-word rule"].set_index("subset")
    assert rows.loc[">=100", "n"] == rows.loc["100-199", "n"] + rows.loc[">=200", "n"]
    notes = (pipeline["results"] / "baseline_validation.md").read_text(encoding="utf-8")
    assert "coverage target" in notes and ">= 100 words" in notes


def _texts(n_words: int, n: int, word: str) -> list[str]:
    return [" ".join(f"{word}{i % 7} beta gamma." for i in range(n_words // 3)) for _ in range(n)]


def test_reference_stats_use_only_texts_the_rule_can_label():
    """Short reference texts (features not length-controlled) must not move the medians."""
    from src.baseline import decidable_reference, feature_frame

    long_h, long_a = _texts(150, 12, "h"), _texts(150, 12, "a")
    short = _texts(30, 40, "s")  # many short texts of both classes
    df = pd.DataFrame({"text": long_h + long_a + short + short,
                       "label": [0] * 12 + [1] * 12 + [0] * 40 + [1] * 40})
    feats = feature_frame(df)
    used, description = decidable_reference(feats, 100)
    assert (used["word_count"] >= 100).all() and len(used) == 24
    assert ">= 100 words" in description
    # Too few long texts per class (tiny smoke runs): all texts are used, and the description says so.
    few, description = decidable_reference(feats.iloc[[0, 12, *range(24, 104)]], 100)
    assert len(few) == 82 and "all reference texts" in description


def test_frozen_baseline_records_reference_filter(pipeline):
    data = json.loads(pipeline["model"].read_text(encoding="utf-8"))
    assert data["reference_filter"].startswith("reference texts with >= 100 words")
    assert data["n_reference_used"] < data["n_reference"]
    assert "validation_metrics_decidable_texts" in data
    for stats in data["reference_stats"].values():
        assert stats["n_human"] + stats["n_ai"] <= data["n_reference_used"]


def test_empty_frames_have_columns():
    from src.baseline import PREDICTION_COLUMNS, feature_frame, predictions

    empty = feature_frame(pd.DataFrame({"text": [], "label": []}))
    assert "word_entropy" in empty.columns and empty.empty
    b = Baseline(reference={})
    assert list(predictions(empty, b).columns) == PREDICTION_COLUMNS


def test_cli_errors_are_clean(tmp_path, capsys):
    from src.baseline import main

    assert main(["eval", "--model-path", str(tmp_path / "missing.json"), "--data-dir", str(tmp_path)]) == 2
    assert "baseline fit" in capsys.readouterr().err
    assert main(["fit", "--data-dir", str(tmp_path), "--model-path", str(tmp_path / "m.json"),
                 "--few-shot-path", str(tmp_path / "f.json"), "--results-dir", str(tmp_path)]) == 2
    assert "python -m src.data" in capsys.readouterr().err


def test_baseline_separates_synthetic_classes(pipeline):
    from src.baseline import evaluate_baseline
    from src.data import load_split

    table, per_text = evaluate_baseline(load_split("test", pipeline["data"]), Baseline.load(pipeline["model"]))
    overall = table[(table["subset"] == "all") & (table["rule"] == "with 100-word rule")].iloc[0]
    assert overall["decided_accuracy"] > 0.8
    assert {"pred_label", "baseline_score"} <= set(per_text.columns)


def test_baseline_cli_eval(pipeline, tmp_path):
    from src.baseline import main

    code = main(["eval", "--data-dir", str(pipeline["data"]), "--model-path", str(pipeline["model"]),
                 "--results-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "baseline_metrics.csv").is_file() and (tmp_path / "baseline_metrics.md").is_file()
