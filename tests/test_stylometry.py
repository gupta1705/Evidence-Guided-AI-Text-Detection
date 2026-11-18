"""Algorithm v2: style features, word log-odds lexicon, fitting, evaluation, CLI and app output."""

import math

import pytest

from src import app, stylometry
from src.baseline import Baseline
from src.config import LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN
from src.data import load_split


def test_style_features_hand_checked():
    f = stylometry.style_features("I don't know, you know? Additionally it is fine.")
    # words: i don't know you know additionally it is fine -> 9 words, 2 sentences
    assert f["first_person_rate"] == pytest.approx(1 / 9)
    assert f["second_person_rate"] == pytest.approx(1 / 9)
    assert f["contraction_rate"] == pytest.approx(1 / 9)
    assert f["assistant_word_rate"] == pytest.approx(1 / 9)
    assert f["comma_rate"] == pytest.approx(1 / 9)
    assert f["exclaim_question_rate"] == pytest.approx(1 / 9)
    assert f["mean_sentence_length"] == pytest.approx(4.5)
    assert set(stylometry.MODEL_FEATURES) - {stylometry.LEXICON_FEATURE} <= set(f)
    empty = stylometry.style_features("")
    assert all(math.isnan(empty[name]) for name in stylometry.STYLE_FEATURES)


def test_lexicon_sign_and_score():
    assert stylometry.fit_lexicon(["rare"], [1]) == {}  # below min_count
    lexicon = stylometry.fit_lexicon(["robot robot cat", "human human cat"], [1, 0], min_count=1)
    assert lexicon["robot"] > 0 > lexicon["human"] and lexicon["cat"] == pytest.approx(0.0)
    assert stylometry.lexicon_score("robot", lexicon) == pytest.approx(lexicon["robot"])
    assert stylometry.lexicon_score("unknown words only", lexicon) == 0.0


@pytest.fixture(scope="module")
def fitted(pipeline):
    reference = load_split("reference", pipeline["data"])
    validation = load_split("validation", pipeline["data"])
    return stylometry.fit_style_model(reference, validation), load_split("test", pipeline["data"])


def test_fit_separates_classes_and_round_trips(fitted, tmp_path):
    model, test = fitted
    assert set(model.weights) == set(stylometry.MODEL_FEATURES) and model.margin >= 0
    labels = [model.predict(t)["label"] for t in test["text"]]
    correct = sum((lab == LABEL_AI) == (y == 1) for lab, y in zip(labels, test["label"]) if lab != LABEL_UNCERTAIN)
    assert correct >= 0.9 * sum(lab != LABEL_UNCERTAIN for lab in labels)
    path = model.save(tmp_path / "s.json")
    again = stylometry.StyleModel.load(path)
    text = test["text"].iloc[0]
    assert again.predict(text)["score"] == pytest.approx(model.predict(text)["score"])
    pred = model.predict(text)
    assert 0 <= pred["probability_ai"] <= 1 and len(pred["top_contributions"]) == 5
    assert pred["label"] in (LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN)


def test_min_words_rule(fitted):
    model, _ = fitted
    model = stylometry.StyleModel.from_dict({**model.to_dict(), "min_words": 1000})
    assert model.predict("a short text")["label"] == LABEL_UNCERTAIN


def test_evaluate_tables(fitted, pipeline):
    model, test = fitted
    table, robust = stylometry.evaluate(test, model, Baseline.load(pipeline["model"]))
    assert set(table["algorithm"]) == {"v1 baseline (4 features, distance rule)", "v2 stylometric logistic regression"}
    assert list(robust["variant"]) == ["swap", "drop", "key", "add", "all 4 variants"]
    assert 0 <= robust["label_unchanged"].min() <= 1


def test_cli_fit_and_eval(pipeline, tmp_path):
    common = ["--data-dir", str(pipeline["data"]), "--model-path", str(tmp_path / "s.json")]
    assert stylometry.main(["fit", *common]) == 0
    assert stylometry.main(["eval", *common, "--baseline-path", str(pipeline["model"]),
                            "--results-dir", str(tmp_path)]) == 0
    assert (tmp_path / "stylometry_metrics.md").is_file() and (tmp_path / "stylometry_robustness.md").is_file()
    assert stylometry.main(["eval", "--model-path", str(tmp_path / "missing.json")]) == 2


def test_app_shows_v2_label(fitted, pipeline):
    model, test = fitted
    _, md = app.compute_statistics(test["text"].iloc[0], Baseline.load(pipeline["model"]), model)
    assert md.startswith("**Algorithm v2 label:**") and "Statistical baseline label" in md
    assert "not available" in app.style_markdown("text", None)
