"""Pure presentation functions of the Gradio app, and app construction without launching."""

import json

import pytest

from src import app
from src.baseline import Baseline
from src.llm import LLMClient, MissingAPIKeyError
from src.prompts import load_few_shot_examples

LONG_TEXT = " ".join(
    f"Sentence number {i} talks about topic {chr(97 + i % 26)} and adds a few more words here." for i in range(12)
)


@pytest.fixture(scope="module")
def artefacts(pipeline):
    return Baseline.load(pipeline["model"]), load_few_shot_examples(pipeline["few_shot"])


def mock_factory(provider, model):
    return LLMClient("mock")


def test_word_count_text():
    assert app.word_count_text("") == "**Words:** 0"
    assert "always labelled 'Uncertain'" in app.word_count_text("just a few words")
    assert "first 300 words" in app.word_count_text("word " * 301)


def test_compute_statistics_with_and_without_baseline(artefacts):
    baseline, _ = artefacts
    table, md = app.compute_statistics(LONG_TEXT, baseline)
    assert list(table.columns) == app.TABLE_HEADERS and len(table) == 6
    assert set(table["Closer to"].iloc[2:]) <= {"human", "AI", "tie", "n/a"}
    assert "Statistical baseline label" in md and "not a probability" in md
    table, md = app.compute_statistics(LONG_TEXT, None)
    assert (table["Human median (IQR)"].iloc[2:] == "not supplied").all()
    assert "not available" in md
    empty, msg = app.compute_statistics("   ", baseline)
    assert empty.empty and msg == "Paste a text first."


def test_assess_with_mock_llm(artefacts):
    baseline, few_shot = artefacts
    out = app.assess_with_llm(LONG_TEXT, "mock", "mock-evidence-v1", "V4", baseline, few_shot, mock_factory)
    assert out["label_md"].startswith("### ")
    assert "JSON valid (strict schema): **yes**" in out["flags_md"]
    assert "Grounded evidence: **100%**" in out["flags_md"]
    assert "quoting the named feature's own numbers: **100%**" in out["flags_md"]
    assert json.loads(out["raw_json"])["label"] in ("Likely AI-generated", "Likely human-written", "Uncertain")
    assert "mock-evidence-v1" in out["model_md"] and "V4" in out["model_md"]


def test_missing_key_is_friendly(artefacts):
    baseline, few_shot = artefacts

    def no_key(provider, model):
        raise MissingAPIKeyError("GROQ_API_KEY is not set: create a free key at console.groq.com/keys")

    out = app.assess_with_llm(LONG_TEXT, "groq", "openai/gpt-oss-120b", "V2", baseline, few_shot, no_key)
    assert "No API key for groq" in out["label_md"] and "statistics above still apply" in out["label_md"]
    assert out["raw_json"] == ""


def test_v3_without_examples_and_empty_text(artefacts):
    baseline, _ = artefacts
    out = app.assess_with_llm(LONG_TEXT, "mock", "", "V3", baseline, None, mock_factory)
    assert "not available yet" in out["label_md"]
    assert app.assess_with_llm("", "mock", "", "V0", baseline, None, mock_factory)["label_md"] == "Paste a text first."


def test_model_choices():
    choices, default = app.model_choices("groq")
    assert default == "openai/gpt-oss-120b" and default in choices


def test_schema_invalid_label_is_shown_as_uncertain(artefacts):
    baseline, few_shot = artefacts

    class Reply:
        provider, requested_model, returned_model, latency_s, json_mode_fallback = "mock", "m", "m", 0.0, False
        text = json.dumps({"label": "Likely AI-generated", "evidence": ["mattr is 0.5."], "limitations": ["x"],
                           "confidence": 0.9})

    class Client:
        def complete(self, *args, **kwargs):
            return Reply()

    out = app.assess_with_llm(LONG_TEXT, "mock", "m", "V0", baseline, few_shot, lambda p, m: Client())
    assert out["label_md"].startswith("### Uncertain")
    assert "named 'Likely AI-generated', but it failed the schema" in out["flags_md"]


def test_build_app_without_launching(artefacts):
    import gradio as gr

    baseline, few_shot = artefacts
    demo = app.build_app(baseline, few_shot, mock_factory, load_artifacts=False)
    assert isinstance(demo, gr.Blocks)
    assert app.BANNER.startswith("Use public or anonymised text only.")


def test_smoke_flag_builds_without_launching(monkeypatch, capsys):
    import gradio as gr

    def no_launch(self, *args, **kwargs):
        raise AssertionError("--smoke must not launch a server")

    monkeypatch.setattr(gr.Blocks, "launch", no_launch)
    assert app.main(["--smoke"]) == 0
    assert "not launched" in capsys.readouterr().out
