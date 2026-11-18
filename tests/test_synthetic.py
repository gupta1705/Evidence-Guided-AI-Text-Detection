"""Optional synthetic / AI-edited data, run offline with the mock provider."""

import pandas as pd

from src.synthetic import EDIT_PROMPT, main


def test_edited_mode_records_provenance_and_prompt(pipeline, tmp_path):
    out = tmp_path / "ai_edited.csv"
    code = main(["--mode", "edited", "--provider", "mock", "--n", "3", "--data-dir", str(pipeline["data"]),
                 "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out, keep_default_na=False)
    assert len(df) == 3
    assert set(df["provenance"]) == {"ai-edited"}
    assert (df["label"] == "").all()  # no binary ground truth for edited text
    assert df["prompt"].str.startswith(EDIT_PROMPT.split("{text}")[0][:40]).all()
    assert {"provider", "requested_model", "returned_model", "temperature", "max_tokens", "original_id"} <= set(df)
    test_ids = set(pd.read_csv(pipeline["data"] / "test.csv")["id"])
    assert set(df["original_id"]) <= test_ids


def test_generated_mode_answers_test_questions(pipeline, tmp_path):
    out = tmp_path / "gen.csv"
    code = main(["--mode", "generated", "--provider", "mock", "--n", "2", "--data-dir", str(pipeline["data"]),
                 "--hc3-path", str(pipeline["hc3"]), "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert set(df["provenance"]) == {"ai-generated-synthetic"} and (df["label"] == 1).all()
    assert df["prompt"].str.contains("QUESTION: Question number").all()
    assert (df["word_count"] >= 20).all()
