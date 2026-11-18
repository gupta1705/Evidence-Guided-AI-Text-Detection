"""End-to-end evaluation with the offline mock provider, plus the metric and bootstrap helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import MAX_TOKENS
from src.evaluate import (
    PROMPT_METRICS,
    bootstrap_weights,
    main,
    messages_sha256,
    metrics_with_ci,
    ratio_with_ci,
    robustness_sample,
    sample_prompt_set,
    wilson_interval,
)


def _args(pipeline, results, *extra):
    return [
        "--provider", "mock", "--data-dir", str(pipeline["data"]), "--model-path", str(pipeline["model"]),
        "--few-shot-path", str(pipeline["few_shot"]), "--results-dir", str(results),
        "--cache-path", str(results / "cache.jsonl"), "--bootstrap", "200", *extra,
    ]


def test_end_to_end_mock_run(pipeline, tmp_path):
    results = tmp_path / "results"
    assert main(_args(pipeline, results, "--n", "12", "--robustness-n", "8")) == 0
    for name in ("baseline_metrics.csv", "baseline_metrics.md", "prompt_comparison.csv", "prompt_comparison.md",
                 "prompt_comparison.png", "robustness_prompts.csv", "robustness_prompts.md", "llm_runs.jsonl",
                 "run_metadata.json", "prompt_sample_ids.csv"):
        assert (results / name).is_file(), name

    comparison = pd.read_csv(results / "prompt_comparison.csv")
    assert comparison["version"].tolist() == ["V0", "V1", "V2", "V3", "V4", "baseline"]
    llm_rows = comparison[comparison["version"] != "baseline"]
    assert (llm_rows["n"] == 12).all()
    assert (llm_rows["json_validity"] == 1.0).all()
    assert (llm_rows["counts_compliance"] == 1.0).all()
    assert (llm_rows["grounded_evidence_rate"] == 1.0).all()
    assert (llm_rows["strict_grounded_rate"] == 1.0).all()
    assert (llm_rows["short_text_compliance"] == 1.0).all()
    for metric in PROMPT_METRICS:
        assert {metric, f"{metric}_ci_low", f"{metric}_ci_high", f"{metric}_n"} <= set(comparison.columns)
    ok = llm_rows.dropna(subset=["coverage_ci_low"])
    assert (ok["coverage_ci_low"] <= ok["coverage"]).all() and (ok["coverage"] <= ok["coverage_ci_high"]).all()
    # An estimate of exactly 1 gets a Wilson interval, not a zero-width bootstrap interval.
    assert (llm_rows["json_validity_ci_low"] < 1.0).all() and (llm_rows["json_validity_ci_high"] == 1.0).all()
    assert np.isnan(comparison.set_index("version").loc["baseline", "counts_compliance"])

    robustness = pd.read_csv(results / "robustness_prompts.csv")
    assert set(zip(robustness["version"], robustness["set"])) == {
        (v, s) for v in ("V0", "V4", "baseline") for s in ("typo", "injection", "style")
    }
    runs = [json.loads(line) for line in (results / "llm_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(runs) == 12 * 5 + 2 * 3 * 8
    first = runs[0]
    for key in ("version", "provider", "requested_model", "returned_model", "temperature", "max_tokens",
                "timestamp", "latency_s", "wait_s", "usage", "raw_text", "pred_label", "parsed_label", "json_valid",
                "request_key", "messages_sha256", "strict_grounded_rate"):
        assert key in first
    assert {r["temperature"] for r in runs} == {0.0} and {r["max_tokens"] for r in runs} == {MAX_TOKENS}
    # Every logged call can be tied to the exact prompt that produced it.
    prompts = {p["messages_sha256"]: p for p in map(json.loads, (results / "llm_prompts.jsonl").open(encoding="utf-8"))}
    assert {r["messages_sha256"] for r in runs} == set(prompts)
    assert all(messages_sha256(p["messages"]) == h for h, p in prompts.items())
    meta = json.loads((results / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["provider"] == "mock" and meta["returned_models"] == ["mock-evidence-v1"]
    assert len(meta["baseline_sha256"]) == 64 and len(meta["few_shot_sha256"]) == 64
    assert meta["robustness"]["sets"] == ["typo", "injection", "style"]
    notes = (results / "prompt_comparison.md").read_text(encoding="utf-8")
    assert "decided examples" in notes and "mock-evidence-v1" in notes and "Wilson" in notes

    # A second run reuses the prompt file instead of duplicating it.
    assert main(_args(pipeline, results, "--n", "12", "--no-robustness")) == 0
    assert sum(1 for _ in (results / "llm_prompts.jsonl").open(encoding="utf-8")) == len(prompts)


def test_malformed_responses_count_as_invalid_and_uncertain(pipeline, tmp_path):
    results = tmp_path / "bad"
    args = _args(pipeline, results, "--n", "6", "--versions", "V0,V2", "--no-robustness",
                 "--mock-malformed-rate", "1.0")
    assert main(args) == 0
    comparison = pd.read_csv(results / "prompt_comparison.csv").set_index("version")
    assert comparison.loc["V0", "json_validity"] == 0.0
    assert comparison.loc["V0", "coverage"] == 0.0
    assert not (results / "robustness_prompts.csv").exists()


def test_dry_run_prints_every_version(pipeline, capsys):
    args = ["--dry-run", "--data-dir", str(pipeline["data"]), "--model-path", str(pipeline["model"]),
            "--few-shot-path", str(pipeline["few_shot"])]
    assert main(args) == 0
    out = capsys.readouterr().out
    for version in ("V0:", "V1:", "V2:", "V3:", "V4:"):
        assert version in out
    assert out.count("<<<BEGIN_TEXT") >= 5 + 6  # one per version plus the V3/V4 worked examples
    assert "not available" not in out


def test_dry_run_without_artifacts(tmp_path, capsys):
    args = ["--dry-run", "--data-dir", str(tmp_path), "--model-path", str(tmp_path / "none.json"),
            "--few-shot-path", str(tmp_path / "none.json")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Human/AI reference ranges: not supplied." in out
    assert "not available" in out  # V3/V4 need few-shot examples


def test_missing_baseline_is_reported(pipeline, tmp_path, capsys):
    args = ["--provider", "mock", "--data-dir", str(pipeline["data"]), "--model-path", str(tmp_path / "x.json")]
    assert main(args) == 2
    assert "baseline fit" in capsys.readouterr().err


def test_empty_test_split_is_reported(pipeline, tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.csv").write_text((pipeline["data"] / "test.csv").read_text(encoding="utf-8").splitlines()[0] + "\n",
                                   encoding="utf-8")
    args = ["--provider", "mock", "--data-dir", str(data), "--model-path", str(pipeline["model"]), "--n", "2"]
    assert main(args) == 2
    assert "no rows" in capsys.readouterr().err


def test_unknown_provider_from_environment_is_reported(pipeline, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    args = ["--data-dir", str(pipeline["data"]), "--model-path", str(pipeline["model"]),
            "--few-shot-path", str(pipeline["few_shot"]), "--results-dir", str(tmp_path), "--n", "2"]
    assert main(args) == 1
    assert "unknown provider 'foo'" in capsys.readouterr().err


def test_mock_defaults_to_its_own_results_dir(pipeline, tmp_path, monkeypatch):
    import src.evaluate as evaluate

    mock_dir, real_dir = tmp_path / "results_mock", tmp_path / "results"
    monkeypatch.setattr(evaluate, "MOCK_RESULTS_DIR", mock_dir)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", real_dir)
    args = ["--provider", "mock", "--data-dir", str(pipeline["data"]), "--model-path", str(pipeline["model"]),
            "--few-shot-path", str(pipeline["few_shot"]), "--cache-path", str(tmp_path / "cache.jsonl"),
            "--bootstrap", "50", "--n", "4", "--versions", "V0", "--no-robustness"]
    assert main(args) == 0
    assert (mock_dir / "prompt_comparison.csv").is_file() and (mock_dir / "llm_runs.jsonl").is_file()
    assert not real_dir.exists()  # an offline check never touches the real results


def test_calls_are_logged_as_they_complete(pipeline, tmp_path, monkeypatch):
    from src.llm import LLMClient

    results = tmp_path / "inc"
    runs, prompts = results / "llm_runs.jsonl", results / "llm_prompts.jsonl"
    seen = []
    original = LLMClient.complete

    def spy(self, messages, **kwargs):
        # lines already in the run log, and whether every logged call's prompt is already stored
        logged = [json.loads(line) for line in runs.open(encoding="utf-8")] if runs.exists() else []
        stored = [json.loads(line) for line in prompts.open(encoding="utf-8")] if prompts.exists() else []
        seen.append((len(logged), {r["messages_sha256"] for r in logged} <= {p["messages_sha256"] for p in stored}))
        return original(self, messages, **kwargs)

    monkeypatch.setattr(LLMClient, "complete", spy)
    assert main(_args(pipeline, results, "--n", "4", "--versions", "V0,V4", "--no-robustness")) == 0
    # each call finds every earlier call (and its prompt) on disk already, so a killed run keeps them
    assert seen == [(i, True) for i in range(8)]


def test_update_readme_skips_robustness_table_of_an_earlier_run(tmp_path):
    from src.evaluate import RESULTS_END, RESULTS_START, update_readme

    results = tmp_path / "results"
    results.mkdir()
    for name in ("baseline_metrics", "prompt_comparison", "robustness_prompts"):
        (results / f"{name}.md").write_text(f"# {name}\n\ntable of {name}\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{RESULTS_START}\nold\n{RESULTS_END}\nend\n", encoding="utf-8")

    meta = {"run_id": "r1", "robustness": {"enabled": False, "n": 0}}
    (results / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert update_readme(results, readme)
    text = readme.read_text(encoding="utf-8")
    assert "table of prompt_comparison" in text and "table of robustness_prompts" not in text
    assert text.startswith("intro\n") and text.endswith(f"{RESULTS_END}\nend\n")

    meta["robustness"]["enabled"] = True
    (results / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert update_readme(results, readme)
    assert "table of robustness_prompts" in readme.read_text(encoding="utf-8")


def test_extra_set_reported_separately(pipeline, tmp_path):
    extra = tmp_path / "ai_edited.csv"
    from src.data import load_split

    test = load_split("test", pipeline["data"]).head(4)
    pd.DataFrame({"id": [f"e{i}" for i in range(4)], "question_id": test["question_id"], "text": test["text"],
                  "provenance": "ai-edited", "label": ""}).to_csv(extra, index=False)
    results = tmp_path / "res"
    assert main(_args(pipeline, results, "--n", "4", "--versions", "V0", "--no-robustness", "--extra", str(extra))) == 0
    table = pd.read_csv(results / "extra_ai_edited.csv")
    assert set(table["version"]) == {"V0", "baseline"} and set(table["provenance"]) == {"ai-edited"}
    assert np.allclose(table[["share_ai", "share_human", "share_uncertain"]].sum(axis=1), 1.0)


def test_sample_prompt_set_is_balanced_with_short_texts(pipeline):
    from src.data import load_split

    test = load_split("test", pipeline["data"])
    sample = sample_prompt_set(test, n=20, short_share=0.2, seed=42)
    assert len(sample) == 20 and (sample["label"] == 0).sum() == 10
    available = [int(((test["label"] == lab) & (test["word_count"] < 100)).sum()) for lab in (0, 1)]
    assert (sample["word_count"] < 100).sum() == sum(min(2, a) for a in available)  # 20% of 20, split by class
    assert sample["id"].tolist() == sample_prompt_set(test, n=20, short_share=0.2, seed=42)["id"].tolist()
    rob = robustness_sample(sample, n=8)
    assert len(rob) == 8 and set(rob["id"]) <= set(sample["id"])


def test_bootstrap_weights_resample_questions():
    qids = np.array([1, 1, 2, 3])
    w = bootstrap_weights(qids, n_boot=50, seed=0)
    assert w.shape == (50, 4)
    assert (w[:, 0] == w[:, 1]).all()  # rows of the same question share a weight
    assert np.allclose(w[:, [0, 2, 3]].sum(axis=1), 3)


def test_metrics_with_ci_point_estimates():
    df = pd.DataFrame({
        "question_id": [1, 2, 3, 4],
        "label_true": [0, 0, 1, 1],
        "pred_label": ["Likely human-written", "Likely AI-generated", "Likely AI-generated", "Uncertain"],
        "short": [False, False, False, True],
        "json_valid": [True, True, False, True],
        "grounded_evidence_rate": [1.0, 0.5, float("nan"), 1.0],
        "n_evidence": [2, 2, 0, 1],
        "n_limitations": [1, 1, 0, 1],
        "has_invented_number": [False, True, False, False],
        "baseline_label": ["Likely human-written", "Uncertain", "Likely AI-generated", "Uncertain"],
    })
    m = metrics_with_ci(df, PROMPT_METRICS, n_boot=100, seed=0)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["decided_accuracy"] == pytest.approx(2 / 3)
    assert m["human_fpr"] == pytest.approx(0.5)
    assert m["json_validity"] == pytest.approx(0.75)
    assert m["grounded_evidence_rate"] == pytest.approx(2.5 / 3)
    assert m["invented_number_rate"] == pytest.approx(1 / 3)
    assert m["short_text_compliance"] == 1.0
    assert m["agreement_with_baseline"] == pytest.approx(0.75)
    assert m["coverage_ci_low"] <= m["coverage"] <= m["coverage_ci_high"]
    assert np.isnan(m["counts_compliance"]) and np.isnan(m["strict_grounded_rate"])  # columns absent
    assert m["decided_accuracy_n"] == 3 and m["human_fpr_n"] == 2 and m["short_text_compliance_n"] == 1
    # short_text_compliance is 1/1: the CI is the Wilson interval for one text, not [1, 1].
    assert m["short_text_compliance_ci_low"] == pytest.approx(wilson_interval(1.0, 1)[0])
    assert m["short_text_compliance_ci_high"] == 1.0


def test_wilson_interval_for_extreme_estimates():
    lo, hi = wilson_interval(0.0, 30)
    assert lo == 0.0 and hi == pytest.approx(0.1135, abs=1e-3)  # close to the exact 0.116 for 0/30
    lo, hi = wilson_interval(1.0, 6)
    assert hi == 1.0 and lo == pytest.approx(0.6097, abs=1e-3)
    assert all(np.isnan(wilson_interval(0.5, 0)))
    # 0 false positives among 30 human texts: no zero-width interval even with a bootstrap.
    num, den = np.zeros(30), np.ones(30)
    r = ratio_with_ci(num, den, bootstrap_weights(np.arange(30), 200, 0))
    assert r["value"] == 0.0 and r["n"] == 30 and r["ci_high"] == pytest.approx(0.1135, abs=1e-3)
    # A non-extreme estimate keeps the bootstrap percentile interval.
    num = np.array([1.0, 0.0] * 15)
    r = ratio_with_ci(num, den, bootstrap_weights(np.arange(30), 200, 0))
    assert r["ci_low"] < 0.5 < r["ci_high"]


def test_label_only_metrics_use_parsed_label():
    from src.config import LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN
    from src.evaluate import _parts
    df = pd.DataFrame({
        "label_true": [1, 0, 1, 0],
        "pred_label": [LABEL_UNCERTAIN] * 4,                      # strict view: all schema-invalid
        "parsed_label": [LABEL_AI, LABEL_AI, LABEL_HUMAN, None],   # labels the responses actually named
    })
    parts = _parts(df)
    ratio = lambda k: parts[k][0].sum() / parts[k][1].sum()
    assert ratio("coverage") == 0.0
    assert ratio("label_validity") == 0.75 and ratio("lenient_coverage") == 0.75
    assert abs(ratio("lenient_decided_accuracy") - 1 / 3) < 1e-9
    assert ratio("lenient_human_fpr") == 0.5


def test_robustness_table_label_only_flips():
    from src.config import LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN
    from src.evaluate import robustness_table
    clean = pd.DataFrame({"version": ["V0", "V0"], "id": ["a", "b"], "question_id": [1, 2],
                          "pred_label": [LABEL_UNCERTAIN] * 2, "parsed_label": [LABEL_AI, LABEL_HUMAN]})
    pert = pd.DataFrame({"version": ["V0", "V0"], "set": ["injection"] * 2, "id": ["a", "b"], "question_id": [1, 2],
                         "pred_label": [LABEL_UNCERTAIN] * 2, "parsed_label": [LABEL_HUMAN, LABEL_HUMAN]})
    row = robustness_table(clean, pert, n_boot=0, seed=0).iloc[0]
    assert row["label_flip_rate"] == 0.0                      # strict view sees nothing
    assert row["lenient_label_flip_rate"] == 0.5              # one named label changed
    assert row["lenient_flip_to_human_rate"] == 1.0           # the AI label became human
