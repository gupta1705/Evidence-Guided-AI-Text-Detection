"""Gradio interface: offline statistics plus an optional LLM assessment for a human reviewer.

* "Compute statistics" runs locally and instantly: the four features against the human/AI reference
  ranges, and the statistical baseline's score and label.
* "Ask the LLM" sends the text and its statistics to the selected provider with the chosen prompt version
  (default V4) and shows the label, evidence, limitations, JSON-validity and grounding flags, the raw JSON
  and the model that actually answered. A missing API key gives a friendly message; the statistics are
  still shown. Nothing is stored and the response cache is bypassed.

All presentation logic lives in pure functions so it can be unit-tested; :func:`build_app` only wires
them to components. Run with ``python -m src.app [--port 7861]`` (binds 127.0.0.1 only);
``python -m src.app --smoke`` builds the interface and exits without launching a server.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable

import pandas as pd

from src.baseline import Baseline, load_baseline
from src.config import (
    LABEL_AI,
    LABEL_HUMAN,
    MAX_WORDS,
    MIN_WORDS_DECISION,
    PROVIDERS,
    SUGGESTED_MODELS,
)
from src.features import FEATURES, compute_features, is_number
from src.grounding import check_grounding_for_prompt
from src.llm import LLMClient, LLMError, MissingAPIKeyError, load_settings
from src.parsing import parse_response
from src.stylometry import StyleModel, load_style_model
from src.prompts import (
    PROMPT_VERSIONS,
    VERSION_NAMES,
    FewShotUnavailable,
    build_messages,
    load_few_shot_examples,
    uses_json_mode,
)
from src.text_utils import count_words, normalize_text

BANNER = (
    "Use public or anonymised text only. When you click 'Ask the LLM', the text is sent to the selected provider."
)
DISCLAIMER = (
    "The statistics and the LLM assessment support a human reviewer. They never prove who wrote a text, and "
    "the baseline score is a distance difference, not a probability."
)
TABLE_HEADERS = ["Feature", "Value", "Human median (IQR)", "AI median (IQR)", "Closer to"]
LLM_PROVIDERS = [p for p in PROVIDERS if p != "mock"] + ["mock"]

ClientFactory = Callable[[str, str], LLMClient]


def default_client_factory(provider: str, model: str) -> LLMClient:
    """App client: no cache (privacy) and no rate-limit wait for single interactive calls."""
    return LLMClient(provider, model or None, rpm=0, cache_path=None, use_cache=False)


# --------------------------------------------------------------------------------------------------
# Pure presentation functions
# --------------------------------------------------------------------------------------------------
def word_count_text(text: str) -> str:
    """Live word-count line shown under the text box."""
    n = count_words(normalize_text(text or ""))
    msg = f"**Words:** {n}"
    if 0 < n < MIN_WORDS_DECISION:
        msg += f" - texts under {MIN_WORDS_DECISION} words are always labelled 'Uncertain'."
    elif n > MAX_WORDS:
        msg += f" - statistics and the prompt use the first {MAX_WORDS} words."
    return msg


def _fmt(value: float) -> str:
    return f"{value:.4f}" if is_number(value) else "-"


def _range(median: float, q1: float, q3: float) -> str:
    if not is_number(median):
        return "not supplied"
    return f"{_fmt(median)} ({_fmt(q1)}–{_fmt(q3)})"


def features_table(features: dict, baseline: Baseline | None) -> pd.DataFrame:
    """Rows: feature, value, human median (IQR), AI median (IQR), closer-to."""
    rows = [["word_count (full text)", str(features["word_count"]), "", "", ""],
            ["sentence_count (full text)", str(features["sentence_count"]), "", "", ""]]
    if baseline is not None:
        explained = baseline.explain(features)
    else:
        explained = [{"feature": n, "value": features.get(n, math.nan), "closer_to": "n/a"} for n in FEATURES]
    for r in explained:
        info = FEATURES[r["feature"]]
        rows.append([
            f"{info.label} ({r['feature']})",
            _fmt(r["value"]),
            _range(r.get("human_median"), r.get("human_q1"), r.get("human_q3")),
            _range(r.get("ai_median"), r.get("ai_q1"), r.get("ai_q3")),
            r["closer_to"],
        ])
    return pd.DataFrame(rows, columns=TABLE_HEADERS)


def baseline_markdown(features: dict, baseline: Baseline | None) -> str:
    """Baseline score, label and the reason for it."""
    if baseline is None:
        return ("**Statistical baseline:** not available - run `python -m src.data` and "
                "`python -m src.baseline fit` first.")
    pred = baseline.predict(features)
    score = f"{pred['score']:.3f}" if math.isfinite(pred["score"]) else "n/a"
    lines = [
        f"**Statistical baseline label:** {pred['label']}",
        f"Score {score} (distance to human reference - distance to AI reference; threshold "
        f"{baseline.threshold:.3f}, uncertainty margin {baseline.margin:.3f}). Reason: {pred['reason']}.",
        f"_{DISCLAIMER}_",
    ]
    notes = []
    if features.get("truncated"):
        notes.append(f"features use the first {baseline.max_words} words")
    if features.get("short_window"):
        notes.append("MATTR fell back to the plain type-token ratio (under 50 words)")
    if features.get("length_controlled") is False:
        notes.append("entropy is not length-controlled (under 100 words)")
    if notes:
        lines.insert(2, "Notes: " + "; ".join(notes) + ".")
    return "\n\n".join(lines)


def style_markdown(text: str, model: StyleModel | None) -> str:
    """Algorithm v2 label, P(AI) and the five features that moved the score most."""
    if model is None:
        return "**Algorithm v2:** not available - run `python -m src.stylometry fit` first."
    pred = model.predict(text)
    terms = ", ".join(f"{name} {value:+.2f}" for name, value in pred["top_contributions"])
    return (f"**Algorithm v2 label:** {pred['label']} (P(AI) {pred['probability_ai']:.0%}; 20 style features + "
            f"word log-odds, logistic weights). Strongest terms (+ = towards AI): {terms}.")


def compute_statistics(text: str, baseline: Baseline | None,
                       style_model: StyleModel | None = None) -> tuple[pd.DataFrame, str]:
    """Offline handler for "Compute statistics"."""
    if not (text or "").strip():
        return pd.DataFrame(columns=TABLE_HEADERS), "Paste a text first."
    features = compute_features(text)
    md = baseline_markdown(features, baseline)
    if style_model is not None:
        md = style_markdown(text, style_model) + "\n\n" + md
    return features_table(features, baseline), md


def label_markdown(label: str) -> str:
    """Rendered LLM label with a reminder that it is not proof."""
    hint = {
        LABEL_AI: "the statistics look more like the AI reference texts",
        LABEL_HUMAN: "the statistics look more like the human reference texts",
    }.get(label, "the evidence is insufficient or conflicting")
    return f"### {label}\n{hint}; manual review is recommended."


def assess_with_llm(
    text: str,
    provider: str,
    model: str,
    version: str,
    baseline: Baseline | None,
    few_shot: list[dict] | None,
    client_factory: ClientFactory = default_client_factory,
) -> dict:
    """Run one prompt version and return everything the "Ask the LLM" panel shows.

    Returns:
        Dict with ``label_md``, ``evidence_md``, ``flags_md``, ``raw_json`` and ``model_md``. Errors
        (missing key, provider failure, missing few-shot examples) are reported in ``label_md``.
    """
    empty = {"label_md": "", "evidence_md": "", "flags_md": "", "raw_json": "", "model_md": ""}
    if not (text or "").strip():
        return {**empty, "label_md": "Paste a text first."}
    features = compute_features(text)
    reference = baseline.reference if baseline is not None else None
    try:
        messages = build_messages(version, text, features, reference, few_shot)
    except FewShotUnavailable as exc:
        return {**empty, "label_md": f"**{version} is not available yet:** {exc}"}
    try:
        client = client_factory(provider, model)
        result = client.complete(messages, version=version, json_mode=uses_json_mode(version))
    except MissingAPIKeyError as exc:
        return {**empty, "label_md": f"**No API key for {provider}.** {exc} The statistics above still apply."}
    except LLMError as exc:
        return {**empty, "label_md": f"**The {provider} call failed:** {exc}"}

    parsed = parse_response(result.text)
    grounding = check_grounding_for_prompt(parsed.evidence, parsed.limitations, messages[-1]["content"])

    def note(item: dict) -> str:
        if not item["grounded"]:
            return " _(not grounded in the supplied statistics)_"
        if not item["strictly_grounded"]:
            return " _(does not quote the named feature's own supplied numbers)_"
        return ""

    evidence = "\n".join(f"- {item['text']}{note(item)}" for item in grounding.items) or "_No evidence returned._"
    limitations = "\n".join(f"- {x}" for x in parsed.limitations) or "_No limitations returned._"

    def pct(rate: float) -> str:
        return "n/a" if math.isnan(rate) else f"{rate:.0%}"

    flags = [
        f"JSON valid (strict schema): **{'yes' if parsed.json_valid else 'no'}**"
        + (f" ({'; '.join(parsed.errors)})" if parsed.errors else ""),
        f"Grounded evidence: **{pct(grounding.grounded_evidence_rate)}**; quoting the named feature's own "
        f"numbers: **{pct(grounding.strict_grounded_rate)}**",
        f"Invented numbers: **{'yes' if grounding.has_invented_number else 'no'}**"
        + (f" ({', '.join(f'{x:g}' for x in grounding.invented_numbers)})" if grounding.invented_numbers else ""),
    ]
    if not parsed.json_valid and parsed.parsed_label:
        flags.append(f"The response named '{parsed.parsed_label}', but it failed the schema, so it counts as "
                     "'Uncertain'.")
    if result.json_mode_fallback:
        flags.append("The provider rejected JSON mode; the call was repeated without it.")
    raw = json.dumps(parsed.parsed, indent=2, ensure_ascii=False) if parsed.parsed is not None else result.text
    return {
        "label_md": label_markdown(parsed.label),
        "evidence_md": f"**Evidence**\n{evidence}\n\n**Limitations**\n{limitations}",
        "flags_md": "  \n".join(flags),
        "raw_json": raw,
        "model_md": (f"Provider **{result.provider}**, requested model `{result.requested_model}`, "
                     f"answered by `{result.returned_model or 'unknown'}`, prompt {version} "
                     f"({PROMPT_VERSIONS[version]['title']}), {result.latency_s:.1f}s."),
    }


def model_choices(provider: str) -> tuple[list[str], str]:
    """Suggested models for a provider and the default one."""
    return SUGGESTED_MODELS.get(provider, []), PROVIDERS[provider].default_model


# --------------------------------------------------------------------------------------------------
# Gradio wiring
# --------------------------------------------------------------------------------------------------
def build_app(
    baseline: Baseline | None = None,
    few_shot: list[dict] | None = None,
    client_factory: ClientFactory = default_client_factory,
    load_artifacts: bool = True,
):
    """Construct (but do not launch) the Gradio Blocks app."""
    import gradio as gr

    if load_artifacts:
        baseline = baseline if baseline is not None else load_baseline()
        few_shot = few_shot if few_shot is not None else load_few_shot_examples()
    style_model = load_style_model() if load_artifacts else None
    settings = load_settings()
    provider0 = settings["provider"] if settings["provider"] in PROVIDERS else "groq"
    choices0, default0 = model_choices(provider0)
    model0 = settings["model"] or default0

    def on_stats(text):
        return compute_statistics(text, baseline, style_model)

    def on_ask(text, provider, model, version):
        table, base_md = compute_statistics(text, baseline, style_model)
        out = assess_with_llm(text, provider, model, version, baseline, few_shot, client_factory)
        return table, base_md, out["label_md"], out["evidence_md"], out["flags_md"], out["raw_json"], out["model_md"]

    def on_provider(provider):
        choices, default = model_choices(provider)
        return gr.update(choices=choices, value=default)

    with gr.Blocks(title="Evidence-Guided AI Text Detection") as demo:
        gr.Markdown("# Statistical AI Text Detection with Evidence-Guided Prompts")
        gr.Markdown(f"**{BANNER}**")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Text to review", lines=12, placeholder="Paste public or anonymised English text")
                words = gr.Markdown(word_count_text(""))
            with gr.Column(scale=1):
                provider = gr.Dropdown(LLM_PROVIDERS, value=provider0, label="Provider")
                model = gr.Dropdown(choices0, value=model0, label="Model", allow_custom_value=True)
                version = gr.Dropdown(VERSION_NAMES, value="V4", label="Prompt version",
                                      info="V0 baseline, V1 +role, V2 +grounding, V3 +few-shot, V4 +structured output")
                stats_btn = gr.Button("Compute statistics", variant="secondary")
                ask_btn = gr.Button("Ask the LLM", variant="primary")
        stats_table = gr.Dataframe(headers=TABLE_HEADERS, label="Statistics (offline, deterministic)",
                                   interactive=False, wrap=True)
        base_md = gr.Markdown()
        gr.Markdown("## LLM assessment")
        label_md = gr.Markdown()
        evidence_md = gr.Markdown()
        flags_md = gr.Markdown()
        with gr.Accordion("Raw JSON", open=False):
            raw_json = gr.Code(language="json", label="Raw response")
        model_md = gr.Markdown()

        text.change(word_count_text, inputs=text, outputs=words)
        provider.change(on_provider, inputs=provider, outputs=model)
        stats_btn.click(on_stats, inputs=text, outputs=[stats_table, base_md])
        ask_btn.click(on_ask, inputs=[text, provider, model, version],
                      outputs=[stats_table, base_md, label_md, evidence_md, flags_md, raw_json, model_md])
    return demo


def main(argv: list[str] | None = None) -> int:
    """Launch the app on 127.0.0.1 (or, with ``--smoke``, only build it)."""
    parser = argparse.ArgumentParser(prog="python -m src.app", description="Launch the Gradio review interface.")
    parser.add_argument("--port", type=int, default=7861, help="port on 127.0.0.1 (default: 7861)")
    parser.add_argument("--smoke", action="store_true",
                        help="build the interface (loading the frozen baseline if present) and exit without launching")
    args = parser.parse_args(argv)
    demo = build_app()
    if args.smoke:
        print(f"app constructed ({type(demo).__name__}); not launched (--smoke)")
        return 0
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
