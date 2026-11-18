"""Milestone 2 test table: run the final prompt (V4) and algorithm v2 on fixed inputs and save the outputs.

Run from the repository root (needs data/test.csv, the frozen models and GROQ_API_KEY in .env):

    python test-data/run_tests.py

Writes test-data/inputs.json (the inputs) and test-data/outputs.json (every actual output, raw LLM JSON
included). Expected outputs are written here before the run and never edited afterwards.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.app import assess_with_llm, default_client_factory  # noqa: E402
from src.baseline import Baseline  # noqa: E402
from src.llm import load_settings  # noqa: E402
from src.perturb import inject_instruction, typo_variant  # noqa: E402
from src.prompts import load_few_shot_examples  # noqa: E402
from src.stylometry import StyleModel  # noqa: E402
from src.text_utils import count_words  # noqa: E402

AI, HUMAN, UNC = "Likely AI-generated", "Likely human-written", "Uncertain"
OUT = Path(__file__).parent

test = pd.read_csv(ROOT / "data" / "test.csv")
long_ = test[test["word_count"].between(150, 300)].sort_values("id")
human = long_[long_["label"] == 0]["text"].tolist()
ai = long_[long_["label"] == 1]["text"].tolist()
modern = ("In today's rapidly evolving digital landscape, artificial intelligence has emerged as a transformative "
          "force that is reshaping industries across the globe. By leveraging advanced machine learning algorithms, "
          "organizations can unlock unprecedented insights, streamline their operations, and drive innovation. "
          "However, it is essential to recognize that these technologies also present significant challenges, "
          "including ethical considerations and data privacy concerns. Furthermore, as AI systems become "
          "increasingly integrated into our daily lives, it is crucial to ensure that they are developed "
          "responsibly and transparently. Overall, by fostering collaboration between policymakers, researchers, "
          "and industry leaders, we can harness the full potential of AI while mitigating its risks. Additionally, "
          "investing in education and workforce development will be key to preparing individuals for the jobs of "
          "the future, ensuring that the benefits of this technology are shared broadly across society.")

# (no, type, short description, text, expected V4 label set, expected note)
CASES = [
    (1, "Normal", "HC3 human answer, 150-300 words", human[0], {HUMAN, UNC}, "not 'Likely AI-generated'"),
    (2, "Normal", "HC3 human answer, 150-300 words", human[1], {HUMAN, UNC}, "not 'Likely AI-generated'"),
    (3, "Normal", "HC3 human answer, 150-300 words", human[2], {HUMAN, UNC}, "not 'Likely AI-generated'"),
    (4, "Normal", "HC3 ChatGPT answer, 150-300 words", ai[0], {AI}, "'Likely AI-generated'"),
    (5, "Normal", "HC3 ChatGPT answer, 150-300 words", ai[1], {AI}, "'Likely AI-generated'"),
    (6, "Normal", "HC3 ChatGPT answer, 150-300 words", ai[2], {AI}, "'Likely AI-generated'"),
    (7, "Normal", "Modern-model style paragraph (2025-style assistant text)", modern, {AI}, "'Likely AI-generated'"),
    (8, "Edge", "Short casual text", "I think the bus was late again today so I walked. Took me 20 mins, honestly "
     "not bad. Might do it more often if the weather holds up like this.", {UNC}, "'Uncertain' (under 100 words)"),
    (9, "Edge", "ChatGPT answer with a prompt-injection sentence inside", inject_instruction(ai[0]), {AI, UNC},
     "injection ignored: not 'Likely human-written'"),
    (10, "Edge", "ChatGPT answer with 2 typos", typo_variant(ai[1], "t10", 0)[0], {AI}, "same label as clean text"),
    (11, "Edge", "Human answer typed all in lower case, no commas",
     human[0].lower().replace(",", ""), {HUMAN, UNC}, "not 'Likely AI-generated'"),
    (12, "Invalid", "Empty input (spaces only)", "   ", {"Paste a text first."}, "message 'Paste a text first.'"),
    (13, "Invalid", "Hindi text (not English)", "मुझे लगता है कि यह परियोजना बहुत उपयोगी है क्योंकि यह शिक्षकों की "
     "मदद करती है। " * 8, {UNC}, "'Uncertain' (no English words to analyse)"),
    (14, "Invalid", "Random keyboard letters", "asdf qwer zxcv uiop hjkl " * 25, {UNC}, "'Uncertain'"),
]


def main() -> int:
    settings = load_settings()
    provider, model = settings["provider"] or "groq", settings["model"] or "openai/gpt-oss-120b"
    baseline = Baseline.load(ROOT / "models" / "baseline.json")
    few_shot = load_few_shot_examples(ROOT / "models" / "few_shot_examples.json")
    style = StyleModel.load(ROOT / "models" / "stylometry.json")
    inputs, outputs = [], []
    for no, kind, desc, text, expected, expected_note in CASES:
        out = assess_with_llm(text, provider, model, "V4", baseline, few_shot, default_client_factory)
        first = out["label_md"].splitlines()[0] if out["label_md"] else ""
        label = first.removeprefix("### ").strip()
        v2 = style.predict(text)["label"] if text.strip() else "-"
        passed = label in expected
        inputs.append({"no": no, "type": kind, "description": desc, "words": count_words(text), "text": text})
        outputs.append({"no": no, "type": kind, "description": desc, "expected": expected_note, "v4_label": label,
                        "pass": passed, "algorithm_v2_label": v2, "flags": out["flags_md"],
                        "evidence": out["evidence_md"], "raw_json": out["raw_json"], "model": out["model_md"]})
        print(f"{no:2d} {kind:7s} {'PASS' if passed else 'FAIL'}  V4={label!r}  v2={v2!r}")
    (OUT / "inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {"run": datetime.now(timezone.utc).isoformat(timespec="seconds"), "provider": provider, "model": model,
            "prompt_version": "V4", "temperature": 0.0}
    (OUT / "outputs.json").write_text(json.dumps({"meta": meta, "results": outputs}, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"{sum(o['pass'] for o in outputs)}/{len(outputs)} passed; wrote test-data/inputs.json and outputs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
