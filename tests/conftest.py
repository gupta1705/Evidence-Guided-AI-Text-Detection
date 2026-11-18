"""Shared fixtures: a synthetic HC3-like corpus and a fitted pipeline in temporary directories.

The synthetic corpus copies HC3's formatting artefacts (pre-tokenised human answers with " . ", " , ",
" n't", spaced slashes and "[link] ( URL_0 )" placeholders; ChatGPT answers whose paragraph breaks are real
line breaks, literal "\\n" escapes or missing altogether) and gives the two classes different writing
statistics, so the whole pipeline can run offline, without the real dataset, a GPU, the network or an API key.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

SYLLABLES = ["ka", "lo", "mi", "ne", "ru", "ta", "po", "se", "di", "fa", "gu", "ve", "ba", "zo", "hi", "we",
             "ar", "el", "on", "us"]


def _vocab(rng: random.Random, n: int) -> list[str]:
    words: set[str] = set()
    while len(words) < n:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


def human_answer(rng: random.Random, vocab: list[str], n_words: int) -> str:
    """Pre-tokenised, bursty human-style answer (varied sentence lengths, wide vocabulary)."""
    sentences, total = [], 0
    while total < n_words:
        length = min(rng.choice([3, 5, 8, 13, 21, 30]), n_words - total) or 1
        words = [rng.choice(vocab) for _ in range(length)]
        if length > 6 and rng.random() < 0.5:
            words.insert(length // 2, ",")
        if rng.random() < 0.3:
            words[0:0] = ["i", "do", "n't", "think"]
            length += 4
        sentences.append(" ".join(words) + " .")
        total += length
    return " ".join(sentences)


PARAGRAPH_BREAKS = ("\n\n", "\\n\\n", "")  # real line breaks, literal escapes, lost breaks ("note.It")


def ai_answer(rng: random.Random, vocab: list[str], n_words: int, paragraph_break: str = "\n\n") -> str:
    """Uniform, repetitive ChatGPT-style answer with paragraph breaks."""
    sentences, total = [], 0
    while total < n_words:
        body = [rng.choice(vocab) for _ in range(6)]
        sentence = "It is important to note that " + " ".join(body) + "."
        sentences.append(sentence)
        total += 12
    paragraphs = [" ".join(sentences[i:i + 3]) for i in range(0, len(sentences), 3)]
    return paragraph_break.join(paragraphs)


def add_link_markup(answer: str) -> str:
    """Pre-tokenised HC3/ELI5 link placeholder and spaced slash, as in real human answers."""
    return answer + " See [ this page ] ( URL_0 ) and/or km / h ."


def write_synthetic_hc3(path: Path, n_questions: int = 150, seed: int = 1234) -> Path:
    """Write an HC3-like ``all.jsonl`` with a few deliberate edge cases."""
    rng = random.Random(seed)
    fmt = random.Random(seed + 1)  # formatting choices only, so the text statistics stay the same
    human_vocab = _vocab(rng, 900)
    ai_vocab = _vocab(rng, 60)
    sources = ["reddit_eli5", "finance", "medicine", "open_qa", "wiki_csai"]
    lines = []
    for q in range(n_questions):
        humans = []
        for _ in range(rng.choice([1, 2])):
            n = rng.randint(110, 240) if rng.random() < 0.6 else rng.randint(30, 95)
            answer = human_answer(rng, human_vocab, n)
            humans.append(add_link_markup(answer) if fmt.random() < 0.3 else answer)
        n_ai = rng.randint(110, 240) if rng.random() < 0.85 else rng.randint(40, 95)
        ais = [ai_answer(rng, ai_vocab, n_ai, fmt.choice(PARAGRAPH_BREAKS))]
        lines.append({"question": f"Question number {q}?", "human_answers": humans, "chatgpt_answers": ais,
                      "index": None, "source": sources[q % len(sources)]})
    # Edge cases: an empty answer, a too-short answer, a duplicate, a question without ChatGPT answers.
    lines[0]["human_answers"].append("")
    lines[1]["human_answers"].append("too short to keep .")
    lines[2]["human_answers"].append(lines[3]["human_answers"][0])
    lines[4]["chatgpt_answers"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in lines:
            fh.write(json.dumps(record) + "\n")
    return path


SMALL_SIZES = {"reference": 80, "validation": 40, "test": 40}


@pytest.fixture(scope="session")
def hc3_file(tmp_path_factory) -> Path:
    """Synthetic HC3 file shared by the session."""
    return write_synthetic_hc3(tmp_path_factory.mktemp("hc3") / "all.jsonl")


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, hc3_file) -> dict:
    """Build the dataset and fit the baseline once; returns the relevant paths and the summary."""
    from src import baseline as baseline_mod
    from src.data import build_dataset

    root = tmp_path_factory.mktemp("pipeline")
    paths = {
        "root": root,
        "data": root / "data",
        "model": root / "models" / "baseline.json",
        "few_shot": root / "models" / "few_shot_examples.json",
        "results": root / "results",
        "hc3": hc3_file,
    }
    paths["summary"] = build_dataset(hc3_file, paths["data"], sizes=SMALL_SIZES, seed=42)
    code = baseline_mod.main([
        "fit", "--data-dir", str(paths["data"]), "--model-path", str(paths["model"]),
        "--few-shot-path", str(paths["few_shot"]), "--results-dir", str(paths["results"]),
    ])
    assert code == 0
    return paths
