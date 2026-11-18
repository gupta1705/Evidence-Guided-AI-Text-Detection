"""Load HC3 (English), normalise and clean every answer, and split by question.

Splits (balanced human/AI, question-disjoint, seed 42):

* ``reference``  - 3,000 texts (1,500 / 1,500): reference statistics for the baseline and the prompts.
* ``validation`` -   600 texts (300 / 300):     threshold, uncertainty margin and few-shot examples.
* ``test``       -   600 texts (300 / 300):     held out for the final evaluation only.

One human and one ChatGPT answer are taken from each selected question, so every question contributes to
exactly one split. Output: ``data/{reference,validation,test}.csv`` and ``data/summary.json``.

Every answer is prepared with :func:`prepare_text`: :func:`src.text_utils.normalize_text` plus the removal
of HC3's anonymised link placeholders (``[anchor] (URL_0)`` -> ``anchor``; bare ``URL_0`` removed), which
occur in about 12% (11.6%) of human answers and almost never (0.1%) in ChatGPT answers.

Usage::

    python -m src.data                 # full build
    python -m src.data --limit 280     # quick build with ~280 texts in total (sizes scaled 5:1:1)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    CACHED_HC3_PATH,
    DATA_DIR,
    DEFAULT_HC3_PATH,
    HC3_URL,
    MIN_WORDS_DATA,
    PROVENANCE_AI,
    PROVENANCE_HUMAN,
    SEED,
    SPLIT_SIZES,
    SPLITS,
    WORD_BANDS,
)
from src.text_utils import NORMALISATION_RULES, count_words, normalize_text

COLUMNS: list[str] = ["id", "question_id", "source", "label", "provenance", "text", "word_count"]

# Formatting artefacts that must be (near) absent from both classes after preparation (< 2% each; tested).
# Values are regular expressions; the first five are the shared project patterns.
ARTEFACT_PATTERNS: dict[str, str] = {
    "space_period": re.escape(" . "),
    "space_comma": re.escape(" , "),
    "space_nt": re.escape(" n't"),
    "double_space": re.escape("  "),
    "newline": r"\n",
    "escaped_newline": r"\\[nrt](?![a-z])",
    "glued_sentence": r"[a-z0-9)\]\"'][.!?][A-Z][a-z]",
    "spaced_slash": r"\w / \w",
    "spaced_single_quote": r"(?<!\S)'(?=[\s.,!?;:)\]]|$)",
    "curly_apostrophe": "’",
    "url_placeholder": r"URL_\d+",
}
# Residual class differences reported for transparency only. They are genuine style or source cues
# (punctuation choice, Reddit markdown), not tokenisation artefacts, so normalisation does not touch them.
INFO_PATTERNS: dict[str, str] = {
    "semicolon": ";",
    "double_hyphen": "--",
    "markdown_asterisk": r"\*",
    "quote_marker": r"(?<!\S)>",
}
_ALL_PATTERNS: dict[str, re.Pattern] = {k: re.compile(v) for k, v in {**ARTEFACT_PATTERNS, **INFO_PATTERNS}.items()}

# HC3 replaced links with placeholders: "[anchor] (URL_0)", "(URL_0)", "URL_0".
_URL_LINK = re.compile(r"\[([^\[\]]*)\]\s*\(\s*URL_\d+\s*\)")
_URL_BRACKETED = re.compile(r"[(\[\"]\s*URL_\d+\s*[)\]\"]")
_URL_BARE = re.compile(r"\bURL_\d+\b")


def remove_url_placeholders(text: str) -> str:
    """Drop HC3's anonymised link placeholders, keeping the anchor text of ``[anchor] (URL_n)`` links."""
    text = _URL_LINK.sub(r"\1", text)
    text = _URL_BRACKETED.sub(" ", text)
    return _URL_BARE.sub(" ", text)


def prepare_text(raw: str) -> str:
    """Normalise one HC3 answer and remove its link placeholders (the result is normalised again)."""
    return normalize_text(remove_url_placeholders(normalize_text(raw)))


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------
def resolve_hc3_path(path: str | Path | None = None, *, download: bool = True) -> Path:
    """Find the HC3 ``all.jsonl`` file.

    An explicit ``path`` must exist. Otherwise ``data/raw/all.jsonl`` is used, then the cached copy at
    ``~/.cache/hc3/all.jsonl``, and as a last resort the file is downloaded into ``data/raw/``.
    """
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"HC3 file not found: {p}")
        return p
    if DEFAULT_HC3_PATH.is_file():
        return DEFAULT_HC3_PATH
    if CACHED_HC3_PATH.is_file():
        return CACHED_HC3_PATH
    if not download:
        raise FileNotFoundError(f"HC3 file not found at {DEFAULT_HC3_PATH} or {CACHED_HC3_PATH}")
    return download_hc3(DEFAULT_HC3_PATH)


def download_hc3(dest: Path, url: str = HC3_URL) -> Path:
    """Download HC3 ``all.jsonl`` (about 74 MB, CC-BY-SA-4.0) to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading HC3 from {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(dest)
    return dest


def iter_hc3(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(question_id, record)`` for each line; the line index is the question id."""
    with open(path, encoding="utf-8") as fh:
        for qid, line in enumerate(fh):
            line = line.strip()
            if line:
                yield qid, json.loads(line)


def artefact_rates(texts: Iterable[str]) -> dict[str, float]:
    """Share of ``texts`` matching each pattern in ``ARTEFACT_PATTERNS`` and ``INFO_PATTERNS``."""
    counts: Counter[str] = Counter()
    n = 0
    for text in texts:
        n += 1
        for key, pattern in _ALL_PATTERNS.items():
            if pattern.search(text):
                counts[key] += 1
    return {key: (counts[key] / n if n else 0.0) for key in _ALL_PATTERNS}


def load_answers(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read every HC3 answer, prepare it (:func:`prepare_text`) and count words.

    Returns:
        A DataFrame with one row per answer (columns ``question_id, source, label, provenance,
        answer_index, text, word_count``) and a dict of raw statistics, including artefact rates
        measured on the raw (un-normalised) answers.
    """
    rows: list[dict] = []
    raw_counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    raw_by_class: dict[str, Counter[str]] = {"human": Counter(), "ai": Counter()}
    for qid, record in iter_hc3(path):
        source = str(record.get("source", "unknown"))
        sources[source] += 1
        for key, label, provenance in (
            ("human_answers", 0, PROVENANCE_HUMAN),
            ("chatgpt_answers", 1, PROVENANCE_AI),
        ):
            cls = "ai" if label else "human"
            for j, answer in enumerate(record.get(key) or []):
                raw = answer if isinstance(answer, str) else ""
                raw_counts[cls] += 1
                for name, pattern in _ALL_PATTERNS.items():
                    if pattern.search(raw):
                        raw_by_class[cls][name] += 1
                text = prepare_text(raw)
                rows.append(
                    {
                        "question_id": qid,
                        "source": source,
                        "label": label,
                        "provenance": provenance,
                        "answer_index": j,
                        "text": text,
                        "word_count": count_words(text),
                    }
                )
    df = pd.DataFrame(rows, columns=["question_id", "source", "label", "provenance", "answer_index", "text", "word_count"])
    raw_rates = {
        cls: {name: (raw_by_class[cls][name] / raw_counts[cls] if raw_counts[cls] else 0.0) for name in _ALL_PATTERNS}
        for cls in ("human", "ai")
    }
    stats = {
        "questions": sum(sources.values()),
        "questions_by_source": dict(sorted(sources.items())),
        "human_answers": raw_counts["human"],
        "ai_answers": raw_counts["ai"],
        "artefact_rates_before_normalisation": raw_rates,
    }
    return df, stats


# --------------------------------------------------------------------------------------------------
# Cleaning and splitting
# --------------------------------------------------------------------------------------------------
def clean_answers(df: pd.DataFrame, min_words: int = MIN_WORDS_DATA) -> tuple[pd.DataFrame, dict]:
    """Drop empty texts, texts under ``min_words`` words, label-conflicting texts and exact duplicates.

    A text that occurs with both labels is removed entirely; of the remaining exact duplicates (after
    normalisation) only the first occurrence is kept.
    """
    stats: dict[str, int] = {}
    empty = df["text"].str.len() == 0
    stats["removed_empty"] = int(empty.sum())
    df = df.loc[~empty]
    short = df["word_count"] < min_words
    stats["removed_under_min_words"] = int(short.sum())
    df = df.loc[~short]

    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting = set(labels_per_text.index[labels_per_text > 1])
    is_conflict = df["text"].isin(conflicting)
    stats["removed_label_conflicts"] = int(is_conflict.sum())
    df = df.loc[~is_conflict]

    df = df.sort_values(["question_id", "label", "answer_index"], kind="stable")
    dup = df.duplicated(subset="text", keep="first")
    stats["removed_duplicates"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)
    stats["remaining_human"] = int((df["label"] == 0).sum())
    stats["remaining_ai"] = int((df["label"] == 1).sum())
    stats["min_words"] = min_words
    return df, stats


def scaled_sizes(limit: int | None, sizes: dict[str, int] | None = None) -> dict[str, int]:
    """Scale the split sizes so they total about ``limit`` texts (even, at least 4 per split)."""
    sizes = dict(sizes or SPLIT_SIZES)
    if not limit:
        return sizes
    total = sum(sizes.values())
    return {k: max(4, 2 * round(v * limit / total / 2)) for k, v in sizes.items()}


def split_by_question(df: pd.DataFrame, sizes: dict[str, int], seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Assign whole questions to balanced splits.

    Questions with at least one human and one AI answer are shuffled (seeded) and consumed in order; each
    gives one randomly chosen human and one randomly chosen AI answer to the current split. If paired
    questions run out, single-class questions fill the remaining quota of a class. No question appears in
    two splits.

    Raises:
        ValueError: if there are not enough answers to fill the requested sizes.
    """
    rng = random.Random(seed)
    by_q: dict[int, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for idx, qid, label in zip(df.index, df["question_id"], df["label"]):
        by_q[int(qid)][int(label)].append(idx)

    paired = sorted(q for q, d in by_q.items() if d[0] and d[1])
    human_only = sorted(q for q, d in by_q.items() if d[0] and not d[1])
    ai_only = sorted(q for q, d in by_q.items() if d[1] and not d[0])
    for queue in (paired, human_only, ai_only):
        rng.shuffle(queue)

    selected: dict[str, list[int]] = {}
    for split in SPLITS:
        size = sizes[split]
        if size % 2:
            raise ValueError(f"split size for {split!r} must be even (balanced classes), got {size}")
        need = {0: size // 2, 1: size // 2}
        chosen: list[int] = []
        while need[0] and need[1] and paired:
            q = paired.pop()
            chosen.append(rng.choice(by_q[q][0]))
            chosen.append(rng.choice(by_q[q][1]))
            need[0] -= 1
            need[1] -= 1
        for label, queue in ((0, human_only), (1, ai_only)):
            while need[label] and (queue or paired):
                q = queue.pop() if queue else paired.pop()
                chosen.append(rng.choice(by_q[q][label]))
                need[label] -= 1
        if need[0] or need[1]:
            raise ValueError(
                f"not enough answers to build the {split!r} split "
                f"(missing {need[0]} human and {need[1]} AI texts); use smaller sizes"
            )
        selected[split] = chosen

    out: dict[str, pd.DataFrame] = {}
    for split, idx in selected.items():
        part = df.loc[idx].copy()
        part["id"] = [
            f"q{q}_{'a' if lab else 'h'}{j}"
            for q, lab, j in zip(part["question_id"], part["label"], part["answer_index"])
        ]
        part = part.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out[split] = part[COLUMNS]
    return out


def word_band(n_words: float) -> str:
    """Word-count band label: ``<100``, ``100-199`` or ``>=200``."""
    for name, low, high in WORD_BANDS:
        if low <= n_words < high:
            return name
    return WORD_BANDS[-1][0]


# --------------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------------
def _class_name(label: int) -> str:
    return "ai" if int(label) == 1 else "human"


def split_summary(df: pd.DataFrame) -> dict:
    """Counts by class, source x class and word-count band x class for one split."""
    classes = df["label"].map(_class_name)
    by_source: dict[str, dict[str, int]] = {}
    for (source, cls), n in Counter(zip(df["source"], classes)).items():
        by_source.setdefault(source, {"human": 0, "ai": 0})[cls] = n
    bands: dict[str, dict[str, int]] = {name: {"human": 0, "ai": 0} for name, _, _ in WORD_BANDS}
    for wc, cls in zip(df["word_count"], classes):
        bands[word_band(wc)][cls] += 1
    return {
        "n": int(len(df)),
        "questions": int(df["question_id"].nunique()),
        "by_class": {"human": int((df["label"] == 0).sum()), "ai": int((df["label"] == 1).sum())},
        "by_source": dict(sorted(by_source.items())),
        "word_count_bands": bands,
        "median_words": {
            "human": float(df.loc[df["label"] == 0, "word_count"].median()) if (df["label"] == 0).any() else None,
            "ai": float(df.loc[df["label"] == 1, "word_count"].median()) if (df["label"] == 1).any() else None,
        },
    }


def disjointness(splits: dict[str, pd.DataFrame]) -> dict:
    """Question-id and text overlap between every pair of splits (all must be zero)."""
    names = list(splits)
    q_overlap: dict[str, int] = {}
    t_overlap: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}-{b}"
            q_overlap[key] = len(set(splits[a]["question_id"]) & set(splits[b]["question_id"]))
            t_overlap[key] = len(set(splits[a]["text"]) & set(splits[b]["text"]))
    return {
        "question_overlap": q_overlap,
        "text_overlap": t_overlap,
        "disjoint": not any(q_overlap.values()) and not any(t_overlap.values()),
    }


def build_dataset(
    hc3_path: str | Path | None = None,
    out_dir: Path = DATA_DIR,
    sizes: dict[str, int] | None = None,
    seed: int = SEED,
    min_words: int = MIN_WORDS_DATA,
) -> dict:
    """Run the full pipeline: load, normalise, clean, split, write CSVs and ``summary.json``.

    Returns:
        The summary dict that was written to ``out_dir / "summary.json"``.
    """
    path = resolve_hc3_path(hc3_path)
    sizes = dict(sizes or SPLIT_SIZES)
    answers, raw_stats = load_answers(path)
    cleaned, clean_stats = clean_answers(answers, min_words=min_words)
    clean_stats["paired_questions"] = int(
        (cleaned.groupby("question_id")["label"].nunique() == 2).sum()
    )
    splits = split_by_question(cleaned, sizes, seed=seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    combined = pd.concat(list(splits.values()), ignore_index=True)
    before = raw_stats.pop("artefact_rates_before_normalisation")
    after = {
        cls: artefact_rates(combined.loc[combined["label"] == lab, "text"])
        for cls, lab in (("human", 0), ("ai", 1))
    }
    summary = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "hc3_path": str(path),
        "sizes_requested": sizes,
        "raw": raw_stats,
        "normalisation_rules": list(NORMALISATION_RULES),
        "hc3_cleaning": "link placeholders removed after normalisation: '[anchor] (URL_n)' -> 'anchor'; "
                        "'(URL_n)', '[URL_n]', '\"URL_n\"' and bare 'URL_n' removed",
        "tokenizer": r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)? on lower-cased normalised text",
        "cleaning": clean_stats,
        "splits": {name: split_summary(part) for name, part in splits.items()},
        "artefact_rates": {
            "patterns": dict(ARTEFACT_PATTERNS),
            "info_patterns": dict(INFO_PATTERNS),
            "note": "patterns are regular expressions; 'patterns' must stay below 2% for both classes after "
                    "preparation; 'info_patterns' are residual style/source differences, reported only",
            "before_normalisation_all_answers": before,
            "after_normalisation_dataset": after,
        },
        "disjointness": disjointness(splits),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


# --------------------------------------------------------------------------------------------------
# Helpers used by the other modules
# --------------------------------------------------------------------------------------------------
def load_split(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one split CSV written by :func:`build_dataset`."""
    path = Path(data_dir) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data` first")
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    df["question_id"] = df["question_id"].astype(int)
    df["label"] = df["label"].astype(int)
    df["word_count"] = df["word_count"].astype(int)
    return df


def sample_balanced(df: pd.DataFrame, n: int | None, seed: int = SEED) -> pd.DataFrame:
    """Return ``n`` rows with (as far as possible) equal human/AI counts; all rows if ``n`` is falsy."""
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    half = n // 2
    humans = df[df["label"] == 0]
    ais = df[df["label"] == 1]
    n_h = min(len(humans), n - half)
    n_a = min(len(ais), n - n_h)
    n_h = min(len(humans), n - n_a)
    parts = [humans.sample(n=n_h, random_state=seed), ais.sample(n=n_a, random_state=seed)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data",
        description="Build question-disjoint, balanced HC3 splits (reference / validation / test).",
    )
    parser.add_argument("--hc3-path", type=Path, default=None,
                        help="path to HC3 all.jsonl (default: data/raw/all.jsonl, then ~/.cache/hc3/all.jsonl, "
                             "then download)")
    parser.add_argument("--limit", type=int, default=None,
                        help="quick build: about N texts in total (split sizes scaled 5:1:1)")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="output directory (default: data/)")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (default: 42)")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS_DATA, help="drop texts under N words (default: 20)")
    args = parser.parse_args(argv)

    sizes = scaled_sizes(args.limit)
    summary = build_dataset(args.hc3_path, args.out_dir, sizes=sizes, seed=args.seed, min_words=args.min_words)
    for name in SPLITS:
        s = summary["splits"][name]
        print(f"{name:<10} n={s['n']:>5}  human={s['by_class']['human']:>5}  ai={s['by_class']['ai']:>5}  "
              f"questions={s['questions']}")
    after = summary["artefact_rates"]["after_normalisation_dataset"]
    print("artefact rates after normalisation (human / ai):")
    for key in ARTEFACT_PATTERNS:
        print(f"  {key:<20} {after['human'][key]:.4f} / {after['ai'][key]:.4f}")
    print("residual style/source differences, reported only (human / ai):")
    for key in INFO_PATTERNS:
        print(f"  {key:<20} {after['human'][key]:.4f} / {after['ai'][key]:.4f}")
    print(f"disjoint splits: {summary['disjointness']['disjoint']}")
    print(f"wrote CSVs and summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
