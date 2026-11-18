"""Algorithm v2: stylometric features + word log-odds + logistic regression (still no neural model).

The v1 baseline (src/baseline.py) compares four features with class medians, weighs them equally and
labels only texts of at least 100 words. v2 keeps the same data, splits and normalisation and changes
three things:

1. **More evidence.** 15 more style features computed in plain Python on the first 300 words: word
   length, function-word, pronoun, contraction, punctuation, digit and sentence-case rates, mean sentence
   length and the rate of typical assistant words ("additionally", "overall", ...).
2. **Word log-odds lexicon.** For every word seen at least 3 times in the reference split,
   ``log P(word | AI) - log P(word | human)`` (multinomial, add-one smoothing). A text's
   ``word_logodds`` is the mean over its known words - a naive-Bayes score that is fully inspectable.
   For the reference texts it is cross-fitted (5 folds by question), so the weight below is not learned
   on a score the lexicon has already seen.
3. **Learned weights.** A logistic regression on the standardised features (fitted on the reference
   split) replaces the equal-weight distance. ``P(AI) = sigmoid(sum w_i z_i + b)``; every term
   ``w_i z_i`` is reported, so the label can be explained feature by feature.

The threshold (on the logit) and the uncertainty margin are fitted on the validation split with the v1
functions (:func:`src.baseline.fit_threshold`, :func:`src.baseline.fit_margin`). Everything is frozen to
``models/stylometry.json``; inference needs only the standard library and numpy.

Usage::

    python -m src.stylometry fit  [--limit N] [--target-accuracy 0.95] [--min-coverage 0.80]
    python -m src.stylometry eval [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import Baseline, fit_margin, fit_threshold, label_metrics, predictions, feature_frame
from src.config import (
    BASELINE_PATH,
    DATA_DIR,
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_UNCERTAIN,
    MAX_WORDS,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    WORD_BANDS,
)
from src.data import load_split, sample_balanced, word_band
from src.features import FEATURE_NAMES, compute_features, is_number
from src.perturb import typo_variant
from src.reporting import write_json, write_table
from src.text_utils import normalize_text, split_sentences, tokenize, truncate_words

STYLOMETRY_PATH: Path = MODELS_DIR / "stylometry.json"
LEXICON_MIN_COUNT = 3
LEXICON_ALPHA = 1.0
N_FOLDS = 5

FUNCTION_WORDS = frozenset(
    "the a an and or but of to in on at for with by from as is are was were be been being it this that these "
    "those there their they them he she his her we our you your i me my not no so if then than too very can "
    "will would should could may might must do does did have has had".split()
)
ASSISTANT_WORDS = ("however", "additionally", "overall", "important", "generally", "typically", "various",
                   "including", "furthermore", "ensure", "provide", "known", "also", "several", "specific",
                   "potential", "factors", "may")
_LOWER_I = re.compile(r"(?<![A-Za-z'])i(?![A-Za-z'])")

STYLE_FEATURES: dict[str, str] = {
    "mean_word_length": "mean word length (letters)",
    "long_word_rate": "share of words with 7+ letters",
    "function_word_rate": "share of common function words",
    "first_person_rate": "rate of I / me / my",
    "second_person_rate": "rate of you / your",
    "contraction_rate": "rate of contractions (don't, it's)",
    "assistant_word_rate": "rate of typical assistant words (additionally, overall, ...)",
    "comma_rate": "commas per word",
    "exclaim_question_rate": "! and ? per word",
    "paren_quote_rate": "brackets and double quotes per word",
    "colon_semicolon_rate": ": and ; per word",
    "digit_rate": "share of numeric tokens",
    "mean_sentence_length": "mean sentence length (words)",
    "lower_start_rate": "share of sentences starting in lower case",
    "lower_i_rate": "rate of a lower-case pronoun 'i'",
}
LEXICON_FEATURE = "word_logodds"
MODEL_FEATURES: list[str] = [*FEATURE_NAMES, *STYLE_FEATURES, LEXICON_FEATURE]


# --------------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------------
def style_features(text: str, max_words: int = MAX_WORDS) -> dict:
    """The four v1 features plus the 15 style features, all on the first ``max_words`` words.

    Returns the dict of :func:`src.features.compute_features` extended with the style features (rates are
    per analysed word; NaN only when the text has no words).
    """
    feats = compute_features(text, max_words)
    head = truncate_words(normalize_text(text), max_words)
    words = tokenize(head)
    n = len(words)
    if n == 0:
        return {**feats, **{name: math.nan for name in STYLE_FEATURES}}
    sentences = [s for s in split_sentences(head) if tokenize(s)]
    counts = Counter(words)
    feats.update(
        mean_word_length=sum(map(len, words)) / n,
        long_word_rate=sum(len(w) >= 7 for w in words) / n,
        function_word_rate=sum(counts[w] for w in FUNCTION_WORDS) / n,
        first_person_rate=(counts["i"] + counts["me"] + counts["my"]) / n,
        second_person_rate=(counts["you"] + counts["your"]) / n,
        contraction_rate=sum(c for w, c in counts.items() if "'" in w) / n,
        assistant_word_rate=sum(counts[w] for w in ASSISTANT_WORDS) / n,
        comma_rate=head.count(",") / n,
        exclaim_question_rate=(head.count("!") + head.count("?")) / n,
        paren_quote_rate=(head.count("(") + head.count('"')) / n,
        colon_semicolon_rate=(head.count(":") + head.count(";")) / n,
        digit_rate=sum(w[0].isdigit() for w in words) / n,
        mean_sentence_length=n / max(len(sentences), 1),
        lower_start_rate=sum(s[:1].islower() for s in sentences) / max(len(sentences), 1),
        lower_i_rate=len(_LOWER_I.findall(head)) / n,
    )
    return feats


def fit_lexicon(texts: list[str], labels: list[int], max_words: int = MAX_WORDS,
                min_count: int = LEXICON_MIN_COUNT, alpha: float = LEXICON_ALPHA) -> dict[str, float]:
    """Word log-odds ``log P(w | AI) - log P(w | human)`` with add-``alpha`` smoothing.

    Only words seen at least ``min_count`` times (both classes together) are kept.
    """
    human, ai = Counter(), Counter()
    for text, y in zip(texts, labels):
        (ai if int(y) == 1 else human).update(tokenize(truncate_words(normalize_text(text), max_words)))
    vocab = [w for w in human.keys() | ai.keys() if human[w] + ai[w] >= min_count]
    if not vocab:
        return {}
    n_h, n_a, v = sum(human[w] for w in vocab), sum(ai[w] for w in vocab), len(vocab)
    return {w: math.log((ai[w] + alpha) / (n_a + alpha * v)) - math.log((human[w] + alpha) / (n_h + alpha * v))
            for w in sorted(vocab)}


def lexicon_score(text: str, lexicon: dict[str, float], max_words: int = MAX_WORDS) -> float:
    """Mean log-odds over the known words of the first ``max_words`` words (0.0 if none is known)."""
    values = [lexicon[w] for w in tokenize(truncate_words(normalize_text(text), max_words)) if w in lexicon]
    return sum(values) / len(values) if values else 0.0


def cross_fitted_lexicon_scores(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold ``word_logodds`` for the rows of ``df``: each fold scored by a lexicon built without it.

    Folds are made by ``question_id`` so that the human and AI answers of a question share a fold.
    """
    questions = df["question_id"].unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(questions), np.arange(len(questions)) % n_folds))
    folds = df["question_id"].map(fold_of).to_numpy()
    scores = np.zeros(len(df))
    for k in range(n_folds):
        train = folds != k
        lexicon = fit_lexicon(df["text"][train].tolist(), df["label"][train].tolist())
        for i in np.flatnonzero(~train):
            scores[i] = lexicon_score(df["text"].iloc[i], lexicon)
    return scores


# --------------------------------------------------------------------------------------------------
# The frozen model
# --------------------------------------------------------------------------------------------------
@dataclass
class StyleModel:
    """Frozen lexicon, standardisation, logistic weights, threshold and margin."""

    lexicon: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]
    weights: dict[str, float]
    intercept: float
    threshold: float = 0.0
    margin: float = 0.0
    min_words: int = 0
    max_words: int = MAX_WORDS
    meta: dict = field(default_factory=dict)

    def features(self, text: str) -> dict:
        """All model inputs for ``text`` (v1 features, style features and ``word_logodds``)."""
        feats = style_features(text, self.max_words)
        feats[LEXICON_FEATURE] = lexicon_score(text, self.lexicon, self.max_words)
        return feats

    def contributions(self, feats: dict) -> dict[str, float]:
        """``w_i * z_i`` per feature; an undefined feature is imputed with the reference mean (term 0)."""
        out = {}
        for name, w in self.weights.items():
            x = feats.get(name)
            z = (x - self.mean[name]) / self.std[name] if is_number(x) else 0.0
            out[name] = w * z
        return out

    def predict_features(self, feats: dict) -> dict:
        """Score, P(AI), label and reason for precomputed :meth:`features`."""
        terms = self.contributions(feats)
        score = self.intercept + sum(terms.values())
        if feats.get("word_count", 0) < self.min_words:
            label, reason = LABEL_UNCERTAIN, f"fewer than {self.min_words} words"
        elif score > self.threshold + self.margin:
            label, reason = LABEL_AI, "score above threshold + margin"
        elif score < self.threshold - self.margin:
            label, reason = LABEL_HUMAN, "score below threshold - margin"
        else:
            label, reason = LABEL_UNCERTAIN, "score inside the uncertainty margin"
        top = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:5]
        return {"score": score, "probability_ai": 1 / (1 + math.exp(-score)), "label": label, "reason": reason,
                "top_contributions": top}

    def predict(self, text: str) -> dict:
        """Apply the frozen model to a raw text."""
        return self.predict_features(self.features(text))

    def to_dict(self) -> dict:
        return {"model": "stylometric logistic regression (algorithm v2)", "features": list(self.weights),
                "weights": self.weights, "intercept": self.intercept, "mean": self.mean, "std": self.std,
                "threshold": self.threshold, "margin": self.margin, "min_words": self.min_words,
                "max_words": self.max_words, **self.meta, "lexicon": self.lexicon}

    @classmethod
    def from_dict(cls, d: dict) -> "StyleModel":
        known = {"model", "features", "weights", "intercept", "mean", "std", "threshold", "margin", "min_words",
                 "max_words", "lexicon"}
        return cls(lexicon=d["lexicon"], mean=d["mean"], std=d["std"], weights=d["weights"],
                   intercept=float(d["intercept"]), threshold=float(d["threshold"]), margin=float(d["margin"]),
                   min_words=int(d.get("min_words", 0)), max_words=int(d.get("max_words", MAX_WORDS)),
                   meta={k: v for k, v in d.items() if k not in known})

    def save(self, path: Path = STYLOMETRY_PATH) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path = STYLOMETRY_PATH) -> "StyleModel":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - run `python -m src.stylometry fit` first")
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def load_style_model(path: Path = STYLOMETRY_PATH) -> StyleModel | None:
    """The frozen v2 model, or None if it has not been fitted yet."""
    try:
        return StyleModel.load(path)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------------------------------
# Fitting and evaluation
# --------------------------------------------------------------------------------------------------
def _matrix(df: pd.DataFrame, lexicon_scores: np.ndarray) -> pd.DataFrame:
    rows = [style_features(t) for t in df["text"]]
    frame = pd.DataFrame(rows)[[*FEATURE_NAMES, *STYLE_FEATURES]].astype(float)
    frame[LEXICON_FEATURE] = lexicon_scores
    return frame


def fit_style_model(reference_df: pd.DataFrame, validation_df: pd.DataFrame, target_accuracy: float = 0.95,
                    min_coverage: float = 0.80, min_words: int = 0) -> StyleModel:
    """Fit lexicon and weights on the reference split, threshold and margin on the validation split."""
    from sklearn.linear_model import LogisticRegression

    ref = reference_df.reset_index(drop=True)
    x_ref = _matrix(ref, cross_fitted_lexicon_scores(ref))
    mean = x_ref.mean()
    std = x_ref.std(ddof=0).replace(0, 1.0)
    z = ((x_ref - mean) / std).fillna(0.0)
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(z.to_numpy(), ref["label"].to_numpy())
    model = StyleModel(
        lexicon=fit_lexicon(ref["text"].tolist(), ref["label"].tolist()),
        mean=mean.to_dict(), std=std.to_dict(), weights=dict(zip(MODEL_FEATURES, map(float, clf.coef_[0]))),
        intercept=float(clf.intercept_[0]), min_words=min_words,
    )
    val_feats = [model.features(t) for t in validation_df["text"]]
    scores = np.array([model.predict_features(f)["score"] for f in val_feats])
    y = validation_df["label"].to_numpy(dtype=int)
    long_ = np.array([f["word_count"] >= min_words for f in val_feats])
    t, bacc = fit_threshold(scores[long_], y[long_])
    delta, margin_info = fit_margin(scores[long_], y[long_], t, target_accuracy, min_coverage)
    model.threshold, model.margin = t, delta
    model.meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_reference": int(len(ref)),
        "n_validation": int(len(validation_df)),
        "lexicon_size": len(model.lexicon),
        "lexicon_definition": f"log P(w|AI) - log P(w|human), add-{LEXICON_ALPHA:g} smoothing, words seen >= "
                              f"{LEXICON_MIN_COUNT} times; cross-fitted ({N_FOLDS} folds by question) for training",
        "fit": {"balanced_accuracy_at_threshold": bacc, "margin": margin_info},
    }
    return model


def metrics_by_band(labels: np.ndarray, y: np.ndarray, word_counts: np.ndarray) -> pd.DataFrame:
    """:func:`label_metrics` for all texts and per word-count band."""
    bands = np.array([word_band(n) for n in word_counts])
    subsets = [("all", np.ones(len(y), dtype=bool))] + [(name, bands == name) for name, _, _ in WORD_BANDS]
    return pd.DataFrame([{"subset": s, **label_metrics(y[m], labels[m])} for s, m in subsets])


def evaluate(df: pd.DataFrame, model: StyleModel, baseline: Baseline | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparison table (v1 baseline vs v2, by band) and a typo-robustness table for v2.

    Robustness: every text gets 4 typo variants (swap, drop, key, add; 2 edited words each, as in the
    prompt robustness set); a text counts as robust only if all variants keep the clean label.
    """
    y = df["label"].to_numpy(dtype=int)
    wc = df["word_count"].to_numpy()
    v2_labels = np.array([model.predict(t)["label"] for t in df["text"]], dtype=object)
    parts = []
    if baseline is not None:
        v1 = predictions(feature_frame(df), baseline)["label"].to_numpy(dtype=object)
        parts.append(metrics_by_band(v1, y, wc).assign(algorithm="v1 baseline (4 features, distance rule)"))
    parts.append(metrics_by_band(v2_labels, y, wc).assign(algorithm="v2 stylometric logistic regression"))
    table = pd.concat(parts, ignore_index=True)
    table = table[["algorithm", *[c for c in table.columns if c != "algorithm"]]]

    rows, stable = [], np.ones(len(df), dtype=bool)
    for j in range(4):
        variants = [typo_variant(t, str(i), j) for t, i in zip(df["text"], df["id"])]
        labels = np.array([model.predict(text)["label"] for text, _ in variants], dtype=object)
        stable &= labels == v2_labels
        rows.append({"variant": variants[0][1] if variants else str(j), **label_metrics(y, labels),
                     "label_unchanged": float((labels == v2_labels).mean())})
    robust = pd.DataFrame(rows)
    robust.loc[len(robust)] = {"variant": "all 4 variants", "label_unchanged": float(stable.mean())}
    return table, robust


def _notes(model: StyleModel) -> list[str]:
    margin = model.meta.get("fit", {}).get("margin", {})
    top = sorted(model.weights.items(), key=lambda kv: -abs(kv[1]))[:6]
    return [
        f"v2: logistic regression on {len(model.weights)} standardised features "
        f"({len(FEATURE_NAMES)} v1 + {len(STYLE_FEATURES)} style + word log-odds, lexicon of "
        f"{model.meta.get('lexicon_size', len(model.lexicon))} words), fitted on the reference split.",
        f"Threshold {model.threshold:.4f} and margin {model.margin:.4f} on the logit, fitted on validation "
        f"(target decided accuracy {margin.get('target_accuracy', math.nan):.2f}, min coverage "
        f"{margin.get('min_coverage', math.nan):.2f}); word rule: "
        + (f"fewer than {model.min_words} words -> Uncertain." if model.min_words else "none (all texts scored)."),
        "Largest standardised weights (+ = towards AI): " + ", ".join(f"{n} {w:+.2f}" for n, w in top) + ".",
        "v1 labels texts under 100 words 'Uncertain'; v2 scores them, so compare the band rows too.",
        "Accuracy and F1 (AI positive) on decided texts; coverage = share decided; human FPR over all human texts.",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.stylometry", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["fit", "eval"])
    parser.add_argument("--limit", type=int, default=None, help="use at most N (balanced) rows per split")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=STYLOMETRY_PATH)
    parser.add_argument("--baseline-path", type=Path, default=BASELINE_PATH, help="v1 baseline for the comparison")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--min-words", type=int, default=0, help="shorter texts are labelled Uncertain (default 0)")
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            model = fit_style_model(sample_balanced(load_split("reference", args.data_dir), args.limit),
                                    sample_balanced(load_split("validation", args.data_dir), args.limit),
                                    args.target_accuracy, args.min_coverage, args.min_words)
            model.save(args.model_path)
            print(f"threshold {model.threshold:.4f}  margin {model.margin:.4f}  "
                  f"validation coverage {model.meta['fit']['margin']['coverage']:.3f}  "
                  f"decided accuracy {model.meta['fit']['margin']['decided_accuracy']:.3f}")
            print(f"wrote {args.model_path}")
            return 0
        model = StyleModel.load(args.model_path)
        test = sample_balanced(load_split("test", args.data_dir), args.limit)
        baseline = Baseline.load(args.baseline_path) if Path(args.baseline_path).is_file() else None
        table, robust = evaluate(test, model, baseline)
        write_table(table, args.results_dir / "stylometry_metrics", "Algorithm v1 vs v2 - test split", _notes(model))
        write_table(robust, args.results_dir / "stylometry_robustness", "Algorithm v2 - typo robustness (test split)",
                    ["Each variant edits 2 words (first letter kept) inside the first 300 words.",
                     "label_unchanged = share of texts whose label is the same as on the clean text."])
        print(table[table["subset"] == "all"].to_string(index=False))
        print(robust.to_string(index=False))
        print(f"wrote {args.results_dir / 'stylometry_metrics.md'} and stylometry_robustness.md")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
