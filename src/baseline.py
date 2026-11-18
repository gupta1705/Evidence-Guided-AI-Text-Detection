"""Transparent statistical baseline: distance to human/AI reference medians, threshold and margin.

Method (from the project form, "Statistical features and decision rule"):

* From the **reference** split, per feature: human and AI medians and quartiles, and the pooled IQR of
  both classes with a denominator floor ``scale = max(pooled_IQR, 0.25 * pooled_std, 1e-6)``. Only
  reference texts with at least ``min_words`` words (default 100) are used - the texts the rule can
  label. Shorter texts have features that are not length-controlled (entropy on fewer than 100 words,
  plain TTR under 50 words) and would distort the medians; for example, the human median of
  ``word_entropy`` on HC3 is below the AI median when short texts are included and above it when they
  are not.
* Distance to a class = mean over the available features of ``|x - median_class| / scale``.
  **Score = distance to the human reference - distance to the AI reference** (larger = more similar to
  the AI reference). The score is *not* a probability of authorship.
* On the **validation** split, using the texts the rule can label (at least ``min_words`` words and two
  valid features): the threshold ``t`` maximises balanced accuracy (candidates are midpoints of the
  sorted scores); the uncertainty margin ``delta`` is the smallest grid value (quantiles of
  ``|score - t|``) whose decided accuracy reaches the target (0.85) with coverage of at least 0.60 *of
  those texts*. Texts under ``min_words`` words are always "Uncertain", so the overall coverage is lower;
  it is reported in ``validation_metrics`` and in every metrics table.
* Decision rule: fewer than ``min_words`` words or fewer than two valid features -> "Uncertain";
  ``score > t + delta`` -> "Likely AI-generated"; ``score < t - delta`` -> "Likely human-written";
  otherwise "Uncertain".
* Everything is frozen to ``models/baseline.json`` and never refitted on test data.

Usage::

    python -m src.baseline fit [--limit N] [--min-words 100] [--target-accuracy 0.85] [--min-coverage 0.6]
    python -m src.baseline eval [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_PATH,
    DATA_DIR,
    FEW_SHOT_PATH,
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_UNCERTAIN,
    MAX_WORDS,
    MIN_WORDS_DECISION,
    RESULTS_DIR,
    SEED,
    WORD_BANDS,
)
from src.data import load_split, sample_balanced, word_band
from src.features import FEATURE_NAMES, FEATURES, compute_features, is_number, valid_feature_count
from src.reporting import write_json, write_table

SCALE_FLOOR = 1e-6
STD_FLOOR_FACTOR = 0.25
DEFAULT_TARGET_ACCURACY = 0.85
DEFAULT_MIN_COVERAGE = 0.60
MIN_REFERENCE_PER_CLASS = 10  # below this (tiny smoke runs) all reference texts are used instead
FEATURE_COLUMNS: list[str] = list(compute_features("").keys())
PREDICTION_COLUMNS: list[str] = ["score", "label", "reason", "distance_human", "distance_ai", "n_valid"]


# --------------------------------------------------------------------------------------------------
# Reference statistics
# --------------------------------------------------------------------------------------------------
def feature_frame(df: pd.DataFrame, max_words: int = MAX_WORDS) -> pd.DataFrame:
    """Compute features for every row of ``df`` (needs a ``text`` column; keeps id/label/... columns)."""
    keep = [c for c in ("id", "question_id", "source", "label", "provenance") if c in df.columns]
    feats = pd.DataFrame([compute_features(t, max_words=max_words) for t in df["text"]], columns=FEATURE_COLUMNS)
    return pd.concat([df[keep].reset_index(drop=True), feats], axis=1)


def decidable_reference(ref_feats: pd.DataFrame, min_words: int) -> tuple[pd.DataFrame, str]:
    """Reference rows the decision rule could label (``word_count >= min_words``) and a description.

    Falls back to every reference row (and says so) when fewer than ``MIN_REFERENCE_PER_CLASS`` texts of
    a class pass the filter, which only happens in tiny smoke runs.
    """
    long_ = ref_feats[ref_feats["word_count"] >= min_words]
    per_class = [int((long_["label"] == lab).sum()) for lab in (0, 1)]
    if min(per_class) >= MIN_REFERENCE_PER_CLASS:
        return long_, f"reference texts with >= {min_words} words (the texts the rule can label)"
    return ref_feats, (f"all reference texts (fewer than {MIN_REFERENCE_PER_CLASS} texts per class have "
                       f">= {min_words} words)")


def reference_stats(feature_df: pd.DataFrame, features: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Per-feature medians, quartiles, pooled IQR / std and the floored scale from labelled rows.

    Args:
        feature_df: Rows with a ``label`` column (0 human, 1 AI) and one column per feature.
        features: Feature names to use (default: all four).
    """
    stats: dict[str, dict[str, float]] = {}
    for name in features or FEATURE_NAMES:
        values = pd.to_numeric(feature_df[name], errors="coerce")
        human = values[feature_df["label"] == 0].dropna().to_numpy(dtype=float)
        ai = values[feature_df["label"] == 1].dropna().to_numpy(dtype=float)
        if len(human) == 0 or len(ai) == 0:
            continue
        pooled = np.concatenate([human, ai])
        q1, q3 = np.quantile(pooled, [0.25, 0.75])
        pooled_iqr = float(q3 - q1)
        pooled_std = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        stats[name] = {
            "human_median": float(np.median(human)),
            "human_q1": float(np.quantile(human, 0.25)),
            "human_q3": float(np.quantile(human, 0.75)),
            "ai_median": float(np.median(ai)),
            "ai_q1": float(np.quantile(ai, 0.25)),
            "ai_q3": float(np.quantile(ai, 0.75)),
            "pooled_iqr": pooled_iqr,
            "pooled_std": pooled_std,
            "scale": max(pooled_iqr, STD_FLOOR_FACTOR * pooled_std, SCALE_FLOOR),
            "n_human": int(len(human)),
            "n_ai": int(len(ai)),
        }
    return stats


# --------------------------------------------------------------------------------------------------
# The frozen baseline
# --------------------------------------------------------------------------------------------------
@dataclass
class Baseline:
    """Frozen reference statistics plus the decision rule."""

    reference: dict[str, dict[str, float]]
    threshold: float = 0.0
    margin: float = 0.0
    min_words: int = MIN_WORDS_DECISION
    max_words: int = MAX_WORDS
    features: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    meta: dict = field(default_factory=dict)

    # ---- scoring ----------------------------------------------------------------------------------
    def distances(self, feats: dict) -> tuple[float, float, int]:
        """Return ``(distance_to_human, distance_to_ai, n_features_used)``; NaN distances if none usable."""
        d_h = d_a = 0.0
        k = 0
        for name in self.features:
            x = feats.get(name)
            ref = self.reference.get(name)
            if ref is None or not is_number(x):
                continue
            d_h += abs(x - ref["human_median"]) / ref["scale"]
            d_a += abs(x - ref["ai_median"]) / ref["scale"]
            k += 1
        if k == 0:
            return math.nan, math.nan, 0
        return d_h / k, d_a / k, k

    def score(self, feats: dict) -> float:
        """Distance to human reference minus distance to AI reference (larger = more AI-like)."""
        d_h, d_a, _ = self.distances(feats)
        return d_h - d_a

    def predict(self, feats: dict, apply_min_words: bool = True) -> dict:
        """Apply the frozen decision rule.

        Args:
            feats: Output of :func:`src.features.compute_features`.
            apply_min_words: If False the word-count precaution is skipped (used to judge the rule).

        Returns:
            Dict with ``score``, ``label``, ``reason``, ``distance_human``, ``distance_ai``, ``n_valid``.
        """
        d_h, d_a, k = self.distances(feats)
        score = d_h - d_a
        n_valid = valid_feature_count(feats)
        word_count = feats.get("word_count", 0)
        t, delta = self.threshold, self.margin
        if apply_min_words and word_count < self.min_words:
            label, reason = LABEL_UNCERTAIN, f"fewer than {self.min_words} words"
        elif n_valid < 2 or k < 2 or not math.isfinite(score):
            label, reason = LABEL_UNCERTAIN, "fewer than 2 valid features"
        elif score > t + delta:
            label, reason = LABEL_AI, "score above threshold + margin"
        elif score < t - delta:
            label, reason = LABEL_HUMAN, "score below threshold - margin"
        else:
            label, reason = LABEL_UNCERTAIN, "score inside the uncertainty margin"
        return {
            "score": score,
            "label": label,
            "reason": reason,
            "distance_human": d_h,
            "distance_ai": d_a,
            "n_valid": n_valid,
        }

    def explain(self, feats: dict) -> list[dict]:
        """Per-feature comparison with the reference medians (for the app and the prompts).

        ``closer_to`` is "human", "AI", "tie" or "n/a" (feature undefined or no reference statistics).
        """
        rows = []
        for name in self.features:
            x = feats.get(name)
            ref = self.reference.get(name, {})
            row = {
                "feature": name,
                "label": FEATURES[name].label if name in FEATURES else name,
                "value": x if is_number(x) else math.nan,
                "human_median": ref.get("human_median", math.nan),
                "human_q1": ref.get("human_q1", math.nan),
                "human_q3": ref.get("human_q3", math.nan),
                "ai_median": ref.get("ai_median", math.nan),
                "ai_q1": ref.get("ai_q1", math.nan),
                "ai_q3": ref.get("ai_q3", math.nan),
                "closer_to": "n/a",
            }
            if ref and is_number(x):
                gap_h = abs(x - ref["human_median"])
                gap_a = abs(x - ref["ai_median"])
                row["closer_to"] = "human" if gap_h < gap_a else "AI" if gap_a < gap_h else "tie"
            rows.append(row)
        return rows

    # ---- persistence ------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "features": self.features,
            "reference_stats": self.reference,
            "threshold": self.threshold,
            "margin": self.margin,
            "min_words": self.min_words,
            "max_words": self.max_words,
            **self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        """Inverse of :meth:`to_dict`."""
        known = {"features", "reference_stats", "threshold", "margin", "min_words", "max_words"}
        return cls(
            reference=d["reference_stats"],
            threshold=float(d["threshold"]),
            margin=float(d["margin"]),
            min_words=int(d.get("min_words", MIN_WORDS_DECISION)),
            max_words=int(d.get("max_words", MAX_WORDS)),
            features=list(d.get("features", FEATURE_NAMES)),
            meta={k: v for k, v in d.items() if k not in known},
        )

    def save(self, path: Path = BASELINE_PATH) -> Path:
        """Write the frozen baseline to JSON."""
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path = BASELINE_PATH) -> "Baseline":
        """Load a frozen baseline; raises FileNotFoundError with a hint if missing."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - run `python -m src.baseline fit` first")
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def load_baseline(path: Path = BASELINE_PATH) -> Baseline | None:
    """Load the frozen baseline, or return None if it has not been fitted yet."""
    try:
        return Baseline.load(path)
    except FileNotFoundError:
        return None


def explain(features: dict, baseline: Baseline | None = None) -> list[dict]:
    """Per-feature value / human median / AI median / closer-to rows (loads the frozen baseline if needed)."""
    baseline = baseline or Baseline.load()
    return baseline.explain(features)


# --------------------------------------------------------------------------------------------------
# Fitting the threshold and the uncertainty margin
# --------------------------------------------------------------------------------------------------
def fit_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold maximising balanced accuracy (predict AI when ``score > t``).

    Candidates are the midpoints between consecutive distinct sorted scores. When several candidates tie,
    the middle one of the tied candidates is chosen (deterministic).

    Returns:
        ``(t, balanced_accuracy)``.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("threshold fitting needs both classes")
    u = np.unique(s)
    candidates = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    pred = s[None, :] > candidates[:, None]
    tpr = (pred & (y == 1)).sum(axis=1) / n1
    tnr = (~pred & (y == 0)).sum(axis=1) / n0
    bacc = (tpr + tnr) / 2
    best = np.flatnonzero(np.isclose(bacc, bacc.max()))
    idx = best[len(best) // 2]
    return float(candidates[idx]), float(bacc[idx])


def fit_margin(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    n_grid: int = 100,
) -> tuple[float, dict]:
    """Choose the uncertainty margin ``delta`` around ``threshold``.

    The grid is ``{0} U quantiles(|score - t|, 0 .. 0.99)``. A text is decided when ``|score - t| > delta``.
    The smallest ``delta`` with decided accuracy >= ``target_accuracy`` and coverage >= ``min_coverage`` is
    chosen; if none qualifies, the ``delta`` with the best decided accuracy among those with coverage >=
    ``min_coverage`` (smallest on ties).

    Returns:
        ``(delta, info)`` where ``info`` has the coverage / decided accuracy reached and whether the
        target was met.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    dist = np.abs(s - threshold)
    grid = np.unique(np.concatenate([[0.0], np.quantile(dist, np.linspace(0.0, 0.99, n_grid))]))
    correct = (s > threshold).astype(int) == y
    rows = []
    for delta in grid:
        decided = dist > delta
        coverage = float(decided.mean()) if len(decided) else 0.0
        acc = float(correct[decided].mean()) if decided.any() else math.nan
        rows.append((float(delta), coverage, acc))

    eligible = [r for r in rows if r[1] >= min_coverage and not math.isnan(r[2])]
    chosen = next((r for r in eligible if r[2] >= target_accuracy), None)
    target_met = chosen is not None
    if chosen is None:
        chosen = max(eligible, key=lambda r: (r[2], -r[0])) if eligible else (0.0, rows[0][1], rows[0][2])
    delta, coverage, acc = chosen
    return delta, {
        "coverage": coverage,
        "decided_accuracy": acc,
        "target_met": target_met,
        "target_accuracy": target_accuracy,
        "min_coverage": min_coverage,
        "grid_size": len(grid),
    }


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------
def label_metrics(y_true: np.ndarray, labels: list[str] | np.ndarray) -> dict:
    """Decided accuracy/F1 (AI positive), coverage, human false-positive rate and AI recall.

    * coverage = share of texts that received a decided (non-"Uncertain") label;
    * accuracy and F1 are computed on decided texts only;
    * human FPR = human texts labelled "Likely AI-generated" / all human texts;
    * AI recall = AI texts labelled "Likely AI-generated" / all AI texts.
    """
    y = np.asarray(y_true, dtype=int)
    lab = np.asarray(labels, dtype=object)
    pred_ai = lab == LABEL_AI
    pred_h = lab == LABEL_HUMAN
    decided = pred_ai | pred_h
    n, n_h, n_a = len(y), int((y == 0).sum()), int((y == 1).sum())
    tp = int((decided & pred_ai & (y == 1)).sum())
    fp = int((decided & pred_ai & (y == 0)).sum())
    fn = int((decided & pred_h & (y == 1)).sum())
    tn = int((decided & pred_h & (y == 0)).sum())
    n_dec = int(decided.sum())
    return {
        "n": n,
        "n_human": n_h,
        "n_ai": n_a,
        "n_decided": n_dec,
        "coverage": n_dec / n if n else math.nan,
        "decided_accuracy": (tp + tn) / n_dec if n_dec else math.nan,
        "decided_f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else math.nan,
        "human_fpr": fp / n_h if n_h else math.nan,
        "ai_recall": tp / n_a if n_a else math.nan,
    }


def predictions(feature_df: pd.DataFrame, baseline: Baseline, apply_min_words: bool = True) -> pd.DataFrame:
    """Score and label every row of a feature frame (an empty frame gives an empty, correctly-columned one)."""
    feature_cols = ["word_count", *FEATURE_NAMES]
    preds = [
        baseline.predict(dict(zip(feature_cols, values)), apply_min_words=apply_min_words)
        for values in feature_df[feature_cols].itertuples(index=False, name=None)
    ]
    return pd.DataFrame(preds, columns=PREDICTION_COLUMNS)


def metrics_table(feature_df: pd.DataFrame, baseline: Baseline) -> pd.DataFrame:
    """Metrics with and without the word-count rule: overall, for texts the rule can label, and by band."""
    rows = []
    bands = feature_df["word_count"].map(word_band)
    long_ = (feature_df["word_count"] >= baseline.min_words).to_numpy()
    for rule_name, apply_rule in ((f"with {baseline.min_words}-word rule", True), ("without word rule", False)):
        labels = predictions(feature_df, baseline, apply_min_words=apply_rule)["label"].to_numpy()
        subsets = [("all", np.ones(len(feature_df), dtype=bool)), (f">={baseline.min_words}", long_)]
        subsets += [(name, (bands == name).to_numpy()) for name, _, _ in WORD_BANDS]
        for subset, mask in subsets:
            m = label_metrics(feature_df["label"].to_numpy()[mask], labels[mask])
            rows.append({"rule": rule_name, "subset": subset, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Fitting pipeline
# --------------------------------------------------------------------------------------------------
def fit_baseline(
    reference_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    min_words: int = MIN_WORDS_DECISION,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[Baseline, pd.DataFrame, pd.DataFrame]:
    """Fit reference statistics, threshold and margin.

    Reference statistics come from the reference texts with at least ``min_words`` words, and ``t`` and
    ``delta`` are fitted on the validation texts the rule would actually score (at least ``min_words``
    words and two valid features), so the coverage target applies to those texts. If either filter
    leaves too few texts of a class (tiny smoke runs), all texts are used and this is recorded in
    ``meta``.

    Returns:
        ``(baseline, validation_metrics_table, validation_feature_frame)``.
    """
    ref_feats = feature_frame(reference_df)
    ref_used, reference_filter = decidable_reference(ref_feats, min_words)
    stats = reference_stats(ref_used)
    baseline = Baseline(reference=stats, min_words=min_words)

    val_feats = feature_frame(validation_df)
    preds = predictions(val_feats, baseline, apply_min_words=False)
    scores = preds["score"].to_numpy(dtype=float)
    y = val_feats["label"].to_numpy(dtype=int)
    scorable = np.isfinite(scores) & (preds["n_valid"].to_numpy() >= 2)
    eligible = scorable & (val_feats["word_count"].to_numpy() >= min_words)
    fit_on = f"validation texts with >= {min_words} words and >= 2 valid features"
    if min(int((y[eligible] == 0).sum()), int((y[eligible] == 1).sum())) < 2:
        eligible = scorable
        fit_on = "all scorable validation texts (too few texts passed the word rule)"

    t, bacc = fit_threshold(scores[eligible], y[eligible])
    delta, margin_info = fit_margin(scores[eligible], y[eligible], t, target_accuracy, min_coverage)
    baseline.threshold, baseline.margin = t, delta

    table = metrics_table(val_feats, baseline)
    overall = table[(table["subset"] == "all")].set_index("rule").to_dict(orient="index")
    long_rows = table[(table["subset"] == f">={min_words}")].set_index("rule").to_dict(orient="index")
    baseline.meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "score_definition": "mean scaled distance to human medians - mean scaled distance to AI medians "
        "(larger = more similar to the AI reference; not a probability)",
        "scale_definition": "max(pooled IQR, 0.25 * pooled std, 1e-6)",
        "reference_filter": reference_filter,
        "n_reference": int(len(reference_df)),
        "n_reference_used": int(len(ref_used)),
        "n_validation": int(len(validation_df)),
        "fit": {
            "fitted_on": fit_on,
            "n_fit": int(eligible.sum()),
            "balanced_accuracy_at_threshold": bacc,
            "margin": {**margin_info, "coverage_refers_to": fit_on},
        },
        "validation_metrics": overall,
        "validation_metrics_decidable_texts": long_rows,
    }
    return baseline, table, val_feats


def validation_notes(baseline: Baseline) -> list[str]:
    """Bullet notes for the validation / test tables."""
    fit = baseline.meta.get("fit", {})
    notes = [
        f"Threshold t = {baseline.threshold:.4f}, margin delta = {baseline.margin:.4f}, "
        f"word rule: fewer than {baseline.min_words} words -> Uncertain.",
        f"Reference medians/IQRs: {baseline.meta.get('reference_filter', 'reference split')}.",
    ]
    if fit:
        margin = fit.get("margin", {})
        notes.append(
            f"t and delta were fitted on {fit.get('fitted_on')} (n = {fit.get('n_fit')}); the margin's "
            f"coverage target ({margin.get('min_coverage', DEFAULT_MIN_COVERAGE):.2f}) applies to those texts "
            f"(reached: {margin.get('coverage', math.nan):.3f}). Overall coverage, which counts the always-"
            f"Uncertain short texts, is the 'all' row."
        )
    notes += [
        f"Rows '>={baseline.min_words}' cover the texts the rule can label; '<100', '100-199', '>=200' are "
        "word-count bands.",
        "Accuracy and F1 (AI positive) are computed on decided texts only; coverage = share decided.",
        "Human FPR = human texts labelled 'Likely AI-generated' / all human texts in the row's subset.",
        "The score is a distance difference, not a probability; labels support manual review only.",
    ]
    return notes


def evaluate_baseline(df: pd.DataFrame, baseline: Baseline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics table and per-text predictions of the frozen baseline on ``df``."""
    feats = feature_frame(df)
    preds = predictions(feats, baseline).rename(columns={"label": "pred_label", "score": "baseline_score"})
    per_text = pd.concat([feats[["id", "question_id", "label", "word_count"]], preds], axis=1)
    return metrics_table(feats, baseline), per_text


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m src.baseline",
        description="Fit (reference + validation) or evaluate (test) the frozen statistical baseline.",
    )
    parser.add_argument("command", choices=["fit", "eval"], help="fit: build models/baseline.json; eval: test metrics")
    parser.add_argument("--limit", type=int, default=None, help="use at most N (balanced) rows per split")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="directory with the split CSVs")
    parser.add_argument("--model-path", type=Path, default=BASELINE_PATH, help="frozen baseline JSON path")
    parser.add_argument("--few-shot-path", type=Path, default=FEW_SHOT_PATH,
                        help="where `fit` writes the V3/V4 few-shot examples (from validation)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="output directory for tables")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS_DECISION,
                        help="texts with fewer words are labelled Uncertain (default: 100)")
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY,
                        help="decided-accuracy target for the margin (default: 0.85)")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help="minimum coverage for the margin (default: 0.60)")
    args = parser.parse_args(argv)
    try:
        return _fit(args) if args.command == "fit" else _eval(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _fit(args: argparse.Namespace) -> int:
    """``fit`` sub-command: reference statistics, threshold, margin and few-shot examples."""
    from src.prompts import build_few_shot_examples, save_few_shot_examples

    reference = sample_balanced(load_split("reference", args.data_dir), args.limit)
    validation = sample_balanced(load_split("validation", args.data_dir), args.limit)
    if reference.empty or validation.empty:
        raise ValueError(f"the reference or validation split in {args.data_dir} is empty - run `python -m src.data`")
    baseline, table, val_feats = fit_baseline(
        reference, validation, args.min_words, args.target_accuracy, args.min_coverage
    )
    baseline.save(args.model_path)
    examples = build_few_shot_examples(validation, baseline, feature_df=val_feats)
    save_few_shot_examples(examples, args.few_shot_path)
    write_table(table, args.results_dir / "baseline_validation", "Baseline - validation split",
                validation_notes(baseline))
    print(f"threshold t = {baseline.threshold:.4f}   margin delta = {baseline.margin:.4f}")
    print(f"reference statistics from: {baseline.meta['reference_filter']} (n={baseline.meta['n_reference_used']})")
    print(f"t and delta fitted on: {baseline.meta['fit']['fitted_on']} (n={baseline.meta['fit']['n_fit']})")
    for subset, metrics in (("all", baseline.meta["validation_metrics"]),
                            (f">={args.min_words} words", baseline.meta["validation_metrics_decidable_texts"])):
        for rule, m in metrics.items():
            print(f"validation [{rule}, {subset}]: coverage={m['coverage']:.3f} "
                  f"decided_acc={m['decided_accuracy']:.3f} human_fpr={m['human_fpr']:.3f}")
    print(f"few-shot examples: {', '.join(e['kind'] for e in examples) or 'none'}")
    print(f"wrote {args.model_path}, {args.few_shot_path} and {args.results_dir / 'baseline_validation.md'}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    """``eval`` sub-command: the frozen baseline on the test split."""
    baseline = Baseline.load(args.model_path)
    test = sample_balanced(load_split("test", args.data_dir), args.limit)
    if test.empty:
        raise ValueError(f"the test split in {args.data_dir} is empty - run `python -m src.data`")
    table, _ = evaluate_baseline(test, baseline)
    write_table(table, args.results_dir / "baseline_metrics", "Baseline - test split", validation_notes(baseline))
    for _, row in table[table["subset"].isin(["all", f">={baseline.min_words}"])].iterrows():
        print(f"test [{row['rule']}, {row['subset']}]: n={row['n']} coverage={row['coverage']:.3f} "
              f"decided_acc={row['decided_accuracy']:.3f} f1={row['decided_f1']:.3f} "
              f"human_fpr={row['human_fpr']:.3f}")
    print(f"wrote {args.results_dir / 'baseline_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
