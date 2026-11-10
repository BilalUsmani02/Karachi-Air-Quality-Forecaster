"""Band-classification statistics: metrics, confusion tables and date-block bootstrap CIs.

Pure functions on band labels (merged indices 0..4: Good, Moderate, USG, Unhealthy, Very
Unhealthy). The point metrics reuse the project's existing definitions:

- accuracy, macro-F1, per-class precision/recall/support and the confusion matrix come from
  `metrics.classification_metrics`;
- the Unhealthy-or-worse **miss rate** (truly band >= 3, forecast below 3) and **false-alarm
  rate** (truly below 3, forecast >= 3) come from `final_test._safety_rates`, the same
  definitions as `fairness.per_sensor_classification`.

Two macro-F1 numbers are reported. `macro_f1_all` averages F1 over all five bands, so a band that
never occurs in the evaluated period scores 0 and drags the average down for every model.
`macro_f1_present` averages only over the bands that occur in that period's `y_true`, which is the
fairer summary for the final test (it has 1 Good day and no Very Unhealthy days).

The bootstrap (`date_block_bootstrap`) recomputes the same metrics from resampled confusion
matrices; `metrics_from_confusion` is tested to agree with the reference functions above.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.config import SEED
from src.data.split import UNHEALTHY_OR_WORSE
from src.evaluate.final_test import BAND_NAMES, _safety_rates
from src.evaluate.metrics import classification_metrics

LABELS: tuple[int, ...] = tuple(range(len(BAND_NAMES)))
N_LABELS = len(LABELS)
CI_METRICS: tuple[str, ...] = ("accuracy", "macro_f1_all", "macro_f1_present", "miss_rate", "false_alarm_rate")


def present_classes(y_true: Sequence[int]) -> list[int]:
    """Band indices that actually occur in `y_true`, in ascending order."""
    seen = set(np.asarray(y_true, dtype=np.int64).tolist())
    return [label for label in LABELS if label in seen]


def band_metrics(y_true: Sequence[int], y_pred: Sequence[int], present: Sequence[int] | None = None) -> dict:
    """Headline band metrics for one model on one row set.

    `present` fixes the classes used by `macro_f1_present`; by default it is the classes that
    occur in `y_true`. Returns n, accuracy, macro_f1_all, macro_f1_present, n_present_classes,
    miss_rate, false_alarm_rate and the 5x5 confusion matrix (rows true, columns predicted).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    present = list(present) if present is not None else present_classes(y_true)
    full = classification_metrics(y_true, y_pred, labels=LABELS)
    present_only = classification_metrics(y_true, y_pred, labels=present)
    safety = _safety_rates(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": full["accuracy"],
        "macro_f1_all": full["macro_f1"],
        "macro_f1_present": present_only["macro_f1"],
        "n_present_classes": len(present),
        "miss_rate": safety["miss_unhealthy"],
        "false_alarm_rate": safety["false_alarm"],
        "confusion_matrix": full["confusion_matrix"],
    }


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def per_class_table(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Per-band precision, recall, F1 and support (zero_division=0, as in `classification_metrics`)."""
    per_class = classification_metrics(y_true, y_pred, labels=LABELS)["per_class"]
    rows = [
        {
            "band": label,
            "band_name": BAND_NAMES[label],
            "precision": per_class[label]["precision"],
            "recall": per_class[label]["recall"],
            "f1": _f1(per_class[label]["precision"], per_class[label]["recall"]),
            "support": per_class[label]["support"],
        }
        for label in LABELS
    ]
    return pd.DataFrame(rows)


def confusion_frames(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(counts, row-normalised recall) confusion tables, indexed by true band, columns predicted.

    A row whose true band never occurs has recall NaN (nothing to divide by), not 0.
    """
    counts = np.asarray(classification_metrics(y_true, y_pred, labels=LABELS)["confusion_matrix"], dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(totals > 0, counts / np.where(totals > 0, totals, 1), np.nan)
    names = list(BAND_NAMES)
    counts_df = pd.DataFrame(counts, index=pd.Index(names, name="true"), columns=names)
    recall_df = pd.DataFrame(recall, index=pd.Index(names, name="true"), columns=names)
    return counts_df, recall_df


def metrics_from_confusion(cm: np.ndarray, present: Sequence[int]) -> dict:
    """The `band_metrics` headline numbers computed straight from a 5x5 confusion matrix.

    Used inside the bootstrap, where calling scikit-learn thousands of times would be slow.
    Rates with an empty denominator are NaN (as in `_safety_rates`).
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    pred_totals, true_totals = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_totals, out=np.zeros_like(tp), where=pred_totals > 0)
    recall = np.divide(tp, true_totals, out=np.zeros_like(tp), where=true_totals > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    k = UNHEALTHY_OR_WORSE
    bad, clean = cm[k:, :].sum(), cm[:k, :].sum()
    return {
        "accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
        "macro_f1_all": float(f1.mean()),
        "macro_f1_present": float(f1[list(present)].mean()) if len(present) else float("nan"),
        "miss_rate": float(cm[k:, :k].sum() / bad) if bad > 0 else float("nan"),
        "false_alarm_rate": float(cm[:k, k:].sum() / clean) if clean > 0 else float("nan"),
    }


def per_date_confusions(dates: Sequence, y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """One flattened 5x5 confusion matrix per calendar date: shape (n_dates, 25).

    Dates are indexed in order of first appearance (`pd.factorize`), so several models scored on
    the same rows share the same date index and can be resampled together (paired).
    """
    codes, uniques = pd.factorize(pd.Series(dates), sort=False)
    cells = np.asarray(y_true, dtype=np.int64) * N_LABELS + np.asarray(y_pred, dtype=np.int64)
    out = np.zeros((len(uniques), N_LABELS * N_LABELS), dtype=np.float64)
    np.add.at(out, (codes, cells), 1.0)
    return out


def date_block_bootstrap(
    frame: pd.DataFrame,
    pred_columns: dict[str, str],
    *,
    true_column: str = "band_true",
    date_column: str = "date",
    pairs: Sequence[tuple[str, str]] = (),
    n_boot: int = 1000,
    seed: int = SEED,
    present: Sequence[int] | None = None,
) -> pd.DataFrame:
    """95% bootstrap confidence intervals, resampling whole calendar dates (block bootstrap).

    Why dates, not rows: the 14 sensors are scored on the same days, and one city-wide smoky or
    clean day moves every sensor together, so rows from the same date are strongly correlated.
    Resampling single rows would treat those ~6 near-copies as independent evidence and give
    intervals that are too narrow. Instead each resample draws `n_dates` dates with replacement
    and keeps **every** row of each drawn date, which preserves the within-day correlation.

    Every model is scored on the *same* resampled dates in each round, so the CI of a paired
    difference (`pairs`, e.g. ("ensemble", "persistence") -> ensemble minus persistence) accounts
    for the two models erring on the same hard days. A difference whose CI contains 0 is within
    resampling noise.

    `pred_columns` maps a model name to its predicted-band column in `frame`. `present` fixes the
    classes for `macro_f1_present` (default: those present in the full `frame`), so every resample
    is scored on the same class set. Percentile intervals (2.5%, 97.5%); a resample with an empty
    denominator (e.g. no Unhealthy day drawn) gives NaN and is ignored. Seeded and deterministic.

    Returns one row per (model, metric) with `point`, `ci_low`, `ci_high`, `n_boot`, `n_dates`;
    difference rows are named "<a> - <b>".
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    y_true = frame[true_column].to_numpy(np.int64)
    present = list(present) if present is not None else present_classes(y_true)
    dates = frame[date_column].to_numpy()

    by_date = {name: per_date_confusions(dates, y_true, frame[col].to_numpy(np.int64)) for name, col in pred_columns.items()}
    n_dates = next(iter(by_date.values())).shape[0]

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_dates, size=(n_boot, n_dates))
    weights = np.stack([np.bincount(row, minlength=n_dates) for row in draws]).astype(np.float64)

    point: dict[str, dict] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    for name, table in by_date.items():
        point[name] = metrics_from_confusion(table.sum(axis=0).reshape(N_LABELS, N_LABELS), present)
        boot_cms = (weights @ table).reshape(n_boot, N_LABELS, N_LABELS)
        rounds = [metrics_from_confusion(cm, present) for cm in boot_cms]
        samples[name] = {m: np.array([r[m] for r in rounds]) for m in CI_METRICS}

    for a, b in pairs:
        label = f"{a} - {b}"
        point[label] = {m: point[a][m] - point[b][m] for m in CI_METRICS}
        samples[label] = {m: samples[a][m] - samples[b][m] for m in CI_METRICS}

    rows = []
    for name in point:
        for metric in CI_METRICS:
            values = samples[name][metric]
            finite = values[np.isfinite(values)]
            low, high = (np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan))
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "point": point[name][metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_boot": n_boot,
                    "n_dates": n_dates,
                }
            )
    return pd.DataFrame(rows)


def rank_by_selection_rule(metrics: pd.DataFrame) -> pd.DataFrame:
    """Order models by the project's selection rule, best first (returns a new frame).

    Priority: fewest missed Unhealthy-or-worse days (`miss_rate`, lower first), then `accuracy`
    (higher first), then `macro_f1_all` (higher first). A missed bad day costs more health-wise
    than a false alarm, so safety comes before overall accuracy.
    """
    return metrics.sort_values(
        ["miss_rate", "accuracy", "macro_f1_all"], ascending=[True, False, False], kind="stable", na_position="last"
    ).reset_index(drop=True)
