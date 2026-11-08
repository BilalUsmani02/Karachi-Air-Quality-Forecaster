"""Regression and classification metrics shared by every model (model contract §3, spec §6.5)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)
from sklearn.metrics import mean_absolute_error as _mae

ArrayLike = np.ndarray | Iterable[float]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike, y_persistence: ArrayLike) -> dict:
    """MAE, RMSE, R², and skill vs the persistence baseline.

    `skill = 1 - MAE_model / MAE_persistence` (contract §3): 0 means no better than
    persistence, 1 would be a perfect model, negative means worse than persistence.
    `y_persistence` must be the persistence model's prediction for the *same* rows
    as `y_true`/`y_pred`, so the two MAEs are directly comparable.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_persistence = np.asarray(y_persistence, dtype=np.float64)

    mae = float(_mae(y_true, y_pred))
    mae_persistence = float(_mae(y_true, y_persistence))
    skill = 1.0 - mae / mae_persistence if mae_persistence > 0 else float("nan")

    return {
        "mae": mae,
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "skill": skill,
    }


def classification_metrics(y_true: ArrayLike, y_pred: ArrayLike, labels: Iterable[int] = range(5)) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/support, and confusion matrix.

    `labels` fixes the full label set (default `0..4`, contract §3) so a fold that
    happens to lack a class still reports it (support 0, precision/recall 0 via
    `zero_division=0`) instead of raising or silently dropping it. Confusion matrix
    is `len(labels) x len(labels)`, rows = true, columns = predicted, as nested
    lists (JSON/CSV friendly).
    """
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, _f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        int(label): {"precision": float(p), "recall": float(r), "support": int(s)}
        for label, p, r, s in zip(labels, precision, recall, support)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
