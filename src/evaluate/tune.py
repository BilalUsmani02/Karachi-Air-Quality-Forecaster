"""Score one model configuration across the CV folds, and search over configurations (Phase 5).

The folds are prepared once (`prepare_folds`): rows sorted by date, the `Preprocessor` fitted on
each fold's training rows, and both targets extracted. Every candidate configuration is then
scored on exactly the same matrices, so a grid search is fair and fast.

Selection uses the **pooled** metric (all folds' test predictions concatenated, scored once):
MAE for regressors, macro-F1 for classifiers. The final-test period is never loaded here
(`project_splits` carves it off), but note that choosing settings on these folds makes the CV
scores slightly optimistic — the untouched final test is what keeps the project honest.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.data.split import project_splits
from src.evaluate.metrics import classification_metrics, regression_metrics
from src.evaluate.run_cv import _to_band
from src.models.dataset import FEATURE_COLUMNS, Preprocessor, make_xy

PERSISTENCE_COLUMN = "pm25_prev_day_mean"


@dataclass(frozen=True)
class PreparedFold:
    index: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray        # PM2.5
    y_test: np.ndarray
    yb_train: np.ndarray       # merged band 0..4
    yb_test: np.ndarray
    persistence_test: np.ndarray
    meta_test: pd.DataFrame    # sensor_id, date


def prepare_folds(df: pd.DataFrame) -> list[PreparedFold]:
    """Build the 8 CV folds once: sorted by date, preprocessed on each fold's training rows."""
    df = df.sort_values(["date", "sensor_id"], kind="stable").reset_index(drop=True)
    folds, _dev, _final_test = project_splits(df)
    prepared = []
    for fold in folds:
        train = fold.train.loc[fold.train["y_pm25"].notna()]
        test = fold.test.loc[fold.test["y_pm25"].notna()]
        X_train_df, y_train = make_xy(train, "y_pm25")
        X_test_df, y_test = make_xy(test, "y_pm25")
        _, yb_train = make_xy(train, "y_band")
        _, yb_test = make_xy(test, "y_band")
        pre = Preprocessor()
        prepared.append(
            PreparedFold(
                index=fold.index,
                X_train=pre.fit_transform(X_train_df),
                X_test=pre.transform(X_test_df),
                y_train=y_train.to_numpy(np.float64),
                y_test=y_test.to_numpy(np.float64),
                yb_train=yb_train.to_numpy(np.int64),
                yb_test=yb_test.to_numpy(np.int64),
                persistence_test=X_test_df[PERSISTENCE_COLUMN].to_numpy(np.float64),
                meta_test=test[["sensor_id", "date"]].reset_index(drop=True),
            )
        )
    return prepared


@dataclass(frozen=True)
class Score:
    """Pooled CV result for one configuration."""

    params: dict
    metrics: dict              # pooled metrics (regression and/or classification)
    predictions: pd.DataFrame  # fold, sensor_id, date, y_true, y_pred (+ band columns)
    seconds: float


def _stack(folds: list[PreparedFold], per_fold: list[np.ndarray], attr: str) -> pd.DataFrame:
    meta = pd.concat([f.meta_test.assign(fold=f.index) for f in folds], ignore_index=True)
    return meta.assign(y_true=np.concatenate([getattr(f, attr) for f in folds]), y_pred=np.concatenate(per_fold))


def score_regressor(
    folds: list[PreparedFold], factory: Callable[..., Any], params: Mapping[str, Any] | None = None
) -> Score:
    """Fit `factory(**params)` on every fold; pooled MAE/RMSE/R²/skill plus regress-then-band metrics."""
    params = dict(params or {})
    start = time.perf_counter()
    preds = [np.asarray(factory(**params).fit(f.X_train, f.y_train).predict(f.X_test), np.float64) for f in folds]
    seconds = time.perf_counter() - start

    table = _stack(folds, preds, "y_test")
    table = table.assign(
        persistence=np.concatenate([f.persistence_test for f in folds]),
        band_true=np.concatenate([f.yb_test for f in folds]),
        band_pred=_to_band(table["y_pred"].to_numpy()),
    )
    metrics = {
        **regression_metrics(table["y_true"], table["y_pred"], table["persistence"]),
        **{f"band_{k}": v for k, v in classification_metrics(table["band_true"], table["band_pred"]).items()},
    }
    return Score(params, metrics, table, seconds)


def score_classifier(
    folds: list[PreparedFold], factory: Callable[..., Any], params: Mapping[str, Any] | None = None
) -> Score:
    """Fit `factory(**params)` on every fold's band labels; pooled classification metrics."""
    params = dict(params or {})
    start = time.perf_counter()
    preds = [np.asarray(factory(**params).fit(f.X_train, f.yb_train).predict(f.X_test), np.int64) for f in folds]
    seconds = time.perf_counter() - start

    table = _stack(folds, preds, "yb_test")
    metrics = classification_metrics(table["y_true"], table["y_pred"])
    return Score(params, metrics, table, seconds)


def param_grid(grid: Mapping[str, list]) -> list[dict]:
    """All combinations of a {name: [values]} grid, in a stable order."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def search(
    folds: list[PreparedFold],
    factory: Callable[..., Any],
    candidates: list[dict],
    *,
    kind: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every candidate; return one row per candidate, best first.

    `kind="regression"` ranks by pooled MAE (lower is better); `kind="classification"` ranks by
    pooled macro-F1 (higher is better). Regression rows also carry `band_macro_f1`.
    """
    if kind not in {"regression", "classification"}:
        raise ValueError(f"kind must be 'regression' or 'classification', got {kind!r}")
    scorer = score_regressor if kind == "regression" else score_classifier
    rows = []
    for i, params in enumerate(candidates, 1):
        score = scorer(folds, factory, params)
        m = score.metrics
        row = {**params, "seconds": round(score.seconds, 2)}
        if kind == "regression":
            row.update(mae=m["mae"], rmse=m["rmse"], r2=m["r2"], skill=m["skill"], band_macro_f1=m["band_macro_f1"])
        else:
            row.update(accuracy=m["accuracy"], macro_f1=m["macro_f1"])
        rows.append(row)
        if verbose:
            key = "mae" if kind == "regression" else "macro_f1"
            print(f"[{i}/{len(candidates)}] {key}={row[key]:.4f} {params}", flush=True)
    result = pd.DataFrame(rows)
    by, ascending = ("mae", True) if kind == "regression" else ("macro_f1", False)
    return result.sort_values(by, ascending=ascending, kind="stable").reset_index(drop=True)
