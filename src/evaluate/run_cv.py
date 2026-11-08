"""Rolling-origin cross-validation runner for the baselines (model contract §3, spec §6.2/§6.4).

For each `project_splits` fold: fit `Preprocessor` on that fold's train rows only,
fit every regressor and classifier, predict the fold's test rows, and collect
predictions. Regressors also get a derived **regress-then-band** class
prediction (`bands_for` on the predicted PM2.5, then `merge_bands`). Metrics are
reported per fold and pooled (every fold's test predictions concatenated, then
scored once) — pooled is the headline number (contract §3).

The final-test period is never touched: `project_splits` carves it off before
any fold is built, and only `dev` rows reach this module.

CLI: `uv run python -m src.evaluate.run_cv` loads `data/processed/features.parquet`
and writes `results/baselines_cv_folds.csv`, `results/baselines_cv_pooled.csv`,
`results/baselines_cv_predictions.parquet`, `results/baselines_summary.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from src.classify.bands import bands_for, merge_bands
from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.data.split import project_splits
from src.evaluate.metrics import classification_metrics, regression_metrics
from src.models.baselines.persistence import PersistenceRegressor
from src.models.baselines.sklearn_models import (
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    ridge,
)
from src.models.dataset import FEATURE_COLUMNS, Preprocessor, make_xy
from src.models.numpy_nn.network import MLPRegressorNP

N_CLASS_LABELS = 5  # band indices 0..4, after the Hazardous -> Very Unhealthy merge (contract §3)
PERSISTENCE_COLUMN = "pm25_prev_day_mean"
PERSISTENCE_MODEL_NAME = "persistence"


class ModelLike(Protocol):
    """The model contract §1 interface — any sklearn/NumPy/PyTorch model satisfies this."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelLike": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


ModelFactory = Callable[[], ModelLike]


@dataclass(frozen=True)
class CVResult:
    """Per-fold and pooled metric tables, plus every fold-test prediction.

    `fold_regression`/`pooled_regression`: one row per (model, fold) / per model;
    columns `mae`, `rmse`, `r2`, `skill`, including the `"persistence"` model.
    `fold_classification`/`pooled_classification`: one row per (model, route,
    fold) / per (model, route); `route` is `"direct"` (classifiers) or
    `"regress_then_band"` (regressors); columns `accuracy`, `macro_f1`,
    `precision_i`/`recall_i`/`support_i` for `i` in `0..4`, `confusion_matrix`
    (JSON string of a 5x5 nested list).
    `predictions`: tidy table of every fold-test row: `fold`, `sensor_id`,
    `date`, `task` (`"regression"`/`"classification"`), `model`, `route`,
    `y_true`, `y_pred`.
    """

    fold_regression: pd.DataFrame
    pooled_regression: pd.DataFrame
    fold_classification: pd.DataFrame
    pooled_classification: pd.DataFrame
    predictions: pd.DataFrame


def _to_band(y_pred_pm25: np.ndarray) -> np.ndarray:
    """Regress-then-band: predicted PM2.5 -> merged band index.

    PM2.5 cannot be negative (`bands.pm25_to_band` raises on it), but a linear
    model can still predict a negative value on an easy day; those are clipped
    to 0 before banding, since that is the physically meaningful reading.
    """
    clipped = np.clip(np.asarray(y_pred_pm25, dtype=np.float64), 0.0, None)
    return merge_bands(bands_for(pd.Series(clipped))).astype(int).to_numpy()


def _regression_records(fold: int, meta: pd.DataFrame, model: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": fold,
            "sensor_id": meta["sensor_id"].to_numpy(),
            "date": meta["date"].to_numpy(),
            "task": "regression",
            "model": model,
            "route": "regression",
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )


def _classification_records(
    fold: int, meta: pd.DataFrame, model: str, route: str, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": fold,
            "sensor_id": meta["sensor_id"].to_numpy(),
            "date": meta["date"].to_numpy(),
            "task": "classification",
            "model": model,
            "route": route,
            "y_true": np.asarray(y_true, dtype=np.int64),
            "y_pred": np.asarray(y_pred, dtype=np.int64),
        }
    )


def cross_validate(
    df: pd.DataFrame,
    regressors: dict[str, ModelFactory],
    classifiers: dict[str, ModelFactory],
) -> CVResult:
    """Run every fold of `project_splits(df)`, fitting fresh models each time.

    `regressors`/`classifiers` map a model name to a zero-argument factory
    returning an unfit model satisfying `ModelLike` (so a tiny dummy class, or
    later the NumPy/PyTorch network's factory, plugs in exactly like sklearn).
    A factory is called once per fold, so folds never share fitted state.
    The persistence baseline is always included under the name `"persistence"`
    and does not need to be passed in `regressors`.
    Never loads the final-test rows: `project_splits` carves them off first.
    """
    # Time order matters: the networks' early stopping holds out the *last* training rows.
    df = df.sort_values(["date", "sensor_id"], kind="stable").reset_index(drop=True)
    folds, _dev, _final_test = project_splits(df)
    persistence_index = FEATURE_COLUMNS.index(PERSISTENCE_COLUMN)

    pred_frames: list[pd.DataFrame] = []

    for fold in folds:
        X_train_df, y_train = make_xy(fold.train, "y_pm25")
        X_test_df, y_test = make_xy(fold.test, "y_pm25")
        _, yb_train = make_xy(fold.train, "y_band")
        _, yb_test = make_xy(fold.test, "y_band")

        # Phase 2 guarantees y_band is NaN iff y_pm25 is NaN, so make_xy's dropna mask is the
        # same for both targets — the four arrays above, and meta below, are row-aligned.
        mask_test = fold.test["y_pm25"].notna()
        meta_test = fold.test.loc[mask_test, ["sensor_id", "date"]].reset_index(drop=True)

        preprocessor = Preprocessor()
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)

        # Persistence reads the raw (unscaled) column directly — see persistence.py.
        persistence = PersistenceRegressor(feature_index=persistence_index)
        persistence.fit(X_train_df.to_numpy(dtype=np.float64), y_train.to_numpy())
        y_pred_persistence = persistence.predict(X_test_df.to_numpy(dtype=np.float64))
        pred_frames.append(_regression_records(fold.index, meta_test, PERSISTENCE_MODEL_NAME, y_test.to_numpy(), y_pred_persistence))
        pred_frames.append(
            _classification_records(
                fold.index, meta_test, PERSISTENCE_MODEL_NAME, "regress_then_band", yb_test.to_numpy(), _to_band(y_pred_persistence)
            )
        )

        for name, factory in regressors.items():
            model = factory()
            model.fit(X_train, y_train.to_numpy())
            y_pred = np.asarray(model.predict(X_test), dtype=np.float64)
            pred_frames.append(_regression_records(fold.index, meta_test, name, y_test.to_numpy(), y_pred))
            pred_frames.append(
                _classification_records(fold.index, meta_test, name, "regress_then_band", yb_test.to_numpy(), _to_band(y_pred))
            )

        for name, factory in classifiers.items():
            model = factory()
            model.fit(X_train, yb_train.to_numpy())
            yb_pred = np.asarray(model.predict(X_test), dtype=np.int64)
            pred_frames.append(_classification_records(fold.index, meta_test, name, "direct", yb_test.to_numpy(), yb_pred))

    predictions = pd.concat(pred_frames, ignore_index=True)
    fold_regression, pooled_regression = _regression_tables(predictions)
    fold_classification, pooled_classification = _classification_tables(predictions)
    return CVResult(fold_regression, pooled_regression, fold_classification, pooled_classification, predictions)


def _regression_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = predictions.loc[predictions["task"] == "regression"]
    persistence = reg.loc[reg["model"] == PERSISTENCE_MODEL_NAME, ["fold", "sensor_id", "date", "y_pred"]].rename(
        columns={"y_pred": "y_persistence"}
    )
    merged = reg.merge(persistence, on=["fold", "sensor_id", "date"], how="left")

    def _score(group: pd.DataFrame) -> pd.Series:
        return pd.Series(regression_metrics(group["y_true"], group["y_pred"], group["y_persistence"]))

    fold_table = _grouped_scores(merged, ["model", "fold"], _score)
    pooled_table = _grouped_scores(merged, ["model"], _score)
    return fold_table.sort_values(["model", "fold"]).reset_index(drop=True), pooled_table.sort_values("model").reset_index(drop=True)


def _grouped_scores(df: pd.DataFrame, keys: Sequence[str], score_fn: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """`df.groupby(keys).apply(score_fn)` without the pandas index-naming churn across versions."""
    rows = []
    for key, group in df.groupby(list(keys)):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row.update(score_fn(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten_classification(metrics: dict) -> dict:
    flat = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
    for label, stats in metrics["per_class"].items():
        flat[f"precision_{label}"] = stats["precision"]
        flat[f"recall_{label}"] = stats["recall"]
        flat[f"support_{label}"] = stats["support"]
    flat["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return flat


def _classification_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clf = predictions.loc[predictions["task"] == "classification"]

    def _score(group: pd.DataFrame) -> dict:
        return _flatten_classification(classification_metrics(group["y_true"], group["y_pred"], labels=range(N_CLASS_LABELS)))

    fold_rows = [
        {"model": model, "route": route, "fold": fold, **_score(group)}
        for (model, route, fold), group in clf.groupby(["model", "route", "fold"])
    ]
    pooled_rows = [
        {"model": model, "route": route, **_score(group)} for (model, route), group in clf.groupby(["model", "route"])
    ]

    fold_table = pd.DataFrame(fold_rows).sort_values(["model", "route", "fold"]).reset_index(drop=True)
    pooled_table = pd.DataFrame(pooled_rows).sort_values(["model", "route"]).reset_index(drop=True)
    return fold_table, pooled_table


# --- CLI ----------------------------------------------------------------------------------


def mlp_numpy() -> MLPRegressorNP:
    """The from-scratch network with its default settings (tuned in Phase 5)."""
    return MLPRegressorNP(random_state=SEED)


def default_regressors() -> dict[str, ModelFactory]:
    return {"ridge": ridge, "random_forest": random_forest_regressor, "mlp_numpy": mlp_numpy}


def default_classifiers() -> dict[str, ModelFactory]:
    return {"logistic_regression": logistic_regression, "random_forest": random_forest_classifier}


def _markdown_table(df: pd.DataFrame, columns: Sequence[str], precision: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [f"{row[c]:.{precision}f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _confusion_markdown(cm: list[list[int]]) -> str:
    n = len(cm)
    header = "| true \\ pred | " + " | ".join(str(i) for i in range(n)) + " |"
    sep = "| --- | " + " | ".join("---" for _ in range(n)) + " |"
    rows = [f"| {i} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    return "\n".join([header, sep, *rows])


def _write_summary(result: CVResult, path: Path) -> None:
    recall_cols = [f"recall_{i}" for i in range(N_CLASS_LABELS)]
    best = result.pooled_classification.sort_values("macro_f1", ascending=False).iloc[0]

    lines = [
        "# Baselines — pooled cross-validation results",
        "",
        "Pooled = every fold's test predictions concatenated, metrics computed once (headline number).",
        "",
        "## Regression",
        _markdown_table(result.pooled_regression, ["model", "mae", "rmse", "r2", "skill"]),
        "",
        "## Classification",
        _markdown_table(result.pooled_classification, ["model", "route", "accuracy", "macro_f1", *recall_cols]),
        "",
        f"## Confusion matrix — best classifier by macro-F1: {best['model']} ({best['route']})",
        _confusion_markdown(json.loads(best["confusion_matrix"])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(features_path: Path | None = None, results_dir: Path | None = None) -> CVResult:
    """Load the feature table, run the baselines, write the four `results/` files."""
    features_path = features_path or PROCESSED_DIR / "features.parquet"
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    result = cross_validate(df, default_regressors(), default_classifiers())

    pd.concat([result.fold_regression, result.fold_classification], ignore_index=True).to_csv(
        results_dir / "baselines_cv_folds.csv", index=False
    )
    pd.concat([result.pooled_regression, result.pooled_classification], ignore_index=True).to_csv(
        results_dir / "baselines_cv_pooled.csv", index=False
    )
    result.predictions.to_parquet(results_dir / "baselines_cv_predictions.parquet", index=False)
    _write_summary(result, results_dir / "baselines_summary.md")

    return result


def main() -> None:
    result = run()
    print("=== Pooled regression ===")
    print(result.pooled_regression.to_string(index=False))
    print()
    print("=== Pooled classification ===")
    print(result.pooled_classification.to_string(index=False))


if __name__ == "__main__":
    main()
