"""Extra ensemble diagnostics, computed from SAVED predictions only (no retraining).

`ensemble_cv.py` rebuilds the production (two-member) ensemble's pooled-CV predictions. This
module answers two further questions about the member models, both from the same saved
per-model files (`results/tuning/<name>_cv_predictions.parquet`, written by `tune.py`):

- **Four-model average.** What if every tuned regressor (Ridge, Random Forest, Gradient
  Boosting, the NumPy network) were averaged with equal weight, instead of just the two that
  were shipped? `four_model_average_metrics` answers this with the same pooled-CV metric code
  used everywhere else in this project.
- **Member correlation.** How different are the mistakes the two shipped members make?
  `member_correlation` is the Pearson correlation between two members' pooled-CV predictions
  on the same rows.

    uv run python -m src.evaluate.ensemble_diagnostics [--tuning-dir results/tuning] [--out results/tuning/ensemble_diagnostics.json]

Writes one JSON file with both results, in the same flat style as `ensemble_cv_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.ensemble_cv import KEY_COLUMNS, TUNING_DIR, cv_predictions_path, load_ensemble_cv
from src.evaluate.metrics import regression_metrics

FOUR_MODELS: tuple[str, ...] = ("ridge", "random_forest", "hist_gb", "mlp_numpy")
CORRELATION_PAIR: tuple[str, str] = ("hist_gb", "mlp_numpy")
OUT_PATH = TUNING_DIR / "ensemble_diagnostics.json"


def four_model_average_metrics(tuning_dir: Path = TUNING_DIR, members: tuple[str, ...] = FOUR_MODELS) -> dict:
    """Pooled-CV MAE / RMSE / R² / skill of the equal-weight average of `members`.

    Reuses `ensemble_cv.load_ensemble_cv`, which already checks that every member's saved file
    covers exactly the same (fold, sensor_id, date) rows with the same `y_true`.
    """
    table = load_ensemble_cv(tuning_dir, members=members)
    metrics = regression_metrics(table["y_true"], table["y_pred"], table["persistence"])
    return {"members": list(members), "n_rows": int(len(table)), **metrics}


def member_correlation(tuning_dir: Path = TUNING_DIR, pair: tuple[str, str] = CORRELATION_PAIR) -> float:
    """Pearson correlation between two members' pooled-CV predictions on the same rows."""
    name_a, name_b = pair
    a = pd.read_parquet(cv_predictions_path(tuning_dir, name_a)).sort_values(KEY_COLUMNS, kind="stable")
    b = pd.read_parquet(cv_predictions_path(tuning_dir, name_b)).sort_values(KEY_COLUMNS, kind="stable")
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if len(a) != len(b) or not a[KEY_COLUMNS].equals(b[KEY_COLUMNS]):
        raise ValueError(f"{name_a} and {name_b} CV predictions do not cover the same rows")
    corr = np.corrcoef(a["y_pred"].to_numpy(np.float64), b["y_pred"].to_numpy(np.float64))[0, 1]
    return float(corr)


def build_diagnostics(tuning_dir: Path = TUNING_DIR) -> dict:
    """Both diagnostics, bundled with a `how` field documenting the method (matches the style of
    `ensemble_cv_metrics.json`)."""
    return {
        "four_model_average": four_model_average_metrics(tuning_dir),
        "member_correlation": {"pair": list(CORRELATION_PAIR), "pearson_r": member_correlation(tuning_dir)},
        "how": (
            "src.evaluate.ensemble_diagnostics: four_model_average_metrics averages the ridge, "
            "random_forest, hist_gb and mlp_numpy pooled-CV predictions "
            "(src.evaluate.ensemble_cv.load_ensemble_cv) and scores them with "
            "src.evaluate.metrics.regression_metrics; member_correlation is the Pearson correlation "
            "between the hist_gb and mlp_numpy pooled-CV predictions, from the same saved files."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Four-model ensemble average and member correlation, from saved CV predictions")
    parser.add_argument("--tuning-dir", default=TUNING_DIR, type=Path)
    parser.add_argument("--out", default=OUT_PATH, type=Path)
    args = parser.parse_args(argv)

    diagnostics = build_diagnostics(args.tuning_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
