"""Rebuild the production ensemble's cross-validation predictions from saved tuning files.

The production model is an equal-weight average of Gradient Boosting (`hist_gb`) and the NumPy
network (`mlp_numpy`), so its CV prediction for a sensor-day is simply the mean of the two
members' saved CV predictions for that same sensor-day. Nothing is retrained.

The member files (`results/tuning/<name>_cv_predictions.parquet`, written by `tune.py`) each hold
the same pooled CV row set: the 8 rolling-origin test blocks (1 Jan - 28 Aug 2026), one row per
labelled sensor-day (1,458 rows). `load_ensemble_cv` checks that the members really are aligned
row-for-row (same `fold`, `sensor_id`, `date`, `y_true`) before averaging, and fails loudly if not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.evaluate.run_cv import _to_band

TUNING_DIR = RESULTS_DIR / "tuning"
ENSEMBLE_MEMBERS: tuple[str, ...] = ("hist_gb", "mlp_numpy")
KEY_COLUMNS = ["fold", "sensor_id", "date"]


def cv_predictions_path(tuning_dir: Path, model: str) -> Path:
    """Path of one model's saved pooled-CV predictions."""
    return Path(tuning_dir) / f"{model}_cv_predictions.parquet"


def average_members(members: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight average of aligned regressor CV tables (tune.py `Score.predictions` format).

    Each table needs `fold`, `sensor_id`, `date`, `y_true`, `y_pred`; `persistence` and
    `band_true` are carried over from the first table when present. Tables are aligned by sorting
    on (`fold`, `sensor_id`, `date`); a different row set or a different `y_true` on the same row
    raises `ValueError`. Returns a new table whose `y_pred` is the mean and `band_pred` its band.
    Inputs are not modified.
    """
    if not members:
        raise ValueError("need at least one member table")
    ordered = [m.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True) for m in members]
    first = ordered[0]
    for other in ordered[1:]:
        if len(other) != len(first) or not other[KEY_COLUMNS].equals(first[KEY_COLUMNS]):
            raise ValueError("ensemble members do not cover the same (fold, sensor_id, date) rows")
        if not np.allclose(other["y_true"].to_numpy(np.float64), first["y_true"].to_numpy(np.float64)):
            raise ValueError("ensemble members disagree on y_true for the same rows")

    mean_pred = np.mean([m["y_pred"].to_numpy(np.float64) for m in ordered], axis=0)
    keep = [c for c in [*KEY_COLUMNS, "y_true", "persistence", "band_true"] if c in first.columns]
    return first[keep].assign(y_pred=mean_pred, band_pred=_to_band(mean_pred))


def load_ensemble_cv(tuning_dir: Path = TUNING_DIR, members: tuple[str, ...] = ENSEMBLE_MEMBERS) -> pd.DataFrame:
    """The production ensemble's pooled CV predictions, rebuilt from the members' saved files."""
    tables = [pd.read_parquet(cv_predictions_path(tuning_dir, name)) for name in members]
    return average_members(tables)
