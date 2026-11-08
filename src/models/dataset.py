"""X/y assembly and preprocessing for the daily feature table (model contract §2).

`make_xy` turns a daily feature frame into model-ready `(X, y)`: it selects
`FEATURE_COLUMNS`, drops rows with no target, and — for `y_band` — applies the
Hazardous -> Very Unhealthy merge (`src.classify.bands.merge_bands`) so classes
are `0..4`. `Preprocessor` then imputes and scales `X` into the float64 NumPy
array every model (sklearn, NumPy, PyTorch) consumes (contract §1).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classify.bands import merge_bands
from src.data.features import COLUMN_ORDER

_KEY_COLUMNS = {"sensor_id", "date"}
_TARGET_COLUMNS = {"y_pm25", "y_hours", "y_band"}
# `pm25_mean_24h` is identical to `pm25_prev_day_mean` by construction (issue time is
# midnight) — drop it so linear models don't see a duplicated feature (roadmap Phase 2 note).
_DUPLICATE_COLUMNS = {"pm25_mean_24h"}

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for col in COLUMN_ORDER if col not in _KEY_COLUMNS | _TARGET_COLUMNS | _DUPLICATE_COLUMNS
)

SENSOR_COLUMNS: tuple[str, ...] = tuple(col for col in FEATURE_COLUMNS if col.startswith("sensor_"))
_SCALED_COLUMNS: tuple[str, ...] = tuple(col for col in FEATURE_COLUMNS if col not in SENSOR_COLUMNS)

_TARGETS = {"y_pm25", "y_band"}


def make_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and target series for `target` in `{"y_pm25", "y_band"}`.

    Rows where the target is NaN are dropped. For `y_band`, the Hazardous (5) ->
    Very Unhealthy (4) merge is applied first, so returned classes are `0..4`.
    The feature frame keeps `FEATURE_COLUMNS` only, in that order; it is not yet
    imputed or scaled — pass it through `Preprocessor` next.
    """
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {_TARGETS}, got {target!r}")

    y_full = merge_bands(df["y_band"]) if target == "y_band" else df["y_pm25"]
    mask = y_full.notna()

    X = df.loc[mask, list(FEATURE_COLUMNS)].reset_index(drop=True)
    y = y_full.loc[mask].reset_index(drop=True)
    y = y.astype(int) if target == "y_band" else y.astype(float)
    return X, y


class Preprocessor:
    """Median-impute + standard-scale `FEATURE_COLUMNS`; one-hot `sensor_*` pass through unscaled.

    Fit on the training rows of one fold only (contract §2) — never on validation
    or test rows — then reused to transform both. `transform`/`fit_transform`
    always return float64 NumPy arrays, column order: scaled features first (in
    `FEATURE_COLUMNS` order, sensors excluded), then the sensor one-hot columns.
    """

    def __init__(self, feature_columns: Sequence[str] = FEATURE_COLUMNS):
        self._feature_columns = tuple(feature_columns)
        sensor_cols = [c for c in self._feature_columns if c.startswith("sensor_")]
        scaled_cols = [c for c in self._feature_columns if c not in sensor_cols]
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("scaled", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), scaled_cols),
                ("sensor", "passthrough", sensor_cols),
            ]
        )

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self._column_transformer.fit(X[list(self._feature_columns)])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self._column_transformer.transform(X[list(self._feature_columns)])
        return np.asarray(transformed, dtype=np.float64)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)
