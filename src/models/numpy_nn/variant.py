"""`MLPVariant`: the NumPy network plus a dropped-feature set and a residual-from-persistence target.

Found in Phase 5 tuning (`docs/report/model_selection.md`, `results/tuning/mlp_numpy_diagnosis.csv`):

- **feature_set** — `"no_calendar"` drops `doy_sin`, `doy_cos` and `month`. The feature table
  starts in June 2025, so early folds never see a March–May day of year; the network maps those
  unseen calendar values onto "winter" and overshoots.
- **target** — `"residual"` learns the change from a *fitted persistence* anchor,
  `a + b · pm25_prev_day_mean`, instead of the PM2.5 level. The model only sees the
  standard-scaled column, but a scaled column is a linear function of the raw one, so a
  least-squares fit of `y` on it recovers the same line as a fit on raw persistence. The anchor
  is fitted on the training rows only.

It satisfies the model contract (`fit(X, y)` / `predict(X)` on the preprocessed matrix).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import SEED
from src.models.dataset import FEATURE_COLUMNS, SENSOR_COLUMNS
from src.models.numpy_nn.network import MLPRegressorNP

PERSISTENCE_COLUMN = "pm25_prev_day_mean"

# Column order of `Preprocessor` output: scaled features first, then the sensor one-hots.
TRANSFORMED_COLUMNS: tuple[str, ...] = (
    *(c for c in FEATURE_COLUMNS if c not in SENSOR_COLUMNS),
    *SENSOR_COLUMNS,
)
CALENDAR_COLUMNS: tuple[str, ...] = ("doy_sin", "doy_cos", "month")
FEATURE_SETS: dict[str, tuple[str, ...]] = {"all": (), "no_calendar": CALENDAR_COLUMNS}
TARGETS = ("level", "residual")
PERSISTENCE_INDEX = TRANSFORMED_COLUMNS.index(PERSISTENCE_COLUMN)


class MLPVariant:
    """`MLPRegressorNP` plus an optional dropped-feature set and a residual-from-persistence target.

    Every other keyword goes straight to `MLPRegressorNP` (`hidden_sizes`, `l2`, ...).
    `regressor_cls` is the network class inside the wrapper; the NumPy-vs-PyTorch report swaps
    in `MLPRegressorTorch` so both networks get exactly the same columns and residual target.
    """

    regressor_cls: type = MLPRegressorNP

    def __init__(self, feature_set: str = "all", target: str = "level", **mlp_params: Any) -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {tuple(FEATURE_SETS)}, got {feature_set!r}")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        self.feature_set = feature_set
        self.target = target
        self.mlp_params = {"random_state": SEED, **mlp_params}

    def _columns(self, n_features: int) -> np.ndarray:
        if n_features != len(TRANSFORMED_COLUMNS):
            raise ValueError(f"expected {len(TRANSFORMED_COLUMNS)} preprocessed features, got {n_features}")
        dropped = set(FEATURE_SETS[self.feature_set])
        return np.array([i for i, c in enumerate(TRANSFORMED_COLUMNS) if c not in dropped])

    def _anchor(self, X: np.ndarray) -> np.ndarray:
        if self.target == "level":
            return np.zeros(len(X))
        return self.anchor_coef_[0] + self.anchor_coef_[1] * X[:, PERSISTENCE_INDEX]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPVariant":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.columns_ = self._columns(X.shape[1])
        design = np.column_stack([np.ones(len(X)), X[:, PERSISTENCE_INDEX]])
        self.anchor_coef_ = np.linalg.lstsq(design, y, rcond=None)[0]
        self.model_ = self.regressor_cls(**self.mlp_params).fit(X[:, self.columns_], y - self._anchor(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("MLPVariant is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X[:, self.columns_]) + self._anchor(X)
