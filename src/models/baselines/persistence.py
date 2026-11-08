"""Persistence baseline: tomorrow's PM2.5 = today's (spec §6.2 step 0).

Design: every other model in this project (sklearn, NumPy, PyTorch) is fit on
`Preprocessor`-scaled `X` (model contract §1) — but scaling `pm25_prev_day_mean`
and then reading it back off would make the "prediction" a z-score, not a PM2.5
value, which would break `regression_metrics` (MAE/skill need real units).
`PersistenceRegressor` therefore still satisfies the §1 interface (`fit`,
`predict`, `get_params`, `set_params`), but is deliberately fed the **unscaled**
feature matrix (same columns/order as `FEATURE_COLUMNS`, imputed or not — Phase 2
guarantees `pm25_prev_day_mean` is never NaN) instead of the preprocessed one.
`src.evaluate.run_cv` is the only caller and does exactly that: it passes this
model the raw `X_*_df.to_numpy()`, while every other model gets `Preprocessor`
output.
"""

from __future__ import annotations

import numpy as np


class PersistenceRegressor:
    """Predicts `X[:, feature_index]` (`pm25_prev_day_mean`'s position in `FEATURE_COLUMNS`).

    `fit` is a no-op (there is nothing to learn) but is kept so this plugs into
    the same `fit`/`predict` loop as every other model.
    """

    def __init__(self, feature_index: int, random_state: int | None = None):
        self.feature_index = feature_index
        self.random_state = random_state  # unused; kept for interface/sklearn-style symmetry

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceRegressor":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)[:, self.feature_index]

    def get_params(self, deep: bool = True) -> dict:
        return {"feature_index": self.feature_index, "random_state": self.random_state}

    def set_params(self, **params) -> "PersistenceRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self
