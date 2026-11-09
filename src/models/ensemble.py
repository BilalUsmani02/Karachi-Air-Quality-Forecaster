"""Averaging ensemble: fit several regressors on the same data and average their predictions.

Different model families make different mistakes (Phase 5: Gradient Boosting and the NumPy
network correlate at r ≈ 0.97, not 1.0), so the average is usually more accurate than either.
Equal weights on purpose: fitting weights on the CV predictions would tune to the same folds
the models were selected on.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class AveragingRegressor:
    """Model-contract regressor whose prediction is the mean of its members' predictions."""

    def __init__(self, members: dict[str, Callable[[], Any]]) -> None:
        if not members:
            raise ValueError("AveragingRegressor needs at least one member")
        self.members = dict(members)

    def get_params(self, deep: bool = False) -> dict:
        return {"members": dict(self.members)}

    def set_params(self, **params: Any) -> "AveragingRegressor":
        if set(params) - {"members"}:
            raise ValueError(f"unknown parameters: {sorted(set(params) - {'members'})}")
        if "members" in params:
            self.members = dict(params["members"])
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AveragingRegressor":
        self.models_ = {name: factory().fit(X, y) for name, factory in self.members.items()}
        return self

    def predict_members(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Each member's prediction, for showing how much the models agree."""
        if not hasattr(self, "models_"):
            raise RuntimeError("AveragingRegressor is not fitted yet; call fit() first")
        return {name: np.asarray(model.predict(X), dtype=np.float64) for name, model in self.models_.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(np.column_stack(list(self.predict_members(X).values())), axis=1)
