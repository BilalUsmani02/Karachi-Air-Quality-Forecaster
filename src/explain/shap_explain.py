"""SHAP explanations for the production ensemble (Phase 6, spec §8).

The spec originally planned DeepExplainer on a PyTorch twin of the NumPy network. The model
that actually ships (`src.models.final.final_ensemble`) is an equal-weight average of a tuned
Gradient Boosting regressor and the tuned NumPy network (`AveragingRegressor`, see
`src/models/ensemble.py`) — DeepExplainer only explains one PyTorch module, not an average of
two unrelated model families, so it cannot explain what the app actually predicts.

Design decision (spec §8, "already made"): explain the real production prediction with
model-agnostic SHAP — `shap.Explainer(model.predict, masker)` with an `Independent` masker on a
small, fixed-seed background sample of *preprocessed* (scaled) rows. For 58 features this
auto-selects (we pin it explicitly) the permutation algorithm, which is fast enough to run live
in the app (≈0.5-1 s per row on a laptop; see `tests/test_explain_shap.py` and
`docs/report/explainability.md` for a measurement on real data).

Column order note (checked against `src/models/dataset.py::Preprocessor` and
`src/models/numpy_nn/variant.py::TRANSFORMED_COLUMNS`): `Preprocessor.transform` puts the
scaled (non-sensor) `FEATURE_COLUMNS` first, in their original order, then the `sensor_*`
one-hots — and `FEATURE_COLUMNS` already ends with the sensor columns (see
`src/data/features.py::COLUMN_ORDER`). So `FEATURE_COLUMNS` order *is* the preprocessed output's
column order; no separate transformed-column list is needed here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import SEED
from src.models.dataset import FEATURE_COLUMNS

BACKGROUND_SIZE = 100
GLOBAL_SAMPLE_SIZE = 300


@dataclass(frozen=True)
class Contribution:
    """One feature's effect on one forecast, in the model's output unit (µg/m³)."""

    feature: str
    raw_value: float
    shap_value: float


@dataclass(frozen=True)
class Explanation:
    """A single forecast's SHAP breakdown.

    `contributions` holds every feature, sorted by `|shap_value|` descending (largest first).
    `prediction` is the model's actual output for the row (not reconstructed from SHAP);
    `additivity_error` is how far `base_value + sum(shap_value)` is from it — SHAP's additivity
    guarantee, which the permutation explainer only satisfies approximately.
    """

    base_value: float
    prediction: float
    contributions: tuple[Contribution, ...]

    @property
    def additivity_error(self) -> float:
        return self.prediction - (self.base_value + sum(c.shap_value for c in self.contributions))


def load_bundle(path: Path) -> dict[str, Any]:
    """Load the production pickle bundle `{"preprocessor", "model", "trained_on"}`."""
    return pickle.loads(Path(path).read_bytes())


def _gb_member(model: Any) -> HistGradientBoostingRegressor:
    """Find the Gradient Boosting member of the ensemble (for the fast, exact TreeExplainer).

    Looked up by type, not by the member's dict key, so this keeps working even if
    `src.models.final.ENSEMBLE_MEMBERS` is ever renamed or reordered.
    """
    members = getattr(model, "models_", None)
    if members is None:
        raise ValueError("model has no fitted `models_` members; is it an AveragingRegressor?")
    for member in members.values():
        if isinstance(member, HistGradientBoostingRegressor):
            return member
    raise ValueError("ensemble has no HistGradientBoostingRegressor member for TreeExplainer")


def build_explainer(
    bundle: dict[str, Any],
    background_X: np.ndarray,
    seed: int = SEED,
    background_size: int = BACKGROUND_SIZE,
) -> shap.Explainer:
    """Model-agnostic SHAP explainer over `bundle["model"].predict`.

    Draws a fixed-seed sample of `background_size` rows from `background_X` (already
    preprocessed) for the `Independent` masker, then wraps the ensemble's `predict` in a
    permutation explainer — the only practical model-agnostic choice at 58 features (exact
    SHAP is O(2^d)).
    """
    model = bundle["model"]
    background_X = np.asarray(background_X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = min(background_size, len(background_X))
    idx = rng.choice(len(background_X), size=n, replace=False)
    masker = shap.maskers.Independent(background_X[idx], max_samples=n)
    return shap.Explainer(model.predict, masker, seed=seed, algorithm="permutation")


def explain_row(explainer: shap.Explainer, X_row_scaled: np.ndarray, X_row_raw: pd.Series) -> Explanation:
    """Explain one preprocessed row.

    `X_row_raw` is the matching row of *unscaled* feature values (a `pandas.Series`, feature
    name -> value, same order as `X_row_scaled`'s columns) — used only to report each
    contribution's raw value, never fed to the model.
    """
    X_row_scaled = np.asarray(X_row_scaled, dtype=np.float64).reshape(1, -1)
    if len(X_row_raw) != X_row_scaled.shape[1]:
        raise ValueError(
            f"X_row_raw has {len(X_row_raw)} entries, X_row_scaled has {X_row_scaled.shape[1]} columns"
        )

    result = explainer(X_row_scaled)
    shap_values = np.asarray(result.values[0], dtype=np.float64)
    base_value = float(np.asarray(result.base_values).reshape(-1)[0])
    prediction = float(np.asarray(explainer.model(X_row_scaled)).reshape(-1)[0])

    raw_values = X_row_raw.to_numpy(dtype=np.float64)
    contributions = tuple(
        sorted(
            (
                Contribution(feature=str(name), raw_value=float(raw), shap_value=float(sv))
                for name, raw, sv in zip(X_row_raw.index, raw_values, shap_values)
            ),
            key=lambda c: abs(c.shap_value),
            reverse=True,
        )
    )
    return Explanation(base_value=base_value, prediction=prediction, contributions=contributions)


def global_importance(
    bundle: dict[str, Any],
    X: np.ndarray,
    sample_size: int = GLOBAL_SAMPLE_SIZE,
    seed: int = SEED,
    background_size: int = BACKGROUND_SIZE,
    feature_names: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Global feature importance by two methods, for comparison in the report.

    - `tree_mean_abs_shap`: `shap.TreeExplainer` on the ensemble's Gradient Boosting member,
      exact, over the *whole* table `X` (fast — no sampling needed for a tree model).
    - `ensemble_mean_abs_shap`: the model-agnostic permutation explainer on the whole ensemble,
      over a fixed-seed sample of `sample_size` rows (expensive per row, so sampled).

    Returns one row per feature, sorted by `ensemble_mean_abs_shap` descending.
    """
    model = bundle["model"]
    X = np.asarray(X, dtype=np.float64)

    tree_explainer = shap.TreeExplainer(_gb_member(model))
    tree_shap = np.asarray(tree_explainer.shap_values(X))
    tree_importance = np.abs(tree_shap).mean(axis=0)

    explainer = build_explainer(bundle, X, seed=seed, background_size=background_size)
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(X))
    sample_idx = rng.choice(len(X), size=n, replace=False)
    ensemble_shap = np.asarray(explainer(X[sample_idx]).values)
    ensemble_importance = np.abs(ensemble_shap).mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "tree_mean_abs_shap": tree_importance,
                "ensemble_mean_abs_shap": ensemble_importance,
            }
        )
        .sort_values("ensemble_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
