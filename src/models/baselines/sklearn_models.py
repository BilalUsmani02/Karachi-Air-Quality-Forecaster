"""Factories for the scikit-learn baselines (spec §6.2 steps 1-2, §6.4).

Each factory returns a fresh, configured, unfit estimator with `random_state=
SEED` (project-wide reproducibility, `src.config`). They already satisfy the
model contract §1 interface (`fit`/`predict`/`get_params`/`set_params`).
Tuning hyperparameters is Phase 5 — these are reasonable defaults, not tuned.
"""

from __future__ import annotations

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from src.config import SEED


def ridge(**overrides) -> Ridge:
    """Ridge regression — the official regression baseline (spec §6.2 step 1)."""
    params = {"random_state": SEED, **overrides}
    return Ridge(**params)


def random_forest_regressor(**overrides) -> RandomForestRegressor:
    params = {"n_estimators": 300, "max_depth": None, "n_jobs": -1, "random_state": SEED, **overrides}
    return RandomForestRegressor(**params)


def logistic_regression(**overrides) -> LogisticRegression:
    """Multinomial logistic regression, balanced for Karachi's class imbalance (spec §6.4)."""
    params = {
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_state": SEED,
        **overrides,
    }
    return LogisticRegression(**params)


def random_forest_classifier(**overrides) -> RandomForestClassifier:
    params = {
        "n_estimators": 300,
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        **overrides,
    }
    return RandomForestClassifier(**params)


def hist_gb_regressor(**overrides) -> HistGradientBoostingRegressor:
    """Gradient-boosted trees (spec §6.2 step 2 extra family) — fast histogram-based boosting.

    `early_stopping=False` by default: Phase 5 tuning scores every candidate on the fixed
    `max_iter`/`learning_rate` grid itself (rolling-origin CV already guards against
    overfitting), rather than letting each candidate silently stop at a different iteration
    count, which would make the grid harder to compare.
    """
    params = {
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "max_iter": 200,
        "early_stopping": False,
        "random_state": SEED,
        **overrides,
    }
    return HistGradientBoostingRegressor(**params)


def hist_gb_classifier(**overrides) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees for the direct band classifier, balanced for class imbalance (§6.4)."""
    params = {
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "max_iter": 200,
        "early_stopping": False,
        "class_weight": "balanced",
        "random_state": SEED,
        **overrides,
    }
    return HistGradientBoostingClassifier(**params)
