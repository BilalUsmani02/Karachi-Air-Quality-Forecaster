"""Factories for the scikit-learn baselines (spec §6.2 steps 1-2, §6.4).

Each factory returns a fresh, configured, unfit estimator with `random_state=
SEED` (project-wide reproducibility, `src.config`). They already satisfy the
model contract §1 interface (`fit`/`predict`/`get_params`/`set_params`).
Tuning hyperparameters is Phase 5 — these are reasonable defaults, not tuned.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
