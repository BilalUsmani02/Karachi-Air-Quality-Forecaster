"""The tuned models chosen in Phase 5, as zero-argument factories (model contract §1).

Settings come from the CV searches in `results/tuning/*_best.json` (8 rolling-origin folds,
Jan–Aug 2026). They are written out here, not read from those files, so the final system does
not depend on tuning output being present.

Pooled CV MAE (µg/m³): persistence 8.70 · Ridge 7.22 · Random Forest 6.54 · NumPy network 6.63 ·
Gradient Boosting 6.30 · **ensemble (Gradient Boosting + NumPy network) 6.16**.
"""

from __future__ import annotations

from src.models.baselines.sklearn_models import (
    hist_gb_regressor,
    random_forest_classifier,
    random_forest_regressor,
    ridge,
)
from src.models.ensemble import AveragingRegressor
from src.models.numpy_nn.variant import MLPVariant

RIDGE_PARAMS = {"alpha": 150.0}
RANDOM_FOREST_PARAMS = {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 5, "max_features": 0.5}
HIST_GB_PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 100, "l2_regularization": 0.1}
MLP_PARAMS = {
    "feature_set": "no_calendar",
    "target": "residual",
    "hidden_sizes": (32,),
    "activation": "relu",
    "loss": "mse",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "l2": 0.1,
    "dropout": 0.0,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "patience": 20,
    "max_epochs": 200,
}
RANDOM_FOREST_CLF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "class_weight": "balanced",
    "min_samples_leaf": 5,
    "max_features": 0.5,
}

# The two ensemble members: the strongest classical model and the from-scratch network.
ENSEMBLE_MEMBERS = ("hist_gb", "mlp_numpy")


def tuned_ridge():
    return ridge(**RIDGE_PARAMS)


def tuned_random_forest():
    return random_forest_regressor(**RANDOM_FOREST_PARAMS)


def tuned_hist_gb():
    return hist_gb_regressor(**HIST_GB_PARAMS)


def tuned_mlp_numpy():
    return MLPVariant(**MLP_PARAMS)


def tuned_random_forest_classifier():
    return random_forest_classifier(**RANDOM_FOREST_CLF_PARAMS)


def final_ensemble() -> AveragingRegressor:
    """The production forecaster: mean of tuned Gradient Boosting and the tuned NumPy network."""
    return AveragingRegressor({"hist_gb": tuned_hist_gb, "mlp_numpy": tuned_mlp_numpy})


TUNED_REGRESSORS = {
    "ridge": tuned_ridge,
    "random_forest": tuned_random_forest,
    "hist_gb": tuned_hist_gb,
    "mlp_numpy": tuned_mlp_numpy,
    "ensemble": final_ensemble,
}
TUNED_CLASSIFIERS = {"random_forest_clf": tuned_random_forest_classifier}
