"""Tests for `src.explain.shap_explain` — synthetic, small, fast (no real model/data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.models.ensemble import AveragingRegressor
from src.explain.shap_explain import (
    Contribution,
    Explanation,
    build_explainer,
    explain_row,
    global_importance,
    load_bundle,
)


class LinearModel:
    """`y = X @ weights` exactly — makes the expected SHAP values easy to reason about."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)

    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.weights


@pytest.fixture
def linear_bundle():
    weights = np.array([5.0, -3.0, 1.0, 0.0])
    return {"model": LinearModel(weights)}


@pytest.fixture
def linear_background():
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, 4))


def _row_raw(values, names=("f0", "f1", "f2", "f3")) -> pd.Series:
    return pd.Series(dict(zip(names, values)))


def test_additivity_on_tiny_model(linear_bundle, linear_background):
    explainer = build_explainer(linear_bundle, linear_background, seed=0, background_size=20)
    row_scaled = np.array([1.0, 2.0, -1.0, 4.0])
    explanation = explain_row(explainer, row_scaled, _row_raw(row_scaled))

    assert explanation.prediction == pytest.approx(float(row_scaled @ linear_bundle["model"].weights))
    assert abs(explanation.additivity_error) < 1e-6


def test_contributions_sorted_by_magnitude(linear_bundle, linear_background):
    explainer = build_explainer(linear_bundle, linear_background, seed=1, background_size=20)
    row_scaled = np.array([2.0, 3.0, -5.0, 10.0])  # f3's weight is 0 -> should rank last
    explanation = explain_row(explainer, row_scaled, _row_raw(row_scaled))

    magnitudes = [abs(c.shap_value) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(explanation.contributions) == 4
    assert explanation.contributions[-1].feature == "f3"  # zero-weight feature, smallest effect
    assert all(isinstance(c, Contribution) for c in explanation.contributions)


def test_build_explainer_is_deterministic_given_seed(linear_bundle, linear_background):
    row_scaled = np.array([1.0, -1.0, 0.5, 2.0])
    raw = _row_raw(row_scaled)

    e1 = build_explainer(linear_bundle, linear_background, seed=7, background_size=15)
    e2 = build_explainer(linear_bundle, linear_background, seed=7, background_size=15)
    exp1 = explain_row(e1, row_scaled, raw)
    exp2 = explain_row(e2, row_scaled, raw)

    shap1 = {c.feature: c.shap_value for c in exp1.contributions}
    shap2 = {c.feature: c.shap_value for c in exp2.contributions}
    for feature in shap1:
        assert shap1[feature] == pytest.approx(shap2[feature])


def test_explain_row_works_on_averaging_regressor():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    y = 4 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=80)

    model = AveragingRegressor(
        {
            "ridge": lambda: Ridge(alpha=1.0),
            "gb": lambda: HistGradientBoostingRegressor(max_iter=20, max_depth=3, random_state=0),
        }
    ).fit(X, y)
    bundle = {"model": model}

    explainer = build_explainer(bundle, X, seed=0, background_size=20)
    row = X[0]
    raw = pd.Series({f"f{i}": v for i, v in enumerate(row)})
    explanation = explain_row(explainer, row, raw)

    assert isinstance(explanation, Explanation)
    assert len(explanation.contributions) == 5
    assert explanation.prediction == pytest.approx(float(model.predict(row.reshape(1, -1))[0]))
    # the permutation explainer is an approximation for a non-linear ensemble, but should still
    # be close to additive
    assert abs(explanation.additivity_error) < 1.0


def test_global_importance_ranks_the_informative_feature_highest():
    rng = np.random.default_rng(5)
    n, d = 150, 4
    X = rng.normal(size=(n, d))
    y = 10 * X[:, 0] + rng.normal(scale=0.05, size=n)  # f0 dominates, f1-f3 are noise

    model = AveragingRegressor(
        {
            "gb": lambda: HistGradientBoostingRegressor(max_iter=30, max_depth=3, random_state=0),
            "ridge": lambda: Ridge(alpha=1.0),
        }
    ).fit(X, y)
    bundle = {"model": model}
    feature_names = ("f0", "f1", "f2", "f3")

    importance = global_importance(bundle, X, sample_size=40, seed=0, feature_names=feature_names)

    assert list(importance.columns) == ["feature", "tree_mean_abs_shap", "ensemble_mean_abs_shap"]
    assert len(importance) == d
    assert importance.iloc[0]["feature"] == "f0"
    assert (importance["ensemble_mean_abs_shap"].to_numpy()[:-1] >= importance["ensemble_mean_abs_shap"].to_numpy()[1:]).all()
    assert importance.set_index("feature").loc["f0", "tree_mean_abs_shap"] > 0


def test_global_importance_raises_without_a_gb_member():
    model = AveragingRegressor({"ridge": lambda: Ridge(alpha=1.0)}).fit(np.zeros((5, 2)), np.zeros(5))
    with pytest.raises(ValueError):
        global_importance({"model": model}, np.zeros((5, 2)))


def test_load_bundle_round_trips_a_pickle(tmp_path):
    import pickle

    path = tmp_path / "bundle.pkl"
    path.write_bytes(pickle.dumps({"preprocessor": None, "model": "m", "trained_on": ["a", "b"]}))

    bundle = load_bundle(path)
    assert bundle == {"preprocessor": None, "model": "m", "trained_on": ["a", "b"]}
