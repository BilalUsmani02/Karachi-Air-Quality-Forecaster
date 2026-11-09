"""Tests for `src.ui.explain_ui` — the pure wrappers the app caches around real SHAP calls.

Monkeypatches `explain_ui.FEATURE_COLUMNS` down to 3 real feature names (so this stays fast,
the same approach `tests/test_explain_shap.py` uses for `src.explain.shap_explain`) but keeps
them *real* registered names (`src.explain.reasons.FEATURE_PHRASES`), so `shap_reasons_for_row`
exercises the actual phrase lookup instead of a stand-in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.ui.explain_ui as explain_ui
from src.ui.explain_ui import build_sensor_explainer, feature_medians, shap_reasons_for_row

FEATURES = ("pm25_lag_1h", "pm25_lag_2h", "pm25_mean_72h")


class _IdentityPreprocessor:
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[list(FEATURES)].to_numpy(dtype=np.float64)


class _LinearModel:
    """y = 5*pm25_lag_1h - 2*pm25_lag_2h — pm25_mean_72h's weight is 0, should rank last."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return 5.0 * X[:, 0] - 2.0 * X[:, 1]


@pytest.fixture(autouse=True)
def _small_feature_set(monkeypatch):
    monkeypatch.setattr(explain_ui, "FEATURE_COLUMNS", FEATURES)


@pytest.fixture
def bundle():
    return {"preprocessor": _IdentityPreprocessor(), "model": _LinearModel()}


@pytest.fixture
def background_raw():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(30, 3)), columns=list(FEATURES))


def test_feature_medians_matches_pandas_median(background_raw):
    medians = feature_medians(background_raw)
    assert medians["pm25_lag_1h"] == pytest.approx(background_raw["pm25_lag_1h"].median())
    assert set(medians) == set(FEATURES)


def test_build_sensor_explainer_scales_the_background_first(bundle, background_raw):
    explainer = build_sensor_explainer(bundle, background_raw, seed=0, background_size=10)
    assert explainer is not None


def test_shap_reasons_for_row_names_the_dominant_feature_first(bundle, background_raw):
    explainer = build_sensor_explainer(bundle, background_raw, seed=0, background_size=10)
    medians = feature_medians(background_raw)

    row = pd.Series({"pm25_lag_1h": 10.0, "pm25_lag_2h": 10.0, "pm25_mean_72h": 10.0})
    reasons = shap_reasons_for_row(explainer, bundle, row, medians, k=2)

    assert len(reasons) == 2
    assert all(isinstance(r, str) and r for r in reasons)
    # pm25_lag_1h's weight (5) dominates pm25_lag_2h's (-2) in magnitude -> named first.
    assert "hour ago" in reasons[0]
