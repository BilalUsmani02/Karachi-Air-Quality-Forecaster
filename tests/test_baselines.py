import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from src.config import SEED
from src.models.baselines.persistence import PersistenceRegressor
from src.models.baselines.sklearn_models import (
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    ridge,
)


# --- PersistenceRegressor --------------------------------------------------------


def test_persistence_predicts_the_chosen_column():
    X = np.array([[1.0, 99.0], [2.0, 50.0], [3.0, 10.0]])
    model = PersistenceRegressor(feature_index=1)
    model.fit(X, y=np.zeros(3))
    np.testing.assert_array_equal(model.predict(X), [99.0, 50.0, 10.0])


def test_persistence_fit_is_a_noop_and_returns_self():
    X = np.ones((2, 2))
    model = PersistenceRegressor(feature_index=0)
    result = model.fit(X, np.zeros(2))
    assert result is model


def test_persistence_get_set_params_roundtrip():
    model = PersistenceRegressor(feature_index=3)
    params = model.get_params()
    assert params["feature_index"] == 3
    model.set_params(feature_index=7)
    assert model.feature_index == 7


# --- sklearn factories -----------------------------------------------------------


def test_ridge_factory_returns_ridge_with_seed():
    model = ridge()
    assert isinstance(model, Ridge)
    assert model.random_state == SEED


def test_ridge_factory_overrides():
    model = ridge(alpha=2.0)
    assert model.alpha == 2.0


def test_random_forest_regressor_factory():
    model = random_forest_regressor()
    assert isinstance(model, RandomForestRegressor)
    assert model.random_state == SEED


def test_logistic_regression_factory_balanced_and_converges():
    model = logistic_regression()
    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"
    assert model.random_state == SEED
    assert model.max_iter >= 1000


def test_random_forest_classifier_factory_balanced():
    model = random_forest_classifier()
    assert isinstance(model, RandomForestClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == SEED


def test_factories_return_fresh_unfit_instances():
    a, b = ridge(), ridge()
    assert a is not b


def test_logistic_regression_fits_without_error_on_tiny_imbalanced_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.array([0] * 50 + [1] * 8 + [2] * 2)  # imbalanced, all 5 classes not required here
    model = logistic_regression()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (60,)
