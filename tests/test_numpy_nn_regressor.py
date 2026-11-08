"""Behaviour tests for ``MLPRegressorNP`` on synthetic data."""

import numpy as np
import pytest
from sklearn.base import clone

from src.models.numpy_nn.estimator_base import BaseMLPRegressor, epoch_batches
from src.models.numpy_nn.network import MLPRegressorNP


def make_data(n=600, d=6, seed=0):
    """y = 40 + 10 sin(x1) + 5 x2² + noise — non-linear, PM2.5-like scale."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 40 + 10 * np.sin(X[:, 0]) + 5 * X[:, 1] ** 2 + rng.normal(scale=1.0, size=n)
    return X, y


def fit_predict(model, X, y):
    """Plain function using the model like any sklearn estimator."""
    return model.fit(X, y).predict(X)


def test_smoke_sklearn_style_fit_predict():
    X, y = make_data(n=200)
    pred = fit_predict(MLPRegressorNP(max_epochs=5), X, y)
    assert pred.shape == (200,) and pred.dtype == np.float64


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"activation": "tanh", "loss": "huber"},
        {"optimizer": "sgd", "learning_rate": 0.05, "dropout": 0.1},
    ],
)
def test_learns_nonlinear_function_far_better_than_the_mean(params):
    X, y = make_data()
    X_test, y_test = make_data(n=300, seed=1)
    model = MLPRegressorNP(max_epochs=150, patience=30, **params).fit(X, y)
    mse_model = np.mean((model.predict(X_test) - y_test) ** 2)
    mse_mean = np.mean((y.mean() - y_test) ** 2)
    assert mse_model < 0.15 * mse_mean


def test_early_stopping_triggers_and_restores_best_weights():
    X, y = make_data(n=150)
    model = MLPRegressorNP(
        hidden_sizes=(128, 128), learning_rate=0.01, max_epochs=500, patience=5, l2=0.0
    ).fit(X, y)
    val = model.history_["val_loss"]
    assert model.n_epochs_ < 500
    assert model.n_epochs_ == model.best_epoch_ + 5
    assert np.argmin(val) + 1 == model.best_epoch_
    n_train = 150 - round(150 * 0.15)
    y_val = (y[n_train:] - model.y_mean_) / model.y_std_
    assert model._loss(X[n_train:], y_val) == pytest.approx(min(val), rel=1e-12)
    assert len(model.history_["train_loss"]) == model.n_epochs_


def test_validation_rows_are_the_last_rows():
    X, y = make_data(n=200)
    base = MLPRegressorNP(max_epochs=3, validation_fraction=0.25).fit(X, y)
    assert base.y_mean_ == pytest.approx(y[:150].mean())

    X2, y2 = X.copy(), y.copy()
    X2[150:] += 5.0  # change only the last 50 rows (the validation set)
    y2[150:] *= 3.0
    other = MLPRegressorNP(max_epochs=3, validation_fraction=0.25).fit(X2, y2)
    assert other.history_["step_loss"] == base.history_["step_loss"]  # training untouched
    assert other.history_["val_loss"] != base.history_["val_loss"]


def test_without_early_stopping_runs_all_epochs_on_all_rows():
    X, y = make_data(n=100)
    model = MLPRegressorNP(early_stopping=False, max_epochs=7, batch_size=30).fit(X, y)
    assert model.n_epochs_ == model.best_epoch_ == 7
    assert model.history_["val_loss"] == []
    assert len(model.history_["step_loss"]) == 7 * 4  # ceil(100 / 30) batches per epoch
    assert model.y_mean_ == pytest.approx(y.mean())


def test_same_seed_same_predictions_different_seed_different():
    X, y = make_data(n=200)
    kw = {"max_epochs": 10, "dropout": 0.2}
    a = MLPRegressorNP(random_state=1, **kw).fit(X, y).predict(X)
    b = MLPRegressorNP(random_state=1, **kw).fit(X, y).predict(X)
    c = MLPRegressorNP(random_state=2, **kw).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_fit_does_not_mutate_inputs():
    X, y = make_data(n=100)
    X0, y0 = X.copy(), y.copy()
    MLPRegressorNP(max_epochs=2).fit(X, y)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)


def test_initial_weights_exported_and_reusable():
    X, y = make_data(n=120)
    a = MLPRegressorNP(max_epochs=3, random_state=5).fit(X, y)
    b = MLPRegressorNP(max_epochs=3, random_state=99).fit(X, y, initial_weights=a.initial_weights_)
    assert [w.shape for w in a.initial_weights_] == [(6, 64), (64,), (64, 32), (32,), (32, 1), (1,)]
    # same start; batches differ by seed, so only the start is shared
    assert all(np.array_equal(p, q) for p, q in zip(a.initial_weights_, b.initial_weights_))
    with pytest.raises(ValueError, match="do not match"):
        MLPRegressorNP(hidden_sizes=(8,)).fit(X, y, initial_weights=a.initial_weights_)


def test_get_set_params_round_trip_and_clone():
    model = MLPRegressorNP(hidden_sizes=(16,), l2=0.01)
    params = model.get_params()
    assert params["hidden_sizes"] == (16,) and params["l2"] == 0.01
    assert "random_state" in params and len(params) == 15
    copy = MLPRegressorNP(**params)
    assert copy.get_params() == params
    assert model.set_params(learning_rate=0.5, loss="huber") is model
    assert model.learning_rate == 0.5 and model.loss == "huber"
    cloned = clone(model)
    assert cloned is not model and cloned.get_params() == model.get_params()
    assert "MLPRegressorNP(hidden_sizes=(16,)" in repr(model)
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(n_layers=3)


def test_y_as_column_and_constant_target():
    X, y = make_data(n=80)
    model = MLPRegressorNP(max_epochs=2).fit(X, y.reshape(-1, 1))
    assert model.predict(X).shape == (80,)
    flat = MLPRegressorNP(max_epochs=2).fit(X, np.full(80, 7.0))
    assert flat.y_std_ == 1.0


def test_unscaled_target_option():
    X, y = make_data(n=80)
    model = MLPRegressorNP(max_epochs=2, scale_y=False).fit(X, y)
    assert (model.y_mean_, model.y_std_) == (0.0, 1.0)


@pytest.mark.parametrize(
    "X,y",
    [
        (np.ones(10), np.ones(10)),  # 1-D X
        (np.ones((10, 3)), np.ones(9)),  # length mismatch
        (np.full((10, 3), np.nan), np.ones(10)),  # NaN in X
        (np.ones((10, 3)), np.r_[np.ones(9), np.inf]),  # inf in y
        (np.ones((0, 3)), np.ones(0)),  # empty
        (np.ones((1, 3)), np.ones(1)),  # too few rows for a validation split
    ],
)
def test_input_validation(X, y):
    with pytest.raises(ValueError):
        MLPRegressorNP(max_epochs=1).fit(X, y)


@pytest.mark.parametrize(
    "bad",
    [
        {"activation": "sigmoid"},
        {"loss": "mae"},
        {"optimizer": "rmsprop"},
        {"hidden_sizes": (8, 0)},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"l2": -1.0},
        {"dropout": 1.0},
        {"validation_fraction": 1.0},
    ],
)
def test_hyperparameter_validation(bad):
    X, y = make_data(n=40)
    with pytest.raises(ValueError):
        MLPRegressorNP(max_epochs=1, **bad).fit(X, y)


def test_predict_checks():
    X, y = make_data(n=60)
    with pytest.raises(RuntimeError, match="not fitted"):
        MLPRegressorNP().predict(X)
    model = MLPRegressorNP(max_epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(X[:, :3])


def test_base_class_hooks_must_be_overridden():
    X, y = make_data(n=40)
    with pytest.raises(NotImplementedError):
        BaseMLPRegressor(max_epochs=1).fit(X, y)


def test_epoch_batches_cover_every_row_once():
    batches = epoch_batches(10, 4, np.random.default_rng(0))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches)) == list(range(10))
