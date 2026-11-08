"""The PyTorch twin, and the proof that the NumPy network trains identically to it."""

import numpy as np
import pytest
import torch

from src.models.numpy_nn.network import MLPRegressorNP
from src.models.torch_nn.compare import compare_numpy_torch
from src.models.torch_nn.model import MLPRegressorTorch

RTOL, ATOL = 1e-6, 1e-10


def make_data(n=160, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 40 + 10 * np.sin(X[:, 0]) + 5 * X[:, 1] ** 2 + rng.normal(size=n)
    return X, y


def fit_pair(X, y, **params):
    numpy_model = MLPRegressorNP(**params).fit(X, y)
    torch_model = MLPRegressorTorch(**params).fit(X, y, initial_weights=numpy_model.initial_weights_)
    return numpy_model, torch_model


# --- equivalence ------------------------------------------------------------------------


@pytest.mark.parametrize("optimizer,lr", [("sgd", 0.05), ("adam", 0.01)])
@pytest.mark.parametrize("loss", ["mse", "huber"])
@pytest.mark.parametrize("l2", [0.0, 0.01])
def test_numpy_and_torch_train_identically(optimizer, lr, loss, l2):
    X, y = make_data()
    # 160 rows, 24 held out, batch 34 -> 4 steps per epoch x 5 epochs = 20 steps
    params = dict(
        hidden_sizes=(16, 8), optimizer=optimizer, learning_rate=lr, loss=loss,
        huber_delta=0.5, l2=l2, batch_size=34, max_epochs=5, patience=100,
    )
    np_model, torch_model = fit_pair(X, y, **params)

    assert len(np_model.history_["step_loss"]) == 20
    np.testing.assert_allclose(np_model.history_["step_loss"], torch_model.history_["step_loss"], rtol=RTOL)
    np.testing.assert_allclose(np_model.history_["val_loss"], torch_model.history_["val_loss"], rtol=RTOL)
    for a, b in zip(np_model.get_weights(), torch_model.get_weights()):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np_model.predict(X), torch_model.predict(X), rtol=RTOL)
    assert not np.allclose(np_model.get_weights()[0], np_model.initial_weights_[0])  # it trained


@pytest.mark.parametrize("activation", ["tanh", "leaky_relu"])
def test_equivalence_other_activations(activation):
    X, y = make_data()
    np_model, torch_model = fit_pair(X, y, hidden_sizes=(8,), activation=activation, max_epochs=3)
    np.testing.assert_allclose(np_model.history_["step_loss"], torch_model.history_["step_loss"], rtol=RTOL)
    np.testing.assert_allclose(np_model.predict(X), torch_model.predict(X), rtol=RTOL)


def test_same_seed_gives_same_start_without_passing_weights():
    X, y = make_data()
    np_model = MLPRegressorNP(hidden_sizes=(8,), max_epochs=2).fit(X, y)
    torch_model = MLPRegressorTorch(hidden_sizes=(8,), max_epochs=2).fit(X, y)
    for a, b in zip(np_model.initial_weights_, torch_model.initial_weights_):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np_model.predict(X), torch_model.predict(X), rtol=RTOL)


# --- the twin on its own ----------------------------------------------------------------


def test_weights_are_loaded_transposed_into_torch_layout():
    X, y = make_data(d=3)
    model = MLPRegressorTorch(hidden_sizes=(4,), max_epochs=1, learning_rate=1e-12).fit(X, y)
    first = model.linears_[0]
    assert tuple(first.weight.shape) == (4, 3)  # torch: (out, in)
    assert first.weight.dtype == torch.float64
    np.testing.assert_allclose(first.weight.detach().numpy().T, model.initial_weights_[0], atol=1e-9)


def test_torch_early_stopping_and_dropout_are_deterministic():
    X, y = make_data()
    kw = dict(hidden_sizes=(16,), dropout=0.3, max_epochs=30, patience=3, random_state=4)
    a = MLPRegressorTorch(**kw).fit(X, y)
    b = MLPRegressorTorch(**kw).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert a.n_epochs_ <= 30 and len(a.history_["val_loss"]) == a.n_epochs_


def test_torch_learns_and_restores_deterministic_mode():
    before = torch.are_deterministic_algorithms_enabled()
    X, y = make_data(n=400)
    model = MLPRegressorTorch(max_epochs=60, learning_rate=0.01).fit(X, y)
    assert np.mean((model.predict(X) - y) ** 2) < 0.2 * np.var(y)
    assert torch.are_deterministic_algorithms_enabled() == before
    assert model.get_params() == MLPRegressorNP(max_epochs=60, learning_rate=0.01).get_params()


# --- benchmark helper -------------------------------------------------------------------


def test_compare_numpy_torch_reports_matching_results():
    X, y = make_data()
    result = compare_numpy_torch(X, y, hidden_sizes=(8,), max_epochs=4)
    assert result["numpy_epochs"] == result["torch_epochs"] == 4
    assert result["numpy_fit_seconds"] > 0 and result["torch_fit_seconds"] > 0
    assert result["numpy_train_loss"] == pytest.approx(result["torch_train_loss"], rel=RTOL)
    assert result["numpy_best_val_loss"] == pytest.approx(result["torch_best_val_loss"], rel=RTOL)
    assert result["max_abs_prediction_diff"] < 1e-8
    assert result["max_abs_weight_diff"] < 1e-8


def test_compare_without_validation():
    X, y = make_data(n=60)
    result = compare_numpy_torch(X, y, hidden_sizes=(4,), max_epochs=2, early_stopping=False)
    assert result["numpy_best_val_loss"] is None and result["torch_best_val_loss"] is None
