"""Unit tests and gradient checks for the NumPy network's building blocks."""

import numpy as np
import pytest

from src.models.numpy_nn import gradcheck as gc
from src.models.numpy_nn.init import he_normal, init_mlp_weights, xavier_uniform
from src.models.numpy_nn.layers import Dense, Dropout, LeakyReLU, Param, ReLU, Tanh
from src.models.numpy_nn.losses import HuberLoss, MSELoss, make_loss
from src.models.numpy_nn.network import Sequential, build_mlp
from src.models.numpy_nn.optimizers import SGD, Adam, l2_penalty, make_optimizer

TOL = 1e-6  # gradient-check threshold on the relative error (float64)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- gradient checks --------------------------------------------------------------------


def test_gradcheck_dense(rng):
    layer = Dense(rng.normal(size=(4, 3)), rng.normal(size=3))
    errors = gc.check_layer(layer, rng.normal(size=(5, 4)))
    assert set(errors) == {"input", "dense.W", "dense.b"}
    assert max(errors.values()) < TOL


@pytest.mark.parametrize("layer", [ReLU(), Tanh(), LeakyReLU(0.1)])
def test_gradcheck_activations(layer, rng):
    errors = gc.check_layer(layer, rng.normal(size=(6, 5)))
    assert errors["input"] < TOL


@pytest.mark.parametrize("loss", [MSELoss(), HuberLoss(0.5), HuberLoss(2.0)])
def test_gradcheck_losses(loss, rng):
    pred = rng.normal(scale=2.0, size=(10, 1))  # mix of residuals inside and outside delta
    assert gc.check_loss(loss, pred, rng.normal(size=10)) < TOL


@pytest.mark.parametrize("activation", ["relu", "tanh", "leaky_relu"])
@pytest.mark.parametrize("loss", [MSELoss(), HuberLoss(1.0)])
def test_gradcheck_full_network_with_l2(activation, loss, rng):
    weights = init_mlp_weights([4, 6, 5, 1], activation, rng)
    weights = [w + rng.normal(scale=0.1, size=w.shape) for w in weights]  # non-zero biases
    network = build_mlp(weights, activation)
    X, y = rng.normal(size=(8, 4)), rng.normal(size=8)
    errors = gc.check_network(network, loss, X, y, l2=0.05)
    assert len(errors) == 6
    assert max(errors.values()) < TOL


def test_gradcheck_detects_a_wrong_gradient(rng):
    class BrokenTanh(Tanh):
        def backward(self, grad_out):
            return grad_out * (1.0 - self._out)  # missing the square

    assert gc.check_layer(BrokenTanh(), rng.normal(size=(4, 3)))["input"] > 1e-2


def test_dropout_gradient_with_fixed_mask(rng):
    X, R = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))

    def f():  # a fresh layer with the same seed draws the same mask every time
        return float(np.sum(Dropout(0.4, np.random.default_rng(7)).forward(X, training=True) * R))

    layer = Dropout(0.4, np.random.default_rng(7))
    layer.forward(X, training=True)
    assert gc.relative_error(layer.backward(R), gc.numerical_gradient(f, X)) < TOL


def test_relative_error_of_two_zeros_is_zero():
    assert gc.relative_error(np.zeros(3), np.zeros(3)) == 0.0


# --- layers -----------------------------------------------------------------------------


def test_dense_forward_matches_formula_and_copies_inputs(rng):
    W, b, X = rng.normal(size=(3, 2)), rng.normal(size=2), rng.normal(size=(4, 3))
    layer = Dense(W, b)
    np.testing.assert_allclose(layer.forward(X), X @ W + b)
    layer.W.value += 1.0
    assert not np.allclose(W, layer.W.value)  # caller's array untouched


def test_dense_rejects_bad_shapes_and_backward_before_forward(rng):
    with pytest.raises(ValueError):
        Dense(rng.normal(size=(3, 2)), np.zeros(3))
    with pytest.raises(RuntimeError):
        Dense(np.ones((2, 2)), np.zeros(2)).backward(np.ones((1, 2)))


def test_relu_tanh_values():
    X = np.array([[-2.0, 0.0, 3.0]])
    np.testing.assert_array_equal(ReLU().forward(X), [[0.0, 0.0, 3.0]])
    np.testing.assert_allclose(LeakyReLU(0.1).forward(X), [[-0.2, 0.0, 3.0]])
    np.testing.assert_allclose(Tanh().forward(X), np.tanh(X))
    relu = ReLU()
    relu.forward(X)
    np.testing.assert_array_equal(relu.backward(np.ones_like(X)), [[0.0, 0.0, 1.0]])  # 0 at 0


def test_dropout_is_identity_at_prediction_and_keeps_mean_in_training():
    X = np.ones((2000, 50))
    layer = Dropout(0.3, np.random.default_rng(1))
    np.testing.assert_array_equal(layer.forward(X, training=False), X)
    out = layer.forward(X, training=True)
    assert set(np.unique(out)) == {0.0, 1.0 / 0.7}
    assert abs(out.mean() - 1.0) < 0.01  # inverted dropout keeps the expected value
    assert abs((out == 0).mean() - 0.3) < 0.01


def test_dropout_rejects_bad_p():
    with pytest.raises(ValueError):
        Dropout(1.0, np.random.default_rng(0))


# --- losses -----------------------------------------------------------------------------


def test_mse_value_and_gradient():
    pred, y = np.array([[1.0], [3.0]]), np.array([0.0, 1.0])
    loss = MSELoss()
    assert loss.forward(pred, y) == pytest.approx((1 + 4) / 2)
    np.testing.assert_allclose(loss.backward(), [[1.0], [2.0]])  # 2 r / N


def test_huber_value_both_regimes():
    pred, y = np.array([[0.5], [3.0]]), np.zeros(2)
    # 0.5 * 0.25 = 0.125 (quadratic) and 1 * (3 - 0.5) = 2.5 (linear), mean = 1.3125
    assert HuberLoss(1.0).forward(pred, y) == pytest.approx(1.3125)


def test_loss_rejects_size_mismatch_and_bad_names():
    with pytest.raises(ValueError):
        MSELoss().forward(np.zeros((3, 1)), np.zeros(4))
    with pytest.raises(ValueError):
        HuberLoss(0.0)
    with pytest.raises(ValueError):
        make_loss("mae")
    assert isinstance(make_loss("huber", 2.0), HuberLoss)


# --- init -------------------------------------------------------------------------------


def test_init_shapes_scales_and_seed():
    rng = np.random.default_rng(3)
    assert np.std(he_normal(1000, 200, rng)) == pytest.approx(np.sqrt(2 / 1000), rel=0.02)
    limit = np.sqrt(6 / 1200)
    assert np.max(np.abs(xavier_uniform(1000, 200, rng))) <= limit
    a = init_mlp_weights([5, 4, 1], "relu", np.random.default_rng(9))
    b = init_mlp_weights([5, 4, 1], "relu", np.random.default_rng(9))
    assert [w.shape for w in a] == [(5, 4), (4,), (4, 1), (1,)]
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not np.any(a[1]) and not np.any(a[3])  # zero biases


# --- optimisers -------------------------------------------------------------------------


def _param(value, grad, decay):
    p = Param("p", np.array(value, dtype=float), decay=decay)
    p.grad = np.array(grad, dtype=float)
    return p


def test_sgd_step_with_l2_on_weights_only():
    w, b = _param([1.0, -2.0], [0.5, 0.5], True), _param([1.0], [0.5], False)
    SGD([w, b], lr=0.1, l2=0.1).step()
    np.testing.assert_allclose(w.value, [1.0 - 0.1 * (0.5 + 0.1), -2.0 - 0.1 * (0.5 - 0.2)])
    np.testing.assert_allclose(b.value, [1.0 - 0.05])
    assert l2_penalty([w, b], 0.1) == pytest.approx(0.05 * np.sum(w.value**2))


def test_adam_first_step_moves_by_learning_rate_times_sign():
    p = _param([1.0, 1.0, 1.0], [3.0, -0.01, 50.0], True)
    Adam([p], lr=0.01).step()
    # bias-corrected m̂ / sqrt(v̂) = g / |g| on step 1, whatever the gradient's size
    np.testing.assert_allclose(p.value, [0.99, 1.01, 0.99], rtol=1e-6)


def test_make_optimizer():
    assert isinstance(make_optimizer("adam", [], 0.1, 0.0), Adam)
    assert isinstance(make_optimizer("sgd", [], 0.1, 0.0), SGD)
    with pytest.raises(ValueError):
        make_optimizer("rmsprop", [], 0.1, 0.0)


# --- Sequential / build_mlp -------------------------------------------------------------


def test_sequential_weights_round_trip_and_validation(rng):
    weights = init_mlp_weights([3, 4, 1], "tanh", rng)
    net = build_mlp(weights, "tanh")
    assert len(net.params_and_grads()) == 4
    new = [w + 1.0 for w in weights]
    net.set_weights(new)
    assert all(np.array_equal(a, b) for a, b in zip(net.get_weights(), new))
    with pytest.raises(ValueError):
        net.set_weights(new[:2])
    with pytest.raises(ValueError):
        net.set_weights([np.zeros((9, 9))] + new[1:])


def test_build_mlp_layer_order_and_errors(rng):
    weights = init_mlp_weights([3, 4, 5, 1], "relu", rng)
    net = build_mlp(weights, "relu", dropout=0.2, rng=rng)
    kinds = [type(layer).__name__ for layer in net.layers]
    assert kinds == ["Dense", "ReLU", "Dropout", "Dense", "ReLU", "Dropout", "Dense"]
    with pytest.raises(ValueError):
        build_mlp(weights[:3])
    with pytest.raises(ValueError):
        build_mlp(weights, dropout=0.2)  # no rng
    assert isinstance(Sequential([]).forward(np.ones((1, 1))), np.ndarray)
