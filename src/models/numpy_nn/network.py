"""The network container and the user-facing ``MLPRegressorNP``.

``Sequential`` chains layers. Forward runs them first to last; backward runs them **last to
first**, each one turning dL/d(its output) into dL/d(its input) — that relay *is*
backpropagation (the chain rule applied one layer at a time).

For the default 2-hidden-layer ReLU network with input ``X (n, d)``:

    Z1 = X  W1 + b1   (n, 64)     A1 = ReLU(Z1)
    Z2 = A1 W2 + b2   (n, 32)     A2 = ReLU(Z2)
    P  = A2 W3 + b3   (n, 1)      L  = mean((P − y)²)

and backward walks the same list in reverse: dL/dP → (W3, b3, dL/dA2) → ReLU mask →
(W2, b2, dL/dA1) → ReLU mask → (W1, b1).
"""

from __future__ import annotations

import numpy as np

from src.models.numpy_nn.estimator_base import BaseMLPRegressor
from src.models.numpy_nn.layers import ACTIVATIONS, Dense, Dropout, Layer, Param
from src.models.numpy_nn.losses import make_loss
from src.models.numpy_nn.optimizers import make_optimizer


class Sequential:
    """A list of layers run in order."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = list(layers)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[Param]:
        return [p for layer in self.layers for p in layer.params()]

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(value, grad)`` pairs for every trainable array, in layer order."""
        return [(p.value, p.grad) for p in self.params()]

    def get_weights(self) -> list[np.ndarray]:
        """Copies of ``[W1, b1, W2, b2, ...]``."""
        return [p.value.copy() for p in self.params()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        params = self.params()
        if len(weights) != len(params):
            raise ValueError(f"expected {len(params)} arrays, got {len(weights)}")
        for p, w in zip(params, weights):
            if np.shape(w) != p.value.shape:
                raise ValueError(f"{p.name}: expected shape {p.value.shape}, got {np.shape(w)}")
            p.value = np.array(w, dtype=np.float64)


def build_mlp(
    weights: list[np.ndarray],
    activation: str = "relu",
    dropout: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Sequential:
    """Build ``Dense → activation [→ Dropout]`` blocks plus a linear output ``Dense``
    from given weights ``[W1, b1, ..., W_out, b_out]`` (NumPy layout, ``W`` is ``(in, out)``).
    """
    if len(weights) < 2 or len(weights) % 2:
        raise ValueError("weights must be [W1, b1, W2, b2, ...]")
    if dropout > 0 and rng is None:
        raise ValueError("dropout needs a seeded rng")
    n_dense = len(weights) // 2
    layers: list[Layer] = []
    for i in range(n_dense):
        layers.append(Dense(weights[2 * i], weights[2 * i + 1], name=f"dense{i + 1}"))
        if i < n_dense - 1:  # hidden layer: add the non-linearity (and dropout)
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(Dropout(dropout, rng))
    return Sequential(layers)


class MLPRegressorNP(BaseMLPRegressor):
    """Multi-layer perceptron regressor written from scratch in NumPy.

    sklearn-style: ``MLPRegressorNP(**params).fit(X, y).predict(X)``.

    Parameters (shared with ``MLPRegressorTorch``): ``hidden_sizes``, ``activation``
    (``"relu" | "tanh" | "leaky_relu"``), ``loss`` (``"mse" | "huber"``), ``huber_delta``,
    ``optimizer`` (``"adam" | "sgd"``), ``learning_rate``, ``batch_size``, ``max_epochs``,
    ``l2`` (weight decay on weights only), ``dropout``, ``early_stopping``,
    ``validation_fraction`` (last rows), ``patience``, ``scale_y``, ``random_state``.

    Fitted attributes: ``history_``, ``n_epochs_``, ``best_epoch_``, ``initial_weights_``,
    ``n_features_in_``, ``y_mean_``, ``y_std_``. See ``estimator_base`` for the training-loop
    conventions.
    """

    def _build(self, weights: list[np.ndarray], dropout_rng: np.random.Generator) -> None:
        self.network_ = build_mlp(weights, self.activation, self.dropout, dropout_rng)
        self.loss_fn_ = make_loss(self.loss, self.huber_delta)
        self.optimizer_ = make_optimizer(
            self.optimizer, self.network_.params(), self.learning_rate, self.l2
        )

    def _train_step(self, Xb: np.ndarray, yb: np.ndarray) -> float:
        """One mini-batch: forward → loss → backward → optimiser update."""
        pred = self.network_.forward(Xb, training=True)
        loss = self.loss_fn_.forward(pred, yb)
        self.network_.backward(self.loss_fn_.backward())
        self.optimizer_.step()
        return loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.loss_fn_.forward(self.network_.forward(X, training=False), y)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return self.network_.forward(X, training=False)[:, 0]

    def get_weights(self) -> list[np.ndarray]:
        return self.network_.get_weights()

    def _set_weights(self, weights: list[np.ndarray]) -> None:
        self.network_.set_weights(weights)
