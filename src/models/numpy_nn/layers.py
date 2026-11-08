"""Network building blocks written with NumPy only: a dense layer, activations and dropout.

Every layer has the same three-method shape, so a container can chain them:

- ``forward(X, training)`` computes the output and **caches** whatever ``backward`` will need.
- ``backward(grad_out)`` receives dL/d(output) and returns dL/d(input) — one step of the chain
  rule. Layers with weights also store dL/d(weight) on their ``Param`` objects.
- ``params()`` lists the trainable ``Param`` objects (empty for activations).

Shapes: ``n`` = rows in the mini-batch, ``d_in`` / ``d_out`` = features in / out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Param:
    """One trainable array plus its gradient.

    ``decay`` says whether L2 regularisation applies to it: True for weight matrices,
    False for biases (shrinking a bias towards zero does not make the model simpler).
    """

    name: str
    value: np.ndarray
    decay: bool
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.value)


class Layer:
    """Base class: no parameters, identity in both directions."""

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[Param]:
        return []


class Dense(Layer):
    """Fully connected layer: ``Z = X @ W + b``.

    Shapes: ``X (n, d_in)``, ``W (d_in, d_out)``, ``b (d_out,)``, ``Z (n, d_out)``.

    Backward, given ``G = dL/dZ`` of shape ``(n, d_out)`` (chain rule on ``Z = XW + b``):

    - ``dL/dW = X.T @ G``  → ``(d_in, d_out)``: each weight's gradient sums, over the batch,
      "input that fed it" × "error that came back".
    - ``dL/db = G.sum(axis=0)`` → ``(d_out,)``: the bias is added to every row, so its
      gradient is the sum of the row errors.
    - ``dL/dX = G @ W.T`` → ``(n, d_in)``: the error sent further back to the previous layer.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray, name: str = "dense") -> None:
        W = np.array(W, dtype=np.float64)  # copy: never share memory with the caller
        b = np.array(b, dtype=np.float64)
        if W.ndim != 2 or b.shape != (W.shape[1],):
            raise ValueError(f"Dense needs W (d_in, d_out) and b (d_out,), got {W.shape} and {b.shape}")
        self.W = Param(f"{name}.W", W, decay=True)
        self.b = Param(f"{name}.b", b, decay=False)
        self._X: np.ndarray | None = None

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self._X = X
        return X @ self.W.value + self.b.value

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise RuntimeError("backward() called before forward()")
        self.W.grad = self._X.T @ grad_out
        self.b.grad = grad_out.sum(axis=0)
        return grad_out @ self.W.value.T

    def params(self) -> list[Param]:
        return [self.W, self.b]


class ReLU(Layer):
    """``A = max(0, Z)``. Backward: the gradient passes where ``Z > 0`` and is blocked elsewhere.

    The derivative at exactly 0 is taken as 0, the same convention PyTorch uses.
    """

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self._mask = X > 0
        return np.where(self._mask, X, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class LeakyReLU(Layer):
    """``A = Z`` if ``Z > 0`` else ``slope * Z``. Keeps a small gradient for negative inputs,
    so a unit can never go completely "dead". ``slope=0.01`` matches PyTorch's default."""

    def __init__(self, slope: float = 0.01) -> None:
        self.slope = slope

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self._mask = X > 0
        return np.where(self._mask, X, self.slope * X)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * np.where(self._mask, 1.0, self.slope)


class Tanh(Layer):
    """``A = tanh(Z)``. Since d tanh(z)/dz = 1 − tanh(z)², backward reuses the cached output."""

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self._out = np.tanh(X)
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1.0 - self._out**2)


class Dropout(Layer):
    """Inverted dropout: in training, zero each unit with probability ``p`` and scale the
    survivors by ``1 / (1 − p)`` so the expected activation is unchanged. At prediction time
    it does nothing (no rescaling needed — that is the point of the "inverted" form).

    Backward multiplies by the same mask and scale, because ``A = X * mask / (1 − p)``.
    The mask comes from a seeded ``np.random.Generator``, so training is reproducible.
    """

    def __init__(self, p: float, rng: np.random.Generator) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
        self.p = p
        self.rng = rng
        self._scale: np.ndarray | float = 1.0

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.p == 0.0:
            self._scale = 1.0
            return X
        keep = self.rng.random(X.shape) >= self.p
        self._scale = keep / (1.0 - self.p)
        return X * self._scale

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._scale


ACTIVATIONS: dict[str, type[Layer]] = {"relu": ReLU, "tanh": Tanh, "leaky_relu": LeakyReLU}
