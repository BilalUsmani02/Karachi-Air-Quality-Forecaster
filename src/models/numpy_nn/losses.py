"""Regression losses with hand-written gradients.

Both losses average over every element of the prediction array, exactly like PyTorch's
``reduction="mean"``. Our predictions have shape ``(n, 1)``, so "every element" is simply
"every row in the batch": ``N = pred.size = n``.

``forward(pred, y)`` returns the loss as a float and caches the residual;
``backward()`` returns dL/d(pred) with the same shape as ``pred``.
"""

from __future__ import annotations

import numpy as np


def _as_column(y: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Reshape targets to match the predictions (``(n,)`` → ``(n, 1)``)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size != like.size:
        raise ValueError(f"pred has {like.size} values but y has {y.size}")
    return y.reshape(like.shape)


class MSELoss:
    """Mean squared error: ``L = (1/N) Σ (pred − y)²``.

    Gradient: ``dL/dpred = 2 (pred − y) / N`` — each prediction is pushed towards its target
    in proportion to how far off it is.
    """

    def forward(self, pred: np.ndarray, y: np.ndarray) -> float:
        self._residual = pred - _as_column(y, pred)
        return float(np.mean(self._residual**2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._residual / self._residual.size


class HuberLoss:
    """Huber loss: squared error for small mistakes, absolute error for large ones.

    With residual ``r = pred − y``:

    - ``|r| <= delta``: ``0.5 r²``           (gradient ``r``)
    - ``|r| >  delta``: ``delta (|r| − delta/2)`` (gradient ``delta · sign(r)``)

    then averaged over the ``N`` elements (so every gradient is also divided by ``N``).
    Big errors — e.g. a smoke-event day — therefore pull on the weights with a capped force
    instead of a force that grows with the error. Same definition as ``torch.nn.HuberLoss``.
    """

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError(f"huber delta must be > 0, got {delta}")
        self.delta = delta

    def forward(self, pred: np.ndarray, y: np.ndarray) -> float:
        self._residual = pred - _as_column(y, pred)
        abs_r = np.abs(self._residual)
        quadratic = 0.5 * self._residual**2
        linear = self.delta * (abs_r - 0.5 * self.delta)
        return float(np.mean(np.where(abs_r <= self.delta, quadratic, linear)))

    def backward(self) -> np.ndarray:
        clipped = np.clip(self._residual, -self.delta, self.delta)
        return clipped / self._residual.size


def make_loss(name: str, huber_delta: float = 1.0) -> MSELoss | HuberLoss:
    """Build a loss from its name (``"mse"`` or ``"huber"``)."""
    if name == "mse":
        return MSELoss()
    if name == "huber":
        return HuberLoss(huber_delta)
    raise ValueError(f"loss must be 'mse' or 'huber', got {name!r}")
