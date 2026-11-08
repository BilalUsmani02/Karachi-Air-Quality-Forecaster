"""``MLPRegressorTorch`` — the PyTorch twin of the from-scratch NumPy network.

Same constructor, same training loop (``BaseMLPRegressor``), same architecture
(``Linear → activation [→ Dropout]`` blocks + a linear output layer), but every piece of maths
is PyTorch's own: ``nn.Linear``, ``nn.ReLU`` / ``nn.Tanh`` / ``nn.LeakyReLU``,
``nn.MSELoss`` / ``nn.HuberLoss`` (``reduction="mean"``), autograd for the gradients and
``torch.optim.SGD`` / ``torch.optim.Adam`` for the updates. If the two models produce the same
losses and weights step after step, the hand-written NumPy maths must be right.

Matching details:

- Float64 on the CPU throughout.
- Weights are exchanged in NumPy layout. ``nn.Linear`` stores ``weight`` as ``(out, in)``
  and computes ``X @ weight.T + bias``, so a NumPy ``W (in, out)`` is loaded as ``W.T``.
- Initial weights come from the same seeded NumPy initialiser as the NumPy model (or are
  passed in via ``fit(..., initial_weights=np_model.initial_weights_)``); mini-batch order
  comes from the same seeded NumPy shuffle. So with equal ``random_state`` the two see
  identical starting points and identical batches.
- L2: parameter groups give the weight matrices ``weight_decay=l2`` and the biases
  ``weight_decay=0``. PyTorch's SGD and Adam add ``weight_decay * W`` to the gradient — the
  same formulation as the NumPy optimisers.
- Dropout uses PyTorch's own random generator (seeded from ``random_state``), so with
  ``dropout > 0`` the two models are each deterministic but no longer step-for-step equal.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn

from src.models.numpy_nn.estimator_base import BaseMLPRegressor

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


@contextlib.contextmanager
def _deterministic() -> Iterator[None]:
    """Turn on PyTorch's deterministic-algorithm mode, then restore the caller's setting."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64)


class MLPRegressorTorch(BaseMLPRegressor):
    """PyTorch MLP regressor with exactly the parameters and behaviour of ``MLPRegressorNP``."""

    def _build(self, weights: list[np.ndarray], dropout_rng: np.random.Generator) -> None:
        # Dropout masks come from torch's generator; seed it from our dropout stream.
        torch.manual_seed(int(dropout_rng.integers(2**31 - 1)))
        n_dense = len(weights) // 2
        layers: list[nn.Module] = []
        for i in range(n_dense):
            W, b = weights[2 * i], weights[2 * i + 1]
            linear = nn.Linear(W.shape[0], W.shape[1], dtype=torch.float64)
            with torch.no_grad():
                linear.weight.copy_(_tensor(W.T))  # NumPy (in, out) -> torch (out, in)
                linear.bias.copy_(_tensor(b))
            layers.append(linear)
            if i < n_dense - 1:
                layers.append(_ACTIVATIONS[self.activation]())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        self.network_ = nn.Sequential(*layers)
        self.linears_ = [m for m in layers if isinstance(m, nn.Linear)]

        if self.loss == "mse":
            self.loss_fn_ = nn.MSELoss(reduction="mean")
        else:
            self.loss_fn_ = nn.HuberLoss(reduction="mean", delta=self.huber_delta)

        groups = [
            {"params": [m.weight for m in self.linears_], "weight_decay": self.l2},
            {"params": [m.bias for m in self.linears_], "weight_decay": 0.0},
        ]
        if self.optimizer == "adam":
            self.optimizer_ = torch.optim.Adam(groups, lr=self.learning_rate)
        else:
            self.optimizer_ = torch.optim.SGD(groups, lr=self.learning_rate)

    def _train_step(self, Xb: np.ndarray, yb: np.ndarray) -> float:
        with _deterministic():
            self.network_.train()
            self.optimizer_.zero_grad()
            pred = self.network_(_tensor(Xb))
            loss = self.loss_fn_(pred, _tensor(yb).reshape(-1, 1))
            loss.backward()  # autograd computes every gradient
            self.optimizer_.step()
        return float(loss.item())

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        self.network_.eval()
        with torch.no_grad():
            pred = self.network_(_tensor(X))
            return float(self.loss_fn_(pred, _tensor(y).reshape(-1, 1)).item())

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self.network_.eval()
        with torch.no_grad():
            return self.network_(_tensor(X))[:, 0].numpy().copy()

    def get_weights(self) -> list[np.ndarray]:
        """Copies of ``[W1, b1, ...]`` in NumPy layout (``W`` transposed back to ``(in, out)``)."""
        weights: list[np.ndarray] = []
        for m in self.linears_:
            weights += [m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()]
        return weights

    def _set_weights(self, weights: list[np.ndarray]) -> None:
        with torch.no_grad():
            for i, m in enumerate(self.linears_):
                m.weight.copy_(_tensor(weights[2 * i].T))
                m.bias.copy_(_tensor(weights[2 * i + 1]))
