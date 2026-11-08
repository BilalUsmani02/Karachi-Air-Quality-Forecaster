"""Optimisers that turn gradients into weight updates: SGD and Adam.

**L2 regularisation** is done the classic way, as "weight decay added to the gradient": for
every parameter with ``decay=True`` (weight matrices, not biases) the optimiser uses

    g = grad + l2 * W

which is the exact gradient of the extra penalty ``(l2 / 2) * Σ W²``. This penalty favours
small weights, i.e. smoother functions that overfit less. It is also precisely what
``torch.optim.SGD`` / ``torch.optim.Adam`` do with ``weight_decay=l2`` (not AdamW's
"decoupled" decay), so the PyTorch twin reproduces it by giving the weights a parameter group
with ``weight_decay=l2`` and the biases one with ``weight_decay=0``.

The optimiser updates the network's own ``Param.value`` arrays in place — that is what
"training" means — but never touches the caller's data.
"""

from __future__ import annotations

import numpy as np

from src.models.numpy_nn.layers import Param


def regularised_grad(param: Param, l2: float) -> np.ndarray:
    """Gradient including the L2 term (``grad + l2 * W`` for weights, plain ``grad`` for biases)."""
    if param.decay and l2 > 0:
        return param.grad + l2 * param.value
    return param.grad


def l2_penalty(params: list[Param], l2: float) -> float:
    """The loss term whose gradient ``regularised_grad`` adds: ``(l2 / 2) * Σ W²`` over weights."""
    return 0.5 * l2 * sum(float(np.sum(p.value**2)) for p in params if p.decay)


class SGD:
    """Plain mini-batch gradient descent: ``W ← W − lr · g``."""

    def __init__(self, params: list[Param], lr: float = 0.01, l2: float = 0.0) -> None:
        self.params = params
        self.lr = lr
        self.l2 = l2

    def step(self) -> None:
        for p in self.params:
            p.value -= self.lr * regularised_grad(p, self.l2)


class Adam:
    """Adam (Kingma & Ba, 2015), with the same defaults as ``torch.optim.Adam``.

    Per parameter it keeps two running averages of the gradient ``g``:

    - ``m ← β1·m + (1 − β1)·g``   (the average direction — "momentum")
    - ``v ← β2·v + (1 − β2)·g²``  (the average squared size)

    Both start at 0, so early on they are biased towards 0; dividing by ``1 − β^t`` (``t`` =
    step count) corrects this. The update is

        W ← W − lr · m̂ / (sqrt(v̂) + eps),   m̂ = m / (1 − β1^t),  v̂ = v / (1 − β2^t)

    so every weight gets its own step size: large where gradients are consistently small,
    small where they are large or noisy.
    """

    def __init__(
        self,
        params: list[Param],
        lr: float = 1e-3,
        l2: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.l2 = l2
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]

    def step(self) -> None:
        self.t += 1
        correction1 = 1.0 - self.beta1**self.t
        correction2 = 1.0 - self.beta2**self.t
        for i, p in enumerate(self.params):
            g = regularised_grad(p, self.l2)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g**2
            m_hat = self.m[i] / correction1
            v_hat = self.v[i] / correction2
            p.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, params: list[Param], lr: float, l2: float) -> SGD | Adam:
    """Build an optimiser from its name (``"adam"`` or ``"sgd"``)."""
    if name == "adam":
        return Adam(params, lr=lr, l2=l2)
    if name == "sgd":
        return SGD(params, lr=lr, l2=l2)
    raise ValueError(f"optimizer must be 'adam' or 'sgd', got {name!r}")
