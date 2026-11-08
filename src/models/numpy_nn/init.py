"""Weight initialisation (He and Xavier/Glorot) from a seeded NumPy generator.

Why it matters: if the starting weights are too large, activations blow up layer after layer;
too small, and they (and the gradients) shrink towards zero. Both schemes pick the spread of
the random weights so the signal keeps roughly the same size through the network.

- **He** (for ReLU): ``W ~ Normal(0, sqrt(2 / fan_in))``. ReLU zeroes about half its inputs,
  so the variance is doubled to compensate.
- **Xavier / Glorot** (for tanh and the linear output layer):
  ``W ~ Uniform(−a, a)`` with ``a = sqrt(6 / (fan_in + fan_out))``, which keeps the variance
  equal going forwards and backwards.

Biases start at zero. ``fan_in`` / ``fan_out`` = number of inputs / outputs of the layer.
"""

from __future__ import annotations

import numpy as np


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He initialisation: ``(fan_in, fan_out)`` weights with std ``sqrt(2 / fan_in)``."""
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot initialisation: uniform in ``±sqrt(6 / (fan_in + fan_out))``."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def init_mlp_weights(
    layer_sizes: list[int], activation: str, rng: np.random.Generator
) -> list[np.ndarray]:
    """Initial ``[W1, b1, W2, b2, ...]`` for an MLP with the given layer widths.

    ``layer_sizes`` = ``[d_in, hidden_1, ..., 1]``. Hidden layers use He for ReLU-type
    activations and Xavier for tanh; the output layer (no activation) uses Xavier.
    """
    weights: list[np.ndarray] = []
    n_layers = len(layer_sizes) - 1
    for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        is_output = i == n_layers - 1
        if not is_output and activation in ("relu", "leaky_relu"):
            W = he_normal(fan_in, fan_out, rng)
        else:
            W = xavier_uniform(fan_in, fan_out, rng)
        weights += [W, np.zeros(fan_out)]
    return weights
