"""Numerical gradient check: proof that the hand-written backward passes are right.

Idea: a derivative is a slope. Nudge one number ``w`` by a tiny ``eps`` up and down, rerun
the forward pass, and the loss changes by about ``2 · eps · dL/dw``. So

    dL/dw ≈ (L(w + eps) − L(w − eps)) / (2 · eps)        (central difference)

This needs only the forward pass, which is easy to get right. If it agrees with the analytic
gradient from ``backward`` for every single number, backward is correct. Agreement is
measured by the **relative error** ``‖a − n‖ / (‖a‖ + ‖n‖)``; in float64 a correct
implementation gives ~1e-8 or smaller, a bug gives something near 1.

The check is slow (two forward passes per number), so it is used on tiny networks in tests.

Run it from the command line to print the relative error of every parameter of a small
2-hidden-layer network (each activation × MSE and Huber, with L2) and save the table:

    uv run python -m src.models.numpy_nn.gradcheck      # -> results/numpy_vs_torch/gradcheck.csv
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR

from src.models.numpy_nn.init import init_mlp_weights
from src.models.numpy_nn.layers import ACTIVATIONS, Layer
from src.models.numpy_nn.losses import HuberLoss, MSELoss, make_loss
from src.models.numpy_nn.network import Sequential, build_mlp
from src.models.numpy_nn.optimizers import l2_penalty, regularised_grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """``‖a − n‖ / (‖a‖ + ‖n‖)``, or 0 when both are exactly zero."""
    denom = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / denom)


def numerical_gradient(f: Callable[[], float], array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of ``f()`` with respect to every entry of ``array``.

    ``array`` is nudged in place and restored, so ``f`` must read it (e.g. it is a layer's
    weight matrix or the input that ``f`` feeds forward).
    """
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + eps
        f_plus = f()
        array[idx] = original - eps
        f_minus = f()
        array[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def check_layer(layer: Layer, X: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Relative errors for a single layer's input gradient and parameter gradients.

    Uses the scalar loss ``L = Σ (output ⊙ R)`` with a fixed random ``R``, so the upstream
    gradient fed to ``backward`` is simply ``R``.
    """
    X = np.array(X, dtype=np.float64)
    R = np.random.default_rng(seed).normal(size=layer.forward(X).shape)

    def f() -> float:
        return float(np.sum(layer.forward(X, training=False) * R))

    layer.forward(X)
    errors = {"input": relative_error(layer.backward(R), numerical_gradient(f, X))}
    for p in layer.params():
        errors[p.name] = relative_error(p.grad.copy(), numerical_gradient(f, p.value))
    return errors


def check_loss(loss: MSELoss | HuberLoss, pred: np.ndarray, y: np.ndarray) -> float:
    """Relative error of ``loss.backward()`` against the numerical dL/dpred."""
    pred = np.array(pred, dtype=np.float64)
    loss.forward(pred, y)
    analytic = loss.backward()
    numeric = numerical_gradient(lambda: loss.forward(pred, y), pred)
    return relative_error(analytic, numeric)


def check_network(
    network: Sequential,
    loss: MSELoss | HuberLoss,
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
) -> dict[str, float]:
    """Relative error for every parameter of a whole network + loss (+ optional L2 term).

    The total loss checked is ``data loss + (l2 / 2) Σ W²``, whose analytic gradient is what
    the optimiser actually uses (``regularised_grad``). Dropout layers must be absent or run
    in prediction mode — a random mask would change between the two nudged forward passes.
    """
    X = np.array(X, dtype=np.float64)
    params = network.params()

    def total_loss() -> float:
        return loss.forward(network.forward(X, training=False), y) + l2_penalty(params, l2)

    total_loss()
    network.backward(loss.backward())
    analytic = {p.name: regularised_grad(p, l2).copy() for p in params}
    return {
        p.name: relative_error(analytic[p.name], numerical_gradient(total_loss, p.value))
        for p in params
    }


GRADCHECK_CSV = RESULTS_DIR / "numpy_vs_torch" / "gradcheck.csv"
CSV_FIELDS = ("loss", "activation", "parameter", "shape", "relative_error")


def run_gradcheck(
    layer_sizes: tuple[int, ...] = (4, 6, 5, 1),
    n_rows: int = 12,
    l2: float = 0.01,
    huber_delta: float = 0.5,
    seed: int = 0,
) -> list[dict]:
    """Relative error of every parameter of a small network, for each activation × loss.

    Random inputs/targets (standard normal) and the He/Xavier initial weights. With
    ``huber_delta=0.5`` some residuals fall on each side of delta, so both branches of the
    Huber gradient are exercised. Returns one row per (loss, activation, parameter).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, layer_sizes[0]))
    y = rng.normal(size=n_rows)
    rows = []
    for loss_name in ("mse", "huber"):
        for activation in ACTIVATIONS:
            weights = init_mlp_weights(list(layer_sizes), activation, np.random.default_rng(seed + 1))
            network = build_mlp(weights, activation)
            errors = check_network(network, make_loss(loss_name, huber_delta), X, y, l2=l2)
            shapes = {p.name: p.value.shape for p in network.params()}
            rows += [
                {"loss": loss_name, "activation": activation, "parameter": name,
                 "shape": "x".join(map(str, shapes[name])), "relative_error": err}
                for name, err in errors.items()
            ]
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({**r, "relative_error": f"{r['relative_error']:.3e}"} for r in rows)
    return path


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description="Gradient check of the NumPy network (central differences).")
    parser.add_argument("--out", type=Path, default=GRADCHECK_CSV)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = run_gradcheck(seed=args.seed)
    print(f"{'loss':<6} {'activation':<11} {'parameter':<10} {'shape':<6} relative error")
    for r in rows:
        print(f"{r['loss']:<6} {r['activation']:<11} {r['parameter']:<10} {r['shape']:<6} {r['relative_error']:.2e}")
    worst = max(r["relative_error"] for r in rows)
    print(f"worst relative error: {worst:.2e} (a correct gradient gives ~1e-8 or less; a bug gives ~1)")
    print(f"saved {write_csv(rows, args.out)}")
    return rows


if __name__ == "__main__":
    main()
