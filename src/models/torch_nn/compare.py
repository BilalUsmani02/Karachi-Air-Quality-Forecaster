"""Side-by-side benchmark of the NumPy network and its PyTorch twin on the same data."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.models.numpy_nn.estimator_base import BaseMLPRegressor
from src.models.numpy_nn.network import MLPRegressorNP
from src.models.torch_nn.model import MLPRegressorTorch


def compare_numpy_torch(X: np.ndarray, y: np.ndarray, **params: Any) -> dict[str, Any]:
    """Fit both models with identical ``params`` (same seed → same init and batches).

    Returns training time, epochs run, final train / validation loss (standardised units,
    see ``estimator_base``) for each model, plus the largest absolute difference between
    their predictions on ``X`` and between their final weights. With ``dropout=0`` both
    differences should be tiny (floating-point noise); with dropout they differ by design.
    """
    result: dict[str, Any] = {}
    models = {"numpy": MLPRegressorNP(**params), "torch": MLPRegressorTorch(**params)}
    predictions = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X, y)
        result[f"{name}_fit_seconds"] = time.perf_counter() - start
        result[f"{name}_epochs"] = model.n_epochs_
        result[f"{name}_train_loss"] = model.history_["train_loss"][-1]
        val = model.history_["val_loss"]
        result[f"{name}_best_val_loss"] = min(val) if val else None
        predictions[name] = model.predict(X)

    result["max_abs_prediction_diff"] = float(np.max(np.abs(predictions["numpy"] - predictions["torch"])))
    result["max_abs_weight_diff"] = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(models["numpy"].get_weights(), models["torch"].get_weights())
    )
    return result


def layer_names(weights: list[np.ndarray]) -> list[str]:
    """``["dense1.W", "dense1.b", "dense2.W", ...]`` — the names ``build_mlp`` gives the params."""
    return [f"dense{i // 2 + 1}.{'W' if i % 2 == 0 else 'b'}" for i in range(len(weights))]


def _abs_diff(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"histories differ in length ({len(a)} vs {len(b)}): training diverged")
    return [float(abs(x - y)) for x, y in zip(a, b)]


def equivalence(
    numpy_net: BaseMLPRegressor,
    torch_net: BaseMLPRegressor,
    numpy_pred: np.ndarray,
    torch_pred: np.ndarray,
) -> dict[str, Any]:
    """How far apart two fitted twins are: predictions, weights per layer, and losses per epoch.

    ``numpy_net`` / ``torch_net`` are the fitted networks themselves (they hold the history and
    weights); the predictions are passed separately so a wrapper (e.g. ``MLPVariant``, which
    adds an anchor back) can supply its final outputs.
    """
    pred_diff = np.abs(np.asarray(numpy_pred, np.float64) - np.asarray(torch_pred, np.float64))
    names = layer_names(numpy_net.get_weights())
    init_diff = {
        n: float(np.max(np.abs(a - b)))
        for n, a, b in zip(names, numpy_net.initial_weights_, torch_net.initial_weights_)
    }
    weight_diff = {
        n: float(np.max(np.abs(a - b)))
        for n, a, b in zip(names, numpy_net.get_weights(), torch_net.get_weights())
    }
    h_np, h_t = numpy_net.history_, torch_net.history_
    train = _abs_diff(h_np["train_loss"], h_t["train_loss"])
    val = _abs_diff(h_np["val_loss"], h_t["val_loss"])
    step = _abs_diff(h_np["step_loss"], h_t["step_loss"])
    return {
        "prediction_diff": {
            "n_rows": int(pred_diff.size),
            "max_abs": float(pred_diff.max()),
            "mean_abs": float(pred_diff.mean()),
        },
        "initial_weight_max_abs_diff": init_diff,
        "weight_max_abs_diff": weight_diff,
        "epochs": {
            "numpy": int(numpy_net.n_epochs_),
            "torch": int(torch_net.n_epochs_),
            "numpy_best_epoch": int(numpy_net.best_epoch_),
            "torch_best_epoch": int(torch_net.best_epoch_),
        },
        "loss_diff": {
            "max_train_epoch": max(train),
            "max_val_epoch": max(val) if val else None,
            "max_step": max(step),
            "n_steps": len(step),
            "train_per_epoch": train,
            "val_per_epoch": val,
        },
    }
