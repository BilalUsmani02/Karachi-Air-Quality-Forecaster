"""Side-by-side benchmark of the NumPy network and its PyTorch twin on the same data."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

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
