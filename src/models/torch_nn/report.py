"""NumPy network vs PyTorch twin on the real feature table, written to ``results/numpy_vs_torch/``.

    uv run python -m src.models.torch_nn

What is compared (like-for-like):

- **Rows.** Training = every labelled row before ``split.FINAL_TEST_START`` (2026-09-01), in
  ``(date, sensor_id)`` order — the same rows and order the final-test run trains the tuned
  network on. The ``Preprocessor`` is fitted on those rows only. Predictions are compared on
  the final-test rows (from 2026-09-01). Only the two models' *outputs* are compared with each
  other; the targets of those rows are never read, so this does not score the final test.
- **Model.** The tuned network from ``final.MLP_PARAMS``: 1 hidden layer of 32 ReLU units,
  MSE, Adam (lr 1e-3), batch 32, L2 0.1, early stopping (last 15 % of rows, patience 20).
  The two Phase 5 extras, ``feature_set="no_calendar"`` and ``target="residual"``, live in the
  ``MLPVariant`` wrapper, not in either network. Both networks are therefore run *inside the
  same wrapper class* (only ``regressor_cls`` differs): the same three calendar columns are
  dropped from both inputs, and both learn the same residual target ``y − (a + b·persistence)``
  with ``(a, b)`` fitted once by least squares on the training rows.
- **Start and batches.** Same ``random_state`` → the shared training loop (``estimator_base``)
  draws identical initial weights and identical shuffled mini-batches for both; the report
  checks the initial weights really are equal (difference 0).

Outputs: ``summary.json``, ``summary.md``, ``loss_curves.png``, ``learning_curve.png``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.data.split import FINAL_TEST_START, final_test_mask
from src.models.dataset import FEATURE_COLUMNS, Preprocessor, make_xy
from src.models.final import MLP_PARAMS
from src.models.numpy_nn.variant import MLPVariant
from src.models.torch_nn import plots
from src.models.torch_nn.compare import equivalence
from src.models.torch_nn.model import MLPRegressorTorch

OUT_DIR = RESULTS_DIR / "numpy_vs_torch"


class MLPVariantTorch(MLPVariant):
    """``MLPVariant`` with the PyTorch twin inside: same columns, same residual target."""

    regressor_cls = MLPRegressorTorch


def prepare(df: pd.DataFrame) -> dict[str, Any]:
    """Training matrix/target (before ``FINAL_TEST_START``) and the comparison rows (from it)."""
    data = df.sort_values(["date", "sensor_id"], kind="stable").reset_index(drop=True)
    is_test = final_test_mask(data, FINAL_TEST_START)
    dev, test = data.loc[~is_test], data.loc[is_test]
    if test.empty:
        raise ValueError(f"no rows on or after {FINAL_TEST_START} to compare predictions on")
    dev = dev.loc[dev["y_pm25"].notna()]
    X_dev_df, y_dev = make_xy(dev, "y_pm25")
    pre = Preprocessor()
    X_dev = pre.fit_transform(X_dev_df)
    X_eval = pre.transform(test[list(FEATURE_COLUMNS)])
    return {
        "X_train": X_dev,
        "y_train": y_dev.to_numpy(np.float64),
        "X_eval": X_eval,
        "train_period": [str(pd.Timestamp(dev["date"].min()).date()), str(pd.Timestamp(dev["date"].max()).date())],
        "eval_period": [str(pd.Timestamp(test["date"].min()).date()), str(pd.Timestamp(test["date"].max()).date())],
    }


def _timed_fit(model: MLPVariant, X: np.ndarray, y: np.ndarray) -> float:
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


def versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }


def run(df: pd.DataFrame, out_dir: Path = OUT_DIR, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit both twins on ``df``, write every output file and return the summary dict."""
    params = dict(MLP_PARAMS if params is None else params)
    params.setdefault("random_state", SEED)
    data = prepare(df)
    X, y, X_eval = data["X_train"], data["y_train"], data["X_eval"]

    numpy_model, torch_model = MLPVariant(**params), MLPVariantTorch(**params)
    seconds = {"numpy": _timed_fit(numpy_model, X, y), "torch": _timed_fit(torch_model, X, y)}
    diff = equivalence(numpy_model.model_, torch_model.model_, numpy_model.predict(X_eval), torch_model.predict(X_eval))
    diff["train_prediction_max_abs_diff"] = float(np.max(np.abs(numpy_model.predict(X) - torch_model.predict(X))))
    if max(diff["initial_weight_max_abs_diff"].values()) != 0.0:
        raise RuntimeError("the two networks did not start from identical weights")

    summary = {
        "what": "NumPy network vs its PyTorch twin, trained from identical initial weights on identical "
        "mini-batches. Differences are in µg/m³ for predictions and standardised units for losses.",
        "seed": params["random_state"],
        "params": _jsonable(params),
        "data": {
            "train_rows": int(len(X)),
            "train_period": data["train_period"],
            "n_inputs_after_wrapper": int(numpy_model.columns_.size),
            "prediction_rows": "final-test rows (date >= FINAL_TEST_START); only the two models' predictions "
            "are compared with each other, targets are not used — this is not a score on the final test",
            "prediction_period": data["eval_period"],
        },
        "like_for_like": [
            "both networks run inside the same MLPVariant wrapper (only regressor_cls differs)",
            "no_calendar: doy_sin, doy_cos, month dropped from both inputs",
            "residual: both learn y - (a + b * pm25_prev_day_mean), (a, b) fitted once on the training rows",
            "float64 on CPU; same seed -> same initial weights (checked equal) and same mini-batch order",
        ],
        **diff,
        "fit_seconds": {**{k: round(v, 3) for k, v in seconds.items()}, "torch_over_numpy": round(seconds["torch"] / seconds["numpy"], 2)},
        "versions": versions(),
    }
    write(summary, numpy_model, torch_model, Path(out_dir))
    return summary


def _fmt(x: float) -> str:
    return "0" if x == 0 else f"{x:.1e}"


def summary_markdown(s: dict[str, Any]) -> str:
    weights = s["weight_max_abs_diff"]
    loss = s["loss_diff"]
    rows = [
        ("Prediction difference, max (µg/m³)", _fmt(s["prediction_diff"]["max_abs"])),
        ("Prediction difference, mean (µg/m³)", _fmt(s["prediction_diff"]["mean_abs"])),
        ("Prediction difference on training rows, max (µg/m³)", _fmt(s["train_prediction_max_abs_diff"])),
        *((f"Final weight difference, max, `{name}`", _fmt(v)) for name, v in weights.items()),
        ("Per-epoch training loss difference, max", _fmt(loss["max_train_epoch"])),
        ("Per-epoch validation loss difference, max", _fmt(loss["max_val_epoch"] or 0.0)),
        (f"Per-step (mini-batch) loss difference, max over {loss['n_steps']} steps", _fmt(loss["max_step"])),
        ("Epochs run (NumPy / PyTorch)", f"{s['epochs']['numpy']} / {s['epochs']['torch']}"),
        ("Best epoch (NumPy / PyTorch)", f"{s['epochs']['numpy_best_epoch']} / {s['epochs']['torch_best_epoch']}"),
        ("Training time, NumPy (s)", f"{s['fit_seconds']['numpy']:.1f}"),
        ("Training time, PyTorch (s)", f"{s['fit_seconds']['torch']:.1f}"),
    ]
    v = s["versions"]
    lines = [
        "# NumPy network vs PyTorch twin (real data)",
        "",
        f"Tuned network (`final.MLP_PARAMS`), {s['data']['train_rows']} training rows "
        f"({s['data']['train_period'][0]} to {s['data']['train_period'][1]}), seed {s['seed']}. "
        "Both start from the same weights and see the same mini-batches.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        *(f"| {k} | {val} |" for k, val in rows),
        "",
        f"Predictions compared on {s['prediction_diff']['n_rows']} final-test rows "
        f"({' to '.join(s['data']['prediction_period'])}). Only the two models' outputs are compared; "
        "the targets are not read, so this is not a final-test score.",
        "",
        "Like-for-like: " + "; ".join(s["like_for_like"]) + ".",
        "",
        f"Python {v['python']}, NumPy {v['numpy']}, PyTorch {v['torch']} ({v['torch_threads']} threads). "
        "Regenerate with `uv run python -m src.models.torch_nn`.",
    ]
    return "\n".join(lines) + "\n"


def write(summary: dict[str, Any], numpy_model: MLPVariant, torch_model: MLPVariant, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    net_np, net_t = numpy_model.model_, torch_model.model_
    plots.loss_curves(net_np.history_, net_t.history_, summary["loss_diff"], out_dir / "loss_curves.png")
    plots.learning_curve(
        net_np.history_, net_np.best_epoch_, out_dir / "learning_curve.png",
        title=f"Tuned NumPy network on real data: {summary['data']['train_rows']} rows, "
        f"{summary['epochs']['numpy']} epochs",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the NumPy network and its PyTorch twin on the real data.")
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "features.parquet")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    run(pd.read_parquet(args.features), args.out_dir)
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))
