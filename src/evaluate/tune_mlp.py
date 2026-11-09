"""Phase 5: diagnose and tune the from-scratch NumPy network on the 8 CV folds.

Two things the network class itself cannot do are added by a thin wrapper, `MLPVariant`,
which still satisfies the model contract (`fit(X, y)` / `predict(X)` on the preprocessed
matrix):

- **feature_set** — `"no_calendar"` drops `doy_sin`, `doy_cos` and `month`. The feature table
  starts in June 2025, so before fold 2 the network has never seen a March–May day of year;
  it maps those unseen calendar values onto "winter" and overshoots (see `diagnose`).
- **target** — `"residual"` learns the change from a *fitted persistence* anchor,
  `a + b · pm25_prev_day_mean`, instead of the PM2.5 level. The model only sees the
  standard-scaled column, but a scaled column is a linear function of the raw one, so an
  ordinary least-squares fit of `y` on it recovers the same line as a fit on raw persistence.
  (Exact `y − persistence` would need the fold's scaler, which the contract does not pass in.)

The search is staged (coarse grid → one-at-a-time sweep around the best → greedy combination
of the sweep's winners), selected by pooled MAE via `tune.search`; band macro-F1 is recorded.

CLI: `uv run python -m src.evaluate.tune_mlp [--features PATH] [--quick]` writes, under
`results/tuning/`: `mlp_numpy_trials.csv`, `mlp_numpy_best.json`,
`mlp_numpy_cv_predictions.parquet` and `mlp_numpy_diagnosis.csv`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.evaluate import tune

from src.models.numpy_nn.variant import (  # noqa: F401  (re-exported for tests)
    CALENDAR_COLUMNS,
    FEATURE_SETS,
    PERSISTENCE_INDEX,
    TARGETS,
    TRANSFORMED_COLUMNS,
    MLPVariant,
)


# --- search space -------------------------------------------------------------------------

BASE_PARAMS: dict[str, Any] = {
    "feature_set": "all", "target": "level", "hidden_sizes": (64, 32), "activation": "relu",
    "loss": "mse", "huber_delta": 1.0, "learning_rate": 1e-3, "batch_size": 32, "l2": 1e-4,
    "dropout": 0.0, "early_stopping": True, "validation_fraction": 0.15, "patience": 20,
    "max_epochs": 200,
}

COARSE_GRID: dict[str, list] = {
    "feature_set": ["all", "no_calendar"],
    "target": ["level", "residual"],
    "hidden_sizes": [(16,), (32,), (64, 32)],
}

# One-at-a-time changes tried around the coarse winner; each entry is a set of overrides.
SWEEP: dict[str, list[dict]] = {
    "activation": [{"activation": "tanh"}],
    "loss": [{"loss": "huber", "huber_delta": 1.0}, {"loss": "huber", "huber_delta": 0.3}],
    "learning_rate": [{"learning_rate": 3e-4}, {"learning_rate": 3e-3}],
    "batch_size": [{"batch_size": 64}, {"batch_size": 128}],
    "l2": [{"l2": 1e-3}, {"l2": 1e-2}, {"l2": 1e-1}],
    "dropout": [{"dropout": 0.1}, {"dropout": 0.3}],
    "patience": [{"patience": 10}, {"patience": 40}],
    "validation_fraction": [{"validation_fraction": 0.25}],
    "early_stopping": [{"early_stopping": False, "max_epochs": 20}, {"early_stopping": False, "max_epochs": 60}],
}

QUICK_COARSE_GRID: dict[str, list] = {"feature_set": ["all", "no_calendar"], "target": ["level", "residual"]}
QUICK_SWEEP: dict[str, list[dict]] = {"l2": [{"l2": 1e-2}], "loss": [{"loss": "huber", "huber_delta": 1.0}]}
QUICK_BASE: dict[str, Any] = {**BASE_PARAMS, "hidden_sizes": (8,), "max_epochs": 5, "patience": 2, "batch_size": 64}
SEED_CHECK = (0, 1, 2)


def _score_table(folds: list[tune.PreparedFold], candidates: list[dict], stage: str, verbose: bool) -> pd.DataFrame:
    if verbose:
        print(f"--- stage: {stage} ({len(candidates)} candidates)", flush=True)
    table = tune.search(folds, MLPVariant, candidates, kind="regression", verbose=verbose)
    return table.assign(stage=stage)


def _params_of(row: pd.Series) -> dict:
    """A trial row's configuration as plain Python values (no NumPy scalars)."""
    return {k: (row[k].item() if isinstance(row[k], np.generic) else row[k]) for k in BASE_PARAMS}


def _same(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in BASE_PARAMS)


def staged_search(
    folds: list[tune.PreparedFold], base: dict, coarse: dict[str, list], sweep: dict[str, list[dict]], verbose: bool = True
) -> pd.DataFrame:
    """Coarse grid → one-at-a-time sweep around its best → greedy combination of the sweep's wins.

    Returns every trial (one row per configuration, stage-labelled), best pooled MAE first.
    """
    stage1 = _score_table(folds, [{**base, **c} for c in tune.param_grid(coarse)], "coarse", verbose)
    centre, centre_mae = _params_of(stage1.iloc[0]), float(stage1.iloc[0]["mae"])

    sweep_candidates, dims = [], []
    for dim, options in sweep.items():
        for override in options:
            candidate = {**centre, **override}
            if not _same(candidate, centre):
                sweep_candidates.append(candidate)
                dims.append(dim)
    stage2 = _score_table(folds, sweep_candidates, "sweep", verbose) if sweep_candidates else stage1.iloc[:0]

    # The best improving override in each dimension, strongest first, then added one at a time.
    winners: dict[str, tuple[float, dict]] = {}
    for candidate, dim in zip(sweep_candidates, dims):
        row = stage2.loc[[_same(_params_of(r), candidate) for _, r in stage2.iterrows()]].iloc[0]
        if row["mae"] < centre_mae and row["mae"] < winners.get(dim, (np.inf, {}))[0]:
            winners[dim] = (float(row["mae"]), {k: candidate[k] for k in BASE_PARAMS if candidate[k] != centre[k]})
    ordered = [override for _, override in sorted(winners.values(), key=lambda t: t[0])]
    combos, current = [], dict(centre)
    for i, override in enumerate(ordered):
        current = {**current, **override}
        if i >= 1:  # a single override was already scored in the sweep
            combos.append(dict(current))
    stage3 = _score_table(folds, combos, "combine", verbose) if combos else stage1.iloc[:0]

    trials = pd.concat([stage1, stage2, stage3], ignore_index=True)
    return trials.sort_values("mae", kind="stable").reset_index(drop=True)


# --- diagnosis ----------------------------------------------------------------------------


def diagnose(folds: list[tune.PreparedFold], base: dict) -> pd.DataFrame:
    """Per-fold train / validation / test error of `base`, with and without calendar features.

    A large train-vs-validation gap means overfitting; a large positive test `bias` means the
    model overshoots; comparing `feature_set` rows isolates the calendar extrapolation.
    """
    rows = []
    for feature_set in ("all", "no_calendar"):
        for f in folds:
            model = MLPVariant(**{**base, "feature_set": feature_set}).fit(f.X_train, f.y_train)
            net = model.model_
            n_fit = net._split(len(f.y_train))
            pred_train, pred_test = model.predict(f.X_train), model.predict(f.X_test)
            val_mae = float(np.mean(np.abs(pred_train[n_fit:] - f.y_train[n_fit:]))) if n_fit < len(f.y_train) else np.nan
            rows.append({
                "feature_set": feature_set,
                "fold": f.index,
                "n_train": len(f.y_train),
                "best_epoch": net.best_epoch_,
                "n_epochs": net.n_epochs_,
                "train_mae": float(np.mean(np.abs(pred_train[:n_fit] - f.y_train[:n_fit]))),
                "val_mae": val_mae,
                "test_mae": float(np.mean(np.abs(pred_test - f.y_test))),
                "test_bias": float(np.mean(pred_test - f.y_test)),
                "persistence_mae": float(np.mean(np.abs(f.persistence_test - f.y_test))),
                "train_target_mean": float(f.y_train.mean()),
                "test_target_mean": float(f.y_test.mean()),
            })
    return pd.DataFrame(rows)


# --- CLI ----------------------------------------------------------------------------------


def per_fold_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Model and persistence MAE per fold from a `Score.predictions` table."""
    err = predictions.assign(
        model=(predictions["y_pred"] - predictions["y_true"]).abs(),
        persistence=(predictions["persistence"] - predictions["y_true"]).abs(),
    )
    return err.groupby("fold")[["model", "persistence"]].mean().reset_index()


def _jsonable(params: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def run(
    features_path: Path | None = None, results_dir: Path | None = None, quick: bool = False, verbose: bool = True
) -> dict:
    """Diagnose, search, rescore the winner, and write the four `results/tuning/` files."""
    features_path = features_path or PROCESSED_DIR / "features.parquet"
    results_dir = results_dir or RESULTS_DIR / "tuning"
    results_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    folds = tune.prepare_folds(pd.read_parquet(features_path))
    base = QUICK_BASE if quick else BASE_PARAMS
    diagnosis = diagnose(folds, base)
    trials = staged_search(folds, base, QUICK_COARSE_GRID if quick else COARSE_GRID, QUICK_SWEEP if quick else SWEEP, verbose)

    best_params = _params_of(trials.iloc[0])
    best = tune.score_regressor(folds, MLPVariant, best_params)
    seed_maes = [
        float(tune.score_regressor(folds, MLPVariant, {**best_params, "random_state": s}).metrics["mae"])
        for s in (SEED_CHECK[:1] if quick else SEED_CHECK)
    ]
    seconds = time.perf_counter() - start

    trials.to_csv(results_dir / "mlp_numpy_trials.csv", index=False)
    diagnosis.to_csv(results_dir / "mlp_numpy_diagnosis.csv", index=False)
    best.predictions.to_parquet(results_dir / "mlp_numpy_cv_predictions.parquet", index=False)
    summary = {
        "params": _jsonable(best_params),
        "pooled": {k: float(v) for k, v in best.metrics.items() if np.isscalar(v)},
        "per_fold_mae": per_fold_mae(best.predictions).round(4).to_dict(orient="records"),
        "seed_check_mae": {"seeds": list(SEED_CHECK[: len(seed_maes)]), "mae": seed_maes},
        "n_trials": len(trials),
        "quick": quick,
        "total_seconds": round(seconds, 1),
    }
    (results_dir / "mlp_numpy_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose and tune the NumPy MLP on the CV folds.")
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "features.parquet")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "tuning")
    parser.add_argument("--quick", action="store_true", help="tiny grid and few epochs (for tests)")
    args = parser.parse_args(argv)
    summary = run(args.features, args.results_dir, quick=args.quick)
    print(json.dumps({k: summary[k] for k in ("params", "pooled", "total_seconds")}, indent=2))


if __name__ == "__main__":
    main()
