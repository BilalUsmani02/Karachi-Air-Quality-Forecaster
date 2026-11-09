"""Hyper-parameter search for the classical (scikit-learn) models (Phase 5, spec §6.2/§6.4).

Reuses `src.evaluate.tune` (`prepare_folds`, `score_regressor`, `score_classifier`,
`param_grid`) to score candidates on the project's 8 rolling-origin CV folds. Each family is
tuned in **sequential stages**: one hyperparameter (or a tightly-coupled pair) is coarse-scanned
at a time, holding the rest at the running best, then the next stage refines around that choice.
This is cheaper than a full combinatorial grid (the brief's "coarse grid -> refine around the
best; don't brute-force") while still covering every hyperparameter named in the brief.

Families (file stem -> factory):

- Regressors (ranked by pooled MAE): `ridge`, `random_forest`, `hist_gb`
  (`HistGradientBoostingRegressor` -- the Phase 5 extra family).
- Classifiers (ranked by pooled macro-F1): `logistic_regression`, `random_forest_clf`,
  `hist_gb_clf` (`HistGradientBoostingClassifier`, `class_weight="balanced"` fixed).

CLI: `uv run python -m src.evaluate.tune_classical [--features PATH] [--quick]` loads
`data/processed/features.parquet` (or `--features`) and writes, per family, to
`results/tuning/`: `<family>_trials.csv` (every candidate scored), `<family>_best.json`
(best params + pooled metrics) and `<family>_cv_predictions.parquet` (the best config's pooled
CV predictions), plus a combined `results/tuning/classical_summary.md`. `--quick` shrinks every
grid to one or two candidates (used by the tests, not for real tuning).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.evaluate.tune import PreparedFold, Score, param_grid, prepare_folds, score_classifier, score_regressor
from src.models.baselines.sklearn_models import (
    hist_gb_classifier,
    hist_gb_regressor,
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    ridge,
)

StageGrid = Callable[[dict], list[dict]]  # (running-best params) -> candidates for this stage


# --- per-family stage grids ------------------------------------------------------------------
#
# Every stage function returns *fully-specified* candidate dicts (it merges the running best
# into each one), so later stages refine around a complete configuration, not a partial one.


def _ridge_stages(quick: bool) -> list[StageGrid]:
    if quick:
        return [lambda best: param_grid({"alpha": [0.1, 10.0]})]

    def coarse(best: dict) -> list[dict]:
        return param_grid({"alpha": [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]})

    def refine(best: dict) -> list[dict]:
        alpha = best["alpha"]
        alphas = sorted({round(alpha * f, 6) for f in (0.3, 0.5, 0.7, 1.5, 2.0, 3.0)})
        return [{**best, "alpha": a} for a in alphas]

    return [coarse, refine]


def _rf_regressor_stages(quick: bool) -> list[StageGrid]:
    if quick:
        return [lambda best: param_grid({"n_estimators": [10], "max_depth": [None], "max_features": [1.0]})]

    def depth(best: dict) -> list[dict]:
        return param_grid({"n_estimators": [100], "max_depth": [None, 15]})

    def leaf(best: dict) -> list[dict]:
        return [{**best, "min_samples_leaf": v} for v in (1, 5)]

    def features(best: dict) -> list[dict]:
        return [{**best, "max_features": v} for v in (1.0, "sqrt", 0.33, 0.5)]

    def n_estimators(best: dict) -> list[dict]:
        return [{**best, "n_estimators": v} for v in (300, 500)]

    return [depth, leaf, features, n_estimators]


def _hist_gb_regressor_stages(quick: bool) -> list[StageGrid]:
    if quick:
        return [lambda best: param_grid({"learning_rate": [0.1], "max_leaf_nodes": [15], "max_iter": [10]})]

    def rate(best: dict) -> list[dict]:
        return param_grid({"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [31], "max_iter": [100]})

    def leaves(best: dict) -> list[dict]:
        return [{**best, "max_leaf_nodes": v} for v in (15, 31, 63)]

    def l2(best: dict) -> list[dict]:
        return [{**best, "l2_regularization": v} for v in (0.0, 0.1, 1.0)]

    def iters(best: dict) -> list[dict]:
        return [{**best, "max_iter": v} for v in (100, 200, 300)]

    return [rate, leaves, l2, iters]


def _logistic_regression_stages(quick: bool) -> list[StageGrid]:
    if quick:
        return [lambda best: param_grid({"C": [1.0], "class_weight": ["balanced"]})]

    def c_coarse(best: dict) -> list[dict]:
        return param_grid({"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": ["balanced"]})

    def class_weight(best: dict) -> list[dict]:
        return [{**best, "class_weight": v} for v in ("balanced", None)]

    def c_refine(best: dict) -> list[dict]:
        c = best["C"]
        cs = sorted({round(c * f, 6) for f in (0.3, 0.5, 1.5, 2.0, 3.0)})
        return [{**best, "C": v} for v in cs]

    return [c_coarse, class_weight, c_refine]


def _rf_classifier_stages(quick: bool) -> list[StageGrid]:
    if quick:
        return [lambda best: param_grid({"n_estimators": [10], "max_depth": [None], "class_weight": ["balanced"]})]

    def depth(best: dict) -> list[dict]:
        return param_grid({"n_estimators": [100], "max_depth": [None, 15]})

    def class_weight(best: dict) -> list[dict]:
        return [{**best, "class_weight": v} for v in ("balanced", "balanced_subsample")]

    def leaf(best: dict) -> list[dict]:
        return [{**best, "min_samples_leaf": v} for v in (1, 5)]

    def features(best: dict) -> list[dict]:
        return [{**best, "max_features": v} for v in (1.0, "sqrt", 0.33, 0.5)]

    def n_estimators(best: dict) -> list[dict]:
        return [{**best, "n_estimators": v} for v in (300, 500)]

    return [depth, class_weight, leaf, features, n_estimators]


def _hist_gb_classifier_stages(quick: bool) -> list[StageGrid]:
    if quick:
        return [lambda best: param_grid({"learning_rate": [0.1], "max_leaf_nodes": [15], "max_iter": [10]})]

    def rate(best: dict) -> list[dict]:
        return param_grid({"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [31], "max_iter": [100]})

    def leaves(best: dict) -> list[dict]:
        return [{**best, "max_leaf_nodes": v} for v in (15, 31, 63)]

    def l2(best: dict) -> list[dict]:
        return [{**best, "l2_regularization": v} for v in (0.0, 0.1, 1.0)]

    def iters(best: dict) -> list[dict]:
        return [{**best, "max_iter": v} for v in (100, 200, 300)]

    return [rate, leaves, l2, iters]


@dataclass(frozen=True)
class FamilySpec:
    """One tunable model family (model contract §1 factory + the stages that search it)."""

    name: str  # file stem under results/tuning/
    kind: str  # "regression" | "classification"
    factory: Callable[..., Any]
    stages: Callable[[bool], list[StageGrid]]


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("ridge", "regression", ridge, _ridge_stages),
    FamilySpec("random_forest", "regression", random_forest_regressor, _rf_regressor_stages),
    FamilySpec("hist_gb", "regression", hist_gb_regressor, _hist_gb_regressor_stages),
    FamilySpec("logistic_regression", "classification", logistic_regression, _logistic_regression_stages),
    FamilySpec("random_forest_clf", "classification", random_forest_classifier, _rf_classifier_stages),
    FamilySpec("hist_gb_clf", "classification", hist_gb_classifier, _hist_gb_classifier_stages),
)


# --- search engine -----------------------------------------------------------------------------


def _score(folds: list[PreparedFold], factory: Callable[..., Any], params: dict, kind: str) -> Score:
    scorer = score_regressor if kind == "regression" else score_classifier
    return scorer(folds, factory, params)


def _trial_row(score: Score, stage: int) -> dict:
    m = score.metrics
    row = {**score.params, "stage": stage, "seconds": round(score.seconds, 2)}
    if "mae" in m:  # regression (score_regressor also carries band_* metrics)
        row.update(mae=m["mae"], rmse=m["rmse"], r2=m["r2"], skill=m["skill"], band_macro_f1=m["band_macro_f1"])
    else:  # classification
        row.update(accuracy=m["accuracy"], macro_f1=m["macro_f1"])
    return row


def multi_stage_search(
    folds: list[PreparedFold], factory: Callable[..., Any], kind: str, stages: list[StageGrid]
) -> tuple[pd.DataFrame, Score]:
    """Run each stage's candidates (built from the previous stage's best), keep every `Score`.

    Returns `(trials, best)`: `trials` has one row per candidate ever scored, across every
    stage, sorted best-first; `best` is the winning candidate's full `Score` (so its pooled
    predictions are available with no extra refit -- every candidate is scored exactly once).
    """
    if kind not in {"regression", "classification"}:
        raise ValueError(f"kind must be 'regression' or 'classification', got {kind!r}")
    key = "mae" if kind == "regression" else "macro_f1"

    running_best: dict = {}
    rows: list[dict] = []
    all_scores: list[Score] = []
    for stage_idx, grid_fn in enumerate(stages):
        candidates = grid_fn(running_best)
        stage_scores = [_score(folds, factory, c, kind) for c in candidates]
        rows.extend(_trial_row(s, stage_idx) for s in stage_scores)
        all_scores.extend(stage_scores)
        stage_values = [s.metrics[key] for s in stage_scores]
        stage_best = stage_scores[int(np.argmin(stage_values) if kind == "regression" else np.argmax(stage_values))]
        running_best = dict(stage_best.params)

    trials = pd.DataFrame(rows).sort_values(key, ascending=(kind == "regression"), kind="stable").reset_index(drop=True)
    best = min(all_scores, key=lambda s: s.metrics[key] if kind == "regression" else -s.metrics[key])
    return trials, best


# --- writers -------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON-serialisable: {value!r}")


def _write_family(results_dir: Path, spec: FamilySpec, trials: pd.DataFrame, best: Score, seconds: float) -> None:
    trials.to_csv(results_dir / f"{spec.name}_trials.csv", index=False)
    payload = {
        "family": spec.name,
        "kind": spec.kind,
        "params": best.params,
        "metrics": best.metrics,
        "fit_seconds": round(best.seconds, 2),
        "search_seconds": round(seconds, 2),
    }
    (results_dir / f"{spec.name}_best.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    best.predictions.to_parquet(results_dir / f"{spec.name}_cv_predictions.parquet", index=False)


def _markdown_table(rows: list[dict], columns: Sequence[str], precision: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [f"{row[c]:.{precision}f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_PHASE3_NOTE = (
    "Phase 3 pooled baselines (`results/baselines_summary.md`): MAE -- RF 6.79, Ridge 7.68, "
    "persistence 8.70; band macro-F1 -- persistence 0.678, Ridge->band 0.657, RF->band 0.598, "
    "RF direct 0.578 (Very Unhealthy recall 0), logistic 0.509."
)


def _write_summary(results_dir: Path, outcomes: dict[str, tuple[Score, float]], total_seconds: float) -> Path:
    reg_names = [f.name for f in FAMILIES if f.kind == "regression"]
    clf_names = [f.name for f in FAMILIES if f.kind == "classification"]
    recall_cols = [f"recall_{i}" for i in range(5)]

    reg_rows = []
    for name in reg_names:
        score, _ = outcomes[name]
        m = score.metrics
        reg_rows.append({"family": name, "params": score.params, "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                          "skill": m["skill"], "band_macro_f1": m["band_macro_f1"]})
    reg_rows.sort(key=lambda r: r["mae"])

    clf_rows = []
    for name in clf_names:
        score, _ = outcomes[name]
        m = score.metrics
        row = {"family": name, "params": score.params, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        row.update({f"recall_{i}": m["per_class"][i]["recall"] for i in range(5)})
        clf_rows.append(row)
    clf_rows.sort(key=lambda r: -r["macro_f1"])

    lines = [
        "# Classical model tuning — pooled cross-validation results (Phase 5)",
        "",
        "Pooled = every fold's test predictions concatenated, metrics computed once. `params` is "
        "the winning configuration from a staged search (coarse scan per hyperparameter, refine "
        "around the best), not a full grid.",
        "",
        _PHASE3_NOTE,
        "",
        f"Total tuning runtime: {total_seconds:.1f}s ({total_seconds / 60:.1f} min).",
        "",
        "## Regressors (ranked by pooled MAE)",
        _markdown_table(reg_rows, ["family", "params", "mae", "rmse", "r2", "skill", "band_macro_f1"]),
        "",
        "## Classifiers (ranked by pooled macro-F1)",
        _markdown_table(clf_rows, ["family", "params", "accuracy", "macro_f1", *recall_cols]),
        "",
    ]
    path = results_dir / "classical_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- CLI -------------------------------------------------------------------------------------


def run(features_path: Path | None = None, results_dir: Path | None = None, quick: bool = False) -> dict[str, Score]:
    """Tune every family and write `results/tuning/`. Returns `{family_name: best_score}`."""
    features_path = features_path or PROCESSED_DIR / "features.parquet"
    results_dir = results_dir or (RESULTS_DIR / "tuning")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    folds = prepare_folds(df)

    outcomes: dict[str, tuple[Score, float]] = {}
    start_all = time.perf_counter()
    for spec in FAMILIES:
        t0 = time.perf_counter()
        trials, best = multi_stage_search(folds, spec.factory, spec.kind, spec.stages(quick))
        elapsed = time.perf_counter() - t0
        _write_family(results_dir, spec, trials, best, elapsed)
        outcomes[spec.name] = (best, elapsed)
        key = "mae" if spec.kind == "regression" else "macro_f1"
        print(f"[{spec.name}] best {key}={best.metrics[key]:.4f} params={best.params} ({elapsed:.1f}s)", flush=True)
    total_seconds = time.perf_counter() - start_all

    _write_summary(results_dir, outcomes, total_seconds)
    print(f"Total tuning runtime: {total_seconds:.1f}s ({total_seconds / 60:.1f} min)", flush=True)
    return {name: score for name, (score, _elapsed) in outcomes.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune the classical (scikit-learn) models (Phase 5).")
    parser.add_argument("--features", type=Path, default=None, help="Path to features.parquet")
    parser.add_argument("--quick", action="store_true", help="Tiny grids, for tests")
    args = parser.parse_args(argv)
    run(features_path=args.features, quick=args.quick)


if __name__ == "__main__":
    main()
