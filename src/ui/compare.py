"""Reading and formatting the stored results tables for the Model comparison page (spec §9).

Every number here comes from a file already written by Phase 4-6 (`results/`, `docs/report/`)
— this module only reads and relabels them for a non-technical reader. It never re-runs any
model or SHAP computation (that would be slow and would not match what was actually scored).
"""

from __future__ import annotations

from pathlib import Path

import json

import pandas as pd

from src.config import RESULTS_DIR

# Display order + plain names for the models in `results/final_test/metrics.csv` and
# `classification.csv`. The app predicts with the ensemble — called out explicitly, since a
# non-technical reader otherwise has no way to know which row is "what I'm using".
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "persistence": "Persistence (forecast = the day before)",
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
    "hist_gb": "Gradient Boosting",
    "mlp_numpy": "NumPy neural network",
    "ensemble": "Ensemble — used by the app",
}
MODEL_ORDER: tuple[str, ...] = tuple(MODEL_DISPLAY_NAMES)
CLASSIFIER_DISPLAY_NAMES: dict[str, str] = {"random_forest_clf": "Random Forest classifier"}

PRODUCTION_MODEL_LABEL = "Gradient Boosting + NumPy network (equal-weight average)"

FAIRNESS_CAVEAT = (
    "Per-site results for the model the app uses, from the 8 cross-validation rounds. Errors are "
    "larger at the more polluted sites, mainly because their pollution levels are higher, not "
    "because of missing readings. \"Missed bad days\" is the share of truly Unhealthy-or-worse days "
    "forecast as cleaner. Treat any single site's forecast as less certain than the city-wide "
    "numbers above suggest."
)

NUMPY_VS_PYTORCH_MISSING = (
    "No saved comparison found — run `uv run python -m src.models.torch_nn` to create it."
)


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def final_test_metrics(results_dir: Path = RESULTS_DIR) -> pd.DataFrame | None:
    """Final-test forecast error per model (`results/final_test/metrics.csv`), relabelled.

    Columns: `model` (display name), `mae`, `rmse`, `r2`, `skill` (vs. persistence).
    """
    df = _read_csv(Path(results_dir) / "final_test" / "metrics.csv")
    if df is None:
        return None
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = df[df["model"].isin(order)].copy()
    df["model"] = df["model"].map(MODEL_DISPLAY_NAMES)
    df["_order"] = df["model"].map({v: order[k] for k, v in MODEL_DISPLAY_NAMES.items()})
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df.round({"mae": 2, "rmse": 2, "r2": 3, "skill": 3})


def mae_bar_data(metrics: pd.DataFrame) -> pd.DataFrame:
    """`metrics` reduced to `model`/`mae`, ready for a bar chart (lower bar = better model)."""
    return metrics[["model", "mae"]].reset_index(drop=True)


ROUTE_NAMES = {
    "regress_then_band": "Predict PM2.5, then its band",
    "direct": "Predict the band directly",
}


def band_route_table(results_dir: Path = RESULTS_DIR) -> pd.DataFrame | None:
    """Accuracy and the two band-forecasting error rates per model/route.

    From `results/final_test/classification.csv`: every `regress_then_band` route (predict
    PM2.5, then look up its band — what the app does), plus the best direct band classifier
    for comparison, with plain column names.
    """
    df = _read_csv(Path(results_dir) / "final_test" / "classification.csv")
    if df is None:
        return None
    df = df[(df["route"] == "regress_then_band") | (df["model"] == "random_forest_clf")].copy()
    df["model"] = df["model"].map(lambda m: CLASSIFIER_DISPLAY_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m)))
    df["route"] = df["route"].map(lambda r: ROUTE_NAMES.get(r, r))
    for rate in ("accuracy", "miss_unhealthy", "false_alarm"):
        df[rate] = (df[rate] * 100).round(1)
    df = df.rename(
        columns={
            "model": "Model",
            "route": "Route",
            "accuracy": "Accuracy (%)",
            "miss_unhealthy": "Unhealthy days missed (%)",
            "false_alarm": "False alarms (%)",
        }
    )
    columns = ["Model", "Route", "Accuracy (%)", "Unhealthy days missed (%)", "False alarms (%)"]
    return df[columns].reset_index(drop=True)


def global_shap_importance(results_dir: Path = RESULTS_DIR, top_n: int = 12) -> pd.DataFrame | None:
    """The top `top_n` features by the ensemble's global SHAP importance, with a plain phrase.

    From `results/explain/global_importance.csv`. Falls back to the raw column name if a
    feature has no registered phrase (`src.explain.reasons.FEATURE_PHRASES`) rather than
    failing the whole page.
    """
    from src.explain.reasons import phrase_for

    df = _read_csv(Path(results_dir) / "explain" / "global_importance.csv")
    if df is None:
        return None
    df = df.sort_values("ensemble_mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    df["what_it_means"] = df["feature"].map(lambda f: _safe_phrase(f, phrase_for))
    return df.rename(columns={"ensemble_mean_abs_shap": "mean_abs_shap"})[
        ["feature", "what_it_means", "mean_abs_shap"]
    ]


def _safe_phrase(feature: str, phrase_for) -> str:
    try:
        return phrase_for(feature)
    except KeyError:
        return feature


def fairness_per_sensor(
    results_dir: Path = RESULTS_DIR, model: str = "ensemble"
) -> pd.DataFrame | None:
    """Per-site results for `model` (`results/fairness/<model>/per_sensor.csv`), worst MAE first.

    Keeps site, days, MAE, the site's mean PM2.5 and, when present, the band miss and
    false-alarm rates (as percentages), with plain column names for the comparison page.
    """
    df = _read_csv(Path(results_dir) / "fairness" / model / "per_sensor.csv")
    if df is None:
        return None
    names = {
        "site": "Site",
        "n": "Days",
        "mae": "Average error (µg/m³)",
        "mean_pm25": "Site's average PM2.5 (µg/m³)",
        "miss_rate": "Missed bad days (%)",
        "false_alarm_rate": "False alarms (%)",
    }
    cols = [c for c in names if c in df.columns]
    out = df.sort_values("mae", ascending=False)[cols].reset_index(drop=True)
    for rate in ("miss_rate", "false_alarm_rate"):
        if rate in out.columns:
            out[rate] = (out[rate] * 100).round(0)
    return out.round({"mae": 1, "mean_pm25": 1}).rename(columns=names)


def numpy_vs_pytorch_note(results_dir: Path = RESULTS_DIR) -> str:
    """Plain-English summary of `results/numpy_vs_torch/summary.json` (real-data twin check)."""
    path = Path(results_dir) / "numpy_vs_torch" / "summary.json"
    if not path.exists():
        return NUMPY_VS_PYTORCH_MISSING
    d = json.loads(path.read_text(encoding="utf-8"))
    pred = d["prediction_diff"]
    weight = max(d["weight_max_abs_diff"].values())
    secs = d["fit_seconds"]
    return (
        "The from-scratch NumPy network was trained side by side with an identical PyTorch network: "
        "same starting weights, same data, same mini-batches. On "
        f"{pred['n_rows']} held-out days their forecasts differed by at most {pred['max_abs']:.1e} µg/m³ "
        f"and their final weights by at most {weight:.1e} — floating-point rounding, not a real "
        f"difference. Both stopped at the same best epoch ({d['epochs']['numpy_best_epoch']}). "
        f"Training took {secs['numpy']:.1f} s in NumPy and {secs['torch']:.1f} s in PyTorch on this "
        "machine (timings vary with load). Source: results/numpy_vs_torch/."
    )
