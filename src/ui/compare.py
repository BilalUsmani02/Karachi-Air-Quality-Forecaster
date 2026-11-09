"""Reading and formatting the stored results tables for the Model comparison page (spec §9).

Every number here comes from a file already written by Phase 4-6 (`results/`, `docs/report/`)
— this module only reads and relabels them for a non-technical reader. It never re-runs any
model or SHAP computation (that would be slow and would not match what was actually scored).
"""

from __future__ import annotations

from pathlib import Path

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

PRODUCTION_MODEL_LABEL = "Gradient Boosting + NumPy network (equal-weight average)"

NUMPY_VS_PYTORCH_NOTE = (
    "The from-scratch NumPy network was checked against an identical PyTorch network trained "
    "on the same data from the same starting weights. Their forecasts matched to within "
    "1e-13 µg/m³ and their final weights to within 6e-16 — floating-point noise, not a real "
    "difference. The NumPy version trained in about 4 seconds; PyTorch took about 38 seconds "
    "for the same result (ROADMAP.md, Phase 4 notes, 8 Nov 2026)."
)

FAIRNESS_CAVEAT = (
    "Checked with the Random Forest model during development. Errors are larger at the more "
    "polluted sites (such as Orangi), mainly because their pollution levels are higher, not "
    "because of missing readings. Treat any single site's forecast as less certain than the "
    "city-wide numbers above suggest."
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
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def mae_bar_data(metrics: pd.DataFrame) -> pd.DataFrame:
    """`metrics` reduced to `model`/`mae`, ready for a bar chart (lower bar = better model)."""
    return metrics[["model", "mae"]].reset_index(drop=True)


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
    df["model"] = df["model"].map(lambda m: MODEL_DISPLAY_NAMES.get(m, m))
    df = df.rename(
        columns={
            "model": "Model",
            "route": "Route",
            "accuracy": "Accuracy",
            "miss_unhealthy": "Unhealthy days missed",
            "false_alarm": "False alarms",
        }
    )
    return df[["Model", "Route", "Accuracy", "Unhealthy days missed", "False alarms"]].reset_index(drop=True)


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
    results_dir: Path = RESULTS_DIR, model: str = "random_forest"
) -> pd.DataFrame | None:
    """Per-sensor forecast error for `model` (`results/fairness/<model>/per_sensor.csv`).

    Columns kept: site, mae, mean_pm25, normalised_mae (MAE relative to that site's own
    pollution level) — sorted worst (highest MAE) first.
    """
    df = _read_csv(Path(results_dir) / "fairness" / model / "per_sensor.csv")
    if df is None:
        return None
    cols = ["site", "mae", "mean_pm25", "normalised_mae"]
    return df[cols].sort_values("mae", ascending=False).reset_index(drop=True)
