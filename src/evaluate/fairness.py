"""Per-sensor fairness / error-disparity check (spec §10).

Reads a tidy predictions table (`CVResult.predictions` format from `run_cv.py`, or the
narrower `Score.predictions` format from `tune.py` — both carry `sensor_id`, `date`,
`y_true`, `y_pred`) already filtered to one model, and reports whether some areas are
forecast worse than others, and why: a higher error could come from a data gap (a sparse,
hard-to-model sensor) or simply from living somewhere more polluted (bigger numbers, bigger
absolute errors even at the same relative accuracy).

Four building blocks:

- `per_sensor_regression` / `per_sensor_classification` — error and safety metrics per sensor.
- `disparity_summary` — the headline numbers: worst/best sensor, and a >25%-above-overall flag.
- `explain_disparity` — joins the per-sensor error against data-quality factors (hourly
  completeness, training rows, mean PM2.5) and reports a Spearman correlation of error vs
  each factor, plus a pollution-normalised MAE so a wetter/dirtier site isn't penalised just
  for being dirtier.

`SENSOR_COMPLETENESS_PCT`, `SENSOR_MEAN_PM25_FALLBACK` and `SENSOR_TRAINING_ROWS_FALLBACK` are
hand-verified project facts (spec §5.2's sensor table; `docs/report/data_exploration.md`'s
coverage table, 8 Nov 2026). Hourly completeness is a property of the raw hourly data, which
never reaches the processed feature table, so it is always taken from this fallback; training
rows and mean PM2.5 use the real `features.parquet` counts when `--features` is given and fall
back to the same hand-verified numbers otherwise.

CLI: `uv run python -m src.evaluate.fairness --predictions PATH --model NAME [--route ROUTE]
[--features PATH] [--out-dir results/fairness]` writes `<out-dir>/<model>/per_sensor.csv`,
`summary.json`, `fairness_report.md` and `per_sensor_mae.png`.

Without `--predictions`, the model is read from the saved tuning CV predictions instead
(`--tuning-dir`, default `results/tuning/`): `--model ensemble` rebuilds the production ensemble
(mean of `hist_gb` and `mlp_numpy`, `ensemble_cv.load_ensemble_cv`); any other name reads
`<model>_cv_predictions.parquet` (a regressor's file). Per-site band metrics (accuracy,
Unhealthy-or-worse miss rate, false-alarm rate) come from the regress-then-band columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.sensors import SENSORS, sensor_by_id
from src.evaluate.ensemble_cv import TUNING_DIR, cv_predictions_path, load_ensemble_cv
from src.evaluate.metrics import classification_metrics

UNHEALTHY_BAND_INDEX = 3  # merged band index >= 3 is Unhealthy or worse (bands.py, spec §10)
DISPARITY_FLAG_THRESHOLD = 0.25  # a sensor is "flagged" when its MAE exceeds the overall by this much

# --- Hand-verified fallbacks (spec §5.2 sensor table; docs/report/data_exploration.md) --------
# Hourly completeness %, from data_exploration.md's "% complete" column (first/last reading to
# 8 Nov 2026) -- not recoverable from the processed daily feature table, so always used.
SENSOR_COMPLETENESS_PCT: dict[str, float] = {
    "S1": 85.6, "S2": 84.7, "S3": 85.0, "S4": 80.8, "S5": 77.9, "S6": 78.2, "S7": 58.5,
    "S8": 57.7, "S9": 49.3, "S10": 85.4, "S11": 43.7, "S12": 74.8, "S13": 35.9, "S14": 32.5,
}
# Mean PM2.5 (ug/m3), from spec §5.2 -- fallback when --features is not given.
SENSOR_MEAN_PM25_FALLBACK: dict[str, float] = {
    "S1": 43.1, "S2": 32.7, "S3": 49.8, "S4": 35.5, "S5": 48.4, "S6": 38.2, "S7": 52.0,
    "S8": 40.8, "S9": 49.8, "S10": 54.6, "S11": 58.6, "S12": 59.6, "S13": 55.9, "S14": 46.3,
}
# Complete days, from spec §5.2 -- fallback proxy for "training rows" when --features is not given.
SENSOR_TRAINING_ROWS_FALLBACK: dict[str, int] = {
    "S1": 417, "S2": 415, "S3": 413, "S4": 391, "S5": 386, "S6": 368, "S7": 284,
    "S8": 266, "S9": 241, "S10": 228, "S11": 197, "S12": 173, "S13": 165, "S14": 163,
}


def _site_names(sensor_ids: pd.Series) -> pd.Series:
    return sensor_ids.map(lambda sid: sensor_by_id(sid).site)


def per_sensor_regression(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor regression error (spec §10): n, MAE, RMSE, bias, skill vs persistence.

    `pred` needs `sensor_id`, `y_true`, `y_pred`; a `persistence` column (the naive
    "tomorrow = today" forecast for the same rows) is optional -- when present,
    `mae_persistence` and `skill = 1 - mae / mae_persistence` are also reported, else NaN.
    Sorted worst-first (highest MAE) so the least fair sensor is always row 0.
    """
    has_persistence = "persistence" in pred.columns

    def _score(group: pd.DataFrame) -> pd.Series:
        err = group["y_pred"].to_numpy(dtype=np.float64) - group["y_true"].to_numpy(dtype=np.float64)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))
        if has_persistence:
            pers_err = group["persistence"].to_numpy(dtype=np.float64) - group["y_true"].to_numpy(dtype=np.float64)
            mae_persistence = float(np.mean(np.abs(pers_err)))
            skill = 1.0 - mae / mae_persistence if mae_persistence > 0 else float("nan")
        else:
            mae_persistence = float("nan")
            skill = float("nan")
        return pd.Series(
            {"n": len(group), "mae": mae, "rmse": rmse, "bias": bias, "mae_persistence": mae_persistence, "skill": skill}
        )

    rows = [{"sensor_id": sid, **_score(group).to_dict()} for sid, group in pred.groupby("sensor_id")]
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    out["site"] = _site_names(out["sensor_id"])
    columns = ["sensor_id", "site", "n", "mae", "rmse", "bias", "mae_persistence", "skill"]
    return out[columns].sort_values("mae", ascending=False).reset_index(drop=True)


def per_sensor_classification(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor classification safety metrics (spec §10): accuracy, macro-F1, miss rate,
    false-alarm rate.

    `pred` needs `sensor_id`, `y_true`, `y_pred` (merged band indices 0..4, bands.py).
    - **Miss rate**: share of truly unhealthy-or-worse days (`y_true >= 3`) the model forecast
      as healthier (`y_pred < 3`) -- the safety-relevant error. NaN (not an error) at a sensor
      with no unhealthy-or-worse days.
    - **False-alarm rate**: share of truly healthy days (`y_true < 3`) the model forecast as
      unhealthy-or-worse (`y_pred >= 3`). NaN at a sensor with no healthy days.
    Sorted worst-first by miss rate (NaN last) so the least safe sensor is row 0.
    """

    def _score(group: pd.DataFrame) -> pd.Series:
        y_true = group["y_true"].to_numpy(dtype=np.int64)
        y_pred = group["y_pred"].to_numpy(dtype=np.int64)
        metrics = classification_metrics(y_true, y_pred, labels=range(5))

        unhealthy_true = y_true >= UNHEALTHY_BAND_INDEX
        healthy_true = ~unhealthy_true
        miss_rate = float(np.mean(y_pred[unhealthy_true] < UNHEALTHY_BAND_INDEX)) if unhealthy_true.any() else float("nan")
        false_alarm_rate = float(np.mean(y_pred[healthy_true] >= UNHEALTHY_BAND_INDEX)) if healthy_true.any() else float("nan")

        return pd.Series(
            {
                "n": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "miss_rate": miss_rate,
                "false_alarm_rate": false_alarm_rate,
            }
        )

    rows = [{"sensor_id": sid, **_score(group).to_dict()} for sid, group in pred.groupby("sensor_id")]
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    out["site"] = _site_names(out["sensor_id"])
    columns = ["sensor_id", "site", "n", "accuracy", "macro_f1", "miss_rate", "false_alarm_rate"]
    return out[columns].sort_values("miss_rate", ascending=False, na_position="last").reset_index(drop=True)


def disparity_summary(per_sensor: pd.DataFrame) -> dict:
    """Headline MAE disparity numbers from a `per_sensor_regression` table (spec §10).

    `overall_mae` is the n-weighted mean of the per-sensor MAE, which equals the pooled MAE
    (every row's absolute error averaged once) since `sum(mae_i * n_i) == sum(abs_error)`.
    `flagged_sensors` lists every sensor whose MAE exceeds `overall_mae` by more than
    `DISPARITY_FLAG_THRESHOLD` (25%, spec §10).
    """
    overall_mae = float((per_sensor["mae"] * per_sensor["n"]).sum() / per_sensor["n"].sum())
    worst = per_sensor.loc[per_sensor["mae"].idxmax()]
    best = per_sensor.loc[per_sensor["mae"].idxmin()]
    flagged = per_sensor.loc[per_sensor["mae"] > overall_mae * (1 + DISPARITY_FLAG_THRESHOLD)]

    def _sensor_entry(row: pd.Series) -> dict:
        return {"sensor_id": row["sensor_id"], "site": row["site"], "mae": float(row["mae"])}

    return {
        "overall_mae": overall_mae,
        "worst_sensor": _sensor_entry(worst),
        "best_sensor": _sensor_entry(best),
        "max_ratio": float(worst["mae"] / overall_mae),
        "min_ratio": float(best["mae"] / overall_mae),
        "flag_threshold": DISPARITY_FLAG_THRESHOLD,
        "flagged_sensors": [_sensor_entry(row) for _, row in flagged.iterrows()],
    }


def build_coverage_table(features_path: Path | None = None) -> pd.DataFrame:
    """Per-sensor data-quality factors: hourly completeness %, training rows, mean PM2.5.

    Hourly completeness always comes from `SENSOR_COMPLETENESS_PCT` (spec §10's "data gaps"
    factor lives in the raw hourly data, which the processed feature table does not carry).
    Training rows and mean PM2.5 use real counts from `features.parquet` (the number of rows
    with a non-missing `y_pm25` target per sensor, and that target's mean) when `features_path`
    is given, else the same hand-verified spec §5.2 numbers used elsewhere in this module.
    """
    if features_path is not None:
        features = pd.read_parquet(features_path)
        target = features.loc[features["y_pm25"].notna()]
        training_rows = target.groupby("sensor_id")["y_pm25"].count()
        mean_pm25 = target.groupby("sensor_id")["y_pm25"].mean()
    else:
        training_rows = pd.Series(SENSOR_TRAINING_ROWS_FALLBACK)
        mean_pm25 = pd.Series(SENSOR_MEAN_PM25_FALLBACK)

    rows = []
    for sensor in SENSORS:
        sid = sensor.id
        rows.append(
            {
                "sensor_id": sid,
                "site": sensor.site,
                "completeness_pct": SENSOR_COMPLETENESS_PCT.get(sid, float("nan")),
                "training_rows": int(training_rows[sid]) if sid in training_rows.index else SENSOR_TRAINING_ROWS_FALLBACK.get(sid, np.nan),
                "mean_pm25": float(mean_pm25[sid]) if sid in mean_pm25.index else SENSOR_MEAN_PM25_FALLBACK.get(sid, np.nan),
            }
        )
    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation, NaN-safe; NaN when fewer than 2 complete pairs remain."""
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    corr, _ = spearmanr(x[mask], y[mask])
    return float(corr)


def explain_disparity(per_sensor: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Join per-sensor error against data-quality factors, and explain it (spec §10).

    Adds `normalised_mae = mae / mean_pm25` (does a sensor's error look worse only because
    its air is, on average, more polluted?), and three Spearman rank correlations of `mae`
    against `completeness_pct`, `training_rows` and `mean_pm25` -- broadcast as constant
    columns so the function keeps a single `DataFrame` return type. A negative correlation
    with completeness/training rows points at a data-gap explanation; a positive correlation
    with mean PM2.5 points at a pollution-level explanation instead.
    """
    merged = per_sensor.merge(coverage.drop(columns="site"), on="sensor_id", how="left")
    merged["normalised_mae"] = merged["mae"] / merged["mean_pm25"]

    merged["corr_mae_vs_completeness"] = _spearman(merged["mae"], merged["completeness_pct"])
    merged["corr_mae_vs_training_rows"] = _spearman(merged["mae"], merged["training_rows"])
    merged["corr_mae_vs_mean_pm25"] = _spearman(merged["mae"], merged["mean_pm25"])
    return merged


# --- Plot -----------------------------------------------------------------------------------


def plot_per_sensor_mae(per_sensor: pd.DataFrame, overall_mae: float, out_path: Path, model_name: str) -> None:
    """Sorted per-sensor MAE bar chart with the overall (pooled) MAE line, units labelled."""
    ordered = per_sensor.sort_values("mae", ascending=False)
    labels = [f"{row.sensor_id}\n{row.site}" for row in ordered.itertuples()]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(ordered)), 5))
    ax.bar(labels, ordered["mae"], color="#4472c4")
    ax.axhline(overall_mae, color="crimson", linestyle="--", label=f"overall MAE = {overall_mae:.1f} µg/m³")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_xlabel("Sensor (site)")
    ax.set_title(f"Per-sensor forecast error — {model_name}")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Report -----------------------------------------------------------------------------------


def _markdown_table(df: pd.DataFrame, columns: list[str], precision: int = 2) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, float):
                cells.append("NaN" if pd.isna(v) else f"{v:.{precision}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _findings(explained: pd.DataFrame, summary: dict) -> list[str]:
    """4-5 plain-English bullets generated from the numbers (spec §10).

    A 6th bullet -- the unhealthy-day miss rate per area -- is appended separately by the
    caller via `_miss_rate_bullet`, since it needs the classification per-sensor table.
    """
    bullets = []

    worst, best = summary["worst_sensor"], summary["best_sensor"]
    bullets.append(
        f"**{worst['site']} ({worst['sensor_id']})** is forecast worst: MAE {worst['mae']:.1f} µg/m³, "
        f"{(summary['max_ratio'] - 1) * 100:.0f}% above the overall MAE of {summary['overall_mae']:.1f} µg/m³."
    )
    bullets.append(
        f"**{best['site']} ({best['sensor_id']})** is forecast best: MAE {best['mae']:.1f} µg/m³, "
        f"{(1 - summary['min_ratio']) * 100:.0f}% below the overall MAE."
    )

    flagged = summary["flagged_sensors"]
    if flagged:
        names = ", ".join(f"{s['site']} ({s['sensor_id']})" for s in flagged)
        bullets.append(
            f"{len(flagged)} sensor(s) exceed the overall MAE by more than {int(summary['flag_threshold'] * 100)}%: {names}."
        )
    else:
        bullets.append(f"No sensor exceeds the overall MAE by more than {int(summary['flag_threshold'] * 100)}%; error is fairly even across sites.")

    corr_gap = explained["corr_mae_vs_completeness"].iloc[0]
    corr_pollution = explained["corr_mae_vs_mean_pm25"].iloc[0]
    if pd.notna(corr_gap) and pd.notna(corr_pollution) and (abs(corr_gap) > 1e-9 or abs(corr_pollution) > 1e-9):
        if abs(corr_pollution) >= abs(corr_gap):
            bullets.append(
                f"Error tracks pollution level more than data gaps: error vs mean PM2.5 rho={corr_pollution:.2f} "
                f"vs error vs completeness rho={corr_gap:.2f} -- dirtier sites simply have bigger numbers to miss by."
            )
        else:
            bullets.append(
                f"Error tracks data gaps more than pollution level: error vs completeness rho={corr_gap:.2f} "
                f"vs error vs mean PM2.5 rho={corr_pollution:.2f} -- sparser sensors are harder to model, not just dirtier."
            )
    else:
        bullets.append("Not enough spread across sensors to separate a data-gap explanation from a pollution-level one.")

    worst_norm = explained.sort_values("normalised_mae", ascending=False).iloc[0]
    bullets.append(
        f"After adjusting for each site's own pollution level (normalised MAE = MAE / mean PM2.5), "
        f"**{worst_norm['site']} ({worst_norm['sensor_id']})** is relatively worst "
        f"(normalised MAE {worst_norm['normalised_mae']:.3f})."
    )

    return bullets


def _miss_rate_bullet(per_sensor_clf: pd.DataFrame) -> str:
    valid = per_sensor_clf.dropna(subset=["miss_rate"])
    if valid.empty:
        return "No sensor had any unhealthy-or-worse day in this window, so a miss rate could not be computed."
    worst = valid.loc[valid["miss_rate"].idxmax()]
    return (
        f"**Unhealthy-day miss rate** (true band Unhealthy-or-worse, forecast healthier) is highest at "
        f"**{worst['site']} ({worst['sensor_id']})**: {worst['miss_rate'] * 100:.0f}% of its unhealthy-or-worse days "
        f"were forecast as safer than they were."
    )


def write_report(
    explained: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    model_name: str,
    route: str | None,
    per_sensor_clf: pd.DataFrame | None,
) -> None:
    """Write `per_sensor.csv`, `summary.json`, `fairness_report.md` and `per_sensor_mae.png`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    full = explained if per_sensor_clf is None else explained.merge(
        per_sensor_clf[["sensor_id", "accuracy", "macro_f1", "miss_rate", "false_alarm_rate"]], on="sensor_id", how="left"
    )
    full.to_csv(out_dir / "per_sensor.csv", index=False)

    summary_out: dict[str, Any] = {"model": model_name, "route": route, **summary}
    if per_sensor_clf is not None:
        summary_out["classification"] = {
            "pooled_accuracy": float(np.average(per_sensor_clf["accuracy"], weights=per_sensor_clf["n"])),
            "worst_miss_rate_sensor": (
                per_sensor_clf.dropna(subset=["miss_rate"]).sort_values("miss_rate", ascending=False).iloc[0][
                    ["sensor_id", "site", "miss_rate"]
                ].to_dict()
                if per_sensor_clf["miss_rate"].notna().any()
                else None
            ),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2, default=float), encoding="utf-8")

    bullets = _findings(explained, summary)
    if per_sensor_clf is not None:
        bullets.append(_miss_rate_bullet(per_sensor_clf))

    reg_cols = ["sensor_id", "site", "n", "mae", "rmse", "bias", "skill", "completeness_pct", "training_rows", "mean_pm25", "normalised_mae"]
    lines = [
        f"# Fairness report — {model_name}" + (f" ({route})" if route else ""),
        "",
        "Per-sensor forecast error (spec §10): checks whether some areas (e.g. Orangi vs AKU) are",
        "forecast worse than others, and whether that is explained by data gaps or by higher pollution.",
        "",
        "## Findings",
        "",
        *[f"- {b}" for b in bullets],
        "",
        "## Per-sensor regression error and data-quality factors",
        "",
        _markdown_table(explained.sort_values("mae", ascending=False), reg_cols),
        "",
    ]
    if per_sensor_clf is not None:
        clf_cols = ["sensor_id", "site", "n", "accuracy", "macro_f1", "miss_rate", "false_alarm_rate"]
        lines += [
            "## Per-sensor classification safety (regress-then-band)",
            "",
            _markdown_table(per_sensor_clf.sort_values("miss_rate", ascending=False, na_position="last"), clf_cols),
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- `skill` = 1 - MAE_model / MAE_persistence (0 = no better than the naive forecast).",
        "- `normalised_mae` = MAE / mean PM2.5, so a wetter/dirtier site is not penalised purely for being dirtier.",
        "- `miss_rate` is NaN at a sensor with no Unhealthy-or-worse day in this window (not an error).",
        "",
    ]
    (out_dir / "fairness_report.md").write_text("\n".join(lines), encoding="utf-8")

    plot_per_sensor_mae(explained, summary["overall_mae"], out_dir / "per_sensor_mae.png", model_name)


# --- CLI ----------------------------------------------------------------------------------


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _filter(predictions: pd.DataFrame, model: str, route: str | None) -> tuple[pd.DataFrame, str]:
    filtered = predictions.loc[predictions["model"] == model]
    if filtered.empty:
        raise ValueError(f"no rows for model={model!r}")
    if route is not None:
        filtered = filtered.loc[filtered["route"] == route]
        if filtered.empty:
            raise ValueError(f"no rows for model={model!r} route={route!r}")

    if "task" in filtered.columns:
        tasks = filtered["task"].unique()
    else:
        # Score.predictions (tune.py) has no `task` column: a regressor's table carries
        # `persistence`/`band_true`/`band_pred`; a classifier's table is already band labels.
        tasks = ["regression" if "persistence" in filtered.columns else "classification"]
    if len(tasks) != 1:
        raise ValueError(
            f"ambiguous task for model={model!r} route={route!r} (found {sorted(tasks)}); pass --route to disambiguate"
        )
    return filtered, tasks[0]


def run(predictions_path: Path, model: str, route: str | None, features_path: Path | None, out_dir: Path) -> Path:
    """Filter `predictions_path` to one model (+route), build the fairness report, write it."""
    predictions = _load_predictions(predictions_path)
    filtered, task = _filter(predictions, model, route)
    model_dir = out_dir / model

    if task == "regression":
        if model == "persistence":
            merged = filtered.assign(persistence=filtered["y_pred"])
        else:
            persistence_rows = predictions.loc[(predictions["model"] == "persistence") & (predictions["task"] == "regression")]
            merged = filtered.merge(
                persistence_rows[["fold", "sensor_id", "date", "y_pred"]].rename(columns={"y_pred": "persistence"}),
                on=["fold", "sensor_id", "date"],
                how="left",
            )

        per_sensor = per_sensor_regression(merged)
        summary = disparity_summary(per_sensor)
        coverage = build_coverage_table(features_path)
        explained = explain_disparity(per_sensor, coverage)

        clf_rows = predictions.loc[
            (predictions["model"] == model) & (predictions["task"] == "classification") & (predictions["route"] == "regress_then_band")
        ]
        per_sensor_clf = per_sensor_classification(clf_rows) if not clf_rows.empty else None

        write_report(explained, summary, model_dir, model, route, per_sensor_clf)
    else:
        per_sensor_clf = per_sensor_classification(filtered)
        model_dir.mkdir(parents=True, exist_ok=True)
        per_sensor_clf.to_csv(model_dir / "per_sensor.csv", index=False)
        pooled_accuracy = float(np.average(per_sensor_clf["accuracy"], weights=per_sensor_clf["n"]))
        (model_dir / "summary.json").write_text(
            json.dumps({"model": model, "route": route, "pooled_accuracy": pooled_accuracy}, indent=2), encoding="utf-8"
        )
        bullet = _miss_rate_bullet(per_sensor_clf)
        clf_cols = ["sensor_id", "site", "n", "accuracy", "macro_f1", "miss_rate", "false_alarm_rate"]
        report = "\n".join(
            [
                f"# Fairness report — {model}" + (f" ({route})" if route else ""),
                "",
                "## Findings",
                "",
                f"- Pooled accuracy: {pooled_accuracy:.3f}.",
                f"- {bullet}",
                "",
                "## Per-sensor classification safety",
                "",
                _markdown_table(per_sensor_clf.sort_values("miss_rate", ascending=False, na_position="last"), clf_cols),
                "",
            ]
        )
        (model_dir / "fairness_report.md").write_text(report, encoding="utf-8")
        plot_per_sensor_mae(
            per_sensor_clf.assign(mae=per_sensor_clf["miss_rate"].fillna(0.0)),
            float(np.average(per_sensor_clf["miss_rate"].fillna(0.0), weights=per_sensor_clf["n"])),
            model_dir / "per_sensor_mae.png",
            model,
        )

    return model_dir


def load_tuning_regressor(tuning_dir: Path, model: str) -> pd.DataFrame:
    """A regressor's pooled CV table from the saved tuning files (`ensemble` = GB + network mean)."""
    if model == "ensemble":
        return load_ensemble_cv(tuning_dir)
    table = pd.read_parquet(cv_predictions_path(tuning_dir, model))
    missing = {"persistence", "band_true", "band_pred"} - set(table.columns)
    if missing:
        raise ValueError(f"{model!r} is not a regressor CV table (missing {sorted(missing)})")
    return table


def run_from_tuning(tuning_dir: Path, model: str, features_path: Path | None, out_dir: Path) -> Path:
    """Fairness report for a regressor scored on the pooled CV rows (tune.py format).

    Writes the same files as `run` does for a regressor: per-sensor MAE and skill, plus per-site
    band accuracy, Unhealthy-or-worse miss rate and false-alarm rate (regress-then-band).
    """
    table = load_tuning_regressor(tuning_dir, model)
    per_sensor = per_sensor_regression(table)
    summary = disparity_summary(per_sensor)
    explained = explain_disparity(per_sensor, build_coverage_table(features_path))
    bands = table[["sensor_id"]].assign(y_true=table["band_true"].to_numpy(), y_pred=table["band_pred"].to_numpy())
    per_sensor_clf = per_sensor_classification(bands)
    model_dir = out_dir / model
    write_report(explained, summary, model_dir, model, "regress_then_band", per_sensor_clf)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-sensor fairness check (spec §10)")
    parser.add_argument("--predictions", default=None, type=Path)
    parser.add_argument("--tuning-dir", default=TUNING_DIR, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--route", default=None)
    parser.add_argument("--features", default=None, type=Path)
    parser.add_argument("--out-dir", default=Path("results/fairness"), type=Path)
    args = parser.parse_args()

    if args.predictions is None:
        out_path = run_from_tuning(args.tuning_dir, args.model, args.features, args.out_dir)
    else:
        out_path = run(args.predictions, args.model, args.route, args.features, args.out_dir)
    print(f"Wrote fairness report to {out_path}")


if __name__ == "__main__":
    main()
