"""Band-classification evidence from SAVED predictions (no retraining, no new final-test run).

    uv run python -m src.evaluate.band_report [--n-boot 2000] [--out-dir results/bands]

Two row sets, always named in every output:

- **Pooled CV** -- the 8 rolling-origin test blocks (1 Jan - 28 Aug 2026), one row per labelled
  sensor-day: 1,458 rows over 240 dates, read from `results/tuning/*_cv_predictions.parquet`.
  The ensemble's CV prediction is the mean of the saved `hist_gb` and `mlp_numpy` predictions
  (`ensemble_cv.load_ensemble_cv`), then banded.
- **Final test** -- 1 Sep - 8 Nov 2026, 442 sensor-days over 69 dates, read from
  `results/final_test/predictions.parquet` (written once by `final_test.py`; only read here).

Models compared: the production ensemble (regress-then-band), persistence -> band (the naive
yardstick), the best direct classifier by the selection rule (`band_stats.rank_by_selection_rule`)
and the official baselines (Ridge -> band for regression, logistic regression for direct
classification; logistic regression was not run on the final test, so it appears in CV only).

Writes `results/bands/`: `metrics.csv`, `per_class.csv`, `bootstrap_ci.csv`,
`confusion_counts.csv`, `confusion_recall.csv`, `confusion_cv.png`, `confusion_final_test.png`
and `summary.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEED
from src.evaluate import band_stats as bs
from src.evaluate.ensemble_cv import KEY_COLUMNS, TUNING_DIR, cv_predictions_path, load_ensemble_cv
from src.evaluate.final_test import BAND_NAMES
from src.evaluate.run_cv import _to_band

OUT_DIR = RESULTS_DIR / "bands"
FINAL_PREDICTIONS = RESULTS_DIR / "final_test" / "predictions.parquet"
DIRECT_CLASSIFIERS: tuple[str, ...] = ("hist_gb_clf", "random_forest_clf", "logistic_regression")
OFFICIAL_BASELINES: tuple[str, ...] = ("ridge", "logistic_regression")  # regression / direct classification
# Single regressors whose CV MAE is within ~10% of the ensemble's — rule 0 in
# docs/report/model_selection.md lets their banded routes compete, so they are compared too.
ELIGIBLE_REGRESSORS: tuple[str, ...] = ("hist_gb", "random_forest", "mlp_numpy")
DISPLAY = {
    "ensemble": "Ensemble -> band (production)",
    "persistence": "Persistence -> band",
    "ridge": "Ridge -> band (tuned regression baseline)",
    "hist_gb": "Gradient Boosting -> band",
    "random_forest": "Random Forest -> band",
    "mlp_numpy": "NumPy network -> band",
    "hist_gb_clf": "Gradient Boosting classifier (direct)",
    "random_forest_clf": "Random Forest classifier (direct)",
    "logistic_regression": "Logistic regression (direct, baseline)",
}
SET_TITLES = {
    "cv": "Pooled CV (8 folds, 1 Jan - 28 Aug 2026)",
    "final_test": "Final test (1 Sep - 8 Nov 2026)",
}


# --- Loading -------------------------------------------------------------------------------


def load_cv_bands(tuning_dir: Path = TUNING_DIR) -> pd.DataFrame:
    """Pooled-CV band table: keys, `band_true`, and one `band_<model>` column per model.

    Every classifier file must cover exactly the ensemble's rows with the same true band;
    a mismatch raises `ValueError` rather than silently scoring different row sets.
    """
    ens = load_ensemble_cv(tuning_dir)
    table = ens[[*KEY_COLUMNS, "band_true"]].assign(
        band_ensemble=ens["band_pred"].to_numpy(np.int64),
        band_persistence=_to_band(ens["persistence"].to_numpy()),
    )
    for name in ("ridge", *ELIGIBLE_REGRESSORS):
        reg = pd.read_parquet(cv_predictions_path(tuning_dir, name))
        table = table.merge(
            reg[[*KEY_COLUMNS, "band_pred"]].rename(columns={"band_pred": f"band_{name}"}), on=KEY_COLUMNS, how="left"
        )
        if table[f"band_{name}"].isna().any():
            raise ValueError(f"{name} CV predictions do not match the ensemble's rows")
    for name in DIRECT_CLASSIFIERS:
        clf = pd.read_parquet(cv_predictions_path(tuning_dir, name)).rename(
            columns={"y_true": f"true_{name}", "y_pred": f"band_{name}"}
        )
        table = table.merge(clf[[*KEY_COLUMNS, f"true_{name}", f"band_{name}"]], on=KEY_COLUMNS, how="left")
        if table[f"band_{name}"].isna().any() or not (table[f"true_{name}"] == table["band_true"]).all():
            raise ValueError(f"{name} CV predictions do not match the ensemble's rows")
        table = table.drop(columns=f"true_{name}")
    band_cols = [c for c in table.columns if c.startswith("band_")]
    return table.astype({c: np.int64 for c in band_cols})


def load_final_bands(path: Path = FINAL_PREDICTIONS) -> pd.DataFrame:
    """Final-test band table (read-only use of the saved one-time predictions)."""
    pred = pd.read_parquet(path)
    keep = ["sensor_id", "date", "band_true", "band_ensemble", "band_persistence", "band_ridge", "band_random_forest_clf"]
    keep += [f"band_{m}" for m in ELIGIBLE_REGRESSORS if f"band_{m}" in pred.columns]
    return pred[keep].copy()


def available_models(table: pd.DataFrame) -> list[str]:
    """Model names with a `band_<model>` column in `table`, in `DISPLAY` order."""
    return [m for m in DISPLAY if f"band_{m}" in table.columns]


# --- Tables --------------------------------------------------------------------------------


def metrics_table(table: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """One row per model: headline band metrics on this row set."""
    present = bs.present_classes(table["band_true"])
    rows = []
    for model in available_models(table):
        m = bs.band_metrics(table["band_true"], table[f"band_{model}"], present=present)
        rows.append(
            {
                "set": set_name,
                "model": model,
                "n": m["n"],
                "n_dates": int(table["date"].nunique()),
                "accuracy": m["accuracy"],
                "macro_f1_all": m["macro_f1_all"],
                "macro_f1_present": m["macro_f1_present"],
                "n_present_classes": m["n_present_classes"],
                "miss_rate": m["miss_rate"],
                "false_alarm_rate": m["false_alarm_rate"],
            }
        )
    return pd.DataFrame(rows)


def best_direct(metrics: pd.DataFrame) -> str:
    """Best direct classifier among those scored, by the selection rule."""
    direct = metrics.loc[metrics["model"].isin(DIRECT_CLASSIFIERS)]
    if direct.empty:
        raise ValueError("no direct classifier in this row set")
    return str(bs.rank_by_selection_rule(direct).iloc[0]["model"])


def per_class_long(table: pd.DataFrame, set_name: str, models: list[str]) -> pd.DataFrame:
    frames = [bs.per_class_table(table["band_true"], table[f"band_{m}"]).assign(set=set_name, model=m) for m in models]
    out = pd.concat(frames, ignore_index=True)
    return out[["set", "model", "band", "band_name", "precision", "recall", "f1", "support"]]


def confusion_long(table: pd.DataFrame, set_name: str, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts, recalls = [], []
    for m in models:
        c, r = bs.confusion_frames(table["band_true"], table[f"band_{m}"])
        counts.append(c.reset_index().assign(set=set_name, model=m))
        recalls.append(r.reset_index().assign(set=set_name, model=m))
    cols = ["set", "model", "true", *BAND_NAMES]
    return pd.concat(counts, ignore_index=True)[cols], pd.concat(recalls, ignore_index=True)[cols]


# --- Figure --------------------------------------------------------------------------------


def plot_confusions(table: pd.DataFrame, models: list[str], title: str, out_path: Path) -> None:
    """One heat-map panel per model: colour = row-normalised recall, text = count and recall %."""
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.6 * len(models) + 1.2, 5.6), squeeze=False, layout="constrained"
    )
    short = ["Good", "Moderate", "USG", "Unhealthy", "Very\nUnhealthy"]
    image = None
    for k, (ax, model) in enumerate(zip(axes[0], models)):
        counts, recall = bs.confusion_frames(table["band_true"], table[f"band_{model}"])
        shown = np.nan_to_num(recall.to_numpy(), nan=0.0)
        image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(BAND_NAMES)):
            for j in range(len(BAND_NAMES)):
                pct = recall.iat[i, j]
                text = "no\ndays" if np.isnan(pct) else f"{int(counts.iat[i, j])}\n{pct:.0%}"
                colour = "white" if shown[i, j] > 0.6 else ("grey" if np.isnan(pct) else "black")
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=colour)
        ax.set_xticks(range(len(short)), short, fontsize=9)
        ax.set_yticks(range(len(short)), short, fontsize=9)
        ax.set_xlabel("Forecast band")
        if k == 0:
            ax.set_ylabel("True band")
        ax.set_title(DISPLAY[model], fontsize=11)
    fig.suptitle(f"{title}: {len(table):,} sensor-days. Each cell: count, and share of that true band", fontsize=12)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="Share of the true band (row recall)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Report --------------------------------------------------------------------------------


def _md(df: pd.DataFrame, precision: int = 3) -> str:
    def cell(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return "NaN" if pd.isna(v) else f"{v:.{precision}f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _ci_text(ci: pd.DataFrame, model: str, metric: str) -> str:
    row = ci.loc[(ci["model"] == model) & (ci["metric"] == metric)].iloc[0]
    return f"{row['point']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def _ci_block(ci: pd.DataFrame) -> str:
    metrics = list(bs.CI_METRICS)
    rows = [{"model": m, **{k: _ci_text(ci, m, k) for k in metrics}} for m in ci["model"].unique()]
    return _md(pd.DataFrame(rows))


def write_summary(results: dict, out_dir: Path, n_boot: int) -> None:
    metrics, per_class, ci = results["metrics"], results["per_class"], results["ci"]
    lines = [
        "# Band classification evidence",
        "",
        "Built by `python -m src.evaluate.band_report` from saved predictions only (no retraining;",
        "the final test was not re-run). The forecast is the 24-hour average PM2.5 for the day,",
        "issued at 00:00 PKT, sorted into 5 bands (Hazardous merged into Very Unhealthy).",
        "",
        "Row sets:",
        "",
    ]
    for set_name, title in SET_TITLES.items():
        sub = metrics.loc[metrics["set"] == set_name].iloc[0]
        lines.append(f"- **{title}**: {int(sub['n']):,} sensor-days over {int(sub['n_dates'])} dates.")
    lines += [
        "",
        "Selection rule (best first): fewest missed Unhealthy-or-worse days, then accuracy, then macro-F1.",
        f"Best direct classifier: CV = `{results['best']['cv']}`; final test = `{results['best']['final_test']}`",
        "(the only direct classifier scored on the final test).",
        "",
        "`macro_f1_all` averages F1 over all 5 bands; `macro_f1_present` only over bands that occur in that",
        "row set's true labels. `miss_rate` = share of truly Unhealthy-or-worse days forecast as a cleaner band;",
        "`false_alarm_rate` = share of cleaner days forecast as Unhealthy-or-worse.",
        "",
    ]
    for set_name, title in SET_TITLES.items():
        sub = metrics.loc[metrics["set"] == set_name].drop(columns="set")
        pc = per_class.loc[per_class["set"] == set_name].drop(columns=["set", "band"])
        lines += [
            f"## {title}",
            "",
            "### Headline metrics",
            "",
            _md(sub),
            "",
            f"### Bootstrap 95% CIs ({n_boot} date-block resamples, seed {results['seed']})",
            "",
            "Point estimate [2.5%, 97.5%]. A difference whose interval contains 0 is within noise.",
            "",
            _ci_block(ci.loc[ci["set"] == set_name]),
            "",
            "### Per-class precision / recall / F1",
            "",
            _md(pc),
            "",
            f"![Confusion matrices]({'confusion_cv.png' if set_name == 'cv' else 'confusion_final_test.png'})",
            "",
        ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(
    cv: pd.DataFrame, final: pd.DataFrame, out_dir: Path = OUT_DIR, n_boot: int = 1000, seed: int = SEED
) -> dict:
    """Compute every table, CI and figure for both row sets and write them to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {"cv": cv, "final_test": final}
    metrics = pd.concat([metrics_table(t, s) for s, t in sets.items()], ignore_index=True)
    best = {s: best_direct(metrics.loc[metrics["set"] == s]) for s in sets}

    per_class, counts, recalls, cis = [], [], [], []
    for set_name, table in sets.items():
        focus = ["ensemble", "persistence", best[set_name]]
        per_class.append(per_class_long(table, set_name, available_models(table)))
        c, r = confusion_long(table, set_name, focus)
        counts.append(c)
        recalls.append(r)
        baselines = [m for m in OFFICIAL_BASELINES if m in available_models(table) and m not in focus]
        eligible = [m for m in ELIGIBLE_REGRESSORS if m in available_models(table)]
        ci = bs.date_block_bootstrap(
            table,
            {m: f"band_{m}" for m in [*focus, *baselines, *eligible]},
            pairs=[("ensemble", m) for m in ["persistence", best[set_name], *baselines, *eligible]],
            n_boot=n_boot,
            seed=seed,
        )
        cis.append(ci.assign(set=set_name))
        plot_confusions(table, focus, SET_TITLES[set_name], out_dir / f"confusion_{set_name}.png")

    results = {
        "metrics": metrics,
        "per_class": pd.concat(per_class, ignore_index=True),
        "confusion_counts": pd.concat(counts, ignore_index=True),
        "confusion_recall": pd.concat(recalls, ignore_index=True),
        "ci": pd.concat(cis, ignore_index=True)[["set", "model", "metric", "point", "ci_low", "ci_high", "n_boot", "n_dates"]],
        "best": best,
        "seed": seed,
    }
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    results["per_class"].to_csv(out_dir / "per_class.csv", index=False)
    results["confusion_counts"].to_csv(out_dir / "confusion_counts.csv", index=False)
    results["confusion_recall"].to_csv(out_dir / "confusion_recall.csv", index=False)
    results["ci"].to_csv(out_dir / "bootstrap_ci.csv", index=False)
    write_summary(results, out_dir, n_boot)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Band-classification evidence from saved predictions")
    parser.add_argument("--tuning-dir", default=TUNING_DIR, type=Path)
    parser.add_argument("--final-predictions", default=FINAL_PREDICTIONS, type=Path)
    parser.add_argument("--out-dir", default=OUT_DIR, type=Path)
    parser.add_argument("--n-boot", default=2000, type=int)
    args = parser.parse_args(argv)

    build_report(load_cv_bands(args.tuning_dir), load_final_bands(args.final_predictions), args.out_dir, args.n_boot)
    print(f"Wrote band report to {args.out_dir}")


if __name__ == "__main__":
    main()
