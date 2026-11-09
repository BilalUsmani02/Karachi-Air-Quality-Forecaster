"""Phase 5 final test: train the tuned models on all development rows, score the final period once.

    uv run python -m src.evaluate.final_test            # refuses to run twice
    uv run python -m src.evaluate.final_test --force    # only if something upstream changed

Development rows = everything before `split.FINAL_TEST_START` (2026-09-01); final test = from
then on. Every setting was chosen on the CV folds before this runs. A marker file records the
run so the test period is not quietly reused for further tuning.

Writes `results/final_test/`: `metrics.csv`, `classification.csv`, `predictions.parquet`,
`summary.md`, `run.json`. Then, separately, `--save-production` refits the chosen ensemble on
**all** rows (development + final test) for the app and saves it to `models/`.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.open_meteo import utc_now_iso
from src.data.split import FINAL_TEST_START, project_splits
from src.evaluate.metrics import classification_metrics, regression_metrics
from src.evaluate.run_cv import _to_band
from src.evaluate.tune import PERSISTENCE_COLUMN
from src.models import final
from src.models.dataset import Preprocessor, make_xy

OUT_DIR = RESULTS_DIR / "final_test"
PRODUCTION_PATH = MODELS_DIR / "final_ensemble.pkl"
BAND_NAMES = ("Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy")


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "sensor_id"], kind="stable").reset_index(drop=True)


def _safety_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Share of truly Unhealthy-or-worse days predicted below it, and of clean days flagged."""
    bad, clean = y_true >= 3, y_true < 3
    return {
        "miss_unhealthy": float((bad & (y_pred < 3)).sum() / bad.sum()) if bad.any() else float("nan"),
        "false_alarm": float((clean & (y_pred >= 3)).sum() / clean.sum()) if clean.any() else float("nan"),
    }


def _class_row(model: str, route: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = classification_metrics(y_true, y_pred)
    row = {"model": model, "route": route, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    row.update({f"recall_{BAND_NAMES[i]}": m["per_class"][i]["recall"] for i in range(5)})
    row.update({f"support_{BAND_NAMES[i]}": m["per_class"][i]["support"] for i in range(5)})
    return {**row, **_safety_rates(y_true, y_pred)}


def evaluate(
    df: pd.DataFrame,
    regressors: dict[str, Callable[[], Any]] | None = None,
    classifiers: dict[str, Callable[[], Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on development rows, predict the final test once. Returns (regression, classification, predictions)."""
    regressors = final.TUNED_REGRESSORS if regressors is None else regressors
    classifiers = final.TUNED_CLASSIFIERS if classifiers is None else classifiers
    _folds, dev, test = project_splits(_sorted(df))
    dev, test = dev.loc[dev["y_pm25"].notna()], test.loc[test["y_pm25"].notna()]

    X_dev_df, y_dev = make_xy(dev, "y_pm25")
    X_test_df, y_test = make_xy(test, "y_pm25")
    _, yb_dev = make_xy(dev, "y_band")
    _, yb_test = make_xy(test, "y_band")
    pre = Preprocessor()
    X_dev, X_test = pre.fit_transform(X_dev_df), pre.transform(X_test_df)
    y_true, yb_true = y_test.to_numpy(np.float64), yb_test.to_numpy(np.int64)
    persistence = X_test_df[PERSISTENCE_COLUMN].to_numpy(np.float64)

    preds = {"persistence": persistence}
    for name, factory in regressors.items():
        preds[name] = np.asarray(factory().fit(X_dev, y_dev.to_numpy(np.float64)).predict(X_test), np.float64)

    reg_rows, class_rows = [], []
    table = test[["sensor_id", "date"]].reset_index(drop=True).assign(y_true=y_true, band_true=yb_true)
    for name, y_pred in preds.items():
        reg_rows.append({"model": name, **regression_metrics(y_true, y_pred, persistence)})
        band = _to_band(y_pred)
        class_rows.append(_class_row(name, "regress_then_band", yb_true, band))
        table[f"pred_{name}"], table[f"band_{name}"] = y_pred, band
    for name, factory in classifiers.items():
        band = np.asarray(factory().fit(X_dev, yb_dev.to_numpy(np.int64)).predict(X_test), np.int64)
        class_rows.append(_class_row(name, "direct", yb_true, band))
        table[f"band_{name}"] = band
    return pd.DataFrame(reg_rows), pd.DataFrame(class_rows), table


def _markdown(df: pd.DataFrame, precision: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.{precision}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results(regression: pd.DataFrame, classification: pd.DataFrame, table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    regression.to_csv(out_dir / "metrics.csv", index=False)
    classification.to_csv(out_dir / "classification.csv", index=False)
    table.to_parquet(out_dir / "predictions.parquet", index=False)
    span = f"{pd.Timestamp(table['date'].min()).date()} to {pd.Timestamp(table['date'].max()).date()}"
    key_cols = ["model", "route", "accuracy", "macro_f1", "miss_unhealthy", "false_alarm"]
    summary = [
        "# Final test (scored once)",
        "",
        f"Period: {span} · {len(table)} sensor-days · trained on all rows before {FINAL_TEST_START}.",
        "",
        "## Forecast error (µg/m³)",
        "",
        _markdown(regression[["model", "mae", "rmse", "r2", "skill"]]),
        "",
        "## Bands",
        "",
        _markdown(classification[key_cols]),
        "",
        "`miss_unhealthy`: share of truly Unhealthy-or-worse days forecast as a cleaner band. "
        "`false_alarm`: share of cleaner days forecast as Unhealthy-or-worse.",
        "",
        "Band support in this period: "
        + ", ".join(f"{BAND_NAMES[i]} {int(classification.iloc[0][f'support_{BAND_NAMES[i]}'])}" for i in range(5))
        + ".",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out_dir / "run.json").write_text(json.dumps({"run_at_utc": utc_now_iso(), "period": span}, indent=2))


def save_production(df: pd.DataFrame, path: Path = PRODUCTION_PATH) -> Path:
    """Refit the preprocessor and the final ensemble on every labelled row and pickle them together."""
    data = _sorted(df)
    data = data.loc[data["y_pm25"].notna()]
    X_df, y = make_xy(data, "y_pm25")
    pre = Preprocessor()
    model = final.final_ensemble().fit(pre.fit_transform(X_df), y.to_numpy(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"preprocessor": pre, "model": model, "trained_on": [str(data["date"].min()), str(data["date"].max())]}
    path.write_bytes(pickle.dumps(bundle))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the final test period once.")
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "features.parquet")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="rerun even though a run is recorded")
    parser.add_argument("--save-production", action="store_true", help="refit the ensemble on all rows and save it")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.features)
    if args.save_production:
        print(f"saved {save_production(df)}")
        return
    if (args.out_dir / "run.json").exists() and not args.force:
        raise SystemExit(f"final test already run ({args.out_dir / 'run.json'}); use --force only if inputs changed")
    regression, classification, table = evaluate(df)
    write_results(regression, classification, table, args.out_dir)
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
