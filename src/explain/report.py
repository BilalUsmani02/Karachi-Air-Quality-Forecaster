"""Build the SHAP explainability report (spec §8, Phase 6).

    uv run python -m src.explain --features PATH --bundle PATH [--out-dir results/explain]

Writes to `--out-dir` (default `results/explain/`):

- `global_importance.csv` — TreeExplainer (Gradient Boosting member, exact, full table) next to
  the model-agnostic ensemble explainer (a 300-row sample), for comparison.
- `shap_summary.png` — bar chart of the global ranking, with readable labels and units.
- `example_explanations.md` — 3-4 real days (a winter smog day, a clean monsoon day, a forecast
  near a band boundary, the single worst forecast), each with its prediction, top plain-English
  reasons and a small contribution table; plus the measured time for one live explanation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.classify.bands import BANDS
from src.config import SEED
from src.explain import reasons
from src.explain.shap_explain import (
    BACKGROUND_SIZE,
    GLOBAL_SAMPLE_SIZE,
    Explanation,
    build_explainer,
    explain_row,
    global_importance,
    load_bundle,
)
from src.models.dataset import make_xy

DEFAULT_OUT_DIR = Path("results/explain")
TOP_N_PLOT = 15
_BAND_BOUNDARIES: tuple[float, ...] = tuple(band.high for band in BANDS[:-1])


def _plot_summary(importance: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of the ensemble explainer's top features, largest at the top."""
    top = importance.head(TOP_N_PLOT).iloc[::-1]
    labels = [reasons.phrase_for(f) for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, top["ensemble_mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP value| (µg/m³)")
    ax.set_title("Global feature importance (ensemble explainer)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_examples(meta: pd.DataFrame) -> list[tuple[str, int]]:
    """Pick 3-4 positional row indices into `meta`/`X`: winter smog, clean monsoon, near a
    band boundary, and the single worst forecast — skipping any season absent from the data,
    and dropping later picks that land on an already-picked row.
    """
    month = pd.DatetimeIndex(meta["date"]).month
    candidates: list[tuple[str, int]] = []

    winter = meta.index[month.isin([12, 1, 2])]
    if len(winter):
        candidates.append(("A winter smog day", int(meta.loc[winter, "pred"].idxmax())))

    monsoon = meta.index[month.isin([7, 8, 9])]
    if len(monsoon):
        candidates.append(("A clean monsoon day", int(meta.loc[monsoon, "pred"].idxmin())))

    boundary_dist = meta["pred"].apply(lambda p: min(abs(p - b) for b in _BAND_BOUNDARIES))
    candidates.append(("A forecast near a band boundary", int(boundary_dist.idxmin())))
    candidates.append(("The single worst forecast in the table", int(meta["pred"].idxmax())))

    seen: set[int] = set()
    examples: list[tuple[str, int]] = []
    for label, idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)
        examples.append((label, idx))
    return examples


def _example_section(label: str, row_meta: pd.Series, explanation: Explanation, reason_texts: list[str]) -> str:
    lines = [
        f"### {label} — {row_meta['sensor_id']} on {pd.Timestamp(row_meta['date']).date()}",
        "",
        f"Forecast: **{explanation.prediction:.1f} µg/m³** (base value {explanation.base_value:.1f} µg/m³, "
        f"additivity error {explanation.additivity_error:+.3f} µg/m³).",
        "",
        "Top reasons:",
        "",
    ]
    lines += [f"- {text}" for text in reason_texts]
    lines += ["", "| Feature | Raw value | SHAP (µg/m³) |", "|---|---|---|"]
    for c in explanation.contributions[:6]:
        feature_label = reasons.phrase_for(c.feature)
        unit = reasons.unit_for(c.feature)
        value = f"{c.raw_value:.2f} {unit}".strip()
        lines.append(f"| {feature_label} | {value} | {c.shap_value:+.2f} |")
    lines.append("")
    return "\n".join(lines)


def _report_markdown(sections: list[str], timing_s: float | None) -> str:
    header = [
        "# Example explanations",
        "",
        "Real forecast days from the feature table, explained with the model-agnostic SHAP "
        "explainer on the production ensemble (spec §8). SHAP values are in µg/m³, the model's "
        "output unit; feature values are shown in their original (unscaled) units.",
        "",
    ]
    if timing_s is not None:
        header += [f"One live explanation took **{timing_s:.2f} s** on this machine.", ""]
    return "\n".join(header + sections)


def run(features_path: Path, bundle_path: Path, out_dir: Path, seed: int = SEED) -> dict:
    """Build every file in `out_dir`. Returns `{"importance": DataFrame, "timing_s": float}`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(bundle_path)
    preprocessor, model = bundle["preprocessor"], bundle["model"]

    df = pd.read_parquet(features_path)
    X_df, _y = make_xy(df, "y_pm25")
    X = preprocessor.transform(X_df)
    mask = df["y_pm25"].notna()  # same mask make_xy used internally, so rows line up with X_df
    meta = df.loc[mask, ["sensor_id", "date"]].reset_index(drop=True)
    meta["pred"] = model.predict(X)
    medians = X_df.median().to_dict()

    importance = global_importance(bundle, X, sample_size=GLOBAL_SAMPLE_SIZE, seed=seed)
    importance.to_csv(out_dir / "global_importance.csv", index=False)
    _plot_summary(importance, out_dir / "shap_summary.png")

    explainer = build_explainer(bundle, X, seed=seed, background_size=BACKGROUND_SIZE)
    sections, timing_s = [], None
    for label, idx in _pick_examples(meta):
        t0 = time.perf_counter()
        explanation = explain_row(explainer, X[idx], X_df.iloc[idx])
        elapsed = time.perf_counter() - t0
        timing_s = elapsed if timing_s is None else timing_s
        reason_texts = reasons.top_reasons(explanation, k=3, feature_medians=medians)
        sections.append(_example_section(label, meta.iloc[idx], explanation, reason_texts))

    (out_dir / "example_explanations.md").write_text(_report_markdown(sections, timing_s), encoding="utf-8")
    return {"importance": importance, "timing_s": timing_s}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the SHAP explainability report (spec §8).")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    result = run(args.features, args.bundle, args.out_dir)
    print(f"wrote {args.out_dir} (one live explanation took {result['timing_s']:.2f} s)")


if __name__ == "__main__":
    main()
