"""Tests for `src.explain.report` (the CLI). Heavy SHAP calls are monkeypatched out — this
file checks file-writing and content, not SHAP numerics (covered by `test_explain_shap.py`).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.explain import report
from src.explain.shap_explain import Contribution, Explanation
from src.models.dataset import FEATURE_COLUMNS, Preprocessor, make_xy
from tests.test_run_cv import _synthetic_frame


class ConstModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 42.0)


def _fake_bundle_and_features(tmp_path):
    """A tiny real (synthetic) feature table and a cheap-but-real preprocessor/model bundle."""
    df = _synthetic_frame(sensor_ids=("S1",), start="2025-06-01", end="2025-08-15")
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)

    X_df, _y = make_xy(df, "y_pm25")
    preprocessor = Preprocessor().fit(X_df)
    bundle = {"preprocessor": preprocessor, "model": ConstModel(), "trained_on": ["2025-06-01", "2025-08-15"]}
    bundle_path = tmp_path / "bundle.pkl"
    bundle_path.write_bytes(pickle.dumps(bundle))
    return features_path, bundle_path


def _fake_importance(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS[:n]),
            "tree_mean_abs_shap": [float(n - i) for i in range(n)],
            "ensemble_mean_abs_shap": [float(n - i) for i in range(n)],
        }
    )


def _fake_explain_row(explainer, row_scaled, row_raw):
    features = FEATURE_COLUMNS[:3]
    contributions = tuple(
        Contribution(feature=f, raw_value=float(row_raw[f]), shap_value=float(3 - i))
        for i, f in enumerate(features)
    )
    return Explanation(base_value=40.0, prediction=42.0, contributions=contributions)


def _patch_heavy_parts(monkeypatch) -> None:
    monkeypatch.setattr(report, "global_importance", lambda *a, **k: _fake_importance())
    monkeypatch.setattr(report, "build_explainer", lambda *a, **k: object())
    monkeypatch.setattr(report, "explain_row", _fake_explain_row)


def test_run_writes_all_expected_files(tmp_path, monkeypatch):
    features_path, bundle_path = _fake_bundle_and_features(tmp_path)
    out_dir = tmp_path / "out"
    _patch_heavy_parts(monkeypatch)

    result = report.run(features_path, bundle_path, out_dir)

    for name in ("global_importance.csv", "shap_summary.png", "example_explanations.md"):
        assert (out_dir / name).exists()
    assert result["timing_s"] is not None and result["timing_s"] >= 0
    assert list(result["importance"].columns) == ["feature", "tree_mean_abs_shap", "ensemble_mean_abs_shap"]


def test_example_explanations_content(tmp_path, monkeypatch):
    features_path, bundle_path = _fake_bundle_and_features(tmp_path)
    out_dir = tmp_path / "out"
    _patch_heavy_parts(monkeypatch)

    report.run(features_path, bundle_path, out_dir)
    text = (out_dir / "example_explanations.md").read_text(encoding="utf-8")

    assert "# Example explanations" in text
    assert "Forecast:" in text
    assert "µg/m³" in text
    assert "live explanation took" in text


def test_main_cli_writes_files(tmp_path, monkeypatch, capsys):
    features_path, bundle_path = _fake_bundle_and_features(tmp_path)
    out_dir = tmp_path / "out_cli"
    _patch_heavy_parts(monkeypatch)

    report.main(["--features", str(features_path), "--bundle", str(bundle_path), "--out-dir", str(out_dir)])

    assert (out_dir / "global_importance.csv").exists()
    assert "wrote" in capsys.readouterr().out


def test_pick_examples_skips_a_missing_season_and_returns_unique_rows():
    dates = pd.date_range("2025-06-01", "2025-08-15", freq="D")  # no Dec/Jan/Feb in range
    meta = pd.DataFrame({"sensor_id": "S1", "date": dates, "pred": np.linspace(5, 100, len(dates))})

    examples = report._pick_examples(meta)
    labels = [label for label, _ in examples]

    assert "A winter smog day" not in labels
    assert "A clean monsoon day" in labels
    indices = [idx for _, idx in examples]
    assert len(set(indices)) == len(indices)


def test_pick_examples_includes_winter_when_present():
    dates = pd.date_range("2025-12-01", "2026-02-15", freq="D")
    meta = pd.DataFrame({"sensor_id": "S1", "date": dates, "pred": np.linspace(5, 300, len(dates))})

    examples = report._pick_examples(meta)
    labels = [label for label, _ in examples]

    assert "A winter smog day" in labels
