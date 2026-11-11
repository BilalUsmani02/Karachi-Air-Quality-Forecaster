"""Tests for the four-model-average / member-correlation diagnostics. Synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import ensemble_diagnostics as ed


def _write_member(root: Path, name: str, y_pred: list[float]) -> None:
    table = pd.DataFrame(
        {
            "sensor_id": ["S1", "S2", "S1"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "fold": [0, 0, 0],
            "y_true": [10.0, 40.0, 60.0],
            "y_pred": y_pred,
            "persistence": [9.0, 30.0, 50.0],
            "band_true": [1, 2, 3],
        }
    )
    table.to_parquet(root / f"{name}_cv_predictions.parquet")


def _write_four_members(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_member(root, "ridge", [8.0, 30.0, 50.0])
    _write_member(root, "random_forest", [12.0, 50.0, 70.0])
    _write_member(root, "hist_gb", [10.0, 40.0, 60.0])
    _write_member(root, "mlp_numpy", [10.0, 40.0, 60.0])
    return root


# --- four_model_average_metrics -------------------------------------------------------------


def test_four_model_average_metrics_matches_hand_computed_mean(tmp_path):
    tuning = _write_four_members(tmp_path / "tuning")
    out = ed.four_model_average_metrics(tuning)

    # mean of the four members per row: (8+12+10+10)/4=10, (30+50+40+40)/4=40, (50+70+60+60)/4=60
    assert out["members"] == list(ed.FOUR_MODELS)
    assert out["n_rows"] == 3
    assert out["mae"] == pytest.approx(0.0, abs=1e-9)
    assert out["skill"] == pytest.approx(1.0, abs=1e-9)


def test_four_model_average_metrics_with_custom_members(tmp_path):
    tuning = _write_four_members(tmp_path / "tuning")
    out = ed.four_model_average_metrics(tuning, members=("hist_gb", "mlp_numpy"))
    assert out["members"] == ["hist_gb", "mlp_numpy"]
    assert out["mae"] == pytest.approx(0.0, abs=1e-9)


# --- member_correlation ----------------------------------------------------------------------


def test_member_correlation_is_one_for_identical_predictions(tmp_path):
    tuning = _write_four_members(tmp_path / "tuning")
    corr = ed.member_correlation(tuning, pair=("hist_gb", "mlp_numpy"))
    assert corr == pytest.approx(1.0, abs=1e-9)


def test_member_correlation_detects_different_mistakes(tmp_path):
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    _write_member(tuning, "hist_gb", [10.0, 20.0, 30.0])
    _write_member(tuning, "mlp_numpy", [30.0, 20.0, 10.0])  # perfectly anti-correlated
    corr = ed.member_correlation(tuning, pair=("hist_gb", "mlp_numpy"))
    assert corr == pytest.approx(-1.0, abs=1e-9)


def test_member_correlation_rejects_misaligned_members(tmp_path):
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    _write_member(tuning, "hist_gb", [10.0, 20.0, 30.0])
    mismatched = pd.DataFrame(
        {
            "sensor_id": ["S1", "S2"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "fold": [0, 0],
            "y_true": [10.0, 40.0],
            "y_pred": [10.0, 40.0],
            "persistence": [9.0, 30.0],
            "band_true": [1, 2],
        }
    )
    mismatched.to_parquet(tuning / "mlp_numpy_cv_predictions.parquet")
    with pytest.raises(ValueError, match="do not cover the same rows"):
        ed.member_correlation(tuning, pair=("hist_gb", "mlp_numpy"))


# --- build_diagnostics / main ----------------------------------------------------------------


def test_build_diagnostics_bundles_both_results(tmp_path):
    tuning = _write_four_members(tmp_path / "tuning")
    out = ed.build_diagnostics(tuning)
    assert set(out) == {"four_model_average", "member_correlation", "how"}
    assert out["member_correlation"]["pair"] == list(ed.CORRELATION_PAIR)
    assert out["four_model_average"]["n_rows"] == 3


def test_main_writes_json_file(tmp_path):
    tuning = _write_four_members(tmp_path / "tuning")
    out_path = tmp_path / "out" / "ensemble_diagnostics.json"
    ed.main(["--tuning-dir", str(tuning), "--out", str(out_path)])

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["four_model_average"]["members"] == list(ed.FOUR_MODELS)
    assert "pearson_r" in written["member_correlation"]
    assert isinstance(written["member_correlation"]["pearson_r"], float)
