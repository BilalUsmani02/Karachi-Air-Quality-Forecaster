"""Tests for the per-sensor fairness check (spec §10). Synthetic data only."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classify.bands import DEFAULT_MERGE, pm25_to_band
from src.data.sensors import sensor_by_id
from src.evaluate import fairness as fr


def _band(pm25: float) -> int:
    merged = pm25_to_band(max(pm25, 0.0))
    return DEFAULT_MERGE.get(merged, merged)


# --- per_sensor_regression ----------------------------------------------------------


def test_per_sensor_regression_hand_computed():
    pred = pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S2", "S2"],
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [12.0, 18.0, 33.0, 36.0],
            "persistence": [10.0, 10.0, 20.0, 20.0],
        }
    )
    out = fr.per_sensor_regression(pred).set_index("sensor_id")

    assert out.loc["S1", "n"] == 2
    assert math.isclose(out.loc["S1", "mae"], (2 + 2) / 2)
    assert math.isclose(out.loc["S1", "rmse"], math.sqrt(((2) ** 2 + (-2) ** 2) / 2))
    assert math.isclose(out.loc["S1", "bias"], ((12 - 10) + (18 - 20)) / 2)
    mae_persistence_s1 = (0 + 10) / 2
    assert math.isclose(out.loc["S1", "mae_persistence"], mae_persistence_s1)
    assert math.isclose(out.loc["S1", "skill"], 1 - out.loc["S1", "mae"] / mae_persistence_s1)
    assert out.loc["S1", "site"] == sensor_by_id("S1").site

    assert math.isclose(out.loc["S2", "mae"], (3 + 4) / 2)


def test_per_sensor_regression_sorted_worst_first():
    pred = pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S2", "S2"],
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [12.0, 18.0, 33.0, 36.0],  # S1 mae 2.0, S2 mae 3.5
        }
    )
    out = fr.per_sensor_regression(pred)
    assert out["sensor_id"].tolist() == ["S2", "S1"]


def test_per_sensor_regression_without_persistence_column_is_nan_not_crash():
    pred = pd.DataFrame({"sensor_id": ["S1", "S1"], "y_true": [10.0, 20.0], "y_pred": [12.0, 18.0]})
    out = fr.per_sensor_regression(pred)
    assert math.isnan(out.loc[0, "mae_persistence"])
    assert math.isnan(out.loc[0, "skill"])


# --- per_sensor_classification -------------------------------------------------------


def test_per_sensor_classification_miss_and_false_alarm_hand_computed():
    # unhealthy-true rows (y_true >= 3): indices 0,1,2,3; missed (y_pred < 3): 0 and 3 -> 2/4
    # healthy-true rows (y_true < 3): indices 4,5; false alarm (y_pred >= 3): index 5 -> 1/2
    pred = pd.DataFrame({"sensor_id": ["S1"] * 6, "y_true": [3, 3, 4, 4, 1, 2], "y_pred": [1, 3, 4, 2, 1, 3]})
    row = fr.per_sensor_classification(pred).iloc[0]
    assert math.isclose(row["miss_rate"], 0.5)
    assert math.isclose(row["false_alarm_rate"], 0.5)


def test_per_sensor_classification_no_unhealthy_days_gives_nan_miss_rate():
    pred = pd.DataFrame({"sensor_id": ["S2"] * 3, "y_true": [0, 1, 2], "y_pred": [0, 1, 1]})
    row = fr.per_sensor_classification(pred).iloc[0]
    assert math.isnan(row["miss_rate"])
    assert not math.isnan(row["false_alarm_rate"])


def test_miss_rate_bullet_when_no_sensor_has_an_unhealthy_day():
    pred = pd.DataFrame({"sensor_id": ["S1", "S1", "S2"], "y_true": [0, 1, 2], "y_pred": [0, 1, 1]})
    per_sensor_clf = fr.per_sensor_classification(pred)
    bullet = fr._miss_rate_bullet(per_sensor_clf)
    assert "could not be computed" in bullet


def test_per_sensor_classification_no_healthy_days_gives_nan_false_alarm_rate():
    pred = pd.DataFrame({"sensor_id": ["S3"] * 3, "y_true": [3, 4, 3], "y_pred": [3, 2, 3]})
    row = fr.per_sensor_classification(pred).iloc[0]
    assert math.isnan(row["false_alarm_rate"])
    assert not math.isnan(row["miss_rate"])


# --- disparity_summary ---------------------------------------------------------------


def test_disparity_summary_flags_sensor_over_25_percent():
    per_sensor = pd.DataFrame({"sensor_id": ["S1", "S2", "S3"], "site": ["A", "B", "C"], "n": [10, 10, 10], "mae": [5.0, 5.0, 10.0]})
    summary = fr.disparity_summary(per_sensor)

    overall = (5.0 + 5.0 + 10.0) / 3
    assert math.isclose(summary["overall_mae"], overall)
    assert summary["worst_sensor"]["sensor_id"] == "S3"
    assert summary["best_sensor"]["sensor_id"] in ("S1", "S2")
    assert [s["sensor_id"] for s in summary["flagged_sensors"]] == ["S3"]


def test_disparity_summary_threshold_is_strict_not_inclusive():
    # overall = 9.0; 25% over = 11.25; neither sensor exceeds it -> no flags
    per_sensor = pd.DataFrame({"sensor_id": ["S1", "S2"], "site": ["A", "B"], "n": [10, 10], "mae": [8.0, 10.0]})
    summary = fr.disparity_summary(per_sensor)
    assert summary["flagged_sensors"] == []


def test_disparity_summary_overall_mae_equals_pooled_mae():
    # overall_mae (n-weighted mean of per-sensor MAE) must equal pooling every row directly.
    pred = pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S1", "S2"],
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [12.0, 18.0, 33.0, 50.0],
        }
    )
    per_sensor = fr.per_sensor_regression(pred)
    summary = fr.disparity_summary(per_sensor)
    pooled_mae = np.mean(np.abs(pred["y_pred"] - pred["y_true"]))
    assert math.isclose(summary["overall_mae"], pooled_mae)


# --- explain_disparity ----------------------------------------------------------------


def _reg_table(sensor_ids, maes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "site": [f"Site {s}" for s in sensor_ids],
            "n": [10] * len(sensor_ids),
            "mae": maes,
            "rmse": maes,
            "bias": [0.0] * len(sensor_ids),
            "mae_persistence": maes,
            "skill": [0.0] * len(sensor_ids),
        }
    )


def test_explain_disparity_normalised_mae():
    per_sensor = _reg_table(["S1", "S2"], [5.0, 10.0])
    coverage = pd.DataFrame(
        {"sensor_id": ["S1", "S2"], "site": ["Site S1", "Site S2"], "completeness_pct": [80.0, 85.0], "training_rows": [100, 110], "mean_pm25": [50.0, 60.0]}
    )
    out = fr.explain_disparity(per_sensor, coverage).set_index("sensor_id")
    assert math.isclose(out.loc["S1", "normalised_mae"], 5.0 / 50.0)
    assert math.isclose(out.loc["S2", "normalised_mae"], 10.0 / 60.0)


def test_explain_disparity_spearman_sign_constructed_example():
    # MAE rises monotonically while completeness and training rows fall, and mean PM2.5 rises:
    # perfect rank correlations of known sign.
    per_sensor = _reg_table(["S1", "S2", "S3", "S4"], [2.0, 4.0, 6.0, 8.0])
    coverage = pd.DataFrame(
        {
            "sensor_id": ["S1", "S2", "S3", "S4"],
            "site": ["A", "B", "C", "D"],
            "completeness_pct": [90.0, 70.0, 50.0, 30.0],
            "training_rows": [400, 300, 200, 100],
            "mean_pm25": [20.0, 40.0, 60.0, 80.0],
        }
    )
    out = fr.explain_disparity(per_sensor, coverage)
    assert out["corr_mae_vs_completeness"].iloc[0] == pytest.approx(-1.0)
    assert out["corr_mae_vs_training_rows"].iloc[0] == pytest.approx(-1.0)
    assert out["corr_mae_vs_mean_pm25"].iloc[0] == pytest.approx(1.0)


def test_spearman_helper_nan_with_fewer_than_two_pairs():
    assert math.isnan(fr._spearman(pd.Series([1.0]), pd.Series([2.0])))
    assert math.isnan(fr._spearman(pd.Series([1.0, np.nan]), pd.Series([2.0, 3.0])))


# --- build_coverage_table --------------------------------------------------------------


def test_build_coverage_table_uses_fallback_without_features_path():
    cov = fr.build_coverage_table(None).set_index("sensor_id")
    assert cov.loc["S11", "completeness_pct"] == fr.SENSOR_COMPLETENESS_PCT["S11"]
    assert cov.loc["S11", "training_rows"] == fr.SENSOR_TRAINING_ROWS_FALLBACK["S11"]
    assert cov.loc["S11", "mean_pm25"] == fr.SENSOR_MEAN_PM25_FALLBACK["S11"]


def test_build_coverage_table_uses_real_features_file(tmp_path):
    features = pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S1", "S2"],
            "y_pm25": [10.0, 20.0, np.nan, 30.0],  # one dropped row for S1 (no target)
        }
    )
    path = tmp_path / "features.parquet"
    features.to_parquet(path)

    cov = fr.build_coverage_table(path).set_index("sensor_id")
    assert cov.loc["S1", "training_rows"] == 2
    assert math.isclose(cov.loc["S1", "mean_pm25"], 15.0)
    # completeness always comes from the hand-verified fallback, not the feature file
    assert cov.loc["S1", "completeness_pct"] == fr.SENSOR_COMPLETENESS_PCT["S1"]
    # a sensor absent from the feature file falls back to the hand-verified numbers
    assert cov.loc["S4", "training_rows"] == fr.SENSOR_TRAINING_ROWS_FALLBACK["S4"]


# --- _findings bullets ----------------------------------------------------------------------


def test_findings_lists_flagged_sensors_by_name():
    per_sensor = _reg_table(["S1", "S11"], [5.0, 10.0])  # S11 flagged (overall 7.5, +25% = 9.375)
    summary = fr.disparity_summary(per_sensor)
    coverage = fr.build_coverage_table(None)
    explained = fr.explain_disparity(per_sensor, coverage)

    bullets = fr._findings(explained, summary)
    assert any("exceed the overall MAE" in b and "S11" in b for b in bullets)


def test_findings_data_gap_dominant_branch():
    # completeness swings fully (rho = -1), mean PM2.5 held flat (rho undefined / ~0) -> gap wins
    per_sensor = _reg_table(["S1", "S2", "S3"], [2.0, 4.0, 6.0])
    coverage = pd.DataFrame(
        {
            "sensor_id": ["S1", "S2", "S3"],
            "site": ["A", "B", "C"],
            "completeness_pct": [90.0, 60.0, 30.0],
            "training_rows": [400, 300, 200],
            "mean_pm25": [50.0, 50.0, 50.1],
        }
    )
    summary = fr.disparity_summary(per_sensor)
    explained = fr.explain_disparity(per_sensor, coverage)
    bullets = fr._findings(explained, summary)
    assert any("Error tracks data gaps more than pollution level" in b for b in bullets)


def test_findings_not_enough_spread_branch():
    per_sensor = _reg_table(["S1", "S2"], [5.0, 5.0])
    coverage = pd.DataFrame(
        {"sensor_id": ["S1", "S2"], "site": ["A", "B"], "completeness_pct": [80.0, 80.0], "training_rows": [100, 100], "mean_pm25": [50.0, 50.0]}
    )
    summary = fr.disparity_summary(per_sensor)
    explained = fr.explain_disparity(per_sensor, coverage)
    bullets = fr._findings(explained, summary)
    assert any("Not enough spread" in b for b in bullets)


# --- CLI / run ----------------------------------------------------------------------------


def _synthetic_predictions_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    rows = []
    for sid, base in [("S1", 20.0), ("S11", 70.0)]:
        for i in range(20):
            date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
            y_true = max(0.0, base + rng.normal(0, 5))
            for model, noise in [("persistence", 3.0), ("random_forest", 6.0)]:
                y_pred = max(0.0, y_true + rng.normal(0, noise))
                rows.append(
                    {
                        "fold": 0, "sensor_id": sid, "date": date, "task": "regression",
                        "model": model, "route": "regression", "y_true": y_true, "y_pred": y_pred,
                    }
                )
                rows.append(
                    {
                        "fold": 0, "sensor_id": sid, "date": date, "task": "classification",
                        "model": model, "route": "regress_then_band", "y_true": _band(y_true), "y_pred": _band(y_pred),
                    }
                )
    path = tmp_path / "pred.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_run_writes_all_four_files(tmp_path):
    pred_path = _synthetic_predictions_path(tmp_path)
    model_dir = fr.run(pred_path, "random_forest", "regression", None, tmp_path / "fairness")

    for name in ("per_sensor.csv", "summary.json", "fairness_report.md", "per_sensor_mae.png"):
        p = model_dir / name
        assert p.exists() and p.stat().st_size > 0

    per_sensor = pd.read_csv(model_dir / "per_sensor.csv")
    assert set(per_sensor["sensor_id"]) == {"S1", "S11"}
    assert "miss_rate" in per_sensor.columns  # regress-then-band companion merged in

    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "random_forest"
    assert "overall_mae" in summary


def test_cli_main_writes_all_four_files(tmp_path, monkeypatch):
    pred_path = _synthetic_predictions_path(tmp_path)
    out_dir = tmp_path / "fairness_cli"
    argv = [
        "fairness", "--predictions", str(pred_path), "--model", "random_forest",
        "--route", "regression", "--out-dir", str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    fr.main()

    model_dir = out_dir / "random_forest"
    for name in ("per_sensor.csv", "summary.json", "fairness_report.md", "per_sensor_mae.png"):
        assert (model_dir / name).exists()


def test_run_raises_when_route_ambiguous(tmp_path):
    pred_path = _synthetic_predictions_path(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        fr.run(pred_path, "random_forest", None, None, tmp_path / "out")


def test_run_raises_for_unknown_model(tmp_path):
    pred_path = _synthetic_predictions_path(tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        fr.run(pred_path, "no_such_model", "regression", None, tmp_path / "out")


def test_run_persistence_self_comparison_has_zero_skill(tmp_path):
    pred_path = _synthetic_predictions_path(tmp_path)
    model_dir = fr.run(pred_path, "persistence", "regression", None, tmp_path / "out")
    per_sensor = pd.read_csv(model_dir / "per_sensor.csv")
    assert (per_sensor["skill"].abs() < 1e-9).all()


def test_run_pure_classification_route_writes_files(tmp_path):
    pred_path = _synthetic_predictions_path(tmp_path)
    model_dir = fr.run(pred_path, "random_forest", "regress_then_band", None, tmp_path / "out")
    for name in ("per_sensor.csv", "summary.json", "fairness_report.md", "per_sensor_mae.png"):
        assert (model_dir / name).exists()


def test_run_accepts_csv_predictions(tmp_path):
    parquet_path = _synthetic_predictions_path(tmp_path)
    csv_path = tmp_path / "pred.csv"
    pd.read_parquet(parquet_path).to_csv(csv_path, index=False)

    model_dir = fr.run(csv_path, "persistence", "regression", None, tmp_path / "out_csv")
    assert (model_dir / "per_sensor.csv").exists()


def test_run_raises_when_route_has_no_matching_rows(tmp_path):
    pred_path = _synthetic_predictions_path(tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        fr.run(pred_path, "random_forest", "direct", None, tmp_path / "out")


def test_filter_detects_task_from_score_predictions_shape_without_task_column():
    # tune.py's Score.predictions has no `task` column: a regressor's table carries
    # `persistence`; a classifier's table (already band labels) does not.
    regressor_like = pd.DataFrame({"sensor_id": ["S1"], "y_true": [1.0], "y_pred": [1.0], "persistence": [1.0]})
    classifier_like = pd.DataFrame({"sensor_id": ["S1"], "y_true": [1], "y_pred": [1]})

    _, task_r = fr._filter(regressor_like.assign(model="m"), "m", None)
    _, task_c = fr._filter(classifier_like.assign(model="m"), "m", None)
    assert task_r == "regression"
    assert task_c == "classification"
