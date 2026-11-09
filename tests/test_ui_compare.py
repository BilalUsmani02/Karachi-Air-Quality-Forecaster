"""Tests for `src.ui.compare` — reading/relabelling the stored results tables.

Builds a tiny `results/` tree in `tmp_path` shaped like the real files
(`results/final_test/metrics.csv` etc.) rather than reading the real project's `results/` —
keeps this test independent of what the real results happen to contain right now.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ui.compare import (
    band_route_table,
    fairness_per_sensor,
    final_test_metrics,
    global_shap_importance,
    mae_bar_data,
)


@pytest.fixture
def results_dir(tmp_path):
    (tmp_path / "final_test").mkdir()
    pd.DataFrame(
        {
            "model": ["persistence", "ridge", "random_forest", "hist_gb", "mlp_numpy", "ensemble"],
            "mae": [5.8, 5.6, 5.4, 4.7, 5.0, 4.6],
            "rmse": [8.6, 7.7, 7.8, 6.8, 7.0, 6.6],
            "r2": [0.76, 0.81, 0.81, 0.85, 0.85, 0.86],
            "skill": [0.0, 0.04, 0.08, 0.19, 0.14, 0.20],
        }
    ).to_csv(tmp_path / "final_test" / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "model": ["persistence", "ensemble", "random_forest_clf"],
            "route": ["regress_then_band", "regress_then_band", "direct"],
            "accuracy": [0.82, 0.85, 0.85],
            "macro_f1": [0.48, 0.49, 0.55],
            "miss_unhealthy": [0.23, 0.15, 0.20],
            "false_alarm": [0.05, 0.04, 0.02],
        }
    ).to_csv(tmp_path / "final_test" / "classification.csv", index=False)

    (tmp_path / "explain").mkdir()
    pd.DataFrame(
        {
            "feature": ["pm25_prev_day_mean", "pm25_lag_1h", "some_unregistered_column"],
            "tree_mean_abs_shap": [5.6, 10.2, 0.05],
            "ensemble_mean_abs_shap": [11.4, 4.9, 0.03],
        }
    ).to_csv(tmp_path / "explain" / "global_importance.csv", index=False)

    (tmp_path / "fairness" / "random_forest").mkdir(parents=True)
    pd.DataFrame(
        {
            "sensor_id": ["S11", "S1"],
            "site": ["Orangi Pilot Project", "Climate Action Center"],
            "mae": [18.5, 5.0],
            "mean_pm25": [58.9, 42.9],
            "normalised_mae": [0.31, 0.12],
        }
    ).to_csv(tmp_path / "fairness" / "random_forest" / "per_sensor.csv", index=False)

    return tmp_path


def test_final_test_metrics_is_relabelled_and_in_display_order(results_dir):
    metrics = final_test_metrics(results_dir)
    assert list(metrics["model"]) == [
        "Persistence (forecast = the day before)",
        "Ridge regression",
        "Random Forest",
        "Gradient Boosting",
        "NumPy neural network",
        "Ensemble — used by the app",
    ]
    assert list(metrics.columns) == ["model", "mae", "rmse", "r2", "skill"]


def test_final_test_metrics_returns_none_when_the_file_is_missing(tmp_path):
    assert final_test_metrics(tmp_path) is None


def test_mae_bar_data_keeps_only_model_and_mae(results_dir):
    bars = mae_bar_data(final_test_metrics(results_dir))
    assert list(bars.columns) == ["model", "mae"]
    assert len(bars) == 6


def test_band_route_table_keeps_regress_then_band_plus_the_best_direct_classifier(results_dir):
    table = band_route_table(results_dir)
    assert set(table["Route"]) == {"regress_then_band", "direct"}
    assert list(table.columns) == ["Model", "Route", "Accuracy", "Unhealthy days missed", "False alarms"]
    assert len(table) == 3


def test_band_route_table_returns_none_when_the_file_is_missing(tmp_path):
    assert band_route_table(tmp_path) is None


def test_global_shap_importance_adds_a_plain_phrase_and_sorts_descending(results_dir):
    importance = global_shap_importance(results_dir, top_n=2)
    assert len(importance) == 2
    assert importance.iloc[0]["feature"] == "pm25_prev_day_mean"
    assert "yesterday" in importance.iloc[0]["what_it_means"].lower()


def test_global_shap_importance_falls_back_to_the_column_name_if_unregistered(results_dir):
    importance = global_shap_importance(results_dir, top_n=3)
    assert importance.iloc[2]["feature"] == "some_unregistered_column"
    assert importance.iloc[2]["what_it_means"] == "some_unregistered_column"


def test_global_shap_importance_returns_none_when_the_file_is_missing(tmp_path):
    assert global_shap_importance(tmp_path) is None


def test_fairness_per_sensor_sorts_worst_first(results_dir):
    fairness = fairness_per_sensor(results_dir, model="random_forest")
    assert list(fairness["site"]) == ["Orangi Pilot Project", "Climate Action Center"]


def test_fairness_per_sensor_returns_none_when_missing(tmp_path):
    assert fairness_per_sensor(tmp_path, model="random_forest") is None
