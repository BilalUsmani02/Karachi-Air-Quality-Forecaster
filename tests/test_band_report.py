"""Tests for the band-classification evidence (band_stats, ensemble_cv, band_report). Synthetic data only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.evaluate import band_report as br
from src.evaluate import band_stats as bs
from src.evaluate.ensemble_cv import average_members, load_ensemble_cv

# --- band_stats -----------------------------------------------------------------------------

Y_TRUE = np.array([0, 1, 1, 2, 3, 3, 4, 1, 2, 3])
Y_PRED = np.array([1, 1, 1, 2, 2, 3, 3, 1, 3, 4])


def test_present_classes_lists_only_seen_bands_in_order():
    assert bs.present_classes([3, 1, 1, 3]) == [1, 3]


def test_band_metrics_hand_computed_safety_rates():
    m = bs.band_metrics(Y_TRUE, Y_PRED)
    # truly >= 3: indices 4,5,6,9 -> predictions 2,3,3,4 -> 1 miss of 4
    assert math.isclose(m["miss_rate"], 1 / 4)
    # truly < 3: 6 days, predictions >= 3 only at index 8 -> 1 of 6
    assert math.isclose(m["false_alarm_rate"], 1 / 6)
    assert math.isclose(m["accuracy"], 5 / 10)
    assert m["n"] == 10 and m["n_present_classes"] == 5


def test_macro_f1_present_ignores_absent_band():
    y_true = np.array([1, 1, 2, 2, 3])
    y_pred = np.array([1, 2, 2, 2, 4])
    m = bs.band_metrics(y_true, y_pred)
    assert m["n_present_classes"] == 3
    assert math.isclose(m["macro_f1_all"], f1_score(y_true, y_pred, labels=range(5), average="macro", zero_division=0))
    assert math.isclose(m["macro_f1_present"], f1_score(y_true, y_pred, labels=[1, 2, 3], average="macro", zero_division=0))
    assert m["macro_f1_present"] > m["macro_f1_all"]


def test_band_metrics_does_not_mutate_inputs():
    y_true, y_pred = Y_TRUE.copy(), Y_PRED.copy()
    bs.band_metrics(y_true, y_pred)
    assert (y_true == Y_TRUE).all() and (y_pred == Y_PRED).all()


def test_per_class_table_f1_matches_sklearn():
    table = bs.per_class_table(Y_TRUE, Y_PRED)
    expected = f1_score(Y_TRUE, Y_PRED, labels=range(5), average=None, zero_division=0)
    assert np.allclose(table["f1"], expected)
    assert table["support"].tolist() == [1, 3, 2, 3, 1]
    assert table["band_name"].iloc[2] == "USG"


def test_confusion_frames_recall_rows_sum_to_one_and_absent_row_is_nan():
    counts, recall = bs.confusion_frames([1, 1, 3], [1, 2, 3])
    assert counts.loc["Moderate", "USG"] == 1
    assert math.isclose(recall.loc["Moderate"].sum(), 1.0)
    assert recall.loc["Good"].isna().all()


@pytest.mark.parametrize("present", [None, [1, 2, 3]])
def test_metrics_from_confusion_matches_reference_functions(present):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 300)
    y_pred = np.clip(y_true + rng.integers(-1, 2, 300), 0, 4)
    present = present if present is not None else bs.present_classes(y_true)
    ref = bs.band_metrics(y_true, y_pred, present=present)
    fast = bs.metrics_from_confusion(np.array(ref["confusion_matrix"]), present)
    for key in bs.CI_METRICS:
        assert math.isclose(fast[key], ref[key], abs_tol=1e-12), key


def test_metrics_from_confusion_empty_denominators_are_nan():
    cm = np.zeros((5, 5))
    cm[1, 1] = 3
    out = bs.metrics_from_confusion(cm, [])
    assert math.isnan(out["miss_rate"]) and math.isnan(out["macro_f1_present"])
    assert out["false_alarm_rate"] == 0.0
    assert math.isnan(bs.metrics_from_confusion(np.zeros((5, 5)), [1])["accuracy"])


def test_per_date_confusions_groups_rows_by_date():
    table = bs.per_date_confusions(["a", "a", "b"], [1, 3, 2], [1, 2, 2])
    assert table.shape == (2, 25)
    assert table[0].sum() == 2 and table[0, 3 * 5 + 2] == 1
    assert table[1, 2 * 5 + 2] == 1


def _bootstrap_frame(n_dates: int = 30, per_date: int = 4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2026-01-01", periods=n_dates), per_date)
    y = rng.integers(0, 5, len(dates))
    good = np.where(rng.random(len(y)) < 0.9, y, np.clip(y - 1, 0, 4))
    bad = np.where(rng.random(len(y)) < 0.5, y, np.clip(y - 1, 0, 4))
    return pd.DataFrame({"date": dates, "band_true": y, "band_a": good, "band_b": bad})


def test_date_block_bootstrap_is_deterministic_and_brackets_point():
    frame = _bootstrap_frame()
    kwargs = dict(pairs=[("a", "b")], n_boot=200, seed=7)
    first = bs.date_block_bootstrap(frame, {"a": "band_a", "b": "band_b"}, **kwargs)
    second = bs.date_block_bootstrap(frame, {"a": "band_a", "b": "band_b"}, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["model"]) == {"a", "b", "a - b"}
    assert (first["n_dates"] == 30).all()
    acc = first.set_index(["model", "metric"])
    assert acc.loc[("a", "accuracy"), "ci_low"] <= acc.loc[("a", "accuracy"), "point"] <= acc.loc[("a", "accuracy"), "ci_high"]
    diff = acc.loc[("a - b", "accuracy")]
    assert math.isclose(diff["point"], acc.loc[("a", "accuracy"), "point"] - acc.loc[("b", "accuracy"), "point"])
    assert diff["ci_low"] > 0  # a is clearly better than b


def test_date_block_bootstrap_resamples_whole_dates():
    # Every row of a date shares one outcome; resampling dates means accuracy takes values k/n_dates only.
    dates = np.repeat(pd.date_range("2026-01-01", periods=5), 3)
    correct = np.repeat([1, 1, 0, 0, 1], 3).astype(bool)
    frame = pd.DataFrame({"date": dates, "band_true": 1, "band_m": np.where(correct, 1, 2)})
    out = bs.date_block_bootstrap(frame, {"m": "band_m"}, n_boot=100, seed=0)
    low = out.set_index("metric").loc["accuracy", "ci_low"]
    assert math.isclose(low * 5, round(low * 5))


def test_date_block_bootstrap_rejects_zero_resamples():
    with pytest.raises(ValueError):
        bs.date_block_bootstrap(_bootstrap_frame(), {"a": "band_a"}, n_boot=0)


def test_date_block_bootstrap_all_nan_metric_gives_nan_ci():
    frame = pd.DataFrame({"date": ["d1", "d2"], "band_true": [1, 1], "band_m": [1, 1]})
    out = bs.date_block_bootstrap(frame, {"m": "band_m"}, n_boot=10).set_index("metric")
    assert math.isnan(out.loc["miss_rate", "ci_low"])


def test_rank_by_selection_rule_orders_miss_then_accuracy_then_macro_f1():
    metrics = pd.DataFrame(
        {
            "model": ["x", "y", "z", "w"],
            "miss_rate": [0.2, 0.1, 0.1, 0.1],
            "accuracy": [0.99, 0.80, 0.80, 0.85],
            "macro_f1_all": [0.9, 0.5, 0.6, 0.1],
        }
    )
    ranked = bs.rank_by_selection_rule(metrics)
    assert ranked["model"].tolist() == ["w", "z", "y", "x"]
    assert metrics["model"].tolist() == ["x", "y", "z", "w"]  # input untouched


# --- ensemble_cv ----------------------------------------------------------------------------


def _member(y_pred: list[float], order: list[int] | None = None) -> pd.DataFrame:
    base = pd.DataFrame(
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
    return base.iloc[order].reset_index(drop=True) if order else base


def test_average_members_means_aligned_rows_and_bands():
    out = average_members([_member([8.0, 30.0, 50.0]), _member([12.0, 50.0, 70.0], order=[2, 0, 1])])
    assert sorted(out["y_pred"]) == [10.0, 40.0, 60.0]
    row = out.set_index(["sensor_id", "date"]).loc[("S2", pd.Timestamp("2026-01-01"))]
    assert row["y_pred"] == 40.0 and row["band_pred"] == 2


def test_average_members_rejects_misaligned_or_disagreeing_members():
    with pytest.raises(ValueError, match="same"):
        average_members([_member([1.0, 2.0, 3.0]), _member([1.0, 2.0, 3.0]).iloc[:2]])
    other = _member([1.0, 2.0, 3.0]).assign(y_true=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="disagree"):
        average_members([_member([1.0, 2.0, 3.0]), other])
    with pytest.raises(ValueError):
        average_members([])


# --- band_report (end to end on synthetic files) --------------------------------------------


def _write_tuning_dir(root: Path, n_dates: int = 12) -> Path:
    rng = np.random.default_rng(3)
    dates = np.repeat(pd.date_range("2026-01-01", periods=n_dates), 3)
    pm = rng.uniform(5, 140, len(dates))
    bands = pd.Series(pm).apply(lambda v: 0 if v <= 9 else 1 if v <= 35.4 else 2 if v <= 55.4 else 3 if v <= 125.4 else 4)
    keys = pd.DataFrame({"sensor_id": ["S1", "S2", "S3"] * n_dates, "date": dates, "fold": 0})
    reg = keys.assign(y_true=pm, persistence=pm * 0.9, band_true=bands.to_numpy())
    root.mkdir(parents=True, exist_ok=True)
    for name, noise in (("hist_gb", 3.0), ("mlp_numpy", -3.0), ("random_forest", 2.0), ("ridge", 8.0)):
        table = reg.assign(y_pred=pm + noise, band_pred=bands.to_numpy())
        table.to_parquet(root / f"{name}_cv_predictions.parquet")
    for name, shift in (("hist_gb_clf", 0), ("random_forest_clf", -1), ("logistic_regression", 1)):
        pred = np.clip(bands.to_numpy() + (rng.random(len(bands)) < 0.3) * shift, 0, 4)
        keys.assign(y_true=bands.to_numpy(), y_pred=pred).to_parquet(root / f"{name}_cv_predictions.parquet")
    return root


def _write_final(path: Path) -> Path:
    rng = np.random.default_rng(4)
    band_true = rng.integers(1, 4, 30)
    frame = pd.DataFrame(
        {
            "sensor_id": ["S1", "S2"] * 15,
            "date": np.repeat(pd.date_range("2026-09-01", periods=15), 2),
            "band_true": band_true,
            "band_ensemble": band_true,
            "band_persistence": np.clip(band_true - 1, 0, 4),
            "band_ridge": band_true,
            "band_hist_gb": band_true,
            "band_random_forest_clf": np.clip(band_true + 1, 0, 4),
            "pred_ensemble": 1.0,
        }
    )
    frame.to_parquet(path)
    return path


def test_load_cv_bands_builds_all_model_columns(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    table = br.load_cv_bands(tuning)
    assert len(table) == 36
    assert br.available_models(table) == list(br.DISPLAY)
    assert table["band_ensemble"].dtype == np.int64


def test_load_cv_bands_rejects_classifier_on_other_rows(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    path = tuning / "hist_gb_clf_cv_predictions.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path)
    with pytest.raises(ValueError, match="hist_gb_clf"):
        br.load_cv_bands(tuning)


def test_load_cv_bands_rejects_ridge_on_other_rows(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    path = tuning / "ridge_cv_predictions.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path)
    with pytest.raises(ValueError, match="ridge"):
        br.load_cv_bands(tuning)


def test_load_ensemble_cv_reads_member_files(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    ens = load_ensemble_cv(tuning)
    assert np.allclose(ens["y_pred"], ens["y_true"])  # +3 and -3 noise average out


def test_best_direct_requires_a_direct_classifier():
    with pytest.raises(ValueError):
        br.best_direct(pd.DataFrame({"model": ["ensemble"], "miss_rate": [0.1], "accuracy": [0.9], "macro_f1_all": [0.5]}))


def test_build_report_writes_every_output(tmp_path):
    cv = br.load_cv_bands(_write_tuning_dir(tmp_path / "tuning"))
    final = br.load_final_bands(_write_final(tmp_path / "final.parquet"))
    out = tmp_path / "bands"
    results = br.build_report(cv, final, out, n_boot=50, seed=1)

    for name in (
        "metrics.csv", "per_class.csv", "bootstrap_ci.csv", "confusion_counts.csv",
        "confusion_recall.csv", "confusion_cv.png", "confusion_final_test.png", "summary.md",
    ):
        assert (out / name).exists(), name
    assert results["best"]["final_test"] == "random_forest_clf"
    final_metrics = results["metrics"].set_index(["set", "model"]).loc["final_test"]
    assert final_metrics.loc["ensemble", "n_present_classes"] == 3
    assert final_metrics.loc["ensemble", "accuracy"] == 1.0
    ci_models = set(results["ci"].loc[results["ci"]["set"] == "cv", "model"])
    assert "ensemble - persistence" in ci_models and "ensemble - logistic_regression" in ci_models
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "36 sensor-days over 12 dates" in summary and "30 sensor-days over 15 dates" in summary
    counts = pd.read_csv(out / "confusion_counts.csv")
    assert set(counts["model"]) >= {"ensemble", "persistence"}


def test_cli_main_writes_report(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    final = _write_final(tmp_path / "final.parquet")
    out = tmp_path / "cli"
    br.main(["--tuning-dir", str(tuning), "--final-predictions", str(final), "--out-dir", str(out), "--n-boot", "20"])
    assert (out / "summary.md").exists()


def test_cv_report_compares_the_ensemble_with_each_eligible_single_regressor(tmp_path):
    cv = br.load_cv_bands(_write_tuning_dir(tmp_path / "tuning"))
    final = br.load_final_bands(_write_final(tmp_path / "final.parquet"))
    results = br.build_report(cv, final, out_dir=tmp_path / "out", n_boot=20)
    ci = results["ci"]
    cv_diffs = set(ci.loc[ci["set"] == "cv", "model"])
    for name in br.ELIGIBLE_REGRESSORS:
        assert any(name in m and "ensemble" in m for m in cv_diffs), name
    assert "band_hist_gb" in final.columns  # final-test columns kept when present
