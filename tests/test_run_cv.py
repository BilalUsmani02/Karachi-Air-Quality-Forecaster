import math

import numpy as np
import pandas as pd
import pytest

from src.classify.bands import bands_for, merge_bands
from src.data.features import COLUMN_ORDER
from src.data.sensors import SENSORS
from src.evaluate import run_cv
from src.evaluate.run_cv import CVResult, cross_validate, run

ALL_SENSOR_IDS = [s.id for s in SENSORS]
FINAL_TEST_START = pd.Timestamp("2026-09-01")


def _synthetic_frame(sensor_ids=("S1",), start="2025-06-01", end="2026-11-08", seed: int = 0) -> pd.DataFrame:
    """A daily-feature-table-shaped frame spanning the project's real split design.

    `y_pm25` follows a slow sine wave (plus noise) so it crosses every band over the
    date range, and `pm25_prev_day_mean` is each sensor's own `y_pm25` shifted by one
    day — never NaN, matching the contract §2 guarantee the persistence baseline
    relies on.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame(
        [{"sensor_id": sensor, "date": date} for date in dates for sensor in sensor_ids]
    )
    n = len(df)

    day_idx = (df["date"] - df["date"].min()).dt.days.to_numpy()
    pm25 = np.clip(40 + 60 * np.sin(2 * np.pi * day_idx / 180) + rng.normal(0, 8, n), 0, None)
    df["y_pm25"] = pm25
    df["y_hours"] = 20
    df["y_band"] = bands_for(pd.Series(pm25)).astype(int)

    for col in COLUMN_ORDER:
        if col in df.columns:
            continue
        if col.startswith("sensor_"):
            df[col] = 0
        elif col == "pm25_prev_day_mean":
            df[col] = df.groupby("sensor_id")["y_pm25"].shift(1).fillna(pm25.mean())
        else:
            df[col] = rng.normal(50, 10, n)
    for sid in ALL_SENSOR_IDS:
        df[f"sensor_{sid}"] = (df["sensor_id"] == sid).astype(int)

    return df[list(COLUMN_ORDER)]


class DummyRegressor:
    """Minimal model-contract §1 object (only `fit`/`predict`) — proves the plug-in point."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyRegressor":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean)


class DummyClassifier:
    """Minimal model-contract §1 object: predicts the most frequent training class."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyClassifier":
        values, counts = np.unique(y, return_counts=True)
        self._mode = int(values[np.argmax(counts)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mode, dtype=int)


@pytest.fixture(scope="module")
def df():
    return _synthetic_frame()


# --- cross_validate structure -----------------------------------------------------


def test_cross_validate_returns_cvresult_with_dummy_models(df):
    result = cross_validate(df, regressors={"dummy": DummyRegressor}, classifiers={"dummy": DummyClassifier})
    assert isinstance(result, CVResult)
    assert set(result.pooled_regression["model"]) == {"persistence", "dummy"}
    assert set(result.pooled_classification["model"]) == {"persistence", "dummy"}
    assert set(result.pooled_classification["route"]) == {"regress_then_band", "direct"}


def test_cross_validate_has_eight_folds(df):
    result = cross_validate(df, {"dummy": DummyRegressor}, {"dummy": DummyClassifier})
    assert result.fold_regression["fold"].nunique() == 8
    assert sorted(result.predictions["fold"].unique()) == list(range(8))


def test_cross_validate_persistence_present_with_no_models_passed(df):
    result = cross_validate(df, regressors={}, classifiers={})
    assert set(result.pooled_regression["model"]) == {"persistence"}
    assert set(result.pooled_classification["model"]) == {"persistence"}
    assert set(result.pooled_classification["route"]) == {"regress_then_band"}


def test_cross_validate_regression_metric_columns(df):
    result = cross_validate(df, {"dummy": DummyRegressor}, {})
    for col in ("mae", "rmse", "r2", "skill"):
        assert col in result.pooled_regression.columns


def test_cross_validate_classification_metric_columns(df):
    result = cross_validate(df, {}, {"dummy": DummyClassifier})
    for col in ("accuracy", "macro_f1", "confusion_matrix"):
        assert col in result.pooled_classification.columns
    for label in range(5):
        assert f"recall_{label}" in result.pooled_classification.columns


def test_persistence_pooled_skill_is_zero_against_itself(df):
    result = cross_validate(df, {}, {})
    row = result.pooled_regression.loc[result.pooled_regression["model"] == "persistence"].iloc[0]
    assert math.isclose(row["skill"], 0.0, abs_tol=1e-9)


# --- never touches the final test period -------------------------------------------


def test_cross_validate_never_touches_final_test_dates(df):
    result = cross_validate(df, {"dummy": DummyRegressor}, {"dummy": DummyClassifier})
    assert pd.to_datetime(result.predictions["date"]).max() < FINAL_TEST_START


# --- regress-then-band mapping ------------------------------------------------------


def test_to_band_matches_bands_for_and_merge():
    preds = np.array([5.0, 20.0, 230.0])
    bands = run_cv._to_band(preds)
    expected = merge_bands(bands_for(pd.Series(preds))).astype(int).to_numpy()
    np.testing.assert_array_equal(bands, expected)
    assert bands[2] == 4  # 230 -> Hazardous (5) -> merged to Very Unhealthy (4)


def test_to_band_clips_negative_predictions_to_good():
    bands = run_cv._to_band(np.array([-50.0]))
    assert bands[0] == 0


def test_cross_validate_regress_then_band_uses_predicted_pm25(df):
    result = cross_validate(df, {"dummy": DummyRegressor}, {})
    regress_then_band = result.predictions.loc[
        (result.predictions["model"] == "dummy") & (result.predictions["route"] == "regress_then_band")
    ]
    regression_rows = result.predictions.loc[
        (result.predictions["model"] == "dummy") & (result.predictions["task"] == "regression")
    ]
    expected = run_cv._to_band(regression_rows["y_pred"].to_numpy())
    np.testing.assert_array_equal(regress_then_band["y_pred"].to_numpy(), expected)


# --- CLI / run() ---------------------------------------------------------------------


def test_run_writes_all_four_result_files(tmp_path, df):
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)
    results_dir = tmp_path / "results"

    result = run(features_path=features_path, results_dir=results_dir)

    assert isinstance(result, CVResult)
    assert (results_dir / "baselines_cv_folds.csv").exists()
    assert (results_dir / "baselines_cv_pooled.csv").exists()
    assert (results_dir / "baselines_cv_predictions.parquet").exists()
    assert (results_dir / "baselines_summary.md").exists()

    summary_text = (results_dir / "baselines_summary.md").read_text(encoding="utf-8")
    assert "persistence" in summary_text
    assert "Confusion matrix" in summary_text

    pooled = pd.read_csv(results_dir / "baselines_cv_pooled.csv")
    assert "persistence" in pooled["model"].values

    predictions = pd.read_parquet(results_dir / "baselines_cv_predictions.parquet")
    assert pd.to_datetime(predictions["date"]).max() < FINAL_TEST_START


def test_main_prints_pooled_tables(monkeypatch, capsys, df):
    fake_result = CVResult(
        fold_regression=pd.DataFrame(),
        pooled_regression=pd.DataFrame({"model": ["persistence"], "mae": [1.0], "rmse": [1.0], "r2": [0.0], "skill": [0.0]}),
        fold_classification=pd.DataFrame(),
        pooled_classification=pd.DataFrame({"model": ["persistence"], "route": ["regress_then_band"], "accuracy": [0.5]}),
        predictions=pd.DataFrame(),
    )
    monkeypatch.setattr(run_cv, "run", lambda: fake_result)
    run_cv.main()
    captured = capsys.readouterr()
    assert "Pooled regression" in captured.out
    assert "Pooled classification" in captured.out
    assert "persistence" in captured.out


class _OrderRecorder:
    """Regressor that records whether the rows it was trained on arrived in time order."""

    seen_sorted: list[bool] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_OrderRecorder":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


def test_cross_validate_feeds_rows_in_time_order(monkeypatch):
    df = _synthetic_frame(sensor_ids=("S1", "S2"))
    shuffled = df.sort_values(["sensor_id", "date"]).reset_index(drop=True)
    seen_dates = []
    real_make_xy = run_cv.make_xy

    def recording_make_xy(frame, target):
        if target == "y_pm25":
            seen_dates.append(frame["date"].to_numpy())
        return real_make_xy(frame, target)

    monkeypatch.setattr(run_cv, "make_xy", recording_make_xy)

    cross_validate(shuffled, {"order": _OrderRecorder}, {})

    assert seen_dates
    for dates in seen_dates:
        assert (np.diff(dates.astype("datetime64[D]").astype(np.int64)) >= 0).all()


def test_default_regressors_include_numpy_network():
    from src.models.numpy_nn.network import MLPRegressorNP

    assert isinstance(run_cv.default_regressors()["mlp_numpy"](), MLPRegressorNP)
