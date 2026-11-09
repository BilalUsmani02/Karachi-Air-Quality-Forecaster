"""Tests for src/live/snapshot.py — the cacheable object the app renders."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.sensors import SENSORS
from src.live import snapshot as snap
from src.models.dataset import FEATURE_COLUMNS


class _IdentityPre:
    def transform(self, X):
        return X.to_numpy(dtype=np.float64)


class _FirstColumnModel:
    """Predicts pm25_prev_day_mean + 1, so expected forecasts are easy to write down."""

    def predict(self, X):
        col = list(FEATURE_COLUMNS).index("pm25_prev_day_mean")
        return X[:, col] + 1.0


BUNDLE = {"preprocessor": _IdentityPre(), "model": _FirstColumnModel(), "trained_on": ["2025-06-20", "2026-11-08"]}


def _features(day: str, sensor_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(len(sensor_ids), len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    frame["pm25_prev_day_mean"] = np.arange(len(sensor_ids), dtype=float) * 10
    frame.insert(0, "date", pd.Timestamp(day))
    frame.insert(0, "sensor_id", sensor_ids)
    frame["y_pm25"] = 50.0
    return frame


def _hourly(sensor_ids: list[str], start: str, hours: int) -> pd.DataFrame:
    times = pd.date_range(start, periods=hours, freq="h", tz="Asia/Karachi")
    return pd.DataFrame(
        [(s, t, 20.0, 20.0) for s in sensor_ids for t in times], columns=["sensor_id", "time", "pm25_raw", "pm25"]
    )


@pytest.fixture
def processed(tmp_path):
    ids = ["S3", "S1", "S2"]
    features = pd.concat([_features("2026-11-07", ids), _features("2026-11-08", ids)], ignore_index=True)
    features.to_parquet(tmp_path / "features.parquet")
    _hourly(ids, "2026-11-04 00:00", 24 * 6).to_parquet(tmp_path / "hourly_clean.parquet")
    return tmp_path


def test_forecast_frame_has_meta_then_features_in_sensor_order():
    frame = snap.forecast_frame(BUNDLE, _features("2026-11-08", ["S3", "S1"]))
    assert list(frame.columns) == [*snap.META_COLUMNS, *FEATURE_COLUMNS]
    assert frame["sensor_id"].tolist() == ["S1", "S3"]  # registry order, not input order
    assert frame.loc[0, "site"] == SENSORS[0].site
    assert frame.loc[0, "forecast_pm25"] == pytest.approx(frame.loc[0, "pm25_prev_day_mean"] + 1)
    assert frame["actual_pm25"].isna().all()


def test_demo_snapshot_uses_latest_day_and_includes_actuals(processed):
    s = snap.demo_snapshot(BUNDLE, processed_dir=processed)
    assert s.source == "demo"
    assert s.forecast_date == date(2026, 11, 8)
    assert len(s.forecasts) == 3
    assert (s.forecasts["actual_pm25"] == 50.0).all()
    assert s.model_trained_on == ("2025-06-20", "2026-11-08")
    assert s.warnings  # demo mode is always labelled


def test_demo_snapshot_recent_window_is_72_hours_before_issue(processed):
    s = snap.demo_snapshot(BUNDLE, processed_dir=processed)
    issue = pd.Timestamp("2026-11-08", tz="Asia/Karachi")
    assert s.recent["time"].max() < issue
    assert s.recent["time"].min() == issue - pd.Timedelta(hours=72)
    assert len(s.recent) == 3 * 72
    assert list(s.recent.columns) == list(snap.RECENT_COLUMNS)


def test_demo_snapshot_for_a_given_day_and_unknown_day(processed):
    assert snap.demo_snapshot(BUNDLE, processed, date(2026, 11, 7)).forecast_date == date(2026, 11, 7)
    with pytest.raises(ValueError, match="no feature rows"):
        snap.demo_snapshot(BUNDLE, processed, date(2020, 1, 1))


def test_save_then_load_round_trip_relabels_live_as_cache(processed, tmp_path):
    live = snap.demo_snapshot(BUNDLE, processed_dir=processed).with_source("live")
    snap.save_snapshot(live, tmp_path / "cache")
    loaded = snap.load_snapshot(tmp_path / "cache")
    assert loaded.source == "cache"
    assert loaded.forecast_date == live.forecast_date
    assert loaded.warnings == live.warnings
    assert loaded.model_trained_on == live.model_trained_on
    pd.testing.assert_frame_equal(loaded.forecasts, live.forecasts)
    pd.testing.assert_frame_equal(loaded.recent, live.recent)


def test_load_keeps_demo_label_and_returns_none_when_empty(processed, tmp_path):
    snap.save_snapshot(snap.demo_snapshot(BUNDLE, processed_dir=processed), tmp_path / "c")
    assert snap.load_snapshot(tmp_path / "c").source == "demo"
    assert snap.load_snapshot(tmp_path / "nothing-here") is None


def test_with_source_adds_a_warning_without_mutating(processed):
    s = snap.demo_snapshot(BUNDLE, processed_dir=processed)
    t = s.with_source("cache", "live fetch failed")
    assert t.warnings == s.warnings + ("live fetch failed",)
    assert s.source == "demo"


def test_snapshot_validates_source_and_columns(processed):
    s = snap.demo_snapshot(BUNDLE, processed_dir=processed)
    with pytest.raises(ValueError, match="source"):
        s.with_source("guess")
    with pytest.raises(ValueError, match="forecasts is missing"):
        snap.Snapshot(s.forecast_date, s.fetched_at, "demo", s.forecasts.drop(columns="site"), s.recent)
    with pytest.raises(ValueError, match="recent is missing"):
        snap.Snapshot(s.forecast_date, s.fetched_at, "demo", s.forecasts, s.recent.drop(columns="pm25"))


def test_load_bundle_and_background(processed, tmp_path):
    import pickle

    path = tmp_path / "b.pkl"
    path.write_bytes(pickle.dumps({"trained_on": ["a", "b"]}))
    assert snap.load_bundle(path) == {"trained_on": ["a", "b"]}
    background = snap.load_background(processed_dir=processed, n=4)
    assert list(background.columns) == list(FEATURE_COLUMNS)
    assert len(background) == 4
