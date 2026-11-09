"""Tests for src/live/pipeline.py — live snapshot building and the app's get_snapshot fallbacks."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import requests

from src.data.clean import assemble_hourly_table
from src.data.features import build_daily_features
from src.data.sensors import sensor_by_id
from src.live import pipeline as pl
from src.live.snapshot import Snapshot, save_snapshot
from src.models.dataset import FEATURE_COLUMNS
from tests.live_fakes import FORECAST_DATE, NOW_UTC, SECRET, TZ, FakeApi, no_sleep, pm25_value
from tests.test_features import multi_sensor_hourly


class _IdentityPre:
    def transform(self, X):
        return X.to_numpy(dtype=np.float64)


class _PrevDayPlusOne:
    def predict(self, X):
        return X[:, list(FEATURE_COLUMNS).index("pm25_prev_day_mean")] + 1.0


BUNDLE = {"preprocessor": _IdentityPre(), "model": _PrevDayPlusOne(), "trained_on": ["2025-06-20", "2026-11-08"]}
THREE = tuple(sensor_by_id(s) for s in ("S1", "S2", "S3"))
ISSUE = pd.Timestamp(FORECAST_DATE).tz_localize(TZ)


def _build(api: FakeApi, sensors=THREE, **kwargs) -> Snapshot:
    return pl.build_live_snapshot(
        now=NOW_UTC, bundle=BUNDLE, api_key=SECRET, sensors=sensors, session=api, sleep=no_sleep, **kwargs
    )


def _assert_rows_equal(live: pd.Series, train: pd.Series) -> None:
    a = live[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    b = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


# --- dates ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "now, expected",
    [
        (pd.Timestamp("2026-11-09 18:59", tz="UTC"), date(2026, 11, 9)),  # 23:59 PKT
        (pd.Timestamp("2026-11-09 19:00", tz="UTC"), date(2026, 11, 10)),  # 00:00 PKT next day
        (datetime(2026, 11, 9, 20, 0), date(2026, 11, 10)),  # naive = UTC
    ],
)
def test_forecast_date_is_today_in_pkt(now, expected):
    assert pl.forecast_date_for(now) == expected


def test_forecast_date_defaults_to_now():
    assert pl.forecast_date_for() == pd.Timestamp.now(tz=TZ).date()


# --- equivalence with training -------------------------------------------------------


def test_live_feature_row_equals_training_row_on_synthetic_hourly():
    hourly = multi_sensor_hourly(["S1", "S2", "S3"], "2025-06-01 00:00", 24 * 20, seed=3)
    hourly.loc[(hourly["sensor_id"] == "S2") & (hourly.index % 11 == 0), ["pm25", "pm25_flag"]] = (np.nan, "missing")
    training, _ = build_daily_features(hourly)
    day = pd.Timestamp("2025-06-12")
    issue = day.tz_localize(TZ)

    # What is visible live: nothing after D's last hour; D's weather/CAMS yes, D's PM2.5 no.
    visible = hourly[hourly["time"] < issue + pd.Timedelta(days=1)]
    visible = visible.assign(pm25=visible["pm25"].mask(visible["time"] >= issue, 9999.0))
    live = pl.features_for_day(visible, day.date())

    expected = training[training["date"] == day].set_index("sensor_id")
    assert sorted(live["sensor_id"]) == sorted(expected.index)
    assert live["y_pm25"].isna().all()
    for _, row in live.iterrows():
        _assert_rows_equal(row, expected.loc[row["sensor_id"]])


def test_live_raw_path_equals_training_raw_path():
    """Raw readings -> clean -> features, live (truncated at D) vs. training (full history)."""
    sensor_id = "S5"
    times = pd.date_range("2026-10-25", "2026-11-15 23:00", freq="h", tz=TZ)
    readings = pd.DataFrame({"time": times, "pm25": [pm25_value(sensor_id, t) for t in times]})
    readings = readings.drop(index=[40, 41, 100, 230]).reset_index(drop=True)  # short gaps, interpolated
    hours = np.arange(len(times), dtype=float)
    weather = pd.DataFrame({"time": times})
    for i, var in enumerate(
        ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure", "precipitation",
         "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]
    ):
        weather[var] = 10 + i + np.cos(hours / 6.0 + i)
    cams = pd.DataFrame({"time": times})
    for i, var in enumerate(["cams_pm2_5", "cams_pm10", "cams_no2", "cams_dust", "cams_aod"]):
        cams[var] = 5 + i + np.sin(hours / 4.0 + i)

    full, _ = assemble_hourly_table(sensor_id, readings, weather, cams)
    training, _ = build_daily_features(full)
    live_hourly = pl.sensor_hourly(sensor_id, readings, weather, cams, FORECAST_DATE)
    live = pl.features_for_day(live_hourly, FORECAST_DATE)

    assert live_hourly["time"].max() == ISSUE + pd.Timedelta(hours=23)
    assert live_hourly.loc[live_hourly["time"] >= ISSUE, "pm25"].isna().all()
    expected = training[training["date"] == pd.Timestamp(FORECAST_DATE)].iloc[0]
    _assert_rows_equal(live.iloc[0], expected)


def test_sensor_hourly_without_readings_is_none():
    empty = pd.DataFrame({"time": pd.DatetimeIndex([], tz=TZ), "pm25": []})
    assert pl.sensor_hourly("S1", empty, empty, empty, FORECAST_DATE) is None


# --- build_live_snapshot -------------------------------------------------------------


def test_build_live_snapshot_happy_path():
    api = FakeApi()
    snapshot = _build(api)

    assert snapshot.source == "live"
    assert snapshot.forecast_date == FORECAST_DATE
    assert snapshot.fetched_at == "2026-11-10T03:00:00+00:00"
    assert snapshot.model_trained_on == ("2025-06-20", "2026-11-08")
    assert snapshot.warnings == (pl.FORECAST_WEATHER_NOTE,)
    assert list(snapshot.forecasts["sensor_id"]) == ["S1", "S2", "S3"]
    np.testing.assert_allclose(snapshot.forecasts["forecast_pm25"], snapshot.forecasts["prev_day_pm25"] + 1.0)
    assert snapshot.forecasts["actual_pm25"].isna().all()

    # Lag 1h is the reading at D-1 23:00 — today's readings (served by the fake) are never used.
    s1 = snapshot.forecasts.iloc[0]
    assert s1["pm25_lag_1h"] == pytest.approx(pm25_value("S1", ISSUE - pd.Timedelta(hours=1)))
    expected_prev = np.mean([pm25_value("S1", ISSUE - pd.Timedelta(hours=h)) for h in range(1, 25)])
    assert s1["pm25_prev_day_mean"] == pytest.approx(expected_prev)
    assert not np.isnan(s1["fc_boundary_layer_height"]) and not np.isnan(s1["pm25_mean_168h"])

    assert len(snapshot.recent) == 3 * 72
    assert snapshot.recent["time"].max() == ISSUE - pd.Timedelta(hours=1)


def test_api_key_only_sent_in_header():
    api = FakeApi()
    _build(api)
    openaq_calls = [c for c in api.calls if "openaq" in c[0]]
    assert openaq_calls and all(c[2] == {"X-API-Key": SECRET} for c in openaq_calls)
    assert all(SECRET not in str(c[1]) and SECRET not in c[0] for c in api.calls)
    weather_call = next(c for c in api.calls if "api.open-meteo.com/v1/forecast" in c[0])
    assert weather_call[1]["past_days"] == pl.LOOKBACK_DAYS and weather_call[1]["forecast_days"] == 2


def test_sensor_with_too_few_readings_yesterday_is_skipped_with_warning():
    yesterday = pd.date_range(ISSUE - pd.Timedelta(days=1), periods=20, freq="h")
    snapshot = _build(FakeApi(missing_hours={"S2": yesterday}))
    assert list(snapshot.forecasts["sensor_id"]) == ["S1", "S3"]
    assert "S2 Zafar Memon DHA: not enough readings yesterday, no forecast." in snapshot.warnings


def test_sensor_without_readings_or_failing_fetch_is_skipped():
    snapshot = _build(FakeApi(no_readings={"S2"}, weather_status={"S3": 500}))
    assert list(snapshot.forecasts["sensor_id"]) == ["S1"]
    assert any(w.startswith("S2 ") and "no readings in the last 10 days" in w for w in snapshot.warnings)
    assert any(w.startswith("S3 ") and "could not be fetched" in w for w in snapshot.warnings)


def test_leading_failures_abort_without_leaking_the_key():
    api = FakeApi(openaq_status={"S1": 401, "S2": 401})
    with pytest.raises(pl.LiveFetchError, match="HTTP 401 from api.openaq.org") as info:
        _build(api)
    assert SECRET not in str(info.value)
    assert not any("13368464" in c[0] for c in api.calls)  # S3 is never requested


def test_no_usable_sensor_raises():
    with pytest.raises(pl.LiveFetchError, match="no live readings"):
        _build(FakeApi(no_readings={"S1", "S2", "S3"}))
    yesterday = pd.date_range(ISSUE - pd.Timedelta(days=1), periods=24, freq="h")
    with pytest.raises(pl.LiveFetchError, match="enough readings yesterday"):
        _build(FakeApi(missing_hours={"S1": yesterday}), sensors=THREE[:1])


def test_failure_after_a_success_does_not_abort():
    snapshot = _build(FakeApi(openaq_status={"S2": 401, "S3": 401}))
    assert list(snapshot.forecasts["sensor_id"]) == ["S1"]


def test_missing_api_key_is_a_live_fetch_error(monkeypatch):
    def no_key():
        raise RuntimeError("OPENAQ_API_KEY is not set")

    monkeypatch.setattr(pl, "get_api_key", no_key)
    with pytest.raises(pl.LiveFetchError, match="no OpenAQ API key"):
        pl.build_live_snapshot(now=NOW_UTC, bundle=BUNDLE, sensors=THREE, session=FakeApi(), sleep=no_sleep)


def test_defaults_load_key_and_bundle(monkeypatch):
    monkeypatch.setattr(pl, "get_api_key", lambda: SECRET)
    monkeypatch.setattr(pl, "load_bundle", lambda: BUNDLE)
    snapshot = pl.build_live_snapshot(now=NOW_UTC, sensors=THREE, session=FakeApi(), sleep=no_sleep)
    assert len(snapshot.forecasts) == 3


def test_short_reason_redacts_and_summarises():
    assert pl._short_reason(RuntimeError(f"bad {SECRET} here"), SECRET) == "bad *** here"
    assert pl._short_reason(ValueError()) == "ValueError"
    assert len(pl._short_reason(RuntimeError("x" * 500))) == pl.REASON_MAX_CHARS
    response = type("R", (), {"status_code": 503, "url": "https://api.openaq.org/v3/x?y=1"})()
    assert pl._short_reason(requests.HTTPError("boom", response=response)) == "HTTP 503 from api.openaq.org"


# --- get_snapshot --------------------------------------------------------------------


def _snapshot(source: str, fetched_at: str = "2026-11-09T19:30:00+00:00") -> Snapshot:
    forecasts = pd.DataFrame(
        {"sensor_id": ["S1"], "site": ["x"], "lat": [1.0], "lon": [2.0], "forecast_pm25": [50.0],
         "prev_day_pm25": [49.0], "actual_pm25": [np.nan]}
    )
    for col in FEATURE_COLUMNS:
        forecasts[col] = 1.0
    recent = pd.DataFrame({"sensor_id": ["S1"], "time": [ISSUE - pd.Timedelta(hours=1)], "pm25": [3.0]})
    return Snapshot(FORECAST_DATE, fetched_at, source, forecasts, recent, ("w",), ("a", "b"))


def _fail(**_kwargs):
    raise pl.LiveFetchError("HTTP 503 from api.openaq.org")


def test_get_snapshot_without_refresh_prefers_cache_then_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "demo_snapshot", lambda bundle: _snapshot("demo"))
    assert pl.get_snapshot(cache_dir=tmp_path, bundle=BUNDLE).source == "demo"
    save_snapshot(_snapshot("live"), tmp_path)
    cached = pl.get_snapshot(cache_dir=tmp_path, bundle=BUNDLE)
    assert cached.source == "cache" and cached.warnings == ("w",)


def test_get_snapshot_refresh_success_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "build_live_snapshot", lambda bundle=None: _snapshot("live"))
    live = pl.get_snapshot(refresh=True, cache_dir=tmp_path, bundle=BUNDLE)
    assert live.source == "live"
    assert (tmp_path / "meta.json").exists()


def test_get_snapshot_refresh_failure_falls_back_to_cache(tmp_path, monkeypatch):
    save_snapshot(_snapshot("live"), tmp_path)
    monkeypatch.setattr(pl, "build_live_snapshot", _fail)
    snapshot = pl.get_snapshot(refresh=True, cache_dir=tmp_path, bundle=BUNDLE)
    assert snapshot.source == "cache"
    assert snapshot.warnings[-1] == (
        "Live update failed (HTTP 503 from api.openaq.org); showing the last saved forecast from "
        "2026-11-09T19:30:00+00:00."
    )


def test_get_snapshot_refresh_failure_without_cache_shows_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "build_live_snapshot", _fail)
    monkeypatch.setattr(pl, "demo_snapshot", lambda bundle: _snapshot("demo"))
    snapshot = pl.get_snapshot(refresh=True, cache_dir=tmp_path, bundle=BUNDLE)
    assert snapshot.source == "demo"
    assert "Live update failed (HTTP 503" in snapshot.warnings[-1] and "demo data" in snapshot.warnings[-1]


def test_get_snapshot_raises_when_nothing_is_available(tmp_path, monkeypatch):
    def no_demo(bundle):
        raise FileNotFoundError("features.parquet")

    monkeypatch.setattr(pl, "build_live_snapshot", _fail)
    monkeypatch.setattr(pl, "demo_snapshot", no_demo)
    with pytest.raises(pl.SnapshotUnavailableError, match="No forecast is available"):
        pl.get_snapshot(refresh=True, cache_dir=tmp_path, bundle=BUNDLE)


def test_get_snapshot_demo_loads_bundle_when_not_given(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "load_bundle", lambda: BUNDLE)
    monkeypatch.setattr(pl, "demo_snapshot", lambda bundle: _snapshot("demo") if bundle is BUNDLE else None)
    assert pl.get_snapshot(cache_dir=tmp_path).source == "demo"


def test_get_snapshot_save_failure_keeps_live_with_warning(tmp_path, monkeypatch):
    def broken_save(snapshot, cache_dir):
        raise PermissionError("read-only disk")

    monkeypatch.setattr(pl, "build_live_snapshot", lambda bundle=None: _snapshot("live"))
    monkeypatch.setattr(pl, "save_snapshot", broken_save)
    snapshot = pl.get_snapshot(refresh=True, cache_dir=tmp_path, bundle=BUNDLE)
    assert snapshot.source == "live"
    assert snapshot.warnings[-1] == "The forecast could not be saved for later (read-only disk)."


def test_lookback_covers_the_168h_window():
    assert timedelta(days=pl.LOOKBACK_DAYS) >= timedelta(hours=168 + 24)
