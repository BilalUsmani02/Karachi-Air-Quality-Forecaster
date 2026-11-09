"""Tests for src/live/fetch.py and the `python -m src.live` CLI (no network: a fake session)."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.data import download_air_quality, download_weather
from src.data.sensors import sensor_by_id
from src.live import __main__ as cli
from src.live import fetch
from src.live.snapshot import Snapshot
from src.models.dataset import FEATURE_COLUMNS
from tests.live_fakes import FORECAST_DATE, SECRET, TZ, FakeApi, no_sleep

S1 = sensor_by_id("S1")


def test_fetch_readings_returns_raw_csv_shape():
    start = FORECAST_DATE - timedelta(days=2)
    frame = fetch.fetch_readings(S1, start, FORECAST_DATE, SECRET, session=FakeApi(), sleep=no_sleep)
    assert {"time", "pm25"} <= set(frame.columns)
    assert str(frame["time"].dt.tz) == TZ
    assert frame["pm25"].dtype == float
    assert len(frame) == 24 * 3
    assert frame["time"].is_monotonic_increasing


def test_fetch_readings_with_no_data_is_empty_but_typed():
    frame = fetch.fetch_readings(
        S1, FORECAST_DATE, FORECAST_DATE, SECRET, session=FakeApi(no_readings={"S1"}), sleep=no_sleep
    )
    assert frame.empty and list(frame.columns) == ["time", "pm25"]
    assert str(frame["time"].dt.tz) == TZ


def test_fetch_weather_uses_forecast_api_with_training_variables():
    api = FakeApi()
    frame = fetch.fetch_weather(S1, 5, 2, session=api, sleep=no_sleep)
    url, params, _ = api.calls[0]
    assert url == fetch.FORECAST_URL
    assert params["past_days"] == 5 and params["forecast_days"] == 2
    assert params["timezone"] == TZ and params["wind_speed_unit"] == "ms"
    assert params["hourly"] == ",".join(download_weather.HOURLY_VARS)
    assert list(frame.columns) == ["time", *download_weather.HOURLY_VARS]
    assert len(frame) == 24 * 7
    assert frame["time"].iloc[0] == pd.Timestamp(FORECAST_DATE - timedelta(days=5)).tz_localize(TZ)


def test_fetch_cams_renames_columns():
    api = FakeApi()
    frame = fetch.fetch_cams(S1, 1, 1, session=api, sleep=no_sleep)
    assert api.calls[0][0] == download_air_quality.URL
    assert list(frame.columns) == ["time", "cams_pm2_5", "cams_pm10", "cams_no2", "cams_dust", "cams_aod"]


def test_fetch_weather_missing_variable_is_an_error():
    with pytest.raises(ValueError, match="boundary_layer_height"):
        fetch.fetch_weather(S1, 1, 1, session=FakeApi(drop_weather_var="boundary_layer_height"), sleep=no_sleep)


# --- CLI -----------------------------------------------------------------------------


def _snapshot() -> Snapshot:
    forecasts = pd.DataFrame(
        {"sensor_id": ["S1", "S2"], "site": ["Climate Action Center", "Zafar Memon DHA"], "lat": [1.0, 2.0],
         "lon": [1.0, 2.0], "forecast_pm25": [8.04, 63.0], "prev_day_pm25": [7.0, 60.0], "actual_pm25": np.nan}
    )
    for col in FEATURE_COLUMNS:
        forecasts[col] = 0.0
    recent = pd.DataFrame({"sensor_id": [], "time": [], "pm25": []})
    return Snapshot(FORECAST_DATE, "2026-11-10T03:00:00+00:00", "live", forecasts, recent, ("Note µg/m³",))


def test_format_snapshot_lists_sensors_bands_and_warnings():
    text = cli.format_snapshot(_snapshot())
    assert "Forecast for 2026-11-10" in text and "source: live" in text
    assert "S1 Climate Action Center" in text and "8.0" in text and "Good" in text
    assert "S2 Zafar Memon DHA" in text and "63.0" in text
    assert text.endswith("- Note µg/m³")


def test_main_passes_refresh_flag(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "get_snapshot", lambda refresh: seen.append(refresh) or _snapshot())
    cli.main(["--refresh"])
    cli.main([])
    assert seen == [True, False]
    assert "S2 Zafar Memon DHA" in capsys.readouterr().out
