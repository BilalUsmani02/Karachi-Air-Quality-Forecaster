"""Live fetchers: the last few days of sensor readings, weather and CAMS for one sensor (Phase 7).

Each returns a frame in the same shape as the raw CSVs the training pipeline reads, so
`clean.assemble_hourly_table` can treat live and historical data identically:

    readings  time, pm25, ...            OpenAQ `/sensors/{id}/hours` (same as `download_sensors`)
    weather   time + download_weather.HOURLY_VARS   Open-Meteo *forecast* API (not the archive)
    cams      time + cams_* columns                 Open-Meteo air-quality API

Weather variables: all eight `download_weather.HOURLY_VARS` (including `boundary_layer_height`)
are served by the forecast API too, so no substitute is needed. The forecast API blends recorded
analysis for past days with a forecast for today and later — the model was trained on the archive
(recorded) weather, so today's `fc_*` inputs are a forecast ("perfect prognosis" gap).

The OpenAQ API key is only ever sent in a request header; it is never logged or put in messages.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
import requests

from src.config import TIMEZONE
from src.data import download_air_quality, download_weather
from src.data.clean import CAMS_COLUMN_RENAME
from src.data.open_meteo import fetch_with_retry, parse_hourly
from src.data.openaq import fetch_hours, parse_records
from src.data.sensors import Sensor

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = download_air_quality.URL

Sleep = Callable[[float], None]


def fetch_readings(
    sensor: Sensor,
    start: date,
    end: date,
    api_key: str,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = time.sleep,
) -> pd.DataFrame:
    """Hourly PM2.5 for [start, end] (inclusive calendar days) as a raw-CSV-shaped frame.

    Always has `time` (tz-aware, PKT) and float `pm25` columns, even when there are no readings.
    """
    records = fetch_hours(sensor.openaq_sensor_id, start, end, api_key, session=session, sleep=sleep)
    frame = parse_records(records)
    if frame.empty:
        return pd.DataFrame({"time": pd.DatetimeIndex([], tz=TIMEZONE), "pm25": np.array([], dtype=float)})
    return frame.assign(pm25=pd.to_numeric(frame["pm25"], errors="coerce").astype(float))


def _open_meteo(
    url: str,
    sensor: Sensor,
    hourly_vars: list[str],
    past_days: int,
    forecast_days: int,
    extra_params: dict | None,
    session: requests.Session | None,
    sleep: Sleep,
) -> pd.DataFrame:
    params = {
        "latitude": sensor.lat,
        "longitude": sensor.lon,
        "hourly": ",".join(hourly_vars),
        "timezone": TIMEZONE,
        "past_days": past_days,
        "forecast_days": forecast_days,
        **(extra_params or {}),
    }
    payload = fetch_with_retry(url, params, session=session, sleep=sleep)
    frame = parse_hourly(payload)
    missing = [v for v in hourly_vars if v not in frame.columns]
    if missing:
        raise ValueError(f"{url} returned no {', '.join(missing)}")
    return frame[["time", *hourly_vars]].astype({v: float for v in hourly_vars})


def fetch_weather(
    sensor: Sensor,
    past_days: int,
    forecast_days: int,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = time.sleep,
) -> pd.DataFrame:
    """Hourly weather for the last `past_days` days, today and the following days (forecast API)."""
    return _open_meteo(
        FORECAST_URL,
        sensor,
        list(download_weather.HOURLY_VARS),
        past_days,
        forecast_days,
        download_weather.EXTRA_PARAMS,
        session,
        sleep,
    )


def fetch_cams(
    sensor: Sensor,
    past_days: int,
    forecast_days: int,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = time.sleep,
) -> pd.DataFrame:
    """Hourly CAMS pollution (columns renamed to `cams_*`, as `clean.load_cams_raw` does)."""
    frame = _open_meteo(
        AIR_QUALITY_URL,
        sensor,
        list(download_air_quality.HOURLY_VARS),
        past_days,
        forecast_days,
        None,
        session,
        sleep,
    )
    return frame.rename(columns=CAMS_COLUMN_RENAME)
