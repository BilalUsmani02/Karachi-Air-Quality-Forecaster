"""A fake HTTP session standing in for OpenAQ and Open-Meteo in the live-pipeline tests (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.data import download_air_quality, download_weather
from src.data.sensors import SENSORS

TZ = "Asia/Karachi"
FORECAST_DATE = date(2026, 11, 10)
NOW_UTC = pd.Timestamp("2026-11-10 03:00", tz="UTC")  # 08:00 PKT on FORECAST_DATE
SECRET = "test-secret-key-123"

_BY_OPENAQ_ID = {s.openaq_sensor_id: s.id for s in SENSORS}


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, url: str = "https://example.org/x"):
        self.status_code = status_code
        self._payload = payload or {}
        self.url = url

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error for url {self.url}", response=self)


def pm25_value(sensor_id: str, time: pd.Timestamp) -> float:
    """A deterministic, never-flat PM2.5 value for (sensor, hour)."""
    hours = (time - pd.Timestamp("2026-01-01", tz=TZ)) / pd.Timedelta(hours=1)
    return float(40 + 15 * np.sin(hours / 5.0) + int(sensor_id[1:]) + (hours % 7) * 0.3)


@dataclass
class FakeApi:
    """Routes GETs by URL. Per-sensor knobs: status codes, hours with no reading, no data at all."""

    forecast_date: date = FORECAST_DATE
    openaq_status: dict[str, int] = field(default_factory=dict)
    weather_status: dict[str, int] = field(default_factory=dict)
    missing_hours: dict[str, pd.DatetimeIndex] = field(default_factory=dict)
    no_readings: set[str] = field(default_factory=set)
    drop_weather_var: str | None = None
    calls: list[tuple[str, dict, dict | None]] = field(default_factory=list)

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), headers))
        if "api.openaq.org" in url:
            return self._openaq(url, params)
        if "air-quality-api" in url:
            return self._open_meteo(params, download_air_quality.HOURLY_VARS, {}, offset=1.0)
        return self._open_meteo(params, download_weather.HOURLY_VARS, self.weather_status, offset=0.0)

    def _openaq(self, url, params):
        sensor_id = _BY_OPENAQ_ID[int(url.split("/sensors/")[1].split("/")[0])]
        status = self.openaq_status.get(sensor_id, 200)
        if status != 200:
            return FakeResponse(status, url=url)
        if sensor_id in self.no_readings or params["page"] > 1:
            return FakeResponse(200, {"results": []}, url=url)
        start = pd.Timestamp(params["datetime_from"], tz=TZ)
        end = pd.Timestamp(params["datetime_to"], tz=TZ)
        missing = self.missing_hours.get(sensor_id, pd.DatetimeIndex([], tz=TZ))
        records = [
            {
                "value": pm25_value(sensor_id, t),
                "period": {"datetimeFrom": {"local": t.isoformat()}},
                "coverage": {"observedCount": 1, "expectedCount": 1},
            }
            for t in pd.date_range(start, end, freq="h", inclusive="left")
            if t not in missing
        ]
        return FakeResponse(200, {"results": records}, url=url)

    def _open_meteo(self, params, variables, statuses, offset):
        sensor_id = next(s.id for s in SENSORS if s.lat == params["latitude"] and s.lon == params["longitude"])
        status = statuses.get(sensor_id, 200)
        if status != 200:
            return FakeResponse(status)
        first = self.forecast_date - timedelta(days=params["past_days"])
        times = pd.date_range(str(first), periods=24 * (params["past_days"] + params["forecast_days"]), freq="h")
        hours = np.arange(len(times), dtype=float)
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for i, var in enumerate(variables):
            if var != self.drop_weather_var:
                hourly[var] = list(10 + i + offset + np.cos(hours / 6.0 + i))
        return FakeResponse(200, {"latitude": params["latitude"], "longitude": params["longitude"], "hourly": hourly})


def no_sleep(_seconds: float) -> None:
    pass
