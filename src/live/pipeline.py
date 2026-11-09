"""The live forecast pipeline: fetch the last few days, build today's feature rows, forecast (Phase 7).

The forecast date D is today's calendar date in PKT; the forecast is "issued" at D 00:00, exactly
as in training (data contract §2). Every lag/rolling/`past_*` input uses hours before D 00:00,
which are complete by now; only `fc_*` uses day D itself — its weather and CAMS, which live come
from a forecast rather than recorded weather.

Live data goes through the *same* code as training: `clean.assemble_hourly_table` (with the grid
extended through D 23:00, so D's issue row and D's weather exist with PM2.5 NaN) and
`features.build_daily_frame`. `get_snapshot` is the one function the app calls.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import pandas as pd
import requests

from src.config import TIMEZONE
from src.data.clean import assemble_hourly_table
from src.data.features import build_daily_frame
from src.data.openaq import get_api_key
from src.data.sensors import SENSORS, Sensor
from src.live import fetch
from src.live.snapshot import (
    CACHE_DIR,
    Snapshot,
    demo_snapshot,
    forecast_frame,
    load_bundle,
    load_snapshot,
    recent_readings,
    save_snapshot,
)

# 168-h rolling mean needs 7 full days before D; 3 more days warm up the spike/flatline filters.
LOOKBACK_DAYS = 10
FORECAST_DAYS = 2  # today (D) and tomorrow, so D's 24 hours are always complete
MAX_LEADING_FAILURES = 2  # if the first sensors all fail, the API is down: stop, don't retry 14x
REASON_MAX_CHARS = 160

FORECAST_WEATHER_NOTE = (
    "Today's weather inputs come from a weather forecast, not recorded weather. The model was "
    "trained on recorded weather, so live accuracy may be a little lower than in testing."
)

# Errors that mean "the live data could not be fetched or understood" rather than a code bug.
FETCH_ERRORS = (requests.RequestException, RuntimeError, ValueError, KeyError)


class LiveFetchError(RuntimeError):
    """The live inputs could not be fetched, or none of them were usable for a forecast."""


class SnapshotUnavailableError(RuntimeError):
    """No live, cached or demo snapshot could be produced."""


# --- dates -------------------------------------------------------------------------


def _as_utc(now: datetime | None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def forecast_date_for(now: datetime | None = None) -> date:
    """Day D: today's calendar date in PKT (naive `now` is taken as UTC)."""
    return _as_utc(now).tz_convert(TIMEZONE).date()


def issue_time(forecast_date: date) -> pd.Timestamp:
    """D 00:00 PKT, the moment the forecast for D is issued."""
    return pd.Timestamp(forecast_date).tz_localize(TIMEZONE)


# --- hourly table and features ------------------------------------------------------


def sensor_hourly(
    sensor_id: str,
    readings: pd.DataFrame,
    weather: pd.DataFrame,
    cams: pd.DataFrame,
    forecast_date: date,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame | None:
    """One sensor's §1 hourly table from LOOKBACK_DAYS before D through D 23:00; None if no readings.

    Readings at or after D 00:00 are dropped (the model never sees D's PM2.5), and the grid is
    extended through D 23:00 so D's issue row and D's weather/CAMS exist.
    """
    issue = issue_time(forecast_date)
    start = issue - pd.Timedelta(days=lookback_days)
    readings = readings[(readings["time"] >= start) & (readings["time"] < issue)]
    if readings.empty:
        return None
    weather = weather[weather["time"] >= start]
    cams = cams[cams["time"] >= start]
    table, _ = assemble_hourly_table(sensor_id, readings, weather, cams, end=issue + pd.Timedelta(hours=23))
    return table


def features_for_day(hourly: pd.DataFrame, forecast_date: date) -> pd.DataFrame:
    """The feature row for D for every sensor in `hourly`, built by the training feature code.

    Any PM2.5 at or after D 00:00 is blanked first, so the row can only use what was known at
    issue time — whatever `hourly` contains. Includes rows whose `pm25_prev_day_mean` is NaN.
    """
    after_issue = hourly["time"] >= issue_time(forecast_date)
    masked = hourly.assign(
        pm25=hourly["pm25"].mask(after_issue),
        pm25_raw=hourly["pm25_raw"].mask(after_issue),
        pm25_flag=hourly["pm25_flag"].mask(after_issue, "missing"),
    )
    frame = build_daily_frame(masked)
    return frame[frame["date"] == pd.Timestamp(forecast_date)].reset_index(drop=True)


# --- fetching -----------------------------------------------------------------------


def _label(sensor: Sensor) -> str:
    return f"{sensor.id} {sensor.site}"


def _short_reason(exc: BaseException, secret: str | None = None) -> str:
    """A one-line reason for a failure, with no secrets in it."""
    response = getattr(exc, "response", None)
    if isinstance(exc, requests.HTTPError) and response is not None:
        reason = f"HTTP {response.status_code} from {urlparse(str(response.url)).netloc or 'the server'}"
    else:
        reason = str(exc) or type(exc).__name__
    if secret:
        reason = reason.replace(secret, "***")
    return reason[:REASON_MAX_CHARS]


def _fetch_sensor(
    sensor: Sensor,
    forecast_date: date,
    api_key: str,
    session: requests.Session | None,
    sleep: Callable[[float], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = forecast_date - timedelta(days=LOOKBACK_DAYS)
    readings = fetch.fetch_readings(sensor, start, forecast_date, api_key, session=session, sleep=sleep)
    weather = fetch.fetch_weather(sensor, LOOKBACK_DAYS, FORECAST_DAYS, session=session, sleep=sleep)
    cams = fetch.fetch_cams(sensor, LOOKBACK_DAYS, FORECAST_DAYS, session=session, sleep=sleep)
    return readings, weather, cams


def live_hourly(
    forecast_date: date,
    api_key: str,
    *,
    sensors: tuple[Sensor, ...] = SENSORS,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch and clean every sensor's recent data into one hourly table; plus per-sensor warnings.

    A sensor that fails to fetch, or has no readings, is skipped with a warning. Raises
    LiveFetchError if every sensor is skipped, or if the first MAX_LEADING_FAILURES all fail.
    """
    tables: list[pd.DataFrame] = []
    warnings: list[str] = []
    reasons: list[str] = []
    for attempt, sensor in enumerate(sensors, start=1):
        try:
            readings, weather, cams = _fetch_sensor(sensor, forecast_date, api_key, session, sleep)
        except FETCH_ERRORS as exc:
            reason = _short_reason(exc, api_key)
            reasons.append(reason)
            warnings.append(f"{_label(sensor)}: live data could not be fetched ({reason}), no forecast.")
            if len(reasons) == attempt and attempt >= MAX_LEADING_FAILURES:
                raise LiveFetchError(f"live data could not be fetched ({reasons[0]})") from None
            continue
        table = sensor_hourly(sensor.id, readings, weather, cams, forecast_date)
        if table is None:
            warnings.append(f"{_label(sensor)}: no readings in the last {LOOKBACK_DAYS} days, no forecast.")
            continue
        tables.append(table)
    if not tables:
        detail = f" ({reasons[0]})" if reasons else ""
        raise LiveFetchError(f"no live readings for any sensor{detail}")
    return pd.concat(tables, ignore_index=True), warnings


# --- snapshots ----------------------------------------------------------------------


def build_live_snapshot(
    now: datetime | None = None,
    bundle: dict[str, Any] | None = None,
    api_key: str | None = None,
    *,
    sensors: tuple[Sensor, ...] = SENSORS,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Snapshot:
    """Fetch live inputs and forecast today's (D's) 24-h mean PM2.5 at every sensor that can be.

    Sensors without a `pm25_prev_day_mean` (too few readings yesterday) are skipped with a
    warning. Raises LiveFetchError if the inputs can't be fetched or no sensor can be forecast.
    """
    fetched_at = _as_utc(now)
    forecast_date = forecast_date_for(fetched_at)
    if api_key is None:
        try:
            api_key = get_api_key()
        except RuntimeError:
            raise LiveFetchError("no OpenAQ API key is set (OPENAQ_API_KEY)") from None
    bundle = load_bundle() if bundle is None else bundle

    hourly, sensor_warnings = live_hourly(forecast_date, api_key, sensors=sensors, session=session, sleep=sleep)
    rows = features_for_day(hourly, forecast_date)
    has_prev_day = rows["pm25_prev_day_mean"].notna()
    for sensor_id in rows.loc[~has_prev_day, "sensor_id"]:
        sensor = next(s for s in sensors if s.id == sensor_id)
        sensor_warnings.append(f"{_label(sensor)}: not enough readings yesterday, no forecast.")
    if not has_prev_day.any():
        raise LiveFetchError("no sensor had enough readings yesterday to forecast today")

    return Snapshot(
        forecast_date=forecast_date,
        fetched_at=fetched_at.isoformat(timespec="seconds"),
        source="live",
        forecasts=forecast_frame(bundle, rows[has_prev_day].reset_index(drop=True)),
        recent=recent_readings(hourly, forecast_date),
        warnings=(FORECAST_WEATHER_NOTE, *sensor_warnings),
        model_trained_on=tuple(bundle.get("trained_on", ("", ""))),
    )


def _demo(bundle: dict[str, Any] | None, warning: str | None = None) -> Snapshot:
    try:
        demo = demo_snapshot(load_bundle() if bundle is None else bundle)
    except (OSError, ValueError, KeyError) as exc:
        raise SnapshotUnavailableError(
            "No forecast is available: the live update failed or was skipped, there is no saved "
            f"forecast, and the demo data could not be loaded ({_short_reason(exc)})."
        ) from exc
    return demo.with_source("demo", warning)


def get_snapshot(refresh: bool = False, cache_dir: Path = CACHE_DIR, bundle: dict[str, Any] | None = None) -> Snapshot:
    """The snapshot the app shows.

    refresh=True: build a live snapshot and save it; if that fails, fall back to the saved one
    (then the demo), with a warning saying why. refresh=False: the saved snapshot, else the demo.
    Raises SnapshotUnavailableError if nothing at all can be shown.
    """
    if not refresh:
        cached = load_snapshot(cache_dir)
        return cached if cached is not None else _demo(bundle)

    try:
        live = build_live_snapshot(bundle=bundle)
    except (LiveFetchError, OSError) as exc:
        reason = _short_reason(exc)
        cached = load_snapshot(cache_dir)
        if cached is not None:
            warning = f"Live update failed ({reason}); showing the last saved forecast from {cached.fetched_at}."
            return cached.with_source(cached.source, warning)
        return _demo(bundle, f"Live update failed ({reason}); no saved forecast yet, so showing demo data.")

    try:
        save_snapshot(live, cache_dir)
    except OSError as exc:
        return live.with_source("live", f"The forecast could not be saved for later ({_short_reason(exc)}).")
    return live
