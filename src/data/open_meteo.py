"""Shared fetch/parse/save logic for the Open-Meteo downloaders (D2 weather, D3 air quality)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import requests

from src.config import TIMEZONE
from src.data.sensors import Sensor

MAX_RETRIES = 5
BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    grid_latitude: float
    grid_longitude: float


def build_params(
    sensor: Sensor,
    start: date,
    end: date,
    hourly_vars: Sequence[str],
    extra_params: dict | None = None,
) -> dict:
    """Build the query params for one request (one sensor, one date chunk)."""
    params = {
        "latitude": sensor.lat,
        "longitude": sensor.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(hourly_vars),
        "timezone": TIMEZONE,
    }
    params.update(extra_params or {})
    return params


def date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-year pieces, each inclusive."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        year_end = date(chunk_start.year, 12, 31)
        chunk_end = min(year_end, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def fetch_with_retry(
    url: str,
    params: dict,
    *,
    headers: dict | None = None,
    max_retries: int = MAX_RETRIES,
    backoff_seconds: float = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    session: requests.Session | None = None,
) -> dict:
    """GET url with retries and exponential backoff on network errors, 429 and 5xx."""
    http = session or requests
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = http.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()
            last_exc = requests.HTTPError(f"{response.status_code} from {url}")
        if attempt < max_retries - 1:
            sleep(backoff_seconds * (2**attempt))
    raise RuntimeError(f"exhausted {max_retries} retries fetching {url}") from last_exc


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo "hourly" JSON block into a tz-aware DataFrame."""
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize(TIMEZONE)
    return frame


def fetch_range(
    url: str,
    sensor: Sensor,
    start: date,
    end: date,
    hourly_vars: Sequence[str],
    *,
    extra_params: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
    session: requests.Session | None = None,
) -> FetchResult:
    """Fetch [start, end] in calendar-year chunks and concatenate into one DataFrame."""
    frames = []
    grid_latitude: float | None = None
    grid_longitude: float | None = None
    for chunk_start, chunk_end in date_chunks(start, end):
        params = build_params(sensor, chunk_start, chunk_end, hourly_vars, extra_params)
        payload = fetch_with_retry(url, params, sleep=sleep, session=session)
        if grid_latitude is None:
            grid_latitude = payload["latitude"]
            grid_longitude = payload["longitude"]
        frames.append(parse_hourly(payload))
    frame = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    return FetchResult(frame=frame, grid_latitude=grid_latitude, grid_longitude=grid_longitude)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_manifest(path: Path, entries: dict) -> None:
    """Merge `entries` (keyed by sensor ID) into the manifest at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update(entries)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
