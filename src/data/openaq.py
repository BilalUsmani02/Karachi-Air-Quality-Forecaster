"""Shared fetch/parse logic for the OpenAQ v3 downloader (D1 sensors, D5 US Consulate).

Hourly aggregates come from `GET /v3/sensors/{sensor_id}/hours`. Checked live against the
API on 8 Nov 2026: the date filters on this endpoint are `datetime_from` / `datetime_to`
(ISO dates, interpreted as local midnight for the sensor) — `date_from`/`date_to` (which
work on `/days`) are silently ignored here and return the sensor's whole history instead of
erroring, so getting the param names right matters.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from src.config import ROOT, TIMEZONE
from src.data.open_meteo import date_chunks, fetch_with_retry

BASE_URL = "https://api.openaq.org/v3"

# OpenAQ's published limits are ~60 requests/minute and ~2,000/hour; this keeps us well under.
PAGE_LIMIT = 1000
REQUEST_DELAY_SECONDS = 1.1

ENV_VAR = "OPENAQ_API_KEY"


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from a .env file. Tolerant of a UTF-8 BOM, comments,
    blank lines, surrounding whitespace and quoted values. Returns {} if the file is missing.
    """
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def get_api_key(root: Path = ROOT) -> str:
    """Read OPENAQ_API_KEY: the environment variable takes precedence, then `<root>/.env`.

    Never prints or logs the key. Raises RuntimeError with fix-it instructions if unset.
    """
    key = os.environ.get(ENV_VAR, "").strip()
    if key:
        return key
    key = read_env_file(root / ".env").get(ENV_VAR, "").strip()
    if key:
        return key
    raise RuntimeError(
        f"{ENV_VAR} is not set. Add `{ENV_VAR}=<your key>` to a .env file in the project "
        f"root, or set the {ENV_VAR} environment variable. Get a free key by registering at "
        "https://explore.openaq.org/register."
    )


def fetch_hours(
    sensor_id: int,
    start: date,
    end: date,
    api_key: str,
    *,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Fetch every hourly aggregate for [start, end] (inclusive calendar days).

    Splits the range into calendar-year chunks and paginates each chunk with `page` until a
    short page (fewer than `PAGE_LIMIT` results) signals the end.
    """
    url = f"{BASE_URL}/sensors/{sensor_id}/hours"
    headers = {"X-API-Key": api_key}
    records: list[dict] = []
    for chunk_start, chunk_end in date_chunks(start, end):
        datetime_to = (chunk_end + timedelta(days=1)).isoformat()
        page = 1
        while True:
            params = {
                "datetime_from": chunk_start.isoformat(),
                "datetime_to": datetime_to,
                "limit": PAGE_LIMIT,
                "page": page,
            }
            payload = fetch_with_retry(url, params, headers=headers, session=session, sleep=sleep)
            page_records = payload.get("results", [])
            records.extend(page_records)
            sleep(request_delay)
            if len(page_records) < PAGE_LIMIT:
                break
            page += 1
    return records


def parse_records(records: list[dict]) -> pd.DataFrame:
    """Turn raw `/hours` result dicts into tidy rows, sorted by time with duplicates dropped.

    Columns: time (tz-aware Asia/Karachi, period start), pm25, observed_count, expected_count
    (from `coverage`), min/max/sd (from `summary`, when present).
    """
    columns = ["time", "pm25", "observed_count", "expected_count", "min", "max", "sd"]
    rows = []
    for record in records:
        local_start = (record.get("period") or {}).get("datetimeFrom", {}).get("local")
        if local_start is None:
            continue
        coverage = record.get("coverage") or {}
        summary = record.get("summary") or {}
        rows.append(
            {
                "time": local_start,
                "pm25": record.get("value"),
                "observed_count": coverage.get("observedCount"),
                "expected_count": coverage.get("expectedCount"),
                "min": summary.get("min"),
                "max": summary.get("max"),
                "sd": summary.get("sd"),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(TIMEZONE)
    frame = frame.sort_values("time").drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    return frame


def fetch_sensor_dataframe(
    sensor_id: int,
    start: date,
    end: date,
    api_key: str,
    *,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Fetch and parse one sensor's hourly PM2.5 history for [start, end]."""
    records = fetch_hours(
        sensor_id, start, end, api_key, session=session, sleep=sleep, request_delay=request_delay
    )
    return parse_records(records)
