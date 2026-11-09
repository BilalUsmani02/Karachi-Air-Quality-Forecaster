"""Plain-English formatting helpers: dates, PKT times, numbers, source labels (spec §9).

Kept separate from `banner.py` and `results.py` because every view needs these — the chart
axis labels, the status banner, the comparison tables and the advice card all format the same
handful of things the same way.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from src.config import TIMEZONE

SOURCE_LABELS: dict[str, str] = {
    "live": "Live data",
    "cache": "Cached data (most recent available, not fetched just now)",
    "demo": "Demo mode (offline sample from stored history)",
}


def format_date_long(day: date) -> str:
    """"Tuesday, 10 November 2026" — a date a non-technical reader can place at a glance.

    Built from `Timestamp` parts rather than a `%-d`/`%e` strftime code, which Windows'
    C runtime does not support (the app must run the same on Windows and Linux/macOS).
    """
    ts = pd.Timestamp(day)
    return f"{ts.day_name()}, {ts.day} {ts.month_name()} {ts.year}"


def format_pkt_timestamp(iso_timestamp: str) -> str:
    """An ISO-8601 timestamp (any timezone, usually UTC) as a readable PKT time.

    e.g. "10 Nov 2026, 14:32 PKT". Falls back to the raw string if it cannot be parsed, so a
    malformed `fetched_at` never crashes the banner.
    """
    try:
        ts = pd.Timestamp(iso_timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.tz_convert(TIMEZONE)
    except (ValueError, TypeError):
        return str(iso_timestamp)
    return local.strftime("%d %b %Y, %H:%M") + " PKT"


def format_pm25(value: float | None) -> str:
    """A PM2.5 value to one decimal place with its unit, or an em dash if unknown."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.1f} µg/m³"


def format_pm25_whole(value: float | None) -> str:
    """A PM2.5 value rounded to a whole number — for the big headline figure."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.0f} µg/m³"


def source_label(source: str) -> str:
    """Human-readable label for a `Snapshot.source` value."""
    return SOURCE_LABELS.get(source, source)


def is_missing(value: float | None) -> bool:
    """True for `None` or NaN — the two ways a reading can be "not available" in a Snapshot."""
    return value is None or (isinstance(value, float) and math.isnan(value))
