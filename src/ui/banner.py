"""The Forecast page's status banner, built from a `Snapshot` (spec §9, item 1).

Separate from `app/views/forecast.py` so the text (and its rules — demo mode always warns,
live/cache never invent a warning that isn't in `snapshot.warnings`, and a non-demo snapshot
dated before today gets a stale-cache warning) is unit-testable without a Streamlit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config import TIMEZONE
from src.live.snapshot import Snapshot
from src.ui.format import format_date_long, format_pkt_timestamp, source_label


@dataclass(frozen=True)
class Banner:
    """Everything the status banner shows, as plain strings — `app/views/forecast.py` just
    renders them with `st.info`/`st.warning`, no formatting logic of its own."""

    heading: str  # "Forecast for Tuesday, 10 November 2026 (24-hour average), issued 00:00 PKT"
    last_updated: str  # "Last updated 10 Nov 2026, 14:32 PKT"
    source_text: str  # "Source: Live data"
    warnings: tuple[str, ...]  # shown one by one
    sensor_notes: tuple[str, ...] = ()  # per-site "no forecast" notes, shown grouped in one box


# The live pipeline ends every per-site skip message with this (src/live/pipeline.py).
SENSOR_NOTE_SUFFIX = ", no forecast."

STALE_SNAPSHOT_WARNING = (
    "This forecast is for {date}, not today. Press Refresh live data to update, or switch to "
    "demo mode."
)


def today_in_pkt() -> date:
    """Today's calendar date in Pakistan Standard Time, for comparing against a snapshot's day."""
    return pd.Timestamp.now(tz=TIMEZONE).date()


def stale_forecast_warning(forecast_date: date, today: date) -> str | None:
    """A warning if `forecast_date` is before `today`, else `None`.

    `today` is a parameter (not read internally) so tests can pin it instead of depending on
    the real clock; `app/views/forecast.py` passes `today_in_pkt()`.
    """
    if forecast_date >= today:
        return None
    return STALE_SNAPSHOT_WARNING.format(date=format_date_long(forecast_date))


def build_banner(snapshot: Snapshot, today: date | None = None) -> Banner:
    """The banner text for `snapshot` — no Streamlit calls, just string building.

    `today` defaults to the real PKT date; demo snapshots never get the stale-cache warning —
    they are deliberately a past day, and already carry their own "Demo mode" warning.
    """
    heading = (
        f"Forecast for {format_date_long(snapshot.forecast_date)} "
        "(24-hour average), issued 00:00 PKT"
    )
    last_updated = f"Last updated {format_pkt_timestamp(snapshot.fetched_at)}"
    source_text = f"Source: {source_label(snapshot.source)}"

    stale = None
    if snapshot.source != "demo":
        stale = stale_forecast_warning(snapshot.forecast_date, today_in_pkt() if today is None else today)

    warnings = tuple(w for w in snapshot.warnings if not w.endswith(SENSOR_NOTE_SUFFIX))
    if stale is not None:
        warnings = (stale, *warnings)

    return Banner(
        heading=heading,
        last_updated=last_updated,
        source_text=source_text,
        warnings=warnings,
        sensor_notes=tuple(w for w in snapshot.warnings if w.endswith(SENSOR_NOTE_SUFFIX)),
    )
