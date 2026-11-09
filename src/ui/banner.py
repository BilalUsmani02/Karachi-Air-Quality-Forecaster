"""The Forecast page's status banner, built from a `Snapshot` (spec §9, item 1).

Separate from `app/views/forecast.py` so the text (and its one rule — demo mode always warns,
live/cache never invent a warning that isn't in `snapshot.warnings`) is unit-testable without a
Streamlit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def build_banner(snapshot: Snapshot) -> Banner:
    """The banner text for `snapshot` — no Streamlit calls, just string building."""
    heading = (
        f"Forecast for {format_date_long(snapshot.forecast_date)} "
        "(24-hour average), issued 00:00 PKT"
    )
    last_updated = f"Last updated {format_pkt_timestamp(snapshot.fetched_at)}"
    source_text = f"Source: {source_label(snapshot.source)}"
    return Banner(
        heading=heading,
        last_updated=last_updated,
        source_text=source_text,
        warnings=tuple(w for w in snapshot.warnings if not w.endswith(SENSOR_NOTE_SUFFIX)),
        sensor_notes=tuple(w for w in snapshot.warnings if w.endswith(SENSOR_NOTE_SUFFIX)),
    )
