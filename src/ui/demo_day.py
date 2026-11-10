"""Which past day the "Demo mode (offline example day)" toggle shows by default (review item 2b).

Pure: works from the processed features table alone (no model, no Streamlit), so picking the
default demo day never needs the model bundle loaded. `app/views/forecast.py` reads the table
once and calls these two functions to build the sidebar's day picker.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.classify.bands import pm25_to_band

# Same threshold as `src.ui.health_alert.USG_BAND_INDEX` — kept as a literal here so this module
# has no dependency on the health-alert module (the two features are independent).
USG_BAND_INDEX = 2


def available_demo_days(features: pd.DataFrame) -> list[date]:
    """Every distinct calendar day in `features["date"]`, earliest first."""
    return sorted({pd.Timestamp(ts).date() for ts in features["date"]})


def _worst_band(values: pd.Series) -> int:
    """The worst (highest) band index among `values`'s non-missing PM2.5 readings, or -1."""
    bands = [pm25_to_band(float(v)) for v in values if pd.notna(v)]
    return max(bands) if bands else -1


def default_demo_day(features: pd.DataFrame) -> date | None:
    """The best day to show by default: the day with the most sensors, preferring one whose
    measured PM2.5 (`y_pm25`) reaches Unhealthy for Sensitive Groups or worse somewhere.

    "Most sensors" matters first — a thin day makes a poor demo of the map and the city-wide
    alert. Among the days tied for the most sensors, one with a USG-or-worse reading is
    preferred (so the demo can show off the health alert), then the most recent such day.
    Returns `None` if `features` is empty.
    """
    if features.empty:
        return None

    by_day: dict[date, list[float]] = {}
    for ts, y in zip(features["date"], features["y_pm25"]):
        by_day.setdefault(pd.Timestamp(ts).date(), []).append(y)

    max_sensors = max(len(values) for values in by_day.values())
    candidates = [day for day, values in by_day.items() if len(values) == max_sensors]

    severe = [day for day in candidates if _worst_band(by_day[day]) >= USG_BAND_INDEX]
    pool = severe if severe else candidates
    return max(pool)
