"""Data shaping for the Forecast page's chart (spec §9, item 7): no plotly, no Streamlit.

`app/components/chart.py` turns a `ChartSeries` into a plotly figure. Keeping the shaping here
means the "which 72 hours, what counts as the forecast segment, how tall do the band stripes
need to be" logic is testable without ever building a `Figure`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.classify.bands import BANDS
from src.expert.engine import EPA_COLORS
from src.live.snapshot import Snapshot

STRIPE_CEILING_MARGIN = 1.3  # draw band stripes a bit above the tallest value on the chart
STRIPE_CEILING_FLOOR = 60.0  # never collapse the stripes below this even on a very clean day


@dataclass(frozen=True)
class BandStripe:
    """One EPA band's background shading, capped to a finite top for a bounded chart."""

    low: float
    high: float  # finite even for the open-ended top band
    color: str
    name: str


@dataclass(frozen=True)
class ChartSeries:
    """Everything `app/components/chart.py` needs to draw the 72 h history + forecast chart."""

    sensor_id: str
    recent_times: tuple[pd.Timestamp, ...]
    recent_values: tuple[float, ...]
    forecast_start: pd.Timestamp  # day D, 00:00 PKT
    forecast_end: pd.Timestamp  # day D+1, 00:00 PKT
    forecast_value: float
    band_stripes: tuple[BandStripe, ...]


def _stripe_ceiling(recent_values: tuple[float, ...], forecast_value: float) -> float:
    values = [v for v in (*recent_values, forecast_value) if not math.isnan(v)]
    tallest = max(values) if values else 0.0
    return max(tallest * STRIPE_CEILING_MARGIN, STRIPE_CEILING_FLOOR)


def _band_stripes(ceiling: float) -> tuple[BandStripe, ...]:
    stripes = []
    for band in BANDS:
        low = band.low
        if low >= ceiling:
            break
        high = min(band.high, ceiling) if math.isfinite(band.high) else ceiling
        stripes.append(BandStripe(low=low, high=high, color=EPA_COLORS[band.index], name=band.name))
    return tuple(stripes)


def build_chart_series(snapshot: Snapshot, sensor_id: str) -> ChartSeries:
    """The chart data for `sensor_id`: its last 72 h of readings plus day D's forecast segment.

    Raises `ValueError` if `sensor_id` has no forecast row in `snapshot.forecasts` — callers
    should check `src.ui.results.available_sensor_ids` first.
    """
    matches = snapshot.forecasts.loc[snapshot.forecasts["sensor_id"] == sensor_id]
    if matches.empty:
        raise ValueError(f"no forecast for sensor {sensor_id!r} in this snapshot")
    forecast_value = float(matches.iloc[0]["forecast_pm25"])

    recent = snapshot.recent.loc[snapshot.recent["sensor_id"] == sensor_id].sort_values("time")
    recent_times = tuple(recent["time"])
    recent_values = tuple(float(v) for v in recent["pm25"])

    tz = recent["time"].dt.tz if len(recent) else "Asia/Karachi"
    forecast_start = pd.Timestamp(snapshot.forecast_date).tz_localize(tz)
    forecast_end = forecast_start + pd.Timedelta(hours=24)

    ceiling = _stripe_ceiling(recent_values, forecast_value)

    return ChartSeries(
        sensor_id=sensor_id,
        recent_times=recent_times,
        recent_values=recent_values,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        forecast_value=forecast_value,
        band_stripes=_band_stripes(ceiling),
    )
