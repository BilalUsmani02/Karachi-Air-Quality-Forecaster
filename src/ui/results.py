"""Per-sensor results: forecast + band + recommendation, built from a `Snapshot` (spec §9-10).

Two shapes, for two different jobs:

- `SensorSummary` — one row per *registered* sensor (`src.data.sensors.SENSORS`, always 14),
  for the picker and the map. A sensor the snapshot has no forecast for still gets a row, so
  the map can grey it out and the dropdown can say "no forecast today" instead of just
  omitting it.
- `SensorResult` — the full detail for the *selected* sensor only: the forecast, the band, and
  the expert system's `Recommendation` for the current profile toggles.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.classify.bands import band_name, pm25_to_band
from src.data.sensors import SENSORS
from src.expert import Recommendation, recommend
from src.expert.engine import EPA_COLORS
from src.live.snapshot import Snapshot
from src.ui.format import is_missing

NO_FORECAST_COLOR = "#B0B0B0"  # grey — a sensor with no row in today's snapshot
NO_FORECAST_LABEL = "No forecast today"

# One plain-English line per EPA band (spec §9, item 3) — what the number means, not just
# what it's called. Paraphrased from the same EPA Air Quality Guide the expert system quotes
# (docs/report/knowledge_sources.md), kept short enough to sit under the big forecast number.
BAND_MEANINGS: dict[str, str] = {
    "Good": "Air quality is good — no health concerns for anyone.",
    "Moderate": "Air quality is acceptable; unusually sensitive people may notice minor effects.",
    "Unhealthy for Sensitive Groups": (
        "People with heart/lung conditions, children and older adults should reduce prolonged "
        "or heavy outdoor exertion."
    ),
    "Unhealthy": "Everyone may begin to notice health effects; sensitive groups more seriously.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
    "Hazardous": "Health emergency: everyone should avoid outdoor exertion.",
}


def band_meaning(name: str) -> str:
    """The one-line plain-English meaning of a band name, or `""` for an unknown name."""
    return BAND_MEANINGS.get(name, "")


@dataclass(frozen=True)
class SensorSummary:
    """One line per registered sensor — enough to draw the map and the dropdown."""

    sensor_id: str
    site: str
    lat: float
    lon: float
    forecast_pm25: float | None  # None if this sensor has no row in the snapshot
    band_name: str  # band name, or NO_FORECAST_LABEL
    color: str  # EPA band colour, or NO_FORECAST_COLOR
    has_forecast: bool


@dataclass(frozen=True)
class SensorResult:
    """The full detail for one sensor's forecast: value, band, and personalised advice."""

    sensor_id: str
    site: str
    lat: float
    lon: float
    forecast_pm25: float
    prev_day_pm25: float | None
    actual_pm25: float | None
    band_index: int
    band_name: str
    color: str
    recommendation: Recommendation


def _nan_to_none(value: float) -> float | None:
    return None if is_missing(value) else float(value)


def all_sensor_summaries(snapshot: Snapshot) -> list[SensorSummary]:
    """One `SensorSummary` per registered sensor, in registry order.

    Sensors absent from `snapshot.forecasts` (no reading today) still get a row, coloured grey
    and labelled "No forecast today" — the map and dropdown show all 14 sites either way.
    """
    by_id = {row.sensor_id: row for _, row in snapshot.forecasts.iterrows()}
    summaries = []
    for sensor in SENSORS:
        row = by_id.get(sensor.id)
        if row is None:
            summaries.append(
                SensorSummary(
                    sensor_id=sensor.id,
                    site=sensor.site,
                    lat=sensor.lat,
                    lon=sensor.lon,
                    forecast_pm25=None,
                    band_name=NO_FORECAST_LABEL,
                    color=NO_FORECAST_COLOR,
                    has_forecast=False,
                )
            )
            continue
        band_idx = pm25_to_band(float(row.forecast_pm25))
        summaries.append(
            SensorSummary(
                sensor_id=sensor.id,
                site=sensor.site,
                lat=sensor.lat,
                lon=sensor.lon,
                forecast_pm25=float(row.forecast_pm25),
                band_name=band_name(band_idx),
                color=EPA_COLORS[band_idx],
                has_forecast=True,
            )
        )
    return summaries


def available_sensor_ids(snapshot: Snapshot) -> list[str]:
    """Sensor IDs that have a forecast today, in registry order."""
    available = set(snapshot.forecasts["sensor_id"])
    return [sensor.id for sensor in SENSORS if sensor.id in available]


def build_sensor_result(
    snapshot: Snapshot,
    sensor_id: str,
    sensitive: bool = False,
    outdoor_activity: bool = False,
) -> SensorResult:
    """The full result for `sensor_id`: forecast, band, and advice for the given profile.

    Raises `ValueError` if `sensor_id` has no row in `snapshot.forecasts` — callers should
    check `available_sensor_ids`/`SensorSummary.has_forecast` first.
    """
    matches = snapshot.forecasts.loc[snapshot.forecasts["sensor_id"] == sensor_id]
    if matches.empty:
        raise ValueError(f"no forecast for sensor {sensor_id!r} in this snapshot")
    row = matches.iloc[0]

    forecast_pm25 = float(row["forecast_pm25"])
    prev_day_pm25 = _nan_to_none(row["prev_day_pm25"])
    actual_pm25 = _nan_to_none(row["actual_pm25"])
    band_idx = pm25_to_band(forecast_pm25)

    recommendation = recommend(
        forecast_pm25=forecast_pm25,
        today_pm25=prev_day_pm25,
        sensitive=sensitive,
        outdoor_activity=outdoor_activity,
    )

    return SensorResult(
        sensor_id=sensor_id,
        site=str(row["site"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        forecast_pm25=forecast_pm25,
        prev_day_pm25=prev_day_pm25,
        actual_pm25=actual_pm25,
        band_index=band_idx,
        band_name=band_name(band_idx),
        color=EPA_COLORS[band_idx],
        recommendation=recommendation,
    )


def feature_row(snapshot: Snapshot, sensor_id: str) -> pd.Series:
    """The raw (unscaled) feature row for `sensor_id`, for the SHAP explainer.

    `META_COLUMNS` are dropped first, so the returned Series is exactly
    `src.models.dataset.FEATURE_COLUMNS`, in order — what `explain_row` expects as its raw row.
    """
    from src.live.snapshot import META_COLUMNS

    matches = snapshot.forecasts.loc[snapshot.forecasts["sensor_id"] == sensor_id]
    if matches.empty:
        raise ValueError(f"no forecast for sensor {sensor_id!r} in this snapshot")
    return matches.iloc[0].drop(labels=list(META_COLUMNS))


def default_sensor_id(snapshot: Snapshot) -> str | None:
    """The first available sensor (registry order), or `None` if the snapshot is empty."""
    ids = available_sensor_ids(snapshot)
    return ids[0] if ids else None


def sensor_site_name(sensor_id: str) -> str:
    """The registered site name for a sensor ID, for picker labels."""
    from src.data.sensors import sensor_by_id

    return sensor_by_id(sensor_id).site
