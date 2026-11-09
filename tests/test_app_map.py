"""Tests for app/components/map.py — the real plotly figure, not a fake.

The smoke tests replace the map, so a bad plotly property only showed up in the browser
("Map unavailable"). Building and serialising the figure here catches that.
"""

from __future__ import annotations

from app.components.map import SELECTED_SIZE, UNSELECTED_SIZE, build_map_figure, clicked_sensor_id
from src.ui.results import SensorSummary


def _summary(sensor_id: str, has_forecast: bool = True) -> SensorSummary:
    return SensorSummary(
        sensor_id=sensor_id,
        site=f"Site {sensor_id}",
        lat=24.9,
        lon=67.0,
        forecast_pm25=40.0 if has_forecast else None,
        band_name="Unhealthy for Sensitive Groups" if has_forecast else "No forecast today",
        color="#FF7E00" if has_forecast else "#B0B0B0",
        has_forecast=has_forecast,
    )


def test_figure_builds_and_serialises_with_the_selected_sensor_larger():
    summaries = [_summary("S1"), _summary("S2", has_forecast=False)]
    fig = build_map_figure(summaries, "S1")
    fig.to_json()  # raises on any invalid plotly property
    assert list(fig.data[0].marker.size) == [SELECTED_SIZE, UNSELECTED_SIZE]
    assert list(fig.data[0].customdata) == ["S1", "S2"]


def test_clicked_sensor_id_reads_customdata_or_point_index():
    summaries = [_summary("S1"), _summary("S2")]
    assert clicked_sensor_id({"selection": {"points": [{"customdata": "S2"}]}}, summaries) == "S2"
    assert clicked_sensor_id({"selection": {"points": [{"point_index": 1}]}}, summaries) == "S2"
    assert clicked_sensor_id({"selection": {"points": []}}, summaries) is None
    assert clicked_sensor_id(None, summaries) is None
