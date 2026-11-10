"""Tests for app/components/map.py — the real plotly figure, not a fake.

The smoke tests replace the map, so a bad plotly property only showed up in the browser
("Map unavailable"). Building and serialising the figure here catches that.
"""

from __future__ import annotations

from app.components.map import (
    SELECTED_SIZE,
    UNSELECTED_SIZE,
    build_map_figure,
    clicked_sensor_id,
    tiles_reachable,
)
from src.ui.map_mode import OFFLINE_STYLE, ONLINE_STYLE
from src.ui.results import SensorSummary


def _summary(sensor_id: str, has_forecast: bool = True) -> SensorSummary:
    return SensorSummary(
        sensor_id=sensor_id,
        site=f"Site {sensor_id}",
        lat=24.9,
        lon=67.0,
        forecast_pm25=40.0 if has_forecast else None,
        band_name="Unhealthy for Sensitive Groups" if has_forecast else "No forecast today",
        band_index=2 if has_forecast else None,
        color="#FF7E00" if has_forecast else "#B0B0B0",
        has_forecast=has_forecast,
    )


def test_figure_builds_and_serialises_with_the_selected_sensor_larger():
    summaries = [_summary("S1"), _summary("S2", has_forecast=False)]
    fig = build_map_figure(summaries, "S1")
    fig.to_json()  # raises on any invalid plotly property
    assert list(fig.data[0].marker.size) == [SELECTED_SIZE, UNSELECTED_SIZE]
    assert list(fig.data[0].customdata) == ["S1", "S2"]
    assert fig.layout.map.style == ONLINE_STYLE
    assert fig.data[0].mode == "markers"


def test_offline_figure_builds_and_serialises_with_tile_free_style_and_labels():
    summaries = [_summary("S1"), _summary("S2", has_forecast=False)]
    fig = build_map_figure(summaries, "S1", offline=True)
    fig.to_json()  # raises on any invalid plotly property
    assert fig.layout.map.style == OFFLINE_STYLE
    assert fig.data[0].mode == "markers+text"
    assert list(fig.data[0].text) == ["Site S1", "Site S2"]


def test_tiles_reachable_returns_false_quickly_for_an_unroutable_host():
    # A non-routable TEST-NET address (RFC 5737): connecting times out rather than refusing,
    # exercising the same "can't tell, assume offline" path a real outage takes.
    assert tiles_reachable(host="192.0.2.1", port=80, timeout=0.2) is False


def test_tiles_reachable_returns_true_when_the_socket_connects(monkeypatch):
    import contextlib

    import app.components.map as map_mod

    monkeypatch.setattr(map_mod.socket, "create_connection", lambda *a, **k: contextlib.nullcontext())
    assert tiles_reachable() is True


def test_clicked_sensor_id_reads_customdata_or_point_index():
    summaries = [_summary("S1"), _summary("S2")]
    assert clicked_sensor_id({"selection": {"points": [{"customdata": "S2"}]}}, summaries) == "S2"
    assert clicked_sensor_id({"selection": {"points": [{"point_index": 1}]}}, summaries) == "S2"
    assert clicked_sensor_id({"selection": {"points": []}}, summaries) is None
    assert clicked_sensor_id(None, summaries) is None
