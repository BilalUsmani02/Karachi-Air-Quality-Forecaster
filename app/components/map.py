"""The sensor map component (spec §9, item 2): a plotly scatter map, EPA-coloured by band.

Building the figure needs plotly (so it lives here, in `app/`, not in the Streamlit-free
`src/ui`); the data behind it (`SensorSummary`) is pure and comes from `src.ui.results`.

`offline=True` (review item 3) swaps `style="open-street-map"`, which needs internet tiles, for
a tile-free style (`src.ui.map_mode.OFFLINE_STYLE`) and adds each site's name as a text label
next to its marker, so the map stays readable with no background at all. Whether to default
that toggle on is decided by `src.ui.map_mode.default_offline_map`, fed by `tiles_reachable`
below — a cheap, timed-out socket probe, never a hard dependency for the map to work.
"""

from __future__ import annotations

import socket
from typing import Any

import plotly.graph_objects as go

from src.ui.map_mode import map_style
from src.ui.results import SensorSummary

SELECTED_SIZE = 24
UNSELECTED_SIZE = 12

TILE_HOST = "tile.openstreetmap.org"
TILE_PORT = 443
PROBE_TIMEOUT_SECONDS = 1.0


def tiles_reachable(
    host: str = TILE_HOST, port: int = TILE_PORT, timeout: float = PROBE_TIMEOUT_SECONDS
) -> bool:
    """A fast, best-effort check: can we open a socket to the map-tile host right now?

    Used only to choose a sensible *default* for the "offline map" toggle — never required to
    work, and never blocks the page for more than `timeout` seconds either way.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_map_figure(
    summaries: list[SensorSummary], selected_sensor_id: str | None, offline: bool = False
) -> go.Figure:
    """A plotly `Scattermap` of every registered sensor, coloured by its forecast band.

    The selected sensor is drawn larger (map markers cannot have outlines); a sensor with no
    forecast today is grey (`src.ui.results.NO_FORECAST_COLOR`) so it reads as "nothing to
    show", not an error. `offline=True` draws a tile-free background plus a site-name label at
    every marker, so the map still reads without internet access.
    """
    sizes = [SELECTED_SIZE if s.sensor_id == selected_sensor_id else UNSELECTED_SIZE for s in summaries]
    texts = [
        f"{s.site}<br>{s.band_name}" + (f": {s.forecast_pm25:.0f} µg/m³" if s.has_forecast else "")
        for s in summaries
    ]
    site_labels = [s.site for s in summaries]

    fig = go.Figure(
        go.Scattermap(
            lat=[s.lat for s in summaries],
            lon=[s.lon for s in summaries],
            mode="markers+text" if offline else "markers",
            marker=dict(size=sizes, color=[s.color for s in summaries]),
            text=site_labels if offline else None,
            textposition="top center",
            hovertext=texts,
            hoverinfo="text",
            customdata=[s.sensor_id for s in summaries],
        )
    )
    fig.update_layout(
        map=dict(
            style=map_style(offline),
            center=dict(lat=sum(s.lat for s in summaries) / len(summaries), lon=sum(s.lon for s in summaries) / len(summaries)),
            zoom=9.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        height=380,
    )
    return fig


def clicked_sensor_id(event: dict[str, Any] | None, summaries: list[SensorSummary]) -> str | None:
    """The sensor ID of a clicked map point, from a `st.plotly_chart(..., on_select="rerun")`
    selection event — or `None` if nothing was clicked (or the event shape is unexpected; the
    map selection is a bonus on top of the dropdown, never required to work)."""
    if not event:
        return None
    try:
        points = event.get("selection", {}).get("points", [])
        if not points:
            return None
        point = points[0]
        custom = point.get("customdata")
        if custom:
            return custom[0] if isinstance(custom, (list, tuple)) else custom
        index = point.get("point_index")
        if index is not None and 0 <= index < len(summaries):
            return summaries[index].sensor_id
    except (AttributeError, TypeError, IndexError, KeyError):
        return None
    return None
