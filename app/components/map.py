"""The sensor map component (spec §9, item 2): a plotly scatter map, EPA-coloured by band.

Building the figure needs plotly (so it lives here, in `app/`, not in the Streamlit-free
`src/ui`); the data behind it (`SensorSummary`) is pure and comes from `src.ui.results`.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from src.ui.results import SensorSummary

SELECTED_SIZE = 24
UNSELECTED_SIZE = 12


def build_map_figure(summaries: list[SensorSummary], selected_sensor_id: str | None) -> go.Figure:
    """A plotly `Scattermap` of every registered sensor, coloured by its forecast band.

    The selected sensor is drawn larger (map markers cannot have outlines); a sensor with no forecast today
    is grey (`src.ui.results.NO_FORECAST_COLOR`) so it reads as "nothing to show", not an error.
    """
    sizes = [SELECTED_SIZE if s.sensor_id == selected_sensor_id else UNSELECTED_SIZE for s in summaries]
    texts = [
        f"{s.site}<br>{s.band_name}" + (f": {s.forecast_pm25:.0f} µg/m³" if s.has_forecast else "")
        for s in summaries
    ]

    fig = go.Figure(
        go.Scattermap(
            lat=[s.lat for s in summaries],
            lon=[s.lon for s in summaries],
            mode="markers",
            marker=dict(size=sizes, color=[s.color for s in summaries]),
            text=texts,
            hoverinfo="text",
            customdata=[s.sensor_id for s in summaries],
        )
    )
    fig.update_layout(
        map=dict(
            style="open-street-map",
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
