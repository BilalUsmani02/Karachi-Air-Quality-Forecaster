"""The 72-hour history + forecast chart component (spec §9, item 7).

Takes a `ChartSeries` (pure data, from `src.ui.chart_data`) and draws it: the measured line,
the forecast drawn as a flat segment across day D's 24 hours, and faint EPA band stripes
behind both so the reader can see at a glance which band the numbers sit in.
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.ui.chart_data import ChartSeries

MEASURED_COLOR = "#1f77b4"
FORECAST_COLOR = "#d62728"


def build_history_figure(series: ChartSeries) -> go.Figure:
    """A plotly figure: last 72 h measured PM2.5, plus tomorrow's forecast as a flat segment."""
    fig = go.Figure()

    for stripe in series.band_stripes:
        fig.add_hrect(
            y0=stripe.low,
            y1=stripe.high,
            fillcolor=stripe.color,
            opacity=0.15,
            line_width=0,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=list(series.recent_times),
            y=list(series.recent_values),
            mode="lines",
            name="Last 72 hours (measured)",
            line=dict(color=MEASURED_COLOR, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[series.forecast_start, series.forecast_end],
            y=[series.forecast_value, series.forecast_value],
            mode="lines",
            name="Tomorrow's forecast (one number for the whole day)",
            line=dict(color=FORECAST_COLOR, width=4, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="Time (Pakistan Standard Time)",
        yaxis_title="PM2.5 (µg/m³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
    )
    return fig
