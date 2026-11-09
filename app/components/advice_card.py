"""Renders a `Recommendation` (advice, notes, disclaimer, rule trace) — spec §9, item 5.

The only module that calls `src.expert.explain` from the app side; kept separate from
`app/views/forecast.py` so the forecast view stays about layout, not advice formatting.
"""

from __future__ import annotations

import streamlit as st

from src.expert import Recommendation, explain
from src.ui.format import format_pm25_whole


def render_band_chip(band_name: str, color: str) -> None:
    """A small, colour-coded pill with the band name — the headline reading, at a glance."""
    st.markdown(
        f"<span style='background-color:{color};color:#111111;padding:4px 14px;"
        "border-radius:999px;font-weight:600;font-size:1.05rem;'>"
        f"{band_name}</span>",
        unsafe_allow_html=True,
    )


def render_forecast_value(forecast_pm25: float, band_name: str, color: str, meaning: str) -> None:
    """The big headline number, its band chip, and a one-line plain-English meaning."""
    st.markdown(f"## {format_pm25_whole(forecast_pm25)}")
    render_band_chip(band_name, color)
    st.caption(meaning)


def render_advice(rec: Recommendation) -> None:
    """Main advice, any extra notes, the always-shown disclaimer, and the rule-trace expander."""
    st.markdown(f"**What to do:** {rec.advice}")
    for note in rec.notes:
        st.caption(f"ℹ️ {note}")
    st.info(rec.disclaimer, icon="⚠️")
    with st.expander("You see this because…"):
        st.text(explain(rec))
