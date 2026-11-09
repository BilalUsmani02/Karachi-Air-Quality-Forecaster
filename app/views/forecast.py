"""The Forecast page (spec §9) — the app's default/home page.

Status banner -> sensor picker (map + dropdown) -> big forecast value + band -> profile
toggles -> advice -> "Why this forecast?" (SHAP) -> 72 h history chart. All the actual logic
lives in `src.ui.*` (pure) and `app.components.*` (plotly figures); this module is just
layout and Streamlit's caching around the slow calls.
"""

from __future__ import annotations

import streamlit as st

from app.components.advice_card import render_advice, render_forecast_value
from app.components.chart import build_history_figure
from app.components.map import build_map_figure, clicked_sensor_id
from app.state import get_background, get_bundle, missing_bundle_message
from src.ui.banner import build_banner
from src.ui.chart_data import build_chart_series
from src.ui.data import get_app_snapshot
from src.ui.explain_ui import build_sensor_explainer, feature_medians, shap_reasons_for_row
from src.ui.format import format_pm25
from src.ui.results import (
    SensorSummary,
    all_sensor_summaries,
    band_meaning,
    build_sensor_result,
    feature_row,
)

SHAP_SEED = 42
SHAP_BACKGROUND_SIZE = 100
SHAP_TOP_K = 3


def render() -> None:
    bundle = get_bundle()
    if bundle is None:
        st.error(missing_bundle_message())
        return

    if "snapshot" not in st.session_state:
        st.session_state["snapshot"] = get_app_snapshot(refresh=False, bundle=bundle)
    snapshot = st.session_state["snapshot"]

    _render_banner(snapshot)

    if st.button("🔄 Refresh live data"):
        with st.spinner("Fetching the latest data…"):
            st.session_state["snapshot"] = get_app_snapshot(refresh=True, bundle=bundle)
        st.rerun()

    st.divider()
    summaries = all_sensor_summaries(snapshot)
    selected_id = _render_sensor_picker(summaries)
    if selected_id is None:
        st.warning("No forecasts are available in this snapshot.")
        return

    chosen = next(s for s in summaries if s.sensor_id == selected_id)
    if not chosen.has_forecast:
        st.warning(f"No forecast is available for **{chosen.site}** today.")
        return

    st.divider()
    sensitive = st.session_state.get("sensitive_profile", False)
    outdoor = st.session_state.get("outdoor_profile", False)
    result = build_sensor_result(snapshot, selected_id, sensitive=sensitive, outdoor_activity=outdoor)

    value_col, profile_col = st.columns([1, 1])
    with value_col:
        st.caption(result.site)
        render_forecast_value(
            result.forecast_pm25, result.band_name, result.color, band_meaning(result.band_name)
        )
        if snapshot.source == "demo" and result.actual_pm25 is not None:
            st.caption(f"What actually happened that day: {format_pm25(result.actual_pm25)}")
    with profile_col:
        _render_profile_toggles()

    st.divider()
    render_advice(result.recommendation)

    st.divider()
    _render_shap_section(bundle, snapshot, selected_id)

    st.divider()
    _render_chart(snapshot, selected_id)


def _render_banner(snapshot) -> None:
    banner = build_banner(snapshot)
    st.info(f"**{banner.heading}**  \n{banner.last_updated}  \n{banner.source_text}")
    for warning in banner.warnings:
        st.warning(warning)


def _render_sensor_picker(summaries: list[SensorSummary]) -> str | None:
    """Map + dropdown, kept in sync through the shared `selected_sensor` session-state key."""
    available = [s for s in summaries if s.has_forecast]
    if not available:
        return None

    known_ids = {s.sensor_id for s in summaries}
    if st.session_state.get("selected_sensor") not in known_ids:
        st.session_state["selected_sensor"] = available[0].sensor_id

    st.caption("Click a dot on the map, or use the dropdown below, to choose a site.")
    try:
        fig = build_map_figure(summaries, st.session_state["selected_sensor"])
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="sensor_map")
        clicked = clicked_sensor_id(event, summaries)
        clicked_summary = next((s for s in summaries if s.sensor_id == clicked), None)
        if clicked_summary is not None and clicked_summary.has_forecast:
            st.session_state["selected_sensor"] = clicked
    except Exception:  # pragma: no cover - defensive: map selection is a bonus, never required
        st.caption("Map unavailable right now — use the dropdown below to pick a site.")

    labels = {s.sensor_id: s.site if s.has_forecast else f"{s.site} (no forecast today)" for s in summaries}
    return st.selectbox(
        "Choose a monitoring site",
        options=[s.sensor_id for s in summaries],
        format_func=lambda sid: labels[sid],
        key="selected_sensor",
    )


def _render_profile_toggles() -> None:
    st.checkbox(
        "I'm in a sensitive group (children, older adults, heart/lung conditions, pregnancy)",
        key="sensitive_profile",
    )
    st.checkbox("I have outdoor plans", key="outdoor_profile")
    st.caption(
        "These two choices stay on this screen only, for this session — never saved to disk, "
        "logs, or the cache."
    )


def _render_shap_section(bundle, snapshot, sensor_id: str) -> None:
    st.subheader("Why this forecast?")
    background = get_background()
    explainer = _cached_explainer(bundle, background)
    medians = feature_medians(background)
    row_raw = feature_row(snapshot, sensor_id)

    with st.spinner("Working out the top reasons…"):
        reasons = _cached_reasons(
            bundle, explainer, row_raw, medians, snapshot.forecast_date, sensor_id, SHAP_TOP_K
        )

    for reason in reasons:
        st.markdown(f"- {reason}")


def _render_chart(snapshot, sensor_id: str) -> None:
    st.subheader("Last 72 hours and tomorrow's forecast")
    series = build_chart_series(snapshot, sensor_id)
    st.plotly_chart(build_history_figure(series), width="stretch")


@st.cache_resource(show_spinner=False)
def _cached_explainer(_bundle, _background):
    """One SHAP explainer per app process — building it is the expensive, reusable part."""
    return build_sensor_explainer(_bundle, _background, seed=SHAP_SEED, background_size=SHAP_BACKGROUND_SIZE)


@st.cache_data(show_spinner=False)
def _cached_reasons(_bundle, _explainer, _row_raw, _medians, forecast_date, sensor_id: str, k: int):
    """Per-sensor SHAP reasons, cached by (forecast_date, sensor_id) — a new day or a
    different sensor recomputes; re-selecting the same sensor on the same day doesn't."""
    return shap_reasons_for_row(_explainer, _bundle, _row_raw, _medians, k=k)
