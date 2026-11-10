"""The Forecast page (spec §9) — the app's default/home page.

Status banner -> sensor picker (map + dropdown) -> big forecast value + band -> profile
toggles -> advice -> "Why this forecast?" (SHAP) -> 72 h history chart. All the actual logic
lives in `src.ui.*` (pure) and `app.components.*` (plotly figures); this module is just
layout and Streamlit's caching around the slow calls.

Two sidebar toggles added in the review pass: "Demo mode" (an offline example day, for a
demo with no internet — review item 2b) and "Offline map" (a tile-free map style, defaulted
smartly — review item 3). `_sync_demo_mode` keeps `st.session_state["snapshot"]` as the one key
every other function reads, swapping in/out a demo snapshot without disturbing the
live/cached one underneath.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from app.components.advice_card import render_advice, render_forecast_value
from app.components.chart import build_history_figure
from app.components.map import build_map_figure, clicked_sensor_id, tiles_reachable
from app.state import get_background, get_bundle, missing_bundle_message
from src.config import PROCESSED_DIR
from src.ui.banner import build_banner
from src.ui.chart_data import build_chart_series
from src.ui.data import get_app_snapshot
from src.ui.demo_day import available_demo_days, default_demo_day
from src.ui.format import format_date_long, format_pm25
from src.ui.health_alert import citywide_alert, site_health_alert
from src.ui.map_mode import default_offline_map
from src.ui.results import (
    SensorResult,
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

    demo_mode = st.sidebar.checkbox("Demo mode (offline example day)", key="demo_mode")
    _sync_demo_mode(bundle, demo_mode)
    snapshot = st.session_state["snapshot"]

    summaries = all_sensor_summaries(snapshot)
    _render_banner(snapshot, summaries)

    if demo_mode:
        st.caption("Demo mode is on — turn it off above to refresh live data.")
    elif st.button("🔄 Refresh live data"):
        with st.spinner("Fetching the latest data…"):
            st.session_state["snapshot"] = get_app_snapshot(refresh=True, bundle=bundle)
        st.rerun()

    st.divider()
    offline_map = _render_offline_map_toggle()
    selected_id = _render_sensor_picker(summaries, offline_map)
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

    _render_site_health_alert(result)

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


def _render_banner(snapshot, summaries: list[SensorSummary]) -> None:
    banner = build_banner(snapshot)
    st.info(f"**{banner.heading}**  \n{banner.last_updated}  \n{banner.source_text}")
    for warning in banner.warnings:
        st.warning(warning)

    citywide = citywide_alert([s.band_index for s in summaries if s.has_forecast])
    if citywide is not None:
        st.warning(citywide)

    if banner.sensor_notes:
        count = len(banner.sensor_notes)
        with st.expander(f"{count} site{'s' if count != 1 else ''} without a forecast today"):
            st.markdown("\n".join(f"- {note}" for note in banner.sensor_notes))


def _render_site_health_alert(result: SensorResult) -> None:
    """A prominent, worded-as-guidance alert once the selected site's band reaches USG+."""
    alert = site_health_alert(result.band_index, result.band_name, result.site)
    if alert is None:
        return
    show = st.error if alert.level == "error" else st.warning
    show(alert.message)


def _sync_demo_mode(bundle, demo_mode: bool) -> None:
    """Switches `st.session_state["snapshot"]` to/from a demo day as the sidebar toggle flips.

    `"snapshot"` stays the one key every other function (and the AppTest smoke tests) read;
    turning demo mode on stashes whatever was there first, turning it off restores that stash
    rather than re-fetching.
    """
    was_demo = st.session_state.get("_was_demo", False)
    if demo_mode and not was_demo:
        st.session_state["_pre_demo_snapshot"] = st.session_state["snapshot"]
    if demo_mode:
        day = _render_demo_day_picker(bundle)
        st.session_state["snapshot"] = _cached_demo_snapshot(bundle, day)
    elif was_demo:
        st.session_state["snapshot"] = st.session_state.pop("_pre_demo_snapshot", st.session_state["snapshot"])
    st.session_state["_was_demo"] = demo_mode


def _render_demo_day_picker(bundle) -> date | None:
    """A sidebar day picker limited to days actually present in the features table."""
    days, default_day = _cached_demo_day_options()
    if not days:
        return None
    default_day = default_day if default_day in days else days[-1]

    keys = [d.isoformat() for d in days]
    labels = {d.isoformat(): format_date_long(d) for d in days}
    chosen_key = st.sidebar.selectbox(
        "Demo day",
        options=keys,
        format_func=lambda k: labels[k],
        index=keys.index(default_day.isoformat()),
        key="demo_day",
    )
    return days[keys.index(chosen_key)]


def _render_offline_map_toggle() -> bool:
    """The "Offline map" sidebar toggle; its default comes from `src.ui.map_mode`.

    The tile-host socket probe (2 s at most) runs once per session, whatever the data source.
    """
    probe = st.session_state.get("_tiles_reachable")
    if probe is None:
        probe = tiles_reachable()
        st.session_state["_tiles_reachable"] = probe
    default = default_offline_map(probe)
    return st.sidebar.checkbox("Offline map (no background tiles)", value=default, key="offline_map")


def _render_sensor_picker(summaries: list[SensorSummary], offline_map: bool) -> str | None:
    """Map + dropdown, kept in sync through the shared `selected_sensor` session-state key."""
    available = [s for s in summaries if s.has_forecast]
    if not available:
        return None

    known_ids = {s.sensor_id for s in summaries}
    if st.session_state.get("selected_sensor") not in known_ids:
        st.session_state["selected_sensor"] = available[0].sensor_id

    st.caption("Click a dot on the map, or use the dropdown below, to choose a site.")
    try:
        fig = build_map_figure(summaries, st.session_state["selected_sensor"], offline=offline_map)
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="sensor_map")
        clicked = clicked_sensor_id(event, summaries)
        clicked_summary = next((s for s in summaries if s.sensor_id == clicked), None)
        if clicked_summary is not None and clicked_summary.has_forecast:
            st.session_state["selected_sensor"] = clicked
    except Exception as exc:  # defensive: map selection is a bonus, never required
        st.caption(f"Map unavailable right now — use the dropdown below to pick a site. ({exc})")

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
    with st.spinner("Working out the top reasons…"):
        # Imported here, not at the top: `shap` takes ~12 s to import, and the forecast and
        # advice above should appear without waiting for it.
        from src.ui.explain_ui import feature_medians

        background = get_background()
        explainer = _cached_explainer(bundle, background)
        medians = feature_medians(background)
        row_raw = feature_row(snapshot, sensor_id)
        reasons = _cached_reasons(
            bundle, explainer, row_raw, medians, snapshot.forecast_date, sensor_id, SHAP_TOP_K
        )

    for reason in reasons:
        st.markdown(f"- {reason}")


def _render_chart(snapshot, sensor_id: str) -> None:
    st.subheader("Last 72 hours and the forecast")
    series = build_chart_series(snapshot, sensor_id)
    st.plotly_chart(build_history_figure(series), width="stretch")


@st.cache_resource(show_spinner=False)
def _cached_explainer(_bundle, _background):
    """One SHAP explainer per app process — building it is the expensive, reusable part."""
    from src.ui.explain_ui import build_sensor_explainer

    return build_sensor_explainer(_bundle, _background, seed=SHAP_SEED, background_size=SHAP_BACKGROUND_SIZE)


@st.cache_data(show_spinner=False)
def _cached_reasons(_bundle, _explainer, _row_raw, _medians, forecast_date, sensor_id: str, k: int):
    """Per-sensor SHAP reasons, cached by (forecast_date, sensor_id) — a new day or a
    different sensor recomputes; re-selecting the same sensor on the same day doesn't."""
    from src.ui.explain_ui import shap_reasons_for_row

    return shap_reasons_for_row(_explainer, _bundle, _row_raw, _medians, k=k)


@st.cache_data(show_spinner=False)
def _cached_demo_day_options() -> tuple[list[date], date | None]:
    """Every day in the features table, and the pure-logic default day — read once per process,
    entirely offline (no model needed to pick the day, only to forecast it)."""
    import pandas as pd

    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    return available_demo_days(features), default_demo_day(features)


@st.cache_data(show_spinner=False)
def _cached_demo_snapshot(_bundle, day: date | None):
    """The demo snapshot for `day` — cached so flipping back to a previously-seen day is instant."""
    from src.live.snapshot import demo_snapshot

    return demo_snapshot(_bundle, forecast_date=day)
