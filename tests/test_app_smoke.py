"""Smoke tests for the three app pages (spec §9): each renders without an exception, with key
text visible, against a fake snapshot and a fake SHAP pipeline — never the real model or a
real `shap` run (slow, and not what this test is checking).

`st.navigation`'s callable-only pages have no backing file, so `AppTest.switch_page` cannot
reach them (it only matches registered *script* paths — see `streamlit/testing/v1/app_test.py
::_resolve_page_hash`). Each page's `render()` is tested directly with `AppTest.from_function`
instead, which is equivalent for a page that is just one function with no navigation chrome of
its own.

`AppTest.from_function` re-executes the function's *source text* as an isolated script, so
every name it needs must be imported inside the function body — nothing from this module's
top level is visible to it.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _run_forecast_page():
    from datetime import date

    import pandas as pd

    import app.state as state
    import app.views.forecast as forecast
    import streamlit as st
    from src.live import snapshot as snap
    from src.models.dataset import FEATURE_COLUMNS
    from tests.conftest import FAKE_BUNDLE, fake_features, fake_hourly

    features = fake_features("2026-11-08", ["S1", "S3", "S5"])
    hourly = fake_hourly(["S1", "S3", "S5"], "2026-11-05 00:00", 72)
    snapshot = snap.Snapshot(
        forecast_date=date(2026, 11, 8),
        fetched_at="2026-11-08T01:00:00+00:00",
        source="demo",
        forecasts=snap.forecast_frame(FAKE_BUNDLE, features, actual=features["y_pm25"]),
        recent=snap.recent_readings(hourly, date(2026, 11, 8)),
        warnings=("Demo mode: showing a past day from the stored data, not a live forecast.",),
        model_trained_on=("2025-06-20", "2026-11-08"),
    )
    st.session_state["snapshot"] = snapshot

    # Replace every slow/real dependency with a fast fake — this test checks *layout and
    # wiring*, never the real model, the real cache, or a real SHAP run.
    fake_background = pd.DataFrame({c: [0.0, 1.0] for c in FEATURE_COLUMNS})
    state.get_bundle = lambda: FAKE_BUNDLE  # type: ignore[assignment]
    state.get_background = lambda: fake_background  # type: ignore[assignment]
    forecast.get_bundle = state.get_bundle
    forecast.get_background = state.get_background
    forecast._cached_explainer = lambda bundle, background: object()
    forecast._cached_reasons = lambda *args, **kwargs: ["A fake reason for testing."]

    forecast.render()


def _run_forecast_page_with_a_usg_site():
    from datetime import date

    import pandas as pd

    import app.state as state
    import app.views.forecast as forecast
    import streamlit as st
    from src.live import snapshot as snap
    from src.models.dataset import FEATURE_COLUMNS
    from tests.conftest import FAKE_BUNDLE, fake_features, fake_hourly

    features = fake_features("2026-11-08", ["S1", "S3", "S5"])
    hourly = fake_hourly(["S1", "S3", "S5"], "2026-11-05 00:00", 72)
    snapshot = snap.Snapshot(
        forecast_date=date(2026, 11, 8),
        fetched_at="2026-11-08T01:00:00+00:00",
        source="demo",
        forecasts=snap.forecast_frame(FAKE_BUNDLE, features, actual=features["y_pm25"]),
        recent=snap.recent_readings(hourly, date(2026, 11, 8)),
        warnings=("Demo mode: showing a past day from the stored data, not a live forecast.",),
        model_trained_on=("2025-06-20", "2026-11-08"),
    )
    st.session_state["snapshot"] = snapshot
    st.session_state["selected_sensor"] = "S5"  # S5's forecast (51) is USG -> triggers the alert

    fake_background = pd.DataFrame({c: [0.0, 1.0] for c in FEATURE_COLUMNS})
    state.get_bundle = lambda: FAKE_BUNDLE  # type: ignore[assignment]
    state.get_background = lambda: fake_background  # type: ignore[assignment]
    forecast.get_bundle = state.get_bundle
    forecast.get_background = state.get_background
    forecast._cached_explainer = lambda bundle, background: object()
    forecast._cached_reasons = lambda *args, **kwargs: ["A fake reason for testing."]

    forecast.render()


def _run_forecast_page_with_a_stale_cached_snapshot():
    from datetime import date

    import pandas as pd

    import app.state as state
    import app.views.forecast as forecast
    import streamlit as st
    from src.live import snapshot as snap
    from src.models.dataset import FEATURE_COLUMNS
    from tests.conftest import FAKE_BUNDLE, fake_features, fake_hourly

    # A date in the far past (not "yesterday") so this stays true regardless of the real clock.
    features = fake_features("2020-01-01", ["S1", "S3", "S5"])
    hourly = fake_hourly(["S1", "S3", "S5"], "2019-12-29 00:00", 72)
    snapshot = snap.Snapshot(
        forecast_date=date(2020, 1, 1),
        fetched_at="2020-01-01T01:00:00+00:00",
        source="cache",
        forecasts=snap.forecast_frame(FAKE_BUNDLE, features, actual=features["y_pm25"]),
        recent=snap.recent_readings(hourly, date(2020, 1, 1)),
        warnings=(),
        model_trained_on=("2025-06-20", "2026-11-08"),
    )
    st.session_state["snapshot"] = snapshot

    fake_background = pd.DataFrame({c: [0.0, 1.0] for c in FEATURE_COLUMNS})
    state.get_bundle = lambda: FAKE_BUNDLE  # type: ignore[assignment]
    state.get_background = lambda: fake_background  # type: ignore[assignment]
    forecast.get_bundle = state.get_bundle
    forecast.get_background = state.get_background
    forecast._cached_explainer = lambda bundle, background: object()
    forecast._cached_reasons = lambda *args, **kwargs: ["A fake reason for testing."]

    forecast.render()


def _run_forecast_page_for_the_demo_toggle():
    from datetime import date

    import pandas as pd

    import app.state as state
    import app.views.forecast as forecast
    import streamlit as st
    from src.live import snapshot as snap
    from src.models.dataset import FEATURE_COLUMNS
    from tests.conftest import FAKE_BUNDLE, fake_features, fake_hourly

    # The "current" (cached) snapshot, already in session_state before the toggle is touched.
    live_features = fake_features("2026-11-08", ["S1", "S3", "S5"])
    live_hourly = fake_hourly(["S1", "S3", "S5"], "2026-11-05 00:00", 72)
    live_snapshot = snap.Snapshot(
        forecast_date=date(2026, 11, 8),
        fetched_at="2026-11-08T01:00:00+00:00",
        source="cache",
        forecasts=snap.forecast_frame(FAKE_BUNDLE, live_features, actual=live_features["y_pm25"]),
        recent=snap.recent_readings(live_hourly, date(2026, 11, 8)),
        warnings=(),
        model_trained_on=("2025-06-20", "2026-11-08"),
    )
    st.session_state["snapshot"] = live_snapshot

    # The snapshot the (faked) demo picker/loader hands back once the toggle is switched on.
    demo_features = fake_features("2026-11-01", ["S1", "S3", "S5"])
    demo_hourly = fake_hourly(["S1", "S3", "S5"], "2026-10-29 00:00", 72)
    demo_snap = snap.Snapshot(
        forecast_date=date(2026, 11, 1),
        fetched_at="2026-11-01T01:00:00+00:00",
        source="demo",
        forecasts=snap.forecast_frame(FAKE_BUNDLE, demo_features, actual=demo_features["y_pm25"]),
        recent=snap.recent_readings(demo_hourly, date(2026, 11, 1)),
        warnings=("Demo mode: showing a past day from the stored data, not a live forecast.",),
        model_trained_on=("2025-06-20", "2026-11-08"),
    )

    fake_background = pd.DataFrame({c: [0.0, 1.0] for c in FEATURE_COLUMNS})
    state.get_bundle = lambda: FAKE_BUNDLE  # type: ignore[assignment]
    state.get_background = lambda: fake_background  # type: ignore[assignment]
    forecast.get_bundle = state.get_bundle
    forecast.get_background = state.get_background
    forecast._cached_explainer = lambda bundle, background: object()
    forecast._cached_reasons = lambda *args, **kwargs: ["A fake reason for testing."]
    # Never touches the real parquet file or the real model: a fast, deterministic stand-in.
    forecast._cached_demo_day_options = lambda: ([date(2026, 11, 1)], date(2026, 11, 1))
    forecast._cached_demo_snapshot = lambda bundle, day: demo_snap

    forecast.render()


def _run_compare_page():
    from app.views import compare

    compare.render()


def _run_about_page():
    import app.state as state
    import app.views.about as about

    state.get_bundle = lambda: {"trained_on": ["2025-06-20 00:00:00", "2026-11-08 00:00:00"]}  # type: ignore[assignment]
    about.get_bundle = state.get_bundle

    about.render()


def test_forecast_page_renders_without_error_and_shows_key_text():
    at = AppTest.from_function(_run_forecast_page, default_timeout=60)
    at.run()

    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
    assert "Forecast for" in page_text
    assert "issued 00:00 PKT" in page_text
    assert "A fake reason for testing." in page_text


def test_forecast_page_shows_the_profile_privacy_note():
    at = AppTest.from_function(_run_forecast_page, default_timeout=60)
    at.run()

    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "never saved to disk" in captions


def test_forecast_page_shows_a_health_alert_for_a_usg_site():
    at = AppTest.from_function(_run_forecast_page_with_a_usg_site, default_timeout=60)
    at.run()

    assert not at.exception
    alert_text = " ".join(w.value for w in at.warning) + " ".join(e.value for e in at.error)
    assert "Health alert" in alert_text
    assert "Unhealthy for Sensitive Groups" in alert_text
    assert "Informational only" in alert_text
    assert "not an official alert" in alert_text


def test_forecast_page_shows_a_stale_warning_for_an_old_cached_snapshot():
    at = AppTest.from_function(_run_forecast_page_with_a_stale_cached_snapshot, default_timeout=60)
    at.run()

    assert not at.exception
    warnings_text = " ".join(w.value for w in at.warning)
    assert "not today" in warnings_text
    assert "Refresh live data" in warnings_text
    assert "demo mode" in warnings_text


def test_demo_mode_toggle_switches_the_page_to_a_demo_snapshot():
    at = AppTest.from_function(_run_forecast_page_for_the_demo_toggle, default_timeout=60)
    at.run()

    assert not at.exception
    info_before = " ".join(i.value for i in at.info)
    assert "Cached data" in info_before

    at.sidebar.checkbox(key="demo_mode").check()
    at.run()

    assert not at.exception
    info_after = " ".join(i.value for i in at.info)
    assert "Demo mode (offline sample from stored history)" in info_after

    # Turning it back off restores the pre-demo (cached) snapshot, not a re-fetch.
    at.sidebar.checkbox(key="demo_mode").uncheck()
    at.run()

    assert not at.exception
    info_restored = " ".join(i.value for i in at.info)
    assert "Cached data" in info_restored


def test_turning_demo_mode_off_restores_the_site_picked_before_the_demo():
    at = AppTest.from_function(_run_forecast_page_for_the_demo_toggle, default_timeout=60)
    at.run()
    at.selectbox(key="selected_sensor").select("S1")
    at.run()

    at.sidebar.checkbox(key="demo_mode").check()
    at.run()
    at.selectbox(key="selected_sensor").select("S5")
    at.run()
    assert at.session_state["selected_sensor"] == "S5"

    at.sidebar.checkbox(key="demo_mode").uncheck()
    at.run()

    assert not at.exception
    assert at.session_state["selected_sensor"] == "S1"
    assert at.selectbox(key="selected_sensor").value == "S1"


def test_compare_page_renders_without_error_and_shows_key_text():
    at = AppTest.from_function(_run_compare_page, default_timeout=60)
    at.run()

    assert not at.exception
    titles = " ".join(t.value for t in at.title)
    page_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Model comparison" in titles
    assert "Gradient Boosting" in page_text


def test_about_page_renders_without_error_and_shows_the_disclaimer():
    at = AppTest.from_function(_run_about_page, default_timeout=60)
    at.run()

    assert not at.exception
    info_text = " ".join(i.value for i in at.info)
    assert "not" in info_text.lower() and "medical advice" in info_text.lower()
    about_text = " ".join(m.value for m in at.markdown)
    assert "20 June 2025" in about_text and "00:00:00" not in about_text
