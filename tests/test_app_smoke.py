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


def _run_compare_page():
    from app.views import compare

    compare.render()


def _run_about_page():
    import app.state as state
    import app.views.about as about

    state.get_bundle = lambda: {"trained_on": ["2025-06-20", "2026-11-08"]}  # type: ignore[assignment]
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
    assert "2025-06-20" in " ".join(m.value for m in at.markdown)
