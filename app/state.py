"""Shared, cached resources used by more than one page: the model bundle and the SHAP
background sample. Both are loaded once per app process (`st.cache_resource`) and reused by
`app/views/forecast.py` and `app/views/about.py`.

Kept as plain module-level names (not wrapped in a class) so tests can monkeypatch
`app.state.load_bundle` / `app.state.load_background` directly, then call
`get_bundle.clear()` / `get_background.clear()` to drop any stale cached value from an
earlier test.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.live.snapshot import BUNDLE_PATH, load_background, load_bundle


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict[str, Any] | None:
    """The production model bundle, or `None` if `models/final_ensemble.pkl` is missing."""
    try:
        return load_bundle()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_background():
    """A fixed-seed sample of raw training rows, for the SHAP background and feature medians."""
    return load_background()


def missing_bundle_message() -> str:
    return (
        f"No trained model found at `{BUNDLE_PATH}`. Run "
        "`uv run python -m src.evaluate.final_test --save-production` to create it, then "
        "reload this page."
    )
