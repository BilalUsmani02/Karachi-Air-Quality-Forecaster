"""Entry point: `uv run streamlit run app/main.py` (spec §9-10).

Three pages, navigated with `st.navigation`/`st.Page`: Forecast (the default/home page),
Model comparison, and About & data. Page config lives here; everything else is in
`app/views/*.py` and the Streamlit-free helpers in `src/ui/*.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts only app/ on the import path; add the repo root so `app.*` and `src.*` import.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.views import about, compare, forecast  # noqa: E402

st.set_page_config(
    page_title="Karachi Air-Quality Forecaster",
    page_icon="🌫️",
    layout="wide",
)

pages = [
    st.Page(forecast.render, title="Forecast", icon="📈", default=True, url_path="forecast"),
    st.Page(compare.render, title="Model comparison", icon="📊", url_path="compare"),
    st.Page(about.render, title="About & data", icon="ℹ️", url_path="about"),
]

st.navigation(pages).run()
