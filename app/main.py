"""Entry point: `uv run streamlit run app/main.py` (spec §9-10).

Three pages, navigated with `st.navigation`/`st.Page`: Forecast (the default/home page),
Model comparison, and About & data. Page config lives here; everything else is in
`app/views/*.py` and the Streamlit-free helpers in `src/ui/*.py`.
"""

from __future__ import annotations

import streamlit as st

from app.views import about, compare, forecast

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
