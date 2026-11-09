"""The Model comparison page (spec §9) — final-test numbers, for a non-technical reader.

Every table/chart reads a file already written by an earlier phase (`src.ui.compare`); this
module never re-trains or re-scores anything, so what it shows always matches the frozen
final-test run.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.ui.compare import (
    FAIRNESS_CAVEAT,
    NUMPY_VS_PYTORCH_NOTE,
    PRODUCTION_MODEL_LABEL,
    band_route_table,
    fairness_per_sensor,
    final_test_metrics,
    global_shap_importance,
    mae_bar_data,
)


def render() -> None:
    st.title("Model comparison")
    st.caption(f"This app forecasts with: **{PRODUCTION_MODEL_LABEL}**.")
    st.write(
        "Numbers below are from the one-time final test (`results/final_test/`, "
        "1 September – 8 November 2026) — scored once, after every tuning choice was frozen, "
        "so it reflects how the models do on data they never saw during development."
    )

    _render_metrics_section()
    _render_bands_section()
    _render_numpy_vs_pytorch_section()
    _render_shap_section()
    _render_fairness_section()


def _render_metrics_section() -> None:
    st.subheader("Forecast error")
    st.caption(
        "Mean absolute error (MAE): how far off the forecast was, on average, in µg/m³ — "
        "lower is better. \"Skill\" is how much better a model did than simply guessing "
        "\"tomorrow will look like yesterday\" (persistence)."
    )
    metrics = final_test_metrics()
    if metrics is None:
        st.warning("No final-test results found yet — run `uv run python -m src.evaluate.final_test`.")
        return

    st.dataframe(metrics, hide_index=True, width="stretch")
    bars = mae_bar_data(metrics)
    fig = go.Figure(go.Bar(x=bars["model"], y=bars["mae"], marker_color="#1f77b4"))
    fig.update_layout(
        yaxis_title="MAE (µg/m³, lower is better)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    st.plotly_chart(fig, width="stretch")


def _render_bands_section() -> None:
    st.subheader("Air-quality band accuracy")
    st.caption(
        "Band accuracy: how often the forecast landed in the right air-quality band "
        "(Good → Hazardous). \"Unhealthy days missed\": a truly Unhealthy-or-worse day that "
        "was forecast as cleaner than it really was — the error that matters most for health. "
        "\"False alarms\": the opposite — a cleaner day forecast as Unhealthy-or-worse."
    )
    bands = band_route_table()
    if bands is None:
        st.warning("No band-accuracy results found yet.")
        return
    st.dataframe(bands, hide_index=True, width="stretch")


def _render_numpy_vs_pytorch_section() -> None:
    st.subheader("From-scratch NumPy network vs. its PyTorch twin")
    st.write(NUMPY_VS_PYTORCH_NOTE)


def _render_shap_section() -> None:
    st.subheader("What the model relies on most (SHAP)")
    st.caption(
        "Mean |SHAP value|: across many forecasts, on average how many µg/m³ this factor "
        "pushed the forecast up or down (either direction can count) — a bigger bar means "
        "more influence on the forecast, not necessarily a \"worse\" factor."
    )
    importance = global_shap_importance()
    if importance is None:
        st.warning("No SHAP importance results found yet — run `uv run python -m src.explain`.")
        return

    fig = go.Figure(
        go.Bar(x=importance["mean_abs_shap"], y=importance["feature"], orientation="h", marker_color="#2ca02c")
    )
    fig.update_layout(
        xaxis_title="Mean |SHAP value| (µg/m³)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        importance.rename(columns={"feature": "feature (column name)", "what_it_means": "what it means"}),
        hide_index=True,
        width="stretch",
    )


def _render_fairness_section() -> None:
    st.subheader("Is the forecast equally good at every site?")
    fairness = fairness_per_sensor()
    if fairness is None:
        st.info("No fairness report found for this model.")
        return
    st.caption(FAIRNESS_CAVEAT)
    st.dataframe(fairness, hide_index=True, width="stretch")
