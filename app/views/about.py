"""The About & data page (spec §9): data sources, model version, how it works, limitations,
privacy and the disclaimer. Static text plus the one dynamic fact — the model's training
window (`bundle["trained_on"]`) — so it never goes stale as the model is retrained.
"""

from __future__ import annotations

import streamlit as st

from app.state import get_bundle, missing_bundle_message
from src.ui.format import format_day


def render() -> None:
    st.title("About & data")

    st.subheader("What this app does")
    st.write(
        "Each midnight it forecasts the coming day's 24-hour average fine-dust level (PM2.5) at monitoring sites "
        "across Karachi, sorts that forecast into an official air-quality band (Good → "
        "Hazardous), and turns the band into plain health advice for your own situation."
    )

    st.subheader("Model version")
    bundle = get_bundle()
    if bundle is None:
        st.error(missing_bundle_message())
    else:
        trained_on = tuple(bundle.get("trained_on") or ())
        start, end = (trained_on + ("", ""))[:2]
        st.write(f"Trained on data from **{format_day(start)}** to **{format_day(end)}**.")
        st.caption(
            "It predicts with an equal-weight average of a tuned Gradient Boosting model and "
            "a neural network written from scratch in NumPy — see the Model comparison page."
        )

    st.subheader("Where the data comes from")
    st.markdown(
        "- **Air quality:** [OpenAQ](https://openaq.org), mirroring 14 low-cost PM2.5 sensors "
        "run by **The Urban Unit** / **AirGradient** around Karachi.\n"
        "- **Weather:** [Open-Meteo](https://open-meteo.com) — recorded weather (ERA5 "
        "reanalysis) for past days, and its weather forecast for the day ahead.\n"
        "- **Atmospheric model:** **CAMS** (Copernicus Atmosphere Monitoring Service) — an "
        "independent modelled estimate of PM2.5, PM10, NO₂ and dust, used as an extra input "
        "feature alongside the sensors."
    )

    st.subheader("How it works, in 4 steps")
    st.markdown(
        "1. **Gather:** the last few days of each sensor's readings, plus weather and CAMS "
        "data for the day before and the forecast day.\n"
        "2. **Forecast:** a trained model combines all of that into one number — the day's "
        "expected 24-hour average PM2.5 for each site.\n"
        "3. **Classify:** that number is sorted into an EPA air-quality band (Good, Moderate, "
        "Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy or Hazardous).\n"
        "4. **Advise:** a rule-based expert system turns the band, the trend, and your own "
        "profile (sensitive group? outdoor plans?) into specific, explained advice."
    )

    st.subheader("Limitations")
    st.markdown(
        "- The rarest, most severe days (**Very Unhealthy**) are the ones the model catches "
        "least often — averaging several models together smooths out extreme forecasts.\n"
        "- Live forecasts use a **weather forecast** for the day, which is never as accurate as "
        "the recorded weather the model was trained and tested with — real-world accuracy "
        "will usually be a little lower than the numbers on the comparison page.\n"
        "- The sensors are **low-cost** air-quality monitors, not reference-grade instruments "
        "— individual hourly readings can be noisy, even though the daily average is reliable "
        "enough to forecast from.\n"
        "- Some sites have much more missing data than others, which can make their forecasts "
        "less certain (see the fairness note on the Model comparison page)."
    )

    st.subheader("Your privacy")
    st.write(
        "The two profile questions on the Forecast page (sensitive group, outdoor plans) are "
        "kept only in this browser tab, for this session. They are never written to disk, "
        "never logged, and never saved into the forecast cache."
    )

    st.subheader("Disclaimer")
    st.info(
        "This tool is informational only. It is **not** an official air-quality alert and "
        "**not** personal medical advice. If you have a heart or lung condition, follow your "
        "own doctor's advice.",
        icon="⚠️",
    )
