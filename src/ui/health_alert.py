"""Health-alert text for a forecast band (review item 1: the PAI proposal's "health alerts").

Two alerts, both worded as guidance rather than an official warning (the project's overview
lists official air-quality warnings as out of scope — see `docs/`):

- `site_health_alert` — one line for the selected site, shown near the top of its result, once
  its forecast band reaches "Unhealthy for Sensitive Groups" (USG) or worse.
- `citywide_alert` — one line for the banner area, counting how many of today's forecast sites
  are USG-or-worse, shown whenever at least one is.

Streamlit-free so both are unit-testable; `app/views/forecast.py` picks `st.error` vs
`st.warning` from `HealthAlert.level`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Band indices from `src.classify.bands.BANDS`: 0 Good, 1 Moderate, 2 USG, 3 Unhealthy,
# 4 Very Unhealthy, 5 Hazardous.
USG_BAND_INDEX = 2
UNHEALTHY_BAND_INDEX = 3

INFORMATIONAL_ONLY = "Informational only — not an official alert."


@dataclass(frozen=True)
class HealthAlert:
    """One alert line, plus which Streamlit call should show it."""

    level: str  # "error" (Unhealthy or worse) or "warning" (USG)
    message: str


def site_health_alert(band_index: int, band_name: str, site: str) -> HealthAlert | None:
    """The alert for one site's forecast band, or `None` below USG (band index < 2).

    USG (index 2) is a warning; "Unhealthy" and worse (index >= 3) is an error, so the colour
    matches how serious the band is.
    """
    if band_index < USG_BAND_INDEX:
        return None
    level = "error" if band_index >= UNHEALTHY_BAND_INDEX else "warning"
    message = (
        f"Health alert: air quality is forecast to be {band_name} at {site} today. "
        f"{INFORMATIONAL_ONLY}"
    )
    return HealthAlert(level=level, message=message)


def count_usg_or_worse(band_indices: list[int]) -> int:
    """How many of `band_indices` are USG (2) or worse."""
    return sum(1 for idx in band_indices if idx is not None and idx >= USG_BAND_INDEX)


def citywide_alert(band_indices: list[int]) -> str | None:
    """"N of M sites are forecast Unhealthy for Sensitive Groups or worse." or `None`.

    `band_indices` is one band index per site that has a forecast today (so M is "sites with a
    forecast", not the full 14-sensor registry). Returns `None` if there are no such sites, or
    none of them reach USG.
    """
    total = len(band_indices)
    if total == 0:
        return None
    worse = count_usg_or_worse(band_indices)
    if worse == 0:
        return None
    return f"{worse} of {total} sites are forecast Unhealthy for Sensitive Groups or worse."
