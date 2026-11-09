"""Plain-English reason text for a SHAP explanation (spec §8, Phase 6).

`FEATURE_PHRASES` turns a feature name into a phrase a non-technical reader can follow;
`FEATURE_UNITS` gives the unit its raw value is shown in. `group_contributions` merges the 14
`sensor_*` one-hots into one "location" contribution (sum of their SHAP values — exactly one is
1 per row, so the sum is really just that sensor's own contribution) and each wind u/v pair into
one "wind direction" contribution per period (yesterday/tomorrow): on their own, a u or v
component is not a number anyone reads naturally. `top_reasons` turns the top-`k` grouped
contributions into sentences such as:

    "Tomorrow's wind speed was low (1.8 m/s), which raised the forecast by about 8 µg/m³."

Direction words ("raised"/"lowered") come from the SHAP value's sign; "low"/"high" compares the
raw value against the training median passed in via `feature_medians`.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.data.features import SENSOR_IDS
from src.explain.shap_explain import Contribution, Explanation

_SENSOR_FEATURES: tuple[str, ...] = tuple(f"sensor_{sid}" for sid in SENSOR_IDS)

LOCATION_LABEL = "this location"
PAST_WIND_LABEL = "yesterday's wind direction"
FC_WIND_LABEL = "tomorrow's wind direction"

# Every column in `src.models.dataset.FEATURE_COLUMNS` must have an entry here (tested in
# `tests/test_explain_reasons.py`) — no feature falls back to showing its raw column name.
FEATURE_PHRASES: dict[str, str] = {
    "pm25_lag_1h": "the PM2.5 level an hour ago",
    "pm25_lag_2h": "the PM2.5 level two hours ago",
    "pm25_lag_3h": "the PM2.5 level three hours ago",
    "pm25_lag_6h": "the PM2.5 level six hours ago",
    "pm25_lag_12h": "the PM2.5 level twelve hours ago",
    "pm25_lag_24h": "the PM2.5 level this time yesterday",
    "pm25_mean_72h": "the average PM2.5 over the last three days",
    "pm25_mean_168h": "the average PM2.5 over the last week",
    "pm25_std_24h": "how much PM2.5 swung up and down yesterday",
    "pm25_prev_day_mean": "yesterday's average PM2.5",
    "city_prev_day_mean": "yesterday's average PM2.5 across the whole city",
    "doy_sin": "the time of year (seasonal cycle)",
    "doy_cos": "the time of year (seasonal cycle)",
    "dow_sin": "the day of the week",
    "dow_cos": "the day of the week",
    "month": "the calendar month",
    "past_temperature_2m": "yesterday's temperature",
    "past_relative_humidity_2m": "yesterday's humidity",
    "past_dew_point_2m": "yesterday's dew point (how damp the air was)",
    "past_surface_pressure": "yesterday's air pressure",
    "past_precipitation": "yesterday's rainfall",
    "past_wind_speed_10m": "yesterday's wind speed",
    "past_wind_u": PAST_WIND_LABEL,
    "past_wind_v": PAST_WIND_LABEL,
    "past_boundary_layer_height": "yesterday's mixing height (how high pollution can spread)",
    "past_cams_pm2_5": "yesterday's modelled PM2.5 (CAMS atmospheric model)",
    "past_cams_pm10": "yesterday's modelled coarse dust (PM10)",
    "past_cams_no2": "yesterday's modelled nitrogen dioxide (traffic and industry pollution)",
    "past_cams_dust": "yesterday's modelled desert dust",
    "past_cams_aod": "yesterday's modelled haziness (aerosol optical depth)",
    "fc_temperature_2m": "tomorrow's temperature",
    "fc_relative_humidity_2m": "tomorrow's humidity",
    "fc_dew_point_2m": "tomorrow's dew point (how damp the air will be)",
    "fc_surface_pressure": "tomorrow's air pressure",
    "fc_precipitation": "tomorrow's rainfall",
    "fc_wind_speed_10m": "tomorrow's wind speed",
    "fc_wind_u": FC_WIND_LABEL,
    "fc_wind_v": FC_WIND_LABEL,
    "fc_boundary_layer_height": "tomorrow's mixing height (how high pollution can spread)",
    "fc_cams_pm2_5": "tomorrow's modelled PM2.5 (CAMS atmospheric model)",
    "fc_cams_pm10": "tomorrow's modelled coarse dust (PM10)",
    "fc_cams_no2": "tomorrow's modelled nitrogen dioxide (traffic and industry pollution)",
    "fc_cams_dust": "tomorrow's modelled desert dust",
    "fc_cams_aod": "tomorrow's modelled haziness (aerosol optical depth)",
    **{f: LOCATION_LABEL for f in _SENSOR_FEATURES},
}

FEATURE_UNITS: dict[str, str] = {
    **{f"pm25_lag_{k}h": "µg/m³" for k in (1, 2, 3, 6, 12, 24)},
    "pm25_mean_72h": "µg/m³",
    "pm25_mean_168h": "µg/m³",
    "pm25_std_24h": "µg/m³",
    "pm25_prev_day_mean": "µg/m³",
    "city_prev_day_mean": "µg/m³",
    "past_temperature_2m": "°C",
    "past_relative_humidity_2m": "%",
    "past_dew_point_2m": "°C",
    "past_surface_pressure": "hPa",
    "past_precipitation": "mm",
    "past_wind_speed_10m": "m/s",
    "past_boundary_layer_height": "m",
    "past_cams_pm2_5": "µg/m³",
    "past_cams_pm10": "µg/m³",
    "past_cams_no2": "µg/m³",
    "past_cams_dust": "µg/m³",
    "fc_temperature_2m": "°C",
    "fc_relative_humidity_2m": "%",
    "fc_dew_point_2m": "°C",
    "fc_surface_pressure": "hPa",
    "fc_precipitation": "mm",
    "fc_wind_speed_10m": "m/s",
    "fc_boundary_layer_height": "m",
    "fc_cams_pm2_5": "µg/m³",
    "fc_cams_pm10": "µg/m³",
    "fc_cams_no2": "µg/m³",
    "fc_cams_dust": "µg/m³",
    # Dimensionless / cyclic-encoded features (doy/dow sin-cos, month, AOD, wind u/v, sensor
    # one-hots) have no entry — `unit_for` returns "" for them.
}

_WIND_GROUPS: dict[str, str] = {
    "past_wind_u": PAST_WIND_LABEL,
    "past_wind_v": PAST_WIND_LABEL,
    "fc_wind_u": FC_WIND_LABEL,
    "fc_wind_v": FC_WIND_LABEL,
}


def phrase_for(feature: str) -> str:
    """The readable phrase for a feature column; raises if none is registered."""
    try:
        return FEATURE_PHRASES[feature]
    except KeyError:
        raise KeyError(f"no readable phrase registered for feature {feature!r}") from None


def unit_for(feature: str) -> str:
    """The display unit for a feature's raw value, or `""` if it has none."""
    return FEATURE_UNITS.get(feature, "")


@dataclass(frozen=True)
class GroupedContribution:
    """One line of plain-English explanation: either one feature, or a merged group.

    `feature` is `None` for a merged group (`location`, `wind direction`) — there, a single raw
    value and median wouldn't mean anything, so `raw_value` is also `None`.
    """

    feature: str | None
    label: str
    shap_value: float
    raw_value: float | None
    unit: str


def group_contributions(contributions: Sequence[Contribution]) -> list[GroupedContribution]:
    """Merge sensor one-hots and wind u/v pairs; sort the result by `|shap_value|` descending."""
    group_sums: dict[str, float] = {}
    singles: list[GroupedContribution] = []

    for c in contributions:
        if c.feature in _SENSOR_FEATURES:
            group_sums[LOCATION_LABEL] = group_sums.get(LOCATION_LABEL, 0.0) + c.shap_value
        elif c.feature in _WIND_GROUPS:
            label = _WIND_GROUPS[c.feature]
            group_sums[label] = group_sums.get(label, 0.0) + c.shap_value
        else:
            singles.append(
                GroupedContribution(
                    feature=c.feature,
                    label=phrase_for(c.feature),
                    shap_value=c.shap_value,
                    raw_value=c.raw_value,
                    unit=unit_for(c.feature),
                )
            )

    groups = [
        GroupedContribution(feature=None, label=label, shap_value=total, raw_value=None, unit="")
        for label, total in group_sums.items()
    ]
    return sorted(singles + groups, key=lambda g: abs(g.shap_value), reverse=True)


def _sentence(g: GroupedContribution, median: float | None) -> str:
    verb = "raised" if g.shap_value >= 0 else "lowered"
    magnitude = abs(g.shap_value)

    effect = f"{verb} the forecast by about {magnitude:.0f} µg/m³."
    if g.raw_value is None:
        text = f"{g.label} {effect}"
    elif math.isnan(g.raw_value):
        # The preprocessor filled the gap with the training median, so say so plainly.
        text = f"{g.label} was missing (a typical value was used), which {effect}"
    else:
        unit = f" {g.unit}" if g.unit else ""
        value = f"{g.raw_value:.1f}{unit}"
        if median is None:
            text = f"{g.label} ({value}) {effect}"
        else:
            level = "low" if g.raw_value < median else "high"
            text = f"{g.label} was {level} ({value}), which {effect}"
    return text[0].upper() + text[1:]


def top_reasons(
    explanation: Explanation,
    k: int = 3,
    feature_medians: Mapping[str, float] | None = None,
) -> list[str]:
    """The `k` largest grouped contributions as plain-English sentences, largest effect first."""
    grouped = group_contributions(explanation.contributions)
    reasons = []
    for g in grouped[:k]:
        median = feature_medians.get(g.feature) if feature_medians and g.feature else None
        reasons.append(_sentence(g, median))
    return reasons
