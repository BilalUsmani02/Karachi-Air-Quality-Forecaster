"""Tests for `src.explain.reasons` — pure text logic, no SHAP computation involved."""

from __future__ import annotations

import pytest

from src.explain import reasons
from src.explain.shap_explain import Contribution, Explanation
from src.models.dataset import FEATURE_COLUMNS, SENSOR_COLUMNS


def _explanation(contributions: list[Contribution]) -> Explanation:
    ordered = tuple(sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True))
    prediction = sum(c.shap_value for c in ordered)
    return Explanation(base_value=0.0, prediction=prediction, contributions=ordered)


def test_every_feature_column_has_a_readable_phrase():
    for feature in FEATURE_COLUMNS:
        phrase = reasons.phrase_for(feature)
        assert phrase != feature  # never falls back to the raw column name
        assert isinstance(phrase, str) and phrase


def test_phrase_for_unknown_feature_raises():
    with pytest.raises(KeyError):
        reasons.phrase_for("not_a_real_feature")


def test_positive_shap_says_raised_negative_says_lowered():
    explanation = _explanation(
        [
            Contribution("fc_wind_speed_10m", raw_value=1.8, shap_value=8.0),
            Contribution("pm25_prev_day_mean", raw_value=20.0, shap_value=-5.0),
        ]
    )
    texts = reasons.top_reasons(explanation, k=2)
    assert "raised" in texts[0]
    assert "lowered" in texts[1]


def test_low_high_wording_relative_to_median():
    explanation = _explanation(
        [
            Contribution("fc_wind_speed_10m", raw_value=1.8, shap_value=8.0),
            Contribution("past_temperature_2m", raw_value=40.0, shap_value=3.0),
        ]
    )
    texts = reasons.top_reasons(
        explanation, k=2, feature_medians={"fc_wind_speed_10m": 3.0, "past_temperature_2m": 28.0}
    )
    assert "was low" in texts[0]  # 1.8 < median 3.0
    assert "was high" in texts[1]  # 40.0 > median 28.0


def test_missing_raw_value_says_missing_not_high_or_low():
    explanation = _explanation([Contribution("fc_cams_no2", raw_value=float("nan"), shap_value=-3.0)])
    text = reasons.top_reasons(explanation, k=1, feature_medians={"fc_cams_no2": 20.0})[0]
    assert "was missing" in text
    assert "nan" not in text
    assert "was high" not in text and "was low" not in text


def test_units_appear_in_the_sentence():
    explanation = _explanation([Contribution("fc_wind_speed_10m", raw_value=1.8, shap_value=8.0)])
    texts = reasons.top_reasons(explanation, k=1, feature_medians={"fc_wind_speed_10m": 3.0})
    assert "m/s" in texts[0]
    assert "8" in texts[0]
    assert "µg/m³" in texts[0]


def test_sensor_one_hots_group_into_one_location_contribution():
    contributions = [
        Contribution(f"sensor_{sid}", raw_value=float(sid == "S1"), shap_value=2.0 if sid == "S1" else 0.0)
        for sid in (c.removeprefix("sensor_") for c in SENSOR_COLUMNS)
    ]
    # give a couple of them non-zero shap to prove the sum, not just the active one-hot, is used
    contributions[1] = Contribution(contributions[1].feature, raw_value=0.0, shap_value=1.5)
    explanation = _explanation(contributions)

    grouped = reasons.group_contributions(explanation.contributions)
    location_groups = [g for g in grouped if g.label == reasons.LOCATION_LABEL]
    assert len(location_groups) == 1
    assert location_groups[0].shap_value == pytest.approx(3.5)
    assert location_groups[0].raw_value is None


def test_wind_u_v_pair_groups_sensibly():
    contributions = [
        Contribution("past_wind_u", raw_value=1.2, shap_value=2.0),
        Contribution("past_wind_v", raw_value=-0.5, shap_value=-0.5),
        Contribution("fc_wind_u", raw_value=0.3, shap_value=1.0),
        Contribution("fc_wind_v", raw_value=0.1, shap_value=1.0),
    ]
    explanation = _explanation(contributions)
    grouped = reasons.group_contributions(explanation.contributions)
    labels = {g.label: g.shap_value for g in grouped}

    assert labels[reasons.PAST_WIND_LABEL] == pytest.approx(1.5)
    assert labels[reasons.FC_WIND_LABEL] == pytest.approx(2.0)
    assert len(grouped) == 2  # the four wind components collapsed into two groups


def test_group_contributions_sorted_by_magnitude():
    contributions = [
        Contribution("fc_wind_speed_10m", raw_value=1.0, shap_value=1.0),
        Contribution("pm25_prev_day_mean", raw_value=1.0, shap_value=-9.0),
        Contribution("past_temperature_2m", raw_value=1.0, shap_value=3.0),
    ]
    explanation = _explanation(contributions)
    grouped = reasons.group_contributions(explanation.contributions)

    magnitudes = [abs(g.shap_value) for g in grouped]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_reasons_respects_k():
    contributions = [
        Contribution("fc_wind_speed_10m", raw_value=1.0, shap_value=5.0),
        Contribution("pm25_prev_day_mean", raw_value=1.0, shap_value=4.0),
        Contribution("past_temperature_2m", raw_value=1.0, shap_value=3.0),
        Contribution("fc_precipitation", raw_value=1.0, shap_value=2.0),
    ]
    explanation = _explanation(contributions)
    assert len(reasons.top_reasons(explanation, k=2)) == 2
    assert len(reasons.top_reasons(explanation, k=3)) == 3


def test_unit_for_returns_empty_string_for_dimensionless_feature():
    assert reasons.unit_for("doy_sin") == ""
    assert reasons.unit_for("fc_wind_speed_10m") == "m/s"
