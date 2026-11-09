"""Tests for `src.expert.facts.Facts` (PROJECT_SPEC.md §7, §10 privacy safeguard)."""

import dataclasses

import pytest

from src.classify.bands import pm25_to_band
from src.expert.facts import PROFILE_FIELDS, Facts

# --- band is computed from the forecast, not supplied --------------------------------------


def test_band_is_derived_from_forecast():
    facts = Facts(forecast_pm25=33.0)
    assert facts.band == pm25_to_band(33.0) == 1  # Moderate


@pytest.mark.parametrize(
    "pm25,expected_band",
    [(5.0, 0), (20.0, 1), (45.0, 2), (90.0, 3), (160.0, 4), (300.0, 5)],
)
def test_band_matches_bands_module_for_each_eq_band(pm25, expected_band):
    assert Facts(forecast_pm25=pm25).band == expected_band


# --- defaults --------------------------------------------------------------------------------


def test_defaults_are_not_sensitive_no_outdoor_no_today():
    facts = Facts(forecast_pm25=20.0)
    assert facts.today_pm25 is None
    assert facts.sensitive is False
    assert facts.outdoor_activity is False


# --- validation --------------------------------------------------------------------------------


@pytest.mark.parametrize("pm25", [-0.1, -1.0, -500.0])
def test_negative_forecast_raises(pm25):
    with pytest.raises(ValueError):
        Facts(forecast_pm25=pm25)


@pytest.mark.parametrize("today", [-0.1, -1.0])
def test_negative_today_raises(today):
    with pytest.raises(ValueError):
        Facts(forecast_pm25=20.0, today_pm25=today)


def test_zero_is_valid_for_forecast_and_today():
    facts = Facts(forecast_pm25=0.0, today_pm25=0.0)
    assert facts.band == 0
    assert facts.today_pm25 == 0.0


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_non_bool_sensitive_raises_type_error(value):
    with pytest.raises(TypeError):
        Facts(forecast_pm25=20.0, sensitive=value)


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_non_bool_outdoor_activity_raises_type_error(value):
    with pytest.raises(TypeError):
        Facts(forecast_pm25=20.0, outdoor_activity=value)


# --- immutability ------------------------------------------------------------------------------


def test_facts_is_frozen():
    facts = Facts(forecast_pm25=20.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.forecast_pm25 = 50.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.sensitive = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.band = 5  # type: ignore[misc]


# --- privacy safeguard (§7, §10): the profile is only two booleans, never anything richer ---


def test_profile_is_exactly_two_boolean_fields():
    assert PROFILE_FIELDS == ("sensitive", "outdoor_activity")


def test_facts_has_no_fields_beyond_the_documented_set():
    field_names = {f.name for f in dataclasses.fields(Facts)}
    assert field_names == {"forecast_pm25", "today_pm25", "sensitive", "outdoor_activity", "band"}


def test_profile_fields_are_plain_booleans_not_richer_data():
    facts = Facts(forecast_pm25=20.0, sensitive=True, outdoor_activity=False)
    for name in PROFILE_FIELDS:
        value = getattr(facts, name)
        assert isinstance(value, bool)
        # the privacy safeguard (§7) only holds if these two facts really are plain yes/no --
        # not, say, a string that could later smuggle in a named medical condition.
        assert not isinstance(value, str)
