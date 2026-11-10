"""Tests for `src.ui.results` — per-sensor summaries and the full advice-bearing result."""

from __future__ import annotations

import pytest

from src.data.sensors import SENSORS
from src.ui.results import (
    NO_FORECAST_COLOR,
    NO_FORECAST_LABEL,
    all_sensor_summaries,
    available_sensor_ids,
    band_meaning,
    build_sensor_result,
    default_sensor_id,
    feature_row,
    sensor_site_name,
)


def test_all_sensor_summaries_covers_every_registered_sensor(fake_snapshot):
    summaries = all_sensor_summaries(fake_snapshot)
    assert [s.sensor_id for s in summaries] == [s.id for s in SENSORS]


def test_a_sensor_with_a_forecast_gets_its_band_and_colour(fake_snapshot):
    summaries = {s.sensor_id: s for s in all_sensor_summaries(fake_snapshot)}
    s1 = summaries["S1"]
    assert s1.has_forecast
    assert s1.forecast_pm25 == pytest.approx(6.0)
    assert s1.band_name == "Good"
    assert s1.band_index == 0
    assert s1.color == "#00E400"


def test_a_sensor_with_a_usg_forecast_has_band_index_two(fake_snapshot):
    summaries = {s.sensor_id: s for s in all_sensor_summaries(fake_snapshot)}
    assert summaries["S5"].band_index == 2
    assert summaries["S5"].band_name == "Unhealthy for Sensitive Groups"


def test_a_sensor_without_a_forecast_is_greyed_out(fake_snapshot):
    summaries = {s.sensor_id: s for s in all_sensor_summaries(fake_snapshot)}
    missing = summaries["S2"]
    assert not missing.has_forecast
    assert missing.forecast_pm25 is None
    assert missing.band_name == NO_FORECAST_LABEL
    assert missing.band_index is None
    assert missing.color == NO_FORECAST_COLOR


def test_available_sensor_ids_is_in_registry_order(fake_snapshot):
    assert available_sensor_ids(fake_snapshot) == ["S1", "S3", "S5"]


def test_default_sensor_id_is_the_first_available_one(fake_snapshot):
    assert default_sensor_id(fake_snapshot) == "S1"


def test_build_sensor_result_matches_the_snapshot_row(fake_snapshot):
    result = build_sensor_result(fake_snapshot, "S3")
    assert result.site == "Urban Resource Center"
    assert result.forecast_pm25 == pytest.approx(31.0)
    assert result.prev_day_pm25 == pytest.approx(30.0)
    assert result.actual_pm25 == pytest.approx(32.0)
    assert result.band_name == "Moderate"
    assert result.color == "#FFFF00"


def test_build_sensor_result_runs_the_expert_system_with_the_given_profile(fake_snapshot):
    result = build_sensor_result(fake_snapshot, "S5", sensitive=True, outdoor_activity=True)
    assert result.recommendation.band_name == "Unhealthy for Sensitive Groups"
    assert result.recommendation.advice  # the expert system actually produced advice
    assert result.recommendation.disclaimer


def test_build_sensor_result_raises_for_a_sensor_with_no_forecast_row(fake_snapshot):
    with pytest.raises(ValueError, match="no forecast"):
        build_sensor_result(fake_snapshot, "S2")


def test_feature_row_drops_meta_columns_and_keeps_feature_order(fake_snapshot):
    from src.models.dataset import FEATURE_COLUMNS

    row = feature_row(fake_snapshot, "S1")
    assert list(row.index) == list(FEATURE_COLUMNS)


def test_feature_row_raises_for_a_sensor_with_no_forecast_row(fake_snapshot):
    with pytest.raises(ValueError, match="no forecast"):
        feature_row(fake_snapshot, "S2")


def test_sensor_site_name_matches_the_registry():
    assert sensor_site_name("S1") == "Climate Action Center"


@pytest.mark.parametrize("name", ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Hazardous"])
def test_band_meaning_is_nonempty_for_every_real_band(name):
    assert band_meaning(name)


def test_band_meaning_is_empty_for_an_unknown_name():
    assert band_meaning("not a band") == ""
