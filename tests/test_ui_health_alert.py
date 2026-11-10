"""Tests for `src.ui.health_alert` — the review's "health alerts" feature."""

from __future__ import annotations

import pytest

from src.ui.health_alert import (
    INFORMATIONAL_ONLY,
    citywide_alert,
    count_usg_or_worse,
    site_health_alert,
)


@pytest.mark.parametrize("band_index,band_name", [(0, "Good"), (1, "Moderate")])
def test_site_health_alert_is_none_below_usg(band_index, band_name):
    assert site_health_alert(band_index, band_name, "Site X") is None


def test_site_health_alert_is_a_warning_for_usg():
    alert = site_health_alert(2, "Unhealthy for Sensitive Groups", "Clifton")
    assert alert is not None
    assert alert.level == "warning"
    assert "Clifton" in alert.message
    assert "Unhealthy for Sensitive Groups" in alert.message
    assert alert.message.endswith(INFORMATIONAL_ONLY)


@pytest.mark.parametrize(
    "band_index,band_name", [(3, "Unhealthy"), (4, "Very Unhealthy"), (5, "Hazardous")]
)
def test_site_health_alert_is_an_error_for_unhealthy_or_worse(band_index, band_name):
    alert = site_health_alert(band_index, band_name, "Clifton")
    assert alert is not None
    assert alert.level == "error"
    assert band_name in alert.message


def test_site_health_alert_wording_is_informational_not_official():
    alert = site_health_alert(3, "Unhealthy", "Clifton")
    assert "Health alert:" in alert.message
    assert "forecast to be" in alert.message  # a forecast, not an observed/official reading
    assert "Informational only" in alert.message
    assert "official alert" in alert.message


def test_count_usg_or_worse_counts_only_usg_and_above():
    assert count_usg_or_worse([0, 1, 2, 3, 4, 5]) == 4


def test_count_usg_or_worse_ignores_none_entries():
    assert count_usg_or_worse([0, None, 2, None]) == 1


def test_citywide_alert_is_none_when_nothing_qualifies():
    assert citywide_alert([]) is None
    assert citywide_alert([0, 1, 1]) is None


def test_citywide_alert_counts_sites_with_a_forecast_as_the_denominator():
    assert citywide_alert([0, 1, 2, 3]) == "2 of 4 sites are forecast Unhealthy for Sensitive Groups or worse."


def test_citywide_alert_when_every_site_qualifies():
    assert citywide_alert([2, 3]) == "2 of 2 sites are forecast Unhealthy for Sensitive Groups or worse."
