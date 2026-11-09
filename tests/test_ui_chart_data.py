"""Tests for `src.ui.chart_data` — data shaping for the 72 h history + forecast chart."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.ui.chart_data import STRIPE_CEILING_FLOOR, build_chart_series


def test_recent_series_is_72_hours_of_the_right_sensor(fake_snapshot):
    series = build_chart_series(fake_snapshot, "S1")
    assert len(series.recent_times) == 72
    assert len(series.recent_values) == 72
    assert all(v == pytest.approx(20.0) for v in series.recent_values)
    assert series.recent_times[0] < series.recent_times[-1]


def test_forecast_segment_spans_day_d_in_pkt(fake_snapshot):
    series = build_chart_series(fake_snapshot, "S1")
    assert series.forecast_value == pytest.approx(6.0)
    assert series.forecast_start == pd.Timestamp("2026-11-08 00:00", tz="Asia/Karachi")
    assert series.forecast_end == pd.Timestamp("2026-11-09 00:00", tz="Asia/Karachi")


def test_band_stripes_cover_from_zero_to_at_least_the_floor_ceiling(fake_snapshot):
    # Values here (recent 20, forecast 6) are small, so the ceiling should hit the floor.
    series = build_chart_series(fake_snapshot, "S1")
    assert series.band_stripes[0].low == 0.0
    assert series.band_stripes[-1].high == pytest.approx(STRIPE_CEILING_FLOOR)
    for lo, hi in zip(series.band_stripes, series.band_stripes[1:]):
        # EPA bands have a genuine 0.1 µg/m³ gap between them (truncated boundaries) — stripes
        # should still be in increasing, non-overlapping order.
        assert lo.high <= hi.low
        assert hi.low - lo.high <= 0.11


def test_band_stripes_grow_with_a_tall_forecast(fake_snapshot):
    # S5's forecast (51) is well above the 60-floor once the 1.3x margin applies at ~66.
    series = build_chart_series(fake_snapshot, "S5")
    assert series.band_stripes[-1].high > STRIPE_CEILING_FLOOR
    assert all(math.isfinite(s.high) for s in series.band_stripes)


def test_build_chart_series_raises_for_a_sensor_with_no_forecast(fake_snapshot):
    with pytest.raises(ValueError, match="no forecast"):
        build_chart_series(fake_snapshot, "S2")
