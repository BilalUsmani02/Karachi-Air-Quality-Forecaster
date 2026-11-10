"""Tests for `src.ui.demo_day` — picking which past day "Demo mode" shows by default."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.ui.demo_day import available_demo_days, default_demo_day


def _features(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """`rows` as (date, sensor_id, y_pm25) -> a minimal features frame for these two functions."""
    return pd.DataFrame(rows, columns=["date", "sensor_id", "y_pm25"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_available_demo_days_is_sorted_and_deduplicated():
    features = _features(
        [("2026-01-02", "S1", 10.0), ("2026-01-01", "S1", 10.0), ("2026-01-01", "S2", 10.0)]
    )
    assert available_demo_days(features) == [date(2026, 1, 1), date(2026, 1, 2)]


def test_default_demo_day_is_none_for_an_empty_table():
    assert default_demo_day(_features([])) is None


def test_default_demo_day_prefers_the_day_with_the_most_sensors():
    features = _features(
        [
            # 2026-01-01: 1 sensor, severe (USG+) — but thin, should lose to 2026-01-02.
            ("2026-01-01", "S1", 90.0),
            # 2026-01-02: 2 sensors, both clean.
            ("2026-01-02", "S1", 5.0),
            ("2026-01-02", "S2", 6.0),
        ]
    )
    assert default_demo_day(features) == date(2026, 1, 2)


def test_default_demo_day_prefers_a_severe_day_among_ties_for_most_sensors():
    features = _features(
        [
            # Both days have 2 sensors; only 2026-01-03 reaches USG (35.5+).
            ("2026-01-02", "S1", 5.0),
            ("2026-01-02", "S2", 6.0),
            ("2026-01-03", "S1", 40.0),
            ("2026-01-03", "S2", 6.0),
        ]
    )
    assert default_demo_day(features) == date(2026, 1, 3)


def test_default_demo_day_breaks_remaining_ties_by_recency():
    features = _features(
        [
            ("2026-01-01", "S1", 5.0),
            ("2026-01-01", "S2", 6.0),
            ("2026-01-05", "S1", 5.0),
            ("2026-01-05", "S2", 6.0),
        ]
    )
    assert default_demo_day(features) == date(2026, 1, 5)


def test_default_demo_day_ignores_missing_readings_when_judging_severity():
    features = _features(
        [
            ("2026-01-01", "S1", float("nan")),
            ("2026-01-01", "S2", 10.0),
            ("2026-01-02", "S1", 5.0),
            ("2026-01-02", "S2", 6.0),
        ]
    )
    # Both days tied at 2 sensors, neither reaches USG -> falls back to recency.
    assert default_demo_day(features) == date(2026, 1, 2)


def test_default_demo_day_single_day_table():
    features = _features([("2026-06-01", "S1", 50.0)])
    assert default_demo_day(features) == date(2026, 6, 1)
