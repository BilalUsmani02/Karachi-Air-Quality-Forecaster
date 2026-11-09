"""Tests for `src.ui.format` — plain-English date/time/number formatting."""

from __future__ import annotations

from datetime import date

import pytest

from src.ui import format as fmt


def test_format_date_long_reads_naturally():
    assert fmt.format_date_long(date(2026, 11, 10)) == "Tuesday, 10 November 2026"


def test_format_pkt_timestamp_converts_utc_to_pkt():
    # 2026-11-08T09:32:00Z -> Asia/Karachi is UTC+5 -> 14:32 PKT.
    assert fmt.format_pkt_timestamp("2026-11-08T09:32:00Z") == "08 Nov 2026, 14:32 PKT"


def test_format_pkt_timestamp_handles_an_already_tz_aware_string():
    assert fmt.format_pkt_timestamp("2026-11-08T14:32:00+05:00") == "08 Nov 2026, 14:32 PKT"


def test_format_pkt_timestamp_assumes_utc_for_a_naive_timestamp():
    # No offset given at all -> treated as UTC, same as a `fetched_at` with no "Z"/"+00:00".
    assert fmt.format_pkt_timestamp("2026-11-08 09:32:00") == "08 Nov 2026, 14:32 PKT"


def test_format_pkt_timestamp_falls_back_on_garbage_input():
    assert fmt.format_pkt_timestamp("not a timestamp") == "not a timestamp"


@pytest.mark.parametrize(
    "value,expected",
    [(42.37, "42.4 µg/m³"), (0.0, "0.0 µg/m³"), (None, "—"), (float("nan"), "—")],
)
def test_format_pm25(value, expected):
    assert fmt.format_pm25(value) == expected


@pytest.mark.parametrize(
    "value,expected", [(42.6, "43 µg/m³"), (5.4, "5 µg/m³"), (None, "—"), (float("nan"), "—")]
)
def test_format_pm25_whole(value, expected):
    assert fmt.format_pm25_whole(value) == expected


@pytest.mark.parametrize(
    "source,expected_substring", [("live", "Live"), ("cache", "Cached"), ("demo", "Demo")]
)
def test_source_label_known_sources(source, expected_substring):
    assert expected_substring in fmt.source_label(source)


def test_source_label_falls_back_to_the_raw_value_for_an_unknown_source():
    assert fmt.source_label("mystery") == "mystery"


def test_is_missing():
    assert fmt.is_missing(None)
    assert fmt.is_missing(float("nan"))
    assert not fmt.is_missing(0.0)
    assert not fmt.is_missing(12.3)
