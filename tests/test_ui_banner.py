"""Tests for `src.ui.banner` — the Forecast page's status banner text."""

from __future__ import annotations

from src.ui.banner import build_banner


def test_banner_heading_names_the_weekday_date_and_issue_time(fake_snapshot):
    banner = build_banner(fake_snapshot)
    assert "Sunday, 8 November 2026" in banner.heading
    assert "24-hour average" in banner.heading
    assert "issued 00:00 PKT" in banner.heading


def test_banner_last_updated_is_in_pkt(fake_snapshot):
    banner = build_banner(fake_snapshot)
    assert banner.last_updated == "Last updated 08 Nov 2026, 06:00 PKT"


def test_banner_source_text_names_the_source(fake_snapshot):
    banner = build_banner(fake_snapshot)
    assert banner.source_text == "Source: Demo mode (offline sample from stored history)"


def test_banner_carries_every_warning_and_nothing_extra(fake_snapshot):
    banner = build_banner(fake_snapshot)
    assert banner.warnings == fake_snapshot.warnings
    assert len(banner.warnings) == 1


def test_banner_has_no_warnings_when_the_snapshot_has_none(fake_snapshot):
    live = fake_snapshot.with_source("live")
    live = type(live)(
        live.forecast_date, live.fetched_at, "live", live.forecasts, live.recent, (), live.model_trained_on
    )
    banner = build_banner(live)
    assert banner.warnings == ()
