"""Tests for `src.ui.banner` — the Forecast page's status banner text."""

from __future__ import annotations

from datetime import date

from src.ui.banner import build_banner, stale_forecast_warning


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
    # today == forecast_date, so the new stale-cache check (tested separately below) stays quiet.
    banner = build_banner(live, today=live.forecast_date)
    assert banner.warnings == ()


def test_per_site_no_forecast_notes_are_grouped_apart_from_warnings(fake_snapshot):
    from src.ui.banner import build_banner

    snapshot = fake_snapshot.with_source("live", "S2 Zafar Memon DHA: no readings in the last 10 days, no forecast.")
    banner = build_banner(snapshot)
    assert banner.sensor_notes == ("S2 Zafar Memon DHA: no readings in the last 10 days, no forecast.",)
    assert all(not w.endswith(", no forecast.") for w in banner.warnings)


def test_stale_forecast_warning_fires_when_the_date_is_in_the_past():
    warning = stale_forecast_warning(date(2026, 11, 8), today=date(2026, 11, 10))
    assert warning is not None
    assert "Sunday, 8 November 2026" in warning
    assert "not today" in warning
    assert "Refresh live data" in warning
    assert "demo mode" in warning


def test_stale_forecast_warning_is_none_for_today_or_a_future_date():
    assert stale_forecast_warning(date(2026, 11, 10), today=date(2026, 11, 10)) is None
    assert stale_forecast_warning(date(2026, 11, 11), today=date(2026, 11, 10)) is None


def test_build_banner_adds_the_stale_warning_first_for_a_past_cached_snapshot(fake_snapshot):
    cached = fake_snapshot.with_source("cache")
    cached = type(cached)(
        date(2020, 1, 1), cached.fetched_at, "cache", cached.forecasts, cached.recent, (), cached.model_trained_on
    )
    banner = build_banner(cached, today=date(2026, 11, 10))
    assert banner.warnings
    assert "not today" in banner.warnings[0]


def test_build_banner_skips_the_stale_warning_for_demo_snapshots_even_when_old(fake_snapshot):
    # fake_snapshot is already a past-dated "demo" snapshot — demo mode never gets this warning.
    banner = build_banner(fake_snapshot, today=date(2026, 11, 10))
    assert all("not today" not in w for w in banner.warnings)
