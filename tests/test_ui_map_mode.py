"""Tests for `src.ui.map_mode` — the offline-map toggle's default and style choice."""

from __future__ import annotations

from src.ui.map_mode import OFFLINE_STYLE, ONLINE_STYLE, default_offline_map, map_style


def test_non_live_source_always_defaults_to_offline_regardless_of_the_probe():
    assert default_offline_map("cache", tiles_reachable=True) is True
    assert default_offline_map("demo", tiles_reachable=True) is True
    assert default_offline_map("cache", tiles_reachable=None) is True


def test_live_source_defaults_online_when_no_probe_was_run():
    assert default_offline_map("live", tiles_reachable=None) is False


def test_live_source_trusts_the_probe():
    assert default_offline_map("live", tiles_reachable=True) is False
    assert default_offline_map("live", tiles_reachable=False) is True


def test_map_style_matches_the_toggle():
    assert map_style(offline=True) == OFFLINE_STYLE
    assert map_style(offline=False) == ONLINE_STYLE
    assert OFFLINE_STYLE != ONLINE_STYLE
