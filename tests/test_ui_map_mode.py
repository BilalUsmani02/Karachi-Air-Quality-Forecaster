"""Tests for `src.ui.map_mode` — the offline-map toggle's default and style choice."""

from __future__ import annotations

from src.ui.map_mode import OFFLINE_STYLE, ONLINE_STYLE, default_offline_map, map_style


def test_defaults_offline_only_when_the_tile_probe_failed():
    assert default_offline_map(tiles_reachable=False) is True
    assert default_offline_map(tiles_reachable=True) is False


def test_defaults_online_when_no_probe_was_run():
    assert default_offline_map(tiles_reachable=None) is False


def test_map_style_matches_the_toggle():
    assert map_style(offline=True) == OFFLINE_STYLE
    assert map_style(offline=False) == ONLINE_STYLE
    assert OFFLINE_STYLE != ONLINE_STYLE
