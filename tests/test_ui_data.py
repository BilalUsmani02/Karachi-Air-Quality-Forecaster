"""Tests for `src.ui.data.get_app_snapshot` — it passes straight through to the live pipeline."""

from __future__ import annotations

import src.ui.data as data_mod
from src.ui.data import get_app_snapshot

SENTINEL_SNAPSHOT = object()
SENTINEL_BUNDLE = object()


def test_passes_refresh_cache_dir_and_bundle_to_the_pipeline(monkeypatch, tmp_path):
    calls = {}

    def fake_get_snapshot(refresh, cache_dir, bundle):
        calls["args"] = (refresh, cache_dir, bundle)
        return SENTINEL_SNAPSHOT

    monkeypatch.setattr(data_mod.pipeline, "get_snapshot", fake_get_snapshot)
    assert get_app_snapshot(refresh=True, bundle=SENTINEL_BUNDLE, cache_dir=tmp_path) is SENTINEL_SNAPSHOT
    assert calls["args"] == (True, tmp_path, SENTINEL_BUNDLE)


def test_defaults_to_no_refresh_and_the_shared_cache(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        data_mod.pipeline, "get_snapshot", lambda refresh, cache_dir, bundle: calls.setdefault("a", (refresh, cache_dir))
    )
    get_app_snapshot()
    assert calls["a"] == (False, data_mod.CACHE_DIR)
