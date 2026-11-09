"""Tests for `src.ui.data.get_app_snapshot` — both branches (spec §9, Phase 7 handoff).

`src.live.pipeline` does not exist on this branch (a parallel agent is building it), so the
`ImportError` branch is the real, naturally-taken one here; the "pipeline exists" branch is
exercised by injecting a fake module into `sys.modules`.
"""

from __future__ import annotations

import sys

import pytest

import src.ui.data as data_mod
from src.ui.data import get_app_snapshot

SENTINEL_CACHED = object()
SENTINEL_DEMO = object()
SENTINEL_BUNDLE = object()


def test_pipeline_module_does_not_exist_yet_on_this_branch():
    # Documents the precondition the ImportError-fallback tests below rely on; if this ever
    # starts passing, `get_app_snapshot` should be simplified to call the pipeline directly.
    with pytest.raises(ImportError):
        import src.live.pipeline  # noqa: F401


def test_falls_back_to_the_cache_when_the_pipeline_is_missing(monkeypatch):
    monkeypatch.setattr(data_mod, "load_snapshot", lambda cache_dir: SENTINEL_CACHED)
    monkeypatch.setattr(
        data_mod, "demo_snapshot", lambda bundle: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    assert get_app_snapshot(refresh=False, bundle=SENTINEL_BUNDLE) is SENTINEL_CACHED


def test_falls_back_to_demo_when_there_is_no_cache_either(monkeypatch):
    monkeypatch.setattr(data_mod, "load_snapshot", lambda cache_dir: None)
    monkeypatch.setattr(
        data_mod, "demo_snapshot", lambda bundle: SENTINEL_DEMO if bundle is SENTINEL_BUNDLE else None
    )
    assert get_app_snapshot(refresh=True, bundle=SENTINEL_BUNDLE) is SENTINEL_DEMO


def test_loads_the_bundle_itself_when_none_is_given_for_the_demo_fallback(monkeypatch):
    monkeypatch.setattr(data_mod, "load_snapshot", lambda cache_dir: None)
    monkeypatch.setattr(data_mod, "load_bundle", lambda: SENTINEL_BUNDLE)
    monkeypatch.setattr(
        data_mod, "demo_snapshot", lambda bundle: SENTINEL_DEMO if bundle is SENTINEL_BUNDLE else None
    )
    assert get_app_snapshot(refresh=False, bundle=None) is SENTINEL_DEMO


def test_uses_the_live_pipeline_when_it_is_importable(monkeypatch):
    calls = {}

    def fake_get_snapshot(refresh, cache_dir, bundle):
        calls["args"] = (refresh, cache_dir, bundle)
        return SENTINEL_CACHED

    import types

    fake_module = types.ModuleType("src.live.pipeline")
    fake_module.get_snapshot = fake_get_snapshot
    monkeypatch.setitem(sys.modules, "src.live.pipeline", fake_module)

    result = get_app_snapshot(refresh=True, bundle=SENTINEL_BUNDLE)

    assert result is SENTINEL_CACHED
    assert calls["args"] == (True, data_mod.CACHE_DIR, SENTINEL_BUNDLE)
