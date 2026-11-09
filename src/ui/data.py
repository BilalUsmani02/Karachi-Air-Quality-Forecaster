"""How the app gets a `Snapshot` to render (spec §9, item 1 — the "Refresh live data" button).

A parallel agent is building `src.live.pipeline.get_snapshot` on its own branch; it does not
exist here yet. `get_app_snapshot` is written so the app works either way: once the pipeline
lands (this module needs no change), it is used directly; until then, an `ImportError` falls
back to the cached snapshot, then the offline demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.live.snapshot import CACHE_DIR, Snapshot, demo_snapshot, load_bundle, load_snapshot


def get_app_snapshot(
    refresh: bool = False,
    bundle: dict[str, Any] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> Snapshot:
    """The snapshot the app renders.

    `refresh=True` (the "Refresh live data" button) only has an effect once
    `src.live.pipeline.get_snapshot` exists — it is the one that actually tries the live APIs.
    Until then (or if a live fetch fails), this falls back to the last cached snapshot
    (`load_snapshot`) and, if there is none, the offline demo (`demo_snapshot`, which needs
    `bundle` — loaded with `load_bundle()` here if the caller didn't already have one).
    """
    try:
        from src.live.pipeline import get_snapshot
    except ImportError:
        cached = load_snapshot(cache_dir)
        if cached is not None:
            return cached
        return demo_snapshot(bundle if bundle is not None else load_bundle())
    return get_snapshot(refresh=refresh, cache_dir=cache_dir, bundle=bundle)
