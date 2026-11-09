"""How the app gets a `Snapshot` to render (spec §9, item 1 — the "Refresh live data" button).

A thin seam over `src.live.pipeline.get_snapshot`, so the app's views and tests depend on one
small function: refresh=True tries the live APIs, then falls back to the cache, then the demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.live import pipeline
from src.live.snapshot import CACHE_DIR, Snapshot


def get_app_snapshot(
    refresh: bool = False,
    bundle: dict[str, Any] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> Snapshot:
    """The snapshot the app renders (see `pipeline.get_snapshot` for the fallback order)."""
    return pipeline.get_snapshot(refresh=refresh, cache_dir=cache_dir, bundle=bundle)
