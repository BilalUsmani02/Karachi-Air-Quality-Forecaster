"""Zip everything needed to run the app on another machine, offline, without a rebuild.

    uv run python scripts/make_demo_kit.py

Packs `models/final_ensemble.pkl`, `data/processed/`, `data/cache/` and `uv.lock` into
`dist/demo_kit_<YYYY-MM-DD>.zip`. stdlib only (`zipfile`, `pathlib`) — no project imports, so it
still runs if the environment is broken.

To restore on another machine: unzip into the project root (same layout — `models/`,
`data/processed/`, `data/cache/`, `uv.lock` land in their normal places), then
`uv sync --frozen` (uses the packed `uv.lock`, doesn't touch package versions) and
`uv run streamlit run app/main.py`. Do **not** run a bare `uv sync` or upgrade packages on
demo day: `models/final_ensemble.pkl` is a pickle tied to the exact NumPy/scikit-learn/PyTorch
versions it was saved with, and a newer library can fail to unpickle it or silently change its
behaviour.
"""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "final_ensemble.pkl"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"
LOCK_FILE = ROOT / "uv.lock"
DIST_DIR = ROOT / "dist"


def _iter_files(path: Path) -> list[Path]:
    """Every regular file under `path` (or `[path]` if `path` is itself a file)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.is_file()]
    return []


def build_demo_kit(root: Path = ROOT, out_dir: Path = DIST_DIR, today: date | None = None) -> Path:
    """Zip the demo kit for `root` into `out_dir`, returning the zip's path.

    Missing inputs (e.g. no live cache yet) are skipped rather than raising, so the script still
    produces a useful kit from a partially-built checkout.
    """
    today = today or date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"demo_kit_{today.isoformat()}.zip"

    sources = [
        root / "models" / "final_ensemble.pkl",
        root / "data" / "processed",
        root / "data" / "cache",
        root / "uv.lock",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source in sources:
            for file_path in _iter_files(source):
                zf.write(file_path, arcname=file_path.relative_to(root).as_posix())

    return zip_path


def main() -> None:
    zip_path = build_demo_kit()
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    with zipfile.ZipFile(zip_path) as zf:
        count = len(zf.namelist())
    print(f"wrote {zip_path} ({size_mb:.1f} MB, {count} files)")


if __name__ == "__main__":
    main()
