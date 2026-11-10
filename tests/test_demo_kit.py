import zipfile
from datetime import date

from scripts.make_demo_kit import build_demo_kit


def _make_fake_project(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "final_ensemble.pkl").write_bytes(b"fake-pickle")

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "features.parquet").write_bytes(b"fake-parquet")
    (processed / "build_report.json").write_text("{}")

    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    (cache / "snapshot.json").write_text("{}")

    (tmp_path / "uv.lock").write_text("fake-lock")
    return tmp_path


def test_build_demo_kit_zips_expected_files(tmp_path):
    root = _make_fake_project(tmp_path)
    out_dir = tmp_path / "dist"

    zip_path = build_demo_kit(root=root, out_dir=out_dir, today=date(2026, 11, 10))

    assert zip_path == out_dir / "demo_kit_2026-11-10.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "models/final_ensemble.pkl" in names
    assert "data/processed/features.parquet" in names
    assert "data/processed/build_report.json" in names
    assert "data/cache/snapshot.json" in names
    assert "uv.lock" in names


def test_build_demo_kit_skips_missing_inputs(tmp_path):
    # No models/, no data/ at all -- only uv.lock exists.
    root = tmp_path
    (root / "uv.lock").write_text("fake-lock")
    out_dir = tmp_path / "dist"

    zip_path = build_demo_kit(root=root, out_dir=out_dir, today=date(2026, 11, 10))

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert names == ["uv.lock"]


def test_build_demo_kit_default_name_uses_today(tmp_path):
    root = _make_fake_project(tmp_path)
    out_dir = tmp_path / "dist"

    zip_path = build_demo_kit(root=root, out_dir=out_dir)

    assert zip_path.name == f"demo_kit_{date.today().isoformat()}.zip"
