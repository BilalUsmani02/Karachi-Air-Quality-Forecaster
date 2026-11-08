import json
from datetime import date, timedelta

import pandas as pd

from src.data import download_air_quality as daq
from src.data.open_meteo import FetchResult
from src.data.sensors import sensor_by_id

SENSOR = sensor_by_id("S1")


def _fake_result(rows: int = 3) -> FetchResult:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="h", tz="Asia/Karachi"),
            "pm2_5": [10.0] * rows,
        }
    )
    return FetchResult(frame=frame, grid_latitude=24.9, grid_longitude=67.1)


def test_default_start_is_first_full_cams_day():
    # Checked live on 8 Nov 2026: CAMS global is null before 2022-08-04, and that first
    # day only starts partway through (05:00 PKT), so 2022-08-05 is the first full day.
    assert daq.DEFAULT_START == date(2022, 8, 5)


def test_default_end_is_yesterday():
    assert daq.default_end() == date.today() - timedelta(days=1)


def test_parse_args_defaults():
    args = daq.parse_args([])
    assert args.start == daq.DEFAULT_START
    assert args.end is None
    assert args.sensors is None
    assert args.force is False


def test_download_one_writes_csv_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(daq, "fetch_range", lambda *a, **k: _fake_result(rows=4))

    daq.download_one(SENSOR, date(2024, 1, 1), date(2024, 1, 2), force=False, out_dir=tmp_path)

    csv_path = tmp_path / "S1.csv"
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 4

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["S1"]
    assert entry["endpoint"] == daq.URL
    assert entry["variables"] == daq.HOURLY_VARS
    assert entry["licence"] == "CC BY 4.0, Open-Meteo / Copernicus CAMS"
    assert entry["row_count"] == 4


def test_download_one_skips_existing_without_force(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(daq, "fetch_range", lambda *a, **k: calls.append(1) or _fake_result())
    (tmp_path / "S1.csv").write_text("time,pm2_5\n")

    daq.download_one(SENSOR, date(2024, 1, 1), date(2024, 1, 2), force=False, out_dir=tmp_path)

    assert calls == []
    assert "skip" in capsys.readouterr().out


def test_download_one_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(daq, "fetch_range", lambda *a, **k: _fake_result(rows=6))
    (tmp_path / "S1.csv").write_text("stale")

    daq.download_one(SENSOR, date(2024, 1, 1), date(2024, 1, 2), force=True, out_dir=tmp_path)

    assert len(pd.read_csv(tmp_path / "S1.csv")) == 6


def test_main_uses_sensor_subset(monkeypatch):
    seen = []
    monkeypatch.setattr(daq, "download_one", lambda sensor, start, end, force: seen.append(sensor.id))
    monkeypatch.setattr(daq, "default_end", lambda: date(2024, 1, 1))

    daq.main(["--sensors", "S4"])

    assert seen == ["S4"]
