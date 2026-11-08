import json
from datetime import date, timedelta

import pandas as pd

from src.data import download_weather as dw
from src.data.open_meteo import FetchResult
from src.data.sensors import sensor_by_id

SENSOR = sensor_by_id("S1")


def _fake_result(rows: int = 3) -> FetchResult:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="h", tz="Asia/Karachi"),
            "temperature_2m": [20.0] * rows,
        }
    )
    return FetchResult(frame=frame, grid_latitude=24.9, grid_longitude=67.1)


def test_default_end_is_yesterday():
    assert dw.default_end() == date.today() - timedelta(days=1)


def test_parse_args_defaults():
    args = dw.parse_args([])
    assert args.start == dw.DEFAULT_START
    assert args.end is None
    assert args.sensors is None
    assert args.force is False


def test_parse_args_overrides():
    args = dw.parse_args(["--start", "2023-01-01", "--end", "2023-02-01", "--sensors", "S1", "S2", "--force"])
    assert args.start == date(2023, 1, 1)
    assert args.end == date(2023, 2, 1)
    assert args.sensors == ["S1", "S2"]
    assert args.force is True


def test_download_one_writes_csv_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dw, "fetch_range", lambda *a, **k: _fake_result(rows=5))

    dw.download_one(SENSOR, date(2024, 1, 1), date(2024, 1, 2), force=False, out_dir=tmp_path)

    csv_path = tmp_path / "S1.csv"
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 5

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["S1"]
    assert entry["endpoint"] == dw.URL
    assert entry["variables"] == dw.HOURLY_VARS
    assert entry["requested_lat"] == SENSOR.lat
    assert entry["requested_lon"] == SENSOR.lon
    assert entry["grid_lat"] == 24.9
    assert entry["grid_lon"] == 67.1
    assert entry["row_count"] == 5
    assert entry["licence"] == dw.LICENCE
    assert entry["start_date"] == "2024-01-01"
    assert entry["end_date"] == "2024-01-02"
    assert "downloaded_at_utc" in entry


def test_download_one_skips_existing_without_force(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(dw, "fetch_range", lambda *a, **k: calls.append(1) or _fake_result())
    out_dir = tmp_path
    out_dir.mkdir(exist_ok=True)
    (out_dir / "S1.csv").write_text("time,temperature_2m\n")

    dw.download_one(SENSOR, date(2024, 1, 1), date(2024, 1, 2), force=False, out_dir=out_dir)

    assert calls == []
    assert "skip" in capsys.readouterr().out


def test_download_one_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(dw, "fetch_range", lambda *a, **k: _fake_result(rows=7))
    (tmp_path / "S1.csv").write_text("stale")

    dw.download_one(SENSOR, date(2024, 1, 1), date(2024, 1, 2), force=True, out_dir=tmp_path)

    assert len(pd.read_csv(tmp_path / "S1.csv")) == 7


def test_main_uses_sensor_subset(monkeypatch):
    seen = []
    monkeypatch.setattr(dw, "download_one", lambda sensor, start, end, force: seen.append(sensor.id))
    monkeypatch.setattr(dw, "default_end", lambda: date(2024, 1, 1))

    dw.main(["--sensors", "S2", "S3"])

    assert seen == ["S2", "S3"]


def test_main_defaults_to_all_sensors(monkeypatch):
    seen = []
    monkeypatch.setattr(dw, "download_one", lambda sensor, start, end, force: seen.append(sensor.id))
    monkeypatch.setattr(dw, "default_end", lambda: date(2024, 1, 1))

    dw.main([])

    assert seen == [s.id for s in dw.SENSORS]
