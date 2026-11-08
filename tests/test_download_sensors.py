import json
from datetime import date

import pandas as pd

from src.data import download_sensors as ds
from src.data.sensors import US_CONSULATE, sensor_by_id

SENSOR = sensor_by_id("S1")


def _fake_frame(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="h", tz="Asia/Karachi"),
            "pm25": [10.0] * rows,
        }
    )


# --- parse_args ----------------------------------------------------------------


def test_parse_args_defaults():
    args = ds.parse_args([])
    assert args.start is None
    assert args.end is None
    assert args.sensors is None
    assert args.consulate is False
    assert args.force is False


def test_parse_args_overrides():
    args = ds.parse_args(
        ["--start", "2025-07-01", "--end", "2025-07-05", "--sensors", "S1", "S2", "--consulate", "--force"]
    )
    assert args.start == date(2025, 7, 1)
    assert args.end == date(2025, 7, 5)
    assert args.sensors == ["S1", "S2"]
    assert args.consulate is True
    assert args.force is True


# --- download_sensor -----------------------------------------------------------


def test_download_sensor_writes_csv_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "fetch_sensor_dataframe", lambda *a, **k: _fake_frame(5))

    ds.download_sensor(SENSOR, date(2025, 6, 1), date(2025, 6, 2), force=False, api_key="key", out_dir=tmp_path)

    csv_path = tmp_path / "S1.csv"
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 5

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["S1"]
    assert entry["openaq_location_id"] == SENSOR.openaq_location_id
    assert entry["openaq_sensor_id"] == SENSOR.openaq_sensor_id
    assert entry["row_count"] == 5
    assert entry["licence"] == ds.LICENCE
    assert entry["provider"] == ds.SENSOR_PROVIDER
    assert entry["start_date"] == "2025-06-01"
    assert entry["end_date"] == "2025-06-02"
    assert "downloaded_at_utc" in entry
    assert str(SENSOR.openaq_sensor_id) in entry["endpoint"]


def test_download_sensor_skips_existing_without_force(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(ds, "fetch_sensor_dataframe", lambda *a, **k: calls.append(1) or _fake_frame())
    (tmp_path / "S1.csv").write_text("time,pm25\n")

    ds.download_sensor(SENSOR, date(2025, 6, 1), date(2025, 6, 2), force=False, api_key="key", out_dir=tmp_path)

    assert calls == []
    assert "skip" in capsys.readouterr().out


def test_download_sensor_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "fetch_sensor_dataframe", lambda *a, **k: _fake_frame(9))
    (tmp_path / "S1.csv").write_text("stale")

    ds.download_sensor(SENSOR, date(2025, 6, 1), date(2025, 6, 2), force=True, api_key="key", out_dir=tmp_path)

    assert len(pd.read_csv(tmp_path / "S1.csv")) == 9


def test_download_sensor_never_prints_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ds, "fetch_sensor_dataframe", lambda *a, **k: _fake_frame(2))

    ds.download_sensor(
        SENSOR, date(2025, 6, 1), date(2025, 6, 2), force=False, api_key="super-secret", out_dir=tmp_path
    )

    assert "super-secret" not in capsys.readouterr().out


# --- download_consulate ---------------------------------------------------------


def test_download_consulate_writes_to_location_id_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "fetch_sensor_dataframe", lambda *a, **k: _fake_frame(4))

    ds.download_consulate(date(2024, 1, 1), date(2024, 1, 2), force=False, api_key="key", out_dir=tmp_path)

    csv_path = tmp_path / f"{US_CONSULATE.openaq_location_id}.csv"
    assert csv_path.exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["US_CONSULATE"]
    assert entry["provider"] == ds.CONSULATE_PROVIDER
    assert entry["openaq_location_id"] == US_CONSULATE.openaq_location_id
    assert entry["openaq_sensor_id"] == US_CONSULATE.openaq_sensor_id


# --- main ------------------------------------------------------------------------


def test_main_downloads_sensor_subset(monkeypatch):
    seen = []
    monkeypatch.setattr(ds, "get_api_key", lambda: "key")
    monkeypatch.setattr(ds, "download_sensor", lambda sensor, start, end, force, api_key: seen.append(sensor.id))

    ds.main(["--sensors", "S2", "S3"])

    assert seen == ["S2", "S3"]


def test_main_defaults_to_all_sensors(monkeypatch):
    seen = []
    monkeypatch.setattr(ds, "get_api_key", lambda: "key")
    monkeypatch.setattr(ds, "download_sensor", lambda sensor, start, end, force, api_key: seen.append(sensor.id))

    ds.main([])

    assert seen == [s.id for s in ds.SENSORS]


def test_main_defaults_use_default_start_and_today(monkeypatch):
    seen = {}
    monkeypatch.setattr(ds, "get_api_key", lambda: "key")

    def fake_download(sensor, start, end, force, api_key):
        seen["start"] = start
        seen["end"] = end

    monkeypatch.setattr(ds, "download_sensor", fake_download)
    monkeypatch.setattr(ds, "default_end", lambda: date(2026, 11, 8))

    ds.main(["--sensors", "S1"])

    assert seen["start"] == ds.DEFAULT_START
    assert seen["end"] == date(2026, 11, 8)


def test_main_consulate_flag_downloads_consulate(monkeypatch):
    called = {}
    monkeypatch.setattr(ds, "get_api_key", lambda: "key")

    def fake_download_consulate(start, end, force, api_key):
        called["start"] = start
        called["end"] = end

    monkeypatch.setattr(ds, "download_consulate", fake_download_consulate)
    monkeypatch.setattr(ds, "download_sensor", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))

    ds.main(["--consulate"])

    assert called["start"] == ds.DEFAULT_CONSULATE_START
    assert called["end"] == ds.DEFAULT_CONSULATE_END


def test_main_propagates_missing_api_key_error(monkeypatch):
    def raise_missing():
        raise RuntimeError("OPENAQ_API_KEY is not set.")

    monkeypatch.setattr(ds, "get_api_key", raise_missing)

    try:
        ds.main([])
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "OPENAQ_API_KEY" in str(exc)
