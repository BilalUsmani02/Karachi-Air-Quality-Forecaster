import json

import numpy as np
import pandas as pd

from src.data import build_features as bf


def _fake_hourly():
    return pd.DataFrame(
        {
            "sensor_id": ["S1", "S1"],
            "time": pd.date_range("2025-07-01", periods=2, freq="h", tz="Asia/Karachi"),
            "pm25": [10.0, 12.0],
        }
    )


def _fake_features():
    return pd.DataFrame(
        {
            "sensor_id": ["S1", "S1"],
            "date": pd.to_datetime(["2025-07-02", "2025-07-03"]),
            "y_pm25": [11.0, 13.0],
        }
    )


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(
        bf, "build_all_hourly", lambda ids, raw_dir: (_fake_hourly(), {"S1": {"ok": np.int64(2)}})
    )
    feature_report = {
        "candidate_rows": {"S1": 3, "total": 3},
        "dropped_for_target": {"S1": 1, "total": 1},
        "dropped_for_prev_day": {"S1": 0, "total": 0},
        "kept": {"S1": 2, "total": 2},
    }
    monkeypatch.setattr(bf, "build_daily_features", lambda hourly: (_fake_features(), feature_report))


def test_build_writes_outputs_and_report(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)

    report = bf.build(raw_dir=tmp_path, out_dir=tmp_path / "processed")

    out = tmp_path / "processed"
    assert len(pd.read_parquet(out / "hourly_clean.parquet")) == 2
    assert len(pd.read_parquet(out / "features.parquet")) == 2
    saved = json.loads((out / "build_report.json").read_text())
    assert saved == report
    assert saved["date_range"] == ["2025-07-02", "2025-07-03"]
    assert saved["cleaning"]["S1"]["ok"] == 2
    assert saved["features"]["kept"]["total"] == 2


def test_jsonable_converts_numpy_scalars_and_nesting():
    result = bf._jsonable({"a": np.int64(3), "b": [np.float64(1.5)], 4: {"c": "x"}})

    assert result == {"a": 3, "b": [1.5], "4": {"c": "x"}}
    json.dumps(result)


def test_main_prints_summary(tmp_path, monkeypatch, capsys):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(bf, "PROCESSED_DIR", tmp_path / "processed")
    real_build = bf.build
    monkeypatch.setattr(bf, "build", lambda ids: real_build(ids, out_dir=tmp_path / "processed"))

    bf.main([])

    out = capsys.readouterr().out
    assert "2 kept of 3 candidates" in out
    assert "2025-07-02 to 2025-07-03" in out
