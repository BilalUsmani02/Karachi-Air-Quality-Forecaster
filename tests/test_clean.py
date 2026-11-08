from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import clean
from src.data.sensors import SENSORS

TZ = "Asia/Karachi"


def _hourly_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz=TZ)


def _series(start: str, values: list[float | None]) -> pd.Series:
    return pd.Series(values, index=_hourly_index(start, len(values)), dtype=float, name="pm25")


# --- loaders -----------------------------------------------------------------


def test_load_sensor_raw_parses_tz_aware_time(tmp_path):
    frame = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 3), "pm25": [1.0, 2.0, 3.0]})
    path = tmp_path / "S1.csv"
    frame.to_csv(path, index=False)

    loaded = clean.load_sensor_raw(path)

    assert str(loaded["time"].dt.tz) == TZ
    assert loaded["pm25"].tolist() == [1.0, 2.0, 3.0]


def test_load_weather_raw_parses_tz_aware_time(tmp_path):
    frame = pd.DataFrame(
        {"time": _hourly_index("2025-05-01 00:00", 2), "temperature_2m": [20.0, 21.0]}
    )
    path = tmp_path / "S1.csv"
    frame.to_csv(path, index=False)

    loaded = clean.load_weather_raw(path)

    assert str(loaded["time"].dt.tz) == TZ
    assert loaded["temperature_2m"].tolist() == [20.0, 21.0]


def test_load_cams_raw_renames_columns_and_parses_time(tmp_path):
    frame = pd.DataFrame(
        {
            "time": _hourly_index("2025-05-01 00:00", 2),
            "pm2_5": [10.0, 11.0],
            "pm10": [20.0, 21.0],
            "nitrogen_dioxide": [1.0, 1.1],
            "dust": [0.1, 0.2],
            "aerosol_optical_depth": [0.01, 0.02],
        }
    )
    path = tmp_path / "S1.csv"
    frame.to_csv(path, index=False)

    loaded = clean.load_cams_raw(path)

    assert list(loaded.columns) == ["time", "cams_pm2_5", "cams_pm10", "cams_no2", "cams_dust", "cams_aod"]
    assert str(loaded["time"].dt.tz) == TZ


# --- regular_hourly_grid ----------------------------------------------------


def test_regular_hourly_grid_spans_weather_start_to_sensor_end():
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 5), "pm25": [1.0] * 5})
    weather_raw = pd.DataFrame({"time": _hourly_index("2025-05-01 00:00", 3), "temperature_2m": [1.0] * 3})
    cams_raw = pd.DataFrame({"time": _hourly_index("2025-05-01 00:00", 3), "cams_pm2_5": [1.0] * 3})

    grid = clean.regular_hourly_grid(sensor_raw, weather_raw, cams_raw)

    assert grid[0] == pd.Timestamp("2025-05-01 00:00", tz=TZ)
    assert grid[-1] == sensor_raw["time"].max()


def test_regular_hourly_grid_is_strictly_hourly():
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 2), "pm25": [1.0, 2.0]})
    weather_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 2), "temperature_2m": [1.0, 2.0]})
    cams_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 2), "cams_pm2_5": [1.0, 2.0]})

    grid = clean.regular_hourly_grid(sensor_raw, weather_raw, cams_raw)

    assert list(grid.to_series().diff().dropna().unique()) == [pd.Timedelta(hours=1)]


# --- flag_negative -----------------------------------------------------------


def test_flag_negative_flags_values_below_zero():
    series = _series("2025-06-01 00:00", [-1.0, 0.0, 5.0, None])
    mask = clean.flag_negative(series)
    assert mask.tolist() == [True, False, False, False]


def test_flag_negative_does_not_mutate_input():
    series = _series("2025-06-01 00:00", [-1.0, 5.0])
    before = series.copy()
    clean.flag_negative(series)
    pd.testing.assert_series_equal(series, before)


# --- flag_spikes --------------------------------------------------------------


def test_flag_spikes_hard_threshold_over_500():
    series = _series("2025-06-01 00:00", [10.0] * 6 + [600.0])
    mask = clean.flag_spikes(series)
    assert mask.iloc[-1] is np.True_ or bool(mask.iloc[-1]) is True


def test_flag_spikes_relative_threshold_needs_min_valid_hours():
    # Only 5 valid trailing hours (just below the 6 needed) -> relative rule not applied.
    series = _series("2025-06-01 00:00", [10.0] * 5 + [None, 60.0])
    mask = clean.flag_spikes(series)
    assert mask.iloc[-1] == False  # noqa: E712


def test_flag_spikes_relative_threshold_flags_with_enough_valid_hours():
    # median 30 -> 5x = 150; 160 clears both the relative and the 100 floor.
    series = _series("2025-06-01 00:00", [30.0] * 6 + [160.0])
    mask = clean.flag_spikes(series)
    assert mask.iloc[-1] == True  # noqa: E712


def test_flag_spikes_low_level_noise_not_flagged():
    # 3 -> 16 is > 5x median but below the 100 abs floor, so it must not be flagged.
    series = _series("2025-06-01 00:00", [3.0] * 6 + [16.0])
    mask = clean.flag_spikes(series)
    assert mask.iloc[-1] == False  # noqa: E712


def test_flag_spikes_spike_right_after_gap_insufficient_history():
    series = _series("2025-06-01 00:00", [10.0, 10.0, None, None, None, None, None, 1000.0])
    mask = clean.flag_spikes(series)
    # Only 2 valid hours precede the gap -> relative rule can't apply, but the hard
    # threshold (>500) still catches it.
    assert mask.iloc[-1] == True  # noqa: E712


def test_flag_spikes_is_causal_future_values_dont_change_past_flags():
    base = [10.0] * 10 + [12.0, 11.0, 10.0]
    series_a = _series("2025-06-01 00:00", base)
    mask_a = clean.flag_spikes(series_a)

    changed = base.copy()
    changed[-1] = 9000.0  # a huge change far in the future
    series_b = _series("2025-06-01 00:00", changed)
    mask_b = clean.flag_spikes(series_b)

    pd.testing.assert_series_equal(mask_a.iloc[:-1], mask_b.iloc[:-1])


def test_flag_spikes_excludes_already_masked_negative_from_window_count():
    # 5 real valid hours + 1 negative that must NOT count as a 6th valid hour.
    baseline = [40.0] * 5
    series = _series("2025-06-01 00:00", baseline + [-5.0, 250.0])

    # Wrong order: spike check sees the raw -5 as an ordinary number -> 6 "valid" hours,
    # median 40, 5x = 200, and 250 clears both that and the 100 floor -> (incorrectly) flagged.
    mask_wrong_order = clean.flag_spikes(series)
    assert mask_wrong_order.iloc[-1] == True  # noqa: E712

    # Correct order: negative is masked to NaN first, leaving only 5 valid hours in the
    # window -> below min_valid_hours, so the relative rule is not applied at all.
    masked = series.mask(clean.flag_negative(series))
    mask_correct_order = clean.flag_spikes(masked)
    assert mask_correct_order.iloc[-1] == False  # noqa: E712


def test_flag_spikes_does_not_mutate_input():
    series = _series("2025-06-01 00:00", [10.0] * 6 + [600.0])
    before = series.copy()
    clean.flag_spikes(series)
    pd.testing.assert_series_equal(series, before)


# --- flag_flatlines ------------------------------------------------------------


def test_flag_flatlines_flags_24_consecutive_identical():
    series = _series("2025-06-01 00:00", [5.0] * 24)
    mask = clean.flag_flatlines(series)
    assert mask.all()


def test_flag_flatlines_does_not_flag_23_consecutive():
    series = _series("2025-06-01 00:00", [5.0] * 23)
    mask = clean.flag_flatlines(series)
    assert not mask.any()


def test_flag_flatlines_nan_breaks_the_run():
    values = [5.0] * 12 + [None] + [5.0] * 12
    series = _series("2025-06-01 00:00", values)
    mask = clean.flag_flatlines(series)
    assert not mask.any()


def test_flag_flatlines_does_not_mutate_input():
    series = _series("2025-06-01 00:00", [5.0] * 24)
    before = series.copy()
    clean.flag_flatlines(series)
    pd.testing.assert_series_equal(series, before)


# --- interpolate_short_gaps -----------------------------------------------------


def test_interpolate_short_gaps_fills_gap_of_3():
    series = _series("2025-06-01 00:00", [10.0, None, None, None, 20.0])
    filled = clean.interpolate_short_gaps(series, max_gap=3)
    assert filled.tolist() == pytest.approx([10.0, 12.5, 15.0, 17.5, 20.0])


def test_interpolate_short_gaps_does_not_fill_gap_of_4():
    series = _series("2025-06-01 00:00", [10.0, None, None, None, None, 20.0])
    filled = clean.interpolate_short_gaps(series, max_gap=3)
    assert filled.iloc[1:5].isna().all()


def test_interpolate_short_gaps_leading_nan_not_filled():
    series = _series("2025-06-01 00:00", [None, None, 10.0, 20.0])
    filled = clean.interpolate_short_gaps(series, max_gap=3)
    assert filled.iloc[:2].isna().all()


def test_interpolate_short_gaps_trailing_nan_not_filled():
    series = _series("2025-06-01 00:00", [10.0, 20.0, None, None])
    filled = clean.interpolate_short_gaps(series, max_gap=3)
    assert filled.iloc[2:].isna().all()


def test_interpolate_short_gaps_does_not_mutate_input():
    series = _series("2025-06-01 00:00", [10.0, None, None, 20.0])
    before = series.copy()
    clean.interpolate_short_gaps(series, max_gap=3)
    pd.testing.assert_series_equal(series, before)


# --- clean_sensor_hourly --------------------------------------------------------


def test_clean_sensor_hourly_ok_for_plain_valid_readings():
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 4), "pm25": [1.0, 0.0, 5.0, 3.0]})
    result = clean.clean_sensor_hourly(sensor_raw)
    assert result["pm25_flag"].tolist() == ["ok", "ok", "ok", "ok"]
    assert result["pm25"].tolist() == [1.0, 0.0, 5.0, 3.0]


def test_clean_sensor_hourly_flags_negative_before_computing_spike_window():
    # A negative reading must be excluded from the spike window's valid-hour count, not
    # counted as an ordinary (small) number. With only 5 real hours before it, the
    # candidate below has too little history for the relative spike rule to apply, so it
    # must come out "ok", not "spike".
    values = [40.0] * 5 + [-5.0, 250.0]
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", len(values)), "pm25": values})
    result = clean.clean_sensor_hourly(sensor_raw)
    assert result["pm25_flag"].iloc[5] == "interpolated"  # 1h gap, bounded both sides
    assert result["pm25_flag"].iloc[6] == "ok"


def test_clean_sensor_hourly_interpolated_overrides_rejection_reason_for_short_gap():
    values = [10.0, -5.0, 10.0]
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 3), "pm25": values})
    result = clean.clean_sensor_hourly(sensor_raw)
    assert result["pm25_flag"].tolist() == ["ok", "interpolated", "ok"]
    assert result["pm25"].iloc[1] == pytest.approx(10.0)


def test_clean_sensor_hourly_long_gap_keeps_missing():
    values = [10.0, None, None, None, None, 10.0]
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", len(values)), "pm25": values})
    result = clean.clean_sensor_hourly(sensor_raw)
    assert result["pm25_flag"].iloc[1:5].tolist() == ["missing"] * 4
    assert result["pm25"].iloc[1:5].isna().all()


def test_clean_sensor_hourly_internal_missing_hour_from_reindex_is_missing():
    # Raw CSV skips an hour entirely (e.g. hour 1); reindexing to the sensor's own hourly
    # grid must surface it as a NaN row flagged "missing" (then possibly interpolated).
    sensor_raw = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2025-06-01 00:00", tz=TZ),
                pd.Timestamp("2025-06-01 02:00", tz=TZ),
            ],
            "pm25": [10.0, 12.0],
        }
    )
    result = clean.clean_sensor_hourly(sensor_raw)
    assert len(result) == 3
    assert result["pm25_flag"].iloc[1] == "interpolated"
    assert result["pm25"].iloc[1] == pytest.approx(11.0)


def test_clean_sensor_hourly_flatline_exactly_24_is_flagged_and_23_is_not():
    values_24 = [7.0] * 24
    sensor_raw_24 = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 24), "pm25": values_24})
    result_24 = clean.clean_sensor_hourly(sensor_raw_24)
    assert (result_24["pm25_flag"] == "flatline").all()

    values_23 = [7.0] * 23
    sensor_raw_23 = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 23), "pm25": values_23})
    result_23 = clean.clean_sensor_hourly(sensor_raw_23)
    assert (result_23["pm25_flag"] == "ok").all()


def test_clean_sensor_hourly_does_not_mutate_input():
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 4), "pm25": [1.0, -2.0, 600.0, 3.0]})
    before = sensor_raw.copy()
    clean.clean_sensor_hourly(sensor_raw)
    pd.testing.assert_frame_equal(sensor_raw, before)


def test_clean_sensor_hourly_columns():
    sensor_raw = pd.DataFrame({"time": _hourly_index("2025-06-01 00:00", 2), "pm25": [1.0, 2.0]})
    result = clean.clean_sensor_hourly(sensor_raw)
    assert list(result.columns) == ["time", "pm25_raw", "pm25", "pm25_flag"]


# --- clean_met -----------------------------------------------------------------


def test_clean_met_interpolates_boundary_layer_height_up_to_6h():
    values = [100.0] + [None] * 6 + [200.0]
    frame = pd.DataFrame(
        {"time": _hourly_index("2025-06-01 00:00", len(values)), "boundary_layer_height": values}
    )
    result = clean.clean_met(frame)
    assert result["boundary_layer_height"].isna().sum() == 0


def test_clean_met_does_not_interpolate_blh_gap_of_7h():
    values = [100.0] + [None] * 7 + [200.0]
    frame = pd.DataFrame(
        {"time": _hourly_index("2025-06-01 00:00", len(values)), "boundary_layer_height": values}
    )
    result = clean.clean_met(frame)
    assert result["boundary_layer_height"].isna().sum() == 7


def test_clean_met_other_columns_limited_to_3h():
    values = [10.0, None, None, None, 20.0]
    frame = pd.DataFrame(
        {"time": _hourly_index("2025-06-01 00:00", len(values)), "temperature_2m": values}
    )
    result = clean.clean_met(frame)
    assert result["temperature_2m"].isna().sum() == 0

    values_4 = [10.0, None, None, None, None, 20.0]
    frame_4 = pd.DataFrame(
        {"time": _hourly_index("2025-06-01 00:00", len(values_4)), "temperature_2m": values_4}
    )
    result_4 = clean.clean_met(frame_4)
    assert result_4["temperature_2m"].isna().sum() == 4


def test_clean_met_does_not_mutate_input():
    values = [10.0, None, None, 20.0]
    frame = pd.DataFrame(
        {"time": _hourly_index("2025-06-01 00:00", len(values)), "temperature_2m": values}
    )
    before = frame.copy()
    clean.clean_met(frame)
    pd.testing.assert_frame_equal(frame, before)


# --- build_hourly_table ----------------------------------------------------------


def _write_raw_sensor_files(raw_dir: Path, sensor_id: str) -> None:
    sensor_raw = pd.DataFrame(
        {"time": _hourly_index("2025-06-01 00:00", 4), "pm25": [1.0, 2.0, 3.0, 4.0]}
    )
    hours = 750  # 2025-05-01 00:00 .. past the sensor's last reading (2025-06-01 03:00)
    weather_raw = pd.DataFrame(
        {
            "time": _hourly_index("2025-05-01 00:00", hours),
            "temperature_2m": [20.0] * hours,
            "boundary_layer_height": [500.0] * hours,
        }
    )
    cams_raw = pd.DataFrame(
        {
            "time": _hourly_index("2025-05-01 00:00", hours),
            "pm2_5": [5.0] * hours,
            "pm10": [10.0] * hours,
            "nitrogen_dioxide": [1.0] * hours,
            "dust": [0.1] * hours,
            "aerosol_optical_depth": [0.02] * hours,
        }
    )
    (raw_dir / "sensors").mkdir(parents=True, exist_ok=True)
    (raw_dir / "weather").mkdir(parents=True, exist_ok=True)
    (raw_dir / "air_quality").mkdir(parents=True, exist_ok=True)
    sensor_raw.to_csv(raw_dir / "sensors" / f"{sensor_id}.csv", index=False)
    weather_raw.to_csv(raw_dir / "weather" / f"{sensor_id}.csv", index=False)
    cams_raw.to_csv(raw_dir / "air_quality" / f"{sensor_id}.csv", index=False)


def test_build_hourly_table_joins_sources_and_includes_pre_sensor_month(tmp_path):
    _write_raw_sensor_files(tmp_path, "S1")

    table, report = clean.build_hourly_table("S1", raw_dir=tmp_path)

    assert list(table.columns) == [
        "sensor_id",
        "time",
        "pm25_raw",
        "pm25",
        "pm25_flag",
        "temperature_2m",
        "boundary_layer_height",
        "cams_pm2_5",
        "cams_pm10",
        "cams_no2",
        "cams_dust",
        "cams_aod",
    ]
    assert (table["sensor_id"] == "S1").all()
    # weather/CAMS start a full ~month before the sensor's first reading.
    pre_sensor_rows = table[table["time"] < pd.Timestamp("2025-06-01 00:00", tz=TZ)]
    assert len(pre_sensor_rows) > 0
    assert (pre_sensor_rows["pm25_flag"] == "missing").all()
    assert pre_sensor_rows["pm25"].isna().all()
    assert pre_sensor_rows["temperature_2m"].notna().all()


def test_build_hourly_table_report_counts_flags(tmp_path):
    _write_raw_sensor_files(tmp_path, "S1")

    _, report = clean.build_hourly_table("S1", raw_dir=tmp_path)

    assert report["sensor_id"] == "S1"
    assert sum(report["flag_counts"].values()) == report["rows"]
    assert report["flag_counts"].get("ok", 0) == 4


# --- build_all_hourly --------------------------------------------------------------


def test_build_all_hourly_defaults_to_full_sensor_registry(monkeypatch):
    seen_ids = []

    def fake_build(sensor_id, raw_dir=clean.RAW_DIR):
        seen_ids.append(sensor_id)
        frame = pd.DataFrame({"sensor_id": [sensor_id], "time": [pd.Timestamp("2025-06-01", tz=TZ)]})
        return frame, {"sensor_id": sensor_id, "rows": 1, "flag_counts": {"ok": 1}}

    monkeypatch.setattr(clean, "build_hourly_table", fake_build)

    clean.build_all_hourly()

    assert seen_ids == [sensor.id for sensor in SENSORS]


def test_build_all_hourly_respects_explicit_sensor_ids(monkeypatch):
    seen_ids = []

    def fake_build(sensor_id, raw_dir=clean.RAW_DIR):
        seen_ids.append(sensor_id)
        frame = pd.DataFrame({"sensor_id": [sensor_id], "time": [pd.Timestamp("2025-06-01", tz=TZ)]})
        return frame, {"sensor_id": sensor_id, "rows": 1, "flag_counts": {"ok": 1}}

    monkeypatch.setattr(clean, "build_hourly_table", fake_build)

    clean.build_all_hourly(sensor_ids=["S2", "S1"])

    assert seen_ids == ["S2", "S1"]


def test_build_all_hourly_concatenates_frames_and_collects_reports(monkeypatch):
    def fake_build(sensor_id, raw_dir=clean.RAW_DIR):
        frame = pd.DataFrame({"sensor_id": [sensor_id, sensor_id], "time": [1, 2]})
        return frame, {"sensor_id": sensor_id, "rows": 2, "flag_counts": {"ok": 2}}

    monkeypatch.setattr(clean, "build_hourly_table", fake_build)

    combined, reports = clean.build_all_hourly(sensor_ids=["S1", "S2"])

    assert len(combined) == 4
    assert set(reports.keys()) == {"S1", "S2"}
    assert reports["S1"]["rows"] == 2
