import math

import numpy as np
import pandas as pd
import pytest

from src.data import features as feat
from src.data.sensors import SENSORS

TZ = "Asia/Karachi"
SENSOR_IDS = [s.id for s in SENSORS]

DEFAULT_WEATHER = {
    "temperature_2m": 25.0,
    "relative_humidity_2m": 40.0,
    "dew_point_2m": 10.0,
    "surface_pressure": 1010.0,
    "precipitation": 0.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 0.0,
    "boundary_layer_height": 500.0,
    "cams_pm2_5": 20.0,
    "cams_pm10": 30.0,
    "cams_no2": 10.0,
    "cams_dust": 1.0,
    "cams_aod": 0.2,
}


def make_hourly(sensor_id: str, start: str, n_hours: int, pm25, flag="ok", **overrides) -> pd.DataFrame:
    """Build a minimal hourly frame (contract §1) for one sensor. `pm25`/`flag` broadcast or array."""
    times = pd.date_range(pd.Timestamp(start, tz=TZ), periods=n_hours, freq="h")
    frame = pd.DataFrame({"sensor_id": sensor_id, "time": times})
    frame["pm25_raw"] = pm25
    frame["pm25"] = pm25
    frame["pm25_flag"] = flag
    for col, value in DEFAULT_WEATHER.items():
        frame[col] = value
    for col, value in overrides.items():
        frame[col] = value
    return frame


def multi_sensor_hourly(sensor_ids, start, n_hours, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for sid in sensor_ids:
        pm25 = rng.uniform(5, 80, n_hours)
        direction = rng.uniform(0, 360, n_hours)
        speed = rng.uniform(0, 10, n_hours)
        frames.append(
            make_hourly(
                sid,
                start,
                n_hours,
                pm25,
                wind_direction_10m=direction,
                wind_speed_10m=speed,
                temperature_2m=rng.uniform(15, 40, n_hours),
                precipitation=rng.uniform(0, 2, n_hours),
                cams_pm2_5=rng.uniform(5, 80, n_hours),
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- exact numeric checks -------------------------------------------------------


def test_pm25_lag_1h_equals_2300_hour_of_prev_day():
    # 10 days, sequential integer PM2.5 so every lag/rolling value is hand-checkable.
    n_hours = 24 * 10
    pm25 = np.arange(n_hours, dtype=float)
    hourly = make_hourly("S1", "2025-06-01 00:00", n_hours, pm25)

    features, _ = feat.build_daily_features(hourly)

    row = features[features["date"] == pd.Timestamp("2025-06-06")].iloc[0]
    # 2025-06-06 00:00 is hour offset 5*24=120; 23:00 on 06-05 is offset 119.
    assert row["pm25_lag_1h"] == 119.0
    assert row["pm25_lag_24h"] == 96.0


def test_y_pm25_equals_mean_of_days_hours():
    n_hours = 24 * 10
    pm25 = np.arange(n_hours, dtype=float)
    hourly = make_hourly("S1", "2025-06-01 00:00", n_hours, pm25)

    features, _ = feat.build_daily_features(hourly)

    row = features[features["date"] == pd.Timestamp("2025-06-06")].iloc[0]
    expected = pm25[5 * 24 : 6 * 24].mean()
    assert row["y_pm25"] == pytest.approx(expected)
    assert row["y_hours"] == 24


def test_rolling_mean_and_std_match_manual_window():
    n_hours = 24 * 10
    pm25 = np.arange(n_hours, dtype=float)
    hourly = make_hourly("S1", "2025-06-01 00:00", n_hours, pm25)

    features, _ = feat.build_daily_features(hourly)

    row = features[features["date"] == pd.Timestamp("2025-06-06")].iloc[0]
    t0 = 5 * 24  # hour offset of 2025-06-06 00:00
    assert row["pm25_mean_24h"] == pytest.approx(pm25[t0 - 24 : t0].mean())
    assert row["pm25_mean_72h"] == pytest.approx(pm25[t0 - 72 : t0].mean())
    assert row["pm25_mean_168h"] == pytest.approx(pm25[t0 - 120 : t0].mean())  # only 120h exist
    assert row["pm25_std_24h"] == pytest.approx(pd.Series(pm25[t0 - 24 : t0]).std())


def test_prev_day_mean_matches_mean_of_d_minus_1():
    n_hours = 24 * 10
    pm25 = np.arange(n_hours, dtype=float)
    hourly = make_hourly("S1", "2025-06-01 00:00", n_hours, pm25)

    features, _ = feat.build_daily_features(hourly)

    row = features[features["date"] == pd.Timestamp("2025-06-06")].iloc[0]
    expected = pm25[4 * 24 : 5 * 24].mean()
    assert row["pm25_prev_day_mean"] == pytest.approx(expected)
    # with a fully-aligned 24h window, pm25_mean_24h covers the identical hours.
    assert row["pm25_mean_24h"] == pytest.approx(row["pm25_prev_day_mean"])


# --- 18-hour target rule --------------------------------------------------------


def _day_with_n_valid_hours(n_valid: int) -> pd.DataFrame:
    """One day (+1 padding day before) where exactly `n_valid` of day 2's 24 hours are valid."""
    pm25 = np.full(48, 20.0)
    flag = np.array(["ok"] * 48, dtype=object)
    n_invalid = 24 - n_valid
    pm25[24 : 24 + n_invalid] = np.nan
    flag[24 : 24 + n_invalid] = "missing"
    return make_hourly("S1", "2025-06-01 00:00", 48, pm25, flag=flag)


def test_18_valid_hours_gives_a_target_17_does_not():
    features_18, report_18 = feat.build_daily_features(_day_with_n_valid_hours(18))
    row = features_18[features_18["date"] == pd.Timestamp("2025-06-02")]
    assert len(row) == 1
    assert row.iloc[0]["y_hours"] == 18
    assert pd.notna(row.iloc[0]["y_pm25"])

    features_17, report_17 = feat.build_daily_features(_day_with_n_valid_hours(17))
    row17 = features_17[features_17["date"] == pd.Timestamp("2025-06-02")]
    assert len(row17) == 0  # dropped: y_pm25 was NaN
    assert report_17["dropped_for_target"]["S1"] >= 1


# --- prev-day 12-hour rule -------------------------------------------------------


def _two_days_with_prev_day_valid_hours(n_valid: int) -> pd.DataFrame:
    """Day 1 (D-1) has `n_valid` valid hours; day 2 (D) is fully valid."""
    pm25 = np.full(48, 20.0)
    flag = np.array(["ok"] * 48, dtype=object)
    n_invalid = 24 - n_valid
    pm25[:n_invalid] = np.nan
    flag[:n_invalid] = "missing"
    return make_hourly("S1", "2025-06-01 00:00", 48, pm25, flag=flag)


def test_prev_day_mean_requires_12_valid_hours():
    features_12, _ = feat.build_daily_features(_two_days_with_prev_day_valid_hours(12))
    row12 = features_12[features_12["date"] == pd.Timestamp("2025-06-02")]
    assert len(row12) == 1  # kept: prev-day mean defined

    features_11, report_11 = feat.build_daily_features(_two_days_with_prev_day_valid_hours(11))
    row11 = features_11[features_11["date"] == pd.Timestamp("2025-06-02")]
    assert len(row11) == 0  # dropped: prev-day mean NaN
    assert report_11["dropped_for_prev_day"]["S1"] >= 1


# --- trailing-window 50% rule ----------------------------------------------------


def test_rolling_mean_24h_requires_half_valid_hours():
    # D-1 (hours 0-23) has 12 valid -> pm25_mean_24h defined; 11 valid -> NaN.
    hourly_12 = _two_days_with_prev_day_valid_hours(12)
    features_12, _ = feat.build_daily_features(hourly_12)
    row12 = features_12[features_12["date"] == pd.Timestamp("2025-06-02")].iloc[0]
    assert pd.notna(row12["pm25_mean_24h"])

    hourly_11 = _two_days_with_prev_day_valid_hours(11)
    day_frame = feat._prepare_hourly(hourly_11)
    lag = feat._hourly_lag_features(day_frame)
    rolling = feat._hourly_rolling_features(day_frame)
    scaffold = feat._day_level_lag_rolling(day_frame, lag, rolling)
    row11 = scaffold.loc[("S1", pd.Timestamp("2025-06-02"))]
    assert pd.isna(row11["pm25_mean_24h"])


# --- wind u/v components ---------------------------------------------------------


@pytest.mark.parametrize(
    "direction,expected_u_sign,expected_v_sign",
    [
        (0.0, 0.0, -1),  # from the north -> blows south -> v negative
        (180.0, 0.0, 1),  # from the south -> blows north -> v positive
        (90.0, -1, 0.0),  # from the east -> blows west -> u negative
        (270.0, 1, 0.0),  # from the west -> blows east -> u positive
    ],
)
def test_wind_u_v_known_directions(direction, expected_u_sign, expected_v_sign):
    hourly = make_hourly("S1", "2025-06-01 00:00", 1, [20.0], wind_direction_10m=[direction], wind_speed_10m=[5.0])
    wind = feat._hourly_wind_components(hourly)
    u, v = wind.iloc[0]["wind_u"], wind.iloc[0]["wind_v"]
    if expected_u_sign == 0.0:
        assert u == pytest.approx(0.0, abs=1e-9)
    else:
        assert np.sign(u) == expected_u_sign
    if expected_v_sign == 0.0:
        assert v == pytest.approx(0.0, abs=1e-9)
    else:
        assert np.sign(v) == expected_v_sign


def test_wind_from_north_gives_negative_v_in_full_pipeline():
    n_hours = 48
    hourly = make_hourly(
        "S1", "2025-06-01 00:00", n_hours, np.full(n_hours, 20.0), wind_direction_10m=0.0, wind_speed_10m=5.0
    )
    features, _ = feat.build_daily_features(hourly)
    row = features[features["date"] == pd.Timestamp("2025-06-02")].iloc[0]
    assert row["past_wind_v"] < 0
    assert row["fc_wind_v"] < 0


# --- one-hot sensor columns -------------------------------------------------------


def test_one_hot_has_all_14_sensor_columns_even_if_absent():
    hourly = make_hourly("S3", "2025-06-01 00:00", 48, np.full(48, 20.0))
    features, _ = feat.build_daily_features(hourly)

    onehot_cols = [f"sensor_{sid}" for sid in SENSOR_IDS]
    assert all(col in features.columns for col in onehot_cols)
    assert len(onehot_cols) == 14
    row = features.iloc[0]
    assert row["sensor_S3"] == 1
    assert sum(row[col] for col in onehot_cols) == 1


# --- calendar dates are PKT, no UTC off-by-one -----------------------------------


def test_dates_are_pkt_calendar_days():
    hourly = make_hourly("S1", "2025-06-01 00:00", 48, np.full(48, 20.0))
    features, _ = feat.build_daily_features(hourly)

    assert pd.Timestamp("2025-06-02") in set(features["date"])
    # midnight PKT (00:00 +05:00) is 19:00 UTC the previous day; the date must stay 06-02 PKT.
    row = features[features["date"] == pd.Timestamp("2025-06-02")].iloc[0]
    assert row["date"] == pd.Timestamp("2025-06-02")
    assert row["date"].tzinfo is None  # midnight-normalised, tz-naive per our column contract


def test_calendar_features_values():
    date_series = pd.Series([pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-04")])
    calendar = feat._calendar_features(date_series)
    assert calendar.loc[0, "month"] == 1
    assert calendar.loc[1, "month"] == 7
    assert calendar["doy_sin"].between(-1, 1).all()
    assert calendar["dow_sin"].between(-1, 1).all()


# --- leakage: altering the future must not change past/present-boundary features ----


def test_future_alteration_does_not_change_past_features_but_changes_fc_and_targets():
    n_hours = 24 * 12
    hourly = multi_sensor_hourly(["S1", "S2"], "2025-06-01 00:00", n_hours, seed=1)

    features_before, _ = feat.build_daily_features(hourly)

    boundary = pd.Timestamp("2025-06-07", tz=TZ)  # day D: altering from here on
    future_mask = hourly["time"] >= boundary
    rng = np.random.default_rng(99)
    altered = hourly.copy()
    n_future = future_mask.sum()
    altered.loc[future_mask, "pm25"] = rng.uniform(5, 80, n_future)
    altered.loc[future_mask, "pm25_raw"] = altered.loc[future_mask, "pm25"]
    altered.loc[future_mask, "temperature_2m"] = rng.uniform(15, 40, n_future)
    altered.loc[future_mask, "wind_direction_10m"] = rng.uniform(0, 360, n_future)
    altered.loc[future_mask, "wind_speed_10m"] = rng.uniform(0, 10, n_future)
    altered.loc[future_mask, "precipitation"] = rng.uniform(0, 2, n_future)
    altered.loc[future_mask, "cams_pm2_5"] = rng.uniform(5, 80, n_future)

    features_after, _ = feat.build_daily_features(altered)

    unchanged_cols = (
        [f"pm25_lag_{k}h" for k in feat.LAG_HOURS]
        + [f"pm25_mean_{w}h" for w in feat.ROLLING_WINDOWS]
        + ["pm25_std_24h", "pm25_prev_day_mean", "city_prev_day_mean"]
        + [f"past_{var}" for var in feat.DAILY_AGG_VARS]
    )
    changed_cols = ["y_pm25"] + [f"fc_{var}" for var in feat.DAILY_AGG_VARS]

    for sid in ["S1", "S2"]:
        before_row = features_before[(features_before["sensor_id"] == sid) & (features_before["date"] == boundary.tz_localize(None))]
        after_row = features_after[(features_after["sensor_id"] == sid) & (features_after["date"] == boundary.tz_localize(None))]
        assert len(before_row) == 1 and len(after_row) == 1
        before_row, after_row = before_row.iloc[0], after_row.iloc[0]

        for col in unchanged_cols:
            b, a = before_row[col], after_row[col]
            if pd.isna(b) and pd.isna(a):
                continue
            assert b == pytest.approx(a), f"{col} changed for {sid}: {b} vs {a}"

        differs = any(
            not (pd.isna(before_row[col]) and pd.isna(after_row[col])) and before_row[col] != pytest.approx(after_row[col])
            for col in changed_cols
        )
        assert differs, f"expected at least one of {changed_cols} to change for {sid}"


def test_future_alteration_does_not_change_earlier_days_at_all():
    n_hours = 24 * 12
    hourly = multi_sensor_hourly(["S1"], "2025-06-01 00:00", n_hours, seed=2)
    features_before, _ = feat.build_daily_features(hourly)

    boundary = pd.Timestamp("2025-06-07", tz=TZ)
    future_mask = hourly["time"] >= boundary
    rng = np.random.default_rng(7)
    altered = hourly.copy()
    altered.loc[future_mask, "pm25"] = rng.uniform(5, 80, future_mask.sum())
    altered.loc[future_mask, "pm25_raw"] = altered.loc[future_mask, "pm25"]
    features_after, _ = feat.build_daily_features(altered)

    before_row = features_before[features_before["date"] < boundary.tz_localize(None)].reset_index(drop=True)
    after_row = features_after[features_after["date"] < boundary.tz_localize(None)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_row, after_row)


# --- report shape and column list -------------------------------------------------


def test_report_counts_candidate_dropped_and_kept():
    hourly = _day_with_n_valid_hours(18)
    _, report = feat.build_daily_features(hourly)

    for key in ("candidate_rows", "dropped_for_target", "dropped_for_prev_day", "kept"):
        assert key in report
        assert "total" in report[key]
        assert "S1" in report[key]

    assert report["candidate_rows"]["total"] == report["kept"]["total"] + report["dropped_for_target"]["total"] + report["dropped_for_prev_day"]["total"]


def test_build_daily_features_column_list_matches_contract():
    hourly = multi_sensor_hourly(["S1", "S2"], "2025-06-01 00:00", 24 * 10, seed=3)
    features, _ = feat.build_daily_features(hourly)
    assert list(features.columns) == feat.COLUMN_ORDER


def test_dropped_rows_are_not_double_counted():
    # A day with both <18 valid hours and (necessarily) a knock-on missing prev-day elsewhere
    # must be counted exactly once, under dropped_for_target.
    hourly = _day_with_n_valid_hours(5)
    _, report = feat.build_daily_features(hourly)
    total_buckets = report["dropped_for_target"]["total"] + report["dropped_for_prev_day"]["total"] + report["kept"]["total"]
    assert total_buckets == report["candidate_rows"]["total"]


# --- performance sanity -----------------------------------------------------------


def test_runs_quickly_for_full_scale_data():
    import time

    hourly = multi_sensor_hourly(SENSOR_IDS, "2025-06-01 00:00", 24 * 60, seed=4)  # 14 sensors x ~1440h
    start = time.perf_counter()
    features, report = feat.build_daily_features(hourly)
    elapsed = time.perf_counter() - start

    assert elapsed < 20
    assert len(features) > 0
    assert report["candidate_rows"]["total"] == 14 * 60
