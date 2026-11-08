"""Daily per-sensor feature table for next-day PM2.5 forecasting (data contract §2).

Builds one row per (sensor_id, date D) from the hourly table produced by `clean.py`. The issue
time is D 00:00 PKT. **Leakage rule:** every lag/rolling/prev-day/`past_*` feature uses only
hours with `time < D 00:00`; only `fc_*` columns and the targets use hours inside day D.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.classify.bands import bands_for
from src.data.sensors import SENSORS

VALID_FLAGS = {"ok", "interpolated"}

LAG_HOURS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (24, 72, 168)
ROLLING_MIN_FRACTION = 0.5  # "need >= 50% valid hours, else NaN" (contract §2)

Y_MIN_VALID_HOURS = 18
PREV_DAY_MIN_VALID_HOURS = 12

SENSOR_IDS = tuple(sensor.id for sensor in SENSORS)

# Daily weather/CAMS aggregates, in contract column order, with wind_direction_10m dropped in
# favour of its derived u/v components (contract §2). All are means except `precipitation`.
DAILY_AGG_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_u",
    "wind_v",
    "boundary_layer_height",
    "cams_pm2_5",
    "cams_pm10",
    "cams_no2",
    "cams_dust",
    "cams_aod",
)
_SUM_VARS = {"precipitation"}

COLUMN_ORDER = (
    ["sensor_id", "date", "y_pm25", "y_hours", "y_band"]
    + [f"pm25_lag_{k}h" for k in LAG_HOURS]
    + [f"pm25_mean_{w}h" for w in ROLLING_WINDOWS]
    + ["pm25_std_24h", "pm25_prev_day_mean", "city_prev_day_mean"]
    + ["doy_sin", "doy_cos", "dow_sin", "dow_cos", "month"]
    + [f"past_{var}" for var in DAILY_AGG_VARS]
    + [f"fc_{var}" for var in DAILY_AGG_VARS]
    + [f"sensor_{sid}" for sid in SENSOR_IDS]
)


# --- hourly prep ---------------------------------------------------------------


def _prepare_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Sort by (sensor_id, time) and add helper columns. Does not mutate `hourly`."""
    prepared = hourly.sort_values(["sensor_id", "time"]).reset_index(drop=True)
    prepared["date"] = prepared["time"].dt.tz_localize(None).dt.normalize()
    prepared["hour"] = prepared["time"].dt.hour
    prepared["_valid"] = prepared["pm25_flag"].isin(VALID_FLAGS) & prepared["pm25"].notna()
    prepared["_valid_pm25"] = prepared["pm25"].where(prepared["_valid"])
    return prepared


def _hourly_wind_components(hourly: pd.DataFrame) -> pd.DataFrame:
    """u/v wind components (meteorological convention: direction = where the wind blows FROM)."""
    speed = hourly["wind_speed_10m"]
    direction_rad = np.deg2rad(hourly["wind_direction_10m"])
    return pd.DataFrame(
        {"wind_u": -speed * np.sin(direction_rad), "wind_v": -speed * np.cos(direction_rad)},
        index=hourly.index,
    )


# --- hour-level PM2.5 lag/rolling features --------------------------------------


def _hourly_lag_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """`pm25_lag_{k}h` for every hour: the valid PM2.5 value k hours earlier for that sensor."""
    grouped = hourly.groupby("sensor_id")["_valid_pm25"]
    return pd.DataFrame({f"pm25_lag_{k}h": grouped.shift(k) for k in LAG_HOURS}, index=hourly.index)


def _hourly_rolling_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window PM2.5 mean/std ending at the hour *before* each row.

    Computed by shifting the valid-PM2.5 series by one hour first, then rolling: at row t the
    window covers [t-window, t-1], i.e. it never includes t itself.
    """
    shifted = hourly.groupby("sensor_id")["_valid_pm25"].shift(1)
    sensor_key = hourly["sensor_id"]
    columns: dict[str, pd.Series] = {}
    for window in ROLLING_WINDOWS:
        min_periods = math.ceil(window * ROLLING_MIN_FRACTION)
        rolled = shifted.groupby(sensor_key).rolling(window, min_periods=min_periods).mean()
        columns[f"pm25_mean_{window}h"] = rolled.reset_index(level=0, drop=True)
    std_min_periods = math.ceil(24 * ROLLING_MIN_FRACTION)
    std_rolled = shifted.groupby(sensor_key).rolling(24, min_periods=std_min_periods).std()
    columns["pm25_std_24h"] = std_rolled.reset_index(level=0, drop=True)
    return pd.DataFrame(columns, index=hourly.index)


def _day_level_lag_rolling(
    hourly: pd.DataFrame, lag_features: pd.DataFrame, rolling_features: pd.DataFrame
) -> pd.DataFrame:
    """Lag/rolling features read off at each day's issue time (the row at D 00:00).

    Defines the day scaffold: a (sensor_id, date) pair exists only if that day's midnight hour
    is present in `hourly` (i.e. the sensor has a defined issue time for it).
    """
    combined = pd.concat([hourly[["sensor_id", "date", "hour"]], lag_features, rolling_features], axis=1)
    midnight = combined[combined["hour"] == 0].drop(columns="hour")
    return midnight.set_index(["sensor_id", "date"])


# --- day-level PM2.5 aggregates (targets, prev-day, city) -----------------------


def _day_pm25_stats(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per (sensor_id, date): mean of valid PM2.5 and count of valid hours."""
    grouped = hourly.groupby(["sensor_id", "date"])
    return pd.DataFrame({"mean": grouped["_valid_pm25"].mean(), "valid_hours": grouped["_valid"].sum()})


def _shift_daily(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Shift a (sensor_id, date)-indexed frame by `periods` calendar days, per sensor.

    Pivots sensors into columns so the shift runs along a single, complete calendar-day axis
    (no gaps), then pivots back — a true calendar-day shift even across any date gaps per sensor.
    """
    wide = frame.unstack(level="sensor_id")
    full_dates = pd.date_range(wide.index.min(), wide.index.max(), freq="D", name="date")
    shifted = wide.reindex(full_dates).shift(periods)
    stacked = shifted.stack(level="sensor_id", future_stack=True)
    stacked = stacked.reorder_levels(["sensor_id", "date"]).sort_index()
    return stacked.reindex(frame.index)


def _targets(day_stats: pd.DataFrame) -> pd.DataFrame:
    """y_pm25 (NaN unless >= 18 valid hours), y_hours, y_band."""
    y_pm25 = day_stats["mean"].where(day_stats["valid_hours"] >= Y_MIN_VALID_HOURS)
    return pd.DataFrame({"y_pm25": y_pm25, "y_hours": day_stats["valid_hours"], "y_band": bands_for(y_pm25)})


def _prev_day_mean(day_stats: pd.DataFrame) -> pd.Series:
    """Mean of D-1 (>= 12 valid hours, else NaN) — also the persistence baseline."""
    shifted = _shift_daily(day_stats, periods=1)
    value = shifted["mean"].where(shifted["valid_hours"] >= PREV_DAY_MIN_VALID_HOURS)
    return value.rename("pm25_prev_day_mean")


def _city_prev_day_mean(prev_day_mean: pd.Series) -> pd.Series:
    """Mean of all sensors' `pm25_prev_day_mean` for the same date."""
    return prev_day_mean.groupby(level="date").transform("mean").rename("city_prev_day_mean")


# --- daily weather/CAMS aggregates (past_*, fc_*) -------------------------------


def _daily_weather_aggregates(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per (sensor_id, date) daily aggregate of each `DAILY_AGG_VARS` variable."""
    wind = _hourly_wind_components(hourly)
    other_vars = [v for v in DAILY_AGG_VARS if v not in ("wind_u", "wind_v")]
    source = hourly[["sensor_id", "date"] + other_vars].copy()
    source["wind_u"] = wind["wind_u"]
    source["wind_v"] = wind["wind_v"]
    grouped = source.groupby(["sensor_id", "date"])
    aggregated = {
        var: (grouped[var].sum(min_count=1) if var in _SUM_VARS else grouped[var].mean())
        for var in DAILY_AGG_VARS
    }
    return pd.DataFrame(aggregated)


# --- calendar and sensor one-hot ------------------------------------------------


def _calendar_features(date_series: pd.Series) -> pd.DataFrame:
    """doy_sin/cos, dow_sin/cos, month for each date (calendar day, PKT)."""
    dt = pd.DatetimeIndex(date_series)
    two_pi = 2 * math.pi
    return pd.DataFrame(
        {
            "doy_sin": np.sin(two_pi * dt.dayofyear / 365.25),
            "doy_cos": np.cos(two_pi * dt.dayofyear / 365.25),
            "dow_sin": np.sin(two_pi * dt.dayofweek / 7),
            "dow_cos": np.cos(two_pi * dt.dayofweek / 7),
            "month": dt.month,
        },
        index=date_series.index,
    )


def _sensor_one_hot(sensor_id_series: pd.Series) -> pd.DataFrame:
    """One-hot sensor columns; all `SENSOR_IDS` exist even if some are absent from the input."""
    return pd.DataFrame(
        {f"sensor_{sid}": (sensor_id_series == sid).astype(int) for sid in SENSOR_IDS},
        index=sensor_id_series.index,
    )


# --- row counting / dropping ----------------------------------------------------


def _counts_by_sensor(mask: pd.Series, sensor_id: pd.Series) -> dict:
    per_sensor = {str(k): int(v) for k, v in mask.groupby(sensor_id).sum().items()}
    return {**per_sensor, "total": int(mask.sum())}


def _drop_incomplete_rows(day_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows with NaN `y_pm25` or `pm25_prev_day_mean`; count candidates/drops/kept.

    A row missing both is counted once, under `dropped_for_target` (checked first).
    """
    sensor_id = day_frame["sensor_id"]
    is_target_na = day_frame["y_pm25"].isna()
    is_prev_day_na = day_frame["pm25_prev_day_mean"].isna()
    dropped_for_target = is_target_na
    dropped_for_prev_day = (~is_target_na) & is_prev_day_na
    kept_mask = ~is_target_na & ~is_prev_day_na

    report = {
        "candidate_rows": _counts_by_sensor(pd.Series(True, index=day_frame.index), sensor_id),
        "dropped_for_target": _counts_by_sensor(dropped_for_target, sensor_id),
        "dropped_for_prev_day": _counts_by_sensor(dropped_for_prev_day, sensor_id),
        "kept": _counts_by_sensor(kept_mask, sensor_id),
    }
    return day_frame[kept_mask], report


# --- entry point -----------------------------------------------------------------


def build_daily_features(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the daily feature table (contract §2) from the hourly table (contract §1).

    Returns `(features, report)`. `report` counts candidate/dropped/kept rows per sensor and
    in total (see `_drop_incomplete_rows`).
    """
    prepared = _prepare_hourly(hourly)

    lag_features = _hourly_lag_features(prepared)
    rolling_features = _hourly_rolling_features(prepared)
    day_lag_rolling = _day_level_lag_rolling(prepared, lag_features, rolling_features)
    scaffold_index = day_lag_rolling.index

    day_stats = _day_pm25_stats(prepared)
    targets = _targets(day_stats)
    prev_day_mean = _prev_day_mean(day_stats)
    city_prev_day_mean = _city_prev_day_mean(prev_day_mean)

    weather_daily = _daily_weather_aggregates(prepared)
    past_weather = _shift_daily(weather_daily, periods=1).add_prefix("past_")
    fc_weather = weather_daily.add_prefix("fc_")

    day_frame = pd.concat(
        [
            targets.reindex(scaffold_index),
            day_lag_rolling,
            prev_day_mean.reindex(scaffold_index),
            city_prev_day_mean.reindex(scaffold_index),
            past_weather.reindex(scaffold_index),
            fc_weather.reindex(scaffold_index),
        ],
        axis=1,
    ).reset_index()

    calendar = _calendar_features(day_frame["date"])
    sensor_onehot = _sensor_one_hot(day_frame["sensor_id"])
    day_frame = pd.concat([day_frame, calendar, sensor_onehot], axis=1)

    kept, report = _drop_incomplete_rows(day_frame)
    return kept[COLUMN_ORDER].reset_index(drop=True), report
