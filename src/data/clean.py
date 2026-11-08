"""Phase 2 cleaning pipeline: raw per-sensor CSVs -> the hourly table (contract §1).

Rules are applied per sensor, in contract order, using only past/present data so the same
code works live: negative -> spike -> flatline -> short-gap interpolation. Weather/CAMS get
their own short-gap interpolation (`clean_met`). `build_hourly_table` joins all three sources
for one sensor onto a shared hourly grid; `build_all_hourly` concatenates every sensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import RAW_DIR, TIMEZONE
from src.data.sensors import SENSORS

CAMS_COLUMN_RENAME = {
    "pm2_5": "cams_pm2_5",
    "pm10": "cams_pm10",
    "nitrogen_dioxide": "cams_no2",
    "dust": "cams_dust",
    "aerosol_optical_depth": "cams_aod",
}

BOUNDARY_LAYER_HEIGHT_COLUMN = "boundary_layer_height"
BOUNDARY_LAYER_HEIGHT_GAP_LIMIT = 6
DEFAULT_MET_GAP_LIMIT = 3
PM25_GAP_LIMIT = 3

# --- loaders -----------------------------------------------------------------


def _read_csv_tz(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(TIMEZONE)
    return frame


def load_sensor_raw(path: Path) -> pd.DataFrame:
    """Load one sensor's raw OpenAQ hourly CSV; `time` parsed tz-aware Asia/Karachi."""
    return _read_csv_tz(path)


def load_weather_raw(path: Path) -> pd.DataFrame:
    """Load one sensor's raw Open-Meteo weather CSV; `time` parsed tz-aware Asia/Karachi."""
    return _read_csv_tz(path)


def load_cams_raw(path: Path) -> pd.DataFrame:
    """Load one sensor's raw CAMS air-quality CSV; columns renamed to `cams_*` (contract §1)."""
    return _read_csv_tz(path).rename(columns=CAMS_COLUMN_RENAME)


# --- grid ----------------------------------------------------------------------


def regular_hourly_grid(
    sensor_raw: pd.DataFrame, weather_raw: pd.DataFrame, cams_raw: pd.DataFrame
) -> pd.DatetimeIndex:
    """Hourly index from the earliest weather/CAMS hour to the sensor's last raw reading.

    Contract §1: this keeps the ~month of weather/CAMS data before the sensor's first
    reading, so lag features exist from the first sensor day.
    """
    start = min(weather_raw["time"].min(), cams_raw["time"].min())
    end = sensor_raw["time"].max()
    return pd.date_range(start, end, freq="h")


def _reindex_on_grid(frame: pd.DataFrame, grid: pd.DatetimeIndex, time_col: str = "time") -> pd.DataFrame:
    reindexed = frame.set_index(time_col).reindex(grid)
    reindexed.index.name = time_col
    return reindexed.reset_index()


# --- cleaning rules (contract §1, each a pure function over a Series) ----------


def flag_negative(pm25: pd.Series) -> pd.Series:
    """True where `pm25` is a valid reading below zero (rule 1)."""
    return pm25.notna() & (pm25 < 0)


def flag_spikes(
    pm25: pd.Series,
    *,
    window: int = 24,
    min_valid_hours: int = 6,
    multiplier: float = 5.0,
    abs_threshold: float = 100.0,
    hard_threshold: float = 500.0,
) -> pd.Series:
    """True where `pm25` is a spike (rule 2): > 500, or > `multiplier`x the trailing median of
    the previous `window` hours (needs >= `min_valid_hours` valid hours there, else not
    applied) and > `abs_threshold`.

    The trailing median is shifted by one hour, so it only ever sees strictly past values —
    changing a future value can never change an earlier hour's flag.
    """
    trailing_median = pm25.shift(1).rolling(window=window, min_periods=min_valid_hours).median()
    hard_spike = pm25 > hard_threshold
    relative_spike = (pm25 > multiplier * trailing_median) & (pm25 > abs_threshold)
    return pm25.notna() & (hard_spike | relative_spike)


def flag_flatlines(pm25: pd.Series, *, min_run: int = 24) -> pd.Series:
    """True where `pm25` belongs to a run of >= `min_run` identical consecutive valid values
    (rule 3). NaN never joins a run (NaN != NaN), so gaps break flatline runs.
    """
    run_id = pm25.ne(pm25.shift()).cumsum()
    run_length = pm25.groupby(run_id).transform("size")
    return pm25.notna() & (run_length >= min_run)


def interpolate_short_gaps(series: pd.Series, *, max_gap: int = 3) -> pd.Series:
    """Linearly interpolate interior NaN runs of at most `max_gap` hours (rule 4).

    A run is filled only when bounded by valid values on both sides; leading/trailing NaN
    runs (no bound on one side) are left as NaN. Returns a new Series; does not mutate input.
    """
    values = series.to_numpy(dtype=float, copy=True)
    is_na = pd.isna(values)
    n = len(values)
    i = 0
    while i < n:
        if not is_na[i]:
            i += 1
            continue
        j = i
        while j < n and is_na[j]:
            j += 1
        gap_len = j - i
        has_left = i > 0
        has_right = j < n
        if gap_len <= max_gap and has_left and has_right:
            left_val = values[i - 1]
            right_val = values[j]
            step = (right_val - left_val) / (gap_len + 1)
            for offset in range(gap_len):
                values[i + offset] = left_val + step * (offset + 1)
        i = j
    return pd.Series(values, index=series.index, name=series.name)


# --- per-sensor / per-table cleaning --------------------------------------------


def clean_sensor_hourly(sensor_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the §1 rules, in contract order, to one sensor's raw PM2.5 series.

    Reindexes to an hourly grid spanning the sensor's own first..last raw reading (internal
    gaps become NaN rows), then applies negative -> spike -> flatline -> interpolation.
    `pm25_flag` records why a cell is what it is; `interpolated` wins over the earlier
    rejection reason when a short gap is filled, since that is the final disposition of the
    hour. Returns columns `time`, `pm25_raw`, `pm25`, `pm25_flag`; does not mutate the input.
    """
    times = sensor_raw["time"]
    grid = pd.date_range(times.min(), times.max(), freq="h")
    raw = sensor_raw.set_index("time")["pm25"].reindex(grid).rename("pm25_raw")

    flag = pd.Series("ok", index=grid, name="pm25_flag", dtype=object)
    flag[raw.isna()] = "missing"

    working = raw.rename("pm25").copy()

    negative_mask = flag_negative(working)
    working = working.mask(negative_mask)
    flag[negative_mask] = "negative"

    spike_mask = flag_spikes(working)
    working = working.mask(spike_mask)
    flag[spike_mask] = "spike"

    flatline_mask = flag_flatlines(working)
    working = working.mask(flatline_mask)
    flag[flatline_mask] = "flatline"

    filled = interpolate_short_gaps(working, max_gap=PM25_GAP_LIMIT)
    interpolated_mask = working.isna() & filled.notna()
    flag[interpolated_mask] = "interpolated"

    return pd.DataFrame(
        {
            "time": grid,
            "pm25_raw": raw.to_numpy(),
            "pm25": filled.to_numpy(),
            "pm25_flag": flag.to_numpy(),
        }
    )


def clean_met(
    frame: pd.DataFrame,
    *,
    time_col: str = "time",
    blh_col: str = BOUNDARY_LAYER_HEIGHT_COLUMN,
    blh_limit: int = BOUNDARY_LAYER_HEIGHT_GAP_LIMIT,
    default_limit: int = DEFAULT_MET_GAP_LIMIT,
) -> pd.DataFrame:
    """Fill short gaps in weather/CAMS columns (rule 7): `boundary_layer_height` up to
    `blh_limit` hours, every other column up to `default_limit` hours. Returns a new frame.
    """
    result = frame.copy()
    for column in result.columns:
        if column == time_col:
            continue
        limit = blh_limit if column == blh_col else default_limit
        result[column] = interpolate_short_gaps(result[column], max_gap=limit)
    return result


# --- table assembly --------------------------------------------------------------


def build_hourly_table(sensor_id: str, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Load, clean and join one sensor's three raw files into the contract §1 hourly table.

    Returns (table, report) where `report` counts rows and `pm25_flag` values.
    """
    sensor_raw = load_sensor_raw(raw_dir / "sensors" / f"{sensor_id}.csv")
    weather_raw = load_weather_raw(raw_dir / "weather" / f"{sensor_id}.csv")
    cams_raw = load_cams_raw(raw_dir / "air_quality" / f"{sensor_id}.csv")

    grid = regular_hourly_grid(sensor_raw, weather_raw, cams_raw)

    pm25_table = _reindex_on_grid(clean_sensor_hourly(sensor_raw), grid)
    pm25_table["pm25_flag"] = pm25_table["pm25_flag"].fillna("missing")

    weather = clean_met(_reindex_on_grid(weather_raw, grid))
    cams = clean_met(_reindex_on_grid(cams_raw, grid))

    table = pm25_table.merge(weather, on="time", how="left").merge(cams, on="time", how="left")
    table.insert(0, "sensor_id", sensor_id)

    report = {
        "sensor_id": sensor_id,
        "rows": len(table),
        "flag_counts": table["pm25_flag"].value_counts().to_dict(),
    }
    return table, report


def build_all_hourly(
    sensor_ids: Sequence[str] | None = None, raw_dir: Path = RAW_DIR
) -> tuple[pd.DataFrame, dict]:
    """Build and concatenate the hourly table for every sensor (default: the full registry).

    Returns (table, reports) where `reports` maps sensor_id -> its `build_hourly_table` report.
    """
    ids = list(sensor_ids) if sensor_ids is not None else [sensor.id for sensor in SENSORS]
    tables = []
    reports = {}
    for sensor_id in ids:
        table, report = build_hourly_table(sensor_id, raw_dir=raw_dir)
        tables.append(table)
        reports[sensor_id] = report
    combined = pd.concat(tables, ignore_index=True)
    return combined, reports
