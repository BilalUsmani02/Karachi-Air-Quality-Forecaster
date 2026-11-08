# Phase 2 data contract

How the Phase 2 modules hand data to each other. Three modules are built in parallel against
this contract, then wired together by `src/data/build_features.py`.

```
raw CSVs ──► clean.py ──► hourly table ──► features.py ──► daily feature table ──► split.py
                                               │
                                     src/classify/bands.py
```

All times are tz-aware `Asia/Karachi` (PKT, UTC+5, no DST). Sensor IDs are `"S1"`…`"S14"`
(`src/data/sensors.py`). Raw file formats: see the headers of `data/raw/sensors/S*.csv`,
`data/raw/weather/S*.csv`, `data/raw/air_quality/S*.csv` (also described in each downloader).

---

## 1. Hourly table — produced by `src/data/clean.py`

One row per **(sensor_id, hour)** on a **regular hourly grid** from that sensor's first to last
raw reading (missing hours are present as rows with NaN). `time` is the start of the hour.

| Column | Type | Meaning |
|---|---|---|
| `sensor_id` | str | `"S1"`… |
| `time` | datetime64[ns, Asia/Karachi] | hour start |
| `pm25_raw` | float | raw OpenAQ hourly value (NaN if no reading) |
| `pm25` | float | cleaned value (NaN if missing or rejected) |
| `pm25_flag` | str | one of `ok`, `missing`, `negative`, `spike`, `flatline`, `interpolated` |
| weather columns | float | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, `precipitation`, `wind_speed_10m` (m/s), `wind_direction_10m` (deg), `boundary_layer_height` (m) — from that sensor's weather file |
| CAMS columns | float | `cams_pm2_5`, `cams_pm10`, `cams_no2`, `cams_dust`, `cams_aod` — from that sensor's air-quality file (renamed) |

Weather/CAMS are joined on `time` for the same sensor. Their grid also includes the ~1 month
**before** the sensor's first reading (the raw weather/CAMS start 2025-05-01) — keep those
hours so 7-day lag features exist from the first sensor day; for those hours `pm25_raw`/`pm25`
are NaN and `pm25_flag` = `missing`.

### Cleaning rules (decided from the Phase 1 exploration)
Applied in this order, to each sensor separately, **using only past and present data** (the
same rules must work live, where the future is unknown):
1. `negative`: value < 0 → NaN.
2. `spike`: value > 500 µg/m³, **or** value > 5 × the trailing median of the previous 24 hours
   (needs ≥ 6 valid hours in that window, else rule not applied) **and** value > 100 µg/m³
   (so low-level noise like 3 → 16 isn't flagged) → NaN.
3. `flatline`: the same value repeated for ≥ 24 consecutive hours → all hours of that run NaN.
4. `interpolated`: gaps of **≤ 3 consecutive hours** between two valid values are filled by
   linear interpolation. Longer gaps stay NaN (`missing`).
5. Zeros are kept as real values (`ok`).
6. No humidity correction (exploration found no humidity inflation — see
   `docs/report/data_exploration.md`). Keep the code path out.
7. Weather/CAMS: `boundary_layer_height` is only populated every few hours → linear
   interpolation with a limit of 6 hours; other weather/CAMS columns interpolate gaps ≤ 3 h.

`clean.py` also returns/logs a per-sensor **cleaning report**: counts of each flag.

---

## 2. Daily feature table — produced by `src/data/features.py`

One row per **(sensor_id, date D)**. The forecast is **issued at 00:00 PKT on day D** and
predicts the **mean PM2.5 over D 00:00–23:59** ("the next 24 hours").

**Leakage rule:** every `past_*`/lag/rolling feature for day D uses only hourly rows with
`time < D 00:00`. Only `fc_*` columns and the targets may use hours inside day D.

| Column | Meaning |
|---|---|
| `sensor_id`, `date` | keys (`date` = D as a `datetime.date` or midnight-normalised timestamp) |
| `y_pm25` | target: mean `pm25` over D's 24 hours; **NaN unless ≥ 18 valid hours** (`ok` or `interpolated`) |
| `y_hours` | number of valid hours behind `y_pm25` |
| `y_band` | int band index of `y_pm25` (§3), NaN if `y_pm25` is NaN |
| `pm25_lag_{1,2,3,6,12,24}h` | cleaned PM2.5 at 1…24 h before the issue time (lag 1 h = hour starting 23:00 on D−1) |
| `pm25_mean_{24,72,168}h`, `pm25_std_24h` | trailing window stats over hours before D (need ≥ 50% valid hours, else NaN) |
| `pm25_prev_day_mean` | mean of D−1 (≥ 12 valid hours, else NaN) — also the **persistence baseline** |
| `city_prev_day_mean` | mean of all sensors' `pm25_prev_day_mean` for D (helps when a sensor's own history is patchy) |
| `doy_sin`, `doy_cos`, `dow_sin`, `dow_cos`, `month` | calendar for D |
| `past_<var>` | D−1 daily aggregate of each weather/CAMS variable (mean; `precipitation` = sum; wind as mean `u`/`v` components, plus mean speed) |
| `fc_<var>` | same aggregates over day D — stands in for the **weather/CAMS forecast** that is available at issue time (training uses ERA5/CAMS actuals: "perfect-prognosis"; the app will use Open-Meteo forecasts) |
| `sensor_<ID>` | one-hot sensor columns (0/1) |

Rows are kept even if some features are NaN; a row is **dropped (and counted)** if
`y_pm25` is NaN **or** `pm25_prev_day_mean` is NaN. Imputation of other NaNs happens later,
fitted on the training split only (Phase 3).

---

## 3. Bands — `src/classify/bands.py`

EPA 2024 PM2.5 breakpoints on the 24-h mean (spec §6.4). Band indices:

| idx | name | PM2.5 (µg/m³) |
|---|---|---|
| 0 | Good | 0.0 – 9.0 |
| 1 | Moderate | 9.1 – 35.4 |
| 2 | Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| 3 | Unhealthy | 55.5 – 125.4 |
| 4 | Very Unhealthy | 125.5 – 225.4 |
| 5 | Hazardous | ≥ 225.5 |

EPA truncates concentrations to one decimal before looking them up, so 9.04 → 9.0 → Good and
35.45 → 35.4 → Moderate. Provide a `merge` mapping (default: Hazardous → Very Unhealthy,
because the training data has zero Hazardous days) applied as a separate step.

---

## 4. Splits — `src/data/split.py`

Works on any frame with `date` and `y_band` columns; never shuffles.
- **Holdout split:** train / validation / test by date cut-offs.
- **Rolling-origin folds:** expanding training window, each fold tests the next block of
  days, with a configurable gap (embargo) between train end and test start.
- A summary per split/fold: date range, row count, band counts.
