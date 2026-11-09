"""The snapshot: everything the app shows, in one object that can be cached to disk (Phase 7).

A `Snapshot` is one forecast run for all sensors: the forecast for day D at each sensor, the raw
feature row behind it (for SHAP), and the last 72 hours of readings (for the chart). It comes
from one of three places, recorded in `source`:

    "live"   built just now from live APIs (`src/live/pipeline.py`)
    "cache"  the last live snapshot, reloaded from disk because a live fetch failed or was skipped
    "demo"   built offline from the processed historical tables (`demo_snapshot`)

The app only ever talks to this module's types, so it works the same whichever source it got.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, ROOT, SEED
from src.data.sensors import SENSORS, sensor_by_id
from src.models.dataset import FEATURE_COLUMNS

CACHE_DIR = ROOT / "data" / "cache"
BUNDLE_PATH = MODELS_DIR / "final_ensemble.pkl"
RECENT_HOURS = 72
BACKGROUND_SIZE = 100
SOURCES = ("live", "cache", "demo")

# Columns of `Snapshot.forecasts`, before the raw FEATURE_COLUMNS that follow them.
META_COLUMNS = (
    "sensor_id",
    "site",
    "lat",
    "lon",
    "forecast_pm25",  # the ensemble's forecast of D's 24-h mean PM2.5 (µg/m³)
    "prev_day_pm25",  # D-1's mean PM2.5 — "today" for the trend rules; NaN if too few hours
    "actual_pm25",  # D's observed mean — known only for demo snapshots, NaN for live ones
)
RECENT_COLUMNS = ("sensor_id", "time", "pm25")


@dataclass(frozen=True)
class Snapshot:
    forecast_date: date  # day D being forecast (calendar day, PKT); issued at D 00:00 PKT
    fetched_at: str  # ISO-8601 UTC time the inputs were fetched (or the demo was built)
    source: str  # one of SOURCES
    forecasts: pd.DataFrame  # one row per sensor: META_COLUMNS + FEATURE_COLUMNS (raw, unscaled)
    recent: pd.DataFrame  # RECENT_COLUMNS: hourly PM2.5 for the RECENT_HOURS before D 00:00
    warnings: tuple[str, ...] = ()
    model_trained_on: tuple[str, str] = ("", "")

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        missing = [c for c in (*META_COLUMNS, *FEATURE_COLUMNS) if c not in self.forecasts.columns]
        if missing:
            raise ValueError(f"forecasts is missing columns: {missing}")
        missing = [c for c in RECENT_COLUMNS if c not in self.recent.columns]
        if missing:
            raise ValueError(f"recent is missing columns: {missing}")

    def with_source(self, source: str, warning: str | None = None) -> "Snapshot":
        """A copy relabelled as `source`, optionally with one more warning."""
        warnings = self.warnings + ((warning,) if warning else ())
        return Snapshot(
            self.forecast_date, self.fetched_at, source, self.forecasts, self.recent, warnings, self.model_trained_on
        )


def load_bundle(path: Path = BUNDLE_PATH) -> dict[str, Any]:
    """The production bundle `{"preprocessor", "model", "trained_on"}` from `final_test --save-production`."""
    return pickle.loads(Path(path).read_bytes())


def predict(bundle: dict[str, Any], raw_X: pd.DataFrame) -> np.ndarray:
    """Forecast PM2.5 for raw (unscaled) FEATURE_COLUMNS rows."""
    X = bundle["preprocessor"].transform(raw_X[list(FEATURE_COLUMNS)])
    return np.asarray(bundle["model"].predict(X), dtype=np.float64)


def forecast_frame(
    bundle: dict[str, Any], features: pd.DataFrame, actual: pd.Series | None = None
) -> pd.DataFrame:
    """Build `Snapshot.forecasts` from one raw feature row per sensor (`sensor_id` + FEATURE_COLUMNS)."""
    sites = [sensor_by_id(s) for s in features["sensor_id"]]
    meta = pd.DataFrame(
        {
            "sensor_id": features["sensor_id"].to_numpy(),
            "site": [s.site for s in sites],
            "lat": [s.lat for s in sites],
            "lon": [s.lon for s in sites],
            "forecast_pm25": predict(bundle, features),
            "prev_day_pm25": features["pm25_prev_day_mean"].to_numpy(dtype=np.float64),
            "actual_pm25": np.nan if actual is None else actual.to_numpy(dtype=np.float64),
        }
    )
    order = {s.id: i for i, s in enumerate(SENSORS)}
    frame = pd.concat([meta, features[list(FEATURE_COLUMNS)].reset_index(drop=True)], axis=1)
    return frame.sort_values("sensor_id", key=lambda s: s.map(order)).reset_index(drop=True)


def save_snapshot(snapshot: Snapshot, cache_dir: Path = CACHE_DIR) -> Path:
    """Write the snapshot to `cache_dir` (forecasts.parquet, recent.parquet, meta.json)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot.forecasts.to_parquet(cache_dir / "forecasts.parquet", index=False)
    snapshot.recent.to_parquet(cache_dir / "recent.parquet", index=False)
    meta = {
        "forecast_date": snapshot.forecast_date.isoformat(),
        "fetched_at": snapshot.fetched_at,
        "source": snapshot.source,
        "warnings": list(snapshot.warnings),
        "model_trained_on": list(snapshot.model_trained_on),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cache_dir


def load_snapshot(cache_dir: Path = CACHE_DIR) -> Snapshot | None:
    """The cached snapshot, relabelled `source="cache"` if it was live; None if there is none."""
    cache_dir = Path(cache_dir)
    try:
        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        forecasts = pd.read_parquet(cache_dir / "forecasts.parquet")
        recent = pd.read_parquet(cache_dir / "recent.parquet")
    except (FileNotFoundError, ValueError):
        return None
    source = "cache" if meta["source"] == "live" else meta["source"]
    return Snapshot(
        forecast_date=date.fromisoformat(meta["forecast_date"]),
        fetched_at=meta["fetched_at"],
        source=source,
        forecasts=forecasts,
        recent=recent,
        warnings=tuple(meta.get("warnings", ())),
        model_trained_on=tuple(meta.get("model_trained_on", ("", ""))),
    )


def recent_readings(hourly: pd.DataFrame, forecast_date: date, hours: int = RECENT_HOURS) -> pd.DataFrame:
    """Hourly PM2.5 (cleaned) for the `hours` before `forecast_date` 00:00 PKT, all sensors."""
    issue = pd.Timestamp(forecast_date).tz_localize(hourly["time"].dt.tz)
    window = hourly[(hourly["time"] >= issue - pd.Timedelta(hours=hours)) & (hourly["time"] < issue)]
    return window[list(RECENT_COLUMNS)].reset_index(drop=True)


def demo_snapshot(
    bundle: dict[str, Any],
    processed_dir: Path = PROCESSED_DIR,
    forecast_date: date | None = None,
) -> Snapshot:
    """An offline snapshot from the processed tables: the latest labelled day (or `forecast_date`).

    The actual PM2.5 for that day is included, so the demo can show forecast vs. what happened.
    """
    features = pd.read_parquet(Path(processed_dir) / "features.parquet")
    hourly = pd.read_parquet(Path(processed_dir) / "hourly_clean.parquet")
    day = pd.Timestamp(forecast_date) if forecast_date else features["date"].max()
    rows = features[features["date"] == day].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no feature rows for {day.date()}")
    trained_on = tuple(bundle.get("trained_on", ("", "")))
    return Snapshot(
        forecast_date=day.date(),
        fetched_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        source="demo",
        forecasts=forecast_frame(bundle, rows, actual=rows["y_pm25"]),
        recent=recent_readings(hourly, day.date()),
        warnings=("Demo mode: showing a past day from the stored data, not a live forecast.",),
        model_trained_on=trained_on,
    )


def load_background(processed_dir: Path = PROCESSED_DIR, n: int = BACKGROUND_SIZE, seed: int = SEED) -> pd.DataFrame:
    """`n` raw FEATURE_COLUMNS rows from the training table, for the SHAP explainer's background."""
    features = pd.read_parquet(Path(processed_dir) / "features.parquet")
    return features[list(FEATURE_COLUMNS)].sample(n=min(n, len(features)), random_state=seed).reset_index(drop=True)
