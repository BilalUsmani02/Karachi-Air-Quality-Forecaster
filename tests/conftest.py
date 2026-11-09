"""Shared fixtures for `tests/test_ui_*.py` and the app smoke test.

A tiny, deterministic fake bundle + `Snapshot`, built the same way as
`tests/test_live_snapshot.py` builds its `BUNDLE`/`_features`/`_hourly` — small enough to be
instant, but real enough (real `forecast_frame`, real `Snapshot`) to exercise the `src.ui.*`
code against actual `Snapshot` shapes rather than a hand-rolled stand-in.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.live import snapshot as snap
from src.models.dataset import FEATURE_COLUMNS

FAKE_SENSOR_IDS = ["S1", "S3", "S5"]  # deliberately skips most of SENSORS: exercises
# "no forecast today" for the other 11 registered sensors.
FAKE_PREV_DAY = {"S1": 5.0, "S3": 30.0, "S5": 50.0}  # -> forecasts 6/31/51 -> Good/Moderate/USG


class _IdentityPreprocessor:
    """Stands in for the real `Preprocessor`: already-scaled columns pass through unchanged."""

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=np.float64)


class _PrevDayPlusOneModel:
    """Forecast = yesterday's mean + 1 — keeps expected numbers easy to write down by hand."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = list(FEATURE_COLUMNS).index("pm25_prev_day_mean")
        return X[:, col] + 1.0


FAKE_BUNDLE: dict = {
    "preprocessor": _IdentityPreprocessor(),
    "model": _PrevDayPlusOneModel(),
    "trained_on": ["2025-06-20", "2026-11-08"],
}


def fake_features(day: str, sensor_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(len(sensor_ids), len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS)
    )
    frame["pm25_prev_day_mean"] = [FAKE_PREV_DAY[s] for s in sensor_ids]
    frame.insert(0, "date", pd.Timestamp(day))
    frame.insert(0, "sensor_id", sensor_ids)
    # Deliberately not forecast + 0: lets "actual vs. forecast" (demo mode) tests tell the two
    # apart instead of them always matching by construction.
    frame["y_pm25"] = frame["pm25_prev_day_mean"] + 2.0
    return frame


def fake_hourly(sensor_ids: list[str], start: str, hours: int, value: float = 20.0) -> pd.DataFrame:
    times = pd.date_range(start, periods=hours, freq="h", tz="Asia/Karachi")
    return pd.DataFrame(
        [(s, t, value, value) for s in sensor_ids for t in times],
        columns=["sensor_id", "time", "pm25_raw", "pm25"],
    )


@pytest.fixture
def fake_snapshot() -> snap.Snapshot:
    """A small "demo" snapshot: S1/S3/S5 have forecasts, the other 11 registered sensors don't."""
    features = fake_features("2026-11-08", FAKE_SENSOR_IDS)
    hourly = fake_hourly(FAKE_SENSOR_IDS, "2026-11-05 00:00", 72)
    return snap.Snapshot(
        forecast_date=date(2026, 11, 8),
        fetched_at="2026-11-08T01:00:00+00:00",
        source="demo",
        forecasts=snap.forecast_frame(FAKE_BUNDLE, features, actual=features["y_pm25"]),
        recent=snap.recent_readings(hourly, date(2026, 11, 8)),
        warnings=("Demo mode: showing a past day from the stored data, not a live forecast.",),
        model_trained_on=("2025-06-20", "2026-11-08"),
    )
