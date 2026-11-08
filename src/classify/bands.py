"""EPA 2024 PM2.5 air-quality bands and merge rules (spec §6.4, data contract §3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Band:
    index: int
    name: str
    low: float
    high: float


BANDS: tuple[Band, ...] = (
    Band(0, "Good", 0.0, 9.0),
    Band(1, "Moderate", 9.1, 35.4),
    Band(2, "Unhealthy for Sensitive Groups", 35.5, 55.4),
    Band(3, "Unhealthy", 55.5, 125.4),
    Band(4, "Very Unhealthy", 125.5, 225.4),
    Band(5, "Hazardous", 225.5, math.inf),
)

# Training data has zero Hazardous days (spec §6.4), so it merges into its neighbour by default.
DEFAULT_MERGE: dict[int, int] = {5: 4}

_NAMES = {band.index: band.name for band in BANDS}


def _truncate_1dp(value: float) -> float:
    """EPA truncates (never rounds) to one decimal before banding, e.g. 9.04 -> 9.0.

    Rounds to 6 dp first to absorb float noise from the `* 10` multiply (e.g. 9.05 -> 90.4999...)
    without changing genuine truncation behaviour.
    """
    return math.floor(round(value * 10, 6)) / 10


def pm25_to_band(value: float) -> int:
    """Return the band index (0-5) for one truncated PM2.5 value.

    Negative values raise ValueError: a negative reading is a cleaning error (clean.py already
    turns negatives into NaN), never a value this function should be asked to band.
    """
    if value < 0:
        raise ValueError(f"PM2.5 cannot be negative: {value}")
    truncated = _truncate_1dp(value)
    for band in BANDS:
        if band.low <= truncated <= band.high:
            return band.index
    raise ValueError(f"PM2.5 value out of range: {value}")  # unreachable: BANDS[-1].high is inf


def bands_for(series: pd.Series) -> pd.Series:
    """Vectorised `pm25_to_band`. NaN in, pandas NA out (nullable Int64 dtype)."""
    return series.apply(lambda v: pd.NA if pd.isna(v) else pm25_to_band(v)).astype("Int64")


def merge_bands(series: pd.Series, mapping: dict[int, int] = DEFAULT_MERGE) -> pd.Series:
    """Remap band indices per `mapping` (default: Hazardous -> Very Unhealthy). NA passes through."""
    return series.replace(mapping).astype("Int64")


def band_name(idx: int) -> str:
    """Human-readable name for a band index."""
    return _NAMES[idx]
