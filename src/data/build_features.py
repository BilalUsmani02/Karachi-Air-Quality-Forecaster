"""CLI: run the Phase 2 pipeline — raw CSVs → cleaned hourly table → daily feature table.

    uv run python -m src.data.build_features
    uv run python -m src.data.build_features --sensors S1 S2

Writes to `data/processed/`:
    hourly_clean.parquet   contract §1 hourly table (all sensors)
    features.parquet       contract §2 daily feature table
    build_report.json      cleaning flag counts and feature-row drop counts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.data.clean import build_all_hourly
from src.data.features import build_daily_features
from src.data.open_meteo import utc_now_iso


def _jsonable(value):
    """Turn numpy/pandas scalars inside nested report dicts into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def build(
    sensor_ids: list[str] | None = None,
    *,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> dict:
    """Run cleaning and feature building, write outputs, and return the build report."""
    hourly, cleaning_report = build_all_hourly(sensor_ids, raw_dir=raw_dir)
    features, feature_report = build_daily_features(hourly)

    out_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out_dir / "hourly_clean.parquet", index=False)
    features.to_parquet(out_dir / "features.parquet", index=False)

    report = _jsonable(
        {
            "built_at_utc": utc_now_iso(),
            "sensors": sorted(hourly["sensor_id"].unique()),
            "hourly_rows": len(hourly),
            "feature_rows": len(features),
            "feature_columns": len(features.columns),
            "date_range": [
                pd.Timestamp(features["date"].min()).date().isoformat(),
                pd.Timestamp(features["date"].max()).date().isoformat(),
            ],
            "cleaning": cleaning_report,
            "features": feature_report,
        }
    )
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned hourly and daily feature tables.")
    parser.add_argument("--sensors", nargs="+", help="sensor IDs (default: all)")
    args = parser.parse_args(argv)

    report = build(args.sensors)
    kept = report["features"]["kept"]["total"]
    candidates = report["features"]["candidate_rows"]["total"]
    print(f"hourly rows: {report['hourly_rows']:,}")
    print(f"daily feature rows: {kept:,} kept of {candidates:,} candidates")
    print(f"dates: {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"wrote {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
