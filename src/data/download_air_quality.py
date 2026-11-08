"""CLI: download D3 modelled pollution (Open-Meteo CAMS air-quality API) per sensor.

    uv run python -m src.data.download_air_quality --start 2022-08-05 --end 2026-11-01
    uv run python -m src.data.download_air_quality --sensors S1 S2 --force
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from src.config import RAW_DIR
from src.data.open_meteo import fetch_range, save_csv, utc_now_iso, write_manifest
from src.data.sensors import SENSORS, Sensor, sensor_by_id

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOURLY_VARS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "dust",
    "aerosol_optical_depth",
]
LICENCE = "CC BY 4.0, Open-Meteo / Copernicus CAMS"

# Checked 8 Nov 2026 by probing the live API over Karachi (24.9338, 67.1135): CAMS global
# is null before 2022-08-04 (and only from 05:00 PKT that day), so 2022-08-05 is the first
# full day of data — not the 2022-08-01 first guessed in the spec.
DEFAULT_START = date(2022, 8, 5)

OUT_DIR = RAW_DIR / "air_quality"


def default_end() -> date:
    return date.today() - timedelta(days=1)


def download_one(sensor: Sensor, start: date, end: date, *, force: bool, out_dir: Path = OUT_DIR) -> None:
    out_path = out_dir / f"{sensor.id}.csv"
    if out_path.exists() and not force:
        print(f"skip {sensor.id}: {out_path} already exists (use --force to overwrite)")
        return

    result = fetch_range(URL, sensor, start, end, HOURLY_VARS)
    save_csv(result.frame, out_path)

    manifest_entry = {
        sensor.id: {
            "endpoint": URL,
            "variables": HOURLY_VARS,
            "requested_lat": sensor.lat,
            "requested_lon": sensor.lon,
            "grid_lat": result.grid_latitude,
            "grid_lon": result.grid_longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "row_count": len(result.frame),
            "downloaded_at_utc": utc_now_iso(),
            "licence": LICENCE,
        }
    }
    write_manifest(out_dir / "manifest.json", manifest_entry)
    print(f"wrote {out_path} ({len(result.frame)} rows)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download D3 modelled air-quality data per sensor.")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--sensors", nargs="+", default=None, help="Sensor IDs, e.g. S1 S2")
    parser.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    end = args.end or default_end()
    sensors = [sensor_by_id(sensor_id) for sensor_id in args.sensors] if args.sensors else SENSORS
    for sensor in sensors:
        download_one(sensor, args.start, end, force=args.force)


if __name__ == "__main__":
    main()
