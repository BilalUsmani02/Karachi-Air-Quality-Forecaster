"""CLI: download D2 historical weather (Open-Meteo archive API) per sensor.

    uv run python -m src.data.download_weather --start 2022-08-01 --end 2026-11-01
    uv run python -m src.data.download_weather --sensors S1 S2 --force
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from src.config import RAW_DIR
from src.data.open_meteo import fetch_range, save_csv, utc_now_iso, write_manifest
from src.data.sensors import SENSORS, Sensor, sensor_by_id

URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
]
EXTRA_PARAMS = {"wind_speed_unit": "ms"}
LICENCE = "CC BY 4.0, Open-Meteo"

# Matches the spec §5.1 default; the archive API itself has decades of history.
DEFAULT_START = date(2022, 8, 1)

OUT_DIR = RAW_DIR / "weather"


def default_end() -> date:
    return date.today() - timedelta(days=1)


def download_one(sensor: Sensor, start: date, end: date, *, force: bool, out_dir: Path = OUT_DIR) -> None:
    out_path = out_dir / f"{sensor.id}.csv"
    if out_path.exists() and not force:
        print(f"skip {sensor.id}: {out_path} already exists (use --force to overwrite)")
        return

    result = fetch_range(URL, sensor, start, end, HOURLY_VARS, extra_params=EXTRA_PARAMS)
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
    parser = argparse.ArgumentParser(description="Download D2 weather data per sensor.")
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
