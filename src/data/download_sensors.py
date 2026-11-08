"""CLI: download D1 PM2.5 sensor data (OpenAQ) and D5 US Consulate reference monitor.

    uv run python -m src.data.download_sensors
    uv run python -m src.data.download_sensors --start 2025-06-01 --end 2026-11-01
    uv run python -m src.data.download_sensors --sensors S1 S2 --force
    uv run python -m src.data.download_sensors --consulate
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path
from typing import Callable

import requests

from src.config import RAW_DIR
from src.data.open_meteo import save_csv, utc_now_iso, write_manifest
from src.data.openaq import BASE_URL, fetch_sensor_dataframe, get_api_key
from src.data.sensors import SENSORS, US_CONSULATE, Sensor, sensor_by_id

LICENCE = "CC BY 4.0, OpenAQ"
SENSOR_PROVIDER = "original provider: AirGradient / The Urban Unit"
CONSULATE_PROVIDER = "original provider: US Department of State via AirNow"

DEFAULT_START = date(2025, 6, 1)
DEFAULT_CONSULATE_START = date(2019, 1, 1)
DEFAULT_CONSULATE_END = date(2025, 4, 1)

OUT_DIR = RAW_DIR / "sensors"
CONSULATE_OUT_DIR = RAW_DIR / "us_consulate"


def default_end() -> date:
    return date.today()


def _download(
    sensor: Sensor,
    start: date,
    end: date,
    *,
    out_path: Path,
    manifest_path: Path,
    force: bool,
    provider: str,
    api_key: str,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if out_path.exists() and not force:
        print(f"skip {sensor.id}: {out_path} already exists (use --force to overwrite)")
        return

    frame = fetch_sensor_dataframe(sensor.openaq_sensor_id, start, end, api_key, session=session, sleep=sleep)
    save_csv(frame, out_path)

    manifest_entry = {
        sensor.id: {
            "endpoint": f"{BASE_URL}/sensors/{sensor.openaq_sensor_id}/hours",
            "openaq_location_id": sensor.openaq_location_id,
            "openaq_sensor_id": sensor.openaq_sensor_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "row_count": len(frame),
            "downloaded_at_utc": utc_now_iso(),
            "licence": LICENCE,
            "provider": provider,
        }
    }
    write_manifest(manifest_path, manifest_entry)
    print(f"wrote {out_path} ({len(frame)} rows)")


def download_sensor(
    sensor: Sensor,
    start: date,
    end: date,
    *,
    force: bool,
    api_key: str,
    out_dir: Path = OUT_DIR,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    out_path = out_dir / f"{sensor.id}.csv"
    _download(
        sensor,
        start,
        end,
        out_path=out_path,
        manifest_path=out_dir / "manifest.json",
        force=force,
        provider=SENSOR_PROVIDER,
        api_key=api_key,
        session=session,
        sleep=sleep,
    )


def download_consulate(
    start: date,
    end: date,
    *,
    force: bool,
    api_key: str,
    out_dir: Path = CONSULATE_OUT_DIR,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    out_path = out_dir / f"{US_CONSULATE.openaq_location_id}.csv"
    _download(
        US_CONSULATE,
        start,
        end,
        out_path=out_path,
        manifest_path=out_dir / "manifest.json",
        force=force,
        provider=CONSULATE_PROVIDER,
        api_key=api_key,
        session=session,
        sleep=sleep,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download D1 sensor data (and optional D5) from OpenAQ.")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--sensors", nargs="+", default=None, help="Sensor IDs, e.g. S1 S2")
    parser.add_argument("--consulate", action="store_true", help="download D5 US Consulate instead of D1")
    parser.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_key = get_api_key()

    if args.consulate:
        start = args.start or DEFAULT_CONSULATE_START
        end = args.end or DEFAULT_CONSULATE_END
        download_consulate(start, end, force=args.force, api_key=api_key)
        return

    start = args.start or DEFAULT_START
    end = args.end or default_end()
    sensors = [sensor_by_id(sensor_id) for sensor_id in args.sensors] if args.sensors else SENSORS
    for sensor in sensors:
        download_sensor(sensor, start, end, force=args.force, api_key=api_key)


if __name__ == "__main__":
    main()
