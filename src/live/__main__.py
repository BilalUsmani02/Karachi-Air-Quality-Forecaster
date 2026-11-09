"""CLI: print today's forecast table.

    uv run python -m src.live            # the saved forecast (or the demo if there is none)
    uv run python -m src.live --refresh  # fetch live data first, falling back if that fails
"""

from __future__ import annotations

import argparse
import sys

from src.classify.bands import band_name, bands_for
from src.live.pipeline import get_snapshot
from src.live.snapshot import Snapshot


def format_snapshot(snapshot: Snapshot) -> str:
    """A short plain-text report: source, a sensor/forecast/band table, then the warnings."""
    lines = [
        f"Forecast for {snapshot.forecast_date} (24-h mean PM2.5) - source: {snapshot.source}, "
        f"fetched {snapshot.fetched_at}",
        "",
        f"{'Sensor':<40} {'Forecast':>9}  Band",
    ]
    forecasts = snapshot.forecasts
    bands = bands_for(forecasts["forecast_pm25"])
    for (_, row), band in zip(forecasts.iterrows(), bands):
        label = f"{row['sensor_id']} {row['site']}"
        lines.append(f"{label:<40} {row['forecast_pm25']:>9.1f}  {band_name(int(band))}")
    if snapshot.warnings:
        lines += ["", "Warnings:", *(f"- {w}" for w in snapshot.warnings)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the PM2.5 forecast for today.")
    parser.add_argument("--refresh", action="store_true", help="fetch live data before printing")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_snapshot(get_snapshot(refresh=args.refresh)))


if __name__ == "__main__":
    main()
