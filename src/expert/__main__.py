"""CLI demo for the expert-system layer.

    uv run python -m src.expert --pm25 33 --today 25 --sensitive --outdoor
"""

from __future__ import annotations

import argparse
import sys

from src.expert import explain, recommend


def main() -> None:
    parser = argparse.ArgumentParser(description="Karachi Air-Quality Forecaster expert system (§7)")
    parser.add_argument("--pm25", required=True, type=float, help="tomorrow's forecast PM2.5 (ug/m3)")
    parser.add_argument("--today", default=None, type=float, help="today's PM2.5 (ug/m3), for the trend rules")
    parser.add_argument("--sensitive", action="store_true", help="person is in a sensitive group")
    parser.add_argument("--outdoor", action="store_true", help="person has outdoor plans")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which garbles µ, ³ and dashes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rec = recommend(
        forecast_pm25=args.pm25,
        today_pm25=args.today,
        sensitive=args.sensitive,
        outdoor_activity=args.outdoor,
    )

    print(f"Band: {rec.band_name} ({rec.color})")
    print(f"Advice: {rec.advice}")
    if rec.notes:
        print("Notes:")
        for note in rec.notes:
            print(f"  - {note}")
    print(f"Disclaimer: {rec.disclaimer}")
    print()
    print(explain(rec))


if __name__ == "__main__":
    main()
