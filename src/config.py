"""Project-wide paths and constants shared by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Pakistan Standard Time (UTC+5, no daylight saving). All data is aligned to this.
TIMEZONE = "Asia/Karachi"

# One seed for every random process, so results are reproducible.
SEED = 42
