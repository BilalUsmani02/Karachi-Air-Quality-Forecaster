# Karachi Air-Quality Forecaster

Forecasts the 24-hour average fine-dust level (PM2.5) for the day ahead, issued at 00:00 PKT,
at 14 real sensors in Karachi (about 9 report at any given time), sorts it into an air-quality
band (Good → Hazardous), and gives personalised, explained health advice.

- **Core model:** a neural network written from scratch in NumPy, checked against PyTorch and
  compared with scikit-learn models.
- **Expert system:** if-then rules turn the band and the user's profile into advice, showing
  which rules fired.
- **Explanations:** SHAP shows which factors pushed each forecast up or down.
- **App:** Streamlit — pick a location, see the forecast, band, advice and reasons.

Shared project for **PAI (CT-264)**, **AIES (CT-361) CCP** and **AIES Lab (CT-361) CCA**.

- What and why: [`PROJECT_SPEC.md`](PROJECT_SPEC.md)
- Plan and progress: [`ROADMAP.md`](ROADMAP.md)

> Advice from this tool is informational only — not official or medical guidance.

---

## Setup

Requires **Python 3.11 or 3.12**.

### Option A — uv (recommended)

```bash
# install uv once: https://docs.astral.sh/uv/getting-started/installation/
uv sync            # creates .venv with Python 3.12 and installs everything from uv.lock
```

Run any command inside the environment with `uv run <command>`.

### Option B — pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
uv run pytest                      # tests + coverage report
uv run streamlit run app/main.py   # web app (from Phase 7)
```

### Run the app (Phase 7)

```bash
uv run streamlit run app/main.py
```

That one command is enough — it works **offline**: on first load it shows a cached forecast
if one exists, or falls back to a demo day from the stored historical data, no internet
connection or API keys needed. Inside the app, use the **🔄 Refresh live data** button on the
Forecast page to try fetching the latest readings and weather; if that fails, it falls back to the cache, then the demo, and says so with a
warning banner. A live refresh takes about 2–3 minutes (OpenAQ limits the request rate) and
needs `OPENAQ_API_KEY` in `.env`; `uv run python -m src.live --refresh` does the same from the
terminal. If `models/final_ensemble.pkl` is missing, the app shows the command to build
it (`uv run python -m src.evaluate.final_test --save-production`) instead of crashing.

### Download the data (Phase 1)

Sensor data needs a free OpenAQ API key: register at https://explore.openaq.org, then put
`OPENAQ_API_KEY=<your key>` in a `.env` file in the project root (gitignored).

```bash
uv run python -m src.data.download_sensors                 # D1: 14 Urban Unit sensors (OpenAQ)
uv run python -m src.data.download_sensors --consulate     # D5: US Consulate reference monitor
uv run python -m src.data.download_weather --start 2025-05-01       # D2: ERA5 weather
uv run python -m src.data.download_air_quality --start 2025-05-01   # D3: CAMS modelled pollution
```

Existing files are skipped; add `--force` to re-download. Each folder gets a `manifest.json`.

### Build the feature table (Phase 2)

```bash
uv run python -m src.data.build_features   # raw → data/processed/{hourly_clean,features}.parquet
```

`data/processed/build_report.json` records how many hours were cleaned and how many daily rows
were kept or dropped. Table formats and cleaning rules: `docs/design/phase2_data_contract.md`.

## Reproduce everything

Every command below, in order, from a bare clone to a running app. Pipeline diagram and a
one-page walkthrough of each stage: `docs/report/figures/pipeline.svg` /
`docs/report/methodology.md`.

```bash
# 1. Data (needs OPENAQ_API_KEY in .env — register at https://explore.openaq.org)
uv run python -m src.data.download_sensors                           # D1: 14 sensors (OpenAQ)
uv run python -m src.data.download_sensors --consulate               # D5: US Consulate monitor
uv run python -m src.data.download_weather --start 2025-05-01        # D2: ERA5 weather
uv run python -m src.data.download_air_quality --start 2025-05-01    # D3: CAMS air quality

# 2. Features
uv run python -m src.data.build_features

# 3. Baselines (persistence, Ridge, RF, logistic, RF classifier) on the CV folds
uv run python -m src.evaluate.run_cv

# 4. Tuning (classical models, then the NumPy MLP)
uv run python -m src.evaluate.tune_classical
uv run python -m src.evaluate.tune_mlp

# 5. Final test — scores the frozen settings once, then saves the production model
uv run python -m src.evaluate.final_test
uv run python -m src.evaluate.final_test --save-production

# 6. Explainability report (SHAP)
uv run python -m src.explain --features data/processed/features.parquet --bundle models/final_ensemble.pkl

# 7. Expert-system demo (prints advice + rule trace for one forecast)
uv run python -m src.expert --pm25 33 --today 25 --sensitive --outdoor

# 8. Live snapshot (today's forecast from live data, needs OPENAQ_API_KEY)
uv run python -m src.live --refresh

# 9. App
uv run streamlit run app/main.py
```

| # | Command | Writes | Roughly takes |
|---|---|---|---|
| 1a | `download_sensors` | `data/raw/sensors/S*.csv` + manifest | ~4-5 min (1 request per sensor per month, rate-limited) |
| 1b | `download_sensors --consulate` | `data/raw/us_consulate/*.csv` + manifest | ~1-2 min (1 sensor, ~75 months) |
| 1c | `download_weather` | `data/raw/weather/S*.csv` + manifest | ~1-2 min (14 sequential requests, no rate limit) |
| 1d | `download_air_quality` | `data/raw/air_quality/S*.csv` + manifest | ~1-2 min |
| 2 | `build_features` | `data/processed/{hourly_clean,features}.parquet`, `build_report.json` | ~1-2 min |
| 3 | `run_cv` | `results/baselines_cv_{folds,pooled,predictions}.*`, `baselines_summary.md` | a few minutes (8 CV folds × 5 baseline models) |
| 4a | `tune_classical` | `results/tuning/*_best.json`, `*_trials.csv`, `classical_summary.md` | ~50 min (grid search over Ridge/RF/HistGB + classifiers) |
| 4b | `tune_mlp` | `results/tuning/mlp_numpy_*` | ~7-8 min |
| 5a | `final_test` | `results/final_test/{metrics.csv,classification.csv,predictions.parquet,summary.md,run.json}` | ~1-2 min (refits each tuned model once on the development set). **Refuses to rerun** once `results/final_test/run.json` exists — pass `--force` only if something upstream genuinely changed |
| 5b | `final_test --save-production` | `models/final_ensemble.pkl` | under a minute (refits the frozen ensemble on all rows) |
| 6 | `src.explain` | `results/explain/{global_importance.csv,shap_summary.png,example_explanations.md}` | ~1-5 min (TreeExplainer is fast; the KernelExplainer cross-check on a 300-row sample is the slow part, ≈4.5 min — `docs/report/explainability.md`) |
| 7 | `src.expert` | stdout only (advice + fired rules) | under a second |
| 8 | `src.live --refresh` | `data/cache/` snapshot | ~2-3 min (OpenAQ rate limit) |
| 9 | `streamlit run app/main.py` | — | first load ~5 s; the SHAP reasons section adds ~15 s the first time it runs |

Existing raw files are skipped on re-run; add `--force` to re-download. Steps 3-6 all read
`data/processed/features.parquet`, so step 2 must run first; step 5b must run before step 6
or the app, since both need `models/final_ensemble.pkl`.

## Project layout

```
├── PROJECT_SPEC.md     what we are building and why
├── ROADMAP.md          phase-by-phase plan and progress
├── docs/
│   ├── briefs/         original course briefs
│   └── report/         report drafts, diagrams
├── data/               (not in git — rebuilt by the download scripts)
│   ├── raw/            downloaded, never edited
│   └── processed/      cleaned feature tables
├── src/
│   ├── config.py       paths, timezone, random seed
│   ├── data/           download, clean, features
│   ├── models/
│   │   ├── baselines/  persistence, ridge, random forest
│   │   ├── numpy_nn/   from-scratch network
│   │   └── torch_nn/   PyTorch twin
│   ├── classify/       AQI band classifier
│   ├── expert/         rule base, inference, explanations
│   ├── explain/        SHAP
│   ├── evaluate/       metrics, comparison tables
│   ├── live/           the Snapshot contract (+ the live-fetch pipeline)
│   └── ui/             pure, Streamlit-free helpers behind app/
├── app/                Streamlit app (views/, components/, main.py)
├── models/             trained models (not in git)
├── results/            metrics tables and figures
├── notebooks/          exploration only
└── tests/
```

## Updating dependencies

Edit `pyproject.toml`, then:

```bash
uv sync
uv export --format requirements-txt --no-hashes --all-groups -o requirements.txt
```

Commit `pyproject.toml`, `uv.lock` and `requirements.txt` together.
