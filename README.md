# Karachi Air-Quality Forecaster

Forecasts tomorrow's average fine-dust level (PM2.5) at 9 sensor sites in Karachi, sorts it
into an air-quality band (Good → Hazardous), and gives personalised, explained health advice.

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

Training commands are added here as each phase lands.

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
