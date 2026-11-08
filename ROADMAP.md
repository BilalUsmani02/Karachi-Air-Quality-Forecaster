# Karachi Air-Quality Forecaster — Roadmap

> Phase-by-phase build plan. Work top to bottom; tick tasks as they are done.
> The **what** and **why** live in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) — this file is the **when** and **in what order**.
> Last updated: 21 October 2026 · **Deadline: ASAP — Core only, no stretch until reports are drafted.**

---

## Status at a glance

| Phase | Name | Status | Started | Finished |
|---|---|---|---|---|
| 0 | Setup | ✅ Done | 21 Oct 2026 | 21 Oct 2026 |
| 1 | Data | ✅ Done | 21 Oct 2026 | 8 Nov 2026 |
| 2 | Clean + features | 🟨 In progress | 8 Nov 2026 | |
| 3 | Baselines | ⬜ Not started | | |
| 4 | NumPy network | ⬜ Not started | | |
| 5 | Improve + classify | ⬜ Not started | | |
| 6 | Rules + explanations | ⬜ Not started | | |
| 7 | App | ⬜ Not started | | |
| 8 | Reports + viva | ⬜ Not started | | |
| 9 | Stretch (optional) | ⬜ Not started | | |

Status key: ⬜ Not started · 🟨 In progress · ✅ Done · ⛔ Blocked (write why in the phase notes)

**Rule:** don't start a phase until the previous phase's *Done when* checks pass — except
Phase 4, which can start as soon as Phase 2 is done (it is the slowest; run it alongside Phase 3).

---

## Phase 0 — Setup

**Goal:** a clean, runnable project skeleton and permission to use one project for three courses.
**Effort:** 1–2 days · **Spec:** §11, §13

- [x] Instructor approval for the shared project (confirmed by Bilal, 21 Oct 2026)
- [x] PAI brief — none exists; `PAI_Overview_original.docx` is the reference
- [x] Deadlines: ASAP (Core only). Group members: added at the end, in Phase 8
- [x] Create the repo (Bilal) and connect this folder to it — github.com/BilalUsmani02/Karachi-Air-Quality-Forecaster
- [x] `.gitignore` (excludes `data/`, `models/`, `.venv/`, caches, `.env`)
- [x] Create the folder layout from spec §13 (+ `models/`, `results/`, `src/config.py`)
- [x] `pyproject.toml` + `uv.lock` + exported `requirements.txt`
- [x] Virtual environment set up (Python 3.12 via uv); versions pinned in `uv.lock`
- [x] `README.md` with install + run steps
- [x] pytest config with coverage on `src/`

**Done when:**
- [x] Approvals received
- [x] `pytest` runs green (3 smoke tests on `src/config.py`)
- [x] First commit pushed

**Notes:**
- 21 Oct 2026 — Python 3.12 chosen over the installed 3.14 for reliable PyTorch/SHAP wheels.
  Installed: NumPy 2.5, pandas 3.0, scikit-learn 1.9, PyTorch 2.14 (CPU), SHAP 0.54, Streamlit 1.66.
- Coverage `fail_under` is 0 for now; raise to 80 in Phase 8 (`pyproject.toml`).

---

## Phase 1 — Data

**Goal:** all raw data downloaded, described and reproducible.
**Effort:** ~1 week (longer if sensor access is hard) · **Spec:** §5.1, §5.2, §11

- [x] **Confirm sensor data access** — OpenAQ mirror, free key; 14 usable sensors (spec §5.2)
- [x] Update `src/data/sensors.py` to the 14 OpenAQ sensors
- [x] `src/data/download_sensors.py` — D1 hourly PM2.5 from OpenAQ (monthly requests)
- [x] D5: US Consulate history (OpenAQ 8156) — 46,937 hourly rows, 2019–2025
- [x] Verify who operates the sensors — The Urban Unit (AirGradient)
- [x] Fill sensor coordinates in spec §5.2 (8 online sensors)
- [x] `src/data/download_weather.py` — D2 (Open-Meteo), per sensor coordinate
- [x] `src/data/download_air_quality.py` — D3 (Open-Meteo / CAMS); data starts 2022-08-05
- [x] D4: EPA + WHO guidance + 22 draft rules in `docs/report/knowledge_sources.md`
- [x] Downloaders retry on failure and never overwrite raw files silently (D2, D3)
- [x] Source table filled (spec §5.1 download record)
- [x] Exploration notebook `notebooks/01_explore_data.ipynb`; findings in `docs/report/data_exploration.md`

**Done when:**
- [x] Re-running the downloaders reproduces `data/raw/` (commands in README)
- [x] Every source is described in the spec table
- [x] You know the usable date range per sensor

**Notes:**
- 21 Oct 2026 — AirGradient public API gives **live** readings only; `measures/past` needs the
  owner's token. OpenAQ mirrors public AirGradient sensors and has a public S3 archive, but finding
  the location IDs needs a free OpenAQ API key → **Bilal to register** at explore.openaq.org and put
  `OPENAQ_API_KEY=...` in `.env`. Fallback: email The Urban Unit for a token / CSV export.
- Open-Meteo weather (archive API) and air-quality (CAMS) APIs both work without a key.
- 8 Nov 2026 — **Unblocked.** OpenAQ key works. 18 Urban Unit sensors mirrored from Jun 2025;
  14 have ≥150 complete days (4,107 sensor-days total). US Consulate reference monitor has
  1,681 complete days (2019–2025). No Hazardous days in the Urban Unit data.
- 8 Nov 2026 — **Phase 1 closed.** Exploration: 4,113 complete sensor-days; sensors agree closely
  (daily r = 0.91–0.99); only 79 spike-hours and 5 zeros, no negatives or flat-lines; CAMS is one
  city-wide value, r = 0.73 with sensors but ~14.5 µg/m³ low; PM2.5 falls with temperature, wind,
  humidity and boundary-layer height (r ≈ −0.5 to −0.65). Weak sensors: S11, S13, S14 (<45% hours).

---

## Phase 2 — Clean + features

**Goal:** one tidy feature table with honest targets and no leakage.
**Effort:** ~1 week · **Spec:** §5.3, §5.4

**Open decision — split:** the notebook proposes train Jun 2025–May 2026, validation Jun–Aug 2026
(monsoon), test Sep–Nov 2026. Risk: monsoon validation has few Unhealthy days, so tuning is done
on the cleanest season, and the worst winter months appear only once (in training). Compare with
time-ordered cross-validation (*rolling-origin*: several train → next-block folds) before fixing.

- [ ] Align all sources to hourly, PKT (UTC+5)
- [ ] Fault flags: negatives, flat-lines, spikes — rule written down
- [ ] Gap handling: short gaps interpolated, long gaps dropped and counted
- [ ] Optional humidity correction for low-cost sensors
- [ ] Features: PM2.5 lags (1, 2, 3, 6, 12, 24 h), rolling mean/std, hour + weekday sin/cos, wind u/v, modelled PM2.5, sensor one-hot
- [ ] Target: next-day 24-h mean PM2.5 (issue time 00:00 PKT, ≥18/24 valid hours)
- [ ] Band label from the target (EPA 2024 table)
- [ ] Time-based train / validation / test split; date ranges recorded
- [ ] Scaler fit on train only
- [ ] Tests: no future data in features, target window correct, split order correct

**Done when:**
- [ ] `data/processed/features.parquet` (or CSV) builds from raw with one command
- [ ] Leakage tests pass
- [ ] Rows dropped at each step are counted and logged

**Notes:**

---

## Phase 3 — Baselines

**Goal:** first honest numbers to beat.
**Effort:** 3–4 days · **Spec:** §6.2 steps 0–2, §6.4, §6.5

- [ ] `src/evaluate/metrics.py` — MAE, RMSE, R², skill vs persistence; accuracy, macro-F1, confusion matrix
- [ ] Persistence model
- [ ] Ridge regression (**official baseline**)
- [ ] Random Forest regressor
- [ ] Logistic regression classifier (band baseline)
- [ ] Random Forest classifier
- [ ] Band counts per class on the training split → decide merges (<~30 samples)
- [ ] Results saved to `results/` as tables (per sensor + overall)

**Done when:**
- [ ] One comparison table with all baselines on the validation set
- [ ] Band merge decision written in spec §6.4
- [ ] Metrics module has tests

**Notes:**

---

## Phase 4 — NumPy network (PAI core)

**Goal:** a from-scratch MLP that provably matches PyTorch.
**Effort:** ~1.5 weeks — hardest phase; start right after Phase 2 · **Spec:** §6.3

- [ ] Dense layer: forward + backward
- [ ] Activations: ReLU + one alternative (tanh or LeakyReLU)
- [ ] Losses: MSE + Huber
- [ ] Weight init: He / Xavier, fixed seed
- [ ] Optimiser: mini-batch SGD, then Adam
- [ ] L2 regularisation; dropout (optional)
- [ ] Early stopping on validation loss
- [ ] **Gradient check** vs numerical gradients (test)
- [ ] PyTorch twin with the same architecture in `src/models/torch_nn/`
- [ ] Same data + same initial weights → losses match within tolerance (test)
- [ ] Compare: final loss, metrics, training time

**Done when:**
- [ ] Gradient check passes for every layer and loss
- [ ] NumPy vs PyTorch match test passes
- [ ] NumPy network added to the comparison table

**Notes:**

---

## Phase 5 — Improve + classify (AIES core)

**Goal:** beat the baselines and answer "number first, or band directly?"
**Effort:** ~1 week · **Spec:** §6.2 steps 5–6, §6.4

- [ ] Hyper-parameter search on validation set (RF, Ridge α, NN size / learning rate / L2)
- [ ] Simple average ensemble of the best models
- [ ] Route 1: regress then band
- [ ] Route 2: direct classifier with class weights (and merged bands)
- [ ] Compare routes: macro-F1, per-class recall, confusion matrices
- [ ] **Final evaluation on the test period — run once**, after all choices are frozen
- [ ] Fairness check: error per sensor / neighbourhood (spec §10)

**Done when:**
- [ ] Final results table (regression + classification) on the test set
- [ ] Best model(s) saved to `models/` with version info
- [ ] Route comparison written up in notes for the AIES report

**Notes:**

---

## Phase 6 — Rules + explanations

**Goal:** every output comes with advice and reasons.
**Effort:** ~1 week · **Spec:** §7, §8

- [ ] Facts model: band, trend, sensitive group, outdoor activity, time of day
- [ ] Rule base with a source (EPA / WHO) on every rule
- [ ] Forward-chaining engine
- [ ] Conflict resolution: most specific wins, ties → safest advice
- [ ] Explanation trace: which rules fired and why
- [ ] SHAP TreeExplainer on the best tree model
- [ ] SHAP DeepExplainer on the PyTorch twin
- [ ] Plain-English reason text ("Low wind raised the forecast by ~8 µg/m³")
- [ ] Global SHAP summary plot for the report
- [ ] Tests: each rule fires on the right facts; conflicts resolve correctly

**Done when:**
- [ ] For any test-set row: forecast → band → advice → rules fired → top SHAP reasons
- [ ] Rule engine has tests

**Notes:**

---

## Phase 7 — App

**Goal:** a demo anyone can use, even offline.
**Effort:** ~1 week · **Spec:** §9, §10

- [ ] Sensor picker: dropdown + map
- [ ] Forecast value + colour-coded band
- [ ] Advice + "You see this because…" rule trace
- [ ] Top SHAP reasons
- [ ] Chart: recent readings + forecast
- [ ] Model comparison page
- [ ] Profile toggles kept in session only (not stored or logged)
- [ ] Cache + "last updated" banner + warning when live fetch fails
- [ ] Disclaimer: informational, not official health advice; data sources + model version shown

**Done when:**
- [ ] App runs from one command in the README
- [ ] Demo works with the internet switched off
- [ ] A non-technical person can read the result without help

**Notes:**

---

## Phase 8 — Reports + viva

**Goal:** submissions that point to real code and real results.
**Effort:** ~1 week · **Spec:** §3, §4, §12

- [ ] Pipeline / methodology diagram
- [ ] **AIES report** (≥10 pages) — lead with classification, expert system, baseline → improvement
- [ ] **PAI report** — lead with the NumPy network and comparison (format per PAI brief)
- [ ] Each report written fresh from shared results — no copied paragraphs
- [ ] Traceability table (spec §3): every row points to real code / section
- [ ] Coverage ≥80% on `src/`
- [ ] Viva prep for every member: inputs/outputs, why each model over alternatives, real-life use
- [ ] Fill group members + ownership in spec §11.1
- [ ] Slides (if required)
- [ ] Final tidy: README, requirements pinned, results reproducible

**Done when:**
- [ ] All requirements A1–A15, L1–L5, P1–P5 checked off
- [ ] Each member can demo and explain any part

**Notes:**

---

## Phase 9 — Stretch (only after Phase 8 is drafted)

- [ ] Gradient Boosting
- [ ] Stacking ensemble
- [ ] NumPy softmax classifier (direct route)
- [ ] Hourly forecasts (1 h / 6 h) for the chart
- [ ] KernelExplainer cross-check on the NumPy network

---

## Change log

| Date | Change |
|---|---|
| 21 Oct 2026 | Roadmap created from spec §14 |
