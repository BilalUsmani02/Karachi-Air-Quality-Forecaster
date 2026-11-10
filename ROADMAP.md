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
| 2 | Clean + features | ✅ Done | 8 Nov 2026 | 8 Nov 2026 |
| 3 | Baselines | ✅ Done | 8 Nov 2026 | 8 Nov 2026 |
| 4 | NumPy network | ✅ Done | 8 Nov 2026 | 8 Nov 2026 |
| 5 | Improve + classify | ✅ Done | 8 Nov 2026 | 10 Nov 2026 |
| 6 | Rules + explanations | ✅ Done | 10 Nov 2026 | 10 Nov 2026 |
| 7 | App | ✅ Done | 10 Nov 2026 | 10 Nov 2026 |
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

**Decided — split:** 8 rolling-origin folds (30-day test blocks, Jan–Aug 2026) for model
selection; final test Sep 1 – Nov 8 2026, scored once. The single holdout was rejected because its
monsoon validation block had zero Unhealthy days (spec §5.3).

- [x] Align all sources to hourly, PKT (UTC+5) — `src/data/clean.py`
- [x] Fault flags: negatives, flat-lines, spikes — rules in `docs/design/phase2_data_contract.md` §1
- [x] Gap handling: gaps ≤3 h interpolated, longer stay missing and are counted
- [x] Humidity correction — decided **not** to apply (no inflation found in Phase 1)
- [x] Features: PM2.5 lags, rolling stats, calendar sin/cos, wind u/v, past-day and forecast-day weather + CAMS, sensor one-hot — `src/data/features.py` (64 columns)
- [x] Target: next-day 24-h mean PM2.5 (issue time 00:00 PKT, ≥18/24 valid hours)
- [x] Band label from the target (EPA 2024 table) — `src/classify/bands.py`
- [x] Time-based split — rolling-origin folds + final test, `src/data/split.py::project_splits`
- [x] Scaler fit on train only — `Preprocessor`, fitted inside each fold (Phase 3)
- [x] Tests: no future data in features, target window correct, split order correct (209 tests)

**Done when:**
- [x] `data/processed/features.parquet` builds from raw with one command (`src.data.build_features`)
- [x] Leakage tests pass (plus 200-row spot check against the hourly table: 0 mismatches)
- [x] Rows dropped at each step are counted and logged (`data/processed/build_report.json`)

**Notes:**
- 8 Nov 2026 — **Phase 2 closed.** Cleaning: 102,019 ok hours, 995 interpolated, 20 spikes removed.
  Feature table: **4,027 sensor-days** (of 7,276 candidates), 2025-06-20 → 2026-11-08, 64 columns.
  Bands: Good 110 · Moderate 2,056 · USG 734 · Unhealthy 1,029 · Very Unhealthy 98 (Hazardous merged).
- `pm25_mean_24h` equals `pm25_prev_day_mean` by construction (issue time is midnight) — drop one
  before fitting linear models.
- `fc_*` features use recorded weather/CAMS for the target day as a stand-in for forecasts
  (*perfect-prognosis*); list as a limitation in the report.

---

## Phase 3 — Baselines

**Goal:** first honest numbers to beat.
**Effort:** 3–4 days · **Spec:** §6.2 steps 0–2, §6.4, §6.5

- [x] `src/evaluate/metrics.py` — MAE, RMSE, R², skill vs persistence; accuracy, macro-F1, confusion matrix
- [x] Persistence model
- [x] Ridge regression (**official baseline**)
- [x] Random Forest regressor
- [x] Logistic regression classifier (band baseline)
- [x] Random Forest classifier
- [x] Band counts per class on the training split → Hazardous merged, Good kept (spec §6.4)
- [x] Results saved to `results/` as tables (per sensor + overall)

**Done when:**
- [x] One comparison table with all baselines on the validation set
- [x] Band merge decision written in spec §6.4
- [x] Metrics module has tests

**Notes:**
- 8 Nov 2026 — pooled over the 8 CV folds (1,458 sensor-days, 1 Jan – 28 Aug 2026; corrected from a mistyped 2,224 on 10 Nov 2026), `results/baselines_summary.md`:

  | Model | MAE µg/m³ | RMSE | R² | Skill vs persistence |
  |---|---|---|---|---|
  | Persistence | 8.70 | 14.70 | 0.749 | 0 |
  | Ridge (official baseline) | 7.68 | 11.10 | 0.857 | +12% |
  | Random Forest | **6.79** | 10.61 | 0.869 | **+22%** |
  | NumPy MLP (untuned) | 11.21 | 15.42 | 0.723 | −29% |

  Bands (macro-F1): persistence 0.678 · Ridge→band 0.657 · RF→band 0.598 · RF direct 0.578 ·
  logistic 0.509 · MLP→band 0.504. Lower MAE did **not** mean better bands: RF/Ridge pull extremes
  toward the middle, so Very Unhealthy recall is 0–28%. Phase 5 must target macro-F1, not just MAE.
- Fixed a row-order bug: `cross_validate` now sorts by date so early stopping holds out the
  latest days, not the last sensors (test added).

---

## Phase 4 — NumPy network (PAI core)

**Goal:** a from-scratch MLP that provably matches PyTorch.
**Effort:** ~1.5 weeks — hardest phase; start right after Phase 2 · **Spec:** §6.3

- [x] Dense layer: forward + backward
- [x] Activations: ReLU + one alternative (tanh or LeakyReLU)
- [x] Losses: MSE + Huber
- [x] Weight init: He / Xavier, fixed seed
- [x] Optimiser: mini-batch SGD, then Adam
- [x] L2 regularisation; dropout (optional)
- [x] Early stopping on validation loss
- [x] **Gradient check** vs numerical gradients (test)
- [x] PyTorch twin with the same architecture in `src/models/torch_nn/`
- [x] Same data + same initial weights → losses match within tolerance (test)
- [x] Compare: final loss, metrics, training time

**Done when:**
- [x] Gradient check passes for every layer and loss
- [x] NumPy vs PyTorch match test passes
- [x] NumPy network added to the comparison table

**Notes:**
- 8 Nov 2026 — gradient check relative errors ~1e-10 (threshold 1e-6); NumPy vs PyTorch on real
  data: max prediction difference 1e-13 µg/m³, max weight difference 6e-16. NumPy trains in ~4 s vs
  ~38 s for PyTorch on the same data.
- Untuned network is the weakest model: good in winter (fold 0 MAE 15.9 vs persistence 19.9) but
  poor in the spring transition (fold 2 MAE 22.3 vs 7.9). Tune in Phase 5 (size, L2, dropout,
  learning rate, Huber loss).

---

## Phase 5 — Improve + classify (AIES core)

**Goal:** beat the baselines and answer "number first, or band directly?"
**Effort:** ~1 week · **Spec:** §6.2 steps 5–6, §6.4

- [x] Hyper-parameter search on the 8 CV folds (Ridge, RF, Gradient Boosting, NumPy network, classifiers)
- [x] Simple average ensemble — Gradient Boosting + NumPy network (`src/models/final.py`)
- [x] Route 1: regress then band
- [x] Route 2: direct classifier with class weights (and merged bands)
- [x] Compare routes — regress-then-band chosen (fewest missed Unhealthy days)
- [x] **Final evaluation on the test period — run once** (10 Nov 2026, `results/final_test/`)
- [x] Fairness check: error per sensor / neighbourhood (`results/fairness/`)

**Done when:**
- [x] Final results table (regression + classification) on the test set
- [x] Best model saved: `models/final_ensemble.pkl` (refit on all rows, gitignored; rebuild with `final_test --save-production`)
- [x] Route comparison written up: `docs/report/model_selection.md`

**Notes:**
- 10 Nov 2026 — **Phase 5 closed.** Full write-up: `docs/report/model_selection.md`.
  - CV MAE after tuning: Ridge 7.22 · RF 6.54 · Gradient Boosting 6.30 · NumPy network 6.63
    (from 11.21: no calendar features, change-from-yesterday target, 32 units, L2 0.1).
  - Final model: **average of Gradient Boosting + NumPy network** — CV MAE 6.16 (+29% vs persistence).
  - Band route: regress-then-band — CV accuracy 82.6%, misses 8% of Unhealthy days (persistence 18%).
  - **Final test (Sep 1 – Nov 8 2026, once):** ensemble MAE 4.63 (+20.5%), band accuracy 84.6%,
    misses 14.8% of Unhealthy days (persistence 22.7%). No Very Unhealthy days in this period.
  - Fairness: errors are larger in more polluted areas (Orangi worst, MAE 18.5 in CV); mostly
    explained by pollution level (ρ = 0.81), not data gaps (ρ = −0.47). `results/fairness/`.
  - The +5 µg/m³ safety margin is **not** applied to the band (it erases "Good"); it goes into
    the advice layer in Phase 6.
- Two tuning sub-agents stalled while waiting on long searches; their code was recovered and the
  searches rerun as background jobs. Test suite now takes ~9 min (tuning tests fit real models).

---

## Phase 6 — Rules + explanations

**Goal:** every output comes with advice and reasons.
**Effort:** ~1 week · **Spec:** §7, §8

- [x] Facts model: band, trend, sensitive group, outdoor activity (time of day dropped — the forecast is a daily mean)
- [x] Rule base with a source (EPA / WHO) on every rule — R01–R22, stored as data (`src/expert/rules.py`)
- [x] Forward-chaining engine (`src/expert/engine.py`) — runs to a fixpoint; the safety margin (R21) is a real chained rule
- [x] Conflict resolution: most specific wins, ties → safest advice, then the rule that checks sensitive-group status
- [x] Explanation trace: which rules fired and why (`src/expert/explain.py`)
- [x] SHAP TreeExplainer on the best tree model (Gradient Boosting member)
- [x] ~~SHAP DeepExplainer on the PyTorch twin~~ → replaced by a model-agnostic explainer on the whole production ensemble (explains what the app actually serves)
- [x] Plain-English reason text ("Yesterday's average PM2.5 was high (137.3 µg/m³), which raised the forecast by about 37 µg/m³.")
- [x] Global SHAP summary plot for the report (`results/explain/shap_summary.png`)
- [x] Tests: each rule fires on the right facts; conflicts resolve correctly (167 expert + 23 SHAP tests)

**Done when:**
- [x] For any test-set row: forecast → band → advice → rules fired → top SHAP reasons
- [x] Rule engine has tests

**Notes:**
- Safety margin: if the forecast is within 5 µg/m³ below the next band (USG or worse), the
  advice follows the worse band and says so; the band *shown* never changes.
- Trend: rising/falling when the change is ≥ max(5 µg/m³, 10% of today).
- Found a gap in the draft rule table: R15 ties with R14/R16/R17 on both specificity and
  severity; a third, documented tie-break resolves it (`docs/report/expert_system.md` §5).
- SHAP: one live explanation takes ~0.7 s; additivity error ~1e-13 µg/m³. Top global factors:
  yesterday's PM2.5, last hour's PM2.5, CAMS PM2.5 and NO₂ forecasts (`docs/report/explainability.md`).
- Full suite: 600 tests pass, 98% coverage.

---

## Phase 7 — App

**Goal:** a demo anyone can use, even offline.
**Effort:** ~1 week · **Spec:** §9, §10

- [x] Sensor picker: dropdown + map (click a dot to choose a site)
- [x] Forecast value + colour-coded band
- [x] Advice + "You see this because…" rule trace
- [x] Top SHAP reasons
- [x] Chart: recent readings + forecast
- [x] Model comparison page
- [x] Profile toggles kept in session only (not stored or logged)
- [x] Cache + "last updated" banner + warning when live fetch fails
- [x] Disclaimer: informational, not official health advice; data sources + model version shown

**Done when:**
- [x] App runs from one command in the README (`uv run streamlit run app/main.py`)
- [x] Demo works with the internet switched off (cache, else a demo day from stored data)
- [ ] A non-technical person can read the result without help — ask a group member or friend to try it

**Notes:**
- Live pipeline (`src/live/`): the forecast covers the day starting at 00:00 PKT, so every input
  is complete when it is issued. Live inputs matched the stored training rows on 13 real days.
  A live refresh takes ~2.5 min (OpenAQ rate limit).
- First live run (10 Nov 2026): 9 of 14 sensors forecast; 4 had no readings for 10 days and 1
  had too few the day before.
- Checked in a real browser, which found 3 bugs the automated tests missed: the app could not
  start (`No module named 'app'`), the map never drew (invalid marker setting), and the wording
  said "tomorrow" for a same-day forecast. All fixed, with tests for the first two.
- First page load takes ~5 s (scikit-learn); SHAP loads afterwards, only for the reasons section.

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
