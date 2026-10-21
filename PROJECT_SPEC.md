# Karachi Air-Quality Forecaster — Project Specification

> One project, three submissions: **PAI (CT-264)**, **AIES (CT-361) CCP** and **AIES Lab (CT-361) CCA**.
> Author: Bilal · Last updated: 21 October 2026
> Original briefs are in [`docs/briefs/`](docs/briefs/).

---

## 1. What we are making

A tool that forecasts how polluted the air in Karachi will be **tomorrow**. It predicts the
**24-hour average** level of fine dust particles (**PM2.5**, µg/m³) at the real sensors
(8 online as of Oct 2026) run by **The Urban Unit**, sorts each forecast into an
air-quality band (Good → Hazardous), and gives plain health advice for it.

At its heart is a **neural network written entirely by hand in NumPy** (no deep-learning
library). Around it sit:

1. **The forecaster** — NumPy neural network predicting PM2.5 per sensor (regression).
2. **The band classifier** — turns the forecast into an AQI category (classification).
3. **The comparison** — the same task solved with scikit-learn and PyTorch, starting from a
   simple baseline and improving through tuning and ensembling.
4. **The expert-system layer** — if-then rules that turn a band + user profile into advice,
   with an explanation of which rules fired.
5. **The explanations** — SHAP breakdown of which factors (wind, humidity, time of day…)
   pushed each forecast up or down.
6. **The web app** — Streamlit: pick a location, see the forecast, the band, the advice and
   the reasons.

### Why it matters

Karachi regularly faces unhealthy air, yet most people only find out how bad it is once they
are already breathing it. A forecast turns that into something people can plan around — when
to go outside, exercise, or keep windows shut.

### One project, three framings

The code base is shared. What changes per course is **which part we lead with** in the
report, the demo and the viva.

| Course | Lead with | Primary task | Supporting parts |
|---|---|---|---|
| **PAI (CT-264)** | The from-scratch NumPy network and its comparison with PyTorch / scikit-learn | Regression: next-day PM2.5 | Bands, rules, app |
| **AIES CCP (CT-361)** | Choosing AI techniques for a **classification** problem + the expert system | **Classification: tomorrow's AQI band** | Regression route, SHAP, app |
| **AIES Lab CCA (CT-361)** | Same as CCP, shown live: inputs/outputs, why each model, real-life use | **Classification: tomorrow's AQI band** | Same as CCP |

Both AIES CLOs say "classification problems", so in AIES material the band classifier is the
**main task** and the PM2.5 regression is presented as one of two routes to it (§6.4) — not
the other way round.

---

## 2. Course requirements

### 2.1 PAI (CT-264) — Semester Project

Source: `docs/briefs/PAI_Overview_original.docx`. **There is no official PAI brief** — this
overview is the agreed reference for PAI (confirmed 21 Oct 2026).

| # | Requirement |
|---|---|
| P1 | Neural network written from scratch in NumPy (no ready-made DL library) |
| P2 | Same task solved with standard tools (PyTorch, scikit-learn) for a fair comparison |
| P3 | Per-forecast explanations with SHAP |
| P4 | Streamlit web app: pick a location, see forecast + reasons |
| P5 | Scope: Karachi only, PM2.5 only, no official health alerts |

### 2.2 AIES (CT-361) — Complex Computing Problem (CCP)

Source: `docs/briefs/AIES_CCP.pdf`

**CLO 2 (C2):** Identify appropriate AI techniques to solve classification problems of moderate complexity.

**Complex-problem attributes:**
- **CP2 Depth of analysis** — no obvious solution; needs conceptual thinking and abstract models.
- **CP3 Depth of knowledge** — needs in-depth computing/domain knowledge and well-founded principles.

| # | Requirement |
|---|---|
| A1 | Real-world problem in an allowed domain (environmental monitoring ✅) |
| A2 | Clear title + problem statement: the issue, real-world impact, AI/ES approach used |
| A3 | Problem description: domain, use case, system goals |
| A4 | **At least two datasets/knowledge sources**, each described (source, data type, relevance) |
| A5 | Explain how datasets are used to build and evaluate the system; suitable metrics |
| A6 | Pre-processing for noise, missing values, inconsistencies |
| A7 | Handle diverse data types for the domain |
| A8 | Acknowledge external factors (environment, user behaviour, dynamic input) |
| A9 | Data privacy (anonymise personal/sensitive data) |
| A10 | Ethics: fairness, transparency, accountability |
| A11 | **Baseline model**, then improvement using **multiple AI approaches** |
| A12 | Real-time monitoring performance depends on network availability (handle it) |
| A13 | **Deliverable:** Python code — preprocessing, model building, evaluation, improvement (tuning/ensembling) |
| A14 | **Deliverable:** simple, user-friendly visualisation/interface (charts, maps, predictions) |
| A15 | **Deliverable:** technical report, **minimum 10 pages** — methodology **with diagram**, models, results, challenges |

**Rubric (/10):**

| Criterion | Satisfactory (5) | Maps to |
|---|---|---|
| 1. Code | Complete, accurate, well-structured; assumptions clear and valid; technical issues handled | CP3 |
| 2. Report | All necessary information, clearly organised | CP2 |

### 2.3 AIES Lab (CT-361) — Complex Computing Activity (CCA, PBL)

Source: `docs/briefs/AIES_Lab_CCA.pdf`

**CLO 3 (C3):** Identify appropriate AI techniques to solve classification problems of moderate complexity.

**Attributes:**
- **CA2 Problem analysis & solution development** — identify inputs/outputs, evaluate
  alternative strategies, develop an effective solution.
- **CA3 Application of AI knowledge** — apply suitable algorithms, explain practical applications.

Same requirements as AIES CCP (A1–A14) with these differences:

| # | Difference |
|---|---|
| L1 | Two datasets **or one sufficiently large** dataset |
| L2 | Explicitly mentions **class imbalance** in pre-processing |
| L3 | Approach list also includes AI agents and chatbots |
| L4 | No 10-page report listed — code + visualisation are the deliverables (assessed live) |
| L5 | Problem statement asks for **context and objectives** and **real-world significance** (CCP: "issue" and "impact") — use the Lab wording in §4 |

**Rubric (C3, each criterion 0–2):**

| Criterion | Top level (2) |
|---|---|
| Problem identification | Correctly identifies the problem and all required inputs/outputs |
| Problem-solving skills | Uses an effective strategy |
| Implementation | Outstanding implementation |
| Deliverables | Generates correct and complete output |
| Depth of knowledge | Thorough grasp; can apply it to real-life situations |

> Be ready to **explain verbally**: inputs/outputs, why each model was chosen over alternatives,
> and real-life use.

---

## 3. Requirement traceability

Where each requirement is satisfied. Keep this table updated as the build progresses.

| Requirement | Satisfied by | Section |
|---|---|---|
| P1 NumPy NN | `src/models/numpy_nn/` | §6.3 |
| P2, A11, A13 Baseline + multiple approaches + tuning/ensembling | Persistence → linear → RF/GBM → NumPy NN → PyTorch → ensemble | §6 |
| P3, A10 SHAP / transparency | `src/explain/` | §8 |
| P4, A14 Interface | Streamlit app | §9 |
| A4, L1 ≥2 datasets | D1 sensors + D4 health knowledge (independent sources); D2/D3 add features | §5.1 |
| A6, L2 Pre-processing, imbalance | Cleaning pipeline, class weights / resampling, band merging | §5.3, §6.4 |
| A7 Diverse data types | Time series, categorical, cyclic, gridded, text rules | §5.4 |
| CLO "classification" | AQI band classifier — **primary task in AIES** | §1, §6.4 |
| "Expert System" | Rule-based advice layer with explanation | §7 |
| A8 External factors | Weather features; limitations section | §5, §11 |
| A9 Privacy | No personal data stored; user profile stays in session | §10 |
| A12 Network availability | Cached data + "last updated" banner + offline mode | §9 |
| A15 Report ≥10 pages | `docs/report/` | §12 |

---

## 4. Problem statement (drafts for reports)

### 4.1 Titles

| Course | Title |
|---|---|
| PAI | Karachi Air-Quality Forecaster: A From-Scratch NumPy Neural Network for Next-Day PM2.5 Forecasting, Benchmarked Against PyTorch and scikit-learn |
| AIES CCP + Lab | Karachi Air-Quality Forecaster: Next-Day AQI Band Classification with a Rule-Based Expert System for Explainable Health Advice |

### 4.2 AIES CCP wording (issue / impact / approach)

**The issue:** Karachi often has PM2.5 levels well above WHO guidelines, but residents only see
current readings, not what is coming.

**Real-world impact:** Fine particles cause respiratory and heart disease. Children, older
people and people with asthma are hit hardest. Knowing tomorrow's air-quality band lets people
plan their exposure.

**AI / ES approach:** A hybrid system. Supervised classifiers predict tomorrow's AQI band at each
sensor, by two routes: (1) a regression model forecasts the 24-h mean PM2.5, which is then
banded; (2) a classifier predicts the band directly. Both start from a baseline and improve
through stronger models, tuning and ensembling. A forward-chaining rule base turns the band and
the user's profile into personalised advice. SHAP and rule traces explain every output.

### 4.3 AIES Lab wording (context and objectives / significance / approach)

**Context and objectives:** Karachi's air is often unhealthy, and residents only see current
readings. Objective: classify tomorrow's air-quality band at 9 sensor sites and give
personalised, explained advice.

**Real-world significance:** Lets sensitive groups (children, older people, people with asthma)
plan outdoor activity a day ahead.

**AI / ES approach:** Same as §4.2.

### 4.4 PAI wording

**Objective:** Build a neural network from scratch in NumPy that forecasts next-day 24-h mean
PM2.5, verify it against an identical PyTorch network, and compare it with scikit-learn
models. Bands, rules and SHAP make the output usable in a Streamlit app.

---

## 5. Data

### 5.1 Sources (≥2 required)

| # | Source | Type | Role | Status |
|---|---|---|---|---|
| D1 | **The Urban Unit** low-cost PM2.5 sensors (AirGradient devices), Karachi — operator confirmed 21 Oct 2026 | Time series, per sensor | **Target** (what we predict) + lag features | Live readings public (AirGradient API, no key). **History needs a key** — owner token (Urban Unit) or OpenAQ mirror (free key). See §5.2 |
| D2 | **Weather** — Open-Meteo Historical Weather API (`archive-api.open-meteo.com`; temperature, humidity, wind speed & direction, pressure, rain, boundary-layer height) | Time series, gridded | Input features | ✅ Works without key (tested 21 Oct 2026) |
| D3 | **Modelled pollution** — CAMS via the Open-Meteo Air Quality API (`air-quality-api.open-meteo.com`; PM2.5, PM10, NO₂, dust) | Time series, gridded | Input features | ✅ Works without key (tested 21 Oct 2026) |
| D4 | **Health-advice knowledge** — US EPA AQI guidance, WHO 2021 air quality guidelines | Documents | Expert-system rule base | — |

Record for each: URL, licence, date range, time step, number of rows, and the date downloaded.

**How we meet "≥2 datasets / knowledge sources" (A4, L1):** D1 (measured sensor data) and D4
(health-guidance knowledge) are fully independent sources. D2 and D3 both come from
Open-Meteo, so in the report present them as **two datasets from one provider** (different
underlying models: weather reanalysis vs CAMS atmospheric composition) — do not rely on them
alone to meet the requirement.

### 5.2 Sensor locations

Source: AirGradient public API (`/public/api/v1/world/locations/measures/current`), contributor
"The Urban Unit", checked 21 Oct 2026. These 8 were online; IDs are AirGradient location IDs.

| ID | Site | AirGradient ID | Lat / Lon |
|---|---|---|---|
| S1 | NED University, Main Campus | 162732 | 24.9338, 67.1135 |
| S2 | Aga Khan University, Main Campus | 162672 | 24.8924, 67.0747 |
| S3 | Orangi Pilot Project | 162802 | 24.9373, 67.0190 |
| S4 | Urban Resource Center | 162379 | 24.8680, 67.0818 |
| S5 | Gulistan-e-Johar | 162753 | 24.9162, 67.1315 |
| S6 | Climate Action Center | 162364 | 24.8638, 67.0566 |
| S7 | Malir Cantonment, Askari 5 | 162762 | 24.9396, 67.1810 |
| S8 | Clifton | 206703 | 24.8103, 67.0321 |

**Not online on 21 Oct 2026** (in the original overview): Aga Khan University sensor 2,
Solid Waste Management, G3 Engineering. Include them if their history turns up in D1; otherwise
the project uses the 8 above. Final count depends on how much history each sensor has.

Weather and modelled data are pulled per coordinate (Open-Meteo snaps to its nearest grid cell).

### 5.3 Pre-processing

- Align all sources to one time step (hourly) and one timezone (PKT, UTC+5).
- Missing values: short gaps → interpolation; long gaps → drop the window and record how much was dropped.
- Outliers / sensor faults: flag impossible values (negative, flat-lined, sudden spikes) and
  document the rule used.
- Low-cost sensor bias: optionally adjust using humidity (common correction for low-cost PM sensors).
- Features: PM2.5 lags (1, 2, 3, 6, 12, 24 h), rolling mean/std, hour of day and weekday
  (sine/cosine), wind as u/v components, modelled PM2.5, sensor ID (one-hot).
- Scaling: fit the scaler on training data only.
- **Split by time, never randomly**: train / validation / test in order. Report the date
  range of each split. No future data may leak into features.
- **Target construction:** for each sensor and each forecast issue time (00:00 PKT), the target
  is the mean of the next 24 hourly readings. Require ≥18 of 24 valid hours (EPA-style
  completeness rule), otherwise drop that row and count it.

### 5.4 Data types handled (A7)

| Type | Example | Handling |
|---|---|---|
| Numeric time series | PM2.5, temperature, humidity | Lags, rolling stats, scaling |
| Categorical | Sensor ID, weekday | One-hot encoding |
| Cyclic | Hour of day, wind direction | Sine/cosine or u/v components |
| Gridded model output | CAMS PM2.5, boundary-layer height | Nearest grid cell to each sensor |
| Text / documents | EPA and WHO guidance | Hand-encoded into if-then rules (§7) |
| Live user input | Sensitive group, outdoor plans | Session-only facts for the rule engine |

---

## 6. Models

### 6.1 Forecast task

- **Regression target:** the **24-hour mean PM2.5 for the next day** at each sensor (§5.3).
  This matches the EPA breakpoints, which are defined on 24-h averages — banding a single
  hourly value would mislabel days.
- **Classification target:** the AQI band of that 24-h mean (§6.4).
- **Inputs:** features from §5.3 available at forecast time (00:00 PKT).
- *Stretch:* hourly forecasts at 1 h / 6 h for the chart only — never banded with the 24-h table.

### 6.2 Ladder of models (baseline → improvement)

| Step | Model | Library | Purpose | Tier |
|---|---|---|---|---|
| 0 | Persistence ("tomorrow = today") | — | Simplest possible baseline | Core |
| 1 | Linear / Ridge regression | scikit-learn | **Official baseline** (A11) | Core |
| 2 | Random Forest (Gradient Boosting = stretch) | scikit-learn | Strong classical model | Core |
| 3 | **MLP from scratch** | NumPy | Core of PAI | Core |
| 4 | Same MLP architecture | PyTorch | Fair check of the NumPy version; needed for SHAP (§8) | Core |
| 5 | Tuned versions (grid/random search on validation set) | — | Improvement (A13) | Core |
| 6 | Ensemble (average of best models; stacking = stretch) | — | Improvement (A13) | Core |

**Core** = the minimum that meets every requirement in all three courses. Build all Core items
end-to-end (data → app) **before** starting any stretch item.

### 6.3 NumPy neural network (PAI core)

Must be hand-written: dense layers, ReLU (and one alternative), forward pass, backpropagation,
MSE/Huber loss, mini-batch gradient descent with Adam, L2 regularisation, dropout (optional),
early stopping on validation loss, weight initialisation (He/Xavier), fixed random seed.

Verification: **gradient check** against numerical gradients; match the PyTorch version's
loss within a small tolerance on the same data and initial weights.

### 6.4 Band classifier (AIES "classification" requirement)

Bands (US EPA PM2.5 breakpoints, 2024 revision, 24-h average, µg/m³):

| Band | PM2.5 |
|---|---|
| Good | 0.0 – 9.0 |
| Moderate | 9.1 – 35.4 |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| Unhealthy | 55.5 – 125.4 |
| Very Unhealthy | 125.5 – 225.4 |
| Hazardous | ≥ 225.5 |

This is the **primary task for AIES** (§1). Two routes, compared:
1. **Regress then band** *(Core)* — apply the table to the 24-h mean forecast from §6.2.
2. **Direct classifier** *(Core: logistic regression baseline + Random Forest;
   stretch: NumPy softmax network)* — predicts the band straight from the features.

Comparing the two routes is the AIES "evaluate alternative strategies" story (CA2): does
predicting the number first beat predicting the category directly?

**Class imbalance (L2):** Karachi has few "Good" days (cut-off is 9.0 µg/m³) and few
"Hazardous" days.
- Count samples per band on the training split first and report the table.
- If a band has fewer than ~30 training samples, **merge it with its neighbour** (e.g.
  Good + Moderate, Very Unhealthy + Hazardous) and state the merge in the report.
- Then handle what remains with class weights (or resampling on the training split only).
- Report **macro-F1**, not only accuracy, plus the confusion matrix.

### 6.5 Metrics

- **Regression:** MAE, RMSE, R², skill score vs persistence; per sensor and overall.
- **Classification:** accuracy, macro-F1, per-class precision/recall, confusion matrix.
- **NumPy vs PyTorch:** final loss, metrics, training time.
- Report all on the **held-out test period** only.

---

## 7. Expert-system layer

A small rule base, run by forward chaining (rules fire when their conditions match the known facts).

- **Facts:** forecast band, trend (rising/falling), user profile (sensitive group? planned outdoor activity?), time of day.
- **Rules (examples):**
  - `IF band ≥ Unhealthy AND sensitive = yes THEN advice = "Stay indoors; keep windows shut"`
  - `IF band = Moderate AND activity = outdoor exercise THEN advice = "Shorten or move indoors"`
  - `IF trend = rising AND band = USG THEN advice += "Conditions worsening — plan ahead"`
- **Conflict resolution:** most-specific rule wins; ties broken by severity (safest advice wins).
- **Explanation facility:** show which rules fired and why ("You see this because…").
- **Sources:** EPA AQI guidance, WHO guidance (D4). Cite each rule's source.
- Advice is informational, **not an official health warning** (out of scope P5).

---

## 8. Explainability

- SHAP values for the best tree model (TreeExplainer — fast, exact) *(Core)*.
- For the network, use **DeepExplainer on the PyTorch twin** — it has the same architecture and
  weights-matched behaviour as the NumPy version (§6.3), so its explanations stand in for both.
  KernelExplainer on the NumPy model only on a small sample (≤100 rows), as a cross-check *(stretch)*.
- Per forecast: top factors pushing up/down, in plain words ("Low wind speed raised the forecast by ~8 µg/m³").
- Global: feature-importance summary plot for the report.

---

## 9. Web app (Streamlit)

- Pick a sensor location (dropdown + map).
- Show: forecast value, band (colour-coded), advice, top SHAP reasons, rules that fired.
- Chart: recent readings + forecast.
- Model comparison page: metrics table and charts for all models.
- Optional profile toggles (sensitive group, outdoor plans) — kept only in the session.
- **Network handling (A12):** cache the last fetched data; if a live fetch fails, show cached data with a "last updated" time and a warning.

---

## 10. Ethics, privacy, fairness

- **Privacy:** sensor and weather data contain no personal data. App profile toggles are not stored or logged.
- **Fairness:** check error per sensor/neighbourhood. Report whether some areas (e.g. Orangi vs AKU) are forecast worse, and why (data gaps, sensor quality).
- **Transparency:** SHAP + rule traces on every output; limitations shown in the app.
- **Accountability:** clear disclaimer that advice is not medical/official guidance; data sources and model versions shown.

---

## 11. Assumptions, constraints and risks

| Item | Note |
|---|---|
| Sensor data access | Urban Unit data may need a request or scraping. **Biggest risk — confirm first.** Fallback: OpenAQ / PurpleAir if Karachi sensors are listed there. |
| Sensor operator | ✅ Confirmed: AirGradient sensors listed under contributor "The Urban Unit" (part of its 160-monitor national network, funded by EPIC's Air Quality Fund). |
| Course reuse | One project for PAI + AIES CCP + AIES Lab — **approved** (21 Oct 2026). Still cite the shared project in each report and write each report fresh. |
| Low-cost sensor accuracy | Readings are noisy and humidity-biased; document the correction. |
| Short history | If the history is short, prefer simpler models; one large dataset is fine for the Lab (L1). |
| External factors | Dust storms, festivals, crop burning, traffic changes — not fully captured by features. |
| Network | Live updates need internet; app falls back to cache. |
| Group work | Projects are done in groups — the Lab is marked per student, so each member must be able to explain every part. See §11.1. |

### 11.1 Group and ownership

Filled in at the end of the project (decided 21 Oct 2026). Groups may differ between PAI and AIES.

| Member | Roll no. | Course(s) | Owns |
|---|---|---|---|
| Bilal | TBD | PAI, AIES, AIES Lab | TBD |
| TBD | | | |

---

## 12. Deliverables checklist

| Deliverable | PAI | AIES CCP | AIES Lab |
|---|---|---|---|
| Python code (preprocessing → models → evaluation → improvement) | ✅ | ✅ | ✅ |
| NumPy-from-scratch network | ✅ | (counts as an approach) | (counts as an approach) |
| Classification + rule-based layer | optional | ✅ | ✅ |
| Streamlit app | ✅ | ✅ | ✅ |
| Technical report (≥10 pages, methodology diagram) | per PAI brief | ✅ | — |
| Live demo / viva | likely | likely | ✅ |

### 12.1 Report outline (reuse across courses, adjust emphasis)

1. Title, group, abstract
2. Introduction — problem, impact, objectives
3. Data — sources table, description, pre-processing
4. Methodology — **pipeline diagram**, features, models, rule base
5. Implementation — tools, code structure, NumPy network details
6. Results — metrics tables, charts, NumPy vs PyTorch, classification results
7. Explainability — SHAP findings
8. Ethics, privacy and fairness
9. Challenges and limitations
10. Conclusion and future work
11. References

Emphasis (see §1 framing table): **PAI** → the from-scratch network and comparison;
results section leads with regression metrics. **AIES** → the choice of AI techniques,
baseline → improvement, the **band classification** (lead the results section with macro-F1
and confusion matrices), and the expert-system layer. Write each report fresh from the shared
results — don't copy paragraphs between them.

---

## 13. Tech stack and project layout

- Python 3.12 (managed by `uv`; `pyproject.toml` + `uv.lock`, with an exported `requirements.txt` for pip), NumPy, pandas, scikit-learn, PyTorch (CPU), SHAP, Streamlit, matplotlib/plotly, requests.
- Tests: **pytest** + pytest-cov, target **80% coverage** for `src/`.
- Files 200–400 lines typical, 800 max; split by feature.

```
Karachi Air Quality Forecaster/
├── PROJECT_SPEC.md          ← this file
├── README.md                ← how to install and run
├── docs/
│   ├── briefs/              ← original course briefs
│   └── report/              ← report drafts, diagrams
├── data/
│   ├── raw/                 ← downloaded, never edited
│   └── processed/           ← cleaned, feature tables
├── src/
│   ├── config.py            ← paths, timezone, random seed
│   ├── data/                ← download, clean, features
│   ├── models/
│   │   ├── baselines/       ← persistence, linear, RF/GBM
│   │   ├── numpy_nn/        ← from-scratch network
│   │   └── torch_nn/        ← PyTorch version
│   ├── classify/            ← AQI band classifier
│   ├── expert/              ← rule base + inference + explanations
│   ├── explain/             ← SHAP
│   └── evaluate/            ← metrics, comparison tables
├── app/                     ← Streamlit app
├── models/                  ← trained models (not in git)
├── results/                 ← metrics tables, figures
├── notebooks/               ← exploration only
└── tests/
```

---

## 14. Build order

Detailed phase-by-phase tasks and progress tracking: [`ROADMAP.md`](ROADMAP.md).

0. ~~Get instructor approval~~ (approved 21 Oct 2026); ~~confirm the sensor operator~~ (done).
1. Confirm data access for D1–D3; download and describe them.
2. Clean data, build 24-h mean targets and features, time-based split.
3. Persistence + linear baseline → first results; band counts per class.
4. Random Forest (regression + direct classifier) + logistic regression baseline.
5. NumPy network (with gradient check) → PyTorch version → compare.
6. Tuning + simple average ensemble.
7. Band classification: regress-then-band vs direct; imbalance handling.
8. Expert-system rules + explanation.
9. SHAP (TreeExplainer, then DeepExplainer on PyTorch).
10. Streamlit app.
11. Report(s) and presentation.
12. **Only then** stretch items: GBM, stacking, NumPy softmax classifier, hourly horizons,
    KernelExplainer cross-check.

Deadlines: **as soon as possible** for all three courses (no fixed dates). Build **Core only**;
stretch items (Phase 9) only if time is left after the reports are drafted.

---

## 15. Open questions

- [ ] Sensor history: OpenAQ free API key (Bilal to register) → check which sensors are mirrored and from when. Fallback: ask The Urban Unit for an AirGradient read token / CSV export.
- [x] Operator confirmed — The Urban Unit (AirGradient sensors).
- [x] Forecast horizon — **decided:** next-day 24-h mean (hourly = stretch, chart only).
- [x] One project for all three courses — **decided** (Bilal, 21 Oct 2026).
- [x] Instructor approval for the shared project — **approved** (21 Oct 2026).
- [x] PAI brief — none exists; the overview is the reference. Deadline: ASAP.
- [x] Deadlines — ASAP for all three. Group members — added at the end (§11.1).
