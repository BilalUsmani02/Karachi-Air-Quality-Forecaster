---
title: "Karachi Air-Quality Forecaster: Day-Ahead AQI Band Classification with a Rule-Based Expert System for Explainable Health Advice"
subtitle: "AIES CCP (CT-361) — Technical Report"
author: "Group members: [to be added]"
date: "November 2026"
---

# Abstract {-}

Karachi's air often carries fine particulate matter (PM2.5) well above international guidelines, yet residents only see current readings, not what tomorrow will bring. This report describes a hybrid AI and expert system that classifies the next day's air quality into one of five US EPA health bands at each of 14 low-cost sensor sites, and turns that band into explained health advice. A supervised model forecasts the 24-hour average PM2.5 for the day ahead, issued at 00:00 Pakistan time, and the forecast is then sorted into a band. We compared this "regress then band" route with classifiers that predict the band directly, starting from simple baselines (persistence, Ridge regression, logistic regression) and improving through Random Forest, Gradient Boosting, a neural network written in NumPy, tuning and ensembling. On eight rolling-origin cross-validation folds (1,458 sensor-days) the chosen ensemble reaches 82.6% band accuracy and misses 8.0% of Unhealthy-or-worse days, against 18.0% for persistence. A 22-rule forward-chaining expert system, SHAP explanations and a Streamlit app complete the system.

# Introduction

## Problem statement

**The issue.** Karachi regularly records PM2.5 levels far above the World Health Organization (WHO) guideline [3]. Public sources show the air quality *now*. Nobody tells a parent at midnight whether the next day will be safe for a school sports match. Across the 4,027 sensor-days we labelled, 28% fell in the Unhealthy band or worse (Section 4.5).

**Real-world impact.** Fine particles reach deep into the lungs and are linked with respiratory and heart disease. Children, older adults and people with heart or lung conditions are hit hardest; the US EPA calls them "sensitive groups" [1][2]. A forecast issued at the start of the day lets these people plan: move exercise indoors, reschedule an outing, keep windows shut. Without one, they learn the air was bad only after they have breathed it.

**The AI and expert-system approach.** We built a hybrid system. Supervised learning predicts the day's air-quality band at each sensor site by two routes: (1) a regression model forecasts the 24-hour mean PM2.5, which is then looked up in the EPA band table, and (2) a classifier predicts the band directly. Both routes start from a baseline and improve through stronger models, tuning and an ensemble. A rule-based expert system then reasons forward from the band and a short user profile (sensitive group? outdoor plans?) to one piece of advice quoted from EPA guidance. SHAP values and rule traces explain every output.

## Objectives

1. Classify each day's 24-hour PM2.5 band, issued at 00:00 PKT, at every reporting sensor site.
2. Beat honest baselines on the measures that matter for health: band accuracy, macro-F1 and, above all, the share of truly unhealthy days that the forecast misses.
3. Turn each forecast band into personalised, sourced and explained advice through an expert system.
4. Present everything in an interface that works even without an internet connection.
5. Report the system's weaknesses openly, including where it is no better than a naive forecast.

## Contributions

- A cleaned, leak-free daily dataset of 4,027 labelled sensor-days from 14 Karachi sensors, joined with weather and atmospheric-model data.
- A side-by-side comparison of nine routes to the band, from persistence to a two-model ensemble, with date-block bootstrap confidence intervals on every difference.
- An explicit, written model-selection rule that ranks routes by missed unhealthy days before accuracy.
- A 22-rule forward-chaining expert system with conflict resolution, a safety margin and a plain-English explanation facility.
- A Streamlit app with live, cached and demo modes, SHAP reasons and health alerts.

# Problem description

## Domain

The domain is environmental monitoring, applied to public health. The quantity of interest is PM2.5: the mass of airborne particles smaller than 2.5 micrometres, in micrograms per cubic metre (µg/m³). Health agencies do not ask the public to interpret raw concentrations. Instead, the US EPA Air Quality Index (AQI) groups the 24-hour average into named, colour-coded bands, each with its own advice [1]. We use the 2024 revision of those breakpoints (Table 1), which lowered the "Good" cut-off to 9.0 µg/m³ [1][5].

| Band | 24-hour PM2.5 (µg/m³) | Index used here |
|---|---|---|
| Good | 0.0 – 9.0 | 0 |
| Moderate | 9.1 – 35.4 | 1 |
| Unhealthy for Sensitive Groups (USG) | 35.5 – 55.4 | 2 |
| Unhealthy | 55.5 – 125.4 | 3 |
| Very Unhealthy | 125.5 – 225.4 | 4 |
| Hazardous | ≥ 225.5 | merged into 4 |

: Table 1. US EPA 2024 PM2.5 breakpoints [1]. Hazardous never occurred in our data and is merged into Very Unhealthy for the classifier.

## Users and use case

The main users are Karachi residents, especially people in sensitive groups and those who care for them (parents, teachers, sports coaches). A typical use: just after midnight, a user opens the app, picks the nearest sensor site on a map, and sees "Tomorrow: Unhealthy for Sensitive Groups". They tick "I'm in a sensitive group" and "I have outdoor plans", and the system replies with one piece of EPA advice, a note if the forecast sits close to a worse band, the top three reasons behind the forecast, and a disclaimer.

## Inputs and outputs

**Inputs at 00:00 PKT on day D**, for each sensor: the sensor's own hourly PM2.5 history up to midnight, the city-wide average of the previous day, the previous day's weather and modelled pollution, the forecast day's weather and modelled pollution, calendar values, and the site's identity. The user adds two yes/no facts.

**Outputs:** (1) the forecast 24-hour mean PM2.5 in µg/m³; (2) the band, the main output for this course; (3) one piece of advice with notes and a disclaimer; (4) the fired rules and the SHAP reasons.

## Why this is a classification problem

The decision a resident faces is categorical: is tomorrow fine, or should I change plans? EPA advice is written per band, not per microgram. A forecast of 54 µg/m³ and one of 56 µg/m³ are almost equal as numbers but fall in different bands with different advice. So the measure of success is whether the band is right, and in particular whether truly unhealthy days are caught. The regression step is a means to that end, not the goal. That is also why we compare it against classifiers that skip the number entirely.

The problem has no obvious solution. The classes are ordered, unevenly sized and drift with the seasons; the data comes from noisy low-cost sensors that drop out for weeks; and the history covers only about 16 months. Choosing a technique therefore needed both analysis (which route, which model, which metric) and domain knowledge (EPA averaging rules, the physics of smog, what a missed bad day costs).

# Data and knowledge sources

## Sources

The brief asks for at least two datasets or knowledge sources. We used five (Table 2). Two of them are fully independent of each other: the measured sensor data (D1) and the health-guidance knowledge base (D4). D2 and D3 come from one provider, Open-Meteo, but from two different underlying models (a weather reanalysis and an atmospheric-chemistry model).

| ID | Source | Type | Size and period | Licence | Relevance | How we use it |
|---|---|---|---|---|---|---|
| D1 | The Urban Unit's AirGradient PM2.5 sensors, via the OpenAQ API v3 [6] | Hourly time series, 14 sites | 102,110 hourly rows; Jun 2025 – Nov 2026 | CC BY 4.0 | The quantity we forecast | Target labels, lag and rolling features, evaluation |
| D2 | Open-Meteo Historical Weather API (ERA5 reanalysis) [7][9] | Hourly gridded time series | 186,816 rows; May 2025 – Nov 2026 | CC BY 4.0 | Wind, rain, temperature and mixing height control how pollution builds up | Previous-day and forecast-day weather features |
| D3 | Open-Meteo Air Quality API (Copernicus CAMS) [8][10] | Hourly gridded model output | 186,816 rows; May 2025 – Nov 2026 | CC BY 4.0 | An independent physical model of PM2.5, NO₂, dust and aerosol depth | Previous-day and forecast-day pollution features |
| D4 | EPA AQI Technical Assistance Document [1], EPA Air Quality Guide for Particle Pollution [2], WHO 2021 guidelines [3], Pakistan NEQS [4] | Text documents | 22 rules | Public documents | Authoritative advice per band | The expert system's rule base, band breakpoints |
| D5 | US Consulate Karachi reference monitor, via OpenAQ [6] | Hourly time series, 1 site | 46,937 rows; 2019 – 2025 | CC BY 4.0 | A reference-grade view of long-term seasons and band frequencies | Context only; it ends before D1 starts |

: Table 2. Data and knowledge sources.

Weather and CAMS data start a month before the first sensor reading so that lag features exist from day one. Every download is logged in a manifest with its endpoint, coordinates and row count.

## Sensors

Fourteen Urban Unit sensors within Karachi had at least 150 complete days and were kept; four others with too little data were dropped. About nine report at any one time. Completeness ranges from 85.6% of expected hours (S1, Climate Action Center) to 32.5% (S14, Aga Khan University IED). The longest single gap is 5,533 hours, or about 230 days, at S13. The sensors agree closely with each other: the daily-mean correlation between any two sites ranges from 0.911 to 0.992 (mean 0.959). This tells us that much of the day-to-day signal is city-wide, which matters when a site goes offline.

## Diverse data types

The system handles six kinds of data, each in its own way (Table 3).

| Data type | Example | How it is handled |
|---|---|---|
| Numeric time series | PM2.5, temperature, humidity | Lags, rolling means and standard deviations, median imputation, standard scaling |
| Categorical | Sensor site | One-hot encoding (14 columns) |
| Cyclic | Day of year, day of week, wind direction | Sine and cosine pairs; wind as east-west (u) and north-south (v) components |
| Gridded model output | CAMS PM2.5, boundary-layer height | Nearest grid cell to each sensor |
| Text documents | EPA and WHO guidance | Hand-encoded as if-then rules with a source on each |
| Live user input | Sensitive group, outdoor plans | Session-only facts for the rule engine, never stored |

: Table 3. Data types and their handling.

## External factors

Several outside factors affect performance, and the design responds to each.

- **Weather** is the largest. In our exploration, daily PM2.5 fell as temperature rose (r = −0.652), as wind speed rose (r = −0.627), as humidity rose (r = −0.625) and as the mixing height rose (r = −0.483). Weather enters the model directly.
- **Season.** Winter smog dominates January (80% of sensor-days Unhealthy or worse in that fold), while the monsoon is clean. This drove the choice of evaluation design (Section 5.2).
- **Events the features cannot see**, such as dust storms, festivals, crop burning or a fire, will produce misses. We say so in the app.
- **The atmospheric model's own bias.** CAMS tracks the city average well (r = 0.733) but reads 14.5 µg/m³ lower than the sensors on average. CAMS also returns one value for the whole city, so it acts as a city-wide feature, not a site-specific one.
- **Dynamic input in live use.** The live app relies on weather forecasts and on the sensor network being online (Section 9.2).

# Pre-processing

## Cleaning rules

Raw readings were aligned to one hourly grid in the Asia/Karachi timezone. Cleaning used only past and present values, so the same code runs on live data. The rules, applied per sensor in this order, were:

1. Negative values are rejected. None occurred.
2. Spikes are rejected: any value above 500 µg/m³, or above five times the median of the previous 24 hours when it also exceeds 100 µg/m³. This removed **20** hourly values.
3. A value repeated for 24 or more hours (a flat-lined sensor) is rejected. None occurred.
4. Gaps of three hours or fewer between valid values are filled by straight-line interpolation. This filled **995** hours. Longer gaps stay missing.
5. Zeros are kept as real readings.
6. No humidity correction is applied. Low-cost sensors can over-read in humid air, so we checked: mean PM2.5 was actually *lower* above 80% humidity (35.6 µg/m³) than below it (50.2 µg/m³). We found no sign of humidity inflation, though monsoon rain probably confounds this.

The cleaned grid has 174,511 sensor-hours: 102,019 valid readings, 995 interpolated, 20 spikes and 71,477 missing (the missing count includes the month of weather-only lead-in before each sensor starts).

## Missing data and the daily table

Each row of the modelling table is one sensor on one day D. The target is the mean of D's 24 cleaned hours, and it is only kept if at least 18 of the 24 hours are valid. This is the EPA-style completeness rule: a "daily average" built from six hours would not be a daily average. Of 7,276 candidate sensor-days, 3,149 were dropped for an incomplete target and 100 because the previous day (the persistence input) was missing, leaving **4,027** labelled rows. Any other missing feature is filled with the median of the *training* rows only, inside each fold.

## The leakage rule

Every history feature for day D uses only hours before D 00:00, the moment the forecast is issued. Only the forecast-day weather and CAMS columns (prefix `fc_`) look inside day D, because a real forecast at midnight would have a weather forecast for the day. In training and testing we feed these columns the *recorded* weather for day D. This is called **perfect prognosis**: the model is evaluated as if the weather forecast were perfect. The live app uses Open-Meteo's actual forecast instead, so live accuracy will be somewhat lower than the numbers in this report. We return to this in Section 11.

## Feature groups

The model sees 58 features in six groups:

- **PM2.5 history (11):** readings 1, 2, 3, 6, 12 and 24 hours before midnight; means over 72 and 168 hours; the standard deviation over 24 hours; the previous day's mean; and the city-wide previous-day mean. (A 24-hour mean duplicated the previous-day mean exactly and was removed.)
- **Calendar (5):** day of year and day of week as sine/cosine pairs, and month.
- **Previous-day weather and CAMS (14):** temperature, humidity, dew point, pressure, rainfall, wind speed and components, mixing height, and CAMS PM2.5, PM10, NO₂, dust and aerosol depth.
- **Forecast-day weather and CAMS (14):** the same variables for day D.
- **Site (14):** one-hot sensor columns.

All non-site features are standard-scaled with statistics from the training rows of each fold.

## Class distribution and imbalance

Table 4 shows the band counts. The classes are clearly unbalanced: Moderate outnumbers Good by about 19 to 1 and Very Unhealthy by about 21 to 1, and Hazardous never occurs.

| Band | All labelled rows | Development period (to 31 Aug 2026) | Pooled CV test rows | Final test |
|---|---|---|---|---|
| Good | 110 | 109 | 65 | 1 |
| Moderate | 2,056 | 1,831 | 762 | 225 |
| USG | 734 | 606 | 292 | 128 |
| Unhealthy | 1,029 | 941 | 314 | 88 |
| Very Unhealthy | 98 | 98 | 25 | 0 |
| Hazardous | 0 | 0 | 0 | 0 |
| **Total** | **4,027** | **3,585** | **1,458** | **442** |

: Table 4. Band counts in the labelled data.

We handled the imbalance in four ways:

1. **Merge empty classes.** Hazardous has zero examples, so it is merged into Very Unhealthy. Good has 109 development rows, well above our 30-row threshold for merging, so it stays its own class.
2. **Class weights.** The direct classifiers (logistic regression, Random Forest, Gradient Boosting) are trained with `class_weight="balanced"`, which weights each row inversely to its class frequency, so a rare-class mistake costs more. For the Random Forest the tuner chose between "balanced" and "balanced_subsample".
3. **Regress then band.** The regression route sidesteps class counts: it learns a continuous number, and the band boundaries are applied afterwards.
4. **Metrics that respect rare classes.** We report macro-F1, which averages F1 over the bands so a rare band counts as much as a common one, and per-class recall, alongside accuracy.

We did not resample (for example by oversampling rare days). With time-series data, copying rare winter days would also copy their neighbours' weather, and the class weights already did that job.

# Methodology

## Pipeline

Figure 1 shows the whole pipeline. Data from the five sources is cleaned, turned into one row per sensor-day, and split by time. Models are trained and tuned on rolling folds, and the chosen ensemble's forecast is turned into a band. The expert system and SHAP explainer then work on that band and forecast, and the Streamlit app shows the result. A live, cached or demo snapshot feeds the app each time it runs.

![Figure 1. End-to-end pipeline. Orange boxes are the parts this course emphasises: band classification, the expert system and the SHAP reasons.](docs/report/figures/pipeline.png){width=100%}

## Evaluation design

Air quality is a time series, so a random train/test split would leak the future into training: a model could learn from 15 January and be tested on 14 January. We split strictly by time.

- **Final test:** 1 September – 8 November 2026, 442 sensor-days over 69 dates. Scored **once**, after every modelling choice was frozen. The scoring script refuses a second run.
- **Model selection:** eight rolling-origin folds over the earlier data. Each fold trains on every day before a 30-day test block, leaves a one-day gap, and tests on the block. The blocks run from 1 January to 28 August 2026. Pooled together, they give **1,458 labelled sensor-days over 240 dates**. All tuning and model choices used these folds only.

We first planned a single validation block for June–August 2026. It turned out to contain **zero** Unhealthy or Very Unhealthy days, so it could not tune a band classifier at all. The rolling folds cover winter smog, the spring transition and the monsoon, so every band appears in the pooled test rows.

**Confidence intervals.** Every headline number carries a 95% bootstrap interval from 2,000 resamples (seed 42). We resample whole *dates*, not single rows, because all sensors on one day share the same weather and smoke and are not independent. Two models are compared on the same resampled dates (a paired difference). If the interval of a difference contains zero, we treat the gap as noise.

## Metrics

For the classification task:

- **Accuracy:** share of sensor-days given the right band.
- **Macro-F1:** the F1 score (the balance of precision and recall) averaged over the five bands with equal weight.
- **Per-class recall:** of the days truly in a band, the share forecast in that band.
- **Missed Unhealthy-or-worse rate:** of the days truly Unhealthy or Very Unhealthy, the share forecast as a cleaner band. This is the health-critical error.
- **False-alarm rate:** of the cleaner days, the share forecast as Unhealthy or worse.

For the regression step we report mean absolute error (MAE), root mean squared error (RMSE), R² and **skill**, the percentage reduction in MAE compared with persistence.

## Baselines

- **Persistence:** tomorrow's average equals today's average. It needs no learning and is surprisingly hard to beat for air quality, since smog does not clear overnight.
- **Ridge regression → band:** a linear model with an L2 penalty; the official regression baseline.
- **Logistic regression (direct):** a multinomial linear classifier with balanced class weights; the official classification baseline.

## Improvement approaches

On top of the baselines we tried four model families, each for a clear reason:

- **Random Forest** (regression and direct classifier): an average of many decision trees, each grown on a bootstrap sample [12][16]. It handles non-linear effects such as "low wind *and* low mixing height" without hand-made interaction terms.
- **Gradient Boosting** (scikit-learn's histogram-based version, regression and direct classifier): trees added one at a time, each correcting the previous ones' errors. Usually the strongest model family on tabular data.
- **A neural network written from scratch in NumPy** [15]: one hidden layer of 32 ReLU units, trained with the Adam optimiser [17], L2 penalty and early stopping. We wrote the forward pass and backpropagation by hand. A numerical gradient check across 36 parameter blocks gave a largest relative error of 7.8 × 10⁻¹⁰, and an identical PyTorch [14] copy started from the same weights matched its predictions to within 1.4 × 10⁻¹⁴ µg/m³. The network is one approach among several here; its derivation is in the project's PAI report.
- **Direct classifiers** (logistic regression, Random Forest, Gradient Boosting) that predict the band without a number.

## Tuning

Each family was tuned on the pooled CV folds with a staged search: a coarse scan per hyperparameter, then a finer scan around the best value. Regressors were tuned on MAE; classifiers on macro-F1. Table 5 shows the winners.

| Model | CV MAE default | CV MAE tuned | Key settings |
|---|---|---|---|
| Persistence | 8.70 | — | — |
| Ridge | 7.68 | 7.22 | alpha 150 |
| Random Forest | 6.79 | 6.54 | 500 trees, depth 15, min leaf 5, half the features per split |
| Gradient Boosting | — | 6.30 | 100 rounds, 31 leaves, L2 0.1 |
| NumPy network | 11.21 | 6.63 | 32 units, L2 0.1, predicts change from previous day, no calendar features |

: Table 5. Tuning results, pooled CV (1,458 sensor-days), MAE in µg/m³.

The untuned network was worse than persistence. Two causes stood out. First, the data starts in June 2025, so the spring folds tested calendar dates the network had never seen; it read them as winter and over-forecast by 21.9 µg/m³ on average in one fold. Dropping the three calendar features cut that bias to −2.5. Second, it overfitted (training error near 3 µg/m³, held-out error 15–17). A smaller network with a heavier weight penalty, trained to predict the *change* from a simple previous-day line, fixed most of this. The result was stable across random seeds (MAE 6.57, 6.63 and 6.70).

## Ensemble

The regressors make partly different mistakes (their predictions correlate at 0.97, not 1), so averaging them helps. An equal-weight average of Gradient Boosting and the NumPy network reached a CV MAE of 6.16 µg/m³, within 0.03 of a four-model average (6.13) and with the best band macro-F1 of any combination (0.674 against 0.642). We kept equal weights on purpose: fitting weights on the same folds would overfit them.

## Model-selection rule

We wrote the rule down before comparing band routes, in priority order:

0. **Eligibility.** A regress-then-band route qualifies only if its regressor's CV MAE is within about 10% of the best (6.16, so up to about 6.8). A model that is clearly worse at the number should not win by leaning towards the worse band. Eligible: the ensemble (6.16), Gradient Boosting (6.30), Random Forest (6.54) and the NumPy network (6.63). Ridge (7.22) and persistence (8.70) stay as baselines. Direct classifiers have no MAE and go straight to rule 1.
1. **Fewest missed Unhealthy-or-worse days.** A missed bad day costs more than a false alarm: people who should have stayed indoors do not. A route is ruled out only if it misses *clearly* more, meaning the bootstrap interval of the difference excludes zero.
2. **Then accuracy.**
3. **Then macro-F1.**

This is the decision we, as designers, had to make: macro-F1 is the textbook measure for imbalanced classes, but for a health tool the cost of errors is not symmetric. Section 7 shows where this choice matters.

# Expert system

The classifier decides *what the air will be like*. The expert system decides *what to tell someone about it*, and shows why. It lives in `src/expert/`.

## Knowledge acquisition

The knowledge came from published documents, not from our own opinions:

- The **EPA AQI Technical Assistance Document** [1] gives the band breakpoints (we checked all six against its breakpoint table) and the official list of sensitive groups.
- The **EPA Air Quality Guide for Particle Pollution** [2] gives plain-language "what should I do" advice per band, separately for sensitive groups and everyone else. Most rules quote it directly.
- The **WHO 2021 guidelines** [3] give stricter health-based limits (5 µg/m³ annual, 15 µg/m³ for 24 hours). WHO has no per-band advice, so it informs the limitations, not the rules.
- **Pakistan's National Environmental Quality Standards** [4] set 35 µg/m³ as the 24-hour PM2.5 limit. We cite it for context; the source document is headed as a draft and we could not confirm its approval date.

Two safeguards shaped the acquisition. Advice is **quoted, not invented**: every advice rule carries a `source` field. And the profile asks only about **broad groups** ("sensitive group: yes/no"), never a named condition. EPA's guide has an extra line for people with asthma; we left it out on purpose because it would require asking about a diagnosis.

## Knowledge representation

Each rule is a frozen data record, not a function: an ID, a list of conditions, a kind, the advice text, a severity from 1 to 5, a source and a "because" sentence. Each condition is a plain `(fact, operator, value)` triple. Keeping rules as data rather than code is what lets the explanation facility print exactly what was checked.

The 22 rules fall into four groups. R01–R17 are candidates for the single main piece of advice, one for each combination of band, sensitive status and outdoor plans. R18–R20 add notes when the trend is rising or falling. R21 is a meta-rule that applies the safety margin. R22 attaches the disclaimer to every answer. Table 6 shows a representative selection.

| ID | IF | THEN | Severity | Source |
|---|---|---|---|---|
| R01 | advice band = Good | "It's a great day to be active outside." | 1 | [2] |
| R03 | advice band = Moderate, sensitive | "Consider making outdoor activities shorter and less intense…" | 2 | [1][2] |
| R06 | advice band = USG, sensitive | "Make outdoor activities shorter and less intense. It's OK to be active outdoors, but take more breaks…" | 3 | [2] |
| R07 | advice band = USG, sensitive, outdoor plans | "Make your planned activity shorter and less intense, take more breaks…" | 3 | [2] |
| R10 | advice band = Unhealthy, sensitive | "Avoid long or intense outdoor activities. Consider rescheduling or moving activities indoors." | 4 | [2] |
| R14 | advice band = Very Unhealthy, sensitive | "Avoid all physical activity outdoors…" | 5 | [2] |
| R15 | advice band ∈ {Very Unhealthy, Hazardous}, outdoor plans | "Cancel or move your planned outdoor activity indoors." | 5 | [2] |
| R18 | trend rising, advice band = USG | append: "Conditions are forecast to worsen… plan ahead." | 3 | design rule |
| R21 | 1 ≤ band ≤ 4, distance to next band ≤ 5 µg/m³ | meta: advice band = band + 1 | 5 | safeguard |
| R22 | always | append the disclaimer | — | safeguard |

: Table 6. Representative rules from the 22-rule base.

The working facts are the forecast PM2.5, the previous day's PM2.5 (optional, for the trend), and the two profile booleans. The band is computed from the forecast with the EPA table.

## Inference by forward chaining

The engine uses **forward chaining**: it starts from known facts, fires every rule whose conditions match, adds the conclusions to its working memory, and repeats until a round adds nothing new (a *fixpoint*). Before any rule runs, the engine seeds three derived facts: the band, the trend (rising or falling if the change exceeds 5 µg/m³ or 10% of the previous day, whichever is larger), and the distance from the forecast up to the next band's lower boundary.

Here is the real trace for a forecast of 33.0 µg/m³ after a day at 25.0 µg/m³, for a sensitive user with outdoor plans, produced by `uv run python -m src.expert --pm25 33 --today 25 --sensitive --outdoor`:

| Round | Rules that fire | Working memory gains |
|---|---|---|
| Seed | — | band = Moderate (1); trend = rising (+8.0 µg/m³); distance to USG = 2.5 |
| 1 | R21 (margin), R22 (disclaimer) | advice band = USG (2); near boundary = true |
| 2 | R06, R07, R18 | candidate advice from R06 and R07; trend note from R18 |
| 3 | none | fixpoint reached, loop stops |

: Table 7. Forward-chaining trace for forecast 33.0, previous day 25.0, sensitive, outdoor plans.

The chain is genuine: R01–R20 condition on the *advice band*, which does not exist until R21 has been evaluated in round 1. When R21 does not match, the engine sets the advice band equal to the band and continues. A cap of ten rounds guards against a rule base that never settles; this one always settles in three.

## Conflict resolution

In round 2 both R06 and R07 matched. Only one can be the main advice, so the engine ranks candidates:

1. **Most specific wins**: the rule with more conditions. R07 (three conditions) beats R06 (two).
2. **Then highest severity**, so the safest advice wins a tie. For a non-sensitive person with outdoor plans on a Very Unhealthy day, R13 (severity 4) loses to R15 (severity 5).
3. **Then the rule that checks sensitive status.** For the two worst bands with outdoor plans, R15 ties with R14, R16 or R17 on both specificity and severity. EPA organises its advice mainly by sensitive-group status, so the rule that checks it wins. This was a real gap in our first draft of the rule table, and we resolved it with a documented choice rather than leaving it to sort order.

A test runs every one of the 24 band × sensitive × outdoor combinations (including Hazardous, since the expert layer reasons on the raw EPA band) and confirms each yields exactly one winner with a source and the disclaimer.

## Safety margin

During model selection we tried adding 5 µg/m³ to every forecast before banding. It cut missed Unhealthy days to 4.1%, but raised false alarms to 9.1% and meant "Good" was never forecast. So the margin is **not** applied to the band shown. It lives in the advice layer instead, as R21: when the forecast is within 5 µg/m³ below the next band's boundary, the *advice* follows the worse band and the app says so. Good never bumps into Moderate, so "a great day to be active outside" can still appear. For example, 122.0 µg/m³ (Unhealthy, 3.5 below Very Unhealthy) triggers the margin, while 120.0 (5.4 below) does not.

## Explanation facility

For every answer the engine can print a "You see this because…" trace: the input facts, every rule that fired with the facts that matched its conditions and its source, whether the margin applied, and why the winner won. For the example above it reports, among other lines:

```
Winning advice rule: R07 (usg_sensitive_outdoor).
  Why it won: R07 beat R06: 3 matching conditions vs 2 - the most-specific rule wins.
  Advice shown: "Make your planned activity shorter and less intense, take more
  breaks, and watch for symptoms such as coughing or shortness of breath."
Additional notes:
  - Near a boundary - showing the more cautious advice for Unhealthy for Sensitive Groups.
  - Conditions are forecast to worsen compared with the day before - plan ahead.
```

The app shows the same trace in an expandable panel under the advice card.

## Limitations of the rule base

- The rules are EPA's general public guidance for broad groups. They are not a model of any one person's risk, and they are not medical advice.
- WHO's guideline is far stricter than EPA's "Good" cut-off (5 against 9.0 µg/m³ for annual exposure). Karachi will exceed the WHO level on most days even when this system says "Good".
- The trend threshold (5 µg/m³ or 10%) and the 5 µg/m³ margin are fixed design values. They have not been tuned against user feedback or a cost study of the advice itself.
- The trend and margin wording (R18–R21) is our own design, not a quotation, and is labelled as such.

# Results

## Band classification on cross-validation

Table 8 compares every route to the band on the pooled CV folds. All five bands occur in these rows, so macro-F1 is over five classes.

| Route | Accuracy | Macro-F1 | Good recall | VU recall | Missed U-or-worse | False alarms |
|---|---|---|---|---|---|---|
| Persistence → band | 78.9% | **0.678** | 72.3% | **40.0%** | 18.0% | 5.4% |
| Logistic regression (direct, baseline) | 73.1% | 0.589 | 76.9% | 20.0% | 23.9% | 4.0% |
| Ridge → band (baseline, tuned) | 79.6% | 0.582 | 61.5% | 0.0% | **5.6%** | 8.7% |
| Random Forest classifier (direct) | 78.3% | 0.651 | **78.5%** | 28.0% | 15.0% | **3.6%** |
| Gradient Boosting classifier (direct) | 82.0% | 0.664 | 61.5% | 20.0% | 10.0% | 6.3% |
| Random Forest → band | 79.6% | 0.609 | 47.7% | 8.0% | 6.2% | 7.4% |
| Gradient Boosting → band | 80.7% | 0.645 | 53.8% | 16.0% | 7.1% | 6.6% |
| NumPy network → band | 80.7% | 0.659 | 52.3% | 20.0% | 13.0% | 4.7% |
| **Ensemble (GB + NumPy) → band** | **82.6%** | 0.674 | 56.9% | 16.0% | 8.0% | 5.6% |

: Table 8. Band results, pooled CV (1,458 sensor-days, 240 dates). VU = Very Unhealthy; U-or-worse = Unhealthy or Very Unhealthy.

Table 9 gives the bootstrap intervals for the chosen ensemble and its differences from the main competitors.

| Comparison | Accuracy | Macro-F1 | Missed U-or-worse | False alarms |
|---|---|---|---|---|
| Ensemble | 0.826 [0.798, 0.855] | 0.674 [0.613, 0.747] | 0.080 [0.043, 0.127] | 0.056 [0.032, 0.084] |
| Ensemble − persistence | **+0.038 [+0.008, +0.069]** | −0.004 [−0.064, +0.083] | **−0.100 [−0.174, −0.035]** | +0.003 [−0.016, +0.022] |
| Ensemble − logistic regression | **+0.095 [+0.065, +0.127]** | **+0.085 [+0.017, +0.183]** | **−0.159 [−0.253, −0.076]** | +0.016 [−0.004, +0.038] |
| Ensemble − Ridge → band | **+0.031 [+0.014, +0.049]** | **+0.091 [+0.034, +0.161]** | +0.024 [+0.003, +0.050] | **−0.030 [−0.052, −0.013]** |
| Ensemble − GB classifier | +0.007 [−0.012, +0.026] | +0.009 [−0.059, +0.056] | −0.021 [−0.054, +0.008] | −0.006 [−0.027, +0.010] |
| Ensemble − RF → band | **+0.031 [+0.012, +0.050]** | **+0.065 [+0.031, +0.102]** | +0.018 [−0.013, +0.050] | −0.018 [−0.043, +0.005] |
| Ensemble − GB → band | **+0.019 [+0.006, +0.034]** | +0.029 [−0.022, +0.061] | +0.009 [−0.013, +0.030] | −0.010 [−0.026, +0.009] |
| Ensemble − NumPy → band | **+0.019 [+0.002, +0.038]** | +0.015 [−0.030, +0.074] | **−0.050 [−0.094, −0.017]** | +0.009 [0.000, +0.019] |

: Table 9. Ensemble and paired differences with 95% date-block bootstrap intervals, pooled CV. Bold differences exclude zero.

**Applying the selection rule.** Among eligible regress-then-band routes and the direct classifiers, rule 1 rules out persistence, logistic regression and the NumPy network alone, because each misses clearly more bad days than the ensemble does (Table 9; the network by 5.0 points). The Random Forest classifier misses nearly twice as many bad days as the ensemble (15.0% against 8.0%) and is also less accurate, so it loses on either rule. Random Forest → band and Gradient Boosting → band miss slightly fewer bad days than the ensemble, but those gaps sit within noise, so rule 1 cannot separate them. On rule 2 the ensemble is clearly more accurate than both. The Gradient Boosting classifier is within noise of the ensemble on every measure; the ensemble wins on rule 2 by a small margin (82.6% against 82.0%). So the production route is the ensemble, then the band.

**Baseline to improvement.** Against the official classification baseline (logistic regression), the ensemble is clearly better on every health measure: 9.5 points more accurate, 0.085 higher macro-F1, and it misses 15.9 points fewer bad days. Against the official regression baseline (Ridge → band), it is more accurate, has a 0.091 higher macro-F1 and 3.0 points fewer false alarms. Ridge does miss fewer bad days (5.6% against 8.0%), but it gets there by calling "Unhealthy" too often: it never forecasts Very Unhealthy and flags 8.7% of cleaner days. It also fails rule 0, with an MAE 17% worse.

**An honest note on macro-F1.** The ensemble does **not** beat persistence on macro-F1. The difference is −0.004, with an interval from −0.064 to +0.083: a tie. Persistence scores well on macro-F1 because it copies yesterday, so it reaches the rare extreme bands about as often as they happen. Its Very Unhealthy recall is 40% against the ensemble's 16%. The ensemble's gains are in accuracy (+3.8 points) and, most importantly, in safety: it misses 10 points fewer Unhealthy-or-worse days, at no measurable cost in false alarms. Had we picked the winner by macro-F1 alone, persistence (a model that learns nothing) would have won. This is why the selection rule puts missed bad days first.

**Interesting side result.** Tuning Ridge for MAE (alpha 150) *lowered* its band macro-F1 from 0.657 to 0.582, because the stronger penalty pulled extreme forecasts towards the middle. A better number does not always mean better bands.

## Confusion matrices

![Figure 2. Pooled CV confusion matrices (1,458 sensor-days). Rows are the true band, columns the forecast band; each cell shows the count and the share of that true band.](results/bands/confusion_cv.png){width=100%}

Figure 2 shows where the errors fall. Almost all of the ensemble's errors are one band away; it never confuses Good with Unhealthy. When it misses an Unhealthy day, it nearly always calls it USG, one band lower. Most Very Unhealthy days are called Unhealthy. This is the expected effect of averaging: the average of two models is pulled towards the middle, so extremes are under-called.

## Per-class recall

| Band | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Good | 65 | 0.949 | 0.569 | 0.712 |
| Moderate | 762 | 0.891 | 0.923 | 0.907 |
| USG | 292 | 0.682 | 0.596 | 0.636 |
| Unhealthy | 314 | 0.774 | 0.914 | 0.838 |
| Very Unhealthy | 25 | 1.000 | 0.160 | 0.276 |

: Table 10. Per-class results for the ensemble, pooled CV.

The ensemble is strong on the two biggest and most important classes: it catches 91.4% of Unhealthy days and 92.3% of Moderate days. Its two weak spots are the extremes. It finds only 57% of Good days, and only 4 of the 25 Very Unhealthy days (16%). When it does call a day Very Unhealthy, it is right every time (precision 1.000), but it says so rarely. USG is the hardest middle band, because it is narrow (20 µg/m³ wide) and sits between two large classes.

## Final test

The final test covers 1 September to 8 November 2026 and was scored once. This period holds 1 Good day, 225 Moderate, 128 USG, 88 Unhealthy and **no** Very Unhealthy days. A band that never occurs has an F1 of zero for every model and drags macro-F1 down for all of them, so we also report macro-F1 over the four bands present.

| Route | Accuracy | Macro-F1 (5) | Macro-F1 (4 present) | Missed U-or-worse | False alarms |
|---|---|---|---|---|---|
| Persistence → band | 81.7% | 0.477 | 0.596 | 22.7% | 4.5% |
| Ridge → band (baseline) | 81.9% | 0.473 | 0.591 | **14.8%** | 5.6% |
| Random Forest classifier (direct) | **84.6%** | **0.550** | **0.688** | 20.5% | **2.0%** |
| Gradient Boosting → band | 83.5% | 0.488 | 0.610 | 18.2% | 4.2% |
| NumPy network → band | 83.0% | 0.485 | 0.606 | 15.9% | 4.0% |
| **Ensemble → band** | **84.6%** | 0.495 | 0.618 | **14.8%** | 4.0% |

: Table 11. Band results on the final test (442 sensor-days, 69 dates).

The ensemble's accuracy is 0.846 [0.786, 0.898], and against persistence it gains +0.029 [−0.005, +0.065] in accuracy and misses −0.080 [−0.182, +0.014] fewer bad days (13 of 88 against persistence's 20). So the CV ranking holds in direction: the ensemble again misses the fewest bad days (tied with Ridge) and ties for best accuracy. But the test is short, and none of these gaps is outside the noise on its own.

The Random Forest classifier, the only direct classifier run on the final test, has the best macro-F1 and the fewest false alarms. Its advantage comes mainly from the USG band (recall 83.6% against the ensemble's 64.8%) and from catching the single Good day. It also misses more bad days (20.5%). Its macro-F1 gain over the ensemble is not significant (ensemble − RF classifier: −0.069 [−0.176, +0.013]). Logistic regression and the Gradient Boosting classifier were not run on the final test, because they were not in the frozen plan and a second run would break the "scored once" rule.

![Figure 3. Final-test confusion matrices (442 sensor-days). The Very Unhealthy row is empty because no such day occurred.](results/bands/confusion_final_test.png){width=100%}

Because the final test has only one Good day and no Very Unhealthy days, the per-class evidence for those two bands comes from CV, not from the final test.

## Regression results

The regression step drives the chosen route, so its accuracy matters too. Table 12 gives the final-test errors.

| Model | MAE | RMSE | R² | Skill vs persistence |
|---|---|---|---|---|
| Persistence | 5.82 | 8.64 | 0.762 | — |
| Ridge (baseline) | 5.60 | 7.72 | 0.810 | +3.8% |
| Random Forest | 5.37 | 7.80 | 0.806 | +7.8% |
| Gradient Boosting | 4.73 | 6.85 | 0.851 | +18.7% |
| NumPy network | 4.98 | 6.97 | 0.846 | +14.5% |
| **Ensemble (GB + NumPy)** | **4.63** | **6.61** | **0.861** | **+20.5%** |

: Table 12. Regression results on the final test, µg/m³.

The ensemble's MAE is 17% lower than the Ridge baseline's on the final test (4.63 against 5.60), and 15% lower on CV (6.16 against 7.22). Errors are smaller in the final test than in CV because September to November is cleaner than January to August, so there is less to miss by.

# Explainability

A forecast that says "Unhealthy tomorrow" with no reason is hard to trust. We explain every forecast with **SHAP** (SHapley Additive exPlanations) [11]. SHAP splits one prediction into a starting point, the *base value* (the average forecast), plus one contribution per feature. The contributions add up exactly to the forecast. A positive contribution pushed the forecast up; a negative one pulled it down.

The production model is an average of a tree model and a neural network, so no single model-specific explainer can open it. We therefore used SHAP's model-agnostic *permutation* explainer on the ensemble's own predict function, with 100 training rows as background. One forecast takes about 0.6–0.8 seconds, fast enough for the app.

## Global importance

![Figure 4. SHAP summary for the production ensemble over a 300-row sample. Each dot is one sensor-day; position shows how much that feature moved the forecast.](results/explain/shap_summary.png){width=90%}

| Rank | Feature | Mean \|SHAP\| (µg/m³) | Meaning |
|---|---|---|---|
| 1 | Previous day's mean PM2.5 | 11.4 | Smog persists from one day to the next |
| 2 | PM2.5 one hour before midnight | 4.9 | The freshest reading |
| 3 | Forecast-day CAMS PM2.5 | 3.0 | An independent physical forecast |
| 4 | Forecast-day CAMS NO₂ | 2.8 | A proxy for traffic and industry |
| 5 | Previous day's PM2.5 variability | 1.9 | Volatile days are harder to extrapolate |
| 6 | 72-hour mean PM2.5 | 1.6 | Longer memory |
| 7 | Forecast-day east-west wind | 1.3 | Which way pollution is blown |
| 8 | Forecast-day wind speed | 1.2 | Calm air traps pollution |

: Table 13. Top global factors, ensemble explainer (`results/explain/global_importance.csv`).

**Agreement with domain knowledge.** Two patterns stand out, and both match atmospheric science. First, persistence dominates: recent PM2.5 outweighs all the weather features combined, which is also why persistence is such a hard baseline. Second, weather works at the margin. Wind, the CAMS forecast and the mixing height (mean |SHAP| 1.1 µg/m³) are what move a forecast away from "same as yesterday". Calm wind and a low mixing height trap pollution near the ground; rain washes it out. As a cross-check we also ran the exact tree explainer on the Gradient Boosting member alone over all 4,027 rows. It agreed on the top two features (in swapped order) and on the two CAMS features just below them.

## A per-forecast example

For sensor S11 (Orangi Pilot Project) on 20 December 2025, a winter smog day, the ensemble forecast 157.3 µg/m³ (Very Unhealthy) from a base value of 49.2 µg/m³:

| Reason | Contribution (µg/m³) |
|---|---|
| Yesterday's average PM2.5 was high (137.3) | +37.5 |
| The reading an hour before midnight was high (195.0) | +19.1 |
| The forecast day's CAMS PM2.5 was high (59.3) | +13.3 |
| The forecast day's CAMS NO₂ was high (55.0) | +11.7 |
| The last three days averaged 120.7 | +6.9 |
| Yesterday's PM2.5 swung a lot (std 63.9) | −6.6 |

: Table 14. SHAP explanation for S11 on 20 December 2025.

On a clean monsoon day (S2, 10 September 2025, forecast 5.0 µg/m³), the top reasons were a low previous-day mean (−17.5) and 86 mm of rain the day before (−6.6). The base value plus all contributions matched the model's output to within about 10⁻¹³ µg/m³. The app shows the top three reasons in this plain-English form.

**Limits.** Correlated features share credit: on a smoggy day, three PM2.5 history features all rise together and SHAP splits the credit among them. SHAP explains the model, not the atmosphere: a large value for wind means the forecast moved because wind changed, not proof that wind caused the pollution.

# Interface

The Streamlit [13] app (`app/`) has three pages.

**Forecast (home page).** A status banner gives the forecast date, when the data was last updated, and its source (live, cache or demo). A map of the sensor sites, coloured by forecast band, sits beside a dropdown; clicking a dot selects the site. The page then shows the forecast value and its colour-coded band, two profile checkboxes, the advice card with its notes, disclaimer and "You see this because…" trace, the top three SHAP reasons, and a chart of the last 72 hours of readings with the forecast.

![Figure 5. The Forecast page: status banner, site map, forecast band and health alert.](docs/report/figures/app_forecast.png){width=100%}

![Figure 6. The advice card with the expert system's explanation and SHAP reasons.](docs/report/figures/app_advice.png){width=100%}

**Model comparison.** Final-test error and band-accuracy tables for every model, the NumPy-versus-PyTorch check, the SHAP importance chart and the per-site fairness chart.

![Figure 7. The Model comparison page.](docs/report/figures/app_compare.png){width=100%}

**About and data.** Data sources, the model version, the four-step method, the limitations, the privacy note and the disclaimer.

## Health alerts

When the selected site's band is USG, the page shows an orange warning; for Unhealthy or worse, a red one. A city-wide line counts how many sites are forecast USG or worse, for example "3 of 9 sites are forecast Unhealthy for Sensitive Groups or worse." Every alert ends with "Informational only — not an official alert." The project is not an official warning service, and the wording says so.

## Network availability

The brief notes that real-time monitoring depends on the network. The app is built to degrade gracefully, in three tiers:

1. **Live:** the "Refresh live data" button fetches the latest sensor readings from OpenAQ and the weather and CAMS forecasts from Open-Meteo, builds features for today and forecasts every site. Requests retry on failure. A full refresh takes about 2–3 minutes because OpenAQ limits the request rate.
2. **Cache:** each successful live snapshot is saved. If a refresh fails, the app shows the last saved forecast and a warning that names the failure and the snapshot's time. If the cached forecast is for an earlier day, a stale-data warning says so.
3. **Demo:** with no cache at all, or when the user switches on "Demo mode", the app forecasts a stored historical day from the processed data, fully offline. Demo mode always shows a warning.

Sites with no recent data are listed as "no forecast today" rather than given a guessed value. An "Offline map" toggle draws the map without background tiles; it switches on by default when a quick connection test to the tile server fails. If the trained model file is missing, the app shows the command to build it instead of crashing.

# Ethics, privacy and fairness

## Privacy

The sensor, weather and atmospheric data contain no personal information: they describe places, not people. The only personal input is the two profile checkboxes. They stay in the browser session; they are never written to disk, logged or saved in the cache. The profile is also limited by design: a test fails if anyone adds a third field, such as a named medical condition, to the facts the engine may see.

## Fairness across sites

A forecast that is good in rich neighbourhoods and poor elsewhere would be unfair. We scored the production ensemble separately at each site over the pooled CV rows. S14 has no rows in the CV blocks, so 13 sites are scored.

![Figure 8. Ensemble MAE per sensor site, pooled CV.](results/fairness/ensemble/per_sensor_mae.png){width=90%}

- **Error is uneven.** MAE ranges from 4.6 µg/m³ at the Climate Action Center (S1) to 15.6 at DG House DHA Phase 5 (S10), against 6.2 overall. Six sites are more than 25% above average, including Orangi Pilot Project (S11, 13.4) and NED University (S7, 9.9).
- **Why.** Error tracks each site's pollution level (rank correlation 0.77 with mean PM2.5) more than its data gaps (−0.39). Dirtier sites have bigger numbers to miss by. The worst sites also have the fewest CV rows (14–66), so their estimates are themselves uncertain. Even after dividing by each site's mean PM2.5, S10 remains the relatively worst (normalised MAE 0.29).
- **Missed bad days are spread more evenly.** The highest miss rate is at Aga Khan University (S4), 5 of its 25 Unhealthy-or-worse days (20%). Elsewhere it is 0–15%.
- **False alarms cluster at sparse, polluted sites:** 36% at S10, 23% at S7 and 17% at S9, against 2–7% at well-covered sites.

In practice, at the sparse, high-pollution sites the forecast leans towards calling a day bad, which is the safer direction for health. Their forecasts still deserve more caution, and the app notes this.

## Transparency

Every forecast comes with its SHAP reasons and its full rule trace. Every rule names its source. The comparison page shows all models, including the ones that did not win, and the About page lists the limitations in plain words.

## Accountability

Every advice panel carries the disclaimer: the advice is general public-health guidance based on EPA and WHO material, "informational only — not an official air-quality alert and not personal medical advice". Health alerts are worded as information, not warnings. The app shows the model version and data sources. The final test was scored once with a script that refuses to run again, so the results in this report could not be tuned after the fact.

# Challenges and limitations

- **Perfect prognosis.** All results use recorded weather for the forecast day. Live forecasts use predicted weather, so live accuracy will be somewhat lower. We could not measure how much lower, because past weather forecasts for Karachi were not part of our data.
- **Short history.** We have about 16 months of sensor data (June 2025 to November 2026) and one winter. The untuned network's failure on unseen calendar dates (Section 5.6) shows how little seasonal coverage there is. A second winter would help every model.
- **A one-season final test.** The final test covers September to early November only. It has one Good day and no Very Unhealthy days, so it cannot test the extremes, and its intervals are wide (69 dates).
- **Extreme bands.** The ensemble catches only 16% of Very Unhealthy days and 57% of Good days on CV. Averaging pulls forecasts to the middle. The safety margin softens this for advice, but the displayed band still under-calls extremes.
- **Macro-F1 tie with persistence.** On the textbook imbalance metric, our system is no better than copying yesterday. We chose a health-first rule instead, and we state that choice openly.
- **Sensor noise and gaps.** These are low-cost sensors, not reference instruments. Of 7,276 candidate sensor-days, 3,149 (43%) had too few valid hours to form a target. One site was offline for about 230 days in a single gap.
- **Sensors offline in live use.** About nine of the 14 sensors report at a time. A site with no recent data gets no forecast.
- **API limits.** OpenAQ rate-limits requests, so a live refresh takes 2–3 minutes. The cache and demo modes exist partly for this reason.
- **No reference overlap.** The US Consulate monitor stopped in March 2025, before our sensors started, so we could not calibrate the low-cost sensors against it directly.
- **Rule thresholds are not tuned.** The 5 µg/m³ margin and the trend threshold are design choices, not the result of a study of user outcomes.

# Conclusion and future work

We built a hybrid AI and expert system that classifies tomorrow's air-quality band at Karachi's sensor sites and turns it into explained, sourced health advice. On 1,458 cross-validation sensor-days, an equal-weight ensemble of Gradient Boosting and a NumPy neural network, followed by EPA banding, reached 82.6% accuracy and missed 8.0% of Unhealthy-or-worse days. That is clearly better than persistence (18.0% missed) and the logistic-regression baseline (23.9% missed). Its macro-F1 ties with persistence, which we report rather than hide, and it under-calls the rarest bands. On the one-time final test, the ranking held in direction (84.6% accuracy, 14.8% missed) but with wide intervals.

Three choices did the most work. Evaluating on rolling folds instead of a single monsoon block made band tuning possible at all. Writing the selection rule down, with missed bad days first, stopped a naive model from winning on the wrong metric. And moving the safety margin from the band into the expert system kept "Good" forecasts while still erring towards caution in the advice.

Future work, in order of value:

1. Collect live weather forecasts every day and measure the real live skill against perfect prognosis.
2. Retrain after a second winter, and re-run a final test that contains Very Unhealthy days.
3. Try ordinal classification or cost-sensitive thresholds aimed at the extreme bands.
4. Give per-site uncertainty ranges, wider where data is sparse.
5. Study the advice margin and trend thresholds with real users.

# References {-}

1. United States Environmental Protection Agency, Office of Air Quality Planning and Standards. *Technical Assistance Document for the Reporting of Daily Air Quality — the Air Quality Index (AQI)*. Research Triangle Park, NC: US EPA; reflects the 2024 PM NAAQS revision. https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf (accessed 8 November 2026).
2. United States Environmental Protection Agency / AirNow. *Air Quality Guide for Particle Pollution*. EPA-452/F-23-002, February 2023. https://www.airnow.gov/sites/default/files/2023-03/air-quality-guide-for-particle-pollution_0.pdf (accessed 8 November 2026).
3. World Health Organization. *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. Geneva: WHO; 2021. ISBN 978-92-4-003422-8. https://iris.who.int/items/5b0ca7ab-830a-44b8-b99f-4ee743b4d7b2 (accessed 8 November 2026).
4. Pakistan Environmental Protection Agency. *National Environmental Quality Standards for Ambient Air* (values effective 1 January 2012), hosted by the Ministry of Climate Change. https://mocc.gov.pk/SiteImage/Misc/files/NEQS%20for%20Ambient%20Air.pdf (accessed 8 November 2026). Approval date not independently confirmed.
5. United States Environmental Protection Agency. *Reconsideration of the National Ambient Air Quality Standards for Particulate Matter* (final rule, 2024; effective 6 May 2024).
6. OpenAQ. OpenAQ API v3, https://api.openaq.org. Data licensed CC BY 4.0; original data from AirGradient / The Urban Unit and the US Department of State (AirNow).
7. Open-Meteo. Historical Weather API, https://archive-api.open-meteo.com. Licensed CC BY 4.0.
8. Open-Meteo. Air Quality API, https://air-quality-api.open-meteo.com. Licensed CC BY 4.0; data from the Copernicus Atmosphere Monitoring Service.
9. Hersbach, H. et al. "The ERA5 global reanalysis." *Quarterly Journal of the Royal Meteorological Society* 146 (2020): 1999–2049.
10. Copernicus Atmosphere Monitoring Service (CAMS), European Centre for Medium-Range Weather Forecasts.
11. Lundberg, S. M. and Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems* 30 (2017).
12. Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12 (2011): 2825–2830.
13. Streamlit. Streamlit open-source app framework, https://streamlit.io.
14. Paszke, A. et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems* 32 (2019).
15. Harris, C. R. et al. "Array programming with NumPy." *Nature* 585 (2020): 357–362.
16. Breiman, L. "Random Forests." *Machine Learning* 45 (2001): 5–32.
17. Kingma, D. P. and Ba, J. "Adam: A Method for Stochastic Optimization." *International Conference on Learning Representations* (2015).

# Appendix A: Requirement traceability {-}

| Brief requirement | Report section | Code / evidence |
|---|---|---|
| CLO 2: AI techniques for a classification problem | 2.4, 5, 7 | `src/classify/bands.py`, `src/evaluate/band_report.py`, `results/bands/` |
| CP2 depth of analysis: abstract models, no obvious solution | 2.4, 5.8, 7.1 | `docs/report/model_selection.md`, `results/bands/bootstrap_ci.csv` |
| CP3 depth of knowledge: domain and computing principles | 2.1, 4.3, 6 | `src/expert/`, `src/data/features.py` |
| Real-world problem in an allowed domain | 1.1, 2.1 | — |
| Clear title | Title page | — |
| Problem statement: issue, impact, AI/ES approach | 1.1 | — |
| Problem description: domain, use case, goals | 1.2, 2 | — |
| At least two datasets / knowledge sources, described | 3.1 | `src/data/download_*.py`, `data/raw/*/manifest.json`, `docs/report/knowledge_sources.md` |
| How datasets build and evaluate the system; metrics | 3.1, 5.2, 5.3 | `src/data/split.py`, `src/evaluate/metrics.py` |
| Noise, missing values, inconsistencies | 4.1, 4.2 | `src/data/clean.py`, `data/processed/build_report.json` |
| Class imbalance (Lab brief) | 4.5 | `src/classify/bands.py`, `src/evaluate/tune_classical.py` |
| Diverse data types | 3.3 | `src/data/features.py`, `src/models/dataset.py`, `src/expert/facts.py` |
| External factors | 3.4, 11 | `src/data/download_weather.py`, `src/data/download_air_quality.py` |
| Data privacy | 10.1 | `src/expert/facts.py`, `app/views/forecast.py` |
| Fairness, transparency, accountability | 8, 10 | `src/evaluate/fairness.py`, `src/explain/`, `src/expert/explain.py`, `results/fairness/ensemble/` |
| Baseline model | 5.4, 7.1 | `src/models/baselines/`, `results/baselines_summary.md` |
| Improvement with multiple AI approaches (tuning, ensembling) | 5.5 – 5.7, 7 | `src/models/`, `src/evaluate/tune_classical.py`, `src/evaluate/tune_mlp.py`, `src/models/ensemble.py`, `src/models/final.py` |
| Expert-system techniques | 6 | `src/expert/rules.py`, `src/expert/engine.py`, `src/expert/explain.py` |
| Network availability for real-time monitoring | 9.2 | `src/live/pipeline.py`, `src/live/snapshot.py`, `src/ui/banner.py`, `src/ui/map_mode.py` |
| Deliverable: Python code (pre-processing to improvement) | 5, Appendix B | `src/`, `tests/` |
| Deliverable: user-friendly visualisation / interface | 9 | `app/`, `src/ui/` |
| Deliverable: technical report of at least 10 pages | this document | `docs/report/aies/aies_report.md` |
| Methodology with diagram | 5.1 | `docs/report/figures/pipeline.png`, `docs/report/figures/make_pipeline.py` |
| Models used | 5.4 – 5.8 | `src/models/final.py` |
| Results | 7, 8 | `results/bands/`, `results/final_test/`, `results/explain/` |
| Challenges | 11 | — |

: Table A1. Each brief requirement mapped to where this report covers it and where the evidence lives.

# Appendix B: How to reproduce {-}

All commands run from the repository root with `uv` (Python 3.12). Step 1 needs a free OpenAQ API key in a `.env` file as `OPENAQ_API_KEY`.

```
# 1. Data
uv run python -m src.data.download_sensors                           # D1: 14 sensors (OpenAQ)
uv run python -m src.data.download_sensors --consulate               # D5: US Consulate monitor
uv run python -m src.data.download_weather --start 2025-05-01        # D2: ERA5 weather
uv run python -m src.data.download_air_quality --start 2025-05-01    # D3: CAMS air quality

# 2. Features
uv run python -m src.data.build_features

# 3. Baselines on the CV folds
uv run python -m src.evaluate.run_cv

# 4. Tuning (classical models, then the NumPy network)
uv run python -m src.evaluate.tune_classical
uv run python -m src.evaluate.tune_mlp

# 5. Final test (scored once), then save the production model
uv run python -m src.evaluate.final_test
uv run python -m src.evaluate.final_test --save-production

# 6. SHAP explanations
uv run python -m src.explain --features data/processed/features.parquet --bundle models/final_ensemble.pkl

# 7. Expert-system demo (advice + rule trace)
uv run python -m src.expert --pm25 33 --today 25 --sensitive --outdoor

# 8. Live snapshot (needs OPENAQ_API_KEY)
uv run python -m src.live --refresh

# 9. App
uv run streamlit run app/main.py
```

Band evidence with bootstrap intervals is rebuilt from the saved predictions, without retraining, by `uv run python -m src.evaluate.band_report --n-boot 2000`; per-site fairness by `uv run python -m src.evaluate.fairness --model ensemble --features data/processed/features.parquet`. The app runs offline with step 9 alone once the processed data and model file exist.
