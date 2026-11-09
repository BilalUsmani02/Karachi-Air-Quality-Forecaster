# Explainability (Phase 6)

Why each forecast comes out the way it does, in plain English — required by both courses
(PAI §8: "explainable AI will highlight the main factors behind each prediction"; AIES A10:
transparency). Numbers below come from `results/explain/` (`uv run python -m src.explain`,
run on the full feature table and the production model).

## What SHAP is, in one example

**SHAP** (SHapley Additive exPlanations) splits one prediction into a starting point — the
**base value**, the average forecast if you knew nothing about the day — plus a list of
**contributions**, one per input feature, that add up exactly to the final number. A feature
that pushed the forecast up gets a positive contribution; one that pulled it down gets a
negative one.

Worked example, sensor S11 on 20 December 2025 (`results/explain/example_explanations.md`):

| | µg/m³ |
|---|---|
| Base value (average forecast) | 49.2 |
| + yesterday's average PM2.5 was high (137.3) | +37.5 |
| + the PM2.5 level an hour ago was high (195.0) | +19.1 |
| + tomorrow's modelled PM2.5 is high (59.3) | +13.3 |
| + tomorrow's modelled nitrogen dioxide is high (55.0) | +11.7 |
| + the remaining 54 features, combined (e.g. `pm25_mean_72h` +6.9, `pm25_std_24h` -6.6, …) | +26.5 |
| = **Forecast** | **157.3** |

Add the base value to every contribution (all 58, not just the four shown) and you get back
the model's actual prediction to within rounding — that's SHAP's **additivity** guarantee. We
checked it on every example in the report: the gap between "base + sum of contributions" and
the model's real output was on the order of **1e-13 µg/m³** (sampled rows, seed 42) — floating-point
noise, i.e. exact for practical purposes.

## Why model-agnostic SHAP on the ensemble, not DeepExplainer on a PyTorch twin

The original plan (`PROJECT_SPEC.md` §8) was `DeepExplainer` on a PyTorch copy of the NumPy
network, on the idea that an identical-architecture twin's explanations would stand in for the
NumPy model. That plan predates the final model choice. The system that actually ships
(`src/models/final.py::final_ensemble`, see `docs/report/model_selection.md`) is an **equal
average of two different model families** — a tuned Gradient Boosting regressor and the tuned
NumPy network — wrapped in `AveragingRegressor`. `DeepExplainer` can only open up one PyTorch
module; it has no notion of "half of a tree model's prediction and half of a neural net's
prediction, averaged", so it cannot explain the number the app actually shows.

Instead, `src/explain/shap_explain.py` wraps the ensemble's own `predict` method and explains
*that*, directly:

```python
masker = shap.maskers.Independent(background, max_samples=100)   # 100 training rows, fixed seed
explainer = shap.Explainer(ensemble.predict, masker, algorithm="permutation")
```

With 58 input features, exact SHAP (`2^58` feature subsets) is impossible, so this is a
**permutation explainer** — it estimates each feature's contribution by shuffling it in and out
across random feature orderings, which scales to any number of features and any model with a
`predict` method. Measured on a laptop, with the production ensemble and a 100-row background:
one row takes **≈0.6-0.8 s** — fast enough to run live when someone picks a location in the app
(the design target was ≲2-3 s).

For the report's global picture (next section), two methods are compared:

- **`TreeExplainer`** on the Gradient Boosting member alone — exact, and fast enough (≈6 s) to
  run over the *entire* feature table (4,027 rows), because it reads the tree structure
  directly instead of sampling.
- The **model-agnostic permutation explainer** on the *whole ensemble*, over a fixed-seed
  300-row sample (≈4.5 minutes — too slow to run on every row for a report figure, but fine
  once offline).

## Global top factors

Ranked by the ensemble explainer's mean |SHAP value| across the 300-row sample
(`results/explain/global_importance.csv`; see also `shap_summary.png`):

| # | Feature | Mean \|SHAP\| (µg/m³) | What it means physically |
|---|---|---|---|
| 1 | `pm25_prev_day_mean` | 11.4 | Yesterday's average PM2.5 — pollution persists day to day (smog doesn't clear overnight). |
| 2 | `pm25_lag_1h` | 4.9 | The most recent hourly reading — the freshest signal of current conditions. |
| 3 | `fc_cams_pm2_5` | 3.0 | Tomorrow's modelled PM2.5 from the CAMS atmospheric model — an independent forecast the model leans on. |
| 4 | `fc_cams_no2` | 2.8 | Tomorrow's modelled nitrogen dioxide — a proxy for traffic/industrial pollution tomorrow. |
| 5 | `pm25_std_24h` | 1.9 | How much PM2.5 swung up and down yesterday — volatile days are harder to extrapolate from. |
| 6 | `pm25_mean_72h` | 1.6 | The last three days' average — a longer-memory smoothed signal than yesterday alone. |
| 7 | `fc_wind_u` (+ `fc_wind_v`) | 1.3 | Tomorrow's wind direction — which way pollution gets blown in or out of the area. |
| 8 | `fc_wind_speed_10m` | 1.2 | Tomorrow's wind speed — calmer air traps pollution near the ground; stronger wind disperses it. |

Two physical patterns stand out. First, **persistence dominates**: the three most recent-PM2.5
features together outweigh every weather feature combined, confirming the project's own
baseline finding that "tomorrow = yesterday" is a strong starting point
(`docs/report/model_selection.md`). Second, **weather matters at the margin**: wind, the CAMS
model's independent forecast and the mixing height (`fc_boundary_layer_height`, mean |SHAP|
1.1 µg/m³ — how high up pollution can spread; a low mixing height traps it near the ground)
are what shift a forecast away from a pure persistence guess, which is exactly the information
a resident can't get by just checking today's reading.

## Agreement between TreeExplainer and the ensemble explainer

The two methods mostly agree on *what* matters, less so on the exact order:

- Both put **yesterday's level and the latest hourly reading** in the top two spots — just
  swapped: the tree ranks `pm25_lag_1h` first and `pm25_prev_day_mean` second; the ensemble
  ranks them the other way round. Both also put the **two CAMS forecast features** (modelled
  PM2.5, modelled NO₂) back-to-back just below that — ranks 5-6 for the tree, ranks 3-4 for the
  ensemble.
- They diverge further down: the tree explainer gives more weight to `doy_cos` (the seasonal
  cycle — rank 7 for the tree, rank 12 for the ensemble) and `city_prev_day_mean` (the citywide
  average — rank 4 for the tree, rank 18 for the ensemble), while the ensemble gives much more
  weight to `pm25_std_24h` (yesterday's volatility — rank 5 for the ensemble, rank 27 for the
  tree alone).
- This divergence is expected: `TreeExplainer` only sees the Gradient Boosting member, while
  the ensemble explainer sees the *average* of that model and the NumPy network, which was
  deliberately tuned with a different, smaller feature set (no calendar features at all — see
  `src/models/numpy_nn/variant.py`) and a different target (the residual from a persistence
  anchor, not the raw level). Averaging two models that lean on different features for the same
  underlying signal is exactly why the ensemble forecasts better than either alone
  (`docs/report/model_selection.md`'s correlation-0.97-not-1.0 finding) — and it shows up here
  as the two explainers partially disagreeing on *why*.

## Limits

- **Correlated features share credit, not causation.** `pm25_lag_1h`, `pm25_prev_day_mean` and
  `pm25_mean_72h` all move together on a smoggy day; SHAP splits the credit for "PM2.5 has been
  high" across all of them rather than crediting one "true" cause, so no single bar in the
  summary plot should be read as the whole story.
- **SHAP explains the model, not the weather.** A high SHAP value for wind means "the model's
  forecast moved because this input changed", not "wind definitely caused this PM2.5 level" —
  the model can only have learned correlations present in ~16 months of data (June 2025 -
  November 2026), and the brief history means its idea of "what matters" is itself a choice the
  model made from limited examples, not an atmospheric-science finding.
- **Forecast-day weather is a stand-in.** Every `fc_*` feature in the explanation is Open-Meteo's
  *recorded* weather for that day used as a perfect-prognosis substitute for tomorrow's weather
  forecast (`docs/report/model_selection.md`). Live SHAP values in the app will look the same in
  shape, but the underlying `fc_*` numbers will carry whatever error is in the live weather
  forecast itself.
- **The permutation explainer is an approximation** for a non-linear model (unlike
  `TreeExplainer`, which is exact for trees); additivity held to ~1e-13 µg/m³ (floating-point
  noise) on every row checked here, but that is empirical, not a guarantee for every possible row.
