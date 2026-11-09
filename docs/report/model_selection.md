# Model selection and final test (Phase 5)

How the forecaster was chosen, and how it did on data it had never seen. Numbers come from
`results/tuning/` (cross-validation) and `results/final_test/` (the one-time final test).

## How models were compared

- **Cross-validation (CV):** 8 rolling-origin folds over Jan–Aug 2026. Each fold trains on every
  day before a 30-day block and tests on that block, with a 1-day gap. All settings were chosen
  on these folds. Scores below are **pooled**: all 8 test blocks scored together (2,224 sensor-days).
- **Final test:** 1 Sep – 8 Nov 2026 (442 sensor-days), trained on everything before it and
  scored **once**, after every choice was frozen (`src/evaluate/final_test.py` refuses a second run).
- **Yardstick:** persistence ("tomorrow = yesterday's average"). *Skill* = how much lower a
  model's mean absolute error (MAE) is than persistence's.

## Tuning (CV, MAE in µg/m³)

| Model | Default | Tuned | Key settings |
|---|---|---|---|
| Persistence | 8.70 | — | — |
| Ridge (official baseline) | 7.68 | 7.22 | alpha 150 |
| Random Forest | 6.79 | 6.54 | 500 trees, depth 15, min leaf 5, 50% of features per split |
| Gradient Boosting | — | **6.30** | 100 rounds, 31 leaves, L2 0.1 |
| NumPy network | 11.21 | 6.63 | 32 hidden units, L2 0.1, change-from-yesterday target, no calendar features |

**Why the untuned network was so bad, and what fixed it** (`results/tuning/mlp_numpy_diagnosis.csv`):

- **Unseen seasons.** The data starts in June 2025, so the folds testing March–May had never seen
  those days of the year. The network read the unfamiliar calendar values as winter and
  over-forecast: in fold 2 its average error was +21.9 µg/m³ too high. Dropping the three
  calendar features cut that bias to −2.5.
- **Overfitting.** Training error was ~3 µg/m³ against ~15–17 on the held-out rows. A smaller
  network (32 units instead of 64+32) with a much larger weight penalty (L2 0.1) fixed most of it.
- **Predicting the change, not the level.** The network learns the difference from a simple
  "yesterday → today" line (fitted on training rows only), which is a much easier target.
- **Stable across random seeds:** MAE 6.57 / 6.63 / 6.70 for seeds 0, 1, 2.

## Ensemble (CV)

Averaging models helps because they make different mistakes (prediction correlation 0.97, not 1).

| Combination (equal-weight average) | MAE | Skill | Band accuracy | Band macro-F1 |
|---|---|---|---|---|
| All four (Ridge + RF + GB + network) | 6.13 | +29.5% | — | 0.642 |
| **Gradient Boosting + NumPy network** | **6.16** | **+29.2%** | **82.7%** | **0.674** |
| Gradient Boosting alone | 6.30 | +27.5% | — | 0.645 |
| NumPy network alone | 6.63 | +23.8% | 80.7% | 0.659 |

**Chosen:** Gradient Boosting + NumPy network. It is within 0.03 µg/m³ of the four-model average,
simpler, has the best band score of any combination, and keeps the from-scratch network in the
final system. Equal weights on purpose: fitting weights on the same folds would overfit them.

## Band route: predict the number, or the band directly? (CV)

| Route | Accuracy | Macro-F1 | Unhealthy days missed | False alarms |
|---|---|---|---|---|
| Persistence → band | 78.9% | **0.678** | 18.0% | 5.4% |
| Random Forest classifier (direct) | 78.3% | 0.651 | 15.0% | 3.6% |
| Gradient Boosting classifier (direct) | 82.0% | 0.664 | 10.0% | 6.3% |
| **Ensemble → band** | **82.6%** | 0.674 | **8.0%** | 5.6% |

**Chosen:** predict PM2.5 with the ensemble, then look up its band. The best direct classifier
(Gradient Boosting) comes close but is behind on every measure. It is the most accurate and
misses the fewest Unhealthy-or-worse days, which is the error that matters most for health.
Its weak spot is the rare extremes: it recognises only 57% of Good days and 16% of the 25
Very Unhealthy days, because averaging models pulls extreme forecasts toward the middle.

**Safety margin.** Adding +5 µg/m³ before banding cuts missed Unhealthy days to 4.1% but raises
false alarms to 9.1% and means "Good" is never forecast. So the margin is **not** applied to the
band shown; it belongs in the advice layer (Phase 6): when a forecast is near a boundary, the
advice follows the worse band and says so.

## Final test (1 Sep – 8 Nov 2026, scored once)

| Model | MAE | RMSE | R² | Skill |
|---|---|---|---|---|
| Persistence | 5.82 | 8.64 | 0.762 | — |
| Ridge (official baseline) | 5.60 | 7.72 | 0.810 | +3.8% |
| Random Forest | 5.37 | 7.80 | 0.806 | +7.8% |
| Gradient Boosting | 4.73 | 6.85 | 0.851 | +18.7% |
| NumPy network | 4.98 | 6.97 | 0.846 | +14.5% |
| **Ensemble (GB + network)** | **4.63** | **6.61** | **0.861** | **+20.5%** |

| Bands | Accuracy | Unhealthy days missed | False alarms |
|---|---|---|---|
| Persistence → band | 81.7% | 22.7% | 4.5% |
| Random Forest classifier (direct) | 84.6% | 20.5% | 2.0% |
| **Ensemble → band** | **84.6%** | **14.8%** | 4.0% |

The ranking from CV holds on unseen data: the ensemble is the most accurate model and misses the
fewest Unhealthy days. Errors are smaller than in CV because Sep–Nov is cleaner than Jan–Aug.

**Limits of the final test.** It contains 1 Good day and no Very Unhealthy days, so macro-F1 over
five bands (≈0.49 for every model) mostly reflects those two absent classes, and per-band results
for Good and Very Unhealthy come from CV instead. Forecast-day weather uses recorded values as a
stand-in for weather forecasts (*perfect-prognosis*), so live accuracy will be somewhat lower.

## Production model

`uv run python -m src.evaluate.final_test --save-production` refits the preprocessor and the
ensemble on **all** labelled rows (Jun 2025 – Nov 2026) and saves `models/final_ensemble.pkl`
for the app. Settings live in `src/models/final.py`.
