# Final test (scored once)

Period: 2026-09-01 to 2026-11-08 · 442 sensor-days · trained on all rows before 2026-09-01.

## Forecast error (µg/m³)

| model | mae | rmse | r2 | skill |
|---|---|---|---|---|
| persistence | 5.819 | 8.642 | 0.762 | 0.000 |
| ridge | 5.597 | 7.719 | 0.810 | 0.038 |
| random_forest | 5.368 | 7.800 | 0.806 | 0.078 |
| hist_gb | 4.730 | 6.849 | 0.851 | 0.187 |
| mlp_numpy | 4.978 | 6.966 | 0.846 | 0.145 |
| ensemble | 4.626 | 6.610 | 0.861 | 0.205 |

## Bands

| model | route | accuracy | macro_f1 | miss_unhealthy | false_alarm |
|---|---|---|---|---|---|
| persistence | regress_then_band | 0.817 | 0.477 | 0.227 | 0.045 |
| ridge | regress_then_band | 0.819 | 0.473 | 0.148 | 0.056 |
| random_forest | regress_then_band | 0.844 | 0.492 | 0.159 | 0.037 |
| hist_gb | regress_then_band | 0.835 | 0.488 | 0.182 | 0.042 |
| mlp_numpy | regress_then_band | 0.830 | 0.485 | 0.159 | 0.040 |
| ensemble | regress_then_band | 0.846 | 0.495 | 0.148 | 0.040 |
| random_forest_clf | direct | 0.846 | 0.550 | 0.205 | 0.020 |

`miss_unhealthy`: share of truly Unhealthy-or-worse days forecast as a cleaner band. `false_alarm`: share of cleaner days forecast as Unhealthy-or-worse.

Band support in this period: Good 1, Moderate 225, USG 128, Unhealthy 88, Very Unhealthy 0.
