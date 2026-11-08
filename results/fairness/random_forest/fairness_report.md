# Fairness report — random_forest (regression)

Per-sensor forecast error (spec §10): checks whether some areas (e.g. Orangi vs AKU) are
forecast worse than others, and whether that is explained by data gaps or by higher pollution.

## Findings

- **Orangi Pilot Project (S11)** is forecast worst: MAE 18.5 µg/m³, 173% above the overall MAE of 6.8 µg/m³.
- **Climate Action Center (S1)** is forecast best: MAE 5.0 µg/m³, 26% below the overall MAE.
- 6 sensor(s) exceed the overall MAE by more than 25%: Orangi Pilot Project (S11), DG House DHA Phase 5 (S10), Urban Unit Karachi office (S13), Karachi Zoo (S12), NED University, Main Campus (S7), Aman Khan Kaka Khail (S9).
- Error tracks pollution level more than data gaps: error vs mean PM2.5 rho=0.81 vs error vs completeness rho=-0.47 -- dirtier sites simply have bigger numbers to miss by.
- After adjusting for each site's own pollution level (normalised MAE = MAE / mean PM2.5), **Orangi Pilot Project (S11)** is relatively worst (normalised MAE 0.314).
- **Unhealthy-day miss rate** (true band Unhealthy-or-worse, forecast healthier) is highest at **Aga Khan University Main Campus (S4)**: 16% of its unhealthy-or-worse days were forecast as safer than they were.

## Per-sensor regression error and data-quality factors

| sensor_id | site | n | mae | rmse | bias | skill | completeness_pct | training_rows | mean_pm25 | normalised_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S11 | Orangi Pilot Project | 14 | 18.52 | 23.96 | -9.87 | 0.10 | 43.70 | 205 | 58.95 | 0.31 |
| S10 | DG House DHA Phase 5 | 52 | 16.09 | 20.24 | -0.64 | 0.31 | 85.40 | 224 | 54.40 | 0.30 |
| S13 | Urban Unit Karachi office | 14 | 14.77 | 18.19 | -8.61 | 0.17 | 35.90 | 156 | 56.41 | 0.26 |
| S12 | Karachi Zoo | 35 | 11.07 | 13.07 | 2.64 | 0.41 | 74.80 | 161 | 60.96 | 0.18 |
| S7 | NED University, Main Campus | 66 | 10.76 | 14.08 | 1.73 | 0.19 | 58.50 | 282 | 52.13 | 0.21 |
| S9 | Aman Khan Kaka Khail | 65 | 10.06 | 13.21 | 3.48 | 0.21 | 49.30 | 233 | 49.82 | 0.20 |
| S2 | Zafar Memon DHA | 222 | 6.34 | 10.25 | 2.54 | 0.18 | 84.70 | 412 | 32.57 | 0.19 |
| S3 | Urban Resource Center | 169 | 6.06 | 9.15 | -1.70 | 0.17 | 85.00 | 409 | 49.79 | 0.12 |
| S5 | Gulistan-e-Johar | 149 | 5.74 | 8.56 | 0.19 | 0.28 | 77.90 | 383 | 48.25 | 0.12 |
| S8 | Malir Cantonment, Askari 5 | 108 | 5.64 | 8.66 | 2.64 | 0.10 | 57.70 | 243 | 40.90 | 0.14 |
| S6 | G3 Engineering Consultants | 172 | 5.57 | 10.87 | -0.05 | 0.16 | 78.20 | 362 | 37.97 | 0.15 |
| S4 | Aga Khan University Main Campus | 203 | 5.12 | 7.71 | 0.65 | 0.17 | 80.80 | 389 | 35.44 | 0.14 |
| S1 | Climate Action Center | 189 | 5.03 | 7.61 | 1.81 | 0.31 | 85.60 | 407 | 42.88 | 0.12 |

## Per-sensor classification safety (regress-then-band)

| sensor_id | site | n | accuracy | macro_f1 | miss_rate | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- |
| S4 | Aga Khan University Main Campus | 203 | 0.84 | 0.46 | 0.16 | 0.02 |
| S7 | NED University, Main Campus | 66 | 0.61 | 0.37 | 0.08 | 0.35 |
| S8 | Malir Cantonment, Askari 5 | 108 | 0.83 | 0.43 | 0.08 | 0.07 |
| S3 | Urban Resource Center | 169 | 0.75 | 0.46 | 0.07 | 0.02 |
| S2 | Zafar Memon DHA | 222 | 0.77 | 0.54 | 0.06 | 0.06 |
| S9 | Aman Khan Kaka Khail | 65 | 0.71 | 0.43 | 0.06 | 0.13 |
| S10 | DG House DHA Phase 5 | 52 | 0.62 | 0.22 | 0.05 | 0.64 |
| S1 | Climate Action Center | 189 | 0.86 | 0.46 | 0.03 | 0.11 |
| S5 | Gulistan-e-Johar | 149 | 0.81 | 0.44 | 0.02 | 0.10 |
| S13 | Urban Unit Karachi office | 14 | 0.64 | 0.23 | 0.00 | 0.75 |
| S12 | Karachi Zoo | 35 | 0.77 | 0.22 | 0.00 | 0.88 |
| S11 | Orangi Pilot Project | 14 | 0.64 | 0.22 | 0.00 | 1.00 |
| S6 | G3 Engineering Consultants | 172 | 0.89 | 0.53 | 0.00 | 0.03 |

## Notes

- `skill` = 1 - MAE_model / MAE_persistence (0 = no better than the naive forecast).
- `normalised_mae` = MAE / mean PM2.5, so a wetter/dirtier site is not penalised purely for being dirtier.
- `miss_rate` is NaN at a sensor with no Unhealthy-or-worse day in this window (not an error).
