# Fairness report — persistence (regression)

Per-sensor forecast error (spec §10): checks whether some areas (e.g. Orangi vs AKU) are
forecast worse than others, and whether that is explained by data gaps or by higher pollution.

## Findings

- **DG House DHA Phase 5 (S10)** is forecast worst: MAE 23.4 µg/m³, 168% above the overall MAE of 8.7 µg/m³.
- **Aga Khan University Main Campus (S4)** is forecast best: MAE 6.2 µg/m³, 29% below the overall MAE.
- 6 sensor(s) exceed the overall MAE by more than 25%: DG House DHA Phase 5 (S10), Orangi Pilot Project (S11), Karachi Zoo (S12), Urban Unit Karachi office (S13), NED University, Main Campus (S7), Aman Khan Kaka Khail (S9).
- Error tracks pollution level more than data gaps: error vs mean PM2.5 rho=0.85 vs error vs completeness rho=-0.29 -- dirtier sites simply have bigger numbers to miss by.
- After adjusting for each site's own pollution level (normalised MAE = MAE / mean PM2.5), **DG House DHA Phase 5 (S10)** is relatively worst (normalised MAE 0.429).
- **Unhealthy-day miss rate** (true band Unhealthy-or-worse, forecast healthier) is highest at **NED University, Main Campus (S7)**: 27% of its unhealthy-or-worse days were forecast as safer than they were.

## Per-sensor regression error and data-quality factors

| sensor_id | site | n | mae | rmse | bias | skill | completeness_pct | training_rows | mean_pm25 | normalised_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S10 | DG House DHA Phase 5 | 52 | 23.35 | 31.86 | 0.69 | 0.00 | 85.40 | 224 | 54.40 | 0.43 |
| S11 | Orangi Pilot Project | 14 | 20.63 | 25.47 | -7.96 | 0.00 | 43.70 | 205 | 58.95 | 0.35 |
| S12 | Karachi Zoo | 35 | 18.71 | 24.27 | 0.80 | 0.00 | 74.80 | 161 | 60.96 | 0.31 |
| S13 | Urban Unit Karachi office | 14 | 17.88 | 23.15 | 4.48 | 0.00 | 35.90 | 156 | 56.41 | 0.32 |
| S7 | NED University, Main Campus | 66 | 13.29 | 17.96 | -0.78 | 0.00 | 58.50 | 282 | 52.13 | 0.25 |
| S9 | Aman Khan Kaka Khail | 65 | 12.74 | 17.68 | 0.84 | 0.00 | 49.30 | 233 | 49.82 | 0.26 |
| S5 | Gulistan-e-Johar | 149 | 7.93 | 12.45 | 0.36 | 0.00 | 77.90 | 383 | 48.25 | 0.16 |
| S2 | Zafar Memon DHA | 222 | 7.72 | 14.58 | -0.13 | 0.00 | 84.70 | 412 | 32.57 | 0.24 |
| S1 | Climate Action Center | 189 | 7.29 | 12.51 | 0.10 | 0.00 | 85.60 | 407 | 42.88 | 0.17 |
| S3 | Urban Resource Center | 169 | 7.26 | 11.24 | 0.04 | 0.00 | 85.00 | 409 | 49.79 | 0.15 |
| S6 | G3 Engineering Consultants | 172 | 6.64 | 13.82 | -0.38 | 0.00 | 78.20 | 362 | 37.97 | 0.17 |
| S8 | Malir Cantonment, Askari 5 | 108 | 6.27 | 10.04 | 0.83 | 0.00 | 57.70 | 243 | 40.90 | 0.15 |
| S4 | Aga Khan University Main Campus | 203 | 6.15 | 9.48 | 0.01 | 0.00 | 80.80 | 389 | 35.44 | 0.17 |

## Per-sensor classification safety (regress-then-band)

| sensor_id | site | n | accuracy | macro_f1 | miss_rate | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- |
| S7 | NED University, Main Campus | 66 | 0.64 | 0.48 | 0.27 | 0.15 |
| S1 | Climate Action Center | 189 | 0.85 | 0.46 | 0.26 | 0.07 |
| S4 | Aga Khan University Main Campus | 203 | 0.82 | 0.50 | 0.24 | 0.03 |
| S8 | Malir Cantonment, Askari 5 | 108 | 0.86 | 0.55 | 0.23 | 0.04 |
| S9 | Aman Khan Kaka Khail | 65 | 0.65 | 0.49 | 0.22 | 0.11 |
| S6 | G3 Engineering Consultants | 172 | 0.87 | 0.61 | 0.17 | 0.02 |
| S2 | Zafar Memon DHA | 222 | 0.80 | 0.69 | 0.17 | 0.03 |
| S3 | Urban Resource Center | 169 | 0.76 | 0.56 | 0.17 | 0.03 |
| S10 | DG House DHA Phase 5 | 52 | 0.56 | 0.27 | 0.16 | 0.36 |
| S12 | Karachi Zoo | 35 | 0.74 | 0.27 | 0.15 | 0.50 |
| S5 | Gulistan-e-Johar | 149 | 0.78 | 0.45 | 0.12 | 0.06 |
| S11 | Orangi Pilot Project | 14 | 0.64 | 0.27 | 0.08 | 1.00 |
| S13 | Urban Unit Karachi office | 14 | 0.79 | 0.44 | 0.00 | 0.25 |

## Notes

- `skill` = 1 - MAE_model / MAE_persistence (0 = no better than the naive forecast).
- `normalised_mae` = MAE / mean PM2.5, so a wetter/dirtier site is not penalised purely for being dirtier.
- `miss_rate` is NaN at a sensor with no Unhealthy-or-worse day in this window (not an error).
