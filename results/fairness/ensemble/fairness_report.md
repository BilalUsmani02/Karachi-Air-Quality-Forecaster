# Fairness report — ensemble (regress_then_band)

Per-sensor forecast error (spec §10): checks whether some areas (e.g. Orangi vs AKU) are
forecast worse than others, and whether that is explained by data gaps or by higher pollution.

## Findings

- **DG House DHA Phase 5 (S10)** is forecast worst: MAE 15.6 µg/m³, 153% above the overall MAE of 6.2 µg/m³.
- **Climate Action Center (S1)** is forecast best: MAE 4.6 µg/m³, 25% below the overall MAE.
- 6 sensor(s) exceed the overall MAE by more than 25%: DG House DHA Phase 5 (S10), Urban Unit Karachi office (S13), Orangi Pilot Project (S11), Karachi Zoo (S12), NED University, Main Campus (S7), Aman Khan Kaka Khail (S9).
- Error tracks pollution level more than data gaps: error vs mean PM2.5 rho=0.77 vs error vs completeness rho=-0.39 -- dirtier sites simply have bigger numbers to miss by.
- After adjusting for each site's own pollution level (normalised MAE = MAE / mean PM2.5), **DG House DHA Phase 5 (S10)** is relatively worst (normalised MAE 0.286).
- **Unhealthy-day miss rate** (true band Unhealthy-or-worse, forecast healthier) is highest at **Aga Khan University Main Campus (S4)**: 20% of its unhealthy-or-worse days were forecast as safer than they were.

## Per-sensor regression error and data-quality factors

| sensor_id | site | n | mae | rmse | bias | skill | completeness_pct | training_rows | mean_pm25 | normalised_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S10 | DG House DHA Phase 5 | 52 | 15.57 | 20.26 | -3.18 | 0.33 | 85.40 | 224 | 54.40 | 0.29 |
| S13 | Urban Unit Karachi office | 14 | 13.63 | 17.17 | -8.51 | 0.24 | 35.90 | 156 | 56.41 | 0.24 |
| S11 | Orangi Pilot Project | 14 | 13.37 | 16.87 | -6.29 | 0.35 | 43.70 | 205 | 58.95 | 0.23 |
| S12 | Karachi Zoo | 35 | 10.00 | 12.10 | -0.31 | 0.47 | 74.80 | 161 | 60.96 | 0.16 |
| S7 | NED University, Main Campus | 66 | 9.87 | 12.53 | -0.58 | 0.26 | 58.50 | 282 | 52.13 | 0.19 |
| S9 | Aman Khan Kaka Khail | 65 | 9.48 | 12.02 | 3.05 | 0.26 | 49.30 | 233 | 49.82 | 0.19 |
| S2 | Zafar Memon DHA | 222 | 5.84 | 9.59 | 0.67 | 0.24 | 84.70 | 412 | 32.57 | 0.18 |
| S5 | Gulistan-e-Johar | 149 | 5.24 | 7.42 | -1.43 | 0.34 | 77.90 | 383 | 48.25 | 0.11 |
| S3 | Urban Resource Center | 169 | 5.24 | 7.91 | -2.18 | 0.28 | 85.00 | 409 | 49.79 | 0.11 |
| S6 | G3 Engineering Consultants | 172 | 4.99 | 9.71 | -0.28 | 0.25 | 78.20 | 362 | 37.97 | 0.13 |
| S4 | Aga Khan University Main Campus | 203 | 4.89 | 7.08 | -0.02 | 0.21 | 80.80 | 389 | 35.44 | 0.14 |
| S8 | Malir Cantonment, Askari 5 | 108 | 4.61 | 6.97 | 0.90 | 0.27 | 57.70 | 243 | 40.90 | 0.11 |
| S1 | Climate Action Center | 189 | 4.59 | 6.78 | 0.08 | 0.37 | 85.60 | 407 | 42.88 | 0.11 |

## Per-sensor classification safety (regress-then-band)

| sensor_id | site | n | accuracy | macro_f1 | miss_rate | false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- |
| S4 | Aga Khan University Main Campus | 203 | 0.84 | 0.45 | 0.20 | 0.03 |
| S8 | Malir Cantonment, Askari 5 | 108 | 0.87 | 0.47 | 0.15 | 0.04 |
| S7 | NED University, Main Campus | 66 | 0.70 | 0.43 | 0.12 | 0.23 |
| S2 | Zafar Memon DHA | 222 | 0.82 | 0.62 | 0.11 | 0.02 |
| S6 | G3 Engineering Consultants | 172 | 0.89 | 0.50 | 0.09 | 0.02 |
| S1 | Climate Action Center | 189 | 0.89 | 0.49 | 0.08 | 0.07 |
| S10 | DG House DHA Phase 5 | 52 | 0.71 | 0.29 | 0.08 | 0.36 |
| S3 | Urban Resource Center | 169 | 0.77 | 0.48 | 0.07 | 0.02 |
| S9 | Aman Khan Kaka Khail | 65 | 0.72 | 0.44 | 0.06 | 0.17 |
| S5 | Gulistan-e-Johar | 149 | 0.84 | 0.48 | 0.05 | 0.05 |
| S13 | Urban Unit Karachi office | 14 | 0.79 | 0.34 | 0.00 | 0.25 |
| S11 | Orangi Pilot Project | 14 | 0.79 | 0.32 | 0.00 | 1.00 |
| S12 | Karachi Zoo | 35 | 0.86 | 0.32 | 0.00 | 0.50 |

## Notes

- `skill` = 1 - MAE_model / MAE_persistence (0 = no better than the naive forecast).
- `normalised_mae` = MAE / mean PM2.5, so a wetter/dirtier site is not penalised purely for being dirtier.
- `miss_rate` is NaN at a sensor with no Unhealthy-or-worse day in this window (not an error).
