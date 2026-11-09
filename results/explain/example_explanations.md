# Example explanations

Real forecast days from the feature table, explained with the model-agnostic SHAP explainer on the production ensemble (spec §8). SHAP values are in µg/m³, the model's output unit; feature values are shown in their original (unscaled) units.

One live explanation took **0.80 s** on this machine.

### A winter smog day — S11 on 2025-12-20

Forecast: **157.3 µg/m³** (base value 49.2 µg/m³, additivity error +0.000 µg/m³).

Top reasons:

- Yesterday's average PM2.5 was high (137.3 µg/m³), which raised the forecast by about 37 µg/m³.
- The PM2.5 level an hour ago was high (195.0 µg/m³), which raised the forecast by about 19 µg/m³.
- The forecast day's modelled PM2.5 (CAMS atmospheric model) was high (59.3 µg/m³), which raised the forecast by about 13 µg/m³.

| Feature | Raw value | SHAP (µg/m³) |
|---|---|---|
| yesterday's average PM2.5 | 137.32 µg/m³ | +37.50 |
| the PM2.5 level an hour ago | 195.00 µg/m³ | +19.14 |
| the forecast day's modelled PM2.5 (CAMS atmospheric model) | 59.30 µg/m³ | +13.35 |
| the forecast day's modelled nitrogen dioxide (traffic and industry pollution) | 54.96 µg/m³ | +11.69 |
| the average PM2.5 over the last three days | 120.65 µg/m³ | +6.93 |
| how much PM2.5 swung up and down yesterday | 63.89 µg/m³ | -6.57 |

### A clean monsoon day — S2 on 2025-09-10

Forecast: **5.0 µg/m³** (base value 49.2 µg/m³, additivity error -0.000 µg/m³).

Top reasons:

- Yesterday's average PM2.5 was low (13.8 µg/m³), which lowered the forecast by about 17 µg/m³.
- Yesterday's rainfall was high (86.0 mm), which lowered the forecast by about 7 µg/m³.
- The PM2.5 level an hour ago was low (4.3 µg/m³), which lowered the forecast by about 7 µg/m³.

| Feature | Raw value | SHAP (µg/m³) |
|---|---|---|
| yesterday's average PM2.5 | 13.84 µg/m³ | -17.49 |
| yesterday's rainfall | 86.00 mm | -6.60 |
| the PM2.5 level an hour ago | 4.28 µg/m³ | -6.55 |
| the forecast day's modelled PM2.5 (CAMS atmospheric model) | 18.32 µg/m³ | -4.99 |
| the average PM2.5 over the last three days | 19.72 µg/m³ | -2.20 |
| the forecast day's wind speed | 3.70 m/s | -1.80 |

### A forecast near a band boundary — S4 on 2026-03-28

Forecast: **9.0 µg/m³** (base value 49.2 µg/m³, additivity error -0.000 µg/m³).

Top reasons:

- Yesterday's average PM2.5 was low (20.5 µg/m³), which lowered the forecast by about 13 µg/m³.
- The forecast day's modelled PM2.5 (CAMS atmospheric model) was low (8.1 µg/m³), which lowered the forecast by about 8 µg/m³.
- The PM2.5 level an hour ago was low (3.9 µg/m³), which lowered the forecast by about 5 µg/m³.

| Feature | Raw value | SHAP (µg/m³) |
|---|---|---|
| yesterday's average PM2.5 | 20.50 µg/m³ | -12.76 |
| the forecast day's modelled PM2.5 (CAMS atmospheric model) | 8.06 µg/m³ | -7.53 |
| the PM2.5 level an hour ago | 3.90 µg/m³ | -5.41 |
| the forecast day's modelled nitrogen dioxide (traffic and industry pollution) | 6.76 µg/m³ | -3.89 |
| the forecast day's mixing height (how high pollution can spread) | 995.83 m | -2.37 |
| the forecast day's wind speed | 4.17 m/s | -2.10 |
