# Data exploration - Phase 1 findings

Generated from `notebooks/01_explore_data.ipynb`. See that notebook for the full analysis, figures and code.

## Coverage table

| Sensor | First reading | Last reading | % complete | Complete days (>=18h) | Longest gap (h) |
|---|---|---|---|---|---|
| S1 | 2025-07-02 | 2026-11-08 | 85.6% | 418 | 577 |
| S2 | 2025-06-21 | 2026-10-27 | 84.7% | 415 | 1343 |
| S3 | 2025-06-19 | 2026-11-08 | 85.0% | 414 | 433 |
| S4 | 2025-06-23 | 2026-10-24 | 80.8% | 391 | 651 |
| S5 | 2025-06-23 | 2026-11-08 | 77.9% | 387 | 1087 |
| S6 | 2025-07-07 | 2026-11-06 | 78.2% | 368 | 724 |
| S7 | 2025-06-23 | 2026-11-08 | 58.5% | 285 | 2351 |
| S8 | 2025-06-23 | 2026-11-07 | 57.7% | 266 | 988 |
| S9 | 2025-06-20 | 2026-11-07 | 49.3% | 241 | 2662 |
| S10 | 2025-06-21 | 2026-03-19 | 85.4% | 228 | 290 |
| S11 | 2025-06-24 | 2026-11-08 | 43.7% | 197 | 4776 |
| S12 | 2025-06-20 | 2026-02-24 | 74.8% | 173 | 287 |
| S13 | 2025-06-23 | 2026-11-08 | 35.9% | 166 | 5533 |
| S14 | 2025-06-19 | 2026-11-08 | 32.5% | 164 | 4558 |
| US_CONSULATE | 2019-05-23 | 2025-03-04 | 71.8% | 1454 | 1635 |

## Findings

### Coverage
- 14 sensors, 2025-06-19 to 2026-11-08.
- 4,113 complete sensor-days total (>=18 valid hours/day); mean completeness 66.4% of all expected hours.
- Best coverage: S1 (85.6%). Worst: S14 (32.5%).
- Longest single gap: S13, 5,533 hours (230.5 days).
- US Consulate: 1,454 complete days, 2019-05-23 to 2025-03-04 (no overlap with the sensor period).

### Faults (measured, not yet cleaned)
- Negatives: 0 across all sensors.
- Zeros: 5.
- Flat-line hours (>=6 h repeated value): 0.
- Spikes (>500 ug/m3 or >5x rolling median): 79.
- Hours with observed_count < expected_count: 0.

### Bands (daily means, >=18 h, EPA 2024 breakpoints)
- Sensor city total: Good 107, Moderate 2096, USG 758, Unhealthy 1049, Very Unhealthy 103, Hazardous 0, out of 4113 sensor-days.
- Hazardous is empty -> merge with Very Unhealthy for classification (spec §6.4).
- Good is present (107 sensor-days) -> check per-sensor training-split counts in Phase 2; merge with Moderate if <30 in the training split.
- US Consulate bands: {'Good': 8, 'Moderate': 805, 'USG': 222, 'Unhealthy': 355, 'Very Unhealthy': 63, 'Hazardous': 1}.

### Sensor agreement
- Pairwise daily-mean correlation between sensors: min 0.911, mean 0.959, max 0.992.
- CAMS spread across the 14 site coordinates: mean 0.0000 ug/m3, max 0.0000 ug/m3 -> confirms CAMS is effectively one city-wide value, not site-specific.
- CAMS vs sensor city-average daily PM2.5: r=0.733, mean bias (CAMS - sensors) = -14.5 ug/m3 over 505 days.

### Weather links (daily-mean PM2.5 vs daily-mean weather)
- temperature_2m: r=-0.652
- wind_speed_10m: r=-0.627
- relative_humidity_2m: r=-0.625
- boundary_layer_height: r=-0.483
- precipitation: r=-0.074

### Raw vs corrected
- PM2.5 vs relative humidity: overall r=-0.265 (n=101,988); RH>80% subset r=0.105 (n=29,674).
- Mean PM2.5 at RH>80% = 35.6 ug/m3 vs RH<=80% = 50.2 ug/m3 -> PM2.5 is actually LOWER at high RH; no sign of humidity inflation in the raw signal, but likely confounded by season (humid spells coincide with monsoon rain, which washes out PM2.5) rather than proof the sensors are already corrected.
- Consulate (2019-2025) vs sensors (2025-2026), same calendar months: sensors read lower on average by -2.8 ug/m3. Caveat: periods do not overlap, so this mixes any real year-to-year trend with sensor-vs-reference offset; treat as suggestive only.

### Recommended Phase 2 decisions
- Fault rules: drop negatives and spikes (>500 ug/m3 or >5x the 25 h rolling median) before imputing; treat flat-line runs >=24 h as missing; keep zeros and under-reported hours (down-weighted) as real.
- Gap handling: interpolate gaps up to a few hours; drop (and log) windows longer than the longest-gap threshold found above rather than filling them.
- Sensors to keep: all 14 (each already clears the >=150-day / >=18h-per-day bar used in spec §5.2); revisit S14 if its 32.5% completeness causes problems once lag features are built.
- Band merges: merge Hazardous into Very Unhealthy everywhere (0-0 days); re-check Good vs Moderate merge once the training split is fixed (<30 samples rule, spec §6.4).
- Train/val/test split (time-ordered, whole seasons kept together): train on the first full annual cycle, validate on the following full season, test on the most recent season, e.g. train 2025-06-19-2026-05-31, validation 2026-06-01-2026-08-31, test 2026-09-01-2026-11-08 (covers monsoon in validation and the start of winter smog season in test) - confirm exact cut-offs once Phase 2 feature lags are built (lags need lead-in data just before each split boundary).

