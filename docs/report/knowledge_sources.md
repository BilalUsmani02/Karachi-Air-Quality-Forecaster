# D4: Health-Guidance Knowledge Sources

Knowledge source for the expert-system layer (§7). Advice text is quoted or
closely paraphrased from EPA/WHO sources; every rule cites a source; the
user profile only ever records sensitive-group status, planned outdoor
activity, and trend — never a named medical condition (§7 safeguards).

## Corrections needed

**None.** The breakpoints in PROJECT_SPEC.md §6.4 were checked against
Table 6 ("Breakpoints for the AQI") of EPA's official *Technical Assistance
Document for the Reporting of Daily Air Quality — the Air Quality Index
(AQI)* [1], which carries the 2024 PM NAAQS revision (Good cut-off lowered
to 9.0 µg/m³, effective 6 May 2024). All six bands and boundaries in §6.4
match the primary source exactly, including the "Hazardous" band being
open-ended (≥ 225.5 µg/m³, with 225.5–325.4 µg/m³ corresponding to AQI
301–500 and no published upper limit beyond that).

| Band | PM2.5 (µg/m³), 24-h average | Spec §6.4 | Match |
|---|---|---|---|
| Good | 0.0–9.0 | 0.0–9.0 | ✅ |
| Moderate | 9.1–35.4 | 9.1–35.4 | ✅ |
| Unhealthy for Sensitive Groups | 35.5–55.4 | 35.5–55.4 | ✅ |
| Unhealthy | 55.5–125.4 | 55.5–125.4 | ✅ |
| Very Unhealthy | 125.5–225.4 | 125.5–225.4 | ✅ |
| Hazardous | ≥ 225.5 | ≥ 225.5 | ✅ |

---

## 1. EPA PM2.5 AQI breakpoints (2024 revision)

Confirmed above. Source: Table 6, *Technical Assistance Document for the
Reporting of Daily Air Quality — the Air Quality Index (AQI)*, U.S. EPA,
Office of Air Quality Planning and Standards [1]. The AQI colour names are
Good (green), Moderate (yellow), Unhealthy for Sensitive Groups (orange),
Unhealthy (red), Very Unhealthy (purple), Hazardous (maroon) — same
document, Table 1.

---

## 2. EPA per-band health guidance for PM2.5

### 2.1 Who counts as "sensitive" (official EPA list)

> "People with heart or lung disease; older adults; and children, including
> teenagers. In addition, some minority groups, people with lower incomes,
> and outdoor workers may experience higher exposure that can worsen
> underlying health conditions." — Table 4, *AQI Technical Assistance
> Document* [1]

> "Sensitive groups include people with heart disease, people with lung
> disease, including asthma and Chronic Obstructive Pulmonary Disease
> (COPD), older adults, children and teenagers, pregnant people, minority
> populations, and outdoor workers." — *Air Quality Guide for Particle
> Pollution*, AirNow/EPA, Feb 2023 [2]

Our expert system only asks "sensitive group: yes/no" (§7 safeguard) — it
never names which condition applies, consistent with this official list
being a group of *categories*, not individual diagnoses.

### 2.2 Official health statement / cautionary statements and activity guidance, by band

Quoted from the *Air Quality Guide for Particle Pollution* [2] (the plainlanguage public leaflet) and, for the "unusually sensitive" wording, from
Table 5 of the *AQI Technical Assistance Document* [1]:

| Band | Who needs to be concerned | What should I do |
|---|---|---|
| Good (0–50) | — | "It's a great day to be active outside." |
| Moderate (51–100) | "Some people who may be unusually sensitive to particle pollution." | **Unusually sensitive:** "Consider making outdoor activities shorter and less intense. Watch for symptoms such as coughing or shortness of breath. These are signs to take it easier." **Everyone else:** "It's a good day to be active outside." |
| Unhealthy for Sensitive Groups (101–150) | "Sensitive groups include people with heart or lung disease, older adults, children and teenagers, minority populations, and outdoor workers." | **Sensitive groups:** "Make outdoor activities shorter and less intense. It's OK to be active outdoors, but take more breaks. Watch for symptoms such as coughing or shortness of breath." |
| Unhealthy (151–200) | "Everyone" | **Sensitive groups:** "Avoid long or intense outdoor activities. Consider rescheduling or moving activities indoors." **Everyone else:** "Reduce long or intense activities. Take more breaks during outdoor activities." |
| Very Unhealthy (201–300) | "Everyone" | **Sensitive groups:** "Avoid all physical activity outdoors. Reschedule to a time when air quality is better or move activities indoors." **Everyone else:** "Avoid long or intense activities. Consider rescheduling or moving activities indoors." |
| Hazardous (301–500) | "Everyone" | **Everyone:** "Avoid all physical activity outdoors." **Sensitive groups:** "Remain indoors and keep activity levels low. Follow tips for keeping particle levels low indoors." |

General ways to reduce exposure on any poor-air-quality day, quoted from
the same leaflet [2]: "Choose a less intense activity. Shorten your
outdoor activities. Reschedule activities. Exercise away from busy roads."

People with asthma get one more official line (Table 5, [1]): "Follow your
asthma action plan and keep quick-relief medicine handy." This is **not**
used in our rule base, since it names a specific condition and profile is
broad-only (§7 safeguard) — flagged here so the project doesn't
accidentally reintroduce it later.

---

## 3. WHO 2021 Global Air Quality Guidelines for PM2.5

From *WHO global air quality guidelines: particulate matter (PM2.5 and
PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*,
Geneva: WHO; 2021 [3], Table 0.1 (p. xvii):

| | Interim target 1 | Interim target 2 | Interim target 3 | Interim target 4 | AQG level |
|---|---|---|---|---|---|
| **Annual**, µg/m³ | 35 | 25 | 15 | 10 | **5** |
| **24-hour**, µg/m³ | 75 | 50 | 37.5 | 25 | **15** |

The 24-hour level is defined as the 99th percentile of the annual
distribution of 24-h averages (≈ 3–4 exceedance days/year). WHO notes the
guideline level is "not a threshold of no effect" — any exposure above it
is associated with some increase in mortality risk, based on linear or
supralinear concentration–response functions down to very low
concentrations (WHO 2021, §3.2). The WHO annual AQG (5 µg/m³) is far
stricter than the EPA "Good" cut-off (9.0 µg/m³); this gap is worth noting
in the limitations/ethics section of the report — Karachi will likely
exceed the WHO guideline on most days even when the EPA band says "Good."

---

## 4. Pakistan NEQS for ambient PM2.5 (optional, verified)

Source: Pakistan Environmental Protection Agency (Pak-EPA) notification,
*National Environmental Quality Standards for Ambient Air*, hosted by the
Ministry of Climate Change [4]. This is the Pak-EPA's own published
standards table, listing values "effective from 1 January 2012" (alongside
superseded 2009 values):

| Averaging time | PM2.5 NEQS (µg/m³) |
|---|---|
| Annual average | 15 |
| 24-hour average | 35 |
| 1-hour average | 15 |

*Note:* the 1-hour value (15) being lower than the 24-hour value (35) is how the source table
reads; it is internally inconsistent and is widely noted as an anomaly. We do not use the
1-hour value anywhere — the project works on 24-hour averages.

**Caveat:** the document itself is headed as a *draft* circulated by Pak-EPA
for public comment before Pakistan Environmental Protection Council
approval, with no visible gazette/approval date in the PDF. The 2012-effective
values match what secondary literature (e.g. SDPI policy brief, legal-review
articles) consistently cites as Pakistan's current PM2.5 NEQS, so they are
very likely the standard in force — but I could not independently verify
the final approval date from a primary source. Use with that caveat if
precision on the adoption date matters; for the report's "≥2 datasets"
discussion this level of certainty is sufficient.

---

## 5. Draft rule table (expert system, forward-chaining)

Facts: `band` (Good/Moderate/USG/Unhealthy/VeryUnhealthy/Hazardous),
`sensitive` (yes/no), `outdoor_activity` (yes/no), `trend` (rising/falling/
steady vs today). Conflict resolution: most-specific rule wins; ties broken
by severity (1=lowest, 5=highest; safest/highest-severity wins). The
margin-of-boundary and disclaimer rules are meta-rules, not part of the
severity contest — the margin rule runs *before* banding, the disclaimer
runs *after* every other rule.

| ID | IF | THEN (advice) | Sev. | Source |
|---|---|---|---|---|
| R01 | band=Good | "It's a great day to be active outside." | 1 | [2] |
| R02 | band=Moderate, sensitive=no | "It's a good day to be active outside." | 1 | [2] |
| R03 | band=Moderate, sensitive=yes | "Consider making outdoor activities shorter and less intense. Watch for symptoms such as coughing or shortness of breath — these are signs to take it easier." | 2 | [1][2] |
| R04 | band=Moderate, sensitive=yes, outdoor_activity=yes | "Shorten your outdoor activity and make it less intense, or exercise away from busy roads." | 2 | [2] |
| R05 | band=USG, sensitive=no | "No specific precaution needed for most people at this level; sensitive groups should take it easier (see below)." | 1 | [1][2] |
| R06 | band=USG, sensitive=yes | "Make outdoor activities shorter and less intense. It's OK to be active outdoors, but take more breaks. Watch for symptoms such as coughing or shortness of breath." | 3 | [2] |
| R07 | band=USG, sensitive=yes, outdoor_activity=yes | "Make your planned activity shorter and less intense, take more breaks, and watch for symptoms such as coughing or shortness of breath." | 3 | [2] |
| R08 | band=USG, sensitive=no, outdoor_activity=yes | "Choose a less intense activity or shorten it; exercise away from busy roads." | 2 | [2] |
| R09 | band=Unhealthy, sensitive=no | "Reduce long or intense activities. Take more breaks during outdoor activities." | 3 | [2] |
| R10 | band=Unhealthy, sensitive=yes | "Avoid long or intense outdoor activities. Consider rescheduling or moving activities indoors." | 4 | [2] |
| R11 | band=Unhealthy, sensitive=no, outdoor_activity=yes | "Shorten and reduce the intensity of your planned activity, or reschedule it." | 3 | [2] |
| R12 | band=Unhealthy, sensitive=yes, outdoor_activity=yes | "Move your planned activity indoors or reschedule it to a day with better air quality." | 4 | [2] |
| R13 | band=VeryUnhealthy, sensitive=no | "Avoid long or intense activities. Consider rescheduling or moving activities indoors." | 4 | [2] |
| R14 | band=VeryUnhealthy, sensitive=yes | "Avoid all physical activity outdoors. Reschedule to a time when air quality is better or move activities indoors." | 5 | [2] |
| R15 | band=VeryUnhealthy OR Hazardous, outdoor_activity=yes | "Cancel or move your planned outdoor activity indoors." | 5 | [2] |
| R16 | band=Hazardous, sensitive=no | "Avoid all physical activity outdoors." | 5 | [2] |
| R17 | band=Hazardous, sensitive=yes | "Remain indoors and keep activity levels low. Follow tips for keeping particle levels low indoors." | 5 | [2] |
| R18 | trend=rising, band=USG | *(append)* "Conditions are forecast to worsen compared with the day before — plan ahead." | 3 | §7 design rule |
| R19 | trend=rising, band ∈ {Unhealthy, VeryUnhealthy, Hazardous} | *(append)* "Air quality is forecast to be worse than the day before. If you're in a sensitive group, consider moving outdoor plans indoors." | 4 | §7 design rule |
| R20 | trend=falling, band ≥ USG | *(append)* "Air quality is forecast to improve compared with the day before." | 1 | §7 design rule |
| R21 | forecast within ±5 µg/m³ of a band boundary | *(meta)* classify into the worse (higher-severity) of the two adjacent bands; app states "near a boundary — showing the more cautious band." | 5 | §7 safeguard ("err on the safe side") |
| R22 | always | *(append, every panel)* "This is general public-health guidance based on EPA and WHO air-quality guidance. It is informational only — not an official air-quality alert and not personal medical advice. If you have a heart or lung condition, follow your own doctor's advice." | — | §7 safeguard |

Notes on the table:
- R18–R20 (trend) implement the spec's "avoid alert fatigue... via the
  trend rule" safeguard (§7). Their *framing* ("plan ahead", "forecast to
  worsen") is a project design choice, not a direct EPA/WHO quote — flagged
  explicitly below.
- R21 implements the "err on the safe side" margin rule (§7) exactly as
  specified: within ±5 µg/m³ of a boundary, forecast margin tuned on
  validation error, use the worse band.
- 22 rules total, within the spec's 15–25 target. Every band has at least
  one general-public and one sensitive-group rule; outdoor-activity and
  trend variants are layered on top rather than cross-multiplied with every
  band, to keep the rule base legible.

---

## 6. Sources

1. United States Environmental Protection Agency, Office of Air Quality
   Planning and Standards. *Technical Assistance Document for the
   Reporting of Daily Air Quality — the Air Quality Index (AQI)*.
   Research Triangle Park, NC: US EPA. Reflects the 2024 PM NAAQS revision
   (effective 6 May 2024); Table 6 gives the PM2.5 breakpoints used in
   §6.4. https://document.airnow.gov/technical-assistance-document-forthe-reporting-of-daily-air-quailty.pdf — accessed 8 November 2026.

2. United States Environmental Protection Agency / AirNow.
   *Air Quality Guide for Particle Pollution*. EPA-452/F-23-002,
   February 2023. https://www.airnow.gov/sites/default/files/2023-03/airquality-guide-for-particle-pollution_0.pdf — accessed 8 November 2026.

3. World Health Organization. *WHO global air quality guidelines:
   particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur
   dioxide and carbon monoxide*. Geneva: World Health Organization; 2021.
   ISBN 978-92-4-003422-8 (electronic).
   https://iris.who.int/items/5b0ca7ab-830a-44b8-b99f-4ee743b4d7b2 —
   accessed 8 November 2026.

4. Government of Pakistan, Pakistan Environmental Protection Agency
   (Pak-EPA). *National Environmental Quality Standards for Ambient Air*
   (notification, values effective 1 January 2012), hosted by the
   Ministry of Climate Change. Approval/gazette date not independently
   confirmed — see caveat in §4.
   https://mocc.gov.pk/SiteImage/Misc/files/NEQS%20for%20Ambient%20Air.pdf
   — accessed 8 November 2026.
