# The expert-system layer (Phase 6)

How the rule-based advice layer works, for the AIES report. Code: `src/expert/`. Tests:
`tests/test_expert_*.py`. This is the "Expert System" half of the AIES submission — the band
classifier (§6.4) decides *what the air is like*; this layer decides *what to tell someone
about it*, and shows exactly why.

## 1. Knowledge sources

Two independent sources feed the rule base (`docs/report/knowledge_sources.md`):

1. **US EPA** — *Technical Assistance Document for the Reporting of Daily Air Quality (AQI)*
   and the *Air Quality Guide for Particle Pollution* leaflet. These give the band boundaries
   (already used by `src/classify/bands.py`) and the plain-language "what should I do" text
   per band, quoted rather than paraphrased, since cited health guidance should say what the
   source actually said.
2. **WHO 2021 Global Air Quality Guidelines** — context on how much stricter WHO's guidance
   is than EPA's bands (used in the limitations section below, not quoted into a rule, since
   it has no per-band "what to do" text of its own).

Every advice rule (R01–R17) carries a `source` field citing which document it came from. Two
small groups of rules are *not* a quote from either document, and are labelled as such:

- **R18–R20** (trend notes) and **R22** (disclaimer) are project design choices, implementing
  the spec's "avoid alert fatigue" and "always disclaim" safeguards (§7) — `source = "§7
  design rule"` / `"§7 safeguard"`.
- **R21** (the safety margin) implements the spec's own "err on the safe side" safeguard
  directly — `source = "§7 safeguard (err on the safe side)"`.

## 2. How a rule is represented

A `Rule` (`src/expert/rules.py`) is a frozen dataclass, not a function:

```python
Rule(
    id="R07", name="usg_sensitive_outdoor",
    conditions=(
        Condition("advice_band", "==", 2),
        Condition("sensitive", "==", True),
        Condition("outdoor_activity", "==", True),
    ),
    kind="advice",
    advice="Make your planned activity shorter and less intense, take more breaks, "
           "and watch for symptoms such as coughing or shortness of breath.",
    severity=3, source="[2]",
    because="the advice band is USG, you are sensitive, and you have outdoor plans",
)
```

Every condition is a plain `(fact, operator, value)` triple — never a lambda — so the
explanation facility (§5) can print exactly what was checked and what it matched against,
instead of just saying "a rule fired."

## 3. Facts

`Facts` (`src/expert/facts.py`) is what the engine is allowed to know, and nothing else:
`forecast_pm25`, `today_pm25` (optional, for trend), and two booleans, `sensitive` and
`outdoor_activity`. `band` is computed from `forecast_pm25` the moment `Facts` is built —
it's arithmetic, not inference. This is also the privacy safeguard from §7/§10: the profile
can never grow a third field or a named medical condition without a test in
`test_expert_facts.py` failing (`test_facts_has_no_fields_beyond_the_documented_set`).

## 4. Forward chaining: a worked example

The engine (`src/expert/engine.py`) keeps a *working memory* — a dict of fact name → value —
and repeatedly checks every rule against it, adding whatever a newly-matching rule concludes,
until a round adds nothing new (a **fixpoint**). Trace for `forecast_pm25=33.0,
today_pm25=25.0, sensitive=True, outdoor_activity=True`:

| Round | Working memory gains | Why |
|---|---|---|
| seed | `band=1` (Moderate), `trend="rising"`, `distance_to_next_boundary=2.5` | computed straight from the forecast, same as `band` |
| 1 | `near_boundary=True`, `advice_band=2` | **R21** matches: `band` is 1 (∈ [1,4]) and `distance_to_next_boundary` (2.5) ≤ 5 |
| 2 | — (no new facts, but rules now fire) | R06, R07, R18 and R22 can now match, because they condition on `advice_band`/`trend`, which only exist as of round 1 |
| 3 | nothing new | **fixpoint** — loop stops |

R01–R20 literally cannot match in round 0: their conditions reference `advice_band`, which
isn't in working memory yet (`Condition.holds` treats a missing fact as "no match", not an
error). That's what makes the margin a genuine forward-chaining step rather than a
pre-processing hack bolted on before "real" inference starts — proven directly in
`test_chain_forecast_to_band_to_near_boundary_to_advice_band_to_advice`
(`tests/test_expert_engine.py`).

A hard cap (`MAX_ITERATIONS = 10`, never reached in practice — this rule base converges in 3
rounds) guards against a rule base that never settles.

## 5. Conflict resolution

R01–R17 are candidates for the *one* main piece of advice; R18–R20 (trend) and R22
(disclaimer) are always appended on top when they match, never part of the contest.

Among the fired candidates:

1. **Most conditions wins** (§7: "most-specific rule wins"). E.g. for Moderate + sensitive +
   outdoor plans, R04 (3 conditions) beats R03 (2 conditions) even though both match.
2. **Tie → highest severity wins** ("safest advice wins"). E.g. Very Unhealthy + not sensitive
   + outdoor plans: R13 (severity 4) loses to R15 (severity 5), which explicitly covers
   cancelling outdoor plans.
3. **Still tied → the rule that also checks `sensitive` wins.** This is a genuine gap in the
   draft rule table (flagged in the final report, not silently patched): R15 ("cancel your
   outdoor plans" — conditions on `advice_band` and `outdoor_activity`, not `sensitive`) ties
   exactly on both specificity (2) and severity (5) with R14, R16 and R17 (the band's
   sensitive-status rule) whenever a sensitive or non-sensitive person in the two worst bands
   also has outdoor plans. EPA's own guide organises its advice primarily by sensitive-group
   status, with activity as a secondary modifier, so this tie is resolved in favour of the
   rule that checks `sensitive`. A final tiebreak on rule id exists underneath this for full
   determinism, though the 22-rule base never actually reaches it.

Every one of the 24 band × sensitive × outdoor combinations (6 bands including Hazardous,
since the expert layer reasons on the raw EPA band, not the merged 5-band classifier target)
is covered and resolves to exactly one winner —
`test_every_combination_yields_one_advice_with_source_and_disclaimer`.

## 6. The safety margin (R21) and why

`docs/report/model_selection.md`'s "Safety margin" section found that adding +5 µg/m³ before
banding *cuts missed Unhealthy days but erases "Good" entirely and raises false alarms* — so
the margin must live in the advice layer, not the displayed band. R21 implements exactly that
split:

- **What's shown** (`Recommendation.band_index` / `band_name` / `color`) is always the raw
  forecast band — never bumped.
- **What advice is chosen** uses `advice_band`, which R21 bumps up by one band when the
  forecast is within 5 µg/m³ *below* the next band's lower boundary — but only when band 1
  (Moderate) or worse is involved, i.e. the bump lands on USG (index 2) or worse. Good (0)
  never bumps into Moderate, so "it's a great day to be active outside" can still be shown —
  the one Phase-5 finding this rule is built to protect against.
- When it fires, a note is added: *"Near a boundary — showing the more cautious advice for
  \<worse band\>."*

Worked boundary cases (`tests/test_expert_engine.py`):

| Forecast | Band | Distance to next boundary | Margin? | Advice band |
|---|---|---|---|---|
| 8.0 | Good | 1.0 (to Moderate) | **No** — Good is excluded by design | Good |
| 30.5 | Moderate | 5.0 (to USG, exactly the threshold) | **Yes** (inclusive) | USG |
| 33.0 | Moderate | 2.5 (to USG) | **Yes** | USG |
| 120.0 | Unhealthy | 5.4 (to Very Unhealthy) | No — just outside ±5 | Unhealthy |
| 122.0 | Unhealthy | 3.5 (to Very Unhealthy) | **Yes** | Very Unhealthy |
| 300.0 | Hazardous | — (no band above it) | No — nothing to push into | Hazardous |

> **Note on the brief's own example:** the task brief that commissioned this layer suggested
> "120.0 → Very Unhealthy advice" as a margin test case. Under the official EPA breakpoints
> (`src/classify/bands.py`), 120.0 is 5.4 µg/m³ below Very Unhealthy's 125.5 lower boundary —
> just outside the ±5 margin — so it correctly stays Unhealthy advice. 122.0 is used instead
> to demonstrate the Unhealthy → Very Unhealthy push; both values are tested explicitly.

## 7. Trend (R18–R20)

`today_pm25` vs `forecast_pm25` gives `trend ∈ {"rising", "falling", "steady", None}`. A
change only counts as rising/falling once it clears **5 µg/m³ or 10% of today's value**,
whichever is reached first (`TREND_ABS_UG_M3`, `TREND_REL_FRACTION` in `engine.py`): this
stops a 3 → 3.2 µg/m³ wobble on a clean day from being flagged "rising", while still catching
a slow percentage drift on a heavily polluted one. `today_pm25 = None` (no reading available)
gives `trend = None` and no trend note — tested directly
(`test_no_today_pm25_means_no_trend_and_no_trend_notes`).

Trend notes are deliberately sparse — only for rising-into-USG-or-worse (R18/R19) or
falling-from-USG-or-worse (R20) — to satisfy the spec's "avoid alert fatigue... via the trend
rule rather than repeating the same warning daily" (§7).

## 8. The explanation facility

`explain(rec)` (`src/expert/explain.py`) builds a plain-English "You see this because…" trace:
the input facts, every rule that fired (its conditions, the facts that satisfied them, and
its source), whether the margin applied, and which advice rule won the contest and why. For
the task's own example (forecast 33.0, today 25.0, sensitive, outdoor plans):

```
You see this because…

Inputs:
  - forecast PM2.5 = 33.0 ug/m3 -> band = Moderate
  - the day before's PM2.5 = 25.0
  - sensitive group = yes
  - outdoor activity planned = yes
  - trend vs the day before = rising

Safety margin: the forecast is within 5 ug/m3 of the next band's lower boundary, so the
advice below uses the more cautious Unhealthy for Sensitive Groups band instead of the
displayed band (R21, §7 safeguard: "near a boundary - showing the more cautious advice").

Rules that fired (5):
  - R21 (safety_margin) fired because the forecast is within 5 ug/m3 of the next band's
    boundary, moving into USG or worse. Conditions: band >= 1, band <= 4,
    distance_to_next_boundary <= 5.0 (matched: band=1, band=1,
    distance_to_next_boundary=2.5). Source: §7 safeguard (err on the safe side).
  - R22 (disclaimer) fired because every panel always carries the disclaimer. ...
  - R06 (usg_sensitive) fired because the advice band is Unhealthy for Sensitive Groups and
    you are sensitive. Conditions: advice_band == 2, sensitive == True (matched:
    advice_band=2, sensitive=yes). Source: [2].
  - R07 (usg_sensitive_outdoor) fired because ... Source: [2].
  - R18 (trend_rising_usg) fired because the trend is rising and the advice band is USG. ...

Winning advice rule: R07 (usg_sensitive_outdoor).
  Why it won: R07 beat R06: 3 matching conditions vs 2 - the most-specific rule wins.
  Advice shown: "Make your planned activity shorter and less intense, take more breaks, and
  watch for symptoms such as coughing or shortness of breath."
Additional notes:
  - Near a boundary - showing the more cautious advice for Unhealthy for Sensitive Groups.
  - Conditions are forecast to worsen compared with the day before - plan ahead.

Disclaimer (always shown): "This is general public-health guidance based on EPA and WHO
air-quality guidance. It is informational only..."
```

`to_dict(rec)` gives the same content as a JSON-serialisable dict (band, advice, notes,
disclaimer, `near_boundary`, `advice_band`, `trend`, and the full `fired` list with each
rule's conditions and matched facts), for the Streamlit app to render as an expandable panel.

## 9. Limitations

- **Not medical advice.** Every panel carries the R22 disclaimer; the profile only ever
  records sensitive-group status and outdoor plans, never a diagnosis (§7, §10).
- **General guidance, not personalised risk.** The rules are EPA's own public guidance for
  broad groups, not a model of any individual's actual health risk.
- **WHO is stricter than the EPA bands used here.** WHO's annual guideline (5 µg/m³) is far
  below EPA's "Good" cut-off (9.0 µg/m³) — Karachi will likely exceed WHO's guideline on most
  days even when this system reports "Good" (`knowledge_sources.md` §3). Worth stating
  plainly in the app, not just the report.
- **The rule base has one genuine ambiguity**, resolved by a documented design choice rather
  than left to silently depend on dict/sort ordering: see §5's tie-break.
- **The 5 µg/m³ / 10% trend threshold and the ±5 µg/m³ margin are both fixed constants**,
  not tuned against live user feedback — they come from the spec's own suggested values and
  Phase 5's margin analysis, not from, e.g., a cost-sensitivity study of missed-vs-false-alarm
  outcomes for advice specifically (that study exists only for the *band*, in
  `model_selection.md`).
