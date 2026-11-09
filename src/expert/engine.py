"""Forward-chaining inference engine (PROJECT_SPEC.md §7).

Starting from `Facts`, the engine builds a working memory (a plain dict of fact name ->
value) and repeatedly asks every rule in `rules.ALL_RULES` whether its conditions now match,
adding whatever a newly-matching rule concludes, until a round adds nothing new (a
*fixpoint*). Two things are added along the way that no other rule could match on from the
start, so a real chain of two rounds is needed, not just one:

    Round 0 (seeded directly from the forecast, no rule needed — `band` works the same way):
        trend                      -- rising/falling/steady vs today (or None)
        distance_to_next_boundary  -- how far the forecast sits below the next band's lower
                                      boundary (infinite if there is no next band)
    Round 1 (R21 matches on `band` + `distance_to_next_boundary` from round 0):
        near_boundary, advice_band -- R21's conclusion; R01-R20 cannot match before this
    Round 2 (R01-R22 now have `advice_band` to condition on):
        every advice/append rule whose conditions hold fires
    Round 3: nothing new matches -> fixpoint reached, loop stops.

This is why the band-boundary safety margin (R21) genuinely has to run as part of the same
forward-chaining loop as the advice rules, instead of as a pre-processing step bolted on
before "real" inference starts.

Conflict resolution among the fired *advice* rules (R01-R17): the rule with the most matching
conditions wins (§7: "most-specific rule wins"); ties go to the higher-severity rule ("safest
advice wins"); the very rare remaining tie — only possible for the two worst bands when
`outdoor_activity=True`, where R15's band-or-worse-plus-outdoor rule ties with that band's
sensitive-status rule on both counts — goes to whichever rule also checks `sensitive`, since
EPA structures its own guidance primarily around sensitive-group status with activity as a
secondary modifier (see docs/report/expert_system.md). A rule id tiebreak is kept underneath
that for full determinism, though the rule set above never actually needs it.

Append rules (R18-R20 trend notes, R22 disclaimer) are never part of that contest: every one
whose conditions match is added as a note, and R22 (conditions `()`) always matches.

The safety margin changes the band used to pick *advice* (`advice_band`) but never the band
*shown* as the headline (`Recommendation.band_index` stays the raw forecast band) — Phase 5's
tuning found that adding the margin before banding erased "Good" forecasts and raised false
alarms (docs/report/model_selection.md's "Safety margin" section), so the margin belongs only
here, in the advice layer, exactly as that section specifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from src.classify.bands import BANDS, band_name
from src.expert.facts import Facts
from src.expert.rules import ALL_RULES, MARGIN_UG_M3, Rule

MAX_ITERATIONS = 10

# Trend fires only on a change that is big enough to matter either in absolute or relative
# terms — whichever threshold is reached first. This avoids calling a 3 -> 3.2 ug/m3 wobble on
# a clean day "rising", while still catching a slow percentage drift on a polluted one.
# Documented and tested per PROJECT_SPEC.md §7's "choose a threshold, document, test".
TREND_ABS_UG_M3 = 5.0
TREND_REL_FRACTION = 0.10

# EPA AQI colour names (knowledge_sources.md §1, citing Table 1 of the AQI Technical
# Assistance Document [1]), indexed by band.
EPA_COLORS: dict[int, str] = {
    0: "#00E400",  # Good - green
    1: "#FFFF00",  # Moderate - yellow
    2: "#FF7E00",  # Unhealthy for Sensitive Groups - orange
    3: "#FF0000",  # Unhealthy - red
    4: "#8F3F97",  # Very Unhealthy - purple
    5: "#7E0023",  # Hazardous - maroon
}


@dataclass(frozen=True)
class FiredRule:
    """A record of one rule firing, kept for the explanation facility (`explain.py`)."""

    rule: Rule
    matched_facts: tuple[tuple[str, Any], ...]  # (fact name, value in working memory) pairs


@dataclass(frozen=True)
class Recommendation:
    """The engine's output: one headline band, one main piece of advice, fully traceable."""

    band_index: int  # the raw forecast band — the margin never changes what is *shown*
    band_name: str
    color: str
    advice: str
    notes: tuple[str, ...]
    disclaimer: str
    fired: tuple[FiredRule, ...]
    chosen: Rule
    chosen_reason: str
    working_memory: Mapping[str, Any]


def _derive_trend(today_pm25: float | None, forecast_pm25: float) -> str | None:
    """Rising/falling/steady vs today, or None if `today_pm25` wasn't given."""
    if today_pm25 is None:
        return None
    delta = forecast_pm25 - today_pm25
    threshold = TREND_ABS_UG_M3
    if today_pm25 > 0:
        threshold = max(TREND_ABS_UG_M3, today_pm25 * TREND_REL_FRACTION)
    if delta >= threshold:
        return "rising"
    if delta <= -threshold:
        return "falling"
    return "steady"


def _distance_to_next_boundary(band_idx: int, forecast_pm25: float) -> float:
    """How far below the next band's lower boundary the forecast sits (§7: "within ±5 µg/m³
    below the next band's lower boundary"); infinite if `band_idx` is the top band (no next)."""
    if band_idx >= len(BANDS) - 1:
        return math.inf
    return BANDS[band_idx + 1].low - forecast_pm25


def _references_sensitive(rule: Rule) -> bool:
    return any(condition.fact == "sensitive" for condition in rule.conditions)


def _matched_facts(rule: Rule, working_memory: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((condition.fact, working_memory[condition.fact]) for condition in rule.conditions)


def _choose_advice(candidates: list[Rule]) -> tuple[Rule, str]:
    """Pick the winning advice rule and explain why (see module docstring for the ordering)."""
    if not candidates:
        raise RuntimeError(
            "no advice rule fired — every band/sensitive/outdoor_activity combination should "
            "match at least one of R01-R17; this means the rule base has a gap"
        )
    ranked = sorted(
        candidates,
        key=lambda r: (-r.specificity, -r.severity, -int(_references_sensitive(r)), r.id),
    )
    winner = ranked[0]
    if len(ranked) == 1:
        return winner, f"{winner.id} was the only advice rule whose conditions matched."
    runner_up = ranked[1]
    if winner.specificity != runner_up.specificity:
        reason = (
            f"{winner.id} beat {runner_up.id}: {winner.specificity} matching conditions vs "
            f"{runner_up.specificity} — the most-specific rule wins."
        )
    elif winner.severity != runner_up.severity:
        reason = (
            f"{winner.id} and {runner_up.id} matched the same number of conditions "
            f"({winner.specificity}); {winner.id} won on higher severity "
            f"({winner.severity} vs {runner_up.severity}) — ties go to the safest advice."
        )
    else:
        reason = (
            f"{winner.id} and {runner_up.id} tied on both specificity ({winner.specificity}) "
            f"and severity ({winner.severity}); {winner.id} won because it checks sensitive-"
            "group status directly, which this rule base treats as the primary axis of EPA "
            "guidance (see docs/report/expert_system.md)."
        )
    return winner, reason


def run(facts: Facts) -> Recommendation:
    """Run forward chaining to a fixpoint and build the final `Recommendation`."""
    working_memory: dict[str, Any] = {
        "forecast_pm25": facts.forecast_pm25,
        "today_pm25": facts.today_pm25,
        "sensitive": facts.sensitive,
        "outdoor_activity": facts.outdoor_activity,
        "band": facts.band,
        "trend": _derive_trend(facts.today_pm25, facts.forecast_pm25),
        "distance_to_next_boundary": _distance_to_next_boundary(facts.band, facts.forecast_pm25),
    }

    fired: list[FiredRule] = []
    fired_ids: set[str] = set()
    advice_band_set = False

    for _ in range(MAX_ITERATIONS):
        newly_matched = [
            rule for rule in ALL_RULES if rule.id not in fired_ids and rule.matches(working_memory)
        ]
        margin_rule = next((rule for rule in newly_matched if rule.kind == "meta"), None)

        if margin_rule is None and not advice_band_set:
            # R21 didn't match (not near a qualifying boundary) — advice_band still has to
            # exist before any R01-R22 condition can be checked, so the engine supplies the
            # default itself: no margin, no change. This is the "no" branch of the same
            # conclusion R21 would have added had it matched.
            working_memory = {**working_memory, "advice_band": working_memory["band"]}
            advice_band_set = True
            continue  # let the next round see R01-R22 match against the now-complete memory

        if margin_rule is not None:
            working_memory = {
                **working_memory,
                "advice_band": working_memory["band"] + 1,
                "near_boundary": True,
            }
            advice_band_set = True

        if not newly_matched:
            break  # fixpoint: nothing new this round

        for rule in newly_matched:
            fired_ids.add(rule.id)
            fired.append(FiredRule(rule=rule, matched_facts=_matched_facts(rule, working_memory)))
    else:
        raise RuntimeError("forward chaining did not reach a fixpoint within MAX_ITERATIONS")

    working_memory.setdefault("near_boundary", False)
    working_memory.setdefault("advice_band", working_memory["band"])

    advice_candidates = [fr.rule for fr in fired if fr.rule.kind == "advice"]
    chosen, chosen_reason = _choose_advice(advice_candidates)

    notes: list[str] = []
    if working_memory["near_boundary"]:
        worse_band = band_name(working_memory["advice_band"])
        notes.append(f"Near a boundary — showing the more cautious advice for {worse_band}.")
    notes.extend(
        fr.rule.advice
        for fr in fired
        if fr.rule.kind == "append" and fr.rule.id != "R22" and fr.rule.advice is not None
    )
    disclaimer = next(fr.rule.advice for fr in fired if fr.rule.id == "R22")

    return Recommendation(
        band_index=facts.band,
        band_name=band_name(facts.band),
        color=EPA_COLORS[facts.band],
        advice=chosen.advice or "",
        notes=tuple(notes),
        disclaimer=disclaimer or "",
        fired=tuple(fired),
        chosen=chosen,
        chosen_reason=chosen_reason,
        working_memory=MappingProxyType(working_memory),
    )
