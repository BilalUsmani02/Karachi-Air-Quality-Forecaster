"""The rule base, as data (PROJECT_SPEC.md §7; `docs/report/knowledge_sources.md` §5).

Every `Rule` is a frozen dataclass, not a function: its `conditions` are a tuple of simple
`(fact, operator, value)` triples that `matches()` below can evaluate *and* that `explain.py`
can print back out, so nothing about why a rule fired is hidden inside a lambda.

R01-R20 are quoted or closely paraphrased from EPA's *Air Quality Guide for Particle
Pollution* and AQI Technical Assistance Document (sources [1][2] in knowledge_sources.md) —
wording is kept faithful, since this is cited guidance, not free paraphrase. R18-R20 (trend)
and R22 (disclaimer) are project design rules, flagged as such via their `source` field.

R21 (the safety margin) is a *meta* rule: its job is to derive a new fact (`advice_band`,
`near_boundary`) rather than to pick advice text, so it has no `advice` and is fired specially
by `engine.py` rather than through the generic advice contest. It is still listed here, with
genuinely inspectable conditions, so the explanation facility can cite it like any other rule.

Rule kinds:
    "advice"  R01-R17 — candidates for the one main recommendation. The engine picks exactly
              one winner per forecast (`engine.py`'s conflict resolution).
    "append"  R18-R20, R22 — notes that are always added on top of the winning advice when
              their conditions match (trend notes) or unconditionally (the disclaimer).
    "meta"    R21 — derives `advice_band` / `near_boundary`; never a candidate itself.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Mapping

# --- Conditions: simple, inspectable (fact, operator, value) triples -----------------------

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "in": lambda actual, allowed: actual in allowed,
}


@dataclass(frozen=True)
class Condition:
    """One predicate over the working memory: `fact <operator> value`, e.g. `("advice_band", "==", 2)`."""

    fact: str
    op: str
    value: Any

    def holds(self, working_memory: Mapping[str, Any]) -> bool:
        """True if `fact` is present in `working_memory` and satisfies the operator.

        A fact that has not been derived yet (e.g. `advice_band` before R21 has run) is
        treated as "does not match" rather than an error — this is what lets forward chaining
        make a rule wait for an earlier round without the engine needing to know the order
        rules were written in.
        """
        if self.fact not in working_memory:
            return False
        return _OPS[self.op](working_memory[self.fact], self.value)

    def __str__(self) -> str:
        return f"{self.fact} {self.op} {self.value!r}"


@dataclass(frozen=True)
class Rule:
    """One row of the rule base. `conditions` ANDed together; `()` means "always matches"."""

    id: str
    name: str
    conditions: tuple[Condition, ...]
    kind: str  # "advice" | "append" | "meta"
    advice: str | None
    severity: int
    source: str
    because: str  # plain-English template for explain.py, e.g. "the band is Good"

    @property
    def specificity(self) -> int:
        """Number of conditions — "most specific rule wins" (§7) is literally this count."""
        return len(self.conditions)

    def matches(self, working_memory: Mapping[str, Any]) -> bool:
        return all(condition.holds(working_memory) for condition in self.conditions)


def C(fact: str, op: str, value: Any) -> Condition:  # noqa: N802 - short alias, used a lot below
    return Condition(fact, op, value)


# --- R01-R17: main advice candidates, one band/sensitive/outdoor combination each ----------
# Conditions are written against `advice_band`, the *post-margin* band (engine.py), not the
# raw forecast band — this is what makes R21's margin actually change which advice wins.

ADVICE_RULES: tuple[Rule, ...] = (
    Rule(
        "R01", "good_everyone",
        (C("advice_band", "==", 0),),
        "advice", "It's a great day to be active outside.", 1, "[2]",
        "the advice band is Good, and EPA says that's a great day to be active outside",
    ),
    Rule(
        "R02", "moderate_not_sensitive",
        (C("advice_band", "==", 1), C("sensitive", "==", False)),
        "advice", "It's a good day to be active outside.", 1, "[2]",
        "the advice band is Moderate and you are not in a sensitive group",
    ),
    Rule(
        "R03", "moderate_sensitive",
        (C("advice_band", "==", 1), C("sensitive", "==", True)),
        "advice",
        "Consider making outdoor activities shorter and less intense. Watch for symptoms "
        "such as coughing or shortness of breath — these are signs to take it easier.",
        2, "[1][2]",
        "the advice band is Moderate and you are in a sensitive group",
    ),
    Rule(
        "R04", "moderate_sensitive_outdoor",
        (C("advice_band", "==", 1), C("sensitive", "==", True), C("outdoor_activity", "==", True)),
        "advice",
        "Shorten your outdoor activity and make it less intense, or exercise away from busy roads.",
        2, "[2]",
        "the advice band is Moderate, you are sensitive, and you have outdoor plans",
    ),
    Rule(
        "R05", "usg_not_sensitive",
        (C("advice_band", "==", 2), C("sensitive", "==", False)),
        "advice",
        "No specific precaution needed for most people at this level; sensitive groups "
        "should take it easier (see below).",
        1, "[1][2]",
        "the advice band is Unhealthy for Sensitive Groups and you are not sensitive",
    ),
    Rule(
        "R06", "usg_sensitive",
        (C("advice_band", "==", 2), C("sensitive", "==", True)),
        "advice",
        "Make outdoor activities shorter and less intense. It's OK to be active outdoors, "
        "but take more breaks. Watch for symptoms such as coughing or shortness of breath.",
        3, "[2]",
        "the advice band is Unhealthy for Sensitive Groups and you are sensitive",
    ),
    Rule(
        "R07", "usg_sensitive_outdoor",
        (C("advice_band", "==", 2), C("sensitive", "==", True), C("outdoor_activity", "==", True)),
        "advice",
        "Make your planned activity shorter and less intense, take more breaks, and watch "
        "for symptoms such as coughing or shortness of breath.",
        3, "[2]",
        "the advice band is USG, you are sensitive, and you have outdoor plans",
    ),
    Rule(
        "R08", "usg_not_sensitive_outdoor",
        (C("advice_band", "==", 2), C("sensitive", "==", False), C("outdoor_activity", "==", True)),
        "advice",
        "Choose a less intense activity or shorten it; exercise away from busy roads.",
        2, "[2]",
        "the advice band is USG, you are not sensitive, and you have outdoor plans",
    ),
    Rule(
        "R09", "unhealthy_not_sensitive",
        (C("advice_band", "==", 3), C("sensitive", "==", False)),
        "advice",
        "Reduce long or intense activities. Take more breaks during outdoor activities.",
        3, "[2]",
        "the advice band is Unhealthy and you are not sensitive",
    ),
    Rule(
        "R10", "unhealthy_sensitive",
        (C("advice_band", "==", 3), C("sensitive", "==", True)),
        "advice",
        "Avoid long or intense outdoor activities. Consider rescheduling or moving "
        "activities indoors.",
        4, "[2]",
        "the advice band is Unhealthy and you are sensitive",
    ),
    Rule(
        "R11", "unhealthy_not_sensitive_outdoor",
        (C("advice_band", "==", 3), C("sensitive", "==", False), C("outdoor_activity", "==", True)),
        "advice",
        "Shorten and reduce the intensity of your planned activity, or reschedule it.",
        3, "[2]",
        "the advice band is Unhealthy, you are not sensitive, and you have outdoor plans",
    ),
    Rule(
        "R12", "unhealthy_sensitive_outdoor",
        (C("advice_band", "==", 3), C("sensitive", "==", True), C("outdoor_activity", "==", True)),
        "advice",
        "Move your planned activity indoors or reschedule it to a day with better air quality.",
        4, "[2]",
        "the advice band is Unhealthy, you are sensitive, and you have outdoor plans",
    ),
    Rule(
        "R13", "very_unhealthy_not_sensitive",
        (C("advice_band", "==", 4), C("sensitive", "==", False)),
        "advice",
        "Avoid long or intense activities. Consider rescheduling or moving activities indoors.",
        4, "[2]",
        "the advice band is Very Unhealthy and you are not sensitive",
    ),
    Rule(
        "R14", "very_unhealthy_sensitive",
        (C("advice_band", "==", 4), C("sensitive", "==", True)),
        "advice",
        "Avoid all physical activity outdoors. Reschedule to a time when air quality is "
        "better or move activities indoors.",
        5, "[2]",
        "the advice band is Very Unhealthy and you are sensitive",
    ),
    Rule(
        "R15", "very_unhealthy_or_hazardous_outdoor",
        (C("advice_band", "in", (4, 5)), C("outdoor_activity", "==", True)),
        "advice",
        "Cancel or move your planned outdoor activity indoors.",
        5, "[2]",
        "the advice band is Very Unhealthy or Hazardous and you have outdoor plans",
    ),
    Rule(
        "R16", "hazardous_not_sensitive",
        (C("advice_band", "==", 5), C("sensitive", "==", False)),
        "advice", "Avoid all physical activity outdoors.", 5, "[2]",
        "the advice band is Hazardous and you are not sensitive",
    ),
    Rule(
        "R17", "hazardous_sensitive",
        (C("advice_band", "==", 5), C("sensitive", "==", True)),
        "advice",
        "Remain indoors and keep activity levels low. Follow tips for keeping particle "
        "levels low indoors.",
        5, "[2]",
        "the advice band is Hazardous and you are sensitive",
    ),
)

# --- R18-R20: trend notes, appended on top of the winning advice ("avoid alert fatigue", §7) -

APPEND_RULES: tuple[Rule, ...] = (
    Rule(
        "R18", "trend_rising_usg",
        (C("trend", "==", "rising"), C("advice_band", "==", 2)),
        "append",
        "Conditions are forecast to worsen compared with the day before — plan ahead.",
        3, "§7 design rule",
        "the trend is rising and the advice band is USG",
    ),
    Rule(
        "R19", "trend_rising_unhealthy_or_worse",
        (C("trend", "==", "rising"), C("advice_band", "in", (3, 4, 5))),
        "append",
        "Air quality is forecast to be worse than the day before. If you're in a sensitive "
        "group, consider moving outdoor plans indoors.",
        4, "§7 design rule",
        "the trend is rising and the advice band is Unhealthy or worse",
    ),
    Rule(
        "R20", "trend_falling_usg_or_worse",
        (C("trend", "==", "falling"), C("advice_band", ">=", 2)),
        "append",
        "Air quality is forecast to improve compared with the day before.",
        1, "§7 design rule",
        "the trend is falling and the advice band is USG or worse",
    ),
)

# --- R21: the safety margin (meta rule — derives facts, chosen specially by engine.py) -----
# `band` and `distance_to_next_boundary` are seeded into working memory straight from the
# forecast (engine.py), exactly like `band` itself — no different from any other input-derived
# fact. What makes this genuinely a *forward-chaining step* is that `advice_band`, the fact
# every advice rule (R01-R20) actually conditions on, only exists in working memory once this
# rule has been evaluated: band ∈ [1, 4] keeps "Good" (0) and "Hazardous" (5) un-bumped, per the
# §7 safeguard ("never push into a worse band from Good→Moderate, so 'Good' advice still
# exists" — Phase 5 found a blanket margin erases it, see docs/report/model_selection.md).
MARGIN_UG_M3 = 5.0

R21 = Rule(
    "R21", "safety_margin",
    (C("band", ">=", 1), C("band", "<=", 4), C("distance_to_next_boundary", "<=", MARGIN_UG_M3)),
    "meta", None, 5, "§7 safeguard (err on the safe side)",
    "the forecast is within 5 µg/m³ of the next band's boundary, moving into USG or worse",
)

# --- R22: disclaimer, appended to every panel, unconditionally -----------------------------

R22 = Rule(
    "R22", "disclaimer",
    (),
    "append",
    "This is general public-health guidance based on EPA and WHO air-quality guidance. "
    "It is informational only — not an official air-quality alert and not personal medical "
    "advice. If you have a heart or lung condition, follow your own doctor's advice.",
    0, "§7 safeguard",
    "every panel always carries the disclaimer",
)

ALL_RULES: tuple[Rule, ...] = ADVICE_RULES + APPEND_RULES + (R21, R22)
RULES_BY_ID: dict[str, Rule] = {rule.id: rule for rule in ALL_RULES}
