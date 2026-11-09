"""The explanation facility (PROJECT_SPEC.md §7: "show which rules fired and why").

`explain(rec)` turns a `Recommendation` into a plain-English "You see this because…" trace:
the input facts, every rule that fired (its conditions, the facts that satisfied them, and
its source), whether the safety margin changed the advice band, and which advice rule won
the conflict and why. `to_dict(rec)` gives the same information as JSON-serialisable data,
for the Streamlit app.
"""

from __future__ import annotations

from typing import Any

from src.expert.engine import FiredRule, Recommendation


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _fired_rule_line(fired_rule: FiredRule) -> str:
    rule = fired_rule.rule
    conditions_text = (
        ", ".join(str(condition) for condition in rule.conditions) if rule.conditions else "always"
    )
    facts_text = ", ".join(f"{name}={_format_value(value)}" for name, value in fired_rule.matched_facts)
    facts_suffix = f" (matched: {facts_text})" if facts_text else ""
    return (
        f"  - {rule.id} ({rule.name}) fired because {rule.because}. "
        f"Conditions: {conditions_text}{facts_suffix}. Source: {rule.source}."
    )


def explain(rec: Recommendation) -> str:
    """Build the full "You see this because…" trace for one recommendation."""
    wm = rec.working_memory
    lines: list[str] = ["You see this because…", ""]

    lines.append("Inputs:")
    lines.append(
        f"  - forecast PM2.5 = {_format_value(wm['forecast_pm25'])} µg/m³ "
        f"-> band = {rec.band_name}"
    )
    lines.append(f"  - today's PM2.5 = {_format_value(wm['today_pm25'])}")
    lines.append(f"  - sensitive group = {_format_value(wm['sensitive'])}")
    lines.append(f"  - outdoor activity planned = {_format_value(wm['outdoor_activity'])}")
    lines.append(f"  - trend vs today = {_format_value(wm['trend'])}")
    lines.append("")

    if wm["near_boundary"]:
        lines.append(
            "Safety margin: the forecast is within 5 µg/m³ of the next band's lower "
            f"boundary, so the advice below uses the more cautious {_band_name_safe(wm['advice_band'])} "
            "band instead of the displayed band (R21, §7 safeguard: \"near a boundary — showing "
            "the more cautious advice\")."
        )
    else:
        lines.append("Safety margin: not applied — the forecast is not close enough to the next band.")
    lines.append("")

    lines.append(f"Rules that fired ({len(rec.fired)}):")
    for fired_rule in rec.fired:
        lines.append(_fired_rule_line(fired_rule))
    lines.append("")

    lines.append(f"Winning advice rule: {rec.chosen.id} ({rec.chosen.name}).")
    lines.append(f"  Why it won: {rec.chosen_reason}")
    lines.append(f"  Advice shown: \"{rec.advice}\"")
    if rec.notes:
        lines.append("Additional notes:")
        for note in rec.notes:
            lines.append(f"  - {note}")
    lines.append("")
    lines.append(f"Disclaimer (always shown): \"{rec.disclaimer}\"")

    return "\n".join(lines)


def _band_name_safe(band_idx: int) -> str:
    from src.classify.bands import band_name

    return band_name(band_idx)


def to_dict(rec: Recommendation) -> dict[str, Any]:
    """A JSON-serialisable view of `rec`, for the Streamlit app and tests."""
    wm = rec.working_memory
    return {
        "band_index": rec.band_index,
        "band_name": rec.band_name,
        "color": rec.color,
        "advice": rec.advice,
        "notes": list(rec.notes),
        "disclaimer": rec.disclaimer,
        "near_boundary": bool(wm["near_boundary"]),
        "advice_band": int(wm["advice_band"]),
        "advice_band_name": _band_name_safe(int(wm["advice_band"])),
        "trend": wm["trend"],
        "chosen_rule_id": rec.chosen.id,
        "chosen_reason": rec.chosen_reason,
        "fired": [
            {
                "id": fired_rule.rule.id,
                "name": fired_rule.rule.name,
                "kind": fired_rule.rule.kind,
                "severity": fired_rule.rule.severity,
                "source": fired_rule.rule.source,
                "conditions": [str(condition) for condition in fired_rule.rule.conditions],
                "matched_facts": {name: value for name, value in fired_rule.matched_facts},
            }
            for fired_rule in rec.fired
        ],
    }
