"""The expert-system layer (PROJECT_SPEC.md §7): a forward-chaining rule base that turns a
PM2.5 forecast and a two-field user profile into plain-English health advice, with a full
explanation of which rules fired and why.

`recommend()` is the one entry point the rest of the project (the Streamlit app, tests) needs:

    >>> from src.expert import recommend
    >>> rec = recommend(forecast_pm25=33.0, today_pm25=25.0, sensitive=True, outdoor_activity=True)
    >>> rec.band_name, rec.advice
    ('Moderate', '...')

See `src/expert/facts.py` (inputs), `rules.py` (the R01-R22 rule base, as data), `engine.py`
(forward chaining, the safety margin, conflict resolution) and `explain.py` (the "You see this
because…" trace) for how each piece works.
"""

from __future__ import annotations

from src.expert.engine import FiredRule, Recommendation, run
from src.expert.explain import explain, to_dict
from src.expert.facts import Facts

__all__ = ["Facts", "FiredRule", "Recommendation", "recommend", "explain", "to_dict", "run"]


def recommend(
    forecast_pm25: float,
    today_pm25: float | None = None,
    sensitive: bool = False,
    outdoor_activity: bool = False,
) -> Recommendation:
    """Run the expert system end to end: build `Facts`, forward-chain, return the `Recommendation`.

    Args:
        forecast_pm25: tomorrow's forecast 24-h mean PM2.5 (µg/m³).
        today_pm25: today's 24-h mean PM2.5, for the trend rules (R18-R20). `None` skips trend.
        sensitive: whether the person is in an EPA-defined sensitive group (§7 safeguard: a
            broad yes/no only — never a named medical condition).
        outdoor_activity: whether the person has outdoor plans.
    """
    facts = Facts(
        forecast_pm25=forecast_pm25,
        today_pm25=today_pm25,
        sensitive=sensitive,
        outdoor_activity=outdoor_activity,
    )
    return run(facts)
