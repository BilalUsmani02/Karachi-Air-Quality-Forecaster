"""Immutable input facts for the expert-system layer (PROJECT_SPEC.md §7).

`Facts` holds exactly what the rule engine is allowed to know about a person: the forecast,
today's reading (for the trend rules), and two broad yes/no profile flags. Nothing else.
This is a deliberate privacy safeguard (§7, §10) — the profile is never a named medical
condition, an age, a location, or anything else that could identify or categorise someone
beyond "do you count as a sensitive group" and "do you have outdoor plans".

`band` is not an extra profile fact: it is simply the EPA band (`src.classify.bands`) that
`forecast_pm25` falls into, computed once here so every rule and explanation refers to the
same number. The engine (`engine.py`) derives further facts — `trend`, `near_boundary`,
`advice_band` — from this starting point; those live in the engine's working memory, not here,
because they come from *inference*, not from input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.classify.bands import pm25_to_band


@dataclass(frozen=True)
class Facts:
    """One person's situation, as the expert system is allowed to see it.

    Raises:
        ValueError: `forecast_pm25` or `today_pm25` is negative (a negative PM2.5 reading is
            a cleaning error upstream, never a value this layer should reason about).
        TypeError: `sensitive` or `outdoor_activity` is not a plain bool (the privacy
            safeguard only holds if the profile can never silently grow a third, richer field).
    """

    forecast_pm25: float
    today_pm25: float | None = None
    sensitive: bool = False
    outdoor_activity: bool = False
    band: int = field(init=False)

    def __post_init__(self) -> None:
        if self.forecast_pm25 < 0:
            raise ValueError(f"forecast_pm25 cannot be negative: {self.forecast_pm25}")
        if self.today_pm25 is not None and self.today_pm25 < 0:
            raise ValueError(f"today_pm25 cannot be negative: {self.today_pm25}")
        if not isinstance(self.sensitive, bool):
            raise TypeError(f"sensitive must be a bool, got {type(self.sensitive).__name__}")
        if not isinstance(self.outdoor_activity, bool):
            raise TypeError(f"outdoor_activity must be a bool, got {type(self.outdoor_activity).__name__}")
        object.__setattr__(self, "band", pm25_to_band(self.forecast_pm25))


# The only two profile facts the expert system is ever allowed to ask for (§7 safeguard:
# "Broad groups only ... never a named medical condition"). `explain.py` and the tests check
# against this constant so the safeguard cannot silently erode as the system grows.
PROFILE_FIELDS: tuple[str, ...] = ("sensitive", "outdoor_activity")
