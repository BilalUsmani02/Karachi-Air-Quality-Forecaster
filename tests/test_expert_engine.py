"""Tests for forward chaining, the safety margin, trend and conflict resolution
(`src.expert.engine`), PROJECT_SPEC.md §7.
"""

import dataclasses

import pytest

from src.expert import recommend
from src.expert.engine import MAX_ITERATIONS, Recommendation, run
from src.expert.facts import Facts
from src.expert.rules import ALL_RULES

# Representative, non-boundary forecasts for each of the six EPA bands (none within 5 ug/m3
# of the next band's lower boundary), so this matrix isn't accidentally testing the margin too.
BAND_FORECASTS = {0: 5.0, 1: 20.0, 2: 45.0, 3: 90.0, 4: 160.0, 5: 300.0}

# Expected winning advice rule for every band x sensitive x outdoor_activity combination,
# worked out by hand from docs/report/knowledge_sources.md §5's specificity/severity table.
EXPECTED_WINNER = {
    (0, False, False): "R01", (0, False, True): "R01", (0, True, False): "R01", (0, True, True): "R01",
    (1, False, False): "R02", (1, False, True): "R02", (1, True, False): "R03", (1, True, True): "R04",
    (2, False, False): "R05", (2, False, True): "R08", (2, True, False): "R06", (2, True, True): "R07",
    (3, False, False): "R09", (3, False, True): "R11", (3, True, False): "R10", (3, True, True): "R12",
    (4, False, False): "R13", (4, False, True): "R15", (4, True, False): "R14", (4, True, True): "R14",
    (5, False, False): "R16", (5, False, True): "R16", (5, True, False): "R17", (5, True, True): "R17",
}


# --- every band x sensitive x outdoor combination: exactly one main advice, source, disclaimer -


@pytest.mark.parametrize(
    "band_idx,sensitive,outdoor_activity",
    [(b, s, o) for b in range(6) for s in (False, True) for o in (False, True)],
)
def test_every_combination_yields_one_advice_with_source_and_disclaimer(band_idx, sensitive, outdoor_activity):
    rec = recommend(
        forecast_pm25=BAND_FORECASTS[band_idx], sensitive=sensitive, outdoor_activity=outdoor_activity
    )
    assert rec.band_index == band_idx
    assert rec.advice  # non-empty: exactly one piece of main advice
    assert rec.chosen.id == EXPECTED_WINNER[(band_idx, sensitive, outdoor_activity)]
    assert rec.chosen.advice == rec.advice
    assert rec.chosen.source  # every winning rule cites a source
    assert rec.disclaimer  # R22 always present


# --- tie-break: R15 (band + outdoor) vs the band's sensitive-status rule -------------------


@pytest.mark.parametrize(
    "band_idx,sensitive,winner,loser",
    [
        (4, True, "R14", "R15"),
        (5, False, "R16", "R15"),
        (5, True, "R17", "R15"),
    ],
)
def test_tie_break_prefers_the_sensitive_conditioned_rule(band_idx, sensitive, winner, loser):
    rec = recommend(forecast_pm25=BAND_FORECASTS[band_idx], sensitive=sensitive, outdoor_activity=True)
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert winner in fired_ids and loser in fired_ids  # both really were in the contest
    assert rec.chosen.id == winner
    assert "tied" in rec.chosen_reason


def test_tie_break_runner_up_losing_rule_still_recorded_with_matched_facts():
    rec = recommend(forecast_pm25=BAND_FORECASTS[4], sensitive=True, outdoor_activity=True)
    loser = next(fr for fr in rec.fired if fr.rule.id == "R15")
    assert dict(loser.matched_facts)["outdoor_activity"] is True


# --- specificity beats severity when they disagree -----------------------------------------


def test_most_specific_wins_even_over_a_lower_severity_margin():
    # Moderate, sensitive, outdoor: R04 (3 conditions, severity 2) must beat R03 (2 conditions,
    # severity 2) and R02 (2 conditions, severity 1) on specificity alone.
    rec = recommend(forecast_pm25=BAND_FORECASTS[1], sensitive=True, outdoor_activity=True)
    assert rec.chosen.id == "R04"
    assert "most-specific" in rec.chosen_reason


# --- safety margin (R21) --------------------------------------------------------------------


def test_margin_moderate_within_5_of_usg_boundary_uses_usg_advice():
    # 33.0 is Moderate (9.1-35.4); USG's lower boundary is 35.5, distance 2.5 <= 5.
    rec = recommend(forecast_pm25=33.0, sensitive=True, outdoor_activity=True)
    assert rec.band_index == 1  # the *displayed* band is still Moderate (model_selection.md)
    assert rec.band_name == "Moderate"
    assert rec.working_memory["near_boundary"] is True
    assert rec.working_memory["advice_band"] == 2  # USG
    assert rec.chosen.id == "R07"  # USG, sensitive, outdoor
    assert any("more cautious" in note for note in rec.notes)


def test_margin_never_applies_good_to_moderate():
    # 8.0 is Good (0-9.0), 1.0 ug/m3 below Moderate's lower boundary (9.1) -- well within 5,
    # but the margin must never fire Good -> Moderate (§7: "Good" advice must still exist).
    rec = recommend(forecast_pm25=8.0, sensitive=True, outdoor_activity=True)
    assert rec.band_index == 0
    assert rec.working_memory["near_boundary"] is False
    assert rec.working_memory["advice_band"] == 0
    assert rec.chosen.id == "R01"
    assert rec.notes == ()


def test_margin_does_not_apply_120_which_is_5_4_away_from_very_unhealthy():
    # 120.0 is Unhealthy (55.5-125.4); Very Unhealthy's lower boundary is 125.5, so the true
    # distance is 5.4 ug/m3 -- just outside the +-5 margin, so this stays Unhealthy advice.
    # (This corrects the brief's "120.0 -> Very Unhealthy advice" example: under the official
    # EPA breakpoints 120.0 does not fall within the margin -- see the final report.)
    rec = recommend(forecast_pm25=120.0, sensitive=False, outdoor_activity=False)
    assert rec.band_index == 3
    assert rec.working_memory["near_boundary"] is False
    assert rec.chosen.id == "R09"  # Unhealthy, not sensitive


def test_margin_applies_at_122_which_is_within_5_of_very_unhealthy():
    # 122.0 is Unhealthy; distance to 125.5 is 3.5 <= 5, so the margin should push the advice
    # (not the displayed band) up to Very Unhealthy.
    rec = recommend(forecast_pm25=122.0, sensitive=False, outdoor_activity=False)
    assert rec.band_index == 3
    assert rec.working_memory["near_boundary"] is True
    assert rec.working_memory["advice_band"] == 4
    assert rec.chosen.id == "R13"  # Very Unhealthy, not sensitive


def test_margin_exactly_on_the_5ug_threshold_is_inclusive():
    # 30.5 is Moderate; USG's lower boundary is 35.5, distance exactly 5.0 -- the margin is
    # "within +-5 ug/m3" so this boundary value must trigger it.
    rec = recommend(forecast_pm25=30.5)
    assert rec.working_memory["near_boundary"] is True
    assert rec.working_memory["advice_band"] == 2


def test_margin_just_outside_the_5ug_threshold_does_not_fire():
    # 30.49 is 5.01 ug/m3 below 35.5 -- just outside the margin.
    rec = recommend(forecast_pm25=30.49)
    assert rec.working_memory["near_boundary"] is False
    assert rec.working_memory["advice_band"] == 1


def test_margin_never_applies_to_hazardous_top_band_no_next_band():
    rec = recommend(forecast_pm25=300.0)
    assert rec.working_memory["near_boundary"] is False
    assert rec.working_memory["advice_band"] == 5


def test_margin_very_unhealthy_can_push_into_hazardous():
    # 222.0 is Very Unhealthy (125.5-225.4); Hazardous starts at 225.5, distance 3.5 <= 5.
    rec = recommend(forecast_pm25=222.0)
    assert rec.band_index == 4
    assert rec.working_memory["near_boundary"] is True
    assert rec.working_memory["advice_band"] == 5


# --- trend (R18-R20) -------------------------------------------------------------------------


def test_trend_rising_usg_fires_r18():
    rec = recommend(forecast_pm25=45.0, today_pm25=20.0)  # delta 25 >= threshold -> rising
    assert rec.working_memory["trend"] == "rising"
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert "R18" in fired_ids
    assert any("worsen" in note for note in rec.notes)


def test_trend_rising_unhealthy_or_worse_fires_r19_not_r18():
    rec = recommend(forecast_pm25=90.0, today_pm25=50.0)  # Unhealthy, rising
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert "R19" in fired_ids
    assert "R18" not in fired_ids
    assert any("worse than the day before" in note for note in rec.notes)


def test_trend_falling_usg_or_worse_fires_r20():
    rec = recommend(forecast_pm25=45.0, today_pm25=70.0)  # delta -25, falling
    assert rec.working_memory["trend"] == "falling"
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert "R20" in fired_ids
    assert any("improve" in note for note in rec.notes)


def test_trend_small_change_is_steady_no_trend_note():
    rec = recommend(forecast_pm25=22.0, today_pm25=20.0)  # delta 2, below threshold
    assert rec.working_memory["trend"] == "steady"
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert not {"R18", "R19", "R20"} & set(fired_ids)


def test_no_today_pm25_means_no_trend_and_no_trend_notes():
    rec = recommend(forecast_pm25=45.0, today_pm25=None)
    assert rec.working_memory["trend"] is None
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert not {"R18", "R19", "R20"} & set(fired_ids)
    assert rec.notes == ()


# --- forward chaining: fixpoint, and a real derived-fact chain -----------------------------


def test_run_terminates_well_within_max_iterations_with_no_duplicate_firings():
    rec = recommend(forecast_pm25=33.0, today_pm25=25.0, sensitive=True, outdoor_activity=True)
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert len(fired_ids) == len(set(fired_ids))  # no rule re-fires
    assert len(fired_ids) <= len(ALL_RULES)


def test_chain_forecast_to_band_to_near_boundary_to_advice_band_to_advice():
    # The textbook chain: a raw forecast -> its band -> the margin rule (R21) derives
    # near_boundary + advice_band from that band -> only then can an advice rule, which
    # conditions on advice_band, fire. Proven by checking advice_band != band and that the
    # rule which actually won conditions on the *bumped* value, not the raw one.
    rec = run(Facts(forecast_pm25=33.0, sensitive=True))
    assert rec.band_index == 1  # band, from the forecast
    assert rec.working_memory["near_boundary"] is True  # R21 matched on that band
    assert rec.working_memory["advice_band"] == 2  # R21's conclusion
    assert any(c.fact == "advice_band" and c.value == 2 for c in rec.chosen.conditions)
    r21_fired = next(fr for fr in rec.fired if fr.rule.id == "R21")
    assert dict(r21_fired.matched_facts)["band"] == 1  # the fact that enabled R21 itself


def test_r21_fires_before_the_advice_rule_it_enables():
    rec = recommend(forecast_pm25=33.0, sensitive=True, outdoor_activity=True)
    fired_ids = [fr.rule.id for fr in rec.fired]
    assert fired_ids.index("R21") < fired_ids.index(rec.chosen.id)


def test_no_advice_rule_can_match_before_the_margin_round():
    # Sanity check on the generic Condition mechanism itself: with only `band` known (no
    # advice_band yet), none of the main advice rules match.
    from src.expert.rules import ADVICE_RULES

    minimal_wm = {"band": 1, "sensitive": True, "outdoor_activity": True}
    assert not any(rule.matches(minimal_wm) for rule in ADVICE_RULES)


# --- Recommendation / FiredRule are immutable, and the colours are EPA's ------------------


def test_recommendation_is_frozen():
    rec = recommend(forecast_pm25=20.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.advice = "something else"  # type: ignore[misc]


def test_fired_rule_is_frozen():
    rec = recommend(forecast_pm25=20.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.fired[0].matched_facts = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "band_idx,color",
    [(0, "#00E400"), (1, "#FFFF00"), (2, "#FF7E00"), (3, "#FF0000"), (4, "#8F3F97"), (5, "#7E0023")],
)
def test_epa_colours_match_band(band_idx, color):
    rec = recommend(forecast_pm25=BAND_FORECASTS[band_idx])
    assert rec.color == color
