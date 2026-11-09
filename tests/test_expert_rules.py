"""Tests for the rule base as data (`src.expert.rules`), PROJECT_SPEC.md §7 /
`docs/report/knowledge_sources.md` §5 — R01-R22.
"""

import pytest

from src.expert.rules import (
    ADVICE_RULES,
    ALL_RULES,
    APPEND_RULES,
    R21,
    R22,
    RULES_BY_ID,
    Condition,
)

ALL_IDS = [f"R{i:02d}" for i in range(1, 23)]

# --- shape: every rule from the knowledge base is present, exactly once --------------------


def test_all_22_rule_ids_present_exactly_once():
    ids = [rule.id for rule in ALL_RULES]
    assert sorted(ids) == sorted(ALL_IDS)
    assert len(ids) == len(set(ids)) == 22


def test_rules_by_id_matches_all_rules():
    assert set(RULES_BY_ID) == set(ALL_IDS)
    for rule_id, rule in RULES_BY_ID.items():
        assert rule.id == rule_id


def test_advice_append_meta_partition_is_exhaustive():
    kinds = {rule.id: rule.kind for rule in ALL_RULES}
    assert set(kinds.values()) <= {"advice", "append", "meta"}
    assert {rule.id for rule in ADVICE_RULES} == {f"R{i:02d}" for i in range(1, 18)}
    assert {rule.id for rule in APPEND_RULES} == {"R18", "R19", "R20"}
    assert R21.kind == "meta"
    assert R22.kind == "append"


# --- conditions are inspectable (fact, operator, value) triples, not opaque lambdas --------


def test_every_condition_is_a_plain_triple():
    for rule in ALL_RULES:
        for condition in rule.conditions:
            assert isinstance(condition, Condition)
            assert isinstance(condition.fact, str)
            assert isinstance(condition.op, str)
            assert str(condition)  # printable, for explain.py


def test_specificity_is_the_condition_count():
    for rule in ALL_RULES:
        assert rule.specificity == len(rule.conditions)


# --- severity: 1-5 for the health-guidance rules, R22 (disclaimer) is outside the contest ---


@pytest.mark.parametrize("rule", ADVICE_RULES + APPEND_RULES + (R21,), ids=lambda r: r.id)
def test_contest_rules_have_valid_severity(rule):
    assert 1 <= rule.severity <= 5


def test_disclaimer_severity_is_outside_the_contest():
    assert R22.severity == 0


# --- every advice rule cites a source and carries non-empty advice text --------------------


@pytest.mark.parametrize("rule", ADVICE_RULES, ids=lambda r: r.id)
def test_advice_rules_have_source_and_text(rule):
    assert rule.source
    assert rule.advice
    assert rule.because


def test_meta_rule_has_no_advice_text():
    assert R21.advice is None


# --- each R01-R20 fires on a hand-built working-memory dict, matching knowledge_sources.md -

ADVICE_MATCH_CASES = [
    ("R01", {"advice_band": 0, "sensitive": False, "outdoor_activity": False}),
    ("R02", {"advice_band": 1, "sensitive": False, "outdoor_activity": False}),
    ("R03", {"advice_band": 1, "sensitive": True, "outdoor_activity": False}),
    ("R04", {"advice_band": 1, "sensitive": True, "outdoor_activity": True}),
    ("R05", {"advice_band": 2, "sensitive": False, "outdoor_activity": False}),
    ("R06", {"advice_band": 2, "sensitive": True, "outdoor_activity": False}),
    ("R07", {"advice_band": 2, "sensitive": True, "outdoor_activity": True}),
    ("R08", {"advice_band": 2, "sensitive": False, "outdoor_activity": True}),
    ("R09", {"advice_band": 3, "sensitive": False, "outdoor_activity": False}),
    ("R10", {"advice_band": 3, "sensitive": True, "outdoor_activity": False}),
    ("R11", {"advice_band": 3, "sensitive": False, "outdoor_activity": True}),
    ("R12", {"advice_band": 3, "sensitive": True, "outdoor_activity": True}),
    ("R13", {"advice_band": 4, "sensitive": False, "outdoor_activity": False}),
    ("R14", {"advice_band": 4, "sensitive": True, "outdoor_activity": False}),
    ("R15", {"advice_band": 4, "sensitive": False, "outdoor_activity": True}),
    ("R15", {"advice_band": 5, "sensitive": False, "outdoor_activity": True}),
    ("R16", {"advice_band": 5, "sensitive": False, "outdoor_activity": False}),
    ("R17", {"advice_band": 5, "sensitive": True, "outdoor_activity": False}),
    ("R18", {"advice_band": 2, "trend": "rising"}),
    ("R19", {"advice_band": 3, "trend": "rising"}),
    ("R19", {"advice_band": 4, "trend": "rising"}),
    ("R19", {"advice_band": 5, "trend": "rising"}),
    ("R20", {"advice_band": 2, "trend": "falling"}),
    ("R20", {"advice_band": 4, "trend": "falling"}),
]


@pytest.mark.parametrize("rule_id,working_memory", ADVICE_MATCH_CASES)
def test_rule_matches_its_hand_built_fact_set(rule_id, working_memory):
    assert RULES_BY_ID[rule_id].matches(working_memory) is True


NON_MATCH_CASES = [
    ("R01", {"advice_band": 1, "sensitive": False, "outdoor_activity": False}),
    ("R03", {"advice_band": 1, "sensitive": False, "outdoor_activity": False}),
    ("R06", {"advice_band": 2, "sensitive": False, "outdoor_activity": False}),
    ("R15", {"advice_band": 3, "sensitive": False, "outdoor_activity": True}),  # Unhealthy, not VU/Haz
    ("R18", {"advice_band": 3, "trend": "rising"}),  # R18 is USG-only
    ("R20", {"advice_band": 1, "trend": "falling"}),  # below USG
]


@pytest.mark.parametrize("rule_id,working_memory", NON_MATCH_CASES)
def test_rule_does_not_match_a_non_satisfying_fact_set(rule_id, working_memory):
    assert RULES_BY_ID[rule_id].matches(working_memory) is False


def test_disclaimer_matches_any_working_memory_including_empty():
    assert R22.matches({}) is True


def test_condition_missing_fact_does_not_match_rather_than_raising():
    # advice_band hasn't been derived yet -- R01 should simply not match, not KeyError.
    assert RULES_BY_ID["R01"].matches({"band": 0}) is False
