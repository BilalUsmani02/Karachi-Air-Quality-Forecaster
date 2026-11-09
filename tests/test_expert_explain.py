"""Tests for the explanation facility (`src.expert.explain`), PROJECT_SPEC.md §7."""

import json

import pytest

from src.expert import explain, recommend, to_dict


@pytest.fixture
def rec():
    # The task's own worked example: near a boundary, sensitive, outdoor plans, rising trend.
    return recommend(forecast_pm25=33.0, today_pm25=25.0, sensitive=True, outdoor_activity=True)


# --- explain(): names every fired rule and its source ---------------------------------------


def test_explain_names_every_fired_rule_id_and_source(rec):
    text = explain(rec)
    for fired_rule in rec.fired:
        assert fired_rule.rule.id in text
        assert fired_rule.rule.source in text


def test_explain_states_the_input_facts(rec):
    text = explain(rec)
    assert "33.0" in text
    assert "Moderate" in text  # the displayed band
    assert "yes" in text  # sensitive / outdoor_activity shown as yes/no, not True/False


def test_explain_states_which_rule_won_and_why(rec):
    text = explain(rec)
    assert rec.chosen.id in text
    assert rec.chosen_reason in text
    assert rec.advice in text


def test_explain_mentions_safety_margin_when_applied(rec):
    assert "margin" in explain(rec).lower()
    assert "Unhealthy for Sensitive Groups" in explain(rec)


def test_explain_states_margin_not_applied_when_it_is_not():
    rec_no_margin = recommend(forecast_pm25=20.0)
    text = explain(rec_no_margin)
    assert "not applied" in text.lower()


def test_explain_includes_the_disclaimer():
    rec_plain = recommend(forecast_pm25=20.0)
    assert rec_plain.disclaimer in explain(rec_plain)


def test_explain_mentions_trend_when_present(rec):
    assert "rising" in explain(rec)


def test_explain_is_a_non_empty_plain_string(rec):
    text = explain(rec)
    assert isinstance(text, str)
    assert text.startswith("You see this because")


# --- to_dict(): JSON-serialisable, for the Streamlit app ------------------------------------


def test_to_dict_is_json_serialisable(rec):
    payload = to_dict(rec)
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)
    assert reloaded["band_name"] == "Moderate"
    assert reloaded["advice_band_name"] == "Unhealthy for Sensitive Groups"
    assert reloaded["near_boundary"] is True
    assert reloaded["chosen_rule_id"] == rec.chosen.id


def test_to_dict_fired_list_matches_recommendation(rec):
    payload = to_dict(rec)
    assert [entry["id"] for entry in payload["fired"]] == [fr.rule.id for fr in rec.fired]
    for entry in payload["fired"]:
        assert entry["source"]
        assert isinstance(entry["conditions"], list)
        assert isinstance(entry["matched_facts"], dict)


def test_to_dict_has_no_trend_key_error_when_today_pm25_is_none():
    rec_no_today = recommend(forecast_pm25=20.0)
    payload = to_dict(rec_no_today)
    assert payload["trend"] is None
    json.dumps(payload)  # must not raise


def test_to_dict_notes_and_disclaimer_match_recommendation(rec):
    payload = to_dict(rec)
    assert payload["notes"] == list(rec.notes)
    assert payload["disclaimer"] == rec.disclaimer
