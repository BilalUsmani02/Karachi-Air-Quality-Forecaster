from datetime import date

import pandas as pd
import pytest

from src.data import openaq as oa

# --- read_env_file -----------------------------------------------------------


def test_read_env_file_missing_returns_empty(tmp_path):
    assert oa.read_env_file(tmp_path / "nope.env") == {}


def test_read_env_file_basic(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAQ_API_KEY=abc123\n")

    assert oa.read_env_file(path) == {"OPENAQ_API_KEY": "abc123"}


def test_read_env_file_handles_quotes_whitespace_comments_and_bom(tmp_path):
    path = tmp_path / ".env"
    content = (
        "# a comment\n"
        "\n"
        '  OPENAQ_API_KEY = "abc 123"  \n'
        "OTHER_KEY='single quoted'\n"
        "NOT_A_LINE_WITHOUT_EQUALS\n"
    )
    path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

    values = oa.read_env_file(path)

    assert values["OPENAQ_API_KEY"] == "abc 123"
    assert values["OTHER_KEY"] == "single quoted"
    assert "NOT_A_LINE_WITHOUT_EQUALS" not in values


# --- get_api_key --------------------------------------------------------------


def test_get_api_key_env_var_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAQ_API_KEY=from_dotenv\n")
    monkeypatch.setenv("OPENAQ_API_KEY", "from_env_var")

    assert oa.get_api_key(root=tmp_path) == "from_env_var"


def test_get_api_key_falls_back_to_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAQ_API_KEY=from_dotenv\n")

    assert oa.get_api_key(root=tmp_path) == "from_dotenv"


def test_get_api_key_missing_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        oa.get_api_key(root=tmp_path)

    message = str(exc_info.value)
    assert "OPENAQ_API_KEY" in message
    assert ".env" in message
    assert "register" in message.lower()


def test_get_api_key_never_appears_in_printed_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAQ_API_KEY", "super-secret-value")

    key = oa.get_api_key(root=tmp_path)

    assert key == "super-secret-value"
    assert "super-secret-value" not in capsys.readouterr().out


# --- fetch_hours (pagination + chunking) --------------------------------------


def test_fetch_hours_stops_on_short_page(monkeypatch):
    calls = []

    def fake_fetch_with_retry(url, params, *, headers=None, session=None, sleep=None):
        calls.append(params)
        page = params["page"]
        if page == 1:
            return {"results": [{"value": 1.0}] * oa.PAGE_LIMIT}
        return {"results": [{"value": 2.0}] * 7}

    monkeypatch.setattr(oa, "fetch_with_retry", fake_fetch_with_retry)
    sleeps = []

    records = oa.fetch_hours(
        123, date(2024, 1, 1), date(2024, 1, 2), "key", sleep=sleeps.append, request_delay=0
    )

    assert len(records) == oa.PAGE_LIMIT + 7
    assert [c["page"] for c in calls] == [1, 2]
    assert sleeps == [0, 0]


def test_month_chunks_split_on_month_boundaries():
    assert oa.month_chunks(date(2024, 1, 30), date(2024, 3, 2)) == [
        (date(2024, 1, 30), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]


def test_month_chunks_single_day_and_bad_range():
    assert oa.month_chunks(date(2025, 12, 31), date(2025, 12, 31)) == [
        (date(2025, 12, 31), date(2025, 12, 31))
    ]
    with pytest.raises(ValueError):
        oa.month_chunks(date(2025, 2, 1), date(2025, 1, 1))


def test_fetch_hours_chunks_by_month(monkeypatch):
    calls = []

    def fake_fetch_with_retry(url, params, *, headers=None, session=None, sleep=None):
        calls.append(params)
        return {"results": []}

    monkeypatch.setattr(oa, "fetch_with_retry", fake_fetch_with_retry)

    oa.fetch_hours(123, date(2022, 12, 31), date(2023, 1, 2), "key", sleep=lambda s: None)

    assert len(calls) == 2
    assert calls[0]["datetime_from"] == "2022-12-31"
    assert calls[0]["datetime_to"] == "2023-01-01"
    assert calls[1]["datetime_from"] == "2023-01-01"
    assert calls[1]["datetime_to"] == "2023-01-03"


def test_fetch_hours_sends_api_key_header(monkeypatch):
    seen_headers = {}

    def fake_fetch_with_retry(url, params, *, headers=None, session=None, sleep=None):
        seen_headers.update(headers or {})
        return {"results": []}

    monkeypatch.setattr(oa, "fetch_with_retry", fake_fetch_with_retry)

    oa.fetch_hours(123, date(2024, 1, 1), date(2024, 1, 1), "my-key", sleep=lambda s: None)

    assert seen_headers == {"X-API-Key": "my-key"}


# --- parse_records ------------------------------------------------------------


def _record(value, local_time, *, observed=24, expected=24, min_=None, max_=None, sd=None):
    return {
        "value": value,
        "period": {"datetimeFrom": {"local": local_time}},
        "coverage": {"observedCount": observed, "expectedCount": expected},
        "summary": {"min": min_, "max": max_, "sd": sd},
    }


def test_parse_records_empty_list_keeps_columns():
    frame = oa.parse_records([])
    assert list(frame.columns) == ["time", "pm25", "observed_count", "expected_count", "min", "max", "sd"]
    assert len(frame) == 0


def test_parse_records_basic_fields_and_tz():
    records = [_record(42.0, "2024-01-01T00:00:00+05:00", min_=40.0, max_=44.0, sd=1.5)]

    frame = oa.parse_records(records)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["pm25"] == 42.0
    assert row["observed_count"] == 24
    assert row["expected_count"] == 24
    assert row["min"] == 40.0
    assert row["max"] == 44.0
    assert row["sd"] == 1.5
    assert str(frame["time"].dt.tz) == "Asia/Karachi"
    assert row["time"].hour == 0


def test_parse_records_missing_coverage_and_summary():
    records = [
        {
            "value": 10.0,
            "period": {"datetimeFrom": {"local": "2024-01-01T01:00:00+05:00"}},
        }
    ]

    frame = oa.parse_records(records)

    assert pd.isna(frame.iloc[0]["observed_count"])
    assert pd.isna(frame.iloc[0]["expected_count"])
    assert pd.isna(frame.iloc[0]["min"])


def test_parse_records_skips_entries_without_period():
    records = [{"value": 1.0}, _record(2.0, "2024-01-01T02:00:00+05:00")]

    frame = oa.parse_records(records)

    assert len(frame) == 1
    assert frame.iloc[0]["pm25"] == 2.0


def test_parse_records_sorts_and_drops_duplicate_timestamps():
    records = [
        _record(2.0, "2024-01-01T01:00:00+05:00"),
        _record(1.0, "2024-01-01T00:00:00+05:00"),
        _record(99.0, "2024-01-01T00:00:00+05:00"),
    ]

    frame = oa.parse_records(records)

    assert len(frame) == 2
    assert frame["pm25"].tolist() == [1.0, 2.0]


# --- fetch_sensor_dataframe ----------------------------------------------------


def test_fetch_sensor_dataframe_combines_fetch_and_parse(monkeypatch):
    monkeypatch.setattr(
        oa, "fetch_hours", lambda *a, **k: [_record(5.0, "2024-01-01T00:00:00+05:00")]
    )

    frame = oa.fetch_sensor_dataframe(123, date(2024, 1, 1), date(2024, 1, 1), "key")

    assert len(frame) == 1
    assert frame.iloc[0]["pm25"] == 5.0
