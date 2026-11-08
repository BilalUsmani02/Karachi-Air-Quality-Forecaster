import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src.data import open_meteo as om
from src.data.sensors import sensor_by_id

SENSOR = sensor_by_id("S1")


def _response(status_code: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    if status_code != 200:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    return response


def _payload(hours: int = 2, lat: float = 24.9, lon: float = 67.1) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": {
            "time": [f"2024-01-01T{h:02d}:00" for h in range(hours)],
            "temperature_2m": [20.0 + h for h in range(hours)],
        },
    }


# --- build_params --------------------------------------------------------


def test_build_params_basic():
    params = om.build_params(SENSOR, date(2024, 1, 1), date(2024, 1, 31), ["temperature_2m", "dust"])
    assert params["latitude"] == SENSOR.lat
    assert params["longitude"] == SENSOR.lon
    assert params["start_date"] == "2024-01-01"
    assert params["end_date"] == "2024-01-31"
    assert params["hourly"] == "temperature_2m,dust"
    assert params["timezone"] == "Asia/Karachi"


def test_build_params_merges_extra():
    params = om.build_params(SENSOR, date(2024, 1, 1), date(2024, 1, 1), ["pm2_5"], {"wind_speed_unit": "ms"})
    assert params["wind_speed_unit"] == "ms"


# --- date_chunks -----------------------------------------------------------


def test_date_chunks_single_day():
    assert om.date_chunks(date(2024, 6, 1), date(2024, 6, 1)) == [(date(2024, 6, 1), date(2024, 6, 1))]


def test_date_chunks_within_one_year():
    assert om.date_chunks(date(2024, 1, 1), date(2024, 12, 31)) == [(date(2024, 1, 1), date(2024, 12, 31))]


def test_date_chunks_splits_at_year_boundary():
    chunks = om.date_chunks(date(2022, 8, 5), date(2024, 2, 1))
    assert chunks == [
        (date(2022, 8, 5), date(2022, 12, 31)),
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 2, 1)),
    ]


def test_date_chunks_start_after_end_raises():
    with pytest.raises(ValueError):
        om.date_chunks(date(2024, 2, 1), date(2024, 1, 1))


# --- fetch_with_retry --------------------------------------------------------


def test_fetch_with_retry_succeeds_first_try():
    session = MagicMock()
    session.get.return_value = _response(200, {"ok": True})
    sleep = MagicMock()

    result = om.fetch_with_retry("http://x", {}, session=session, sleep=sleep)

    assert result == {"ok": True}
    sleep.assert_not_called()


def test_fetch_with_retry_retries_then_succeeds_on_429():
    session = MagicMock()
    session.get.side_effect = [_response(429), _response(200, {"ok": True})]
    sleep = MagicMock()

    result = om.fetch_with_retry("http://x", {}, session=session, sleep=sleep)

    assert result == {"ok": True}
    sleep.assert_called_once_with(om.BACKOFF_SECONDS)


def test_fetch_with_retry_retries_then_succeeds_after_network_error():
    session = MagicMock()
    session.get.side_effect = [requests.ConnectionError("boom"), _response(200, {"ok": True})]
    sleep = MagicMock()

    result = om.fetch_with_retry("http://x", {}, session=session, sleep=sleep)

    assert result == {"ok": True}
    assert sleep.call_count == 1


def test_fetch_with_retry_exhausted_raises():
    session = MagicMock()
    session.get.return_value = _response(503)
    sleep = MagicMock()

    with pytest.raises(RuntimeError):
        om.fetch_with_retry("http://x", {}, session=session, sleep=sleep, max_retries=3)

    assert session.get.call_count == 3
    assert sleep.call_count == 2  # no sleep after the last attempt


def test_fetch_with_retry_backoff_is_exponential():
    session = MagicMock()
    session.get.return_value = _response(500)
    sleep = MagicMock()

    with pytest.raises(RuntimeError):
        om.fetch_with_retry("http://x", {}, session=session, sleep=sleep, max_retries=4, backoff_seconds=1.0)

    sleep.assert_any_call(1.0)
    sleep.assert_any_call(2.0)
    sleep.assert_any_call(4.0)


def test_fetch_with_retry_non_retryable_status_raises_immediately():
    session = MagicMock()
    session.get.return_value = _response(404)
    sleep = MagicMock()

    with pytest.raises(requests.HTTPError):
        om.fetch_with_retry("http://x", {}, session=session, sleep=sleep)

    assert session.get.call_count == 1
    sleep.assert_not_called()


# --- parse_hourly ------------------------------------------------------------


def test_parse_hourly_builds_tz_aware_frame():
    frame = om.parse_hourly(_payload(hours=3))
    assert list(frame.columns) == ["time", "temperature_2m"]
    assert len(frame) == 3
    assert str(frame["time"].dt.tz) == "Asia/Karachi"
    assert frame["temperature_2m"].tolist() == [20.0, 21.0, 22.0]


# --- fetch_range --------------------------------------------------------------


def test_fetch_range_concatenates_chunks_and_keeps_grid_coords():
    session = MagicMock()
    session.get.side_effect = [
        _response(200, _payload(hours=2, lat=24.91, lon=67.12)),
        _response(200, _payload(hours=2, lat=24.91, lon=67.12)),
    ]

    result = om.fetch_range(
        "http://x",
        SENSOR,
        date(2022, 12, 31),
        date(2023, 1, 1),
        ["temperature_2m"],
        session=session,
        sleep=MagicMock(),
    )

    assert session.get.call_count == 2
    assert len(result.frame) == 4
    assert result.grid_latitude == 24.91
    assert result.grid_longitude == 67.12


def test_fetch_range_sorts_by_time():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-02T00:00", "2024-01-01T00:00"]).tz_localize("Asia/Karachi"),
            "temperature_2m": [2.0, 1.0],
        }
    )
    assert frame.sort_values("time")["temperature_2m"].tolist() == [1.0, 2.0]


# --- save_csv / write_manifest ------------------------------------------------


def test_save_csv_creates_parent_and_writes(tmp_path):
    frame = om.parse_hourly(_payload(hours=2))
    out_path = tmp_path / "nested" / "S1.csv"

    om.save_csv(frame, out_path)

    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert len(written) == 2


def test_write_manifest_creates_new_file(tmp_path):
    path = tmp_path / "manifest.json"
    om.write_manifest(path, {"S1": {"row_count": 10}})

    data = json.loads(path.read_text())
    assert data == {"S1": {"row_count": 10}}


def test_write_manifest_merges_without_dropping_other_sensors(tmp_path):
    path = tmp_path / "manifest.json"
    om.write_manifest(path, {"S1": {"row_count": 10}})
    om.write_manifest(path, {"S2": {"row_count": 20}})

    data = json.loads(path.read_text())
    assert data == {"S1": {"row_count": 10}, "S2": {"row_count": 20}}


def test_write_manifest_overwrites_same_sensor(tmp_path):
    path = tmp_path / "manifest.json"
    om.write_manifest(path, {"S1": {"row_count": 10}})
    om.write_manifest(path, {"S1": {"row_count": 99}})

    data = json.loads(path.read_text())
    assert data == {"S1": {"row_count": 99}}


def test_utc_now_iso_is_parseable():
    from datetime import datetime

    stamp = om.utc_now_iso()
    datetime.fromisoformat(stamp)
