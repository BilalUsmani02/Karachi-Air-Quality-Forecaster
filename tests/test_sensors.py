import pytest

from src.data.sensors import SENSORS, Sensor, sensor_by_id


def test_eight_sensors():
    assert len(SENSORS) == 8


def test_ids_are_unique_and_sequential():
    assert [sensor.id for sensor in SENSORS] == [f"S{i}" for i in range(1, 9)]


def test_sensor_is_frozen():
    sensor = SENSORS[0]
    with pytest.raises(AttributeError):
        sensor.lat = 0.0


def test_coordinates_are_in_karachi():
    for sensor in SENSORS:
        assert 24.5 < sensor.lat < 25.2
        assert 66.7 < sensor.lon < 67.3


def test_s1_full_precision_coordinates():
    s1 = sensor_by_id("S1")
    assert s1.lat == 24.9337634173323
    assert s1.lon == 67.11354155158995
    assert s1.site == "NED University, Main Campus"
    assert s1.airgradient_id == 162732


def test_sensor_by_id_unknown_raises():
    with pytest.raises(ValueError):
        sensor_by_id("S9")


def test_sensor_by_id_matches_registry():
    for sensor in SENSORS:
        assert sensor_by_id(sensor.id) is sensor


def test_sensor_equality_by_value():
    assert Sensor("S1", "x", 1, 1.0, 1.0) == Sensor("S1", "x", 1, 1.0, 1.0)
