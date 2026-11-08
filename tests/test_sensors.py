import pytest

from src.data.sensors import SENSORS, US_CONSULATE, Sensor, sensor_by_id


def test_fourteen_sensors():
    assert len(SENSORS) == 14


def test_ids_are_unique_and_sequential():
    assert [sensor.id for sensor in SENSORS] == [f"S{i}" for i in range(1, 15)]


def test_sensor_is_frozen():
    sensor = SENSORS[0]
    with pytest.raises(AttributeError):
        sensor.lat = 0.0


def test_coordinates_are_in_karachi():
    for sensor in SENSORS:
        assert 24.5 < sensor.lat < 25.2
        assert 66.7 < sensor.lon < 67.3


def test_openaq_ids_are_unique():
    location_ids = [sensor.openaq_location_id for sensor in SENSORS]
    sensor_ids = [sensor.openaq_sensor_id for sensor in SENSORS]
    assert len(set(location_ids)) == len(SENSORS)
    assert len(set(sensor_ids)) == len(SENSORS)


def test_s1_values():
    s1 = sensor_by_id("S1")
    assert s1.site == "Climate Action Center"
    assert s1.openaq_location_id == 4932783
    assert s1.openaq_sensor_id == 13490916
    assert s1.lat == 24.8638
    assert s1.lon == 67.0566


def test_s14_values():
    s14 = sensor_by_id("S14")
    assert s14.site == "Aga Khan University IED"
    assert s14.openaq_location_id == 4793983
    assert s14.openaq_sensor_id == 13370571


def test_sensor_by_id_unknown_raises():
    with pytest.raises(ValueError):
        sensor_by_id("S15")


def test_sensor_by_id_matches_registry():
    for sensor in SENSORS:
        assert sensor_by_id(sensor.id) is sensor


def test_sensor_equality_by_value():
    assert Sensor("S1", "x", 1, 2, 1.0, 1.0) == Sensor("S1", "x", 1, 2, 1.0, 1.0)


def test_us_consulate_is_kept_out_of_sensors():
    assert US_CONSULATE not in SENSORS
    assert US_CONSULATE.id == "US_CONSULATE"


def test_us_consulate_values():
    assert US_CONSULATE.openaq_location_id == 8156
    assert US_CONSULATE.openaq_sensor_id == 23747
    assert US_CONSULATE.lat == 24.8415
    assert US_CONSULATE.lon == 67.0091
    assert 24.5 < US_CONSULATE.lat < 25.2
    assert 66.7 < US_CONSULATE.lon < 67.3
