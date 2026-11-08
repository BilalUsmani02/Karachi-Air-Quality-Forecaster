"""Immutable registry of the 8 AirGradient sensors run by The Urban Unit (spec §5.2)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    id: str
    site: str
    airgradient_id: int
    lat: float
    lon: float


SENSORS: tuple[Sensor, ...] = (
    Sensor("S1", "NED University, Main Campus", 162732, 24.9337634173323, 67.11354155158995),
    Sensor("S2", "Aga Khan University, Main Campus", 162672, 24.892402, 67.074663),
    Sensor("S3", "Orangi Pilot Project", 162802, 24.9373111, 67.0189743),
    Sensor("S4", "Urban Resource Center", 162379, 24.868, 67.081833),
    Sensor("S5", "Gulistan-e-Johar", 162753, 24.916229, 67.131546),
    Sensor("S6", "Climate Action Center", 162364, 24.86377, 67.056642),
    Sensor("S7", "Malir Cantonment, Askari 5", 162762, 24.939636, 67.180983),
    Sensor("S8", "Clifton", 206703, 24.810257, 67.032087),
)

_BY_ID = {sensor.id: sensor for sensor in SENSORS}


def sensor_by_id(sensor_id: str) -> Sensor:
    """Look up a sensor by its ID (e.g. "S1"); raises ValueError if unknown."""
    try:
        return _BY_ID[sensor_id]
    except KeyError:
        raise ValueError(f"unknown sensor id: {sensor_id!r}") from None
