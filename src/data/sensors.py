"""Immutable registry of the 14 OpenAQ-mirrored PM2.5 sensors run by The Urban Unit (spec §5.2)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    id: str
    site: str
    openaq_location_id: int
    openaq_sensor_id: int
    lat: float
    lon: float


SENSORS: tuple[Sensor, ...] = (
    Sensor("S1", "Climate Action Center", 4932783, 13490916, 24.8638, 67.0566),
    Sensor("S2", "Zafar Memon DHA", 4814327, 13387396, 24.8147, 67.0671),
    Sensor("S3", "Urban Resource Center", 4791924, 13368464, 24.8680, 67.0818),
    Sensor("S4", "Aga Khan University Main Campus", 4837113, 13406419, 24.8924, 67.0747),
    Sensor("S5", "Gulistan-e-Johar", 4838712, 13408226, 24.9162, 67.1315),
    Sensor("S6", "G3 Engineering Consultants", 4985886, 13533643, 24.8070, 67.0332),
    Sensor("S7", "NED University, Main Campus", 4837117, 13406434, 24.9338, 67.1135),
    Sensor("S8", "Malir Cantonment, Askari 5", 4840233, 13409122, 24.9396, 67.1810),
    Sensor("S9", "Aman Khan Kaka Khail", 4808325, 13381251, 24.9532, 67.0637),
    Sensor("S10", "DG House DHA Phase 5", 4815820, 13388480, 24.7956, 67.0481),
    Sensor("S11", "Orangi Pilot Project", 4847173, 13414781, 24.9373, 67.0190),
    Sensor("S12", "Karachi Zoo", 4804767, 13378900, 24.8761, 67.0231),
    Sensor("S13", "Urban Unit Karachi office", 4838122, 13407059, 24.8641, 67.0709),
    Sensor("S14", "Aga Khan University IED", 4793983, 13370571, 24.8921, 67.0709),
)

_BY_ID = {sensor.id: sensor for sensor in SENSORS}


def sensor_by_id(sensor_id: str) -> Sensor:
    """Look up a sensor by its ID (e.g. "S1"); raises ValueError if unknown."""
    try:
        return _BY_ID[sensor_id]
    except KeyError:
        raise ValueError(f"unknown sensor id: {sensor_id!r}") from None


# D5: reference-grade monitor, kept separate from SENSORS (not a prediction target).
# PM2.5 sensor ID confirmed live via GET /v3/locations/8156 (checked 8 Nov 2026).
US_CONSULATE = Sensor("US_CONSULATE", "Karachi US Consulate", 8156, 23747, 24.8415, 67.0091)
