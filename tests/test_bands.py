import math

import pandas as pd
import pytest

from src.classify import bands as b

# --- pm25_to_band: boundaries -------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_index",
    [
        (0.0, 0),
        (9.0, 0),
        (9.04, 0),  # truncates to 9.0 -> Good
        (9.05, 0),  # truncates to 9.0 -> Good
        (9.1, 1),  # Moderate
        (35.4, 1),
        (35.45, 1),  # truncates to 35.4 -> Moderate
        (35.5, 2),  # Unhealthy for Sensitive Groups
        (55.4, 2),
        (55.5, 3),  # Unhealthy
        (125.4, 3),
        (125.5, 4),  # Very Unhealthy
        (225.4, 4),
        (225.5, 5),  # Hazardous
        (500.0, 5),
        (1000.0, 5),
    ],
)
def test_pm25_to_band_boundaries(value, expected_index):
    assert b.pm25_to_band(value) == expected_index


def test_pm25_to_band_names_match_breakpoints():
    assert b.band_name(b.pm25_to_band(9.04)) == "Good"
    assert b.band_name(b.pm25_to_band(9.1)) == "Moderate"
    assert b.band_name(b.pm25_to_band(35.45)) == "Moderate"
    assert b.band_name(b.pm25_to_band(35.5)) == "Unhealthy for Sensitive Groups"
    assert b.band_name(b.pm25_to_band(225.5)) == "Hazardous"


def test_bands_are_contiguous_and_cover_zero_to_infinity():
    assert b.BANDS[0].low == 0.0
    assert math.isinf(b.BANDS[-1].high)
    for earlier, later in zip(b.BANDS, b.BANDS[1:]):
        # one-decimal breakpoints step by 0.1 with no gap or overlap
        assert round(later.low - earlier.high, 1) == 0.1


# --- pm25_to_band: negative values ---------------------------------------------


@pytest.mark.parametrize("value", [-0.1, -1.0, -500.0])
def test_pm25_to_band_negative_raises(value):
    with pytest.raises(ValueError):
        b.pm25_to_band(value)


# --- bands_for: vectorised, NaN handling ---------------------------------------


def test_bands_for_matches_scalar_function():
    series = pd.Series([0.0, 9.04, 9.1, 35.45, 35.5, 225.5])
    result = b.bands_for(series)
    expected = [b.pm25_to_band(v) for v in series]
    assert result.tolist() == expected


def test_bands_for_nan_in_gives_na_out():
    series = pd.Series([9.0, float("nan"), 35.5])
    result = b.bands_for(series)
    assert result.dtype == "Int64"
    assert result.iloc[0] == 0
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 2


def test_bands_for_empty_series():
    result = b.bands_for(pd.Series([], dtype="float64"))
    assert len(result) == 0
    assert result.dtype == "Int64"


# --- merge_bands ----------------------------------------------------------------


def test_merge_bands_default_merges_hazardous_into_very_unhealthy():
    series = pd.Series([0, 1, 2, 3, 4, 5], dtype="Int64")
    merged = b.merge_bands(series)
    assert merged.tolist() == [0, 1, 2, 3, 4, 4]


def test_merge_bands_custom_mapping():
    series = pd.Series([0, 1, 5], dtype="Int64")
    merged = b.merge_bands(series, mapping={0: 1, 5: 4})
    assert merged.tolist() == [1, 1, 4]


def test_merge_bands_preserves_na():
    series = pd.Series([0, pd.NA, 5], dtype="Int64")
    merged = b.merge_bands(series)
    assert merged.iloc[0] == 0
    assert pd.isna(merged.iloc[1])
    assert merged.iloc[2] == 4


def test_merge_bands_default_mapping_is_hazardous_to_very_unhealthy():
    assert b.DEFAULT_MERGE == {5: 4}


# --- band_name -------------------------------------------------------------------


def test_band_name_all_indices():
    expected = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    assert [b.band_name(i) for i in range(6)] == expected


def test_band_name_unknown_index_raises():
    with pytest.raises(KeyError):
        b.band_name(99)
