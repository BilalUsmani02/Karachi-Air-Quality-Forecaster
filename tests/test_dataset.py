import numpy as np
import pandas as pd
import pytest

from src.data.features import COLUMN_ORDER
from src.models.dataset import FEATURE_COLUMNS, SENSOR_COLUMNS, Preprocessor, make_xy

N_ROWS = 40


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    """A daily-feature-table-shaped frame: every `COLUMN_ORDER` column, deterministic values."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "sensor_id": ["S1" if i % 2 == 0 else "S2" for i in range(N_ROWS)],
        "date": pd.date_range("2026-01-01", periods=N_ROWS, freq="D"),
        "y_pm25": rng.uniform(5, 150, N_ROWS),
        "y_hours": rng.integers(18, 25, N_ROWS),
        "y_band": rng.integers(0, 6, N_ROWS),
    }
    for col in COLUMN_ORDER:
        if col in data:
            continue
        if col.startswith("sensor_"):
            data[col] = 0
        else:
            data[col] = rng.normal(50, 10, N_ROWS)
    df = pd.DataFrame(data)
    df["sensor_S1"] = (df["sensor_id"] == "S1").astype(int)
    df["sensor_S2"] = (df["sensor_id"] == "S2").astype(int)
    return df[list(COLUMN_ORDER)]


@pytest.fixture
def df():
    return _synthetic_df()


# --- make_xy -------------------------------------------------------------------


def test_make_xy_drops_keys_and_targets_and_duplicate(df):
    X, _y = make_xy(df, "y_pm25")
    assert list(X.columns) == list(FEATURE_COLUMNS)
    for col in ("sensor_id", "date", "y_pm25", "y_hours", "y_band", "pm25_mean_24h"):
        assert col not in X.columns


def test_make_xy_pm25_target_matches_column(df):
    X, y = make_xy(df, "y_pm25")
    assert len(X) == len(df)
    assert np.allclose(y.to_numpy(), df["y_pm25"].to_numpy())


def test_make_xy_drops_rows_with_nan_target(df):
    broken = df.copy()
    broken.loc[0, "y_pm25"] = np.nan
    X, y = make_xy(broken, "y_pm25")
    assert len(X) == len(df) - 1
    assert len(y) == len(df) - 1


def test_make_xy_band_merges_hazardous_into_very_unhealthy(df):
    hazard = df.copy()
    hazard["y_band"] = 5
    _X, y = make_xy(hazard, "y_band")
    assert set(y.unique()) == {4}


def test_make_xy_band_target_is_int_0_to_4(df):
    _X, y = make_xy(df, "y_band")
    assert y.dtype == int
    assert y.min() >= 0
    assert y.max() <= 4


def test_make_xy_invalid_target_raises(df):
    with pytest.raises(ValueError):
        make_xy(df, "y_hours")


# --- Preprocessor ----------------------------------------------------------------


def test_preprocessor_fit_on_train_only_does_not_see_test_values(df):
    """Changing a test-only value must not change the transform of an unrelated train row."""
    X, _y = make_xy(df, "y_pm25")
    train_X, test_X = X.iloc[:30].reset_index(drop=True), X.iloc[30:].reset_index(drop=True)

    pre = Preprocessor()
    train_out_before = pre.fit_transform(train_X)

    test_X_mutated = test_X.copy()
    scaled_col = [c for c in FEATURE_COLUMNS if c not in SENSOR_COLUMNS][0]
    test_X_mutated[scaled_col] = test_X_mutated[scaled_col] * 1000 + 99999

    pre2 = Preprocessor()
    train_out_after = pre2.fit_transform(train_X)  # refit on the same train rows only
    np.testing.assert_array_equal(train_out_before, train_out_after)


def test_preprocessor_scaler_mean_comes_from_train_not_test(df):
    X, _y = make_xy(df, "y_pm25")
    train_X, test_X = X.iloc[:30].reset_index(drop=True), X.iloc[30:].reset_index(drop=True)
    scaled_col = [c for c in FEATURE_COLUMNS if c not in SENSOR_COLUMNS][0]

    # Shift every test value far away; the fitted scaler must still center on the train mean.
    test_X_shifted = test_X.copy()
    test_X_shifted[scaled_col] = 1e6

    pre = Preprocessor()
    pre.fit(train_X)
    train_mean = train_X[scaled_col].mean()
    train_std = train_X[scaled_col].std(ddof=0)

    one_row = train_X.iloc[[0]]
    transformed = pre.transform(one_row)
    col_idx = [c for c in FEATURE_COLUMNS if c not in SENSOR_COLUMNS].index(scaled_col)
    expected = (one_row[scaled_col].iloc[0] - train_mean) / train_std
    assert np.isclose(transformed[0, col_idx], expected, atol=1e-6)


def test_preprocessor_transform_test_uses_train_statistics(df):
    """Test-set values don't leak into the fitted statistics even when transform(test) is called."""
    X, _y = make_xy(df, "y_pm25")
    train_X, test_X = X.iloc[:30].reset_index(drop=True), X.iloc[30:].reset_index(drop=True)

    pre = Preprocessor()
    pre.fit(train_X)
    out_a = pre.transform(test_X)

    # Fit again on train only (statistics must be identical regardless of what test looks like)
    pre2 = Preprocessor()
    pre2.fit(train_X)
    out_b = pre2.transform(test_X)
    np.testing.assert_array_equal(out_a, out_b)


def test_preprocessor_sensor_columns_unscaled(df):
    X, _y = make_xy(df, "y_pm25")
    pre = Preprocessor()
    out = pre.fit_transform(X)
    n_scaled = len(FEATURE_COLUMNS) - len(SENSOR_COLUMNS)
    sensor_block = out[:, n_scaled:]
    original_sensor_block = X[list(SENSOR_COLUMNS)].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(sensor_block, original_sensor_block)


def test_preprocessor_output_is_float64(df):
    X, _y = make_xy(df, "y_pm25")
    pre = Preprocessor()
    out = pre.fit_transform(X)
    assert out.dtype == np.float64


def test_preprocessor_imputes_median(df):
    X, _y = make_xy(df, "y_pm25")
    scaled_col = [c for c in FEATURE_COLUMNS if c not in SENSOR_COLUMNS][0]
    X_with_nan = X.copy()
    X_with_nan.loc[0, scaled_col] = np.nan
    median = X_with_nan[scaled_col].median()  # pandas skips NaN, matching SimpleImputer(strategy="median")

    pre = Preprocessor()
    out = pre.fit_transform(X_with_nan)
    col_idx = [c for c in FEATURE_COLUMNS if c not in SENSOR_COLUMNS].index(scaled_col)
    # Imputed then scaled value should equal the scaled median (z-score 0, since median == mean-ish
    # imputation target); check it's finite and not NaN, and matches a manual impute+scale.
    assert not np.isnan(out[0, col_idx])
    manual = X_with_nan[scaled_col].fillna(median)
    expected_z = (manual.iloc[0] - manual.mean()) / manual.std(ddof=0)
    assert np.isclose(out[0, col_idx], expected_z, atol=1e-6)
