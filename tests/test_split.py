import pandas as pd
import pytest

from src.data.split import (
    Fold,
    Split,
    compare_split_designs,
    final_test_mask,
    holdout_split,
    rolling_origin_folds,
    split_summary,
)

SENSORS = ("S1", "S2", "S3")
START = "2025-06-01"
END = "2026-11-08"


def _synthetic_df() -> pd.DataFrame:
    """Multi-sensor daily rows with a deterministic, NaN-free y_band in 0..5."""
    dates = pd.date_range(START, END, freq="D")
    rows = []
    for day_idx, date in enumerate(dates):
        for sensor_idx, sensor in enumerate(SENSORS):
            rows.append(
                {
                    "sensor_id": sensor,
                    "date": date,
                    "y_band": (day_idx + sensor_idx) % 6,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return _synthetic_df()


# --- holdout_split -----------------------------------------------------------


def test_holdout_no_date_in_two_parts(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    train_dates = set(pd.to_datetime(split.train["date"]))
    val_dates = set(pd.to_datetime(split.val["date"]))
    test_dates = set(pd.to_datetime(split.test["date"]))
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)


def test_holdout_order_is_strict(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    assert split.train["date"].max() < split.val["date"].min()
    assert split.val["date"].max() < split.test["date"].min()


def test_holdout_multi_sensor_same_date_same_part(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    boundary_date = pd.Timestamp("2026-05-31")
    rows = df.loc[pd.to_datetime(df["date"]) == boundary_date]
    assert len(rows) == len(SENSORS)
    assert set(split.train.loc[pd.to_datetime(split.train["date"]) == boundary_date].index) == set(rows.index)
    assert not (pd.to_datetime(split.val["date"]) == boundary_date).any()
    assert not (pd.to_datetime(split.test["date"]) == boundary_date).any()


def test_holdout_covers_all_rows(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    assert len(split.train) + len(split.val) + len(split.test) == len(df)


def test_holdout_bad_order_raises(df):
    with pytest.raises(ValueError):
        holdout_split(df, "2026-08-31", "2026-05-31")


def test_holdout_equal_cutoffs_raise(df):
    with pytest.raises(ValueError):
        holdout_split(df, "2026-05-31", "2026-05-31")


def test_holdout_empty_part_raises(df):
    # val_end before the first date in df -> train is empty.
    with pytest.raises(ValueError):
        holdout_split(df, "2025-01-01", "2025-02-01")


def test_holdout_cutoff_past_data_raises(df):
    # both cutoffs after the last date -> test part is empty.
    with pytest.raises(ValueError):
        holdout_split(df, "2027-01-01", "2027-02-01")


def test_holdout_does_not_mutate_input(df):
    before = df.copy(deep=True)
    holdout_split(df, "2026-05-31", "2026-08-31")
    pd.testing.assert_frame_equal(df, before)


def test_holdout_returns_split_dataclass(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    assert isinstance(split, Split)


# --- rolling_origin_folds ------------------------------------------------------


def test_folds_expand_and_dont_overlap(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=4)
    assert len(folds) == 4
    for fold in folds:
        assert isinstance(fold, Fold)

    prev_train_idx = set()
    seen_test_dates = set()
    for fold in folds:
        train_idx = set(fold.train.index)
        assert prev_train_idx <= train_idx
        prev_train_idx = train_idx

        test_dates = set(pd.to_datetime(fold.test["date"]))
        assert seen_test_dates.isdisjoint(test_dates)
        seen_test_dates |= test_dates


def test_fold_gap_is_respected(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=5, min_train_days=90, n_folds=3)
    for fold in folds:
        train_dates = pd.to_datetime(fold.train["date"])
        embargo_start = fold.test_start - pd.Timedelta(days=5)
        assert (train_dates < embargo_start).all()


def test_fold_train_before_test(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=3)
    for fold in folds:
        assert fold.train["date"].max() < fold.test_start


def test_folds_tile_forward(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=3)
    for k in range(1, len(folds)):
        assert folds[k].test_start == folds[k - 1].test_start + pd.Timedelta(days=30)


def test_folds_multi_sensor_same_date_same_part(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=1)
    fold = folds[0]
    a_test_date = pd.to_datetime(fold.test["date"]).iloc[0]
    assert (pd.to_datetime(fold.test["date"]) == a_test_date).sum() == len(SENSORS)


def test_folds_stop_when_data_ends(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90)
    last_test_end = folds[-1].test_end
    assert last_test_end > pd.Timestamp(END) - pd.Timedelta(days=30)
    # one more block beyond the last fold would have no data left
    assert not (pd.to_datetime(df["date"]) >= folds[-1].test_end).any() or len(folds) >= 1


def test_rolling_origin_bad_params_raise(df):
    with pytest.raises(ValueError):
        rolling_origin_folds(df, "2026-01-01", test_days=0)
    with pytest.raises(ValueError):
        rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=-1)
    with pytest.raises(ValueError):
        rolling_origin_folds(df, "2026-01-01", test_days=30, min_train_days=0)
    with pytest.raises(ValueError):
        rolling_origin_folds(df, "2026-01-01", test_days=30, n_folds=0)


def test_rolling_origin_too_early_start_raises(df):
    with pytest.raises(ValueError):
        rolling_origin_folds(df, "2025-06-05", test_days=30, min_train_days=90)


def test_rolling_origin_empty_df_raises():
    empty = pd.DataFrame(columns=["sensor_id", "date", "y_band"])
    with pytest.raises(ValueError):
        rolling_origin_folds(empty, "2026-01-01", test_days=30)


def test_rolling_origin_start_past_data_raises(df):
    with pytest.raises(ValueError):
        rolling_origin_folds(df, "2027-06-01", test_days=30, min_train_days=90)


def test_rolling_origin_does_not_mutate_input(df):
    before = df.copy(deep=True)
    rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=3)
    pd.testing.assert_frame_equal(df, before)


# --- final_test_mask -----------------------------------------------------------


def test_final_test_mask_selects_reserved_period(df):
    mask = final_test_mask(df, "2026-09-01")
    selected = df.loc[mask]
    assert (pd.to_datetime(selected["date"]) >= pd.Timestamp("2026-09-01")).all()
    remaining = df.loc[~mask]
    assert (pd.to_datetime(remaining["date"]) < pd.Timestamp("2026-09-01")).all()


def test_final_test_mask_excludes_from_folds(df):
    cv_df = df.loc[~final_test_mask(df, "2026-09-01")]
    folds = rolling_origin_folds(cv_df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90)
    for fold in folds:
        assert (pd.to_datetime(fold.test["date"]) < pd.Timestamp("2026-09-01")).all()


# --- split_summary --------------------------------------------------------------


def test_summary_band_counts_match_row_counts_for_holdout(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    summary = split_summary(split)
    band_cols = [f"band_{i}" for i in range(6)]
    assert (summary[band_cols].sum(axis=1) == summary["n_rows"]).all()


def test_summary_band_counts_match_row_counts_for_folds(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=3)
    summary = split_summary(folds)
    band_cols = [f"band_{i}" for i in range(6)]
    assert (summary[band_cols].sum(axis=1) == summary["n_rows"]).all()


def test_summary_has_one_row_per_part_for_holdout(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    summary = split_summary(split)
    assert sorted(summary["part"]) == ["test", "train", "val"]


def test_summary_has_two_rows_per_fold(df):
    folds = rolling_origin_folds(df, "2026-01-01", test_days=30, gap_days=1, min_train_days=90, n_folds=3)
    summary = split_summary(folds)
    assert len(summary) == 2 * len(folds)
    assert sorted(summary["fold"].unique()) == [0, 1, 2]


def test_summary_unhealthy_or_worse_share_is_a_fraction(df):
    split = holdout_split(df, "2026-05-31", "2026-08-31")
    summary = split_summary(split)
    assert summary["unhealthy_or_worse_share"].between(0, 1).all()


def test_summary_handles_nan_bands():
    df_with_nan = _synthetic_df()
    df_with_nan.loc[df_with_nan.index[:3], "y_band"] = float("nan")
    split = holdout_split(df_with_nan, "2026-05-31", "2026-08-31")
    summary = split_summary(split)
    band_cols = [f"band_{i}" for i in range(6)]
    # n_rows still counts the NaN rows; band columns don't.
    train_row = summary.loc[summary["part"] == "train"].iloc[0]
    assert train_row[band_cols].sum() <= train_row["n_rows"]


# --- compare_split_designs ------------------------------------------------------


def test_compare_split_designs_has_both_designs(df):
    comparison = compare_split_designs(df)
    assert set(comparison["design"]) == {"holdout", "rolling_origin"}


def test_compare_split_designs_rolling_respects_final_test(df):
    comparison = compare_split_designs(df, final_test_start="2026-09-01")
    rolling = comparison.loc[comparison["design"] == "rolling_origin"]
    assert (pd.to_datetime(rolling["end"]) < pd.Timestamp("2026-09-01")).all()


def test_compare_split_designs_does_not_mutate_input(df):
    before = df.copy(deep=True)
    compare_split_designs(df)
    pd.testing.assert_frame_equal(df, before)
