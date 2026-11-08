"""Time-based train/val/test splitting for the daily feature table (contract §4).

Works on any frame with a `date` column (one row per sensor-day) and, for the
summary helpers, a `y_band` column (int band index 0-5, contract §3). Splits are
computed on calendar dates and applied to **all sensors at once**, so the same
date is never split across two parts — there is no sensor-wise leakage.

All functions are pure: they read `df`, never assign into it, and return new
DataFrames (boolean-mask indexing already copies, so the input is untouched).
Nothing is ever shuffled — order is time, start to finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

DATE_COL = "date"
BAND_COL = "y_band"
N_BANDS = 6
UNHEALTHY_OR_WORSE = 3  # band index >= 3 is "Unhealthy" or worse (contract §3)

DateLike = str | pd.Timestamp


def _dates(df: pd.DataFrame) -> pd.Series:
    """The `date` column as `Timestamp`s, without touching `df` itself."""
    return pd.to_datetime(df[DATE_COL])


@dataclass(frozen=True)
class Split:
    """Train/validation/test parts of a daily feature table.

    Each part is a DataFrame (a filtered copy of the input, not a view), kept
    simple so downstream code (models, scalers) can use it directly without
    re-indexing into the original frame.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class Fold:
    """One rolling-origin cross-validation fold (contract §4)."""

    index: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def holdout_split(df: pd.DataFrame, train_end: DateLike, val_end: DateLike) -> Split:
    """Split by date cut-offs: train <= train_end < val <= val_end < test.

    Both cut-offs are inclusive end dates of their own part (e.g. `train_end=
    "2026-05-31"` keeps 31 May in train, 1 June starts val). Applies to every
    sensor at once, so no date appears in two parts.

    Raises `ValueError` if the cut-offs are not ordered, or if any part would
    be empty (e.g. a cut-off outside the data's date range).
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if not train_end_ts < val_end_ts:
        raise ValueError(f"train_end ({train_end_ts.date()}) must be before val_end ({val_end_ts.date()})")

    dates = _dates(df)
    train_mask = dates <= train_end_ts
    val_mask = (dates > train_end_ts) & (dates <= val_end_ts)
    test_mask = dates > val_end_ts

    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not mask.any():
            raise ValueError(
                f"{name} part is empty for train_end={train_end_ts.date()}, val_end={val_end_ts.date()} "
                f"(data spans {dates.min().date()} to {dates.max().date()})"
            )

    return Split(train=df.loc[train_mask].copy(), val=df.loc[val_mask].copy(), test=df.loc[test_mask].copy())


def final_test_mask(df: pd.DataFrame, test_start: DateLike) -> pd.Series:
    """Boolean mask selecting the reserved final-test period (`date >= test_start`).

    Filter it out of anything used for model selection, e.g.:
    `cv_df = df.loc[~final_test_mask(df, "2026-09-01")]`
    so rolling-origin folds (and the holdout val/train split, if reused for tuning)
    never see the final test period. It is touched once, after all choices are
    frozen (spec §5.3, roadmap Phase 5).
    """
    return _dates(df) >= pd.Timestamp(test_start)


def rolling_origin_folds(
    df: pd.DataFrame,
    first_test_start: DateLike,
    test_days: int,
    n_folds: int | None = None,
    gap_days: int = 1,
    min_train_days: int = 90,
) -> list[Fold]:
    """Expanding-window, time-ordered cross-validation folds.

    Fold `k` trains on every date strictly before `test_start_k - gap_days` and
    tests on the `test_days`-day block `[test_start_k, test_start_k + test_days)`.
    `test_start_0 = first_test_start`; later folds tile forward by `test_days`
    (`test_start_{k+1} = test_start_k + test_days`), so test blocks never overlap
    and each fold's train set is a superset of the previous one's.

    Stops when the next test block has no rows left in `df`, or once `n_folds`
    folds have been produced, whichever comes first.

    `gap_days` is an embargo: it keeps the `gap_days` immediately before a test
    block out of training, since neighbouring days are correlated (lag/rolling
    features on day `test_start - 1` would otherwise leak into a fold whose test
    block starts the next day). `min_train_days` is the minimum calendar-day span
    the first fold's training window must cover, so an early `first_test_start`
    fails loudly instead of silently training on too little history.

    Raises `ValueError` on bad parameters, or if the very first fold would not
    have `min_train_days` of training history.
    """
    if test_days < 1:
        raise ValueError(f"test_days must be >= 1, got {test_days}")
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
    if min_train_days < 1:
        raise ValueError(f"min_train_days must be >= 1, got {min_train_days}")
    if n_folds is not None and n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if df.empty:
        raise ValueError("df has no rows")

    dates = _dates(df)
    data_min, data_max = dates.min(), dates.max()
    first_test_start_ts = pd.Timestamp(first_test_start)
    gap = pd.Timedelta(days=gap_days)
    span = pd.Timedelta(days=test_days)

    first_train_end_excl = first_test_start_ts - gap
    first_train_span_days = (first_train_end_excl - data_min).days
    if first_train_span_days < min_train_days:
        raise ValueError(
            f"first_test_start={first_test_start_ts.date()} leaves only {first_train_span_days} "
            f"train day(s) before the gap (need >= {min_train_days}); data starts {data_min.date()}"
        )

    folds: list[Fold] = []
    k = 0
    while n_folds is None or len(folds) < n_folds:
        test_start = first_test_start_ts + k * span
        test_end = test_start + span
        test_mask = (dates >= test_start) & (dates < test_end)
        if not test_mask.any():
            break

        train_end_excl = test_start - gap
        train_mask = dates < train_end_excl

        folds.append(
            Fold(
                index=k,
                train=df.loc[train_mask].copy(),
                test=df.loc[test_mask].copy(),
                train_start=dates[train_mask].min(),
                train_end=dates[train_mask].max(),
                test_start=test_start,
                test_end=test_end,
            )
        )
        k += 1

    if not folds:
        raise ValueError(
            f"no folds produced: first_test_start={first_test_start_ts.date()} is past the data's "
            f"last date ({data_max.date()})"
        )
    return folds


def _part_summary(frame: pd.DataFrame, part: str, fold: int | None = None) -> dict:
    dates = _dates(frame)
    bands = frame[BAND_COL] if BAND_COL in frame.columns else pd.Series(dtype=float)
    band_counts = bands.value_counts(dropna=True).reindex(range(N_BANDS), fill_value=0).astype(int)
    known = int(band_counts.sum())
    unhealthy_or_worse = int(band_counts.loc[UNHEALTHY_OR_WORSE:].sum())

    row = {
        "fold": fold,
        "part": part,
        "start": dates.min(),
        "end": dates.max(),
        "n_days": int(dates.nunique()),
        "n_rows": len(frame),
        "unhealthy_or_worse_share": (unhealthy_or_worse / known) if known else float("nan"),
    }
    for band in range(N_BANDS):
        row[f"band_{band}"] = int(band_counts.loc[band])
    return row


def split_summary(parts: Split | Sequence[Fold]) -> pd.DataFrame:
    """One row per part (holdout) or per train/test block per fold (rolling-origin).

    Columns: `fold` (NaN for a holdout `Split`), `part` (`train`/`val`/`test`),
    `start`, `end`, `n_days`, `n_rows`, `band_0`..`band_5`, and
    `unhealthy_or_worse_share` (share of rows with a known band that are band
    index >= 3, i.e. Unhealthy or worse). Rows with no `y_band` (e.g. still NaN)
    are counted in `n_rows` but excluded from the band counts and the share.
    """
    if isinstance(parts, Split):
        rows = [_part_summary(parts.train, "train"), _part_summary(parts.val, "val"), _part_summary(parts.test, "test")]
    else:
        rows = []
        for fold in parts:
            rows.append(_part_summary(fold.train, "train", fold.index))
            rows.append(_part_summary(fold.test, "test", fold.index))
    return pd.DataFrame(rows)


def compare_split_designs(
    df: pd.DataFrame,
    train_end: DateLike = "2026-05-31",
    val_end: DateLike = "2026-08-31",
    first_test_start: DateLike = "2026-01-01",
    test_days: int = 30,
    gap_days: int = 1,
    min_train_days: int = 90,
    final_test_start: DateLike = "2026-09-01",
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Holdout summary next to a rolling-origin summary, for a human to compare.

    Defaults follow the recommendation in `docs/report/data_exploration.md`: a
    plain holdout (train through May 2026, val = monsoon Jun-Aug 2026, test =
    Sep-Nov 2026) next to 30-day rolling-origin folds starting early 2026, with
    Sep-Nov 2026 reserved as the final test in both cases (never used to pick
    between them). Only `df`'s `date`/`y_band` columns are required; this feeds
    a decision, not a model, so it stays a thin wrapper over `holdout_split`,
    `rolling_origin_folds` and `split_summary`.
    """
    holdout = holdout_split(df, train_end, val_end)
    holdout_rows = split_summary(holdout)
    holdout_rows.insert(0, "design", "holdout")

    cv_df = df.loc[~final_test_mask(df, final_test_start)].copy()
    folds = rolling_origin_folds(
        cv_df, first_test_start, test_days, n_folds=n_folds, gap_days=gap_days, min_train_days=min_train_days
    )
    rolling_rows = split_summary(folds)
    rolling_rows.insert(0, "design", "rolling_origin")

    return pd.concat([holdout_rows, rolling_rows], ignore_index=True)
