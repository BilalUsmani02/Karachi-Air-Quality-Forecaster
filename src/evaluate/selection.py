"""The project's model-selection rule, applied in code (see `docs/report/model_selection.md`).

Rule, in priority order:

0. **Eligibility.** A regress-then-band route qualifies only if its regressor's pooled-CV MAE is
   within `mae_tolerance` (10%) of the best route's MAE. Direct band classifiers have no MAE and
   go straight to rule 1.
1. **Fewest missed Unhealthy-or-worse days.** The eligible route with the lowest miss rate is the
   *leader*. Another route is ruled out only if it misses *clearly* more: the 95% date-block
   bootstrap interval of (route - leader) miss rate lies entirely above zero.
2. **Then accuracy** (highest point estimate among the routes left).
3. **Then macro-F1** over all five bands (only if accuracy is tied).

Everything here is pure: `select_model` takes the saved CV metrics and the bootstrap differences
and returns the decision with a reason for every step; it reads and writes nothing.
`band_report` computes the inputs, calls it and saves the result as `selection.json`.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

MAE_TOLERANCE = 0.10
REQUIRED_COLUMNS: tuple[str, ...] = ("model", "kind", "cv_mae", "miss_rate", "accuracy", "macro_f1_all")
KINDS: tuple[str, ...] = ("regress_then_band", "direct")


def _check_candidates(candidates: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidates table is missing columns: {missing}")
    if candidates.empty:
        raise ValueError("no candidate models")
    if candidates["model"].duplicated().any():
        raise ValueError("candidate model names must be unique")
    bad = set(candidates["kind"]) - set(KINDS)
    if bad:
        raise ValueError(f"unknown kind(s): {sorted(bad)}")
    reg = candidates.loc[candidates["kind"] == "regress_then_band", "cv_mae"]
    if reg.isna().any():
        raise ValueError("every regress-then-band candidate needs a CV MAE")


def apply_rule0(candidates: pd.DataFrame, mae_tolerance: float = MAE_TOLERANCE) -> dict:
    """Rule 0: which regress-then-band routes are close enough to the best CV MAE."""
    reg = candidates.loc[candidates["kind"] == "regress_then_band"]
    best_mae = float(reg["cv_mae"].min()) if not reg.empty else float("nan")
    ceiling = best_mae * (1 + mae_tolerance)
    excluded = {}
    for row in reg.itertuples(index=False):
        if row.cv_mae > ceiling:
            excluded[row.model] = {
                "cv_mae": float(row.cv_mae),
                "above_best_pct": float(100 * (row.cv_mae / best_mae - 1)),
            }
    eligible = [m for m in candidates["model"] if m not in excluded]
    return {
        "best_mae": best_mae,
        "mae_tolerance": mae_tolerance,
        "mae_ceiling": ceiling,
        "eligible": eligible,
        "excluded": excluded,
        "direct_skip_rule0": [m for m in candidates.loc[candidates["kind"] == "direct", "model"]],
    }


def miss_rate_leader(candidates: pd.DataFrame) -> str:
    """The route with the lowest miss rate (ties broken by accuracy, then macro-F1, then input order)."""
    ranked = candidates.sort_values(
        ["miss_rate", "accuracy", "macro_f1_all"], ascending=[True, False, False], kind="stable"
    )
    return str(ranked.iloc[0]["model"])


def miss_difference(diffs: pd.DataFrame, model: str, leader: str) -> dict:
    """The (model - leader) miss-rate difference and its 95% CI, from bootstrap difference rows.

    Accepts either orientation ("model - leader" or "leader - model", which is negated).
    Raises `ValueError` if neither pair was bootstrapped: rule 1 must not pass a route silently.
    """
    rows = diffs.loc[diffs["metric"] == "miss_rate"].set_index("model")
    forward, backward = f"{model} - {leader}", f"{leader} - {model}"
    if forward in rows.index:
        r = rows.loc[forward]
        return {"point": float(r["point"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])}
    if backward in rows.index:
        r = rows.loc[backward]
        return {"point": -float(r["point"]), "ci_low": -float(r["ci_high"]), "ci_high": -float(r["ci_low"])}
    raise ValueError(f"no bootstrap miss-rate difference for {model} vs {leader}")


def apply_rule1(candidates: pd.DataFrame, diffs: pd.DataFrame) -> dict:
    """Rule 1: drop routes whose miss rate is clearly (CI excludes zero) above the leader's."""
    leader = miss_rate_leader(candidates)
    comparisons, survivors = {}, []
    for model in candidates["model"]:
        if model == leader:
            survivors.append(model)
            continue
        diff = miss_difference(diffs, model, leader)
        clearly_worse = diff["ci_low"] > 0
        comparisons[model] = {**diff, "ruled_out": bool(clearly_worse)}
        if not clearly_worse:
            survivors.append(model)
    return {"leader": leader, "comparisons": comparisons, "survivors": survivors}


def _best_by(frame: pd.DataFrame, column: str) -> list[str]:
    top = frame[column].max()
    return [str(m) for m in frame.loc[np.isclose(frame[column], top, rtol=0, atol=1e-12), "model"]]


def select_model(candidates: pd.DataFrame, diffs: pd.DataFrame, mae_tolerance: float = MAE_TOLERANCE) -> dict:
    """Apply rules 0-3 and return the chosen model with the reason for each step.

    `candidates`: one row per route with `model`, `kind` ("regress_then_band" or "direct"),
    `cv_mae` (NaN for direct classifiers), `miss_rate`, `accuracy`, `macro_f1_all`, all on the
    pooled CV rows. `diffs`: bootstrap rows in `band_stats.date_block_bootstrap` format
    (`model` = "a - b", `metric`, `point`, `ci_low`, `ci_high`); it must hold the miss-rate
    difference between the rule-1 leader and every other eligible route. Inputs are not modified.
    """
    _check_candidates(candidates)
    rule0 = apply_rule0(candidates, mae_tolerance)
    eligible = candidates.loc[candidates["model"].isin(rule0["eligible"])]
    rule1 = apply_rule1(eligible, diffs)
    left = eligible.loc[eligible["model"].isin(rule1["survivors"])]

    steps = {
        "rule0_eligibility": {
            **rule0,
            "reason": (
                f"Regress-then-band routes need a CV MAE within {mae_tolerance:.0%} of the best "
                f"({rule0['best_mae']:.2f}, ceiling {rule0['mae_ceiling']:.2f}). "
                f"Excluded: {', '.join(rule0['excluded']) or 'none'}. Direct classifiers skip this rule."
            ),
        },
        "rule1_missed_bad_days": {
            **rule1,
            "reason": (
                f"Leader on missed Unhealthy-or-worse days: {rule1['leader']}. Ruled out (95% CI of the "
                f"difference above zero): {', '.join(m for m, c in rule1['comparisons'].items() if c['ruled_out']) or 'none'}."
            ),
        },
    }

    by_accuracy = _best_by(left, "accuracy")
    steps["rule2_accuracy"] = {
        "accuracy": {str(m): float(a) for m, a in zip(left["model"], left["accuracy"])},
        "best": by_accuracy,
        "reason": f"Highest accuracy among {', '.join(left['model'])}: {', '.join(by_accuracy)}.",
    }
    if len(by_accuracy) == 1:
        chosen, decided_by = by_accuracy[0], "rule2_accuracy"
    else:
        tied = left.loc[left["model"].isin(by_accuracy)]
        by_f1 = _best_by(tied, "macro_f1_all")
        chosen, decided_by = by_f1[0], "rule3_macro_f1"
        steps["rule3_macro_f1"] = {
            "macro_f1_all": {str(m): float(f) for m, f in zip(tied["model"], tied["macro_f1_all"])},
            "best": by_f1,
            "reason": (
                f"Accuracy tied between {', '.join(by_accuracy)}; highest macro-F1: {', '.join(by_f1)}"
                + ("." if len(by_f1) == 1 else "; still tied, so the first in input order is kept.")
            ),
        }
    return {"chosen": chosen, "decided_by": decided_by, "steps": steps}


def candidates_table(
    metrics: pd.DataFrame,
    cv_mae: Mapping[str, float],
    regressors: Iterable[str],
    direct: Iterable[str],
) -> pd.DataFrame:
    """Build the `select_model` input from a `band_report.metrics_table` frame (one row set).

    `regressors` are the regress-then-band routes (each needs an entry in `cv_mae`); `direct`
    are direct classifiers. Models missing from `metrics` are skipped.
    """
    regressors, direct = list(regressors), list(direct)
    rows = []
    for row in metrics.itertuples(index=False):
        if row.model in regressors:
            kind, mae = "regress_then_band", float(cv_mae[row.model])
        elif row.model in direct:
            kind, mae = "direct", float("nan")
        else:
            continue
        rows.append(
            {
                "model": row.model,
                "kind": kind,
                "cv_mae": mae,
                "miss_rate": float(row.miss_rate),
                "accuracy": float(row.accuracy),
                "macro_f1_all": float(row.macro_f1_all),
            }
        )
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
