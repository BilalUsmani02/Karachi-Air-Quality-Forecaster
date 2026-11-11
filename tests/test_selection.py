"""Tests for the model-selection rule in code (`src.evaluate.selection`) and its band_report wiring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import band_report as br
from src.evaluate import selection as sel
from tests.test_band_report import _write_final, _write_tuning_dir

NAN = float("nan")


def _cands(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(sel.REQUIRED_COLUMNS))


def _diffs(entries: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """{"a - b": (point, low, high)} -> bootstrap-format miss-rate rows (plus a decoy accuracy row)."""
    rows = [{"model": k, "metric": "miss_rate", "point": p, "ci_low": lo, "ci_high": hi} for k, (p, lo, hi) in entries.items()]
    rows.append({"model": "decoy", "metric": "accuracy", "point": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    return pd.DataFrame(rows)


def test_rule0_excludes_regressors_more_than_ten_percent_worse_and_keeps_direct():
    c = _cands(
        [
            ("ens", "regress_then_band", 6.0, 0.10, 0.80, 0.6),
            ("ok", "regress_then_band", 6.59, 0.10, 0.70, 0.6),
            ("ridge", "regress_then_band", 6.61, 0.01, 0.75, 0.5),
            ("clf", "direct", NAN, 0.10, 0.70, 0.6),
        ]
    )
    r = sel.apply_rule0(c)
    assert r["best_mae"] == 6.0 and r["mae_ceiling"] == pytest.approx(6.6)
    assert r["eligible"] == ["ens", "ok", "clf"]
    assert set(r["excluded"]) == {"ridge"}
    assert r["excluded"]["ridge"]["above_best_pct"] == pytest.approx(10.1667, abs=1e-3)
    assert r["direct_skip_rule0"] == ["clf"]


def test_rule0_ineligible_route_cannot_win_even_with_lowest_miss_rate():
    c = _cands(
        [
            ("ens", "regress_then_band", 6.0, 0.08, 0.82, 0.6),
            ("ridge", "regress_then_band", 7.2, 0.05, 0.90, 0.9),
        ]
    )
    out = sel.select_model(c, _diffs({}))
    assert out["chosen"] == "ens"
    assert out["steps"]["rule1_missed_bad_days"]["leader"] == "ens"


def test_rule1_rules_out_only_routes_whose_ci_excludes_zero():
    c = _cands(
        [
            ("leader", "regress_then_band", 6.0, 0.05, 0.70, 0.6),
            ("close", "regress_then_band", 6.1, 0.07, 0.80, 0.6),
            ("worse", "direct", NAN, 0.15, 0.95, 0.9),
        ]
    )
    d = _diffs({"close - leader": (0.02, -0.01, 0.05), "worse - leader": (0.10, 0.02, 0.20)})
    out = sel.select_model(c, d)
    r1 = out["steps"]["rule1_missed_bad_days"]
    assert r1["survivors"] == ["leader", "close"]
    assert r1["comparisons"]["worse"]["ruled_out"] and not r1["comparisons"]["close"]["ruled_out"]
    # "worse" has the best accuracy but was removed at rule 1, so rule 2 picks "close".
    assert out["chosen"] == "close" and out["decided_by"] == "rule2_accuracy"


def test_miss_difference_accepts_reversed_pair_and_negates_it():
    d = _diffs({"leader - x": (-0.03, -0.06, -0.01)})
    got = sel.miss_difference(d, "x", "leader")
    assert got == pytest.approx({"point": 0.03, "ci_low": 0.01, "ci_high": 0.06})


def test_missing_bootstrap_pair_raises_instead_of_passing_silently():
    c = _cands([("a", "direct", NAN, 0.05, 0.7, 0.6), ("b", "direct", NAN, 0.06, 0.8, 0.6)])
    with pytest.raises(ValueError, match="b vs a"):
        sel.select_model(c, _diffs({}))


def test_rule3_macro_f1_breaks_an_accuracy_tie():
    c = _cands(
        [
            ("a", "regress_then_band", 6.0, 0.05, 0.80, 0.60),
            ("b", "regress_then_band", 6.0, 0.06, 0.80, 0.65),
        ]
    )
    out = sel.select_model(c, _diffs({"b - a": (0.01, -0.01, 0.03)}))
    assert out["chosen"] == "b" and out["decided_by"] == "rule3_macro_f1"
    assert out["steps"]["rule3_macro_f1"]["best"] == ["b"]


def test_full_tie_keeps_first_in_input_order_and_says_so():
    c = _cands([("a", "direct", NAN, 0.05, 0.8, 0.6), ("b", "direct", NAN, 0.05, 0.8, 0.6)])
    out = sel.select_model(c, _diffs({"b - a": (0.0, -0.01, 0.01)}))
    assert out["chosen"] == "a"
    assert "still tied" in out["steps"]["rule3_macro_f1"]["reason"]


def test_leader_tie_on_miss_rate_goes_to_higher_accuracy():
    c = _cands([("a", "direct", NAN, 0.05, 0.7, 0.6), ("b", "direct", NAN, 0.05, 0.8, 0.6)])
    assert sel.miss_rate_leader(c) == "b"


@pytest.mark.parametrize(
    "frame, match",
    [
        (pd.DataFrame({"model": ["a"]}), "missing columns"),
        (_cands([]), "no candidate"),
        (_cands([("a", "direct", NAN, 0.1, 0.8, 0.6), ("a", "direct", NAN, 0.1, 0.8, 0.6)]), "unique"),
        (_cands([("a", "other", NAN, 0.1, 0.8, 0.6)]), "unknown kind"),
        (_cands([("a", "regress_then_band", NAN, 0.1, 0.8, 0.6)]), "CV MAE"),
    ],
)
def test_select_model_validates_its_input(frame, match):
    with pytest.raises(ValueError, match=match):
        sel.select_model(frame, _diffs({}))


def test_select_model_does_not_mutate_inputs():
    c = _cands([("a", "direct", NAN, 0.05, 0.7, 0.6), ("b", "direct", NAN, 0.06, 0.8, 0.6)])
    d = _diffs({"b - a": (0.01, -0.01, 0.03)})
    c0, d0 = c.copy(), d.copy()
    sel.select_model(c, d)
    pd.testing.assert_frame_equal(c, c0)
    pd.testing.assert_frame_equal(d, d0)


def test_candidates_table_labels_kinds_and_skips_unknown_models():
    metrics = pd.DataFrame(
        {
            "model": ["ensemble", "hist_gb_clf", "other"],
            "miss_rate": [0.1, 0.2, 0.3],
            "accuracy": [0.8, 0.7, 0.6],
            "macro_f1_all": [0.6, 0.5, 0.4],
        }
    )
    t = sel.candidates_table(metrics, {"ensemble": 6.2}, ["ensemble"], ["hist_gb_clf"])
    assert list(t["model"]) == ["ensemble", "hist_gb_clf"]
    assert list(t["kind"]) == ["regress_then_band", "direct"]
    assert t.loc[0, "cv_mae"] == 6.2 and np.isnan(t.loc[1, "cv_mae"])


# --- band_report wiring --------------------------------------------------------------------


def test_load_cv_mae_reads_every_regress_then_band_route(tmp_path):
    mae = br.load_cv_mae(_write_tuning_dir(tmp_path / "tuning"))
    assert set(mae) == set(br.SELECTION_REGRESSORS)
    assert mae["ensemble"] == pytest.approx(0.0)  # +3 and -3 noise cancel in the average
    assert mae["hist_gb"] == pytest.approx(3.0) and mae["ridge"] == pytest.approx(8.0)


def test_build_report_writes_selection_json_with_a_reason_per_step(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    out = tmp_path / "bands"
    results = br.build_report(
        br.load_cv_bands(tuning),
        br.load_final_bands(_write_final(tmp_path / "final.parquet")),
        out,
        n_boot=30,
        seed=1,
        cv_mae=br.load_cv_mae(tuning),
    )
    saved = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert saved["chosen"] == results["selection"]["chosen"]
    assert saved["n_boot"] == 30 and saved["seed"] == 1
    for step in ("rule0_eligibility", "rule1_missed_bad_days", "rule2_accuracy"):
        assert saved["steps"][step]["reason"]
    # Ridge (+8 noise) and persistence are far above the ensemble's MAE, so rule 0 drops them.
    assert {"ridge", "persistence"} <= set(saved["steps"]["rule0_eligibility"]["excluded"])
    assert "selection.json" in (out / "summary.md").read_text(encoding="utf-8")


def test_build_report_without_cv_mae_skips_selection(tmp_path):
    tuning = _write_tuning_dir(tmp_path / "tuning")
    out = tmp_path / "bands"
    results = br.build_report(
        br.load_cv_bands(tuning), br.load_final_bands(_write_final(tmp_path / "final.parquet")), out, n_boot=10
    )
    assert "selection" not in results and not (out / "selection.json").exists()
