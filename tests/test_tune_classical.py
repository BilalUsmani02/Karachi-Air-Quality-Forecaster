import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import tune, tune_classical
from src.evaluate.tune import prepare_folds
from tests.test_run_cv import _synthetic_frame


class MeanRegressor:
    def __init__(self, offset: float = 0.0, **_ignored):
        self.offset = offset

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_ + self.offset)


class ConstantClassifier:
    def __init__(self, label: int = 1, **_ignored):
        self.label = label

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.label)


@pytest.fixture(scope="module")
def folds():
    df = _synthetic_frame(sensor_ids=("S1", "S2")).sort_values(["sensor_id", "date"])
    return prepare_folds(df)


@pytest.fixture(scope="module")
def features_path(tmp_path_factory):
    df = _synthetic_frame(sensor_ids=("S1",))
    path = tmp_path_factory.mktemp("data") / "features.parquet"
    df.to_parquet(path)
    return path


# --- multi_stage_search -------------------------------------------------------------------


def test_multi_stage_search_picks_best_regressor_across_stages(folds):
    stage0_candidates = [{"offset": 50.0}, {"offset": 20.0}]
    stages = [
        lambda best: stage0_candidates,
        lambda best: [{**best, "offset": best["offset"] / 2}, {**best, "offset": 0.0}],
    ]
    trials, best = tune_classical.multi_stage_search(folds, MeanRegressor, "regression", stages)

    assert len(trials) == 4
    assert set(trials["stage"]) == {0, 1}
    assert trials["mae"].is_monotonic_increasing

    # Stage 1's candidates depend on stage 0's winner, so recompute the full set by hand and
    # check the search found the true pooled-MAE minimum across every candidate it tried.
    stage0_best = min(stage0_candidates, key=lambda p: tune.score_regressor(folds, MeanRegressor, p).metrics["mae"])
    stage1_candidates = [{**stage0_best, "offset": stage0_best["offset"] / 2}, {**stage0_best, "offset": 0.0}]
    all_candidates = stage0_candidates + stage1_candidates
    expected_best = min(all_candidates, key=lambda p: tune.score_regressor(folds, MeanRegressor, p).metrics["mae"])
    assert best.params == expected_best


def test_multi_stage_search_picks_best_classifier_across_stages(folds):
    stages = [lambda best: [{"label": 4}, {"label": 1}]]
    trials, best = tune_classical.multi_stage_search(folds, ConstantClassifier, "classification", stages)

    assert best.params["label"] == 1
    assert trials["macro_f1"].is_monotonic_decreasing


def test_multi_stage_search_best_has_predictions(folds):
    stages = [lambda best: [{"offset": 0.0}]]
    _trials, best = tune_classical.multi_stage_search(folds, MeanRegressor, "regression", stages)
    assert len(best.predictions) == sum(len(f.y_test) for f in folds)


def test_multi_stage_search_rejects_unknown_kind(folds):
    with pytest.raises(ValueError):
        tune_classical.multi_stage_search(folds, MeanRegressor, "ranking", [lambda best: [{}]])


# --- per-family quick stage grids ------------------------------------------------------------


@pytest.mark.parametrize("spec", tune_classical.FAMILIES, ids=lambda s: s.name)
def test_quick_stages_are_tiny(spec):
    stages = spec.stages(True)
    assert len(stages) == 1
    assert len(stages[0]({})) <= 2


def test_family_names_match_the_deliverable_list():
    names = {f.name for f in tune_classical.FAMILIES}
    assert names == {
        "ridge",
        "random_forest",
        "hist_gb",
        "logistic_regression",
        "random_forest_clf",
        "hist_gb_clf",
    }


def test_regression_and_classification_kinds_are_split():
    kinds = {f.name: f.kind for f in tune_classical.FAMILIES}
    assert kinds["ridge"] == kinds["random_forest"] == kinds["hist_gb"] == "regression"
    assert kinds["logistic_regression"] == kinds["random_forest_clf"] == kinds["hist_gb_clf"] == "classification"


# --- full (non-quick) stage grids: checked without fitting anything ----------------------------


@pytest.mark.parametrize("spec", tune_classical.FAMILIES, ids=lambda s: s.name)
def test_full_stages_refine_a_complete_running_best(spec):
    best: dict = {}
    for stage in spec.stages(False):
        candidates = stage(best)
        assert candidates, "every stage proposes at least one candidate"
        assert all(set(best) <= set(c) for c in candidates), "later stages keep earlier choices"
        assert len({tuple(sorted(map(str, c.items()))) for c in candidates}) == len(candidates)
        best = candidates[-1]
    spec.factory(**best)  # the final configuration is a valid estimator


# --- run() end-to-end on synthetic data, --quick (run once, shared) ----------------------------


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory, features_path):
    results_dir = tmp_path_factory.mktemp("tuning")
    outcomes = tune_classical.run(features_path=features_path, results_dir=results_dir, quick=True)
    return outcomes, results_dir


def test_run_quick_writes_all_family_files(quick_run):
    outcomes, results_dir = quick_run

    assert set(outcomes) == {f.name for f in tune_classical.FAMILIES}
    for spec in tune_classical.FAMILIES:
        assert (results_dir / f"{spec.name}_trials.csv").exists()
        assert (results_dir / f"{spec.name}_best.json").exists()
        assert (results_dir / f"{spec.name}_cv_predictions.parquet").exists()
    assert (results_dir / "classical_summary.md").exists()


def test_run_quick_best_json_has_params_and_metrics(quick_run):
    _, results_dir = quick_run

    payload = json.loads((results_dir / "ridge_best.json").read_text(encoding="utf-8"))
    assert payload["family"] == "ridge"
    assert payload["kind"] == "regression"
    assert "alpha" in payload["params"]
    assert {"mae", "rmse", "r2", "skill", "band_macro_f1"} <= set(payload["metrics"])

    clf_payload = json.loads((results_dir / "logistic_regression_best.json").read_text(encoding="utf-8"))
    assert {"accuracy", "macro_f1", "per_class"} <= set(clf_payload["metrics"])


def test_run_quick_predictions_parquet_is_pooled_cv(quick_run):
    _, results_dir = quick_run

    preds = pd.read_parquet(results_dir / "random_forest_cv_predictions.parquet")
    assert {"fold", "sensor_id", "date", "y_true", "y_pred"} <= set(preds.columns)


def test_run_quick_summary_mentions_every_family(quick_run):
    _, results_dir = quick_run

    text = (results_dir / "classical_summary.md").read_text(encoding="utf-8")
    for spec in tune_classical.FAMILIES:
        assert spec.name in text


# --- CLI ---------------------------------------------------------------------------------------


def test_main_passes_quick_and_features_through(monkeypatch, tmp_path):
    captured = {}

    def fake_run(features_path=None, quick=False):
        captured["features_path"] = features_path
        captured["quick"] = quick

    monkeypatch.setattr(tune_classical, "run", fake_run)
    features_arg = tmp_path / "features.parquet"
    tune_classical.main(["--features", str(features_arg), "--quick"])

    assert captured["quick"] is True
    assert captured["features_path"] == features_arg


def test_main_defaults_features_to_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(tune_classical, "run", lambda features_path=None, quick=False: captured.update(
        features_path=features_path, quick=quick
    ))
    tune_classical.main([])

    assert captured["features_path"] is None
    assert captured["quick"] is False
