import numpy as np
import pandas as pd
import pytest

from src.data.split import FINAL_TEST_START
from src.evaluate import tune
from tests.test_run_cv import _synthetic_frame


class MeanRegressor:
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_ + self.offset)


class ConstantClassifier:
    def __init__(self, label: int = 1):
        self.label = label

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.label)


@pytest.fixture(scope="module")
def folds():
    df = _synthetic_frame(sensor_ids=("S1", "S2")).sort_values(["sensor_id", "date"])
    return tune.prepare_folds(df)


def test_prepare_folds_builds_eight_time_ordered_folds_without_final_test(folds):
    assert len(folds) == 8
    for f in folds:
        dates = f.meta_test["date"]
        assert dates.max() < pd.Timestamp(FINAL_TEST_START)
        assert dates.is_monotonic_increasing
        assert len(f.X_test) == len(f.y_test) == len(f.yb_test) == len(f.persistence_test)
        assert set(np.unique(f.yb_train)) <= set(range(5))


def test_score_regressor_pools_all_folds_and_adds_band_metrics(folds):
    score = tune.score_regressor(folds, MeanRegressor)

    assert len(score.predictions) == sum(len(f.y_test) for f in folds)
    assert {"mae", "rmse", "r2", "skill", "band_macro_f1", "band_accuracy"} <= set(score.metrics)
    expected_mae = np.mean(np.abs(score.predictions["y_true"] - score.predictions["y_pred"]))
    assert score.metrics["mae"] == pytest.approx(expected_mae)


def test_score_classifier_uses_band_targets(folds):
    score = tune.score_classifier(folds, ConstantClassifier, {"label": 1})

    assert (score.predictions["y_pred"] == 1).all()
    assert score.metrics["accuracy"] == pytest.approx((score.predictions["y_true"] == 1).mean())


def test_param_grid_is_cartesian_and_ordered():
    assert tune.param_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_search_ranks_regressors_by_mae(folds):
    result = tune.search(folds, MeanRegressor, [{"offset": 50.0}, {"offset": 0.0}], kind="regression", verbose=False)

    assert list(result["offset"]) == [0.0, 50.0]
    assert result["mae"].is_monotonic_increasing


def test_search_ranks_classifiers_by_macro_f1(folds, capsys):
    result = tune.search(folds, ConstantClassifier, [{"label": 4}, {"label": 1}], kind="classification")

    assert result["macro_f1"].is_monotonic_decreasing
    assert "[2/2]" in capsys.readouterr().out


def test_search_rejects_unknown_kind(folds):
    with pytest.raises(ValueError):
        tune.search(folds, MeanRegressor, [{}], kind="ranking")
