import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.split import FINAL_TEST_START
from src.evaluate import final_test
from src.models import final
from src.models.ensemble import AveragingRegressor
from tests.test_run_cv import _synthetic_frame


class Const:
    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


class MeanClassifier:
    def fit(self, X, y):
        self.label_ = int(np.bincount(y).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)


def test_averaging_regressor_averages_members():
    model = AveragingRegressor({"a": lambda: Const(10.0), "b": lambda: Const(20.0)}).fit(np.zeros((3, 2)), np.zeros(3))

    assert model.predict(np.zeros((3, 2))).tolist() == [15.0, 15.0, 15.0]
    assert set(model.predict_members(np.zeros((1, 2)))) == {"a", "b"}


def test_averaging_regressor_params_and_errors():
    model = AveragingRegressor({"a": lambda: Const(1.0)})
    assert set(model.get_params()["members"]) == {"a"}
    model.set_params(members={"b": lambda: Const(2.0)})
    assert set(model.members) == {"b"}
    with pytest.raises(ValueError):
        model.set_params(alpha=1)
    with pytest.raises(ValueError):
        AveragingRegressor({})
    with pytest.raises(RuntimeError):
        AveragingRegressor({"a": lambda: Const(1.0)}).predict(np.zeros((1, 2)))


def test_final_factories_build_the_tuned_models():
    assert final.tuned_ridge().alpha == 150.0
    assert final.tuned_hist_gb().max_iter == 100
    assert final.tuned_mlp_numpy().target == "residual"
    assert set(final.final_ensemble().members) == set(final.ENSEMBLE_MEMBERS)
    assert final.tuned_random_forest_classifier().class_weight == "balanced"


@pytest.fixture(scope="module")
def df():
    return _synthetic_frame(sensor_ids=("S1", "S2"))


def test_evaluate_trains_before_and_scores_only_the_final_period(df):
    regression, classification, table = final_test.evaluate(
        df, regressors={"const": lambda: Const(40.0)}, classifiers={"mode": MeanClassifier}
    )

    assert table["date"].min() >= pd.Timestamp(FINAL_TEST_START)
    assert set(regression["model"]) == {"persistence", "const"}
    assert regression.set_index("model").loc["persistence", "skill"] == pytest.approx(0.0)
    assert set(classification["route"]) == {"regress_then_band", "direct"}
    assert {"miss_unhealthy", "false_alarm", "recall_Good"} <= set(classification.columns)
    assert (table["band_const"] == table["band_const"].iloc[0]).all()


def test_safety_rates_definitions():
    rates = final_test._safety_rates(np.array([3, 4, 1, 0]), np.array([2, 4, 3, 0]))
    assert rates == {"miss_unhealthy": 0.5, "false_alarm": 0.5}
    assert np.isnan(final_test._safety_rates(np.array([0, 1]), np.array([0, 1]))["miss_unhealthy"])


def test_main_writes_once_then_refuses(tmp_path, df, monkeypatch):
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(final, "TUNED_REGRESSORS", {"const": lambda: Const(40.0)})
    monkeypatch.setattr(final, "TUNED_CLASSIFIERS", {})
    out = tmp_path / "final"

    final_test.main(["--features", str(path), "--out-dir", str(out)])

    for name in ("metrics.csv", "classification.csv", "predictions.parquet", "summary.md", "run.json"):
        assert (out / name).exists()
    assert "Final test" in (out / "summary.md").read_text(encoding="utf-8")
    assert "period" in json.loads((out / "run.json").read_text())
    with pytest.raises(SystemExit):
        final_test.main(["--features", str(path), "--out-dir", str(out)])
    final_test.main(["--features", str(path), "--out-dir", str(out), "--force"])


def _const_five():
    return Const(5.0)


def _small_ensemble():
    return AveragingRegressor({"c": _const_five})


def test_save_production_pickles_preprocessor_and_model(tmp_path, df, monkeypatch):
    monkeypatch.setattr(final, "final_ensemble", _small_ensemble)
    path = final_test.save_production(df, tmp_path / "m.pkl")

    bundle = pickle.loads(path.read_bytes())
    assert {"preprocessor", "model", "trained_on"} <= set(bundle)
    assert bundle["model"].predict(np.zeros((2, 3))).tolist() == [5.0, 5.0]


def test_main_save_production_flag(tmp_path, df, monkeypatch, capsys):
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(final_test, "save_production", lambda frame: tmp_path / "saved.pkl")

    final_test.main(["--features", str(path), "--save-production"])

    assert "saved" in capsys.readouterr().out
